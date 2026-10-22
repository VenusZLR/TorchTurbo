"""NPU sparse flash attention adaptation for packed QSA."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice, get_device_runtime

_QSA_DENSE_MASK_MAX_BYTES = 256 * 1024 * 1024


def qsa_dense_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    softmax_scale: float | None,
    target_dtype: torch.dtype | None,
    *,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Apply selected QSA edges with bounded internal masks and fused FA."""
    from torch_npu import npu_fusion_attention

    if query.dtype == torch.float32 and target_dtype is not None:
        query, key, value = (tensor.to(target_dtype) for tensor in (query, key, value))
    batch_size, query_length, num_heads, head_dim = query.shape
    key_length = key.shape[1]
    chunk_size = max(1, _QSA_DENSE_MASK_MAX_BYTES // (batch_size * (key_length + 1)))
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    outputs = []
    for start in range(0, query_length, chunk_size):
        stop = min(start + chunk_size, query_length)
        indices = token_ids[:, start:stop]
        valid_queries = (indices >= 0).any(dim=-1)
        indices = torch.where(indices >= 0, indices, key_length)
        blocked = torch.ones(
            batch_size,
            stop - start,
            key_length + 1,
            dtype=torch.bool,
            device=query.device,
        )
        blocked.scatter_(-1, indices, False)
        # A padding query attends one real row, then its output is zeroed.
        # This avoids undefined softmax on a fully masked row and keeps its
        # Q/K/V gradients zero without dropping the SP backward collective.
        blocked[..., 0].logical_and_(valid_queries)
        attention_mask = blocked[..., :key_length].unsqueeze(1).contiguous()
        del blocked
        output = npu_fusion_attention(
            query[:, start:stop].transpose(1, 2).contiguous(),
            key,
            value,
            num_heads,
            "BNSD",
            atten_mask=attention_mask,
            scale=scale,
            keep_prob=1.0 - dropout,
            sparse_mode=1,
        )[0].transpose(1, 2)
        output = torch.where(valid_queries[:, :, None, None], output, 0)
        outputs.append(output)
    return torch.cat(outputs, dim=1)


def _sparse_query_head_groups(num_heads_per_kv: int) -> tuple[int, ...]:
    """Choose native SFA head counts for each GQA group."""
    if num_heads_per_kv == 12:
        return (16,)
    if num_heads_per_kv in (1, 2, 4, 8, 16, 32, 64, 128):
        return (num_heads_per_kv,)
    return ()


def can_use_sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    dropout: float,
    target_dtype: torch.dtype | None,
    *,
    softmax_scale: float | None = None,
) -> bool:
    """Check the common forward/backward SFA kernel contract."""
    dtype = query.dtype
    correct_dtype = dtype == torch.float32 and target_dtype is not None
    if correct_dtype:
        dtype = target_dtype
    num_query_heads, query_dim = query.shape[2:]
    num_kv_heads, key_dim = key.shape[2:]
    num_heads_per_kv = num_query_heads // num_kv_heads
    return (
        get_device_runtime().turbo_device is TurboDevice.NPU
        and dtype in (torch.bfloat16, torch.float16)
        and (correct_dtype or key.dtype == value.dtype == dtype)
        and dropout == 0.0
        and (softmax_scale is None or softmax_scale > 0)
        and query_dim == key_dim
        and key_dim + value.shape[-1] <= 512
        and num_query_heads % num_kv_heads == 0
        and bool(_sparse_query_head_groups(num_heads_per_kv))
        and 0 < token_ids.shape[-1] <= 8192
        and query.shape[0] * query.shape[1] <= 128 * 1024
        and key.shape[0] * key.shape[1] < 1024 * 1024
    )


def qsa_sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    softmax_scale: float | None,
    target_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Map independent Q/K/V to shared MLA KV without expanding selected rows."""
    batch_size, query_length, num_query_heads, key_dim = query.shape
    num_kv_heads, value_dim = value.shape[2:]
    dtype = query.dtype
    if dtype == torch.float32 and target_dtype is not None:
        dtype = target_dtype
        query, key, value = (tensor.to(dtype) for tensor in (query, key, value))
    output = query.new_zeros(batch_size * query_length, num_query_heads, value_dim)
    query_indices = torch.nonzero(
        (token_ids >= 0).any(dim=-1).flatten(), as_tuple=False
    ).flatten()
    if query_indices.numel() == 0:
        # Padding-only ranks must still join the K/V backward collective.
        zero = query[:, :0].sum() + key[:, :0].sum() + value[:, :0].sum()
        return output.view(batch_size, query_length, num_query_heads, value_dim) + zero

    import torch_npu

    selected_query = query.flatten(0, 1).index_select(0, query_indices)
    batch_offsets = (
        torch.arange(batch_size, dtype=token_ids.dtype, device=token_ids.device)
        * key.shape[1]
    ).view(batch_size, 1, 1)
    num_kv_tokens = key.shape[0] * key.shape[1]
    global_ids = torch.where(token_ids >= 0, token_ids + batch_offsets, num_kv_tokens)
    sparse_indices = global_ids.flatten(0, 1).index_select(0, query_indices)
    # Backward kernels share the portable 1024..8192, multiple-of-1024 widths.
    sparse_width = math.ceil(token_ids.shape[-1] / 1024) * 1024
    sparse_indices = (
        F.pad(
            sparse_indices,
            (0, sparse_width - token_ids.shape[-1]),
            value=num_kv_tokens,
        )
        .to(torch.int32)
        .unsqueeze(1)
        .contiguous()
    )
    key, value = key.flatten(0, 1), value.flatten(0, 1)
    # CANN backward skips gathering when KV length does not exceed sparse width.
    # Its sparse gather also requires every index to address a real row.
    kernel_kv_length = max(num_kv_tokens + 1, sparse_width + 1)
    query_lengths = torch.tensor(
        [selected_query.shape[0]], dtype=torch.int32, device=query.device
    )
    kv_lengths = torch.tensor(
        [kernel_kv_length], dtype=torch.int32, device=query.device
    )
    num_heads_per_kv = num_query_heads // num_kv_heads
    head_groups = _sparse_query_head_groups(num_heads_per_kv)
    scale = key_dim**-0.5 if softmax_scale is None else softmax_scale
    # A constant RoPE score masks the dummy row exactly at every positive scale.
    query_ropes = {}
    for group_size in head_groups:
        query_rope = query.new_zeros(selected_query.shape[0], group_size, 64)
        query_rope[..., 0] = 1
        query_ropes[group_size] = query_rope
    key_rope = key.new_zeros(kernel_kv_length, 1, 64)
    key_rope[num_kv_tokens, 0, 0] = -torch.inf
    outputs = []
    for head in range(num_kv_heads):
        head_start = head * num_heads_per_kv
        # Q has zeros over the V block, so logits use K alone. Autograd sums
        # the shared key/value derivatives into their disjoint K and V blocks.
        shared_kv = F.pad(
            torch.cat((key[:, head : head + 1], value[:, head : head + 1]), dim=-1),
            (0, 512 - key_dim - value_dim, 0, 0, 0, kernel_kv_length - num_kv_tokens),
        ).contiguous()
        for group_size in head_groups:
            remaining_heads = (head + 1) * num_heads_per_kv - head_start
            real_heads = min(group_size, remaining_heads)
            head_query = F.pad(
                selected_query[:, head_start : head_start + real_heads],
                (0, 512 - key_dim, 0, group_size - real_heads),
            ).contiguous()
            head_output, *_ = torch_npu.npu_sparse_flash_attention(
                head_query,
                shared_kv,
                shared_kv,
                sparse_indices,
                scale,
                block_table=None,
                actual_seq_lengths_query=query_lengths,
                actual_seq_lengths_kv=kv_lengths,
                query_rope=query_ropes[group_size],
                key_rope=key_rope,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="TND",
                sparse_mode=0,
                attention_mode=2,
                return_softmax_lse=True,
            )
            outputs.append(head_output[:, :real_heads, key_dim : key_dim + value_dim])
            head_start += real_heads
    output = output.index_copy(0, query_indices, torch.cat(outputs, dim=1))
    return output.view(batch_size, query_length, num_query_heads, value_dim)
