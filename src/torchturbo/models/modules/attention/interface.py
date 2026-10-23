"""HF-compatible attention with optional maximum-logit collection."""

import math
from typing import Any

import torch
from torch import Tensor

from torchturbo.ops import ops
from torchturbo.utils.flash_attn_utils import unpad_varlen_inputs
from torchturbo.utils.functools import cache


@cache
def _position_sequence_lengths(
    position_ids: Tensor,
    batch_size: int,
    _version: int | None,
) -> tuple[Tensor | None, Tensor | None, int | Tensor | None, int | Tensor | None]:
    from transformers.modeling_flash_attention_utils import (
        _is_packed_sequence,
        prepare_fa_kwargs_from_position_ids,
    )

    # HF detects flattened packing; also retain packing within multiple batch rows.
    packed = (
        _is_packed_sequence(position_ids, batch_size)
        if batch_size == 1
        else torch.any(position_ids[:, 1:] == 0)
    )
    if not packed:
        return None, None, None, None
    position_ids = position_ids.expand(batch_size, -1)
    if batch_size > 1:
        # Batch rows remain separate sequences even when RoPE positions start at an offset.
        position_ids = position_ids.clone()
        position_ids[:, 0] = 0
    lengths, max_lengths = prepare_fa_kwargs_from_position_ids(position_ids)
    return (*lengths, *max_lengths)


@cache
def _matching_cumulative_lengths(
    cu_seq_lens_q: Tensor,
    cu_seq_lens_k: Tensor,
    _query_version: int | None,
    _key_version: int | None,
) -> bool:
    return cu_seq_lens_q is cu_seq_lens_k or torch.equal(cu_seq_lens_q, cu_seq_lens_k)


def attention_forward(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attention_mask: Tensor | None,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Tensor | None = None,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    use_top_left_mask: bool = False,
    softcap: float | None = None,
    deterministic: bool | None = None,
    cu_seq_lens_q: Tensor | None = None,
    cu_seq_lens_k: Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    target_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
    *,
    return_max_logits: bool = False,
    **kwargs: Any,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run HF-compatible attention, optionally returning per-head maxima.

    Inputs and outputs use BSND layout. Ordinary attention delegates to the
    original HF helper. Maximum-logit attention follows HF's mask-first
    dispatch, passing sequence lengths directly to the registered provider.
    Maximum-logit attention currently supports full MLA self-attention with
    equal Q/K/V head counts, after RoPE and any sequence-parallel exchange.
    It owns no distributed communication or optimizer statistics.

    Args:
        query_states: Query activations in BSND layout.
        key_states: Key activations in BSND layout.
        value_states: Value activations in BSND layout.
        attention_mask: HF mask; maximum logits support integer sequence IDs
            with zero padding, or 4D masks for the CPU reference.
        query_length: Query sequence length, matching the HF helper contract.
        is_causal: Whether each sequence uses causal attention.
        dropout: Attention probability dropout.
        position_ids: Token positions used for packing when no mask or lengths
            are supplied; zero resets delimit packed sequences.
        softmax_scale: The actual scale applied to the QK product.
        sliding_window: HF sliding-window option; unsupported with maxima.
        use_top_left_mask: HF causal alignment compatibility option.
        softcap: HF logit softcap; unsupported with maxima.
        deterministic: HF deterministic option; unsupported with maxima.
        cu_seq_lens_q: Cumulative query lengths for padding-free input.
        cu_seq_lens_k: Matching cumulative key lengths.
        max_length_q: Maximum query segment length.
        max_length_k: Maximum key segment length.
        target_dtype: HF dtype correction for FP32 inputs.
        attn_implementation: Existing HF implementation for the default path.
        return_max_logits: Return detached FP32 maxima in execution-head order.
        **kwargs: Additional HF backend arguments for the default path.

    Returns:
        The attention output, or (output, head maxima) when requested.
    """
    if not return_max_logits:
        from transformers.modeling_flash_attention_utils import (
            _flash_attention_forward,
        )

        query, key, value = query_states, key_states, value_states
        query_padding = 0
        if attention_mask is None and all(
            item is not None
            for item in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
        ):
            query, key, value, query_padding = unpad_varlen_inputs(
                query, key, value, cu_seq_lens_q, cu_seq_lens_k,
            )
            if query is not query_states:
                position_ids = None
                query_length = query.shape[1]

        output = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length,
            is_causal,
            dropout=dropout,
            position_ids=position_ids,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            softcap=softcap,
            deterministic=deterministic,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            target_dtype=target_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        if query is not query_states:
            output = torch.nn.functional.pad(
                output.flatten(0, 1), (0, 0, 0, 0, 0, query_padding),
            )
            output = output.reshape(*query_states.shape[:-1], value_states.shape[-1])
        return output

    from transformers.modeling_flash_attention_utils import fa_peft_integration_check

    query, key, value = query_states, key_states, value_states
    output_shape = value.shape
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Maximum-logit attention expects Q/K/V in BSND layout.")
    if query.shape != key.shape or query.shape[:3] != value.shape[:3]:
        raise ValueError("Maximum logits require full self-attention with equal Q/K/V heads.")
    if query_length != query.shape[1]:
        raise ValueError("Maximum-logit query_length must match the input sequence length.")
    scaling = softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5
    if not math.isfinite(scaling) or scaling <= 0 or not 0 <= dropout < 1:
        raise ValueError("Maximum logits require positive finite scaling and 0 <= dropout < 1.")
    if sliding_window is not None or softcap not in (None, 0.0):
        raise ValueError("Maximum logits require full attention without logit softcapping.")
    if kwargs.get("s_aux") is not None:
        raise ValueError("Maximum logits do not support attention sinks.")
    if deterministic:
        raise ValueError("Explicit deterministic attention is not supported with maximum logits.")
    is_causal = is_causal and not (use_top_left_mask and query_length == 1)
    if attention_mask is not None and attention_mask.ndim not in (2, 4):
        raise ValueError("Maximum logits require a 2D or 4D attention mask.")
    if attention_mask is not None and attention_mask.ndim == 4:
        batch_size, seq_length, num_heads, _ = query.shape
        if (
            attention_mask.shape[0] not in (1, batch_size)
            or attention_mask.shape[1] not in (1, num_heads)
            or attention_mask.shape[2:] != (seq_length, seq_length)
        ):
            raise ValueError("The 4D mask must broadcast to [batch, heads, sequence, sequence].")
        if not attention_mask.is_floating_point() and attention_mask.dtype != torch.bool:
            raise ValueError("The 4D mask must be additive floating-point or boolean.")

    batch_size, seq_length = query.shape[:2]
    varlen_kwargs = {}
    query_padding = 0
    if attention_mask is not None:
        if attention_mask.ndim == 2 and attention_mask.shape != (batch_size, seq_length):
            raise ValueError("Maximum-logit attention expects a [batch, sequence] padding or packing mask.")
    elif cu_seq_lens_q is not None or cu_seq_lens_k is not None:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            raise ValueError("Maximum logits require matching query/key sequence lengths.")
        for lengths in (cu_seq_lens_q, cu_seq_lens_k):
            if lengths.ndim != 1 or lengths.numel() < 2 or lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("Cumulative sequence lengths must be a 1D integer tensor with at least two entries.")
        if cu_seq_lens_q.shape != cu_seq_lens_k.shape:
            raise ValueError("Maximum logits require matching query/key sequence lengths.")
        if (
            torch.is_inference_mode_enabled()
            or cu_seq_lens_q.is_inference()
            or cu_seq_lens_k.is_inference()
        ):
            matching_lengths = _matching_cumulative_lengths.__wrapped__(
                cu_seq_lens_q, cu_seq_lens_k, None, None,
            )
        else:
            matching_lengths = _matching_cumulative_lengths(
                cu_seq_lens_q, cu_seq_lens_k,
                cu_seq_lens_q._version, cu_seq_lens_k._version,
            )
        if not matching_lengths:
            raise ValueError("Maximum logits require matching query/key sequence lengths.")
    elif position_ids is not None:
        if position_ids.ndim != 2 or position_ids.shape[0] not in (1, batch_size):
            raise ValueError("Maximum-logit attention expects position_ids shaped [batch, sequence].")
        if position_ids.shape[1] != seq_length:
            raise ValueError("Maximum-logit attention position_ids must span the full attention sequence.")
        # Inference tensors have no mutation counters; other inputs can reuse HF metadata across layers.
        if torch.is_inference_mode_enabled() or position_ids.is_inference():
            prepare_lengths = _position_sequence_lengths.__wrapped__
            version = None
        else:
            prepare_lengths = _position_sequence_lengths
            version = position_ids._version
        cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k = prepare_lengths(
            position_ids, batch_size, version,
        )
    if attention_mask is None and cu_seq_lens_q is not None:
        query, key, value, query_padding = unpad_varlen_inputs(
            query, key, value, cu_seq_lens_q, cu_seq_lens_k,
        )
        varlen_kwargs = dict(
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
    query, key, value = fa_peft_integration_check(query, key, value, target_dtype)
    output, max_logits = ops.attention_with_max_logits(
        query, key, value, attention_mask, scaling, dropout, is_causal, **varlen_kwargs
    )
    if query_padding:
        output = torch.nn.functional.pad(
            output.flatten(0, 1), (0, 0, 0, 0, 0, query_padding),
        )
    return output.reshape(output_shape), max_logits
