"""QSA sparse flash attention adaptation and independent Q/K/V gradients."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

import torchturbo.runtime.runtime as runtime_impl
from torchturbo.models.qwen4_exp import npu, qsa
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
from torchturbo.runtime.npu import NpuRuntime

from . import test_qsa_attention
from .test_model import _token_ids_to_mask

flash_calls = test_qsa_attention.flash_calls


def _cpu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, **kwargs
):
    """Replace only the unavailable NPU kernel, including its MLA contract."""
    assert key is value
    assert query.shape[-1] == key.shape[-1] == value.shape[-1] == 512
    assert key.shape[1] == 1
    assert query.shape[1] in (1, 2, 4, 8, 16, 32, 64, 128)
    assert all(tensor.is_contiguous() for tensor in (query, key, value, sparse_indices))
    assert sparse_indices.dtype == torch.int32
    assert sparse_indices.shape[:2] == (query.shape[0], 1)
    assert sparse_indices.shape[-1] in (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
    for name, tensor in (
        ("actual_seq_lengths_query", query),
        ("actual_seq_lengths_kv", key),
    ):
        lengths = kwargs[name]
        assert isinstance(lengths, torch.Tensor)
        assert lengths.dtype == torch.int32
        assert lengths.device == tensor.device
        assert lengths.tolist() == [tensor.shape[0]]
    assert kwargs["layout_query"] == kwargs["layout_kv"] == "TND"
    assert kwargs["block_table"] is None
    assert kwargs["sparse_block_size"] == 1
    assert kwargs["sparse_mode"] == 0
    assert kwargs["attention_mode"] == 2
    assert kwargs["return_softmax_lse"] is True
    query_rope, key_rope = kwargs["query_rope"], kwargs["key_rope"]
    assert query_rope.shape == (*query.shape[:-1], 64)
    assert key_rope.shape == (*key.shape[:-1], 64)
    assert query_rope.is_contiguous() and key_rope.is_contiguous()
    assert query_rope.dtype == key_rope.dtype == query.dtype == key.dtype
    assert not query_rope.requires_grad and not key_rope.requires_grad
    assert torch.all(query_rope[..., 0] == 1)
    assert torch.count_nonzero(query_rope[..., 1:]) == 0
    dummy_rows = torch.nonzero(torch.isneginf(key_rope[:, 0, 0])).flatten()
    assert dummy_rows.numel() == 1
    dummy_index = int(dummy_rows[0])
    assert torch.count_nonzero(key_rope[:dummy_index]) == 0
    assert torch.count_nonzero(key_rope[dummy_index + 1 :]) == 0
    assert torch.count_nonzero(key_rope[dummy_index, :, 1:]) == 0
    assert torch.count_nonzero(key[dummy_index:]) == 0
    assert key.shape[0] == max(dummy_index + 1, sparse_indices.shape[-1] + 1)
    assert (sparse_indices >= 0).all()
    assert sparse_indices.max() <= dummy_index
    mask = _token_ids_to_mask(sparse_indices[:, 0].unsqueeze(0), key.shape[0])
    rope_scores = query_rope.transpose(0, 1) @ key_rope[:, 0].transpose(0, 1)
    attention_mask = (rope_scores * scale_value).unsqueeze(0)
    attention_mask = attention_mask.masked_fill(~mask, -torch.inf)
    repeats = query.shape[1]
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0),
            value.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0),
            attn_mask=attention_mask,
            scale=scale_value,
        )[0].transpose(0, 1)
    stats = torch.zeros(query.shape[0], query.shape[1], 1, dtype=torch.float32)
    return output, stats, stats


@pytest.fixture
def sfa_calls(monkeypatch):
    calls = []

    def sparse_flash(query, key, value, sparse_indices, scale_value, **kwargs):
        calls.append((query, key, sparse_indices, scale_value))
        return _cpu_sparse_flash_attention(
            query, key, value, sparse_indices, scale_value, **kwargs
        )

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_sparse_flash_attention=sparse_flash),
    )
    return calls


def _sparse_attention_forward(
    query, key, value, token_ids, *, softmax_scale=None, target_dtype=None
):
    """Exercise the retained SFA adapter independently of fused QSA dispatch."""
    return npu.qsa_sparse_flash_attention(
        query, key, value, token_ids, softmax_scale, target_dtype
    )


def _reference_attention(inputs, token_ids, scale=None, target_dtype=None):
    dtype = (
        target_dtype
        if inputs[0].dtype == torch.float32 and target_dtype is not None
        else inputs[0].dtype
    )
    query, key, value = [tensor.to(dtype) for tensor in inputs]
    repeats = query.shape[2] // key.shape[2]
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.repeat_interleave(repeats, dim=2).transpose(1, 2),
            value.repeat_interleave(repeats, dim=2).transpose(1, 2),
            attn_mask=_token_ids_to_mask(token_ids, key.shape[1]),
            scale=scale,
        ).transpose(1, 2)


@pytest.mark.parametrize(
    "dimensions,head_groups",
    [
        ((4, 2, 4, 3), (2, 2)),
        ((16, 2, 256, 256), (8, 8)),
        ((24, 2, 256, 256), (16, 16)),
    ],
)
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("scale", [None, 0.31, 1e-6])
def test_sfa_keeps_global_kv_and_matches_independent_gradients(
    sfa_calls, flash_calls, dimensions, head_groups, outer_checkpoint, scale
):
    torch.manual_seed(61)
    num_q_heads, num_kv_heads, key_dim, value_dim = dimensions
    inputs = [
        torch.randn(2, length, heads, dim, dtype=torch.double, requires_grad=True)
        for length, heads, dim in (
            (5, num_q_heads, key_dim),
            (7, num_kv_heads, key_dim),
            (7, num_kv_heads, value_dim),
        )
    ]
    # Query ownership is local; these selections contain global keys, multiple
    # packed segments, arbitrary block order, and trailing query padding.
    ids = torch.tensor(
        [
            [[0, -1, -1], [2, 0, 1], [3, -1, -1], [5, 3, 4], [-1, -1, -1]],
            [[1, 0, -1], [2, 1, -1], [4, 3, -1], [-1, -1, -1], [-1, -1, -1]],
        ],
        dtype=torch.int32,
    )
    if outer_checkpoint:
        actual = checkpoint(
            _sparse_attention_forward,
            *inputs,
            ids,
            use_reentrant=False,
            softmax_scale=scale,
        )
    else:
        actual = _sparse_attention_forward(*inputs, ids, softmax_scale=scale)
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    expected = _reference_attention(reference_inputs, ids, scale)
    torch.testing.assert_close(actual, expected)
    assert tuple(call[0].shape[1] for call in sfa_calls) == head_groups
    assert flash_calls == []
    for query, key, sparse_ids, scale_value in sfa_calls:
        assert query.shape[0] == 7
        assert key.shape == (1025, 1, 512)
        assert scale_value == (key_dim**-0.5 if scale is None else scale)
        assert sparse_ids[0, 0, :3].tolist() == [0, 14, 14]
        assert sparse_ids[4, 0, :3].tolist() == [8, 7, 14]
        assert torch.count_nonzero(query[..., key_dim:]) == 0
        assert torch.count_nonzero(key[..., key_dim + value_dim :]) == 0
    if head_groups == (16, 16):
        assert all(torch.count_nonzero(call[0][:, 12:]) == 0 for call in sfa_calls)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, probe)
    expected_grads = torch.autograd.grad(expected, reference_inputs, probe)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    if outer_checkpoint:
        assert len(sfa_calls) > len(head_groups)


@pytest.mark.parametrize(
    "input_dtype,target_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float16),
        (torch.bfloat16, None),
        (torch.float16, None),
    ],
)
@pytest.mark.parametrize(
    "grad_inputs",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("padding_only", [False, True])
@pytest.mark.parametrize("heads", [(4, 2, 2), (24, 2, 2)])
def test_sfa_dtype_partial_gradients_and_padding(
    sfa_calls, flash_calls, input_dtype, target_dtype, grad_inputs, padding_only, heads
):
    torch.manual_seed(62)
    inputs = [
        torch.randn(1, 4, heads, 4, dtype=input_dtype, requires_grad=grad)
        for heads, grad in zip(heads, grad_inputs, strict=True)
    ]
    ids = torch.tensor([[[0, -1], [0, 1], [-1, -1], [-1, -1]]], dtype=torch.int32)
    if padding_only:
        ids.fill_(-1)
    actual = _sparse_attention_forward(*inputs, ids, target_dtype=target_dtype)
    reference_inputs = [
        tensor.detach().clone().requires_grad_(grad)
        for tensor, grad in zip(inputs, grad_inputs, strict=True)
    ]
    expected = _reference_attention(reference_inputs, ids, target_dtype=target_dtype)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    if any(grad_inputs):
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, [tensor for tensor in inputs if tensor.requires_grad], probe
        )
        expected_grads = torch.autograd.grad(
            expected,
            [tensor for tensor in reference_inputs if tensor.requires_grad],
            probe,
        )
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.03, atol=0.03)
    assert len(sfa_calls) == (0 if padding_only else 2)
    assert flash_calls == []


def test_sfa_no_grad_context(sfa_calls, flash_calls):
    inputs = [torch.randn(1, 2, heads, 4, requires_grad=True) for heads in (4, 2, 2)]
    ids = torch.tensor([[[0, -1], [0, 1]]], dtype=torch.int32)
    with torch.no_grad():
        actual = _sparse_attention_forward(*inputs, ids)
        expected = _reference_attention(inputs, ids)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    assert len(sfa_calls) == 2
    assert flash_calls == []


@pytest.mark.parametrize(
    "width,padded_width", [(1024, 1024), (1025, 2048), (2051, 3072), (8192, 8192)]
)
@pytest.mark.parametrize("heads", [(4, 2, 2), (24, 2, 2)])
def test_sfa_pads_sparse_width_for_backward(
    sfa_calls, flash_calls, width, padded_width, heads
):
    inputs = [
        torch.randn(1, 2, heads, 4, dtype=torch.double, requires_grad=True)
        for heads in heads
    ]
    ids = torch.full((1, 2, width), -1, dtype=torch.int32)
    ids[0, 0, 0] = 0
    ids[0, 1, :2] = torch.tensor([1, 0], dtype=torch.int32)
    actual = _sparse_attention_forward(*inputs, ids)
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    expected = _reference_attention(reference_inputs, ids)
    torch.testing.assert_close(actual, expected)
    assert all(call[2].shape[-1] == padded_width for call in sfa_calls)
    assert flash_calls == []
    actual_grads = torch.autograd.grad(actual.sum(), inputs)
    expected_grads = torch.autograd.grad(expected.sum(), reference_inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("num_q_heads", [4, 24])
def test_sfa_kernel_errors_propagate(monkeypatch, sfa_calls, flash_calls, num_q_heads):
    def kernel_error(*args, **kwargs):
        raise RuntimeError("SFA kernel failure")

    monkeypatch.setattr(
        sys.modules["torch_npu"], "npu_sparse_flash_attention", kernel_error
    )
    inputs = [torch.randn(1, 2, heads, 4) for heads in (num_q_heads, 2, 2)]
    ids = torch.tensor([[[0, -1], [0, 1]]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="SFA kernel failure"):
        _sparse_attention_forward(*inputs, ids)
    assert flash_calls == []


def test_cpu_qsa_uses_fa2_without_importing_torch_npu(monkeypatch, flash_calls):
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", CpuRuntime())
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    inputs = [torch.randn(1, 2, heads, 4, dtype=torch.double) for heads in (4, 2, 2)]
    ids = torch.tensor([[[0, -1], [0, 1]]], dtype=torch.int32)
    actual = qsa.qsa_attention_forward(*inputs, ids)
    expected = _reference_attention(inputs, ids)
    torch.testing.assert_close(actual, expected)
    assert flash_calls == [(2, 3)]


@pytest.mark.parametrize("scale", [0.0, -0.31])
def test_nonpositive_scale_uses_fa2(monkeypatch, flash_calls, scale):
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", NpuRuntime())
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    inputs = [
        torch.randn(1, 2, heads, 4, dtype=torch.bfloat16, requires_grad=True)
        for heads in (4, 2, 2)
    ]
    ids = torch.tensor([[[0, -1], [0, 1]]], dtype=torch.int32)
    actual = qsa.qsa_attention_forward(*inputs, ids, softmax_scale=scale)
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    expected = _reference_attention(reference_inputs, ids, scale)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, probe)
    expected_grads = torch.autograd.grad(expected, reference_inputs, probe)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0.03, atol=0.03)
    assert flash_calls


@pytest.mark.parametrize(
    "runtime_device,dtype,target_dtype,dropout,q_heads,kv_heads,key_dim,value_dim,width,q_length,kv_length,eligible",
    [
        ("npu", torch.bfloat16, None, 0, 16, 2, 256, 256, 2051, 17, 23, True),
        ("npu", torch.bfloat16, None, 0, 24, 2, 256, 256, 2051, 17, 23, True),
        ("npu", torch.float16, None, 0, 24, 2, 256, 256, 2051, 16384, 32768, True),
        ("npu", torch.bfloat16, None, 0, 25, 2, 256, 256, 2051, 17, 23, False),
        ("npu", torch.float32, torch.float16, 0, 4, 1, 32, 16, 4, 17, 23, True),
        ("npu", torch.bfloat16, torch.float16, 0, 4, 1, 32, 16, 4, 17, 23, True),
        ("npu", torch.bfloat16, None, 0, 128, 1, 32, 16, 8192, 131072, 1048575, True),
        ("npu", torch.bfloat16, None, 0, 128, 1, 32, 16, 8192, 131072, 1048576, False),
        ("cpu", torch.bfloat16, None, 0, 16, 2, 256, 256, 4, 17, 23, False),
        ("cuda", torch.bfloat16, None, 0, 16, 2, 256, 256, 4, 17, 23, False),
        ("npu", torch.float64, torch.bfloat16, 0, 4, 1, 32, 16, 4, 17, 23, False),
        ("npu", torch.float32, None, 0, 16, 2, 256, 256, 4, 17, 23, False),
        ("npu", torch.bfloat16, None, 0.1, 16, 2, 256, 256, 4, 17, 23, False),
        ("npu", torch.bfloat16, None, 0, 6, 2, 32, 32, 4, 17, 23, False),
        ("npu", torch.bfloat16, None, 0, 256, 1, 32, 32, 4, 17, 23, False),
        ("npu", torch.bfloat16, None, 0, 16, 2, 256, 257, 4, 17, 23, False),
        ("npu", torch.bfloat16, None, 0, 16, 2, 256, 256, 8193, 17, 23, False),
        ("npu", torch.bfloat16, None, 0, 16, 2, 256, 256, 4, 131073, 23, False),
        ("npu", torch.bfloat16, None, 0, 16, 2, 256, 256, 4, 17, 1048577, False),
    ],
)
def test_sfa_eligibility(
    monkeypatch,
    runtime_device,
    dtype,
    target_dtype,
    dropout,
    q_heads,
    kv_heads,
    key_dim,
    value_dim,
    width,
    q_length,
    kv_length,
    eligible,
):
    runtimes = {"cpu": CpuRuntime, "cuda": CudaRuntime, "npu": NpuRuntime}
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", runtimes[runtime_device]())
    # Tensor metadata differs deliberately to verify runtime-based dispatch.
    query = SimpleNamespace(
        device=SimpleNamespace(type="npu" if runtime_device == "cpu" else "cpu"),
        dtype=dtype,
        shape=(1, q_length, q_heads, key_dim),
    )
    key = SimpleNamespace(dtype=dtype, shape=(1, kv_length, kv_heads, key_dim))
    value = SimpleNamespace(dtype=dtype, shape=(1, kv_length, kv_heads, value_dim))
    ids = SimpleNamespace(shape=(1, q_length, width))
    assert (
        npu.can_use_sparse_flash_attention(
            query, key, value, ids, dropout, target_dtype
        )
        is eligible
    )


def test_sfa_eligibility_requires_matching_qk_dimensions(monkeypatch):
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", NpuRuntime())
    query = SimpleNamespace(
        device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(1, 2, 4, 32)
    )
    key = SimpleNamespace(dtype=torch.bfloat16, shape=(1, 2, 1, 16))
    value = SimpleNamespace(dtype=torch.bfloat16, shape=(1, 2, 1, 16))
    ids = SimpleNamespace(shape=(1, 2, 4))
    assert not npu.can_use_sparse_flash_attention(query, key, value, ids, 0, None)


@pytest.mark.skipif(
    os.environ.get("TORCH_TURBO_TEST_QSA_SFA_NPU") != "1",
    reason="opt-in native NPU SFA forward/backward",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_q_heads,head_groups", [(16, (8, 8)), (24, (16, 16))])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
@pytest.mark.parametrize("padding_only", [False, True])
@pytest.mark.parametrize("kv_length", [17, 4096])
def test_native_sfa_default_heads_and_gradients(
    monkeypatch,
    dtype,
    num_q_heads,
    head_groups,
    checkpoint_mode,
    padding_only,
    kv_length,
    record_property,
):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", NpuRuntime())
    torch.manual_seed(63)
    inputs = [
        torch.randn(1, length, heads, 256).to(dtype).float().requires_grad_()
        for length, heads in ((17, num_q_heads), (kv_length, 2), (kv_length, 2))
    ]
    ids = torch.full((1, 17, 2051), -1, dtype=torch.int32)
    if not padding_only:
        for start, stop in ((0, 7), (7, 15)):
            for row in range(start, stop):
                count = min(row - start + 1, 5)
                ids[0, row, :count] = (
                    start + torch.randperm(row - start + 1)[:count].int()
                )
                if start:
                    ids[0, row, :count] += kv_length - 17
    device_inputs = [
        tensor.detach().to(device="npu", dtype=dtype).requires_grad_()
        for tensor in inputs
    ]
    calls = []
    native_op = torch_npu.npu_sparse_flash_attention

    def sparse_flash(*args, **kwargs):
        calls.append((args[0].shape, args[1].shape, args[3].shape))
        return native_op(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_sparse_flash_attention", sparse_flash)
    if checkpoint_mode is None:
        actual = _sparse_attention_forward(*device_inputs, ids.to("npu"))
    else:
        actual = checkpoint(
            _sparse_attention_forward,
            *device_inputs,
            ids.to("npu"),
            use_reentrant=checkpoint_mode,
        )
    expected = _reference_attention(inputs, ids)
    assert calls == (
        []
        if padding_only
        else [
            (
                torch.Size([15, heads, 512]),
                torch.Size([max(kv_length + 1, 3073), 1, 512]),
                torch.Size([15, 1, 3072]),
            )
            for heads in head_groups
        ]
    )
    probe = torch.randn_like(expected).to(dtype).float()
    with torch.autograd.set_detect_anomaly(True):
        actual.backward(probe.to("npu", dtype=dtype))
    expected.backward(probe)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.02, atol=0.02)
    record_property(
        "output_max_abs_error", (actual.float().cpu() - expected).abs().max().item()
    )
    for name, device_input, reference in zip(
        ("query", "key", "value"), device_inputs, inputs, strict=True
    ):
        torch.testing.assert_close(
            device_input.grad.float().cpu(), reference.grad, rtol=0.03, atol=0.03
        )
        record_property(
            f"{name}_grad_max_abs_error",
            (device_input.grad.float().cpu() - reference.grad).abs().max().item(),
        )
