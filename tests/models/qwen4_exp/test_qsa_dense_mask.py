"""Fused masked QSA must preserve independent Q/K/V and padding gradients."""

from __future__ import annotations

import os
import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from torchturbo.models.qwen4_exp import npu, qsa
from torchturbo.runtime import TurboDevice

from .test_qsa_sfa import _reference_attention


def _cpu_fusion_attention(query, key, value, heads, layout, **kwargs):
    """Substitute the fused NPU primitive while retaining production masks."""
    assert layout == "BNSD" and heads == query.shape[1]
    assert kwargs["sparse_mode"] == 1
    repeats = query.shape[1] // key.shape[1]
    with sdpa_kernel(SDPBackend.MATH):
        result = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(repeats, dim=1),
            value.repeat_interleave(repeats, dim=1),
            attn_mask=~kwargs["atten_mask"],
            scale=kwargs["scale"],
            dropout_p=1.0 - kwargs["keep_prob"],
        )
    return (result,)


def _inputs(batch_size, key_length, dtype, padding_only, head_dim):
    torch.manual_seed(126)
    inputs = [
        torch.randn(batch_size, length, heads, head_dim)
        .to(dtype)
        .float()
        .requires_grad_()
        for length, heads in ((17, 24), (key_length, 2), (key_length, 2))
    ]
    ids = torch.full((batch_size, 17, 2051), -1, dtype=torch.int32)
    if not padding_only:
        for batch in range(batch_size):
            for start, stop in ((0, 7), (7, 15)):
                for row in range(start, stop):
                    count = min(row - start + 1, 5)
                    selected = start + torch.randperm(row - start + 1)[:count].int()
                    if start:
                        selected += key_length - 17
                    ids[batch, row, :count] = selected
    return inputs, ids


@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
@pytest.mark.parametrize("padding_only", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_internal_mask_preserves_selection_and_independent_gradients(
    monkeypatch, checkpoint_mode, padding_only, batch_size
):
    calls = []

    def fused(query, key, value, heads, layout, **kwargs):
        assert layout == "BNSD" and heads == 24
        assert kwargs["sparse_mode"] == 1 and kwargs["keep_prob"] == 1
        mask = kwargs["atten_mask"]
        assert mask.is_contiguous() and mask.dtype == torch.bool
        assert mask.shape[-2] <= 5
        assert not mask.all(dim=-1).any()
        calls.append(mask.shape)
        return _cpu_fusion_attention(query, key, value, heads, layout, **kwargs)

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=fused)
    )
    monkeypatch.setattr(npu, "_QSA_DENSE_MASK_MAX_BYTES", batch_size * 18 * 5)
    inputs, ids = _inputs(batch_size, 17, torch.float32, padding_only, 8)
    reference = [t.detach().clone().requires_grad_() for t in inputs]
    function = partial(
        npu.qsa_dense_flash_attention, softmax_scale=0.19, target_dtype=None
    )
    output = (
        function(*inputs, ids)
        if checkpoint_mode is None
        else checkpoint(function, *inputs, ids, use_reentrant=checkpoint_mode)
    )
    expected = _reference_attention(reference, ids, 0.19)
    torch.testing.assert_close(output, expected)
    probe = torch.randn_like(output)
    output.backward(probe)
    expected.backward(probe)
    for actual, ref in zip(inputs, reference):
        torch.testing.assert_close(actual.grad, ref.grad)
    assert len(calls) >= 4


@pytest.mark.parametrize("padding_only", [False, True])
def test_npu_dispatch_uses_fused_attention(monkeypatch, padding_only):
    monkeypatch.setattr(
        npu, "get_device_runtime", lambda: SimpleNamespace(turbo_device=TurboDevice.NPU)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_fusion_attention=_cpu_fusion_attention),
    )

    def unexpected_sparse(*args, **kwargs):
        pytest.fail("Production QSA must dispatch to fused attention")

    monkeypatch.setattr(npu, "qsa_sparse_flash_attention", unexpected_sparse)
    inputs, ids = _inputs(1, 17, torch.bfloat16, padding_only, 8)
    reference = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    actual = qsa.qsa_attention_forward(*inputs, ids, target_dtype=torch.bfloat16)
    expected = _reference_attention(reference, ids, target_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    for tensor, ref in zip(inputs, reference, strict=True):
        torch.testing.assert_close(tensor.grad, ref.grad, rtol=0.03, atol=0.03)


@pytest.mark.skipif(
    os.getenv("TORCH_TURBO_TEST_QSA_SFA_NPU") != "1", reason="opt-in native NPU"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
@pytest.mark.parametrize("padding_only", [False, True])
@pytest.mark.parametrize("key_length", [17, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_native_dense_mask_output_and_gradients(
    monkeypatch, dtype, checkpoint_mode, padding_only, key_length, batch_size
):
    pytest.importorskip("torch_npu")
    monkeypatch.setattr(
        npu, "_QSA_DENSE_MASK_MAX_BYTES", batch_size * (key_length + 1) * 5
    )
    inputs, ids = _inputs(batch_size, key_length, dtype, padding_only, 256)
    device_inputs = [
        t.detach().to(device="npu", dtype=dtype).requires_grad_() for t in inputs
    ]
    function = partial(
        npu.qsa_dense_flash_attention, softmax_scale=None, target_dtype=None
    )
    output = (
        function(*device_inputs, ids.to("npu"))
        if checkpoint_mode is None
        else checkpoint(
            function, *device_inputs, ids.to("npu"), use_reentrant=checkpoint_mode
        )
    )
    expected = _reference_attention(inputs, ids)
    probe = torch.randn_like(expected).to(dtype).float()
    output.backward(probe.to("npu", dtype=dtype))
    expected.backward(probe)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float().cpu(), expected, rtol=0.02, atol=0.02)
    for actual, ref in zip(device_inputs, inputs):
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(
            actual.grad.float().cpu(), ref.grad, rtol=0.04, atol=0.04
        )
