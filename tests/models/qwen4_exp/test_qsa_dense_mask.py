"""Fused masked QSA must preserve independent Q/K/V and padding gradients."""

from __future__ import annotations

import os
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from torchturbo.models.qwen4_exp import qsa

from .test_model import _token_ids_to_mask


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

    original = qsa.attention_forward

    def masked(query, key, value, mask, query_length, is_causal, **kwargs):
        assert mask.is_contiguous() and mask.dtype == torch.bool
        assert mask.shape[-2] <= 5
        assert mask.any(dim=-1).all()
        assert not is_causal
        calls.append(mask.shape)
        return original(query, key, value, mask, query_length, is_causal, **kwargs)

    monkeypatch.setattr(qsa, "attention_forward", masked)
    monkeypatch.setattr(qsa, "_QSA_DENSE_MASK_MAX_BYTES", batch_size * 18 * 5)
    inputs, ids = _inputs(batch_size, 17, torch.float32, padding_only, 8)
    reference = [t.detach().clone().requires_grad_() for t in inputs]
    function = partial(
        qsa._qsa_masked_attention, softmax_scale=0.19, target_dtype=None
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


@pytest.mark.skipif(
    os.getenv("TORCH_TURBO_TEST_QSA_MASKED_NPU") != "1", reason="opt-in native NPU"
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
        qsa, "_QSA_DENSE_MASK_MAX_BYTES", batch_size * (key_length + 1) * 5
    )
    inputs, ids = _inputs(batch_size, key_length, dtype, padding_only, 256)
    device_inputs = [
        t.detach().to(device="npu", dtype=dtype).requires_grad_() for t in inputs
    ]
    function = partial(
        qsa._qsa_masked_attention, softmax_scale=None, target_dtype=None
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
