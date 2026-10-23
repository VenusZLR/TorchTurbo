"""Shared 4D attention masks preserve GQA, causal alignment and gradients."""

import pytest
import torch
import torch.nn.functional as F

from torchturbo.models.modules.attention.interface import attention_forward


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("query_length,key_length", [(3, 7), (7, 3), (1, 7)])
@pytest.mark.parametrize("alignment", ["none", "bottom_right", "top_left"])
def test_masked_gqa_matches_expanded_reference(additive, query_length, key_length, alignment):
    torch.manual_seed(41)
    inputs = [
        torch.randn(2, length, heads, 8, dtype=torch.float64, requires_grad=True)
        for length, heads in ((query_length, 6), (key_length, 2), (key_length, 2))
    ]
    reference = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    allowed = torch.rand(2, 1, query_length, key_length) > 0.3
    allowed[:, :, -1] = False
    mask = (
        torch.randn(2, 1, query_length, key_length, dtype=torch.float64).masked_fill(~allowed, -torch.inf)
        if additive else allowed
    )
    causal = alignment != "none"
    top_left = alignment == "top_left"
    output = attention_forward(
        *inputs, mask, query_length, causal, softmax_scale=0.19, use_top_left_mask=top_left
    )
    if causal and not (top_left and query_length == 1):
        offset = 0 if top_left else key_length - query_length
        allowed = allowed & (
            torch.arange(key_length)[None, :] <= torch.arange(query_length)[:, None] + offset
        )
    reference_mask = mask.masked_fill(~allowed, -torch.inf) if additive else allowed
    query, key, value = reference
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.repeat_interleave(3, 2).transpose(1, 2),
        value.repeat_interleave(3, 2).transpose(1, 2),
        attn_mask=reference_mask, scale=0.19,
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)
    probe = torch.randn_like(output)
    actual_gradients = torch.autograd.grad(output, inputs, probe)
    reference_gradients = torch.autograd.grad(expected, reference, probe)
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(output[:, -1]) == 0


def test_masked_attention_applies_target_dtype():
    inputs = [torch.randn(1, 3, 2, 8, requires_grad=True) for _ in range(3)]
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    actual = attention_forward(*inputs, mask, 3, False, target_dtype=torch.bfloat16)
    expected = F.scaled_dot_product_attention(
        *(tensor.to(torch.bfloat16).transpose(1, 2) for tensor in inputs), attn_mask=mask
    ).transpose(1, 2)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    actual.float().sum().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in inputs)


@pytest.mark.parametrize("options", [
    {"sliding_window": 3}, {"softcap": 5.0}, {"deterministic": True}, {"s_aux": torch.ones(2)},
])
def test_masked_attention_rejects_unsupported_backend_options(options):
    inputs = [torch.randn(1, 3, 2, 8) for _ in range(3)]
    with pytest.raises(ValueError, match="4D masked SDPA"):
        attention_forward(*inputs, torch.ones(1, 1, 3, 3, dtype=torch.bool), 3, False, **options)
