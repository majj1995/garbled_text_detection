import pytest
import torch

from poor_word.models.mil import AttentionMilPool


def test_attention_masks_padding_and_normalizes_nonempty_bags() -> None:
    model = AttentionMilPool(feature_dim=3, hidden_dim=5)
    features = torch.tensor([[[0.1, 1.0, 0.0], [0.8, 0.0, 1.0], [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])

    logits, attention = model(features, mask)

    assert logits.shape == (1,)
    assert attention.shape == (1, 3)
    assert torch.all(attention >= 0)
    assert attention[0, 2].item() == 0.0
    assert attention[0, :2].sum().item() == 1.0
    assert attention[0, 1] > attention[0, 0]


def test_zero_character_bag_has_finite_deterministic_fallback() -> None:
    model = AttentionMilPool(feature_dim=2, hidden_dim=4)
    mask = torch.zeros((2, 3), dtype=torch.bool)
    first, first_attention = model(torch.randn(2, 3, 2), mask)
    second, second_attention = model(torch.randn(2, 3, 2), mask)

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.equal(first_attention, torch.zeros_like(first_attention))
    assert torch.equal(second_attention, torch.zeros_like(second_attention))


def test_instance_permutation_preserves_bag_logit_and_permutes_attention() -> None:
    torch.manual_seed(9)
    model = AttentionMilPool(feature_dim=3, hidden_dim=5)
    features = torch.rand(1, 4, 3)
    mask = torch.tensor([[True, True, True, False]])
    permutation = torch.tensor([2, 0, 3, 1])

    original_logit, original_attention = model(features, mask)
    permuted_logit, permuted_attention = model(features[:, permutation], mask[:, permutation])

    assert torch.allclose(original_logit, permuted_logit, atol=1e-6)
    assert torch.allclose(original_attention[:, permutation], permuted_attention, atol=1e-6)


def test_increasing_one_evidence_value_cannot_reduce_bag_risk() -> None:
    torch.manual_seed(2)
    model = AttentionMilPool(feature_dim=3, hidden_dim=6)
    features = torch.rand(2, 4, 3)
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    raised = features.clone()
    raised[0, 1, 0] = torch.clamp(raised[0, 1, 0] + 0.2, max=1.0)
    raised[1, 0, 0] = torch.clamp(raised[1, 0, 0] + 0.2, max=1.0)

    before, _ = model(features, mask)
    after, _ = model(raised, mask)

    assert torch.all(after >= before)


def test_model_rejects_non_boolean_or_mismatched_masks() -> None:
    model = AttentionMilPool(feature_dim=1)
    features = torch.zeros((1, 2, 1))
    for mask in (torch.ones((1, 2)), torch.ones((1, 3), dtype=torch.bool)):
        with pytest.raises(ValueError):
            model(features, mask)
