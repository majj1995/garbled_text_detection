"""Permutation-invariant attention pooling for character evidence bags."""

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class AttentionMilPool(nn.Module):
    """Pool padded character evidence while preserving monotonic base-risk evidence."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = feature_dim
        context_dim = max(1, feature_dim - 1)
        self.raw_attention_strength = nn.Parameter(torch.tensor(0.0))
        self.context_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.raw_evidence_scale = nn.Parameter(torch.tensor(0.0))
        self.empty_bag_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape [B,N,feature_dim]")
        if mask.dtype is not torch.bool or mask.shape != features.shape[:2]:
            raise ValueError("mask must be bool with shape [B,N]")
        valid_evidence = features[..., 0][mask]
        if valid_evidence.numel() and (
            not torch.isfinite(valid_evidence).all()
            or (valid_evidence < 0).any()
            or (valid_evidence > 1).any()
        ):
            raise ValueError("valid base evidence must be finite and within [0,1]")
        if self.feature_dim == 1:
            context = torch.zeros_like(features)
        else:
            context = features[..., 1:]
        attention_strength = torch.sigmoid(self.raw_attention_strength)
        raw_attention = attention_strength * features[..., 0]
        masked_attention = raw_attention.masked_fill(~mask, -torch.inf)
        nonempty = mask.any(dim=1)
        attention = torch.zeros_like(raw_attention)
        if nonempty.any():
            attention[nonempty] = torch.softmax(masked_attention[nonempty], dim=1)

        context_score = self.context_head(context).squeeze(-1)
        valid_count = mask.sum(dim=1).clamp_min(1).to(features.dtype)
        pooled_context = (mask * context_score).sum(dim=1) / valid_count
        evidence = features[..., 0].masked_fill(~mask, -torch.inf)
        max_evidence = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        attended_evidence = (attention * features[..., 0]).sum(dim=1)
        if nonempty.any():
            max_evidence[nonempty] = evidence[nonempty].max(dim=1).values
        monotonic_pool = 0.5 * (attended_evidence + max_evidence)
        monotonic = functional.softplus(self.raw_evidence_scale) * monotonic_pool
        logits = pooled_context + monotonic
        logits = torch.where(nonempty, logits, self.empty_bag_logit.expand_as(logits))
        return logits, attention
