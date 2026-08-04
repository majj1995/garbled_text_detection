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
        self.attention_network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.context_head = nn.Linear(context_dim, 1)
        self.raw_evidence_scale = nn.Parameter(torch.tensor(0.0))
        self.empty_bag_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape [B,N,feature_dim]")
        if mask.dtype is not torch.bool or mask.shape != features.shape[:2]:
            raise ValueError("mask must be bool with shape [B,N]")
        if self.feature_dim == 1:
            context = torch.zeros_like(features)
        else:
            context = features[..., 1:]
        raw_attention = self.attention_network(context).squeeze(-1)
        masked_attention = raw_attention.masked_fill(~mask, -torch.inf)
        nonempty = mask.any(dim=1)
        attention = torch.zeros_like(raw_attention)
        if nonempty.any():
            attention[nonempty] = torch.softmax(masked_attention[nonempty], dim=1)

        context_score = self.context_head(context).squeeze(-1)
        pooled_context = (attention * context_score).sum(dim=1)
        evidence = features[..., 0].masked_fill(~mask, -torch.inf)
        max_evidence = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        if nonempty.any():
            max_evidence[nonempty] = evidence[nonempty].max(dim=1).values
        monotonic = functional.softplus(self.raw_evidence_scale) * max_evidence
        logits = pooled_context + monotonic
        logits = torch.where(nonempty, logits, self.empty_bag_logit.expand_as(logits))
        return logits, attention
