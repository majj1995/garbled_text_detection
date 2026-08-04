from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torchvision.models import (  # type: ignore[import-untyped]
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)


class GlyphEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        *,
        pretrained: bool = False,
        cache_dir: Path = Path("models/cache"),
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        backbone = convnext_tiny(weights=None)
        if pretrained:
            cache_dir.mkdir(parents=True, exist_ok=True)
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            state_dict = torch.hub.load_state_dict_from_url(
                weights.url,
                model_dir=str(cache_dir),
                map_location="cpu",
                check_hash=True,
            )
            backbone.load_state_dict(state_dict)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(768),
            nn.Linear(768, embedding_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("glyph encoder input must have shape [N, 3, H, W]")
        if inputs.shape[-2:] != (96, 96):
            inputs = functional.interpolate(
                inputs,
                size=(96, 96),
                mode="bilinear",
                align_corners=False,
            )
        features = self.features(inputs)
        pooled = self.avgpool(features)
        embeddings = self.projection(pooled)
        return functional.normalize(embeddings, p=2, dim=1)
