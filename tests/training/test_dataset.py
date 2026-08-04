from pathlib import Path

import torch

from poor_word.training.dataset import GlyphDataset


def test_dataset_returns_three_views_and_label(generated_manifest: Path) -> None:
    item = GlyphDataset(generated_manifest)[0]

    assert item.views.shape == (3, 96, 96)
    assert item.views.dtype == torch.float32
    assert item.decision in {"PASS", "BLOCK"}
    assert len(item.base_char) == 1
