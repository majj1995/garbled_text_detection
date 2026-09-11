import json
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from PIL import Image

from poor_word.training.train_glyph import TrainConfig, train_glyph


def test_train_smoke_writes_checkpoint_and_metrics(
    generated_manifest: Path, tmp_path: Path
) -> None:
    artifacts = train_glyph(
        TrainConfig(
            manifest=generated_manifest,
            output_dir=tmp_path / "train",
            epochs=1,
            max_steps=2,
            batch_size=4,
            seed=11,
            pretrained=False,
            device="cpu",
        )
    )

    assert artifacts.checkpoint.exists()
    assert artifacts.metrics.exists()
    assert artifacts.prototype_bank.exists()


def test_validation_pass_rows_are_scored_but_never_fitted_as_prototypes(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rows: list[dict[str, object]] = []
    for index, (character, decision, source_id) in enumerate(
        [
            ("A", "PASS", "0"),
            ("A", "BLOCK", "0"),
            ("B", "PASS", "2"),
            ("B", "BLOCK", "2"),
        ]
    ):
        pixels = np.zeros((64, 64), dtype=np.uint8)
        pixels[8 + index : 48, 20 + index : 25 + index] = 255
        image = dataset_root / f"image-{index}.png"
        mask = dataset_root / f"mask-{index}.png"
        Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2)).save(image)
        Image.fromarray(pixels).save(mask)
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "base_char": character,
                "decision": decision,
                "source_asset_ids": [source_id],
                "image_path": image.name,
                "mask_path": mask.name,
            }
        )
    manifest = dataset_root / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)

    artifacts = train_glyph(
        TrainConfig(
            manifest=manifest,
            output_dir=tmp_path / "training",
            epochs=1,
            max_steps=1,
            batch_size=2,
            seed=3,
            pretrained=False,
            device="cpu",
            embedding_dim=8,
        )
    )

    with np.load(artifacts.prototype_bank, allow_pickle=False) as payload:
        assert payload["center_labels"].tolist() == ["A"]
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert metrics["train_nearest_prototype_accuracy"] == 1.0
    assert metrics["validation_nearest_prototype_accuracy"] == 0.0


def test_legacy_random_training_still_accepts_an_all_pass_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "all-pass"
    dataset_root.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(2):
        pixels = np.zeros((64, 64), dtype=np.uint8)
        pixels[8:48, 20 + index : 25 + index] = 255
        image = dataset_root / f"image-{index}.png"
        mask = dataset_root / f"mask-{index}.png"
        Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2)).save(image)
        Image.fromarray(pixels).save(mask)
        rows.append(
            {
                "sample_id": f"normal-{index}",
                "base_char": "A",
                "decision": "PASS",
                "source_asset_ids": ["0"],
                "image_path": image.name,
                "mask_path": mask.name,
            }
        )
    manifest = dataset_root / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)

    artifacts = train_glyph(
        TrainConfig(
            manifest=manifest,
            output_dir=tmp_path / "all-pass-training",
            epochs=1,
            max_steps=1,
            batch_size=2,
            device="cpu",
            embedding_dim=8,
        )
    )

    assert artifacts.prototype_bank.exists()
    checkpoint = torch.load(artifacts.checkpoint, map_location="cpu", weights_only=False)
    assert "experimental_only" not in checkpoint
    assert "production_allowed" not in checkpoint
