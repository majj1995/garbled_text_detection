from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from PIL import Image

from poor_word.training.train_glyph import TrainConfig, train_glyph


def _v2_manifest(tmp_path: Path, *, role: str = "train") -> Path:
    root = tmp_path / f"dataset-{role}"
    root.mkdir()
    specs = [
        ("normal-1", "PASS", "none", "identity", 0),
        ("normal-2", "PASS", "none", "identity", 0),
        ("block-1", "BLOCK", "extra_stroke", "add_stroke", 7),
        ("block-2", "BLOCK", "missing_stroke", "erase_segment", 9),
    ]
    rows: list[dict[str, object]] = []
    for index, (sample_id, decision, anomaly, operator, changed) in enumerate(specs):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[8 + index : 48 + index, 20 + index : 25 + index] = 255
        image_path = root / f"image-{index}.png"
        mask_path = root / f"mask-{index}.png"
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(rgb[:, :, 0]).save(mask_path)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_path.name,
                "mask_path": mask_path.name,
                "base_char": "甲",
                "rendered_char": "甲",
                "decision": decision,
                "anomaly_kind": anomaly,
                "operator": operator,
                "changed_pixels": changed,
                "seed": index,
                "bbox": {"x0": 8, "y0": 8, "x1": 50, "y1": 55},
                "source_asset_ids": ["stroke-source-v2"],
                "schema_version": "glyph-dataset-v2",
                "dataset_id": "dataset-v2-test",
                "split_role": role,
                "source_group_id": "source-group-甲",
                "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                "training_eligible": role == "train",
                "production_allowed": False,
                "label_provenance": "synthetic_rule_v2",
            }
        )
    manifest = root / f"{role}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "glyph-dataset-v2",
                "dataset_id": "dataset-v2-test",
                "experimental_only": True,
                "production_allowed": False,
                "source_provenance": "synthetic stroke fixture",
                "manifests": {
                    manifest.name: {
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "row_count": len(rows),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize(
    ("role", "sampler", "allow_experimental", "message"),
    [
        ("train", "paired", False, "allow_experimental"),
        ("train", "random", True, "paired"),
        ("test", "paired", True, "split_role.*train"),
    ],
)
def test_v2_training_requires_explicit_experimental_paired_train_input(
    tmp_path: Path,
    role: str,
    sampler: str,
    allow_experimental: bool,
    message: str,
) -> None:
    manifest = _v2_manifest(tmp_path, role=role)

    with pytest.raises(ValueError, match=message):
        train_glyph(
            TrainConfig(
                manifest=manifest,
                output_dir=tmp_path / "output",
                epochs=1,
                max_steps=1,
                batch_size=4,
                device="cpu",
                sampler=sampler,  # type: ignore[arg-type]
                allow_experimental=allow_experimental,
            )
        )


def test_v2_training_records_exact_membership_sampling_and_progress(tmp_path: Path) -> None:
    manifest = _v2_manifest(tmp_path)
    output = tmp_path / "output"
    messages: list[str] = []

    artifacts = train_glyph(
        TrainConfig(
            manifest=manifest,
            output_dir=output,
            epochs=1,
            max_steps=1,
            batch_size=4,
            seed=41,
            pretrained=False,
            device="cpu",
            embedding_dim=8,
            sampler="paired",
            allow_experimental=True,
            log_every=1,
        ),
        progress=messages.append,
    )

    membership_path = output / "training_membership.json"
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    assert membership == {
        "schema_version": "glyph-training-membership-v2",
        "dataset_id": "dataset-v2-test",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sample_ids": ["normal-1", "normal-2", "block-1", "block-2"],
        "source_group_ids": ["source-group-甲"] * 4,
        "pixel_sha256": [str(row["pixel_sha256"]) for row in pq.read_table(manifest).to_pylist()],
    }
    membership_sha = hashlib.sha256(membership_path.read_bytes()).hexdigest()
    checkpoint = torch.load(artifacts.checkpoint, map_location="cpu", weights_only=False)
    prototype_metadata = json.loads(
        artifacts.prototype_bank.with_suffix(".json").read_text(encoding="utf-8")
    )
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    progress_events = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert checkpoint["training_membership_sha256"] == membership_sha
    assert prototype_metadata["training_membership_sha256"] == membership_sha
    assert (
        prototype_metadata["prototype_bank_sha256"]
        == hashlib.sha256(artifacts.prototype_bank.read_bytes()).hexdigest()
    )
    assert prototype_metadata["experimental_only"] == "true"
    assert prototype_metadata["production_allowed"] == "false"
    assert checkpoint["experimental_only"] is True
    assert checkpoint["production_allowed"] is False
    assert metrics["experimental_only"] is True
    assert metrics["production_allowed"] is False
    assert metrics["dataset_schema_version"] == "glyph-dataset-v2"
    assert metrics["dataset_id"] == "dataset-v2-test"
    assert metrics["normal_draws"] == 2
    assert metrics["eligible_normal_draws"] == 2
    assert metrics["eligible_normal_fraction"] == 1.0
    assert metrics["positive_pairs"] == 1
    assert metrics["batches_without_positive_pairs"] == 0
    assert any("startup" in message for message in messages)
    assert any("epoch=1 step=1" in message and "loss=" in message for message in messages)
    assert any("embedding" in message for message in messages)
    assert any("prototype" in message for message in messages)
    assert any(event["phase"] == "embedding_progress" for event in progress_events)
    assert progress_events[-1]["phase"] == "complete"
    assert progress_events[-1]["steps"] == 1
    assert progress_events[-1]["train_nearest_prototype_accuracy"] == 1.0


def test_training_refuses_to_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    manifest = _v2_manifest(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = output / "encoder.pt"
    checkpoint.write_bytes(b"existing-model")

    with pytest.raises(FileExistsError, match="fresh output"):
        train_glyph(
            TrainConfig(
                manifest=manifest,
                output_dir=output,
                epochs=1,
                max_steps=1,
                batch_size=4,
                device="cpu",
                sampler="paired",
                allow_experimental=True,
            )
        )

    assert checkpoint.read_bytes() == b"existing-model"
