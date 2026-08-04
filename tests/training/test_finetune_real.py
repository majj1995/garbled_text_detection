import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from PIL import Image

from poor_word.training.finetune_real import RealFineTuneConfig, finetune_real_fold
from poor_word.training.train_glyph import GlyphClassifier, TrainConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _inputs(tmp_path: Path) -> RealFineTuneConfig:
    crop_root = tmp_path / "crops"
    synthetic_root = tmp_path / "synthetic"
    (crop_root / "images").mkdir(parents=True)
    (synthetic_root / "images").mkdir(parents=True)
    (synthetic_root / "masks").mkdir(parents=True)
    for crop_id, color in (
        ("held-pass", "white"),
        ("held-block", "black"),
        ("train-pass", "red"),
        ("train-review", "blue"),
    ):
        Image.new("RGB", (14, 18), color).save(crop_root / "images" / f"{crop_id}.png")
    for sample_id, color in (("syn-pass", "white"), ("syn-block", "black")):
        Image.new("RGB", (16, 16), color).save(synthetic_root / "images" / f"{sample_id}.png")
        Image.new("L", (16, 16), "white").save(
            synthetic_root / "masks" / f"{sample_id}.png"
        )

    real = tmp_path / "real.parquet"
    _write_parquet(
        real,
        [
            {"image_id": "held-image", "split_role": "DEV", "training_eligible": True},
            {"image_id": "train-image", "split_role": "DEV", "training_eligible": True},
            {
                "image_id": "locked-image",
                "split_role": "LOCKED_TEST",
                "training_eligible": True,
            },
        ],
    )
    crops = crop_root / "crops.parquet"
    _write_parquet(
        crops,
        [
            {
                "crop_id": crop_id,
                "image_id": image_id,
                "crop_path": f"images/{crop_id}.png",
            }
            for crop_id, image_id in (
                ("held-pass", "held-image"),
                ("held-block", "held-image"),
                ("train-pass", "train-image"),
                ("train-review", "train-image"),
                # This path must never be opened.
                ("locked", "locked-image"),
            )
        ],
    )
    gold = tmp_path / "gold.parquet"
    _write_parquet(
        gold,
        [
            {
                "crop_id": "held-pass",
                "image_id": "held-image",
                "crop_path": "images/held-pass.png",
                "decision": "PASS",
                "anomaly_kind": "NONE",
            },
            {
                "crop_id": "held-block",
                "image_id": "held-image",
                "crop_path": "images/held-block.png",
                "decision": "BLOCK",
                "anomaly_kind": "missing_stroke",
            },
            {
                "crop_id": "train-pass",
                "image_id": "train-image",
                "crop_path": "images/train-pass.png",
                "decision": "PASS",
                "anomaly_kind": "NONE",
            },
            {
                "crop_id": "train-review",
                "image_id": "train-image",
                "crop_path": "images/train-review.png",
                "decision": "REVIEW",
                "anomaly_kind": "unknown",
            },
            {
                "crop_id": "locked",
                "image_id": "locked-image",
                "crop_path": "images/does-not-exist.png",
                "decision": "BLOCK",
                "anomaly_kind": "invented_character",
            },
        ],
    )
    folds = tmp_path / "folds.parquet"
    _write_parquet(
        folds,
        [
            {"image_id": "held-image", "fold": 0, "split_role": "DEV"},
            {"image_id": "train-image", "fold": 1, "split_role": "DEV"},
            {"image_id": "locked-image", "fold": -1, "split_role": "LOCKED_TEST"},
        ],
    )
    synthetic = synthetic_root / "manifest.parquet"
    _write_parquet(
        synthetic,
        [
            {
                "sample_id": "syn-pass",
                "image_path": "images/syn-pass.png",
                "mask_path": "masks/syn-pass.png",
                "base_char": "A",
                "decision": "PASS",
                "source_asset_ids": ["fixture"],
            },
            {
                "sample_id": "syn-block",
                "image_path": "images/syn-block.png",
                "mask_path": "masks/syn-block.png",
                "base_char": "A",
                "decision": "BLOCK",
                "source_asset_ids": ["fixture"],
            },
        ],
    )
    prior = tmp_path / "adapted.pt"
    classifier = GlyphClassifier(
        1,
        TrainConfig(
            manifest=synthetic,
            output_dir=tmp_path,
            embedding_dim=8,
            device="cpu",
        ),
    )
    torch.save(
        {
            "model_state": classifier.state_dict(),
            "char_to_id": {"A": 0},
            "config": {"embedding_dim": 8},
            "calibrated_for_block_decisions": False,
        },
        prior,
    )
    return RealFineTuneConfig(
        real_manifest=real,
        crop_manifest=crops,
        gold_manifest=gold,
        fold_manifest=folds,
        synthetic_manifest=synthetic,
        adapted_checkpoint=prior,
        output_dir=tmp_path / "fold-0",
        epochs=1,
        max_steps=1,
        batch_size=2,
        device="cpu",
        embedding_dim=8,
        seed=17,
    )


def test_finetune_excludes_held_out_and_review_labels_and_never_opens_locked_crop(
    tmp_path: Path,
) -> None:
    config = _inputs(tmp_path)

    artifacts = finetune_real_fold(config, held_out_fold=0)

    checkpoint = torch.load(artifacts.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["held_out_fold"] == 0
    assert checkpoint["parent_checkpoint_sha256"] == _sha256(config.adapted_checkpoint)
    assert checkpoint["fold_manifest_sha256"] == _sha256(config.fold_manifest)
    assert checkpoint["real_training_crop_ids"] == ["train-pass"]
    assert checkpoint["scoring_crop_ids"] == ["held-block", "held-pass"]
    assert "train-review" not in checkpoint["real_training_crop_ids"]
    scores = pq.read_table(artifacts.scores).to_pylist()
    assert {row["crop_id"] for row in scores} == {"held-pass", "held-block"}
    assert all(0.0 <= row["risk_score"] <= 1.0 for row in scores)
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert metrics["real_supervised_count"] == 1
    assert metrics["real_review_excluded_count"] == 1
    assert metrics["synthetic_replay_count"] == 2


@pytest.mark.parametrize(
    ("manifest_name", "field", "value", "match"),
    [
        ("fold_manifest", "fold", "0", "malformed fold"),
        ("fold_manifest", "split_role", "UNKNOWN", "malformed split_role"),
        ("real_manifest", "training_eligible", "true", "training_eligible"),
    ],
)
def test_finetune_rejects_malformed_linkage_before_writing_artifacts(
    tmp_path: Path, manifest_name: str, field: str, value: object, match: str
) -> None:
    config = _inputs(tmp_path)
    path = getattr(config, manifest_name)
    rows = pq.read_table(path).to_pylist()
    if field in {"fold", "training_eligible"}:
        for row in rows:
            row[field] = (
                str(row[field]).lower()
                if field == "training_eligible"
                else str(row[field])
            )
    else:
        rows[0][field] = value
    _write_parquet(path, rows)

    with pytest.raises(ValueError, match=match):
        finetune_real_fold(config, held_out_fold=0)
    assert not config.output_dir.exists()


def test_finetune_rejects_duplicate_crop_links(tmp_path: Path) -> None:
    config = _inputs(tmp_path)
    rows = pq.read_table(config.crop_manifest).to_pylist()
    rows.append(dict(rows[0]))
    _write_parquet(config.crop_manifest, rows)

    with pytest.raises(ValueError, match="duplicate crop_id"):
        finetune_real_fold(config, held_out_fold=0)
    assert not config.output_dir.exists()
