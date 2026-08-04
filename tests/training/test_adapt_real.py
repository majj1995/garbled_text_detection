import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from PIL import Image

from poor_word.training.adapt_real import AdaptConfig, adapt_real_encoder
from poor_word.training.train_glyph import GlyphClassifier, TrainConfig


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _prior_checkpoint(path: Path) -> Path:
    config = TrainConfig(manifest=path, output_dir=path.parent, embedding_dim=8, device="cpu")
    model = GlyphClassifier(2, config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "char_to_id": {"好": 0, "坏": 1},
            "config": config.model_dump(mode="json"),
        },
        path,
    )
    return path


def _real_crop_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    crop_root = tmp_path / "crops"
    images = crop_root / "images"
    images.mkdir(parents=True)
    colors = (("dev-a", "white"), ("dev-b", "gray"), ("dev-c", "black"), ("locked", "red"))
    for name, color in colors:
        Image.new("RGB", (16, 12), color).save(images / f"{name}.png")
    crops = crop_root / "crops.parquet"
    _write_parquet(
        crops,
        [
            {"crop_id": name, "image_id": image_id, "crop_path": f"images/{name}.png"}
            for name, image_id in (
                ("dev-a", "image-a"),
                ("dev-b", "image-b"),
                ("dev-c", "image-c"),
                ("locked", "image-locked"),
            )
        ],
    )
    manifest = tmp_path / "real-manifest.parquet"
    _write_parquet(
        manifest,
        [
            {
                "image_id": image_id,
                "training_eligible": eligible,
                "split_role": split_role,
            }
            for image_id, eligible, split_role in (
                ("image-a", True, "DEV"),
                ("image-b", True, "DEV"),
                ("image-c", True, "DEV"),
                ("image-locked", False, "LOCKED_TEST"),
            )
        ],
    )
    checkpoint = _prior_checkpoint(tmp_path / "prior.pt")
    return crops, manifest, checkpoint, images / "locked.png"


def test_adaptation_uses_eligible_crops_and_never_opens_locked_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: loading a locked-test crop into adaptation or diagnostics."""
    crops, manifest, checkpoint, locked_crop = _real_crop_inputs(tmp_path)
    from poor_word.training import adapt_real

    real_open = adapt_real.Image.open

    def reject_locked(path: str | Path, *args: object, **kwargs: object) -> Image.Image:
        if Path(path) == locked_crop:
            raise AssertionError("locked crop must not be opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(adapt_real.Image, "open", reject_locked)
    artifacts = adapt_real_encoder(
        AdaptConfig(
            crop_manifest=crops,
            real_manifest=manifest,
            prior_checkpoint=checkpoint,
            output_dir=tmp_path / "adapted",
            epochs=2,
            max_steps=2,
            batch_size=2,
            seed=37,
            device="cpu",
        )
    )

    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    saved = torch.load(artifacts.checkpoint, map_location="cpu")
    assert artifacts.checkpoint.exists()
    assert metrics["eligible_crop_count"] == 3
    assert metrics["train_crop_count"] == 2
    assert metrics["diagnostic_crop_count"] == 1
    assert metrics["steps"] == 2
    assert len(metrics["loss_history"]) == 2
    assert metrics["held_out_embedding_drift"]["mean_cosine_distance"] >= 0.0
    assert metrics["collapse_guard"]["passed"] is True
    assert saved["parent_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert saved["crop_manifest_sha256"] == hashlib.sha256(crops.read_bytes()).hexdigest()
    assert saved["real_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert saved["calibrated_for_block_decisions"] is False


def test_adaptation_rejects_duplicate_source_image_ids(tmp_path: Path) -> None:
    """Break caught: ambiguous crop-to-source eligibility joins."""
    crops, manifest, checkpoint, _locked = _real_crop_inputs(tmp_path)
    rows = pq.read_table(manifest).to_pylist()
    rows.append(dict(rows[0]))
    _write_parquet(manifest, rows)

    with pytest.raises(ValueError, match="duplicate image_id"):
        adapt_real_encoder(
            AdaptConfig(
                crop_manifest=crops,
                real_manifest=manifest,
                prior_checkpoint=checkpoint,
                output_dir=tmp_path / "adapted",
                max_steps=1,
                device="cpu",
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("training_eligible", "false"),
        ("split_role", None),
        ("split_role", "UNRECOGNIZED_ROLE"),
    ),
)
def test_adaptation_rejects_malformed_source_eligibility(
    tmp_path: Path, field: str, value: object
) -> None:
    """Break caught: malformed source values can become training admission."""
    crops, manifest, checkpoint, _locked = _real_crop_inputs(tmp_path)
    rows = pq.read_table(manifest).to_pylist()
    for row in rows:
        row[field] = value
    _write_parquet(manifest, rows)

    with pytest.raises(ValueError, match="malformed"):
        adapt_real_encoder(
            AdaptConfig(
                crop_manifest=crops,
                real_manifest=manifest,
                prior_checkpoint=checkpoint,
                output_dir=tmp_path / "adapted",
                max_steps=1,
                batch_size=2,
                device="cpu",
            )
        )


def test_adaptation_rejects_singleton_train_set_after_diagnostic_split(tmp_path: Path) -> None:
    """Break caught: a one-crop contrastive batch writes an unadapted artifact."""
    crops, manifest, checkpoint, _locked = _real_crop_inputs(tmp_path)
    crop_rows = [row for row in pq.read_table(crops).to_pylist() if row["crop_id"] != "dev-c"]
    source_rows = [
        row for row in pq.read_table(manifest).to_pylist() if row["image_id"] != "image-c"
    ]
    _write_parquet(crops, crop_rows)
    _write_parquet(manifest, source_rows)

    with pytest.raises(ValueError, match="at least two training crops"):
        adapt_real_encoder(
            AdaptConfig(
                crop_manifest=crops,
                real_manifest=manifest,
                prior_checkpoint=checkpoint,
                output_dir=tmp_path / "adapted",
                max_steps=1,
                batch_size=2,
                device="cpu",
            )
        )
    assert not (tmp_path / "adapted" / "encoder.pt").exists()


def test_adaptation_rejects_singleton_batch_size(tmp_path: Path) -> None:
    """Break caught: batch size one silently records a zero contrastive step."""
    crops, manifest, checkpoint, _locked = _real_crop_inputs(tmp_path)

    with pytest.raises(ValueError, match="greater than or equal to 2"):
        AdaptConfig(
            crop_manifest=crops,
            real_manifest=manifest,
            prior_checkpoint=checkpoint,
            output_dir=tmp_path / "adapted",
            batch_size=1,
            device="cpu",
        )


def test_adaptation_drops_singleton_tail_batch(tmp_path: Path) -> None:
    """Break caught: a singleton tail is counted as a zero-loss adaptation step."""
    crops, manifest, checkpoint, locked_crop = _real_crop_inputs(tmp_path)
    Image.new("RGB", (16, 12), "blue").save(locked_crop.parent / "dev-d.png")
    crop_rows = pq.read_table(crops).to_pylist()
    crop_rows.append(
        {"crop_id": "dev-d", "image_id": "image-d", "crop_path": "images/dev-d.png"}
    )
    source_rows = pq.read_table(manifest).to_pylist()
    source_rows.append(
        {"image_id": "image-d", "training_eligible": True, "split_role": "DEV"}
    )
    _write_parquet(crops, crop_rows)
    _write_parquet(manifest, source_rows)

    artifacts = adapt_real_encoder(
        AdaptConfig(
            crop_manifest=crops,
            real_manifest=manifest,
            prior_checkpoint=checkpoint,
            output_dir=tmp_path / "adapted",
            epochs=1,
            max_steps=2,
            batch_size=2,
            device="cpu",
        )
    )

    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert metrics["steps"] == 1
    assert len(metrics["loss_history"]) == 1
    assert metrics["loss_history"][0]["contrastive"] > 0.0
