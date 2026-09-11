"""V2 generation stays deterministic, bounded, licensed, and experimental."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from poor_word.data.manifest import SourceLock
from poor_word.glyphs.generate_v2 import V2GenerationConfig, generate_v2_dataset
from poor_word.glyphs.v2_manifest import validate_v2_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def generation_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build a tiny exact-byte source lock from real Make Me a Hanzi records."""
    repo = Path(__file__).resolve().parents[2]
    wanted = {"永", "一"}
    graphics = tmp_path / "graphics.txt"
    with (repo / "data/raw/makemeahanzi_graphics.txt").open(encoding="utf-8") as stream:
        selected = [line for line in stream if json.loads(line)["character"] in wanted]
    graphics.write_text("".join(selected), encoding="utf-8")
    payload = graphics.read_bytes()
    lock = SourceLock(
        source_id="makemeahanzi_graphics",
        declared_url="https://example.org/graphics.txt",
        resolved_url="https://example.org/graphics.txt",
        output_name="graphics.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="Arphic-1999",
        production_allowed=False,
    )
    lock_path = tmp_path / "graphics.lock.json"
    lock_path.write_text(lock.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return {
        "graphics": graphics,
        "lock": lock_path,
        "license": repo / "data/licenses/makemeahanzi.ARPHICPL.txt",
    }


def _config(
    tmp_path: Path,
    inputs: dict[str, Path],
    *,
    output_name: str = "dataset",
    characters: tuple[str, ...] = ("永",),
    **overrides: Any,
) -> V2GenerationConfig:
    values = {
        "output_dir": tmp_path / output_name,
        "graphics_path": inputs["graphics"],
        "source_lock_path": inputs["lock"],
        "license_path": inputs["license"],
        "characters": characters,
        "train_normal_per_char": 2,
        "eval_normal_per_char": 1,
        "train_abnormal_per_operator": 0,
        "eval_abnormal_per_operator": 0,
        "max_attempts": 2,
        "allow_experimental": True,
    }
    return V2GenerationConfig.model_validate({**values, **overrides})


def test_generation_requires_explicit_experimental_permission(
    tmp_path: Path, generation_inputs: dict[str, Path]
) -> None:
    """Break caught: the non-production source starts generation without an explicit opt-in."""
    config = _config(tmp_path, generation_inputs).model_copy(update={"allow_experimental": False})

    with pytest.raises(ValueError, match="experimental"):
        generate_v2_dataset(config)

    assert not config.output_dir.exists()


def test_generation_publishes_three_deterministic_valid_manifests(
    tmp_path: Path, generation_inputs: dict[str, Path]
) -> None:
    """Break caught: split identity, hashes, licensing, or reproducible layers drift."""
    first_config = _config(tmp_path, generation_inputs, output_name="first")
    second_config = _config(tmp_path, generation_inputs, output_name="second")

    first = generate_v2_dataset(first_config)
    second = generate_v2_dataset(second_config)

    assert first.dataset_id == second.dataset_id
    assert first.run_path == first_config.output_dir / "run.json"
    all_pixels: set[str] = set()
    all_groups: dict[str, str] = {}
    expected_rows = {"train": 2, "calibration": 1, "test": 1}
    for role, first_manifest, second_manifest in (
        ("train", first.train_manifest, second.train_manifest),
        ("calibration", first.calibration_manifest, second.calibration_manifest),
        ("test", first.test_manifest, second.test_manifest),
    ):
        assert first_manifest.name == f"{role}.parquet"
        assert _sha256(first_manifest) == _sha256(second_manifest)
        rows = pq.read_table(first_manifest).to_pylist()
        validate_v2_manifest(first_manifest, rows)
        assert len(rows) == expected_rows[role]
        assert {row["split_role"] for row in rows} == {role}
        assert {row["decision"] for row in rows} == {"PASS"}
        assert {row["training_eligible"] for row in rows} == {role == "train"}
        for row in rows:
            image_path = first_config.output_dir / row["image_path"]
            mask_path = first_config.output_dir / row["mask_path"]
            # The constant "v2-" version prefix must not collapse all assets
            # into one directory when the 3500-character profile is generated.
            expected_shard = row["sample_id"].removeprefix("v2-")[:2]
            assert image_path.parent.name == expected_shard
            assert mask_path.parent.name == expected_shard
            rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            assert row["pixel_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
            assert row["image_sha256"] == _sha256(image_path)
            assert row["mask_sha256"] == _sha256(mask_path)
            assert np.array_equal(mask > 0, rgb[:, :, 0] > 8)
            assert row["pixel_sha256"] not in all_pixels
            all_pixels.add(row["pixel_sha256"])
            prior_role = all_groups.setdefault(row["source_group_id"], role)
            assert prior_role == role

    run = json.loads(first.run_path.read_text(encoding="utf-8"))
    assert run["schema_version"] == "glyph-dataset-v2"
    assert run["experimental_only"] is True and run["production_allowed"] is False
    assert run["dataset_id"] == first.dataset_id
    assert run["grouping_limitation"]["same_characters_across_roles"] is True
    assert run["source_lock_sha256"] == {
        generation_inputs["lock"].name: _sha256(generation_inputs["lock"])
    }
    assert (first_config.output_dir / generation_inputs["lock"].name).read_bytes() == (
        generation_inputs["lock"].read_bytes()
    )
    assert (first_config.output_dir / "ARPHICPL.txt").read_bytes() == generation_inputs[
        "license"
    ].read_bytes()
    expected_code = {
        "generate_v2.py",
        "corrupt.py",
        "stroke_source.py",
        "stroke_corrupt.py",
        "stroke_break.py",
        "stroke_bridge.py",
    }
    assert expected_code <= set(run["code_sha256"])
    layer_path = first_config.output_dir / run["character_layers"]["永"]["path"]
    with np.load(layer_path, allow_pickle=False) as archive:
        layers = [archive[key] for key in sorted(archive.files) if key.startswith("stroke_")]
        assert layers and all(layer.dtype == np.uint8 for layer in layers)
        assert str(archive["source_char"]) == "永"
        assert str(archive["license_id"]) == "Arphic-1999"
        assert "2026-" in str(archive["modification"])


def test_block_generation_is_bounded_and_records_real_difference_masks(
    tmp_path: Path, generation_inputs: dict[str, Path]
) -> None:
    """Break caught: failed operators loop/fallback, or BLOCK masks cease matching edits."""
    config = _config(
        tmp_path,
        generation_inputs,
        characters=("一",),
        train_normal_per_char=1,
        eval_normal_per_char=1,
        train_abnormal_per_operator=1,
        eval_abnormal_per_operator=1,
        max_attempts=1,
    )

    artifacts = generate_v2_dataset(config)

    run = json.loads(artifacts.run_path.read_text(encoding="utf-8"))
    assert run["attempts"]["abnormal"] <= 15
    assert run["counts"]["expected_block"] == 15
    assert run["counts"]["actual_block"] <= 15
    assert run["counts"]["skipped_block"] == (
        run["counts"]["expected_block"] - run["counts"]["actual_block"]
    )
    assert run["complete"] == (run["counts"]["actual_block"] == 15)
    assert (
        sum(run["skip_reasons"].values())
        == run["attempts"]["abnormal"] - run["counts"]["actual_block"]
    )
    for manifest in (
        artifacts.train_manifest,
        artifacts.calibration_manifest,
        artifacts.test_manifest,
    ):
        for row in pq.read_table(manifest).to_pylist():
            if row["decision"] != "BLOCK":
                continue
            image = np.asarray(
                Image.open(config.output_dir / row["image_path"]).convert("RGB"), dtype=np.uint8
            )
            changed = np.asarray(
                Image.open(config.output_dir / row["mask_path"]).convert("L"), dtype=np.uint8
            )
            assert row["changed_pixels"] == int(np.count_nonzero(changed))
            assert row["changed_pixels"] > 0
            assert row["corruption_seed"] == row["seed"]
            assert row["selected_stroke_ids"]
            assert json.loads(row["metrics"])["input_changed_pixels"] >= 8
            assert image.shape == (128, 128, 3)


def test_existing_output_is_never_modified(
    tmp_path: Path, generation_inputs: dict[str, Path]
) -> None:
    """Break caught: a fresh V2 run overwrites a previous dataset or review artifact."""
    config = _config(tmp_path, generation_inputs)
    config.output_dir.mkdir()
    marker = config.output_dir / "human-review.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate_v2_dataset(config)

    assert marker.read_text(encoding="utf-8") == "keep"
