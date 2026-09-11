"""V2 manifests must keep split intent, provenance and actual input bytes intact."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from poor_word.training.dataset import GlyphDataset


def _fixture(tmp_path: Path, role: str = "train", **overrides):
    rgb = np.zeros((128, 128, 3), np.uint8)
    rgb[40:90, 60:68] = 255
    Image.fromarray(rgb).save(tmp_path / "image.png")
    Image.fromarray(rgb[:, :, 0]).save(tmp_path / "mask.png")
    row = {
        "sample_id": "sample-1",
        "image_path": "image.png",
        "mask_path": "mask.png",
        "base_char": "一",
        "rendered_char": "一",
        "decision": "PASS",
        "anomaly_kind": "none",
        "operator": "identity",
        "changed_pixels": 0,
        "seed": 1,
        "bbox": {"x0": 60, "y0": 40, "x1": 68, "y1": 90},
        "source_asset_ids": ["makemeahanzi_graphics"],
        "schema_version": "glyph-dataset-v2",
        "dataset_id": "dataset-1",
        "split_role": role,
        "source_group_id": "group-1",
        "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "image_sha256": hashlib.sha256((tmp_path / "image.png").read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256((tmp_path / "mask.png").read_bytes()).hexdigest(),
        "training_eligible": role == "train",
        "production_allowed": False,
        "label_provenance": "synthetic_rule_v2",
        **overrides,
    }
    manifest = tmp_path / f"{role}.parquet"
    pq.write_table(pa.Table.from_pylist([row]), manifest)
    _metadata(manifest)
    return manifest


def _metadata(manifest):
    (manifest.parent / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "glyph-dataset-v2",
                "dataset_id": "dataset-1",
                "experimental_only": True,
                "production_allowed": False,
                "manifests": {
                    manifest.name: {
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "row_count": pq.read_metadata(manifest).num_rows,
                    }
                },
            }
        )
    )


def test_v2_split_does_not_drop_training_characters_using_the_old_hash_split(tmp_path):
    dataset = GlyphDataset(_fixture(tmp_path))
    assert dataset.schema_version == "glyph-dataset-v2"
    assert dataset.dataset_id == "dataset-1"
    assert dataset.split_role == "train"
    assert dataset.split_indices() == ((0,), ())
    assert dataset[0].views.shape == (3, 96, 96)


def test_v2_test_rows_are_never_returned_as_training_indices(tmp_path):
    dataset = GlyphDataset(_fixture(tmp_path, "test"))
    assert dataset.split_indices() == ((), (0,))


@pytest.mark.parametrize(
    "overrides",
    [
        {"decision": "REVIEW"},
        {"training_eligible": False},
        {"production_allowed": True},
        {"label_provenance": "human_review"},
        {"schema_version": "glyph-dataset-v9"},
        {"dataset_id": "different"},
        {"image_path": "../outside.png"},
        {"mask_sha256": "invalid"},
    ],
)
def test_v2_refuses_unreviewed_mislabelled_or_misrouted_rows(tmp_path, overrides):
    manifest = _fixture(tmp_path, **overrides)
    with pytest.raises(ValueError):
        GlyphDataset(manifest)


def test_v2_refuses_manifest_modified_since_generation(tmp_path):
    manifest = _fixture(tmp_path)
    rows = pq.read_table(manifest).to_pylist()
    rows[0]["sample_id"] = "changed"
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    with pytest.raises(ValueError, match="hash"):
        GlyphDataset(manifest)


def test_v2_refuses_duplicate_sample_identity(tmp_path):
    manifest = _fixture(tmp_path)
    rows = pq.read_table(manifest).to_pylist() * 2
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    _metadata(manifest)
    with pytest.raises(ValueError, match="duplicate"):
        GlyphDataset(manifest)


@pytest.mark.parametrize("asset", ["image.png", "mask.png"])
def test_v2_refuses_changed_asset_instead_of_training_wrong_pixels(tmp_path, asset):
    dataset = GlyphDataset(_fixture(tmp_path))
    Image.new("RGB" if asset == "image.png" else "L", (128, 128), 100).save(tmp_path / asset)
    with pytest.raises(ValueError, match="hash"):
        dataset[0]


def test_v2_pixel_hash_describes_decoded_rgb_not_encoded_png(tmp_path):
    dataset = GlyphDataset(_fixture(tmp_path, pixel_sha256="0" * 64))
    with pytest.raises(ValueError, match="pixel"):
        dataset[0]
