"""Train-time affine views must preserve provenance and recompute derived channels."""

import hashlib
import importlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from poor_word.training.dataset import GlyphDataset


def _augmentation():
    return importlib.import_module("poor_word.training.augmentation")


def _glyph():
    gray = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(gray, (38, 30), (86, 96), 255, 7)
    return gray


def _parameters(**overrides):
    return {"angle": 0.0, "scale": 1.0, "translate_x": 0.0, "translate_y": 0.0} | overrides


def test_integer_translation_moves_ink_without_changing_intensity_or_topology():
    module = _augmentation()
    gray = _glyph()
    before = gray.copy()
    result = module.apply_affine(gray, _parameters(translate_x=2.0, translate_y=-1.0))
    expected = np.zeros_like(gray)
    expected[:-1, 2:] = gray[1:, :-2]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(gray, before)


def test_cropping_even_a_small_stroke_is_rejected():
    module = _augmentation()
    gray = _glyph()
    gray[20:25, 1:4] = 255  # Tiny, meaningful dot; total ink retention alone would miss its loss.
    with pytest.raises(module.UnsafeAugmentation, match="clipping"):
        module.apply_affine(gray, _parameters(translate_x=-2.0))


def test_rendered_topology_change_is_rejected_even_if_total_ink_is_retained():
    module = _augmentation()
    # A model-resolution one-pixel opening closes after half-pixel translation/interpolation.
    gray = np.zeros((96, 96), dtype=np.uint8)
    cv2.rectangle(gray, (24, 24), (72, 72), 255, 5)
    gray[20:28, 48] = 0
    with pytest.raises(module.UnsafeAugmentation, match="topology"):
        module.apply_affine(gray, _parameters(translate_x=0.5))


def test_no_visible_anomaly_in_reference_is_not_silently_accepted():
    module = _augmentation()
    gray = _glyph()
    with pytest.raises(module.UnsafeAugmentation, match="edit_visibility"):
        module.apply_affine(gray, _parameters(translate_x=1.0), reference=gray.copy())


def test_reference_missing_ink_area_is_protected_from_clipping():
    module = _augmentation()
    reference = _glyph()
    reference[20:26, 1:4] = 255
    # Erased dot is invisible in the observed foreground but remains part of the anomaly.
    with pytest.raises(module.UnsafeAugmentation, match="clipping"):
        module.apply_affine(_glyph(), _parameters(translate_x=-2.0), reference=reference)


def test_seeded_views_change_across_draws_but_never_consume_global_rng():
    module = _augmentation()
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    outputs = []
    for draw in range(8):
        seed = module.augmentation_seed(41, 1, 0, draw, "normal-1")
        image, trace = module.augment_affine(_glyph(), seed=seed)
        repeated, repeated_trace = module.augment_affine(_glyph(), seed=seed)
        np.testing.assert_array_equal(image, repeated)
        assert trace == repeated_trace
        outputs.append(image.tobytes())
    assert len(set(outputs)) > 1
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])
    seeds = {
        module.augmentation_seed(41, epoch, step, draw, sample)
        for epoch in (1, 2)
        for step in (0, 1)
        for draw in (0, 1)
        for sample in ("normal-1", "normal-2")
    }
    assert len(seeds) == 16


def test_bounded_rejection_returns_original_with_auditable_fallback():
    module = _augmentation()
    # Every possible mild warp clips full-canvas ink or violates its topology/area.
    gray = np.full((128, 128), 255, dtype=np.uint8)
    image, trace = module.augment_affine(gray, seed=42)
    np.testing.assert_array_equal(image, gray)
    assert not trace.applied
    assert trace.attempts == 4
    assert len(trace.rejection_reasons) == 4
    assert trace.parameters == {}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, role="train"):
    root.mkdir(exist_ok=True)
    original = _glyph()
    altered = original.copy()
    altered[25:44, 76:94] = 0
    archive = root / "layers.npz"
    np.savez(archive, stroke_000=original, source_char=np.asarray("口"))
    rows = []
    for index, (decision, gray) in enumerate((("PASS", original), ("BLOCK", altered))):
        image = root / f"image-{index}.png"
        mask = root / f"mask-{index}.png"
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        edit = original != altered
        Image.fromarray(rgb).save(image)
        Image.fromarray(((gray > 8) if decision == "PASS" else edit).astype(np.uint8) * 255).save(
            mask
        )
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "base_char": "口",
                "rendered_char": "口",
                "decision": decision,
                "anomaly_kind": "none" if index == 0 else "broken_stroke",
                "operator": "identity" if index == 0 else "break_stroke",
                "changed_pixels": 0 if index == 0 else int(edit.sum()),
                "seed": index,
                "image_path": image.name,
                "mask_path": mask.name,
                "bbox": {"x0": 30, "y0": 24, "x1": 95, "y1": 105},
                "source_asset_ids": ["fixture"],
                "schema_version": "glyph-dataset-v2",
                "dataset_id": "affine-fixture",
                "split_role": role,
                "source_group_id": "group-口",
                "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "image_sha256": _sha(image),
                "mask_sha256": _sha(mask),
                "training_eligible": role == "train",
                "production_allowed": False,
                "label_provenance": "synthetic_rule_v2",
                "source_layers_path": archive.name,
                "source_layers_sha256": _sha(archive),
                "metrics": json.dumps(
                    {
                        "appearance_scale": 1.0,
                        "appearance_rotation_degrees": 0.0,
                        "appearance_translate_x": 0.0,
                        "appearance_translate_y": 0.0,
                    }
                ),
            }
        )
    manifest = root / f"{role}.parquet"
    _write_manifest(manifest, rows)
    return manifest


def _write_manifest(manifest, rows):
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    (manifest.parent / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "glyph-dataset-v2",
                "dataset_id": "affine-fixture",
                "experimental_only": True,
                "production_allowed": False,
                "manifests": {manifest.name: {"sha256": _sha(manifest), "row_count": len(rows)}},
            }
        )
    )


@pytest.mark.parametrize("index", [0, 1])
def test_training_view_is_dynamic_but_plain_reads_and_source_files_remain_unchanged(
    tmp_path, index
):
    dataset = GlyphDataset(_manifest(tmp_path))
    original = dataset[index]
    sources = {path: _sha(path) for path in tmp_path.iterdir() if path.is_file()}
    views = []
    for epoch in range(1, 5):
        item, _trace = dataset.augmented_item(index, seed=42, epoch=epoch, step=0, draw=0)
        assert (item.sample_id, item.base_char, item.decision, item.label_id) == (
            original.sample_id,
            original.base_char,
            original.decision,
            original.label_id,
        )
        assert item.views.shape == (3, 96, 96)
        assert item.views.dtype == torch.float32
        # Independently derive the binary and edge channels from the actual transformed gray.
        gray = np.rint(item.views[0].numpy() * 255).astype(np.uint8)
        np.testing.assert_array_equal(item.views[1].numpy(), (gray > 8).astype(np.float32))
        np.testing.assert_array_equal(item.views[2].numpy(), cv2.Canny(gray, 50, 150) / 255)
        views.append(item.views.numpy().tobytes())
    assert len(set(views)) > 1
    assert torch.equal(dataset[index].views, original.views)
    assert {path: _sha(path) for path in sources} == sources


@pytest.mark.parametrize("role", ["calibration", "test"])
def test_training_augmentation_rejects_evaluation_roles(tmp_path, role):
    dataset = GlyphDataset(_manifest(tmp_path, role))
    with pytest.raises(ValueError, match="train"):
        dataset.augmented_item(0, seed=1, epoch=1, step=0, draw=0)


@pytest.mark.parametrize("defect", ["missing", "tampered", "escaping", "wrong_reference"])
def test_block_augmentation_requires_verified_aligned_normal_reference(tmp_path, defect):
    manifest = _manifest(tmp_path)
    rows = pq.read_table(manifest).to_pylist()
    if defect == "missing":
        rows[1].pop("source_layers_path")
    elif defect == "tampered":
        (tmp_path / "layers.npz").write_bytes(b"tampered")
    elif defect == "escaping":
        rows[1]["source_layers_path"] = "../layers.npz"
    else:
        metrics = json.loads(rows[1]["metrics"])
        metrics["appearance_translate_x"] = 2.0
        rows[1]["metrics"] = json.dumps(metrics)
    _write_manifest(manifest, rows)
    dataset = GlyphDataset(manifest)
    with pytest.raises(ValueError):
        dataset.augmented_item(1, seed=1, epoch=1, step=0, draw=0)


@pytest.mark.parametrize("defect", ["character", "dimensions"])
@pytest.mark.parametrize("prime_cache", [False, True])
def test_source_metadata_validation_is_independent_of_cache_access_order(
    tmp_path, defect, prime_cache
):
    manifest = _manifest(tmp_path)
    rows = pq.read_table(manifest).to_pylist()
    reference = _glyph()
    altered = reference.copy()
    altered[48:64, 80:94] = 0
    if defect == "dimensions":
        reference, altered = reference[:120, :120], altered[:120, :120]
    rgb = np.repeat(altered[:, :, None], 3, axis=2)
    edit = reference != altered
    image, mask = tmp_path / "other.png", tmp_path / "other-mask.png"
    Image.fromarray(rgb).save(image)
    Image.fromarray(edit.astype(np.uint8) * 255).save(mask)
    other = rows[1] | {
        "sample_id": "other-sample",
        "base_char": "目" if defect == "character" else "口",
        "rendered_char": "目" if defect == "character" else "口",
        "source_group_id": "other-group",
        "image_path": image.name,
        "mask_path": mask.name,
        "image_sha256": _sha(image),
        "mask_sha256": _sha(mask),
        "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "changed_pixels": int(edit.sum()),
    }
    _write_manifest(manifest, [*rows, other])
    dataset = GlyphDataset(manifest)
    if prime_cache:
        dataset.augmented_item(1, seed=1, epoch=1, step=0, draw=0)
    with pytest.raises(
        ValueError, match=f"source layers {defect} mismatch|source layers dimensions"
    ):
        dataset.augmented_item(2, seed=1, epoch=1, step=0, draw=1)
