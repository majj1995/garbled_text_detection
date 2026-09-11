"""Strong gap plugs need new solid ink, not a large overlapping drawing mask."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from poor_word.glyphs import stroke_bridge
from poor_word.glyphs.stroke_corrupt import _width

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/previews/glyph-mixed-20260910"
WEAK_IDS = (
    "sb361978ba729",
    "s58b7f878c5b1",
    "s453475d07331",
    "sf7cd087529b7",
    "s4ea729dc042e",
)


def _planner(layers, seed=2):
    points = np.argwhere(np.maximum.reduce(layers) >= 128)
    return stroke_bridge.BridgePlanner(
        layers, [_width(x) for x in layers], float(np.ptp(points, axis=0).max() + 1), seed
    )


def test_gap_band_extends_beyond_both_banks_and_is_not_the_old_short_plug() -> None:
    layers = []
    for x in (38, 50):
        layer = np.zeros((96, 96), np.uint8)
        layer[34:61, x : x + 5] = 255
        layers.append(layer)
    planner = _planner(tuple(layers))
    passage = next(p for p in planner.passages if p.gate.axis == 0)
    proposal = planner._block(passage)
    assert proposal is not None
    added = proposal.layers[-1] >= 128
    y = passage.gate.row
    # Literal locations outside both five-pixel banks. The old bank +/-2 plug
    # leaves these columns blank; the stronger band visibly crosses both banks.
    assert added[y, 33] and added[y, 59]
    assert np.count_nonzero(added[y - 4 : y + 5, 43:50]) == 9 * 7
    for before, after in zip(layers, proposal.layers, strict=False):
        np.testing.assert_array_equal(before, after)


def test_solid_measurements_are_of_the_largest_real_new_component() -> None:
    original = np.zeros((96, 96), np.uint8)
    original[10:80, 20:25] = 255
    original[10:80, 35:40] = 255
    candidate = original.copy()
    candidate[43:53, 22:43] = 255
    metrics = stroke_bridge._new_solid_region(original, candidate, 5.0)
    assert metrics is not None
    # The central 10x10 region is largest; the 3x10 overhang is separate.
    assert metrics["new_solid_area_96"] == 100
    assert metrics["new_solid_length_96"] == 10
    assert metrics["new_core_pixels_96"] == 130


@pytest.mark.parametrize(
    "kind", ["overlap", "halo", "old_edge", "islands", "thin_line", "blob_with_whisker"]
)
def test_overlap_antialiasing_or_scattered_ink_cannot_count_as_a_solid_new_band(kind) -> None:
    original = np.zeros((96, 96), np.uint8)
    original[15:75, 15:30] = 255
    candidate = original.copy()
    if kind == "overlap":
        candidate[20:70, 16:29] = 255
    elif kind == "halo":
        candidate[20:70, 35:45] = 64
    elif kind == "old_edge":
        original[20:70, 35:45] = 100
        candidate[20:70, 35:45] = 255
    elif kind == "islands":
        candidate[20:70:3, 35:75:3] = 255
    elif kind == "thin_line":
        candidate[45, 35:80] = 255
    else:
        candidate[45:52, 45:52] = 255
        candidate[48, 52:78] = 255
    assert stroke_bridge._new_solid_region(original, candidate, 5.0) is None


def test_current_close_opening_pixels_stay_identical_for_archived_sites() -> None:
    rows = [json.loads(line) for line in (ARCHIVE / "candidates.jsonl").read_text().splitlines()]
    for row in rows:
        if row["bridge_mode"] != "close_opening":
            continue
        with np.load(ARCHIVE / row["stroke_archive"]["path"], allow_pickle=False) as data:
            layers = tuple(
                data[f"before_{i:03}"].copy() for i in range(row["original_stroke_count"])
            )
        planner = _planner(layers, row["seed"])
        proposal = planner.propose(int(row["metrics"]["attempts"]) - 1, None)
        assert proposal is not None and proposal.mode == "close_opening"
        rgb = np.repeat(np.maximum.reduce(proposal.layers)[:, :, None], 3, axis=2)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["pixel_sha256"]


@pytest.mark.parametrize("identifier", WEAK_IDS)
def test_old_noise_like_site_cannot_survive_unchanged(identifier) -> None:
    row = next(
        json.loads(line)
        for line in (ARCHIVE / "candidates.jsonl").read_text().splitlines()
        if json.loads(line)["candidate_id"] == identifier
    )
    with np.load(ARCHIVE / row["stroke_archive"]["path"], allow_pickle=False) as data:
        layers = tuple(data[f"before_{i:03}"].copy() for i in range(row["original_stroke_count"]))
    planner = _planner(layers, row["seed"])
    proposal = planner.propose(int(row["metrics"]["attempts"]) - 1, None)
    if proposal is None:
        return  # Insufficient new solid ink must be an explicit fixed-site skip.
    assert proposal.mode == "block_gap"
    rgb = np.repeat(np.maximum.reduce(proposal.layers)[:, :, None], 3, axis=2)
    assert hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]
    for key in ("gate_start_x_96", "gate_start_y_96", "gate_end_x_96", "gate_end_y_96"):
        assert proposal.metrics[key] == row["metrics"][key]
    assert proposal.metrics["new_solid_area_96"] >= 32


@pytest.mark.parametrize("size", [64, 128, 256])
def test_resizing_a_strong_gap_band_keeps_new_solid_ink_at_the_actual_input_size(size) -> None:
    layers = []
    for x in (38, 50):
        layer = np.zeros((96, 96), np.uint8)
        layer[34:61, x : x + 5] = 255
        layers.append(cv2.resize(layer, (size, size), interpolation=cv2.INTER_AREA))
    planner = _planner(tuple(layers))
    proposal = next(
        (p for item in planner.passages if (p := planner._block(item)) is not None), None
    )
    assert proposal is not None
    assert proposal.metrics["new_solid_length_96"] >= 10
    assert proposal.metrics["new_solid_area_96"] >= 32
