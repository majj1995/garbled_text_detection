"""Local blank-space topology, not character legality or semantic confidence."""

import hashlib
import io

import cv2
import numpy as np
import pytest

from poor_word.glyphs.corrupt import CorruptionNotApplicable
from poor_word.glyphs.stroke_corrupt import corrupt_stroke_layers


def _u(size: int = 128) -> tuple[np.ndarray, ...]:
    alpha = np.zeros((128, 128), np.uint8)
    cv2.polylines(alpha, [np.array([[28, 25], [28, 100], [99, 100], [99, 25]])], False, 255, 9)
    return (cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA),)


def _gap(long: bool = False, wide: bool = False) -> tuple[np.ndarray, ...]:
    layers = []
    for x in (42, 98 if wide else 62):
        layer = np.zeros((128, 128), np.uint8)
        cv2.line(layer, (x, 20 if long else 43), (x, 108 if long else 84), 255, 9)
        layers.append(layer)
    return tuple(layers)


def _foreground(image: np.ndarray, threshold: int) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA) >= threshold


def _outside(foreground: np.ndarray) -> np.ndarray:
    _, regions = cv2.connectedComponents((~foreground).astype(np.uint8), connectivity=4)
    labels = np.unique(np.concatenate((regions[0], regions[-1], regions[:, 0], regions[:, -1])))
    return np.isin(regions, labels[labels != 0])


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("size", [64, 128, 256])
def test_close_opening_encloses_a_large_persistent_region_in_one_connected_u_or_c(
    rotate: bool,
    size: int,
) -> None:
    layers = _u(size)
    if rotate:
        layers = tuple(np.rot90(x).copy() for x in layers)
    result = corrupt_stroke_layers(layers, "bridge", 3, bridge_mode="close_opening")
    assert result.bridge_mode == "close_opening"
    for threshold in (9, 128):
        before = _foreground(layers[0], threshold)
        after = _foreground(result.image, threshold)
        assert cv2.connectedComponents(before.astype(np.uint8), connectivity=8)[0] == 2
        assert cv2.connectedComponents(after.astype(np.uint8), connectivity=8)[0] == 2
        captured = _outside(before) & ~_outside(after) & ~after
        assert captured.sum() >= 250
        assert cv2.distanceTransform(captured.astype(np.uint8), cv2.DIST_L2, 5).max() >= 6
        assert not after[48, 48]
        assert _outside(before)[48, 48] and not _outside(after)[48, 48]
    assert result.metrics["retained_region_area_96"] >= 250
    assert result.metrics["region_seed_x_96"] >= result.metrics["roi_x0_96"]
    assert result.metrics["region_seed_x_96"] < result.metrics["roi_x1_96"]


def test_block_gap_consumes_a_substantial_preselected_narrow_passage() -> None:
    layers = _gap()
    result = corrupt_stroke_layers(layers, "bridge", 2, bridge_mode="block_gap")
    assert result.bridge_mode == "block_gap"
    for threshold in (9, 128):
        before = _foreground(np.maximum.reduce(layers), threshold)
        after = _foreground(result.image, threshold)
        # This literal original corridor lies between the two independent bars.
        assert not before[36:62, 37:42].any()
        assert np.count_nonzero(after[36:62, 37:42]) >= 26 * 5 * 0.30
    assert result.metrics["blocked_length_fraction"] >= 0.30
    assert result.metrics["gap_consumed_fraction"] >= 0.30


def test_thin_local_banks_are_not_disqualified_by_unrelated_full_glyph_extent() -> None:
    layers = []
    for x in (42, 53):
        layer = np.zeros((128, 128), np.uint8)
        cv2.line(layer, (x, 43), (x, 65), 255, 3)
        layers.append(layer)
    distant = np.zeros((128, 128), np.uint8)
    cv2.line(distant, (8, 115), (119, 115), 255, 3)
    layers.append(distant)
    result = corrupt_stroke_layers(tuple(layers), "bridge", 2, bridge_mode="block_gap")
    assert result.metrics["gap_consumed_fraction"] >= 0.30
    assert result.metrics["input_changed_fraction"] >= 0.025
    assert result.metrics["local_stroke_width_96"] < 5


def test_substantial_gap_swallowing_does_not_require_both_original_ports_to_survive() -> None:
    layers = []
    for x in (42, 62):
        layer = np.zeros((128, 128), np.uint8)
        cv2.line(layer, (x, 50), (x, 66), 255, 9)
        layers.append(layer)
    result = corrupt_stroke_layers(tuple(layers), "bridge", 2, bridge_mode="block_gap")
    assert result.metrics["blocked_length_fraction"] >= 0.50
    assert result.metrics["gap_consumed_fraction"] >= 0.50


@pytest.mark.parametrize("mode", [None, "close_opening", "block_gap"])
@pytest.mark.parametrize("wide", [False, True])
def test_no_mode_falls_back_to_a_meaningless_short_bridge_between_long_parallel_bars(
    mode: str | None,
    wide: bool,
) -> None:
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers(_gap(long=True, wide=wide), "bridge", 2, bridge_mode=mode)


def test_antialiased_closed_mouth_is_not_an_original_opening_at_both_input_thresholds() -> None:
    layer = _u()[0].copy()
    cv2.line(layer, (28, 25), (99, 25), 32, 6)
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers((layer,), "bridge", 5, bridge_mode="close_opening")


def test_native_closure_that_reopens_after_area_resize_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from poor_word.glyphs.stroke_bridge import BridgePlanner

    # Inject a real rasterization failure at the drawing boundary: the 1px white
    # center seals natively, but AREA averaging leaves only a weak broad halo.
    # The observable assertion is rejection of that output, not a mock call.
    def weak_raster(planner, gate, thickness):
        factor = planner.original.shape[0] / 96
        first, last = gate.point(gate.left - 2), gate.point(gate.right + 2)
        a = (round(first[1] * factor), round(first[0] * factor))
        b = (round(last[1] * factor), round(last[0] * factor))
        alpha = np.zeros_like(planner.original)
        cv2.line(alpha, a, b, 64, round(thickness * factor))
        cv2.line(alpha, a, b, 255, 1)
        return alpha

    monkeypatch.setattr(BridgePlanner, "_alpha", weak_raster)
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers(_u(512), "bridge", 3, bridge_mode="close_opening")


def test_tiny_u_opening_cannot_be_promoted_to_a_meaningful_sealed_region() -> None:
    layer = np.zeros((128, 128), np.uint8)
    cv2.polylines(layer, [np.array([[50, 50], [50, 59], [57, 59], [57, 50]])], False, 255, 3)
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers((layer,), "bridge", 5, bridge_mode="close_opening")


def test_remote_thick_stroke_parts_cannot_turn_thin_mouth_banks_into_a_thick_bar() -> None:
    layer = np.zeros((128, 128), np.uint8)
    for a, b in [
        ((28, 20), (31, 64)),
        ((23, 64), (42, 110)),
        ((95, 20), (98, 64)),
        ((83, 64), (102, 110)),
        ((23, 91), (102, 110)),
    ]:
        cv2.rectangle(layer, a, b, 255, -1)
    try:
        result = corrupt_stroke_layers((layer,), "bridge", 0, bridge_mode="close_opening")
    except CorruptionNotApplicable:
        return  # The wide mouth may have no safe thin-bridge solution.
    assert result.metrics["local_stroke_width_96"] <= 4.5
    added = _foreground(result.edited_layers[-1], 128)
    assert cv2.distanceTransform(added.astype(np.uint8), cv2.DIST_L2, 5).max() <= 3


@pytest.mark.parametrize("seed", [1, 3])
def test_sloping_continuing_banks_are_not_mistaken_for_mouth_endpoints(seed: int) -> None:
    layer = np.zeros((128, 128), np.uint8)
    cv2.polylines(layer, [np.array([[20, 20], [50, 100], [78, 100], [108, 20]])], False, 255, 9)
    try:
        result = corrupt_stroke_layers((layer,), "bridge", seed, bridge_mode="close_opening")
    except CorruptionNotApplicable:
        return
    # The original mouth is y=15 at 96px; a bar near y=54 cuts a mid-wall pocket.
    assert result.metrics["gate_start_y_96"] <= 25
    assert result.metrics["gate_end_y_96"] <= 25


@pytest.mark.parametrize("mode,layers", [("close_opening", _u()), ("block_gap", _gap())])
def test_each_mode_is_deterministic_and_npz_reconstructs_independent_unchanged_original_layers(
    mode: str,
    layers: tuple[np.ndarray, ...],
) -> None:
    originals = tuple(x.copy() for x in layers)
    first = corrupt_stroke_layers(layers, "bridge", 3, bridge_mode=mode)
    second = corrupt_stroke_layers(layers, "bridge", 3, bridge_mode=mode)
    np.testing.assert_array_equal(first.image, second.image)
    assert first.metrics == second.metrics
    for actual, expected in zip(layers, originals, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(first.edited_layers[: len(layers)], originals, strict=True):
        np.testing.assert_array_equal(actual, expected)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, layers=np.stack(first.edited_layers))
    buffer.seek(0)
    with np.load(buffer) as loaded:
        np.testing.assert_array_equal(np.maximum.reduce(loaded["layers"]), first.image[:, :, 0])
    assert all(isinstance(value, float) and np.isfinite(value) for value in first.metrics.values())


def test_explicit_mode_never_silently_falls_back_to_the_other_subtype() -> None:
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers(_gap(), "bridge", 2, bridge_mode="close_opening")
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers(_u(), "bridge", 2, bridge_mode="block_gap")


def test_rejects_invalid_mode_and_mode_on_non_bridge() -> None:
    with pytest.raises(ValueError):
        corrupt_stroke_layers(_u(), "bridge", 2, bridge_mode="ordinary")
    with pytest.raises(ValueError):
        corrupt_stroke_layers(_u(), "erase_segment", 2, bridge_mode="close_opening")


@pytest.mark.parametrize(
    "operator,digest",
    [
        ("erase_segment", "670f40b9f39c513596871a8d483e0850d80391bdc14d9d7a9a01ec7714a6d9ae"),
        ("add_stroke", "c21433ccae6800399efaa34fb21800b9bfafe06c594409e2c68a4669b8c02e65"),
        ("break_stroke", "48535fed98d8fb0f8edc6dcf9301f21a9026519cb9acfb98b498362553ea5d2a"),
        ("component_shift", "1f0ffa094597d925122c2f7293cfceb7ec60bdaf579c9e894e1bcbeceea00efe"),
    ],
)
def test_frozen_other_operators_keep_the_pre_change_pixels(operator: str, digest: str) -> None:
    layers = []
    for a, b, width in [
        ((25, 20), (25, 105), 9),
        ((94, 20), (94, 105), 9),
        ((25, 56), (94, 56), 9),
        ((43, 88), (75, 88), 7),
    ]:
        layer = np.zeros((128, 128), np.uint8)
        cv2.line(layer, a, b, 255, width)
        layers.append(layer)
    result = corrupt_stroke_layers(tuple(layers), operator, 17)
    assert hashlib.sha256(result.image.tobytes()).hexdigest() == digest
    assert result.bridge_mode is None
