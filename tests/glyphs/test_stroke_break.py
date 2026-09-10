"""BREAK must split independent source ink, not merely unjoin legal contacts."""

import hashlib

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from poor_word.glyphs.corrupt import CorruptionNotApplicable
from poor_word.glyphs.stroke_break import _parts, _visible_gap
from poor_word.glyphs.stroke_corrupt import corrupt_stroke_layers


def _stroke(points: list[tuple[int, int]], width: int = 9) -> NDArray[np.uint8]:
    layer = np.zeros((128, 128), np.uint8)
    cv2.polylines(layer, [np.asarray(points, np.int32)], False, 255, width)
    return layer


def _box(character: str) -> tuple[NDArray[np.uint8], ...]:
    # Independent handwritten strokes: left vertical, horizontal-fold, bottom.
    layers = [
        _stroke([(27, 25), (27, 103)]),
        _stroke([(27, 25), (101, 25), (101, 103)]),
        _stroke([(27, 103), (101, 103)]),
    ]
    if character in ("目", "田"):
        layers.append(_stroke([(27, 64), (101, 64)]))
    if character == "目":
        layers.append(_stroke([(27, 45), (101, 45)]))
    if character == "田":
        layers.append(_stroke([(64, 25), (64, 103)]))
    return tuple(layers)


def _part_areas(layer: NDArray[np.uint8], threshold: int) -> list[int]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (layer >= threshold).astype(np.uint8), connectivity=8
    )
    return sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)


@pytest.mark.parametrize("character", ["口", "目", "田"])
@pytest.mark.parametrize("seed", [0, 1, 7, 17])
def test_box_components_split_selected_stroke_into_two_substantial_parts(
    character: str, seed: int
) -> None:
    # Removing the source-split gate would accept a terminal-only corner opening.
    layers = _box(character)
    result = corrupt_stroke_layers(layers, "break_stroke", seed)
    index = result.selected_stroke_indices[0]
    for size in (128, 96):
        before, after = [
            cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
            for x in (layers[index], result.edited_layers[index])
        ]
        for threshold in (9, 128):
            assert len(_part_areas(before, threshold)) == 1
            parts = _part_areas(after, threshold)
            assert len(parts) == 2
            assert min(parts) >= np.count_nonzero(before >= threshold) * 0.12
    for other in range(len(layers)):
        if other != index:
            np.testing.assert_array_equal(result.edited_layers[other], layers[other])


@pytest.mark.parametrize("points", [[(20, 64), (108, 64)], [(20, 25), (103, 25), (103, 108)]])
@pytest.mark.parametrize("size", [32, 64, 128, 192])
@pytest.mark.parametrize("turns,offset", [(0, 0), (1, 5), (2, -5), (3, 0)])
def test_long_and_folded_single_strokes_support_interior_breaks_under_transforms(
    points: list[tuple[int, int]], size: int, turns: int, offset: int
) -> None:
    # A contact-only planner incorrectly rejects these independent stroke bodies.
    layer = np.rot90(_stroke(points), turns).copy()
    layer = cv2.warpAffine(layer, np.array([[1.0, 0.0, offset], [0.0, 1.0, offset]]), (128, 128))
    layer = cv2.resize(layer, (size, size), interpolation=cv2.INTER_AREA)
    result = corrupt_stroke_layers((layer,), "break_stroke", 17)
    after = cv2.resize(result.edited_layers[0], (96, 96), interpolation=cv2.INTER_AREA)
    for threshold in (9, 128):
        areas = _part_areas(after, threshold)
        assert len(areas) == 2
        assert min(areas) >= 40


def test_identical_covering_stroke_leaves_no_visible_break_and_is_skipped() -> None:
    # A selected-layer-only gate would accept a gap hidden by the other layer.
    layer = _stroke([(20, 64), (108, 64)])
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers((layer, layer.copy()), "break_stroke", 17)


def test_short_terminal_contact_cannot_qualify_as_two_substantial_pieces() -> None:
    layers = (_stroke([(50, 55), (62, 55)]), _stroke([(62, 55), (62, 67)]))
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers(layers, "break_stroke", 17)


def test_tiny_preexisting_island_cannot_qualify_a_shortened_main_stroke() -> None:
    layer = _stroke([(50, 55), (62, 55)])
    layer[35, 35] = 255
    assert _part_areas(layer, 128)[-1] == 1
    assert len(_part_areas(layer, 128)) == 2
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers((layer,), "break_stroke", 17)


@pytest.mark.parametrize("threshold", [9, 128])
def test_new_one_pixel_secondary_fragment_does_not_pass_the_split_gate(threshold: int) -> None:
    # The old 瞬 mechanism: raw 2CC alone is not evidence of two stroke bodies.
    before = np.zeros((96, 96), np.uint8)
    before[43:52, 15:81] = 255
    after = before.copy()
    after[:, 16:29] = 0
    after[44:52, 15] = 0
    assert len(_part_areas(before, threshold)) == 1
    assert _part_areas(after, threshold) == [468, 1]
    assert _parts(before, after, 9.0, 1.0, threshold) is None


@pytest.mark.parametrize("threshold", [9, 128])
def test_terminal_shortening_is_not_an_independent_source_split(threshold: int) -> None:
    before = np.zeros((96, 96), np.uint8)
    before[43:52, 15:81] = 255
    after = before.copy()
    after[:, 15:29] = 0
    assert _part_areas(after, threshold) == [468]
    assert _parts(before, after, 9.0, 1.0, threshold) is None


@pytest.mark.parametrize("threshold", [9, 128])
def test_covering_ink_cannot_hide_a_gap_between_two_substantial_source_parts(
    threshold: int,
) -> None:
    # Literal bank labels isolate the composition gate; no planner helper builds expectations.
    parts = np.zeros((96, 96), np.int32)
    parts[43:52, 15:42] = 1
    parts[43:52, 54:81] = 2
    composite = (parts > 0).astype(np.uint8) * 255
    roi = np.zeros((96, 96), np.bool_)
    roi[30:66, 30:66] = True
    assert _visible_gap(composite, parts, roi, 9.0, threshold) == 12.0
    composite[43:52, 42:54] = 255
    assert _visible_gap(composite, parts, roi, 9.0, threshold) is None


def test_substantial_area_without_substantial_length_is_rejected() -> None:
    before = np.zeros((96, 96), np.uint8)
    before[40:55, 15:81] = 255
    after = before.copy()
    after[:, 30:43] = 0
    assert _part_areas(after, 128) == [570, 225]
    assert _parts(before, after, 15.0, 1.0, 128) is None


def test_faint_cover_is_still_a_closed_gap_at_the_foreground_threshold() -> None:
    parts = np.zeros((96, 96), np.int32)
    parts[43:52, 15:42] = 1
    parts[43:52, 54:81] = 2
    composite = (parts > 0).astype(np.uint8) * 255
    composite[43:52, 42:54] = 32
    roi = np.zeros((96, 96), np.bool_)
    roi[30:66, 30:66] = True
    assert _visible_gap(composite, parts, roi, 9.0, 128) == 12.0
    assert _visible_gap(composite, parts, roi, 9.0, 9) is None


@pytest.mark.parametrize("threshold", [9, 128])
def test_detached_ink_between_banks_cannot_count_as_clear_gap(threshold: int) -> None:
    parts = np.zeros((96, 96), np.int32)
    parts[43:52, 15:42] = 1
    parts[43:52, 54:81] = 2
    composite = (parts > 0).astype(np.uint8) * 255
    composite[43:52, 43:53] = 255
    roi = np.zeros((96, 96), np.bool_)
    roi[30:66, 30:66] = True
    # Only one empty column remains on each side of the detached middle ink.
    assert not composite[47, 42]
    assert not composite[47, 53]
    assert np.all(composite[47, 43:53])
    assert _visible_gap(composite, parts, roi, 9.0, threshold) is None


def test_public_break_does_not_report_bank_distance_as_clearance_over_detached_ink() -> None:
    source = np.zeros((96, 96), np.uint8)
    source[43:52, 15:81] = 255
    rest = np.zeros_like(source)
    rest[43:52, 46:59] = 255
    result = corrupt_stroke_layers((source, rest), "break_stroke", 0)
    assert result.selected_stroke_indices == (0,)
    # A different interior candidate can be valid; examine the actual accepted gap.
    for threshold in (9, 128):
        selected_row = result.edited_layers[0][47] >= threshold
        starts = np.flatnonzero(np.diff(selected_row.astype(np.int32)) == 1) + 1
        ends = np.flatnonzero(np.diff(selected_row.astype(np.int32)) == -1)
        assert len(starts) == len(ends) == 2
        interval = result.image[47, ends[0] + 1 : starts[1], 0]
        blank = np.pad((interval < threshold).astype(np.int32), (1, 1))
        blank_lengths = np.flatnonzero(np.diff(blank) == -1) - np.flatnonzero(np.diff(blank) == 1)
        assert len(blank_lengths) == 1
        assert blank_lengths[0] >= 3
        assert result.metrics[f"break_visible_gap_96_t{threshold}"] == blank_lengths[0]


def test_measurements_describe_actual_96_pixel_parts_and_visible_empty_columns() -> None:
    layer = _stroke([(20, 64), (108, 64)])
    result = corrupt_stroke_layers((layer,), "break_stroke", 17)
    small = cv2.resize(result.edited_layers[0], (96, 96), interpolation=cv2.INTER_AREA)
    for threshold in (9, 128):
        areas = _part_areas(small, threshold)
        assert result.metrics[f"break_before_parts_96_t{threshold}"] == 1
        assert result.metrics[f"break_after_parts_96_t{threshold}"] == 2
        assert sorted(
            result.metrics[f"break_part_{part}_area_96_t{threshold}"] for part in (1, 2)
        ) == sorted(areas)
        row = small[48] >= threshold
        starts = np.flatnonzero(np.diff(row.astype(np.int32)) == 1) + 1
        ends = np.flatnonzero(np.diff(row.astype(np.int32)) == -1)
        assert len(starts) == len(ends) == 2
        blank_columns = int(starts[1] - ends[0] - 1)
        assert result.metrics[f"break_visible_gap_96_t{threshold}"] == blank_columns
        assert blank_columns >= 3
    native_x = round((result.metrics["break_center_x_96"] + 0.5) * 128 / 96 - 0.5)
    native_y = round((result.metrics["break_center_y_96"] + 0.5) * 128 / 96 - 0.5)
    distance = cv2.distanceTransform((layer >= 128).astype(np.uint8), cv2.DIST_L2, 5)
    assert result.metrics["local_stroke_width"] == distance[native_y, native_x] * 2
    assert result.metrics["local_stroke_width_96"] == result.metrics["local_stroke_width"] * 0.75


@pytest.mark.parametrize(
    "operator,digest",
    [
        ("erase_segment", "670f40b9f39c513596871a8d483e0850d80391bdc14d9d7a9a01ec7714a6d9ae"),
        ("component_shift", "1f0ffa094597d925122c2f7293cfceb7ec60bdaf579c9e894e1bcbeceea00efe"),
        ("add_stroke", "c21433ccae6800399efaa34fb21800b9bfafe06c594409e2c68a4669b8c02e65"),
        ("bridge", "e4409117ab1a002d914789c5d0beb314fd278d546225201c2faf1bd49bd8641a"),
    ],
)
def test_non_break_operators_keep_baseline_pixels(operator: str, digest: str) -> None:
    if operator == "bridge":
        points = [[(28, 25), (28, 100)], [(28, 100), (99, 100)], [(99, 100), (99, 25)]]
        layers = tuple(_stroke(p) for p in points)
    else:
        layers = (
            _stroke([(25, 20), (25, 105)]),
            _stroke([(94, 20), (94, 105)]),
            _stroke([(25, 56), (94, 56)]),
            _stroke([(43, 88), (75, 88)], 7),
        )
    result = corrupt_stroke_layers(layers, operator, 17)
    assert hashlib.sha256(result.image.tobytes()).hexdigest() == digest
