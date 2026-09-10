"""Independent layers test geometry, never whether a character is legally written."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from skimage.measure import euler_number, label

from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable
from poor_word.glyphs.stroke_corrupt import corrupt_stroke_layers


def _line(start: tuple[int, int], end: tuple[int, int], width: int = 9) -> NDArray[np.uint8]:
    layer = np.zeros((128, 128), dtype=np.uint8)
    cv2.line(layer, start, end, 255, width)
    return layer


def _layers() -> tuple[NDArray[np.uint8], ...]:
    return (
        _line((25, 20), (25, 105)),
        _line((94, 20), (94, 105)),
        _line((25, 56), (94, 56)),
        _line((43, 88), (75, 88), 7),
    )


def _bridge_layers() -> tuple[NDArray[np.uint8], ...]:
    return (
        _line((28, 25), (28, 100)),
        _line((28, 100), (99, 100)),
        _line((99, 100), (99, 25)),
    )


def _input(layer: NDArray[np.uint8], threshold: int = 8) -> NDArray[np.bool_]:
    return cv2.resize(layer, (96, 96), interpolation=cv2.INTER_AREA) > threshold


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_operator_preserves_inputs_and_returns_exact_auditable_visible_changes(
    operator: str,
) -> None:
    layers = _bridge_layers() if operator == "bridge" else _layers()
    originals = tuple(layer.copy() for layer in layers)
    first = corrupt_stroke_layers(layers, operator, 17)
    second = corrupt_stroke_layers(layers, operator, 17)
    old = np.maximum.reduce(layers)
    new = np.maximum.reduce(first.edited_layers)
    np.testing.assert_array_equal(first.image, np.repeat(new[:, :, None], 3, axis=2))
    np.testing.assert_array_equal(first.changed_mask, (new != old).astype(np.uint8))
    assert first.changed_pixels == np.count_nonzero(new != old)
    assert first.operator == operator
    assert first.image.dtype == first.changed_mask.dtype == np.uint8
    np.testing.assert_array_equal(first.image, second.image)
    assert first.metrics == second.metrics
    assert first.selected_stroke_indices == second.selected_stroke_indices
    for source, original in zip(layers, originals, strict=True):
        np.testing.assert_array_equal(source, original)
    assert 0.025 <= first.metrics["changed_fraction"] <= 0.60
    assert np.count_nonzero(_input(old) != _input(new)) >= 8
    a, b = [cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA) for x in (old, new)]
    assert np.count_nonzero(np.abs(a.astype(float) - b) >= 32) >= 8
    assert np.count_nonzero(cv2.Canny(a, 50, 150) != cv2.Canny(b, 50, 150)) >= 8
    assert all(isinstance(value, float) and np.isfinite(value) for value in first.metrics.values())


def test_erasing_whole_stroke_redraws_crossings_from_surviving_layers() -> None:
    layers = _layers()
    result = corrupt_stroke_layers(layers, "erase_segment", 4)
    assert len(result.selected_stroke_indices) == 1
    selected = result.selected_stroke_indices[0]
    remaining = tuple(layer for index, layer in enumerate(layers) if index != selected)
    assert len(result.edited_layers) == len(remaining)
    for actual, expected in zip(result.edited_layers, remaining, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(result.image[:, :, 0], np.maximum.reduce(remaining))
    overlap = (layers[selected] > 0) & (np.maximum.reduce(remaining) > 0)
    assert overlap.any()
    assert np.all(result.image[:, :, 0][overlap] == 255)


def test_addition_adds_two_to_four_complete_copies_with_real_new_ink_and_contact() -> None:
    layers = _layers()
    result = corrupt_stroke_layers(layers, "add_stroke", 17)
    added = result.edited_layers[len(layers) :]
    assert 2 <= len(added) <= 4
    for before, after in zip(layers, result.edited_layers[: len(layers)], strict=True):
        np.testing.assert_array_equal(before, after)
    original_core = _input(np.maximum.reduce(layers), 127)
    for index, layer in enumerate(added):
        core = _input(layer, 127)
        others = np.maximum.reduce(layers + added[:index] + added[index + 1 :])
        assert np.count_nonzero(core & ~_input(others)) >= original_core.sum() * 0.025
        assert np.count_nonzero(core & original_core) >= core.sum() * 0.03
        assert np.count_nonzero(core & ~original_core) >= core.sum() * 0.25
        assert label(core, connectivity=2).max() == 1


@pytest.mark.parametrize("seed", [0, 4, 17, 39])
def test_shift_is_large_and_collides_at_actual_input_resolution(seed: int) -> None:
    layers = _layers()
    result = corrupt_stroke_layers(layers, "component_shift", seed)
    assert len(result.selected_stroke_indices) == 1
    selected = result.selected_stroke_indices[0]
    for index, original in enumerate(layers):
        if index != selected:
            np.testing.assert_array_equal(result.edited_layers[index], original)
    moved = result.edited_layers[selected]
    rest = np.maximum.reduce(tuple(x for i, x in enumerate(layers) if i != selected))
    moved_core, rest_core = _input(moved, 127), _input(rest, 127)
    overlap = np.count_nonzero(moved_core & rest_core) / moved_core.sum()
    assert 0.15 <= overlap <= 0.45
    assert result.metrics["input_overlap_fraction"] == pytest.approx(overlap)
    assert result.metrics["shift_distance"] >= max(
        result.metrics["local_stroke_width"] * 2, result.metrics["glyph_scale"] * 0.15
    )
    assert np.count_nonzero(moved) >= np.count_nonzero(layers[selected]) * 0.98


def test_single_stroke_cannot_be_moved_or_deleted_as_an_entire_character() -> None:
    layers = (_line((20, 64), (105, 64)),)
    for operator in ("component_shift", "erase_segment", "break_stroke"):
        with pytest.raises(CorruptionNotApplicable):
            corrupt_stroke_layers(layers, operator, 3)


def test_break_targets_actual_junction_and_changes_only_one_stroke_layer() -> None:
    layers = (_line((20, 64), (108, 64)), _line((64, 20), (64, 108)))
    result = corrupt_stroke_layers(layers, "break_stroke", 17)
    assert len(result.selected_stroke_indices) == 1
    index = result.selected_stroke_indices[0]
    np.testing.assert_array_equal(result.edited_layers[1 - index], layers[1 - index])
    cut = (layers[index] > 0) & (result.edited_layers[index] == 0)
    junction = (layers[0] > 0) & (layers[1] > 0)
    assert np.count_nonzero(cut & junction) >= junction.sum() * 0.75
    before = _input(np.maximum.reduce(layers))
    after = _input(result.image[:, :, 0])
    assert label(before, connectivity=2).max() == 1
    assert label(after, connectivity=2).max() >= 2
    assert after.sum() >= before.sum() * 0.4


def test_bridge_has_an_auditable_added_layer_and_changes_input_topology() -> None:
    layers = _bridge_layers()
    result = corrupt_stroke_layers(layers, "bridge", 17)
    assert len(result.edited_layers) > len(layers)
    before = _input(np.maximum.reduce(layers))
    after = _input(result.image[:, :, 0])
    assert label(after, connectivity=2).max() == label(before, connectivity=2).max()
    assert euler_number(after, connectivity=2) < euler_number(before, connectivity=2)


@pytest.mark.parametrize(
    "layers",
    [
        (),
        (np.zeros((128, 128), dtype=np.float32),),
        (np.zeros((20, 20), dtype=np.uint8),),
        (np.zeros((64, 65), dtype=np.uint8),),
        (np.zeros((64, 64), dtype=np.uint8), np.zeros((128, 128), dtype=np.uint8)),
    ],
)
def test_rejects_invalid_layer_inputs(layers: tuple[NDArray[np.uint8], ...]) -> None:
    with pytest.raises(ValueError):
        corrupt_stroke_layers(layers, "erase_segment", 0)


def test_blank_layers_and_unknown_operator_fail_explicitly() -> None:
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers((np.zeros((64, 64), dtype=np.uint8),), "add_stroke", 0)
    with pytest.raises(ValueError):
        corrupt_stroke_layers(_layers(), "made_up", 0)


def test_break_does_not_report_a_junction_edit_when_most_contact_survives() -> None:
    layers = []
    for points, width in [
        ([[98, 65], [22, 25]], 6),
        ([[92, 81], [35, 32], [45, 85], [89, 22]], 11),
        ([[56, 41], [64, 61]], 10),
    ]:
        layer = np.zeros((128, 128), np.uint8)
        cv2.polylines(layer, [np.array(points, np.int32)], False, 255, width)
        layers.append(layer)
    try:
        result = corrupt_stroke_layers(tuple(layers), "break_stroke", 0)
    except CorruptionNotApplicable:
        return  # A complex junction is allowed to be conservatively skipped.
    selected = result.selected_stroke_indices[0]
    rest = np.maximum.reduce([x for i, x in enumerate(layers) if i != selected])
    contact = (layers[selected] >= 128) & (
        cv2.dilate((rest >= 128).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    )
    count, regions = cv2.connectedComponents(contact.astype(np.uint8), connectivity=8)
    removal_fractions = [
        np.count_nonzero((regions == i) & (result.edited_layers[selected] == 0))
        / np.count_nonzero(regions == i)
        for i in range(1, count)
    ]
    assert max(removal_fractions) >= 0.75


def test_break_can_open_main_closed_loop_without_disconnect_in_component_count() -> None:
    layers = (
        _line((25, 25), (100, 25)),
        _line((100, 25), (100, 100)),
        _line((100, 100), (25, 100)),
        _line((25, 100), (25, 25)),
    )
    result = corrupt_stroke_layers(layers, "break_stroke", 0)
    before, after = _input(np.maximum.reduce(layers)), _input(result.image[:, :, 0])
    assert label(before, connectivity=2).max() == label(after, connectivity=2).max() == 1
    assert euler_number(before, connectivity=2) == 0
    assert euler_number(after, connectivity=2) == 1


def test_shift_skips_when_border_and_width_leave_only_weak_or_clipped_moves() -> None:
    first = np.zeros((32, 32), np.uint8)
    second = first.copy()
    first[1:31, 1:17] = 255
    second[1:31, 15:31] = 255
    with pytest.raises(CorruptionNotApplicable):
        corrupt_stroke_layers((first, second), "component_shift", 17)


@pytest.mark.parametrize("size", [32, 64, 192])
@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_resized_layers_keep_visible_training_inputs(size: int, operator: str) -> None:
    original = _bridge_layers() if operator == "bridge" else _layers()
    layers = tuple(cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA) for x in original)
    if operator == "bridge" and size == 32:
        # Coarse original banks do not support this wide mouth safely.
        with pytest.raises(CorruptionNotApplicable):
            corrupt_stroke_layers(layers, operator, 0)
        return
    result = corrupt_stroke_layers(layers, operator, 0)
    before, after = _input(np.maximum.reduce(layers)), _input(result.image[:, :, 0])
    assert np.count_nonzero(before != after) >= 8


def test_local_rng_does_not_consume_global_numpy_state() -> None:
    state = np.random.get_state()
    corrupt_stroke_layers(_layers(), "add_stroke", 17)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
