"""Geometry contracts: these tests do not claim synthetic glyphs are illegal Chinese."""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from skimage.measure import euler_number, label

from poor_word.domain import BoundingBox
from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable
from poor_word.glyphs.corrupt_v2 import Severity, V2CorruptionResult, corrupt_glyph_v2
from poor_word.glyphs.render import RenderedGlyph, render_glyph


def _run(
    glyph: RenderedGlyph, operator: str, seed: int = 17, severity: str = "medium"
) -> V2CorruptionResult:
    return corrupt_glyph_v2(glyph, operator, seed, cast(Severity, severity))


def _glyph(mask: NDArray[np.uint8]) -> RenderedGlyph:
    rows, columns = np.nonzero(mask)
    bbox = BoundingBox(
        x0=int(columns.min()) if len(rows) else 0,
        y0=int(rows.min()) if len(rows) else 0,
        x1=int(columns.max()) + 1 if len(rows) else mask.shape[1],
        y1=int(rows.max()) + 1 if len(rows) else mask.shape[0],
    )
    return RenderedGlyph(np.repeat((mask * 255)[:, :, None], 3, axis=2), mask, bbox)


def _bar(width: int = 12, size: int = 128) -> RenderedGlyph:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 2 - width // 2 : size // 2 + (width + 1) // 2, size // 5 : size * 4 // 5] = 1
    return _glyph(mask)


def _parts(size: int = 128) -> RenderedGlyph:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[22:106, 28:40] = 1
    mask[22:34, 28:100] = 1
    mask[68:80, 60:104] = 1
    mask[105:109, 103:107] = 1  # Tiny dot must not be the only shifted choice.
    if size != 128:
        mask = np.asarray(Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST))
    return _glyph(mask)


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_each_operator_returns_exact_rgb_changes_without_mutating_input(operator: str) -> None:
    glyph = _parts()
    original_image, original_mask = glyph.image.copy(), glyph.mask.copy()
    first = _run(glyph, operator)
    second = _run(glyph, operator)
    actual = np.any(first.image != original_image, axis=2)

    assert first.image.dtype == np.uint8
    assert first.mask.dtype == first.changed_mask.dtype == np.uint8
    assert set(np.unique(first.mask)) <= {0, 1}
    assert np.array_equal(first.changed_mask, actual.astype(np.uint8))
    assert first.changed_pixels == int(actual.sum()) > 0
    assert np.array_equal(first.mask, np.any(first.image > 0, axis=2).astype(np.uint8))
    assert np.array_equal(first.image, second.image)
    assert first.metrics == second.metrics
    assert np.array_equal(glyph.image, original_image)
    assert np.array_equal(glyph.mask, original_mask)
    assert first.operator == operator and first.severity == "medium"
    assert first.metrics["local_stroke_width"] > 0
    assert 0.025 <= first.metrics["changed_fraction"] <= 0.55
    assert first.metrics["input_changed_fraction"] >= 0.02
    assert all(isinstance(value, float) and np.isfinite(value) for value in first.metrics.values())


@pytest.mark.parametrize("width", [4, 12, 20])
@pytest.mark.parametrize("operator", ["break_stroke", "erase_segment"])
def test_removal_cuts_through_entire_stroke_instead_of_making_a_hole(
    width: int, operator: str
) -> None:
    glyph = _bar(width)
    result = _run(glyph, operator)
    assert label(result.mask, connectivity=2).max() == 2
    assert np.count_nonzero(result.mask) < np.count_nonzero(glyph.mask)
    assert not np.any(result.image > glyph.image)
    erased_columns = np.flatnonzero(np.any(glyph.mask, axis=0) & ~np.any(result.mask, axis=0))
    assert len(erased_columns) >= width * (0.7 if operator == "break_stroke" else 1.7)
    assert np.all(np.diff(erased_columns) == 1)
    assert width * 0.65 <= result.metrics["local_stroke_width"] <= width * 1.3


def test_erase_removes_longer_segment_than_break() -> None:
    glyph = _bar(8)
    broken = _run(glyph, "break_stroke")
    erased = _run(glyph, "erase_segment")
    assert erased.changed_pixels > broken.changed_pixels * 1.5


def test_visibility_floor_does_not_collapse_erase_and_break_into_the_same_edit() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    for row in range(18, 110, 12):
        mask[row : row + 4, 18:110] = 1
    glyph = _glyph(mask)
    broken = _run(glyph, "break_stroke")
    erased = _run(glyph, "erase_segment")
    assert erased.changed_pixels > broken.changed_pixels * 1.4


@pytest.mark.parametrize("operator", ["break_stroke", "erase_segment", "add_stroke"])
def test_strong_severity_increases_geometric_effect_on_same_simple_stroke(operator: str) -> None:
    glyph = _bar(8)
    medium = _run(glyph, operator, severity="medium")
    strong = _run(glyph, operator, severity="strong")
    assert strong.changed_pixels > medium.changed_pixels * 1.15


@pytest.mark.parametrize("operator", sorted(OPERATORS))
@pytest.mark.parametrize("size", [64, 256])
def test_changes_remain_visible_after_both_model_input_resize_paths(
    operator: str, size: int
) -> None:
    glyph = _parts(size)
    result = _run(glyph, operator)
    area_original = cv2.resize(glyph.image, (96, 96), interpolation=cv2.INTER_AREA)
    area_result = cv2.resize(result.image, (96, 96), interpolation=cv2.INTER_AREA)
    linear_original = np.asarray(
        Image.fromarray(glyph.image).resize((96, 96), Image.Resampling.BILINEAR)
    )
    linear_result = np.asarray(
        Image.fromarray(result.image).resize((96, 96), Image.Resampling.BILINEAR)
    )
    for original, candidate in [(area_original, area_result), (linear_original, linear_result)]:
        difference = np.max(np.abs(candidate.astype(float) - original), axis=2)
        assert np.count_nonzero(difference >= 32) >= 8


def test_addition_matches_thick_local_stroke_scale() -> None:
    result = _run(_bar(16), "add_stroke")
    assert result.metrics["local_stroke_width"] >= 12
    assert result.metrics["segment_width"] >= 10
    added = result.changed_mask.astype(bool)
    assert label(added, connectivity=2).max() <= 2
    assert result.changed_pixels >= 100


def test_bridge_connects_separated_strokes_with_scale_matched_width() -> None:
    glyph = _parts()
    result = _run(glyph, "bridge")
    assert label(result.mask, connectivity=2).max() < label(glyph.mask, connectivity=2).max()
    assert result.metrics["segment_width"] >= 6
    assert not np.any(result.image < glyph.image)


def test_bridge_topology_survives_model_resize_not_only_pixel_changes() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[28:100, 28:40] = mask[28:100, 88:100] = 1
    mask[28:40, 28:100] = mask[88:100, 28:100] = 1
    glyph = _glyph(mask)
    try:
        result = _run(glyph, "bridge", 4)
    except CorruptionNotApplicable:
        # A bounded skip is valid; a tiny closed corner that vanishes is not.
        return
    for mode in [cv2.INTER_AREA, cv2.INTER_LINEAR]:
        original = cv2.resize(glyph.image, (96, 96), interpolation=mode).max(axis=2) >= 128
        candidate = cv2.resize(result.image, (96, 96), interpolation=mode).max(axis=2) >= 128
        assert euler_number(candidate, connectivity=2) < euler_number(original, connectivity=2)
    original = (
        np.asarray(Image.fromarray(glyph.image).resize((96, 96), Image.Resampling.BILINEAR)).max(
            axis=2
        )
        >= 128
    )
    candidate = (
        np.asarray(Image.fromarray(result.image).resize((96, 96), Image.Resampling.BILINEAR)).max(
            axis=2
        )
        >= 128
    )
    assert euler_number(candidate, connectivity=2) < euler_number(original, connectivity=2)


def _assert_bridge_changes_training_mask(glyph: RenderedGlyph, seed: int) -> None:
    try:
        result = _run(glyph, "bridge", seed)
    except CorruptionNotApplicable:
        return  # Rejecting every candidate is preferable to a vanished bridge.
    original, candidate = [
        cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            (96, 96),
            interpolation=cv2.INTER_AREA,
        )
        > 8
        for image in (glyph.image, result.image)
    ]
    assert label(candidate, connectivity=2).max() < label(
        original, connectivity=2
    ).max() or euler_number(candidate, connectivity=2) < euler_number(original, connectivity=2)


def test_bridge_topology_survives_actual_training_mask_threshold() -> None:
    # Subpixel, antialiased grid: a hole in the >=128 core can be closed by
    # its faint edge at the training pipeline's >8 foreground threshold.
    canvas = Image.new("L", (512, 512))
    draw = ImageDraw.Draw(canvas)
    for rectangle in (
        (98, 98, 129, 413),
        (384, 98, 415, 413),
        (242, 98, 273, 413),
        (98, 98, 415, 129),
        (98, 384, 415, 415),
        (98, 242, 415, 273),
    ):
        draw.rectangle(rectangle, fill=255)
    gray = np.asarray(canvas.resize((128, 128), Image.Resampling.BOX))
    mask = (gray > 0).astype(np.uint8)
    glyph = RenderedGlyph(np.repeat(gray[:, :, None], 3, axis=2), mask, _glyph(mask).bbox)
    _assert_bridge_changes_training_mask(glyph, 13)


def test_tian_bridge_does_not_validate_a_core_only_corner_hole() -> None:
    font = Path(__file__).resolve().parents[2] / "data/raw/NotoSansCJKsc-Regular.otf"
    if not font.is_file():
        pytest.skip("optional locally downloaded CJK font is unavailable")
    _assert_bridge_changes_training_mask(render_glyph("田", font, 17), 3)


def test_component_shift_ignores_tiny_dot_and_preserves_stationary_pixels() -> None:
    glyph = _parts()
    result = _run(glyph, "component_shift")
    assert np.array_equal(result.image[105:109, 103:107], glyph.image[105:109, 103:107])
    assert result.metrics["moved_component_fraction"] >= 0.1
    assert result.metrics["moved_component_fraction"] <= 0.75
    assert 0 <= result.metrics["overlap_fraction"] <= 0.4
    assert np.count_nonzero(result.image == glyph.image) > glyph.image.size * 0.9


def test_component_shift_can_overlap_another_component() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[24:104, 30:46] = 1
    mask[44:88, 52:64] = 1
    glyph = _glyph(mask)
    results = []
    for seed in range(24):
        try:
            results.append(_run(glyph, "component_shift", seed, "strong"))
        except CorruptionNotApplicable:
            pass
    assert results
    assert any(result.metrics["overlap_fraction"] > 0 for result in results)
    assert any(result.metrics["overlap_fraction"] == 0 for result in results)


def test_local_width_distinguishes_thin_horizontal_and_thick_vertical_segments() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[60:64, 18:110] = 1
    mask[18:110, 58:72] = 1
    glyph = _glyph(mask)
    widths = [_run(glyph, "break_stroke", seed).metrics["local_stroke_width"] for seed in range(16)]
    assert min(widths) < 7
    assert max(widths) > 10


def test_antialiased_input_keeps_edges_and_reports_rgb_changes_exactly() -> None:
    glyph = _bar(10)
    image = glyph.image.copy()
    image[58, 25:102] = 80
    image[68, 25:102] = 80
    softened = RenderedGlyph(image, np.any(image > 0, axis=2).astype(np.uint8), glyph.bbox)
    result = _run(softened, "add_stroke")
    actual = np.any(result.image != image, axis=2)
    assert np.array_equal(result.changed_mask, actual)
    assert np.any((result.image[:, :, 0] > 0) & (result.image[:, :, 0] < 255) & actual)
    assert np.array_equal(result.image[~actual], image[~actual])


@pytest.mark.parametrize("operator", sorted(OPERATORS))
@pytest.mark.parametrize("kind", ["empty", "full", "tiny"])
def test_degenerate_input_is_not_applicable(operator: str, kind: str) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    if kind == "full":
        mask[:] = 1
    elif kind == "tiny":
        mask[15, 15] = 1
    with pytest.raises(CorruptionNotApplicable):
        _run(_glyph(mask), operator)


def test_component_shift_does_not_move_a_whole_single_component() -> None:
    with pytest.raises(CorruptionNotApplicable):
        _run(_bar(), "component_shift")


@pytest.mark.parametrize(
    ("operator", "seed", "severity"),
    [("invalid", 1, "medium"), ("add_stroke", -1, "medium"), ("add_stroke", 1, "low")],
)
def test_invalid_configuration_is_rejected(operator: str, seed: int, severity: str) -> None:
    with pytest.raises(ValueError):
        _run(_bar(), operator, seed, severity)


def test_inconsistent_image_and_mask_are_rejected() -> None:
    glyph = _bar()
    bad = RenderedGlyph(glyph.image, np.zeros_like(glyph.mask), glyph.bbox)
    with pytest.raises(ValueError):
        _run(bad, "add_stroke")
