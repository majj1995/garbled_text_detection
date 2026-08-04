import numpy as np
import pytest
from skimage.measure import label

from poor_word.domain import BoundingBox
from poor_word.glyphs.corrupt import OPERATORS, corrupt_glyph
from poor_word.glyphs.render import RenderedGlyph


@pytest.fixture
def rendered_glyph() -> RenderedGlyph:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[6:26, 13:18] = 1
    mask[13:18, 6:26] = 1
    mask[2:5, 2:5] = 1
    image = np.repeat((mask * 255)[:, :, None], 3, axis=2)
    return RenderedGlyph(
        image=image,
        mask=mask,
        bbox=BoundingBox(x0=2, y0=2, x1=26, y1=26),
    )


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_corruption_is_deterministic_and_changes_foreground(
    rendered_glyph: RenderedGlyph, operator: str
) -> None:
    first = corrupt_glyph(rendered_glyph, operator=operator, seed=17)
    second = corrupt_glyph(rendered_glyph, operator=operator, seed=17)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.changed_mask, second.changed_mask)
    assert 8 <= first.changed_pixels <= int(rendered_glyph.mask.sum() * 0.35)
    assert first.operator == operator


def test_bridge_does_not_increase_component_count(rendered_glyph: RenderedGlyph) -> None:
    result = corrupt_glyph(rendered_glyph, operator="bridge", seed=17)

    assert label(result.mask, connectivity=2).max() <= label(
        rendered_glyph.mask, connectivity=2
    ).max()


def test_component_shift_preserves_foreground_area(rendered_glyph: RenderedGlyph) -> None:
    result = corrupt_glyph(rendered_glyph, operator="component_shift", seed=17)

    original_area = int(np.count_nonzero(rendered_glyph.mask))
    shifted_area = int(np.count_nonzero(result.mask))
    assert abs(shifted_area - original_area) / original_area <= 0.02


@pytest.mark.parametrize("operator", ["erase_segment", "break_stroke"])
def test_erase_operators_reduce_foreground(
    rendered_glyph: RenderedGlyph, operator: str
) -> None:
    result = corrupt_glyph(rendered_glyph, operator=operator, seed=17)

    assert np.count_nonzero(result.mask) < np.count_nonzero(rendered_glyph.mask)
