from pathlib import Path

import numpy as np

from poor_word.glyphs.render import render_glyph


def test_render_is_seed_deterministic() -> None:
    first = render_glyph("A", None, seed=9)
    second = render_glyph("A", None, seed=9)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert first.bbox == second.bbox
    assert int(first.mask.sum()) > 0


def test_render_real_chinese_font_has_tight_nonempty_bbox() -> None:
    font_path = Path("data/raw/NotoSansCJKsc-Regular.otf")

    rendered = render_glyph("文", font_path, seed=17)

    assert rendered.image.shape == (128, 128, 3)
    assert rendered.mask.shape == (128, 128)
    assert rendered.mask.dtype == np.uint8
    assert rendered.bbox.x1 - rendered.bbox.x0 < 128
    assert rendered.bbox.y1 - rendered.bbox.y0 < 128
    assert int(rendered.mask.sum()) > 0
