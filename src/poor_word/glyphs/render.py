from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from poor_word.domain import BoundingBox


@dataclass(frozen=True)
class RenderedGlyph:
    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    bbox: BoundingBox


def _load_font(
    font_path: Path | None, font_size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is None:
        return ImageFont.load_default(size=font_size)
    return ImageFont.truetype(str(font_path), size=font_size)


def render_glyph(
    char: str,
    font_path: Path | None,
    seed: int,
    canvas_size: int = 128,
) -> RenderedGlyph:
    if len(char) != 1:
        raise ValueError("char must contain exactly one Unicode code point")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")

    random = np.random.default_rng(seed)
    font_size = int(random.integers(78, 105))
    font = _load_font(font_path, font_size)
    canvas = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    width = right - left
    height = bottom - top
    x = (canvas_size - width) // 2 - left
    y = (canvas_size - height) // 2 - top
    draw.text((x, y), char, fill=255, font=font)

    grayscale = np.asarray(canvas, dtype=np.uint8).copy()
    mask = (grayscale > 0).astype(np.uint8)
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError(f"font did not render a visible glyph for {char!r}")

    bbox = BoundingBox(
        x0=int(columns.min()),
        y0=int(rows.min()),
        x1=int(columns.max()) + 1,
        y1=int(rows.max()) + 1,
    )
    image = np.repeat(grayscale[:, :, np.newaxis], repeats=3, axis=2)
    return RenderedGlyph(image=image, mask=mask, bbox=bbox)
