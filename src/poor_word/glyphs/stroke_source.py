"""Load Make Me a Hanzi outlines and rasterize intact, independently owned strokes.

The source uses a 1024-unit, Y-up coordinate system (SVG view Y = 900 - Y).
All strokes share one outline-bounds transform; medians never determine the ink.
Only path-data strings are parsed, never XML or external SVG resources.
Supported commands are M/L/H/V/C/S/Q/T/Z and their relative lowercase forms;
elliptical arcs (A/a) are explicitly unsupported, and absent from the source data.
"""

# The two drawing libraries expose an untyped pen protocol.
# mypy: disallow-untyped-calls=False, disallow-subclassing-any=False

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aggdraw  # type: ignore[import-not-found]
import numpy as np
from fontTools.pens.basePen import BasePen  # type: ignore[import-untyped]
from fontTools.pens.boundsPen import BoundsPen  # type: ignore[import-untyped]
from fontTools.pens.recordingPen import RecordingPen  # type: ignore[import-untyped]
from fontTools.pens.transformPen import TransformPen  # type: ignore[import-untyped]
from fontTools.svgLib.path import parse_path  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image

_NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(rf"[MmLlHhVvCcSsQqTtAaZz]|{_NUMBER}")
_SEPARATOR = re.compile(r"[\s,]*")
_SUPERSAMPLING = 4
_GLYPH_SIZE = 96.0


@dataclass(frozen=True)
class StrokeRecord:
    character: str
    strokes: tuple[str, ...]
    medians: tuple[tuple[tuple[float, float], ...], ...]


def _coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("coordinates must be numbers")
    number = float(value)
    # Allow overshoot beyond the nominal em square, including negative descenders.
    if not math.isfinite(number) or not -1024 <= number <= 2048:
        raise ValueError("coordinates must be finite and within [-1024, 2048]")
    return number


def _parse_outline(path: str) -> tuple[Any, tuple[float, float, float, float]]:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("stroke path must be a nonempty SVG path-data string")
    end = 0
    normalized_tokens = []
    for match in _TOKEN.finditer(path):
        if _SEPARATOR.fullmatch(path[end : match.start()]) is None:
            raise ValueError("unsupported SVG path syntax")
        token = match.group()
        if token in ("A", "a"):
            raise ValueError("SVG arc commands are not supported")
        if len(token) > 1 or not token.isalpha():
            token = repr(_coordinate(float(token)))
        normalized_tokens.append(token)
        end = match.end()
    if _SEPARATOR.fullmatch(path[end:]) is None or path.lstrip()[0] not in "Mm":
        raise ValueError("unsupported SVG path syntax")
    pen = RecordingPen()
    try:
        # fontTools tokenizes leading-zero numbers differently from SVG. Pass
        # canonical, separated numeric tokens without changing the stored source.
        parse_path(" ".join(normalized_tokens), pen)
    except (ValueError, TypeError, IndexError, AssertionError, ZeroDivisionError) as exc:
        raise ValueError("invalid SVG stroke path") from exc
    if not pen.value or any(operator == "endPath" for operator, _ in pen.value):
        raise ValueError("stroke contours must be closed")
    for _, points in pen.value:
        for point in points:
            for coordinate in point:
                _coordinate(coordinate)
    bounds_pen = BoundsPen(None)
    pen.replay(bounds_pen)
    bounds = bounds_pen.bounds
    if bounds is None or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("stroke must have a nondegenerate outline")
    return pen, (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))


def _validate_record(raw: object) -> StrokeRecord:
    if not isinstance(raw, dict):
        raise ValueError("each JSONL record must be an object")
    character = raw.get("character")
    if not isinstance(character, str) or len(character) != 1 or character.isspace():
        raise ValueError("character must be exactly one non-whitespace Unicode code point")
    if 0xD800 <= ord(character) <= 0xDFFF:
        raise ValueError("character cannot be an unpaired surrogate")
    strokes, medians = raw.get("strokes"), raw.get("medians")
    if not isinstance(strokes, (list, tuple)) or not strokes:
        raise ValueError("strokes must be a nonempty sequence")
    if not isinstance(medians, (list, tuple)) or len(strokes) != len(medians):
        raise ValueError("strokes and medians must have matching counts")
    for stroke in strokes:
        _parse_outline(stroke)
    checked_medians = []
    for median in medians:
        if not isinstance(median, (list, tuple)) or len(median) < 2:
            raise ValueError("each median must contain at least two points")
        points = []
        for point in median:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("each median point must contain exactly two coordinates")
            points.append((_coordinate(point[0]), _coordinate(point[1])))
        checked_medians.append(tuple(points))
    return StrokeRecord(character, tuple(strokes), tuple(checked_medians))


def load_stroke_records(
    path: Path,
    characters: tuple[str, ...] | None = None,
    *,
    progress: Callable[[int], None] | None = None,
) -> dict[str, StrokeRecord]:
    """Validate JSONL, selecting requested characters or raising on missing glyphs.

    Even records outside the selection are validated, and duplicate characters
    anywhere in the source are rejected. Extra source metadata is ignored.
    Progress receives the processed row count every 250 records and at completion.
    """
    selected = None if characters is None else set(characters)
    seen: set[str] = set()
    records: dict[str, StrokeRecord] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = _validate_record(json.loads(line))
                if record.character in seen:
                    raise ValueError(f"duplicate character {record.character!r}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen.add(record.character)
            if selected is None or record.character in selected:
                records[record.character] = record
            if progress is not None and line_number % 250 == 0:
                progress(line_number)
    if progress is not None and len(seen) % 250:
        progress(len(seen))
    if selected is not None and (missing := selected - records.keys()):
        raise ValueError(f"missing stroke characters: {', '.join(sorted(missing))}")
    return records


class _RasterPen(BasePen):
    """Preserve each full multi-contour outline in one nonzero-winding AGG path."""

    def __init__(self) -> None:
        super().__init__(None)
        self.path = aggdraw.Path()

    def _moveTo(self, point: tuple[float, float]) -> None:
        self.path.moveto(*point)

    def _lineTo(self, point: tuple[float, float]) -> None:
        self.path.lineto(*point)

    def _curveToOne(
        self, first: tuple[float, float], second: tuple[float, float], end: tuple[float, float]
    ) -> None:
        self.path.curveto(*first, *second, *end)

    def _closePath(self) -> None:
        self.path.close()


def render_stroke_layers(
    record: StrokeRecord, canvas_size: int = 128
) -> tuple[NDArray[np.uint8], ...]:
    """Render per-stroke 2D alpha at 4x resolution, with one shared 96px fit.

    Composite with ``np.maximum.reduce(layers)`` to obtain white ink on black.
    Pixels at crossings belong independently to every original covering stroke.
    """
    if isinstance(canvas_size, bool) or not isinstance(canvas_size, int) or canvas_size <= 96:
        raise ValueError("canvas_size must be an integer greater than 96")
    _validate_record(
        {
            "character": record.character,
            "strokes": record.strokes,
            "medians": record.medians,
        }
    )
    outlines = [_parse_outline(path) for path in record.strokes]
    left = min(bounds[0] for _, bounds in outlines)
    bottom = min(bounds[1] for _, bounds in outlines)
    right = max(bounds[2] for _, bounds in outlines)
    top = max(bounds[3] for _, bounds in outlines)
    scale = _GLYPH_SIZE / max(right - left, top - bottom)
    # SVG view bounds are [left, 900-top, right, 900-bottom]. Center after Y flip.
    view_top = 900.0 - top
    offset_x = (canvas_size - (right - left) * scale) / 2 - left * scale
    offset_y = (canvas_size - (top - bottom) * scale) / 2 - view_top * scale
    transform = (
        scale * _SUPERSAMPLING,
        0,
        0,
        -scale * _SUPERSAMPLING,
        offset_x * _SUPERSAMPLING,
        (offset_y + 900 * scale) * _SUPERSAMPLING,
    )
    layers = []
    for outline, _ in outlines:
        pen = _RasterPen()
        outline.replay(TransformPen(pen, transform))
        canvas = Image.new("L", (canvas_size * _SUPERSAMPLING,) * 2)
        draw = aggdraw.Draw(canvas)
        draw.path(pen.path, aggdraw.Brush(255))
        draw.flush()
        image = canvas.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
        layer = np.array(image, dtype=np.uint8)
        if not np.any(layer):
            raise ValueError(f"stroke rendered no visible ink for {record.character!r}")
        layers.append(layer)
    return tuple(layers)
