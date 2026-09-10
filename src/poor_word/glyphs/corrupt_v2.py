"""Bounded, local-geometry corruption proposals, not Chinese validity labels.

Skeleton neighbourhoods approximate straight *segments*, not semantic strokes or
radicals. Local distance-transform widths intentionally vary within a glyph (for
example between thin horizontal and thick vertical strokes in a serif font).
Successful proposals satisfy geometric and 96-pixel visibility checks only. They
must never be interpreted as automatic BLOCK labels or guaranteed illegal glyphs.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize

from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable
from poor_word.glyphs.render import RenderedGlyph

Severity = Literal["medium", "strong"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ByteArray = NDArray[np.uint8]


@dataclass(frozen=True)
class V2CorruptionResult:
    image: ByteArray
    mask: ByteArray
    changed_mask: ByteArray
    changed_pixels: int
    operator: str
    severity: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class _Geometry:
    core: BoolArray
    points: FloatArray
    widths: FloatArray
    scale: float
    antialiased: bool


@dataclass(frozen=True)
class _Segment:
    center: FloatArray  # Coordinates throughout this module are (row, column).
    tangent: FloatArray
    width: float


@dataclass(frozen=True)
class _Proposal:
    image: ByteArray
    metrics: dict[str, float]
    segment: _Segment | None = None
    gap: float = 0.0


def _labels(mask: BoolArray) -> NDArray[np.int32]:
    return np.asarray(label(mask, connectivity=2), dtype=np.int32)  # type: ignore[no-untyped-call]


def _euler(mask: BoolArray) -> int:
    return int(euler_number(mask, connectivity=2))  # type: ignore[no-untyped-call]


def _geometry(rendered: RenderedGlyph) -> _Geometry:
    gray = np.max(rendered.image, axis=2)
    core = gray >= max(32.0, float(gray.max()) * 0.5)
    distance = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    skeleton = np.asarray(skeletonize(core), dtype=np.bool_)  # type: ignore[no-untyped-call]
    points = np.argwhere(skeleton)
    widths = distance[points[:, 0], points[:, 1]].astype(np.float64) * 2.0
    foreground = np.argwhere(rendered.mask)
    scale = float(np.max(np.ptp(foreground, axis=0) + 1))
    return _Geometry(
        core=core,
        points=points.astype(np.float64),
        widths=widths,
        scale=scale,
        antialiased=bool(np.any((gray > 0) & (gray < gray.max()))),
    )


def _segment_at(geometry: _Geometry, index: int) -> _Segment | None:
    center = geometry.points[index]
    width = float(geometry.widths[index])
    offsets = geometry.points - center
    radius = max(width * 1.4, geometry.scale * 0.055)
    neighbors = offsets[np.sum(offsets * offsets, axis=1) <= radius * radius]
    if len(neighbors) < 3 or width > geometry.scale * 0.45:
        return None
    values, vectors = np.linalg.eigh(np.cov(neighbors, rowvar=False))
    # Junctions and compact blobs are not reliable straight-segment candidates.
    if values[-1] < 1.0 or values[-1] < max(0.1, values[0]) * 4.0:
        return None
    tangent = vectors[:, -1]
    return _Segment(center, tangent, width)


def _inside(point: FloatArray, shape: tuple[int, ...], margin: float = 1.0) -> bool:
    return bool(np.all(point >= margin) and np.all(point <= np.array(shape[:2]) - 1 - margin))


def _core_at(core: BoolArray, point: FloatArray) -> bool:
    if not _inside(point, core.shape, margin=0):
        return False
    row, column = np.rint(point).astype(int)
    return bool(core[row, column])


def _stroke_alpha(
    shape: tuple[int, ...], start: FloatArray, end: FloatArray, width: float, antialiased: bool
) -> ByteArray | None:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 0:
        return None
    normal = np.array((-direction[1], direction[0])) / length
    # A subtly tapered, flat-ended segment; no circular deletion/addition blobs.
    corners = np.array(
        [
            start + normal * width * 0.48,
            end + normal * width * 0.52,
            end - normal * width * 0.52,
            start - normal * width * 0.48,
        ]
    )
    if not all(_inside(corner, shape) for corner in corners):
        return None
    factor = 4 if antialiased else 1
    canvas = Image.new("L", (shape[1] * factor, shape[0] * factor))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(float(p[1] * factor), float(p[0] * factor)) for p in corners], fill=255)
    if factor > 1:
        canvas = canvas.resize((shape[1], shape[0]), Image.Resampling.BOX)
    return np.asarray(canvas, dtype=np.uint8)


def _add_image(image: ByteArray, alpha: ByteArray) -> ByteArray:
    # Keep source pixels and their grayscale edge profile exactly where unchanged.
    intensity = np.max(image, axis=(0, 1)).astype(np.float64)
    ink = np.rint(alpha[:, :, None].astype(float) * intensity / 255.0).astype(np.uint8)
    return np.maximum(image, ink)


def _cut(
    rendered: RenderedGlyph, geometry: _Geometry, segment: _Segment, operator: str, strong: bool
) -> _Proposal | None:
    width = segment.width
    if operator == "break_stroke":
        gap = max(width * (1.65 if strong else 1.0), geometry.scale * (0.07 if strong else 0.04))
    else:
        gap = min(
            max(width * (3.2 if strong else 2.1), geometry.scale * (0.24 if strong else 0.16)),
            geometry.scale * (0.60 if strong else 0.46),
        )
    # A thin serif horizontal needs a longer gap than a thick vertical to
    # survive the input-scale visibility floor; do not silently discard it.
    visible_fraction = (
        (0.045 if strong else 0.03) if operator == "break_stroke" else (0.075 if strong else 0.05)
    )
    visible_area = np.count_nonzero(rendered.mask) * visible_fraction
    gap = max(gap, float(visible_area) / width)
    shoulders = gap / 2.0 + width * 0.65
    if not all(
        _core_at(geometry.core, segment.center + sign * segment.tangent * shoulders)
        for sign in (-1, 1)
    ):
        return None
    start = segment.center - segment.tangent * gap / 2.0
    end = segment.center + segment.tangent * gap / 2.0
    alpha = _stroke_alpha(rendered.mask.shape, start, end, width * 1.85, geometry.antialiased)
    if alpha is None:
        return None
    image = np.rint(rendered.image.astype(float) * (1.0 - alpha[:, :, None] / 255.0)).astype(
        np.uint8
    )
    return _Proposal(
        image,
        {
            "local_stroke_width": width,
            "segment_length": gap,
            "gap_length": gap,
            "segment_width": width * 1.85,
        },
        segment,
        gap,
    )


def _add(
    rendered: RenderedGlyph,
    geometry: _Geometry,
    segment: _Segment,
    random: np.random.Generator,
    strong: bool,
) -> _Proposal | None:
    angle = float(random.choice(np.array([-1.0, -0.5, 0.5, 1.0]))) * np.pi / 2.0
    cosine, sine = np.cos(angle), np.sin(angle)
    tangent = segment.tangent
    direction = np.array(
        [cosine * tangent[0] - sine * tangent[1], sine * tangent[0] + cosine * tangent[1]]
    )
    length = max(
        segment.width * (2.9 if strong else 2.0), geometry.scale * (0.32 if strong else 0.22)
    )
    width = segment.width * (1.1 if strong else 0.9)
    alpha = _stroke_alpha(
        rendered.mask.shape,
        segment.center,
        segment.center + direction * length,
        width,
        geometry.antialiased,
    )
    if alpha is None:
        return None
    return _Proposal(
        _add_image(rendered.image, alpha),
        {"local_stroke_width": segment.width, "segment_length": length, "segment_width": width},
    )


def _components(rendered: RenderedGlyph) -> list[NDArray[np.int64]]:
    labeled = _labels(rendered.mask.astype(bool))
    area = int(np.count_nonzero(rendered.mask))
    return [
        coordinates
        for index in range(1, int(labeled.max()) + 1)
        if len(coordinates := np.argwhere(labeled == index)) >= max(4, area * 0.025)
    ]


def _nearest(first: NDArray[np.int64], second: NDArray[np.int64]) -> tuple[FloatArray, FloatArray]:
    distances, indices = cKDTree(first).query(second, k=1)
    index = int(np.argmin(distances))
    return first[int(indices[index])].astype(float), second[index].astype(float)


def _width_near(geometry: _Geometry, point: FloatArray) -> float:
    index = int(np.argmin(np.sum((geometry.points - point) ** 2, axis=1)))
    return float(geometry.widths[index])


def _bridge(
    rendered: RenderedGlyph,
    geometry: _Geometry,
    components: list[NDArray[np.int64]],
    random: np.random.Generator,
    strong: bool,
) -> _Proposal | None:
    pairs = list(combinations(range(len(components)), 2))
    if pairs:
        first, second = pairs[int(random.integers(len(pairs)))]
        start, end = _nearest(components[first], components[second])
    else:
        indices = random.choice(len(geometry.points), size=2, replace=False)
        start, end = geometry.points[indices]
    distance = float(np.linalg.norm(end - start))
    local_width = min(_width_near(geometry, start), _width_near(geometry, end))
    if not local_width * 0.65 <= distance <= geometry.scale * 0.5:
        return None
    direction = (end - start) / distance
    width = local_width * (1.1 if strong else 0.85)
    alpha = _stroke_alpha(
        rendered.mask.shape,
        start - direction * local_width * 0.55,
        end + direction * local_width * 0.55,
        width,
        geometry.antialiased,
    )
    if alpha is None:
        return None
    image = _add_image(rendered.image, alpha)
    return _Proposal(
        image,
        {"local_stroke_width": local_width, "segment_width": width, "segment_length": distance},
    )


def _shift(
    rendered: RenderedGlyph,
    geometry: _Geometry,
    components: list[NDArray[np.int64]],
    random: np.random.Generator,
    strong: bool,
) -> _Proposal | None:
    original_area = int(np.count_nonzero(rendered.mask))
    eligible = [part for part in components if 0.08 <= len(part) / original_area <= 0.7]
    if not eligible or len(components) < 2:
        return None
    # Sample substantial parts, weighted toward area, instead of always choosing the smallest.
    areas = np.array([len(part) for part in eligible], dtype=float)
    component = eligible[int(random.choice(len(eligible), p=areas / areas.sum()))]
    belongs = np.zeros(rendered.mask.shape, dtype=bool)
    belongs[component[:, 0], component[:, 1]] = True
    point_indices = np.rint(geometry.points).astype(int)
    widths = geometry.widths[belongs[point_indices[:, 0], point_indices[:, 1]]]
    width = (
        float(np.median(widths)) if len(widths) else _width_near(geometry, component.mean(axis=0))
    )
    stationary = rendered.mask.astype(bool) & ~belongs
    if random.random() < 0.5:
        start, end = _nearest(component, np.argwhere(stationary))
        direction = end - start
        separation = float(np.linalg.norm(direction))
        distance = separation + width * (0.22 if strong else 0.12)
        if distance > geometry.scale * 0.24:
            return None
        offset = np.rint(direction / max(separation, 1.0) * distance).astype(int)
    else:
        angle = float(random.integers(8)) * np.pi / 4
        distance = max(
            width * (1.4 if strong else 0.95), geometry.scale * (0.13 if strong else 0.085)
        )
        distance *= float(random.uniform(0.85, 1.1))
        offset = np.rint(np.array((np.sin(angle), np.cos(angle))) * distance).astype(int)
    shifted = component + offset
    if np.any(shifted < 1) or np.any(shifted >= np.array(rendered.mask.shape) - 1):
        return None
    overlap = float(np.mean(stationary[shifted[:, 0], shifted[:, 1]]))
    if overlap > (0.4 if strong else 0.3):
        return None
    image = rendered.image.copy()
    image[belongs] = 0
    image[shifted[:, 0], shifted[:, 1]] = np.maximum(
        image[shifted[:, 0], shifted[:, 1]], rendered.image[component[:, 0], component[:, 1]]
    )
    return _Proposal(
        image,
        {
            "local_stroke_width": width,
            "overlap_fraction": overlap,
            "moved_component_fraction": len(component) / original_area,
            "shift_distance": float(np.linalg.norm(offset)),
            "shift_row": float(offset[0]),
            "shift_column": float(offset[1]),
        },
    )


def _gap_clear(image: ByteArray, segment: _Segment, gap: float, scale: float = 1.0) -> bool:
    center, width = segment.center * scale, segment.width * scale
    rows, columns = np.indices(image.shape[:2])
    along = (rows - center[0]) * segment.tangent[0] + (columns - center[1]) * segment.tangent[1]
    across = -(rows - center[0]) * segment.tangent[1] + (columns - center[1]) * segment.tangent[0]
    interior = (np.abs(along) < max(0.6, gap * scale * 0.30)) & (np.abs(across) < width * 0.70)
    return bool(np.any(interior) and not np.any(np.max(image, axis=2)[interior] >= 96))


def _verify(
    rendered: RenderedGlyph,
    geometry: _Geometry,
    proposal: _Proposal,
    operator: str,
    severity: Severity,
    attempt: int,
) -> V2CorruptionResult | None:
    image = proposal.image
    changed = np.any(image != rendered.image, axis=2)
    changed_pixels = int(changed.sum())
    area = int(np.count_nonzero(rendered.mask))
    changed_fraction = changed_pixels / area
    strong = severity == "strong"
    if not (0.025 <= changed_fraction <= (0.70 if strong else 0.55)):
        return None
    mask = np.any(image > 0, axis=2)
    core = np.max(image, axis=2) >= max(32.0, float(rendered.image.max()) * 0.5)
    new_area = int(mask.sum())
    if new_area < area * (0.35 if strong else 0.48) or np.mean(mask) > 0.8:
        return None
    if operator in {"erase_segment", "break_stroke"}:
        if proposal.segment is None or not _gap_clear(image, proposal.segment, proposal.gap):
            return None
    if operator == "bridge":
        if not (
            int(_labels(core).max()) < int(_labels(geometry.core).max())
            or _euler(core) < _euler(geometry.core)
        ):
            return None
        # Match GlyphDataset's actual mask view exactly: RGB luminance first,
        # AREA resize second, then include faint antialiased foreground >8.
        # A new hole in the >=128 core can vanish when that edge is included.
        original_training_mask, changed_training_mask = [
            cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                (96, 96),
                interpolation=cv2.INTER_AREA,
            )
            > 8
            for rgb in (rendered.image, image)
        ]
        if not (
            int(_labels(changed_training_mask).max()) < int(_labels(original_training_mask).max())
            or _euler(changed_training_mask) < _euler(original_training_mask)
        ):
            return None
    if operator == "add_stroke":
        if new_area <= area or int(_labels(core).max()) > int(_labels(geometry.core).max()):
            return None
    input_fractions = []
    input_pixels = []
    for original_small, changed_small in (
        (
            cv2.resize(rendered.image, (96, 96), interpolation=cv2.INTER_AREA),
            cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA),
        ),
        (
            np.asarray(Image.fromarray(rendered.image).resize((96, 96), Image.Resampling.BILINEAR)),
            np.asarray(Image.fromarray(image).resize((96, 96), Image.Resampling.BILINEAR)),
        ),
    ):
        difference = np.max(np.abs(changed_small.astype(float) - original_small), axis=2)
        visible = int(np.count_nonzero(difference >= 32))
        fraction = visible / max(1, int(np.count_nonzero(np.max(original_small, axis=2) >= 128)))
        if visible < 8 or fraction < 0.025:
            return None
        original_input_core = np.max(original_small, axis=2) >= 128
        changed_input_core = np.max(changed_small, axis=2) >= 128
        if operator == "bridge":
            if not (
                int(_labels(changed_input_core).max()) < int(_labels(original_input_core).max())
                or _euler(changed_input_core) < _euler(original_input_core)
            ):
                return None
        if operator == "add_stroke":
            if int(_labels(changed_input_core).max()) > int(_labels(original_input_core).max()):
                return None
        if proposal.segment is not None and rendered.mask.shape[0] == rendered.mask.shape[1]:
            if not _gap_clear(
                np.asarray(changed_small, dtype=np.uint8),
                proposal.segment,
                proposal.gap,
                96 / image.shape[0],
            ):
                return None
        input_pixels.append(visible)
        input_fractions.append(fraction)
    return V2CorruptionResult(
        image=image,
        mask=mask.astype(np.uint8),
        changed_mask=changed.astype(np.uint8),
        changed_pixels=changed_pixels,
        operator=operator,
        severity=severity,
        metrics={
            **proposal.metrics,
            "changed_fraction": float(changed_fraction),
            "input_changed_fraction": float(min(input_fractions)),
            "input_changed_pixels": float(min(input_pixels)),
            "foreground_delta_fraction": float((new_area - area) / area),
            "attempts": float(attempt),
            "structure_verified": 1.0,
        },
    )


def corrupt_glyph_v2(
    rendered: RenderedGlyph, operator: str, seed: int, severity: Severity = "medium"
) -> V2CorruptionResult:
    """Propose a deterministic local edit or skip after at most 96 candidates.

    ``changed_fraction`` uses actual changed RGB positions / original foreground
    support. ``input_changed_fraction`` is the smaller significant-change ratio
    across the project's AREA and bilinear 96x96 resize paths (intensity delta
    at least 32, denominator original input pixels at least 128). Metrics are
    descriptive, not confidence estimates of Chinese illegality. Bridges also
    require a topology change in the training mask: RGB2GRAY, AREA 96x96, >8.
    """
    if operator not in OPERATORS:
        raise ValueError(f"unknown corruption operator: {operator}")
    if severity not in {"medium", "strong"}:
        raise ValueError("severity must be medium or strong")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        rendered.mask.ndim != 2
        or rendered.image.ndim != 3
        or rendered.image.shape != (*rendered.mask.shape, 3)
        or rendered.image.dtype != np.uint8
        or rendered.mask.dtype != np.uint8
        or not np.all((rendered.mask == 0) | (rendered.mask == 1))
    ):
        raise ValueError("rendered image must be uint8 RGB with a matching uint8 binary mask")
    if not np.array_equal(rendered.mask.astype(bool), np.any(rendered.image > 0, axis=2)):
        raise ValueError("rendered mask must match nonzero RGB foreground support")
    area = int(np.count_nonzero(rendered.mask))
    if area < 16 or np.mean(rendered.mask) > 0.8 or min(rendered.mask.shape) < 16:
        raise CorruptionNotApplicable("glyph has insufficient usable foreground or background")
    geometry = _geometry(rendered)
    if len(geometry.points) < 3:
        raise CorruptionNotApplicable("glyph has no usable local skeleton segments")
    random = np.random.default_rng(seed)
    components = _components(rendered) if operator in {"bridge", "component_shift"} else []
    for attempt in range(1, 97):
        proposal: _Proposal | None
        strong = severity == "strong"
        if operator == "component_shift":
            proposal = _shift(rendered, geometry, components, random, strong)
        elif operator == "bridge":
            proposal = _bridge(rendered, geometry, components, random, strong)
        else:
            segment = _segment_at(geometry, int(random.integers(len(geometry.points))))
            if segment is None:
                continue
            if operator == "add_stroke":
                proposal = _add(rendered, geometry, segment, random, strong)
            else:
                proposal = _cut(rendered, geometry, segment, operator, strong)
        if proposal is not None:
            result = _verify(rendered, geometry, proposal, operator, severity, attempt)
            if result is not None:
                return result
    raise CorruptionNotApplicable(
        f"{operator} ({severity}) found no scale-matched, structure-checked visible candidate "
        f"after 96 attempts for seed {seed}"
    )
