from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from skimage.draw import disk as draw_disk
from skimage.draw import line
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, skeletonize

from poor_word.glyphs.render import RenderedGlyph

OPERATORS = frozenset(
    {"erase_segment", "add_stroke", "break_stroke", "bridge", "component_shift"}
)


class CorruptionNotApplicable(RuntimeError):
    """Raised when an operator cannot satisfy its labeling postconditions."""


@dataclass(frozen=True)
class CorruptionResult:
    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    changed_mask: NDArray[np.uint8]
    changed_pixels: int
    operator: str


CandidateFactory = Callable[
    [NDArray[np.bool_], np.random.Generator], NDArray[np.bool_] | None
]


def _skeleton_coordinates(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    return np.argwhere(skeletonize(mask)).astype(np.int64, copy=False)


def _erase_segment(
    mask: NDArray[np.bool_], random: np.random.Generator
) -> NDArray[np.bool_] | None:
    coordinates = _skeleton_coordinates(mask)
    if not len(coordinates):
        return None
    center = coordinates[int(random.integers(len(coordinates)))]
    radius = int(random.integers(2, 5))
    rows, columns = draw_disk((int(center[0]), int(center[1])), radius, shape=mask.shape)
    candidate = mask.copy()
    candidate[rows, columns] = False
    return candidate


def _draw_thick_line(
    shape: tuple[int, ...],
    start: NDArray[np.int64],
    end: NDArray[np.int64],
    width: int,
) -> NDArray[np.bool_]:
    rows, columns = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
    line_mask = np.zeros(shape, dtype=bool)
    line_mask[rows, columns] = True
    if width > 1:
        line_mask = dilation(line_mask, footprint=disk(width - 1))
    return line_mask


def _add_stroke(
    mask: NDArray[np.bool_], random: np.random.Generator
) -> NDArray[np.bool_] | None:
    coordinates = _skeleton_coordinates(mask)
    if len(coordinates) < 2:
        return None
    indices = random.choice(len(coordinates), size=2, replace=False)
    start, end = coordinates[indices]
    distance = float(np.linalg.norm(start - end))
    if distance < 8:
        return None
    stroke = _draw_thick_line(mask.shape, start, end, width=int(random.integers(1, 4)))
    return np.logical_or(mask, stroke)


def _local_tangent(
    coordinates: NDArray[np.int64], center: NDArray[np.int64]
) -> NDArray[np.float64] | None:
    offsets = coordinates - center
    neighbors = offsets[np.sum(offsets * offsets, axis=1) <= 25]
    if len(neighbors) < 2:
        return None
    covariance = np.cov(neighbors, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, -1]
    norm = float(np.linalg.norm(tangent))
    return None if norm == 0 else tangent / norm


def _break_stroke(
    mask: NDArray[np.bool_], random: np.random.Generator
) -> NDArray[np.bool_] | None:
    coordinates = _skeleton_coordinates(mask)
    if len(coordinates) < 2:
        return None
    center = coordinates[int(random.integers(len(coordinates)))]
    tangent = _local_tangent(coordinates, center)
    if tangent is None:
        return None
    perpendicular = np.array((-tangent[1], tangent[0]))
    half_length = int(random.integers(4, 8))
    start = np.rint(center - perpendicular * half_length).astype(np.int64)
    end = np.rint(center + perpendicular * half_length).astype(np.int64)
    start = np.clip(start, (0, 0), np.array(mask.shape) - 1)
    end = np.clip(end, (0, 0), np.array(mask.shape) - 1)
    eraser = _draw_thick_line(mask.shape, start, end, width=int(random.integers(1, 3)))
    candidate = mask.copy()
    candidate[eraser] = False
    return candidate


def _nearest_component_points(
    components: list[NDArray[np.int64]],
) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
    best: tuple[float, NDArray[np.int64], NDArray[np.int64]] | None = None
    for first_index, first in enumerate(components):
        tree = cKDTree(first)
        for second in components[first_index + 1 :]:
            distances, indices = tree.query(second, k=1)
            second_index = int(np.argmin(distances))
            first_point = first[int(indices[second_index])]
            second_point = second[second_index]
            distance = float(distances[second_index])
            if best is None or distance < best[0]:
                best = (distance, first_point, second_point)
    return None if best is None else (best[1], best[2])


def _bridge(
    mask: NDArray[np.bool_], random: np.random.Generator
) -> NDArray[np.bool_] | None:
    labeled = label(mask, connectivity=2)
    components = [region.coords.astype(np.int64, copy=False) for region in regionprops(labeled)]
    if len(components) >= 2:
        endpoints = _nearest_component_points(components)
        if endpoints is None:
            return None
        start, end = endpoints
    else:
        coordinates = _skeleton_coordinates(mask)
        if len(coordinates) < 2:
            return None
        indices = random.choice(len(coordinates), size=2, replace=False)
        start, end = coordinates[indices]
        if np.linalg.norm(start - end) < min(mask.shape) * 0.35:
            return None
    bridge = _draw_thick_line(mask.shape, start, end, width=int(random.integers(1, 3)))
    return np.logical_or(mask, bridge)


def _component_shift(
    mask: NDArray[np.bool_], random: np.random.Generator
) -> NDArray[np.bool_] | None:
    labeled = label(mask, connectivity=2)
    regions = sorted(regionprops(labeled), key=lambda region: region.area)
    if len(regions) < 2:
        return None
    component = regions[0].coords.astype(np.int64, copy=False)
    distance = int(random.integers(4, 13))
    direction = int(random.integers(8))
    offsets = np.array(
        ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)),
        dtype=np.int64,
    )
    shifted = component + offsets[direction] * distance
    if np.any(shifted < 0) or np.any(shifted >= np.array(mask.shape)):
        return None

    candidate = mask.copy()
    candidate[component[:, 0], component[:, 1]] = False
    if np.any(candidate[shifted[:, 0], shifted[:, 1]]):
        return None
    candidate[shifted[:, 0], shifted[:, 1]] = True
    return candidate


_CANDIDATE_FACTORIES: dict[str, CandidateFactory] = {
    "erase_segment": _erase_segment,
    "add_stroke": _add_stroke,
    "break_stroke": _break_stroke,
    "bridge": _bridge,
    "component_shift": _component_shift,
}


def _build_result(
    rendered: RenderedGlyph,
    original: NDArray[np.bool_],
    candidate: NDArray[np.bool_],
    operator: str,
) -> CorruptionResult:
    changed = np.logical_xor(original, candidate)
    image = rendered.image.copy()
    image[np.logical_and(original, np.logical_not(candidate))] = 0
    image[np.logical_and(np.logical_not(original), candidate)] = 255
    return CorruptionResult(
        image=image.astype(np.uint8, copy=False),
        mask=candidate.astype(np.uint8),
        changed_mask=changed.astype(np.uint8),
        changed_pixels=int(np.count_nonzero(changed)),
        operator=operator,
    )


def corrupt_glyph(rendered: RenderedGlyph, operator: str, seed: int) -> CorruptionResult:
    if operator not in _CANDIDATE_FACTORIES:
        raise ValueError(f"unknown corruption operator: {operator}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if rendered.mask.ndim != 2 or rendered.image.shape[:2] != rendered.mask.shape:
        raise ValueError("rendered image and mask shapes must agree")

    original = rendered.mask.astype(bool)
    foreground_pixels = int(np.count_nonzero(original))
    maximum_changed = int(foreground_pixels * 0.35)
    random = np.random.Generator(np.random.PCG64(seed))
    factory = _CANDIDATE_FACTORIES[operator]

    for _ in range(20):
        candidate = factory(original, random)
        if candidate is None:
            continue
        changed_pixels = int(np.count_nonzero(np.logical_xor(original, candidate)))
        if 8 <= changed_pixels <= maximum_changed:
            return _build_result(rendered, original, candidate, operator)

    raise CorruptionNotApplicable(
        f"{operator} could not change between 8 and {maximum_changed} pixels for seed {seed}"
    )
