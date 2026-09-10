"""Auditable, bounded stroke-layer proposals, for human REVIEW only.

Inputs are independent white-on-black alpha layers, not a segmentation inferred
from the final bitmap. Recomposition always uses maximum alpha; deleting a layer
therefore cannot erase another stroke's intersection pixels. Operator names are
legacy editing names, not five semantic categories of malformed Chinese.

Metrics describe geometry and visibility, never linguistic validity, OCR
confidence, or training labels. Every successful result still needs human review.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.measure import euler_number
from skimage.morphology import skeletonize

from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable
from poor_word.glyphs.stroke_bridge import BridgePlanner

ByteArray = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class StrokeCorruptionResult:
    image: ByteArray
    changed_mask: ByteArray
    changed_pixels: int
    operator: str
    metrics: dict[str, float]
    selected_stroke_indices: tuple[int, ...]
    edited_layers: tuple[ByteArray, ...]
    bridge_mode: str | None = None


@dataclass(frozen=True)
class _Proposal:
    layers: tuple[ByteArray, ...]
    selected: tuple[int, ...]
    metrics: dict[str, float]
    bridge_mode: str | None = None


def _small(alpha: ByteArray) -> ByteArray:
    # Equal RGB channels mean RGB2GRAY is exactly this alpha, including rounding.
    return np.asarray(cv2.resize(alpha, (96, 96), interpolation=cv2.INTER_AREA), dtype=np.uint8)


def _topology(core: BoolArray) -> tuple[int, int]:
    components = int(cv2.connectedComponents(core.astype(np.uint8), connectivity=8)[0]) - 1
    return components, int(euler_number(core, connectivity=2))  # type: ignore[no-untyped-call]


def _width(layer: ByteArray) -> float:
    core = layer >= 128
    skeleton = np.asarray(skeletonize(core), dtype=np.bool_)  # type: ignore[no-untyped-call]
    distance = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    return float(np.median(distance[skeleton]) * 2) if skeleton.any() else 1.0


def _rest(layers: tuple[ByteArray, ...], selected: int) -> ByteArray:
    return np.asarray(
        np.maximum.reduce([layer for i, layer in enumerate(layers) if i != selected]),
        dtype=np.uint8,
    )


def _transform(layer: ByteArray, matrix: NDArray[np.float64]) -> ByteArray | None:
    points = np.argwhere(layer > 0)[:, ::-1].astype(float)
    transformed = points @ matrix[:, :2].T + matrix[:, 2]
    # Reject clipping instead of silently truncating a transformed whole stroke.
    if not len(points) or np.any(transformed < 1) or np.any(transformed > layer.shape[0] - 2):
        return None
    return np.asarray(
        cv2.warpAffine(layer, matrix, (layer.shape[1], layer.shape[0]), flags=cv2.INTER_LINEAR),
        dtype=np.uint8,
    )


def _erase(layers: tuple[ByteArray, ...], index: int, width: float) -> _Proposal | None:
    if len(layers) < 2:
        return None
    return _Proposal(
        tuple(layer for i, layer in enumerate(layers) if i != index),
        (index,),
        {"removed_stroke_count": 1.0, "local_stroke_width": width},
    )


def _shift(
    layers: tuple[ByteArray, ...],
    index: int,
    width: float,
    scale: float,
    random: np.random.Generator,
) -> _Proposal | None:
    if len(layers) < 2:
        return None
    source = layers[index]
    rest = _rest(layers, index)
    source_points, target_points = np.argwhere(source >= 128), np.argwhere(rest >= 128)
    if not len(source_points) or not len(target_points):
        return None
    # Anchor a source ink point to another stroke's ink: collisions are intentional.
    offset = (
        target_points[random.integers(len(target_points))]
        - source_points[random.integers(len(source_points))]
    )
    distance = float(np.linalg.norm(offset))
    if distance < max(2 * width, scale * 0.15):
        return None
    matrix = np.array([[1.0, 0.0, offset[1]], [0.0, 1.0, offset[0]]])
    moved = _transform(source, matrix)
    if moved is None:
        return None
    core, rest_core = _small(moved) >= 128, _small(rest) >= 128
    overlap = float(np.count_nonzero(core & rest_core) / max(1, core.sum()))
    if not 0.15 <= overlap <= 0.45:
        return None
    native_core = moved >= 128
    native_overlap = float(np.count_nonzero(native_core & (rest >= 128)) / native_core.sum())
    edited = tuple(moved if i == index else layer for i, layer in enumerate(layers))
    return _Proposal(
        edited,
        (index,),
        {
            "local_stroke_width": width,
            "shift_distance": distance,
            "shift_dx": float(offset[1]),
            "shift_dy": float(offset[0]),
            "overlap_fraction": native_overlap,
            "input_overlap_fraction": overlap,
            "moved_stroke_count": 1.0,
        },
    )


def _add(
    layers: tuple[ByteArray, ...],
    widths: list[float],
    scale: float,
    random: np.random.Generator,
) -> _Proposal | None:
    original = np.maximum.reduce(layers)
    original_small = _small(original)
    original_core = original_small >= 128
    original_points = np.argwhere(original >= 128)
    area = int(original_core.sum())
    added: list[ByteArray] = []
    sources: list[int] = []
    current = original.copy()
    for _ in range(int(random.integers(2, 5))):
        index = int(random.integers(len(layers)))
        source = layers[index]
        points = np.argwhere(source >= 128)
        if len(points) < max(12, np.count_nonzero(original >= 128) * 0.05):
            return None
        anchor = points[random.integers(len(points))][::-1].astype(float)
        target = original_points[random.integers(len(original_points))][::-1].astype(float)
        angle = float(random.choice(np.array([-90, -45, 0, 45, 90, 180])))
        matrix = np.asarray(
            cv2.getRotationMatrix2D((float(anchor[0]), float(anchor[1])), angle, 1.0),
            dtype=np.float64,
        )
        matrix[:, 2] += target - anchor
        transformed = _transform(source, matrix)
        if transformed is None:
            return None
        core = _small(transformed) >= 128
        contact = int(np.count_nonzero(core & original_core))
        new = int(np.count_nonzero(core & ~(_small(current) > 8)))
        if contact < max(4, core.sum() * 0.03) or new < max(12, area * 0.04, core.sum() * 0.30):
            return None
        added.append(transformed)
        sources.append(index)
        current = np.maximum(current, transformed)
    # Every added stroke must remain independently visible even after later copies.
    for index, stroke in enumerate(added):
        others = np.maximum.reduce(layers + tuple(added[:index] + added[index + 1 :]))
        unique = np.count_nonzero((_small(stroke) >= 128) & ~(_small(others) > 8))
        if unique < max(12, area * 0.025):
            return None
    # Prevent overlapping copies from forming an unstructured, unusually thick blob.
    old_distance = cv2.distanceTransform((original >= 128).astype(np.uint8), cv2.DIST_L2, 5)
    new_distance = cv2.distanceTransform((current >= 128).astype(np.uint8), cv2.DIST_L2, 5)
    if new_distance.max() > max(float(old_distance.max()) * 1.6, float(np.median(widths))):
        return None
    return _Proposal(
        layers + tuple(added),
        tuple(sources),
        {
            "added_stroke_count": float(len(added)),
            "local_stroke_width": float(np.median([widths[i] for i in sources])),
            "minimum_added_unique_fraction": float(
                min(
                    np.count_nonzero(
                        (_small(stroke) >= 128)
                        & ~(
                            _small(np.maximum.reduce(layers + tuple(added[:i] + added[i + 1 :])))
                            > 8
                        )
                    )
                    / max(1, area)
                    for i, stroke in enumerate(added)
                )
            ),
        },
    )


def _break(
    layers: tuple[ByteArray, ...],
    index: int,
    width: float,
    scale: float,
    random: np.random.Generator,
) -> _Proposal | None:
    if len(layers) < 2:
        return None
    source = layers[index]
    rest = _rest(layers, index)
    core = source >= 128
    # Include one-pixel abutments, but never choose a free-standing segment.
    contact = core & (cv2.dilate((rest >= 128).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    count, regions, stats, _ = cv2.connectedComponentsWithStats(
        contact.astype(np.uint8), connectivity=8
    )
    if count < 2:
        return None
    # Prefer substantial junctions over a tangential contact at a tiny serif tip.
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    chosen = int(order[int(random.integers(min(3, len(order))))])
    junction = np.argwhere(regions == chosen)
    center = junction.mean(axis=0)
    skeleton = np.argwhere(skeletonize(core)).astype(float)  # type: ignore[no-untyped-call]
    neighbors = skeleton[np.linalg.norm(skeleton - center, axis=1) <= max(width * 3, scale * 0.18)]
    if len(neighbors) < 3:
        return None
    _, vectors = np.linalg.eigh(np.cov(neighbors, rowvar=False))
    tangent = vectors[:, -1]
    normal = np.array([-tangent[1], tangent[0]])
    overlap_length = float(np.ptp((junction - center) @ tangent)) + 1
    gap = max(width * 1.2, overlap_length + width * 0.9, scale * 0.07)
    gap *= float(random.uniform(1.0, 1.35))
    yy, xx = np.indices(source.shape)
    offsets = np.stack((yy - center[0], xx - center[1]), axis=-1)
    cut = (np.abs(offsets @ tangent) <= gap / 2) & (np.abs(offsets @ normal) <= width * 1.1)
    removed_contact_fraction = float(np.count_nonzero(cut & (regions == chosen)) / len(junction))
    if removed_contact_fraction < 0.75:
        return None
    edited_stroke = source.copy()
    edited_stroke[cut] = 0
    if np.count_nonzero(edited_stroke >= 128) < np.count_nonzero(core) * 0.35:
        return None
    edited = tuple(edited_stroke if i == index else layer for i, layer in enumerate(layers))
    before, after = [_small(np.maximum.reduce(x)) > 8 for x in (layers, edited)]
    before_components, before_euler = _topology(before)
    after_components, after_euler = _topology(after)
    if not (after_components > before_components or after_euler > before_euler):
        return None
    return _Proposal(
        edited,
        (index,),
        {
            "local_stroke_width": width,
            "gap_length": gap,
            "junction_removed_fraction": removed_contact_fraction,
            "input_component_delta": float(after_components - before_components),
            "input_euler_delta": float(after_euler - before_euler),
            "broken_stroke_count": 1.0,
        },
    )


def _bridge(planner: BridgePlanner, attempt: int, mode: str | None) -> _Proposal | None:
    proposal = planner.propose(attempt, mode)
    if proposal is None:
        return None
    return _Proposal(proposal.layers, proposal.selected, proposal.metrics, proposal.mode)


def _visible(original: ByteArray, candidate: ByteArray) -> dict[str, float] | None:
    changed = original != candidate
    area = max(1, np.count_nonzero(original > 8))
    fraction = float(changed.sum() / area)
    if not 0.025 <= fraction <= 0.60:
        return None
    a, b = _small(original), _small(candidate)
    small_area = max(1, np.count_nonzero(a >= 128))
    changed_pixels = int(np.count_nonzero(np.abs(a.astype(float) - b) >= 32))
    foreground_pixels = int(np.count_nonzero((a > 8) != (b > 8)))
    edge_pixels = int(np.count_nonzero(cv2.Canny(a, 50, 150) != cv2.Canny(b, 50, 150)))
    if changed_pixels < max(8, small_area * 0.025) or min(foreground_pixels, edge_pixels) < 8:
        return None
    # Retain a substantial body, not only dust left after a destructive operation.
    _, _, stats, _ = cv2.connectedComponentsWithStats((b > 8).astype(np.uint8), connectivity=8)
    if len(stats) < 2 or stats[1:, cv2.CC_STAT_AREA].max() < small_area * 0.25:
        return None
    return {
        "changed_fraction": fraction,
        "input_changed_fraction": float(changed_pixels / small_area),
        "input_changed_pixels": float(changed_pixels),
        "input_foreground_changed_pixels": float(foreground_pixels),
        "input_edge_changed_pixels": float(edge_pixels),
    }


def corrupt_stroke_layers(
    layers: tuple[ByteArray, ...], operator: str, seed: int, *, bridge_mode: str | None = None
) -> StrokeCorruptionResult:
    """Return one geometrically visible REVIEW proposal, or skip after 96 tries.

    Square uint8 alpha layers must share a canvas of at least 32 pixels. Empty
    ink is inapplicable, malformed arguments are ValueError. All returned layers
    own their data; neither inputs nor NumPy's global random state are modified.
    ``selected_stroke_indices`` always refers to original layers (for addition,
    it records the source of each transformed copy, and may repeat an index).
    """
    if operator not in OPERATORS:
        raise ValueError(f"unknown stroke operator: {operator}")
    if bridge_mode not in (None, "close_opening", "block_gap"):
        raise ValueError("unknown bridge mode")
    if bridge_mode is not None and operator != "bridge":
        raise ValueError("bridge_mode is only supported for bridge")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a nonempty tuple")
    shape = layers[0].shape if isinstance(layers[0], np.ndarray) else ()
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 32:
        raise ValueError("stroke layers must be square canvases of at least 32 pixels")
    if any(
        not isinstance(x, np.ndarray) or x.dtype != np.uint8 or x.shape != shape for x in layers
    ):
        raise ValueError("stroke layers must be same-shape uint8 alpha arrays")
    original = np.maximum.reduce(layers)
    points = np.argwhere(original >= 128)
    if not len(points):
        raise CorruptionNotApplicable("no visible stroke core")
    scale = float(np.ptp(points, axis=0).max() + 1)
    widths = [_width(x) for x in layers]
    weights = np.asarray([np.count_nonzero(x >= 128) ** 1.5 for x in layers], dtype=float)
    weights /= weights.sum()
    random = np.random.default_rng(seed)
    planner = BridgePlanner(layers, widths, scale, seed) if operator == "bridge" else None
    for attempt in range(96):
        index = int(random.choice(len(layers), p=weights))
        if operator == "erase_segment":
            # Try the substantial strokes first; small dots remain eligible if
            # those removals cannot preserve a visible character body.
            if attempt < 8:
                substantial = np.flatnonzero(weights >= weights.max() * 0.75)
                index = int(random.choice(substantial))
            proposal = _erase(layers, index, widths[index])
        elif operator == "component_shift":
            proposal = _shift(layers, index, widths[index], scale, random)
        elif operator == "add_stroke":
            proposal = _add(layers, widths, scale, random)
        elif operator == "break_stroke":
            proposal = _break(layers, index, widths[index], scale, random)
        else:
            assert planner is not None
            proposal = _bridge(planner, attempt, bridge_mode)
        if proposal is None:
            continue
        candidate = np.maximum.reduce(proposal.layers)
        visibility = _visible(original, candidate)
        if visibility is None:
            continue
        changed = (original != candidate).astype(np.uint8)
        return StrokeCorruptionResult(
            image=np.repeat(candidate[:, :, None], 3, axis=2),
            changed_mask=changed,
            changed_pixels=int(changed.sum()),
            operator=operator,
            metrics={
                **proposal.metrics,
                **visibility,
                "glyph_scale": scale,
                "attempts": float(attempt + 1),
            },
            selected_stroke_indices=proposal.selected,
            edited_layers=tuple(layer.copy() for layer in proposal.layers),
            bridge_mode=proposal.bridge_mode,
        )
    raise CorruptionNotApplicable(f"{operator}: no valid stroke-layer proposal after 96 attempts")
