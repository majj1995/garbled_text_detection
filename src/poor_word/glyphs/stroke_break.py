"""Conservative independent-stroke interior breaks; geometry is REVIEW evidence only."""

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize

ByteArray = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]


def _skeleton(core: BoolArray) -> BoolArray:
    return np.asarray(skeletonize(core), dtype=np.bool_)  # type: ignore[no-untyped-call]


def _parts(
    before: ByteArray, after: ByteArray, width: float, units: float, threshold: int
) -> tuple[NDArray[np.int32], dict[str, float]] | None:
    """Require one continuous stroke to become two area/length-supported bodies."""
    original = before >= threshold
    before_count = int(cv2.connectedComponents(original.astype(np.uint8), connectivity=8)[0]) - 1
    if before_count != 1:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (after >= threshold).astype(np.uint8), connectivity=8
    )
    source_skeleton = _skeleton(original)
    minimum_area = max(16 * units**2, width**2 * 0.8, original.sum() * 0.12)
    minimum_length = max(8 * units, width * 1.5)
    significant = []
    metrics = {"before_parts": 1.0}
    for index in range(1, count):
        part = labels == index
        area = float(stats[index, cv2.CC_STAT_AREA])
        length = float(_skeleton(part).sum())
        support = float(np.count_nonzero(source_skeleton & part))
        if area < minimum_area or length < minimum_length or support < minimum_length * 0.75:
            continue
        significant.append(index)
        number = len(significant)
        metrics.update(
            {
                f"part_{number}_area": area,
                f"part_{number}_length": length,
                f"part_{number}_support": support,
            }
        )
    # Do not let a tiny island qualify as a second body, or extra fragments hide
    # behind the significant-parts count. A bounded skip is preferable to dust.
    if len(significant) != 2 or count != 3:
        return None
    metrics["after_parts"] = 2.0
    return np.asarray(labels, dtype=np.int32), metrics


def _visible_gap(
    composite: ByteArray, parts: NDArray[np.int32], roi: BoolArray, width: float, threshold: int
) -> float | None:
    """Opposite source banks must remain locally separated after ALL layers merge.

    Global components/hole counts are intentionally irrelevant: a box's other
    strokes can reconnect its banks far away while the local body break is clear.
    """
    local_ink = (composite >= threshold) & roi
    _, labels = cv2.connectedComponents(local_ink.astype(np.uint8), connectivity=8)
    banks = []
    for index in (1, 2):
        support = (parts == index) & roi
        if support.sum() < max(4, width**2 * 0.3):
            return None
        regions = np.unique(labels[support & local_ink])
        regions = regions[regions > 0]
        if not len(regions):
            return None
        banks.append(np.isin(labels, regions))
    if np.any(banks[0] & banks[1]):
        return None
    # Bank-to-bank distance is clear space only when it accounts for ALL local
    # foreground. Detached third ink can occupy almost the entire interval while
    # leaving one-pixel slits at both ends; it must never inflate visible clearance.
    # Conservatively skip any extra local region, even one off the direct corridor.
    if np.any(local_ink & ~(banks[0] | banks[1])):
        return None
    distance = cv2.distanceTransform((~banks[0]).astype(np.uint8), cv2.DIST_L2, 5)
    gap = float(distance[banks[1]].min()) - 1.0
    return gap if gap >= max(3.0, width * 0.65) else None


def propose_break(
    source: ByteArray, rest: ByteArray, random: np.random.Generator
) -> tuple[ByteArray, dict[str, float]] | None:
    """One bounded candidate on a selected source stroke, without mutating inputs."""
    core = source >= 128
    skeleton = _skeleton(core)
    points = np.argwhere(skeleton).astype(float)
    if len(points) < 16:
        return None
    distance = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    degree = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    endpoints = np.argwhere(skeleton & (degree == 2))
    # Sample source geometry, never intersection centers or absolute quadrants.
    # Curved neighborhoods get priority; straight interior bodies remain valid.
    candidates = []
    for candidate in random.choice(len(points), size=min(8, len(points)), replace=False):
        center = points[candidate]
        width = float(distance[int(center[0]), int(center[1])] * 2)
        if len(endpoints) and np.linalg.norm(endpoints - center, axis=1).min() < width * 2:
            continue
        neighbors = points[np.linalg.norm(points - center, axis=1) <= width * 2.5]
        if len(neighbors) < 5:
            continue
        values, vectors = np.linalg.eigh(np.cov(neighbors, rowvar=False))
        curvature = float(values[0] / max(values[1], 1.0))
        candidates.append((curvature, center, width, vectors[:, -1]))
    if not candidates:
        return None
    _, center, width, tangent = max(candidates, key=lambda x: x[0])
    normal = np.array([-tangent[1], tangent[0]])
    factor = source.shape[0] / 96.0
    gap = max(width * 1.35, 5 * factor) * float(random.uniform(1.0, 1.3))
    yy, xx = np.indices(source.shape)
    offsets = np.stack((yy - center[0], xx - center[1]), axis=-1)
    along, across = offsets @ tangent, offsets @ normal
    cut = (np.abs(along) <= gap / 2) & (np.abs(across) <= width * 1.5)
    edited = source.copy()
    edited[cut] = 0
    metrics = {
        "local_stroke_width": width,
        "local_stroke_width_96": width / factor,
        "break_center_x_96": (float(center[1]) + 0.5) / factor - 0.5,
        "break_center_y_96": (float(center[0]) + 0.5) / factor - 0.5,
        "gap_length": gap,
        "gap_length_96": gap / factor,
        "broken_stroke_count": 1.0,
    }
    for threshold in (9, 128):
        split = _parts(source, edited, width, factor, threshold)
        if split is None:
            return None
        metrics.update(
            {f"break_{key}_native_t{threshold}": value for key, value in split[1].items()}
        )
    source96, edited96, composite96 = [
        np.asarray(cv2.resize(x, (96, 96), interpolation=cv2.INTER_AREA), dtype=np.uint8)
        for x in (source, edited, np.maximum(edited, rest))
    ]
    roi = (np.abs(along) <= gap / 2 + width * 1.5) & (np.abs(across) <= width * 2)
    roi96 = cv2.resize(roi.astype(np.uint8), (96, 96), interpolation=cv2.INTER_NEAREST) > 0
    for threshold in (9, 128):
        split = _parts(source96, edited96, width / factor, 1.0, threshold)
        if split is None:
            return None
        labels, evidence = split
        visible_gap = _visible_gap(composite96, labels, roi96, width / factor, threshold)
        if visible_gap is None:
            return None
        metrics.update({f"break_{key}_96_t{threshold}": value for key, value in evidence.items()})
        metrics[f"break_visible_gap_96_t{threshold}"] = visible_gap
    return edited, metrics
