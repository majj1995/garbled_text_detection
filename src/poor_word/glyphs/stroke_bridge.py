"""Conservative blank-space bridge subtypes; geometry never establishes illegality.

All planning coordinates are fixed on the *original* 96-pixel input. Horizontal
and vertical scan gaps select ink banks, a gate, and a maximal original passage.
Closing an opening uses a one-pixel virtual gate to identify the original exterior
pocket before drawing any ink. Blocking a gap uses the full original bank-to-bank
scan run, never a convenient post-edit crop. Foreground uses 8-connectivity and
background uses the complementary 4-connectivity, at both >8 and >=128.
"""

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize

ByteArray = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class BridgeProposal:
    layers: tuple[ByteArray, ...]
    selected: tuple[int, ...]
    metrics: dict[str, float]
    mode: str


@dataclass(frozen=True)
class _Gate:
    axis: int
    row: int
    left: int
    right: int
    width: float
    selected: tuple[int, ...]

    def point(self, across: int, along: int | None = None) -> tuple[int, int]:
        row = self.row if along is None else along
        return (row, across) if self.axis == 0 else (across, row)


@dataclass(frozen=True)
class _Passage:
    gate: _Gate
    region: BoolArray
    first: tuple[int, int]
    last: tuple[int, int]
    span: int
    thickness: float


def _small(alpha: ByteArray) -> ByteArray:
    return np.asarray(cv2.resize(alpha, (96, 96), interpolation=cv2.INTER_AREA), dtype=np.uint8)


def _regions(background: BoolArray) -> NDArray[np.int32]:
    return np.asarray(
        cv2.connectedComponents(background.astype(np.uint8), connectivity=4)[1], dtype=np.int32
    )


def _exterior(background: BoolArray) -> BoolArray:
    labels = _regions(background)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return np.isin(labels, border[border != 0])


def _roi(region: BoolArray) -> dict[str, float]:
    yy, xx = np.nonzero(region)
    return {
        "roi_x0_96": float(xx.min()),
        "roi_y0_96": float(yy.min()),
        "roi_x1_96": float(xx.max() + 1),
        "roi_y1_96": float(yy.max() + 1),
    }


def _radius(region: BoolArray) -> float:
    return float(cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5).max())


class BridgePlanner:
    """Precompute original gap geometry once; each proposal checks one candidate."""

    def __init__(
        self, layers: tuple[ByteArray, ...], widths: list[float], scale: float, seed: int
    ) -> None:
        self.layers = layers
        self.original = np.asarray(np.maximum.reduce(layers), dtype=np.uint8)
        self.small = _small(self.original)
        self.ink = self.small >= 128
        self.scale = scale * 96 / self.original.shape[0]
        small_layers = tuple(_small(layer) for layer in layers)
        self.layer_cores = tuple(layer >= 128 for layer in small_layers)
        # Measure the actual banks near each gate, not the median of a complete
        # stroke whose remote bottom, hook, or flare may be many times thicker.
        geometry = []
        for alpha in small_layers:
            core = alpha >= 128
            skeleton = np.asarray(skeletonize(core), dtype=np.uint8)  # type: ignore[no-untyped-call]
            points = np.argwhere(skeleton)
            distances = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
            neighbors = cv2.filter2D(skeleton, -1, np.ones((3, 3), np.uint8)) - skeleton
            endpoints = np.argwhere((skeleton > 0) & (neighbors <= 1))
            geometry.append((points, distances[points[:, 0], points[:, 1]] * 2, endpoints))
        self.endpoints = tuple(item[2] for item in geometry)
        gates: list[_Gate] = []
        for axis in (0, 1):
            view = self.ink if axis == 0 else self.ink.T
            for row in range(2, 94):
                occupied = np.flatnonzero(view[row])
                for left, right in pairwise(occupied):
                    if right - left < 4:
                        continue
                    start = (row, int(left)) if axis == 0 else (int(left), row)
                    end = (row, int(right)) if axis == 0 else (int(right), row)
                    selected = tuple(
                        i for i, x in enumerate(small_layers) if x[start] >= 128 or x[end] >= 128
                    )
                    if not selected:
                        continue
                    local_widths = []
                    for point in (start, end):
                        for i in selected:
                            points, values, _ = geometry[i]
                            if small_layers[i][point] < 128 or not len(points):
                                continue
                            distance = np.linalg.norm(points - point, axis=1)
                            nearby = distance <= float(distance.min()) + 2.0
                            local_widths.append(float(np.median(values[nearby])))
                    width = float(np.median(local_widths))
                    if right - left > min(self.scale * 0.80, width * 8):
                        continue
                    gates.append(_Gate(axis, row, int(left), int(right), width, selected))
        self.close_gates = [gate for gate in gates if self._near_mouth(gate)]
        self.passages = self._passages(gates)
        rng = np.random.default_rng(seed)
        self.close_gates = [
            self.close_gates[int(i)] for i in rng.permutation(len(self.close_gates))
        ]
        self.passages = [self.passages[int(i)] for i in rng.permutation(len(self.passages))]

    def _near_mouth(self, gate: _Gate) -> bool:
        if gate.right - gate.left < max(4, gate.width * 0.70):
            return False
        # A true bank tip must be near a source-stroke skeleton endpoint. A
        # thickness step or displaced sloping wall is not an endpoint merely
        # because the fixed pixel column becomes blank farther along the wall.
        for point in (gate.point(gate.left), gate.point(gate.right)):
            if not any(
                len(self.endpoints[i])
                and self.layer_cores[i][point]
                and np.linalg.norm(self.endpoints[i] - point, axis=1).min()
                <= max(3, gate.width * 1.25)
                for i in gate.selected
            ):
                return False
        # A bar placed halfway down an open U is not its mouth. At least one
        # outward normal direction must leave both original banks near their tips.
        distance = max(3, round(gate.width * 1.5))
        radius = max(2, round(gate.width * 0.70))
        yy, xx = np.indices(self.ink.shape)
        for sign in (-1, 1):
            outside_row = gate.row + sign * distance
            if not 1 <= outside_row < 95:
                continue
            first, last = gate.point(gate.left, outside_row), gate.point(gate.right, outside_row)
            # Probe neighborhoods, not just the old bank columns: sloping walls
            # move sideways while continuing and must not masquerade as tips.
            first_neighborhood = (yy - first[0]) ** 2 + (xx - first[1]) ** 2 <= radius**2
            last_neighborhood = (yy - last[0]) ** 2 + (xx - last[1]) ** 2 <= radius**2
            if not np.any(self.ink & first_neighborhood) and not np.any(
                self.ink & last_neighborhood
            ):
                return True
        return False

    def _passages(self, gates: list[_Gate]) -> list[_Passage]:
        eligible = [x for x in gates if 3 <= x.right - x.left - 1 <= x.width * 1.35]
        runs: list[list[_Gate]] = []
        for gate in eligible:
            matches = [
                run
                for run in runs
                if run[-1].axis == gate.axis
                and run[-1].row == gate.row - 1
                and abs(run[-1].left - gate.left) <= 2
                and abs(run[-1].right - gate.right) <= 2
            ]
            if matches:
                min(matches, key=lambda x: abs(x[-1].left - gate.left)).append(gate)
            else:
                runs.append([gate])
        passages = []
        for run in runs:
            gate = run[len(run) // 2]
            span = len(run)
            # Thickness follows the local banks, not an unrelated whole-glyph
            # bounding box. Global significance is checked on the actual output.
            thickness = max(gate.width * 1.4, span * 0.35)
            # Long uniform parallel rails need an oversized patch to consume a
            # meaningful fraction: skip them, rather than insert a short bridge.
            if span < max(8, gate.width * 1.6) or thickness > gate.width * 1.8:
                continue
            region = np.zeros((96, 96), np.bool_)
            for item in run:
                if gate.axis == 0:
                    region[item.row, item.left + 1 : item.right] = True
                else:
                    region[item.left + 1 : item.right, item.row] = True
            first_gate, last_gate = run[1], run[-2]
            first = first_gate.point((first_gate.left + first_gate.right) // 2)
            last = last_gate.point((last_gate.left + last_gate.right) // 2)
            passages.append(_Passage(gate, region, first, last, span, thickness))
        return passages

    def _alpha(self, gate: _Gate, thickness: float) -> ByteArray:
        factor = self.original.shape[0] / 96
        # Include two original-input ink pixels at both banks; endpoint caps
        # cannot rely on faint antialiasing to seal at the >=128 threshold.
        start, end = gate.point(gate.left - 2), gate.point(gate.right + 2)
        alpha = np.zeros_like(self.original)
        cv2.line(
            alpha,
            (round(start[1] * factor), round(start[0] * factor)),
            (round(end[1] * factor), round(end[0] * factor)),
            (255,),
            max(2, round(thickness * factor)),
            lineType=cv2.LINE_AA,
        )
        return alpha

    def _metrics(self, gate: _Gate, region: BoolArray) -> dict[str, float]:
        start, end = gate.point(gate.left), gate.point(gate.right)
        return {
            **_roi(region),
            "gate_start_x_96": float(start[1]),
            "gate_start_y_96": float(start[0]),
            "gate_end_x_96": float(end[1]),
            "gate_end_y_96": float(end[0]),
            "local_stroke_width_96": gate.width,
            "local_stroke_width": gate.width * self.original.shape[0] / 96,
            "gate_width_96": float(gate.right - gate.left - 1),
            "added_stroke_count": 1.0,
        }

    def _close(self, gate: _Gate) -> BridgeProposal | None:
        # Plan with a zero-width mathematical gate on original geometry. This
        # selects an original exterior pocket and its ROI before the actual edit.
        virtual = np.zeros((96, 96), np.uint8)
        a, b = gate.point(gate.left), gate.point(gate.right)
        cv2.line(virtual, (a[1], a[0]), (b[1], b[0]), (1,), 1)
        original_blank = self.small < 9
        original_exterior = _exterior(original_blank)
        pockets = original_exterior & ~_exterior(original_blank & ~(virtual > 0)) & ~(virtual > 0)
        regions = _regions(pockets)
        counts = np.bincount(regions.ravel())
        if len(counts) < 2:
            return None
        index = int(np.argmax(counts[1:])) + 1
        region = regions == index
        area = int(region.sum())
        if area < max(48, self.ink.sum() * 0.06, gate.width**2 * 1.5):
            return None
        if _radius(region) < max(3, gate.width * 0.75):
            return None
        distances = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
        seed = tuple(int(x) for x in np.unravel_index(distances.argmax(), region.shape))
        alpha = self._alpha(gate, max(3, gate.width * 0.90))
        candidate = _small(np.maximum(self.original, alpha))
        retained_areas = []
        retained_widths = []
        for threshold in (9, 128):
            before_blank, after_blank = self.small < threshold, candidate < threshold
            if not _exterior(before_blank)[seed] or not after_blank[seed]:
                return None
            after_regions = _regions(after_blank)
            retained = (after_regions == after_regions[seed]) & region
            if _exterior(after_blank)[seed] or retained.sum() < max(48, area * 0.60):
                return None
            radius = _radius(retained)
            if radius < max(3, gate.width * 0.65):
                return None
            retained_areas.append(float(retained.sum()))
            retained_widths.append(radius * 2)
        metrics = {
            **self._metrics(gate, region),
            "original_region_area_96": float(area),
            "retained_region_area_96": min(retained_areas),
            "retained_region_width_96": min(retained_widths),
            "retained_region_fraction": min(retained_areas) / area,
            "region_seed_x_96": float(seed[1]),
            "region_seed_y_96": float(seed[0]),
        }
        return BridgeProposal((*self.layers, alpha), gate.selected, metrics, "close_opening")

    def _block(self, passage: _Passage) -> BridgeProposal | None:
        gate, region = passage.gate, passage.region
        alpha = self._alpha(gate, passage.thickness)
        candidate = _small(np.maximum(self.original, alpha))
        fractions, lengths = [], []
        for threshold in (9, 128):
            before = (self.small < threshold) & region
            after = (candidate < threshold) & region
            labels = _regions(before)
            if not labels[passage.first] or labels[passage.first] != labels[passage.last]:
                return None
            labels_after = _regions(after)
            if labels_after[passage.first] and (
                labels_after[passage.first] == labels_after[passage.last]
            ):
                return None
            consumed = before & ~after
            fraction = float(consumed.sum() / max(1, before.sum()))
            # Whole cross-sections disappear over substantial original-run length.
            axis = 1 if gate.axis == 0 else 0
            closed_sections = np.any(before, axis=axis) & ~np.any(after, axis=axis)
            length = float(closed_sections.sum())
            if fraction < 0.30 or length < max(gate.width, passage.span * 0.30):
                return None
            fractions.append(fraction)
            lengths.append(length)
        metrics = {
            **self._metrics(gate, region),
            "original_gap_area_96": float(region.sum()),
            "gap_consumed_fraction": min(fractions),
            "blocked_length_96": min(lengths),
            "original_gap_length_96": float(passage.span),
            "blocked_length_fraction": min(lengths) / passage.span,
            "passage_start_x_96": float(passage.first[1]),
            "passage_start_y_96": float(passage.first[0]),
            "passage_end_x_96": float(passage.last[1]),
            "passage_end_y_96": float(passage.last[0]),
        }
        return BridgeProposal((*self.layers, alpha), gate.selected, metrics, "block_gap")

    def propose(self, attempt: int, mode: str | None) -> BridgeProposal | None:
        selected_mode = mode or ("close_opening" if attempt % 2 == 0 else "block_gap")
        index = attempt if mode is not None else attempt // 2
        if selected_mode == "close_opening":
            if index >= len(self.close_gates):
                return None
            return self._close(self.close_gates[index])
        if index >= len(self.passages):
            return None
        return self._block(self.passages[index])
