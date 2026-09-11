"""Deterministic, conservative affine views; no label-changing corruption operators."""

import hashlib
import json
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]
POLICY_VERSION = "glyph-affine-v1"
MAX_ATTEMPTS = 4


class UnsafeAugmentation(ValueError):
    """A proposed warp failed a pixel-level preservation check; use another or the original."""


@dataclass(frozen=True)
class AugmentationTrace:
    seed: int
    applied: bool
    attempts: int
    rejection_reasons: tuple[str, ...]
    parameters: dict[str, float]


def augmentation_policy() -> dict[str, object]:
    return {
        "version": POLICY_VERSION,
        "rotation_degrees": [-2.0, 2.0],
        "scale": [0.97, 1.03],
        "translation_fraction": [-2 / 128, 2 / 128],
        "max_attempts": MAX_ATTEMPTS,
        "interpolation": "opencv_INTER_LINEAR",
        "seed_key": ["version", "training_seed", "epoch", "step", "draw", "sample_id"],
        "guards": ["clipping", "topology_96", "ink_mass_96", "reference_edit_visibility_96"],
        "ink_mass_ratio": [0.9, 1.1],
        "min_edit_retention": 0.85,
        "fallback": "original_verified_image",
        "scope": "optimization_batches_only",
        "semantic_guarantee": False,
    }


def augmentation_seed(seed: int, epoch: int, step: int, draw: int, sample_id: str) -> int:
    if any(value < 0 for value in (seed, epoch, step, draw)) or not sample_id:
        raise ValueError("augmentation requires non-negative coordinates and a sample identity")
    key = json.dumps([POLICY_VERSION, seed, epoch, step, draw, sample_id], ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _validate_gray(gray: GrayImage) -> None:
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 8:
        raise ValueError("augmentation requires a uint8 grayscale image of at least 8x8")
    if not np.any(gray > 8):
        raise ValueError("augmentation requires visible foreground")


def _model_gray(gray: GrayImage) -> GrayImage:
    return np.asarray(cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA), dtype=np.uint8)


def _topology(gray: GrayImage) -> tuple[int, int]:
    # Match the actual mask channel's >8 support. Reject newly joined/broken components
    # or opened/closed holes, including small regions rather than dropping them as noise.
    binary = (gray > 8).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return (0, 0)
    holes = int(np.count_nonzero(hierarchy[0, :, 3] >= 0))
    return len(contours) - holes, holes


def _warp_checked(
    gray: GrayImage, matrix: NDArray[np.float64], scale: float
) -> tuple[GrayImage, GrayImage, GrayImage]:
    height, width = gray.shape
    rows, cols = np.nonzero(gray > 0)
    corners = np.asarray(
        [
            [cols.min(), rows.min(), 1],
            [cols.max(), rows.min(), 1],
            [cols.min(), rows.max(), 1],
            [cols.max(), rows.max(), 1],
        ],
        dtype=np.float64,
    )
    mapped = corners @ matrix.T
    # One-pixel margin includes the bilinear interpolation footprint.
    if (mapped.min(axis=0) < 1).any() or (mapped.max(axis=0) > [width - 2, height - 2]).any():
        raise UnsafeAugmentation("clipping")
    warped = np.asarray(
        cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        ),
        dtype=np.uint8,
    )
    before, after = _model_gray(gray), _model_gray(warped)
    if _topology(before) != _topology(after):
        raise UnsafeAugmentation("topology")
    expected_ink = float(before.sum()) * scale**2
    if not 0.9 * expected_ink <= float(after.sum()) <= 1.1 * expected_ink:
        raise UnsafeAugmentation("ink_mass")
    return warped, before, after


def apply_affine(
    gray: GrayImage, parameters: dict[str, float], *, reference: GrayImage | None = None
) -> GrayImage:
    """Apply a bounded proposal, checking both observed ink and any erased reference ink.

    Reference is the original legal glyph aligned to this exact anomalous raster, not a
    calibration image. Pixel/topology checks are conservative heuristics, not human labels.
    """
    _validate_gray(gray)
    if reference is not None:
        _validate_gray(reference)
        if reference.shape != gray.shape:
            raise ValueError("reference and image dimensions differ")
    angle, scale, tx, ty = (parameters[k] for k in ("angle", "scale", "translate_x", "translate_y"))
    height, width = gray.shape
    if (
        not np.all(np.isfinite([angle, scale, tx, ty]))
        or abs(angle) > 2.0
        or not 0.97 <= scale <= 1.03
        or abs(tx) > width * 2 / 128
        or abs(ty) > height * 2 / 128
    ):
        raise ValueError("affine parameters exceed the fixed conservative policy")
    matrix = np.asarray(
        cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale), dtype=np.float64
    )
    matrix[:, 2] += (tx, ty)
    warped, before, after = _warp_checked(gray, matrix, scale)
    if reference is not None:
        _, ref_before, ref_after = _warp_checked(reference, matrix, scale)
        old_delta = np.abs(before.astype(np.int16) - ref_before.astype(np.int16))
        new_delta = np.abs(after.astype(np.int16) - ref_after.astype(np.int16))
        # Measure the actual appearance difference, not merely the warped edit mask:
        # a mask can survive even when interpolation erased the anomaly it described.
        old_visible, new_visible = int((old_delta >= 32).sum()), int((new_delta >= 32).sum())
        if (
            old_visible == 0
            or new_visible < max(1, 0.85 * scale**2 * old_visible)
            or float(new_delta.sum()) < 0.85 * scale**2 * float(old_delta.sum())
        ):
            raise UnsafeAugmentation("edit_visibility")
    return warped


def augment_affine(
    gray: GrayImage, *, seed: int, reference: GrayImage | None = None
) -> tuple[GrayImage, AugmentationTrace]:
    _validate_gray(gray)
    rng = np.random.default_rng(seed)
    height, width = gray.shape
    reasons: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        parameters = {
            "angle": float(rng.uniform(-2.0, 2.0)),
            "scale": float(rng.uniform(0.97, 1.03)),
            "translate_x": float(rng.uniform(-width * 2 / 128, width * 2 / 128)),
            "translate_y": float(rng.uniform(-height * 2 / 128, height * 2 / 128)),
        }
        try:
            image = apply_affine(gray, parameters, reference=reference)
            if np.array_equal(image, gray):
                raise UnsafeAugmentation("unchanged")
        except UnsafeAugmentation as error:
            reasons.append(str(error))
            continue
        return image, AugmentationTrace(seed, True, attempt, tuple(reasons), parameters)
    return gray.copy(), AugmentationTrace(seed, False, MAX_ATTEMPTS, tuple(reasons), {})
