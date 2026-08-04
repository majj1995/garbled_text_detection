import base64
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray

from poor_word.domain import BoundingBox
from poor_word.ocr.types import OcrAudit, OcrCandidate, OcrCharacter, OcrLine, OcrResult

OcrImage = str | NDArray[np.uint8]


class PaddleBackend(Protocol):
    def predict(self, image: OcrImage) -> dict[str, object]: ...


class AuditablePaddleBackend(PaddleBackend, Protocol):
    def capabilities(self) -> dict[str, object]: ...


class HttpPaddleBackend:
    def __init__(self, endpoint: str, timeout_seconds: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _image_bytes(image: OcrImage) -> bytes:
        if isinstance(image, str):
            return Path(image).read_bytes()
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("could not encode OCR image as PNG")
        return encoded.tobytes()

    def predict(self, image: OcrImage) -> dict[str, object]:
        payload = {"file": base64.b64encode(self._image_bytes(image)).decode("ascii")}
        response = httpx.post(
            f"{self.endpoint}/predict",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return cast(dict[str, object], response.json())

    def capabilities(self) -> dict[str, object]:
        response = httpx.get(f"{self.endpoint}/health", timeout=self.timeout_seconds)
        response.raise_for_status()
        return cast(dict[str, object], response.json())


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    return list(value)


def _as_int(value: object, field: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be numeric")
    return int(value)


def _as_float(value: object, field: str) -> float:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _box(value: object) -> BoundingBox:
    coordinates = _sequence(value, "character.box")
    if len(coordinates) != 4:
        raise ValueError("character.box must contain four coordinates")
    return BoundingBox(
        x0=_as_int(coordinates[0], "character.box[0]"),
        y0=_as_int(coordinates[1], "character.box[1]"),
        x1=_as_int(coordinates[2], "character.box[2]"),
        y1=_as_int(coordinates[3], "character.box[3]"),
    )


def _candidate(value: object) -> OcrCandidate:
    pair = _sequence(value, "character.top_k item")
    if len(pair) != 2:
        raise ValueError("top_k items must contain text and confidence")
    return OcrCandidate(
        text=str(pair[0]), confidence=_as_float(pair[1], "candidate.confidence")
    )


def _character(value: object) -> OcrCharacter:
    raw = _mapping(value, "character")
    raw_logits = raw.get("logits")
    logits = (
        tuple(
            _as_float(item, "character.logits item")
            for item in _sequence(raw_logits, "character.logits")
        )
        if raw_logits is not None
        else None
    )
    return OcrCharacter(
        text=str(raw.get("text", "")),
        box=_box(raw.get("box")),
        confidence=(
            _as_float(raw["confidence"], "character.confidence")
            if raw.get("confidence") is not None
            else None
        ),
        top_k=tuple(_candidate(item) for item in _sequence(raw.get("top_k", []), "top_k")),
        logits=logits,
    )


def _line(value: object) -> OcrLine:
    raw = _mapping(value, "line")
    polygon = tuple(
        (
            _as_float(point[0], "line.polygon x"),
            _as_float(point[1], "line.polygon y"),
        )
        for item in _sequence(raw.get("polygon"), "line.polygon")
        if len(point := _sequence(item, "line.polygon point")) == 2
    )
    return OcrLine(
        text=str(raw.get("text", "")),
        confidence=_as_float(raw.get("confidence", 0.0), "line.confidence"),
        polygon=polygon,
        characters=tuple(
            _character(item) for item in _sequence(raw.get("characters", []), "characters")
        ),
    )


class PaddleV5Adapter:
    def __init__(
        self,
        backend: PaddleBackend | None = None,
        *,
        endpoint: str | None = None,
    ) -> None:
        if backend is None:
            service_url: str = (
                endpoint
                if endpoint is not None
                else os.environ.get("POOR_WORD_OCR_URL", "http://127.0.0.1:8765")
            )
            backend = HttpPaddleBackend(service_url)
        self.backend = backend

    def recognize(self, image: OcrImage) -> OcrResult:
        raw = self.backend.predict(image)
        stage_raw = _mapping(raw.get("stage_ms", {}), "stage_ms")
        return OcrResult(
            model_name=str(raw.get("model_name", "unknown")),
            lines=tuple(_line(item) for item in _sequence(raw.get("lines", []), "lines")),
            stage_ms={
                str(key): _as_float(value, f"stage_ms.{key}")
                for key, value in stage_raw.items()
            },
        )

    def audit(
        self,
        images: tuple[OcrImage, ...],
        *,
        warmup: int = 10,
        runs: int = 30,
    ) -> OcrAudit:
        if not images:
            raise ValueError("audit requires at least one image")
        if warmup < 0 or runs <= 0:
            raise ValueError("warmup must be non-negative and runs must be positive")
        if not hasattr(self.backend, "capabilities"):
            raise ValueError("OCR backend does not expose capability metadata")

        backend = cast(AuditablePaddleBackend, self.backend)
        capabilities = backend.capabilities()
        for index in range(warmup):
            self.recognize(images[index % len(images)])

        durations: list[float] = []
        results: list[OcrResult] = []
        for index in range(runs):
            started = time.perf_counter()
            result = self.recognize(images[index % len(images)])
            durations.append((time.perf_counter() - started) * 1000.0)
            results.append(result)

        characters = [
            character
            for result in results
            for line_result in result.lines
            for character in line_result.characters
        ]
        character_boxes_available = bool(characters)
        logits_available = bool(characters) and all(
            character.logits_available for character in characters
        )
        gaps: list[str] = []
        if not character_boxes_available:
            gaps.append("character_box_adapter")
        if not logits_available:
            gaps.append("raw_logits_adapter")

        return OcrAudit(
            paddleocr_version=str(capabilities.get("paddleocr_version", "unknown")),
            paddlepaddle_version=str(capabilities.get("paddlepaddle_version", "unknown")),
            cuda_version=(
                str(capabilities["cuda_version"])
                if capabilities.get("cuda_version") is not None
                else None
            ),
            gpu_name=(
                str(capabilities["gpu_name"])
                if capabilities.get("gpu_name") is not None
                else None
            ),
            detection_model_name=str(capabilities.get("detection_model_name", "unknown")),
            recognition_model_name=str(
                capabilities.get("recognition_model_name", "unknown")
            ),
            character_boxes_available=character_boxes_available,
            logits_available=logits_available,
            latency_p50_ms=float(np.percentile(durations, 50)),
            latency_p95_ms=float(np.percentile(durations, 95)),
            peak_gpu_memory_mb=(
                _as_float(capabilities["peak_gpu_memory_mb"], "peak_gpu_memory_mb")
                if capabilities.get("peak_gpu_memory_mb") is not None
                else None
            ),
            required_capability_gaps=tuple(gaps),
        )
