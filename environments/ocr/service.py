import base64
import io
import os
import threading
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "PP-OCRv5_server_rec"
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class PredictRequest(BaseModel):
    file: str


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _result_payload(result: Any) -> Mapping[str, Any]:
    payload = result.json
    if callable(payload):
        payload = payload()
    if not isinstance(payload, Mapping):
        raise ValueError("PaddleOCR result.json must be an object")
    nested = payload.get("res", payload)
    if not isinstance(nested, Mapping):
        raise ValueError("PaddleOCR result.res must be an object")
    return nested


def _polygon_from_box(box: Sequence[float]) -> list[list[float]]:
    x0, y0, x1, y1 = (float(value) for value in box)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _normalize_lines(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    texts = list(_plain(payload.get("rec_texts", [])))
    scores = list(_plain(payload.get("rec_scores", [])))
    polygons = list(_plain(payload.get("rec_polys", [])))
    boxes = list(_plain(payload.get("rec_boxes", [])))
    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        polygon = polygons[index] if index < len(polygons) else _polygon_from_box(boxes[index])
        lines.append(
            {
                "text": str(text),
                "confidence": float(scores[index]),
                "polygon": polygon,
                "characters": [],
            }
        )
    return lines


class PaddleRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline: Any = None
        self._paddle: Any = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            import paddle
            from paddleocr import PaddleOCR

            self._paddle = paddle
            self._pipeline = PaddleOCR(
                text_detection_model_name=DETECTION_MODEL,
                text_recognition_model_name=RECOGNITION_MODEL,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=os.environ.get("POOR_WORD_OCR_DEVICE", "gpu:0"),
            )

    def predict(self, image: np.ndarray[Any, np.dtype[np.uint8]]) -> dict[str, Any]:
        self._load()
        started = time.perf_counter()
        results = list(self._pipeline.predict(image))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(results) != 1:
            raise ValueError(f"expected one PaddleOCR result, got {len(results)}")
        return {
            "model_name": "PP-OCRv5_server",
            "lines": _normalize_lines(_result_payload(results[0])),
            "stage_ms": {"total": elapsed_ms},
        }

    def health(self) -> dict[str, Any]:
        self._load()
        paddle = self._paddle
        compiled_with_cuda = bool(paddle.device.is_compiled_with_cuda())
        gpu_name = None
        peak_memory_mb = None
        if compiled_with_cuda:
            try:
                gpu_name = str(paddle.device.cuda.get_device_name())
            except Exception:
                gpu_name = "unknown"
            try:
                peak_memory_mb = float(paddle.device.cuda.max_memory_allocated()) / (1024**2)
            except Exception:
                peak_memory_mb = None
        return {
            "status": "ok",
            "paddleocr_version": version("paddleocr"),
            "paddlepaddle_version": str(paddle.__version__),
            "cuda_version": str(paddle.version.cuda()) if compiled_with_cuda else None,
            "gpu_name": gpu_name,
            "detection_model_name": DETECTION_MODEL,
            "recognition_model_name": RECOGNITION_MODEL,
            "peak_gpu_memory_mb": peak_memory_mb,
        }


app = FastAPI(title="poor-word PP-OCRv5 runtime", version="0.1.0")
runtime = PaddleRuntime()


@app.get("/health")
def health() -> dict[str, Any]:
    return runtime.health()


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        image_bytes = base64.b64decode(request.file, validate=True)
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("image payload is empty or exceeds 32 MiB")
        image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
        return runtime.predict(image)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, workers=1)
