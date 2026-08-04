import sys

from poor_word.ocr.paddle_v5 import PaddleV5Adapter


class FakePaddleBackend:
    def capabilities(self) -> dict[str, object]:
        return {
            "paddleocr_version": "3.3.0",
            "paddlepaddle_version": "3.3.0",
            "cuda_version": "12.6",
            "gpu_name": "NVIDIA L20",
            "detection_model_name": "PP-OCRv5_server_det",
            "recognition_model_name": "PP-OCRv5_server_rec",
            "peak_gpu_memory_mb": 1024.0,
        }

    def predict(self, image: str) -> dict[str, object]:
        return {
            "model_name": "PP-OCRv5_server",
            "lines": [
                {
                    "text": "优惠",
                    "confidence": 0.85,
                    "polygon": [[0, 0], [80, 0], [80, 32], [0, 32]],
                    "characters": [
                        {
                            "text": "优",
                            "box": [0, 0, 40, 32],
                            "top_k": [["优", 0.8]],
                            "logits": [1.0],
                        },
                        {
                            "text": "惠",
                            "box": [40, 0, 80, 32],
                            "top_k": [["惠", 0.9]],
                            "logits": [1.0],
                        },
                    ],
                }
            ],
            "stage_ms": {"total": 12.5},
        }


def test_adapter_preserves_character_boxes_and_raw_candidates() -> None:
    result = PaddleV5Adapter(backend=FakePaddleBackend()).recognize("poster.png")

    assert result.model_name == "PP-OCRv5_server"
    assert result.lines[0].text == "优惠"
    assert [character.text for character in result.lines[0].characters] == ["优", "惠"]
    assert result.lines[0].characters[0].top_k[0].text == "优"
    assert result.lines[0].characters[0].logits_available is True


def test_cpu_adapter_does_not_import_paddle() -> None:
    PaddleV5Adapter(backend=FakePaddleBackend())

    assert "paddle" not in sys.modules
    assert "paddleocr" not in sys.modules


def test_audit_reports_measured_capabilities() -> None:
    audit = PaddleV5Adapter(backend=FakePaddleBackend()).audit(
        ("poster.png",), warmup=1, runs=3
    )

    assert audit.detection_model_name == "PP-OCRv5_server_det"
    assert audit.recognition_model_name == "PP-OCRv5_server_rec"
    assert audit.character_boxes_available is True
    assert audit.logits_available is True
    assert audit.latency_p95_ms >= audit.latency_p50_ms >= 0
    assert audit.required_capability_gaps == ()


def test_audit_reports_standard_line_only_capability_gaps() -> None:
    backend = FakePaddleBackend()
    backend.predict = lambda image: {  # type: ignore[method-assign]
        "model_name": "PP-OCRv5_server",
        "lines": [
            {
                "text": "优惠",
                "confidence": 0.85,
                "polygon": [[0, 0], [80, 0], [80, 32], [0, 32]],
                "characters": [],
            }
        ],
    }

    audit = PaddleV5Adapter(backend=backend).audit(("poster.png",), warmup=0, runs=1)

    assert audit.character_boxes_available is False
    assert audit.logits_available is False
    assert audit.required_capability_gaps == ("character_box_adapter", "raw_logits_adapter")
