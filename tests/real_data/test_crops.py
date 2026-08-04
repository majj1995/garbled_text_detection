import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from PIL import Image

from poor_word.ocr.types import OcrAudit
from poor_word.real_data.crops import extract_character_crops
from poor_word.real_data.ingest import import_real_dataset


def _image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 10), "white")
    image.putpixel((0, 0), (1, 2, 3))
    image.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, *, characters: list[dict[str, object]]) -> tuple[Path, Path]:
    source_dir = tmp_path / "source"
    image_hash = _image(source_dir / "images/one.png")
    record = {
        "image_id": "one",
        "image_path": "images/one.png",
        "expected_sha256": image_hash,
        "image_label": "ABNORMAL",
        "split_role": "DEV",
        "source_id": "business_seed",
        "source_group_id": "upload-one",
        "license_id": "LicenseRef-Proprietary",
        "production_allowed": True,
        "product_id": "product-one",
        "campaign_id": "campaign-one",
        "template_id": "template-one",
        "label_provenance": "human-image-review-v1",
        "training_eligible": True,
        "characters": characters,
    }
    records = source_dir / "records.jsonl"
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")
    imported = import_real_dataset(records, tmp_path / "dataset")
    return imported.manifest, source_dir


def test_extract_character_crops_clamps_padding_links_hash_and_is_byte_identical(
    tmp_path: Path,
) -> None:
    manifest, image_root = _manifest(
        tmp_path,
        characters=[
            {
                "annotation_id": "one-char-1",
                "box": {"x0": 1, "y0": 2, "x1": 7, "y1": 9},
                "decision": "BLOCK",
                "annotator_id": "reviewer-1",
                "text": "字",
                "anomaly_kind": "missing_stroke",
            }
        ],
    )

    first = extract_character_crops(manifest, image_root, tmp_path / "crops", padding=3)
    first_png = next((tmp_path / "crops" / "images").glob("*.png"))
    first_bytes = first_png.read_bytes()
    row = pq.read_table(first.manifest).to_pylist()[0]

    assert Image.open(first_png).size == (10, 10)
    assert row["crop_box"] == {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert (
        row["source_image_sha256"]
        == hashlib.sha256((image_root / "images/one.png").read_bytes()).hexdigest()
    )
    assert row["annotation_source"] == "reviewed"
    assert row["source_annotation_id"] == "one-char-1"

    second = extract_character_crops(manifest, image_root, tmp_path / "crops", padding=3)
    assert first.manifest.read_bytes() == second.manifest.read_bytes()
    assert first_bytes == first_png.read_bytes()


def test_extract_character_crops_refuses_unaudited_or_line_only_ocr_boxes(tmp_path: Path) -> None:
    manifest, image_root = _manifest(tmp_path, characters=[])
    audit = OcrAudit(
        paddleocr_version="3.2.0",
        paddlepaddle_version="3.1.1",
        cuda_version=None,
        gpu_name=None,
        detection_model_name="PP-OCRv5_server_det",
        recognition_model_name="PP-OCRv5_server_rec",
        character_boxes_available=False,
        logits_available=False,
        latency_p50_ms=1,
        latency_p95_ms=2,
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(audit.model_dump_json(), encoding="utf-8")
    results_path = tmp_path / "ocr.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "image_id": "one",
                "result": {
                    "model_name": "PP-OCRv5_server_rec",
                    "lines": [
                        {
                            "text": "两个",
                            "confidence": 0.9,
                            "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
                            "characters": [],
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="character boxes unavailable"):
        extract_character_crops(
            manifest,
            image_root,
            tmp_path / "crops",
            ocr_results=results_path,
            ocr_audit=audit_path,
        )

    audited = audit.model_copy(update={"character_boxes_available": True})
    audit_path.write_text(audited.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="no character boxes"):
        extract_character_crops(
            manifest,
            image_root,
            tmp_path / "crops",
            ocr_results=results_path,
            ocr_audit=audit_path,
        )


def test_ocr_crops_carry_validated_model_and_audit_provenance(tmp_path: Path) -> None:
    manifest, image_root = _manifest(tmp_path, characters=[])
    audit = OcrAudit(
        paddleocr_version="3.2.0",
        paddlepaddle_version="3.1.1",
        cuda_version="12.6",
        gpu_name="NVIDIA L20",
        detection_model_name="PP-OCRv5_server_det",
        recognition_model_name="PP-OCRv5_server_rec",
        character_boxes_available=True,
        logits_available=False,
        latency_p50_ms=1,
        latency_p95_ms=2,
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(audit.model_dump_json(), encoding="utf-8")
    results_path = tmp_path / "ocr.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "image_id": "one",
                "result": {
                    "model_name": "glyph-locator-2026-08",
                    "lines": [
                        {
                            "text": "字",
                            "confidence": 0.9,
                            "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
                            "characters": [
                                {"text": "字", "box": {"x0": 1, "y0": 1, "x1": 8, "y1": 8}}
                            ],
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = extract_character_crops(
        manifest,
        image_root,
        tmp_path / "crops",
        ocr_results=results_path,
        ocr_audit=audit_path,
    )

    row = pq.read_table(artifacts.manifest).to_pylist()[0]
    crop_audit = json.loads(artifacts.audit.read_text(encoding="utf-8"))
    assert row["ocr_model_name"] == "glyph-locator-2026-08"
    assert row["ocr_audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert crop_audit["ocr_audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert crop_audit["ocr_audit"]["recognition_model_name"] == "PP-OCRv5_server_rec"
