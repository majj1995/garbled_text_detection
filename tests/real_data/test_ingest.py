import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from PIL import Image

from poor_word.real_data.ingest import import_real_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(image_id: str, image_path: str, *, expected_sha256: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "image_path": image_path,
        "expected_sha256": expected_sha256,
        "image_label": "ABNORMAL",
        "split_role": "DEV",
        "source_id": "business_seed",
        "source_group_id": f"upload-{image_id}",
        "license_id": "LicenseRef-Proprietary",
        "production_allowed": True,
        "product_id": f"product-{image_id}",
        "campaign_id": "campaign-1",
        "template_id": "template-1",
        "label_provenance": "human-image-review-v1",
        "training_eligible": True,
        "characters": [
            {
                "annotation_id": f"{image_id}-char-1",
                "box": {"x0": 2, "y0": 3, "x1": 20, "y1": 22},
                "decision": "BLOCK",
                "annotator_id": "reviewer-1",
                "anomaly_kind": "missing_stroke",
            }
        ],
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )


def test_import_decodes_hashes_and_writes_deterministic_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_hash = _write_image(input_dir / "images/one.png", (10, 20, 30))
    second_hash = _write_image(input_dir / "images/two.png", (40, 50, 60))
    source = input_dir / "records.jsonl"
    _write_jsonl(
        source,
        [
            _record("two", "images/two.png", expected_sha256=second_hash),
            _record("one", "images/one.png", expected_sha256=first_hash),
        ],
    )

    first = import_real_dataset(source, tmp_path / "dataset-a")
    second = import_real_dataset(source, tmp_path / "dataset-b")

    rows = pq.read_table(first.manifest).to_pylist()
    assert [row["image_id"] for row in rows] == ["one", "two"]
    assert rows[0]["image_path"] == "images/one.png"
    assert rows[0]["image_sha256"] == first_hash
    assert rows[0]["width"] == 32
    assert rows[0]["height"] == 24
    first_metadata = json.loads(first.dataset_metadata.read_text(encoding="utf-8"))
    second_metadata = json.loads(second.dataset_metadata.read_text(encoding="utf-8"))
    assert first_metadata["dataset_id"] == second_metadata["dataset_id"]
    assert first_metadata["row_count"] == 2
    assert first.validation_report.exists()


def test_import_rejects_duplicate_ids_hash_mismatch_and_out_of_bounds_box(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    image_hash = _write_image(input_dir / "images/one.png", (10, 20, 30))
    source = input_dir / "records.jsonl"
    record = _record("one", "images/one.png", expected_sha256=image_hash)
    _write_jsonl(source, [record, record])
    with pytest.raises(ValueError, match="duplicate image_id"):
        import_real_dataset(source, tmp_path / "duplicate")

    bad_hash = _record("one", "images/one.png", expected_sha256="0" * 64)
    _write_jsonl(source, [bad_hash])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        import_real_dataset(source, tmp_path / "bad-hash")

    bad_box = _record("one", "images/one.png", expected_sha256=image_hash)
    characters = bad_box["characters"]
    assert isinstance(characters, list)
    characters[0]["box"]["x1"] = 40  # type: ignore[index]
    _write_jsonl(source, [bad_box])
    with pytest.raises(ValueError, match="outside image bounds"):
        import_real_dataset(source, tmp_path / "bad-box")
