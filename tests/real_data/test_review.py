import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from PIL import Image

from poor_word.real_data.review import (
    build_review_queue,
    export_review_queue,
    import_review_labels,
)


def _scores() -> list[dict[str, object]]:
    return [
        {
            "crop_id": "a",
            "image_id": "one",
            "crop_path": "images/a.png",
            "risk_score": 0.50,
            "style_id": "rare",
            "score_model_id": "glyph-v1",
            "score_artifact_sha256": "a" * 64,
        },
        {
            "crop_id": "b",
            "image_id": "two",
            "crop_path": "images/b.png",
            "risk_score": 0.49,
            "style_id": "common",
            "score_model_id": "glyph-v1",
            "score_artifact_sha256": "a" * 64,
        },
        {
            "crop_id": "a",
            "image_id": "one",
            "crop_path": "images/a.png",
            "risk_score": 0.10,
            "style_id": "rare",
            "score_model_id": "glyph-v1",
            "score_artifact_sha256": "a" * 64,
        },
    ]


def test_build_review_queue_deduplicates_and_prioritizes_disagreement_then_coverage() -> None:
    queue = build_review_queue(
        _scores(),
        {
            "a": {
                "disagreement": 0.2,
                "disagreement_model_id": "ensemble-v1",
                "disagreement_artifact_sha256": "b" * 64,
            },
            "b": {
                "disagreement": 0.9,
                "disagreement_model_id": "ensemble-v1",
                "disagreement_artifact_sha256": "b" * 64,
            },
        },
        limit=2,
        seed=7,
    )

    assert [candidate.crop_id for candidate in queue] == ["b", "a"]
    assert len({candidate.crop_id for candidate in queue}) == 2
    assert all(candidate.label is None for candidate in queue)
    assert queue[0].score_model_id == "glyph-v1"
    assert queue[0].disagreement_model_id == "ensemble-v1"


def _crop_manifest(tmp_path: Path) -> Path:
    image_dir = tmp_path / "crop-images"
    (image_dir / "images").mkdir(parents=True)
    for crop_id, color in (("a", "red"), ("b", "blue")):
        Image.new("RGB", (8, 8), color).save(image_dir / "images" / f"{crop_id}.png")
    manifest = tmp_path / "crops.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "crop_id": "a",
                    "crop_path": "images/a.png",
                    "source_image_sha256": "1" * 64,
                },
                {
                    "crop_id": "b",
                    "crop_path": "images/b.png",
                    "source_image_sha256": "2" * 64,
                },
            ]
        ),
        manifest,
    )
    return manifest


def test_review_import_requires_queue_version_rejects_duplicates_and_preserves_gold_owner(
    tmp_path: Path,
) -> None:
    crop_manifest = _crop_manifest(tmp_path)
    queue = build_review_queue(
        _scores()[:2],
        {
            "a": {
                "disagreement": 0.2,
                "disagreement_model_id": "ensemble-v1",
                "disagreement_artifact_sha256": "b" * 64,
            },
            "b": {
                "disagreement": 0.9,
                "disagreement_model_id": "ensemble-v1",
                "disagreement_artifact_sha256": "b" * 64,
            },
        },
        limit=2,
        seed=7,
    )
    exported = export_review_queue(
        queue, crop_manifest, tmp_path / "crop-images", tmp_path / "queues"
    )

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "queue_version": exported.queue_version,
                "crop_id": "a",
                "label": "PASS",
                "annotator_id": "alice",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    first = import_review_labels(labels, exported.queue, tmp_path / "gold")
    row = pq.read_table(first.gold_crops).to_pylist()[0]
    assert row["decision"] == "PASS"
    assert row["queue_version"] == exported.queue_version
    assert row["score_model_id"] == "glyph-v1"
    assert row["disagreement_model_id"] == "ensemble-v1"
    first_audit = json.loads(first.audit.read_text(encoding="utf-8"))
    assert first_audit["accepted_label_count"] == 1
    assert first_audit["review_count"] == 0

    labels.write_text(
        json.dumps(
            {
                "queue_version": "wrong-version",
                "crop_id": "a",
                "label": "BLOCK",
                "annotator_id": "bob",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="queue version"):
        import_review_labels(
            labels, exported.queue, tmp_path / "gold", existing_gold=first.gold_crops
        )

    labels.write_text(
        "".join(
            json.dumps(
                {
                    "queue_version": exported.queue_version,
                    "crop_id": "a",
                    "label": label,
                    "annotator_id": "bob",
                }
            )
            + "\n"
            for label in ("BLOCK", "BLOCK")
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate crop label"):
        import_review_labels(
            labels, exported.queue, tmp_path / "gold", existing_gold=first.gold_crops
        )

    labels.write_text(
        json.dumps(
            {
                "queue_version": exported.queue_version,
                "crop_id": "a",
                "label": "BLOCK",
                "annotator_id": "bob",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="accepted gold label"):
        import_review_labels(
            labels, exported.queue, tmp_path / "gold", existing_gold=first.gold_crops
        )


def test_review_import_keeps_review_out_of_gold_rows(tmp_path: Path) -> None:
    crop_manifest = _crop_manifest(tmp_path)
    queue = build_review_queue(
        _scores()[:1],
        {
            "a": {
                "disagreement": 0.2,
                "disagreement_model_id": "ensemble-v1",
                "disagreement_artifact_sha256": "b" * 64,
            }
        },
        limit=1,
        seed=7,
    )
    exported = export_review_queue(
        queue, crop_manifest, tmp_path / "crop-images", tmp_path / "queues"
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "queue_version": exported.queue_version,
                "crop_id": "a",
                "label": "REVIEW",
                "annotator_id": "alice",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    imported = import_review_labels(labels, exported.queue, tmp_path / "gold")
    assert pq.read_table(imported.gold_crops).num_rows == 0
    audit = json.loads(imported.audit.read_text(encoding="utf-8"))
    assert audit["accepted_label_count"] == 0
    assert audit["review_count"] == 1


def test_export_preserves_priority_order_and_import_rejects_tampered_queue_content(
    tmp_path: Path,
) -> None:
    crop_manifest = _crop_manifest(tmp_path)
    queue = build_review_queue(
        _scores()[:2],
        {
            "a": {
                "disagreement": 0.2,
                "disagreement_model_id": "ensemble-v1",
                "disagreement_artifact_sha256": "b" * 64,
            },
            "b": {
                "disagreement": 0.9,
                "disagreement_model_id": "ensemble-v1",
                "disagreement_artifact_sha256": "b" * 64,
            },
        },
        limit=2,
        seed=7,
    )
    exported = export_review_queue(
        queue, crop_manifest, tmp_path / "crop-images", tmp_path / "queues"
    )
    rows = [json.loads(line) for line in exported.jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["crop_id"] for row in rows] == ["b", "a"]
    assert [row["priority_rank"] for row in rows] == [1, 2]

    rows[0]["risk_score"] = 0.01
    exported.jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "queue_version": exported.queue_version,
                "crop_id": "b",
                "label": "PASS",
                "annotator_id": "alice",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content hash"):
        import_review_labels(labels, exported.queue, tmp_path / "gold")
