import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from poor_word.evaluation.real_report import RealSeedReportConfig, evaluate_real_seed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _fixture(tmp_path: Path) -> RealSeedReportConfig:
    real = _write(
        tmp_path / "real.parquet",
        [
            {
                "image_id": "normal",
                "image_label": "NORMAL",
                "split_role": "DEV",
                "training_eligible": True,
                "source_id": "business",
                "license_id": "Proprietary",
                "production_allowed": True,
            },
            {
                "image_id": "abnormal",
                "image_label": "ABNORMAL",
                "split_role": "DEV",
                "training_eligible": True,
                "source_id": "business",
                "license_id": "Proprietary",
                "production_allowed": True,
            },
            {
                "image_id": "zero",
                "image_label": "NORMAL",
                "split_role": "DEV",
                "training_eligible": True,
                "source_id": "business",
                "license_id": "Proprietary",
                "production_allowed": True,
            },
            {
                "image_id": "locked",
                "image_label": "ABNORMAL",
                "split_role": "LOCKED_TEST",
                "training_eligible": False,
                "source_id": "business",
                "license_id": "Proprietary",
                "production_allowed": True,
            },
        ],
    )
    folds = _write(
        tmp_path / "folds.parquet",
        [
            {"image_id": "normal", "fold": 0, "split_role": "DEV"},
            {"image_id": "abnormal", "fold": 1, "split_role": "DEV"},
            {"image_id": "zero", "fold": 0, "split_role": "DEV"},
            {"image_id": "locked", "fold": -1, "split_role": "LOCKED_TEST"},
        ],
    )
    fold_hash = _sha(folds)
    crops = _write(tmp_path / "crops.parquet", [{"crop_id": "p"}])
    gold = _write(tmp_path / "gold.parquet", [{"crop_id": "p"}])
    char_rows = [
        {
            "crop_id": "pass",
            "image_id": "normal",
            "fold": 0,
            "decision": "PASS",
            "anomaly_kind": "NONE",
            "risk_score": 0.1,
            "model_id": "real-fold-0",
            "checkpoint_sha256": "1" * 64,
            "fold_manifest_sha256": fold_hash,
        },
        {
            "crop_id": "block",
            "image_id": "abnormal",
            "fold": 1,
            "decision": "BLOCK",
            "anomaly_kind": "invented_character",
            "risk_score": 0.9,
            "model_id": "real-fold-1",
            "checkpoint_sha256": "2" * 64,
            "fold_manifest_sha256": fold_hash,
        },
        {
            "crop_id": "review",
            "image_id": "abnormal",
            "fold": 1,
            "decision": "REVIEW",
            "anomaly_kind": "NONE",
            "risk_score": 0.5,
            "model_id": "real-fold-1",
            "checkpoint_sha256": "2" * 64,
            "fold_manifest_sha256": fold_hash,
        },
    ]
    for row in char_rows:
        row["parent_checkpoint_sha256"] = "c" * 64
        row["gold_manifest_sha256"] = _sha(gold)
    char_oof = _write(tmp_path / "char-oof.parquet", char_rows)
    ocr = _write(
        tmp_path / "ocr.parquet",
        [
            {
                **{
                    key: row[key]
                    for key in ("crop_id", "image_id", "fold", "decision", "anomaly_kind")
                },
                "text": text,
                "ocr_confidence": confidence,
                "ocr_model_id": "PP-OCRv5_server_rec",
                "ocr_audit_sha256": "a" * 64,
                "visual_anomaly_score": visual,
            }
            for row, text, confidence, visual in zip(
                char_rows,
                ("常", "�", "龘"),
                (0.95, 0.2, 0.9),
                (0.1, 0.8, None),
                strict=True,
            )
        ],
    )
    image_oof = _write(
        tmp_path / "image-oof.parquet",
        [
            {
                "image_id": image_id,
                "fold": fold,
                "held_out_fold": fold,
                "image_label": label,
                "risk_score": score,
                "zero_character": zero,
                "checkpoint_sha256": str(fold + 3) * 64,
                "real_manifest_sha256": _sha(real),
                "fold_manifest_sha256": fold_hash,
                "feature_manifest_sha256": _sha(char_oof),
            }
            for image_id, fold, label, score, zero in (
                ("normal", 0, "NORMAL", 0.1, False),
                ("abnormal", 1, "ABNORMAL", 0.9, False),
                ("zero", 0, "NORMAL", 0.2, True),
            )
        ],
    )
    common = tmp_path / "common.txt"
    common.write_text("常\n用\n", encoding="utf-8")
    source_lock = tmp_path / "source.lock.json"
    source_lock.write_text('{"sha256":"' + "b" * 64 + '"}\n', encoding="utf-8")
    dependency_lock = tmp_path / "uv.lock"
    dependency_lock.write_text("version = 1\n", encoding="utf-8")
    ocr_audit = tmp_path / "ocr-audit.json"
    ocr_audit.write_text(
        json.dumps(
            {
                "detection_model_name": "PP-OCRv5_server_det",
                "recognition_model_name": "PP-OCRv5_server_rec",
                "character_boxes_available": True,
                "endpoint": "http://127.0.0.1:8765",
            }
        ),
        encoding="utf-8",
    )
    audited_rows = pq.read_table(ocr).to_pylist()
    for row in audited_rows:
        row["ocr_audit_sha256"] = _sha(ocr_audit)
    _write(ocr, audited_rows)
    return RealSeedReportConfig(
        real_manifest=real,
        fold_manifest=folds,
        crop_manifest=crops,
        gold_manifest=gold,
        character_oof=char_oof,
        image_oof=image_oof,
        ocr_manifest=ocr,
        ocr_audit=ocr_audit,
        common_chars=common,
        source_lock=source_lock,
        dependency_lock=dependency_lock,
        output_dir=tmp_path / "report",
        prevalence=0.001,
        threshold=0.5,
    )


def test_report_compares_identical_oof_rows_and_is_explicitly_inconclusive(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    artifacts = evaluate_real_seed(config)
    report = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert report["status"] == "inconclusive"
    assert report["character_metrics"]["glyph"]["counts"] == {
        "positive": 1,
        "negative": 1,
        "review": 1,
        "total": 3,
    }
    assert report["character_metrics"]["glyph"]["aucpr"] == 1.0
    assert report["character_metrics"]["glyph"]["confusion"] == {
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 0,
    }
    assert report["character_metrics"]["glyph"]["anomaly_kind_recall"] == {
        "invented_character": 1.0
    }
    assert report["image_metrics"]["mil"]["counts"]["zero_character"] == 1
    assert report["image_metrics"]["mil"]["base_rate_precision"] == 1.0
    assert report["locked_test_access"] == {
        "image_ids": ["locked"],
        "count": 1,
        "evaluation_score_rows": 0,
        "pixels_accessed": False,
        "labels_accessed": False,
    }
    assert report["provenance"]["inputs"]["real_manifest"] == _sha(config.real_manifest)
    assert report["source_summary"]["production_allowed"] == {"true": 4}
    assert artifacts.provenance_path.is_file()
    assert "Not a pilot approval" in markdown
    assert "0.1%" in markdown
    assert "不得自动封禁罕见字" in markdown
    assert "PP-OCRv5_server_rec" in markdown
    assert "uv run poor-word train adapt-real" in markdown
    assert "uv run poor-word train real-oof" in markdown
    assert "uv run poor-word train mil" in markdown
    assert "uv run poor-word evaluate real-seed" in markdown


@pytest.mark.parametrize("manifest_name", ["character_oof", "image_oof", "ocr_manifest"])
def test_report_rejects_missing_extra_or_locked_score_rows(
    tmp_path: Path, manifest_name: str
) -> None:
    config = _fixture(tmp_path)
    path = getattr(config, manifest_name)
    rows = pq.read_table(path).to_pylist()
    if manifest_name == "image_oof":
        rows.pop()
    elif manifest_name == "ocr_manifest":
        rows.append(dict(rows[0]))
    else:
        rows[0]["image_id"] = "locked"
    _write(path, rows)

    with pytest.raises(ValueError):
        evaluate_real_seed(config)
    assert not config.output_dir.exists()


def test_report_rejects_fold_or_provenance_mismatch(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    rows = pq.read_table(config.image_oof).to_pylist()
    rows[0]["held_out_fold"] = 1
    _write(config.image_oof, rows)
    with pytest.raises(ValueError, match="fold"):
        evaluate_real_seed(config)


def test_l20_commands_are_auditable_not_claimed_as_executed(tmp_path: Path) -> None:
    report = json.loads(
        evaluate_real_seed(_fixture(tmp_path)).json_path.read_text(encoding="utf-8")
    )
    commands = "\n".join(report["l20_commands"])
    assert "--device cuda" in commands
    assert "--endpoint http://127.0.0.1:8765" in commands
    assert report["l20_commands_executed_by_report"] is False
