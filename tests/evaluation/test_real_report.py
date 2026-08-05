import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

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


def _trusted_fixture(tmp_path: Path) -> RealSeedReportConfig:
    real = _write(
        tmp_path / "real.parquet",
        [
            {
                "image_id": image_id,
                "image_label": label,
                "split_role": role,
                "training_eligible": role == "DEV",
                "source_id": "business",
                "license_id": "Proprietary",
                "production_allowed": True,
            }
            for image_id, label, role in (
                ("normal", "NORMAL", "DEV"),
                ("abnormal", "ABNORMAL", "DEV"),
                ("zero", "NORMAL", "DEV"),
                ("locked", "ABNORMAL", "LOCKED_TEST"),
            )
        ],
    )
    folds = _write(
        tmp_path / "folds.parquet",
        [
            {
                "image_id": image_id,
                "fold": fold,
                "split_role": role,
                "component_id": f"component-{image_id}",
            }
            for image_id, fold, role in (
                ("normal", 0, "DEV"),
                ("abnormal", 1, "DEV"),
                ("zero", 0, "DEV"),
                ("locked", -1, "LOCKED_TEST"),
            )
        ],
    )
    crop_rows = [
        {"crop_id": "pass", "image_id": "normal", "crop_path": "images/pass.png"},
        {"crop_id": "block", "image_id": "abnormal", "crop_path": "images/block.png"},
        {"crop_id": "review", "image_id": "abnormal", "crop_path": "images/review.png"},
        {
            "crop_id": "locked-crop",
            "image_id": "locked",
            "crop_path": "images/locked-secret.png",
        },
    ]
    crops = _write(tmp_path / "crops.parquet", crop_rows)
    gold = _write(
        tmp_path / "gold.parquet",
        [
            {**crop_rows[0], "decision": "PASS", "anomaly_kind": "NONE"},
            {
                **crop_rows[1],
                "decision": "BLOCK",
                "anomaly_kind": "invented_character",
            },
            {**crop_rows[2], "decision": "REVIEW", "anomaly_kind": "NONE"},
            {
                **crop_rows[3],
                "decision": "BLOCK",
                "anomaly_kind": "invented_character",
            },
        ],
    )
    trusted_hashes = {
        "real_manifest_sha256": _sha(real),
        "crop_manifest_sha256": _sha(crops),
        "gold_manifest_sha256": _sha(gold),
        "fold_manifest_sha256": _sha(folds),
    }
    char_specs = [
        ("pass", "normal", "images/pass.png", 0, "PASS", "NONE", 0.1),
        (
            "block",
            "abnormal",
            "images/block.png",
            1,
            "BLOCK",
            "invented_character",
            0.9,
        ),
        ("review", "abnormal", "images/review.png", 1, "REVIEW", "NONE", 0.5),
    ]
    char_rows: list[dict[str, object]] = []
    char_root = tmp_path / "char-models"
    char_inventory: list[dict[str, object]] = []
    for fold in (0, 1):
        directory = char_root / f"fold-{fold}"
        directory.mkdir(parents=True)
        ids = sorted(item[0] for item in char_specs if item[3] == fold)
        checkpoint = directory / "model.pt"
        torch.save(
            {
                "held_out_fold": fold,
                "scoring_crop_ids": ids,
                "real_training_crop_ids": sorted(
                    item[0]
                    for item in char_specs
                    if item[3] != fold and item[4] in {"PASS", "BLOCK"}
                ),
                **trusted_hashes,
            },
            checkpoint,
        )
        checkpoint_hash = _sha(checkpoint)
        fold_rows = [
            {
                "crop_id": crop_id,
                "image_id": image_id,
                "crop_path": crop_path,
                "fold": row_fold,
                "decision": decision,
                "anomaly_kind": anomaly_kind,
                "risk_score": risk,
                "model_id": f"real-fold-{fold}",
                "checkpoint_sha256": checkpoint_hash,
                "parent_checkpoint_sha256": "c" * 64,
                **trusted_hashes,
            }
            for crop_id, image_id, crop_path, row_fold, decision, anomaly_kind, risk in char_specs
            if row_fold == fold
        ]
        char_rows.extend(fold_rows)
        scores = _write(directory / "scores.parquet", fold_rows)
        metrics = directory / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "held_out_fold": fold,
                    "checkpoint_sha256": checkpoint_hash,
                    "scores_sha256": _sha(scores),
                    "scoring_crop_ids": ids,
                }
            ),
            encoding="utf-8",
        )
        char_inventory.append(
            {
                "held_out_fold": fold,
                "checkpoint": f"fold-{fold}/model.pt",
                "checkpoint_sha256": checkpoint_hash,
                "metrics": f"fold-{fold}/metrics.json",
                "metrics_sha256": _sha(metrics),
                "scores": f"fold-{fold}/scores.parquet",
                "scores_sha256": _sha(scores),
            }
        )
    char_oof = _write(tmp_path / "char-oof.parquet", char_rows)
    character_inventory = char_root / "model-inventory.json"
    character_inventory.write_text(
        json.dumps({"schema_version": 1, "kind": "character_oof", "folds": char_inventory}),
        encoding="utf-8",
    )

    nested_root = tmp_path / "nested-models"
    nested_entries: list[dict[str, object]] = []
    for outer_fold in (0, 1):
        feature_rows: list[dict[str, object]] = []
        model_entries: list[dict[str, object]] = []
        for scoring_fold in (0, 1):
            excluded = sorted({outer_fold, scoring_fold})
            directory = nested_root / f"outer-fold-{outer_fold}" / f"character-fold-{scoring_fold}"
            directory.mkdir(parents=True)
            scoring_specs = [item for item in char_specs if item[3] == scoring_fold]
            scoring_ids = sorted(item[0] for item in scoring_specs)
            checkpoint = directory / "model.pt"
            torch.save(
                {
                    "held_out_fold": scoring_fold,
                    "excluded_folds": excluded,
                    "scoring_crop_ids": scoring_ids,
                    "real_training_crop_ids": sorted(
                        item[0]
                        for item in char_specs
                        if item[3] not in excluded and item[4] in {"PASS", "BLOCK"}
                    ),
                    "fold_manifest_sha256": _sha(folds),
                },
                checkpoint,
            )
            checkpoint_hash = _sha(checkpoint)
            score_rows = [
                {
                    "crop_id": crop_id,
                    "image_id": image_id,
                    "fold": row_fold,
                    "risk_score": risk,
                    "model_id": f"real-fold-{scoring_fold}",
                    "checkpoint_sha256": checkpoint_hash,
                    "fold_manifest_sha256": _sha(folds),
                    "excluded_folds": excluded,
                }
                for crop_id, image_id, _path, row_fold, _decision, _kind, risk in scoring_specs
            ]
            scores = _write(directory / "scores.parquet", score_rows)
            metrics = directory / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "held_out_fold": scoring_fold,
                        "excluded_folds": excluded,
                        "checkpoint_sha256": checkpoint_hash,
                        "scores_sha256": _sha(scores),
                    }
                ),
                encoding="utf-8",
            )
            feature_rows.extend({**row, "outer_fold": outer_fold} for row in score_rows)
            prefix = f"outer-fold-{outer_fold}/character-fold-{scoring_fold}"
            model_entries.append(
                {
                    "scoring_fold": scoring_fold,
                    "excluded_folds": excluded,
                    "checkpoint": f"{prefix}/model.pt",
                    "checkpoint_sha256": checkpoint_hash,
                    "metrics": f"{prefix}/metrics.json",
                    "metrics_sha256": _sha(metrics),
                    "scores": f"{prefix}/scores.parquet",
                    "scores_sha256": _sha(scores),
                    "scoring_crop_ids": scoring_ids,
                }
            )
        features = _write(
            nested_root / f"outer-fold-{outer_fold}" / "features.parquet",
            feature_rows,
        )
        nested_entries.append(
            {
                "outer_fold": outer_fold,
                "features": f"outer-fold-{outer_fold}/features.parquet",
                "features_sha256": _sha(features),
                "models": model_entries,
            }
        )
    nested_manifest = nested_root / "nested-manifest.json"
    nested_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "nested_character_oof",
                "real_manifest_sha256": _sha(real),
                "fold_manifest_sha256": _sha(folds),
                "outer_folds": nested_entries,
            }
        ),
        encoding="utf-8",
    )

    image_specs = [
        ("normal", 0, "NORMAL", 0.1, False),
        ("zero", 0, "NORMAL", 0.2, True),
        ("abnormal", 1, "ABNORMAL", 0.9, False),
    ]
    image_rows: list[dict[str, object]] = []
    image_root = tmp_path / "image-models"
    image_inventory_rows: list[dict[str, object]] = []
    for fold in (0, 1):
        directory = image_root / f"fold-{fold}"
        directory.mkdir(parents=True)
        ids = sorted(item[0] for item in image_specs if item[1] == fold)
        feature_entry = nested_entries[fold]
        feature_hash = str(feature_entry["features_sha256"])
        image_hashes = {
            "real_manifest_sha256": _sha(real),
            "fold_manifest_sha256": _sha(folds),
            "feature_manifest_sha256": feature_hash,
            "nested_manifest_sha256": _sha(nested_manifest),
        }
        checkpoint = directory / "model.pt"
        torch.save(
            {
                "held_out_fold": fold,
                "validation_image_ids": ids,
                "train_image_ids": sorted(item[0] for item in image_specs if item[1] != fold),
                **image_hashes,
            },
            checkpoint,
        )
        checkpoint_hash = _sha(checkpoint)
        fold_rows = [
            {
                "image_id": image_id,
                "fold": row_fold,
                "held_out_fold": row_fold,
                "image_label": label,
                "risk_score": risk,
                "zero_character": zero,
                "checkpoint_sha256": checkpoint_hash,
                **image_hashes,
            }
            for image_id, row_fold, label, risk, zero in image_specs
            if row_fold == fold
        ]
        image_rows.extend(fold_rows)
        scores = _write(directory / "image-scores.parquet", fold_rows)
        metrics = directory / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "held_out_fold": fold,
                    "checkpoint_sha256": checkpoint_hash,
                    "image_scores_sha256": _sha(scores),
                    "validation_image_ids": ids,
                    **image_hashes,
                }
            ),
            encoding="utf-8",
        )
        image_inventory_rows.append(
            {
                "held_out_fold": fold,
                "checkpoint": f"fold-{fold}/model.pt",
                "checkpoint_sha256": checkpoint_hash,
                "metrics": f"fold-{fold}/metrics.json",
                "metrics_sha256": _sha(metrics),
                "scores": f"fold-{fold}/image-scores.parquet",
                "scores_sha256": _sha(scores),
                "nested_feature_manifest": feature_entry["features"],
                "nested_feature_manifest_sha256": feature_hash,
            }
        )
    image_oof = _write(tmp_path / "image-oof.parquet", image_rows)
    image_inventory = image_root / "model-inventory.json"
    image_inventory.write_text(
        json.dumps({"schema_version": 1, "kind": "image_oof", "folds": image_inventory_rows}),
        encoding="utf-8",
    )

    ocr_audit = tmp_path / "ocr-audit.json"
    ocr_audit.write_text(
        json.dumps(
            {
                "paddleocr_version": "3.2.0",
                "paddlepaddle_version": "3.1.1",
                "cuda_version": "12.6",
                "gpu_name": "NVIDIA L20",
                "detection_model_name": "PP-OCRv5_server_det",
                "recognition_model_name": "PP-OCRv5_server_rec",
                "character_boxes_available": True,
                "logits_available": False,
                "latency_p50_ms": 30.0,
                "latency_p95_ms": 50.0,
                "required_capability_gaps": ["raw_logits_adapter"],
            }
        ),
        encoding="utf-8",
    )
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
                "ocr_audit_sha256": _sha(ocr_audit),
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
    common = tmp_path / "common_chars_3500.txt"
    characters = ["常", "用"] + [
        chr(0x4E00 + index) for index in range(3498) if chr(0x4E00 + index) not in {"常", "用"}
    ]
    while len(characters) < 3500:
        characters.append(chr(0x6000 + len(characters)))
    common.write_text("".join(characters[:3500]), encoding="utf-8")
    source_lock = tmp_path / "source.lock.json"
    source_lock.write_text(
        json.dumps(
            {
                "source_id": "common_chars_3500",
                "declared_url": "https://example.invalid/common.txt",
                "resolved_url": "https://example.invalid/common.txt",
                "output_name": common.name,
                "sha256": _sha(common),
                "size_bytes": common.stat().st_size,
                "license_id": "Apache-2.0",
                "production_allowed": True,
            }
        ),
        encoding="utf-8",
    )
    dependency_lock = tmp_path / "uv.lock"
    dependency_lock.write_text("version = 1\n", encoding="utf-8")
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
        character_model_inventory=character_inventory,
        image_model_inventory=image_inventory,
        nested_character_manifest=nested_manifest,
        output_dir=tmp_path / "report",
        prevalence=0.001,
        threshold=0.5,
    )


_fixture = _trusted_fixture


def test_report_config_requires_nested_character_provenance(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    payload = config.model_dump(exclude={"nested_character_manifest"})

    with pytest.raises(ValueError, match="nested_character_manifest"):
        RealSeedReportConfig.model_validate(payload)


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
    assert "uv run poor-word train mil-oof" in markdown
    assert "--held-out-fold" not in markdown
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


@pytest.mark.parametrize("mutation", ["delete", "flip_label", "fake_crop"])
def test_character_scores_cannot_redefine_the_trusted_gold_cohort(
    tmp_path: Path, mutation: str
) -> None:
    config = _fixture(tmp_path)
    character = pq.read_table(config.character_oof).to_pylist()
    ocr = pq.read_table(config.ocr_manifest).to_pylist()
    if mutation == "delete":
        character.pop(0)
        ocr.pop(0)
    elif mutation == "flip_label":
        character[1]["decision"] = "PASS"
        ocr[1]["decision"] = "PASS"
    else:
        fake = dict(character[0], crop_id="fabricated")
        character.append(fake)
        ocr.append(dict(ocr[0], crop_id="fabricated"))
    _write(config.character_oof, character)
    _write(config.ocr_manifest, ocr)
    with pytest.raises(ValueError, match=r"trusted|cohort|gold"):
        evaluate_real_seed(config)


def test_low_confidence_pass_is_tn_for_block_policy_not_fp(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    rows = pq.read_table(config.ocr_manifest).to_pylist()
    next(row for row in rows if row["decision"] == "PASS")["ocr_confidence"] = 0.01
    _write(config.ocr_manifest, rows)
    report = json.loads(evaluate_real_seed(config).json_path.read_text(encoding="utf-8"))
    assert report["character_metrics"]["ocr_plus_rule_block"]["confusion"]["tn"] == 1
    assert report["character_metrics"]["ocr_plus_rule_block"]["confusion"]["fp"] == 0
    assert "diagnostic" in report["character_metrics"]["ocr_confidence_review_risk"]["scope"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recognition_model_name", "PP-OCRv5_mobile_rec"),
        ("detection_model_name", "PP-OCRv5_mobile_det"),
        ("character_boxes_available", False),
    ],
)
def test_report_rejects_unaudited_server_ocr(tmp_path: Path, field: str, value: object) -> None:
    config = _fixture(tmp_path)
    audit = json.loads(config.ocr_audit.read_text(encoding="utf-8"))
    audit[field] = value
    config.ocr_audit.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match=r"OCR|server|character boxes"):
        evaluate_real_seed(config)


def test_report_rejects_malformed_ocr_audit(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config.ocr_audit.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_real_seed(config)


def test_report_rejects_component_crossing_folds(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    rows = pq.read_table(config.fold_manifest).to_pylist()
    rows[1]["component_id"] = rows[0]["component_id"]
    _write(config.fold_manifest, rows)
    with pytest.raises(ValueError, match="component_id crosses folds"):
        evaluate_real_seed(config)


def test_report_projects_sensitive_columns_only_after_locked_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path)
    original = pq.read_table

    def guarded(path: Path, *args: object, **kwargs: object) -> pa.Table:
        columns = kwargs.get("columns")
        filters = kwargs.get("filters")
        names = set(columns) if isinstance(columns, list) else set()
        if Path(path) == config.fold_manifest:
            assert "image_label" not in names
        if Path(path) in {config.crop_manifest, config.gold_manifest} and names & {
            "crop_path",
            "decision",
            "anomaly_kind",
        }:
            assert filters is not None
            assert "locked-crop" not in str(filters)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded)
    evaluate_real_seed(config)


@pytest.mark.parametrize("artifact", ["common", "source_lock", "character_checkpoint"])
def test_report_rejects_untrusted_catalog_or_model_artifact(tmp_path: Path, artifact: str) -> None:
    config = _fixture(tmp_path)
    if artifact == "common":
        config.common_chars.write_text("常用", encoding="utf-8")
    elif artifact == "source_lock":
        payload = json.loads(config.source_lock.read_text(encoding="utf-8"))
        payload["production_allowed"] = False
        config.source_lock.write_text(json.dumps(payload), encoding="utf-8")
    else:
        inventory = json.loads(config.character_model_inventory.read_text(encoding="utf-8"))
        checkpoint = config.character_model_inventory.parent / inventory["folds"][0]["checkpoint"]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        evaluate_real_seed(config)
