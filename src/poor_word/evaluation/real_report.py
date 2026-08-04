# ruff: noqa: E501, RUF001
"""Strict real-seed OOF comparison and provenance-rich phase reporting."""

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

from poor_word.domain import Decision
from poor_word.evaluation.baselines import BaselineInput, score_baselines
from poor_word.evaluation.metrics import base_rate_precision
from poor_word.real_data.schema import ImageLabel, SplitRole


class RealSeedReportConfig(BaseModel):
    """Explicit immutable inputs to the real-seed phase report."""

    model_config = ConfigDict(frozen=True)

    real_manifest: Path
    fold_manifest: Path
    crop_manifest: Path
    gold_manifest: Path
    character_oof: Path
    image_oof: Path
    ocr_manifest: Path
    ocr_audit: Path
    common_chars: Path
    source_lock: Path
    dependency_lock: Path
    output_dir: Path
    additional_source_artifacts: dict[str, Path] = Field(default_factory=dict)
    model_artifacts: dict[str, Path] = Field(default_factory=dict)
    prevalence: float = Field(default=0.001, gt=0.0, lt=1.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    minimum_negative_count: int = Field(default=10_000, ge=10_000)
    minimum_real_positive_count: int = Field(default=60, ge=1)


@dataclass(frozen=True)
class RealSeedReportArtifacts:
    json_path: Path
    markdown_path: Path
    provenance_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _required_text(row: dict[str, Any], field: str, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} has malformed {field}")
    return value


def _required_probability(row: dict[str, Any], field: str, source: str) -> float:
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{source} has malformed {field}")
    return float(value)


def _wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return [max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator)]


def _metric_summary(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    label_field: str,
    positive_value: str,
    threshold: float,
    prevalence: float,
    review_value: str | None = None,
    zero_field: str | None = None,
) -> dict[str, object]:
    supervised = [row for row in rows if review_value is None or row[label_field] != review_value]
    labels = np.asarray([row[label_field] == positive_value for row in supervised], dtype=np.int64)
    scores = np.asarray([row[score_field] for row in supervised], dtype=np.float64)
    predictions = scores >= threshold
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    positives = tp + fn
    negatives = fp + tn
    recall = tp / positives if positives else None
    fpr = fp / negatives if negatives else None
    two_class = positives > 0 and negatives > 0
    aucpr = float(average_precision_score(labels, scores)) if two_class else None
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in supervised:
        if row[label_field] == positive_value:
            by_kind[str(row.get("anomaly_kind", "IMAGE_ABNORMAL"))].append(row)
    kind_recall = {
        kind: sum(item[score_field] >= threshold for item in values) / len(values)
        for kind, values in sorted(by_kind.items())
    }
    review_count = (
        sum(row[label_field] == review_value for row in rows) if review_value is not None else 0
    )
    counts: dict[str, int] = {
        "positive": positives,
        "negative": negatives,
        "review": review_count,
        "total": len(rows),
    }
    if zero_field is not None:
        counts["zero_character"] = sum(row[zero_field] is True for row in rows)
    return {
        "counts": counts,
        "review_rate": review_count / len(rows) if rows else 0.0,
        "aucpr": aucpr,
        "status": "ok" if two_class else "inconclusive_single_class",
        "diagnostic_threshold": threshold,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "recall": recall,
        "recall_wilson_95": _wilson(tp, positives),
        "fpr": fpr,
        "fpr_wilson_95": _wilson(fp, negatives),
        "anomaly_kind_recall": kind_recall,
        "base_rate_precision": (
            base_rate_precision(prevalence, recall, fpr)
            if recall is not None and fpr is not None
            else None
        ),
    }


def _load_real_contract(
    config: RealSeedReportConfig,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, object], Counter[str]]:
    dataset = ds.dataset(config.real_manifest, format="parquet")
    metadata_columns = [
        "image_id",
        "split_role",
        "training_eligible",
        "source_id",
        "license_id",
        "production_allowed",
    ]
    metadata = cast(list[dict[str, Any]], dataset.to_table(columns=metadata_columns).to_pylist())
    locked_ids: list[str] = []
    sources: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    production: Counter[str] = Counter()
    seen: set[str] = set()
    for row in metadata:
        image_id = _required_text(row, "image_id", "real manifest")
        if image_id in seen:
            raise ValueError(f"real manifest has duplicate image_id: {image_id}")
        seen.add(image_id)
        role = _required_text(row, "split_role", "real manifest")
        if role == SplitRole.LOCKED_TEST.value:
            locked_ids.append(image_id)
        sources[_required_text(row, "source_id", "real manifest")] += 1
        licenses[_required_text(row, "license_id", "real manifest")] += 1
        allowed = row.get("production_allowed")
        if not isinstance(allowed, bool):
            raise ValueError("real manifest has malformed production_allowed")
        production[str(allowed).lower()] += 1

    safe_rows = cast(
        list[dict[str, Any]],
        dataset.to_table(
            columns=["image_id", "image_label", "split_role", "training_eligible"],
            filter=ds.field("split_role") != SplitRole.LOCKED_TEST.value,
        ).to_pylist(),
    )
    eligible: dict[str, dict[str, Any]] = {}
    role_counts: Counter[str] = Counter()
    allowed_roles = {
        SplitRole.DEV.value,
        SplitRole.IMAGE_ONLY.value,
        SplitRole.NORMAL_REPLAY.value,
    }
    for row in safe_rows:
        role = _required_text(row, "split_role", "real manifest")
        if row.get("training_eligible") is not True or role not in allowed_roles:
            continue
        image_id = _required_text(row, "image_id", "real manifest")
        label = _required_text(row, "image_label", "real manifest")
        if label not in {item.value for item in ImageLabel}:
            raise ValueError("real manifest has malformed image_label")
        eligible[image_id] = row
        role_counts[role] += 1
    source_summary: dict[str, object] = {
        "source_id": dict(sorted(sources.items())),
        "license_id": dict(sorted(licenses.items())),
        "production_allowed": dict(sorted(production.items())),
    }
    return eligible, sorted(locked_ids), source_summary, role_counts


def _load_folds(config: RealSeedReportConfig) -> dict[str, int]:
    rows = cast(list[dict[str, Any]], pq.read_table(config.fold_manifest).to_pylist())
    folds: dict[str, int] = {}
    for row in rows:
        image_id = _required_text(row, "image_id", "fold manifest")
        fold = row.get("fold")
        role = _required_text(row, "split_role", "fold manifest")
        if image_id in folds or type(fold) is not int or fold < -1:
            raise ValueError("fold manifest has duplicate ID or malformed fold")
        if role == SplitRole.LOCKED_TEST.value and fold != -1:
            raise ValueError("locked-test fold must be -1")
        if role != SplitRole.LOCKED_TEST.value and fold < 0:
            raise ValueError("development fold must be nonnegative")
        folds[image_id] = fold
    return folds


def _load_character_rows(
    config: RealSeedReportConfig,
    folds: dict[str, int],
    locked: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = cast(list[dict[str, Any]], pq.read_table(config.character_oof).to_pylist())
    by_id: dict[str, dict[str, Any]] = {}
    fold_hash = _sha256(config.fold_manifest)
    for row in rows:
        crop_id = _required_text(row, "crop_id", "character OOF")
        image_id = _required_text(row, "image_id", "character OOF")
        fold = row.get("fold")
        if crop_id in by_id:
            raise ValueError(f"character OOF has duplicate crop_id: {crop_id}")
        if image_id in locked:
            raise ValueError("character OOF contains locked-test score")
        if image_id not in folds or type(fold) is not int or fold != folds[image_id]:
            raise ValueError("character OOF fold mismatch")
        decision = _required_text(row, "decision", "character OOF")
        if decision not in {item.value for item in Decision}:
            raise ValueError("character OOF has malformed decision")
        _required_probability(row, "risk_score", "character OOF")
        if _required_text(row, "model_id", "character OOF") != f"real-fold-{fold}":
            raise ValueError("character OOF model/fold mismatch")
        _valid_sha(row.get("checkpoint_sha256"), "character checkpoint_sha256")
        _valid_sha(row.get("parent_checkpoint_sha256"), "character parent_checkpoint_sha256")
        if (
            _valid_sha(row.get("gold_manifest_sha256"), "character gold_manifest_sha256")
            != _sha256(config.gold_manifest)
        ):
            raise ValueError("character OOF gold manifest hash mismatch")
        if (
            _valid_sha(row.get("fold_manifest_sha256"), "character fold_manifest_sha256")
            != fold_hash
        ):
            raise ValueError("character OOF fold manifest hash mismatch")
        by_id[crop_id] = row

    ocr_rows = cast(list[dict[str, Any]], pq.read_table(config.ocr_manifest).to_pylist())
    ocr_by_id: dict[str, dict[str, Any]] = {}
    audit_hash = _sha256(config.ocr_audit)
    for row in ocr_rows:
        crop_id = _required_text(row, "crop_id", "OCR manifest")
        if crop_id in ocr_by_id:
            raise ValueError(f"OCR manifest has duplicate crop_id: {crop_id}")
        expected = by_id.get(crop_id)
        if expected is None:
            raise ValueError("OCR manifest contains extra crop_id")
        for field in ("image_id", "fold", "decision", "anomaly_kind"):
            if row.get(field) != expected.get(field):
                raise ValueError(f"OCR and character OOF {field} mismatch")
        _required_probability(row, "ocr_confidence", "OCR manifest")
        _required_text(row, "ocr_model_id", "OCR manifest")
        if _valid_sha(row.get("ocr_audit_sha256"), "ocr_audit_sha256") != audit_hash:
            raise ValueError("OCR manifest audit hash mismatch")
        ocr_by_id[crop_id] = row
    if set(ocr_by_id) != set(by_id):
        raise ValueError("OCR manifest must cover exactly the character OOF IDs")
    return [by_id[key] for key in sorted(by_id)], [ocr_by_id[key] for key in sorted(by_id)]


def _load_image_rows(
    config: RealSeedReportConfig,
    eligible: dict[str, dict[str, Any]],
    folds: dict[str, int],
    locked: set[str],
) -> list[dict[str, Any]]:
    rows = cast(list[dict[str, Any]], pq.read_table(config.image_oof).to_pylist())
    by_id: dict[str, dict[str, Any]] = {}
    trusted_hashes = {
        "real_manifest_sha256": _sha256(config.real_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
        "feature_manifest_sha256": _sha256(config.character_oof),
    }
    for row in rows:
        image_id = _required_text(row, "image_id", "image OOF")
        if image_id in by_id:
            raise ValueError(f"image OOF has duplicate image_id: {image_id}")
        if image_id in locked:
            raise ValueError("image OOF contains locked-test score")
        expected = eligible.get(image_id)
        if expected is None:
            raise ValueError("image OOF contains ineligible image")
        fold = row.get("fold")
        if type(fold) is not int or fold != folds.get(image_id):
            raise ValueError("image OOF assigned fold mismatch")
        if row.get("held_out_fold") != fold:
            raise ValueError("image OOF scoring fold mismatch")
        if row.get("image_label") != expected["image_label"]:
            raise ValueError("image OOF label mismatch")
        if not isinstance(row.get("zero_character"), bool):
            raise ValueError("image OOF has malformed zero_character")
        _required_probability(row, "risk_score", "image OOF")
        _valid_sha(row.get("checkpoint_sha256"), "image checkpoint_sha256")
        for field, value in trusted_hashes.items():
            if _valid_sha(row.get(field), field) != value:
                raise ValueError(f"image OOF {field} mismatch")
        by_id[image_id] = row
    if set(by_id) != set(eligible):
        raise ValueError("image OOF must contain exactly one row per eligible development image")
    return [by_id[key] for key in sorted(by_id)]


def _input_hashes(config: RealSeedReportConfig) -> dict[str, str]:
    paths = {
        "real_manifest": config.real_manifest,
        "fold_manifest": config.fold_manifest,
        "crop_manifest": config.crop_manifest,
        "gold_manifest": config.gold_manifest,
        "character_oof": config.character_oof,
        "image_oof": config.image_oof,
        "ocr_manifest": config.ocr_manifest,
        "ocr_audit": config.ocr_audit,
        "common_chars_3500": config.common_chars,
        "source_lock": config.source_lock,
        "dependency_lock": config.dependency_lock,
        **{f"source:{name}": path for name, path in config.additional_source_artifacts.items()},
        **{f"model:{name}": path for name, path in config.model_artifacts.items()},
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"required provenance file does not exist: {name}={path}")
        hashes[name] = _sha256(path)
    return hashes


def _l20_commands() -> list[str]:
    return [
        "uv run poor-word ocr audit --endpoint http://127.0.0.1:8765 --image-dir data/real/seed/images --warmup 10 --runs 30 --output artifacts/ocr-audit-l20.json  # PP-OCRv5_server_det + PP-OCRv5_server_rec",
        "uv run poor-word train adapt-real --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet --real-manifest data/real/versioned/seed-v1/manifest.parquet --prior-checkpoint artifacts/glyph-mvp-v1/encoder.pt --output-dir artifacts/glyph-real-adapt-v1 --device cuda --epochs 20 --batch-size 64",
        "uv run poor-word train real-oof --real-manifest data/real/versioned/seed-v1/manifest.parquet --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet --gold-manifest data/real/versioned/seed-v1/gold-v1/gold-crops.parquet --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet --synthetic-manifest data/generated/mvp-v1/manifest.parquet --adapted-checkpoint artifacts/glyph-real-adapt-v1/encoder.pt --output-dir artifacts/real-oof-v1 --device cuda --epochs 10 --batch-size 64",
        "uv run poor-word train mil --real-manifest data/real/versioned/seed-v1/manifest.parquet --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet --feature-manifest artifacts/real-oof-v1/oof/oof.parquet --held-out-fold 0 --output-dir artifacts/mil-v1/fold-0 --device cuda --epochs 30 --batch-size 32",
        "uv run poor-word evaluate real-seed --real-manifest data/real/versioned/seed-v1/manifest.parquet --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet --gold-manifest data/real/versioned/seed-v1/gold-v1/gold-crops.parquet --character-oof artifacts/real-oof-v1/oof/oof.parquet --image-oof artifacts/mil-v1/image-oof.parquet --ocr-manifest artifacts/ocr-character-scores.parquet --ocr-audit artifacts/ocr-audit-l20.json --common-chars data/raw/common_chars_3500.txt --source-lock data/locks/common_chars_3500.lock.json --dependency-lock uv.lock --prevalence 0.001 --output-dir artifacts/real-seed-report-v1",
    ]


def _markdown(report: dict[str, object]) -> str:
    failures = cast(list[str], report["failures"])
    commands = cast(list[str], report["l20_commands"])
    hashes = cast(dict[str, str], cast(dict[str, object], report["provenance"])["inputs"])
    return "\n".join(
        [
            "# Not a pilot approval",
            "",
            "本报告仅为真实种子开发集 OOF 诊断。现网异常基率按 0.1%（0.001）重算 PPV，",
            "不得把平衡集或挖掘样本的 observed precision 当作生产 precision。",
            "",
            "## 为什么仍为 inconclusive",
            "",
            *(f"- {item}" for item in failures),
            "",
            "合法 Unicode CJK 但不在 3500 常用字表的字符只进入 REVIEW；不得自动封禁罕见字。",
            "文字规则异常与视觉乱码证据在 JSON 中分别记录。",
            "",
            "## 下一步",
            "",
            "补足至少 10,000 个真实正常/正常 replay 负样本和足量真实 BLOCK，完成受控 locked-test 运行后再申请 pilot。",
            "",
            "## 输入 SHA-256",
            "",
            *(f"- `{name}`: `{value}`" for name, value in sorted(hashes.items())),
            "",
            "## NVIDIA L20 / Python 3.12 / uv 命令（仅审计，未由本报告执行）",
            "",
            "PP-OCRv5 使用 server 模型；OCR audit 命令显式记录 endpoint 和模型名称。",
            "",
            "```bash",
            *commands,
            "```",
            "",
        ]
    )


def evaluate_real_seed(config: RealSeedReportConfig) -> RealSeedReportArtifacts:
    """Validate aligned development OOF artifacts and publish an auditable report."""
    if config.output_dir.exists():
        raise ValueError(f"report output directory already exists: {config.output_dir}")
    input_hashes = _input_hashes(config)
    eligible, locked_ids, source_summary, role_counts = _load_real_contract(config)
    folds = _load_folds(config)
    if not set(eligible).issubset(folds) or not set(locked_ids).issubset(folds):
        raise ValueError("fold manifest does not cover real manifest IDs")
    character_rows, ocr_rows = _load_character_rows(config, folds, set(locked_ids))
    image_rows = _load_image_rows(config, eligible, folds, set(locked_ids))
    common_chars = frozenset(
        line.strip()
        for line in config.common_chars.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not common_chars:
        raise ValueError("common-character catalog is empty")

    baseline_rows: list[dict[str, Any]] = []
    for row in ocr_rows:
        baseline = score_baselines(
            BaselineInput(
                crop_id=str(row["crop_id"]),
                image_id=str(row["image_id"]),
                fold=int(row["fold"]),
                text=str(row["text"]),
                ocr_confidence=float(row["ocr_confidence"]),
                ocr_model_id=str(row["ocr_model_id"]),
                ocr_audit_sha256=str(row["ocr_audit_sha256"]),
                visual_anomaly_score=row.get("visual_anomaly_score"),
            ),
            common_chars,
            block_threshold=config.threshold,
        )
        baseline_rows.append({**row, **baseline.model_dump()})

    character_metrics = {
        "glyph": _metric_summary(
            character_rows,
            score_field="risk_score",
            label_field="decision",
            positive_value=Decision.BLOCK.value,
            review_value=Decision.REVIEW.value,
            threshold=config.threshold,
            prevalence=config.prevalence,
        ),
        "ocr_confidence": _metric_summary(
            baseline_rows,
            score_field="ocr_confidence_risk",
            label_field="decision",
            positive_value=Decision.BLOCK.value,
            review_value=Decision.REVIEW.value,
            threshold=config.threshold,
            prevalence=config.prevalence,
        ),
        "membership": _metric_summary(
            baseline_rows,
            score_field="membership_risk",
            label_field="decision",
            positive_value=Decision.BLOCK.value,
            review_value=Decision.REVIEW.value,
            threshold=config.threshold,
            prevalence=config.prevalence,
        ),
        "ocr_plus_rule": _metric_summary(
            baseline_rows,
            score_field="combined_risk",
            label_field="decision",
            positive_value=Decision.BLOCK.value,
            review_value=Decision.REVIEW.value,
            threshold=config.threshold,
            prevalence=config.prevalence,
        ),
    }
    image_metrics = {
        "mil": _metric_summary(
            image_rows,
            score_field="risk_score",
            label_field="image_label",
            positive_value=ImageLabel.ABNORMAL.value,
            threshold=config.threshold,
            prevalence=config.prevalence,
            zero_field="zero_character",
        )
    }

    glyph_counts = cast(dict[str, int], character_metrics["glyph"]["counts"])
    image_counts = cast(dict[str, int], image_metrics["mil"]["counts"])
    failures: list[str] = ["locked test was not run by this task"]
    warnings: list[str] = []
    if glyph_counts["negative"] < config.minimum_negative_count:
        failures.append(
            f"character normal support {glyph_counts['negative']} < {config.minimum_negative_count}; FPR<=1e-4 cannot be established"
        )
    if image_counts["negative"] < config.minimum_negative_count:
        failures.append(
            f"image normal support {image_counts['negative']} < {config.minimum_negative_count}; FPR<=1e-4 cannot be established"
        )
    if glyph_counts["positive"] < config.minimum_real_positive_count:
        failures.append(
            f"real character BLOCK support {glyph_counts['positive']} < {config.minimum_real_positive_count}"
        )
    if image_counts["positive"] < config.minimum_real_positive_count:
        failures.append(
            f"real abnormal image support {image_counts['positive']} < {config.minimum_real_positive_count}"
        )
    if role_counts[SplitRole.NORMAL_REPLAY.value] == 0:
        failures.append("normal replay is absent")
    rare_count = sum(row["out_of_catalog"] is True for row in baseline_rows)
    if rare_count:
        warnings.append(f"{rare_count} OCR characters are out_of_catalog and review-only")

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "inconclusive" if failures else "development_gate_only",
        "pilot_approved": False,
        "prevalence": config.prevalence,
        "threshold": config.threshold,
        "character_metrics": character_metrics,
        "image_metrics": image_metrics,
        "baseline_policy": {
            "rare_cjk": "out_of_catalog/needs_review; never auto-BLOCK without visual evidence",
            "membership_block_score_semantics": "valid rare CJK membership risk stays below block threshold",
            "text_and_visual_evidence_separate": True,
            "eligible_ids_sha256": hashlib.sha256(
                "\n".join(str(row["crop_id"]) for row in character_rows).encode()
            ).hexdigest(),
        },
        "source_summary": source_summary,
        "locked_test_access": {
            "image_ids": locked_ids,
            "count": len(locked_ids),
            "evaluation_score_rows": 0,
            "pixels_accessed": False,
            "labels_accessed": False,
        },
        "provenance": {
            "inputs": input_hashes,
            "character_model_checkpoints": sorted(
                {str(row["checkpoint_sha256"]) for row in character_rows}
            ),
            "image_model_checkpoints": sorted(
                {str(row["checkpoint_sha256"]) for row in image_rows}
            ),
            "self_hashes": "report-provenance.json",
        },
        "failures": failures,
        "warnings": warnings,
        "l20_commands": _l20_commands(),
        "l20_commands_executed_by_report": False,
    }
    markdown = _markdown(report)
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = config.output_dir.with_name(f".{config.output_dir.name}.part-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        json_path = staging / "report.json"
        markdown_path = staging / "report.md"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(markdown, encoding="utf-8")
        provenance_path = staging / "report-provenance.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "report_json_sha256": _sha256(json_path),
                    "report_markdown_sha256": _sha256(markdown_path),
                    "input_sha256": input_hashes,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(config.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RealSeedReportArtifacts(
        config.output_dir / "report.json",
        config.output_dir / "report.md",
        config.output_dir / "report-provenance.json",
    )
