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
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

from poor_word.data.manifest import load_source_lock
from poor_word.domain import Decision
from poor_word.evaluation.baselines import BaselineInput, score_baselines
from poor_word.evaluation.metrics import base_rate_precision
from poor_word.glyphs.catalog import load_common_chars
from poor_word.ocr.types import OcrAudit
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
    character_model_inventory: Path
    image_model_inventory: Path
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


def _diagnostic_ranking_summary(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    threshold: float,
    prevalence: float,
) -> dict[str, object]:
    summary = _metric_summary(
        rows,
        score_field=score_field,
        label_field="decision",
        positive_value=Decision.BLOCK.value,
        review_value=Decision.REVIEW.value,
        threshold=threshold,
        prevalence=prevalence,
    )
    for field in (
        "confusion",
        "recall",
        "recall_wilson_95",
        "fpr",
        "fpr_wilson_95",
        "base_rate_precision",
        "anomaly_kind_recall",
        "diagnostic_threshold",
    ):
        summary.pop(field)
    summary["scope"] = "diagnostic ranking/review risk only; never an automatic BLOCK policy"
    return summary


def _load_real_contract(
    config: RealSeedReportConfig,
) -> tuple[
    dict[str, dict[str, Any]],
    list[str],
    dict[str, object],
    Counter[str],
    dict[str, str],
]:
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
    roles: dict[str, str] = {}
    for row in metadata:
        image_id = _required_text(row, "image_id", "real manifest")
        if image_id in seen:
            raise ValueError(f"real manifest has duplicate image_id: {image_id}")
        seen.add(image_id)
        role = _required_text(row, "split_role", "real manifest")
        if role not in {item.value for item in SplitRole}:
            raise ValueError("real manifest has malformed split_role")
        roles[image_id] = role
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
    return eligible, sorted(locked_ids), source_summary, role_counts, roles


def _load_folds(
    config: RealSeedReportConfig, real_roles: dict[str, str]
) -> dict[str, dict[str, Any]]:
    rows = cast(
        list[dict[str, Any]],
        pq.read_table(
            config.fold_manifest,
            columns=["image_id", "fold", "split_role", "component_id"],
        ).to_pylist(),
    )
    folds: dict[str, dict[str, Any]] = {}
    component_folds: dict[str, int] = {}
    for row in rows:
        image_id = _required_text(row, "image_id", "fold manifest")
        fold = row.get("fold")
        role = _required_text(row, "split_role", "fold manifest")
        component = _required_text(row, "component_id", "fold manifest")
        if image_id in folds or image_id not in real_roles or type(fold) is not int or fold < -1:
            raise ValueError("fold manifest has duplicate ID or malformed fold")
        if role != real_roles[image_id]:
            raise ValueError("real/fold split_role mismatch")
        previous = component_folds.setdefault(component, fold)
        if previous != fold:
            raise ValueError("component_id crosses folds")
        if role == SplitRole.LOCKED_TEST.value and fold != -1:
            raise ValueError("locked-test fold must be -1")
        if role != SplitRole.LOCKED_TEST.value and fold < 0:
            raise ValueError("development fold must be nonnegative")
        folds[image_id] = row
    if set(folds) != set(real_roles):
        raise ValueError("fold manifest IDs do not exactly match real manifest")
    return folds


def _load_character_rows(
    config: RealSeedReportConfig,
    folds: dict[str, dict[str, Any]],
    locked: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    crop_links_raw = cast(
        list[dict[str, Any]],
        pq.read_table(config.crop_manifest, columns=["crop_id", "image_id"]).to_pylist(),
    )
    gold_links_raw = cast(
        list[dict[str, Any]],
        pq.read_table(config.gold_manifest, columns=["crop_id", "image_id"]).to_pylist(),
    )
    crop_links: dict[str, str] = {}
    for row in crop_links_raw:
        crop_id = _required_text(row, "crop_id", "crop manifest")
        image_id = _required_text(row, "image_id", "crop manifest")
        if crop_id in crop_links or image_id not in folds:
            raise ValueError("crop manifest has duplicate crop_id or unknown image_id")
        crop_links[crop_id] = image_id
    gold_links: dict[str, str] = {}
    allowed_ids: set[str] = set()
    for row in gold_links_raw:
        crop_id = _required_text(row, "crop_id", "gold manifest")
        image_id = _required_text(row, "image_id", "gold manifest")
        if crop_id in gold_links:
            raise ValueError("gold manifest has duplicate crop_id")
        if crop_links.get(crop_id) != image_id:
            raise ValueError("gold/crop trusted linkage mismatch")
        gold_links[crop_id] = image_id
        fold_row = folds.get(image_id)
        if fold_row is None:
            raise ValueError("gold manifest has unknown image_id")
        if (
            image_id not in locked
            and fold_row["split_role"] == SplitRole.DEV.value
            and cast(int, fold_row["fold"]) >= 0
        ):
            allowed_ids.add(crop_id)
    if not allowed_ids:
        raise ValueError("trusted gold cohort has no development crops")
    filters = [("crop_id", "in", sorted(allowed_ids))]
    crop_rows = cast(
        list[dict[str, Any]],
        pq.read_table(
            config.crop_manifest,
            columns=["crop_id", "image_id", "crop_path"],
            filters=filters,
        ).to_pylist(),
    )
    gold_columns = ["crop_id", "image_id", "crop_path", "decision"]
    if "anomaly_kind" in pq.ParquetFile(config.gold_manifest).schema_arrow.names:
        gold_columns.append("anomaly_kind")
    gold_rows = cast(
        list[dict[str, Any]],
        pq.read_table(config.gold_manifest, columns=gold_columns, filters=filters).to_pylist(),
    )
    crops = {str(row["crop_id"]): row for row in crop_rows}
    gold = {str(row["crop_id"]): row for row in gold_rows}
    if set(crops) != allowed_ids or set(gold) != allowed_ids:
        raise ValueError("trusted filtered crop/gold cohort is incomplete")
    trusted: dict[str, dict[str, Any]] = {}
    for crop_id in sorted(allowed_ids):
        crop = crops[crop_id]
        label = gold[crop_id]
        image_id = crop_links[crop_id]
        if crop.get("image_id") != image_id or label.get("image_id") != image_id:
            raise ValueError("trusted crop/gold image linkage changed during filtered read")
        crop_path = _required_text(crop, "crop_path", "crop manifest")
        if _required_text(label, "crop_path", "gold manifest") != crop_path:
            raise ValueError("trusted crop/gold path mismatch")
        decision = _required_text(label, "decision", "gold manifest")
        if decision not in {item.value for item in Decision}:
            raise ValueError("trusted gold has malformed decision")
        anomaly_raw = label.get("anomaly_kind")
        anomaly_kind = (
            anomaly_raw
            if isinstance(anomaly_raw, str) and anomaly_raw
            else ("NONE" if decision == Decision.PASS.value else "unknown")
        )
        trusted[crop_id] = {
            "crop_id": crop_id,
            "image_id": image_id,
            "crop_path": crop_path,
            "decision": decision,
            "anomaly_kind": anomaly_kind,
            "fold": folds[image_id]["fold"],
        }

    rows = cast(list[dict[str, Any]], pq.read_table(config.character_oof).to_pylist())
    by_id: dict[str, dict[str, Any]] = {}
    trusted_hashes = {
        "real_manifest_sha256": _sha256(config.real_manifest),
        "crop_manifest_sha256": _sha256(config.crop_manifest),
        "gold_manifest_sha256": _sha256(config.gold_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
    }
    for row in rows:
        crop_id = _required_text(row, "crop_id", "character OOF")
        image_id = _required_text(row, "image_id", "character OOF")
        fold = row.get("fold")
        if crop_id in by_id:
            raise ValueError(f"character OOF has duplicate crop_id: {crop_id}")
        expected = trusted.get(crop_id)
        if expected is None:
            raise ValueError("character OOF contains a crop outside trusted gold cohort")
        if image_id in locked:
            raise ValueError("character OOF contains locked-test score")
        if type(fold) is not int or fold != expected["fold"]:
            raise ValueError("character OOF fold mismatch")
        decision = _required_text(row, "decision", "character OOF")
        if decision not in {item.value for item in Decision}:
            raise ValueError("character OOF has malformed decision")
        for field in ("image_id", "crop_path", "decision", "anomaly_kind"):
            if row.get(field) != expected[field]:
                raise ValueError(f"character OOF does not match trusted gold {field}")
        _required_probability(row, "risk_score", "character OOF")
        if _required_text(row, "model_id", "character OOF") != f"real-fold-{fold}":
            raise ValueError("character OOF model/fold mismatch")
        _valid_sha(row.get("checkpoint_sha256"), "character checkpoint_sha256")
        _valid_sha(row.get("parent_checkpoint_sha256"), "character parent_checkpoint_sha256")
        for field, value in trusted_hashes.items():
            if _valid_sha(row.get(field), f"character {field}") != value:
                raise ValueError(f"character OOF {field} mismatch")
        by_id[crop_id] = row
    if set(by_id) != set(trusted):
        raise ValueError("character OOF does not exactly cover trusted gold cohort")

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
        if _required_text(row, "ocr_model_id", "OCR manifest") != "PP-OCRv5_server_rec":
            raise ValueError("OCR manifest must use PP-OCRv5_server_rec")
        if _valid_sha(row.get("ocr_audit_sha256"), "ocr_audit_sha256") != audit_hash:
            raise ValueError("OCR manifest audit hash mismatch")
        ocr_by_id[crop_id] = row
    if set(ocr_by_id) != set(by_id):
        raise ValueError("OCR manifest must cover exactly the character OOF IDs")
    return [by_id[key] for key in sorted(by_id)], [ocr_by_id[key] for key in sorted(by_id)]


def _load_image_rows(
    config: RealSeedReportConfig,
    eligible: dict[str, dict[str, Any]],
    folds: dict[str, dict[str, Any]],
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
        fold_row = folds.get(image_id)
        if type(fold) is not int or fold_row is None or fold != fold_row["fold"]:
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
        "character_model_inventory": config.character_model_inventory,
        "image_model_inventory": config.image_model_inventory,
        **{f"source:{name}": path for name, path in config.additional_source_artifacts.items()},
        **{f"model:{name}": path for name, path in config.model_artifacts.items()},
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"required provenance file does not exist: {name}={path}")
        hashes[name] = _sha256(path)
    return hashes


def _inventory_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"model inventory {field} must be a safe relative path")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"model inventory {field} escapes its directory") from error
    if not resolved.is_file():
        raise ValueError(f"model inventory {field} does not exist")
    return resolved


def _validate_model_inventory(
    inventory_path: Path,
    rows: list[dict[str, Any]],
    *,
    kind: str,
    id_field: str,
    score_name: str,
    trusted_hashes: dict[str, str],
) -> dict[str, str]:
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ValueError(f"{kind} model inventory is malformed")
    entries = payload.get("folds")
    if not isinstance(entries, list):
        raise ValueError(f"{kind} model inventory has malformed folds")
    expected_by_fold: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        fold = row.get("fold")
        item_id = row.get(id_field)
        if type(fold) is not int or not isinstance(item_id, str):
            raise ValueError(f"{kind} OOF row has malformed fold or ID")
        expected_by_fold[fold][item_id] = row
    if len(entries) != len(expected_by_fold):
        raise ValueError(f"{kind} model inventory fold count mismatch")
    root = inventory_path.parent
    seen: set[int] = set()
    model_hashes: dict[str, str] = {}
    for entry_raw in entries:
        if not isinstance(entry_raw, dict):
            raise ValueError(f"{kind} model inventory entry is malformed")
        entry = cast(dict[str, Any], entry_raw)
        fold = entry.get("held_out_fold")
        if type(fold) is not int or fold in seen or fold not in expected_by_fold:
            raise ValueError(f"{kind} model inventory has duplicate or unexpected fold")
        seen.add(fold)
        checkpoint_path = _inventory_path(root, entry.get("checkpoint"), "checkpoint")
        metrics_path = _inventory_path(root, entry.get("metrics"), "metrics")
        scores_path = _inventory_path(root, entry.get("scores"), "scores")
        checkpoint_hash = _sha256(checkpoint_path)
        metrics_hash = _sha256(metrics_path)
        scores_hash = _sha256(scores_path)
        for field, actual_hash in (
            ("checkpoint_sha256", checkpoint_hash),
            ("metrics_sha256", metrics_hash),
            ("scores_sha256", scores_hash),
        ):
            if _valid_sha(entry.get(field), f"inventory {field}") != actual_hash:
                raise ValueError(f"{kind} inventory {field} mismatch")
        checkpoint_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint_raw, dict):
            raise ValueError(f"{kind} checkpoint is malformed")
        checkpoint = cast(dict[str, Any], checkpoint_raw)
        if checkpoint.get("held_out_fold") != fold:
            raise ValueError(f"{kind} checkpoint held_out_fold mismatch")
        expected_ids = sorted(expected_by_fold[fold])
        checkpoint_id_field = (
            "scoring_crop_ids" if kind == "character_oof" else "validation_image_ids"
        )
        if checkpoint.get(checkpoint_id_field) != expected_ids:
            raise ValueError(f"{kind} checkpoint scoring IDs mismatch trusted contract")
        training_field = "real_training_crop_ids" if kind == "character_oof" else "train_image_ids"
        expected_training = sorted(
            item_id
            for other_fold, items in expected_by_fold.items()
            if other_fold != fold
            for item_id, item in items.items()
            if kind == "image_oof"
            or item.get("decision") in {Decision.PASS.value, Decision.BLOCK.value}
        )
        if checkpoint.get(training_field) != expected_training:
            raise ValueError(f"{kind} checkpoint training/scoring isolation failed")
        for field, expected_hash in trusted_hashes.items():
            if checkpoint.get(field) != expected_hash:
                raise ValueError(f"{kind} checkpoint {field} mismatch")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError(f"{kind} metrics are malformed")
        metrics_ids = "scoring_crop_ids" if kind == "character_oof" else "validation_image_ids"
        score_hash_field = "scores_sha256" if kind == "character_oof" else "image_scores_sha256"
        if (
            metrics.get("held_out_fold") != fold
            or metrics.get("checkpoint_sha256") != checkpoint_hash
            or metrics.get(score_hash_field) != scores_hash
            or metrics.get(metrics_ids) != expected_ids
        ):
            raise ValueError(f"{kind} metrics provenance mismatch")
        fold_rows = cast(list[dict[str, Any]], pq.read_table(scores_path).to_pylist())
        actual_by_id = {str(row[id_field]): row for row in fold_rows}
        if len(actual_by_id) != len(fold_rows) or set(actual_by_id) != set(expected_ids):
            raise ValueError(f"{kind} fold scores do not exactly match trusted IDs")
        for item_id in expected_ids:
            actual_row = actual_by_id[item_id]
            merged = expected_by_fold[fold][item_id]
            if actual_row.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError(f"{kind} fold score checkpoint hash mismatch")
            for field in (id_field, "image_id", "fold", score_name, "checkpoint_sha256"):
                if actual_row.get(field) != merged.get(field):
                    raise ValueError(f"{kind} merged/fold score {field} mismatch")
        model_hashes[f"fold-{fold}"] = checkpoint_hash
    if seen != set(expected_by_fold):
        raise ValueError(f"{kind} model inventory is missing folds")
    return model_hashes


def _l20_commands() -> list[str]:
    return [
        "uv run poor-word ocr audit --endpoint http://127.0.0.1:8765 --image-dir data/real/seed/images --warmup 10 --runs 30 --output artifacts/ocr-audit-l20.json  # PP-OCRv5_server_det + PP-OCRv5_server_rec",
        "uv run poor-word train adapt-real --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet --real-manifest data/real/versioned/seed-v1/manifest.parquet --prior-checkpoint artifacts/glyph-mvp-v1/encoder.pt --output-dir artifacts/glyph-real-adapt-v1 --device cuda --epochs 20 --batch-size 64",
        "uv run poor-word train real-oof --real-manifest data/real/versioned/seed-v1/manifest.parquet --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet --gold-manifest data/real/versioned/seed-v1/gold-v1/gold-crops.parquet --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet --synthetic-manifest data/generated/mvp-v1/manifest.parquet --adapted-checkpoint artifacts/glyph-real-adapt-v1/encoder.pt --output-dir artifacts/real-oof-v1 --device cuda --epochs 10 --batch-size 64",
        "uv run poor-word train mil-oof --real-manifest data/real/versioned/seed-v1/manifest.parquet --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet --feature-manifest artifacts/real-oof-v1/oof/oof.parquet --output-dir artifacts/mil-oof-v1 --device cuda --epochs 30 --batch-size 32",
        "uv run poor-word evaluate real-seed --real-manifest data/real/versioned/seed-v1/manifest.parquet --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet --gold-manifest data/real/versioned/seed-v1/gold-v1/gold-crops.parquet --character-oof artifacts/real-oof-v1/oof/oof.parquet --image-oof artifacts/mil-oof-v1/image-oof.parquet --ocr-manifest artifacts/ocr-character-scores.parquet --ocr-audit artifacts/ocr-audit-l20.json --common-chars data/raw/common_chars_3500.txt --source-lock data/locks/common_chars_3500.lock.json --dependency-lock uv.lock --character-inventory artifacts/real-oof-v1/model-inventory.json --image-inventory artifacts/mil-oof-v1/model-inventory.json --prevalence 0.001 --output-dir artifacts/real-seed-report-v1",
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
    audit = OcrAudit.model_validate_json(config.ocr_audit.read_text(encoding="utf-8"))
    if audit.detection_model_name != "PP-OCRv5_server_det":
        raise ValueError("OCR audit must use PP-OCRv5_server_det")
    if audit.recognition_model_name != "PP-OCRv5_server_rec":
        raise ValueError("OCR audit must use PP-OCRv5_server_rec")
    if audit.character_boxes_available is not True:
        raise ValueError("OCR audit must confirm character boxes")
    eligible, locked_ids, source_summary, role_counts, real_roles = _load_real_contract(config)
    folds = _load_folds(config, real_roles)
    if not set(eligible).issubset(folds) or not set(locked_ids).issubset(folds):
        raise ValueError("fold manifest does not cover real manifest IDs")
    character_rows, ocr_rows = _load_character_rows(config, folds, set(locked_ids))
    image_rows = _load_image_rows(config, eligible, folds, set(locked_ids))
    character_model_hashes = _validate_model_inventory(
        config.character_model_inventory,
        character_rows,
        kind="character_oof",
        id_field="crop_id",
        score_name="risk_score",
        trusted_hashes={
            "real_manifest_sha256": _sha256(config.real_manifest),
            "crop_manifest_sha256": _sha256(config.crop_manifest),
            "gold_manifest_sha256": _sha256(config.gold_manifest),
            "fold_manifest_sha256": _sha256(config.fold_manifest),
        },
    )
    image_model_hashes = _validate_model_inventory(
        config.image_model_inventory,
        image_rows,
        kind="image_oof",
        id_field="image_id",
        score_name="risk_score",
        trusted_hashes={
            "real_manifest_sha256": _sha256(config.real_manifest),
            "fold_manifest_sha256": _sha256(config.fold_manifest),
            "feature_manifest_sha256": _sha256(config.character_oof),
        },
    )
    common_chars = frozenset(load_common_chars(config.common_chars, expected_count=3500))
    source_lock = load_source_lock(config.source_lock)
    if source_lock.output_name != config.common_chars.name:
        raise ValueError("common-character source lock output_name mismatch")
    if source_lock.sha256 != _sha256(config.common_chars):
        raise ValueError("common-character source lock SHA-256 mismatch")
    if source_lock.size_bytes != config.common_chars.stat().st_size:
        raise ValueError("common-character source lock size mismatch")
    if source_lock.production_allowed is not True:
        raise ValueError("common-character source is not approved for production")
    source_summary["common_chars_catalog"] = {
        "source_id": source_lock.source_id,
        "license_id": source_lock.license_id,
        "production_allowed": source_lock.production_allowed,
    }

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
        "ocr_confidence_review_risk": _diagnostic_ranking_summary(
            baseline_rows,
            score_field="ocr_confidence_risk",
            threshold=config.threshold,
            prevalence=config.prevalence,
        ),
        "membership_review_risk": _diagnostic_ranking_summary(
            baseline_rows,
            score_field="membership_risk",
            threshold=config.threshold,
            prevalence=config.prevalence,
        ),
        "ocr_plus_rule_review_risk": _diagnostic_ranking_summary(
            baseline_rows,
            score_field="combined_risk",
            threshold=config.threshold,
            prevalence=config.prevalence,
        ),
        "ocr_plus_rule_block": _metric_summary(
            baseline_rows,
            score_field="auto_block_score",
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
            "character_model_checkpoints": sorted(character_model_hashes.values()),
            "image_model_checkpoints": sorted(image_model_hashes.values()),
            "self_hashes": "report-provenance.json",
        },
        "failures": failures,
        "warnings": warnings,
        "ocr_capabilities": {
            "detection_model_name": audit.detection_model_name,
            "recognition_model_name": audit.recognition_model_name,
            "character_boxes_available": audit.character_boxes_available,
            "logits_available": audit.logits_available,
            "required_capability_gaps": list(audit.required_capability_gaps),
        },
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
