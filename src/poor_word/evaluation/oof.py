"""Strict collection and diagnostics for character-level out-of-fold scores."""

import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

from poor_word.domain import Decision
from poor_word.real_data.schema import SplitRole
from poor_word.training.finetune_real import FoldModelArtifacts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_aucpr(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    *,
    seed: int,
    samples: int = 200,
) -> list[float | None]:
    if len(np.unique(labels)) != 2:
        return [None, None]
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(labels), size=len(labels))
        selected_labels = labels[indices]
        if len(np.unique(selected_labels)) != 2:
            continue
        values.append(float(average_precision_score(selected_labels, scores[indices])))
    if not values:
        return [None, None]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _fold_metrics(rows: list[dict[str, Any]], fold: int) -> dict[str, object]:
    labeled = [
        row
        for row in rows
        if row["decision"] in {Decision.PASS.value, Decision.BLOCK.value}
    ]
    labels = np.asarray(
        [row["decision"] == Decision.BLOCK.value for row in labeled], dtype=np.int64
    )
    scores = np.asarray([row["risk_score"] for row in labeled], dtype=np.float64)
    two_class = len(np.unique(labels)) == 2
    aucpr = float(average_precision_score(labels, scores)) if two_class else None
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        if row["decision"] == Decision.BLOCK.value:
            by_kind[str(row["anomaly_kind"])].append(row)
    recalls = {
        kind: float(np.mean([float(row["risk_score"]) >= 0.5 for row in kind_rows]))
        for kind, kind_rows in sorted(by_kind.items())
    }
    return {
        "row_count": len(rows),
        "labeled_count": len(labeled),
        "pass_count": int((labels == 0).sum()),
        "block_count": int((labels == 1).sum()),
        "review_count": len(rows) - len(labeled),
        "aucpr": aucpr,
        "bootstrap_aucpr_95": _bootstrap_aucpr(labels, scores, seed=20260804 + fold),
        "anomaly_kind_recall": recalls,
        "diagnostic_threshold": 0.5,
        "status": "ok" if two_class else "inconclusive_single_class",
    }


def collect_oof_scores(
    fold_artifacts: list[FoldModelArtifacts] | tuple[FoldModelArtifacts, ...],
    fold_manifest: Path,
    output_dir: Path | None = None,
) -> Path:
    """Validate fold provenance and atomically publish one score per held-out crop."""
    if not fold_artifacts:
        raise ValueError("at least one fold artifact is required")
    destination = output_dir or fold_artifacts[0].checkpoint.parent.parent / "oof"
    if destination.exists():
        raise ValueError(f"OOF output directory already exists: {destination}")
    manifest_hash = _sha256(fold_manifest)
    fold_rows = cast(list[dict[str, Any]], pq.read_table(fold_manifest).to_pylist())
    assignments: dict[str, tuple[int, str]] = {}
    valid_roles = {role.value for role in SplitRole}
    for row in fold_rows:
        image_id = row.get("image_id")
        fold = row.get("fold")
        role = row.get("split_role")
        if not isinstance(image_id, str) or not image_id or image_id in assignments:
            raise ValueError("fold manifest has missing or duplicate image_id")
        if type(fold) is not int:
            raise ValueError("fold manifest has malformed fold")
        if not isinstance(role, str) or role not in valid_roles:
            raise ValueError("fold manifest has malformed split_role")
        assignments[image_id] = (fold, role)

    rows: list[dict[str, Any]] = []
    seen_folds: set[int] = set()
    seen_crops: set[str] = set()
    for artifact in sorted(fold_artifacts, key=lambda item: item.held_out_fold):
        fold = artifact.held_out_fold
        if fold < 0 or fold in seen_folds:
            raise ValueError(f"duplicate or invalid held-out fold artifact: {fold}")
        seen_folds.add(fold)
        checkpoint_hash = _sha256(artifact.checkpoint)
        checkpoint_raw = torch.load(
            artifact.checkpoint, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint_raw, dict):
            raise ValueError(f"fold {fold} checkpoint metadata is malformed")
        checkpoint = cast(dict[str, object], checkpoint_raw)
        if checkpoint.get("held_out_fold") != fold:
            raise ValueError(f"fold {fold} checkpoint held-out fold mismatch")
        if checkpoint.get("fold_manifest_sha256") != manifest_hash:
            raise ValueError(f"fold {fold} checkpoint fold manifest hash mismatch")
        checkpoint_scoring = checkpoint.get("scoring_crop_ids")
        if not isinstance(checkpoint_scoring, list) or not all(
            isinstance(crop_id, str) and crop_id for crop_id in checkpoint_scoring
        ):
            raise ValueError(f"fold {fold} checkpoint scoring IDs are malformed")
        if tuple(checkpoint_scoring) != artifact.scoring_crop_ids:
            raise ValueError(f"fold {fold} artifact/checkpoint scoring IDs mismatch")
        artifact_rows = cast(list[dict[str, Any]], pq.read_table(artifact.scores).to_pylist())
        actual_ids: set[str] = set()
        for row in artifact_rows:
            crop_id = row.get("crop_id")
            image_id = row.get("image_id")
            row_fold = row.get("fold")
            risk = row.get("risk_score")
            decision = row.get("decision")
            anomaly_kind = row.get("anomaly_kind")
            if not isinstance(crop_id, str) or not crop_id:
                raise ValueError("OOF score has malformed crop_id")
            if crop_id in seen_crops or crop_id in actual_ids:
                raise ValueError(f"duplicate OOF crop_id: {crop_id}")
            if not isinstance(image_id, str) or image_id not in assignments:
                raise ValueError(f"OOF score has unknown image_id: {crop_id}")
            assigned_fold, role = assignments[image_id]
            if role == SplitRole.LOCKED_TEST.value or assigned_fold == -1:
                raise ValueError(f"locked-test row mixed into OOF scores: {crop_id}")
            if row_fold != fold or assigned_fold != fold:
                raise ValueError(f"OOF score/model/fold mismatch: {crop_id}")
            if row.get("fold_manifest_sha256") != manifest_hash:
                raise ValueError(f"OOF fold manifest hash mismatch: {crop_id}")
            if row.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError(f"OOF checkpoint hash mismatch: {crop_id}")
            if decision not in {item.value for item in Decision}:
                raise ValueError(f"OOF score has malformed decision: {crop_id}")
            if not isinstance(anomaly_kind, str) or not anomaly_kind:
                raise ValueError(f"OOF score has malformed anomaly_kind: {crop_id}")
            if (
                isinstance(risk, bool)
                or not isinstance(risk, (int, float))
                or not np.isfinite(risk)
                or not 0 <= risk <= 1
            ):
                raise ValueError(f"OOF score has malformed risk_score: {crop_id}")
            actual_ids.add(crop_id)
            rows.append(row)
        expected_ids = set(artifact.scoring_crop_ids)
        if len(expected_ids) != len(artifact.scoring_crop_ids) or actual_ids != expected_ids:
            raise ValueError(f"fold {fold} has missing or unexpected OOF crop scores")
        seen_crops.update(actual_ids)

    development_folds = {
        fold for fold, role in assignments.values() if fold >= 0 and role != SplitRole.LOCKED_TEST
    }
    if seen_folds != development_folds:
        raise ValueError("OOF fold artifacts do not cover every development fold")

    rows.sort(key=lambda row: (int(row["fold"]), str(row["crop_id"])))
    metrics_by_fold = {
        str(fold): _fold_metrics([row for row in rows if row["fold"] == fold], fold)
        for fold in sorted(seen_folds)
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        output = staging / "oof.parquet"
        pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd", version="2.6")
        (staging / "metrics.json").write_text(
            json.dumps(
                {
                    "fold_manifest_sha256": manifest_hash,
                    "folds": metrics_by_fold,
                    "oof_row_count": len(rows),
                    "locked_test_rows": 0,
                    "metric_scope": "development OOF only",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "oof.parquet"
