"""Nested character-score features for leakage-safe image-level MIL OOF."""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from pydantic import BaseModel, ConfigDict, Field

from poor_word.real_data.schema import SplitRole
from poor_word.training.finetune_real import RealFineTuneConfig, finetune_real_fold


class NestedOofConfig(BaseModel):
    """Inputs for all outer-fold-aware character features used by MIL OOF."""

    model_config = ConfigDict(frozen=True)

    real_manifest: Path
    crop_manifest: Path
    gold_manifest: Path
    fold_manifest: Path
    synthetic_manifest: Path
    adapted_checkpoint: Path
    output_dir: Path
    epochs: int = Field(default=10, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=64, ge=2)
    seed: int = Field(default=20260804, ge=0)
    device: str = "cuda"
    learning_rate: float = Field(default=1e-4, gt=0)


@dataclass(frozen=True)
class NestedOofArtifacts:
    manifest: Path
    output_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_folds(path: Path) -> list[int]:
    rows = cast(
        list[dict[str, Any]],
        pq.read_table(path, columns=["image_id", "fold", "split_role"]).to_pylist(),
    )
    folds: set[int] = set()
    image_ids: set[str] = set()
    for row in rows:
        image_id = row.get("image_id")
        fold = row.get("fold")
        role = row.get("split_role")
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ValueError("fold manifest has malformed or duplicate image_id")
        image_ids.add(image_id)
        if type(fold) is not int or not isinstance(role, str):
            raise ValueError("fold manifest has malformed fold or split_role")
        if role == SplitRole.LOCKED_TEST.value:
            if fold != -1:
                raise ValueError("locked-test image must have fold=-1")
            continue
        if fold < 0:
            raise ValueError("development image has negative fold")
        folds.add(fold)
    if folds != set(range(5)):
        raise ValueError("nested character OOF requires development folds 0..4")
    return sorted(folds)


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError("nested character artifact escapes output root") from error


def train_nested_real_oof(config: NestedOofConfig) -> NestedOofArtifacts:
    """Atomically train all ordered outer/scoring-fold character models.

    For outer MIL fold ``k``, rows belonging to scoring fold ``j`` are emitted
    only by a model whose real supervised training excludes ``{k, j}``.
    """
    if config.output_dir.exists():
        raise ValueError(f"nested OOF output directory already exists: {config.output_dir}")
    for path in (
        config.real_manifest,
        config.crop_manifest,
        config.gold_manifest,
        config.fold_manifest,
        config.synthetic_manifest,
        config.adapted_checkpoint,
    ):
        if not path.is_file():
            raise ValueError(f"required input does not exist: {path}")
    folds = _development_folds(config.fold_manifest)
    hashes = {
        "real_manifest_sha256": _sha256(config.real_manifest),
        "crop_manifest_sha256": _sha256(config.crop_manifest),
        "gold_manifest_sha256": _sha256(config.gold_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
        "synthetic_manifest_sha256": _sha256(config.synthetic_manifest),
        "parent_checkpoint_sha256": _sha256(config.adapted_checkpoint),
    }
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = config.output_dir.with_name(f".{config.output_dir.name}.part-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    outer_entries: list[dict[str, object]] = []
    try:
        for outer_fold in folds:
            outer_dir = staging / f"outer-fold-{outer_fold}"
            outer_dir.mkdir()
            feature_rows: list[dict[str, object]] = []
            model_entries: list[dict[str, object]] = []
            scoring_ids_by_fold: dict[int, set[str]] = {}
            training_ids_by_fold: dict[int, tuple[str, ...]] = {}
            excluded_by_fold: dict[int, tuple[int, ...]] = {}
            for scoring_fold in folds:
                excluded_folds = tuple(sorted({outer_fold, scoring_fold}))
                artifact = finetune_real_fold(
                    RealFineTuneConfig(
                        real_manifest=config.real_manifest,
                        crop_manifest=config.crop_manifest,
                        gold_manifest=config.gold_manifest,
                        fold_manifest=config.fold_manifest,
                        synthetic_manifest=config.synthetic_manifest,
                        adapted_checkpoint=config.adapted_checkpoint,
                        output_dir=outer_dir / f"character-fold-{scoring_fold}",
                        epochs=config.epochs,
                        max_steps=config.max_steps,
                        batch_size=config.batch_size,
                        seed=config.seed,
                        device=config.device,
                        learning_rate=config.learning_rate,
                    ),
                    scoring_fold,
                    None if scoring_fold == outer_fold else outer_fold,
                )
                if artifact.excluded_folds != excluded_folds:
                    raise ValueError("nested character artifact excluded-fold contract mismatch")
                checkpoint_raw = torch.load(
                    artifact.checkpoint, map_location="cpu", weights_only=True
                )
                if not isinstance(checkpoint_raw, dict):
                    raise ValueError("nested character checkpoint is malformed")
                checkpoint = cast(dict[str, Any], checkpoint_raw)
                training_ids = checkpoint.get("real_training_crop_ids")
                if (
                    checkpoint.get("held_out_fold") != scoring_fold
                    or checkpoint.get("excluded_folds") != list(excluded_folds)
                    or checkpoint.get("scoring_crop_ids") != list(artifact.scoring_crop_ids)
                    or not isinstance(training_ids, list)
                    or not all(isinstance(item, str) and item for item in training_ids)
                    or len(training_ids) != len(set(training_ids))
                ):
                    raise ValueError("nested character checkpoint membership is malformed")
                scoring_ids_by_fold[scoring_fold] = set(artifact.scoring_crop_ids)
                training_ids_by_fold[scoring_fold] = tuple(training_ids)
                excluded_by_fold[scoring_fold] = excluded_folds
                score_rows = cast(
                    list[dict[str, Any]],
                    pq.read_table(
                        artifact.scores,
                        columns=[
                            "crop_id",
                            "image_id",
                            "fold",
                            "risk_score",
                            "model_id",
                            "checkpoint_sha256",
                            "fold_manifest_sha256",
                            "excluded_folds",
                        ],
                    ).to_pylist(),
                )
                expected_crop_ids = set(artifact.scoring_crop_ids)
                actual_crop_ids = {str(row.get("crop_id")) for row in score_rows}
                if actual_crop_ids != expected_crop_ids:
                    raise ValueError("nested character scores do not exactly cover held-out crops")
                for row in score_rows:
                    if row.get("fold") != scoring_fold or row.get("excluded_folds") != list(
                        excluded_folds
                    ):
                        raise ValueError("nested character score provenance mismatch")
                    if row.get("fold_manifest_sha256") != hashes["fold_manifest_sha256"]:
                        raise ValueError("nested character score fold manifest mismatch")
                    feature_rows.append({**row, "outer_fold": outer_fold})
                model_entries.append(
                    {
                        "scoring_fold": scoring_fold,
                        "excluded_folds": list(excluded_folds),
                        "checkpoint": _relative(staging, artifact.checkpoint),
                        "checkpoint_sha256": _sha256(artifact.checkpoint),
                        "metrics": _relative(staging, artifact.metrics),
                        "metrics_sha256": _sha256(artifact.metrics),
                        "scores": _relative(staging, artifact.scores),
                        "scores_sha256": _sha256(artifact.scores),
                        "scoring_crop_ids": list(artifact.scoring_crop_ids),
                    }
                )
            all_scoring_ids = set().union(*scoring_ids_by_fold.values())
            for scoring_fold, training_ids in training_ids_by_fold.items():
                forbidden = set().union(
                    *(scoring_ids_by_fold[fold] for fold in excluded_by_fold[scoring_fold])
                )
                if not set(training_ids).issubset(all_scoring_ids) or set(training_ids) & forbidden:
                    raise ValueError("nested character training includes an excluded-fold crop")
            feature_rows.sort(
                key=lambda row: (cast(int, row["fold"]), str(row["crop_id"]))
            )
            features = outer_dir / "features.parquet"
            pq.write_table(
                pa.Table.from_pylist(feature_rows),
                features,
                compression="zstd",
                version="2.6",
            )
            outer_entries.append(
                {
                    "outer_fold": outer_fold,
                    "features": _relative(staging, features),
                    "features_sha256": _sha256(features),
                    "models": model_entries,
                }
            )
        manifest = staging / "nested-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "nested_character_oof",
                    "outer_folds": outer_entries,
                    **hashes,
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
    return NestedOofArtifacts(config.output_dir / "nested-manifest.json", config.output_dir)
