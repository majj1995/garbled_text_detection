"""Leakage-safe image-level MIL training from held-out character evidence."""

import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn import functional as functional

from poor_word.models.mil import AttentionMilPool
from poor_word.real_data.schema import ImageLabel, SplitRole


class MilTrainConfig(BaseModel):
    """Immutable inputs for one held-out MIL fold."""

    model_config = ConfigDict(frozen=True)

    real_manifest: Path
    fold_manifest: Path
    feature_manifest: Path
    output_dir: Path
    held_out_fold: int = Field(ge=0)
    outer_fold: int | None = Field(default=None, ge=0)
    nested_manifest_sha256: str | None = None
    epochs: int = Field(default=30, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1)
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=1e-4, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    normal_instance_weight: float = Field(default=0.25, ge=0)
    hidden_dim: int = Field(default=32, ge=1)
    seed: int = Field(default=20260804, ge=0)
    device: str = "cuda"


class MilOofConfig(BaseModel):
    """Inputs for atomic training and collection of all five MIL OOF folds."""

    model_config = ConfigDict(frozen=True)

    real_manifest: Path
    fold_manifest: Path
    feature_manifest: Path | None = None
    nested_feature_dir: Path | None = None
    output_dir: Path
    epochs: int = Field(default=30, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1)
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=1e-4, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    normal_instance_weight: float = Field(default=0.25, ge=0)
    hidden_dim: int = Field(default=32, ge=1)
    seed: int = Field(default=20260804, ge=0)
    device: str = "cuda"


@dataclass(frozen=True)
class MilArtifacts:
    checkpoint: Path
    metrics: Path
    attention_candidates: Path
    image_scores: Path


@dataclass(frozen=True)
class MilOofArtifacts:
    image_oof: Path
    inventory: Path
    metrics: Path


@dataclass(frozen=True)
class MilLoss:
    total: Tensor
    bag: Tensor
    normal_instance: Tensor
    abnormal_instance: Tensor


@dataclass(frozen=True)
class _Feature:
    crop_id: str
    risk_score: float


@dataclass(frozen=True)
class _Bag:
    image_id: str
    label: float
    features: tuple[_Feature, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: dict[str, Any], field: str, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} has missing or malformed {field}")
    return value


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def compute_mil_loss(
    bag_logits: Tensor,
    instance_logits: Tensor,
    mask: Tensor,
    image_labels: Tensor,
    *,
    pos_weight: float,
    normal_instance_weight: float = 0.25,
) -> MilLoss:
    """Apply bag supervision to every image and negative pressure only to normal instances."""
    if bag_logits.ndim != 1 or image_labels.shape != bag_logits.shape:
        raise ValueError("bag logits and image labels must have shape [B]")
    if instance_logits.shape != mask.shape or mask.dtype is not torch.bool:
        raise ValueError("instance logits and boolean mask must have shape [B,N]")
    if image_labels.dtype not in {torch.float32, torch.float64}:
        raise ValueError("image labels must be floating point")
    weight = torch.tensor(pos_weight, device=bag_logits.device, dtype=bag_logits.dtype)
    bag = functional.binary_cross_entropy_with_logits(
        bag_logits, image_labels.to(bag_logits.dtype), pos_weight=weight
    )
    normal_mask = mask & (image_labels == 0).unsqueeze(1)
    if normal_mask.any():
        normal_instance = functional.binary_cross_entropy_with_logits(
            instance_logits[normal_mask], torch.zeros_like(instance_logits[normal_mask])
        )
    else:
        normal_instance = instance_logits.sum() * 0.0
    abnormal_instance = instance_logits.sum() * 0.0
    total = bag + normal_instance_weight * normal_instance
    return MilLoss(total, bag, normal_instance, abnormal_instance)


def _load_inputs(
    config: MilTrainConfig,
) -> tuple[list[_Bag], list[_Bag], dict[str, int], list[dict[str, object]]]:
    if config.feature_manifest is None:
        raise ValueError("MIL fold requires a feature manifest")
    for path in (config.real_manifest, config.fold_manifest, config.feature_manifest):
        if not path.is_file():
            raise ValueError(f"required manifest does not exist: {path}")
    real_dataset = ds.dataset(config.real_manifest, format="parquet")
    real_metadata = cast(
        list[dict[str, Any]],
        real_dataset.to_table(
            columns=["image_id", "split_role", "training_eligible"]
        ).to_pylist(),
    )
    real_rows = cast(
        list[dict[str, Any]],
        real_dataset.to_table(
            columns=["image_id", "image_label", "split_role", "training_eligible"],
            filter=ds.field("split_role") != SplitRole.LOCKED_TEST.value,
        ).to_pylist(),
    )
    fold_rows = cast(
        list[dict[str, Any]],
        pq.read_table(
            config.fold_manifest,
            columns=["image_id", "fold", "split_role", "component_id"],
        ).to_pylist(),
    )

    real: dict[str, dict[str, Any]] = {}
    allowed_roles = {role.value for role in SplitRole}
    allowed_labels = {label.value for label in ImageLabel}
    for row in real_metadata:
        image_id = _required_text(row, "image_id", "real manifest")
        if image_id in real:
            raise ValueError(f"real manifest has duplicate image_id: {image_id}")
        role = row.get("split_role")
        eligible = row.get("training_eligible")
        if not isinstance(role, str) or role not in allowed_roles:
            raise ValueError("real manifest has malformed split_role")
        if not isinstance(eligible, bool):
            raise ValueError("real manifest has malformed training_eligible")
        real[image_id] = row
    safe_ids: set[str] = set()
    for row in real_rows:
        image_id = _required_text(row, "image_id", "real manifest")
        if image_id in safe_ids or image_id not in real:
            raise ValueError("real manifest has duplicate or unknown development image_id")
        safe_ids.add(image_id)
        if row.get("split_role") != real[image_id]["split_role"]:
            raise ValueError("real manifest development projection mismatch")
        label = row.get("image_label")
        if not isinstance(label, str) or label not in allowed_labels:
            raise ValueError("real manifest has malformed image_label")
        real[image_id]["image_label"] = label
    expected_safe = {
        image_id
        for image_id, row in real.items()
        if row["split_role"] != SplitRole.LOCKED_TEST.value
    }
    if safe_ids != expected_safe:
        raise ValueError("real manifest development projection is incomplete")

    folds: dict[str, dict[str, Any]] = {}
    component_folds: dict[str, int] = {}
    for row in fold_rows:
        image_id = _required_text(row, "image_id", "fold manifest")
        if image_id in folds:
            raise ValueError(f"fold manifest has duplicate image_id: {image_id}")
        if image_id not in real:
            raise ValueError(f"fold manifest has unknown image_id: {image_id}")
        fold = row.get("fold")
        role = row.get("split_role")
        if type(fold) is not int or fold < -1:
            raise ValueError("fold manifest has malformed fold")
        if not isinstance(role, str) or role not in allowed_roles:
            raise ValueError("fold manifest has malformed split_role")
        if role != real[image_id]["split_role"]:
            raise ValueError("real and fold manifest split_role mismatch")
        component = row.get("component_id")
        if not isinstance(component, str) or not component:
            raise ValueError("fold manifest requires nonempty component_id")
        previous = component_folds.setdefault(component, fold)
        if previous != fold:
            raise ValueError("component_id crosses folds")
        folds[image_id] = row
    if set(folds) != set(real):
        raise ValueError("fold manifest image IDs do not match real manifest")

    feature_columns = [
        "crop_id",
        "image_id",
        "fold",
        "risk_score",
        "model_id",
        "checkpoint_sha256",
        "fold_manifest_sha256",
    ]
    feature_identities = cast(
        list[dict[str, Any]],
        pq.read_table(config.feature_manifest, columns=["crop_id", "image_id"]).to_pylist(),
    )
    seen_feature_ids: set[str] = set()
    for identity in feature_identities:
        crop_id = _required_text(identity, "crop_id", "feature manifest")
        image_id = _required_text(identity, "image_id", "feature manifest")
        if crop_id in seen_feature_ids:
            raise ValueError(f"feature manifest has duplicate crop_id: {crop_id}")
        seen_feature_ids.add(crop_id)
        if image_id not in real:
            raise ValueError(f"feature manifest has unknown image_id: {image_id}")
        if image_id not in safe_ids:
            raise ValueError(f"feature manifest contains locked-test feature: {crop_id}")
    feature_filters: list[tuple[str, str, object]] = [("image_id", "in", sorted(safe_ids))]
    if config.outer_fold is not None:
        feature_columns.extend(["held_out_fold", "outer_fold", "excluded_folds"])
        feature_filters.append(("outer_fold", "=", config.outer_fold))
    feature_rows = cast(
        list[dict[str, Any]],
        pq.read_table(
            config.feature_manifest,
            columns=feature_columns,
            filters=feature_filters,
        ).to_pylist(),
    )

    fold_hash = _sha256(config.fold_manifest)
    features: dict[str, list[_Feature]] = {image_id: [] for image_id in real}
    crop_ids: set[str] = set()
    feature_models: set[tuple[str, str, int, tuple[int, ...]]] = set()
    fold_model_claims: dict[int, tuple[str, str, tuple[int, ...]]] = {}
    for row in feature_rows:
        crop_id = _required_text(row, "crop_id", "feature manifest")
        image_id = _required_text(row, "image_id", "feature manifest")
        if crop_id in crop_ids:
            raise ValueError(f"feature manifest has duplicate crop_id: {crop_id}")
        crop_ids.add(crop_id)
        if image_id not in real:
            raise ValueError(f"feature manifest has unknown image_id: {image_id}")
        image_fold = folds[image_id]["fold"]
        if image_fold == -1 or real[image_id]["split_role"] == SplitRole.LOCKED_TEST.value:
            raise ValueError(f"feature manifest contains locked-test feature: {crop_id}")
        feature_fold = row.get("held_out_fold", row.get("fold"))
        if type(feature_fold) is not int:
            raise ValueError("feature manifest has malformed scoring fold")
        if feature_fold != image_fold:
            raise ValueError("feature scoring fold does not match assigned image fold")
        excluded_folds: tuple[int, ...] = (feature_fold,)
        if config.outer_fold is not None:
            if row.get("outer_fold") != config.outer_fold:
                raise ValueError("nested feature outer fold mismatch")
            raw_excluded = row.get("excluded_folds")
            if (
                not isinstance(raw_excluded, list)
                or not all(type(item) is int for item in raw_excluded)
                or tuple(sorted(set(raw_excluded))) != tuple(raw_excluded)
            ):
                raise ValueError("nested feature has malformed excluded_folds")
            excluded_folds = tuple(raw_excluded)
            expected_excluded = tuple(sorted({config.outer_fold, feature_fold}))
            if excluded_folds != expected_excluded:
                raise ValueError("nested feature excluded-fold contract mismatch")
        risk = row.get("risk_score", row.get("evidence"))
        if (
            not isinstance(risk, (int, float))
            or isinstance(risk, bool)
            or not math.isfinite(risk)
            or not 0.0 <= risk <= 1.0
        ):
            raise ValueError("feature manifest requires finite risk_score within [0,1]")
        risk32 = float(torch.tensor(float(risk), dtype=torch.float32).item())
        if not math.isfinite(risk32) or not 0.0 <= risk32 <= 1.0:
            raise ValueError("feature manifest requires finite risk_score within [0,1]")
        model_id = _required_text(row, "model_id", "feature manifest")
        if model_id != f"real-fold-{feature_fold}":
            raise ValueError("feature manifest requires canonical model_id for scoring fold")
        checkpoint_hash = _required_text(row, "checkpoint_sha256", "feature manifest")
        if len(checkpoint_hash) != 64 or any(
            char not in "0123456789abcdef" for char in checkpoint_hash
        ):
            raise ValueError("feature manifest has malformed checkpoint_sha256")
        claimed_fold_hash = _required_text(row, "fold_manifest_sha256", "feature manifest")
        if claimed_fold_hash != fold_hash:
            raise ValueError("feature fold manifest hash mismatch")
        claim = (model_id, checkpoint_hash, excluded_folds)
        previous_claim = fold_model_claims.setdefault(feature_fold, claim)
        if previous_claim != claim:
            raise ValueError("feature manifest has conflicting model provenance for scoring fold")
        feature_models.add((model_id, checkpoint_hash, feature_fold, excluded_folds))
        features[image_id].append(_Feature(crop_id, risk32))

    train: list[_Bag] = []
    validation: list[_Bag] = []
    for image_id in sorted(real):
        row = real[image_id]
        fold = folds[image_id]["fold"]
        role = row["split_role"]
        if (
            fold == -1
            or role == SplitRole.LOCKED_TEST.value
            or row["training_eligible"] is not True
        ):
            continue
        if role not in {
            SplitRole.DEV.value,
            SplitRole.IMAGE_ONLY.value,
            SplitRole.NORMAL_REPLAY.value,
        }:
            continue
        bag = _Bag(
            image_id,
            float(row["image_label"] == ImageLabel.ABNORMAL.value),
            tuple(sorted(features[image_id], key=lambda item: item.crop_id)),
        )
        (validation if fold == config.held_out_fold else train).append(bag)
    if not validation:
        raise ValueError("held-out fold has no eligible validation images")
    normal = sum(bag.label == 0 for bag in train)
    abnormal = sum(bag.label == 1 for bag in train)
    if not normal or not abnormal:
        raise ValueError("training fold requires both NORMAL and ABNORMAL images")
    validation_labels = {bag.label for bag in validation}
    if validation_labels != {0.0, 1.0}:
        raise ValueError("validation fold requires both NORMAL and ABNORMAL images")
    model_provenance = []
    for model_id, checkpoint_hash, scoring_fold, excluded_folds in sorted(feature_models):
        item: dict[str, object] = {
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_hash,
            "scoring_fold": scoring_fold,
        }
        if config.outer_fold is not None:
            item["excluded_folds"] = list(excluded_folds)
        model_provenance.append(item)
    return train, validation, {"NORMAL": normal, "ABNORMAL": abnormal}, model_provenance


def _batch(
    bags: list[_Bag], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, list[list[str]]]:
    width = max(1, max(len(bag.features) for bag in bags))
    values = torch.zeros((len(bags), width, 1), dtype=torch.float32, device=device)
    mask = torch.zeros((len(bags), width), dtype=torch.bool, device=device)
    crop_ids: list[list[str]] = []
    for bag_index, bag in enumerate(bags):
        ids: list[str] = []
        for feature_index, feature in enumerate(bag.features):
            values[bag_index, feature_index, 0] = feature.risk_score
            mask[bag_index, feature_index] = True
            ids.append(feature.crop_id)
        crop_ids.append(ids)
    labels = torch.tensor([bag.label for bag in bags], dtype=torch.float32, device=device)
    return values, mask, labels, crop_ids


def _instance_logits(model: AttentionMilPool, features: Tensor) -> Tensor:
    scale = functional.softplus(model.raw_evidence_scale)
    if model.feature_dim == 1:
        context = torch.zeros_like(features)
    else:
        context = features[..., 1:]
    return cast(
        Tensor,
        scale * features[..., 0] + model.context_head(context).squeeze(-1),
    )


def _evaluate(
    model: AttentionMilPool,
    bags: list[_Bag],
    device: torch.device,
    pos_weight: float,
    instance_weight: float,
) -> tuple[float, list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    features, mask, labels, crop_ids = _batch(bags, device)
    with torch.inference_mode():
        logits, attention = model(features, mask)
        loss = compute_mil_loss(
            logits,
            _instance_logits(model, features),
            mask,
            labels,
            pos_weight=pos_weight,
            normal_instance_weight=instance_weight,
        )
    candidates: list[dict[str, object]] = []
    image_scores: list[dict[str, object]] = []
    for index, bag in enumerate(bags):
        risk_score = float(torch.sigmoid(logits[index]).cpu())
        image_scores.append(
            {
                "image_id": bag.image_id,
                "fold": None,
                "held_out_fold": None,
                "image_label": ImageLabel.ABNORMAL.value
                if bag.label == 1.0
                else ImageLabel.NORMAL.value,
                "risk_score": risk_score,
                "zero_character": not crop_ids[index],
            }
        )
        if not crop_ids[index]:
            continue
        best = int(torch.argmax(attention[index]).item())
        candidates.append(
            {
                "image_id": bag.image_id,
                "crop_id": crop_ids[index][best],
                "held_out_fold": None,
                "attention": float(attention[index, best].cpu()),
                "image_risk_score": risk_score,
                "label_source": "mil_attention_candidate",
                "is_gold": False,
            }
        )
    return float(loss.total.cpu()), candidates, image_scores


def _parquet_bytes(rows: list[dict[str, object]]) -> bytes:
    schema = pa.schema(
        [
            ("image_id", pa.string()),
            ("crop_id", pa.string()),
            ("held_out_fold", pa.int64()),
            ("attention", pa.float64()),
            ("image_risk_score", pa.float64()),
            ("label_source", pa.string()),
            ("is_gold", pa.bool_()),
            ("checkpoint_sha256", pa.string()),
            ("real_manifest_sha256", pa.string()),
            ("fold_manifest_sha256", pa.string()),
            ("feature_manifest_sha256", pa.string()),
            ("nested_manifest_sha256", pa.string()),
        ]
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), sink, compression="zstd", version="2.6"
    )
    return cast(bytes, sink.getvalue().to_pybytes())


def _image_score_parquet_bytes(rows: list[dict[str, object]]) -> bytes:
    schema = pa.schema(
        [
            ("image_id", pa.string()),
            ("fold", pa.int64()),
            ("held_out_fold", pa.int64()),
            ("image_label", pa.string()),
            ("risk_score", pa.float64()),
            ("zero_character", pa.bool_()),
            ("checkpoint_sha256", pa.string()),
            ("real_manifest_sha256", pa.string()),
            ("fold_manifest_sha256", pa.string()),
            ("feature_manifest_sha256", pa.string()),
            ("nested_manifest_sha256", pa.string()),
        ]
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), sink, compression="zstd", version="2.6"
    )
    return cast(bytes, sink.getvalue().to_pybytes())


def train_mil_fold(config: MilTrainConfig) -> MilArtifacts:
    """Train and atomically publish the best image-level MIL model for one fold."""
    if config.output_dir.exists():
        raise ValueError(f"MIL output directory already exists: {config.output_dir}")
    if config.feature_manifest is None:
        raise ValueError("MIL fold requires a feature manifest")
    train, validation, counts, feature_model_provenance = _load_inputs(config)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {config.device}")
    device = torch.device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = AttentionMilPool(1, config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    pos_weight = counts["NORMAL"] / counts["ABNORMAL"]
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    steps = 0
    history: list[dict[str, object]] = []
    stop_reason = "epochs_complete"
    generator = random.Random(config.seed)
    for epoch in range(config.epochs):
        order = list(train)
        generator.shuffle(order)
        model.train()
        train_losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            selected = order[start : start + config.batch_size]
            features, mask, labels, _ = _batch(selected, device)
            logits, _ = model(features, mask)
            losses = compute_mil_loss(
                logits,
                _instance_logits(model, features),
                mask,
                labels,
                pos_weight=pos_weight,
                normal_instance_weight=config.normal_instance_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            train_losses.append(float(losses.total.detach().cpu()))
            steps += 1
            if config.max_steps is not None and steps >= config.max_steps:
                break
        validation_loss, _, _ = _evaluate(
            model, validation, device, pos_weight, config.normal_instance_weight
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(train_losses) / len(train_losses),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                stop_reason = "early_stopping"
                break
        if config.max_steps is not None and steps >= config.max_steps:
            stop_reason = "max_steps" if stale_epochs < config.patience else stop_reason
            break
    if best_state is None:
        raise RuntimeError("MIL training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    _, candidates, image_scores = _evaluate(
        model, validation, device, pos_weight, config.normal_instance_weight
    )
    for row in candidates:
        row["held_out_fold"] = config.held_out_fold
    for row in image_scores:
        row["fold"] = config.held_out_fold
        row["held_out_fold"] = config.held_out_fold

    hashes = {
        "real_manifest_sha256": _sha256(config.real_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
        "feature_manifest_sha256": _sha256(config.feature_manifest),
    }
    if config.nested_manifest_sha256 is not None:
        hashes["nested_manifest_sha256"] = config.nested_manifest_sha256
    train_ids = sorted(bag.image_id for bag in train)
    validation_ids = sorted(bag.image_id for bag in validation)
    zero_validation = sorted(bag.image_id for bag in validation if not bag.features)
    repo_root = Path(__file__).resolve().parents[3]
    provenance = {
        "seed": config.seed,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "dependency_lock_sha256": _sha256(repo_root / "uv.lock")
        if (repo_root / "uv.lock").is_file()
        else None,
    }
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = config.output_dir.with_name(f".{config.output_dir.name}.part-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        checkpoint = staging / "model.pt"
        torch.save(
            {
                "model_state": best_state,
                "feature_dim": 1,
                "hidden_dim": config.hidden_dim,
                "held_out_fold": config.held_out_fold,
                "train_image_ids": train_ids,
                "validation_image_ids": validation_ids,
                "class_counts": counts,
                "pos_weight": pos_weight,
                "best_epoch": best_epoch,
                "stop_reason": stop_reason,
                "feature_model_provenance": feature_model_provenance,
                **hashes,
                **provenance,
            },
            checkpoint,
        )
        checkpoint_hash = _sha256(checkpoint)
        for row in candidates:
            row.update({"checkpoint_sha256": checkpoint_hash, **hashes})
        for row in image_scores:
            row.update({"checkpoint_sha256": checkpoint_hash, **hashes})
        candidate_path = staging / "attention-candidates.parquet"
        candidate_path.write_bytes(_parquet_bytes(candidates))
        candidate_hash = _sha256(candidate_path)
        image_score_path = staging / "image-scores.parquet"
        image_score_path.write_bytes(_image_score_parquet_bytes(image_scores))
        image_score_hash = _sha256(image_score_path)
        metrics = {
            "held_out_fold": config.held_out_fold,
            "train_image_ids": train_ids,
            "validation_image_ids": validation_ids,
            "zero_character_validation_image_ids": zero_validation,
            "class_counts": counts,
            "pos_weight": pos_weight,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "stop_reason": stop_reason,
            "steps": steps,
            "history": history,
            "checkpoint_sha256": checkpoint_hash,
            "attention_candidates_sha256": candidate_hash,
            "image_scores_sha256": image_score_hash,
            "feature_model_provenance": feature_model_provenance,
            **hashes,
            **provenance,
        }
        (staging / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(config.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return MilArtifacts(
        config.output_dir / "model.pt",
        config.output_dir / "metrics.json",
        config.output_dir / "attention-candidates.parquet",
        config.output_dir / "image-scores.parquet",
    )


def _mil_oof_contract(config: MilOofConfig) -> tuple[dict[int, dict[str, str]], set[str]]:
    real_dataset = ds.dataset(config.real_manifest, format="parquet")
    real_metadata = cast(
        list[dict[str, Any]],
        real_dataset.to_table(
            columns=["image_id", "split_role", "training_eligible"]
        ).to_pylist(),
    )
    real_rows = cast(
        list[dict[str, Any]],
        real_dataset.to_table(
            columns=["image_id", "image_label", "split_role", "training_eligible"],
            filter=ds.field("split_role") != SplitRole.LOCKED_TEST.value,
        ).to_pylist(),
    )
    fold_rows = cast(
        list[dict[str, Any]],
        pq.read_table(
            config.fold_manifest,
            columns=["image_id", "fold", "split_role", "component_id"],
        ).to_pylist(),
    )
    real: dict[str, dict[str, Any]] = {}
    locked: set[str] = set()
    for row in real_metadata:
        image_id = _required_text(row, "image_id", "real manifest")
        if image_id in real:
            raise ValueError(f"real manifest has duplicate image_id: {image_id}")
        if not isinstance(row.get("training_eligible"), bool):
            raise ValueError("real manifest has malformed training_eligible")
        if row.get("split_role") not in {item.value for item in SplitRole}:
            raise ValueError("real manifest has malformed split_role")
        real[image_id] = row
        if row["split_role"] == SplitRole.LOCKED_TEST.value:
            locked.add(image_id)
    safe_ids: set[str] = set()
    for row in real_rows:
        image_id = _required_text(row, "image_id", "real manifest")
        if image_id in safe_ids or image_id not in real:
            raise ValueError("real manifest has duplicate or unknown development image_id")
        safe_ids.add(image_id)
        if row.get("split_role") != real[image_id]["split_role"]:
            raise ValueError("real manifest development projection mismatch")
        if row.get("image_label") not in {item.value for item in ImageLabel}:
            raise ValueError("real manifest has malformed image_label")
        real[image_id]["image_label"] = row["image_label"]
    if safe_ids != set(real) - locked:
        raise ValueError("real manifest development projection is incomplete")
    expected: dict[int, dict[str, str]] = defaultdict(dict)
    seen: set[str] = set()
    component_folds: dict[str, int] = {}
    for row in fold_rows:
        image_id = _required_text(row, "image_id", "fold manifest")
        if image_id in seen or image_id not in real:
            raise ValueError("fold manifest has duplicate or unknown image_id")
        seen.add(image_id)
        fold = row.get("fold")
        if type(fold) is not int:
            raise ValueError("fold manifest has malformed fold")
        role = _required_text(row, "split_role", "fold manifest")
        if role != real[image_id]["split_role"]:
            raise ValueError("real/fold split_role mismatch")
        component = _required_text(row, "component_id", "fold manifest")
        previous = component_folds.setdefault(component, fold)
        if previous != fold:
            raise ValueError("component_id crosses folds")
        if role == SplitRole.LOCKED_TEST.value:
            if fold != -1:
                raise ValueError("locked-test image must have fold=-1")
            continue
        if fold < 0:
            raise ValueError("development image has negative fold")
        if real[image_id]["training_eligible"] is True and role in {
            SplitRole.DEV.value,
            SplitRole.IMAGE_ONLY.value,
            SplitRole.NORMAL_REPLAY.value,
        }:
            expected[fold][image_id] = str(real[image_id]["image_label"])
    if seen != set(real):
        raise ValueError("fold manifest IDs do not exactly match real manifest")
    if set(expected) != set(range(5)):
        raise ValueError("MIL OOF requires development folds 0..4")
    return dict(expected), locked


def validate_nested_feature_dir(
    nested_feature_dir: Path,
    real_manifest: Path,
    fold_manifest: Path,
) -> tuple[dict[int, Path], str]:
    """Validate actual nested checkpoints, metrics, scores, and feature membership."""
    manifest = nested_feature_dir / "nested-manifest.json"
    if not manifest.is_file():
        raise ValueError("nested feature directory is missing nested-manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "nested_character_oof":
        raise ValueError("nested feature manifest has invalid kind")
    for field, path in (
        ("real_manifest_sha256", real_manifest),
        ("fold_manifest_sha256", fold_manifest),
    ):
        if payload.get(field) != _sha256(path):
            raise ValueError(f"nested feature manifest {field} mismatch")
    entries = payload.get("outer_folds")
    if not isinstance(entries, list):
        raise ValueError("nested feature manifest has malformed outer_folds")
    resolved: dict[int, Path] = {}
    root = nested_feature_dir.resolve()

    def inventory_path(entry: dict[str, Any], field: str) -> Path:
        relative = entry.get(field)
        if not isinstance(relative, str) or not relative:
            raise ValueError("nested model inventory has malformed artifact path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("nested model artifact path escapes directory") from error
        if not path.is_file():
            raise ValueError("nested model inventory references a missing artifact")
        return path

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("nested feature manifest entry is malformed")
        outer_fold = entry.get("outer_fold")
        relative = entry.get("features")
        expected_hash = entry.get("features_sha256")
        if type(outer_fold) is not int or not isinstance(relative, str) or not relative:
            raise ValueError("nested feature manifest entry is malformed")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("nested feature path escapes directory") from error
        if not candidate.is_file() or not isinstance(expected_hash, str):
            raise ValueError("nested feature manifest references a missing feature table")
        if _sha256(candidate) != expected_hash:
            raise ValueError("nested feature table hash mismatch")
        models = entry.get("models")
        if not isinstance(models, list):
            raise ValueError("nested feature manifest has malformed model inventory")
        validated_models: set[tuple[int, tuple[int, ...], str]] = set()
        scoring_ids_by_fold: dict[int, set[str]] = {}
        training_ids_by_fold: dict[int, tuple[str, ...]] = {}
        excluded_by_fold: dict[int, tuple[int, ...]] = {}
        for model_entry in models:
            if not isinstance(model_entry, dict):
                raise ValueError("nested model inventory entry is malformed")
            scoring_fold = model_entry.get("scoring_fold")
            raw_excluded = model_entry.get("excluded_folds")
            if (
                type(scoring_fold) is not int
                or not isinstance(raw_excluded, list)
                or not all(type(value) is int for value in raw_excluded)
            ):
                raise ValueError("nested model inventory has malformed fold provenance")
            excluded = tuple(raw_excluded)
            if excluded != tuple(sorted({outer_fold, scoring_fold})):
                raise ValueError("nested model inventory excluded-fold contract mismatch")
            checkpoint_path = inventory_path(model_entry, "checkpoint")
            metrics_path = inventory_path(model_entry, "metrics")
            scores_path = inventory_path(model_entry, "scores")
            for field, artifact_path in (
                ("checkpoint_sha256", checkpoint_path),
                ("metrics_sha256", metrics_path),
                ("scores_sha256", scores_path),
            ):
                if model_entry.get(field) != _sha256(artifact_path):
                    raise ValueError("nested model inventory artifact hash mismatch")
            checkpoint_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint_raw, dict):
                raise ValueError("nested model checkpoint is malformed")
            checkpoint = cast(dict[str, Any], checkpoint_raw)
            training_ids = checkpoint.get("real_training_crop_ids")
            if (
                checkpoint.get("held_out_fold") != scoring_fold
                or checkpoint.get("excluded_folds") != list(excluded)
                or checkpoint.get("fold_manifest_sha256") != _sha256(fold_manifest)
                or not isinstance(training_ids, list)
                or not all(isinstance(item, str) and item for item in training_ids)
                or len(training_ids) != len(set(training_ids))
            ):
                raise ValueError("nested model checkpoint provenance mismatch")
            metrics_raw = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(metrics_raw, dict) or (
                metrics_raw.get("held_out_fold") != scoring_fold
                or metrics_raw.get("excluded_folds") != list(excluded)
                or metrics_raw.get("checkpoint_sha256") != _sha256(checkpoint_path)
                or metrics_raw.get("scores_sha256") != _sha256(scores_path)
            ):
                raise ValueError("nested model metrics provenance mismatch")
            score_rows = cast(
                list[dict[str, Any]],
                pq.read_table(
                    scores_path,
                    columns=[
                        "crop_id",
                        "fold",
                        "excluded_folds",
                        "checkpoint_sha256",
                        "fold_manifest_sha256",
                    ],
                ).to_pylist(),
            )
            scoring_ids = model_entry.get("scoring_crop_ids")
            if not isinstance(scoring_ids, list) or {
                row.get("crop_id") for row in score_rows
            } != set(scoring_ids):
                raise ValueError("nested model scores do not match inventory crop IDs")
            if any(
                row.get("fold") != scoring_fold
                or row.get("excluded_folds") != list(excluded)
                or row.get("checkpoint_sha256") != _sha256(checkpoint_path)
                or row.get("fold_manifest_sha256") != _sha256(fold_manifest)
                for row in score_rows
            ):
                raise ValueError("nested model score provenance mismatch")
            model_key = (scoring_fold, excluded, _sha256(checkpoint_path))
            if model_key in validated_models:
                raise ValueError("nested model inventory has duplicate fold provenance")
            validated_models.add(model_key)
            scoring_ids_by_fold[scoring_fold] = set(cast(list[str], scoring_ids))
            training_ids_by_fold[scoring_fold] = tuple(training_ids)
            excluded_by_fold[scoring_fold] = excluded
        if {fold for fold, _excluded, _hash in validated_models} != set(range(5)):
            raise ValueError("nested model inventory requires scoring folds 0..4")
        all_scoring_ids = set().union(*scoring_ids_by_fold.values())
        for scoring_fold, training_ids in training_ids_by_fold.items():
            forbidden = set().union(
                *(scoring_ids_by_fold[fold] for fold in excluded_by_fold[scoring_fold])
            )
            if not set(training_ids).issubset(all_scoring_ids) or set(training_ids) & forbidden:
                raise ValueError("nested model training includes an excluded-fold crop")
        feature_rows = cast(
            list[dict[str, Any]],
            pq.read_table(
                candidate,
                columns=[
                    "crop_id",
                    "fold",
                    "outer_fold",
                    "excluded_folds",
                    "checkpoint_sha256",
                ],
            ).to_pylist(),
        )
        if not feature_rows or any(
            row.get("outer_fold") != outer_fold
            or type(row.get("fold")) is not int
            or not isinstance(row.get("excluded_folds"), list)
            or (
                row.get("fold"),
                tuple(cast(list[int], row["excluded_folds"])),
                row.get("checkpoint_sha256"),
            )
            not in validated_models
            for row in feature_rows
        ):
            raise ValueError("nested feature rows are not bound to model inventory")
        if outer_fold in resolved:
            raise ValueError("nested feature manifest has duplicate outer fold")
        resolved[outer_fold] = candidate
    if set(resolved) != set(range(5)):
        raise ValueError("nested feature manifest requires outer folds 0..4")
    return resolved, _sha256(manifest)


def _nested_feature_paths(config: MilOofConfig) -> tuple[dict[int, Path], str]:
    """Resolve the required nested feature directory for MIL OOF."""
    if config.feature_manifest is not None:
        raise ValueError("MIL OOF rejects ordinary feature manifests; use nested feature directory")
    if config.nested_feature_dir is None:
        raise ValueError("MIL OOF requires a nested feature directory")
    return validate_nested_feature_dir(
        config.nested_feature_dir,
        config.real_manifest,
        config.fold_manifest,
    )


def train_mil_oof(config: MilOofConfig) -> MilOofArtifacts:
    """Train five MIL folds and atomically publish complete image-level OOF scores."""
    if config.output_dir.exists():
        raise ValueError(f"MIL OOF output directory already exists: {config.output_dir}")
    for path in (config.real_manifest, config.fold_manifest):
        if not path.is_file():
            raise ValueError(f"required manifest does not exist: {path}")
    expected, locked = _mil_oof_contract(config)
    nested_features, nested_manifest_hash = _nested_feature_paths(config)
    nested_root = cast(Path, config.nested_feature_dir).resolve()
    trusted_hashes = {
        "real_manifest_sha256": _sha256(config.real_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
        "nested_manifest_sha256": nested_manifest_hash,
    }
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = config.output_dir.with_name(f".{config.output_dir.name}.part-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    rows: list[dict[str, Any]] = []
    inventory_folds: list[dict[str, object]] = []
    try:
        for fold in range(5):
            artifact = train_mil_fold(
                MilTrainConfig(
                    real_manifest=config.real_manifest,
                    fold_manifest=config.fold_manifest,
                    feature_manifest=nested_features[fold],
                    output_dir=staging / f"fold-{fold}",
                    held_out_fold=fold,
                    outer_fold=fold,
                    nested_manifest_sha256=nested_manifest_hash,
                    epochs=config.epochs,
                    max_steps=config.max_steps,
                    batch_size=config.batch_size,
                    patience=config.patience,
                    min_delta=config.min_delta,
                    learning_rate=config.learning_rate,
                    normal_instance_weight=config.normal_instance_weight,
                    hidden_dim=config.hidden_dim,
                    seed=config.seed,
                    device=config.device,
                )
            )
            checkpoint_hash = _sha256(artifact.checkpoint)
            checkpoint_raw = torch.load(artifact.checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint_raw, dict):
                raise ValueError(f"MIL fold {fold} checkpoint is malformed")
            checkpoint = cast(dict[str, Any], checkpoint_raw)
            expected_ids = sorted(expected[fold])
            if checkpoint.get("held_out_fold") != fold:
                raise ValueError(f"MIL fold {fold} checkpoint held_out_fold mismatch")
            if checkpoint.get("validation_image_ids") != expected_ids:
                raise ValueError(f"MIL fold {fold} validation IDs mismatch")
            train_ids = checkpoint.get("train_image_ids")
            if not isinstance(train_ids, list) or not all(
                isinstance(item, str) for item in train_ids
            ):
                raise ValueError(f"MIL fold {fold} train IDs are malformed")
            if set(train_ids) & set(expected_ids) or set(train_ids) & locked:
                raise ValueError(f"MIL fold {fold} training/scoring isolation failed")
            per_fold_hashes = {
                **trusted_hashes,
                "feature_manifest_sha256": _sha256(nested_features[fold]),
            }
            for field, trusted in per_fold_hashes.items():
                if checkpoint.get(field) != trusted:
                    raise ValueError(f"MIL fold {fold} checkpoint {field} mismatch")

            fold_rows = cast(list[dict[str, Any]], pq.read_table(artifact.image_scores).to_pylist())
            actual_ids: set[str] = set()
            for row in fold_rows:
                image_id = _required_text(row, "image_id", "MIL image scores")
                if image_id in actual_ids or image_id not in expected[fold]:
                    raise ValueError(f"MIL fold {fold} has duplicate or unexpected image score")
                actual_ids.add(image_id)
                if row.get("fold") != fold or row.get("held_out_fold") != fold:
                    raise ValueError(f"MIL fold {fold} score fold mismatch")
                if row.get("image_label") != expected[fold][image_id]:
                    raise ValueError(f"MIL fold {fold} score label mismatch")
                if row.get("checkpoint_sha256") != checkpoint_hash:
                    raise ValueError(f"MIL fold {fold} score checkpoint hash mismatch")
                for field, trusted in per_fold_hashes.items():
                    if row.get(field) != trusted:
                        raise ValueError(f"MIL fold {fold} score {field} mismatch")
                rows.append(row)
            if actual_ids != set(expected_ids):
                raise ValueError(f"MIL fold {fold} image scores do not exactly cover validation")
            metrics_raw = json.loads(artifact.metrics.read_text(encoding="utf-8"))
            if not isinstance(metrics_raw, dict):
                raise ValueError(f"MIL fold {fold} metrics are malformed")
            if (
                metrics_raw.get("held_out_fold") != fold
                or metrics_raw.get("checkpoint_sha256") != checkpoint_hash
                or metrics_raw.get("image_scores_sha256") != _sha256(artifact.image_scores)
                or metrics_raw.get("validation_image_ids") != expected_ids
            ):
                raise ValueError(f"MIL fold {fold} metrics provenance mismatch")
            for field, trusted in per_fold_hashes.items():
                if metrics_raw.get(field) != trusted:
                    raise ValueError(f"MIL fold {fold} metrics {field} mismatch")
            inventory_folds.append(
                {
                    "held_out_fold": fold,
                    "checkpoint": f"fold-{fold}/model.pt",
                    "checkpoint_sha256": checkpoint_hash,
                    "metrics": f"fold-{fold}/metrics.json",
                    "metrics_sha256": _sha256(artifact.metrics),
                    "scores": f"fold-{fold}/image-scores.parquet",
                    "scores_sha256": _sha256(artifact.image_scores),
                    "validation_image_ids": expected_ids,
                    "nested_feature_manifest": str(
                        nested_features[fold].resolve().relative_to(nested_root)
                    ),
                    "nested_feature_manifest_sha256": _sha256(nested_features[fold]),
                }
            )
        rows.sort(key=lambda row: (int(row["fold"]), str(row["image_id"])))
        image_oof = staging / "image-oof.parquet"
        pq.write_table(pa.Table.from_pylist(rows), image_oof, compression="zstd", version="2.6")
        inventory = staging / "model-inventory.json"
        inventory.write_text(
            json.dumps(
                {"schema_version": 1, "kind": "image_oof", "folds": inventory_folds},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metrics = staging / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "image_oof_sha256": _sha256(image_oof),
                    "image_count": len(rows),
                    "locked_test_rows": 0,
                    "folds": list(range(5)),
                    **trusted_hashes,
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
    return MilOofArtifacts(
        config.output_dir / "image-oof.parquet",
        config.output_dir / "model-inventory.json",
        config.output_dir / "metrics.json",
    )
