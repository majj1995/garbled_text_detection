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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
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
    for path in (config.real_manifest, config.fold_manifest, config.feature_manifest):
        if not path.is_file():
            raise ValueError(f"required manifest does not exist: {path}")
    real_rows = cast(list[dict[str, Any]], pq.read_table(config.real_manifest).to_pylist())
    fold_rows = cast(list[dict[str, Any]], pq.read_table(config.fold_manifest).to_pylist())
    feature_rows = cast(list[dict[str, Any]], pq.read_table(config.feature_manifest).to_pylist())

    real: dict[str, dict[str, Any]] = {}
    allowed_roles = {role.value for role in SplitRole}
    allowed_labels = {label.value for label in ImageLabel}
    for row in real_rows:
        image_id = _required_text(row, "image_id", "real manifest")
        if image_id in real:
            raise ValueError(f"real manifest has duplicate image_id: {image_id}")
        label = row.get("image_label")
        role = row.get("split_role")
        eligible = row.get("training_eligible")
        if not isinstance(label, str) or label not in allowed_labels:
            raise ValueError("real manifest has malformed image_label")
        if not isinstance(role, str) or role not in allowed_roles:
            raise ValueError("real manifest has malformed split_role")
        if not isinstance(eligible, bool):
            raise ValueError("real manifest has malformed training_eligible")
        real[image_id] = row

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
        if component is not None:
            if not isinstance(component, str) or not component:
                raise ValueError("fold manifest has malformed component_id")
            previous = component_folds.setdefault(component, fold)
            if previous != fold:
                raise ValueError("component_id crosses folds")
        folds[image_id] = row
    if set(folds) != set(real):
        raise ValueError("fold manifest image IDs do not match real manifest")

    fold_hash = _sha256(config.fold_manifest)
    features: dict[str, list[_Feature]] = {image_id: [] for image_id in real}
    crop_ids: set[str] = set()
    feature_models: set[tuple[str, str, int]] = set()
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
        risk = row.get("risk_score", row.get("evidence"))
        if not isinstance(risk, (int, float)) or isinstance(risk, bool) or not math.isfinite(risk):
            raise ValueError("feature manifest requires finite risk_score")
        model_id = _required_text(row, "model_id", "feature manifest")
        checkpoint_hash = _required_text(row, "checkpoint_sha256", "feature manifest")
        if len(checkpoint_hash) != 64 or any(
            char not in "0123456789abcdef" for char in checkpoint_hash
        ):
            raise ValueError("feature manifest has malformed checkpoint_sha256")
        claimed_fold_hash = _required_text(row, "fold_manifest_sha256", "feature manifest")
        if claimed_fold_hash != fold_hash:
            raise ValueError("feature fold manifest hash mismatch")
        feature_models.add((model_id, checkpoint_hash, feature_fold))
        features[image_id].append(_Feature(crop_id, float(risk)))

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
    model_provenance = [
        {
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_hash,
            "scoring_fold": scoring_fold,
        }
        for model_id, checkpoint_hash, scoring_fold in sorted(feature_models)
    ]
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
) -> tuple[float, list[dict[str, object]]]:
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
    for index, bag in enumerate(bags):
        if not crop_ids[index]:
            continue
        best = int(torch.argmax(attention[index]).item())
        candidates.append(
            {
                "image_id": bag.image_id,
                "crop_id": crop_ids[index][best],
                "held_out_fold": None,
                "attention": float(attention[index, best].cpu()),
                "image_risk_score": float(torch.sigmoid(logits[index]).cpu()),
                "label_source": "mil_attention_candidate",
                "is_gold": False,
            }
        )
    return float(loss.total.cpu()), candidates


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
        validation_loss, _ = _evaluate(
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
    _, candidates = _evaluate(model, validation, device, pos_weight, config.normal_instance_weight)
    for row in candidates:
        row["held_out_fold"] = config.held_out_fold

    hashes = {
        "real_manifest_sha256": _sha256(config.real_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
        "feature_manifest_sha256": _sha256(config.feature_manifest),
    }
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
        candidate_path = staging / "attention-candidates.parquet"
        candidate_path.write_bytes(_parquet_bytes(candidates))
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
    )
