"""Leakage-safe fold fine-tuning with synthetic replay and reviewed real crops."""

import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn import functional as functional

from poor_word.domain import Decision
from poor_word.real_data.schema import SplitRole
from poor_word.training.dataset import GlyphDataset
from poor_word.training.train_glyph import GlyphClassifier, TrainConfig


class RealFineTuneConfig(BaseModel):
    """Immutable inputs for one real-data OOF fold."""

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
    synthetic_replay_weight: float = Field(default=1.0, gt=0)
    real_supervision_weight: float = Field(default=2.0, gt=0)
    embedding_dim: int | None = Field(default=None, ge=2)


@dataclass(frozen=True)
class FoldModelArtifacts:
    held_out_fold: int
    checkpoint: Path
    scores: Path
    metrics: Path
    scoring_crop_ids: tuple[str, ...]
    excluded_folds: tuple[int, ...] = ()


@dataclass(frozen=True)
class _RealCrop:
    crop_id: str
    image_id: str
    path: Path
    decision: str
    anomaly_kind: str


@dataclass(frozen=True)
class _ValidatedInputs:
    training: tuple[_RealCrop, ...]
    scoring: tuple[_RealCrop, ...]
    review_excluded_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: dict[str, Any], field: str, manifest: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest} has missing or malformed {field}")
    return value


def _load_unique(
    path: Path,
    key: str,
    manifest: str,
    *,
    columns: list[str],
    allowed_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    filters = [(key, "in", sorted(allowed_ids))] if allowed_ids is not None else None
    rows = cast(
        list[dict[str, Any]],
        pq.read_table(path, columns=columns, filters=filters).to_pylist(),
    )
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = _required_text(row, key, manifest)
        if value in result:
            raise ValueError(f"{manifest} has duplicate {key}: {value}")
        if allowed_ids is not None and value not in allowed_ids:
            raise ValueError(f"{manifest} filter returned disallowed {key}: {value}")
        result[value] = row
    if allowed_ids is not None and set(result) != allowed_ids:
        missing = sorted(allowed_ids - set(result))
        raise ValueError(f"{manifest} filtered read is missing IDs: {missing}")
    return result


def _safe_path(root: Path, relative: str, crop_id: str) -> Path:
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"crop path escapes crop manifest root: {crop_id}") from error
    return resolved


def _validated_inputs(
    config: RealFineTuneConfig,
    held_out_fold: int,
    additionally_excluded_fold: int | None = None,
) -> _ValidatedInputs:
    if held_out_fold < 0:
        raise ValueError("held_out_fold must be a development fold")
    if additionally_excluded_fold is not None and additionally_excluded_fold < 0:
        raise ValueError("additional excluded fold must be a development fold")
    excluded_folds = {held_out_fold}
    if additionally_excluded_fold is not None:
        excluded_folds.add(additionally_excluded_fold)
    valid_roles = {role.value for role in SplitRole}
    real = _load_unique(
        config.real_manifest,
        "image_id",
        "real manifest",
        columns=["image_id", "split_role", "training_eligible"],
    )
    for row in real.values():
        role = row.get("split_role")
        eligible = row.get("training_eligible")
        if not isinstance(role, str) or role not in valid_roles:
            raise ValueError("real manifest has malformed split_role")
        if not isinstance(eligible, bool):
            raise ValueError("real manifest has malformed training_eligible")

    folds = _load_unique(
        config.fold_manifest,
        "image_id",
        "fold manifest",
        columns=["image_id", "fold", "split_role"],
    )
    if set(folds) != set(real):
        raise ValueError("fold manifest image IDs do not exactly match real manifest")
    for image_id, row in folds.items():
        fold = row.get("fold")
        role = row.get("split_role")
        if type(fold) is not int:  # bool is deliberately not accepted as an integer fold.
            raise ValueError(f"fold manifest has malformed fold for image_id={image_id}")
        if not isinstance(role, str) or role not in valid_roles:
            raise ValueError("fold manifest has malformed split_role")
        if role != real[image_id]["split_role"]:
            raise ValueError(f"split_role mismatch for image_id={image_id}")
        if role == SplitRole.LOCKED_TEST.value and fold != -1:
            raise ValueError(f"locked-test image must have fold=-1: {image_id}")
        if role != SplitRole.LOCKED_TEST.value and fold < 0:
            raise ValueError(f"development image has invalid negative fold: {image_id}")

    crop_links = _load_unique(
        config.crop_manifest,
        "crop_id",
        "crop manifest",
        columns=["crop_id", "image_id"],
    )
    for crop_id, row in crop_links.items():
        image_id = _required_text(row, "image_id", "crop manifest")
        if image_id not in real:
            raise ValueError(f"crop image_id is missing from real manifest: {crop_id}")

    gold_links = _load_unique(
        config.gold_manifest,
        "crop_id",
        "gold manifest",
        columns=["crop_id", "image_id"],
    )
    allowed_ids: set[str] = set()
    for crop_id, row in gold_links.items():
        source = crop_links.get(crop_id)
        if source is None:
            raise ValueError(f"gold crop is missing from crop manifest: {crop_id}")
        image_id = _required_text(row, "image_id", "gold manifest")
        if image_id != _required_text(source, "image_id", "crop manifest"):
            raise ValueError(f"gold/crop image_id mismatch for crop_id={crop_id}")
        fold = cast(int, folds[image_id]["fold"])
        role = cast(str, folds[image_id]["split_role"])
        if fold >= 0 and role == SplitRole.DEV.value:
            allowed_ids.add(crop_id)

    if not allowed_ids:
        raise ValueError("no reviewed development crops are available")
    crops = _load_unique(
        config.crop_manifest,
        "crop_id",
        "crop manifest",
        columns=["crop_id", "image_id", "crop_path"],
        allowed_ids=allowed_ids,
    )
    gold_columns = ["crop_id", "image_id", "crop_path", "decision"]
    if "anomaly_kind" in pq.ParquetFile(config.gold_manifest).schema_arrow.names:
        gold_columns.append("anomaly_kind")
    gold = _load_unique(
        config.gold_manifest,
        "crop_id",
        "gold manifest",
        columns=gold_columns,
        allowed_ids=allowed_ids,
    )
    training: list[_RealCrop] = []
    scoring: list[_RealCrop] = []
    review_excluded_count = 0
    crop_root = config.crop_manifest.parent.resolve()
    for crop_id, row in gold.items():
        source = crops.get(crop_id)
        if source is None:
            raise ValueError(f"gold crop is missing from crop manifest: {crop_id}")
        image_id = _required_text(row, "image_id", "gold manifest")
        source_image_id = _required_text(source, "image_id", "crop manifest")
        if image_id != source_image_id:
            raise ValueError(f"gold/crop image_id mismatch for crop_id={crop_id}")
        fold = cast(int, folds[image_id]["fold"])
        role = cast(str, folds[image_id]["split_role"])
        gold_path = _required_text(row, "crop_path", "gold manifest")
        source_path = _required_text(source, "crop_path", "crop manifest")
        if gold_path != source_path:
            raise ValueError(f"gold/crop path mismatch for crop_id={crop_id}")
        decision = _required_text(row, "decision", "gold manifest")
        if decision not in {item.value for item in Decision}:
            raise ValueError(f"gold manifest has invalid decision for crop_id={crop_id}")
        anomaly_raw = row.get("anomaly_kind")
        anomaly_kind = (
            anomaly_raw
            if isinstance(anomaly_raw, str) and anomaly_raw
            else ("NONE" if decision == Decision.PASS.value else "unknown")
        )
        path = _safe_path(crop_root, source_path, crop_id)
        item = _RealCrop(crop_id, image_id, path, decision, anomaly_kind)
        if role == SplitRole.DEV.value and fold == held_out_fold:
            scoring.append(item)
        elif (
            role == SplitRole.DEV.value
            and fold not in excluded_folds
            and real[image_id]["training_eligible"] is True
            and decision in {Decision.PASS.value, Decision.BLOCK.value}
        ):
            training.append(item)
        elif role == SplitRole.DEV.value and decision == Decision.REVIEW.value:
            review_excluded_count += 1

    return _ValidatedInputs(
        training=tuple(sorted(training, key=lambda item: item.crop_id)),
        scoring=tuple(sorted(scoring, key=lambda item: item.crop_id)),
        review_excluded_count=review_excluded_count,
    )


def _load_real_image(item: _RealCrop) -> Tensor:
    if not item.path.is_file():
        raise FileNotFoundError(f"eligible real crop does not exist: {item.path}")
    try:
        with Image.open(item.path) as opened:
            rgb = opened.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
        values = np.asarray(rgb, dtype=np.float32) / 255.0
    except Exception as error:
        raise ValueError(f"could not load real crop {item.crop_id}: {error}") from error
    return torch.from_numpy(np.moveaxis(values, -1, 0).copy())


def _restore_model(
    path: Path, config: RealFineTuneConfig
) -> tuple[GlyphClassifier, dict[str, int], int]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("adapted checkpoint must be a mapping")
    checkpoint = cast(dict[str, object], raw)
    expected_real_hash = _sha256(config.real_manifest)
    expected_crop_hash = _sha256(config.crop_manifest)
    real_hash = checkpoint.get("real_manifest_sha256")
    crop_hash = checkpoint.get("crop_manifest_sha256")
    if not isinstance(real_hash, str):
        raise ValueError("adapted checkpoint is missing real_manifest_sha256")
    if real_hash != expected_real_hash:
        raise ValueError("adapted checkpoint real manifest hash mismatch")
    if not isinstance(crop_hash, str):
        raise ValueError("adapted checkpoint is missing crop_manifest_sha256")
    if crop_hash != expected_crop_hash:
        raise ValueError("adapted checkpoint crop manifest hash mismatch")
    if checkpoint.get("calibrated_for_block_decisions") is not False:
        raise ValueError("adapted checkpoint calibrated flag must be exactly false")
    model_config = checkpoint.get("config")
    catalog = checkpoint.get("char_to_id")
    state = checkpoint.get("model_state")
    if (
        not isinstance(model_config, dict)
        or not isinstance(catalog, dict)
        or not isinstance(state, dict)
    ):
        raise ValueError("adapted checkpoint is missing config, character catalog, or model_state")
    embedding_dim = model_config.get("embedding_dim")
    if not isinstance(embedding_dim, int) or embedding_dim < 2:
        raise ValueError("adapted checkpoint has invalid embedding_dim")
    if config.embedding_dim is not None and config.embedding_dim != embedding_dim:
        raise ValueError("configured embedding_dim does not match adapted checkpoint")
    if not catalog or not all(
        isinstance(character, str) and character and type(index) is int
        for character, index in catalog.items()
    ):
        raise ValueError("adapted checkpoint has invalid character catalog")
    if sorted(catalog.values()) != list(range(len(catalog))):
        raise ValueError("adapted checkpoint catalog indices must be unique and contiguous")
    typed_catalog = cast(dict[str, int], catalog)
    model = GlyphClassifier(
        len(typed_catalog),
        TrainConfig(
            manifest=config.synthetic_manifest,
            output_dir=config.output_dir,
            embedding_dim=embedding_dim,
            device=config.device,
        ),
    )
    model.load_state_dict(cast(dict[str, Tensor], state), strict=True)
    return model, typed_catalog, embedding_dim


def _real_batch(items: list[_RealCrop], device: torch.device) -> tuple[Tensor, Tensor]:
    images = torch.stack([_load_real_image(item) for item in items]).to(device)
    labels = torch.tensor(
        [item.decision == Decision.BLOCK.value for item in items],
        dtype=torch.float32,
        device=device,
    )
    return images, labels


def _parquet_bytes(rows: list[dict[str, object]], schema: pa.Schema) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        sink,
        compression="zstd",
        version="2.6",
    )
    return cast(bytes, sink.getvalue().to_pybytes())


def finetune_real_fold(
    config: RealFineTuneConfig,
    held_out_fold: int,
    additionally_excluded_fold: int | None = None,
) -> FoldModelArtifacts:
    """Fine-tune one fold and score only its held-out reviewed development crops."""
    if config.output_dir.exists():
        raise ValueError(f"fold output directory already exists: {config.output_dir}")
    random.seed(config.seed + held_out_fold)
    np.random.seed(config.seed + held_out_fold)
    torch.manual_seed(config.seed + held_out_fold)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {config.device}")
    device = torch.device(config.device)
    inputs = _validated_inputs(config, held_out_fold, additionally_excluded_fold)
    excluded_folds = tuple(
        sorted(
            {
                held_out_fold,
                *(
                    []
                    if additionally_excluded_fold is None
                    else [additionally_excluded_fold]
                ),
            }
        )
    )
    synthetic = GlyphDataset(config.synthetic_manifest)
    model, catalog, embedding_dim = _restore_model(config.adapted_checkpoint, config)
    if set(synthetic.char_to_id) - set(catalog):
        raise ValueError("synthetic manifest contains characters absent from adapted checkpoint")
    model = model.to(device)
    risk_head = nn.Linear(embedding_dim, 1).to(device)
    cpu_smoke = device.type == "cpu" and config.max_steps is not None and config.max_steps <= 2
    if cpu_smoke:
        for parameter in model.encoder.features.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [
            *[parameter for parameter in model.parameters() if parameter.requires_grad],
            *risk_head.parameters(),
        ],
        lr=config.learning_rate,
    )
    generator = torch.Generator().manual_seed(config.seed + held_out_fold)
    history: list[dict[str, float]] = []
    step = 0
    real_indices = list(range(len(inputs.training)))
    synthetic_indices = list(range(len(synthetic)))
    model.train()
    risk_head.train()
    for _epoch in range(config.epochs):
        if real_indices:
            real_order = torch.randperm(len(real_indices), generator=generator).tolist()
        else:
            real_order = []
        synthetic_order = torch.randperm(len(synthetic_indices), generator=generator).tolist()
        largest_pool = max(len(real_order), len(synthetic_order))
        batches = max(1, (largest_pool + config.batch_size - 1) // config.batch_size)
        for batch_index in range(batches):
            real_count = min(len(real_order), max(1, config.batch_size // 2)) if real_order else 0
            synthetic_count = config.batch_size - real_count
            if synthetic_count < 1:
                synthetic_count = 1
                real_count = config.batch_size - 1
            real_selected = (
                [
                    inputs.training[
                        real_order[(batch_index * real_count + offset) % len(real_order)]
                    ]
                    for offset in range(real_count)
                ]
                if real_order
                else []
            )
            synthetic_selected = [
                synthetic[
                    synthetic_order[(batch_index * synthetic_count + offset) % len(synthetic_order)]
                ]
                for offset in range(synthetic_count)
            ]

            synthetic_views = torch.stack([item.views for item in synthetic_selected]).to(device)
            synthetic_embeddings, synthetic_logits = model(synthetic_views)
            synthetic_binary = torch.tensor(
                [item.decision == Decision.BLOCK.value for item in synthetic_selected],
                dtype=torch.float32,
                device=device,
            )
            synthetic_risk = risk_head(synthetic_embeddings).squeeze(1)
            synthetic_loss = functional.binary_cross_entropy_with_logits(
                synthetic_risk, synthetic_binary
            )
            legal_positions = [
                index
                for index, item in enumerate(synthetic_selected)
                if item.decision == Decision.PASS.value
            ]
            if legal_positions:
                expected = torch.tensor(
                    [catalog[synthetic_selected[index].base_char] for index in legal_positions],
                    dtype=torch.long,
                    device=device,
                )
                classification = functional.cross_entropy(
                    synthetic_logits[legal_positions], expected
                )
            else:
                classification = synthetic_logits.sum() * 0.0
            if real_selected:
                real_views, real_labels = _real_batch(real_selected, device)
                real_embeddings, _ = model(real_views)
                real_loss = functional.binary_cross_entropy_with_logits(
                    risk_head(real_embeddings).squeeze(1), real_labels
                )
            else:
                real_loss = synthetic_logits.sum() * 0.0
            loss = (
                config.synthetic_replay_weight * (synthetic_loss + 0.25 * classification)
                + config.real_supervision_weight * real_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            history.append(
                {
                    "step": float(step),
                    "synthetic_binary": float(synthetic_loss.detach()),
                    "synthetic_classification": float(classification.detach()),
                    "real_binary": float(real_loss.detach()),
                    "total": float(loss.detach()),
                }
            )
            step += 1
            if config.max_steps is not None and step >= config.max_steps:
                break
        if config.max_steps is not None and step >= config.max_steps:
            break
    if not history:
        raise ValueError("fine-tuning produced no optimization steps")

    model.eval()
    risk_head.eval()
    raw_scores: list[tuple[_RealCrop, float]] = []
    with torch.inference_mode():
        for item in inputs.scoring:
            image = _load_real_image(item).unsqueeze(0).to(device)
            embedding, _ = model(image)
            risk = float(torch.sigmoid(risk_head(embedding).squeeze()).cpu())
            raw_scores.append((item, risk))

    hashes = {
        "parent_checkpoint_sha256": _sha256(config.adapted_checkpoint),
        "real_manifest_sha256": _sha256(config.real_manifest),
        "crop_manifest_sha256": _sha256(config.crop_manifest),
        "gold_manifest_sha256": _sha256(config.gold_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
        "synthetic_manifest_sha256": _sha256(config.synthetic_manifest),
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
                "model_state": model.state_dict(),
                "risk_head_state": risk_head.state_dict(),
                "char_to_id": catalog,
                "config": {"embedding_dim": embedding_dim},
                "held_out_fold": held_out_fold,
                "excluded_folds": list(excluded_folds),
                "real_training_crop_ids": [item.crop_id for item in inputs.training],
                "scoring_crop_ids": [item.crop_id for item in inputs.scoring],
                **hashes,
            },
            checkpoint,
        )
        checkpoint_hash = _sha256(checkpoint)
        score_rows = [
            {
                "crop_id": item.crop_id,
                "image_id": item.image_id,
                "crop_path": str(item.path.relative_to(config.crop_manifest.parent.resolve())),
                "fold": held_out_fold,
                "excluded_folds": list(excluded_folds),
                "decision": item.decision,
                "anomaly_kind": item.anomaly_kind,
                "risk_score": risk,
                "model_id": f"real-fold-{held_out_fold}",
                "checkpoint_sha256": checkpoint_hash,
                "parent_checkpoint_sha256": hashes["parent_checkpoint_sha256"],
                "real_manifest_sha256": hashes["real_manifest_sha256"],
                "crop_manifest_sha256": hashes["crop_manifest_sha256"],
                "fold_manifest_sha256": hashes["fold_manifest_sha256"],
                "gold_manifest_sha256": hashes["gold_manifest_sha256"],
                "synthetic_manifest_sha256": hashes["synthetic_manifest_sha256"],
            }
            for item, risk in raw_scores
        ]
        score_schema = pa.schema(
            [
                ("crop_id", pa.string()),
                ("image_id", pa.string()),
                ("crop_path", pa.string()),
                ("fold", pa.int64()),
                ("excluded_folds", pa.list_(pa.int64())),
                ("decision", pa.string()),
                ("anomaly_kind", pa.string()),
                ("risk_score", pa.float64()),
                ("model_id", pa.string()),
                ("checkpoint_sha256", pa.string()),
                ("parent_checkpoint_sha256", pa.string()),
                ("real_manifest_sha256", pa.string()),
                ("crop_manifest_sha256", pa.string()),
                ("fold_manifest_sha256", pa.string()),
                ("gold_manifest_sha256", pa.string()),
                ("synthetic_manifest_sha256", pa.string()),
            ]
        )
        scores = staging / "scores.parquet"
        scores.write_bytes(_parquet_bytes(score_rows, score_schema))
        metrics = staging / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "held_out_fold": held_out_fold,
                    "excluded_folds": list(excluded_folds),
                    "steps": step,
                    "loss_history": history,
                    "real_supervised_count": len(inputs.training),
                    "real_training_review_excluded_count": inputs.review_excluded_count,
                    "synthetic_replay_count": len(synthetic),
                    "scoring_count": len(inputs.scoring),
                    "checkpoint_sha256": checkpoint_hash,
                    "scores_sha256": _sha256(scores),
                    "scoring_crop_ids": [item.crop_id for item in inputs.scoring],
                    "cpu_smoke_backbone_frozen": cpu_smoke,
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
    return FoldModelArtifacts(
        held_out_fold=held_out_fold,
        checkpoint=config.output_dir / "model.pt",
        scores=config.output_dir / "scores.parquet",
        metrics=config.output_dir / "metrics.json",
        scoring_crop_ids=tuple(item.crop_id for item in inputs.scoring),
        excluded_folds=excluded_folds,
    )
