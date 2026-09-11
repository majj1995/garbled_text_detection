import hashlib
import json
import random
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.nn import functional as functional

from poor_word.models.glyph_encoder import GlyphEncoder
from poor_word.models.prototypes import PrototypeBank
from poor_word.training.dataset import GlyphDataset, GlyphItem
from poor_word.training.sampling import build_sampling_run, sampling_counts


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: Path
    output_dir: Path
    epochs: int = Field(default=20, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=256, ge=1)
    seed: int = Field(default=20260804, ge=0)
    pretrained: bool = False
    device: str = "cuda"
    learning_rate: float = Field(default=3e-4, gt=0)
    embedding_dim: int = Field(default=256, ge=2)
    sampler: Literal["random", "paired"] = "random"
    allow_experimental: bool = False
    log_every: int = Field(default=25, ge=1)


@dataclass(frozen=True)
class TrainArtifacts:
    checkpoint: Path
    metrics: Path
    prototype_bank: Path


class GlyphClassifier(nn.Module):
    def __init__(self, character_count: int, config: TrainConfig) -> None:
        super().__init__()
        self.encoder = GlyphEncoder(
            embedding_dim=config.embedding_dim,
            pretrained=config.pretrained,
        )
        self.classifier = nn.Linear(config.embedding_dim, character_count)

    def forward(self, views: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.encoder(views)
        return embeddings, self.classifier(embeddings)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _supervised_contrastive(embeddings: Tensor, labels: Tensor) -> Tensor:
    if len(embeddings) < 2:
        return embeddings.sum() * 0.0
    similarities = embeddings @ embeddings.T / 0.1
    eye = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    if not bool(positives.any()):
        return embeddings.sum() * 0.0
    similarities = similarities.masked_fill(eye, float("-inf"))
    log_prob = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positive_count)
    return per_anchor[positives.any(dim=1)].mean()


def _batch(items: list[GlyphItem], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    views = torch.stack([item.views for item in items]).to(device)
    labels = torch.tensor([item.label_id for item in items], dtype=torch.long, device=device)
    legal = torch.tensor(
        [item.decision == "PASS" for item in items], dtype=torch.bool, device=device
    )
    return views, labels, legal


def _embed_dataset(
    model: GlyphClassifier,
    dataset: GlyphDataset,
    device: torch.device,
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    results: list[NDArray[np.float32]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            items = [dataset[index] for index in range(start, stop)]
            views = torch.stack([item.views for item in items]).to(device)
            embeddings, _ = model(views)
            results.append(embeddings.cpu().numpy().astype(np.float32))
            if progress is not None:
                progress(stop, len(dataset))
    return np.concatenate(results, axis=0).astype(np.float32, copy=False)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    part = path.with_name(f"{path.name}.part")
    try:
        part.write_text(
            f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _nearest_accuracy(
    bank: PrototypeBank,
    embeddings: NDArray[np.float32],
    dataset: GlyphDataset,
    indices: list[int],
) -> float | None:
    pass_indices = [index for index in indices if dataset.rows[index]["decision"] == "PASS"]
    if not pass_indices:
        return None
    scores = bank.score(embeddings[pass_indices].astype(np.float32))
    expected = [str(dataset.rows[index]["base_char"]) for index in pass_indices]
    return float(
        np.mean(
            [
                actual == target
                for actual, target in zip(scores.nearest_chars, expected, strict=True)
            ]
        )
    )


def _ood_aucpr(
    bank: PrototypeBank,
    embeddings: NDArray[np.float32],
    dataset: GlyphDataset,
    indices: list[int],
) -> float | None:
    if not indices:
        return None
    labels = np.asarray(
        [dataset.rows[index]["decision"] == "BLOCK" for index in indices], dtype=np.int64
    )
    if len(np.unique(labels)) != 2:
        return None
    scores = bank.score(embeddings[indices].astype(np.float32))
    return float(average_precision_score(labels, scores.nearest_distance))


def _dataset_provenance(dataset: GlyphDataset) -> dict[str, object]:
    source_ids = sorted(
        {str(source_id) for row in dataset.rows for source_id in row.get("source_asset_ids", [])}
    )
    provenance: dict[str, object] = {"source_asset_ids": source_ids}
    if dataset.schema_version is not None:
        provenance["dataset_schema_version"] = dataset.schema_version
    if dataset.dataset_id is not None:
        provenance["dataset_id"] = dataset.dataset_id
    label_provenance = sorted(
        {str(row["label_provenance"]) for row in dataset.rows if row.get("label_provenance")}
    )
    if label_provenance:
        provenance["label_provenance"] = label_provenance
    if dataset.run_metadata.get("source_provenance") is not None:
        provenance["source_provenance"] = dataset.run_metadata["source_provenance"]
    return provenance


def train_glyph(
    config: TrainConfig, *, progress: Callable[[str], None] | None = None
) -> TrainArtifacts:
    started = time.perf_counter()
    output_artifacts = (
        "encoder.pt",
        "metrics.json",
        "prototypes.npz",
        "prototypes.json",
        "training_membership.json",
        "progress.jsonl",
    )
    if any((config.output_dir / name).exists() for name in output_artifacts):
        raise FileExistsError(
            "training requires a fresh output directory; existing artifacts remain"
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {config.device}")
    device = torch.device(config.device)
    dataset = GlyphDataset(config.manifest)
    train_indices, validation_indices = dataset.split_indices()
    if dataset.schema_version is not None:
        if not config.allow_experimental:
            raise ValueError("V2 training requires allow_experimental=True")
        if config.sampler != "paired":
            raise ValueError("V2 training requires sampler='paired'")
        if dataset.split_role != "train":
            raise ValueError("V2 training requires split_role='train'")
    elif not train_indices:
        train_indices = tuple(range(len(dataset)))
    if not train_indices:
        raise ValueError("training split contains no samples")
    training_decisions = {str(dataset.rows[index]["decision"]) for index in train_indices}
    if "PASS" not in training_decisions:
        raise ValueError("training membership must contain PASS samples")
    if config.sampler == "paired" and "BLOCK" not in training_decisions:
        raise ValueError("training membership must contain BLOCK samples")

    sampling_run = build_sampling_run(
        dataset.rows,
        train_indices,
        sampler=config.sampler,
        batch_size=config.batch_size,
        seed=config.seed,
        epochs=config.epochs,
        max_steps=config.max_steps,
    )
    sample_stats = sampling_counts(dataset.rows, sampling_run.batches)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = config.output_dir / "progress.jsonl"

    def emit(message: str, **event: Any) -> None:
        if progress is not None:
            progress(message)
        payload = {"message": message, "elapsed_seconds": time.perf_counter() - started, **event}
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n")

    emit(
        f"startup device={device} sampler={config.sampler} train_samples={len(train_indices)}",
        phase="startup",
        sampler=config.sampler,
        train_samples=len(train_indices),
    )

    membership_sha: str | None = None
    if dataset.schema_version is not None:
        membership_path = config.output_dir / "training_membership.json"
        _atomic_json(
            membership_path,
            {
                "schema_version": "glyph-training-membership-v2",
                "dataset_id": dataset.dataset_id,
                "manifest_sha256": _sha256(config.manifest),
                "sample_ids": [str(dataset.rows[index]["sample_id"]) for index in train_indices],
                "source_group_ids": [
                    str(dataset.rows[index]["source_group_id"]) for index in train_indices
                ],
                "pixel_sha256": [
                    str(dataset.rows[index]["pixel_sha256"]) for index in train_indices
                ],
            },
        )
        membership_sha = _sha256(membership_path)

    model = GlyphClassifier(len(dataset.char_to_id), config).to(device)
    cpu_smoke = device.type == "cpu" and config.max_steps is not None and config.max_steps <= 2
    if cpu_smoke:
        for parameter in model.encoder.features.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )

    history: list[dict[str, float]] = []
    step = 0
    last_log = started
    model.train()
    for epoch in sampling_run.epochs:
        for selected in epoch.batches:
            items = [dataset[index] for index in selected]
            views, labels, legal = _batch(items, device)
            embeddings, logits = model(views)
            zero = logits.sum() * 0.0
            classification = (
                functional.cross_entropy(logits[legal], labels[legal]) if legal.any() else zero
            )
            contrastive = (
                _supervised_contrastive(embeddings[legal], labels[legal]) if legal.any() else zero
            )
            energy = -torch.logsumexp(logits, dim=1)
            if (~legal).any():
                legal_reference = (
                    energy[legal].mean().detach() if legal.any() else energy.new_tensor(-2.0)
                )
                energy_margin = functional.relu(1.0 - (energy[~legal] - legal_reference)).mean()
            else:
                energy_margin = zero
            loss = classification + 0.5 * contrastive + 0.5 * energy_margin
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            entry = {
                "step": float(step),
                "classification": float(classification.detach()),
                "contrastive": float(contrastive.detach()),
                "energy_margin": float(energy_margin.detach()),
                "total": float(loss.detach()),
            }
            history.append(entry)
            step += 1
            now = time.perf_counter()
            if step == 1 or step % config.log_every == 0 or now - last_log >= 30.0:
                emit(
                    f"epoch={epoch.epoch} step={step} loss={entry['total']:.6f} "
                    f"elapsed={now - started:.1f}s",
                    phase="training",
                    epoch=epoch.epoch,
                    **entry,
                )
                last_log = now

    checkpoint = config.output_dir / "encoder.pt"
    checkpoint_part = checkpoint.with_name(f"{checkpoint.name}.part")
    checkpoint_payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "char_to_id": dataset.char_to_id,
        "config": config.model_dump(mode="json"),
        **_dataset_provenance(dataset),
    }
    if dataset.experimental_only:
        checkpoint_payload.update(experimental_only=True, production_allowed=False)
    if membership_sha is not None:
        checkpoint_payload["training_membership_sha256"] = membership_sha
    torch.save(checkpoint_payload, checkpoint_part)
    checkpoint_part.replace(checkpoint)

    emit("embedding phase started", phase="embedding_start")
    embeddings = _embed_dataset(
        model,
        dataset,
        device,
        min(config.batch_size, 64),
        progress=lambda completed, total: emit(
            f"embedding samples={completed}/{total}",
            phase="embedding_progress",
            completed=completed,
            total=total,
        ),
    )
    emit(f"embedding phase complete samples={len(dataset)}", phase="embedding_complete")
    pass_indices = [index for index in train_indices if dataset.rows[index]["decision"] == "PASS"]
    emit(f"prototype fit started samples={len(pass_indices)}", phase="prototype_fit_start")
    bank = PrototypeBank(random_state=config.seed)
    bank.fit(
        embeddings[pass_indices].astype(np.float32),
        [str(dataset.rows[index]["base_char"]) for index in pass_indices],
    )
    emit("prototype fit complete", phase="prototype_fit_complete")
    prototype_path = config.output_dir / "prototypes.npz"
    catalog_hash = hashlib.sha256("".join(sorted(dataset.char_to_id)).encode("utf-8")).hexdigest()
    prototype_metadata = {
        "catalog_sha256": catalog_hash,
        "encoder_checkpoint_sha256": _sha256(checkpoint),
        "source_manifest_sha256": _sha256(config.manifest),
        "creation_command": "poor-word train glyph",
    }
    if membership_sha is not None:
        prototype_metadata["training_membership_sha256"] = membership_sha
    if dataset.schema_version is not None:
        prototype_metadata["dataset_schema_version"] = dataset.schema_version
    if dataset.dataset_id is not None:
        prototype_metadata["dataset_id"] = dataset.dataset_id
    if dataset.experimental_only:
        prototype_metadata["experimental_only"] = "true"
        prototype_metadata["production_allowed"] = "false"
    bank.save(prototype_path, metadata=prototype_metadata)
    if dataset.experimental_only:
        prototype_metadata["prototype_bank_sha256"] = _sha256(prototype_path)
        _atomic_json(prototype_path.with_suffix(".json"), prototype_metadata)

    train_list = list(train_indices)
    validation_list = list(validation_indices)
    emit("metric scoring started", phase="metric_scoring_start")
    metrics_payload: dict[str, object] = {
        "seed": config.seed,
        "device": str(device),
        "git_commit": _git_commit(),
        "uv_lock_sha256": _sha256(Path("uv.lock")),
        "manifest_sha256": _sha256(config.manifest),
        "loss_history": history,
        "train_nearest_prototype_accuracy": _nearest_accuracy(
            bank, embeddings, dataset, train_list
        ),
        "validation_nearest_prototype_accuracy": _nearest_accuracy(
            bank, embeddings, dataset, validation_list
        ),
        "train_synthetic_ood_aucpr": _ood_aucpr(bank, embeddings, dataset, train_list),
        "validation_synthetic_ood_aucpr": _ood_aucpr(bank, embeddings, dataset, validation_list),
        "elapsed_seconds": time.perf_counter() - started,
        "steps": step,
        "cpu_smoke_backbone_frozen": cpu_smoke,
        "sampler": config.sampler,
        **_dataset_provenance(dataset),
        **sample_stats,
    }
    if dataset.experimental_only:
        metrics_payload.update(experimental_only=True, production_allowed=False)
    if membership_sha is not None:
        metrics_payload["training_membership_sha256"] = membership_sha
    metrics_path = config.output_dir / "metrics.json"
    _atomic_json(metrics_path, metrics_payload)
    emit("metric scoring complete", phase="metric_scoring_complete")
    emit(
        "training complete",
        phase="complete",
        steps=step,
        train_nearest_prototype_accuracy=metrics_payload["train_nearest_prototype_accuracy"],
        validation_nearest_prototype_accuracy=metrics_payload[
            "validation_nearest_prototype_accuracy"
        ],
        train_synthetic_ood_aucpr=metrics_payload["train_synthetic_ood_aucpr"],
        validation_synthetic_ood_aucpr=metrics_payload["validation_synthetic_ood_aucpr"],
        **sample_stats,
    )
    return TrainArtifacts(
        checkpoint=checkpoint,
        metrics=metrics_path,
        prototype_bank=prototype_path,
    )
