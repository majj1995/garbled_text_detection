import hashlib
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

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
    model: GlyphClassifier, dataset: GlyphDataset, device: torch.device, batch_size: int
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
    return np.concatenate(results, axis=0).astype(np.float32, copy=False)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
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


def train_glyph(config: TrainConfig) -> TrainArtifacts:
    started = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {config.device}")
    device = torch.device(config.device)
    dataset = GlyphDataset(config.manifest)
    train_indices, validation_indices = dataset.split_indices()
    if not train_indices:
        train_indices = tuple(range(len(dataset)))

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
    generator = torch.Generator().manual_seed(config.seed)
    model.train()
    for _epoch in range(config.epochs):
        order = torch.randperm(len(train_indices), generator=generator).tolist()
        for start in range(0, len(order), config.batch_size):
            batch_order = order[start : start + config.batch_size]
            selected = [train_indices[order_index] for order_index in batch_order]
            items = [dataset[index] for index in selected]
            views, labels, legal = _batch(items, device)
            embeddings, logits = model(views)
            zero = logits.sum() * 0.0
            classification = (
                functional.cross_entropy(logits[legal], labels[legal])
                if legal.any()
                else zero
            )
            contrastive = (
                _supervised_contrastive(embeddings[legal], labels[legal]) if legal.any() else zero
            )
            energy = -torch.logsumexp(logits, dim=1)
            if (~legal).any():
                legal_reference = (
                    energy[legal].mean().detach()
                    if legal.any()
                    else energy.new_tensor(-2.0)
                )
                energy_margin = functional.relu(1.0 - (energy[~legal] - legal_reference)).mean()
            else:
                energy_margin = zero
            loss = classification + 0.5 * contrastive + 0.5 * energy_margin
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "step": float(step),
                    "classification": float(classification.detach()),
                    "contrastive": float(contrastive.detach()),
                    "energy_margin": float(energy_margin.detach()),
                    "total": float(loss.detach()),
                }
            )
            step += 1
            if config.max_steps is not None and step >= config.max_steps:
                break
        if config.max_steps is not None and step >= config.max_steps:
            break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "encoder.pt"
    checkpoint_part = checkpoint.with_name(f"{checkpoint.name}.part")
    torch.save(
        {
            "model_state": model.state_dict(),
            "char_to_id": dataset.char_to_id,
            "config": config.model_dump(mode="json"),
        },
        checkpoint_part,
    )
    checkpoint_part.replace(checkpoint)

    embeddings = _embed_dataset(model, dataset, device, min(config.batch_size, 64))
    pass_indices = [index for index, row in enumerate(dataset.rows) if row["decision"] == "PASS"]
    if not pass_indices:
        raise ValueError("training manifest must contain PASS samples")
    bank = PrototypeBank(random_state=config.seed)
    bank.fit(
        embeddings[pass_indices].astype(np.float32),
        [str(dataset.rows[index]["base_char"]) for index in pass_indices],
    )
    prototype_path = config.output_dir / "prototypes.npz"
    catalog_hash = hashlib.sha256(
        "".join(sorted(dataset.char_to_id)).encode("utf-8")
    ).hexdigest()
    bank.save(
        prototype_path,
        metadata={
            "catalog_sha256": catalog_hash,
            "encoder_checkpoint_sha256": _sha256(checkpoint),
            "source_manifest_sha256": _sha256(config.manifest),
            "creation_command": "poor-word train glyph",
        },
    )

    validation_pass = [
        index
        for index in validation_indices
        if dataset.rows[index]["decision"] == "PASS"
    ]
    validation_accuracy: float | None = None
    if validation_pass:
        scores = bank.score(embeddings[validation_pass].astype(np.float32))
        expected = [str(dataset.rows[index]["base_char"]) for index in validation_pass]
        validation_accuracy = float(
            np.mean(
                [
                    actual == target
                    for actual, target in zip(scores.nearest_chars, expected, strict=True)
                ]
            )
        )
    all_scores = bank.score(embeddings.astype(np.float32))
    ood_labels = np.asarray(
        [row["decision"] == "BLOCK" for row in dataset.rows], dtype=np.int64
    )
    ood_aucpr = (
        float(average_precision_score(ood_labels, all_scores.nearest_distance))
        if len(np.unique(ood_labels)) == 2
        else None
    )
    lock_path = Path("uv.lock")
    metrics_path = config.output_dir / "metrics.json"
    _atomic_json(
        metrics_path,
        {
            "seed": config.seed,
            "device": str(device),
            "git_commit": _git_commit(),
            "uv_lock_sha256": _sha256(lock_path),
            "manifest_sha256": _sha256(config.manifest),
            "loss_history": history,
            "validation_nearest_prototype_accuracy": validation_accuracy,
            "synthetic_ood_aucpr": ood_aucpr,
            "elapsed_seconds": time.perf_counter() - started,
            "steps": step,
            "cpu_smoke_backbone_frozen": cpu_smoke,
        },
    )
    return TrainArtifacts(
        checkpoint=checkpoint,
        metrics=metrics_path,
        prototype_bank=prototype_path,
    )
