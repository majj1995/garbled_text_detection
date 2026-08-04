"""Self-supervised, photometric-only adaptation on eligible real character crops."""

import hashlib
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn import functional as functional

from poor_word.training.train_glyph import GlyphClassifier, TrainConfig


class AdaptConfig(BaseModel):
    """Inputs and limits for unlabeled real-crop encoder adaptation."""

    model_config = ConfigDict(frozen=True)

    crop_manifest: Path
    real_manifest: Path
    prior_checkpoint: Path
    output_dir: Path
    epochs: int = Field(default=20, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=64, ge=1)
    seed: int = Field(default=20260804, ge=0)
    device: str = "cuda"
    learning_rate: float = Field(default=3e-5, gt=0)
    diagnostic_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    min_embedding_variance: float = Field(default=1e-8, gt=0.0)


@dataclass(frozen=True)
class AdaptArtifacts:
    checkpoint: Path
    metrics: Path


@dataclass(frozen=True)
class _Crop:
    crop_id: str
    path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


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


def _load_eligible_crops(crop_manifest: Path, real_manifest: Path) -> list[_Crop]:
    source_rows = cast(list[dict[str, Any]], pq.read_table(real_manifest).to_pylist())
    sources: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("real manifest has missing image_id")
        if image_id in sources:
            raise ValueError(f"real manifest has duplicate image_id: {image_id}")
        if "training_eligible" not in row or "split_role" not in row:
            raise ValueError("real manifest row is missing training eligibility or split role")
        sources[image_id] = row

    root = crop_manifest.parent.resolve()
    crops: list[_Crop] = []
    seen_crop_ids: set[str] = set()
    for row in cast(list[dict[str, Any]], pq.read_table(crop_manifest).to_pylist()):
        image_id = row.get("image_id")
        crop_id = row.get("crop_id")
        crop_path = row.get("crop_path")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("crop manifest has missing image_id")
        if image_id not in sources:
            raise ValueError(f"crop image_id is missing from real manifest: {image_id}")
        if (
            not isinstance(crop_id, str)
            or not crop_id
            or not isinstance(crop_path, str)
            or not crop_path
        ):
            raise ValueError("crop manifest row is missing crop_id or crop_path")
        if crop_id in seen_crop_ids:
            raise ValueError(f"crop manifest has duplicate crop_id: {crop_id}")
        seen_crop_ids.add(crop_id)
        source = sources[image_id]
        if not bool(source["training_eligible"]) or str(source["split_role"]) == "LOCKED_TEST":
            continue
        resolved = (root / crop_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"crop path escapes crop manifest root: {crop_id}") from error
        crops.append(_Crop(crop_id=crop_id, path=resolved))
    if len(crops) < 2:
        raise ValueError("at least two eligible real crops are required for adaptation")
    return sorted(crops, key=lambda crop: crop.crop_id)


def _diagnostic_split(
    crops: list[_Crop], seed: int, fraction: float
) -> tuple[list[_Crop], list[_Crop]]:
    ranked = sorted(
        crops,
        key=lambda crop: hashlib.sha256(f"{seed}\0{crop.crop_id}".encode()).digest(),
    )
    diagnostic_count = min(len(crops) - 1, max(1, round(len(crops) * fraction)))
    diagnostic_ids = {crop.crop_id for crop in ranked[:diagnostic_count]}
    diagnostic = [crop for crop in crops if crop.crop_id in diagnostic_ids]
    train = [crop for crop in crops if crop.crop_id not in diagnostic_ids]
    return train, diagnostic


def _load_crop(crop: _Crop) -> Tensor:
    if not crop.path.is_file():
        raise FileNotFoundError(f"eligible crop does not exist: {crop.path}")
    try:
        with Image.open(crop.path) as opened:
            rgb = opened.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    except Exception as error:
        raise ValueError(f"could not load eligible crop {crop.crop_id}: {error}") from error
    return torch.from_numpy(np.moveaxis(array, -1, 0).copy())


def _load_batch(crops: list[_Crop], device: torch.device) -> Tensor:
    return torch.stack([_load_crop(crop) for crop in crops]).to(device)


def _photometric_view(images: Tensor, generator: torch.Generator) -> Tensor:
    """Apply style-only changes; this deliberately contains no geometric operation."""
    count = len(images)
    shape = (count, 1, 1, 1)
    random_values = torch.rand((count, 4), generator=generator, device="cpu").to(images.device)
    brightness = 0.9 + 0.2 * random_values[:, 0].reshape(shape)
    contrast = 0.9 + 0.2 * random_values[:, 1].reshape(shape)
    grayscale_mix = 0.15 * random_values[:, 2].reshape(shape)
    noise = torch.randn(images.shape, generator=generator, device="cpu").to(images.device) * 0.01
    output = images * brightness
    mean = output.mean(dim=(2, 3), keepdim=True)
    output = (output - mean) * contrast + mean
    grayscale = output.mean(dim=1, keepdim=True)
    output = output * (1.0 - grayscale_mix) + grayscale * grayscale_mix
    output = (output + noise).clamp(0.0, 1.0)
    if bool((random_values[:, 3] > 0.5).any()):
        blurred = functional.avg_pool2d(output, kernel_size=3, stride=1, padding=1)
        apply_blur = (random_values[:, 3] > 0.5).reshape(shape)
        output = torch.where(apply_blur, blurred, output)
    return output


def _contrastive_loss(first: Tensor, second: Tensor) -> Tensor:
    if len(first) < 2:
        return first.sum() * 0.0
    embeddings = torch.cat((first, second), dim=0)
    similarities = embeddings @ embeddings.T / 0.15
    count = len(embeddings)
    diagonal = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    similarities = similarities.masked_fill(diagonal, float("-inf"))
    targets = torch.cat(
        (torch.arange(len(first), 2 * len(first)), torch.arange(0, len(first)))
    ).to(embeddings.device)
    return functional.cross_entropy(similarities, targets)


def _embedding_summary(
    encoder: torch.nn.Module, crops: list[_Crop], device: torch.device
) -> Tensor:
    rows: list[Tensor] = []
    encoder.eval()
    with torch.inference_mode():
        for crop in crops:
            rows.append(encoder(_load_batch([crop], device)).cpu())
    return torch.cat(rows, dim=0)


def _restore_classifier(checkpoint_path: Path) -> tuple[GlyphClassifier, dict[str, object]]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("prior checkpoint must be a mapping")
    checkpoint = cast(dict[str, object], raw)
    config = checkpoint.get("config")
    catalog = checkpoint.get("char_to_id")
    state = checkpoint.get("model_state")
    if not isinstance(config, dict) or not isinstance(catalog, dict) or not isinstance(state, dict):
        raise ValueError("prior checkpoint is missing config, character catalog, or model_state")
    embedding_dim = config.get("embedding_dim")
    if not isinstance(embedding_dim, int) or embedding_dim < 2:
        raise ValueError("prior checkpoint config has invalid embedding_dim")
    catalog_is_valid = catalog and all(
        isinstance(key, str) and isinstance(value, int) for key, value in catalog.items()
    )
    if not catalog_is_valid:
        raise ValueError("prior checkpoint has invalid character catalog")
    classifier = GlyphClassifier(
        len(catalog),
        TrainConfig(manifest=Path("."), output_dir=Path("."), embedding_dim=embedding_dim),
    )
    classifier.load_state_dict(cast(dict[str, Tensor], state), strict=True)
    return classifier, checkpoint


def adapt_real_encoder(config: AdaptConfig) -> AdaptArtifacts:
    """Adapt a prior glyph encoder without consuming labels or locked-test pixels."""
    started = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {config.device}")
    device = torch.device(config.device)
    eligible = _load_eligible_crops(config.crop_manifest, config.real_manifest)
    train_crops, diagnostic_crops = _diagnostic_split(
        eligible, config.seed, config.diagnostic_fraction
    )
    classifier, prior = _restore_classifier(config.prior_checkpoint)
    encoder = classifier.encoder.to(device)
    cpu_smoke = device.type == "cpu" and config.max_steps is not None and config.max_steps <= 2
    if cpu_smoke:
        for parameter in encoder.features.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in encoder.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    before = _embedding_summary(encoder, diagnostic_crops, device)
    generator = torch.Generator().manual_seed(config.seed)
    history: list[dict[str, float]] = []
    step = 0
    encoder.train()
    for _epoch in range(config.epochs):
        order = torch.randperm(len(train_crops), generator=generator).tolist()
        for start in range(0, len(order), config.batch_size):
            selected = [train_crops[index] for index in order[start : start + config.batch_size]]
            images = _load_batch(selected, device)
            first = encoder(_photometric_view(images, generator))
            second = encoder(_photometric_view(images, generator))
            loss = _contrastive_loss(first, second)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            history.append({"step": float(step), "contrastive": float(loss.detach())})
            step += 1
            if config.max_steps is not None and step >= config.max_steps:
                break
        if config.max_steps is not None and step >= config.max_steps:
            break

    after = _embedding_summary(encoder, diagnostic_crops, device)
    drift = 1.0 - (before * after).sum(dim=1)
    all_embeddings = _embedding_summary(encoder, eligible, device)
    variance = float(all_embeddings.var(dim=0, unbiased=False).mean())
    norm_mean = float(all_embeddings.norm(dim=1).mean())
    norm_std = float(all_embeddings.norm(dim=1).std(unbiased=False))
    collapse_passed = variance >= config.min_embedding_variance and 0.98 <= norm_mean <= 1.02
    if not collapse_passed:
        raise ValueError(
            "collapse guard failed: "
            f"embedding_variance={variance:.6g}, norm_mean={norm_mean:.6g}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "encoder.pt"
    checkpoint_part = checkpoint_path.with_name(f"{checkpoint_path.name}.part")
    torch.save(
        {
            "model_state": classifier.state_dict(),
            "char_to_id": prior["char_to_id"],
            "config": prior["config"],
            "parent_checkpoint_sha256": _sha256(config.prior_checkpoint),
            "crop_manifest_sha256": _sha256(config.crop_manifest),
            "real_manifest_sha256": _sha256(config.real_manifest),
            "calibrated_for_block_decisions": False,
        },
        checkpoint_part,
    )
    checkpoint_part.replace(checkpoint_path)
    metrics_path = config.output_dir / "metrics.json"
    lock_path = Path("uv.lock")
    _atomic_json(
        metrics_path,
        {
            "seed": config.seed,
            "device": str(device),
            "git_commit": _git_commit(),
            "uv_lock_sha256": _sha256(lock_path) if lock_path.is_file() else "unknown",
            "parent_checkpoint_sha256": _sha256(config.prior_checkpoint),
            "crop_manifest_sha256": _sha256(config.crop_manifest),
            "real_manifest_sha256": _sha256(config.real_manifest),
            "eligible_crop_count": len(eligible),
            "train_crop_count": len(train_crops),
            "diagnostic_crop_count": len(diagnostic_crops),
            "diagnostic_subset": "deterministic eligible-only heldout subset",
            "loss_history": history,
            "steps": step,
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_smoke_backbone_frozen": cpu_smoke,
            "augmentation": "brightness, contrast, grayscale/color, noise, blur only; no geometry",
            "held_out_embedding_drift": {
                "mean_cosine_distance": float(drift.mean()),
                "max_cosine_distance": float(drift.max()),
            },
            "collapse_guard": {
                "passed": collapse_passed,
                "embedding_dimension_variance": variance,
                "embedding_norm_mean": norm_mean,
                "embedding_norm_std": norm_std,
                "minimum_embedding_variance": config.min_embedding_variance,
            },
            "calibrated_for_block_decisions": False,
        },
    )
    return AdaptArtifacts(checkpoint=checkpoint_path, metrics=metrics_path)
