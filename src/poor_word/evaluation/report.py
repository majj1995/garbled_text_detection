import hashlib
import json
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from poor_word.data.manifest import load_source_lock
from poor_word.evaluation.metrics import ThresholdEvaluation, evaluate_thresholds
from poor_word.models.prototypes import PrototypeBank
from poor_word.training.dataset import GlyphDataset
from poor_word.training.train_glyph import GlyphClassifier, TrainConfig


class ReportContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_hashes: dict[str, str] = Field(default_factory=dict)
    asset_hashes: dict[str, str] = Field(default_factory=dict)
    model_hashes: dict[str, str] = Field(default_factory=dict)
    source_license_decisions: dict[str, str] = Field(default_factory=dict)
    ocr_capabilities: dict[str, str | bool | float | None] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    failures: tuple[str, ...] = ()
    reproduction_commands: tuple[str, ...]


class MvpReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_scope: Literal["synthetic_only", "real_seed_evaluation"]
    production_prevalence: float = Field(gt=0.0, lt=1.0)
    synthetic_metrics: ThresholdEvaluation
    real_data_metrics: ThresholdEvaluation | None = None
    context: ReportContext


@dataclass(frozen=True)
class ReportArtifacts:
    json_path: Path
    markdown_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], payload)


def _score_manifest(
    manifest: Path,
    artifacts_dir: Path,
    *,
    device_name: str,
    batch_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device_name}")
    checkpoint_path = artifacts_dir / "encoder.pt"
    prototype_path = artifacts_dir / "prototypes.npz"
    if not checkpoint_path.is_file() or not prototype_path.is_file():
        raise FileNotFoundError("artifacts directory must contain encoder.pt and prototypes.npz")

    raw_checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("encoder checkpoint must be a mapping")
    checkpoint = cast(dict[str, object], raw_checkpoint)
    raw_config = checkpoint.get("config")
    raw_char_to_id = checkpoint.get("char_to_id")
    raw_state = checkpoint.get("model_state")
    if not isinstance(raw_config, dict) or not isinstance(raw_char_to_id, dict):
        raise ValueError("encoder checkpoint is missing config or character catalog")
    if not isinstance(raw_state, dict):
        raise ValueError("encoder checkpoint is missing model_state")
    if not all(
        isinstance(character, str) and isinstance(index, int)
        for character, index in raw_char_to_id.items()
    ):
        raise ValueError("encoder character catalog must map strings to integers")
    char_to_id = cast(dict[str, int], raw_char_to_id)
    config_payload = {**cast(dict[str, object], raw_config)}
    config_payload.update({"pretrained": False, "device": device_name})
    config = TrainConfig.model_validate(config_payload)
    model = GlyphClassifier(len(char_to_id), config).to(device)
    model.load_state_dict(cast(dict[str, Tensor], raw_state))
    model.eval()
    dataset = GlyphDataset(manifest)
    if dataset.char_to_id != char_to_id:
        raise ValueError("manifest character catalog does not match encoder checkpoint")
    bank, prototype_metadata = PrototypeBank.load(prototype_path)
    _validate_prototype_metadata(prototype_metadata, checkpoint_path, char_to_id)

    started = time.perf_counter()
    embedded: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            views = torch.stack([dataset[index].views for index in range(start, stop)]).to(
                device
            )
            embeddings, _ = model(views)
            embedded.append(embeddings.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(embedded, axis=0).astype(np.float32, copy=False)
    scores = bank.score(embeddings).nearest_distance.astype(np.float64)
    elapsed_seconds = time.perf_counter() - started
    labels = np.asarray(
        [row["decision"] == "BLOCK" for row in dataset.rows], dtype=np.int64
    )
    return labels, scores, elapsed_seconds


def _validate_prototype_metadata(
    metadata: dict[str, str], checkpoint_path: Path, char_to_id: dict[str, int]
) -> None:
    if metadata.get("encoder_checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError("prototype bank checkpoint hash does not match encoder.pt")
    catalog_hash = hashlib.sha256("".join(sorted(char_to_id)).encode("utf-8")).hexdigest()
    if metadata.get("catalog_sha256") != catalog_hash:
        raise ValueError("prototype bank character catalog does not match encoder.pt")


def _manifest_source_ids(manifest: Path) -> set[str]:
    rows = GlyphDataset(manifest).rows
    return {
        str(source_id)
        for row in rows
        for source_id in cast(list[str], row["source_asset_ids"])
    }


def _source_context(
    repo_root: Path, manifest: Path, run_manifest: Path | None
) -> tuple[dict[str, str], dict[str, str]]:
    hashes: dict[str, str] = {}
    decisions: dict[str, str] = {}
    referenced_source_ids = _manifest_source_ids(manifest)
    recorded_lock_hashes: dict[str, str] = {}
    if run_manifest is not None:
        run_payload = _json_object(run_manifest)
        raw_lock_hashes = run_payload.get("source_lock_sha256")
        if not isinstance(raw_lock_hashes, dict) or not all(
            isinstance(name, str) and isinstance(digest, str)
            for name, digest in raw_lock_hashes.items()
        ):
            raise ValueError("generation run manifest is missing source_lock_sha256")
        recorded_lock_hashes = cast(dict[str, str], raw_lock_hashes)
        unsafe_names = [
            name
            for name in recorded_lock_hashes
            if Path(name).name != name or not name.endswith(".lock.json")
        ]
        if unsafe_names:
            raise ValueError("source lock names must be plain .lock.json file names")
        lock_paths = [repo_root / "data/locks" / name for name in sorted(recorded_lock_hashes)]
    else:
        lock_paths = [
            repo_root / "data/locks" / f"{source_id}.lock.json"
            for source_id in sorted(referenced_source_ids)
        ]

    found_source_ids: set[str] = set()
    for lock_path in lock_paths:
        if not lock_path.is_file():
            raise FileNotFoundError(f"required source lock is missing: {lock_path}")
        recorded_hash = recorded_lock_hashes.get(lock_path.name)
        if recorded_hash is not None and _sha256(lock_path) != recorded_hash:
            raise ValueError(f"source lock changed since generation: {lock_path.name}")
        lock = load_source_lock(lock_path)
        found_source_ids.add(lock.source_id)
        hashes[lock.source_id] = lock.sha256
        decisions[lock.source_id] = (
            f"{lock.license_id}; "
            f"production_allowed={'true' if lock.production_allowed else 'false'}"
        )
    missing_source_ids = referenced_source_ids - found_source_ids
    if missing_source_ids:
        raise ValueError(
            f"generation provenance omits manifest assets: {', '.join(sorted(missing_source_ids))}"
        )
    return hashes, decisions


def evaluate_glyph_artifacts(
    manifest: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    prevalence: float,
    device: str = "cpu",
    batch_size: int = 64,
    ocr_audit_path: Path | None = None,
    repo_root: Path | None = None,
) -> ReportArtifacts:
    """Score a manifest and emit a provenance-rich synthetic MVP report."""
    root = Path.cwd() if repo_root is None else repo_root
    labels, scores, elapsed_seconds = _score_manifest(
        manifest, artifacts_dir, device_name=device, batch_size=batch_size
    )
    dataset_hashes = {"synthetic_manifest": _sha256(manifest)}
    run_manifest = manifest.parent / "run.json"
    if run_manifest.is_file():
        dataset_hashes["generation_run"] = _sha256(run_manifest)
    model_hashes = {
        name: _sha256(artifacts_dir / name)
        for name in ("encoder.pt", "prototypes.npz", "prototypes.json", "metrics.json")
        if (artifacts_dir / name).is_file()
    }
    asset_hashes, license_decisions = _source_context(
        root, manifest, run_manifest if run_manifest.is_file() else None
    )

    failures: list[str] = []
    ocr_capabilities: dict[str, str | bool | float | None]
    if ocr_audit_path is None:
        ocr_capabilities = {"audit_status": "not_supplied"}
        failures.append("PP-OCRv5 L20 audit was not supplied to this report")
    else:
        audit = _json_object(ocr_audit_path)
        selected_keys = (
            "paddleocr_version",
            "paddlepaddle_version",
            "cuda_version",
            "gpu_name",
            "detection_model_name",
            "recognition_model_name",
            "character_boxes_available",
            "logits_available",
            "latency_p50_ms",
            "latency_p95_ms",
            "peak_gpu_memory_mb",
        )
        ocr_capabilities = {
            key: cast(str | bool | float | None, audit[key])
            for key in selected_keys
            if key in audit
        }
        gaps = audit.get("required_capability_gaps", [])
        if isinstance(gaps, list):
            failures.extend(f"PP-OCRv5 capability gap: {gap}" for gap in gaps)
        if audit.get("character_boxes_available") is not True:
            failures.append("PP-OCRv5 character boxes unavailable")
        if audit.get("logits_available") is not True:
            failures.append("PP-OCRv5 raw logits unavailable")
        dataset_hashes["ocr_audit"] = _sha256(ocr_audit_path)

    negative_count = int((labels == 0).sum())
    if negative_count < 10_000:
        failures.append(
            f"Only {negative_count} synthetic normal samples: insufficient empirical "
            "resolution for the 0.01% MVP FPR gate"
        )
    if negative_count < 40_000:
        failures.append(
            f"Only {negative_count} synthetic normal samples: insufficient empirical "
            "resolution for the 0.0025% pilot FPR gate"
        )
    failures.append(
        "This command evaluates the supplied synthetic manifest; use a separately versioned "
        "holdout and real seed set before making deployment claims"
    )

    command = (
        "uv run poor-word evaluate glyph "
        f"--manifest {shlex.quote(str(manifest))} "
        f"--artifacts {shlex.quote(str(artifacts_dir))} "
        f"--prevalence {prevalence} --device {device} --batch-size {batch_size} "
        f"--output-dir {shlex.quote(str(output_dir))}"
    )
    if ocr_audit_path is not None:
        command += f" --ocr-audit {shlex.quote(str(ocr_audit_path))}"
    context = ReportContext(
        dataset_hashes=dataset_hashes,
        asset_hashes=asset_hashes,
        model_hashes=model_hashes,
        source_license_decisions=license_decisions,
        ocr_capabilities=ocr_capabilities,
        latency={
            f"offline_{device.replace(':', '_')}_ms_per_glyph": elapsed_seconds
            * 1000.0
            / len(labels)
        },
        failures=tuple(dict.fromkeys(failures)),
        reproduction_commands=(command,),
    )
    report = build_mvp_report(labels, scores, prevalence=prevalence, context=context)
    return write_mvp_report(report, output_dir)


def build_mvp_report(
    labels: NDArray[np.int64] | list[int],
    scores: NDArray[np.float64] | list[float],
    *,
    prevalence: float,
    context: ReportContext,
    real_labels: NDArray[np.int64] | list[int] | None = None,
    real_scores: NDArray[np.float64] | list[float] | None = None,
) -> MvpReport:
    if (real_labels is None) != (real_scores is None):
        raise ValueError("real_labels and real_scores must be supplied together")
    real_metrics = (
        evaluate_thresholds(real_labels, real_scores, prevalence=prevalence)
        if real_labels is not None and real_scores is not None
        else None
    )
    return MvpReport(
        claim_scope="real_seed_evaluation" if real_metrics is not None else "synthetic_only",
        production_prevalence=prevalence,
        synthetic_metrics=evaluate_thresholds(labels, scores, prevalence=prevalence),
        real_data_metrics=real_metrics,
        context=context,
    )


def _gate_markdown(name: str, evaluation: ThresholdEvaluation) -> list[str]:
    gate = evaluation.mvp_gate if name == "Offline MVP" else evaluation.trial_gate
    point = gate.point
    return [
        f"### {name}: {gate.status.upper()}",
        "",
        f"- Threshold: `{point.threshold:.8g}`",
        f"- Raw counts: TP={point.tp}, FP={point.fp}, TN={point.tn}, FN={point.fn}",
        f"- Recall: {point.recall:.6%}",
        f"- FPR: {point.fpr:.6%}",
        f"- Production-base-rate precision: {point.production_precision:.6%}",
        f"- Negative sample support: {evaluation.negative_count}/{gate.required_negative_count}",
        "",
    ]


def _mapping_markdown(title: str, values: Mapping[str, object]) -> list[str]:
    lines = [f"## {title}", ""]
    if not values:
        return [*lines, "- None", ""]
    return [*lines, *(f"- `{key}`: `{value}`" for key, value in sorted(values.items())), ""]


def render_markdown(report: MvpReport) -> str:
    synthetic = report.synthetic_metrics
    scope_heading = (
        "Not a production claim"
        if report.claim_scope == "synthetic_only"
        else "Real seed evaluation (pilot validation still required)"
    )
    lines = [
        "# Malformed Chinese Glyph MVP Report",
        "",
        f"## {scope_heading}",
        "",
        "Synthetic results measure controlled corruptions. They do not establish production "
        "performance on AIGC traffic.",
        "",
        "## Production prevalence",
        "",
        f"- Assumed anomaly prevalence: {report.production_prevalence:.6%}",
        "- Precision below is recalculated from recall and FPR at that prevalence.",
        "- Evaluation-set class balance is not used as the production base rate.",
        "",
        "## Synthetic split metrics",
        "",
        f"- Positive samples: {synthetic.positive_count}",
        f"- Negative samples: {synthetic.negative_count}",
        f"- Synthetic AUROC: {synthetic.auroc:.6f}",
        f"- Synthetic AUCPR: {synthetic.aucpr:.6f}",
        "",
        *_gate_markdown("Offline MVP", synthetic),
        *_gate_markdown("Pilot trial", synthetic),
    ]
    if report.real_data_metrics is not None:
        real = report.real_data_metrics
        lines.extend(
            [
                "## Real seed data metrics",
                "",
                f"- Positive samples: {real.positive_count}",
                f"- Negative samples: {real.negative_count}",
                f"- AUROC: {real.auroc:.6f}",
                f"- AUCPR: {real.aucpr:.6f}",
                "",
                *_gate_markdown("Offline MVP", real),
                *_gate_markdown("Pilot trial", real),
            ]
        )
    lines.extend(_mapping_markdown("Dataset hashes", report.context.dataset_hashes))
    lines.extend(_mapping_markdown("Asset hashes", report.context.asset_hashes))
    lines.extend(_mapping_markdown("Model hashes", report.context.model_hashes))
    lines.extend(
        _mapping_markdown(
            "Source license decisions", report.context.source_license_decisions
        )
    )
    lines.extend(_mapping_markdown("PP-OCRv5 L20 capabilities", report.context.ocr_capabilities))
    lines.extend(_mapping_markdown("Latency", report.context.latency))
    lines.extend(["## Known failures and capability gaps", ""])
    lines.extend(
        [*(f"- {failure}" for failure in report.context.failures), ""]
        if report.context.failures
        else ["- None recorded", ""]
    )
    lines.extend(["## Reproduction commands", ""])
    for command in report.context.reproduction_commands:
        lines.extend(["```bash", command, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _atomic_text(path: Path, text: str) -> None:
    part = path.with_name(f"{path.name}.part")
    try:
        part.write_text(text, encoding="utf-8")
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def write_mvp_report(report: MvpReport, output_dir: Path) -> ReportArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    _atomic_text(json_path, f"{report.model_dump_json(indent=2)}\n")
    _atomic_text(markdown_path, render_markdown(report))
    return ReportArtifacts(json_path=json_path, markdown_path=markdown_path)
