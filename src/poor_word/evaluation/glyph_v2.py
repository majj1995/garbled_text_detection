"""Calibration-locked evaluation for experimental synthetic V2 glyph datasets."""

import hashlib
import json
import math
import shlex
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

from poor_word.evaluation.metrics import base_rate_precision
from poor_word.glyphs.catalog import load_common_chars
from poor_word.glyphs.v2_manifest import V2_SCHEMA
from poor_word.models.prototypes import PrototypeBank
from poor_word.training.dataset import GlyphDataset
from poor_word.training.train_glyph import GlyphClassifier, TrainConfig

REPORT_SCHEMA = "glyph-evaluation-report-v2"
MEMBERSHIP_SCHEMA = "glyph-training-membership-v2"


@dataclass(frozen=True)
class CalibrationThreshold:
    threshold: float
    normal_count: int
    allowed_false_positives: int
    false_positives: int
    empirical_fpr: float


@dataclass(frozen=True)
class GlyphV2ReportArtifacts:
    json_path: Path
    markdown_path: Path
    scores_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability(name: str, value: float) -> float:
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return float(value)


def _fpr_budget(value: float) -> float:
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("max_fpr must be finite and in (0, 1]")
    return float(value)


def select_calibrated_threshold(
    normal_scores: Sequence[float] | NDArray[np.float64], *, max_fpr: float
) -> CalibrationThreshold:
    """Choose the lowest ``score >= threshold`` boundary within an empirical FPR budget."""
    budget = _fpr_budget(max_fpr)
    values = np.asarray(normal_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("calibration normal scores must be a non-empty finite vector")
    allowed = math.floor(budget * len(values))
    descending = np.sort(values)[::-1]
    if allowed < len(values):
        first_excluded = float(descending[allowed])
        threshold = float(np.nextafter(first_excluded, np.inf))
        if not np.isfinite(threshold):
            raise ValueError("calibration scores are too large for a finite threshold")
    else:
        threshold = float(descending[-1])
    false_positives = int(np.count_nonzero(values >= threshold))
    if false_positives > allowed:  # defensive check around floating-point boundaries
        raise RuntimeError("tie-aware threshold selection exceeded its false-positive budget")
    return CalibrationThreshold(
        threshold=threshold,
        normal_count=len(values),
        allowed_false_positives=allowed,
        false_positives=false_positives,
        empirical_fpr=false_positives / len(values),
    )


def _validated_labels_scores(
    labels: NDArray[np.int64], scores: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    label_values = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    if (
        label_values.ndim != 1
        or score_values.ndim != 1
        or len(label_values) == 0
        or len(label_values) != len(score_values)
    ):
        raise ValueError("labels and scores must be equal non-empty vectors")
    if not np.all(np.isin(label_values, (0, 1))):
        raise ValueError("labels must contain only zero and one")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("scores must be finite")
    return label_values, score_values


def _locked_metrics(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    *,
    threshold: float,
    prevalence: float,
) -> dict[str, int | float | None]:
    label_values, score_values = _validated_labels_scores(labels, scores)
    prevalence_value = _probability("prevalence", prevalence)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = score_values >= threshold
    positive = label_values == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    positive_count = int(np.count_nonzero(positive))
    negative_count = len(label_values) - positive_count
    fn = positive_count - tp
    tn = negative_count - fp
    recall = tp / positive_count if positive_count else None
    fpr = fp / negative_count if negative_count else None
    two_classes = positive_count > 0 and negative_count > 0
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "fpr": fpr,
        "auroc": float(roc_auc_score(label_values, score_values)) if two_classes else None,
        "aucpr": (
            float(average_precision_score(label_values, score_values)) if two_classes else None
        ),
        "projected_production_ppv": (
            base_rate_precision(prevalence_value, recall, fpr)
            if recall is not None and fpr is not None
            else None
        ),
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required artifact is missing: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _string_list(payload: dict[str, Any], key: str, *, unique: bool = True) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"training membership {key} must be a list of non-empty strings")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"training membership {key} must not contain duplicates")
    return cast(list[str], value)


def _validate_training_artifacts(artifacts_dir: Path, dataset_id: str) -> dict[str, Any]:
    paths = {
        "checkpoint": artifacts_dir / "encoder.pt",
        "prototypes": artifacts_dir / "prototypes.npz",
        "prototype_metadata": artifacts_dir / "prototypes.json",
        "training_membership": artifacts_dir / "training_membership.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required artifact is missing: {path}")
    membership = _read_object(paths["training_membership"])
    if membership.get("schema_version") != MEMBERSHIP_SCHEMA:
        raise ValueError("training membership has an invalid schema_version")
    if membership.get("dataset_id") != dataset_id:
        raise ValueError("training membership dataset_id differs from evaluation splits")
    manifest_hash = membership.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError("training membership manifest_sha256 is malformed")
    sample_ids = _string_list(membership, "sample_ids")
    source_group_ids = _string_list(membership, "source_group_ids", unique=False)
    pixel_hashes = _string_list(membership, "pixel_sha256")
    if not (len(sample_ids) == len(source_group_ids) == len(pixel_hashes)):
        raise ValueError("training membership identity lists must have equal length")
    if any(
        len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        for value in pixel_hashes
    ):
        raise ValueError("training membership pixel_sha256 contains a malformed hash")
    membership_hash = _sha256(paths["training_membership"])
    raw_checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("encoder checkpoint must be a mapping")
    checkpoint = cast(dict[str, Any], raw_checkpoint)
    prototype_metadata = _read_object(paths["prototype_metadata"])
    if checkpoint.get("dataset_schema_version") != V2_SCHEMA:
        raise ValueError("checkpoint dataset schema is not glyph-dataset-v2")
    if checkpoint.get("experimental_only") is not True:
        raise ValueError("checkpoint must retain experimental_only=true")
    if checkpoint.get("production_allowed") is not False:
        raise ValueError("checkpoint must retain production_allowed=false")
    if prototype_metadata.get("dataset_schema_version") != V2_SCHEMA:
        raise ValueError("prototype metadata dataset schema is not glyph-dataset-v2")
    if prototype_metadata.get("experimental_only") != "true":
        raise ValueError("prototype metadata must retain experimental_only=true")
    if prototype_metadata.get("production_allowed") != "false":
        raise ValueError("prototype metadata must retain production_allowed=false")
    if prototype_metadata.get("encoder_checkpoint_sha256") != _sha256(paths["checkpoint"]):
        raise ValueError("prototype metadata encoder_checkpoint_sha256 mismatch")
    if prototype_metadata.get("prototype_bank_sha256") != _sha256(paths["prototypes"]):
        raise ValueError("prototype metadata prototype_bank_sha256 mismatch")
    checkpoint_catalog = checkpoint.get("char_to_id")
    if isinstance(checkpoint_catalog, dict):
        catalog_hash = hashlib.sha256(
            "".join(sorted(str(key) for key in checkpoint_catalog)).encode("utf-8")
        ).hexdigest()
        if prototype_metadata.get("catalog_sha256") != catalog_hash:
            raise ValueError("prototype metadata catalog_sha256 mismatch")
    if prototype_metadata.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("prototype metadata source_manifest_sha256 mismatch")
    for name, payload in (("checkpoint", checkpoint), ("prototype metadata", prototype_metadata)):
        if payload.get("dataset_id") != dataset_id:
            raise ValueError(f"{name} dataset_id differs from evaluation splits")
        if payload.get("training_membership_sha256") != membership_hash:
            raise ValueError(f"{name} training_membership_sha256 mismatch")
    return {
        "paths": paths,
        "membership": membership,
        "sample_ids": set(sample_ids),
        "source_group_ids": set(source_group_ids),
        "pixel_sha256": set(pixel_hashes),
        "hashes": {name: _sha256(path) for name, path in paths.items()},
        "checkpoint": checkpoint,
    }


def _row_identity_sets(dataset: GlyphDataset) -> dict[str, set[str]]:
    return {
        "sample_id": {str(row["sample_id"]) for row in dataset.rows},
        "source_group_id": {str(row["source_group_id"]) for row in dataset.rows},
        "pixel_sha256": {str(row["pixel_sha256"]) for row in dataset.rows},
    }


def _reject_overlap(left: dict[str, set[str]], right: dict[str, set[str]], label: str) -> None:
    for key in ("sample_id", "source_group_id", "pixel_sha256"):
        overlap = left[key] & right[key]
        if overlap:
            raise ValueError(f"{label} overlap on {key}: {sorted(overlap)[0]}")


def _score_manifest(
    manifest: Path,
    artifacts_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
    """Run the existing encoder/prototype architecture with bounded prototype scoring."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device_name}")
    checkpoint_path = artifacts_dir / "encoder.pt"
    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("encoder checkpoint must be a mapping")
    checkpoint = cast(dict[str, Any], raw)
    raw_config = checkpoint.get("config")
    raw_catalog = checkpoint.get("char_to_id")
    raw_state = checkpoint.get("model_state")
    if (
        not isinstance(raw_config, dict)
        or not isinstance(raw_catalog, dict)
        or not isinstance(raw_state, dict)
    ):
        raise ValueError("encoder checkpoint is missing config, character catalog, or model_state")
    if not all(
        isinstance(key, str) and isinstance(value, int) for key, value in raw_catalog.items()
    ):
        raise ValueError("encoder character catalog must map strings to integers")
    catalog = cast(dict[str, int], raw_catalog)
    config = TrainConfig.model_validate(
        {**cast(dict[str, Any], raw_config), "pretrained": False, "device": device_name}
    )
    model = GlyphClassifier(len(catalog), config).to(device)
    model.load_state_dict(raw_state)
    model.eval()
    dataset = GlyphDataset(manifest)
    if dataset.char_to_id != catalog:
        raise ValueError("manifest character catalog does not match encoder checkpoint")
    bank, metadata = PrototypeBank.load(artifacts_dir / "prototypes.npz")
    if metadata.get("encoder_checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError("prototype bank checkpoint hash does not match encoder.pt")

    started = time.perf_counter()
    emit = progress if progress is not None else lambda _message: None
    embeddings: list[NDArray[np.float32]] = []
    last_log = 0.0
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            views = torch.stack([dataset[index].views for index in range(start, stop)]).to(device)
            values, _ = model(views)
            embeddings.append(values.cpu().numpy().astype(np.float32, copy=False))
            now = time.monotonic()
            if now - last_log >= 5.0 or stop == len(dataset):
                emit(f"inference={stop}/{len(dataset)}")
                last_log = now
    all_embeddings = np.concatenate(embeddings, axis=0)
    score_blocks: list[NDArray[np.float64]] = []
    for start in range(0, len(dataset), 4096):
        stop = min(start + 4096, len(dataset))
        scored = bank.score(all_embeddings[start:stop])
        score_blocks.append(scored.nearest_distance.astype(np.float64))
        emit(f"prototype_scoring={stop}/{len(dataset)}")
    labels = np.asarray([row["decision"] == "BLOCK" for row in dataset.rows], dtype=np.int64)
    return labels, np.concatenate(score_blocks), time.perf_counter() - started


def _groups(
    rows: list[dict[str, Any]], predicted: NDArray[np.bool_], key: str
) -> dict[str, dict[str, int | float | None]]:
    names = sorted(
        {
            str(row[key])
            for row in rows
            if row["decision"] == "BLOCK" and row.get(key) not in (None, "")
        }
    )
    result: dict[str, dict[str, int | float | None]] = {}
    for name in names:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["decision"] == "BLOCK" and str(row.get(key)) == name
        ]
        tp = int(np.count_nonzero(predicted[indices]))
        result[name] = {
            "count": len(indices),
            "tp": tp,
            "fn": len(indices) - tp,
            "recall": tp / len(indices) if indices else None,
        }
    return result


def _split_report(
    dataset: GlyphDataset,
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    *,
    threshold: float,
    prevalence: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    report: dict[str, Any] = dict(
        _locked_metrics(labels, scores, threshold=threshold, prevalence=prevalence)
    )
    predicted = scores >= threshold
    report.update(
        {
            "elapsed_seconds": elapsed_seconds,
            "by_anomaly_operator": _groups(dataset.rows, predicted, "operator"),
            "by_bridge_mode": _groups(dataset.rows, predicted, "bridge_mode"),
        }
    )
    return report


def _source_hashes(*datasets: GlyphDataset) -> dict[str, str]:
    result: dict[str, str] = {}
    for dataset in datasets:
        run = dataset.run_metadata
        for key in ("source_assets", "source_asset_sha256", "source_lock_sha256"):
            raw = run.get(key)
            if not isinstance(raw, dict):
                continue
            for name, digest in raw.items():
                if not isinstance(name, str) or not isinstance(digest, str):
                    raise ValueError(f"generation run {key} must map strings to hashes")
                if name in result and result[name] != digest:
                    raise ValueError(f"generation runs disagree on source hash for {name}")
                result[name] = digest
    return result


def _shared_run_provenance(*datasets: GlyphDataset) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("source", "license", "grouping_limitation"):
        values = [dataset.run_metadata.get(key) for dataset in datasets]
        present = [value for value in values if value is not None]
        if present and any(value != present[0] for value in present[1:]):
            raise ValueError(f"generation runs disagree on {key} provenance")
        if present:
            result[key] = present[0]
    return result


def _score_rows(
    dataset: GlyphDataset,
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, label, score in zip(dataset.rows, labels, scores, strict=True):
        rows.append(
            {
                "schema_version": REPORT_SCHEMA,
                "dataset_id": dataset.dataset_id,
                "split_role": dataset.split_role,
                "sample_id": str(row["sample_id"]),
                "source_group_id": str(row["source_group_id"]),
                "pixel_sha256": str(row["pixel_sha256"]),
                "decision": str(row["decision"]),
                "label": int(label),
                "score": float(score),
                "threshold": threshold,
                "predicted_block": bool(score >= threshold),
                "operator": str(row["operator"]),
                "anomaly_kind": str(row["anomaly_kind"]),
                "bridge_mode": row.get("bridge_mode"),
                "label_provenance": str(row["label_provenance"]),
            }
        )
    return rows


def _commands(
    calibration: Path,
    test: Path,
    artifacts: Path,
    output: Path,
    *,
    checkpoint: dict[str, Any],
    run: dict[str, Any],
    prevalence: float,
    max_fpr: float,
    device: str,
    batch_size: int,
) -> list[str]:
    quote = shlex.quote
    run_config = run.get("config")
    fresh_generated = calibration.parent.with_name(f"{calibration.parent.name}-reproduced")
    fresh_artifacts = artifacts.with_name(f"{artifacts.name}-reproduced")
    fresh_output = output.with_name(f"{output.name}-reproduced")
    profile: str | None = None
    if isinstance(run_config, dict):
        characters = run_config.get("characters")
        counts = (
            run_config.get("train_normal_per_char"),
            run_config.get("eval_normal_per_char"),
            run_config.get("train_abnormal_per_operator"),
            run_config.get("eval_abnormal_per_operator"),
            run_config.get("max_attempts"),
        )
        if characters == list("永明林国春田合口") and counts == (4, 2, 1, 1, 8):
            profile = "smoke"
        elif counts == (8, 4, 2, 1, 8):
            catalog_path = Path("data/raw/common_chars_3500.txt")
            try:
                canonical_mvp = list(load_common_chars(catalog_path))
            except (FileNotFoundError, ValueError):
                canonical_mvp = None
            if characters == canonical_mvp:
                profile = "mvp"
    if profile is None:
        generate_command = (
            f"# Custom generation: replay the V2GenerationConfig recorded in "
            f"{quote(str(calibration.parent / 'run.json'))} via the Python API."
        )
        reproduced_calibration = calibration
        reproduced_test = test
        reproduced_train = None
    else:
        assert isinstance(run_config, dict)
        generate_command = (
            f"uv run poor-word glyphs generate-v2 --profile {profile} "
            f"--seed {run_config['seed']} --allow-experimental "
            f"--graphics {quote(str(run_config['graphics_path']))} "
            f"--source-lock {quote(str(run_config['source_lock_path']))} "
            f"--license {quote(str(run_config['license_path']))} "
            f"--output-dir {quote(str(fresh_generated))}"
        )
        reproduced_calibration = fresh_generated / "calibration.parquet"
        reproduced_test = fresh_generated / "test.parquet"
        reproduced_train = fresh_generated / "train.parquet"
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint config is required for reproduction commands")
    training_manifest = reproduced_train or checkpoint_config["manifest"]
    max_steps = checkpoint_config.get("max_steps")
    max_steps_option = f" --max-steps {max_steps}" if max_steps is not None else ""
    pretrained_option = "--pretrained" if checkpoint_config["pretrained"] else "--no-pretrained"
    cli_training = (
        checkpoint_config.get("learning_rate", 3e-4) == 3e-4
        and checkpoint_config.get("embedding_dim", 256) == 256
    )
    if cli_training:
        training_command = (
            f"uv run poor-word train glyph --manifest {quote(str(training_manifest))} "
            f"--sampler {checkpoint_config['sampler']} --allow-experimental "
            f"--epochs {checkpoint_config['epochs']}{max_steps_option} "
            f"--batch-size {checkpoint_config['batch_size']} --seed {checkpoint_config['seed']} "
            f"{pretrained_option} --device {quote(str(checkpoint_config['device']))} "
            f"--log-every {checkpoint_config['log_every']} "
            f"--output-dir {quote(str(fresh_artifacts))}"
        )
    else:
        training_command = (
            "# Custom training: replay the TrainConfig stored in "
            f"{quote(str(artifacts / 'encoder.pt'))} via the Python API, changing only "
            f"output_dir to {quote(str(fresh_artifacts))}."
        )
    return [
        generate_command,
        training_command,
        "uv run poor-word evaluate glyph-v2 "
        f"--calibration-manifest {quote(str(reproduced_calibration))} "
        f"--test-manifest {quote(str(reproduced_test))} "
        f"--artifacts {quote(str(fresh_artifacts))} "
        f"--allow-experimental --max-fpr {max_fpr:.17g} --prevalence {prevalence:.17g} "
        f"--device {quote(device)} --batch-size {batch_size} "
        f"--output-dir {quote(str(fresh_output))}",
    ]


def _group_markdown(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "All rows use the single calibration-locked threshold.",
        "",
        "| Group | Count | TP | FN | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if not groups:
        lines.append("| None | 0 | 0 | 0 | — |")
    for name, item in groups.items():
        recall = "—" if item["recall"] is None else f"{item['recall']:.4%}"
        lines.append(f"| {name} | {item['count']} | {item['tp']} | {item['fn']} | {recall} |")
    return [*lines, ""]


def _render_markdown(report: dict[str, Any]) -> str:
    threshold = report["threshold"]
    calibration = report["calibration"]
    test = report["test"]
    support = report["normal_support"]
    warnings = cast(list[str], report["warnings"])
    lines = [
        "# Experimental glyph V2 calibrated evaluation",
        "",
        "## Scope",
        "",
        "This is a synthetic-rule V2 evaluation, not a production gate or production claim.",
        "The splits use the same catalog/source font; only renditions and source groups "
        "are disjoint.",
        "Synthetic weak labels and correlated variants cannot establish real-world FPR.",
        "",
        "## Calibration-locked threshold",
        "",
        f"- Threshold: {threshold['value']:.17g} (`score >= threshold`)",
        "- Source: calibration normal scores only; no calibration positives or test data "
        "selected it.",
        f"- Calibration empirical FPR: {threshold['observed_fpr']:.6%} "
        f"({threshold['observed_fp']}/{threshold['normal_count']})",
        f"- Calibration normals/resolution: {support['calibration']['count']} / "
        f"{support['calibration']['empirical_fpr_resolution']:.6%}",
        f"- Test normals/resolution: {support['test']['count']} / "
        f"{support['test']['empirical_fpr_resolution']}",
        "",
        *_group_markdown(
            "Calibration recall by anomaly operator", calibration["by_anomaly_operator"]
        ),
        *_group_markdown("Calibration recall by bridge mode", calibration["by_bridge_mode"]),
        "## Locked test results",
        "",
        f"- TP={test['tp']} FP={test['fp']} TN={test['tn']} FN={test['fn']}",
        f"- Recall={test['recall']}; FPR={test['fpr']}",
        f"- AUROC={test['auroc']}; AUCPR={test['aucpr']}",
        f"- PPV={test['projected_production_ppv']} at prevalence "
        f"{report['production_prevalence']}; projected, not measured deployment precision.",
        "",
        *_group_markdown("Test recall by anomaly operator", test["by_anomaly_operator"]),
        *_group_markdown("Test recall by bridge mode", test["by_bridge_mode"]),
        "## Warnings",
        "",
        *(f"- {warning}" for warning in warnings),
        "",
        "## Reproduction commands",
        "",
    ]
    for command in cast(list[str], report["reproduction_commands"]):
        lines.extend(["```bash", command, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def evaluate_glyph_v2(
    calibration_manifest: Path,
    test_manifest: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    prevalence: float = 0.001,
    max_fpr: float = 0.0001,
    device: str = "cpu",
    batch_size: int = 64,
    allow_experimental: bool = False,
    progress: Callable[[str], None] | None = None,
) -> GlyphV2ReportArtifacts:
    """Validate, calibrate on normal calibration scores, then score the locked test split."""
    output = output_dir.resolve()
    if output_dir.is_symlink() or output.exists():
        raise FileExistsError(f"evaluation output already exists; choose a fresh path: {output}")
    if not allow_experimental:
        raise ValueError("V2 synthetic evaluation requires allow_experimental=True")
    if calibration_manifest.resolve() == test_manifest.resolve():
        raise ValueError("calibration and test manifests must be distinct input paths")
    prevalence_value = _probability("prevalence", prevalence)
    _fpr_budget(max_fpr)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    emit = progress if progress is not None else lambda _message: None
    emit("Validating experimental V2 splits, assets, and training membership...")
    calibration = GlyphDataset(calibration_manifest)
    test = GlyphDataset(test_manifest)
    for expected_role, dataset in (("calibration", calibration), ("test", test)):
        if dataset.schema_version != V2_SCHEMA or not dataset.experimental_only:
            raise ValueError(f"{expected_role} manifest must use experimental {V2_SCHEMA}")
        if dataset.split_role != expected_role:
            raise ValueError(f"expected {expected_role} split, got {dataset.split_role!r}")
    if calibration.dataset_id != test.dataset_id or calibration.dataset_id is None:
        raise ValueError("calibration and test splits must have the same dataset_id")
    if calibration.char_to_id != test.char_to_id:
        raise ValueError("calibration and test character catalogs must match exactly")
    artifact = _validate_training_artifacts(artifacts_dir, calibration.dataset_id)
    checkpoint_catalog = artifact["checkpoint"].get("char_to_id")
    if checkpoint_catalog is not None and checkpoint_catalog != calibration.char_to_id:
        raise ValueError("evaluation character catalog differs from the training checkpoint")
    calibration_ids = _row_identity_sets(calibration)
    test_ids = _row_identity_sets(test)
    membership_ids = {
        "sample_id": artifact["sample_ids"],
        "source_group_id": artifact["source_group_ids"],
        "pixel_sha256": artifact["pixel_sha256"],
    }
    _reject_overlap(membership_ids, calibration_ids, "training/calibration")
    _reject_overlap(membership_ids, test_ids, "training/test")
    _reject_overlap(calibration_ids, test_ids, "calibration/test")

    emit(f"Scoring calibration split ({len(calibration)} glyphs)...")
    calibration_labels, calibration_scores, calibration_elapsed = _score_manifest(
        calibration_manifest,
        artifacts_dir,
        device_name=device,
        batch_size=batch_size,
        progress=lambda message: emit(f"calibration {message}"),
    )
    expected_calibration_labels = np.asarray(
        [row["decision"] == "BLOCK" for row in calibration.rows], dtype=np.int64
    )
    if not np.array_equal(calibration_labels, expected_calibration_labels):
        raise ValueError("calibration scorer labels differ from manifest decisions")
    normal_scores = calibration_scores[calibration_labels == 0]
    selected = select_calibrated_threshold(normal_scores, max_fpr=max_fpr)
    emit(f"Calibration complete; locked threshold={selected.threshold:.17g}.")
    emit(f"Scoring locked test split ({len(test)} glyphs)...")
    test_labels, test_scores, test_elapsed = _score_manifest(
        test_manifest,
        artifacts_dir,
        device_name=device,
        batch_size=batch_size,
        progress=lambda message: emit(f"test {message}"),
    )
    expected_test_labels = np.asarray(
        [row["decision"] == "BLOCK" for row in test.rows], dtype=np.int64
    )
    if not np.array_equal(test_labels, expected_test_labels):
        raise ValueError("test scorer labels differ from manifest decisions")
    required_normals = math.ceil(1.0 / max_fpr)
    warnings = [
        "Synthetic weak labels only; no production gate pass is claimed.",
        "Variants from the same catalog/source font are correlated and cannot establish real FPR.",
    ]
    test_normal_count = int(np.count_nonzero(test_labels == 0))
    if selected.normal_count < required_normals:
        warnings.append(
            f"Calibration has {selected.normal_count} normals, below "
            f"ceil(1/max_fpr)={required_normals}."
        )
    if test_normal_count < required_normals:
        warnings.append(
            f"Test has {test_normal_count} normals, below ceil(1/max_fpr)={required_normals}."
        )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "claim_scope": "synthetic_rule_v2_only",
        "dataset_id": calibration.dataset_id,
        "dataset_schema_version": V2_SCHEMA,
        "experimental_only": True,
        "production_allowed": False,
        "label_provenance": "synthetic_rule_v2",
        "production_prevalence": prevalence_value,
        "model": {
            "encoder": "GlyphClassifier/GlyphEncoder",
            "prototype_bank": "PrototypeBank",
            "score": "nearest_prototype_cosine_distance",
        },
        "threshold": {
            "value": selected.threshold,
            "source": "calibration_normal_scores",
            "comparison": "score >= threshold",
            "max_fpr": max_fpr,
            "normal_count": selected.normal_count,
            "allowed_fp": selected.allowed_false_positives,
            "observed_fp": selected.false_positives,
            "observed_fpr": selected.empirical_fpr,
            "required_normal_count_for_one_fpr_event": required_normals,
        },
        "calibration": _split_report(
            calibration,
            calibration_labels,
            calibration_scores,
            threshold=selected.threshold,
            prevalence=prevalence_value,
            elapsed_seconds=calibration_elapsed,
        ),
        "test": _split_report(
            test,
            test_labels,
            test_scores,
            threshold=selected.threshold,
            prevalence=prevalence_value,
            elapsed_seconds=test_elapsed,
        ),
        "normal_support": {
            "calibration": {
                "count": selected.normal_count,
                "empirical_fpr_resolution": 1.0 / selected.normal_count,
            },
            "test": {
                "count": test_normal_count,
                "empirical_fpr_resolution": 1.0 / test_normal_count if test_normal_count else None,
            },
        },
        "provenance": {
            "manifests": {
                "calibration": _sha256(calibration_manifest),
                "test": _sha256(test_manifest),
            },
            "generation_runs": {
                "calibration": _sha256(calibration_manifest.parent / "run.json"),
                "test": _sha256(test_manifest.parent / "run.json"),
            },
            "source_asset_sha256": _source_hashes(calibration, test),
            "generation_source": _shared_run_provenance(calibration, test),
            "model_artifacts": artifact["hashes"],
            "training_manifest_sha256": artifact["membership"]["manifest_sha256"],
        },
        "warnings": warnings,
        "reproduction_commands": _commands(
            calibration_manifest,
            test_manifest,
            artifacts_dir,
            output_dir,
            checkpoint=artifact["checkpoint"],
            run=calibration.run_metadata,
            prevalence=prevalence_value,
            max_fpr=max_fpr,
            device=device,
            batch_size=batch_size,
        ),
    }
    score_rows = [
        *_score_rows(calibration, calibration_labels, calibration_scores, selected.threshold),
        *_score_rows(test, test_labels, test_scores, selected.threshold),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        json_path = staging / "report.json"
        markdown_path = staging / "report.md"
        scores_path = staging / "scores.parquet"
        json_path.write_text(
            f"{json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_render_markdown(report), encoding="utf-8")
        pq.write_table(pa.Table.from_pylist(score_rows), scores_path)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    emit(f"Published V2 evaluation to {output}.")
    return GlyphV2ReportArtifacts(
        json_path=output / "report.json",
        markdown_path=output / "report.md",
        scores_path=output / "scores.parquet",
    )
