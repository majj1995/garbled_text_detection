"""Read-only model diagnostics on synthetic glyphs, not a deployment acceptance test."""

# Chinese punctuation is intentional in the user-facing reports below.
# ruff: noqa: RUF001

import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from poor_word.evaluation.metrics import evaluate_thresholds
from poor_word.evaluation.report import _validate_prototype_metadata
from poor_word.glyphs.corrupt import OPERATORS
from poor_word.models.prototypes import PrototypeBank
from poor_word.training.dataset import GlyphDataset
from poor_word.training.pair_coverage import audit_pair_coverage
from poor_word.training.train_glyph import GlyphClassifier, TrainConfig


class DiagnosticConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: Path
    train_manifest: Path
    artifacts_dir: Path
    output_dir: Path
    device: str = "cpu"
    batch_size: int = Field(default=64, ge=1)
    threshold: float | None = Field(default=None, allow_inf_nan=False)
    max_fpr: float = Field(default=0.0001, gt=0, le=1)
    examples_per_kind: int = Field(default=10, ge=0)
    max_false_positives: int = Field(default=50, ge=0)
    seed: int = Field(default=20260910, ge=0)


@dataclass(frozen=True)
class DiagnosticArtifacts:
    json_path: Path
    markdown_path: Path
    scores_path: Path
    examples_path: Path


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    identifiers: set[str] = set()
    for row in rows:
        identifier = row.get("sample_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("diagnostic manifest requires unique, non-empty sample_id values")
        identifiers.add(identifier)
        if row.get("decision") not in {"PASS", "BLOCK"}:
            raise ValueError("synthetic diagnostics only accepts PASS/BLOCK labels")
        if row["decision"] == "BLOCK" and row.get("operator") not in OPERATORS:
            raise ValueError("BLOCK sample has an unsupported synthetic operator")


def summarize_scores(
    rows: list[dict[str, Any]],
    scores: NDArray[np.float64],
    *,
    threshold: float | None,
    max_fpr: float,
) -> dict[str, Any]:
    """Apply one global >= threshold; never fit a threshold separately by error type."""
    _validate_rows(rows)
    if not np.isfinite(max_fpr) or not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be finite and in (0, 1]")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    labels = np.asarray([row["decision"] == "BLOCK" for row in rows], dtype=np.int64)
    evaluation = evaluate_thresholds(labels, scores, prevalence=0.001)
    if threshold is None:
        selected = max(
            (point for point in evaluation.threshold_table if point.fpr <= max_fpr),
            key=lambda point: (point.recall, -point.fpr, point.threshold),
        )
        applied_threshold = selected.threshold
    else:
        applied_threshold = threshold
    predicted = scores >= applied_threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    positives = int(positive.sum())
    negatives = len(rows) - positives
    by_operator: dict[str, dict[str, int | float | None]] = {}
    for operator in sorted(OPERATORS):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["decision"] == "BLOCK" and row["operator"] == operator
        ]
        detected = int(predicted[indices].sum())
        by_operator[operator] = {
            "count": len(indices),
            "tp": detected,
            "fn": len(indices) - detected,
            "recall": detected / len(indices) if indices else None,
        }
    return {
        "threshold": applied_threshold,
        "threshold_source": "explicit" if threshold is not None else "selected_on_diagnostic_set",
        "max_fpr": max_fpr,
        "overall": {
            "positive_count": positives,
            "negative_count": negatives,
            "tp": tp,
            "fp": fp,
            "tn": negatives - fp,
            "fn": positives - tp,
            "recall": tp / positives,
            "fpr": fp / negatives,
            "fpr_constraint_met": fp / negatives <= max_fpr,
            "auroc": evaluation.auroc,
            "aucpr": evaluation.aucpr,
        },
        "by_operator": by_operator,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"manifest image/mask path escapes source root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"missing diagnostic image or mask: {relative}")
    return path


def export_error_examples(
    rows: list[dict[str, Any]],
    scores: NDArray[np.float64],
    *,
    threshold: float,
    source_root: Path,
    output_dir: Path,
    examples_per_kind: int,
    max_false_positives: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Copy unchanged error images; FN sampling is seeded and FP ranking is by score."""
    _validate_rows(rows)
    if scores.shape != (len(rows),) or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite and aligned with manifest rows")
    if not np.isfinite(threshold) or min(examples_per_kind, max_false_positives, seed) < 0:
        raise ValueError("invalid example selection configuration")
    rng = np.random.default_rng(seed)
    selected: list[tuple[str, int]] = []
    for operator in sorted(OPERATORS):
        missed = sorted(
            (
                i
                for i, row in enumerate(rows)
                if row["decision"] == "BLOCK"
                and row["operator"] == operator
                and scores[i] < threshold
            ),
            key=lambda index: str(rows[index]["sample_id"]),
        )
        rng.shuffle(missed)
        selected.extend(("FN", i) for i in missed[:examples_per_kind])
    false_positives = sorted(
        (i for i, row in enumerate(rows) if row["decision"] == "PASS" and scores[i] >= threshold),
        key=lambda index: (-float(scores[index]), str(rows[index]["sample_id"])),
    )
    selected.extend(("FP", i) for i in false_positives[:max_false_positives])
    sources = [
        (
            _source_path(source_root, str(rows[i]["image_path"])),
            _source_path(source_root, str(rows[i]["mask_path"])),
        )
        for _, i in selected
    ]
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for ordinal, ((error_type, index), (image_path, mask_path)) in enumerate(
        zip(selected, sources, strict=True), start=1
    ):
        row = rows[index]
        group = f"FN/{row['operator']}" if error_type == "FN" else "FP"
        stem = f"{ordinal:03d}-{hashlib.sha256(str(row['sample_id']).encode()).hexdigest()[:12]}"
        image_relative = Path("examples") / group / f"{stem}.png"
        mask_relative = image_relative.with_name(f"{stem}-mask.png")
        (output_dir / image_relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, output_dir / image_relative)
        shutil.copyfile(mask_path, output_dir / mask_relative)
        records.append(
            {
                "sample_id": row["sample_id"],
                "base_char": row["base_char"],
                "operator": row["operator"],
                "anomaly_kind": row.get("anomaly_kind"),
                "changed_pixels": row.get("changed_pixels"),
                "error_type": error_type,
                "score": float(scores[index]),
                "threshold": threshold,
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "exported_image": image_relative.as_posix(),
                "exported_mask": mask_relative.as_posix(),
                "mask_semantics": "changed_pixels" if error_type == "FN" else "foreground",
                "source_image_sha256": _sha256(image_path),
                "source_mask_sha256": _sha256(mask_path),
            }
        )
    (examples_dir / "index.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    index_lines = [
        "# 错例人工抽查",
        "",
        "原图未修改；来源字只表示合成起点，不是对当前图中文字的人工确认。",
        "",
        "FN 掩码白色表示变更像素；FP 掩码白色表示正常字前景。",
        "",
        "| 类型 | 算子 | 来源字 | 分数 | 原图 | 掩码 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        character = str(record["base_char"]).replace("|", "\\|").replace("\n", " ")
        image_link = Path(record["exported_image"]).relative_to("examples").as_posix()
        mask_link = Path(record["exported_mask"]).relative_to("examples").as_posix()
        index_lines.append(
            f"| {record['error_type']} | {record['operator']} | {character} | "
            f"{record['score']:.8g} | [查看]({image_link}) | [查看]({mask_link}) |"
        )
    (examples_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return records


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _infer_scores(
    model: GlyphClassifier,
    bank: PrototypeBank,
    dataset: GlyphDataset,
    config: DiagnosticConfig,
    progress: Callable[[str], None],
) -> tuple[NDArray[np.float64], list[str]]:
    blocks: list[NDArray[np.float32]] = []
    last_log = 0.0
    with torch.inference_mode():
        for start in range(0, len(dataset), config.batch_size):
            stop = min(start + config.batch_size, len(dataset))
            views = torch.stack([dataset[i].views for i in range(start, stop)]).to(config.device)
            embeddings, _ = model(views)
            blocks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            now = time.monotonic()
            if now - last_log >= 5 or stop == len(dataset):
                progress(f"inference={stop}/{len(dataset)}")
                last_log = now
    values = np.concatenate(blocks, axis=0)
    distance_blocks: list[NDArray[np.float64]] = []
    nearest_chars: list[str] = []
    # Bound the large prototype-distance matrix independently of GPU inference batches.
    for start in range(0, len(dataset), 4096):
        stop = min(start + 4096, len(dataset))
        scored = bank.score(values[start:stop])
        distance_blocks.append(scored.nearest_distance.astype(np.float64))
        nearest_chars.extend(scored.nearest_chars)
        progress(f"prototype_scoring={stop}/{len(dataset)}")
    return np.concatenate(distance_blocks), nearest_chars


_OPERATOR_LABELS = {
    "erase_segment": "缺笔",
    "add_stroke": "添笔",
    "break_stroke": "断笔",
    "bridge": "粘连",
    "component_shift": "部件位移",
}


def _render_diagnostics(report: dict[str, Any]) -> str:
    summary = report["summary"]
    overall = summary["overall"]
    pair = report["pair_coverage"]
    lines = [
        "# 字形诊断报告",
        "",
        "## Not a production claim",
        "",
        "仅用于合成数据错误分析。诊断集选出的阈值和择优结果不是独立验收或上线证明。",
        "",
        "## 总体结果",
        "",
        f"- 阈值：{summary['threshold']:.17g}（{summary['threshold_source']}）",
        f"- 误报率上限：{summary['max_fpr']:.6%}",
        f"- Recall：{overall['recall']:.6%}；FPR：{overall['fpr']:.6%}",
        f"- TP={overall['tp']} FP={overall['fp']} TN={overall['tn']} FN={overall['fn']}",
        f"- AUROC={overall['auroc']:.6f}；AUCPR={overall['aucpr']:.6f}",
        "",
        "## 按异常类型统计",
        "",
        "以下所有类型共用上述阈值，不分别调阈值。",
        "",
        "| 类型 | 样本数 | 检出 TP | 漏检 FN | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for operator, item in summary["by_operator"].items():
        recall = "—" if item["recall"] is None else f"{item['recall']:.4%}"
        lines.append(
            f"| {_OPERATOR_LABELS[operator]} ({operator}) | {item['count']} | "
            f"{item['tp']} | {item['fn']} | {recall} |"
        )
    lines.extend(
        [
            "",
            "## 训练采样配对覆盖率",
            "",
            "这是当前代码依据原训练配置的元数据重放，不是历史损失日志，也不能证明学习或坍塌。",
            "",
            f"- 重放步数：{pair['replayed_steps']}；"
            f"与训练记录一致：{pair['expected_steps_matches']}",
            f"- 正常样本抽取次数：{pair['normal_draws']}",
            f"- 有同字正样本的正常样本占比：{pair['eligible_normal_fraction']:.4%}",
            f"- 没有同字正样本对的批次占比：{pair['batches_without_positive_pairs_fraction']:.4%}",
            f"- 无序正样本对累计数：{pair['positive_pairs']}",
            "",
            "## 错例人工抽查",
            "",
            f"导出 {report['examples_count']} 张错例及对应掩码："
            "[查看抽查索引](examples/index.md)。",
            "",
            "FN 按每类固定种子随机抽样；FP 超过导出上限时优先导出分数最高者。",
            "完整逐样本分数保存在 scores.parquet；来源与文件哈希见 examples/index.json。",
            "",
            "请检查：是否仍为合法字、是否变成另一个合法字、修改是否过轻、是否符合业务异常定义。",
            "样本抽查不能直接估算标签错误率；合成来源字不等于人工金标。",
            "",
            "## 限制与提醒",
            "",
            *(f"- {warning}" for warning in report["warnings"]),
        ]
    )
    return "\n".join(lines) + "\n"


def diagnose_glyph(
    config: DiagnosticConfig, *, progress: Callable[[str], None] | None = None
) -> DiagnosticArtifacts:
    """Infer once, diagnose errors, and atomically publish to a new output directory."""
    emit = progress if progress is not None else lambda _message: None
    output = config.output_dir.resolve()
    if config.output_dir.is_symlink() or output.exists():
        raise FileExistsError(f"diagnostic output already exists; choose a new directory: {output}")
    for source_root in (config.manifest.parent, config.train_manifest.parent, config.artifacts_dir):
        if output.is_relative_to(source_root.resolve()):
            raise ValueError("diagnostic output must be outside model and dataset directories")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {config.device}")
    paths = {
        "evaluation_manifest": config.manifest,
        "training_manifest": config.train_manifest,
        "checkpoint": config.artifacts_dir / "encoder.pt",
        "prototypes": config.artifacts_dir / "prototypes.npz",
        "prototype_metadata": config.artifacts_dir / "prototypes.json",
        "training_metrics": config.artifacts_dir / "metrics.json",
    }
    emit("Validating model, manifests and training provenance...")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    metrics = _read_json(paths["training_metrics"])
    if metrics.get("manifest_sha256") != hashes["training_manifest"]:
        raise ValueError("training manifest does not match the saved training metrics")
    expected_steps = metrics.get("steps")
    if type(expected_steps) is not int or expected_steps < 1:
        raise ValueError("training metrics must contain a positive integer steps count")
    saved = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    training_config = TrainConfig.model_validate(saved["config"])
    bank, metadata = PrototypeBank.load(paths["prototypes"])
    _validate_prototype_metadata(metadata, paths["checkpoint"], saved["char_to_id"])
    if metadata.get("source_manifest_sha256") != hashes["training_manifest"]:
        raise ValueError("training manifest does not match prototype provenance")
    dataset = GlyphDataset(config.manifest)
    training_dataset = GlyphDataset(config.train_manifest)
    for data in (dataset, training_dataset):
        _validate_rows(data.rows)
        if data.char_to_id != saved["char_to_id"]:
            raise ValueError("manifest character catalog does not match checkpoint")
    if {row["decision"] for row in dataset.rows} != {"PASS", "BLOCK"}:
        raise ValueError("diagnostic manifest must contain both PASS and BLOCK samples")
    # Validate every evaluation path before any pixel is read, even if not exported later.
    for row in dataset.rows:
        _source_path(dataset.root, str(row["image_path"]))
        _source_path(dataset.root, str(row["mask_path"]))
    emit("Replaying training sampler (metadata only; no retraining)...")
    pair = audit_pair_coverage(training_dataset, training_config, expected_steps=expected_steps)
    inference_config = training_config.model_copy(
        update={"pretrained": False, "device": config.device}
    )
    model = GlyphClassifier(len(saved["char_to_id"]), inference_config)
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    emit(f"Starting inference on {len(dataset)} glyphs; no weight download.")
    scores, nearest_chars = _infer_scores(model, bank, dataset, config, emit)
    summary = summarize_scores(
        dataset.rows, scores, threshold=config.threshold, max_fpr=config.max_fpr
    )
    threshold = float(summary["threshold"])
    warnings = [
        "同字体、同合成规则的数据不能证明真实 AIGC 业务效果。",
        "低误报尾部样本有限；经验 FPR 不等于真实 FPR 的统计保证。",
        "配对覆盖率使用当前代码和 PyTorch 重放；跨版本可能与原训练不同。",
        *cast(list[str], pair["warnings"]),
    ]
    if config.threshold is None:
        warnings.append("阈值在当前诊断集上选择；该集合已作为开发诊断集，不是独立验收集。")
    if not summary["overall"]["fpr_constraint_met"]:
        warnings.append("显式阈值未满足配置的 FPR 上限；报告仍保留真实结果。")
    if hashes["evaluation_manifest"] == hashes["training_manifest"]:
        warnings.append("诊断直接复用了训练清单，结果仅用于工程检查。")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        scored_rows = []
        for row, score, nearest in zip(dataset.rows, scores, nearest_chars, strict=True):
            predicted = "BLOCK" if score >= threshold else "PASS"
            positive = row["decision"] == "BLOCK"
            error_type = (
                ("TP" if positive else "FP")
                if predicted == "BLOCK"
                else ("FN" if positive else "TN")
            )
            scored_rows.append(
                {
                    **row,
                    "prototype_distance": float(score),
                    "nearest_char": nearest,
                    "predicted_decision": predicted,
                    "error_type": error_type,
                }
            )
        pq.write_table(
            pa.Table.from_pylist(scored_rows), staging / "scores.parquet", compression="zstd"
        )
        emit("Exporting unchanged error images and diagnostic report...")
        examples = export_error_examples(
            dataset.rows,
            scores,
            threshold=threshold,
            source_root=dataset.root,
            output_dir=staging,
            examples_per_kind=config.examples_per_kind,
            max_false_positives=config.max_false_positives,
            seed=config.seed,
        )
        report = {
            "claim_scope": "synthetic_diagnostic_only",
            "summary": summary,
            "pair_coverage": pair,
            "examples_count": len(examples),
            "warnings": warnings,
            "provenance": {
                "sha256": hashes,
                "paths": {k: str(v) for k, v in paths.items()},
                "training_git_commit": metrics.get("git_commit"),
                "torch_version": str(torch.__version__),
                "scores_sha256": _sha256(staging / "scores.parquet"),
            },
            "config": config.model_dump(mode="json"),
            "training_config": training_config.model_dump(mode="json"),
        }
        (staging / "diagnostics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "diagnostics.md").write_text(_render_diagnostics(report), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"diagnostic output was created concurrently: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    emit("Diagnostics complete.")
    return DiagnosticArtifacts(
        json_path=output / "diagnostics.json",
        markdown_path=output / "diagnostics.md",
        scores_path=output / "scores.parquet",
        examples_path=output / "examples/index.md",
    )
