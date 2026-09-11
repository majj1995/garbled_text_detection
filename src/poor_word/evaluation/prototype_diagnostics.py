"""Read-only, paired prototype-compression diagnostics for experimental V2 glyphs."""

# Chinese punctuation is intentional in the user-facing reports below.
# ruff: noqa: RUF001

import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from poor_word.evaluation.glyph_v2 import (
    _reject_overlap,
    _row_identity_sets,
    _sha256,
    _validate_training_artifacts,
)
from poor_word.glyphs.v2_manifest import V2_SCHEMA
from poor_word.models.prototypes import PrototypeBank
from poor_word.training.dataset import GlyphDataset
from poor_word.training.train_glyph import GlyphClassifier, TrainConfig

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class PrototypeDiagnosticArtifacts:
    json_path: Path
    markdown_path: Path
    summary_lines: tuple[str, ...]


def _unit_rows(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings must have non-zero norm")
    return cast(FloatArray, values / norms)


def compare_prototype_distances(
    bank: PrototypeBank,
    queries: FloatArray,
    query_characters: Sequence[str],
    train_embeddings: FloatArray,
    train_characters: Sequence[str],
) -> list[dict[str, Any]]:
    """Compare three cosine distances on each query, without fitting a new bank.

    Same-character references deliberately exclude other legal characters. The global
    score remains the existing bank's score, so the first two columns are not equivalent.
    """
    values = _unit_rows(queries)
    references = _unit_rows(train_embeddings)
    # Read existing centers only; do not fit, normalize, or mutate the saved bank.
    centers = bank._centers
    center_labels = np.asarray(bank._center_labels)
    if (
        centers is None
        or centers.ndim != 2
        or len(centers) == 0
        or len(centers) != len(center_labels)
        or centers.shape[1] != values.shape[1]
        or references.shape[1] != values.shape[1]
        or not np.all(np.isfinite(centers))
        or not np.allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("prototype centers must be finite unit vectors with matching dimensions")
    if len(values) != len(query_characters) or len(references) != len(train_characters):
        raise ValueError("embedding rows must match character labels")
    global_scores = bank.score(queries)
    train_labels = np.asarray(train_characters)
    result: list[dict[str, Any]] = []
    for index, character in enumerate(query_characters):
        own_centers = centers[center_labels == character]
        own_indices = np.flatnonzero(train_labels == character)
        if not len(own_centers) or not len(own_indices):
            raise ValueError(f"missing same-character normal references or prototypes: {character}")
        prototype_distances = 1.0 - values[index] @ own_centers.T
        raw_distances = 1.0 - values[index] @ references[own_indices].T
        nearest = int(np.argmin(raw_distances))
        result.append(
            {
                "global_score": float(global_scores.nearest_distance[index]),
                "global_nearest_char": global_scores.nearest_chars[index],
                "same_char_prototype_distance": float(np.min(prototype_distances)),
                "same_char_train_nn_distance": float(raw_distances[nearest]),
                "nearest_train_index": int(own_indices[nearest]),
                "train_normal_count": len(own_indices),
                "prototype_count": len(own_centers),
            }
        )
    return result


def _embed_selected(
    model: GlyphClassifier,
    dataset: GlyphDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    emit: Callable[[str], None],
) -> FloatArray:
    blocks: list[FloatArray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            views = torch.stack([dataset[index].views for index in batch]).to(device)
            embeddings, _ = model(views)
            blocks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            emit(f"{dataset.split_role}_normal_inference={start + len(batch)}/{len(indices)}")
    return np.concatenate(blocks)


def _summary_line(row: dict[str, Any]) -> str:
    return (
        f"{row['base_char']}: train={row['train_normal_count']} "
        f"proto={row['prototype_count']} cal={row['calibration_normal_count']} "
        f"global={row['global_score']:.6f} "
        f"own_proto={row['same_char_prototype_distance']:.6f} "
        f"own_train_nn={row['same_char_train_nn_distance']:.6f} "
        f"nearest_char={row['global_nearest_char']}"
    )


def _markdown(report: dict[str, Any], summary_lines: tuple[str, ...]) -> str:
    lines = [
        "# V2 正常字原型诊断",
        "",
        "冻结现有编码器，仅以训练集 PASS 样本为参考、校准集 PASS 样本为查询。",
        "不读取测试集，不重新训练、拟合原型或选择阈值。仅限合成实验诊断，不代表生产效果。",
        "",
        "## 每字最高全局分数样本",
        "",
        "每一行的三个距离来自**同一张**图片，不是分别取三个最大值。距离越大表示越不相似。",
        "global：现有全字表原型评分；own_proto：同字聚类原型距离；",
        "own_train_nn：同字原始训练特征最近邻距离。",
        "",
        "```text",
        *summary_lines,
        "```",
        "",
        "## 全部选中正常校准样本",
        "",
        "| 字 | 样本 ID | global | own_proto | own_train_nn | 全局最近字 | 最近训练样本 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["samples"]:
        # JSON contains unescaped identities and full-precision scores for machine use.
        safe = {
            key: str(row[key]).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            for key in ("base_char", "sample_id", "global_nearest_char", "nearest_train_sample_id")
        }
        lines.append(
            f"| {safe['base_char']} | {safe['sample_id']} | {row['global_score']:.6f} | "
            f"{row['same_char_prototype_distance']:.6f} | "
            f"{row['same_char_train_nn_distance']:.6f} | "
            f"{safe['global_nearest_char']} | {safe['nearest_train_sample_id']} |"
        )
    lines.extend(
        [
            "",
            "## 如何解读",
            "",
            "- 同一高分样本的 own_train_nn 若明显低于 own_proto，说明原型压缩可能损失了覆盖；",
            "  还需比较它与 global，不能据此保证整体误报率或召回率改善。",
            "- own_train_nn 仍较高时，应继续排查训练正常变体覆盖与编码器表征；"
            "不是单凭距离证明根因。",
            "- global 可以匹配其他合法字，理论上不高于 own_proto（允许浮点误差）。",
            "- 同字比较使用合成数据已知字标签，是离线诊断，不是线上已知正确字的假设。",
            "- 不删除高分正常样本，不修改标签，不以这批选中样本重新调阈值。",
            "- 摘要每字只显示一张，完整列表用于查看其他变体，尤其同字多个高分的情况。",
            "- provenance 与逐样本图片/训练参考路径见 diagnostics.json；未导出或复制原图。",
        ]
    )
    return "\n".join(lines) + "\n"


def diagnose_prototypes(
    *,
    train_manifest: Path,
    calibration_manifest: Path,
    artifacts_dir: Path,
    output_dir: Path,
    characters: str,
    device_name: str = "cpu",
    batch_size: int = 64,
    allow_experimental: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PrototypeDiagnosticArtifacts:
    """Validate provenance, embed selected normals, then publish to a new directory."""
    if not allow_experimental:
        raise ValueError("V2 diagnostics require --allow-experimental")
    selected = list(dict.fromkeys(char for char in characters if not char.isspace()))
    if not selected:
        raise ValueError("characters must contain at least one character")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = output_dir.resolve()
    if output_dir.is_symlink() or output.exists():
        raise FileExistsError(f"diagnostic output already exists; choose a new directory: {output}")
    for source in (train_manifest.parent, calibration_manifest.parent, artifacts_dir):
        if output.is_relative_to(source.resolve()):
            raise ValueError("diagnostic output must be outside model and dataset directories")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device_name}")
    emit = progress if progress is not None else lambda _message: None
    emit("Validating training/calibration provenance; no test data will be read...")
    train = GlyphDataset(train_manifest)
    calibration = GlyphDataset(calibration_manifest)
    for dataset, role in ((train, "train"), (calibration, "calibration")):
        if dataset.schema_version != V2_SCHEMA or dataset.split_role != role:
            raise ValueError(f"expected a V2 {role} manifest")
    if train.dataset_id != calibration.dataset_id:
        raise ValueError("training and calibration dataset_id differ")
    artifacts = _validate_training_artifacts(artifacts_dir, str(train.dataset_id))
    membership = artifacts["membership"]
    if _sha256(train_manifest) != membership["manifest_sha256"]:
        raise ValueError("training manifest does not match saved training membership")
    actual_members = {
        (row["sample_id"], row["source_group_id"], row["pixel_sha256"]) for row in train.rows
    }
    saved_members = set(
        zip(
            membership["sample_ids"],
            membership["source_group_ids"],
            membership["pixel_sha256"],
            strict=True,
        )
    )
    if actual_members != saved_members:
        raise ValueError("training manifest identities do not match saved membership")
    _reject_overlap(_row_identity_sets(train), _row_identity_sets(calibration), "train/calibration")
    checkpoint = artifacts["checkpoint"]
    catalog = checkpoint.get("char_to_id")
    if train.char_to_id != catalog or calibration.char_to_id != catalog:
        raise ValueError("manifest character catalogs do not match encoder checkpoint")
    train_indices = [
        i
        for i, r in enumerate(train.rows)
        if r["decision"] == "PASS" and r["base_char"] in selected
    ]
    cal_indices = [
        i
        for i, r in enumerate(calibration.rows)
        if r["decision"] == "PASS" and r["base_char"] in selected
    ]
    for char in selected:
        if not any(train.rows[i]["base_char"] == char for i in train_indices) or not any(
            calibration.rows[i]["base_char"] == char for i in cal_indices
        ):
            raise ValueError(f"missing normal train or calibration samples for character: {char}")
    config = TrainConfig.model_validate(
        {
            **checkpoint["config"],
            "pretrained": False,
            "device": device_name,
        }
    )
    model = GlyphClassifier(len(train.char_to_id), config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.requires_grad_(False)
    model.eval()
    bank, _ = PrototypeBank.load(artifacts["paths"]["prototypes"])
    emit(
        f"selected_characters={len(selected)} train_normals={len(train_indices)} "
        f"calibration_normals={len(cal_indices)} device={device}"
    )
    train_embeddings = _embed_selected(model, train, train_indices, device, batch_size, emit)
    cal_embeddings = _embed_selected(model, calibration, cal_indices, device, batch_size, emit)
    comparisons = compare_prototype_distances(
        bank,
        cal_embeddings,
        [str(calibration.rows[i]["base_char"]) for i in cal_indices],
        train_embeddings,
        [str(train.rows[i]["base_char"]) for i in train_indices],
    )
    samples = []
    for index, scores in zip(cal_indices, comparisons, strict=True):
        row = calibration.rows[index]
        reference = train.rows[train_indices[scores.pop("nearest_train_index")]]
        samples.append(
            {
                **scores,
                "sample_id": row["sample_id"],
                "base_char": row["base_char"],
                "image_path": str((calibration.root / row["image_path"]).resolve()),
                "pixel_sha256": row["pixel_sha256"],
                "nearest_train_sample_id": reference["sample_id"],
                "nearest_train_image_path": str((train.root / reference["image_path"]).resolve()),
                "nearest_train_pixel_sha256": reference["pixel_sha256"],
                "calibration_normal_count": sum(
                    calibration.rows[i]["base_char"] == row["base_char"] for i in cal_indices
                ),
            }
        )
    samples.sort(key=lambda row: (selected.index(row["base_char"]), -row["global_score"]))
    summary = [next(row for row in samples if row["base_char"] == char) for char in selected]
    summary_lines = tuple(_summary_line(row) for row in summary)
    input_paths = {
        "training_manifest": train_manifest,
        "calibration_manifest": calibration_manifest,
        "training_run": train_manifest.parent / "run.json",
        "calibration_run": calibration_manifest.parent / "run.json",
        **artifacts["paths"],
    }
    report = {
        "schema_version": "glyph-prototype-diagnostics-v2",
        "dataset_id": train.dataset_id,
        "experimental_only": True,
        "production_allowed": False,
        "test_data_used": False,
        "encoder_frozen": True,
        "threshold_selected": False,
        "prototype_bank_refitted": False,
        "claim_scope": "selected_calibration_normals_only",
        "characters": selected,
        "reference_count": len(train_indices),
        "query_count": len(cal_indices),
        "summary": summary,
        "samples": samples,
        "runtime": {
            "device": str(device),
            "batch_size": batch_size,
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in input_paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging_name:
        staging = Path(staging_name)
        (staging / "diagnostics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "diagnostics.md").write_text(_markdown(report, summary_lines), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"diagnostic output appeared during inference: {output}")
        staging.rename(output)
    return PrototypeDiagnosticArtifacts(
        json_path=output / "diagnostics.json",
        markdown_path=output / "diagnostics.md",
        summary_lines=summary_lines,
    )
