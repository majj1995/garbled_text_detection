"""Native stroke-layer candidates for human calibration, never training labels."""

# Chinese punctuation is intentional in the offline review page.
# ruff: noqa: RUF001

import hashlib
import html
import inspect
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, PngImagePlugin
from pydantic import BaseModel, ConfigDict, Field, model_validator

from poor_word.data.manifest import load_source_lock
from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable
from poor_word.glyphs.preview_v2 import PreviewArtifacts, _overview, save_preview_images

_ARPHIC_SHA256 = "3a5e90c0957524a89e48203febcd4492ca4393678abaa7e5b4d70f3ff32b386d"
_VERSION = "stroke-layer-v3"
_BRIDGE_LABELS = {"close_opening": "错误封口", "block_gap": "间隙堵塞"}
_LABELS = {
    "add_stroke": "多笔添加",
    "erase_segment": "整笔删除",
    "break_stroke": "单笔内部断裂",
    "component_shift": "重叠位移",
    "bridge": "粘连",
}


class StrokePreviewConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    graphics_path: Path
    source_lock_path: Path
    license_path: Path
    characters: tuple[str, ...]
    per_operator: int = Field(default=10, ge=1, le=100)
    bridges_only: bool = False  # In this mode per_operator is the quota for each subtype.
    breaks_only: bool = False
    max_attempts_per_slot: int = Field(default=48, ge=1, le=1000)
    seed: int = Field(default=20260910, ge=0)

    @model_validator(mode="after")
    def valid_characters(self) -> "StrokePreviewConfig":
        if self.bridges_only and self.breaks_only:
            raise ValueError("bridges_only and breaks_only are mutually exclusive")
        if not self.characters or any(len(c) != 1 or c.isspace() for c in self.characters):
            raise ValueError("characters must contain single non-whitespace characters")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("characters must be distinct")
        return self


def save_stroke_assets(
    directory: Path,
    prefix: str,
    original: NDArray[np.uint8],
    candidate: NDArray[np.uint8],
    selected: NDArray[np.uint8],
    notice: str,
) -> dict[str, dict[str, str]]:
    """Keep the V2 input views, plus a whole-stroke audit view and PNG license notices."""
    if selected.dtype != np.uint8 or selected.shape != original.shape[:2]:
        raise ValueError("selected strokes must be uint8 alpha matching the original canvas")
    if not notice.strip():
        raise ValueError("a dated modification notice is required")
    selected_path = directory / f"images/{prefix}-selected_strokes.png"
    if selected_path.exists():
        raise FileExistsError(f"preview asset already exists: {selected_path}")
    files = save_preview_images(directory, prefix, original, candidate)
    with selected_path.open("xb") as stream:
        Image.fromarray(selected).save(stream, format="PNG")
    files["selected_strokes"] = {"path": f"images/{prefix}-selected_strokes.png", "sha256": ""}
    for info in files.values():
        path = directory / info["path"]
        _stamp_png(path, notice)
        info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def _stamp_png(path: Path, notice: str) -> None:
    with Image.open(path) as existing:
        pixels = existing.copy()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Copyright", "Copyright (C) 1999 Arphic Technology Co., Ltd.")
    metadata.add_text("License", "Arphic-1999; see ARPHICPL.txt")
    metadata.add_text("Modification", notice)
    pixels.save(path, format="PNG", pnginfo=metadata)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _stroke_html(rows: list[dict[str, Any]], complete: bool, date: str) -> str:
    cards = []
    views = (
        ("original", "原字 128px"),
        ("candidate", "修改字 128px"),
        ("selected_strokes", "操作涉及的原始完整笔画"),
        ("mask", "实际变化 mask"),
        ("original_96", "原字 96px"),
        ("mask_96", "96px 变化 mask"),
        ("foreground_96", "模型前景通道"),
        ("edges_96", "模型边缘通道"),
    )
    for row in rows:
        files = row["files"]
        figures = "".join(
            f'<figure><img src="{html.escape(files[key]["path"], quote=True)}" '
            f'alt="{label}"><figcaption>{label}</figcaption></figure>'
            for key, label in views
        )
        description = html.escape(
            f"来源字：{row['base_char']}；"
            f"{_BRIDGE_LABELS.get(str(row.get('bridge_mode') or ''), _LABELS[row['operator']])}；"
            f"原笔画序号（从 1 开始）：{[i + 1 for i in row['selected_stroke_indices']]}"
        )
        cards.append(
            f"<article><h2>{html.escape(row['candidate_id'])}</h2>"
            f'<img src="{html.escape(files["candidate_96"]["path"], quote=True)}" '
            'width="96" height="96" alt="待复核字形 96px">'
            "<p>先判断：明显结构异常 / 合法可接受 / 不确定或不可读。</p>"
            f"<details><summary>判断后展开原字、完整笔画和 mask</summary><p>{description}</p>"
            f'<div class="views">{figures}</div>'
            f"<pre>{html.escape(_json(row['metrics']))}</pre>"
            f'<a href="{html.escape(row["stroke_archive"]["path"], quote=True)}">'
            "下载操作前后笔画层（NPZ）</a></details></article>"
        )
    status = "配额完整" if complete else "配额未满，请查看 run.json 的跳过原因"
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>笔画级字形候选预览</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;"
        "padding:0 20px;background:#f6f7f8;color:#202428}h1{font-size:26px}"
        "h2{font:15px monospace}article{background:white;border:1px solid #d8dde2;"
        "border-radius:8px;padding:20px;margin:20px 0}summary{cursor:pointer;color:#174e83}"
        ".views{display:flex;flex-wrap:wrap;gap:16px}figure{margin:8px 0;min-width:128px}"
        "figcaption{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"
        "</style></head><body><h1>笔画级 · 人工校准预览</h1>"
        f"<p>{len(rows)} 个候选；{status}；全部为 REVIEW，不是异常训练集。</p>"
        "<p>先独立看 96×96 修改字，再展开对照；浏览器缩放设为 100%。"
        "还能猜出原字不等于字形合法；变成另一个合法字也不属于乱码。不确定项保留 REVIEW。</p>"
        "<p>使用 Make Me a Hanzi 原生笔画轮廓（Arphic 来源），不是 Noto 跨字体映射。"
        "当前不含复杂广告背景，几何检查不能证明汉字非法。</p>"
        "<p>粘连候选分为错误封口和间隙堵塞：前者封闭原先通向外部的空白区域，"
        "后者吞并原图选定的局部笔画间隙。类型和几何证据只在展开后展示，"
        "是否破坏汉字结构仍由人工判断。</p>"
        "<p>断笔针对同一笔画的内部：要求原先连续的笔画形成两段有面积和长度的部分，"
        "并在实际 96px 输入中保留清晰断口。仅截短笔端、分离两笔接头或留下微小碎点不算。"
        "这些检查仍不能代替人工判断。</p>"
        '<p><a href="overview.png">全部候选总览</a> · '
        '<a href="review-template.jsonl" download>审阅模板</a> · '
        '<a href="run.json">配置与跳过统计</a> · '
        '<a href="source.json">来源锁定信息</a> · <a href="ARPHICPL.txt">APL 许可全文</a></p>'
        f"<p>Copyright (C) 1999 Arphic Technology Co., Ltd. "
        f"{html.escape(date)}：通过笔画级操作和栅格化产生本预览。"
        "图形衍生物按 Arphic-1999 提供，可依该许可复制和修改，不提供担保。</p>"
        + "".join(cards)
        + "</body></html>\n"
    )


def generate_stroke_preview(
    config: StrokePreviewConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> PreviewArtifacts:
    """Generate a fresh, licensed, review-only native-stroke preview, with no fallback."""
    emit = progress if progress is not None else lambda _message: None
    output = config.output_dir.resolve()
    if output.exists() or config.output_dir.is_symlink():
        raise FileExistsError(f"preview output already exists; use a new directory: {output}")
    emit("Checking locked stroke source and license; no model, OCR or GPU required...")
    lock = load_source_lock(config.source_lock_path)
    source_bytes = config.graphics_path.read_bytes()
    if (
        lock.source_id != "makemeahanzi_graphics"
        or lock.license_id != "Arphic-1999"
        or hashlib.sha256(source_bytes).hexdigest() != lock.sha256
        or len(source_bytes) != lock.size_bytes
    ):
        raise ValueError("stroke source does not match the source lock")
    license_bytes = config.license_path.read_bytes()
    if hashlib.sha256(license_bytes).hexdigest() != _ARPHIC_SHA256:
        raise ValueError("stroke source requires the unmodified Arphic license")
    from poor_word.glyphs import stroke_break, stroke_bridge, stroke_corrupt, stroke_source

    records = stroke_source.load_stroke_records(
        config.graphics_path,
        config.characters,
        progress=lambda count: emit(f"validated_stroke_records={count}"),
    )
    emit(f"Usable characters={len(records)}; building review candidates...")
    slots: list[tuple[str, str | None]]
    if config.bridges_only:
        slots = [("bridge", mode) for mode in _BRIDGE_LABELS for _ in range(config.per_operator)]
    elif config.breaks_only:
        slots = [("break_stroke", None) for _ in range(config.per_operator)]
    else:
        slots = [
            (operator, None) for operator in sorted(OPERATORS) for _ in range(config.per_operator)
        ]
    rng = np.random.default_rng(config.seed)
    date = datetime.now(UTC).date().isoformat()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    attempted = 0
    seen: set[str] = set()
    last_log = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "ARPHICPL.txt").write_bytes(license_bytes)
        (staging / "source.json").write_text(
            lock.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (staging / "layers").mkdir()
        for slot, (operator, requested_mode) in enumerate(slots):
            for _attempt in range(config.max_attempts_per_slot):
                character = config.characters[int(rng.integers(len(config.characters)))]
                seed = int(rng.integers(2**63 - 1))
                attempted += 1
                try:
                    layers = stroke_source.render_stroke_layers(records[character])
                    original = np.repeat(np.maximum.reduce(layers)[:, :, None], 3, axis=2)
                    result = stroke_corrupt.corrupt_stroke_layers(
                        layers, operator, seed, bridge_mode=requested_mode
                    )
                    actual_mode = result.bridge_mode
                    if operator == "bridge" and actual_mode not in _BRIDGE_LABELS:
                        raise ValueError("bridge result is missing a supported subtype")
                    if requested_mode is not None and actual_mode != requested_mode:
                        raise ValueError("bridge result does not match the requested subtype")
                    actual = np.any(original != result.image, axis=2)
                    if not np.array_equal(actual, result.changed_mask.astype(bool)):
                        raise ValueError("stroke result has an inconsistent change mask")
                    if int(actual.sum()) != result.changed_pixels:
                        raise ValueError("stroke result has an inconsistent change count")
                    composed = np.repeat(
                        np.maximum.reduce(result.edited_layers)[:, :, None], 3, axis=2
                    )
                    if not np.array_equal(composed, result.image):
                        raise ValueError("edited layers do not reproduce the result image")
                    pixel_hash = hashlib.sha256(result.image.tobytes()).hexdigest()
                    if pixel_hash in seen:
                        raise CorruptionNotApplicable("duplicate_candidate_pixels")
                except CorruptionNotApplicable as error:
                    skipped.append(
                        {
                            "slot": slot,
                            "operator": operator,
                            "bridge_mode": requested_mode,
                            "base_char": character,
                            "seed": seed,
                            "reason": str(error),
                        }
                    )
                    if time.monotonic() - last_log >= 5:
                        emit(
                            f"preview={len(rows)}/{len(slots)} "
                            f"attempts={attempted} skipped={len(skipped)}"
                        )
                        last_log = time.monotonic()
                    continue
                identity = _json(
                    [_VERSION, lock.sha256, character, operator, actual_mode, seed, pixel_hash]
                )
                identifier = "s" + hashlib.sha256(identity.encode()).hexdigest()[:12]
                selected = result.selected_stroke_indices
                selected_mask = np.maximum.reduce([layers[index] for index in selected])
                notice = (
                    f"{date}: normalized and rasterized Make Me a Hanzi outlines; "
                    f"applied {operator} ({actual_mode or 'not a bridge'}), "
                    f"source stroke indices (zero based) {list(selected)}, "
                    f"seed {seed}; see candidates.jsonl for metrics and layer archive."
                )
                files = save_stroke_assets(
                    staging, identifier, original, result.image, selected_mask, notice
                )
                archive = staging / f"layers/{identifier}.npz"
                arrays = {
                    **{f"before_{i:03}": layer for i, layer in enumerate(layers)},
                    **{f"after_{i:03}": layer for i, layer in enumerate(result.edited_layers)},
                }
                np.savez_compressed(
                    archive,
                    **arrays,
                    license=np.array("Arphic-1999"),
                    modification=np.array(notice),
                    source_char=np.array(character),
                    bridge_mode=np.array(actual_mode or ""),
                )
                rows.append(
                    {
                        "candidate_id": identifier,
                        "generator_version": _VERSION,
                        "decision": "REVIEW",
                        "training_eligible": False,
                        "label_provenance": "synthetic_stroke_candidate_unreviewed",
                        "base_char": character,
                        "operator": operator,
                        "bridge_mode": actual_mode,
                        "seed": seed,
                        "source_style": "Make Me a Hanzi (Arphic-derived; not Noto)",
                        "source_sha256": lock.sha256,
                        "original_stroke_count": len(layers),
                        "selected_stroke_indices": list(selected),
                        "changed_pixels": result.changed_pixels,
                        "pixel_sha256": pixel_hash,
                        "metrics": result.metrics,
                        "files": files,
                        "license_id": "Arphic-1999",
                        "modification": notice,
                        "stroke_archive": {
                            "path": f"layers/{identifier}.npz",
                            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        },
                    }
                )
                seen.add(pixel_hash)
                emit(
                    f"preview={len(rows)}/{len(slots)} attempts={attempted} skipped={len(skipped)}"
                )
                last_log = time.monotonic()
                break
        complete = len(rows) == len(slots)
        order = np.random.default_rng(config.seed + 1).permutation(len(rows)).tolist()
        display = [rows[index] for index in order]
        (staging / "candidates.jsonl").write_text(
            "".join(_json(row) + "\n" for row in rows), encoding="utf-8"
        )
        reviews = [
            {
                "candidate_id": row["candidate_id"],
                "decision": "REVIEW",
                "confirmed_char": None,
                "annotator_id": "",
                "reason": "",
                "candidate_sha256": row["files"]["candidate"]["sha256"],
            }
            for row in display
        ]
        (staging / "review-template.jsonl").write_text(
            "".join(_json(row) + "\n" for row in reviews), encoding="utf-8"
        )
        run = {
            "purpose": "native_stroke_calibration_not_training_or_acceptance",
            "generator_version": _VERSION,
            "created_date": date,
            "complete": complete,
            "expected_count": len(slots),
            "candidate_count": len(rows),
            "missing_count": len(slots) - len(rows),
            "attempted_count": attempted,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "by_operator": {
                op: sum(row["operator"] == op for row in rows) for op in sorted(OPERATORS)
            },
            "by_bridge_mode": {
                mode: sum(row["bridge_mode"] == mode for row in rows) for mode in _BRIDGE_LABELS
            },
            "expected_by_bridge_mode": (
                {mode: config.per_operator for mode in _BRIDGE_LABELS}
                if config.bridges_only
                else {}
            ),
            "missing_by_bridge_mode": (
                {
                    mode: config.per_operator - sum(row["bridge_mode"] == mode for row in rows)
                    for mode in _BRIDGE_LABELS
                }
                if config.bridges_only
                else {}
            ),
            "config": config.model_dump(mode="json"),
            "source": lock.model_dump(),
            "provenance": {
                "code_sha256": {
                    Path(inspect.getfile(module)).name: hashlib.sha256(
                        Path(inspect.getfile(module)).read_bytes()
                    ).hexdigest()
                    for module in (stroke_corrupt, stroke_source, stroke_bridge, stroke_break)
                },
                "preview_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_views_code_sha256": hashlib.sha256(
                    Path(inspect.getfile(save_preview_images)).read_bytes()
                ).hexdigest(),
                "numpy_version": np.__version__,
                "packages": {
                    package: version(package)
                    for package in (
                        "numpy",
                        "fonttools",
                        "aggdraw",
                        "pillow",
                        "opencv-python-headless",
                        "scikit-image",
                    )
                },
            },
            "warnings": [
                "All candidates remain REVIEW; geometry does not establish Chinese illegality.",
                "Native source outlines only; no Noto transfer or complex background.",
                "This small calibration set is not a production FPR or model-quality evaluation.",
            ],
        }
        (staging / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _overview(staging, display)
        _stamp_png(
            staging / "overview.png",
            f"{date}: contact sheet of modified native stroke candidates; see candidates.jsonl.",
        )
        (staging / "index.html").write_text(_stroke_html(display, complete, date), encoding="utf-8")
        (staging / "index.md").write_text(
            "# 笔画级人工校准预览\n\n打开 [index.html](index.html)，"
            "先独立判断修改字，再展开原字及完整笔画。\n\n"
            "全部为 REVIEW，不是训练集；明显异常可记 BLOCK，合法可接受记 PASS，"
            "不确定或不可读保持 REVIEW。"
            "若成为另一个合法字，记录 confirmed_char，不能把来源字直接作为识字标签。\n\n"
            "本阶段不导入标签、不训练、不做跨 Noto 字体迁移。\n\n"
            "图形源自 Make Me a Hanzi / Arphic 字体，"
            "衍生图形按 [Arphic-1999](ARPHICPL.txt) 提供，不提供担保。\n",
            encoding="utf-8",
        )
        if output.exists() or config.output_dir.is_symlink():
            raise FileExistsError(f"preview output appeared during generation: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    emit(
        f"Stroke preview {'complete' if complete else 'incomplete'}: "
        f"{len(rows)}/{len(slots)}; all REVIEW."
    )
    return PreviewArtifacts(
        output / "candidates.jsonl",
        output / "index.html",
        output / "overview.png",
        len(rows),
        complete,
    )
