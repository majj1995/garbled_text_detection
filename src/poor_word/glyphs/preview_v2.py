"""Isolated, unlabelled V2 candidate previews; never a training manifest."""

# Chinese punctuation is intentional in the human-facing review pages.
# ruff: noqa: RUF001

import hashlib
import html
import json
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pillow_version
from pydantic import BaseModel, ConfigDict, Field, model_validator

from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable
from poor_word.glyphs.render import render_glyph


class PreviewConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    characters: tuple[str, ...]
    font_paths: tuple[Path | None, ...]
    per_operator: int = Field(default=10, ge=1, le=100)
    max_attempts_per_slot: int = Field(default=64, ge=1, le=1000)
    seed: int = Field(default=20260910, ge=0)

    @model_validator(mode="after")
    def valid_pools(self) -> "PreviewConfig":
        if not self.characters or any(len(c) != 1 or c.isspace() for c in self.characters):
            raise ValueError("characters must contain non-whitespace Unicode characters")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("characters must not contain duplicates")
        fonts = tuple(path.resolve() if path is not None else None for path in self.font_paths)
        if not fonts or len(set(fonts)) != len(fonts):
            raise ValueError("font_paths must be non-empty and distinct")
        return self


@dataclass(frozen=True)
class PreviewSlot:
    operator: str
    severity: Literal["medium", "strong"]
    font_index: int


def preview_slots(config: PreviewConfig) -> tuple[PreviewSlot, ...]:
    """Plan quotas without consulting a classifier or cherry-picking its errors."""
    return tuple(
        PreviewSlot(
            operator=operator,
            severity="medium" if index % 2 == 0 else "strong",
            font_index=(index // 2 + operator_index) % len(config.font_paths),
        )
        for operator_index, operator in enumerate(sorted(OPERATORS))
        for index in range(config.per_operator)
    )


def save_preview_images(
    directory: Path,
    prefix: str,
    original: NDArray[np.uint8],
    candidate: NDArray[np.uint8],
) -> dict[str, dict[str, str]]:
    """Export RGB-exact differences and the current trainer's actual 96px input views."""
    if re.fullmatch(r"[a-z0-9-]+", prefix) is None:
        raise ValueError("preview prefix must be a plain lowercase identifier")
    if (
        original.dtype != np.uint8
        or candidate.dtype != np.uint8
        or original.shape != candidate.shape
        or original.ndim != 3
        or original.shape[2] != 3
        or min(original.shape[:2]) < 1
    ):
        raise ValueError("preview images must be matching, non-empty RGB uint8 arrays")
    original_96 = cv2.resize(
        cv2.cvtColor(original, cv2.COLOR_RGB2GRAY),
        (96, 96),
        interpolation=cv2.INTER_AREA,
    )
    candidate_96 = cv2.resize(
        cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY),
        (96, 96),
        interpolation=cv2.INTER_AREA,
    )
    arrays = {
        "original": original,
        "candidate": candidate,
        "mask": np.any(original != candidate, axis=2).astype(np.uint8) * 255,
        "original_96": original_96,
        "candidate_96": candidate_96,
        "mask_96": (original_96 != candidate_96).astype(np.uint8) * 255,
        "foreground_96": (candidate_96 > 8).astype(np.uint8) * 255,
        "edges_96": cv2.Canny(candidate_96, threshold1=50, threshold2=150),
    }
    paths = {key: f"images/{prefix}-{key}.png" for key in arrays}
    for relative in paths.values():
        if (directory / relative).exists():
            raise FileExistsError(f"preview asset already exists: {relative}")
    (directory / "images").mkdir(parents=True, exist_ok=True)
    exported: dict[str, dict[str, str]] = {}
    for key, array in arrays.items():
        path = directory / paths[key]
        with path.open("xb") as stream:
            Image.fromarray(array).save(stream, format="PNG")
        exported[key] = {
            "path": paths[key],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return exported


@dataclass(frozen=True)
class PreviewArtifacts:
    candidates_path: Path
    html_path: Path
    overview_path: Path
    candidate_count: int
    complete: bool


def _font_metadata(paths: tuple[Path | None, ...]) -> list[dict[str, Any]]:
    fonts: list[dict[str, Any]] = []
    for path in paths:
        if path is None:
            fonts.append({"name": "Pillow builtin (smoke only)", "path": None, "sha256": None})
            continue
        if not path.is_file():
            raise FileNotFoundError(f"preview font is missing: {path}; fetch it or use --font")
        try:
            font = ImageFont.truetype(str(path), 96)
        except OSError as error:
            raise ValueError(f"invalid preview font: {path}") from error
        fonts.append(
            {
                "name": " ".join(part for part in font.getname() if part) or path.stem,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    hashes = [info["sha256"] for info in fonts]
    if len(set(hashes)) != len(hashes):
        raise ValueError("font files must have distinct content, not just different filenames")
    return fonts


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _overview(directory: Path, rows: list[dict[str, Any]]) -> None:
    columns, cell_width, cell_height = 5, 116, 124
    canvas = Image.new(
        "RGB", (columns * cell_width, max(1, (len(rows) + 4) // 5) * cell_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        with Image.open(directory / row["files"]["candidate_96"]["path"]) as glyph:
            canvas.paste(glyph.convert("RGB"), (x + 10, y + 4))
        draw.text((x + 6, y + 103), row["candidate_id"][:13], fill="black")
    canvas.save(directory / "overview.png")


_OPERATOR_LABELS = {
    "add_stroke": "添笔",
    "break_stroke": "断笔",
    "erase_segment": "缺笔",
    "bridge": "粘连",
    "component_shift": "位移",
}


def _preview_html(rows: list[dict[str, Any]], *, complete: bool) -> str:
    cards = []
    for row in rows:
        identifier = html.escape(row["candidate_id"])
        files = row["files"]
        figures = []
        for key, label in (
            ("original", "原字 · 128px"),
            ("candidate", "修改字 · 128px"),
            ("mask", "实际 RGB 变化 mask"),
            ("original_96", "原字 · 输入尺寸"),
            ("mask_96", "96px 灰度变化 mask"),
            ("foreground_96", "模型前景通道"),
            ("edges_96", "模型边缘通道"),
        ):
            figures.append(
                f'<figure><img src="{html.escape(files[key]["path"], quote=True)}" '
                f'alt="{label}"><figcaption>{label}</figcaption></figure>'
            )
        description = html.escape(
            f"来源字：{row['base_char']}；{_OPERATOR_LABELS[row['operator']]}；"
            f"强度：{row['severity']}；字体：{row['font']['name']}"
        )
        cards.append(
            f'<article><h2>{identifier}</h2><img class="actual" '
            f'src="{html.escape(files["candidate_96"]["path"], quote=True)}" '
            'width="96" height="96" alt="待判断字形，模型实际输入尺寸">'
            '<p class="prompt">先独立判断：明显结构异常 / 合法可接受 / 不确定或不可读。</p>'
            f"<details><summary>判断后再展开原字、操作与 mask</summary><p>{description}</p>"
            f'<div class="views">{"".join(figures)}</div>'
            f"<pre>{html.escape(_json(row['metrics']))}</pre></details></article>"
        )
    state = "配额完整" if complete else "配额未填满，请检查 run.json 中的跳过原因"
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>字形 V2 候选预览</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;"
        "padding:0 20px;background:#f6f7f8;color:#202428}"
        "h1{font-size:26px}h2{font:15px monospace}"
        "article{background:white;border:1px solid #d8dde2;"
        "border-radius:8px;padding:20px;margin:20px 0}"
        ".actual{display:block;width:96px;height:96px}summary{cursor:pointer;color:#174e83}"
        ".views{display:flex;flex-wrap:wrap;gap:16px}"
        "figure{margin:8px 0;min-width:128px}figcaption{font-size:13px}"
        "figure img{max-width:none}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.prompt{color:#4b5560}"
        "</style></head><body><h1>字形 V2 · 人工校准预览</h1>"
        f"<p>{len(rows)} 个候选；{state}。全部为 REVIEW，未自动认定为非法汉字。</p>"
        "<p>请先看 96×96 字形，再展开原字和 mask；浏览器缩放设为 100%。来源字只是合成起点，"
        "变成另一个合法字不属于乱码。不可读和不确定项保留 REVIEW。</p>"
        "<p>mask 的白色只表示改动位置，不是“非法字”的证据。当前是纯字形校准，不含复杂广告背景。</p>"
        '<p><a href="overview.png">查看全部候选缩略图</a> · '
        '<a href="review-template.jsonl" download>审阅记录模板</a> · '
        '<a href="run.json">生成配置及跳过统计</a></p>' + "".join(cards) + "</body></html>\n"
    )


def generate_preview(
    config: PreviewConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> PreviewArtifacts:
    """Generate bounded, unreviewed candidates into a fresh, atomically published directory."""
    emit = progress if progress is not None else lambda _message: None
    output = config.output_dir.resolve()
    if output.exists() or config.output_dir.is_symlink():
        raise FileExistsError(f"preview output already exists; choose a new directory: {output}")
    emit("Validating preview fonts; no model, OCR or GPU required...")
    fonts = _font_metadata(config.font_paths)
    # Keep unreviewed candidates in a separate workflow, never train_glyph/generate_dataset.
    from poor_word.glyphs import corrupt_v2

    slots = preview_slots(config)
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempted_count = 0
    last_log = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for slot_index, slot in enumerate(slots):
            font_path = config.font_paths[slot.font_index]
            for _attempt in range(config.max_attempts_per_slot):
                character = config.characters[int(rng.integers(len(config.characters)))]
                sample_seed = int(rng.integers(0, 2**63 - 1))
                attempted_count += 1
                reason: str | None = None
                try:
                    try:
                        original = render_glyph(character, font_path, sample_seed)
                    except ValueError as error:
                        raise CorruptionNotApplicable("font_blank_or_missing_glyph") from error
                    try:
                        missing = render_glyph("\U0010ffff", font_path, sample_seed)
                    except ValueError:
                        missing = None
                    if missing is not None and np.array_equal(original.image, missing.image):
                        raise CorruptionNotApplicable("font_missing_glyph")
                    result = corrupt_v2.corrupt_glyph_v2(
                        original,
                        slot.operator,
                        sample_seed,
                        slot.severity,
                    )
                    actual_changed = np.any(original.image != result.image, axis=2)
                    if not np.array_equal(actual_changed, result.changed_mask.astype(bool)):
                        raise ValueError("V2 changed mask does not match the actual image")
                    if int(actual_changed.sum()) != result.changed_pixels:
                        raise ValueError("V2 changed pixel count does not match the actual image")
                    fingerprint = hashlib.sha256(result.image.tobytes()).hexdigest()
                    if fingerprint in seen:
                        raise CorruptionNotApplicable("duplicate_candidate_pixels")
                except CorruptionNotApplicable as error:
                    reason = str(error)
                if reason is not None:
                    skipped.append(
                        {
                            "slot": slot_index,
                            "operator": slot.operator,
                            "severity": slot.severity,
                            "font_index": slot.font_index,
                            "base_char": character,
                            "seed": sample_seed,
                            "reason": reason,
                        }
                    )
                    now = time.monotonic()
                    if now - last_log >= 5:
                        emit(
                            f"preview={len(records)}/{len(slots)} "
                            f"attempts={attempted_count} skipped={len(skipped)}"
                        )
                        last_log = now
                    continue
                identity = _json(
                    [
                        "stroke-aware-v2",
                        character,
                        fonts[slot.font_index]["sha256"] or fonts[slot.font_index]["name"],
                        slot.operator,
                        slot.severity,
                        sample_seed,
                        fingerprint,
                    ]
                )
                identifier = "p" + hashlib.sha256(identity.encode()).hexdigest()[:12]
                files = save_preview_images(staging, identifier, original.image, result.image)
                seen.add(fingerprint)
                records.append(
                    {
                        "candidate_id": identifier,
                        "generator_version": "stroke-aware-v2",
                        "decision": "REVIEW",
                        "training_eligible": False,
                        "label_provenance": "synthetic_candidate_unreviewed",
                        "base_char": character,
                        "operator": slot.operator,
                        "severity": slot.severity,
                        "font_index": slot.font_index,
                        "font": fonts[slot.font_index],
                        "seed": sample_seed,
                        "changed_pixels": result.changed_pixels,
                        "metrics": result.metrics,
                        "files": files,
                    }
                )
                emit(
                    f"preview={len(records)}/{len(slots)} "
                    f"attempts={attempted_count} skipped={len(skipped)}"
                )
                last_log = time.monotonic()
                break
        complete = len(records) == len(slots)
        order = np.random.default_rng(config.seed + 1).permutation(len(records)).tolist()
        display_rows = [records[index] for index in order]
        (staging / "candidates.jsonl").write_text(
            "".join(_json(row) + "\n" for row in records),
            encoding="utf-8",
        )
        (staging / "review-template.jsonl").write_text(
            "".join(
                _json(
                    {
                        "candidate_id": row["candidate_id"],
                        "decision": "REVIEW",
                        "confirmed_char": None,
                        "annotator_id": "",
                        "reason": "",
                        "candidate_sha256": row["files"]["candidate"]["sha256"],
                    }
                )
                + "\n"
                for row in display_rows
            ),
            encoding="utf-8",
        )
        run = {
            "purpose": "synthetic_candidate_calibration_not_training_or_acceptance",
            "complete": complete,
            "expected_count": len(slots),
            "candidate_count": len(records),
            "missing_count": len(slots) - len(records),
            "attempted_count": attempted_count,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "by_operator": {
                op: sum(row["operator"] == op for row in records) for op in sorted(OPERATORS)
            },
            "by_severity": dict(Counter(row["severity"] for row in records)),
            "by_font_index": dict(Counter(str(row["font_index"]) for row in records)),
            "config": config.model_dump(mode="json"),
            "fonts": fonts,
            "provenance": {
                "generator_code_sha256": hashlib.sha256(
                    Path(corrupt_v2.__file__).read_bytes()
                ).hexdigest(),
                "preview_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pillow_version": pillow_version,
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
            },
            "warnings": [
                "All candidates remain REVIEW. "
                "Geometric checks do not establish character illegality.",
                "This small preview is for development calibration, "
                "not independent acceptance or production FPR estimation.",
                "No complex background augmentation in this calibration run.",
                *(
                    ["Only one font was supplied; cross-font diversity is not covered."]
                    if len(fonts) < 2
                    else []
                ),
            ],
        }
        (staging / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _overview(staging, display_rows)
        (staging / "index.html").write_text(
            _preview_html(display_rows, complete=complete), encoding="utf-8"
        )
        (staging / "index.md").write_text(
            "# 字形 V2 人工校准预览\n\n"
            f"候选 {len(records)}/{len(slots)}；全部为 REVIEW，不是训练集。\n\n"
            "用浏览器打开同目录的 [index.html](index.html)，"
            "先判断 96×96 字形，再展开原字和 mask。\n\n"
            "[总览](overview.png) · [审阅模板](review-template.jsonl) · "
            "[配置及跳过统计](run.json)\n\n"
            "明显结构异常可记录 BLOCK；合法可接受记 PASS；不确定或不可读保留 REVIEW。"
            "若变成另一个合法字，记录 confirmed_char，不能继续把来源字当识字标签。\n\n"
            "此处不执行标签导入或训练。背景干扰和真实业务效果需要后续单独验证。\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"preview output was created concurrently: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    emit(
        f"Preview {'complete' if complete else 'incomplete'}: "
        f"{len(records)}/{len(slots)} candidates; all REVIEW."
    )
    return PreviewArtifacts(
        candidates_path=output / "candidates.jsonl",
        html_path=output / "index.html",
        overview_path=output / "overview.png",
        candidate_count=len(records),
        complete=complete,
    )
