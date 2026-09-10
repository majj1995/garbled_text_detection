"""Reproduce a same-site strength comparison from archived accepted break attempts.

This is an offline calibration script, not the production first-valid sampler.
Earlier proposals are intentionally ignored. Failed original sites stay skipped.
"""

# Chinese punctuation belongs to the human review page.
# ruff: noqa: RUF001

import argparse
import hashlib
import html
import inspect
import json
import shutil
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from poor_word.glyphs import preview_v2, stroke_break, stroke_corrupt, stroke_preview
from poor_word.glyphs.stroke_corrupt import ByteArray, StrokeCorruptionResult, _visible

VERSION = "stroke-break-longer-v1"


def replay_sample(
    layers: tuple[ByteArray, ...],
    row: dict[str, Any],
    *,
    length_multiplier: float = 1.15,
) -> StrokeCorruptionResult | None:
    """Keep the archived attempt, selected stroke and center; apply every current gate."""
    attempts = int(row["metrics"]["attempts"])
    if row["operator"] != "break_stroke" or not 1 <= attempts <= 96:
        raise ValueError("expected an archived accepted break attempt")
    weights = np.asarray([np.count_nonzero(x >= 128) ** 1.5 for x in layers], dtype=float)
    weights /= weights.sum()
    random = np.random.default_rng(row["seed"])
    # Same draw sequence as the original sampler: length changes consume no extra RNG draws.
    proposal = None
    for _ in range(attempts):
        index = int(random.choice(len(layers), p=weights))
        rest = (
            np.maximum.reduce([x for i, x in enumerate(layers) if i != index])
            if len(layers) > 1
            else np.zeros_like(layers[0])
        )
        proposal = stroke_break.propose_break(
            layers[index], rest, random, length_multiplier=length_multiplier
        )
    if [index] != row["selected_stroke_indices"]:
        raise ValueError("sampling sequence changed: selected stroke does not match baseline")
    if proposal is None:
        return None
    edited, metrics = proposal
    for key in ("break_center_x_96", "break_center_y_96", "local_stroke_width"):
        if metrics[key] != row["metrics"][key]:
            raise ValueError(f"sampling sequence changed: {key} does not match baseline")
    if not np.isclose(metrics["gap_length"], row["metrics"]["gap_length"] * length_multiplier):
        raise ValueError("comparison did not preserve the requested cut length ratio")
    after = tuple(edited.copy() if i == index else x.copy() for i, x in enumerate(layers))
    original, candidate = np.maximum.reduce(layers), np.maximum.reduce(after)
    visibility = _visible(original, candidate)
    if visibility is None:
        return None
    changed = (original != candidate).astype(np.uint8)
    points = np.argwhere(original >= 128)
    return StrokeCorruptionResult(
        image=np.repeat(candidate[:, :, None], 3, axis=2),
        changed_mask=changed,
        changed_pixels=int(changed.sum()),
        operator="break_stroke",
        metrics={
            **metrics,
            **visibility,
            "glyph_scale": float(np.ptp(points, axis=0).max() + 1),
            "attempts": float(attempts),
        },
        selected_stroke_indices=(index,),
        edited_layers=after,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked(directory: Path, info: dict[str, str]) -> Path:
    path = (directory / info["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or _sha(path) != info["sha256"]:
        raise ValueError(f"baseline hash/path mismatch: {info['path']}")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _comparison_page(directory: Path, pairs: list[dict[str, Any]], date: str) -> None:
    cards = []
    sheet = Image.new("RGB", (1020, ((len(pairs) + 2) // 3) * 148), "white")
    draw = ImageDraw.Draw(sheet)
    for number, pair in enumerate(pairs, 1):
        x, y = ((number - 1) % 3) * 340, ((number - 1) // 3) * 148
        draw.text((x + 8, y + 5), f"{number:02}   original       old          +15%", fill="black")
        figures = []
        for column, (key, label) in enumerate(
            [("original_96", "原字"), ("old_96", "旧断笔"), ("new_96", "加长15%")]
        ):
            info = pair["files"].get(key)
            if info:
                with Image.open(directory / info["path"]) as image:
                    sheet.paste(image.convert("RGB"), (x + 8 + column * 108, y + 30))
                content = f'<img width="96" height="96" src="{html.escape(info["path"])}">'
            else:
                draw.text((x + 8 + column * 108, y + 66), "SKIP", fill="#8a3030")
                content = '<div class="skip">未通过保护检查<br>保留旧图，不替换</div>'
            figures.append(f"<figure>{content}<figcaption>{label}</figcaption></figure>")
        status = "同一笔画、同一切点" if pair["status"] == "accepted" else "本次跳过增强"
        if pair.get("pixel_changed") is False:
            status = "长度参数已加长，像素不变"
        masks = "".join(
            f'<figure><img width="96" height="96" src="{html.escape(pair["files"][key]["path"])}">'
            f"<figcaption>{label}</figcaption></figure>"
            for key, label in [("old_mask_96", "旧变化 mask"), ("new_mask_96", "新变化 mask")]
            if key in pair["files"]
        )
        cards.append(
            f"<article><h2>{number:02} · {html.escape(pair['base_char'])} · {status}</h2>"
            f'<div class="views">{"".join(figures)}</div>'
            f'<details><summary>变化 mask 与配对记录</summary><div class="views">{masks}</div>'
            f"<pre>{html.escape(json.dumps(pair, ensure_ascii=False, indent=2))}</pre>"
            "</details></article>"
        )
    sheet.save(directory / "overview.png")
    stroke_preview._stamp_png(directory / "overview.png", f"{date}: same-site break comparison.")
    accepted = sum(pair["status"] == "accepted" for pair in pairs)
    changed = sum(pair.get("pixel_changed") is True for pair in pairs)
    (directory / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>断笔切口加长15% · 同位置对照</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:32px auto;"
        "padding:0 16px;color:#202428;background:#f6f7f8}h1{font-size:25px}h2{font-size:18px}"
        "article{padding:18px;background:white;border:1px solid #ddd;margin:16px 0;"
        "border-radius:8px}.views{display:flex;flex-wrap:wrap;gap:20px}figure{margin:8px 0;"
        "text-align:center}figcaption{font-size:14px}.skip{width:96px;height:96px;"
        "font-size:12px;background:#f9eeee;display:grid;place-content:center}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}"
        "</style><h1>断笔切口加长15% · 同位置对照</h1>"
        f"<p>{len(pairs)}个原字：{accepted}张通过检查（{changed}张像素变化，"
        f"{accepted - changed}张像素不变），{len(pairs) - accepted}张跳过；"
        "没有换字、换笔或换切点。</p>"
        "<p>从左到右：原字 / 旧断笔 / 加长后。仅沿笔画方向增加切除长度，横向宽度与保护门槛不变。"
        "15%指连续坐标下的切口长度，不保证像素面积或可见空隙也恰好增加15%。</p>"
        "<p>这是旧样本的配对校准，不是新的独立评测；新候选全部为REVIEW，不自动继承旧样本人工判断。"
        "同位置未通过检查的样本保留空位，不放宽规则。</p>"
        '<p><a href="overview.png">总览图</a> · <a href="candidates.html">先盲看增强后的候选</a> · '
        '<a href="review-template.jsonl">审阅模板</a> · <a href="run.json">运行记录</a> · '
        '<a href="ARPHICPL.txt">Arphic-1999 许可</a></p>'
        f"<p>Copyright (C) 1999 Arphic Technology Co., Ltd. {date}：生成笔画对照；"
        "图形衍生物按Arphic-1999提供，无担保；不含复杂广告背景。</p>"
        + "".join(cards)
        + "</html>\n",
        encoding="utf-8",
    )


def generate_comparison(baseline: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"use a fresh output directory: {output}")
    baseline = baseline.resolve()
    rows = [json.loads(line) for line in (baseline / "candidates.jsonl").read_text().splitlines()]
    for row in rows:
        _checked(baseline, row["stroke_archive"])
        for info in row["files"].values():
            _checked(baseline, info)
    if not rows or len({row["base_char"] for row in rows}) != len(rows):
        raise ValueError("baseline must contain distinct source characters")
    if _sha(baseline / "ARPHICPL.txt") != stroke_preview._ARPHIC_SHA256:
        raise ValueError("baseline license hash mismatch")
    date = datetime.now(UTC).date().isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    candidates, pairs = [], []
    try:
        (staging / "images").mkdir()
        (staging / "layers").mkdir()
        for name in ("ARPHICPL.txt", "source.json"):
            shutil.copyfile(baseline / name, staging / name)
        for row in rows:
            archive = baseline / row["stroke_archive"]["path"]
            with np.load(archive, allow_pickle=False) as data:
                layers = tuple(
                    data[f"before_{i:03}"].copy() for i in range(row["original_stroke_count"])
                )
                old = np.repeat(
                    np.maximum.reduce([data[f"after_{i:03}"] for i in range(len(layers))])[
                        :, :, None
                    ],
                    3,
                    axis=2,
                )
            if hashlib.sha256(old.tobytes()).hexdigest() != row["pixel_sha256"]:
                raise ValueError("baseline recomposed pixel hash mismatch")
            pair = {
                "baseline_candidate_id": row["candidate_id"],
                "base_char": row["base_char"],
                "seed": row["seed"],
                "status": "skipped",
                "candidate_id": None,
                "baseline_pixel_sha256": row["pixel_sha256"],
                "files": {},
                "baseline_metrics": row["metrics"],
                "selected_stroke_indices": row["selected_stroke_indices"],
            }
            for old_key, key in [
                ("original_96", "original_96"),
                ("candidate_96", "old_96"),
                ("mask_96", "old_mask_96"),
            ]:
                info = row["files"][old_key]
                name = f"images/baseline-{row['candidate_id']}-{key}.png"
                shutil.copyfile(baseline / info["path"], staging / name)
                pair["files"][key] = {"path": name, "sha256": _sha(staging / name)}
            result = replay_sample(layers, row)
            if result is None:
                pair["reason"] = "archived_site_failed_unchanged_safety_gates; no replacement"
                pairs.append(pair)
                continue
            digest = hashlib.sha256(result.image.tobytes()).hexdigest()
            identifier = (
                "s"
                + hashlib.sha256(f"{VERSION}:{row['candidate_id']}:{digest}".encode()).hexdigest()[
                    :12
                ]
            )
            notice = (
                f"{date}: extended the archived interior break length by 15%; "
                f"same source stroke and site as {row['candidate_id']}; "
                "all safety gates retained; see pairs.jsonl. No training label."
            )
            selected = result.selected_stroke_indices[0]
            original = np.repeat(np.maximum.reduce(layers)[:, :, None], 3, axis=2)
            files = stroke_preview.save_stroke_assets(
                staging, identifier, original, result.image, layers[selected], notice
            )
            archive_name = f"layers/{identifier}.npz"
            np.savez_compressed(
                staging / archive_name,
                **{f"before_{i:03}": x for i, x in enumerate(layers)},
                **{f"after_{i:03}": x for i, x in enumerate(result.edited_layers)},
                license=np.array("Arphic-1999"),
                modification=np.array(notice),
                source_char=np.array(row["base_char"]),
                bridge_mode=np.array(""),
            )
            candidates.append(
                {
                    **row,
                    "candidate_id": identifier,
                    "generator_version": VERSION,
                    "generation_mode": "archived_accepted_attempt_replay",
                    "baseline_candidate_id": row["candidate_id"],
                    "decision": "REVIEW",
                    "training_eligible": False,
                    "label_provenance": "synthetic_stroke_candidate_unreviewed",
                    "files": files,
                    "pixel_sha256": digest,
                    "changed_pixels": result.changed_pixels,
                    "metrics": result.metrics,
                    "modification": notice,
                    "stroke_archive": {
                        "path": archive_name,
                        "sha256": _sha(staging / archive_name),
                    },
                }
            )
            pair.update(
                status="accepted",
                candidate_id=identifier,
                same_site=True,
                pixel_changed=digest != row["pixel_sha256"],
                metrics=result.metrics,
                pixel_sha256=digest,
                length_ratio=1.15,
                changed_pixel_ratio=result.changed_pixels / row["changed_pixels"],
            )
            pair["files"].update(new_96=files["candidate_96"], new_mask_96=files["mask_96"])
            pairs.append(pair)
        _write_jsonl(staging / "candidates.jsonl", candidates)
        _write_jsonl(staging / "pairs.jsonl", pairs)
        _write_jsonl(
            staging / "review-template.jsonl",
            [
                {
                    "candidate_id": row["candidate_id"],
                    "decision": "REVIEW",
                    "confirmed_char": None,
                    "annotator_id": "",
                    "reason": "",
                    "candidate_sha256": row["files"]["candidate"]["sha256"],
                }
                for row in candidates
            ],
        )
        run = {
            "purpose": "same_site_calibration_not_training_or_independent_evaluation",
            "generator_version": VERSION,
            "created_date": date,
            "baseline_count": len(rows),
            "candidate_count": len(candidates),
            "pixel_changed_count": sum(pair.get("pixel_changed") is True for pair in pairs),
            "pixel_unchanged_count": sum(pair.get("pixel_changed") is False for pair in pairs),
            "skipped_count": len(rows) - len(candidates),
            "length_multiplier": 1.15,
            "sampling": (
                "archived accepted attempt; earlier valid proposals ignored; no replacement"
            ),
            "baseline_files_sha256": {
                name: _sha(baseline / name)
                for name in ("candidates.jsonl", "run.json", "source.json")
            },
            "code_sha256": {
                Path(inspect.getfile(module)).name: _sha(Path(inspect.getfile(module)))
                for module in (stroke_break, stroke_corrupt, stroke_preview, preview_v2)
            },
            "replay_code_sha256": _sha(Path(__file__)),
            "packages": {
                name: version(name)
                for name in ("numpy", "pillow", "opencv-python-headless", "scikit-image")
            },
            "skipped": [
                {key: pair[key] for key in ("baseline_candidate_id", "base_char", "reason")}
                for pair in pairs
                if pair["status"] == "skipped"
            ],
        }
        (staging / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        (staging / "candidates.html").write_text(
            stroke_preview._stroke_html(candidates, len(candidates) == len(rows), date),
            encoding="utf-8",
        )
        _comparison_page(staging, pairs, date)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during generation: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=Path("docs/previews/glyph-breaks-20260910")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate_comparison(args.baseline, args.output_dir)
    print(
        f"pairs={result['baseline_count']} passed={result['candidate_count']} "
        f"pixel_changed={result['pixel_changed_count']} "
        f"pixel_unchanged={result['pixel_unchanged_count']} "
        f"skipped={result['skipped_count']} preview={args.output_dir / 'index.html'}",
        flush=True,
    )
    raise SystemExit(2 if result["skipped_count"] else 0)
