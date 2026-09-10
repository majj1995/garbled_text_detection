"""Build an offline 1x/1.15x/2x comparison at the same archived break sites."""

# Chinese punctuation belongs to the human review page.
# ruff: noqa: RUF001

import argparse
import hashlib
import html
import inspect
import json
import runpy
import shutil
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from poor_word.glyphs import preview_v2, stroke_break, stroke_corrupt, stroke_preview

VERSION = "stroke-break-double-v1"
_REPLAY_PATH = Path(__file__).with_name("replay_break_comparison.py")
replay_sample = runpy.run_path(str(_REPLAY_PATH))["replay_sample"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked(directory: Path, info: dict[str, str]) -> Path:
    path = (directory / info["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or _sha(path) != info["sha256"]:
        raise ValueError(f"archive hash/path mismatch: {info['path']}")
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_layers(directory: Path, row: dict[str, Any]) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    archive = _checked(directory, row["stroke_archive"])
    count = int(row["original_stroke_count"])
    with np.load(archive, allow_pickle=False) as data:
        if str(data["source_char"]) != row["base_char"] or str(data["license"]) != "Arphic-1999":
            raise ValueError("archive source/license metadata mismatch")
        before = tuple(data[f"before_{index:03}"].copy() for index in range(count))
        after = tuple(data[f"after_{index:03}"].copy() for index in range(count))
    one_x = np.repeat(np.maximum.reduce(after)[:, :, None], 3, axis=2)
    if hashlib.sha256(one_x.tobytes()).hexdigest() != row["pixel_sha256"]:
        raise ValueError("baseline recomposed pixel hash mismatch")
    return before, one_x


def _validate_inputs(
    baseline: Path, prior: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = _rows(baseline / "candidates.jsonl")
    prior_pairs = _rows(prior / "pairs.jsonl")
    prior_rows = _rows(prior / "candidates.jsonl")
    if (
        not baseline_rows
        or len(baseline_rows) != len(prior_pairs)
        or len({row["candidate_id"] for row in baseline_rows}) != len(baseline_rows)
        or len({row["base_char"] for row in baseline_rows}) != len(baseline_rows)
    ):
        raise ValueError("comparison inputs must contain one prior pair per distinct baseline slot")
    if _sha(baseline / "ARPHICPL.txt") != stroke_preview._ARPHIC_SHA256:
        raise ValueError("baseline license hash mismatch")
    if _sha(prior / "ARPHICPL.txt") != stroke_preview._ARPHIC_SHA256:
        raise ValueError("prior comparison license hash mismatch")
    if (baseline / "source.json").read_bytes() != (prior / "source.json").read_bytes():
        raise ValueError("comparison source metadata mismatch")

    baseline_run = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    prior_run = json.loads((prior / "run.json").read_text(encoding="utf-8"))
    for name, expected in prior_run.get("baseline_files_sha256", {}).items():
        if _sha(baseline / name) != expected:
            raise ValueError(f"prior comparison baseline hash mismatch: {name}")
    if baseline_run.get("candidate_count") != len(baseline_rows):
        raise ValueError("baseline run count mismatch")

    prior_by_id = {row["candidate_id"]: row for row in prior_rows}
    if len(prior_by_id) != len(prior_rows):
        raise ValueError("duplicate prior candidate id")
    accepted_ids: set[str] = set()
    for baseline_row, pair in zip(baseline_rows, prior_pairs, strict=True):
        _load_layers(baseline, baseline_row)
        for info in baseline_row["files"].values():
            _checked(baseline, info)
        for info in pair["files"].values():
            _checked(prior, info)
        if (
            pair["baseline_candidate_id"] != baseline_row["candidate_id"]
            or pair["base_char"] != baseline_row["base_char"]
            or pair["seed"] != baseline_row["seed"]
            or pair["selected_stroke_indices"] != baseline_row["selected_stroke_indices"]
            or pair["baseline_pixel_sha256"] != baseline_row["pixel_sha256"]
            or pair["baseline_metrics"] != baseline_row["metrics"]
        ):
            raise ValueError("prior comparison pair does not match its baseline slot")
        if pair["files"]["original_96"]["sha256"] != baseline_row["files"]["original_96"]["sha256"]:
            raise ValueError("prior comparison original image does not match baseline")
        if pair["files"]["old_96"]["sha256"] != baseline_row["files"]["candidate_96"]["sha256"]:
            raise ValueError("prior comparison 1x image does not match baseline")
        if pair["status"] == "skipped":
            if pair.get("candidate_id") is not None or "new_96" in pair["files"]:
                raise ValueError("skipped prior slot unexpectedly contains a replacement")
            continue
        if pair["status"] != "accepted" or not pair.get("same_site"):
            raise ValueError("unsupported prior comparison status")
        prior_row = prior_by_id.get(pair["candidate_id"])
        if prior_row is None:
            raise ValueError("accepted prior pair is missing its candidate")
        accepted_ids.add(prior_row["candidate_id"])
        for info in prior_row["files"].values():
            _checked(prior, info)
        _checked(prior, prior_row["stroke_archive"])
        for key in ("base_char", "seed", "selected_stroke_indices"):
            if prior_row[key] != baseline_row[key]:
                raise ValueError(f"prior comparison changed archived {key}")
        if (
            prior_row["baseline_candidate_id"] != baseline_row["candidate_id"]
            or prior_row["pixel_sha256"] != pair["pixel_sha256"]
            or pair.get("length_ratio") != 1.15
            or not np.isclose(
                prior_row["metrics"]["gap_length"],
                baseline_row["metrics"]["gap_length"] * 1.15,
            )
        ):
            raise ValueError("prior comparison strength or pixel pairing mismatch")
        for key in ("break_center_x_96", "break_center_y_96", "local_stroke_width"):
            if prior_row["metrics"][key] != baseline_row["metrics"][key]:
                raise ValueError(f"prior comparison changed archived site metric: {key}")
        if pair["files"]["new_96"]["sha256"] != prior_row["files"]["candidate_96"]["sha256"]:
            raise ValueError("prior comparison candidate image pairing mismatch")
    if accepted_ids != set(prior_by_id):
        raise ValueError("prior comparison contains an unpaired candidate")
    return baseline_rows, prior_pairs


def _copy_asset(source: Path, staging: Path, name: str) -> dict[str, str]:
    target = staging / f"images/{name}.png"
    shutil.copyfile(source, target)
    return {"path": f"images/{name}.png", "sha256": _sha(target)}


def _comparison_page(directory: Path, pairs: list[dict[str, Any]], date: str) -> None:
    header_height = 32
    sheet = Image.new("RGB", (540, header_height + max(1, len(pairs)) * 124), "white")
    draw = ImageDraw.Draw(sheet)
    for x, label in ((76, "Original"), (188, "1x"), (300, "1.15x"), (412, "2x")):
        draw.text((x, 8), label, fill="black")
    cards: list[str] = []
    views = (
        ("original_96", "原字"),
        ("one_x_96", "1×"),
        ("prior_1_15_96", "1.15×"),
        ("double_96", "2×"),
    )
    for number, pair in enumerate(pairs, 1):
        y = header_height + (number - 1) * 124
        draw.text((4, y + 6), f"{number:02}", fill="black")
        figures = []
        for column, (key, label) in enumerate(views):
            info = pair["files"].get(key)
            x = 76 + column * 112
            if info is None:
                draw.text((x + 25, y + 50), "SKIP", fill="#8a3030")
                content = '<div class="skip">SKIP<br>不替换</div>'
            else:
                with Image.open(directory / info["path"]) as image:
                    sheet.paste(image.convert("RGB"), (x, y + 24))
                content = f'<img width="96" height="96" src="{html.escape(info["path"])}">'
            figures.append(f"<figure>{content}<figcaption>{label}</figcaption></figure>")
        cards.append(
            f"<article><h2>{number:02} · {html.escape(pair['base_char'])}</h2>"
            f'<div class="views">{"".join(figures)}</div>'
            "<details><summary>固定尝试记录</summary>"
            f"<pre>{html.escape(json.dumps(pair, ensure_ascii=False, indent=2))}</pre>"
            "</details></article>"
        )
    sheet.save(directory / "overview.png")
    stroke_preview._stamp_png(
        directory / "overview.png", f"{date}: fixed-site 1x/1.15x/2x comparison."
    )
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    (directory / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>断笔切口 1× / 1.15× / 2× 同位置对照</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;max-width:1040px;margin:32px auto;"
        "padding:0 16px;color:#202428;background:#f6f7f8}h1{font-size:25px}h2{font-size:18px}"
        "article{padding:18px;background:white;border:1px solid #ddd;margin:16px 0;"
        "border-radius:8px}.views{display:flex;flex-wrap:wrap;gap:18px}figure{margin:8px 0;"
        "text-align:center}.skip{width:96px;height:96px;background:#f9eeee;display:grid;"
        "place-content:center;color:#8a3030}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "font-size:12px}summary{cursor:pointer}</style>"
        "<h1>断笔切口 1× / 1.15× / 2× · 同笔画同切点</h1>"
        f"<p>固定保留 {run['slot_count']} 个原始槽位；2× 有 "
        f"{run['double_candidate_count']} 个通过、"
        f"{run['double_skipped_count']} 个跳过。跳过项不换字、不换笔画、不换切点。</p>"
        "<p>从左到右为原字、原始 1×、旧 1.15×、新 2×。仅沿笔画方向的切除长度改变；"
        "横向宽度、随机抽样序列和全部保护门槛不变。连续坐标长度倍率不代表可见像素面积倍率。</p>"
        "<p>这是固定旧尝试的离线校准。生产采样返回首个合格提案，强度变化时可能更早接受，"
        "所以两者有意不同。所有新 2× 候选均为 REVIEW、非训练数据。</p>"
        '<p><a href="overview.png">总览图</a> · <a href="candidates.html">2×候选盲看页</a> · '
        '<a href="review-template.jsonl">审阅模板</a> · <a href="pairs.jsonl">完整配对</a> · '
        '<a href="reproduce.md">复现说明</a> · <a href="ARPHICPL.txt">Arphic-1999许可</a></p>'
        f"<p>Copyright (C) 1999 Arphic Technology Co., Ltd. {date}：固定位置生成强度对照；"
        "图形衍生物按 Arphic-1999 提供，无担保。</p>" + "".join(cards) + "</html>\n",
        encoding="utf-8",
    )


def _reproduce_text(run: dict[str, Any]) -> str:
    skipped = "、".join(item["base_char"] for item in run["double_skipped"])
    return f"""# 断笔切口 1× / 1.15× / 2×：固定位置对照

打开 [index.html](index.html)。本档案保留原始 20 个字符槽位，逐项展示原字、原始 1×、
既有 1.15× 和新 2×；任何倍率未通过保护检查时都留空，不重新选字、笔画或切点。

- 2× 实际通过 {run["double_candidate_count"]} 项，跳过 {run["double_skipped_count"]} 项：{skipped}。
- 通过的 2× 中，{run["double_pixel_changed_vs_one_x_count"]} 项相对 1× 像素变化，
  {run["double_pixel_unchanged_vs_one_x_count"]} 项像素相同。
- 既有 1.15× 保持原档案结果：17 项通过，其中 16 项像素变化；“脖”参数变化但像素相同；3 项跳过。
- 2× 只改变沿笔画方向的切口长度；横向宽度、RNG 抽样、笔端排除、分段和可见性门槛不变。
- 全部新图均为 `REVIEW`、`training_eligible=false`；没有训练、训练 manifest 或标签继承。

## 为什么固定旧尝试

生产接口逐次提案并返回第一个合格结果。倍率改变后，更早的提案可能通过，原提案也可能失败，
所以同种子重新调用生产采样可能换笔或换位置。本离线脚本从原始 NPZ 的 `before_###` 层出发，
按原种子重放到每条记录的 `metrics.attempts`，忽略此前提案，只评估原先接受的那次尝试。
它核对来源字符、种子、笔画、切点、局部宽度和长度倍率；失败就跳过。这种固定尝试比较与生产的
“首个合格提案”采样有意不同。

## 文件与复现

`pairs.jsonl` 始终有 20 行；`candidates.jsonl` 只含通过全部保护检查的新 2× 候选。
每个新候选有 9 个视图及可重组的 `layers/*.npz` 前后笔画；旧 1×/1.15× 显示资产按原字节复制。
`run.json` 记录输入、源码与依赖哈希。修改任一输入资产会中止生成。

在项目根目录、使用匹配 `run.json` 的源码和依赖时执行：

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/poor-word-uv-cache uv run --offline python \\
  docs/previews/compare_break_strengths.py \\
  --baseline docs/previews/glyph-breaks-20260910 \\
  --prior docs/previews/glyph-breaks-longer-20260910 \\
  --output-dir artifacts/glyph-breaks-double-replay
```

预期打印 `slots=20 double_passed=7 double_changed=7 double_unchanged=0 double_skipped=13`，
并因明确存在跳过项返回退出码 2；这不是崩溃，也不会补位。日期进入 PNG/NPZ 修改说明，
跨日期生成时文件哈希可不同，但像素哈希、固定位置和通过/跳过结果应一致。

图形源自 Make Me a Hanzi / Arphic。分享时保留 [Arphic-1999 许可全文](ARPHICPL.txt)；
新图包含日期和修改说明，不提供担保。
"""


def generate_strength_comparison(baseline: Path, prior: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"use a fresh output directory: {output}")
    baseline, prior = baseline.resolve(), prior.resolve()
    baseline_rows, prior_pairs = _validate_inputs(baseline, prior)
    date = datetime.now(UTC).date().isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    candidates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    try:
        (staging / "images").mkdir()
        (staging / "layers").mkdir()
        for name in ("ARPHICPL.txt", "source.json"):
            shutil.copyfile(baseline / name, staging / name)
        for baseline_row, prior_pair in zip(baseline_rows, prior_pairs, strict=True):
            layers, _one_x = _load_layers(baseline, baseline_row)
            baseline_id = baseline_row["candidate_id"]
            files = {
                "original_96": _copy_asset(
                    baseline / baseline_row["files"]["original_96"]["path"],
                    staging,
                    f"{baseline_id}-original-96",
                ),
                "one_x_96": _copy_asset(
                    baseline / baseline_row["files"]["candidate_96"]["path"],
                    staging,
                    f"{baseline_id}-one-x-96",
                ),
                "one_x_mask_96": _copy_asset(
                    baseline / baseline_row["files"]["mask_96"]["path"],
                    staging,
                    f"{baseline_id}-one-x-mask-96",
                ),
            }
            pair: dict[str, Any] = {
                "baseline_candidate_id": baseline_id,
                "base_char": baseline_row["base_char"],
                "seed": baseline_row["seed"],
                "selected_stroke_indices": baseline_row["selected_stroke_indices"],
                "baseline_metrics": baseline_row["metrics"],
                "one_x_pixel_sha256": baseline_row["pixel_sha256"],
                "prior_1_15_status": prior_pair["status"],
                "prior_1_15_candidate_id": prior_pair.get("candidate_id"),
                "prior_1_15_pixel_changed": prior_pair.get("pixel_changed"),
                "prior_1_15_pixel_sha256": prior_pair.get("pixel_sha256"),
                "double_status": "skipped",
                "double_candidate_id": None,
                "files": files,
            }
            if prior_pair["status"] == "accepted":
                pair["files"]["prior_1_15_96"] = _copy_asset(
                    prior / prior_pair["files"]["new_96"]["path"],
                    staging,
                    f"{baseline_id}-prior-one-fifteen-96",
                )
                pair["files"]["prior_1_15_mask_96"] = _copy_asset(
                    prior / prior_pair["files"]["new_mask_96"]["path"],
                    staging,
                    f"{baseline_id}-prior-one-fifteen-mask-96",
                )
            result = replay_sample(layers, baseline_row, length_multiplier=2.0)
            if result is None:
                pair["double_reason"] = (
                    "archived_site_failed_unchanged_safety_gates; no replacement"
                )
                pairs.append(pair)
                continue
            digest = hashlib.sha256(result.image.tobytes()).hexdigest()
            identifier = (
                "s" + hashlib.sha256(f"{VERSION}:{baseline_id}:{digest}".encode()).hexdigest()[:12]
            )
            notice = (
                f"{date}: doubled the archived interior break length; same source stroke and "
                f"site as {baseline_id}; transverse width and all safety gates retained; "
                "see pairs.jsonl. REVIEW only; no training label."
            )
            selected = result.selected_stroke_indices[0]
            original = np.repeat(np.maximum.reduce(layers)[:, :, None], 3, axis=2)
            new_files = stroke_preview.save_stroke_assets(
                staging, identifier, original, result.image, layers[selected], notice
            )
            archive_name = f"layers/{identifier}.npz"
            np.savez_compressed(
                staging / archive_name,
                **{f"before_{index:03}": layer for index, layer in enumerate(layers)},
                **{f"after_{index:03}": layer for index, layer in enumerate(result.edited_layers)},
                license=np.array("Arphic-1999"),
                modification=np.array(notice),
                source_char=np.array(baseline_row["base_char"]),
                bridge_mode=np.array(""),
            )
            candidate = {
                **baseline_row,
                "candidate_id": identifier,
                "generator_version": VERSION,
                "generation_mode": "archived_accepted_attempt_replay",
                "baseline_candidate_id": baseline_id,
                "decision": "REVIEW",
                "training_eligible": False,
                "label_provenance": "synthetic_stroke_candidate_unreviewed",
                "length_multiplier": 2.0,
                "files": new_files,
                "pixel_sha256": digest,
                "changed_pixels": result.changed_pixels,
                "metrics": result.metrics,
                "modification": notice,
                "stroke_archive": {
                    "path": archive_name,
                    "sha256": _sha(staging / archive_name),
                },
            }
            candidates.append(candidate)
            pair.update(
                double_status="accepted",
                double_candidate_id=identifier,
                double_same_site=True,
                double_pixel_changed_vs_one_x=digest != baseline_row["pixel_sha256"],
                double_pixel_sha256=digest,
                double_metrics=result.metrics,
                length_ratio=2.0,
            )
            pair["files"].update(
                double_96=new_files["candidate_96"], double_mask_96=new_files["mask_96"]
            )
            pairs.append(pair)

        run = {
            "purpose": "fixed_attempt_strength_calibration_not_training_or_evaluation",
            "generator_version": VERSION,
            "created_date": date,
            "slot_count": len(pairs),
            "double_candidate_count": len(candidates),
            "double_pixel_changed_vs_one_x_count": sum(
                pair.get("double_pixel_changed_vs_one_x") is True for pair in pairs
            ),
            "double_pixel_unchanged_vs_one_x_count": sum(
                pair.get("double_pixel_changed_vs_one_x") is False for pair in pairs
            ),
            "double_skipped_count": len(pairs) - len(candidates),
            "prior_1_15_candidate_count": sum(
                pair["prior_1_15_status"] == "accepted" for pair in pairs
            ),
            "prior_1_15_pixel_changed_count": sum(
                pair["prior_1_15_pixel_changed"] is True for pair in pairs
            ),
            "prior_1_15_pixel_unchanged_count": sum(
                pair["prior_1_15_pixel_changed"] is False for pair in pairs
            ),
            "length_multipliers": [1.0, 1.15, 2.0],
            "sampling": (
                "original archived accepted attempt only; earlier proposals ignored; "
                "no reselection or replacement; production uses first-valid sampling"
            ),
            "input_files_sha256": {
                "baseline": {
                    name: _sha(baseline / name)
                    for name in ("candidates.jsonl", "run.json", "source.json")
                },
                "prior_1_15": {
                    name: _sha(prior / name)
                    for name in ("candidates.jsonl", "pairs.jsonl", "run.json", "source.json")
                },
            },
            "code_sha256": {
                Path(inspect.getfile(module)).name: _sha(Path(inspect.getfile(module)))
                for module in (stroke_break, stroke_corrupt, stroke_preview, preview_v2)
            },
            "comparison_code_sha256": _sha(Path(__file__)),
            "replay_code_sha256": _sha(_REPLAY_PATH),
            "packages": {
                name: version(name)
                for name in ("numpy", "pillow", "opencv-python-headless", "scikit-image")
            },
            "double_skipped": [
                {
                    "baseline_candidate_id": pair["baseline_candidate_id"],
                    "base_char": pair["base_char"],
                    "reason": pair["double_reason"],
                }
                for pair in pairs
                if pair["double_status"] == "skipped"
            ],
        }
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
        (staging / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "reproduce.md").write_text(_reproduce_text(run), encoding="utf-8")
        (staging / "candidates.html").write_text(
            stroke_preview._stroke_html(candidates, len(candidates) == len(pairs), date),
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
    parser.add_argument(
        "--prior", type=Path, default=Path("docs/previews/glyph-breaks-longer-20260910")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate_strength_comparison(args.baseline, args.prior, args.output_dir)
    print(
        f"slots={result['slot_count']} double_passed={result['double_candidate_count']} "
        f"double_changed={result['double_pixel_changed_vs_one_x_count']} "
        f"double_unchanged={result['double_pixel_unchanged_vs_one_x_count']} "
        f"double_skipped={result['double_skipped_count']} "
        f"preview={args.output_dir / 'index.html'}",
        flush=True,
    )
    raise SystemExit(2 if result["double_skipped_count"] else 0)
