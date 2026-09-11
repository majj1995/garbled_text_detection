"""Replay five archived block-gap gates with the strengthened blocker at the same sites.

This is a fixed-site offline calibration, not the production first-valid sampler.
Any site that fails the current unchanged safety gates stays skipped without replacement.
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

from poor_word.glyphs import preview_v2, stroke_bridge, stroke_corrupt, stroke_preview
from poor_word.glyphs.stroke_corrupt import ByteArray, StrokeCorruptionResult

VERSION = "stroke-block-gap-strength-v1"
BASELINE_IDS = (
    "sb361978ba729",
    "s58b7f878c5b1",
    "s453475d07331",
    "sf7cd087529b7",
    "s4ea729dc042e",
)
BASELINE_CANDIDATES_SHA256 = "94a025874eb147dd712c56625786b42914dca7f7e5708d7e7f7f27776f1ee177"
SITE_KEYS = (
    "gate_start_x_96",
    "gate_start_y_96",
    "gate_end_x_96",
    "gate_end_y_96",
    "gate_width_96",
    "local_stroke_width_96",
    "roi_x0_96",
    "roi_y0_96",
    "roi_x1_96",
    "roi_y1_96",
    "original_gap_length_96",
    "passage_start_x_96",
    "passage_start_y_96",
    "passage_end_x_96",
    "passage_end_y_96",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _archived_site(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: row["metrics"][key] for key in SITE_KEYS},
        "selected_stroke_indices": row["selected_stroke_indices"],
    }


def _passage_site(passage: Any) -> dict[str, Any]:
    gate = passage.gate
    start = gate.point(gate.left)
    end = gate.point(gate.right)
    yy, xx = np.nonzero(passage.region)
    return {
        "gate_start_x_96": float(start[1]),
        "gate_start_y_96": float(start[0]),
        "gate_end_x_96": float(end[1]),
        "gate_end_y_96": float(end[0]),
        "gate_width_96": float(gate.right - gate.left - 1),
        "local_stroke_width_96": gate.width,
        "roi_x0_96": float(xx.min()),
        "roi_y0_96": float(yy.min()),
        "roi_x1_96": float(xx.max() + 1),
        "roi_y1_96": float(yy.max() + 1),
        "original_gap_length_96": float(passage.span),
        "passage_start_x_96": float(passage.first[1]),
        "passage_start_y_96": float(passage.first[0]),
        "passage_end_x_96": float(passage.last[1]),
        "passage_end_y_96": float(passage.last[0]),
        "selected_stroke_indices": list(gate.selected),
    }


def _planner(layers: tuple[ByteArray, ...], row: dict[str, Any]) -> Any:
    original = np.maximum.reduce(layers)
    points = np.argwhere(original >= 128)
    if not len(points):
        raise ValueError("archived layers have no visible stroke core")
    widths = [stroke_corrupt._width(layer) for layer in layers]
    scale = float(np.ptp(points, axis=0).max() + 1)
    return stroke_bridge.BridgePlanner(layers, widths, scale, row["seed"])


def _matching_passage(planner: Any, row: dict[str, Any]) -> Any:
    wanted = _archived_site(row)
    matches = [passage for passage in planner.passages if _passage_site(passage) == wanted]
    if len(matches) != 1:
        raise ValueError(
            f"archived gate site no longer maps to exactly one passage: {row['candidate_id']}"
        )
    return matches[0]


def _replay(
    layers: tuple[ByteArray, ...], row: dict[str, Any]
) -> tuple[StrokeCorruptionResult | None, dict[str, Any]]:
    if row["operator"] != "bridge" or row["bridge_mode"] != "block_gap":
        raise ValueError("expected an archived accepted block_gap bridge")
    planner = _planner(layers, row)
    passage = _matching_passage(planner, row)
    replayed_site = _passage_site(passage)
    proposal = planner._block(passage)
    if proposal is None:
        return None, replayed_site
    if proposal.mode != "block_gap" or list(proposal.selected) != row["selected_stroke_indices"]:
        raise ValueError("strengthened proposal changed the archived bridge subtype or strokes")
    original = np.asarray(np.maximum.reduce(layers), dtype=np.uint8)
    candidate = np.asarray(np.maximum.reduce(proposal.layers), dtype=np.uint8)
    visibility = stroke_corrupt._visible(original, candidate)
    if visibility is None:
        return None, replayed_site
    for key in SITE_KEYS:
        if proposal.metrics[key] != row["metrics"][key]:
            raise ValueError(f"strengthened proposal changed archived site metric: {key}")
    changed = (original != candidate).astype(np.uint8)
    points = np.argwhere(original >= 128)
    return (
        StrokeCorruptionResult(
            image=np.repeat(candidate[:, :, None], 3, axis=2),
            changed_mask=changed,
            changed_pixels=int(changed.sum()),
            operator="bridge",
            metrics={
                **proposal.metrics,
                **visibility,
                "glyph_scale": float(np.ptp(points, axis=0).max() + 1),
                "attempts": row["metrics"]["attempts"],
            },
            selected_stroke_indices=proposal.selected,
            edited_layers=tuple(layer.copy() for layer in proposal.layers),
            bridge_mode="block_gap",
        ),
        replayed_site,
    )


def replay_sample(
    layers: tuple[ByteArray, ...], row: dict[str, Any]
) -> StrokeCorruptionResult | None:
    """Replay only the archived passage, then apply the generic visibility gate unchanged."""
    return _replay(layers, row)[0]


def _load_inputs(
    baseline: Path,
) -> tuple[list[dict[str, Any]], dict[str, tuple[tuple[ByteArray, ...], np.ndarray]]]:
    candidates_path = baseline / "candidates.jsonl"
    if _sha(candidates_path) != BASELINE_CANDIDATES_SHA256:
        raise ValueError("baseline candidates hash mismatch")
    all_rows = [
        json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["candidate_id"]: row for row in all_rows}
    if len(by_id) != len(all_rows) or not all(
        candidate_id in by_id for candidate_id in BASELINE_IDS
    ):
        raise ValueError("baseline does not contain the exact five archived candidate ids")
    if _sha(baseline / "ARPHICPL.txt") != stroke_preview._ARPHIC_SHA256:
        raise ValueError("baseline license hash mismatch")
    source = json.loads((baseline / "source.json").read_text(encoding="utf-8"))
    if source.get("license_id") != "Arphic-1999":
        raise ValueError("baseline source metadata license mismatch")

    rows = [by_id[candidate_id] for candidate_id in BASELINE_IDS]
    loaded: dict[str, tuple[tuple[ByteArray, ...], np.ndarray]] = {}
    for row in rows:
        if row["source_sha256"] != source.get("sha256"):
            raise ValueError("baseline row/source hash mismatch")
        archive = _checked(baseline, row["stroke_archive"])
        for key in ("original_96", "candidate_96", "mask_96"):
            _checked(baseline, row["files"][key])
        with np.load(archive, allow_pickle=False) as data:
            count = int(row["original_stroke_count"])
            if (
                str(data["source_char"]) != row["base_char"]
                or str(data["bridge_mode"]) != "block_gap"
                or str(data["license"]) != "Arphic-1999"
            ):
                raise ValueError("baseline archive metadata mismatch")
            layers = tuple(data[f"before_{index:03}"].copy() for index in range(count))
            after_keys = sorted(key for key in data.files if key.startswith("after_"))
            old_alpha = np.maximum.reduce([data[key] for key in after_keys])
        old = np.repeat(old_alpha[:, :, None], 3, axis=2)
        if hashlib.sha256(old.tobytes()).hexdigest() != row["pixel_sha256"]:
            raise ValueError("baseline recomposed pixel hash mismatch")
        _matching_passage(_planner(layers, row), row)
        loaded[row["candidate_id"]] = (layers, old)
    return rows, loaded


def _copy_asset(source: Path, staging: Path, name: str) -> dict[str, str]:
    target = staging / f"images/{name}.png"
    shutil.copyfile(source, target)
    return {"path": f"images/{name}.png", "sha256": _sha(target)}


def _comparison_page(directory: Path, pairs: list[dict[str, Any]], date: str) -> None:
    header_height = 32
    sheet = Image.new("RGB", (428, header_height + len(pairs) * 124), "white")
    draw = ImageDraw.Draw(sheet)
    for x, label in ((76, "Original"), (188, "Old"), (300, "New")):
        draw.text((x, 8), label, fill="black")
    cards: list[str] = []
    views = (("original_96", "原字"), ("old_96", "旧堵塞"), ("new_96", "增强堵塞"))
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
            "<details><summary>固定门位记录</summary>"
            f"<pre>{html.escape(json.dumps(pair, ensure_ascii=False, indent=2))}</pre>"
            "</details></article>"
        )
    sheet.save(directory / "overview.png")
    stroke_preview._stamp_png(
        directory / "overview.png", f"{date}: fixed-site old/new block-gap comparison."
    )
    accepted = sum(pair["status"] == "accepted" for pair in pairs)
    (directory / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>间隙堵塞增强 · 同门位对照</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;max-width:760px;margin:32px auto;"
        "padding:0 16px;color:#202428;background:#f6f7f8}h1{font-size:25px}h2{font-size:18px}"
        "article{padding:18px;background:white;border:1px solid #ddd;margin:16px 0;"
        "border-radius:8px}.views{display:flex;gap:16px}figure{margin:8px 0;text-align:center}"
        ".skip{width:96px;height:96px;background:#f9eeee;display:grid;place-content:center;"
        "color:#8a3030}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"
        "summary{cursor:pointer}</style><h1>间隙堵塞增强 · 同一原始门位</h1>"
        f"<p>固定5个旧门位：{accepted}个通过，{len(pairs) - accepted}个跳过。"
        "跳过项不换字、不换笔画、不换门位。</p>"
        "<p>从左到右为原字、旧堵塞和增强堵塞。新图只在原始 before 笔画层上重放当前"
        " block_gap，并继续使用通用可见性门槛；全部为 REVIEW，不是训练数据。</p>"
        '<p><a href="overview.png">总览图</a> · <a href="candidates.html">增强候选盲看页</a> · '
        '<a href="review-template.jsonl">审阅模板</a> · <a href="pairs.jsonl">配对记录</a> · '
        '<a href="run.json">运行记录</a> · <a href="ARPHICPL.txt">Arphic-1999许可</a></p>'
        f"<p>Copyright (C) 1999 Arphic Technology Co., Ltd. {date}：固定门位生成强度对照；"
        "图形衍生物按 Arphic-1999 提供，无担保。</p>" + "".join(cards) + "</html>\n",
        encoding="utf-8",
    )


def generate_gap_comparison(baseline: Path, output: Path) -> dict[str, Any]:
    """Generate fixed original/old/new views while leaving the archived directory untouched."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"use a fresh output directory: {output}")
    baseline = baseline.resolve()
    rows, loaded = _load_inputs(baseline)
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
        for row in rows:
            baseline_id = row["candidate_id"]
            layers, _old = loaded[baseline_id]
            result, replayed_site = _replay(layers, row)
            files = {
                "original_96": _copy_asset(
                    baseline / row["files"]["original_96"]["path"],
                    staging,
                    f"{baseline_id}-original-96",
                ),
                "old_96": _copy_asset(
                    baseline / row["files"]["candidate_96"]["path"],
                    staging,
                    f"{baseline_id}-old-96",
                ),
                "old_mask_96": _copy_asset(
                    baseline / row["files"]["mask_96"]["path"],
                    staging,
                    f"{baseline_id}-old-mask-96",
                ),
            }
            pair: dict[str, Any] = {
                "baseline_candidate_id": baseline_id,
                "base_char": row["base_char"],
                "seed": row["seed"],
                "selected_stroke_indices": row["selected_stroke_indices"],
                "status": "skipped",
                "candidate_id": None,
                "same_site": True,
                "archived_site": _archived_site(row),
                "replayed_site": replayed_site,
                "baseline_source_sha256": row["source_sha256"],
                "baseline_archive_sha256": row["stroke_archive"]["sha256"],
                "baseline_original_96_sha256": row["files"]["original_96"]["sha256"],
                "baseline_old_96_sha256": row["files"]["candidate_96"]["sha256"],
                "baseline_pixel_sha256": row["pixel_sha256"],
                "files": files,
            }
            if result is None:
                pair["reason"] = "archived_site_failed_current_safety_gates; no replacement"
                pairs.append(pair)
                continue
            digest = hashlib.sha256(result.image.tobytes()).hexdigest()
            identifier = (
                "s" + hashlib.sha256(f"{VERSION}:{baseline_id}:{digest}".encode()).hexdigest()[:12]
            )
            notice = (
                f"{date}: strengthened block_gap at the exact archived gate from {baseline_id}; "
                "original seed, passage and safety gates retained; see pairs.jsonl. "
                "REVIEW only; no training label."
            )
            original = np.repeat(np.maximum.reduce(layers)[:, :, None], 3, axis=2)
            selected = np.maximum.reduce(
                [layers[index] for index in result.selected_stroke_indices]
            )
            candidate_files = stroke_preview.save_stroke_assets(
                staging, identifier, original, result.image, selected, notice
            )
            archive_name = f"layers/{identifier}.npz"
            np.savez_compressed(
                staging / archive_name,
                **{f"before_{index:03}": layer for index, layer in enumerate(layers)},
                **{f"after_{index:03}": layer for index, layer in enumerate(result.edited_layers)},
                license=np.array("Arphic-1999"),
                modification=np.array(notice),
                source_char=np.array(row["base_char"]),
                bridge_mode=np.array("block_gap"),
                baseline_candidate_id=np.array(baseline_id),
            )
            candidate = {
                **row,
                "candidate_id": identifier,
                "generator_version": VERSION,
                "generation_mode": "exact_archived_block_gap_site_replay",
                "baseline_candidate_id": baseline_id,
                "decision": "REVIEW",
                "training_eligible": False,
                "label_provenance": "synthetic_stroke_candidate_unreviewed",
                "files": candidate_files,
                "pixel_sha256": digest,
                "changed_pixels": result.changed_pixels,
                "edited_stroke_count": len(result.edited_layers),
                "metrics": result.metrics,
                "modification": notice,
                "stroke_archive": {
                    "path": archive_name,
                    "sha256": _sha(staging / archive_name),
                },
            }
            candidates.append(candidate)
            pair.update(
                status="accepted",
                candidate_id=identifier,
                pixel_sha256=digest,
                pixel_changed=digest != row["pixel_sha256"],
                metrics=result.metrics,
            )
            pair["files"].update(
                new_96=candidate_files["candidate_96"],
                new_mask_96=candidate_files["mask_96"],
            )
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
        used_paths = {"ARPHICPL.txt", "candidates.jsonl", "run.json", "source.json"}
        for row in rows:
            used_paths.add(row["stroke_archive"]["path"])
            used_paths.update(
                row["files"][key]["path"] for key in ("original_96", "candidate_96", "mask_96")
            )
        source_modules = (stroke_bridge, stroke_corrupt, stroke_preview, preview_v2)
        run = {
            "purpose": "fixed_old_gate_strength_calibration_not_training_or_evaluation",
            "generator_version": VERSION,
            "created_date": date,
            "slot_count": len(rows),
            "candidate_count": len(candidates),
            "skipped_count": len(rows) - len(candidates),
            "complete": len(candidates) == len(rows),
            "baseline_candidate_ids": list(BASELINE_IDS),
            "sampling": "exact archived passage; no retry or replacement",
            "baseline_files_sha256": {name: _sha(baseline / name) for name in sorted(used_paths)},
            "source_code_sha256": {
                str(Path(inspect.getfile(module)).resolve().relative_to(REPO_ROOT)): _sha(
                    Path(inspect.getfile(module))
                )
                for module in source_modules
            },
            "replay_code_sha256": _sha(Path(__file__)),
            "packages": {
                name: version(name)
                for name in ("numpy", "pillow", "opencv-python-headless", "scikit-image")
            },
            "skipped": [
                {
                    "baseline_candidate_id": pair["baseline_candidate_id"],
                    "base_char": pair["base_char"],
                    "reason": pair["reason"],
                }
                for pair in pairs
                if pair["status"] == "skipped"
            ],
        }
        (staging / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("docs/previews/glyph-mixed-20260910"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_gap_comparison(args.baseline, args.output_dir)
    print(
        f"sites={result['slot_count']} passed={result['candidate_count']} "
        f"skipped={result['skipped_count']} preview={args.output_dir / 'index.html'}",
        flush=True,
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
