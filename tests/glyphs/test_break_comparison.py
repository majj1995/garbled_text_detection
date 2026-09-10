"""The calibration comparison must isolate cut length, never silently relocate cuts."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs/previews/glyph-breaks-20260910"
SCRIPT = ROOT / "docs/previews/replay_break_comparison.py"


def _sample(character: str):
    rows = [json.loads(line) for line in (BASELINE / "candidates.jsonl").read_text().splitlines()]
    row = next(row for row in rows if row["base_char"] == character)
    with np.load(BASELINE / row["stroke_archive"]["path"], allow_pickle=False) as data:
        layers = tuple(data[f"before_{i:03}"].copy() for i in range(row["original_stroke_count"]))
        old = tuple(data[f"after_{i:03}"].copy() for i in range(row["original_stroke_count"]))
    return row, layers, old


def test_comparison_keeps_the_old_cut_even_when_an_earlier_alternative_now_passes() -> None:
    # Public seed replay changes the cut in 锌; a length comparison must not do that.
    replay = runpy.run_path(str(SCRIPT))["replay_sample"]
    row, layers, old = _sample("锌")
    result = replay(layers, row)
    assert result is not None
    assert list(result.selected_stroke_indices) == row["selected_stroke_indices"]
    for key in ("break_center_x_96", "break_center_y_96", "local_stroke_width"):
        assert result.metrics[key] == row["metrics"][key]
    assert result.metrics["gap_length"] == pytest.approx(row["metrics"]["gap_length"] * 1.15)
    selected = result.selected_stroke_indices[0]
    assert np.all(result.edited_layers[selected] <= old[selected])
    assert np.any(result.edited_layers[selected] < old[selected])
    for i, layer in enumerate(layers):
        if i != selected:
            np.testing.assert_array_equal(result.edited_layers[i], layer)


@pytest.mark.parametrize("character", ["拎", "叛", "宝"])
def test_rejected_original_site_is_skipped_instead_of_replaced(character: str) -> None:
    replay = runpy.run_path(str(SCRIPT))["replay_sample"]
    row, layers, _ = _sample(character)
    assert replay(layers, row) is None


def test_comparison_preserves_all_baseline_slots_and_never_writes_training_labels(tmp_path) -> None:
    generate = runpy.run_path(str(SCRIPT))["generate_comparison"]
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in BASELINE.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "comparison"
    run = generate(BASELINE, output)
    assert run["baseline_count"] == 20
    assert run["candidate_count"] == 17
    assert run["skipped_count"] == 3
    # Continuous length changes do not necessarily remove another rasterized pixel.
    assert run["pixel_changed_count"] == 16
    assert run["pixel_unchanged_count"] == 1
    pairs = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 20
    unchanged = [pair for pair in pairs if pair.get("pixel_changed") is False]
    assert [pair["base_char"] for pair in unchanged] == ["脖"]
    assert unchanged[0]["pixel_sha256"] == unchanged[0]["baseline_pixel_sha256"]
    assert {pair["base_char"] for pair in pairs if pair["status"] == "skipped"} == {
        "拎",
        "叛",
        "宝",
    }
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 17
    assert all(row["decision"] == "REVIEW" and not row["training_eligible"] for row in rows)
    assert not list(output.rglob("*.parquet"))
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    with pytest.raises(FileExistsError):
        generate(BASELINE, output)


def test_comparison_refuses_a_modified_baseline_archive(tmp_path) -> None:
    generate = runpy.run_path(str(SCRIPT))["generate_comparison"]
    row, _, _ = _sample("寸")
    baseline = tmp_path / "baseline"
    archive = baseline / row["stroke_archive"]["path"]
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"not the archived stroke layers")
    (baseline / "candidates.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash"):
        generate(baseline, tmp_path / "comparison")
