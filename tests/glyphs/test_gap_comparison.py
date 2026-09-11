"""The gap-strength comparison replays five archived sites without replacement."""

import hashlib
import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs/previews/glyph-mixed-20260910"
SCRIPT = ROOT / "docs/previews/replay_gap_comparison.py"
KNOWN_IDS = (
    "sb361978ba729",
    "s58b7f878c5b1",
    "s453475d07331",
    "sf7cd087529b7",
    "s4ea729dc042e",
)
SITE_KEYS = (
    "gate_start_x_96",
    "gate_start_y_96",
    "gate_end_x_96",
    "gate_end_y_96",
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


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _baseline_by_id() -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in _rows(BASELINE / "candidates.jsonl")}


def _hashes(directory: Path) -> dict[Path, str]:
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory):
    """Generate once because reconstructing all five planners is intentionally nontrivial."""
    generate = runpy.run_path(str(SCRIPT))["generate_gap_comparison"]
    before = _hashes(BASELINE)
    output = tmp_path_factory.mktemp("gap-comparison") / "comparison"
    run = generate(BASELINE, output)
    return output, run, before


def test_replay_uses_the_archived_gate_and_never_replaces_a_rejected_site() -> None:
    """Break caught: stronger blocking silently moves to another passage that happens to pass."""
    replay = runpy.run_path(str(SCRIPT))["replay_sample"]
    baseline = _baseline_by_id()

    for candidate_id in KNOWN_IDS:
        row = baseline[candidate_id]
        with np.load(BASELINE / row["stroke_archive"]["path"], allow_pickle=False) as data:
            layers = tuple(
                data[f"before_{index:03}"].copy() for index in range(row["original_stroke_count"])
            )
        result = replay(layers, row)
        if result is None:
            continue
        assert list(result.selected_stroke_indices) == row["selected_stroke_indices"]
        assert result.bridge_mode == "block_gap"
        for key in SITE_KEYS:
            assert result.metrics[key] == row["metrics"][key]
        assert len(result.edited_layers) == len(layers) + 1
        for actual, original in zip(result.edited_layers, layers, strict=False):
            np.testing.assert_array_equal(actual, original)


def test_comparison_records_every_known_site_as_pass_or_skip(generated) -> None:
    """Break caught: failed old gates disappear, get replacement samples, or count as success."""
    output, run, before = generated
    pairs = _rows(output / "pairs.jsonl")
    candidates = _rows(output / "candidates.jsonl")

    assert [pair["baseline_candidate_id"] for pair in pairs] == list(KNOWN_IDS)
    assert {pair["status"] for pair in pairs} <= {"accepted", "skipped"}
    assert run["slot_count"] == 5
    assert run["candidate_count"] + run["skipped_count"] == 5
    assert run["complete"] is (run["skipped_count"] == 0)
    assert len(candidates) == run["candidate_count"]
    assert {row["baseline_candidate_id"] for row in candidates} == {
        pair["baseline_candidate_id"] for pair in pairs if pair["status"] == "accepted"
    }
    for pair in pairs:
        assert pair["archived_site"] == pair["replayed_site"]
        if pair["status"] == "skipped":
            assert pair["candidate_id"] is None
            assert "new_96" not in pair["files"]
            assert pair["reason"].endswith("no replacement")
        else:
            assert pair["candidate_id"]
            assert "new_96" in pair["files"]
    assert before == _hashes(BASELINE)
    assert not list(output.rglob("*.parquet"))


def test_accepted_candidates_are_review_only_licensed_and_replayable(generated) -> None:
    """Break caught: comparison images become labels or omit audited views/layer metadata."""
    output, run, _ = generated
    rows = _rows(output / "candidates.jsonl")
    expected_views = {
        "original",
        "candidate",
        "selected_strokes",
        "mask",
        "original_96",
        "candidate_96",
        "mask_96",
        "foreground_96",
        "edges_96",
    }
    for row in rows:
        assert row["decision"] == "REVIEW"
        assert row["training_eligible"] is False
        assert row["edited_stroke_count"] == row["original_stroke_count"] + 1
        assert set(row["files"]) == expected_views
        for info in row["files"].values():
            path = output / info["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
            with Image.open(path) as image:
                assert image.info["License"] == "Arphic-1999; see ARPHICPL.txt"
                assert image.info["Modification"] == row["modification"]
        archive = output / row["stroke_archive"]["path"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["stroke_archive"]["sha256"]
        with np.load(archive, allow_pickle=False) as data:
            assert str(data["license"]) == "Arphic-1999"
            assert str(data["modification"]) == row["modification"]
            after = [data[key] for key in sorted(data.files) if key.startswith("after_")]
            candidate = np.repeat(np.maximum.reduce(after)[:, :, None], 3, axis=2)
        assert hashlib.sha256(candidate.tobytes()).hexdigest() == row["pixel_sha256"]
    assert len(rows) == run["candidate_count"]


def test_comparison_pairs_exact_input_and_source_hashes(generated) -> None:
    """Break caught: run metadata claims provenance unrelated to the bytes actually replayed."""
    output, run, _ = generated
    baseline = _baseline_by_id()
    pairs = _rows(output / "pairs.jsonl")

    for name, digest in run["baseline_files_sha256"].items():
        assert hashlib.sha256((BASELINE / name).read_bytes()).hexdigest() == digest
    for name, digest in run["source_code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == run["replay_code_sha256"]
    for pair in pairs:
        row = baseline[pair["baseline_candidate_id"]]
        assert pair["baseline_archive_sha256"] == row["stroke_archive"]["sha256"]
        assert pair["baseline_original_96_sha256"] == row["files"]["original_96"]["sha256"]
        assert pair["baseline_old_96_sha256"] == row["files"]["candidate_96"]["sha256"]


def test_overview_has_three_ascii_headers_and_numeric_only_row_labels(generated) -> None:
    """Break caught: CJK fallback changes the comparison grid or reveals character labels."""
    output, _, _ = generated
    with Image.open(output / "overview.png") as image:
        actual = np.asarray(image.convert("RGB"))
    assert actual.shape == (32 + 5 * 124, 428, 3)

    expected_header = Image.new("RGB", (428, 32), "white")
    draw = ImageDraw.Draw(expected_header)
    for x, label in ((76, "Original"), (188, "Old"), (300, "New")):
        draw.text((x, 8), label, fill="black")
    np.testing.assert_array_equal(actual[:32], np.asarray(expected_header))

    expected_label = Image.new("RGB", (60, 124), "white")
    ImageDraw.Draw(expected_label).text((4, 6), "01", fill="black")
    np.testing.assert_array_equal(actual[32 : 32 + 124, :60], np.asarray(expected_label))


def test_comparison_rejects_a_tampered_archived_site(tmp_path: Path) -> None:
    """Break caught: replay proceeds after the immutable before-layer archive changes."""
    generate = runpy.run_path(str(SCRIPT))["generate_gap_comparison"]
    baseline = tmp_path / "baseline"
    shutil.copytree(BASELINE, baseline)
    row = _baseline_by_id()[KNOWN_IDS[0]]
    (baseline / row["stroke_archive"]["path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        generate(baseline, tmp_path / "comparison")


def test_comparison_requires_a_fresh_output_directory(generated) -> None:
    """Break caught: regenerating the calibration overwrites existing review work."""
    output, _, _ = generated
    generate = runpy.run_path(str(SCRIPT))["generate_gap_comparison"]
    with pytest.raises(FileExistsError):
        generate(BASELINE, output)
