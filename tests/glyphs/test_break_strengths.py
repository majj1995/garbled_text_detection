"""The 1x/1.15x/2x archive compares fixed attempts without replacements."""

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs/previews/glyph-breaks-20260910"
PRIOR = ROOT / "docs/previews/glyph-breaks-longer-20260910"
SCRIPT = ROOT / "docs/previews/compare_break_strengths.py"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _hashes(directory: Path) -> dict[Path, str]:
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory):
    generate = runpy.run_path(str(SCRIPT))["generate_strength_comparison"]
    before = (_hashes(BASELINE), _hashes(PRIOR))
    output = tmp_path_factory.mktemp("break-strengths") / "comparison"
    run = generate(BASELINE, PRIOR, output)
    return output, run, before


def test_double_cut_replay_isolates_the_archived_stroke_and_site() -> None:
    replay = runpy.run_path(str(SCRIPT))["replay_sample"]
    row = next(row for row in _rows(BASELINE / "candidates.jsonl") if row["base_char"] == "寸")
    with np.load(BASELINE / row["stroke_archive"]["path"], allow_pickle=False) as data:
        layers = tuple(
            data[f"before_{index:03}"].copy() for index in range(row["original_stroke_count"])
        )
    result = replay(layers, row, length_multiplier=2.0)
    assert result is not None
    assert list(result.selected_stroke_indices) == row["selected_stroke_indices"]
    assert result.metrics["break_center_x_96"] == row["metrics"]["break_center_x_96"]
    assert result.metrics["break_center_y_96"] == row["metrics"]["break_center_y_96"]
    assert result.metrics["local_stroke_width"] == row["metrics"]["local_stroke_width"]
    assert result.metrics["gap_length"] == pytest.approx(row["metrics"]["gap_length"] * 2)
    selected = result.selected_stroke_indices[0]
    for index, layer in enumerate(layers):
        if index != selected:
            np.testing.assert_array_equal(result.edited_layers[index], layer)


def test_comparison_keeps_all_original_slots_and_publishes_only_fixed_attempt_passes(
    generated,
) -> None:
    output, run, before = generated
    assert run["slot_count"] == 20
    assert run["double_candidate_count"] == 7
    assert run["double_pixel_changed_vs_one_x_count"] == 7
    assert run["double_pixel_unchanged_vs_one_x_count"] == 0
    assert run["double_skipped_count"] == 13
    assert run["prior_1_15_candidate_count"] == 17
    assert run["prior_1_15_pixel_changed_count"] == 16
    assert run["prior_1_15_pixel_unchanged_count"] == 1

    baseline_rows = _rows(BASELINE / "candidates.jsonl")
    prior_pairs = _rows(PRIOR / "pairs.jsonl")
    pairs = _rows(output / "pairs.jsonl")
    assert [pair["base_char"] for pair in pairs] == [row["base_char"] for row in baseline_rows]
    assert len(pairs) == 20
    assert {pair["base_char"] for pair in pairs if pair["double_status"] == "skipped"} == {
        "貌",
        "锌",
        "呕",
        "蛤",
        "煌",
        "拎",
        "脖",
        "叛",
        "疲",
        "宝",
        "班",
        "佐",
        "堰",
    }
    for baseline, prior, pair in zip(baseline_rows, prior_pairs, pairs, strict=True):
        assert pair["baseline_candidate_id"] == baseline["candidate_id"]
        assert pair["seed"] == baseline["seed"]
        assert pair["selected_stroke_indices"] == baseline["selected_stroke_indices"]
        assert pair["prior_1_15_status"] == prior["status"]
        for copied_key, source, source_key in (
            ("original_96", baseline, "original_96"),
            ("one_x_96", baseline, "candidate_96"),
            ("one_x_mask_96", baseline, "mask_96"),
        ):
            copied = output / pair["files"][copied_key]["path"]
            original = BASELINE / source["files"][source_key]["path"]
            assert copied.read_bytes() == original.read_bytes()
        if prior["status"] == "accepted":
            copied = output / pair["files"]["prior_1_15_96"]["path"]
            original = PRIOR / prior["files"]["new_96"]["path"]
            assert copied.read_bytes() == original.read_bytes()

    neck = next(pair for pair in pairs if pair["base_char"] == "脖")
    assert neck["prior_1_15_pixel_changed"] is False
    assert neck["double_status"] == "skipped"
    assert before == (_hashes(BASELINE), _hashes(PRIOR))
    assert not list(output.rglob("*.parquet"))


def test_every_double_candidate_is_review_only_with_nine_views_and_replayable_layers(
    generated,
) -> None:
    output, _, _ = generated
    rows = _rows(output / "candidates.jsonl")
    assert len(rows) == 7
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
        assert row["length_multiplier"] == 2.0
        assert set(row["files"]) == expected_views
        assert all(
            hashlib.sha256((output / info["path"]).read_bytes()).hexdigest() == info["sha256"]
            for info in row["files"].values()
        )
        archive = output / row["stroke_archive"]["path"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["stroke_archive"]["sha256"]
        with np.load(archive, allow_pickle=False) as data:
            count = row["original_stroke_count"]
            assert str(data["source_char"]) == row["base_char"]
            assert str(data["license"]) == "Arphic-1999"
            assert all(f"before_{index:03}" in data for index in range(count))
            assert all(f"after_{index:03}" in data for index in range(count))
            recomposed = np.repeat(
                np.maximum.reduce([data[f"after_{index:03}"] for index in range(count)])[
                    :, :, None
                ],
                3,
                axis=2,
            )
        assert hashlib.sha256(recomposed.tobytes()).hexdigest() == row["pixel_sha256"]


def test_overview_has_ascii_column_headers_and_numeric_only_row_labels(generated) -> None:
    output, _, _ = generated
    with Image.open(output / "overview.png") as image:
        actual = np.asarray(image.convert("RGB"))
    assert actual.shape == (32 + 20 * 124, 540, 3)

    expected_header = Image.new("RGB", (540, 32), "white")
    header_draw = ImageDraw.Draw(expected_header)
    for x, label in ((76, "Original"), (188, "1x"), (300, "1.15x"), (412, "2x")):
        header_draw.text((x, 8), label, fill="black")
    np.testing.assert_array_equal(actual[:32], np.asarray(expected_header))

    expected_label = Image.new("RGB", (60, 124), "white")
    ImageDraw.Draw(expected_label).text((4, 6), "01", fill="black")
    np.testing.assert_array_equal(actual[32 : 32 + 124, :60], np.asarray(expected_label))


def test_comparison_rejects_a_tampered_prior_strength_asset(tmp_path: Path) -> None:
    generate = runpy.run_path(str(SCRIPT))["generate_strength_comparison"]
    prior = tmp_path / "prior"
    shutil.copytree(PRIOR, prior)
    pair = _rows(prior / "pairs.jsonl")[0]
    asset = prior / pair["files"]["new_96"]["path"]
    asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        generate(BASELINE, prior, tmp_path / "comparison")


def test_comparison_requires_a_fresh_output_directory(generated) -> None:
    output, _, _ = generated
    generate = runpy.run_path(str(SCRIPT))["generate_strength_comparison"]
    with pytest.raises(FileExistsError):
        generate(BASELINE, PRIOR, output)
