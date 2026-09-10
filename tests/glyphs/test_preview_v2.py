"""Preview candidates must be balanced and isolated from training labels."""

import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


def _preview() -> Any:
    module_name = "poor_word.glyphs.preview_v2"
    assert importlib.util.find_spec(module_name) is not None, "V2 preview is missing"
    return importlib.import_module(module_name)


def test_preview_slots_cover_all_types_levels_and_fonts(tmp_path: Path) -> None:
    """Catch selecting all strong or single-font examples while claiming balanced preview."""
    module = _preview()
    config = module.PreviewConfig(
        output_dir=tmp_path / "preview",
        characters=("A", "B"),
        font_paths=(Path("sans.otf"), Path("serif.otf")),
    )
    slots = module.preview_slots(config)
    assert len(slots) == 50
    assert Counter(slot.operator for slot in slots) == {
        "add_stroke": 10,
        "break_stroke": 10,
        "bridge": 10,
        "component_shift": 10,
        "erase_segment": 10,
    }
    assert Counter(slot.severity for slot in slots) == {"medium": 25, "strong": 25}
    for operator in {slot.operator for slot in slots}:
        selected = [slot for slot in slots if slot.operator == operator]
        assert {slot.font_index for slot in selected} == {0, 1}
        for font_index in (0, 1):
            assert {s.severity for s in selected if s.font_index == font_index} == {
                "medium",
                "strong",
            }


@pytest.mark.parametrize(
    "override",
    [
        {"characters": ()},
        {"characters": ("AB",)},
        {"characters": ("A", "A")},
        {"font_paths": ()},
        {"font_paths": (Path("same.otf"), Path("same.otf"))},
        {"per_operator": 0},
        {"max_attempts_per_slot": 0},
        {"seed": -1},
    ],
)
def test_invalid_preview_configuration_is_rejected(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    """Catch accepting empty pools, fake font diversity, or unbounded/empty attempt budgets."""
    values = {
        "output_dir": tmp_path / "preview",
        "characters": ("A",),
        "font_paths": (None,),
        **override,
    }
    with pytest.raises(ValueError):
        _preview().PreviewConfig(**values)


def test_preview_slot_order_is_reproducible(tmp_path: Path) -> None:
    """Catch global RNG state changing the declared quota plan."""
    module = _preview()
    config = module.PreviewConfig(
        output_dir=tmp_path / "preview",
        characters=("A",),
        font_paths=(None,),
    )
    assert module.preview_slots(config) == module.preview_slots(config)


def test_preview_images_preserve_pixels_and_mask_actual_changes(tmp_path: Path) -> None:
    """Catch a copied/dilated diagnostic mask disagreeing with actual RGB changes."""
    original = np.zeros((128, 128, 3), dtype=np.uint8)
    original[15:110, 40:55] = 255
    candidate = original.copy()
    candidate[40:60, 45:80] = (180, 200, 220)
    files = _preview().save_preview_images(tmp_path, "p001", original, candidate)
    assert np.array_equal(np.asarray(Image.open(tmp_path / files["original"]["path"])), original)
    assert np.array_equal(np.asarray(Image.open(tmp_path / files["candidate"]["path"])), candidate)
    expected = np.zeros((128, 128), dtype=np.uint8)
    expected[40:60, 45:80] = 255
    assert np.array_equal(np.asarray(Image.open(tmp_path / files["mask"]["path"])), expected)
    for info in files.values():
        assert hashlib.sha256((tmp_path / info["path"]).read_bytes()).hexdigest() == info["sha256"]


def test_preview_96_views_match_existing_training_preprocessing(tmp_path: Path) -> None:
    """Catch showing the user a different resize/foreground/edge view than the model consumes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from poor_word.training.dataset import GlyphDataset

    original = np.zeros((128, 128, 3), dtype=np.uint8)
    candidate = original.copy()
    candidate[15:90, 40:65] = (130, 190, 250)
    files = _preview().save_preview_images(tmp_path, "p001", original, candidate)
    manifest = tmp_path / "fixture.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "p001",
                    "image_path": files["candidate"]["path"],
                    "mask_path": files["mask"]["path"],
                    "base_char": "A",
                    "decision": "PASS",
                }
            ]
        ),
        manifest,
    )
    views = GlyphDataset(manifest)[0].views.numpy()
    for channel, key in enumerate(("candidate_96", "foreground_96", "edges_96")):
        actual = np.asarray(Image.open(tmp_path / files[key]["path"]))
        assert np.array_equal(actual, np.rint(views[channel] * 255).astype(np.uint8))


@pytest.mark.parametrize("prefix", ["../escape", "/absolute", "a/b"])
def test_preview_image_paths_cannot_escape_output(tmp_path: Path, prefix: str) -> None:
    """Catch candidate IDs being interpreted as filesystem paths."""
    original = np.zeros((128, 128, 3), dtype=np.uint8)
    candidate = np.full_like(original, 255)
    with pytest.raises(ValueError):
        _preview().save_preview_images(tmp_path, prefix, original, candidate)
    assert not list(tmp_path.iterdir())


def test_preview_images_never_overwrite_existing_asset(tmp_path: Path) -> None:
    """Catch a repeated export silently replacing an earlier review image."""
    original = np.zeros((128, 128, 3), dtype=np.uint8)
    candidate = np.full_like(original, 255)
    module = _preview()
    files = module.save_preview_images(tmp_path, "p001", original, candidate)
    path = tmp_path / files["original"]["path"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.save_preview_images(tmp_path, "p001", candidate, original)
    assert path.read_bytes() == before


def test_preview_run_exports_review_only_records_reproducibly(tmp_path: Path) -> None:
    """Catch automatic BLOCK labels, training manifests, or nondeterministic candidate selection."""
    module = _preview()
    config = module.PreviewConfig(
        output_dir=tmp_path / "first",
        characters=tuple("ABEFHX=:%ij"),
        font_paths=(None,),
        per_operator=2,
        seed=19,
        max_attempts_per_slot=64,
    )
    progress: list[str] = []
    first = module.generate_preview(config, progress=progress.append)
    second = module.generate_preview(config.model_copy(update={"output_dir": tmp_path / "second"}))
    rows = [json.loads(line) for line in first.candidates_path.read_text().splitlines()]
    assert first.complete and first.candidate_count == 10
    assert first.candidates_path.read_bytes() == second.candidates_path.read_bytes()
    assert len({row["candidate_id"] for row in rows}) == 10
    assert set(Counter(row["operator"] for row in rows).values()) == {2}
    assert all(row["decision"] == "REVIEW" and row["training_eligible"] is False for row in rows)
    assert all(row["label_provenance"] == "synthetic_candidate_unreviewed" for row in rows)
    assert not list(config.output_dir.glob("*.parquet"))
    assert first.html_path.is_file() and first.overview_path.is_file()
    assert (config.output_dir / "review-template.jsonl").is_file()
    assert (config.output_dir / "run.json").is_file()
    assert progress
    for row in rows:
        for info in row["files"].values():
            path = config.output_dir / info["path"]
            assert (
                path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
            )


def test_preview_run_never_replaces_old_output(tmp_path: Path) -> None:
    """Catch existing review notes being lost when rerunning a command."""
    module = _preview()
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "notes.txt"
    marker.write_text("human review")
    config = module.PreviewConfig(output_dir=destination, characters=("A",), font_paths=(None,))
    with pytest.raises(FileExistsError):
        module.generate_preview(config)
    assert marker.read_text() == "human review"


def test_missing_font_is_reported_before_creating_output(tmp_path: Path) -> None:
    """Catch silently falling back to a font with no Chinese support."""
    module = _preview()
    config = module.PreviewConfig(
        output_dir=tmp_path / "preview",
        characters=("A",),
        font_paths=(tmp_path / "absent.otf",),
    )
    with pytest.raises(FileNotFoundError):
        module.generate_preview(config)
    assert not config.output_dir.exists()


def test_unsupported_glyphs_leave_an_explicit_incomplete_preview(tmp_path: Path) -> None:
    """Catch tofu glyphs or forced invalid candidates used to fill quotas."""
    module = _preview()
    config = module.PreviewConfig(
        output_dir=tmp_path / "preview",
        characters=("\U0010ffff",),
        font_paths=(None,),
        per_operator=1,
        max_attempts_per_slot=2,
    )
    result = module.generate_preview(config)
    assert result.complete is False and result.candidate_count == 0
    run = json.loads((config.output_dir / "run.json").read_text())
    assert run["attempted_count"] == 10
    assert run["missing_count"] == 5
    assert run["skipped_count"] == 10
    assert result.candidates_path.read_text() == ""


def test_blank_glyph_is_skipped_not_a_partially_published_run(tmp_path: Path) -> None:
    """Catch zero-width/missing font glyphs aborting all other candidate work."""
    module = _preview()
    font = Path(__file__).resolve().parents[2] / "data/raw/NotoSansCJKsc-Regular.otf"
    config = module.PreviewConfig(
        output_dir=tmp_path / "preview",
        characters=("\u200b",),
        font_paths=(font,),
        per_operator=1,
        max_attempts_per_slot=1,
    )
    result = module.generate_preview(config)
    assert not result.complete and result.candidate_count == 0


def test_preview_cleanup_after_failed_image_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch leaking partial review directories after a disk/write failure."""
    module = _preview()
    config = module.PreviewConfig(
        output_dir=tmp_path / "preview",
        characters=tuple("ABEFHX=:%ij"),
        font_paths=(None,),
        per_operator=1,
        seed=19,
    )

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(module, "save_preview_images", fail_write)
    with pytest.raises(OSError, match="disk full"):
        module.generate_preview(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".preview-*"))


def test_preview_cli_writes_real_chinese_review_candidates(tmp_path: Path) -> None:
    """Catch a CLI wired to v1 auto-BLOCK generation or the wrong font/quotas."""
    from typer.testing import CliRunner

    from poor_word.cli import app

    font = Path(__file__).resolve().parents[2] / "data/raw/NotoSansCJKsc-Regular.otf"
    destination = tmp_path / "preview"
    result = CliRunner().invoke(
        app,
        [
            "glyphs",
            "preview-v2",
            "--font",
            str(font),
            "--characters",
            "天地合蛤明口日田木本林森水永王玉中字",
            "--per-operator",
            "2",
            "--seed",
            "19",
            "--output-dir",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line) for line in (destination / "candidates.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 10 and all(row["decision"] == "REVIEW" for row in rows)
    assert (destination / "index.html").is_file()
    assert "preview=10/10" in result.output


def test_preview_cli_incomplete_run_returns_nonzero_with_audit(tmp_path: Path) -> None:
    """Catch claiming a successful 50-sample preview after exhausting unsuitable glyphs."""
    from typer.testing import CliRunner

    from poor_word.cli import app

    font = Path(__file__).resolve().parents[2] / "data/raw/NotoSansCJKsc-Regular.otf"
    destination = tmp_path / "preview"
    result = CliRunner().invoke(
        app,
        [
            "glyphs",
            "preview-v2",
            "--font",
            str(font),
            "--characters",
            "\U0010ffff",
            "--per-operator",
            "1",
            "--max-attempts-per-slot",
            "1",
            "--output-dir",
            str(destination),
        ],
    )
    assert result.exit_code == 2
    assert json.loads((destination / "run.json").read_text())["complete"] is False


def test_candidate_ids_do_not_depend_on_font_install_directory(tmp_path: Path) -> None:
    """Catch server and developer previews getting different IDs for identical fonts and pixels."""
    module = _preview()
    source = Path(__file__).resolve().parents[2] / "data/raw/NotoSansCJKsc-Regular.otf"
    first_font, second_font = tmp_path / "first.otf", tmp_path / "second.otf"
    first_font.symlink_to(source)
    second_font.symlink_to(source)
    config = module.PreviewConfig(
        output_dir=tmp_path / "first",
        characters=tuple("合蛤日田木林水王口"),
        font_paths=(first_font,),
        per_operator=1,
        seed=19,
    )
    first = module.generate_preview(config)
    second = module.generate_preview(
        config.model_copy(
            update={
                "output_dir": tmp_path / "second",
                "font_paths": (second_font,),
            }
        )
    )
    first_rows = [json.loads(line) for line in first.candidates_path.read_text().splitlines()]
    second_rows = [json.loads(line) for line in second.candidates_path.read_text().splitlines()]
    assert first.complete and second.complete
    assert [row["files"]["candidate"]["sha256"] for row in first_rows] == [
        row["files"]["candidate"]["sha256"] for row in second_rows
    ]
    assert [row["candidate_id"] for row in first_rows] == [
        row["candidate_id"] for row in second_rows
    ]
