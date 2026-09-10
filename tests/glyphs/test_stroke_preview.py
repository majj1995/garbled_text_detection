"""Stroke previews are licensed, reproducible review artifacts, never training labels."""

import hashlib
import importlib
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


def _module() -> Any:
    name = "poor_word.glyphs.stroke_preview"
    assert importlib.util.find_spec(name) is not None, "stroke preview implementation is missing"
    return importlib.import_module(name)


@pytest.mark.parametrize(
    "override",
    [
        {"characters": ()},
        {"characters": ("日", "日")},
        {"characters": ("日月",)},
        {"characters": (" ",)},
        {"per_operator": 0},
        {"per_operator": 101},
        {"max_attempts_per_slot": 0},
        {"seed": -1},
    ],
)
def test_invalid_config_cannot_start_a_preview(tmp_path: Path, override: dict[str, Any]) -> None:
    """Catch empty/fake character pools or an unbounded/empty generation request."""
    module = _module()
    with pytest.raises(ValueError):
        module.StrokePreviewConfig(
            **{
                "output_dir": tmp_path / "preview",
                "graphics_path": tmp_path / "graphics.txt",
                "source_lock_path": tmp_path / "source.lock.json",
                "license_path": tmp_path / "APL.txt",
                "characters": ("日",),
                **override,
            }
        )


def test_export_keeps_exact_pixels_selected_strokes_and_png_notice(tmp_path: Path) -> None:
    """Catch wrong masks, truncated selected strokes, or absent APL modification notices."""
    original = np.zeros((128, 128, 3), dtype=np.uint8)
    original[55:66, 20:100] = 255
    candidate = original.copy()
    candidate[20:100, 55:66] = 255
    selected = original[:, :, 0].copy()
    notice = "2026-09-10: rendered and modified by add_stroke; selected source stroke 0."
    files = _module().save_stroke_assets(tmp_path, "p001", original, candidate, selected, notice)
    assert len(files) == 9
    assert np.array_equal(np.asarray(Image.open(tmp_path / files["candidate"]["path"])), candidate)
    assert np.array_equal(
        np.asarray(Image.open(tmp_path / files["selected_strokes"]["path"])), selected
    )
    assert np.array_equal(
        np.asarray(Image.open(tmp_path / files["mask"]["path"])) > 0,
        np.any(original != candidate, axis=2),
    )
    for info in files.values():
        path = tmp_path / info["path"]
        with Image.open(path) as saved:
            assert saved.info["Modification"] == notice
            assert saved.info["License"] == "Arphic-1999; see ARPHICPL.txt"
            assert "Arphic" in saved.info["Copyright"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]


def test_stroke_export_refuses_to_overwrite_selected_asset(tmp_path: Path) -> None:
    """Catch an extra stroke-specific asset bypassing the old exporter's no-overwrite guard."""
    module = _module()
    directory = tmp_path / "images"
    directory.mkdir()
    marker = directory / "p001-selected_strokes.png"
    marker.write_bytes(b"human asset")
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.raises(FileExistsError):
        module.save_stroke_assets(tmp_path, "p001", rgb, rgb, rgb[:, :, 0], "test modification")
    assert marker.read_bytes() == b"human asset"
    assert len(list(directory.iterdir())) == 1


@pytest.fixture
def config(tmp_path: Path) -> Any:
    """Use a small real-source subset, preserving the published SVG/median schema."""
    from poor_word.data.manifest import SourceLock

    repo = Path(__file__).resolve().parents[2]
    characters = tuple("永明田林国回合蛤日木困井一")
    wanted = set(characters)
    source = tmp_path / "graphics.txt"
    with (repo / "data/raw/makemeahanzi_graphics.txt").open(encoding="utf-8") as stream:
        selected = [line for line in stream if json.loads(line)["character"] in wanted]
    source.write_text("".join(selected), encoding="utf-8")
    payload = source.read_bytes()
    lock = SourceLock(
        source_id="makemeahanzi_graphics",
        declared_url="https://example.org/graphics.txt",
        resolved_url="https://example.org/graphics.txt",
        output_name="graphics.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="Arphic-1999",
        production_allowed=False,
    )
    lock_path = tmp_path / "graphics.lock.json"
    lock_path.write_text(lock.model_dump_json(), encoding="utf-8")
    return _module().StrokePreviewConfig(
        output_dir=tmp_path / "preview",
        graphics_path=source,
        source_lock_path=lock_path,
        license_path=repo / "data/licenses/makemeahanzi.ARPHICPL.txt",
        characters=characters,
        per_operator=1,
        seed=19,
    )


def test_real_stroke_preview_is_review_only_and_auditable(config: Any, tmp_path: Path) -> None:
    """Catch missing quotas, hidden training labels, or archives that cannot reproduce the PNG."""
    module = _module()
    progress: list[str] = []
    first = module.generate_stroke_preview(config, progress=progress.append)
    second = module.generate_stroke_preview(
        config.model_copy(update={"output_dir": tmp_path / "second"})
    )
    assert first.complete and first.candidate_count == 5
    rows = [json.loads(line) for line in first.candidates_path.read_text().splitlines()]
    other = [json.loads(line) for line in second.candidates_path.read_text().splitlines()]
    assert len({row["operator"] for row in rows}) == 5
    assert [row["candidate_id"] for row in rows] == [row["candidate_id"] for row in other]
    assert (config.output_dir / "ARPHICPL.txt").read_bytes() == config.license_path.read_bytes()
    assert (config.output_dir / "review-template.jsonl").is_file()
    assert not list(config.output_dir.rglob("*.parquet"))
    assert any("preview=5/5" in line for line in progress)
    run = json.loads((config.output_dir / "run.json").read_text())
    assert run["provenance"]["packages"]["numpy"] == np.__version__
    assert {"fonttools", "aggdraw", "pillow"} <= run["provenance"]["packages"].keys()
    with Image.open(first.overview_path) as overview:
        assert overview.info["License"] == "Arphic-1999; see ARPHICPL.txt"
    for row in rows:
        assert row["decision"] == "REVIEW" and row["training_eligible"] is False
        assert row["label_provenance"] == "synthetic_stroke_candidate_unreviewed"
        assert row["selected_stroke_indices"]
        for info in [*row["files"].values(), row["stroke_archive"]]:
            path = config.output_dir / info["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        with np.load(config.output_dir / row["stroke_archive"]["path"], allow_pickle=False) as data:
            after = [data[key] for key in sorted(data.files) if key.startswith("after_")]
            expected = np.maximum.reduce(after)
            assert np.array_equal(
                np.asarray(Image.open(config.output_dir / row["files"]["candidate"]["path"]))[
                    :, :, 0
                ],
                expected,
            )
            assert str(data["license"]) == "Arphic-1999"

    class OfflinePage(HTMLParser):
        image_count = 0
        details_count = 0

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            values = dict(attrs)
            assert tag not in {"script", "iframe", "object", "embed"}
            if tag == "details":
                self.details_count += 1
                assert "open" not in values
            if tag == "img":
                self.image_count += 1
            for key in ("src", "href"):
                if key in values:
                    target = values[key]
                    assert target and ":" not in target and not target.startswith("/")
                    assert (config.output_dir / target).is_file()

    page = OfflinePage()
    page.feed(first.html_path.read_text())
    assert page.image_count == 45 and page.details_count == 5


def test_existing_stroke_review_notes_are_not_overwritten(config: Any) -> None:
    """Catch regeneration destroying user review work."""
    config.output_dir.mkdir()
    marker = config.output_dir / "human.txt"
    marker.write_text("keep my review")
    with pytest.raises(FileExistsError):
        _module().generate_stroke_preview(config)
    assert marker.read_text() == "keep my review"


def test_graphics_hash_mismatch_is_rejected_before_output(config: Any) -> None:
    """Catch silent drift from the declared public stroke source."""
    config.graphics_path.write_bytes(config.graphics_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source"):
        _module().generate_stroke_preview(config)
    assert not config.output_dir.exists()


def test_missing_license_blocks_distribution_before_output(config: Any, tmp_path: Path) -> None:
    """Catch unlicensed derived images being published with a successful result."""
    with pytest.raises(FileNotFoundError):
        _module().generate_stroke_preview(
            config.model_copy(update={"license_path": tmp_path / "missing.txt"})
        )
    assert not config.output_dir.exists()


def test_incomplete_stroke_preview_never_fills_with_weak_fallback(config: Any) -> None:
    """Catch falling back to V1/V2 pixel edits when a whole-stroke operation is inapplicable."""
    config = config.model_copy(update={"characters": ("一",), "max_attempts_per_slot": 1})
    result = _module().generate_stroke_preview(config)
    assert not result.complete and result.candidate_count < 5
    run = json.loads((config.output_dir / "run.json").read_text())
    assert run["missing_count"] > 0 and run["skipped_count"] > 0


def test_failed_stroke_export_does_not_leave_partial_output(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch exposing a partial preview or leaving owned staging data after a disk error."""
    module = _module()

    def disk_full(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(module, "save_stroke_assets", disk_full)
    with pytest.raises(OSError, match="disk full"):
        module.generate_stroke_preview(config)
    assert not config.output_dir.exists()
    assert not list(config.output_dir.parent.glob(".preview-*"))


def test_stroke_cli_runs_real_review_pipeline(config: Any) -> None:
    """Catch wiring the new command to a legacy auto-labelled glyph generator."""
    from typer.testing import CliRunner

    from poor_word.cli import app

    result = CliRunner().invoke(
        app,
        [
            "glyphs",
            "preview-strokes",
            "--output-dir",
            str(config.output_dir),
            "--graphics",
            str(config.graphics_path),
            "--source-lock",
            str(config.source_lock_path),
            "--license",
            str(config.license_path),
            "--characters",
            "".join(config.characters),
            "--per-operator",
            "1",
            "--seed",
            "19",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (config.output_dir / "candidates.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 5 and all(row["decision"] == "REVIEW" for row in rows)
    assert "preview=5/5" in result.output


def test_stroke_cli_reports_unfilled_quotas_without_success(config: Any) -> None:
    """Catch a zero/partial candidate run being reported as a complete preview."""
    from typer.testing import CliRunner

    from poor_word.cli import app

    result = CliRunner().invoke(
        app,
        [
            "glyphs",
            "preview-strokes",
            "--output-dir",
            str(config.output_dir),
            "--graphics",
            str(config.graphics_path),
            "--source-lock",
            str(config.source_lock_path),
            "--license",
            str(config.license_path),
            "--characters",
            "一",
            "--per-operator",
            "1",
            "--max-attempts-per-slot",
            "1",
        ],
    )
    assert result.exit_code == 2, result.output
    assert json.loads((config.output_dir / "run.json").read_text())["complete"] is False
