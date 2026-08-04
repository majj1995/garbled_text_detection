from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import poor_word.cli as cli
from poor_word.evaluation.report import ReportArtifacts
from poor_word.ocr.types import OcrAudit

runner = CliRunner()


def test_data_lock_and_fetch_commands_delegate_to_locked_downloader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = SimpleNamespace(source_id="fixture")
    lock = SimpleNamespace(
        sha256="a" * 64,
        size_bytes=12,
        license_id="OFL-1.1",
        resolved_url="https://example.invalid/font.otf",
    )
    fetched = tmp_path / "font.otf"
    monkeypatch.setattr(cli, "_get_source", lambda *_args: source)
    monkeypatch.setattr(cli, "lock_source", lambda *_args, **_kwargs: lock)
    lock_result = runner.invoke(cli.app, ["data", "lock", "--source-id", "fixture"])
    assert lock_result.exit_code == 0
    assert "license=OFL-1.1" in lock_result.stdout

    monkeypatch.setattr(cli, "load_source_lock", lambda _path: lock)
    monkeypatch.setattr(cli, "fetch_locked_source", lambda *_args: fetched)
    fetch_result = runner.invoke(cli.app, ["data", "fetch", "--source-id", "fixture"])
    assert fetch_result.exit_code == 0
    assert str(fetched) in fetch_result.stdout


def test_get_source_rejects_unknown_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_source_specs", lambda _path: ())
    with pytest.raises(Exception, match="unknown source_id"):
        cli._get_source("missing", Path("sources.toml"))


def test_glyph_generation_command_writes_run_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.parquet"
    run = tmp_path / "run.json"
    monkeypatch.setattr(cli, "load_common_chars", lambda _path: tuple("天地玄黄宇宙洪荒日月"))
    monkeypatch.setattr(cli, "generate_dataset", lambda _config: manifest)
    monkeypatch.setattr(cli, "_write_run_metadata", lambda *_args: run)

    result = runner.invoke(
        cli.app,
        ["glyphs", "generate", "--profile", "smoke", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert f"manifest={manifest}" in result.stdout
    assert f"run={run}" in result.stdout


def test_ocr_train_and_evaluate_commands_emit_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "glyph.png").write_bytes(b"png")
    audit = OcrAudit(
        paddleocr_version="3.2.0",
        paddlepaddle_version="3.1.1",
        cuda_version="12.6",
        gpu_name="NVIDIA L20",
        detection_model_name="PP-OCRv5_server_det",
        recognition_model_name="PP-OCRv5_server_rec",
        character_boxes_available=True,
        logits_available=False,
        latency_p50_ms=30,
        latency_p95_ms=50,
        required_capability_gaps=("raw_logits_adapter",),
    )
    backend = SimpleNamespace(audit=lambda *_args, **_kwargs: audit)
    monkeypatch.setattr(cli, "PaddleV5Adapter", lambda **_kwargs: backend)
    audit_path = tmp_path / "audit.json"
    audit_result = runner.invoke(
        cli.app,
        [
            "ocr",
            "audit",
            "--image-dir",
            str(image_dir),
            "--output",
            str(audit_path),
            "--runs",
            "1",
        ],
    )
    assert audit_result.exit_code == 0
    assert audit_path.exists()

    train_paths = SimpleNamespace(
        checkpoint=tmp_path / "encoder.pt",
        prototype_bank=tmp_path / "prototypes.npz",
        metrics=tmp_path / "metrics.json",
    )
    monkeypatch.setattr(cli, "train_glyph", lambda _config: train_paths)
    train_result = runner.invoke(
        cli.app,
        [
            "train",
            "glyph",
            "--manifest",
            str(tmp_path / "manifest.parquet"),
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )
    assert train_result.exit_code == 0
    assert f"checkpoint={train_paths.checkpoint}" in train_result.stdout

    report_paths = ReportArtifacts(
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
    )
    monkeypatch.setattr(cli, "evaluate_glyph_artifacts", lambda *_args, **_kwargs: report_paths)
    evaluate_result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "glyph",
            "--manifest",
            str(tmp_path / "manifest.parquet"),
            "--artifacts",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )
    assert evaluate_result.exit_code == 0
    assert f"json={report_paths.json_path}" in evaluate_result.stdout


def test_real_data_import_command_emits_versioned_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = SimpleNamespace(
        manifest=tmp_path / "manifest.parquet",
        dataset_metadata=tmp_path / "dataset.json",
        validation_report=tmp_path / "validation.json",
    )
    monkeypatch.setattr(cli, "import_real_dataset", lambda *_args: paths)

    result = runner.invoke(
        cli.app,
        [
            "real-data",
            "import",
            "--input",
            str(tmp_path / "records.jsonl"),
            "--output-dir",
            str(tmp_path / "versioned"),
        ],
    )

    assert result.exit_code == 0
    assert f"manifest={paths.manifest}" in result.stdout
    assert f"validation={paths.validation_report}" in result.stdout


def test_real_data_split_command_emits_fold_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = SimpleNamespace(
        folds=tmp_path / "folds.parquet",
        audit=tmp_path / "split-audit.json",
    )
    monkeypatch.setattr(cli, "assign_group_folds", lambda *_args, **_kwargs: paths)

    result = runner.invoke(
        cli.app,
        [
            "real-data",
            "split",
            "--manifest",
            str(tmp_path / "manifest.parquet"),
            "--image-root",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "split"),
        ],
    )

    assert result.exit_code == 0
    assert f"folds={paths.folds}" in result.stdout
    assert f"audit={paths.audit}" in result.stdout
