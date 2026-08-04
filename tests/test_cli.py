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


def test_adapt_real_command_emits_adapted_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Break caught: the adaptation CLI fails to pass its required manifest inputs."""
    artifacts = SimpleNamespace(
        checkpoint=tmp_path / "encoder.pt", metrics=tmp_path / "metrics.json"
    )
    monkeypatch.setattr(cli, "adapt_real_encoder", lambda _config: artifacts)

    result = runner.invoke(
        cli.app,
        [
            "train",
            "adapt-real",
            "--crop-manifest",
            str(tmp_path / "crops.parquet"),
            "--real-manifest",
            str(tmp_path / "real.parquet"),
            "--prior-checkpoint",
            str(tmp_path / "prior.pt"),
            "--output-dir",
            str(tmp_path / "adapted"),
            "--device",
            "cpu",
            "--max-steps",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert f"checkpoint={artifacts.checkpoint}" in result.stdout
    assert f"metrics={artifacts.metrics}" in result.stdout


def test_adapt_real_command_rejects_singleton_batch_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Break caught: CLI accepts a batch size that cannot form a contrastive pair."""
    monkeypatch.setattr(cli, "adapt_real_encoder", lambda _config: None)

    result = runner.invoke(
        cli.app,
        [
            "train",
            "adapt-real",
            "--crop-manifest",
            str(tmp_path / "crops.parquet"),
            "--real-manifest",
            str(tmp_path / "real.parquet"),
            "--prior-checkpoint",
            str(tmp_path / "prior.pt"),
            "--output-dir",
            str(tmp_path / "adapted"),
            "--batch-size",
            "1",
        ],
    )

    assert result.exit_code == 2


def test_real_oof_command_runs_all_five_folds_and_emits_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts: list[SimpleNamespace] = []

    def fake_finetune(config: object, held_out_fold: int) -> SimpleNamespace:
        artifact = SimpleNamespace(
            held_out_fold=held_out_fold,
            checkpoint=tmp_path / f"fold-{held_out_fold}.pt",
            scores=tmp_path / f"fold-{held_out_fold}.parquet",
            metrics=tmp_path / f"fold-{held_out_fold}.json",
        )
        artifacts.append(artifact)
        return artifact

    oof = tmp_path / "oof" / "oof.parquet"
    monkeypatch.setattr(cli, "finetune_real_fold", fake_finetune)
    monkeypatch.setattr(cli, "collect_oof_scores", lambda *_args, **_kwargs: oof)
    result = runner.invoke(
        cli.app,
        [
            "train",
            "real-oof",
            "--real-manifest",
            str(tmp_path / "real.parquet"),
            "--crop-manifest",
            str(tmp_path / "crops.parquet"),
            "--gold-manifest",
            str(tmp_path / "gold.parquet"),
            "--fold-manifest",
            str(tmp_path / "folds.parquet"),
            "--synthetic-manifest",
            str(tmp_path / "synthetic.parquet"),
            "--adapted-checkpoint",
            str(tmp_path / "adapted.pt"),
            "--output-dir",
            str(tmp_path / "real-oof"),
            "--device",
            "cpu",
            "--max-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert [artifact.held_out_fold for artifact in artifacts] == list(range(5))
    assert f"oof={oof}" in result.stdout


def test_real_oof_help_lists_all_provenance_inputs() -> None:
    result = runner.invoke(cli.app, ["train", "real-oof", "--help"])

    assert result.exit_code == 0
    for option in (
        "--real-manifest",
        "--crop-manifest",
        "--gold-manifest",
        "--fold-manifest",
        "--synthetic-manifest",
        "--adapted-checkpoint",
    ):
        assert option in result.stdout


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


def test_real_data_mine_command_delegates_trusted_inputs_and_emits_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = SimpleNamespace(
        queue=tmp_path / "mined" / "queue.parquet",
        metadata=tmp_path / "mined" / "mining-audit.json",
        review_scores=tmp_path / "mined" / "review-scores.jsonl",
        review_disagreements=tmp_path / "mined" / "review-disagreements.jsonl",
    )
    captured: dict[str, object] = {}

    def fake_mine(scores: Path, policy: object, **kwargs: object) -> SimpleNamespace:
        captured.update(scores=scores, policy=policy, **kwargs)
        return artifacts

    monkeypatch.setattr(cli, "mine_candidates", fake_mine)
    result = runner.invoke(
        cli.app,
        [
            "real-data",
            "mine",
            "--scores",
            str(tmp_path / "scores.parquet"),
            "--real-manifest",
            str(tmp_path / "real.parquet"),
            "--fold-manifest",
            str(tmp_path / "folds.parquet"),
            "--output-dir",
            str(tmp_path / "mined"),
            "--overall-limit",
            "75",
            "--per-product-cap",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert captured["real_manifest"] == tmp_path / "real.parquet"
    assert captured["fold_manifest"] == tmp_path / "folds.parquet"
    policy = captured["policy"]
    assert isinstance(policy, cli.MiningPolicy)
    assert policy.overall_limit == 75
    assert policy.per_product_cap == 5
    assert f"queue={artifacts.queue}" in result.stdout
    assert f"review_scores={artifacts.review_scores}" in result.stdout


def test_real_data_mining_yield_delegates_review_import_lineage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = SimpleNamespace(
        dataset_version=tmp_path / "v2" / "dataset-version.json",
        audit=tmp_path / "v2" / "mining-yield-audit.json",
    )
    captured: dict[str, object] = {}

    def fake_record(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured.update(kwargs)
        return artifacts

    monkeypatch.setattr(cli, "record_mining_yield", fake_record)
    result = runner.invoke(
        cli.app,
        [
            "real-data",
            "mining-yield",
            "--candidate-queue",
            str(tmp_path / "queue.parquet"),
            "--reviewed-gold",
            str(tmp_path / "gold-crops.parquet"),
            "--import-audit",
            str(tmp_path / "import-audit.json"),
            "--base-real-manifest",
            str(tmp_path / "manifest.parquet"),
            "--output-dir",
            str(tmp_path / "v2"),
        ],
    )

    assert result.exit_code == 0
    assert captured["base_real_manifest"] == tmp_path / "manifest.parquet"
    assert captured["previous_gold_manifest"] is None
    assert f"dataset_version={artifacts.dataset_version}" in result.stdout


def test_real_data_crops_and_review_commands_emit_versioned_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    crops = SimpleNamespace(
        manifest=tmp_path / "crops.parquet", audit=tmp_path / "crops-audit.json"
    )
    monkeypatch.setattr(cli, "extract_character_crops", lambda *_args, **_kwargs: crops)
    crop_result = runner.invoke(
        cli.app,
        [
            "real-data",
            "crops",
            "--manifest",
            str(tmp_path / "manifest.parquet"),
            "--image-root",
            str(tmp_path / "images"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert crop_result.exit_code == 0
    assert f"crops={crops.manifest}" in crop_result.stdout

    queue = SimpleNamespace(
        queue=tmp_path / "queue",
        queue_version="version-1",
        csv=tmp_path / "queue.csv",
        jsonl=tmp_path / "queue.jsonl",
        contact_sheet=tmp_path / "contact-sheet.png",
    )
    monkeypatch.setattr(cli, "build_review_queue", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(cli, "export_review_queue", lambda *_args, **_kwargs: queue)
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        '{"crop_id":"a","image_id":"one","crop_path":"images/a.png","risk_score":0.5,"style_id":"x"}\n',
        encoding="utf-8",
    )
    disagreements = tmp_path / "disagreements.jsonl"
    disagreements.write_text(
        '{"crop_id":"a","disagreement":0.1,"disagreement_model_id":"ensemble-v1",'
        '"disagreement_artifact_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}\n',
        encoding="utf-8",
    )
    export_result = runner.invoke(
        cli.app,
        [
            "review",
            "export",
            "--crops",
            str(tmp_path / "crops.parquet"),
            "--crop-root",
            str(tmp_path / "root"),
            "--scores",
            str(scores),
            "--disagreements",
            str(disagreements),
            "--output-dir",
            str(tmp_path / "queues"),
        ],
    )
    assert export_result.exit_code == 0
    assert "queue_version=version-1" in export_result.stdout

    gold = SimpleNamespace(
        gold_crops=tmp_path / "gold-crops.parquet", audit=tmp_path / "audit.json"
    )
    monkeypatch.setattr(cli, "import_review_labels", lambda *_args, **_kwargs: gold)
    import_result = runner.invoke(
        cli.app,
        [
            "review",
            "import",
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--queue",
            str(tmp_path / "queue"),
            "--output-dir",
            str(tmp_path / "gold"),
        ],
    )
    assert import_result.exit_code == 0
    assert f"gold={gold.gold_crops}" in import_result.stdout


def test_mil_command_delegates_trusted_manifests_and_emits_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    help_result = runner.invoke(cli.app, ["train", "mil", "--help"])
    assert help_result.exit_code == 0
    assert "--feature-manifest" in help_result.stdout
    assert "--held-out-fold" in help_result.stdout

    artifacts = SimpleNamespace(
        checkpoint=tmp_path / "model.pt",
        metrics=tmp_path / "metrics.json",
        attention_candidates=tmp_path / "attention-candidates.parquet",
    )
    captured: list[object] = []

    def fake_train(config: object) -> object:
        captured.append(config)
        return artifacts

    monkeypatch.setattr(cli, "train_mil_fold", fake_train)
    result = runner.invoke(
        cli.app,
        [
            "train",
            "mil",
            "--real-manifest",
            str(tmp_path / "real.parquet"),
            "--fold-manifest",
            str(tmp_path / "folds.parquet"),
            "--feature-manifest",
            str(tmp_path / "oof.parquet"),
            "--output-dir",
            str(tmp_path / "mil"),
            "--held-out-fold",
            "2",
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--learning-rate",
            "0.002",
            "--normal-instance-weight",
            "0.4",
            "--hidden-dim",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    assert captured[0].held_out_fold == 2
    assert captured[0].learning_rate == 0.002
    assert captured[0].normal_instance_weight == 0.4
    assert captured[0].hidden_dim == 7
    assert f"checkpoint={artifacts.checkpoint}" in result.stdout
    assert f"metrics={artifacts.metrics}" in result.stdout
    assert f"attention_candidates={artifacts.attention_candidates}" in result.stdout
