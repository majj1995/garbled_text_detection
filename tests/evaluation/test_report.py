import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from poor_word.evaluation.report import (
    ReportContext,
    _source_context,
    _validate_prototype_metadata,
    build_mvp_report,
    evaluate_glyph_artifacts,
    render_markdown,
    write_mvp_report,
)
from poor_word.glyphs.generate import GenerationConfig, generate_dataset
from poor_word.training.train_glyph import TrainConfig, train_glyph


def test_synthetic_report_is_not_a_production_claim(tmp_path: Path) -> None:
    labels = np.asarray([1] * 5 + [0] * 10_000, dtype=np.int64)
    scores = np.asarray(
        [0.99, 0.96, 0.001, 0.001, 0.001]
        + [0.97]
        + [0.01] * 9_999,
        dtype=np.float64,
    )
    context = ReportContext(
        dataset_hashes={"synthetic_manifest": "a" * 64},
        asset_hashes={"font": "b" * 64},
        model_hashes={"encoder": "c" * 64},
        source_license_decisions={"font": "approved"},
        ocr_capabilities={
            "detection_model_name": "PP-OCRv5_server_det",
            "recognition_model_name": "PP-OCRv5_server_rec",
            "character_boxes_available": False,
            "raw_logits_available": False,
        },
        latency={"offline_cpu_ms_per_glyph": 12.5},
        failures=("PP-OCRv5 character boxes unavailable",),
        reproduction_commands=("uv run poor-word evaluate glyph ...",),
    )

    report = build_mvp_report(labels, scores, prevalence=0.001, context=context)
    artifacts = write_mvp_report(report, tmp_path)

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert payload["claim_scope"] == "synthetic_only"
    assert payload["synthetic_metrics"]["mvp_gate"]["passed"] is True
    assert payload["synthetic_metrics"]["mvp_gate"]["point"]["tp"] == 2
    assert payload["synthetic_metrics"]["aucpr"] is not None
    assert payload["real_data_metrics"] is None
    assert "Not a production claim" in markdown
    assert "Production prevalence" in markdown
    assert "Observed/balanced precision" not in markdown
    assert "PP-OCRv5 character boxes unavailable" in markdown


def test_real_data_section_is_only_added_when_supplied() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    context = ReportContext(reproduction_commands=("reproduce",))

    report = build_mvp_report(
        labels,
        scores,
        prevalence=0.001,
        context=context,
        real_labels=labels,
        real_scores=scores,
    )

    assert report.claim_scope == "real_seed_evaluation"
    assert report.real_data_metrics is not None
    assert "Real seed data metrics" in render_markdown(report)


def test_real_data_arrays_must_be_supplied_together() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        build_mvp_report(
            [0, 1],
            [0.1, 0.9],
            prevalence=0.001,
            context=ReportContext(reproduction_commands=("reproduce",)),
            real_labels=[0, 1],
        )


def test_evaluate_glyph_artifacts_restores_checkpoint_and_collects_audit(
    tmp_path: Path,
) -> None:
    generated_dir = tmp_path / "generated data"
    manifest = generate_dataset(
        GenerationConfig(
            output_dir=generated_dir,
            characters=("A", "B"),
            font_paths=(None,),
            normal_per_char=1,
            abnormal_per_operator=1,
            operators=("erase_segment", "add_stroke"),
            seed=71,
            source_asset_ids=("fixture_font",),
        )
    )
    (generated_dir / "run.json").write_text('{"seed": 71}\n', encoding="utf-8")
    train_dir = tmp_path / "trained artifacts"
    train_glyph(
        TrainConfig(
            manifest=manifest,
            output_dir=train_dir,
            epochs=1,
            max_steps=1,
            batch_size=4,
            seed=71,
            device="cpu",
        )
    )
    lock_dir = tmp_path / "data/locks"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / "fixture_font.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "source_id": "fixture_font",
                "declared_url": "https://example.com/font.otf",
                "resolved_url": "https://cdn.example.com/font.otf",
                "output_name": "font.otf",
                "sha256": "d" * 64,
                "size_bytes": 123,
                "license_id": "OFL-1.1",
                "production_allowed": True,
            }
        ),
        encoding="utf-8",
    )
    (generated_dir / "run.json").write_text(
        json.dumps(
            {
                "seed": 71,
                "source_lock_sha256": {
                    lock_path.name: hashlib.sha256(lock_path.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "ocr-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "paddleocr_version": "3.2.0",
                "paddlepaddle_version": "3.1.1",
                "cuda_version": "12.6",
                "gpu_name": "NVIDIA L20",
                "detection_model_name": "PP-OCRv5_server_det",
                "recognition_model_name": "PP-OCRv5_server_rec",
                "character_boxes_available": False,
                "logits_available": False,
                "latency_p50_ms": 40.0,
                "latency_p95_ms": 55.0,
                "peak_gpu_memory_mb": 2048.0,
                "required_capability_gaps": ["character_box_adapter", "raw_logits_adapter"],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "report output"
    paths = evaluate_glyph_artifacts(
        manifest,
        train_dir,
        output,
        prevalence=0.001,
        batch_size=2,
        ocr_audit_path=audit_path,
        repo_root=tmp_path,
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["context"]["asset_hashes"]["fixture_font"] == "d" * 64
    assert payload["context"]["ocr_capabilities"]["gpu_name"] == "NVIDIA L20"
    assert "PP-OCRv5 raw logits unavailable" in payload["context"]["failures"]
    assert "'" in payload["context"]["reproduction_commands"][0]
    assert paths.markdown_path.exists()

    run_manifest = generated_dir / "run.json"
    run_manifest.write_text(
        json.dumps(
            {
                "source_lock_sha256": {
                    "fixture_font.lock.json": "0" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed since generation"):
        _source_context(tmp_path, manifest, run_manifest)

    lock_path.write_text('{"source_id": "fixture_font"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="declared_url"):
        _source_context(tmp_path, manifest, None)


def test_prototype_metadata_must_match_checkpoint_and_catalog(tmp_path: Path) -> None:
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "encoder_checkpoint_sha256": "0" * 64,
        "catalog_sha256": "1" * 64,
    }

    with pytest.raises(ValueError, match="checkpoint hash"):
        _validate_prototype_metadata(metadata, checkpoint, {"A": 0})
