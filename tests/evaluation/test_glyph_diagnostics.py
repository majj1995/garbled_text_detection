"""Diagnostic scores must use one global threshold and preserve error provenance."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from poor_word.glyphs.generate import GenerationConfig, generate_dataset
from poor_word.training.train_glyph import TrainConfig, train_glyph


def _diagnostics() -> Any:
    module_name = "poor_word.evaluation.glyph_diagnostics"
    assert importlib.util.find_spec(module_name) is not None, "glyph diagnostics is missing"
    return importlib.import_module(module_name)


def _rows() -> list[dict[str, Any]]:
    return [
        {"sample_id": "n1", "decision": "PASS", "operator": "identity"},
        {"sample_id": "n2", "decision": "PASS", "operator": "identity"},
        {"sample_id": "a1", "decision": "BLOCK", "operator": "add_stroke"},
        {"sample_id": "a2", "decision": "BLOCK", "operator": "add_stroke"},
        {"sample_id": "b1", "decision": "BLOCK", "operator": "break_stroke"},
    ]


def test_fixed_threshold_is_shared_by_every_anomaly_type() -> None:
    """Catch per-type threshold fitting, reversed scores, and equality treated as PASS."""
    result = _diagnostics().summarize_scores(
        _rows(), np.array([0.8, 0.1, 0.9, 0.3, 0.8]), threshold=0.8, max_fpr=0.0001
    )
    assert result["threshold_source"] == "explicit"
    assert result["overall"]["tp"] == 2
    assert result["overall"]["fp"] == 1
    assert result["overall"]["fn"] == 1
    assert result["overall"]["tn"] == 1
    assert result["overall"]["fpr_constraint_met"] is False
    assert result["by_operator"]["add_stroke"] == {"count": 2, "tp": 1, "fn": 1, "recall": 0.5}
    assert result["by_operator"]["break_stroke"]["recall"] == 1.0
    assert result["by_operator"]["bridge"]["count"] == 0
    assert result["by_operator"]["bridge"]["recall"] is None


def test_automatic_threshold_respects_fpr_even_when_recall_is_low() -> None:
    """Catch reuse of the gate's unconstrained failed-threshold fallback or split ties."""
    result = _diagnostics().summarize_scores(
        _rows(), np.array([0.8, 0.1, 0.9, 0.8, 0.7]), threshold=None, max_fpr=0.0001
    )
    assert result["threshold_source"] == "selected_on_diagnostic_set"
    assert result["threshold"] == pytest.approx(0.9)
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 0
    assert result["overall"]["recall"] == pytest.approx(1 / 3)
    assert result["overall"]["fpr_constraint_met"] is True


def test_markdown_threshold_round_trips_without_creating_false_positives() -> None:
    """Catch rounded display of a nextafter sentinel turning excluded tied scores into FP."""
    module = _diagnostics()
    rows = _rows()
    scores = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    summary = module.summarize_scores(rows, scores, threshold=None, max_fpr=0.0001)
    report = {
        "summary": summary,
        "pair_coverage": {
            "replayed_steps": 1,
            "expected_steps_matches": True,
            "normal_draws": 2,
            "eligible_normal_fraction": 1.0,
            "batches_without_positive_pairs_fraction": 0.0,
            "positive_pairs": 1,
        },
        "examples_count": 0,
        "warnings": [],
    }
    markdown = module._render_diagnostics(report)
    # Parse the displayed numeric value exactly as a user would copy it to --threshold.
    threshold_line = next(line for line in markdown.splitlines() if line.startswith("- 阈值"))
    displayed = float(threshold_line.split("：", 1)[1].split("（", 1)[0])  # noqa: RUF001
    reused = module.summarize_scores(rows, scores, threshold=displayed, max_fpr=0.0001)
    assert reused["overall"]["fp"] == 0
    assert reused["overall"]["tp"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "review", "nan", "length"])
def test_summary_rejects_ambiguous_or_invalid_score_rows(mutation: str) -> None:
    """Catch silent misalignment, dropped REVIEW rows, and non-finite scores."""
    rows = _rows()
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    if mutation == "duplicate":
        rows[1]["sample_id"] = "n1"
    elif mutation == "review":
        rows[1]["decision"] = "REVIEW"
    elif mutation == "nan":
        scores[0] = np.nan
    else:
        scores = scores[:-1]
    with pytest.raises(ValueError):
        _diagnostics().summarize_scores(rows, scores, threshold=None, max_fpr=0.0001)


def _image_rows(root: Path) -> list[dict[str, Any]]:
    root.mkdir()
    rows = _rows()
    for index, row in enumerate(rows):
        image_path = f"{row['sample_id']}.png"
        mask_path = f"{row['sample_id']}-mask.png"
        Image.new("RGB", (12, 12), (index * 40, 0, 0)).save(root / image_path)
        Image.new("L", (12, 12), index * 40).save(root / mask_path)
        row.update(image_path=image_path, mask_path=mask_path, base_char="A", changed_pixels=8)
    return rows


def test_exports_only_errors_without_modifying_images_and_records_provenance(
    tmp_path: Path,
) -> None:
    """Catch exporting true positives, mislabeled mask semantics, or editing the source image."""
    root = tmp_path / "source"
    rows = _image_rows(root)
    scores = np.array([0.8, 0.1, 0.9, 0.3, 0.2])
    output = tmp_path / "output"
    records = _diagnostics().export_error_examples(
        rows,
        scores,
        threshold=0.8,
        source_root=root,
        output_dir=output,
        examples_per_kind=10,
        max_false_positives=50,
        seed=7,
    )
    assert {(r["sample_id"], r["error_type"]) for r in records} == {
        ("n1", "FP"),
        ("a2", "FN"),
        ("b1", "FN"),
    }
    for record in records:
        original = root / record["image_path"]
        exported = output / record["exported_image"]
        assert exported.read_bytes() == original.read_bytes()
        assert record["source_image_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
        assert (output / record["exported_mask"]).read_bytes() == (
            root / record["mask_path"]
        ).read_bytes()
        assert record["mask_semantics"] == (
            "changed_pixels" if record["error_type"] == "FN" else "foreground"
        )
    assert json.loads((output / "examples/index.json").read_text()) == records


def test_error_example_selection_is_capped_and_reproducible(tmp_path: Path) -> None:
    """Catch exceeding per-kind caps, exporting all normals, or non-deterministic sampling."""
    root = tmp_path / "source"
    rows = _image_rows(root)
    scores = np.array([0.9, 0.8, 0.2, 0.3, 0.1])
    outputs = []
    for run in range(2):
        outputs.append(
            _diagnostics().export_error_examples(
                rows,
                scores,
                threshold=0.7,
                source_root=root,
                output_dir=tmp_path / str(run),
                examples_per_kind=1,
                max_false_positives=1,
                seed=7,
            )
        )
    assert [r["sample_id"] for r in outputs[0]] == [r["sample_id"] for r in outputs[1]]
    assert sum(r["error_type"] == "FP" for r in outputs[0]) == 1
    assert sum(r["operator"] == "add_stroke" for r in outputs[0]) == 1


def test_error_export_rejects_paths_outside_source_root(tmp_path: Path) -> None:
    """Catch using a malicious manifest path to export unrelated local files."""
    root = tmp_path / "source"
    rows = _image_rows(root)
    secret = tmp_path / "outside.png"
    secret.write_bytes(b"outside")
    rows[0]["image_path"] = "../outside.png"
    with pytest.raises(ValueError, match="escapes"):
        _diagnostics().export_error_examples(
            rows,
            np.array([0.9, 0.1, 0.2, 0.3, 0.4]),
            threshold=0.8,
            source_root=root,
            output_dir=tmp_path / "out",
            examples_per_kind=10,
            max_false_positives=50,
            seed=7,
        )


@pytest.fixture(scope="module")
def diagnostic_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("glyph_diagnostics")
    manifests = []
    for name, seed in [("train", 71), ("holdout", 72)]:
        manifests.append(
            generate_dataset(
                GenerationConfig(
                    output_dir=root / name,
                    characters=("A", "B"),
                    font_paths=(None,),
                    normal_per_char=2,
                    abnormal_per_operator=1,
                    operators=("erase_segment", "add_stroke"),
                    seed=seed,
                    source_asset_ids=("fixture_font",),
                )
            )
        )
    artifacts = root / "artifacts"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_glyph(
            TrainConfig(
                manifest=manifests[0],
                output_dir=artifacts,
                epochs=1,
                max_steps=1,
                batch_size=4,
                seed=71,
                device="cpu",
            )
        )
    finally:
        torch.set_num_threads(previous_threads)
    return manifests[0], manifests[1], artifacts


def test_diagnostics_runs_inference_exports_aligned_scores_and_replays_training(
    diagnostic_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Catch wrong model/input restoration, duplicate or missing score rows, and input writes."""
    module = _diagnostics()
    train_manifest, manifest, artifacts = diagnostic_inputs
    inputs = [train_manifest, manifest, *(artifacts.iterdir())]
    hashes_before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.is_file()}
    output = tmp_path / "diagnostics"
    messages = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = module.diagnose_glyph(
            module.DiagnosticConfig(
                manifest=manifest,
                train_manifest=train_manifest,
                artifacts_dir=artifacts,
                output_dir=output,
                device="cpu",
                threshold=2.0,
                batch_size=2,
                examples_per_kind=1,
            ),
            progress=messages.append,
        )
    finally:
        torch.set_num_threads(previous_threads)
    report = json.loads(result.json_path.read_text())
    score_rows = pq.read_table(result.scores_path).to_pylist()
    original_rows = pq.read_table(manifest).to_pylist()
    assert [row["sample_id"] for row in score_rows] == [row["sample_id"] for row in original_rows]
    assert all(row["predicted_decision"] == "PASS" for row in score_rows)
    assert report["summary"]["overall"]["tp"] == 0
    assert report["summary"]["overall"]["fp"] == 0
    assert report["pair_coverage"]["replayed_steps"] == 1
    assert report["pair_coverage"]["expected_steps_matches"] is True
    assert report["claim_scope"] == "synthetic_diagnostic_only"
    assert "Not a production claim" in result.markdown_path.read_text()
    assert report["examples_count"] > 0
    assert messages
    assert hashes_before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes_before}
    with pytest.raises(FileExistsError):
        module.diagnose_glyph(
            module.DiagnosticConfig(
                manifest=manifest,
                train_manifest=train_manifest,
                artifacts_dir=artifacts,
                output_dir=output,
                device="cpu",
            )
        )


def test_diagnostics_rejects_wrong_training_manifest_before_creating_output(
    diagnostic_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Catch a sampler audit silently replayed on holdout instead of actual training data."""
    module = _diagnostics()
    _, manifest, artifacts = diagnostic_inputs
    output = tmp_path / "wrong-source"
    with pytest.raises(ValueError, match="training manifest"):
        module.diagnose_glyph(
            module.DiagnosticConfig(
                manifest=manifest,
                train_manifest=manifest,
                artifacts_dir=artifacts,
                output_dir=output,
                device="cpu",
            )
        )
    assert not output.exists()


def test_diagnostic_cli_produces_usable_artifacts(
    diagnostic_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Catch omitted CLI wiring/options or a command which reports success without artifacts."""
    from poor_word.cli import app

    train_manifest, manifest, artifacts = diagnostic_inputs
    output = tmp_path / "cli-diagnostics"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = CliRunner().invoke(
            app,
            [
                "evaluate",
                "diagnose-glyph",
                "--manifest",
                str(manifest),
                "--train-manifest",
                str(train_manifest),
                "--artifacts",
                str(artifacts),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
                "--examples-per-kind",
                "1",
            ],
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert result.exit_code == 0, result.output
    assert (output / "diagnostics.json").is_file()
    assert (output / "scores.parquet").is_file()
    assert (output / "examples/index.json").is_file()
    assert "diagnostics.md" in result.stdout


def test_failed_export_cleans_only_its_staging_directory(
    diagnostic_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch half-published reports or accidental input deletion after an output I/O error."""
    module = _diagnostics()
    train_manifest, manifest, artifacts = diagnostic_inputs
    output = tmp_path / "failed-diagnostics"
    original = hashlib.sha256((artifacts / "encoder.pt").read_bytes()).hexdigest()

    def fail_copy(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated disk full")

    monkeypatch.setattr(module.shutil, "copyfile", fail_copy)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(OSError, match="disk full"):
            module.diagnose_glyph(
                module.DiagnosticConfig(
                    manifest=manifest,
                    train_manifest=train_manifest,
                    artifacts_dir=artifacts,
                    output_dir=output,
                    device="cpu",
                    threshold=2.0,
                )
            )
    finally:
        torch.set_num_threads(previous_threads)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-diagnostics-*"))
    assert hashlib.sha256((artifacts / "encoder.pt").read_bytes()).hexdigest() == original


def test_output_must_not_be_created_inside_input_data(
    diagnostic_inputs: tuple[Path, Path, Path],
) -> None:
    """Catch a diagnostic run writing into the supposedly unchanged input dataset."""
    module = _diagnostics()
    train_manifest, manifest, artifacts = diagnostic_inputs
    output = manifest.parent / "do-not-create"
    with pytest.raises(ValueError, match="outside"):
        module.diagnose_glyph(
            module.DiagnosticConfig(
                manifest=manifest,
                train_manifest=train_manifest,
                artifacts_dir=artifacts,
                output_dir=output,
                device="cpu",
            )
        )
    assert not output.exists()


def test_single_class_manifest_is_rejected_before_model_inference(
    diagnostic_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch wasting inference on a manifest which cannot support binary diagnostics."""
    module = _diagnostics()
    train_manifest, original, artifacts = diagnostic_inputs
    data_root = tmp_path / "normal-only"
    data_root.mkdir()
    rows = [row for row in pq.read_table(original).to_pylist() if row["decision"] == "PASS"]
    # Copy fixture pixels so they stay inside the diagnostic dataset's root.
    for row in rows:
        for key in ["image_path", "mask_path"]:
            relative = Path(row[key])
            (data_root / relative).parent.mkdir(parents=True, exist_ok=True)
            (data_root / relative).write_bytes((original.parent / relative).read_bytes())
    manifest = data_root / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)

    def unexpected_inference(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("invalid data reached inference")

    monkeypatch.setattr(module, "_infer_scores", unexpected_inference)
    with pytest.raises(ValueError, match="both PASS and BLOCK"):
        module.diagnose_glyph(
            module.DiagnosticConfig(
                manifest=manifest,
                train_manifest=train_manifest,
                artifacts_dir=artifacts,
                output_dir=tmp_path / "invalid-output",
                device="cpu",
            )
        )
