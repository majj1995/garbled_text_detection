"""Commands must route V2 profiles and explicit experimental boundaries correctly."""

from types import SimpleNamespace

from typer.testing import CliRunner

import poor_word.cli as cli

runner = CliRunner()


def test_generate_v2_smoke_routes_new_rules_and_three_output_manifests(monkeypatch, tmp_path):
    seen = []

    def generate(config, *, progress):
        seen.append(config)
        progress("new-stroke-generator")
        return SimpleNamespace(
            train_manifest=tmp_path / "train.parquet",
            calibration_manifest=tmp_path / "calibration.parquet",
            test_manifest=tmp_path / "test.parquet",
            run_path=tmp_path / "run.json",
        )

    monkeypatch.setattr(cli, "generate_v2_dataset", generate, raising=False)
    result = runner.invoke(
        cli.app,
        [
            "glyphs",
            "generate-v2",
            "--profile",
            "smoke",
            "--allow-experimental",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "new-stroke-generator" in result.output
    assert len(seen) == 1 and len(seen[0].characters) == 8
    assert seen[0].allow_experimental is True
    assert seen[0].train_normal_per_char >= 2
    assert "train.parquet" in result.output and "calibration.parquet" in result.output
    assert "test.parquet" in result.output


def test_generate_v2_refuses_missing_experimental_opt_in_before_loading_data(tmp_path):
    result = runner.invoke(
        cli.app,
        [
            "glyphs",
            "generate-v2",
            "--output-dir",
            str(tmp_path / "new"),
        ],
    )
    assert result.exit_code != 0
    assert "--allow-experimental" in result.output
    assert not (tmp_path / "new").exists()


def test_train_glyph_routes_paired_sampler_and_continuous_progress(monkeypatch, tmp_path):
    seen = []

    def train(config, *, progress):
        seen.append(config)
        progress("step=1 loss=1.0")
        return SimpleNamespace(
            checkpoint=tmp_path / "encoder.pt",
            prototype_bank=tmp_path / "prototypes.npz",
            metrics=tmp_path / "metrics.json",
        )

    monkeypatch.setattr(cli, "train_glyph", train)
    result = runner.invoke(
        cli.app,
        [
            "train",
            "glyph",
            "--manifest",
            str(tmp_path / "train.parquet"),
            "--output-dir",
            str(tmp_path),
            "--sampler",
            "paired",
            "--allow-experimental",
            "--log-every",
            "10",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen[0].sampler == "paired" and seen[0].allow_experimental is True
    assert seen[0].log_every == 10
    assert "step=1 loss=1.0" in result.output


def test_evaluate_v2_routes_both_splits_instead_of_tuning_on_test(monkeypatch, tmp_path):
    seen = []

    def evaluate(calibration_manifest, test_manifest, artifacts_dir, output_dir, **kwargs):
        seen.append((calibration_manifest, test_manifest, kwargs))
        return SimpleNamespace(
            json_path=output_dir / "report.json", markdown_path=output_dir / "report.md"
        )

    monkeypatch.setattr(cli, "evaluate_glyph_v2", evaluate, raising=False)
    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "glyph-v2",
            "--calibration-manifest",
            str(tmp_path / "calibration.parquet"),
            "--test-manifest",
            str(tmp_path / "test.parquet"),
            "--artifacts",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "report"),
            "--allow-experimental",
            "--max-fpr",
            "0.0001",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen[0][0] == tmp_path / "calibration.parquet"
    assert seen[0][1] == tmp_path / "test.parquet"
    assert seen[0][2]["allow_experimental"] is True
    assert seen[0][2]["max_fpr"] == 0.0001
    assert "report.json" in result.output
