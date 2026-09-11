"""Calibrated V2 evaluation keeps threshold choice separate from locked testing."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from poor_word.evaluation.glyph_v2 import (
    _commands,
    _locked_metrics,
    evaluate_glyph_v2,
    select_calibrated_threshold,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(
    root: Path,
    role: str,
    records: list[tuple[str, str, str, str, str | None]],
    *,
    dataset_id: str = "dataset-1",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (sample_id, group_id, pixel_token, decision, bridge_mode) in enumerate(records):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[2:14, 2 + index % 8] = int(pixel_token, 16)
        image = root / f"{sample_id}.png"
        mask = root / f"{sample_id}-mask.png"
        Image.fromarray(rgb).save(image)
        Image.fromarray(rgb[:, :, 0]).save(mask)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": image.name,
                "mask_path": mask.name,
                "base_char": "文",
                "rendered_char": "文",
                "decision": decision,
                "anomaly_kind": "none" if decision == "PASS" else "bridge",
                "operator": "identity" if decision == "PASS" else "bridge",
                "bridge_mode": bridge_mode,
                "changed_pixels": 0 if decision == "PASS" else 12,
                "seed": index,
                "bbox": {"x0": 1, "y0": 1, "x1": 15, "y1": 15},
                "source_asset_ids": ["frozenstroke_catalog", "fixture_font"],
                "schema_version": "glyph-dataset-v2",
                "dataset_id": dataset_id,
                "split_role": role,
                "source_group_id": group_id,
                "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "image_sha256": _sha256(image),
                "mask_sha256": _sha256(mask),
                "training_eligible": role == "train",
                "production_allowed": False,
                "label_provenance": "synthetic_rule_v2",
            }
        )
    manifest = root / f"{role}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "glyph-dataset-v2",
                "dataset_id": dataset_id,
                "experimental_only": True,
                "production_allowed": False,
                "config": {
                    "output_dir": str(root),
                    "graphics_path": "data/raw/makemeahanzi_graphics.txt",
                    "source_lock_path": "data/locks/makemeahanzi_graphics.lock.json",
                    "license_path": "data/licenses/makemeahanzi.ARPHICPL.txt",
                    "characters": list("永明林国春田合口"),
                    "seed": 41,
                    "train_normal_per_char": 4,
                    "eval_normal_per_char": 2,
                    "train_abnormal_per_operator": 1,
                    "eval_abnormal_per_operator": 1,
                    "max_attempts": 8,
                    "allow_experimental": True,
                },
                "source_assets": {
                    "frozenstroke_catalog": "a" * 64,
                    "fixture_font": "b" * 64,
                },
                "manifests": {
                    manifest.name: {
                        "sha256": _sha256(manifest),
                        "row_count": len(rows),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_artifacts(
    root: Path,
    *,
    sample_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
    pixel_hashes: list[str] | None = None,
    dataset_id: str = "dataset-1",
) -> Path:
    root.mkdir(parents=True)
    membership = root / "training_membership.json"
    membership.write_text(
        json.dumps(
            {
                "schema_version": "glyph-training-membership-v2",
                "dataset_id": dataset_id,
                "manifest_sha256": "c" * 64,
                "sample_ids": sample_ids or ["train-1"],
                "source_group_ids": group_ids or ["train-group"],
                "pixel_sha256": pixel_hashes or ["d" * 64],
            }
        ),
        encoding="utf-8",
    )
    membership_hash = _sha256(membership)
    checkpoint_path = root / "encoder.pt"
    torch.save(
        {
            "dataset_id": dataset_id,
            "dataset_schema_version": "glyph-dataset-v2",
            "experimental_only": True,
            "production_allowed": False,
            "training_membership_sha256": membership_hash,
            "config": {
                "manifest": "/original/train.parquet",
                "output_dir": str(root),
                "epochs": 2,
                "max_steps": 2,
                "batch_size": 8,
                "seed": 41,
                "pretrained": False,
                "device": "cpu",
                "sampler": "paired",
                "allow_experimental": True,
                "log_every": 1,
            },
        },
        checkpoint_path,
    )
    prototype_path = root / "prototypes.npz"
    prototype_path.write_bytes(b"prototype fixture")
    (root / "prototypes.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_schema_version": "glyph-dataset-v2",
                "training_membership_sha256": membership_hash,
                "encoder_checkpoint_sha256": _sha256(checkpoint_path),
                "prototype_bank_sha256": _sha256(prototype_path),
                "catalog_sha256": hashlib.sha256("文".encode()).hexdigest(),
                "source_manifest_sha256": "c" * 64,
                "experimental_only": "true",
                "production_allowed": "false",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_threshold_is_the_least_strict_tie_safe_value_within_the_fpr_budget() -> None:
    # Regresses choosing an arbitrary tied row and thereby admitting too many false positives.
    selected = select_calibrated_threshold([0.1, 0.2, 0.2, 0.3], max_fpr=0.25)

    assert selected.threshold == np.nextafter(0.2, np.inf)
    assert selected.allowed_false_positives == 1
    assert selected.false_positives == 1
    assert selected.empirical_fpr == 0.25


def test_unique_boundary_steps_above_the_first_excluded_normal_score() -> None:
    # Regresses selecting the included 0.9 score and missing anomalies between 0.8 and 0.9.
    selected = select_calibrated_threshold([0.9, 0.8], max_fpr=0.5)

    assert selected.threshold == np.nextafter(0.8, np.inf)
    assert selected.false_positives == 1


def test_all_tied_normals_use_nextafter_when_no_false_positive_is_allowed() -> None:
    # Regresses using >= max(normal), which would flag every tied normal.
    selected = select_calibrated_threshold([0.2, 0.2, 0.2], max_fpr=0.1)

    assert selected.threshold == np.nextafter(0.2, np.inf)
    assert selected.false_positives == 0


@pytest.mark.parametrize("scores", [[], [0.1, np.inf], [0.1, np.nan]])
def test_threshold_rejects_empty_or_non_finite_normal_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        select_calibrated_threshold(scores, max_fpr=0.1)


def test_locked_metrics_use_one_threshold_and_project_ppv_from_prevalence() -> None:
    # Regresses re-optimizing on test labels or reporting balanced-set precision as deployment PPV.
    metrics = _locked_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.1, 0.7, 0.8, 0.2], dtype=np.float64),
        threshold=0.7,
        prevalence=0.001,
    )

    assert metrics == {
        "positive_count": 2,
        "negative_count": 2,
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
        "recall": 0.5,
        "fpr": 0.5,
        "auroc": 0.75,
        "aucpr": pytest.approx(5 / 6),
        "projected_production_ppv": pytest.approx(0.001),
    }


def test_test_scores_and_labels_cannot_change_the_calibration_threshold() -> None:
    first = select_calibrated_threshold([0.1, 0.4, 0.3, 0.2], max_fpr=0.25)
    _locked_metrics(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0.9, 0.1], dtype=np.float64),
        threshold=first.threshold,
        prevalence=0.001,
    )
    second = select_calibrated_threshold([0.1, 0.4, 0.3, 0.2], max_fpr=0.25)
    _locked_metrics(
        np.asarray([1, 0], dtype=np.int64),
        np.asarray([0.99, 0.98], dtype=np.float64),
        threshold=second.threshold,
        prevalence=0.001,
    )

    assert first.threshold == second.threshold == np.nextafter(0.3, np.inf)


def test_evaluator_publishes_both_roles_and_explicit_synthetic_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _write_split(
        tmp_path / "calibration",
        "calibration",
        [
            ("cal-n1", "cal-g1", "1", "PASS", None),
            ("cal-n2", "cal-g2", "2", "PASS", None),
            ("cal-a1", "cal-g3", "3", "BLOCK", "close_opening"),
        ],
    )
    test = _write_split(
        tmp_path / "test",
        "test",
        [
            ("test-n1", "test-g1", "4", "PASS", None),
            ("test-a1", "test-g2", "5", "BLOCK", "block_gap"),
        ],
    )
    artifacts = _write_artifacts(tmp_path / "artifacts")

    def scores(manifest: Path, *_args: Any, **_kwargs: Any):
        if manifest == calibration:
            return np.asarray([0, 0, 1]), np.asarray([0.1, 0.2, 0.8]), 0.01
        return np.asarray([0, 1]), np.asarray([0.05, 0.7]), 0.02

    monkeypatch.setattr("poor_word.evaluation.glyph_v2._score_manifest", scores)
    progress: list[str] = []
    output = evaluate_glyph_v2(
        calibration,
        test,
        artifacts,
        tmp_path / "report",
        prevalence=0.001,
        max_fpr=0.5,
        allow_experimental=True,
        progress=progress.append,
    )

    report = json.loads(output.json_path.read_text(encoding="utf-8"))
    scored = pq.read_table(output.scores_path).to_pylist()
    markdown = output.markdown_path.read_text(encoding="utf-8")
    assert report["schema_version"] == "glyph-evaluation-report-v2"
    assert report["threshold"]["source"] == "calibration_normal_scores"
    assert report["model"]["score"] == "nearest_prototype_cosine_distance"
    assert report["test"]["tp"] == 1
    assert report["test"]["fp"] == 0
    assert report["test"]["by_anomaly_operator"]["bridge"]["recall"] == 1.0
    assert report["test"]["by_bridge_mode"]["block_gap"]["recall"] == 1.0
    assert {row["split_role"] for row in scored} == {"calibration", "test"}
    assert all(row["threshold"] == report["threshold"]["value"] for row in scored)
    assert "same catalog/source font" in markdown
    assert "projected, not measured deployment precision" in markdown
    assert "production gate passed" not in markdown.lower()
    assert "| bridge | 1 | 1 | 0 | 100.0000% |" in markdown
    assert "| block_gap | 1 | 1 | 0 | 100.0000% |" in markdown
    generate, train, evaluate = report["reproduction_commands"]
    assert generate.startswith("uv run poor-word glyphs generate-v2 --profile smoke")
    assert "-reproduced" in generate
    assert train.startswith("uv run poor-word train glyph")
    assert "--epochs 2 --max-steps 2 --batch-size 8" in train
    assert "--no-pretrained" in train
    assert "-reproduced" in train
    assert evaluate.startswith("uv run poor-word evaluate glyph-v2")
    assert "-reproduced" in evaluate
    assert any("Validating" in message for message in progress)
    assert any("calibration split" in message for message in progress)
    assert any("locked test split" in message for message in progress)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dataset_schema_version", "glyph-dataset-v1", "schema"),
        ("experimental_only", False, "experimental"),
        ("production_allowed", True, "production"),
    ],
)
def test_evaluator_requires_v2_nonproduction_checkpoint_scope(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    # Regresses evaluating an artifact trained outside the explicitly experimental V2 contract.
    calibration = _write_split(tmp_path / "cal", "calibration", [("c", "cg", "1", "PASS", None)])
    test = _write_split(tmp_path / "test", "test", [("t", "tg", "2", "PASS", None)])
    artifacts = _write_artifacts(tmp_path / "artifacts")
    checkpoint_path = artifacts / "encoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=match):
        evaluate_glyph_v2(
            calibration,
            test,
            artifacts,
            tmp_path / "out",
            allow_experimental=True,
        )


def test_repeated_training_source_groups_are_valid_membership_not_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regresses rejecting the trainer's one-source-group-entry-per-row membership format.
    calibration = _write_split(tmp_path / "cal", "calibration", [("c-n", "c-g", "1", "PASS", None)])
    test = _write_split(tmp_path / "test", "test", [("t-n", "t-g", "2", "PASS", None)])
    artifacts = _write_artifacts(
        tmp_path / "artifacts",
        sample_ids=["train-1", "train-2"],
        group_ids=["train-group", "train-group"],
        pixel_hashes=["c" * 64, "d" * 64],
    )
    monkeypatch.setattr(
        "poor_word.evaluation.glyph_v2._score_manifest",
        lambda manifest, *_args, **_kwargs: (
            np.asarray([0]),
            np.asarray([0.1 if manifest == calibration else 0.2]),
            0.01,
        ),
    )

    output = evaluate_glyph_v2(
        calibration,
        test,
        artifacts,
        tmp_path / "out",
        max_fpr=0.5,
        allow_experimental=True,
    )

    assert output.json_path.is_file()


def test_evaluator_requires_explicit_experimental_permission(tmp_path: Path) -> None:
    calibration = _write_split(tmp_path / "cal", "calibration", [("c", "cg", "1", "PASS", None)])
    test = _write_split(tmp_path / "test", "test", [("t", "tg", "2", "PASS", None)])

    with pytest.raises(ValueError, match="allow_experimental"):
        evaluate_glyph_v2(calibration, test, tmp_path / "artifacts", tmp_path / "out")


@pytest.mark.parametrize(
    ("sample_ids", "group_ids", "pixel_hashes", "match"),
    [
        (["cal-n"], ["train-group"], ["d" * 64], "sample_id"),
        (["train-1"], ["cal-group"], ["d" * 64], "source_group_id"),
    ],
)
def test_evaluator_rejects_actual_training_membership_overlap(
    tmp_path: Path,
    sample_ids: list[str],
    group_ids: list[str],
    pixel_hashes: list[str],
    match: str,
) -> None:
    calibration = _write_split(
        tmp_path / "cal", "calibration", [("cal-n", "cal-group", "1", "PASS", None)]
    )
    test = _write_split(tmp_path / "test", "test", [("test-n", "test-group", "2", "PASS", None)])
    artifacts = _write_artifacts(
        tmp_path / "artifacts",
        sample_ids=sample_ids,
        group_ids=group_ids,
        pixel_hashes=pixel_hashes,
    )

    with pytest.raises(ValueError, match=match):
        evaluate_glyph_v2(
            calibration,
            test,
            artifacts,
            tmp_path / "out",
            allow_experimental=True,
        )


def test_evaluator_rejects_training_pixel_overlap(tmp_path: Path) -> None:
    calibration = _write_split(
        tmp_path / "cal", "calibration", [("cal-n", "cal-group", "1", "PASS", None)]
    )
    test = _write_split(tmp_path / "test", "test", [("test-n", "test-group", "2", "PASS", None)])
    calibration_pixel = str(pq.read_table(calibration).to_pylist()[0]["pixel_sha256"])
    artifacts = _write_artifacts(tmp_path / "artifacts", pixel_hashes=[calibration_pixel])

    with pytest.raises(ValueError, match="pixel_sha256"):
        evaluate_glyph_v2(
            calibration,
            test,
            artifacts,
            tmp_path / "out",
            allow_experimental=True,
        )


def test_evaluator_rejects_calibration_test_group_overlap(tmp_path: Path) -> None:
    calibration = _write_split(
        tmp_path / "cal", "calibration", [("cal-n", "shared-group", "1", "PASS", None)]
    )
    test = _write_split(tmp_path / "test", "test", [("test-n", "shared-group", "2", "PASS", None)])
    artifacts = _write_artifacts(tmp_path / "artifacts")

    with pytest.raises(ValueError, match=r"calibration/test.*source_group_id"):
        evaluate_glyph_v2(
            calibration,
            test,
            artifacts,
            tmp_path / "out",
            allow_experimental=True,
        )


def test_evaluator_rejects_tampered_membership_binding(tmp_path: Path) -> None:
    calibration = _write_split(tmp_path / "cal", "calibration", [("c", "cg", "1", "PASS", None)])
    test = _write_split(tmp_path / "test", "test", [("t", "tg", "2", "PASS", None)])
    artifacts = _write_artifacts(tmp_path / "artifacts")
    membership = artifacts / "training_membership.json"
    payload = json.loads(membership.read_text(encoding="utf-8"))
    payload["sample_ids"] = ["different"]
    membership.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training_membership_sha256"):
        evaluate_glyph_v2(
            calibration,
            test,
            artifacts,
            tmp_path / "out",
            allow_experimental=True,
        )


def test_evaluator_rejects_substituted_prototype_bank_before_scoring(tmp_path: Path) -> None:
    calibration = _write_split(tmp_path / "cal", "calibration", [("c", "cg", "1", "PASS", None)])
    test = _write_split(tmp_path / "test", "test", [("t", "tg", "2", "PASS", None)])
    artifacts = _write_artifacts(tmp_path / "artifacts")
    (artifacts / "prototypes.npz").write_bytes(b"substituted bank")

    with pytest.raises(ValueError, match="prototype_bank_sha256"):
        evaluate_glyph_v2(
            calibration,
            test,
            artifacts,
            tmp_path / "out",
            allow_experimental=True,
        )


def test_evaluator_rejects_train_role_duplicate_paths_and_existing_output(tmp_path: Path) -> None:
    train = _write_split(tmp_path / "train", "train", [("x", "xg", "1", "PASS", None)])
    artifacts = _write_artifacts(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="distinct"):
        evaluate_glyph_v2(train, train, artifacts, tmp_path / "out", allow_experimental=True)
    with pytest.raises(ValueError, match="calibration split"):
        evaluate_glyph_v2(
            train,
            _write_split(tmp_path / "test", "test", [("t", "tg", "2", "PASS", None)]),
            artifacts,
            tmp_path / "out",
            allow_experimental=True,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        evaluate_glyph_v2(train, train, artifacts, existing, allow_experimental=True)


def test_reproduction_does_not_call_arbitrary_3500_character_catalog_mvp(tmp_path: Path) -> None:
    # Regresses treating catalog length as catalog identity and claiming a false CLI replay.
    commands = _commands(
        tmp_path / "calibration.parquet",
        tmp_path / "test.parquet",
        tmp_path / "model",
        tmp_path / "report",
        checkpoint={
            "config": {
                "manifest": "train.parquet",
                "epochs": 2,
                "max_steps": 1,
                "batch_size": 8,
                "seed": 7,
                "pretrained": False,
                "device": "cpu",
                "sampler": "paired",
                "log_every": 1,
                "learning_rate": 3e-4,
                "embedding_dim": 256,
            }
        },
        run={
            "config": {
                "characters": [chr(0x4E00 + index) for index in range(3500)],
                "train_normal_per_char": 8,
                "eval_normal_per_char": 4,
                "train_abnormal_per_operator": 2,
                "eval_abnormal_per_operator": 1,
                "max_attempts": 8,
            }
        },
        prevalence=0.001,
        max_fpr=0.0001,
        device="cpu",
        batch_size=64,
    )

    assert commands[0].startswith("# Custom generation:")
    assert "--profile mvp" not in commands[0]


def test_reproduction_labels_non_cli_training_hyperparameters_as_python_api(tmp_path: Path) -> None:
    # Regresses omitting a non-default embedding size from a command claimed to be exact.
    commands = _commands(
        tmp_path / "calibration.parquet",
        tmp_path / "test.parquet",
        tmp_path / "model",
        tmp_path / "report",
        checkpoint={
            "config": {
                "manifest": "train.parquet",
                "epochs": 2,
                "max_steps": 1,
                "batch_size": 8,
                "seed": 7,
                "pretrained": False,
                "device": "cpu",
                "sampler": "paired",
                "log_every": 1,
                "learning_rate": 3e-4,
                "embedding_dim": 8,
            }
        },
        run={"config": {}},
        prevalence=0.001,
        max_fpr=0.0001,
        device="cpu",
        batch_size=64,
    )

    assert commands[1].startswith("# Custom training:")
    assert "--embedding-dim" not in commands[1]
