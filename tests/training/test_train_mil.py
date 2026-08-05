import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from poor_word.training.train_mil import MilTrainConfig, compute_mil_loss, train_mil_fold


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _inputs(tmp_path: Path) -> MilTrainConfig:
    real = _write(
        tmp_path / "real.parquet",
        [
            {
                "image_id": "normal-train",
                "image_label": "NORMAL",
                "split_role": "DEV",
                "training_eligible": True,
            },
            {
                "image_id": "abnormal-train",
                "image_label": "ABNORMAL",
                "split_role": "DEV",
                "training_eligible": True,
            },
            {
                "image_id": "normal-valid",
                "image_label": "NORMAL",
                "split_role": "DEV",
                "training_eligible": True,
            },
            {
                "image_id": "abnormal-valid",
                "image_label": "ABNORMAL",
                "split_role": "DEV",
                "training_eligible": True,
            },
            {
                "image_id": "empty-valid",
                "image_label": "NORMAL",
                "split_role": "DEV",
                "training_eligible": True,
            },
            {
                "image_id": "locked",
                "image_label": "LOCKED_LABEL_MUST_NOT_BE_READ",
                "split_role": "LOCKED_TEST",
                "training_eligible": False,
            },
        ],
    )
    folds = _write(
        tmp_path / "folds.parquet",
        [
            {"image_id": "normal-train", "fold": 1, "split_role": "DEV", "component_id": "c1"},
            {"image_id": "abnormal-train", "fold": 1, "split_role": "DEV", "component_id": "c2"},
            {"image_id": "normal-valid", "fold": 0, "split_role": "DEV", "component_id": "c3"},
            {"image_id": "abnormal-valid", "fold": 0, "split_role": "DEV", "component_id": "c4"},
            {"image_id": "empty-valid", "fold": 0, "split_role": "DEV", "component_id": "c5"},
            {
                "image_id": "locked",
                "fold": -1,
                "split_role": "LOCKED_TEST",
                "component_id": "locked",
            },
        ],
    )
    fold_hash = _sha256(folds)
    features = _write(
        tmp_path / "oof.parquet",
        [
            {
                "crop_id": crop_id,
                "image_id": image_id,
                "fold": fold,
                "risk_score": risk,
                "model_id": f"real-fold-{fold}",
                "checkpoint_sha256": str(fold + 1) * 64,
                "fold_manifest_sha256": fold_hash,
                "decision": decision,
                "anomaly_kind": "invented_character",
            }
            for crop_id, image_id, fold, risk, decision in (
                ("nt-1", "normal-train", 1, 0.1, "BLOCK"),
                ("at-1", "abnormal-train", 1, 0.8, "PASS"),
                ("at-2", "abnormal-train", 1, 0.2, "PASS"),
                ("nv-1", "normal-valid", 0, 0.2, "BLOCK"),
                ("av-1", "abnormal-valid", 0, 0.9, "PASS"),
            )
        ],
    )
    return MilTrainConfig(
        real_manifest=real,
        fold_manifest=folds,
        feature_manifest=features,
        output_dir=tmp_path / "mil",
        held_out_fold=0,
        epochs=4,
        max_steps=2,
        batch_size=2,
        patience=1,
        min_delta=100.0,
        seed=19,
        device="cpu",
        hidden_dim=4,
    )


def test_loss_uses_image_labels_and_only_normal_instance_pressure() -> None:
    bag_logits = torch.tensor([-1.0, 1.0], requires_grad=True)
    instance_logits = torch.tensor([[-2.0, 3.0], [-4.0, 4.0]], requires_grad=True)
    mask = torch.ones((2, 2), dtype=torch.bool)
    labels = torch.tensor([0.0, 1.0])

    losses = compute_mil_loss(bag_logits, instance_logits, mask, labels, pos_weight=1.0)
    losses.total.backward()

    assert losses.normal_instance.item() > 0
    assert instance_logits.grad is not None
    assert torch.count_nonzero(instance_logits.grad[0]).item() == 2
    assert torch.count_nonzero(instance_logits.grad[1]).item() == 0
    assert losses.abnormal_instance.item() == 0.0


def test_training_is_fold_safe_balanced_and_publishes_best_atomically(tmp_path: Path) -> None:
    config = _inputs(tmp_path)
    artifacts = train_mil_fold(config)

    checkpoint = torch.load(artifacts.checkpoint, map_location="cpu", weights_only=True)
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert checkpoint["train_image_ids"] == ["abnormal-train", "normal-train"]
    assert checkpoint["validation_image_ids"] == ["abnormal-valid", "empty-valid", "normal-valid"]
    assert "locked" not in checkpoint["train_image_ids"] + checkpoint["validation_image_ids"]
    assert checkpoint["class_counts"] == {"ABNORMAL": 1, "NORMAL": 1}
    assert checkpoint["pos_weight"] == 1.0
    assert checkpoint["best_epoch"] == metrics["best_epoch"]
    assert checkpoint["stop_reason"] == "early_stopping"
    assert metrics["checkpoint_sha256"] == _sha256(artifacts.checkpoint)
    assert metrics["real_manifest_sha256"] == _sha256(config.real_manifest)
    assert metrics["fold_manifest_sha256"] == _sha256(config.fold_manifest)
    assert metrics["feature_manifest_sha256"] == _sha256(config.feature_manifest)
    assert checkpoint["feature_model_provenance"] == [
        {"checkpoint_sha256": "1" * 64, "model_id": "real-fold-0", "scoring_fold": 0},
        {"checkpoint_sha256": "2" * 64, "model_id": "real-fold-1", "scoring_fold": 1},
    ]
    assert metrics["feature_model_provenance"] == checkpoint["feature_model_provenance"]

    candidates = pq.read_table(artifacts.attention_candidates).to_pylist()
    assert {row["image_id"] for row in candidates} == {"abnormal-valid", "normal-valid"}
    assert all(row["label_source"] == "mil_attention_candidate" for row in candidates)
    assert all(row["is_gold"] is False for row in candidates)
    assert {"decision", "annotator_id", "anomaly_kind"}.isdisjoint(
        pq.read_schema(artifacts.attention_candidates).names
    )
    candidate_hashes = {
        "checkpoint_sha256": _sha256(artifacts.checkpoint),
        "real_manifest_sha256": _sha256(config.real_manifest),
        "fold_manifest_sha256": _sha256(config.fold_manifest),
        "feature_manifest_sha256": _sha256(config.feature_manifest),
    }
    assert all(
        all(row[field] == value for field, value in candidate_hashes.items())
        for row in candidates
    )
    assert metrics["attention_candidates_sha256"] == _sha256(
        artifacts.attention_candidates
    )
    image_scores = pq.read_table(artifacts.image_scores).to_pylist()
    assert {row["image_id"] for row in image_scores} == {
        "abnormal-valid",
        "empty-valid",
        "normal-valid",
    }
    assert len(image_scores) == 3
    empty = next(row for row in image_scores if row["image_id"] == "empty-valid")
    assert empty["zero_character"] is True
    assert empty["image_label"] == "NORMAL"
    assert empty["fold"] == 0
    assert all(row["checkpoint_sha256"] == _sha256(artifacts.checkpoint) for row in image_scores)
    assert metrics["image_scores_sha256"] == _sha256(artifacts.image_scores)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("unknown_image", "unknown image_id"),
        ("duplicate_crop", "duplicate crop_id"),
        ("wrong_fold", "scoring fold"),
        ("bad_risk", "finite risk_score"),
        ("bad_hash", "fold manifest hash"),
        ("locked_feature", "locked-test feature"),
    ],
)
def test_training_rejects_untrusted_feature_linkage_before_output(
    tmp_path: Path, mutation: str, match: str
) -> None:
    config = _inputs(tmp_path)
    rows = pq.read_table(config.feature_manifest).to_pylist()
    if mutation == "unknown_image":
        rows[0]["image_id"] = "missing"
    elif mutation == "duplicate_crop":
        rows.append(dict(rows[0]))
    elif mutation == "wrong_fold":
        rows[0]["fold"] = 0
    elif mutation == "bad_risk":
        rows[0]["risk_score"] = float("nan")
    elif mutation == "bad_hash":
        rows[0]["fold_manifest_sha256"] = "0" * 64
    else:
        rows.append(
            {
                "crop_id": "locked-1",
                "image_id": "locked",
                "fold": -1,
                "risk_score": 1.0,
                "model_id": "real-fold--1",
                "checkpoint_sha256": "f" * 64,
                "fold_manifest_sha256": _sha256(config.fold_manifest),
                "decision": "BLOCK",
                "anomaly_kind": "invented_character",
            }
        )
    _write(config.feature_manifest, rows)

    with pytest.raises(ValueError, match=match):
        train_mil_fold(config)
    assert not config.output_dir.exists()


def test_missing_feature_rows_form_zero_character_bag(tmp_path: Path) -> None:
    artifacts = train_mil_fold(_inputs(tmp_path))
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert metrics["zero_character_validation_image_ids"] == ["empty-valid"]


def test_training_rejects_single_class_training_fold(tmp_path: Path) -> None:
    config = _inputs(tmp_path)
    rows = pq.read_table(config.real_manifest).to_pylist()
    for row in rows:
        if row["image_id"] == "abnormal-train":
            row["training_eligible"] = False
    _write(config.real_manifest, rows)
    with pytest.raises(ValueError, match="both NORMAL and ABNORMAL"):
        train_mil_fold(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("component", [None, ""])
def test_training_requires_nonempty_component_id_before_output(
    tmp_path: Path, component: object
) -> None:
    config = _inputs(tmp_path)
    rows = pq.read_table(config.fold_manifest).to_pylist()
    if component is None:
        for row in rows:
            row.pop("component_id")
    else:
        rows[0]["component_id"] = component
    _write(config.fold_manifest, rows)

    with pytest.raises(ValueError, match="component_id"):
        train_mil_fold(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("risk", [-0.1, 1.1, 1e300, float("nan")])
def test_training_rejects_risk_outside_float32_probability_contract(
    tmp_path: Path, risk: float
) -> None:
    config = _inputs(tmp_path)
    rows = pq.read_table(config.feature_manifest).to_pylist()
    rows[0]["risk_score"] = risk
    _write(config.feature_manifest, rows)

    with pytest.raises(ValueError, match=r"risk_score.*\[0,1\]"):
        train_mil_fold(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("mutation", ["wrong_model_id", "second_model", "second_checkpoint"])
def test_training_requires_one_canonical_model_per_scoring_fold(
    tmp_path: Path, mutation: str
) -> None:
    config = _inputs(tmp_path)
    rows = pq.read_table(config.feature_manifest).to_pylist()
    if mutation == "wrong_model_id":
        rows[0]["model_id"] = "other-model"
    else:
        clone = dict(rows[0])
        clone["crop_id"] = "nt-2"
        if mutation == "second_model":
            clone["model_id"] = "real-fold-1-alternate"
        else:
            clone["checkpoint_sha256"] = "e" * 64
        rows.append(clone)
    _write(config.feature_manifest, rows)

    with pytest.raises(
        ValueError, match=r"canonical model|conflicting model provenance"
    ):
        train_mil_fold(config)
    assert not config.output_dir.exists()


def test_training_rejects_single_class_validation_before_model_selection(
    tmp_path: Path,
) -> None:
    config = _inputs(tmp_path)
    rows = pq.read_table(config.real_manifest).to_pylist()
    for row in rows:
        if row["image_id"] == "abnormal-valid":
            row["training_eligible"] = False
    _write(config.real_manifest, rows)

    with pytest.raises(ValueError, match="validation fold requires both NORMAL and ABNORMAL"):
        train_mil_fold(config)
    assert not config.output_dir.exists()
