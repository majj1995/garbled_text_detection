import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from poor_word.evaluation.oof import collect_oof_scores
from poor_word.training.finetune_real import FoldModelArtifacts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _trusted_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "real": _write(
            tmp_path / "real.parquet",
            [
                {"image_id": "a", "split_role": "DEV", "training_eligible": True},
                {"image_id": "b", "split_role": "DEV", "training_eligible": True},
                {"image_id": "c", "split_role": "DEV", "training_eligible": True},
                {
                    "image_id": "locked",
                    "split_role": "LOCKED_TEST",
                    "training_eligible": True,
                },
            ],
        ),
        "crops": _write(
            tmp_path / "crops.parquet",
            [
                {"crop_id": "a-1", "image_id": "a", "crop_path": "images/a.png"},
                {"crop_id": "b-1", "image_id": "b", "crop_path": "images/b.png"},
                {"crop_id": "c-1", "image_id": "c", "crop_path": "images/c.png"},
                {
                    "crop_id": "locked-1",
                    "image_id": "locked",
                    "crop_path": "images/secret.png",
                },
            ],
        ),
        "gold": _write(
            tmp_path / "gold.parquet",
            [
                {
                    "crop_id": "a-1",
                    "image_id": "a",
                    "crop_path": "images/a.png",
                    "decision": "PASS",
                    "anomaly_kind": "NONE",
                },
                {
                    "crop_id": "b-1",
                    "image_id": "b",
                    "crop_path": "images/b.png",
                    "decision": "BLOCK",
                    "anomaly_kind": "missing_stroke",
                },
                {
                    "crop_id": "c-1",
                    "image_id": "c",
                    "crop_path": "images/c.png",
                    "decision": "PASS",
                    "anomaly_kind": "NONE",
                },
                {
                    "crop_id": "locked-1",
                    "image_id": "locked",
                    "crop_path": "images/secret.png",
                    "decision": "BLOCK",
                    "anomaly_kind": "invented_character",
                },
            ],
        ),
        "folds": _write(
            tmp_path / "folds.parquet",
            [
                {"image_id": "a", "fold": 0, "split_role": "DEV"},
                {"image_id": "b", "fold": 0, "split_role": "DEV"},
                {"image_id": "c", "fold": 1, "split_role": "DEV"},
                {"image_id": "locked", "fold": -1, "split_role": "LOCKED_TEST"},
            ],
        ),
        "synthetic": _write(tmp_path / "synthetic.parquet", [{"sample_id": "s"}]),
    }
    adapted = tmp_path / "adapted.pt"
    adapted.write_bytes(b"trusted adapted checkpoint")
    paths["adapted"] = adapted
    return paths


def _artifact(
    root: Path,
    trusted: dict[str, Path],
    fold: int,
    rows: list[dict[str, object]],
    training_ids: list[str],
) -> FoldModelArtifacts:
    directory = root / f"fold-{fold}"
    directory.mkdir(parents=True)
    checkpoint = directory / "model.pt"
    scoring_ids = tuple(str(row["crop_id"]) for row in rows)
    torch.save(
        {
            "held_out_fold": fold,
            "parent_checkpoint_sha256": _sha256(trusted["adapted"]),
            "real_manifest_sha256": _sha256(trusted["real"]),
            "crop_manifest_sha256": _sha256(trusted["crops"]),
            "gold_manifest_sha256": _sha256(trusted["gold"]),
            "fold_manifest_sha256": _sha256(trusted["folds"]),
            "synthetic_manifest_sha256": _sha256(trusted["synthetic"]),
            "real_training_crop_ids": training_ids,
            "scoring_crop_ids": list(scoring_ids),
        },
        checkpoint,
    )
    checkpoint_hash = _sha256(checkpoint)
    for row in rows:
        row.update(
            {
                "fold": fold,
                "checkpoint_sha256": checkpoint_hash,
                "parent_checkpoint_sha256": _sha256(trusted["adapted"]),
                "gold_manifest_sha256": _sha256(trusted["gold"]),
                "fold_manifest_sha256": _sha256(trusted["folds"]),
                "model_id": f"real-fold-{fold}",
            }
        )
    scores = _write(directory / "scores.parquet", rows)
    metrics = directory / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")
    return FoldModelArtifacts(fold, checkpoint, scores, metrics, scoring_ids)


def _artifacts(tmp_path: Path, trusted: dict[str, Path]) -> list[FoldModelArtifacts]:
    return [
        _artifact(
            tmp_path,
            trusted,
            0,
            [
                {
                    "crop_id": "a-1",
                    "image_id": "a",
                    "decision": "PASS",
                    "anomaly_kind": "NONE",
                    "risk_score": 0.1,
                },
                {
                    "crop_id": "b-1",
                    "image_id": "b",
                    "decision": "BLOCK",
                    "anomaly_kind": "missing_stroke",
                    "risk_score": 0.9,
                },
            ],
            ["c-1"],
        ),
        _artifact(
            tmp_path,
            trusted,
            1,
            [
                {
                    "crop_id": "c-1",
                    "image_id": "c",
                    "decision": "PASS",
                    "anomaly_kind": "NONE",
                    "risk_score": 0.2,
                }
            ],
            ["a-1", "b-1"],
        ),
    ]


def _collect(
    artifacts: list[FoldModelArtifacts], trusted: dict[str, Path], output: Path
) -> Path:
    return collect_oof_scores(
        artifacts,
        trusted["folds"],
        trusted["real"],
        trusted["crops"],
        trusted["gold"],
        trusted["adapted"],
        trusted["synthetic"],
        output,
    )


def _rewrite_checkpoint(
    artifact: FoldModelArtifacts,
    mutate: object,
) -> None:
    checkpoint = torch.load(artifact.checkpoint, map_location="cpu", weights_only=True)
    assert callable(mutate)
    mutate(checkpoint)
    torch.save(checkpoint, artifact.checkpoint)
    rows = pq.read_table(artifact.scores).to_pylist()
    for row in rows:
        row["checkpoint_sha256"] = _sha256(artifact.checkpoint)
    _write(artifact.scores, rows)


def test_collect_oof_uses_trusted_contract_and_reports_fold_metrics(tmp_path: Path) -> None:
    trusted = _trusted_inputs(tmp_path)
    artifacts = _artifacts(tmp_path, trusted)

    output = _collect(artifacts, trusted, tmp_path / "oof")

    rows = pq.read_table(output).to_pylist()
    assert [row["crop_id"] for row in rows] == ["a-1", "b-1", "c-1"]
    metrics = json.loads((output.parent / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["folds"]["0"]["aucpr"] == pytest.approx(1.0)
    assert metrics["folds"]["0"]["anomaly_kind_recall"]["missing_stroke"] == 1.0
    assert metrics["folds"]["1"]["status"] == "inconclusive_single_class"
    assert metrics["locked_test_rows"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "synchronized_drop",
        "label",
        "anomaly_kind",
        "training_contains_heldout",
        "parent_hash",
        "gold_hash",
        "model_id",
    ],
)
def test_collect_oof_rejects_artifact_claims_that_disagree_with_trusted_inputs(
    tmp_path: Path, mutation: str
) -> None:
    trusted = _trusted_inputs(tmp_path)
    artifacts = _artifacts(tmp_path, trusted)
    target = artifacts[0]
    rows = pq.read_table(target.scores).to_pylist()
    if mutation == "synchronized_drop":
        rows = rows[:1]
        target = FoldModelArtifacts(
            target.held_out_fold,
            target.checkpoint,
            target.scores,
            target.metrics,
            ("a-1",),
        )
        artifacts[0] = target
        _rewrite_checkpoint(target, lambda data: data.update(scoring_crop_ids=["a-1"]))
        rows = pq.read_table(target.scores).to_pylist()[:1]
    elif mutation == "label":
        rows[0]["decision"] = "BLOCK"
    elif mutation == "anomaly_kind":
        rows[1]["anomaly_kind"] = "invented_character"
    elif mutation == "training_contains_heldout":
        _rewrite_checkpoint(
            target, lambda data: data.update(real_training_crop_ids=["a-1", "c-1"])
        )
        rows = pq.read_table(target.scores).to_pylist()
    elif mutation == "parent_hash":
        _rewrite_checkpoint(target, lambda data: data.update(parent_checkpoint_sha256="0" * 64))
        rows = pq.read_table(target.scores).to_pylist()
    elif mutation == "gold_hash":
        rows[0]["gold_manifest_sha256"] = "0" * 64
    else:
        rows[0]["model_id"] = "forged-model"
    _write(target.scores, rows)

    with pytest.raises(ValueError):
        _collect(artifacts, trusted, tmp_path / "oof")
    assert not (tmp_path / "oof" / "oof.parquet").exists()
