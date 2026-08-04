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


def _artifact(
    root: Path,
    fold_manifest: Path,
    fold: int,
    rows: list[dict[str, object]],
) -> FoldModelArtifacts:
    directory = root / f"fold-{fold}"
    directory.mkdir(parents=True)
    checkpoint = directory / "model.pt"
    scoring_ids = tuple(str(row["crop_id"]) for row in rows)
    torch.save(
        {
            "held_out_fold": fold,
            "fold_manifest_sha256": _sha256(fold_manifest),
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
                "fold_manifest_sha256": _sha256(fold_manifest),
                "model_id": f"real-fold-{fold}",
            }
        )
    scores = directory / "scores.parquet"
    pq.write_table(pa.Table.from_pylist(rows), scores)
    metrics = directory / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")
    return FoldModelArtifacts(
        held_out_fold=fold,
        checkpoint=checkpoint,
        scores=scores,
        metrics=metrics,
        scoring_crop_ids=scoring_ids,
    )


def test_collect_oof_writes_exactly_one_dev_score_and_inconclusive_single_class_metrics(
    tmp_path: Path,
) -> None:
    folds = tmp_path / "folds.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"image_id": "a", "fold": 0, "split_role": "DEV"},
                {"image_id": "b", "fold": 0, "split_role": "DEV"},
                {"image_id": "c", "fold": 1, "split_role": "DEV"},
                {"image_id": "locked", "fold": -1, "split_role": "LOCKED_TEST"},
            ]
        ),
        folds,
    )
    artifacts = [
        _artifact(
            tmp_path,
            folds,
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
        ),
        _artifact(
            tmp_path,
            folds,
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
        ),
    ]

    output = collect_oof_scores(artifacts, folds, tmp_path / "oof")

    rows = pq.read_table(output).to_pylist()
    assert [row["crop_id"] for row in rows] == ["a-1", "b-1", "c-1"]
    assert len({row["crop_id"] for row in rows}) == 3
    metrics = json.loads((output.parent / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["folds"]["0"]["aucpr"] == pytest.approx(1.0)
    assert metrics["folds"]["0"]["bootstrap_aucpr_95"][0] is not None
    assert metrics["folds"]["0"]["anomaly_kind_recall"]["missing_stroke"] == 1.0
    assert metrics["folds"]["1"]["aucpr"] is None
    assert metrics["folds"]["1"]["status"] == "inconclusive_single_class"
    assert metrics["locked_test_rows"] == 0


@pytest.mark.parametrize(
    "mutation", ["duplicate", "wrong_hash", "wrong_fold", "checkpoint_fold", "locked"]
)
def test_collect_oof_fails_closed_on_invalid_fold_outputs(
    tmp_path: Path, mutation: str
) -> None:
    folds = tmp_path / "folds.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"image_id": "a", "fold": 0, "split_role": "DEV"},
                {"image_id": "locked", "fold": -1, "split_role": "LOCKED_TEST"},
            ]
        ),
        folds,
    )
    rows = [
        {
            "crop_id": "a-1",
            "image_id": "a",
            "decision": "PASS",
            "anomaly_kind": "NONE",
            "risk_score": 0.1,
        }
    ]
    artifact = _artifact(tmp_path, folds, 0, rows)
    stored = pq.read_table(artifact.scores).to_pylist()
    if mutation == "duplicate":
        stored.append(dict(stored[0]))
    elif mutation == "wrong_hash":
        stored[0]["fold_manifest_sha256"] = "0" * 64
    elif mutation == "wrong_fold":
        stored[0]["fold"] = 1
    elif mutation == "checkpoint_fold":
        torch.save(
            {
                "held_out_fold": 1,
                "fold_manifest_sha256": _sha256(folds),
                "scoring_crop_ids": ["a-1"],
            },
            artifact.checkpoint,
        )
        stored[0]["checkpoint_sha256"] = _sha256(artifact.checkpoint)
    else:
        stored[0]["image_id"] = "locked"
        stored[0]["fold"] = -1
    pq.write_table(pa.Table.from_pylist(stored), artifact.scores)

    with pytest.raises(ValueError):
        collect_oof_scores([artifact], folds, tmp_path / "oof")
    assert not (tmp_path / "oof" / "oof.parquet").exists()
