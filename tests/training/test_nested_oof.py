import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

import poor_word.training.nested_oof as nested_module
from poor_word.training.nested_oof import NestedOofConfig, train_nested_real_oof
from poor_word.training.train_mil import MilOofConfig, _nested_feature_paths


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _config(tmp_path: Path) -> NestedOofConfig:
    folds = _write(
        tmp_path / "folds.parquet",
        [
            {"image_id": f"image-{fold}", "fold": fold, "split_role": "DEV"}
            for fold in range(5)
        ],
    )
    inputs = []
    for name in ("real", "crops", "gold", "synthetic", "adapted"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        inputs.append(path)
    return NestedOofConfig(
        real_manifest=inputs[0],
        crop_manifest=inputs[1],
        gold_manifest=inputs[2],
        fold_manifest=folds,
        synthetic_manifest=inputs[3],
        adapted_checkpoint=inputs[4],
        output_dir=tmp_path / "nested",
        device="cpu",
        max_steps=1,
    )


@pytest.mark.parametrize("leak_excluded_crop", [False, True])
def test_nested_oof_binds_each_outer_feature_to_actual_training_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leak_excluded_crop: bool,
) -> None:
    config = _config(tmp_path)

    def fake_finetune(
        fold_config: object,
        held_out_fold: int,
        additionally_excluded_fold: int | None,
    ) -> object:
        excluded = sorted(
            {
                held_out_fold,
                *(
                    []
                    if additionally_excluded_fold is None
                    else [additionally_excluded_fold]
                ),
            }
        )
        output = fold_config.output_dir
        output.mkdir(parents=True)
        checkpoint = output / "model.pt"
        training_ids = [f"crop-{fold}" for fold in range(5) if fold not in excluded]
        if leak_excluded_crop and additionally_excluded_fold is not None:
            training_ids.append(f"crop-{additionally_excluded_fold}")
        torch.save(
            {
                "held_out_fold": held_out_fold,
                "excluded_folds": excluded,
                "fold_manifest_sha256": _sha(config.fold_manifest),
                "scoring_crop_ids": [f"crop-{held_out_fold}"],
                "real_training_crop_ids": training_ids,
            },
            checkpoint,
        )
        scores = _write(
            output / "scores.parquet",
            [
                {
                    "crop_id": f"crop-{held_out_fold}",
                    "image_id": f"image-{held_out_fold}",
                    "fold": held_out_fold,
                    "risk_score": 0.5,
                    "model_id": f"real-fold-{held_out_fold}",
                    "checkpoint_sha256": _sha(checkpoint),
                    "fold_manifest_sha256": _sha(config.fold_manifest),
                    "excluded_folds": excluded,
                }
            ],
        )
        metrics = output / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "held_out_fold": held_out_fold,
                    "excluded_folds": excluded,
                    "checkpoint_sha256": _sha(checkpoint),
                    "scores_sha256": _sha(scores),
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            checkpoint=checkpoint,
            scores=scores,
            metrics=metrics,
            scoring_crop_ids=(f"crop-{held_out_fold}",),
            excluded_folds=tuple(excluded),
        )

    monkeypatch.setattr(nested_module, "finetune_real_fold", fake_finetune)
    if leak_excluded_crop:
        with pytest.raises(ValueError, match="excluded-fold crop"):
            train_nested_real_oof(config)
        assert not config.output_dir.exists()
        return
    artifacts = train_nested_real_oof(config)

    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["kind"] == "nested_character_oof"
    for outer in manifest["outer_folds"]:
        outer_fold = outer["outer_fold"]
        rows = pq.read_table(config.output_dir / outer["features"]).to_pylist()
        assert {
            (row["fold"], tuple(row["excluded_folds"])) for row in rows
        } == {(fold, tuple(sorted({outer_fold, fold}))) for fold in range(5)}
    resolved, manifest_hash = _nested_feature_paths(
        MilOofConfig(
            real_manifest=config.real_manifest,
            fold_manifest=config.fold_manifest,
            nested_feature_dir=config.output_dir,
            output_dir=tmp_path / "mil-oof",
            device="cpu",
            max_steps=1,
        )
    )
    assert set(resolved) == set(range(5))
    assert manifest_hash == _sha(artifacts.manifest)
