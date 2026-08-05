import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

import poor_word.training.train_mil as mil_module
from poor_word.training.train_mil import MilOofConfig, train_mil_oof


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _config(tmp_path: Path) -> MilOofConfig:
    real = _write(
        tmp_path / "real.parquet",
        [
            {
                "image_id": f"image-{fold}",
                "image_label": "ABNORMAL" if fold % 2 else "NORMAL",
                "split_role": "DEV",
                "training_eligible": True,
            }
            for fold in range(5)
        ]
        + [
            {
                "image_id": "locked",
                "image_label": "LOCKED_LABEL_MUST_NOT_BE_READ",
                "split_role": "LOCKED_TEST",
                "training_eligible": False,
            }
        ],
    )
    folds = _write(
        tmp_path / "folds.parquet",
        [
            {
                "image_id": f"image-{fold}",
                "fold": fold,
                "split_role": "DEV",
                "component_id": f"component-{fold}",
            }
            for fold in range(5)
        ]
        + [
            {
                "image_id": "locked",
                "fold": -1,
                "split_role": "LOCKED_TEST",
                "component_id": "locked",
            }
        ],
    )
    features = _write(tmp_path / "features.parquet", [{"crop_id": "placeholder"}])
    return MilOofConfig(
        real_manifest=real,
        fold_manifest=folds,
        feature_manifest=features,
        output_dir=tmp_path / "mil-oof",
        device="cpu",
        max_steps=1,
    )


def test_mil_oof_runs_all_folds_and_atomically_collects_every_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)

    def fake_train(fold_config: object) -> object:
        fold = fold_config.held_out_fold
        output = fold_config.output_dir
        output.mkdir()
        checkpoint = output / "model.pt"
        validation_ids = [f"image-{fold}"]
        torch.save(
            {
                "held_out_fold": fold,
                "validation_image_ids": validation_ids,
                "train_image_ids": [f"image-{other}" for other in range(5) if other != fold],
                "real_manifest_sha256": _sha(config.real_manifest),
                "fold_manifest_sha256": _sha(config.fold_manifest),
                "feature_manifest_sha256": _sha(config.feature_manifest),
            },
            checkpoint,
        )
        score = _write(
            output / "image-scores.parquet",
            [
                {
                    "image_id": f"image-{fold}",
                    "fold": fold,
                    "held_out_fold": fold,
                    "image_label": "ABNORMAL" if fold % 2 else "NORMAL",
                    "risk_score": 0.8 if fold % 2 else 0.2,
                    "zero_character": fold == 4,
                    "checkpoint_sha256": _sha(checkpoint),
                    "real_manifest_sha256": _sha(config.real_manifest),
                    "fold_manifest_sha256": _sha(config.fold_manifest),
                    "feature_manifest_sha256": _sha(config.feature_manifest),
                }
            ],
        )
        metrics = output / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "held_out_fold": fold,
                    "validation_image_ids": validation_ids,
                    "checkpoint_sha256": _sha(checkpoint),
                    "image_scores_sha256": _sha(score),
                }
            ),
            encoding="utf-8",
        )
        candidates = _write(output / "attention-candidates.parquet", [])
        return SimpleNamespace(
            checkpoint=checkpoint,
            metrics=metrics,
            attention_candidates=candidates,
            image_scores=score,
        )

    monkeypatch.setattr(mil_module, "train_mil_fold", fake_train)
    artifacts = train_mil_oof(config)

    rows = pq.read_table(artifacts.image_oof).to_pylist()
    assert [row["image_id"] for row in rows] == [f"image-{fold}" for fold in range(5)]
    inventory = json.loads(artifacts.inventory.read_text(encoding="utf-8"))
    assert inventory["kind"] == "image_oof"
    assert [item["held_out_fold"] for item in inventory["folds"]] == list(range(5))
    assert all("locked" not in str(row) for row in rows)
    assert (
        _sha(artifacts.image_oof)
        == json.loads(artifacts.metrics.read_text(encoding="utf-8"))["image_oof_sha256"]
    )


def test_mil_oof_failure_removes_whole_staging_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        mil_module,
        "train_mil_fold",
        lambda _config: (_ for _ in ()).throw(RuntimeError("fold failed")),
    )
    with pytest.raises(RuntimeError, match="fold failed"):
        train_mil_oof(config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".mil-oof.part-*"))
