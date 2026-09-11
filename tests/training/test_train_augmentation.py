from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from poor_word.training.dataset import GlyphDataset
from poor_word.training.train_glyph import TrainConfig, train_glyph


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, *, role: str = "train", v2: bool = True) -> Path:
    root = tmp_path / f"dataset-{role}-{'v2' if v2 else 'legacy'}"
    root.mkdir()
    reference = np.zeros((128, 128), dtype=np.uint8)
    reference[28:100, 42:50] = 255
    reference[28:36, 42:88] = 255
    altered = reference.copy()
    altered[28:36, 62:72] = 0
    archive = root / "layers.npz"
    np.savez(archive, stroke_000=reference, source_char=np.asarray("甲"))

    rows: list[dict[str, object]] = []
    specs = (("PASS", reference), ("PASS", reference), ("BLOCK", altered), ("BLOCK", altered))
    for index, (decision, gray) in enumerate(specs):
        sample_gray = gray.copy()
        sample_gray[110, 10 + index] = 255
        image_path = root / f"image-{index}.png"
        mask_path = root / f"mask-{index}.png"
        rgb = np.repeat(sample_gray[:, :, None], 3, axis=2)
        edit_mask = reference != altered
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(
            ((sample_gray > 8) if decision == "PASS" else edit_mask).astype(np.uint8) * 255
        ).save(mask_path)
        row: dict[str, object] = {
            "sample_id": f"sample-{index}",
            "base_char": "甲",
            "decision": decision,
            "source_asset_ids": ["fixture"],
            "image_path": image_path.name,
            "mask_path": mask_path.name,
        }
        if v2:
            row.update(
                rendered_char="甲",
                anomaly_kind="none" if decision == "PASS" else "broken_stroke",
                operator="identity" if decision == "PASS" else "break_stroke",
                changed_pixels=0 if decision == "PASS" else int(edit_mask.sum()),
                seed=index,
                bbox={"x0": 28, "y0": 28, "x1": 88, "y1": 100},
                schema_version="glyph-dataset-v2",
                dataset_id="train-affine-fixture",
                split_role=role,
                source_group_id="group-甲",
                pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                image_sha256=_sha256(image_path),
                mask_sha256=_sha256(mask_path),
                training_eligible=role == "train",
                production_allowed=False,
                label_provenance="synthetic_rule_v2",
                source_layers_path=archive.name,
                source_layers_sha256=_sha256(archive),
                metrics=json.dumps(
                    {
                        "appearance_scale": 1.0,
                        "appearance_rotation_degrees": 0.0,
                        "appearance_translate_x": 0.0,
                        "appearance_translate_y": 0.0,
                    }
                ),
            )
        rows.append(row)

    manifest = root / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    if v2:
        (root / "run.json").write_text(
            json.dumps(
                {
                    "schema_version": "glyph-dataset-v2",
                    "dataset_id": "train-affine-fixture",
                    "experimental_only": True,
                    "production_allowed": False,
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


class _TinyEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Identity()

    def forward(self, views: Tensor) -> Tensor:
        means = views.mean(dim=(1, 2, 3))
        columns = [means + float(index + 1) for index in range(self.embedding_dim)]
        return torch.nn.functional.normalize(torch.stack(columns, dim=1), dim=1)


class _TinyClassifier(nn.Module):
    instances: ClassVar[list[_TinyClassifier]] = []

    def __init__(self, character_count: int, config: TrainConfig) -> None:
        super().__init__()
        self.encoder = _TinyEncoder(config.embedding_dim)
        self.classifier = nn.Linear(config.embedding_dim, character_count)
        self.forward_inputs: list[Tensor] = []
        self.instances.append(self)

    def forward(self, views: Tensor) -> tuple[Tensor, Tensor]:
        self.forward_inputs.append(views.detach().cpu().clone())
        embeddings = self.encoder(views)
        return embeddings, self.classifier(embeddings)


@pytest.fixture(autouse=True)
def _use_tiny_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("poor_word.training.train_glyph")
    _TinyClassifier.instances.clear()
    monkeypatch.setattr(module, "GlyphClassifier", _TinyClassifier)


def _config(manifest: Path, output_dir: Path, **overrides: Any) -> TrainConfig:
    values: dict[str, Any] = {
        "manifest": manifest,
        "output_dir": output_dir,
        "epochs": 1,
        "max_steps": 1,
        "batch_size": 4,
        "seed": 17,
        "pretrained": False,
        "device": "cpu",
        "embedding_dim": 2,
        "sampler": "paired",
        "allow_experimental": True,
        "log_every": 1,
    }
    values.update(overrides)
    return TrainConfig(**values)


@pytest.mark.parametrize(
    ("v2", "role", "message"),
    [(False, "train", "V2"), (True, "test", "train")],
)
def test_affine_training_rejects_ineligible_manifests_before_model_or_output(
    tmp_path: Path,
    v2: bool,
    role: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: affine mode reaches initialization for legacy or evaluation data."""
    manifest = _manifest(tmp_path, role=role, v2=v2)
    output = tmp_path / "output"
    module = importlib.import_module("poor_word.training.train_glyph")

    def unexpected_model(*_args: object, **_kwargs: object) -> nn.Module:
        raise AssertionError("model must not be initialized")

    monkeypatch.setattr(module, "GlyphClassifier", unexpected_model)

    with pytest.raises(ValueError, match=message):
        train_glyph(_config(manifest, output, augmentation="affine"))

    assert not output.exists()


def test_default_none_never_requests_augmented_items_and_records_disabled_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the backwards-compatible default enters the augmentation path."""
    manifest = _manifest(tmp_path)

    def unexpected_augmentation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("augmentation must be opt-in")

    monkeypatch.setattr(GlyphDataset, "augmented_item", unexpected_augmentation, raising=False)
    artifacts = train_glyph(_config(manifest, tmp_path / "output"))

    checkpoint = torch.load(artifacts.checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert checkpoint["config"]["augmentation"] == "none"
    assert checkpoint["augmentation"] == "none"
    assert checkpoint["augmentation_policy"] is None
    assert metrics["augmentation"] == "none"
    assert metrics["augmentation_policy"] is None
    assert metrics["augmentation_counts"] == {
        "draws": 0,
        "applied": 0,
        "fallback": 0,
        "rejected_attempts": 0,
        "by_decision": {},
        "rejection_reasons": {},
    }


def test_affine_training_uses_augmented_optimization_views_but_plain_bank_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: augmentation leaks into prototype fitting or skips an optimization draw."""
    augmentation = importlib.import_module("poor_word.training.augmentation")
    manifest = _manifest(tmp_path)
    calls: list[tuple[int, int, int, int, int]] = []

    def augmented_item(
        dataset: GlyphDataset,
        index: int,
        *,
        seed: int,
        epoch: int,
        step: int,
        draw: int,
    ) -> tuple[object, object]:
        item = dataset[index]
        calls.append((index, seed, epoch, step, draw))
        trace = augmentation.AugmentationTrace(
            seed=1000 + draw,
            applied=draw in {0, 2},
            attempts=(1, 4, 2, 4)[draw],
            rejection_reasons=(
                (),
                ("clipping", "topology", "clipping", "topology"),
                ("edit_visibility",),
                ("clipping", "topology", "edit_visibility", "clipping"),
            )[draw],
            parameters=(
                {"angle": 1.0, "scale": 1.0, "translate_x": 0.0, "translate_y": 0.0}
                if draw in {0, 2}
                else {}
            ),
        )
        return replace(item, views=torch.full_like(item.views, 0.75)), trace

    monkeypatch.setattr(GlyphDataset, "augmented_item", augmented_item, raising=False)
    messages: list[str] = []
    artifacts = train_glyph(
        _config(
            manifest,
            tmp_path / "output",
            augmentation="affine",
            epochs=2,
            max_steps=2,
        ),
        progress=messages.append,
    )

    assert len(_TinyClassifier.instances) == 1
    recorded = _TinyClassifier.instances[0].forward_inputs
    assert len(recorded) == 3
    assert torch.equal(recorded[0], torch.full_like(recorded[0], 0.75))
    assert torch.equal(recorded[1], torch.full_like(recorded[1], 0.75))
    plain_dataset = GlyphDataset(manifest)
    expected_plain = torch.stack(
        [plain_dataset[index].views for index in range(len(plain_dataset))]
    )
    assert torch.equal(recorded[2], expected_plain)
    assert [(seed, epoch, step, draw) for _, seed, epoch, step, draw in calls] == [
        (17, 1, 0, 0),
        (17, 1, 0, 1),
        (17, 1, 0, 2),
        (17, 1, 0, 3),
        (17, 2, 1, 0),
        (17, 2, 1, 1),
        (17, 2, 1, 2),
        (17, 2, 1, 3),
    ]
    assert {index for index, *_ in calls} == {0, 1, 2, 3}

    checkpoint = torch.load(artifacts.checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    expected_counts = {
        "draws": 8,
        "applied": 4,
        "fallback": 4,
        "rejected_attempts": 18,
        "by_decision": {
            "BLOCK": {"draws": 4, "applied": 2, "fallback": 2, "rejected_attempts": 10},
            "PASS": {"draws": 4, "applied": 2, "fallback": 2, "rejected_attempts": 8},
        },
        "rejection_reasons": {"clipping": 8, "edit_visibility": 4, "topology": 6},
    }
    assert checkpoint["augmentation"] == "affine"
    assert checkpoint["config"]["augmentation"] == "affine"
    assert checkpoint["augmentation_policy"] == augmentation.augmentation_policy()
    assert metrics["augmentation"] == "affine"
    assert metrics["augmentation_policy"] == augmentation.augmentation_policy()
    assert metrics["augmentation_counts"] == expected_counts
    assert any("augmentation affine" in message and "applied=4" in message for message in messages)
