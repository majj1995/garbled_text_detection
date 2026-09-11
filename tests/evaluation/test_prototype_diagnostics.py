"""Frozen, paired diagnostics must not turn calibration normals into references."""

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
from torch import nn

from poor_word.evaluation import prototype_diagnostics as module
from poor_word.models.prototypes import PrototypeBank
from poor_word.training.train_glyph import TrainConfig


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TinyClassifier(nn.Module):
    """Replace only expensive ConvNeXt, retaining checkpoint loading and tensor inference."""

    def __init__(self, character_count: int, config: TrainConfig) -> None:
        super().__init__()
        assert character_count == 3
        assert config.pretrained is False  # A saved pretrained=True must never download weights.
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert not self.training and not torch.is_grad_enabled()
        assert not self.weight.requires_grad
        level = views[:, 0].mean(dim=(1, 2)) * self.weight
        embedding = torch.nn.functional.normalize(torch.stack((level, 1 - level), dim=1))
        return embedding, embedding


def _split(root: Path, role: str, samples: list[tuple[str, str, int]]) -> Path:
    root.mkdir()
    rows = []
    for index, (char, decision, level) in enumerate(samples):
        sample_id = f"{role}-{index}"
        rgb = np.full((16, 16, 3), level, dtype=np.uint8)
        path = root / f"{sample_id}.png"
        mask = root / f"{sample_id}-mask.png"
        Image.fromarray(rgb).save(path)
        Image.fromarray(rgb[:, :, 0]).save(mask)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": path.name,
                "mask_path": mask.name,
                "base_char": char,
                "rendered_char": char,
                "decision": decision,
                "anomaly_kind": "none" if decision == "PASS" else "bridge",
                "operator": "identity" if decision == "PASS" else "bridge",
                "changed_pixels": 0 if decision == "PASS" else 12,
                "seed": index,
                "bbox": {"x0": 0, "y0": 0, "x1": 16, "y1": 16},
                "source_asset_ids": ["fixture"],
                "schema_version": "glyph-dataset-v2",
                "dataset_id": "diagnostic-fixture",
                "split_role": role,
                "source_group_id": sample_id,
                "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "image_sha256": _hash(path),
                "mask_sha256": _hash(mask),
                "training_eligible": role == "train",
                "production_allowed": False,
                "label_provenance": "synthetic_rule_v2",
            }
        )
    manifest = root / f"{role}.parquet"
    _write_rows(manifest, rows)
    return manifest


def _write_rows(manifest: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    (manifest.parent / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "glyph-dataset-v2",
                "dataset_id": "diagnostic-fixture",
                "experimental_only": True,
                "production_allowed": False,
                "manifests": {
                    manifest.name: {"sha256": _hash(manifest), "row_count": len(rows)},
                    "test.parquet": {"sha256": "a" * 64, "row_count": 1},
                },
            }
        )
    )


def _bind_artifacts(artifacts: Path, membership: dict[str, Any]) -> None:
    path = artifacts / "training_membership.json"
    path.write_text(json.dumps(membership))
    checkpoint = torch.load(artifacts / "encoder.pt", weights_only=True)
    checkpoint["training_membership_sha256"] = _hash(path)
    torch.save(checkpoint, artifacts / "encoder.pt")
    metadata = {
        "dataset_schema_version": "glyph-dataset-v2",
        "dataset_id": "diagnostic-fixture",
        "experimental_only": "true",
        "production_allowed": "false",
        "training_membership_sha256": _hash(path),
        "encoder_checkpoint_sha256": _hash(artifacts / "encoder.pt"),
        "prototype_bank_sha256": _hash(artifacts / "prototypes.npz"),
        "source_manifest_sha256": membership["manifest_sha256"],
        "catalog_sha256": hashlib.sha256("字文永".encode()).hexdigest(),
    }
    (artifacts / "prototypes.json").write_text(json.dumps(metadata))


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    train = _split(
        tmp_path / "train",
        "train",
        [
            ("文", "PASS", 40),
            ("文", "PASS", 80),
            ("字", "PASS", 120),
            ("永", "PASS", 150),
            ("文", "BLOCK", 200),
            ("字", "BLOCK", 210),
        ],
    )
    cal = _split(
        tmp_path / "cal",
        "calibration",
        [
            ("文", "PASS", 50),
            ("文", "PASS", 100),
            ("字", "PASS", 130),
            ("永", "PASS", 160),
            ("文", "BLOCK", 220),
        ],
    )
    artifacts = tmp_path / "model"
    artifacts.mkdir()
    config = TrainConfig(manifest=train, output_dir=artifacts, embedding_dim=2, device="cpu")
    torch.save(
        {
            "dataset_schema_version": "glyph-dataset-v2",
            "dataset_id": "diagnostic-fixture",
            "experimental_only": True,
            "production_allowed": False,
            "config": {**config.model_dump(mode="json"), "pretrained": True},
            "char_to_id": {"字": 0, "文": 1, "永": 2},
            "model_state": TinyClassifier(3, config).state_dict(),
        },
        artifacts / "encoder.pt",
    )
    np.savez(
        artifacts / "prototypes.npz",
        centers=np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32),
        center_labels=np.asarray(["文", "文", "字", "永"]),
        max_prototypes_per_char=np.asarray(8),
        random_state=np.asarray(0),
    )
    rows = pq.read_table(train).to_pylist()
    _bind_artifacts(
        artifacts,
        {
            "schema_version": "glyph-training-membership-v2",
            "dataset_id": "diagnostic-fixture",
            "manifest_sha256": _hash(train),
            "sample_ids": [r["sample_id"] for r in rows],
            "source_group_ids": [r["source_group_id"] for r in rows],
            "pixel_sha256": [r["pixel_sha256"] for r in rows],
        },
    )
    monkeypatch.setattr(module, "GlyphClassifier", TinyClassifier)
    return dict(
        train_manifest=train,
        calibration_manifest=cal,
        artifacts_dir=artifacts,
        output_dir=tmp_path / "diagnosis",
        characters="文字文",
        allow_experimental=True,
    )


def test_cosine_comparison_keeps_global_own_prototype_and_raw_references_distinct(tmp_path: Path):
    # Query 文=[.6,.8]: global nearest 字=.2, own prototype=.4, own raw exact match=0.
    np.savez(
        tmp_path / "bank.npz",
        centers=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        center_labels=np.asarray(["文", "字"]),
        max_prototypes_per_char=8,
        random_state=0,
    )
    (tmp_path / "bank.json").write_text("{}")
    bank, _ = PrototypeBank.load(tmp_path / "bank.npz")
    compared = module.compare_prototype_distances(
        bank,
        np.asarray([[0.6, 0.8]], dtype=np.float32),
        ["文"],
        np.asarray([[0.6, 0.8], [0, 1]], dtype=np.float32),
        ["文", "字"],
    )
    assert compared[0]["global_score"] == pytest.approx(0.2)
    assert compared[0]["same_char_prototype_distance"] == pytest.approx(0.4)
    assert compared[0]["same_char_train_nn_distance"] == pytest.approx(0, abs=1e-6)
    assert compared[0]["global_nearest_char"] == "字"
    assert compared[0]["nearest_train_index"] == 0


def test_diagnostic_loads_only_selected_normals_and_leaves_all_inputs_unchanged(inputs):
    # Invalid unused assets catch accidentally inferring BLOCKs, other chars, or test data.
    for key in ("train_manifest", "calibration_manifest"):
        root = inputs[key].parent
        (root / "test.parquet").write_bytes(b"not readable parquet")
        for row in pq.read_table(inputs[key]).to_pylist():
            if row["decision"] == "BLOCK" or row["base_char"] == "永":
                (root / row["image_path"]).write_bytes(b"not an image")
    source_files = [
        p
        for key in ("train_manifest", "calibration_manifest", "artifacts_dir")
        for p in (inputs[key].parent if key.endswith("manifest") else inputs[key]).iterdir()
        if p.is_file()
    ]
    before = {p: _hash(p) for p in source_files}
    logs = []
    result = module.diagnose_prototypes(**inputs, batch_size=1, progress=logs.append)
    report = json.loads(result.json_path.read_text())
    assert len(report["samples"]) == 3
    assert report["characters"] == ["文", "字"]
    assert report["reference_count"] == 3
    assert report["query_count"] == 3
    assert report["test_data_used"] is False and report["production_allowed"] is False
    assert {r["sample_id"] for r in report["samples"]} == {
        "calibration-0",
        "calibration-1",
        "calibration-2",
    }
    for row in report["samples"]:
        assert row["nearest_train_sample_id"] in {"train-0", "train-1", "train-2"}
        assert row["train_normal_count"] == (2 if row["base_char"] == "文" else 1)
    assert len(result.summary_lines) == 2
    assert len(logs) >= 3
    # Summary must show all three values for the SAME highest-global query, not separate maxima.
    for summary in report["summary"]:
        own_rows = [r for r in report["samples"] if r["base_char"] == summary["base_char"]]
        assert summary == max(own_rows, key=lambda r: r["global_score"])
    assert result.markdown_path.is_file()
    assert {p: _hash(p) for p in source_files} == before


@pytest.mark.parametrize(
    "change, message",
    [
        ({"allow_experimental": False}, "allow-experimental"),
        ({"characters": ""}, "characters"),
        ({"characters": "无"}, "normal"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_invalid_request_cannot_create_diagnostic_output(inputs, change, message):
    with pytest.raises(ValueError, match=message):
        module.diagnose_prototypes(**(inputs | change))
    assert not inputs["output_dir"].exists()


def test_existing_and_nested_output_directories_are_never_modified(inputs):
    output = inputs["output_dir"]
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        module.diagnose_prototypes(**inputs)
    assert marker.read_text() == "keep"
    for root in (inputs["train_manifest"].parent, inputs["artifacts_dir"]):
        with pytest.raises(ValueError, match="outside"):
            module.diagnose_prototypes(**(inputs | {"output_dir": root / "new"}))
        assert not (root / "new").exists()


def test_train_manifest_must_match_checkpoint_membership_hash(inputs):
    artifacts = inputs["artifacts_dir"]
    membership = json.loads((artifacts / "training_membership.json").read_text())
    membership["manifest_sha256"] = "f" * 64
    _bind_artifacts(artifacts, membership)
    with pytest.raises(ValueError, match="training manifest"):
        module.diagnose_prototypes(**inputs)
    assert not inputs["output_dir"].exists()


def test_membership_identity_associations_not_just_sets_must_match(inputs):
    artifacts = inputs["artifacts_dir"]
    membership = json.loads((artifacts / "training_membership.json").read_text())
    membership["pixel_sha256"].reverse()
    _bind_artifacts(artifacts, membership)
    with pytest.raises(ValueError, match="membership"):
        module.diagnose_prototypes(**inputs)


def test_test_role_cannot_be_passed_as_calibration(inputs):
    manifest = inputs["calibration_manifest"]
    rows = pq.read_table(manifest).to_pylist()
    for row in rows:
        row["split_role"] = "test"
    _write_rows(manifest, rows)
    with pytest.raises(ValueError, match="calibration"):
        module.diagnose_prototypes(**inputs)


def test_calibration_source_groups_cannot_overlap_training(inputs):
    manifest = inputs["calibration_manifest"]
    rows = pq.read_table(manifest).to_pylist()
    rows[0]["source_group_id"] = "train-0"
    _write_rows(manifest, rows)
    with pytest.raises(ValueError, match="overlap"):
        module.diagnose_prototypes(**inputs)


def test_selected_image_corruption_fails_without_publishing_partial_results(inputs):
    (inputs["calibration_manifest"].parent / "calibration-0.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.diagnose_prototypes(**inputs)
    assert not inputs["output_dir"].exists()
    assert not list(inputs["output_dir"].parent.glob(".diagnosis-*"))
