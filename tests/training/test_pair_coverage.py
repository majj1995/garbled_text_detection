import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from poor_word.training.dataset import GlyphDataset
from poor_word.training.pair_coverage import audit_pair_coverage
from poor_word.training.train_glyph import TrainConfig


def _dataset(tmp_path: Path, samples: list[tuple[str, str, str]]) -> GlyphDataset:
    rows = [
        {
            "sample_id": f"sample-{index}",
            "base_char": character,
            "decision": decision,
            "source_asset_ids": [source],
            "image_path": f"missing/image-{index}.png",
            "mask_path": f"missing/mask-{index}.png",
        }
        for index, (character, decision, source) in enumerate(samples)
    ]
    manifest = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    return GlyphDataset(manifest)


def _config(
    dataset: GlyphDataset, *, batch_size: int, epochs: int = 1, max_steps: int | None = None
) -> TrainConfig:
    return TrainConfig(
        manifest=dataset.manifest,
        output_dir=dataset.root / "unused-training-output",
        seed=7,
        batch_size=batch_size,
        epochs=epochs,
        max_steps=max_steps,
        device="cpu",
    )


def test_distinct_normal_labels_have_no_eligible_anchors(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [("甲", "PASS", "0"), ("乙", "PASS", "0"), ("丁", "PASS", "0")])

    result = audit_pair_coverage(dataset, _config(dataset, batch_size=3))

    assert result["train_samples"] == 3
    assert result["normal_draws"] == 3
    assert result["eligible_normal_draws"] == 0
    assert result["eligible_normal_fraction"] == 0.0
    assert result["positive_pairs"] == 0
    assert result["batches_without_positive_pairs"] == 1
    assert result["batches_without_positive_pairs_fraction"] == 1.0
    assert result["expected_steps_matches"] is None


def test_pairs_are_unordered_and_exclude_block_samples(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [
            ("甲", "PASS", "0"),
            ("甲", "PASS", "0"),
            ("甲", "PASS", "0"),
            ("乙", "PASS", "0"),
            ("乙", "BLOCK", "0"),
            ("甲", "BLOCK", "0"),
        ],
    )

    result = audit_pair_coverage(dataset, _config(dataset, batch_size=6))

    assert result["normal_draws"] == 4
    assert result["eligible_normal_draws"] == 3
    assert result["eligible_normal_fraction"] == 0.75
    assert result["positive_pairs"] == 3
    assert result["batches_without_positive_pairs"] == 0
    assert result["batches_without_positive_pairs_fraction"] == 0.0


def test_no_normal_samples_has_zero_eligible_fraction(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [("甲", "BLOCK", "0"), ("甲", "BLOCK", "0")])

    result = audit_pair_coverage(dataset, _config(dataset, batch_size=2))

    assert result["normal_draws"] == 0
    assert result["eligible_normal_draws"] == 0
    assert result["eligible_normal_fraction"] == 0.0
    assert result["positive_pairs"] == 0
    assert result["batches_without_positive_pairs"] == 1


def test_validation_samples_are_excluded_when_training_split_exists(tmp_path: Path) -> None:
    # For 甲, source 0 hashes into training and source 3 into validation.
    dataset = _dataset(tmp_path, [("甲", "PASS", "0"), ("甲", "PASS", "3")])

    result = audit_pair_coverage(dataset, _config(dataset, batch_size=2))

    assert result["train_samples"] == 1
    assert result["normal_draws"] == 1
    assert result["positive_pairs"] == 0


def test_empty_training_split_falls_back_to_all_samples(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [("甲", "PASS", "3"), ("甲", "PASS", "3")])

    result = audit_pair_coverage(dataset, _config(dataset, batch_size=2))

    assert result["train_samples"] == 2
    assert result["normal_draws"] == 2
    assert result["eligible_normal_draws"] == 2
    assert result["positive_pairs"] == 1


def test_sampler_replay_includes_partial_batches_and_stops_mid_epoch(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [
            ("甲", "PASS", "0"),
            ("甲", "PASS", "0"),
            ("乙", "PASS", "0"),
            ("乙", "BLOCK", "0"),
            ("乙", "PASS", "0"),
        ],
    )
    config = _config(dataset, batch_size=2, epochs=4, max_steps=5)

    result = audit_pair_coverage(dataset, config, expected_steps=6)

    # Seed 7 orders: [0, 1, 3, 2, 4], then [3, 4, 1, 2, 0].
    # First epoch has (甲,甲), (BLOCK,乙), (乙); second stops after
    # (BLOCK,乙), (甲,乙). Only the first batch supplies a positive pair.
    assert result["replayed_steps"] == 5
    assert result["train_samples"] == 5
    assert result["batch_size"] == 2
    assert result["epochs_started"] == 2
    assert result["epochs_completed"] == 1
    assert result["normal_draws"] == 7
    assert result["eligible_normal_draws"] == 2
    assert result["eligible_normal_fraction"] == pytest.approx(2 / 7)
    assert result["positive_pairs"] == 1
    assert result["batches_without_positive_pairs"] == 4
    assert result["batches_without_positive_pairs_fraction"] == 0.8
    assert result["expected_steps_matches"] is False
    assert result["per_epoch"] == [
        {
            "epoch": 1,
            "completed": True,
            "replayed_steps": 3,
            "normal_draws": 4,
            "eligible_normal_draws": 2,
            "eligible_normal_fraction": 0.5,
            "positive_pairs": 1,
            "batches_without_positive_pairs": 2,
            "batches_without_positive_pairs_fraction": 2 / 3,
        },
        {
            "epoch": 2,
            "completed": False,
            "replayed_steps": 2,
            "normal_draws": 3,
            "eligible_normal_draws": 0,
            "eligible_normal_fraction": 0.0,
            "positive_pairs": 0,
            "batches_without_positive_pairs": 2,
            "batches_without_positive_pairs_fraction": 1.0,
        },
    ]


@pytest.mark.parametrize("max_steps", [None, 4, 20])
def test_epoch_limit_and_step_limit_on_epoch_boundary(
    tmp_path: Path, max_steps: int | None
) -> None:
    dataset = _dataset(tmp_path, [("甲", "PASS", "0"), ("甲", "PASS", "0"), ("甲", "PASS", "0")])
    config = _config(dataset, batch_size=2, epochs=2, max_steps=max_steps)

    result = audit_pair_coverage(dataset, config, expected_steps=4)

    assert result["replayed_steps"] == 4
    assert result["epochs_started"] == 2
    assert result["epochs_completed"] == 2
    assert result["normal_draws"] == 6
    assert result["eligible_normal_draws"] == 4
    assert result["positive_pairs"] == 2
    assert result["batches_without_positive_pairs"] == 2
    assert result["expected_steps_matches"] is True


def test_replay_is_deterministic_without_changing_global_torch_rng(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [("甲", "PASS", "0"), ("甲", "PASS", "0"), ("乙", "PASS", "0")],
    )
    config = _config(dataset, batch_size=2, epochs=3)
    rng_state = torch.get_rng_state().clone()

    first = audit_pair_coverage(dataset, config)
    second = audit_pair_coverage(dataset, config)

    assert first == second
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_replay_needs_no_pixels_and_returns_json_safe_data(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [("甲", "PASS", "0"), ("甲", "PASS", "0")])
    config = _config(dataset, batch_size=2)
    # The real dataset fails if any caller attempts to read these absent pixels.
    with pytest.raises(ValueError, match="image or mask file is missing"):
        dataset[0]

    result = audit_pair_coverage(dataset, config)

    assert result["positive_pairs"] == 1
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert not config.output_dir.exists()


def test_paired_replay_reports_complete_positive_pair_coverage(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        [
            ("甲", "PASS", "0"),
            ("甲", "PASS", "0"),
            ("乙", "PASS", "0"),
            ("乙", "PASS", "0"),
            ("甲", "BLOCK", "0"),
            ("乙", "BLOCK", "0"),
        ],
    )
    config = _config(dataset, batch_size=4).model_copy(update={"sampler": "paired"})

    result = audit_pair_coverage(dataset, config, expected_steps=2)

    assert result["replayed_steps"] == 2
    assert result["normal_draws"] == 4
    assert result["eligible_normal_draws"] == 4
    assert result["eligible_normal_fraction"] == 1.0
    assert result["positive_pairs"] == 2
    assert result["batches_without_positive_pairs"] == 0
    assert result["expected_steps_matches"] is True
