from pathlib import Path

import pytest

from poor_word.evaluation.glyph_v2 import _commands


@pytest.mark.parametrize(
    ("saved_augmentation", "expected_option"),
    [(None, "--augmentation none"), ("affine", "--augmentation affine")],
)
def test_reproduction_command_carries_actual_training_augmentation(
    tmp_path: Path, saved_augmentation: str | None, expected_option: str
) -> None:
    """Break caught: reproduction silently changes or omits the checkpoint's augmentation."""
    config = {
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
    if saved_augmentation is not None:
        config["augmentation"] = saved_augmentation

    commands = _commands(
        tmp_path / "calibration.parquet",
        tmp_path / "test.parquet",
        tmp_path / "model",
        tmp_path / "report",
        checkpoint={"config": config},
        run={"config": {}},
        prevalence=0.001,
        max_fpr=0.0001,
        device="cpu",
        batch_size=64,
    )

    assert expected_option in commands[1]
