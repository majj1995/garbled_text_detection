from pathlib import Path

from poor_word.training.train_glyph import TrainConfig, train_glyph


def test_train_smoke_writes_checkpoint_and_metrics(
    generated_manifest: Path, tmp_path: Path
) -> None:
    artifacts = train_glyph(
        TrainConfig(
            manifest=generated_manifest,
            output_dir=tmp_path / "train",
            epochs=1,
            max_steps=2,
            batch_size=4,
            seed=11,
            pretrained=False,
            device="cpu",
        )
    )

    assert artifacts.checkpoint.exists()
    assert artifacts.metrics.exists()
    assert artifacts.prototype_bank.exists()
