from pathlib import Path

import numpy as np

from poor_word.models.prototypes import PrototypeBank


def test_nearest_prototype_and_margin() -> None:
    bank = PrototypeBank(max_prototypes_per_char=2, random_state=5)
    bank.fit(
        np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        ["甲", "甲", "乙"],
    )

    scores = bank.score(np.array([[1.0, 0.0]], dtype=np.float32))

    assert scores.nearest_chars == ("甲",)
    assert scores.nearest_distance[0] < scores.second_distance[0]
    assert scores.margin[0] > 0


def test_prototype_bank_round_trips_with_audit_metadata(tmp_path: Path) -> None:
    bank = PrototypeBank(max_prototypes_per_char=2, random_state=5)
    bank.fit(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ["甲", "乙"],
    )
    path = tmp_path / "prototypes.npz"
    metadata = {
        "catalog_sha256": "a" * 64,
        "encoder_checkpoint_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "creation_command": "poor-word train glyph",
    }

    bank.save(path, metadata=metadata)
    loaded, loaded_metadata = PrototypeBank.load(path)
    scores = loaded.score(np.array([[1.0, 0.0]], dtype=np.float32))

    assert scores.nearest_chars == ("甲",)
    assert loaded_metadata == metadata
