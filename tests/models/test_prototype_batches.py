"""Large catalogs must not allocate a full dataset-by-prototype distance matrix."""

import json

import numpy as np
import pytest

from poor_word.models.prototypes import PrototypeBank


def _bank(tmp_path):
    path = tmp_path / "bank.npz"
    np.savez_compressed(
        path,
        centers=np.array([[0, 1], [1, 0], [-1, 0], [0, -1]], dtype=np.float32),
        center_labels=np.array(["乙", "甲", "甲", "丙"]),
        max_prototypes_per_char=np.array(2),
        random_state=np.array(0),
    )
    path.with_suffix(".json").write_text(json.dumps({}))
    return PrototypeBank.load(path)[0]


@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_batched_score_keeps_distinct_class_runner_up_and_unsorted_loaded_labels(
    tmp_path, batch_size
):
    bank = _bank(tmp_path)
    scores = bank.score(
        np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0.6, 0.8]], dtype=np.float32),
        batch_size=batch_size,
    )
    assert scores.nearest_chars == ("甲", "甲", "乙", "丙", "乙")
    np.testing.assert_allclose(scores.nearest_distance, [0, 0, 0, 0, 0.2], atol=1e-7)
    np.testing.assert_allclose(scores.second_distance, [1, 1, 1, 1, 0.4], atol=1e-7)


def test_score_bounds_the_real_matrix_multiplication_working_set(monkeypatch, tmp_path):
    bank = _bank(tmp_path)
    multiply = np.matmul
    observed = []

    def bounded(a, b):
        observed.append(len(a))
        return multiply(a, b)

    monkeypatch.setattr(np, "matmul", bounded)
    scores = bank.score(np.tile([[1.0, 0.0]], (11, 1)).astype(np.float32), batch_size=3)
    assert len(scores.nearest_chars) == 11
    assert observed == [3, 3, 3, 2]


def test_score_rejects_nonpositive_working_set_limit(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        _bank(tmp_path).score(np.array([[1, 0]], dtype=np.float32), batch_size=0)
