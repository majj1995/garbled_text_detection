from __future__ import annotations

from typing import Any

import pytest

from poor_word.training.sampling import build_sampling_run, sampling_counts


def _rows(spec: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"sample_id": f"sample-{index}", "base_char": character, "decision": decision}
        for index, (character, decision) in enumerate(spec)
    ]


def test_paired_sampler_gives_every_normal_draw_a_distinct_positive_partner() -> None:
    rows = _rows(
        [
            ("甲", "PASS"),
            ("甲", "PASS"),
            ("乙", "PASS"),
            ("乙", "PASS"),
            ("甲", "BLOCK"),
            ("乙", "BLOCK"),
            ("丙", "BLOCK"),
        ]
    )

    run = build_sampling_run(
        rows, tuple(range(len(rows))), sampler="paired", batch_size=4, seed=17, epochs=1
    )
    counts = sampling_counts(rows, run.batches)

    assert counts == {
        "normal_draws": 4,
        "eligible_normal_draws": 4,
        "eligible_normal_fraction": 1.0,
        "positive_pairs": 2,
        "batches_without_positive_pairs": 0,
    }
    assert {index for batch in run.batches for index in batch if index < 4} == {0, 1, 2, 3}
    assert {index for batch in run.batches for index in batch if index >= 4} == {4, 5, 6}
    for batch in run.batches:
        assert len(batch) == 4
        normal = [index for index in batch if rows[index]["decision"] == "PASS"]
        abnormal = [index for index in batch if rows[index]["decision"] == "BLOCK"]
        assert len(normal) == len(abnormal) == 2
        assert normal[0] != normal[1]
        assert rows[normal[0]]["base_char"] == rows[normal[1]]["base_char"]


def test_paired_sampler_is_deterministic_by_seed_and_epoch() -> None:
    rows = _rows(
        [("甲", "PASS"), ("甲", "PASS"), ("乙", "PASS"), ("乙", "PASS")] + [("甲", "BLOCK")] * 6
    )
    arguments = (rows, tuple(range(len(rows))))

    first = build_sampling_run(*arguments, sampler="paired", batch_size=4, seed=23, epochs=3)
    second = build_sampling_run(*arguments, sampler="paired", batch_size=4, seed=23, epochs=3)
    different = build_sampling_run(*arguments, sampler="paired", batch_size=4, seed=24, epochs=3)

    assert first == second
    assert first != different
    assert first.epochs[0].batches != first.epochs[1].batches


def test_paired_sampler_rejects_a_character_with_only_one_normal_sample() -> None:
    rows = _rows([("甲", "PASS"), ("甲", "PASS"), ("乙", "PASS"), ("甲", "BLOCK")])

    with pytest.raises(ValueError, match=r"乙.*only one PASS"):
        build_sampling_run(
            rows, tuple(range(len(rows))), sampler="paired", batch_size=4, seed=1, epochs=1
        )


@pytest.mark.parametrize(
    ("spec", "batch_size", "message"),
    [
        ([("甲", "PASS"), ("甲", "PASS")], 4, "BLOCK"),
        ([("甲", "BLOCK"), ("甲", "BLOCK")], 4, "PASS"),
        ([("甲", "PASS"), ("甲", "PASS"), ("甲", "BLOCK")], 2, "divisible by 4"),
        ([("甲", "PASS"), ("甲", "PASS"), ("甲", "BLOCK")], 6, "divisible by 4"),
    ],
)
def test_paired_sampler_rejects_invalid_training_membership(
    spec: list[tuple[str, str]], batch_size: int, message: str
) -> None:
    rows = _rows(spec)

    with pytest.raises(ValueError, match=message):
        build_sampling_run(
            rows, tuple(range(len(rows))), sampler="paired", batch_size=batch_size, seed=1, epochs=1
        )


def test_random_sampler_preserves_the_previous_seeded_permutation_and_partial_batches() -> None:
    rows = _rows([("甲", "PASS")] * 5)

    run = build_sampling_run(
        rows, tuple(range(5)), sampler="random", batch_size=2, seed=7, epochs=2, max_steps=5
    )

    assert run.batches == ((0, 1), (3, 2), (4,), (3, 4), (1, 2))
    assert [(epoch.completed, len(epoch.batches)) for epoch in run.epochs] == [
        (True, 3),
        (False, 2),
    ]
