"""Metadata-only replay of the current glyph trainer's positive-pair exposure."""

from collections import Counter

from poor_word.training.dataset import GlyphDataset
from poor_word.training.sampling import build_sampling_run, sampling_counts
from poor_word.training.train_glyph import TrainConfig


def _coverage_fields(counts: Counter[str]) -> dict[str, object]:
    normal_draws = counts["normal_draws"]
    steps = counts["replayed_steps"]
    return {
        "replayed_steps": steps,
        "normal_draws": normal_draws,
        "eligible_normal_draws": counts["eligible_normal_draws"],
        "eligible_normal_fraction": (
            counts["eligible_normal_draws"] / normal_draws if normal_draws else 0.0
        ),
        "positive_pairs": counts["positive_pairs"],
        "batches_without_positive_pairs": counts["batches_without_positive_pairs"],
        "batches_without_positive_pairs_fraction": (
            counts["batches_without_positive_pairs"] / steps if steps else 0.0
        ),
    }


def audit_pair_coverage(
    dataset: GlyphDataset,
    config: TrainConfig,
    *,
    expected_steps: int | None = None,
) -> dict[str, object]:
    """Replay training batches without loading images or executing a model.

    A normal draw is a PASS sample. An eligible normal draw has another PASS
    sample with the same base character in its batch. Positive pairs are
    unordered and count repeated exposure across batches and epochs.

    This reproduces the current ``train_glyph`` split/fallback, local seeded
    generator, epoch permutations, partial batches, and step limit. It is not
    recorded training loss or evidence of representation learning or collapse.
    """
    train_indices, _ = dataset.split_indices()
    if not train_indices:
        train_indices = tuple(range(len(dataset)))

    run = build_sampling_run(
        dataset.rows,
        train_indices,
        sampler=config.sampler,
        batch_size=config.batch_size,
        seed=config.seed,
        epochs=config.epochs,
        max_steps=config.max_steps,
    )
    totals: Counter[str] = Counter()
    per_epoch: list[dict[str, object]] = []
    epochs_completed = 0
    for epoch in run.epochs:
        epoch_counts = Counter(sampling_counts(dataset.rows, epoch.batches))
        epoch_counts["replayed_steps"] = len(epoch.batches)
        totals.update(epoch_counts)
        epochs_completed += int(epoch.completed)
        per_epoch.append(
            {"epoch": epoch.epoch, "completed": epoch.completed, **_coverage_fields(epoch_counts)}
        )

    expected_steps_matches = (
        totals["replayed_steps"] == expected_steps if expected_steps is not None else None
    )
    warnings = (
        [
            f"Replayed {totals['replayed_steps']} steps, but the recorded run reports "
            f"{expected_steps}; this replay may not describe that run."
        ]
        if expected_steps_matches is False
        else []
    )
    return {
        "audit_type": "deterministic_sampler_replay",
        "interpretation": (
            "Metadata-only replay of the current train_glyph sampler, not recorded "
            "training loss or proof of learning or collapse."
        ),
        "train_samples": len(train_indices),
        "batch_size": config.batch_size,
        "epochs_started": len(per_epoch),
        "epochs_completed": epochs_completed,
        **_coverage_fields(totals),
        "expected_steps_matches": expected_steps_matches,
        "per_epoch": per_epoch,
        "warnings": warnings,
    }
