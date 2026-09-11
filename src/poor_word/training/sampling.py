"""Deterministic batch planning shared by glyph training and coverage audit."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import torch

SamplerKind = Literal["random", "paired"]


@dataclass(frozen=True)
class SamplingEpoch:
    epoch: int
    batches: tuple[tuple[int, ...], ...]
    completed: bool


@dataclass(frozen=True)
class SamplingRun:
    epochs: tuple[SamplingEpoch, ...]

    @property
    def batches(self) -> tuple[tuple[int, ...], ...]:
        return tuple(batch for epoch in self.epochs for batch in epoch.batches)


def _permuted(values: list[int], generator: torch.Generator) -> list[int]:
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[position] for position in order]


def _paired_epoch(
    rows: list[dict[str, Any]],
    train_indices: tuple[int, ...],
    batch_size: int,
    generator: torch.Generator,
) -> tuple[tuple[int, ...], ...]:
    if batch_size < 4 or batch_size % 4:
        raise ValueError("paired batch_size must be at least 4 and divisible by 4")
    normals: dict[str, list[int]] = defaultdict(list)
    abnormals: list[int] = []
    for index in train_indices:
        decision = str(rows[index]["decision"])
        if decision == "PASS":
            normals[str(rows[index]["base_char"])].append(index)
        elif decision == "BLOCK":
            abnormals.append(index)
    if not normals:
        raise ValueError("paired sampling requires at least one PASS sample")
    if not abnormals:
        raise ValueError("paired sampling requires at least one BLOCK sample")
    singletons = sorted(character for character, indices in normals.items() if len(indices) == 1)
    if singletons:
        joined = ", ".join(singletons)
        raise ValueError(f"paired sampling cannot use {joined}: character has only one PASS sample")

    pairs: list[tuple[int, int]] = []
    for character in sorted(normals):
        shuffled = _permuted(normals[character], generator)
        for start in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[start], shuffled[start + 1]))
        if len(shuffled) % 2:
            pairs.append((shuffled[-1], shuffled[0]))
    pair_order = torch.randperm(len(pairs), generator=generator).tolist()
    pairs = [pairs[position] for position in pair_order]
    abnormals = _permuted(abnormals, generator)

    pair_slots = batch_size // 4
    abnormal_slots = batch_size // 2
    batch_count = max(
        math.ceil(len(pairs) / pair_slots),
        math.ceil(len(abnormals) / abnormal_slots),
    )
    batches: list[tuple[int, ...]] = []
    for batch_index in range(batch_count):
        selected_pairs = [
            pairs[(batch_index * pair_slots + offset) % len(pairs)] for offset in range(pair_slots)
        ]
        selected_abnormal = [
            abnormals[(batch_index * abnormal_slots + offset) % len(abnormals)]
            for offset in range(abnormal_slots)
        ]
        normal_indices = [index for pair in selected_pairs for index in pair]
        batches.append(tuple(normal_indices + selected_abnormal))
    return tuple(batches)


def build_sampling_run(
    rows: list[dict[str, Any]],
    train_indices: tuple[int, ...],
    *,
    sampler: SamplerKind,
    batch_size: int,
    seed: int,
    epochs: int,
    max_steps: int | None = None,
) -> SamplingRun:
    """Build the exact finite batch sequence consumed by one training run."""
    if not train_indices:
        raise ValueError("training split contains no samples")
    generator = torch.Generator().manual_seed(seed)
    planned_epochs: list[SamplingEpoch] = []
    steps = 0
    for epoch in range(epochs):
        if sampler == "random":
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            full_batches = tuple(
                tuple(
                    train_indices[order_index] for order_index in order[start : start + batch_size]
                )
                for start in range(0, len(order), batch_size)
            )
        elif sampler == "paired":
            full_batches = _paired_epoch(rows, train_indices, batch_size, generator)
        else:
            raise ValueError(f"unknown sampler: {sampler}")
        remaining = len(full_batches) if max_steps is None else max(max_steps - steps, 0)
        batches = full_batches[:remaining]
        completed = len(batches) == len(full_batches)
        planned_epochs.append(SamplingEpoch(epoch=epoch + 1, batches=batches, completed=completed))
        steps += len(batches)
        if max_steps is not None and steps >= max_steps:
            break
    return SamplingRun(epochs=tuple(planned_epochs))


def sampling_counts(
    rows: list[dict[str, Any]], batches: tuple[tuple[int, ...], ...]
) -> dict[str, int | float]:
    totals: Counter[str] = Counter()
    for batch in batches:
        normal_counts = Counter(
            str(rows[index]["base_char"])
            for index in batch
            if str(rows[index]["decision"]) == "PASS"
        )
        positive_pairs = sum(count * (count - 1) // 2 for count in normal_counts.values())
        normal_draws = sum(normal_counts.values())
        totals.update(
            {
                "normal_draws": normal_draws,
                "eligible_normal_draws": sum(
                    count for count in normal_counts.values() if count >= 2
                ),
                "positive_pairs": positive_pairs,
                "batches_without_positive_pairs": int(positive_pairs == 0),
            }
        )
    normal_draws = totals["normal_draws"]
    return {
        "normal_draws": normal_draws,
        "eligible_normal_draws": totals["eligible_normal_draws"],
        "eligible_normal_fraction": (
            totals["eligible_normal_draws"] / normal_draws if normal_draws else 0.0
        ),
        "positive_pairs": totals["positive_pairs"],
        "batches_without_positive_pairs": totals["batches_without_positive_pairs"],
    }
