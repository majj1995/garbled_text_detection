import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class GlyphScores:
    nearest_chars: tuple[str, ...]
    nearest_distance: FloatArray
    second_distance: FloatArray
    margin: FloatArray


def _normalize(rows: FloatArray) -> FloatArray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings and prototypes must have non-zero norm")
    return cast(FloatArray, (rows / norms).astype(np.float32, copy=False))


class PrototypeBank:
    def __init__(self, max_prototypes_per_char: int = 8, random_state: int = 0) -> None:
        if max_prototypes_per_char <= 0:
            raise ValueError("max_prototypes_per_char must be positive")
        self.max_prototypes_per_char = max_prototypes_per_char
        self.random_state = random_state
        self._centers: FloatArray | None = None
        self._center_labels: tuple[str, ...] = ()

    def fit(self, embeddings: FloatArray, labels: list[str]) -> None:
        values = cast(FloatArray, np.asarray(embeddings, dtype=np.float32))
        if values.ndim != 2 or len(values) != len(labels) or not len(values):
            raise ValueError("embeddings must be non-empty [N, D] and match labels")
        if any(not label for label in labels):
            raise ValueError("prototype labels must not be empty")
        values = _normalize(values)

        centers: list[FloatArray] = []
        center_labels: list[str] = []
        labels_array = np.asarray(labels)
        for character in sorted(set(labels)):
            character_values = values[labels_array == character]
            if len(character_values) < 4:
                character_centers = character_values.mean(axis=0, keepdims=True)
            else:
                cluster_count = min(
                    self.max_prototypes_per_char,
                    max(1, int(np.sqrt(len(character_values)))),
                )
                model = MiniBatchKMeans(
                    n_clusters=cluster_count,
                    random_state=self.random_state,
                    n_init=10,
                    batch_size=max(256, len(character_values)),
                )
                model.fit(character_values)
                character_centers = model.cluster_centers_
            normalized_centers = _normalize(
                np.asarray(character_centers, dtype=np.float32)
            )
            centers.append(normalized_centers)
            center_labels.extend([character] * len(normalized_centers))

        self._centers = np.concatenate(centers, axis=0).astype(np.float32, copy=False)
        self._center_labels = tuple(center_labels)

    def score(self, embeddings: FloatArray) -> GlyphScores:
        if self._centers is None:
            raise ValueError("prototype bank must be fitted before scoring")
        values = cast(FloatArray, np.asarray(embeddings, dtype=np.float32))
        if values.ndim != 2 or values.shape[1] != self._centers.shape[1]:
            raise ValueError("score embeddings must have shape [N, prototype_dim]")
        values = _normalize(values)
        prototype_distances = 1.0 - values @ self._centers.T
        characters = tuple(sorted(set(self._center_labels)))
        class_distances = np.column_stack(
            [
                prototype_distances[
                    :, np.asarray(self._center_labels, dtype=object) == character
                ].min(axis=1)
                for character in characters
            ]
        ).astype(np.float32, copy=False)
        order = np.argsort(class_distances, axis=1)
        nearest_indices = order[:, 0]
        nearest = class_distances[np.arange(len(values)), nearest_indices]
        if len(characters) > 1:
            second = class_distances[np.arange(len(values)), order[:, 1]]
        else:
            second = np.full(len(values), np.inf, dtype=np.float32)
        return GlyphScores(
            nearest_chars=tuple(characters[int(index)] for index in nearest_indices),
            nearest_distance=nearest.astype(np.float32, copy=False),
            second_distance=second.astype(np.float32, copy=False),
            margin=(second - nearest).astype(np.float32, copy=False),
        )

    def save(self, path: Path, *, metadata: dict[str, str]) -> None:
        if self._centers is None:
            raise ValueError("prototype bank must be fitted before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        part_path = path.with_name(f"{path.name}.part")
        try:
            with part_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    centers=self._centers,
                    center_labels=np.asarray(self._center_labels),
                    max_prototypes_per_char=np.asarray(self.max_prototypes_per_char),
                    random_state=np.asarray(self.random_state),
                )
            part_path.replace(path)
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise

        metadata_path = path.with_suffix(".json")
        metadata_part = metadata_path.with_name(f"{metadata_path.name}.part")
        try:
            metadata_part.write_text(
                f"{json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)}\n",
                encoding="utf-8",
            )
            metadata_part.replace(metadata_path)
        except BaseException:
            metadata_part.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> tuple["PrototypeBank", dict[str, str]]:
        with np.load(path, allow_pickle=False) as payload:
            bank = cls(
                max_prototypes_per_char=int(payload["max_prototypes_per_char"]),
                random_state=int(payload["random_state"]),
            )
            bank._centers = np.asarray(payload["centers"], dtype=np.float32)
            bank._center_labels = tuple(str(item) for item in payload["center_labels"])
        raw_metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if not isinstance(raw_metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_metadata.items()
        ):
            raise ValueError("prototype metadata must be a string-to-string object")
        return bank, cast(dict[str, str], raw_metadata)
