import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image

from poor_word.real_data.schema import ImageLabel, SplitRole


@dataclass(frozen=True)
class FoldArtifacts:
    folds: Path
    audit: Path


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def perceptual_hash(path: Path) -> int:
    """Return a deterministic 64-bit DCT perceptual hash."""
    grayscale = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    resized = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cast(NDArray[np.float32], cv2.dct(resized))[:8, :8]
    flattened = coefficients.reshape(-1)
    median = float(np.median(flattened[1:]))
    bits = flattened > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(manifest: Path) -> list[dict[str, object]]:
    rows = cast(list[dict[str, object]], pq.read_table(manifest).to_pylist())
    if not rows:
        raise ValueError("real-data manifest contains no rows")
    required = {
        "image_id",
        "image_path",
        "image_sha256",
        "image_label",
        "split_role",
        "product_id",
        "campaign_id",
        "template_id",
        "source_group_id",
    }
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"real-data manifest row is missing: {', '.join(missing)}")
    return sorted(rows, key=lambda row: str(row["image_id"]))


def _validated_hashes(
    rows: list[dict[str, object]], image_root: Path
) -> list[int]:
    root = image_root.resolve()
    hashes: list[int] = []
    for row in rows:
        path = (root / str(row["image_path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"image path escapes image root: {row['image_id']}") from error
        if not path.is_file():
            raise FileNotFoundError(f"real-data image does not exist: {path}")
        if _sha256(path) != str(row["image_sha256"]):
            raise ValueError(f"image changed after import: {row['image_id']}")
        hashes.append(perceptual_hash(path))
    return hashes


def _union_equal_values(
    rows: list[dict[str, object]], field: str, groups: _UnionFind
) -> None:
    first_by_value: dict[str, int] = {}
    for index, row in enumerate(rows):
        value = str(row[field])
        if value in first_by_value:
            groups.union(first_by_value[value], index)
        else:
            first_by_value[value] = index


def _union_near_hashes(hashes: list[int], groups: _UnionFind, max_distance: int) -> None:
    # Eleven bands guarantee one matching band for a 64-bit pair at Hamming <= 10.
    if max_distance > 10:
        raise ValueError("pHash distance above 10 is unsupported by the exact band index")
    widths = (6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5)
    buckets: dict[tuple[int, int], list[int]] = {}
    candidates: set[tuple[int, int]] = set()
    shift = 64
    for band_index, width in enumerate(widths):
        shift -= width
        mask = (1 << width) - 1
        for index, value in enumerate(hashes):
            key = (band_index, (value >> shift) & mask)
            for other in buckets.setdefault(key, []):
                candidates.add((other, index))
            buckets[key].append(index)
    for first, second in sorted(candidates):
        if (hashes[first] ^ hashes[second]).bit_count() <= max_distance:
            groups.union(first, second)


def _components(rows: list[dict[str, object]], hashes: list[int]) -> list[list[int]]:
    groups = _UnionFind(len(rows))
    for field in (
        "image_sha256",
        "product_id",
        "campaign_id",
        "template_id",
        "source_group_id",
    ):
        _union_equal_values(rows, field, groups)
    _union_near_hashes(hashes, groups, max_distance=10)
    by_root: dict[int, list[int]] = {}
    for index in range(len(rows)):
        by_root.setdefault(groups.find(index), []).append(index)
    return sorted(by_root.values(), key=lambda values: str(rows[values[0]]["image_id"]))


def _component_id(rows: list[dict[str, object]], indices: list[int]) -> str:
    identity = "\0".join(sorted(str(rows[index]["image_id"]) for index in indices))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _assign(
    rows: list[dict[str, object]], components: list[list[int]], folds: int, seed: int
) -> tuple[dict[int, int], dict[str, str]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    assignments: dict[int, int] = {}
    component_ids: dict[str, str] = {}
    eligible: list[list[int]] = []
    for indices in components:
        roles = {str(rows[index]["split_role"]) for index in indices}
        has_locked = SplitRole.LOCKED_TEST.value in roles
        if has_locked and roles != {SplitRole.LOCKED_TEST.value}:
            image_ids = ", ".join(str(rows[index]["image_id"]) for index in indices)
            raise ValueError(f"locked-test leakage component mixes roles: {image_ids}")
        component_id = _component_id(rows, indices)
        for index in indices:
            component_ids[str(rows[index]["image_id"])] = component_id
        if has_locked:
            assignments.update({index: -1 for index in indices})
        else:
            eligible.append(indices)

    label_counts = [
        {ImageLabel.NORMAL.value: 0, ImageLabel.ABNORMAL.value: 0} for _ in range(folds)
    ]
    total_counts = [0] * folds
    ordered = sorted(
        eligible,
        key=lambda indices: (
            -len(indices),
            hashlib.sha256(
                f"{seed}:".encode()
                + ",".join(str(rows[index]["image_id"]) for index in indices).encode()
            ).hexdigest(),
        ),
    )
    global_counts = {
        label: sum(row["image_label"] == label for row in rows)
        for label in (ImageLabel.NORMAL.value, ImageLabel.ABNORMAL.value)
    }
    for indices in ordered:
        component_counts = {
            label: sum(rows[index]["image_label"] == label for index in indices)
            for label in global_counts
        }
        selected = min(
            range(folds),
            key=lambda fold: (
                sum(
                    (label_counts[fold][label] + component_counts[label])
                    / max(1, global_counts[label])
                    for label in global_counts
                    if component_counts[label]
                ),
                total_counts[fold],
                fold,
            ),
        )
        for index in indices:
            assignments[index] = selected
        total_counts[selected] += len(indices)
        for label, count in component_counts.items():
            label_counts[selected][label] += count
    return assignments, component_ids


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable split artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    try:
        part.write_bytes(payload)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def assign_group_folds(
    manifest: Path,
    image_root: Path,
    output_dir: Path,
    *,
    folds: int = 5,
    seed: int = 20260804,
) -> FoldArtifacts:
    rows = _load_rows(manifest)
    hashes = _validated_hashes(rows, image_root)
    components = _components(rows, hashes)
    assignments, component_ids = _assign(rows, components, folds, seed)
    output_rows = [
        {
            "image_id": str(row["image_id"]),
            "fold": assignments[index],
            "component_id": component_ids[str(row["image_id"])],
            "phash": f"{hashes[index]:016x}",
            "image_label": str(row["image_label"]),
            "split_role": str(row["split_role"]),
        }
        for index, row in enumerate(rows)
    ]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(output_rows), sink, compression="zstd", version="2.6")
    parquet_bytes = cast(bytes, sink.getvalue().to_pybytes())
    fold_counts = {
        str(fold): sum(row["fold"] == fold for row in output_rows)
        for fold in range(-1, folds)
        if any(row["fold"] == fold for row in output_rows)
    }
    audit = {
        "manifest_sha256": _sha256(manifest),
        "folds": folds,
        "seed": seed,
        "component_count": len(components),
        "fold_counts": fold_counts,
        "leakage_violations": [],
    }
    folds_path = output_dir / "folds.parquet"
    audit_path = output_dir / "split-audit.json"
    _write_immutable(folds_path, parquet_bytes)
    _write_immutable(
        audit_path,
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n",
    )
    return FoldArtifacts(folds=folds_path, audit=audit_path)
