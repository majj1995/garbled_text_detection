"""Bounded experimental datasets from frozen native stroke corruption rules."""

import hashlib
import inspect
import io
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image, PngImagePlugin
from pydantic import BaseModel, ConfigDict, Field, model_validator

from poor_word.data.manifest import load_source_lock
from poor_word.domain import AnomalyKind, BoundingBox
from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable

_ARPHIC_SHA256 = "3a5e90c0957524a89e48203febcd4492ca4393678abaa7e5b4d70f3ff32b386d"
_SCHEMA_VERSION = "glyph-dataset-v2"
_SOURCE_ASSET_ID = "makemeahanzi_graphics"
_ROLES: tuple[Literal["train", "calibration", "test"], ...] = (
    "train",
    "calibration",
    "test",
)
_OPERATOR_KINDS = {
    "erase_segment": AnomalyKind.MISSING_STROKE.value,
    "add_stroke": AnomalyKind.EXTRA_STROKE.value,
    "break_stroke": AnomalyKind.BROKEN_STROKE.value,
    "bridge": AnomalyKind.BRIDGE.value,
    "component_shift": AnomalyKind.COMPONENT_SHIFT.value,
}


class V2GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    graphics_path: Path
    source_lock_path: Path
    license_path: Path
    characters: tuple[str, ...]
    seed: int = Field(default=20260911, ge=0)
    train_normal_per_char: int = Field(default=8, ge=1, le=100)
    eval_normal_per_char: int = Field(default=4, ge=1, le=100)
    train_abnormal_per_operator: int = Field(default=2, ge=0, le=100)
    eval_abnormal_per_operator: int = Field(default=1, ge=0, le=100)
    max_attempts: int = Field(default=8, ge=1, le=100)
    allow_experimental: bool = False

    @model_validator(mode="after")
    def validate_characters(self) -> "V2GenerationConfig":
        if not self.characters or any(
            len(character) != 1 or character.isspace() for character in self.characters
        ):
            raise ValueError("characters must contain single non-whitespace Unicode characters")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("characters must be distinct")
        return self


@dataclass(frozen=True)
class V2GenerationArtifacts:
    train_manifest: Path
    calibration_manifest: Path
    test_manifest: Path
    run_path: Path
    dataset_id: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_seed(base_seed: int, *identity: object) -> int:
    digest = hashlib.sha256(_canonical_json([base_seed, *identity])).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _identifier(prefix: str, *identity: object) -> str:
    return prefix + hashlib.sha256(_canonical_json(identity)).hexdigest()


def _bbox(mask: NDArray[np.bool_]) -> dict[str, int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("generated glyph has no visible foreground")
    return BoundingBox(
        x0=int(columns.min()),
        y0=int(rows.min()),
        x1=int(columns.max()) + 1,
        y1=int(rows.max()) + 1,
    ).model_dump()


def _appearance(alpha: NDArray[np.uint8], seed: int) -> tuple[NDArray[np.uint8], dict[str, float]]:
    random = np.random.default_rng(seed)
    scale = float(random.uniform(0.85, 1.0))
    angle = float(random.uniform(-3.0, 3.0))
    translate_x = float(random.uniform(-3.0, 3.0))
    translate_y = float(random.uniform(-3.0, 3.0))
    center = ((alpha.shape[1] - 1) / 2.0, (alpha.shape[0] - 1) / 2.0)
    matrix = np.asarray(cv2.getRotationMatrix2D(center, angle, scale), dtype=np.float64)
    matrix[:, 2] += (translate_x, translate_y)
    transformed = cv2.warpAffine(
        alpha,
        matrix,
        (alpha.shape[1], alpha.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,),
    )
    return np.asarray(transformed, dtype=np.uint8), {
        "appearance_scale": scale,
        "appearance_rotation_degrees": angle,
        "appearance_translate_x": translate_x,
        "appearance_translate_y": translate_y,
    }


def _png_bytes(array: NDArray[np.uint8], notice: str, source_sha256: str) -> bytes:
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Copyright", "Copyright (C) 1999 Arphic Technology Co., Ltd.")
    metadata.add_text("License", "Arphic-1999; see ARPHICPL.txt")
    metadata.add_text("Modification", notice)
    metadata.add_text(
        "Source",
        f"Make Me a Hanzi graphics.txt sha256={source_sha256}; see source lock in dataset root",
    )
    buffer = io.BytesIO()
    Image.fromarray(array).save(
        buffer,
        format="PNG",
        pnginfo=metadata,
        compress_level=9,
        optimize=False,
    )
    return buffer.getvalue()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_assets(
    root: Path,
    role: str,
    sample_id: str,
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    notice: str,
    source_sha256: str,
) -> tuple[str, str, str, str]:
    prefix = sample_id.removeprefix("v2-")[:2]
    image_relative = Path("images") / role / prefix / f"{sample_id}.png"
    mask_relative = Path("masks") / role / prefix / f"{sample_id}.png"
    image_bytes = _png_bytes(image, notice, source_sha256)
    mask_bytes = _png_bytes(mask, notice, source_sha256)
    _write_bytes(root / image_relative, image_bytes)
    _write_bytes(root / mask_relative, mask_bytes)
    return (
        image_relative.as_posix(),
        mask_relative.as_posix(),
        _sha256_bytes(image_bytes),
        _sha256_bytes(mask_bytes),
    )


def _npy_bytes(array: NDArray[Any]) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)  # type: ignore[no-untyped-call]
    return buffer.getvalue()


def _write_layer_archive(
    path: Path,
    layers: tuple[NDArray[np.uint8], ...],
    character: str,
    notice: str,
    source_sha256: str,
) -> str:
    arrays: dict[str, NDArray[Any]] = {
        **{f"stroke_{index:03}": layer for index, layer in enumerate(layers)},
        "source_char": np.array(character),
        "source_sha256": np.array(source_sha256),
        "license_id": np.array("Arphic-1999"),
        "modification": np.array(notice),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, array in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array), compresslevel=9)
    return _sha256_file(path)


def _manifest_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.string()),
            ("image_path", pa.string()),
            ("mask_path", pa.string()),
            ("base_char", pa.string()),
            ("rendered_char", pa.string()),
            ("decision", pa.string()),
            ("anomaly_kind", pa.string()),
            ("operator", pa.string()),
            ("changed_pixels", pa.int64()),
            ("seed", pa.int64()),
            (
                "bbox",
                pa.struct(
                    [
                        ("x0", pa.int64()),
                        ("y0", pa.int64()),
                        ("x1", pa.int64()),
                        ("y1", pa.int64()),
                    ]
                ),
            ),
            ("source_asset_ids", pa.list_(pa.string())),
            ("schema_version", pa.string()),
            ("dataset_id", pa.string()),
            ("split_role", pa.string()),
            ("source_group_id", pa.string()),
            ("pixel_sha256", pa.string()),
            ("image_sha256", pa.string()),
            ("mask_sha256", pa.string()),
            ("training_eligible", pa.bool_()),
            ("production_allowed", pa.bool_()),
            ("label_provenance", pa.string()),
            ("appearance_seed", pa.int64()),
            ("corruption_seed", pa.int64()),
            ("selected_stroke_ids", pa.list_(pa.int32())),
            ("metrics", pa.string()),
            ("bridge_mode", pa.string()),
            ("source_layers_path", pa.string()),
            ("source_layers_sha256", pa.string()),
        ]
    )


def _row(
    *,
    root: Path,
    dataset_id: str,
    role: str,
    character: str,
    source_group_id: str,
    source_sha256: str,
    source_layers_path: str,
    source_layers_sha256: str,
    image_alpha: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    operator: str,
    anomaly_kind: str,
    changed_pixels: int,
    seed: int,
    appearance_seed: int,
    corruption_seed: int | None,
    selected_stroke_ids: tuple[int, ...],
    metrics: dict[str, float],
    bridge_mode: str | None,
    replica: int,
    attempt: int,
    date: str,
) -> dict[str, object]:
    image = np.repeat(image_alpha[:, :, None], 3, axis=2)
    pixel_sha256 = _sha256_bytes(image.tobytes())
    decision = "PASS" if operator == "identity" else "BLOCK"
    sample_id = _identifier(
        "v2-",
        dataset_id,
        role,
        character,
        operator,
        replica,
        attempt,
        seed,
        appearance_seed,
        pixel_sha256,
    )
    notice = (
        f"{date}: experimental glyph-dataset-v2 sample {sample_id}; normalized and "
        f"rasterized Make Me a Hanzi outlines, applied {operator}, appearance seed "
        f"{appearance_seed}; production_allowed=false."
    )
    image_path, mask_path, image_sha256, mask_sha256 = _write_assets(
        root, role, sample_id, image, mask, notice, source_sha256
    )
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "mask_path": mask_path,
        "base_char": character,
        "rendered_char": character,
        "decision": decision,
        "anomaly_kind": anomaly_kind,
        "operator": operator,
        "changed_pixels": changed_pixels,
        "seed": seed,
        "bbox": _bbox(image_alpha > 8),
        "source_asset_ids": [_SOURCE_ASSET_ID],
        "schema_version": _SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "split_role": role,
        "source_group_id": source_group_id,
        "pixel_sha256": pixel_sha256,
        "image_sha256": image_sha256,
        "mask_sha256": mask_sha256,
        "training_eligible": role == "train",
        "production_allowed": False,
        "label_provenance": "synthetic_rule_v2",
        "appearance_seed": appearance_seed,
        "corruption_seed": corruption_seed,
        "selected_stroke_ids": list(selected_stroke_ids),
        "metrics": json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "bridge_mode": bridge_mode,
        "source_layers_path": source_layers_path,
        "source_layers_sha256": source_layers_sha256,
    }


def _role_quota(config: V2GenerationConfig, role: str, *, abnormal: bool) -> int:
    if abnormal:
        return (
            config.train_abnormal_per_operator
            if role == "train"
            else config.eval_abnormal_per_operator
        )
    return config.train_normal_per_char if role == "train" else config.eval_normal_per_char


def _code_hashes(modules: tuple[Any, ...]) -> dict[str, str]:
    return {
        Path(inspect.getfile(module)).name: _sha256_file(Path(inspect.getfile(module)))
        for module in modules
    }


def generate_v2_dataset(
    config: V2GenerationConfig,
    progress: Callable[[str], None] | None = None,
) -> V2GenerationArtifacts:
    """Build a fresh, atomic V2 root without changing the frozen five rule modules."""
    if not config.allow_experimental:
        raise ValueError("V2 generation requires allow_experimental=True")
    output = config.output_dir.resolve()
    if output.exists() or config.output_dir.is_symlink():
        raise FileExistsError(f"V2 output already exists; choose a fresh directory: {output}")
    emit = progress if progress is not None else lambda _message: None
    emit("Checking locked stroke source and unmodified Arphic license...")
    lock_bytes = config.source_lock_path.read_bytes()
    lock = load_source_lock(config.source_lock_path)
    source_bytes = config.graphics_path.read_bytes()
    if (
        lock.source_id != _SOURCE_ASSET_ID
        or lock.license_id != "Arphic-1999"
        or lock.production_allowed is not False
        or _sha256_bytes(source_bytes) != lock.sha256
        or len(source_bytes) != lock.size_bytes
    ):
        raise ValueError("stroke source does not match its non-production source lock")
    license_bytes = config.license_path.read_bytes()
    if _sha256_bytes(license_bytes) != _ARPHIC_SHA256:
        raise ValueError("stroke source requires the unmodified Arphic license")

    from poor_word.glyphs import corrupt, stroke_break, stroke_bridge, stroke_corrupt, stroke_source

    records = stroke_source.load_stroke_records(
        config.graphics_path,
        config.characters,
        progress=lambda count: emit(f"validated_stroke_records={count}"),
    )
    code_hashes = _code_hashes(
        (corrupt, stroke_source, stroke_corrupt, stroke_break, stroke_bridge)
    )
    code_hashes[Path(__file__).name] = _sha256_file(Path(__file__))
    identity_config = {
        "characters": list(config.characters),
        "seed": config.seed,
        "train_normal_per_char": config.train_normal_per_char,
        "eval_normal_per_char": config.eval_normal_per_char,
        "train_abnormal_per_operator": config.train_abnormal_per_operator,
        "eval_abnormal_per_operator": config.eval_abnormal_per_operator,
        "max_attempts": config.max_attempts,
        "operators": sorted(OPERATORS),
    }
    dataset_id = _identifier(
        "glyph-v2-",
        identity_config,
        lock.model_dump(mode="json"),
        _sha256_bytes(lock_bytes),
        _sha256_bytes(license_bytes),
        code_hashes,
    )
    date = datetime.now(UTC).date().isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    schema = _manifest_schema()
    manifest_parts = {role: staging / f"{role}.parquet.part" for role in _ROLES}
    writers: dict[str, pq.ParquetWriter] = {}
    seen_pixels: set[str] = set()
    seen_sample_ids: set[str] = set()
    counts = Counter[str]()
    skip_reasons = Counter[str]()
    by_type: dict[str, dict[str, dict[str, Any]]] = {
        role: {
            operator: {
                "expected": len(config.characters) * _role_quota(config, role, abnormal=True),
                "actual": 0,
                "skipped": 0,
                "attempts": 0,
                "skip_reasons": {},
            }
            for operator in sorted(OPERATORS)
        }
        for role in _ROLES
    }
    character_layers: dict[str, dict[str, object]] = {}
    try:
        _write_bytes(staging / config.source_lock_path.name, lock_bytes)
        _write_bytes(staging / "ARPHICPL.txt", license_bytes)
        writers = {
            role: pq.ParquetWriter(manifest_parts[role], schema, compression="zstd", version="2.6")
            for role in _ROLES
        }
        for character_index, character in enumerate(config.characters, 1):
            # This is deliberately the only render call for this character.
            layers = stroke_source.render_stroke_layers(records[character])
            original = np.asarray(np.maximum.reduce(layers), dtype=np.uint8)
            layer_relative = Path("layers") / f"u{ord(character):04x}.npz"
            layer_notice = (
                f"{date}: normalized and rasterized Make Me a Hanzi outlines once for "
                f"experimental glyph-dataset-v2 dataset {dataset_id}; no source-rule edit; "
                "production_allowed=false."
            )
            layer_hash = _write_layer_archive(
                staging / layer_relative, layers, character, layer_notice, lock.sha256
            )
            character_layers[character] = {
                "path": layer_relative.as_posix(),
                "sha256": layer_hash,
                "stroke_count": len(layers),
                "source_sha256": lock.sha256,
                "license_id": "Arphic-1999",
                "modification": layer_notice,
            }
            rows_by_role: dict[str, list[dict[str, object]]] = {role: [] for role in _ROLES}
            for role in _ROLES:
                group_id = _identifier("group-", dataset_id, role, character)
                normal_quota = _role_quota(config, role, abnormal=False)
                counts["expected_normal"] += normal_quota
                for replica in range(normal_quota):
                    normal_row: dict[str, object] | None = None
                    for attempt in range(config.max_attempts):
                        counts["normal_attempts"] += 1
                        appearance_seed = _stable_seed(
                            config.seed, character, role, "identity", replica, attempt, "appearance"
                        )
                        transformed, appearance_metrics = _appearance(original, appearance_seed)
                        if not np.any(transformed > 8):
                            continue
                        pixel_hash = _sha256_bytes(
                            np.repeat(transformed[:, :, None], 3, axis=2).tobytes()
                        )
                        if pixel_hash in seen_pixels:
                            continue
                        mask = ((transformed > 8).astype(np.uint8) * 255).astype(np.uint8)
                        normal_row = _row(
                            root=staging,
                            dataset_id=dataset_id,
                            role=role,
                            character=character,
                            source_group_id=group_id,
                            source_sha256=lock.sha256,
                            source_layers_path=layer_relative.as_posix(),
                            source_layers_sha256=layer_hash,
                            image_alpha=transformed,
                            mask=mask,
                            operator="identity",
                            anomaly_kind=AnomalyKind.NONE.value,
                            changed_pixels=0,
                            seed=appearance_seed,
                            appearance_seed=appearance_seed,
                            corruption_seed=None,
                            selected_stroke_ids=(),
                            metrics=appearance_metrics,
                            bridge_mode=None,
                            replica=replica,
                            attempt=attempt,
                            date=date,
                        )
                        break
                    if normal_row is None:
                        raise RuntimeError(
                            f"normal quota could not be filled for {character!r}/{role}/"
                            f"replica {replica} after {config.max_attempts} attempts"
                        )
                    pixel = str(normal_row["pixel_sha256"])
                    sample = str(normal_row["sample_id"])
                    if pixel in seen_pixels or sample in seen_sample_ids:
                        raise RuntimeError("internal V2 identity collision")
                    seen_pixels.add(pixel)
                    seen_sample_ids.add(sample)
                    rows_by_role[role].append(normal_row)
                    counts["actual_normal"] += 1

                abnormal_quota = _role_quota(config, role, abnormal=True)
                counts["expected_block"] += len(OPERATORS) * abnormal_quota
                for operator in sorted(OPERATORS):
                    type_stats = by_type[role][operator]
                    for replica in range(abnormal_quota):
                        abnormal_row: dict[str, object] | None = None
                        for attempt in range(config.max_attempts):
                            counts["abnormal_attempts"] += 1
                            type_stats["attempts"] = int(type_stats["attempts"]) + 1
                            corruption_seed = _stable_seed(
                                config.seed,
                                character,
                                role,
                                operator,
                                replica,
                                attempt,
                                "corruption",
                            )
                            appearance_seed = _stable_seed(
                                config.seed,
                                character,
                                role,
                                operator,
                                replica,
                                attempt,
                                "appearance",
                            )
                            try:
                                result = stroke_corrupt.corrupt_stroke_layers(
                                    layers, operator, corruption_seed
                                )
                                original_transformed, appearance_metrics = _appearance(
                                    original, appearance_seed
                                )
                                candidate_transformed, _ = _appearance(
                                    np.asarray(
                                        np.maximum.reduce(result.edited_layers), dtype=np.uint8
                                    ),
                                    appearance_seed,
                                )
                                visibility = stroke_corrupt._visible(
                                    original_transformed, candidate_transformed
                                )
                                if visibility is None:
                                    raise CorruptionNotApplicable(
                                        "appearance_transform_removed_visible_edit"
                                    )
                                changed = original_transformed != candidate_transformed
                                changed_pixels = int(np.count_nonzero(changed))
                                if changed_pixels == 0:
                                    raise CorruptionNotApplicable(
                                        "appearance_transform_removed_all_changed_pixels"
                                    )
                                pixel_hash = _sha256_bytes(
                                    np.repeat(
                                        candidate_transformed[:, :, None], 3, axis=2
                                    ).tobytes()
                                )
                                if pixel_hash in seen_pixels:
                                    raise CorruptionNotApplicable("duplicate_decoded_rgb_pixels")
                            except CorruptionNotApplicable as error:
                                reason = str(error) or "corruption_not_applicable"
                                skip_reasons[reason] += 1
                                reasons = type_stats["skip_reasons"]
                                assert isinstance(reasons, dict)
                                reasons[reason] = int(reasons.get(reason, 0)) + 1
                                continue
                            abnormal_row = _row(
                                root=staging,
                                dataset_id=dataset_id,
                                role=role,
                                character=character,
                                source_group_id=group_id,
                                source_sha256=lock.sha256,
                                source_layers_path=layer_relative.as_posix(),
                                source_layers_sha256=layer_hash,
                                image_alpha=candidate_transformed,
                                mask=(changed.astype(np.uint8) * 255).astype(np.uint8),
                                operator=operator,
                                anomaly_kind=_OPERATOR_KINDS[operator],
                                changed_pixels=changed_pixels,
                                seed=corruption_seed,
                                appearance_seed=appearance_seed,
                                corruption_seed=corruption_seed,
                                selected_stroke_ids=result.selected_stroke_indices,
                                metrics={**result.metrics, **visibility, **appearance_metrics},
                                bridge_mode=result.bridge_mode,
                                replica=replica,
                                attempt=attempt,
                                date=date,
                            )
                            break
                        if abnormal_row is None:
                            type_stats["skipped"] = int(type_stats["skipped"]) + 1
                            counts["skipped_block"] += 1
                            continue
                        pixel = str(abnormal_row["pixel_sha256"])
                        sample = str(abnormal_row["sample_id"])
                        if pixel in seen_pixels or sample in seen_sample_ids:
                            raise RuntimeError("internal V2 identity collision")
                        seen_pixels.add(pixel)
                        seen_sample_ids.add(sample)
                        rows_by_role[role].append(abnormal_row)
                        type_stats["actual"] = int(type_stats["actual"]) + 1
                        counts["actual_block"] += 1
            for role in _ROLES:
                writers[role].write_table(pa.Table.from_pylist(rows_by_role[role], schema=schema))
            emit(
                f"characters={character_index}/{len(config.characters)} "
                f"normal={counts['actual_normal']}/{counts['expected_normal']} "
                f"block={counts['actual_block']}/{counts['expected_block']}"
            )

        for writer in writers.values():
            writer.close()
        writers.clear()
        manifests: dict[str, dict[str, object]] = {}
        for role in _ROLES:
            final_manifest = staging / f"{role}.parquet"
            manifest_parts[role].replace(final_manifest)
            row_count = pq.read_metadata(final_manifest).num_rows
            manifests[final_manifest.name] = {
                "sha256": _sha256_file(final_manifest),
                "row_count": row_count,
            }
        complete = counts["skipped_block"] == 0
        run = {
            "schema_version": _SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "experimental_only": True,
            "production_allowed": False,
            "complete": complete,
            "created_date": date,
            "manifests": manifests,
            "config": config.model_dump(mode="json"),
            "source_lock_sha256": {config.source_lock_path.name: _sha256_bytes(lock_bytes)},
            "source": lock.model_dump(mode="json"),
            "license": {
                "path": "ARPHICPL.txt",
                "license_id": "Arphic-1999",
                "sha256": _sha256_bytes(license_bytes),
                "copied_exact_bytes": True,
            },
            "code_sha256": code_hashes,
            "counts": {
                "expected_normal": counts["expected_normal"],
                "actual_normal": counts["actual_normal"],
                "expected_block": counts["expected_block"],
                "actual_block": counts["actual_block"],
                "skipped_block": counts["skipped_block"],
            },
            "attempts": {
                "normal": counts["normal_attempts"],
                "abnormal": counts["abnormal_attempts"],
                "max_per_slot": config.max_attempts,
            },
            "skip_reasons": dict(sorted(skip_reasons.items())),
            "by_type": by_type,
            "character_layers": character_layers,
            "grouping_limitation": {
                "same_characters_across_roles": True,
                "description": (
                    "Closed-catalog synthetic roles reuse the same source character outlines; "
                    "role-specific source_group_id values prevent group crossing, but these are "
                    "not source-independent or out-of-distribution splits."
                ),
            },
            "modification_scope": (
                "New experimental raster derivatives only: frozen stroke source and five rule "
                "modules were not edited; shared affine appearance transforms were applied to "
                "each original/candidate pair and rechecked in input space."
            ),
            "label_scope": (
                "Synthetic rule labels are experimental, not human-reviewed, and do not imply "
                "production clearance, linguistic invalidity, recall, or false-positive guarantees."
            ),
        }
        (staging / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists() or config.output_dir.is_symlink():
            raise FileExistsError(f"V2 output appeared during generation: {output}")
        staging.rename(output)
    except BaseException:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    emit(
        f"V2 dataset {'complete' if complete else 'partial'}: "
        f"normal={counts['actual_normal']}, block={counts['actual_block']}; experimental only."
    )
    return V2GenerationArtifacts(
        train_manifest=output / "train.parquet",
        calibration_manifest=output / "calibration.parquet",
        test_manifest=output / "test.parquet",
        run_path=output / "run.json",
        dataset_id=dataset_id,
    )
