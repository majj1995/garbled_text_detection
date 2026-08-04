import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, UnidentifiedImageError

from poor_word.real_data.schema import ImageLabel, RealImageRecord, SplitRole


@dataclass(frozen=True)
class RealDatasetArtifacts:
    manifest: Path
    dataset_metadata: Path
    validation_report: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable dataset artifact already has different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    try:
        part.write_bytes(payload)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _read_records(path: Path) -> tuple[RealImageRecord, ...]:
    records: list[RealImageRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(RealImageRecord.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid real-data record at line {line_number}: {error}") from error
    if not records:
        raise ValueError("real-data input must contain at least one record")
    image_ids = [record.image_id for record in records]
    duplicates = sorted(
        image_id for image_id in set(image_ids) if image_ids.count(image_id) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate image_id values: {', '.join(duplicates)}")
    return tuple(sorted(records, key=lambda record: record.image_id))


def _decode_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not decode real-data image: {path}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"real-data image has invalid dimensions: {path}")
    return width, height


def _validated_rows(input_path: Path) -> list[dict[str, object]]:
    root = input_path.parent.resolve()
    rows: list[dict[str, object]] = []
    for record in _read_records(input_path):
        image_path = (root / record.image_path).resolve()
        try:
            relative_path = image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"image_path escapes real-data input directory: {record.image_path}"
            ) from error
        if not image_path.is_file():
            raise FileNotFoundError(f"real-data image does not exist: {image_path}")
        image_hash = _sha256(image_path)
        if record.expected_sha256 is not None and record.expected_sha256 != image_hash:
            raise ValueError(f"SHA-256 mismatch for image_id={record.image_id}")
        width, height = _decode_image(image_path)
        for annotation in record.characters:
            if annotation.box.x1 > width or annotation.box.y1 > height:
                raise ValueError(
                    f"character box outside image bounds for image_id={record.image_id}"
                )
        row = cast(dict[str, object], record.model_dump(mode="json"))
        row.pop("expected_sha256", None)
        row.update(
            {
                "image_path": relative_path.as_posix(),
                "image_sha256": image_hash,
                "width": width,
                "height": height,
            }
        )
        rows.append(row)
    return rows


def _parquet_bytes(rows: list[dict[str, object]]) -> bytes:
    table = pa.Table.from_pylist(rows)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", version="2.6")
    return cast(bytes, sink.getvalue().to_pybytes())


def _budget_report(rows: list[dict[str, object]]) -> dict[str, object]:
    def count(label: ImageLabel, role: SplitRole) -> int:
        return sum(
            row["image_label"] == label.value and row["split_role"] == role.value
            for row in rows
        )

    counts = {
        "development_abnormal": count(ImageLabel.ABNORMAL, SplitRole.DEV),
        "locked_test_abnormal": count(ImageLabel.ABNORMAL, SplitRole.LOCKED_TEST),
        "development_normal": count(ImageLabel.NORMAL, SplitRole.DEV),
        "image_only_abnormal": count(ImageLabel.ABNORMAL, SplitRole.IMAGE_ONLY),
        "normal_replay": count(ImageLabel.NORMAL, SplitRole.NORMAL_REPLAY),
    }
    targets = {
        "development_abnormal": 60,
        "locked_test_abnormal": 20,
        "development_normal": 20,
    }
    warnings = [
        f"{name}: observed={counts[name]}, approved_seed_target={target}"
        for name, target in targets.items()
        if counts[name] != target
    ]
    return {
        "status": "valid_with_warnings" if warnings else "valid",
        "counts": counts,
        "warnings": warnings,
    }


def import_real_dataset(input_jsonl: Path, output_dir: Path) -> RealDatasetArtifacts:
    """Validate real records and write immutable, provenance-rich dataset artifacts."""
    rows = _validated_rows(input_jsonl)
    manifest_bytes = _parquet_bytes(rows)
    dataset_id = hashlib.sha256(_canonical_json(rows)).hexdigest()
    manifest_path = output_dir / "manifest.parquet"
    metadata_path = output_dir / "dataset.json"
    validation_path = output_dir / "validation.json"
    metadata = {
        "dataset_id": dataset_id,
        "row_count": len(rows),
        "input_jsonl_sha256": _sha256(input_jsonl),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "image_sha256": {str(row["image_id"]): str(row["image_sha256"]) for row in rows},
    }
    validation = _budget_report(rows)
    _write_immutable(manifest_path, manifest_bytes)
    _write_immutable(metadata_path, _canonical_json(metadata) + b"\n")
    _write_immutable(validation_path, _canonical_json(validation) + b"\n")
    return RealDatasetArtifacts(
        manifest=manifest_path,
        dataset_metadata=metadata_path,
        validation_report=validation_path,
    )
