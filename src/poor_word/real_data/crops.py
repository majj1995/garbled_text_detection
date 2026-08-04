"""Immutable character crops derived from reviewed or audited OCR coordinates."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image

from poor_word.domain import BoundingBox, Decision
from poor_word.ocr.types import OcrAudit, OcrResult


@dataclass(frozen=True)
class CropArtifacts:
    manifest: Path
    audit: Path


@dataclass(frozen=True)
class _AuditedOcrInputs:
    results: dict[str, OcrResult]
    audit: OcrAudit | None
    audit_sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable crop artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    try:
        part.write_bytes(payload)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _load_manifest(manifest: Path) -> list[dict[str, object]]:
    rows = cast(list[dict[str, object]], pq.read_table(manifest).to_pylist())
    required = {"image_id", "image_path", "image_sha256", "characters"}
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"real-data manifest row is missing: {', '.join(missing)}")
    return sorted(rows, key=lambda row: str(row["image_id"]))


def _load_audited_ocr(ocr_results: Path | None, ocr_audit: Path | None) -> _AuditedOcrInputs:
    if (ocr_results is None) != (ocr_audit is None):
        raise ValueError("ocr results and ocr audit must be supplied together")
    if ocr_results is None or ocr_audit is None:
        return _AuditedOcrInputs(results={}, audit=None, audit_sha256=None)
    try:
        audit = OcrAudit.model_validate_json(ocr_audit.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"invalid OCR audit: {error}") from error
    if not audit.character_boxes_available:
        raise ValueError("OCR audit reports character boxes unavailable")

    results: dict[str, OcrResult] = {}
    for number, line in enumerate(ocr_results.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            image_id = str(raw["image_id"])
            result = OcrResult.model_validate(raw["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid OCR result at line {number}: {error}") from error
        if not image_id or image_id in results:
            raise ValueError(f"duplicate or empty OCR image_id at line {number}")
        results[image_id] = result
    return _AuditedOcrInputs(
        results=results,
        audit=audit,
        audit_sha256=_sha256(ocr_audit),
    )


def _crop_box(box: BoundingBox, width: int, height: int, padding: int) -> BoundingBox:
    if padding < 0:
        raise ValueError("padding must be non-negative")
    x0 = max(0, box.x0 - padding)
    y0 = max(0, box.y0 - padding)
    x1 = min(width, box.x1 + padding)
    y1 = min(height, box.y1 + padding)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("character box does not overlap image")
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _crop_id(
    image_id: str, source_image_sha256: str, source_kind: str, source_id: str, box: BoundingBox
) -> str:
    identity = {
        "box": box.model_dump(mode="json"),
        "image_id": image_id,
        "source_id": source_id,
        "source_image_sha256": source_image_sha256,
        "source_kind": source_kind,
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _reviewed_candidates(
    row: dict[str, object],
) -> list[tuple[str, BoundingBox, str, str, str | None, str | None]]:
    candidates: list[tuple[str, BoundingBox, str, str, str | None, str | None]] = []
    characters = row.get("characters") or []
    if not isinstance(characters, list):
        raise ValueError(f"characters must be a list for image_id={row['image_id']}")
    for character in characters:
        if not isinstance(character, dict):
            raise ValueError(f"invalid reviewed character for image_id={row['image_id']}")
        try:
            annotation_id = str(character["annotation_id"])
            box = BoundingBox.model_validate(character["box"])
            decision = Decision(str(character["decision"]))
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"invalid reviewed character for image_id={row['image_id']}: {error}"
            ) from error
        text = character.get("text")
        candidates.append(
            (annotation_id, box, "reviewed", decision.value, str(text) if text else None, None)
        )
    return candidates


def _ocr_candidates(
    image_id: str, result: OcrResult
) -> list[tuple[str, BoundingBox, str, str, str | None, str | None]]:
    candidates: list[tuple[str, BoundingBox, str, str, str | None, str | None]] = []
    for line_index, line in enumerate(result.lines):
        for character_index, character in enumerate(line.characters):
            candidates.append(
                (
                    f"ocr-{image_id}-{line_index}-{character_index}",
                    character.box,
                    "ocr",
                    Decision.REVIEW.value,
                    character.text,
                    result.model_name,
                )
            )
    return candidates


def extract_character_crops(
    manifest: Path,
    image_root: Path,
    output_dir: Path,
    *,
    padding: int = 2,
    ocr_results: Path | None = None,
    ocr_audit: Path | None = None,
) -> CropArtifacts:
    """Write deterministic crops from reviewed boxes and optionally audited OCR boxes.

    OCR input is intentionally accepted only as a JSONL mapping of image IDs to the
    OCR-neutral ``OcrResult`` schema.  Line polygons are never converted into boxes.
    """
    rows = _load_manifest(manifest)
    audited_ocr = _load_audited_ocr(ocr_results, ocr_audit)
    root = image_root.resolve()
    output_rows: list[dict[str, object]] = []

    for row in rows:
        image_id = str(row["image_id"])
        image_path = (root / str(row["image_path"])).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"image path escapes image root: {image_id}") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"real-data image does not exist: {image_path}")
        source_hash = _sha256(image_path)
        if source_hash != str(row["image_sha256"]):
            raise ValueError(f"image changed after import: {image_id}")

        candidates = _reviewed_candidates(row)
        if image_id in audited_ocr.results:
            ocr_candidates = _ocr_candidates(image_id, audited_ocr.results[image_id])
            if not ocr_candidates and not candidates:
                raise ValueError(f"OCR result has no character boxes for image_id={image_id}")
            candidates.extend(ocr_candidates)
        if not candidates:
            continue

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        for source_id, raw_box, annotation_source, decision, text, ocr_model_name in candidates:
            crop_box = _crop_box(raw_box, width, height, padding)
            crop_id = _crop_id(image_id, source_hash, annotation_source, source_id, crop_box)
            png = _png_bytes(image.crop((crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1)))
            crop_path = Path("images") / f"{crop_id}.png"
            _write_immutable(output_dir / crop_path, png)
            output_rows.append(
                {
                    "crop_id": crop_id,
                    "image_id": image_id,
                    "crop_path": crop_path.as_posix(),
                    "source_image_path": str(row["image_path"]),
                    "source_image_sha256": source_hash,
                    "source_annotation_id": source_id,
                    "annotation_source": annotation_source,
                    "decision": decision,
                    "text": text,
                    "crop_box": crop_box.model_dump(mode="json"),
                    "crop_sha256": hashlib.sha256(png).hexdigest(),
                    "ocr_model_name": ocr_model_name,
                    "ocr_audit_sha256": (
                        audited_ocr.audit_sha256 if annotation_source == "ocr" else None
                    ),
                }
            )

    if not output_rows:
        raise ValueError("no reviewed or audited OCR character boxes available for crop extraction")
    output_rows.sort(key=lambda row: str(row["crop_id"]))
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(output_rows), sink, compression="zstd", version="2.6")
    manifest_bytes = cast(bytes, sink.getvalue().to_pybytes())
    manifest_path = output_dir / "crops.parquet"
    audit_path = output_dir / "crops-audit.json"
    _write_immutable(manifest_path, manifest_bytes)
    _write_immutable(
        audit_path,
        _canonical_json(
            {
                "crop_count": len(output_rows),
                "manifest_sha256": _sha256(manifest),
                "ocr_results_sha256": _sha256(ocr_results) if ocr_results is not None else None,
                "ocr_audit_sha256": audited_ocr.audit_sha256,
                "ocr_audit": (
                    audited_ocr.audit.model_dump(mode="json")
                    if audited_ocr.audit is not None
                    else None
                ),
                "padding": padding,
            }
        )
        + b"\n",
    )
    return CropArtifacts(manifest=manifest_path, audit=audit_path)
