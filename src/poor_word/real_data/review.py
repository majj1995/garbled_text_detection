"""Versioned review bundles and conflict-safe human label imports."""

import csv
import hashlib
import io
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, ImageDraw

from poor_word.domain import Decision


@dataclass(frozen=True)
class ReviewCandidate:
    crop_id: str
    image_id: str
    crop_path: str
    risk_score: float
    style_id: str
    disagreement: float
    label: None = None


@dataclass(frozen=True)
class ReviewExportArtifacts:
    queue: Path
    queue_version: str
    csv: Path
    jsonl: Path
    contact_sheet: Path


@dataclass(frozen=True)
class ReviewImportArtifacts:
    gold_crops: Path
    audit: Path


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable review artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    try:
        part.write_bytes(payload)
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _required_text(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"review score requires non-empty {field}")
    return value


def build_review_queue(
    scores: Sequence[Mapping[str, object]],
    disagreements: Mapping[str, float],
    limit: int,
    seed: int,
) -> tuple[ReviewCandidate, ...]:
    """Rank unique candidates by disagreement, normal-boundary proximity and style coverage."""
    if limit <= 0:
        raise ValueError("review queue limit must be positive")
    candidates: dict[str, ReviewCandidate] = {}
    for raw in scores:
        crop_id = _required_text(raw, "crop_id")
        risk_score_raw = raw.get("risk_score")
        if not isinstance(risk_score_raw, (int, float)) or not 0 <= risk_score_raw <= 1:
            raise ValueError(f"review score has invalid risk_score for crop_id={crop_id}")
        disagreement = disagreements.get(crop_id, 0.0)
        if not isinstance(disagreement, (int, float)) or disagreement < 0:
            raise ValueError(f"review score has invalid disagreement for crop_id={crop_id}")
        candidate = ReviewCandidate(
            crop_id=crop_id,
            image_id=_required_text(raw, "image_id"),
            crop_path=_required_text(raw, "crop_path"),
            risk_score=float(risk_score_raw),
            style_id=_required_text(raw, "style_id"),
            disagreement=float(disagreement),
        )
        previous = candidates.get(crop_id)
        if previous is not None and (
            previous.image_id != candidate.image_id
            or previous.crop_path != candidate.crop_path
            or previous.style_id != candidate.style_id
        ):
            raise ValueError(f"duplicate crop_id has inconsistent metadata: {crop_id}")
        if previous is None or (candidate.disagreement, candidate.risk_score) > (
            previous.disagreement,
            previous.risk_score,
        ):
            candidates[crop_id] = candidate

    style_counts = Counter(candidate.style_id for candidate in candidates.values())
    randomized_ties = {
        crop_id: random.Random(f"{seed}:{crop_id}").random() for crop_id in candidates
    }
    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.disagreement,
            abs(candidate.risk_score - 0.5),
            style_counts[candidate.style_id],
            randomized_ties[candidate.crop_id],
            candidate.crop_id,
        ),
    )
    return tuple(ranked[:limit])


def _load_crop_rows(crop_manifest: Path) -> dict[str, dict[str, object]]:
    rows = cast(list[dict[str, object]], pq.read_table(crop_manifest).to_pylist())
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        crop_id = row.get("crop_id")
        crop_path = row.get("crop_path")
        if (
            not isinstance(crop_id, str)
            or not crop_id
            or not isinstance(crop_path, str)
            or not crop_path
        ):
            raise ValueError("crop manifest rows require crop_id and crop_path")
        if crop_id in by_id:
            raise ValueError(f"crop manifest has duplicate crop_id: {crop_id}")
        by_id[crop_id] = row
    return by_id


def _safe_crop_path(root: Path, relative: str) -> Path:
    path = (root.resolve() / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"crop path escapes crop root: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(f"crop image does not exist: {path}")
    return path


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    fields = [
        "queue_version",
        "crop_id",
        "image_id",
        "crop_path",
        "risk_score",
        "style_id",
        "disagreement",
        "label",
        "annotator_id",
        "source_image_sha256",
        "crop_sha256",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _contact_sheet(rows: list[dict[str, object]], crop_root: Path) -> bytes:
    cell = 128
    columns = min(4, max(1, len(rows)))
    rows_count = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows_count * (cell + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        with Image.open(_safe_crop_path(crop_root, str(row["crop_path"]))) as opened:
            image = opened.convert("RGB")
        image.thumbnail((cell, cell))
        x = (index % columns) * cell + (cell - image.width) // 2
        y = (index // columns) * (cell + 20) + (cell - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            ((index % columns) * cell + 2, (index // columns) * (cell + 20) + cell),
            str(row["crop_id"])[:12],
            fill="black",
        )
    output = io.BytesIO()
    sheet.save(output, format="PNG", optimize=False)
    return output.getvalue()


def export_review_queue(
    queue: Sequence[ReviewCandidate],
    crop_manifest: Path,
    crop_root: Path,
    output_dir: Path,
) -> ReviewExportArtifacts:
    """Export an immutable, reviewable CSV/JSONL/contact-sheet bundle."""
    if not queue:
        raise ValueError("cannot export an empty review queue")
    crop_rows = _load_crop_rows(crop_manifest)
    records: list[dict[str, object]] = []
    for candidate in queue:
        source = crop_rows.get(candidate.crop_id)
        if source is None:
            raise ValueError(f"review candidate is absent from crop manifest: {candidate.crop_id}")
        if str(source["crop_path"]) != candidate.crop_path:
            raise ValueError(
                f"review candidate crop_path does not match manifest: {candidate.crop_id}"
            )
        _safe_crop_path(crop_root, candidate.crop_path)
        records.append(
            {
                **asdict(candidate),
                "source_image_sha256": str(source.get("source_image_sha256", "")),
                "crop_sha256": str(source.get("crop_sha256", "")),
            }
        )
    records.sort(key=lambda record: str(record["crop_id"]))
    version_payload = {
        "crop_manifest_sha256": _sha256(crop_manifest),
        "candidates": records,
    }
    queue_version = hashlib.sha256(_canonical_json(version_payload)).hexdigest()
    for record in records:
        record["queue_version"] = queue_version
    queue_dir = output_dir / f"queue-{queue_version}"
    jsonl = queue_dir / "queue.jsonl"
    csv_path = queue_dir / "queue.csv"
    contact_sheet = queue_dir / "contact-sheet.png"
    metadata = queue_dir / "queue.json"
    _write_immutable(
        jsonl,
        b"".join(_canonical_json(record) + b"\n" for record in records),
    )
    _write_immutable(csv_path, _csv_bytes(records))
    _write_immutable(contact_sheet, _contact_sheet(records, crop_root))
    _write_immutable(
        metadata,
        _canonical_json(
            {
                "candidate_count": len(records),
                "crop_manifest_sha256": _sha256(crop_manifest),
                "queue_version": queue_version,
            }
        )
        + b"\n",
    )
    return ReviewExportArtifacts(
        queue=queue_dir,
        queue_version=queue_version,
        csv=csv_path,
        jsonl=jsonl,
        contact_sheet=contact_sheet,
    )


def _load_queue(queue_dir: Path) -> tuple[str, dict[str, dict[str, object]]]:
    try:
        metadata = json.loads((queue_dir / "queue.json").read_text(encoding="utf-8"))
        version = metadata["queue_version"]
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid review queue metadata: {error}") from error
    if not isinstance(version, str) or not version:
        raise ValueError("invalid review queue version")
    records: dict[str, dict[str, object]] = {}
    for number, line in enumerate(
        (queue_dir / "queue.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
            crop_id = row["crop_id"]
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid review queue row at line {number}: {error}") from error
        if (
            not isinstance(crop_id, str)
            or crop_id in records
            or row.get("queue_version") != version
        ):
            raise ValueError(f"invalid review queue row at line {number}")
        records[crop_id] = cast(dict[str, object], row)
    return version, records


def _read_label_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid label JSONL at line {number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"invalid label JSONL at line {number}")
        rows.append(cast(dict[str, object], row))
    return rows


def _load_existing_gold(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    rows = cast(list[dict[str, object]], pq.read_table(path).to_pylist())
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        crop_id = row.get("crop_id")
        if not isinstance(crop_id, str) or crop_id in result:
            raise ValueError("existing gold has missing or duplicate crop_id")
        decision = Decision(str(row.get("decision")))
        if decision not in {Decision.PASS, Decision.BLOCK}:
            raise ValueError("existing gold contains unresolved decision")
        result[crop_id] = row
    return result


_GOLD_SCHEMA = pa.schema(
    [
        ("crop_id", pa.string()),
        ("image_id", pa.string()),
        ("crop_path", pa.string()),
        ("decision", pa.string()),
        ("annotator_id", pa.string()),
        ("queue_version", pa.string()),
        ("queue_manifest_sha256", pa.string()),
        ("source_image_sha256", pa.string()),
        ("crop_sha256", pa.string()),
        ("risk_score", pa.float64()),
        ("style_id", pa.string()),
    ]
)


def import_review_labels(
    labels: Path,
    queue: Path,
    output_dir: Path,
    *,
    existing_gold: Path | None = None,
) -> ReviewImportArtifacts:
    """Validate human labels and write only PASS/BLOCK rows to versioned gold data."""
    queue_version, queue_rows = _load_queue(queue)
    gold = _load_existing_gold(existing_gold)
    seen: set[str] = set()
    pending: list[tuple[str, str, Decision]] = []
    for row in _read_label_rows(labels):
        crop_id = row.get("crop_id")
        queue_value = row.get("queue_version")
        annotator_id = row.get("annotator_id")
        if not isinstance(crop_id, str) or not crop_id or crop_id not in queue_rows:
            raise ValueError("label references crop outside exported queue")
        if queue_value != queue_version:
            raise ValueError("label queue version does not match exported queue")
        if crop_id in seen:
            raise ValueError(f"duplicate crop label: {crop_id}")
        seen.add(crop_id)
        if not isinstance(annotator_id, str) or not annotator_id:
            raise ValueError(f"label requires annotator_id for crop_id={crop_id}")
        try:
            decision = Decision(str(row.get("label")))
        except ValueError as error:
            raise ValueError(f"invalid review label for crop_id={crop_id}") from error
        pending.append((crop_id, annotator_id, decision))

    for crop_id, annotator_id, decision in pending:
        if decision is Decision.REVIEW:
            continue
        existing = gold.get(crop_id)
        if existing is not None:
            if (
                str(existing.get("annotator_id")) != annotator_id
                or str(existing.get("decision")) != decision.value
            ):
                raise ValueError(f"cannot overwrite accepted gold label for crop_id={crop_id}")
            continue
        source = queue_rows[crop_id]
        risk_score = source.get("risk_score")
        if not isinstance(risk_score, (int, float)):
            raise ValueError(f"invalid queued risk score for crop_id={crop_id}")
        gold[crop_id] = {
            "crop_id": crop_id,
            "image_id": str(source["image_id"]),
            "crop_path": str(source["crop_path"]),
            "decision": decision.value,
            "annotator_id": annotator_id,
            "queue_version": queue_version,
            "queue_manifest_sha256": _sha256(queue / "queue.jsonl"),
            "source_image_sha256": str(source.get("source_image_sha256", "")),
            "crop_sha256": str(source.get("crop_sha256", "")),
            "risk_score": float(risk_score),
            "style_id": str(source["style_id"]),
        }

    output_rows = [gold[crop_id] for crop_id in sorted(gold)]
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(output_rows, schema=_GOLD_SCHEMA),
        sink,
        compression="zstd",
        version="2.6",
    )
    parquet_bytes = cast(bytes, sink.getvalue().to_pybytes())
    gold_version = hashlib.sha256(parquet_bytes).hexdigest()
    gold_dir = output_dir / f"gold-{gold_version}"
    gold_path = gold_dir / "gold-crops.parquet"
    audit_path = gold_dir / "import-audit.json"
    _write_immutable(gold_path, parquet_bytes)
    _write_immutable(
        audit_path,
        _canonical_json(
            {
                "accepted_gold_count": len(output_rows),
                "labels_sha256": _sha256(labels),
                "queue_version": queue_version,
                "review_count": len(seen)
                - sum(1 for row in output_rows if row["queue_version"] == queue_version),
            }
        )
        + b"\n",
    )
    return ReviewImportArtifacts(gold_crops=gold_path, audit=audit_path)
