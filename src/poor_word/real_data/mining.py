"""Leakage-safe hard-example mining and reviewed-yield versioning."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from poor_word.real_data.schema import ImageLabel, SplitRole

_BUCKET_ORDER = (
    "ABNORMAL_HIGH_ATTENTION",
    "NEW_STYLE_CLUSTER",
    "DISAGREEMENT",
    "THRESHOLD_BAND",
    "NORMAL_FALSE_POSITIVE",
)
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class MiningPolicy:
    normal_false_positive_threshold: float = 0.8
    disagreement_threshold: float = 0.25
    threshold_band_low: float = 0.45
    threshold_band_high: float = 0.55
    abnormal_attention_threshold: float = 0.8
    style_novelty_threshold: float = 0.8
    overall_limit: int = 500
    per_product_cap: int = 25
    per_template_cap: int = 10
    per_source_cap: int = 50
    seed: int = 20260804
    base_rate_context: float = 0.001

    def __post_init__(self) -> None:
        probabilities = (
            self.normal_false_positive_threshold,
            self.disagreement_threshold,
            self.threshold_band_low,
            self.threshold_band_high,
            self.abnormal_attention_threshold,
            self.style_novelty_threshold,
            self.base_rate_context,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("mining thresholds and base rate must be finite values in [0, 1]")
        if self.threshold_band_low > self.threshold_band_high:
            raise ValueError("threshold band low must not exceed high")
        if any(
            value <= 0
            for value in (
                self.overall_limit,
                self.per_product_cap,
                self.per_template_cap,
                self.per_source_cap,
            )
        ):
            raise ValueError("mining limits and caps must be positive")
        if self.seed < 0:
            raise ValueError("mining seed must be non-negative")


@dataclass(frozen=True)
class MiningArtifacts:
    queue: Path
    metadata: Path
    review_scores: Path
    review_disagreements: Path


@dataclass(frozen=True)
class DatasetVersionArtifacts:
    dataset_version: Path
    audit: Path


@dataclass(frozen=True)
class _Candidate:
    row: dict[str, object]
    reasons: tuple[str, ...]
    priority: float


_QUEUE_SCHEMA = pa.schema(
    [
        ("queue_version", pa.string()),
        ("priority_rank", pa.int64()),
        ("priority", pa.float64()),
        ("reason_buckets", pa.list_(pa.string())),
        ("crop_id", pa.string()),
        ("image_id", pa.string()),
        ("crop_path", pa.string()),
        ("crop_sha256", pa.string()),
        ("source_image_sha256", pa.string()),
        ("image_label", pa.string()),
        ("split_role", pa.string()),
        ("fold", pa.int64()),
        ("source_id", pa.string()),
        ("product_id", pa.string()),
        ("template_id", pa.string()),
        ("risk_score", pa.float64()),
        ("alternate_risk_score", pa.float64()),
        ("attention_score", pa.float64()),
        ("style_cluster_id", pa.string()),
        ("style_novelty_score", pa.float64()),
        ("style_is_unseen", pa.bool_()),
        ("score_model_id", pa.string()),
        ("score_checkpoint_sha256", pa.string()),
        ("alternate_model_id", pa.string()),
        ("alternate_checkpoint_sha256", pa.string()),
        ("real_manifest_sha256", pa.string()),
        ("fold_manifest_sha256", pa.string()),
        ("score_manifest_sha256", pa.string()),
        ("label_source", pa.string()),
        ("is_gold", pa.bool_()),
    ]
)


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


def _parquet_bytes(rows: Sequence[Mapping[str, object]], schema: pa.Schema) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=schema),
        sink,
        compression="zstd",
        version="2.6",
    )
    return cast(bytes, sink.getvalue().to_pybytes())


def _load_rows(path: Path, artifact: str) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact} does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return cast(list[dict[str, object]], pq.read_table(path).to_pylist())
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, object]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid {artifact} JSONL at line {number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{artifact} JSONL row {number} must be an object")
            rows.append(cast(dict[str, object], value))
        return rows
    raise ValueError(f"{artifact} must be Parquet or JSONL")


def _text(row: Mapping[str, object], field: str, artifact: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{artifact} requires non-empty {field}")
    return value


def _sha(row: Mapping[str, object], field: str, artifact: str) -> str:
    value = _text(row, field, artifact)
    if len(value) != 64 or any(character not in _SHA256_CHARS for character in value):
        raise ValueError(f"{artifact} requires lowercase SHA-256 {field}")
    return value


def _score(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"score row requires numeric {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"score row {field} must be finite")
    if not 0 <= result <= 1:
        raise ValueError(f"score row {field} must be in [0, 1]")
    return result


def _relative_path(row: Mapping[str, object], field: str) -> str:
    value = _text(row, field, "score row")
    path = Path(value)
    if path.is_absolute() or not path.name or ".." in path.parts:
        raise ValueError(f"score row {field} must be a safe relative file path")
    return path.as_posix()


def _index_rows(
    rows: Sequence[Mapping[str, object]], artifact: str, required: frozenset[str]
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{artifact} row is missing: {', '.join(missing)}")
        image_id = _text(row, "image_id", artifact)
        if image_id in result:
            raise ValueError(f"{artifact} has duplicate image_id: {image_id}")
        result[image_id] = row
    if not result:
        raise ValueError(f"{artifact} contains no rows")
    return result


def _candidate_reasons(
    image_label: str,
    risk: float,
    alternate: float,
    attention: float,
    novelty: float,
    unseen: bool,
    policy: MiningPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        image_label == ImageLabel.ABNORMAL.value
        and attention >= policy.abnormal_attention_threshold
    ):
        reasons.append("ABNORMAL_HIGH_ATTENTION")
    if unseen and novelty >= policy.style_novelty_threshold:
        reasons.append("NEW_STYLE_CLUSTER")
    if abs(risk - alternate) >= policy.disagreement_threshold:
        reasons.append("DISAGREEMENT")
    if policy.threshold_band_low <= risk <= policy.threshold_band_high:
        reasons.append("THRESHOLD_BAND")
    if image_label == ImageLabel.NORMAL.value and risk >= policy.normal_false_positive_threshold:
        reasons.append("NORMAL_FALSE_POSITIVE")
    return tuple(reasons)


def _priority(
    reasons: Sequence[str],
    risk: float,
    alternate: float,
    attention: float,
    novelty: float,
    policy: MiningPolicy,
) -> float:
    signals: list[float] = []
    if "ABNORMAL_HIGH_ATTENTION" in reasons:
        signals.append(attention)
    if "NEW_STYLE_CLUSTER" in reasons:
        signals.append(novelty)
    if "DISAGREEMENT" in reasons:
        signals.append(abs(risk - alternate))
    if "THRESHOLD_BAND" in reasons:
        midpoint = (policy.threshold_band_low + policy.threshold_band_high) / 2
        width = max((policy.threshold_band_high - policy.threshold_band_low) / 2, 1e-12)
        signals.append(1 - min(abs(risk - midpoint) / width, 1))
    if "NORMAL_FALSE_POSITIVE" in reasons:
        signals.append(risk)
    return max(signals)


def _validate_and_build_candidates(
    score_rows: Sequence[Mapping[str, object]],
    real_rows: Mapping[str, Mapping[str, object]],
    fold_rows: Mapping[str, Mapping[str, object]],
    real_hash: str,
    fold_hash: str,
    score_hash: str,
    policy: MiningPolicy,
) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    seen_crop_ids: set[str] = set()
    locked_skipped = 0
    roles = {role.value for role in SplitRole}
    labels = {label.value for label in ImageLabel}
    for raw in score_rows:
        crop_id = _text(raw, "crop_id", "score row")
        if crop_id in seen_crop_ids:
            raise ValueError(f"score manifest has duplicate crop_id: {crop_id}")
        seen_crop_ids.add(crop_id)
        image_id = _text(raw, "image_id", "score row")
        real = real_rows.get(image_id)
        fold = fold_rows.get(image_id)
        if real is None or fold is None:
            raise ValueError(f"score row references unknown image_id: {image_id}")
        trusted_role = _text(fold, "split_role", "fold manifest")
        trusted_fold = fold.get("fold")
        if trusted_role not in roles or not isinstance(trusted_fold, int):
            raise ValueError(f"fold manifest has invalid role/fold for image_id={image_id}")
        if trusted_role == SplitRole.LOCKED_TEST.value or trusted_fold == -1:
            if trusted_role != SplitRole.LOCKED_TEST.value or trusted_fold != -1:
                raise ValueError(f"locked-test fold/role mismatch for image_id={image_id}")
            locked_skipped += 1
            continue
        if trusted_fold < 0:
            raise ValueError(f"development fold must be non-negative for image_id={image_id}")

        image_label = _text(real, "image_label", "real manifest")
        real_role = _text(real, "split_role", "real manifest")
        if image_label not in labels or real_role not in roles:
            raise ValueError(f"real manifest has invalid enum for image_id={image_id}")
        if real_role != trusted_role or _text(fold, "image_label", "fold manifest") != image_label:
            raise ValueError(f"real/fold manifest role or label mismatch for image_id={image_id}")
        if real.get("production_allowed") is not True or real.get("training_eligible") is not True:
            raise ValueError(f"score row references ineligible source for image_id={image_id}")
        if raw.get("fold") != trusted_fold:
            raise ValueError(f"score row fold mismatch for image_id={image_id}")
        for field, expected in (
            ("image_label", image_label),
            ("split_role", real_role),
            ("source_id", _text(real, "source_id", "real manifest")),
            ("product_id", _text(real, "product_id", "real manifest")),
            ("template_id", _text(real, "template_id", "real manifest")),
            ("source_image_sha256", _sha(real, "image_sha256", "real manifest")),
        ):
            if raw.get(field) != expected:
                raise ValueError(f"score row {field} mismatch for image_id={image_id}")
        if raw.get("real_manifest_sha256") != real_hash:
            raise ValueError("score row real manifest hash mismatch")
        if raw.get("fold_manifest_sha256") != fold_hash:
            raise ValueError("score row fold manifest hash mismatch")

        risk = _score(raw, "risk_score")
        alternate = _score(raw, "alternate_risk_score")
        attention = _score(raw, "attention_score")
        novelty = _score(raw, "style_novelty_score")
        unseen = raw.get("style_is_unseen")
        if not isinstance(unseen, bool):
            raise ValueError("score row requires boolean style_is_unseen")
        reasons = _candidate_reasons(
            image_label, risk, alternate, attention, novelty, unseen, policy
        )
        row: dict[str, object] = {
            "crop_id": crop_id,
            "image_id": image_id,
            "crop_path": _relative_path(raw, "crop_path"),
            "crop_sha256": _sha(raw, "crop_sha256", "score row"),
            "source_image_sha256": _sha(raw, "source_image_sha256", "score row"),
            "image_label": image_label,
            "split_role": real_role,
            "fold": trusted_fold,
            "source_id": _text(raw, "source_id", "score row"),
            "product_id": _text(raw, "product_id", "score row"),
            "template_id": _text(raw, "template_id", "score row"),
            "risk_score": risk,
            "alternate_risk_score": alternate,
            "attention_score": attention,
            "style_cluster_id": _text(raw, "style_cluster_id", "score row"),
            "style_novelty_score": novelty,
            "style_is_unseen": unseen,
            "score_model_id": _text(raw, "score_model_id", "score row"),
            "score_checkpoint_sha256": _sha(raw, "score_checkpoint_sha256", "score row"),
            "alternate_model_id": _text(raw, "alternate_model_id", "score row"),
            "alternate_checkpoint_sha256": _sha(raw, "alternate_checkpoint_sha256", "score row"),
            "real_manifest_sha256": real_hash,
            "fold_manifest_sha256": fold_hash,
            "score_manifest_sha256": score_hash,
            "label_source": "mining_candidate",
            "is_gold": False,
        }
        if not reasons:
            continue
        candidates.append(
            _Candidate(
                row=row,
                reasons=reasons,
                priority=_priority(reasons, risk, alternate, attention, novelty, policy),
            )
        )
    return candidates, locked_skipped


def _rank_key(candidate: _Candidate, seed: int) -> tuple[float, float, str]:
    tie = random.Random(f"{seed}:{candidate.row['crop_id']}").random()
    return (-candidate.priority, tie, str(candidate.row["crop_id"]))


def _suppress_duplicates(
    candidates: Sequence[_Candidate], seed: int
) -> tuple[list[_Candidate], int]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            min(_BUCKET_ORDER.index(reason) for reason in candidate.reasons),
            *_rank_key(candidate, seed),
        ),
    )
    crop_hashes: set[str] = set()
    image_hashes: set[str] = set()
    kept: list[_Candidate] = []
    suppressed = 0
    for candidate in ordered:
        crop_hash = str(candidate.row["crop_sha256"])
        image_hash = str(candidate.row["source_image_sha256"])
        if crop_hash in crop_hashes or image_hash in image_hashes:
            suppressed += 1
            continue
        crop_hashes.add(crop_hash)
        image_hashes.add(image_hash)
        kept.append(candidate)
    return kept, suppressed


def _apply_caps_and_coverage(
    candidates: Sequence[_Candidate], policy: MiningPolicy
) -> tuple[list[_Candidate], Counter[str]]:
    by_bucket = {
        bucket: sorted(
            (candidate for candidate in candidates if bucket in candidate.reasons),
            key=lambda candidate: _rank_key(candidate, policy.seed),
        )
        for bucket in _BUCKET_ORDER
    }
    positions = {bucket: 0 for bucket in _BUCKET_ORDER}
    selected: list[_Candidate] = []
    handled: set[str] = set()
    product_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    suppressed: Counter[str] = Counter()
    while len(selected) < policy.overall_limit:
        progressed = False
        for bucket in _BUCKET_ORDER:
            values = by_bucket[bucket]
            while positions[bucket] < len(values):
                candidate = values[positions[bucket]]
                positions[bucket] += 1
                crop_id = str(candidate.row["crop_id"])
                if crop_id in handled:
                    continue
                handled.add(crop_id)
                product = str(candidate.row["product_id"])
                template = str(candidate.row["template_id"])
                source = str(candidate.row["source_id"])
                cap_reasons: list[str] = []
                if product_counts[product] >= policy.per_product_cap:
                    cap_reasons.append("product")
                if template_counts[template] >= policy.per_template_cap:
                    cap_reasons.append("template")
                if source_counts[source] >= policy.per_source_cap:
                    cap_reasons.append("source")
                if cap_reasons:
                    suppressed.update(cap_reasons)
                else:
                    selected.append(candidate)
                    product_counts[product] += 1
                    template_counts[template] += 1
                    source_counts[source] += 1
                progressed = True
                break
            if len(selected) >= policy.overall_limit:
                break
        if not progressed:
            break
    return selected, suppressed


def _publish_directory(output_dir: Path, files: Mapping[str, bytes], artifact: str) -> None:
    if output_dir.exists():
        if not output_dir.is_dir() or set(path.name for path in output_dir.iterdir()) != set(files):
            raise ValueError(f"immutable {artifact} already differs: {output_dir}")
        if any((output_dir / name).read_bytes() != payload for name, payload in files.items()):
            raise ValueError(f"immutable {artifact} already differs: {output_dir}")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        for name, payload in files.items():
            (staging / name).write_bytes(payload)
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def mine_candidates(
    scores: Path,
    policy: MiningPolicy,
    *,
    real_manifest: Path,
    fold_manifest: Path,
    output_dir: Path,
) -> MiningArtifacts:
    """Mine a deterministic human-review queue without creating character gold."""
    real_hash = _sha256(real_manifest)
    fold_hash = _sha256(fold_manifest)
    score_hash = _sha256(scores)
    real_rows = _index_rows(
        _load_rows(real_manifest, "real manifest"),
        "real manifest",
        frozenset(
            {
                "image_id",
                "image_sha256",
                "image_label",
                "split_role",
                "source_id",
                "product_id",
                "template_id",
                "production_allowed",
                "training_eligible",
            }
        ),
    )
    fold_rows = _index_rows(
        _load_rows(fold_manifest, "fold manifest"),
        "fold manifest",
        frozenset({"image_id", "fold", "image_label", "split_role"}),
    )
    if set(real_rows) != set(fold_rows):
        raise ValueError("real and fold manifests must contain identical image_id values")
    candidates, locked_skipped = _validate_and_build_candidates(
        _load_rows(scores, "score manifest"),
        real_rows,
        fold_rows,
        real_hash,
        fold_hash,
        score_hash,
        policy,
    )
    deduplicated, duplicate_suppression_count = _suppress_duplicates(candidates, policy.seed)
    selected, cap_suppression_counts = _apply_caps_and_coverage(deduplicated, policy)
    rows_without_version = [
        {
            "priority_rank": rank,
            "priority": candidate.priority,
            "reason_buckets": list(candidate.reasons),
            **candidate.row,
        }
        for rank, candidate in enumerate(selected, start=1)
    ]
    queue_version = hashlib.sha256(_canonical_json(rows_without_version)).hexdigest()
    queue_rows = [{"queue_version": queue_version, **row} for row in rows_without_version]
    queue_bytes = _parquet_bytes(queue_rows, _QUEUE_SCHEMA)
    score_jsonl = b"".join(
        _canonical_json(
            {
                "crop_id": row["crop_id"],
                "image_id": row["image_id"],
                "crop_path": row["crop_path"],
                "risk_score": row["risk_score"],
                "style_id": row["style_cluster_id"],
                "score_model_id": row["score_model_id"],
                "score_artifact_sha256": row["score_checkpoint_sha256"],
            }
        )
        + b"\n"
        for row in queue_rows
    )
    disagreement_jsonl = b"".join(
        _canonical_json(
            {
                "crop_id": row["crop_id"],
                "disagreement": abs(
                    cast(float, row["risk_score"]) - cast(float, row["alternate_risk_score"])
                ),
                "disagreement_model_id": row["alternate_model_id"],
                "disagreement_artifact_sha256": row["alternate_checkpoint_sha256"],
            }
        )
        + b"\n"
        for row in queue_rows
    )
    bucket_counts = Counter(reason for candidate in selected for reason in candidate.reasons)
    metadata = {
        "queue_version": queue_version,
        "candidate_count": len(queue_rows),
        "bucket_counts": {bucket: bucket_counts[bucket] for bucket in _BUCKET_ORDER},
        "cap_suppression_counts": {
            key: cap_suppression_counts[key] for key in ("product", "template", "source")
        },
        "duplicate_suppression_count": duplicate_suppression_count,
        "locked_test_skipped_count": locked_skipped,
        "policy": asdict(policy),
        "seed": policy.seed,
        "real_manifest_sha256": real_hash,
        "fold_manifest_sha256": fold_hash,
        "score_manifest_sha256": score_hash,
        "output_queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "output_review_scores_sha256": hashlib.sha256(score_jsonl).hexdigest(),
        "output_review_disagreements_sha256": hashlib.sha256(disagreement_jsonl).hexdigest(),
        "base_rate_context": policy.base_rate_context,
        "production_prevalence_estimate": None,
        "sampling_warning": "mined candidate yield is not a production prevalence estimate",
    }
    files = {
        "queue.parquet": queue_bytes,
        "review-scores.jsonl": score_jsonl,
        "review-disagreements.jsonl": disagreement_jsonl,
        "mining-audit.json": _canonical_json(metadata) + b"\n",
    }
    _publish_directory(output_dir, files, "mining artifact")
    return MiningArtifacts(
        queue=output_dir / "queue.parquet",
        metadata=output_dir / "mining-audit.json",
        review_scores=output_dir / "review-scores.jsonl",
        review_disagreements=output_dir / "review-disagreements.jsonl",
    )


def _load_gold(path: Path, artifact: str) -> dict[str, dict[str, object]]:
    rows = _load_rows(path, artifact)
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        crop_id = _text(row, "crop_id", artifact)
        if crop_id in result:
            raise ValueError(f"{artifact} has duplicate crop_id: {crop_id}")
        decision = row.get("decision")
        if decision not in {"PASS", "BLOCK"}:
            raise ValueError(f"{artifact} contains unresolved or invalid decision")
        _text(row, "annotator_id", artifact)
        result[crop_id] = row
    return result


def _read_import_audit(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid review import audit: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("review import audit must be an object")
    audit = cast(dict[str, object], raw)
    _text(audit, "queue_version", "review import audit")
    for field in ("accepted_label_count", "new_gold_count", "review_count"):
        value = audit.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"review import audit requires non-negative integer {field}")
    return audit


def record_mining_yield(
    candidate_queue: Path,
    reviewed_gold_manifest: Path,
    import_audit: Path,
    *,
    base_real_manifest: Path,
    output_dir: Path,
    previous_gold_manifest: Path | None = None,
) -> DatasetVersionArtifacts:
    """Version trusted review yield; never create or modify a gold manifest."""
    queue_rows = _load_rows(candidate_queue, "mining queue")
    queue: dict[str, dict[str, object]] = {}
    for row in queue_rows:
        crop_id = _text(row, "crop_id", "mining queue")
        if crop_id in queue:
            raise ValueError(f"mining queue has duplicate crop_id: {crop_id}")
        if row.get("label_source") != "mining_candidate" or row.get("is_gold") is not False:
            raise ValueError("mining queue provenance is invalid")
        queue[crop_id] = row
    if not queue:
        raise ValueError("mining queue contains no candidates")

    current = _load_gold(reviewed_gold_manifest, "reviewed gold manifest")
    previous = (
        _load_gold(previous_gold_manifest, "previous gold manifest")
        if previous_gold_manifest is not None
        else {}
    )
    for crop_id, old in previous.items():
        if current.get(crop_id) != old:
            raise ValueError(f"reviewed gold changed previous gold row: {crop_id}")
    new = {crop_id: row for crop_id, row in current.items() if crop_id not in previous}
    audit = _read_import_audit(import_audit)
    if audit["new_gold_count"] != len(new) or audit["accepted_label_count"] != len(new):
        raise ValueError("review import audit counts do not match new reviewed gold")
    queue_version = str(audit["queue_version"])
    for crop_id, row in new.items():
        candidate = queue.get(crop_id)
        if candidate is None:
            raise ValueError(f"new reviewed gold is outside mining queue: {crop_id}")
        if row.get("queue_version") != queue_version:
            raise ValueError(f"review queue provenance mismatch for crop_id={crop_id}")
        comparisons = (
            ("image_id", "image_id"),
            ("crop_sha256", "crop_sha256"),
            ("source_image_sha256", "source_image_sha256"),
            ("score_model_id", "score_model_id"),
            ("score_artifact_sha256", "score_checkpoint_sha256"),
        )
        for gold_field, queue_field in comparisons:
            if row.get(gold_field) != candidate.get(queue_field):
                raise ValueError(f"reviewed gold provenance mismatch for crop_id={crop_id}")

    pass_count = sum(row["decision"] == "PASS" for row in new.values())
    block_count = sum(row["decision"] == "BLOCK" for row in new.values())
    accepted = audit["accepted_label_count"]
    unresolved = cast(int, audit["review_count"])
    reviewed = accepted + unresolved
    counts = {
        "accepted_block": block_count,
        "accepted_pass": pass_count,
        "accepted_total": accepted,
        "reviewed_total": reviewed,
        "unresolved_review": unresolved,
    }
    lineage = {
        "base_real_manifest_sha256": _sha256(base_real_manifest),
        "candidate_queue_sha256": _sha256(candidate_queue),
        "review_import_audit_sha256": _sha256(import_audit),
        "previous_gold_manifest_sha256": (
            _sha256(previous_gold_manifest) if previous_gold_manifest is not None else None
        ),
        "new_gold_manifest_sha256": _sha256(reviewed_gold_manifest),
    }
    version_identity = hashlib.sha256(
        _canonical_json({"lineage": lineage, "counts": counts})
    ).hexdigest()
    payload = {
        "dataset_version": version_identity,
        "lineage": lineage,
        "counts": counts,
        "accepted_yield": accepted / reviewed if reviewed else None,
        "yield_denominator": ("accepted_label_count + review_count in the trusted import audit"),
        "yield_status": "observed_import_batch" if reviewed else "no_reviewed_rows",
        "base_rate_context": 0.001,
        "production_prevalence_estimate": None,
        "sampling_warning": "mined candidate yield is not a production prevalence estimate",
    }
    audit_payload = {
        **payload,
        "queue_candidate_count": len(queue),
        "previous_gold_count": len(previous),
        "current_gold_count": len(current),
        "new_gold_count": len(new),
    }
    files = {
        "dataset-version.json": _canonical_json(payload) + b"\n",
        "mining-yield-audit.json": _canonical_json(audit_payload) + b"\n",
    }
    _publish_directory(output_dir, files, "dataset version artifact")
    return DatasetVersionArtifacts(
        dataset_version=output_dir / "dataset-version.json",
        audit=output_dir / "mining-yield-audit.json",
    )
