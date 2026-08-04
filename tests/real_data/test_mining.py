import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from PIL import Image

from poor_word.real_data.mining import (
    MiningPolicy,
    mine_candidates,
    record_mining_yield,
)
from poor_word.real_data.review import (
    ReviewCandidate,
    export_review_queue,
    import_review_labels,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _manifests(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    definitions = [
        ("normal-fp", "NORMAL", "DEV", 0),
        ("disagree", "NORMAL", "NORMAL_REPLAY", 1),
        ("threshold", "NORMAL", "DEV", 2),
        ("attention", "ABNORMAL", "IMAGE_ONLY", 3),
        ("new-style", "ABNORMAL", "DEV", 4),
        ("normal-extra", "NORMAL", "DEV", 0),
        ("crop-duplicate", "ABNORMAL", "DEV", 1),
        ("locked", "ABNORMAL", "LOCKED_TEST", -1),
    ]
    real_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for index, (image_id, label, role, fold) in enumerate(definitions, start=1):
        image_sha = f"{index:064x}"
        real_rows.append(
            {
                "image_id": image_id,
                "image_sha256": image_sha,
                "image_label": label,
                "split_role": role,
                "source_id": f"source-{index}",
                "product_id": "shared-product" if image_id == "normal-extra" else f"p-{index}",
                "template_id": f"t-{index}",
                "production_allowed": role != "LOCKED_TEST",
                "training_eligible": role != "LOCKED_TEST",
            }
        )
        fold_rows.append(
            {
                "image_id": image_id,
                "fold": fold,
                "image_label": label,
                "split_role": role,
            }
        )
    # Exercise the product cap against normal-fp.
    real_rows[0]["product_id"] = "shared-product"
    return (
        _write_parquet(tmp_path / "manifest.parquet", real_rows),
        _write_parquet(tmp_path / "folds.parquet", fold_rows),
        real_rows,
    )


def _score_rows(
    real_manifest: Path, fold_manifest: Path, real_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    real_hash = _sha(real_manifest)
    fold_hash = _sha(fold_manifest)
    by_id = {str(row["image_id"]): row for row in real_rows}

    def row(
        image_id: str,
        risk: float,
        alternate: float,
        attention: float,
        novelty: float,
        unseen: bool,
        *,
        crop_sha: str,
    ) -> dict[str, object]:
        source = by_id[image_id]
        return {
            "crop_id": f"crop-{image_id}",
            "image_id": image_id,
            "crop_path": f"crops/{image_id}.png",
            "crop_sha256": crop_sha,
            "duplicate_group_id": f"group-{image_id}",
            "source_image_sha256": source["image_sha256"],
            "image_label": source["image_label"],
            "split_role": source["split_role"],
            "fold": next(
                value
                for key, value in {
                    "normal-fp": 0,
                    "disagree": 1,
                    "threshold": 2,
                    "attention": 3,
                    "new-style": 4,
                    "normal-extra": 0,
                    "crop-duplicate": 1,
                    "locked": -1,
                }.items()
                if key == image_id
            ),
            "source_id": source["source_id"],
            "product_id": source["product_id"],
            "template_id": source["template_id"],
            "risk_score": risk,
            "alternate_risk_score": alternate,
            "attention_score": attention,
            "style_cluster_id": f"style-{image_id}",
            "style_novelty_score": novelty,
            "style_is_unseen": unseen,
            "score_model_id": "glyph-oof-v1",
            "score_checkpoint_sha256": "a" * 64,
            "alternate_model_id": "mil-v1",
            "alternate_checkpoint_sha256": "b" * 64,
            "real_manifest_sha256": real_hash,
            "fold_manifest_sha256": fold_hash,
        }

    duplicate_hash = "d" * 64
    rows = [
        row("normal-fp", 0.95, 0.91, 0.1, 0.1, False, crop_sha="1" * 64),
        row("disagree", 0.72, 0.20, 0.2, 0.1, False, crop_sha="2" * 64),
        row("threshold", 0.51, 0.50, 0.1, 0.1, False, crop_sha="3" * 64),
        row("attention", 0.61, 0.60, 0.91, 0.1, False, crop_sha="4" * 64),
        row("new-style", 0.63, 0.60, 0.2, 0.93, True, crop_sha=duplicate_hash),
        row("normal-extra", 0.84, 0.82, 0.1, 0.1, False, crop_sha="6" * 64),
        row("crop-duplicate", 0.59, 0.58, 0.2, 0.82, True, crop_sha=duplicate_hash),
        row("locked", 0.99, 0.01, 0.99, 0.99, True, crop_sha="8" * 64),
    ]
    rows[4]["duplicate_group_id"] = "equivalent-new-style"
    rows[6].update(
        crop_sha256="7" * 64,
        duplicate_group_id="equivalent-new-style",
        attention_score=0.82,
    )
    return rows


def test_mining_covers_five_buckets_deduplicates_caps_and_never_emits_labels(
    tmp_path: Path,
) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    scores = _write_parquet(tmp_path / "scores.parquet", _score_rows(real, folds, real_rows))
    policy = MiningPolicy(
        overall_limit=10,
        per_product_cap=1,
        per_template_cap=2,
        per_source_cap=2,
        seed=7,
    )

    artifacts = mine_candidates(
        scores,
        policy,
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )

    rows = pq.read_table(artifacts.queue).to_pylist()
    reasons = {reason for row in rows for reason in row["reason_buckets"]}
    assert reasons == {
        "ABNORMAL_HIGH_ATTENTION",
        "DISAGREEMENT",
        "NEW_STYLE_CLUSTER",
        "NORMAL_FALSE_POSITIVE",
        "THRESHOLD_BAND",
    }
    assert {row["image_id"] for row in rows}.isdisjoint(
        {"locked", "normal-extra", "crop-duplicate"}
    )
    assert len({row["crop_id"] for row in rows}) == len(rows)
    assert len({row["crop_sha256"] for row in rows}) == len(rows)
    assert all(
        row["label_source"] == "mining_candidate" and row["is_gold"] is False for row in rows
    )
    forbidden = {"decision", "annotator_id", "label", "gold_path"}
    assert forbidden.isdisjoint(pq.read_schema(artifacts.queue).names)
    assert not any("gold" in path.name.lower() for path in artifacts.queue.parent.iterdir())

    metadata = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
    assert metadata["bucket_counts"]["NEW_STYLE_CLUSTER"] == 1
    assert metadata["duplicate_suppression_count"] == 1
    assert metadata["cap_suppression_counts"]["product"] == 1
    assert (
        metadata["cap_suppression_semantics"]
        == "trigger counts; one candidate may trigger multiple caps"
    )
    assert metadata["eligible_candidate_count"] == 7
    assert metadata["selected_count"] == len(rows)
    assert metadata["overall_limit_suppression_count"] == 0
    assert metadata["eligible_candidate_count"] == (
        metadata["selected_count"]
        + metadata["duplicate_suppression_count"]
        + metadata["cap_suppressed_candidate_count"]
        + metadata["overall_limit_suppression_count"]
    )
    assert metadata["production_prevalence_estimate"] is None
    assert metadata["base_rate_context"] == 0.001
    assert (
        metadata["sampling_warning"]
        == "mined candidate yield is not a production prevalence estimate"
    )
    assert metadata["output_queue_sha256"] == _sha(artifacts.queue)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate crop_id"),
        (lambda rows: rows[0].update(risk_score=float("nan")), "finite"),
        (lambda rows: rows[0].update(crop_path="../escape.png"), "relative"),
        (lambda rows: rows[0].update(real_manifest_sha256="f" * 64), "manifest hash"),
        (lambda rows: rows[0].update(image_id="unknown"), "unknown image_id"),
        (lambda rows: rows[0].update(source_id=""), "source_id"),
        (
            lambda rows: rows[0].update(
                risk_score=0.1,
                alternate_risk_score=0.1,
                score_model_id="",
            ),
            "score_model_id",
        ),
        (
            lambda rows: rows[1].update(
                alternate_model_id=rows[1]["score_model_id"],
                alternate_checkpoint_sha256=rows[1]["score_checkpoint_sha256"],
            ),
            "independent",
        ),
    ],
)
def test_mining_fails_closed_on_untrusted_score_rows(
    tmp_path: Path, mutation: object, message: str
) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    rows = _score_rows(real, folds, real_rows)
    mutation(rows)  # type: ignore[operator]
    scores = _write_parquet(tmp_path / "scores.parquet", rows)

    with pytest.raises(ValueError, match=message):
        mine_candidates(
            scores,
            MiningPolicy(),
            real_manifest=real,
            fold_manifest=folds,
            output_dir=tmp_path / "mined",
        )


@pytest.mark.parametrize("bad_fold", [True, 99, None])
def test_mining_requires_exact_contiguous_development_folds(
    tmp_path: Path, bad_fold: object
) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    fold_rows = pq.read_table(folds).to_pylist()
    fold_rows[0]["fold"] = bad_fold
    folds = _write_jsonl(tmp_path / "folds.jsonl", fold_rows)
    scores = _write_parquet(tmp_path / "scores.parquet", _score_rows(real, folds, real_rows))

    with pytest.raises(ValueError, match="fold"):
        mine_candidates(
            scores,
            MiningPolicy(fold_count=5),
            real_manifest=real,
            fold_manifest=folds,
            output_dir=tmp_path / "mined",
        )


def test_mining_keeps_distinct_crops_from_one_image_and_deduplicates_by_global_priority(
    tmp_path: Path,
) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    rows = _score_rows(real, folds, real_rows)
    second_crop = dict(rows[0])
    second_crop.update(
        crop_id="crop-normal-fp-second",
        crop_path="crops/normal-fp-second.png",
        crop_sha256="9" * 64,
        duplicate_group_id="group-normal-fp-second",
        risk_score=0.94,
    )
    rows.append(second_crop)
    scores = _write_parquet(tmp_path / "scores.parquet", rows)

    artifacts = mine_candidates(
        scores,
        MiningPolicy(
            overall_limit=20,
            per_product_cap=20,
            per_template_cap=20,
            per_source_cap=20,
        ),
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )

    mined = pq.read_table(artifacts.queue).to_pylist()
    ids = {row["crop_id"] for row in mined}
    assert {"crop-normal-fp", "crop-normal-fp-second"}.issubset(ids)
    assert "crop-new-style" in ids
    assert "crop-crop-duplicate" not in ids


def test_mining_publish_is_idempotent_but_refuses_overwrite(tmp_path: Path) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    scores = _write_parquet(tmp_path / "scores.parquet", _score_rows(real, folds, real_rows))
    first = mine_candidates(
        scores,
        MiningPolicy(),
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )
    second = mine_candidates(
        scores,
        MiningPolicy(),
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )
    assert first == second

    changed_scores = _write_parquet(
        tmp_path / "changed.parquet", _score_rows(real, folds, real_rows)[:-1]
    )
    with pytest.raises(ValueError, match="immutable mining artifact"):
        mine_candidates(
            changed_scores,
            MiningPolicy(),
            real_manifest=real,
            fold_manifest=folds,
            output_dir=tmp_path / "mined",
        )


def _review_cycle(
    tmp_path: Path,
    candidate_rows: list[dict[str, object]],
    labels: list[tuple[str, str]],
) -> tuple[Path, Path, Path, Path]:
    crop_root = tmp_path / "review-crops"
    crop_manifest_rows: list[dict[str, object]] = []
    queue: list[ReviewCandidate] = []
    by_id = {str(row["crop_id"]): row for row in candidate_rows}
    for crop_id, _ in labels:
        row = by_id[crop_id]
        path = crop_root / str(row["crop_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(path)
        crop_manifest_rows.append(
            {
                "crop_id": crop_id,
                "crop_path": row["crop_path"],
                "source_image_sha256": row["source_image_sha256"],
                "crop_sha256": row["crop_sha256"],
            }
        )
        queue.append(
            ReviewCandidate(
                crop_id=crop_id,
                image_id=str(row["image_id"]),
                crop_path=str(row["crop_path"]),
                risk_score=float(row["risk_score"]),
                style_id=str(row["style_cluster_id"]),
                disagreement=abs(float(row["risk_score"]) - float(row["alternate_risk_score"])),
                score_model_id=str(row["score_model_id"]),
                score_artifact_sha256=str(row["score_checkpoint_sha256"]),
                disagreement_model_id=str(row["alternate_model_id"]),
                disagreement_artifact_sha256=str(row["alternate_checkpoint_sha256"]),
            )
        )
    crop_manifest = _write_parquet(tmp_path / "review-crops.parquet", crop_manifest_rows)
    exported = export_review_queue(queue, crop_manifest, crop_root, tmp_path / "review-queue")
    review_labels = tmp_path / "review-labels.jsonl"
    review_labels.write_text(
        "".join(
            json.dumps(
                {
                    "queue_version": exported.queue_version,
                    "crop_id": crop_id,
                    "label": decision,
                    "annotator_id": "alice",
                }
            )
            + "\n"
            for crop_id, decision in labels
        ),
        encoding="utf-8",
    )
    imported = import_review_labels(review_labels, exported.queue, tmp_path / "review-import")
    return exported.queue, review_labels, imported.gold_crops, imported.audit


def test_record_mining_yield_binds_review_queue_labels_and_gold(tmp_path: Path) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    scores = _write_parquet(tmp_path / "scores.parquet", _score_rows(real, folds, real_rows))
    mined = mine_candidates(
        scores,
        MiningPolicy(),
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )
    candidates = pq.read_table(mined.queue).to_pylist()
    review_queue, review_labels, gold, audit = _review_cycle(
        tmp_path,
        candidates,
        [(str(candidates[0]["crop_id"]), "BLOCK"), (str(candidates[1]["crop_id"]), "REVIEW")],
    )

    version = record_mining_yield(
        mined.queue,
        review_queue,
        review_labels,
        gold,
        audit,
        base_real_manifest=real,
        output_dir=tmp_path / "dataset-v2",
    )
    payload = json.loads(version.dataset_version.read_text(encoding="utf-8"))
    assert payload["counts"] == {
        "accepted_block": 1,
        "accepted_pass": 0,
        "accepted_total": 1,
        "reviewed_total": 2,
        "unresolved_review": 1,
    }
    assert payload["accepted_yield"] == pytest.approx(1 / 2)
    assert (
        payload["yield_denominator"]
        == "accepted_label_count + review_count in the trusted import audit"
    )
    assert payload["production_prevalence_estimate"] is None
    assert not hasattr(version, "gold_crops")

    assert payload["lineage"]["review_queue_jsonl_sha256"] == _sha(review_queue / "queue.jsonl")
    assert payload["lineage"]["review_labels_sha256"] == _sha(review_labels)


def test_record_mining_yield_rejects_review_outside_mining_queue(tmp_path: Path) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    scores = _write_parquet(tmp_path / "scores.parquet", _score_rows(real, folds, real_rows))
    mined = mine_candidates(
        scores,
        MiningPolicy(),
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )
    candidates = pq.read_table(mined.queue).to_pylist()
    outside = dict(candidates[0])
    outside.update(
        crop_id="outside-mining",
        crop_path="crops/outside.png",
        crop_sha256="e" * 64,
        duplicate_group_id="outside-group",
    )
    review_queue, review_labels, gold, audit = _review_cycle(
        tmp_path,
        [candidates[0], outside],
        [(str(candidates[0]["crop_id"]), "PASS"), ("outside-mining", "REVIEW")],
    )

    with pytest.raises(ValueError, match="outside mining queue"):
        record_mining_yield(
            mined.queue,
            review_queue,
            review_labels,
            gold,
            audit,
            base_real_manifest=real,
            output_dir=tmp_path / "rejected",
        )


@pytest.mark.parametrize(
    "tamper",
    ["audit-queue-version", "labels-sha", "queue-priority", "gold-queue-hash"],
)
def test_record_mining_yield_rejects_tampered_review_lineage(tmp_path: Path, tamper: str) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    scores = _write_parquet(tmp_path / "scores.parquet", _score_rows(real, folds, real_rows))
    mined = mine_candidates(
        scores,
        MiningPolicy(),
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )
    candidates = pq.read_table(mined.queue).to_pylist()
    review_queue, review_labels, gold, audit = _review_cycle(
        tmp_path, candidates, [(str(candidates[0]["crop_id"]), "PASS")]
    )
    if tamper in {"audit-queue-version", "labels-sha"}:
        payload = json.loads(audit.read_text(encoding="utf-8"))
        field = "queue_version" if tamper == "audit-queue-version" else "labels_sha256"
        payload[field] = "f" * 64
        audit.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "queue-priority":
        queue_jsonl = review_queue / "queue.jsonl"
        row = json.loads(queue_jsonl.read_text(encoding="utf-8"))
        row["priority_rank"] = 9
        queue_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    else:
        gold_rows = pq.read_table(gold).to_pylist()
        gold_rows[0]["queue_manifest_sha256"] = "f" * 64
        _write_parquet(gold, gold_rows)

    with pytest.raises(ValueError):
        record_mining_yield(
            mined.queue,
            review_queue,
            review_labels,
            gold,
            audit,
            base_real_manifest=real,
            output_dir=tmp_path / "rejected",
        )


def test_record_mining_yield_allows_accepted_label_already_in_previous_gold(
    tmp_path: Path,
) -> None:
    real, folds, real_rows = _manifests(tmp_path)
    scores = _write_parquet(tmp_path / "scores.parquet", _score_rows(real, folds, real_rows))
    mined = mine_candidates(
        scores,
        MiningPolicy(),
        real_manifest=real,
        fold_manifest=folds,
        output_dir=tmp_path / "mined",
    )
    candidates = pq.read_table(mined.queue).to_pylist()
    review_queue, review_labels, previous_gold, _ = _review_cycle(
        tmp_path, candidates, [(str(candidates[0]["crop_id"]), "PASS")]
    )
    imported = import_review_labels(
        review_labels,
        review_queue,
        tmp_path / "review-import-repeat",
        existing_gold=previous_gold,
    )

    version = record_mining_yield(
        mined.queue,
        review_queue,
        review_labels,
        imported.gold_crops,
        imported.audit,
        base_real_manifest=real,
        previous_gold_manifest=previous_gold,
        output_dir=tmp_path / "dataset-v2",
    )

    payload = json.loads(version.dataset_version.read_text(encoding="utf-8"))
    assert payload["counts"]["accepted_pass"] == 1
    assert payload["counts"]["accepted_total"] == 1
