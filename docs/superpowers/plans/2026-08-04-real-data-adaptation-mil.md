# Real-data Adaptation, Quick Review, MIL, and Hard-negative Mining Plan

> Execute with `superpowers:executing-plans`. Use test-driven development and one
> `feature:`/`fix:`/`refactor:` commit per task.

**Goal:** Turn approximately 100 real abnormal/normal seed images plus larger image-level
weak labels into leakage-safe character supervision, an adapted glyph scorer, image-level
MIL evidence, and repeatable hard-negative review queues without ever promoting unreviewed
pseudo-labels to gold data.

**Architecture:** Real data is imported into a versioned immutable Parquet manifest with
image, group, split-role, license, and optional character annotations. Group-aware five-fold
splits and perceptual near-duplicate components prevent template/product leakage. Reviewed
character crops adapt the existing glyph encoder; image-label-only bags train a separate
attention MIL head. Mining writes candidates to a review queue, never back into gold labels.
All outputs carry source/model/manifest hashes. The locked 20-image character test set is
read-only to training code.

**Constraints:**

- Expected real budget: 60 development abnormal, 20 locked-test abnormal, and 20 difficult
  normal images; counts are audited rather than fabricated when the available set differs.
- Existing image labels mean normal=`all visible text normal`, abnormal=`at least one region
  abnormal`; an abnormal bag does not identify which character is bad.
- Quick labels are exactly `PASS`, `BLOCK`, or `REVIEW`. Only human-confirmed PASS/BLOCK
  become character gold labels.
- Character crops require real coordinates from reviewed annotations or an OCR capability
  that explicitly reports character boxes. Line-level PP-OCRv5 boxes must not be silently
  split into fake character boxes.
- Same product, campaign, template, source group, and perceptual near-duplicate component
  stay in one fold.
- Thresholds remain calibrated on real out-of-fold predictions and large normal replay;
  synthetic or balanced precision is never reported as production precision.
- This phase does not implement the sequence language model, final fusion policy, or API.

---

## Task 1: Add the validated real-data contract and immutable importer

**Files:**

- Create `src/poor_word/real_data/__init__.py`
- Create `src/poor_word/real_data/schema.py`
- Create `src/poor_word/real_data/ingest.py`
- Modify `src/poor_word/cli.py`
- Create `tests/real_data/test_schema.py`
- Create `tests/real_data/test_ingest.py`
- Modify `README.md`

**Interfaces:**

- JSONL input records contain `image_id`, `image_path`, `image_label`, `split_role`,
  `source_id`, `source_group_id`, `product_id`, `campaign_id`, `template_id`, and optional
  character records. `source_id` identifies licensing provenance; `source_group_id` is the
  leakage-control batch/generator/feed identity.
- `CharacterAnnotation` contains a positive-area box, `PASS|BLOCK|REVIEW`, optional text,
  anomaly kind, and annotator provenance.
- `import_real_dataset(input_jsonl, output_dir) -> RealDatasetArtifacts` writes
  `manifest.parquet`, `dataset.json`, and `validation.json` atomically.
- CLI: `poor-word real-data import --input ... --output-dir ...`.

**TDD steps:**

1. Test schema rejection for duplicate image IDs, invalid boxes, missing grouping keys,
   abnormal characters in a normal image, and character gold labels without annotator IDs.
2. Test that the importer decodes every image, hashes its bytes, normalizes paths relative
   to the dataset root, preserves image-only abnormal labels, and writes deterministic
   Parquet rows.
3. Test that locked-test records cannot be marked as training eligible and that missing
   licenses or production decisions fail closed.
4. Implement frozen Pydantic schemas, atomic writes, dataset statistics, source hashes, and
   validation findings. Do not copy source images.
5. Add the CLI and exact example manifest to README.
6. Run `pytest tests/real_data/test_schema.py tests/real_data/test_ingest.py -q`, Ruff, and
   Mypy; commit `feature: import validated real seed data`.

## Task 2: Build leakage-safe group folds and near-duplicate components

**Files:**

- Create `src/poor_word/real_data/split.py`
- Create `tests/real_data/test_split.py`
- Modify `src/poor_word/cli.py`
- Modify `README.md`

**Interfaces:**

- `assign_group_folds(manifest, folds=5, seed=20260804) -> FoldArtifacts`.
- CLI: `poor-word real-data split --manifest ... --folds 5 --output ...`.

**TDD steps:**

1. Test that exact SHA duplicates and perceptual-hash neighbors form connected components.
2. Test that product/campaign/template/source/duplicate relationships are transitively
   closed and never cross folds.
3. Test deterministic stratified greedy assignment by image label and split role; locked
   test remains `fold=-1` and is never placed in development folds.
4. Implement 64-bit pHash, union-find grouping, deterministic fold balancing, leakage audit,
   and `folds.parquet` plus `split-audit.json`.
5. Run targeted and full CPU tests; commit `feature: assign leakage-safe real data folds`.

## Task 3: Add character crop extraction and a human quick-review queue

**Files:**

- Create `src/poor_word/real_data/crops.py`
- Create `src/poor_word/real_data/review.py`
- Create `tests/real_data/test_crops.py`
- Create `tests/real_data/test_review.py`
- Modify `src/poor_word/cli.py`
- Modify `README.md`

**Interfaces:**

- `extract_character_crops(...)` accepts reviewed character boxes or audited OCR character
  boxes only and writes immutable crop PNGs plus `crops.parquet`.
- `build_review_queue(scores, disagreements, limit, seed)` prioritizes model disagreement,
  near-threshold normal crops, and under-covered styles.
- CLI commands: `real-data crops`, `review export`, and `review import`.

**TDD steps:**

1. Test clamped crop geometry, padding, image hash linkage, and byte-identical reruns.
2. Test explicit refusal when OCR audit says character boxes unavailable; never equal-split
   line boxes.
3. Test queue deduplication, deterministic priority ordering, label vocabulary, and that
   REVIEW remains unresolved.
4. Test optimistic concurrency: imported labels must reference the exported queue version
   and cannot overwrite a different annotator's accepted gold label.
5. Implement CSV/JSONL review bundles, contact-sheet thumbnails, import validation, and
   reviewed `gold-crops.parquet` lineage.
6. Run tests and commit `feature: build character quick-review workflow`.

## Task 4: Add unlabeled real-crop self-supervised adaptation

**Files:**

- Create `src/poor_word/training/adapt_real.py`
- Create `tests/training/test_adapt_real.py`
- Modify `src/poor_word/cli.py`
- Modify `README.md`

**Interfaces:**

- `AdaptConfig` and `adapt_real_encoder(config) -> AdaptArtifacts`.
- CLI: `poor-word train adapt-real`.

**TDD steps:**

1. Write a two-step CPU smoke test that loads the prior encoder, uses all eligible unlabeled
   real crops, and never reads locked-test rows.
2. Implement two-view contrastive adaptation with mild photometric/style transforms only;
   do not perform geometry that changes character identity.
3. Add collapse guards, deterministic seeds, checkpoint-parent hashes, crop-manifest hashes,
   loss history, and held-out embedding-drift diagnostics.
4. Add the L20 command and document that the adapted checkpoint is not calibrated for BLOCK.
5. Run tests and commit `feature: adapt glyph encoder to unlabeled real crops`.

## Task 5: Fine-tune with reviewed crops and generate five-fold OOF scores

**Files:**

- Create `src/poor_word/training/finetune_real.py`
- Create `src/poor_word/evaluation/oof.py`
- Create `tests/training/test_finetune_real.py`
- Create `tests/evaluation/test_oof.py`
- Modify `src/poor_word/cli.py`

**Interfaces:**

- `finetune_real_fold(config, held_out_fold) -> FoldModelArtifacts`.
- `collect_oof_scores(fold_artifacts, fold_manifest) -> oof.parquet`.
- CLI: `poor-word train real-oof`.

**TDD steps:**

1. Test fold exclusion, parent-checkpoint provenance, and locked-test isolation.
2. Fine-tune with synthetic replay plus reviewed real PASS/BLOCK; REVIEW has zero supervised
   loss. Use weighted sampling without duplicating metric rows.
3. Produce one and only one OOF score per development character and reject model/fold hash
   mismatches.
4. Report per-fold counts, AUCPR, anomaly-kind recall, and bootstrap intervals without using
   the locked test for model selection.
5. Run tests and commit `feature: generate real character oof scores`.

## Task 6: Train an image-level weakly supervised MIL head

**Files:**

- Create `src/poor_word/models/mil.py`
- Create `src/poor_word/training/train_mil.py`
- Create `tests/models/test_mil.py`
- Create `tests/training/test_train_mil.py`
- Modify `src/poor_word/cli.py`

**Interfaces:**

- `AttentionMilPool` maps variable-length character evidence to an image risk logit and
  returns normalized instance attention for inspection.
- CLI: `poor-word train mil`.

**TDD steps:**

1. Test padding masks, zero-character bags, attention normalization, permutation invariance,
   and monotonic max-evidence fallback.
2. Build bags only from image labels and OOF/held-out character features. Normal bags apply
   all-instance negative pressure; abnormal bags apply positive bag loss only.
3. Test that the highest-attention instance is never written as a gold label.
4. Add group-fold training, class-balanced loss, early stopping, and artifact hashes.
5. Run tests and commit `feature: train weakly supervised image mil head`.

## Task 7: Mine hard negatives and uncertain abnormal candidates safely

**Files:**

- Create `src/poor_word/real_data/mining.py`
- Create `tests/real_data/test_mining.py`
- Modify `src/poor_word/cli.py`
- Modify `README.md`

**Interfaces:**

- `mine_candidates(scores, policy) -> MiningArtifacts`.
- CLI: `poor-word real-data mine`.

**TDD steps:**

1. Test policy buckets: normal false positives, model disagreement, threshold band, abnormal
   bag high-attention, new font/style cluster, and duplicate suppression.
2. Cap candidates per product/template/source so one campaign cannot dominate review.
3. Write only queue candidates with model/data hashes; prohibit direct writes to gold.
4. Measure yield after review and generate the next immutable dataset version.
5. Run tests and commit `feature: mine real glyph hard examples`.

## Task 8: Compare OCR/rule baselines and emit the real-seed phase report

**Files:**

- Create `src/poor_word/evaluation/baselines.py`
- Create `src/poor_word/evaluation/real_report.py`
- Create `tests/evaluation/test_baselines.py`
- Create `tests/evaluation/test_real_report.py`
- Modify `src/poor_word/cli.py`
- Modify `README.md`

**Interfaces:**

- Baselines: OCR confidence, Unicode/3,500-character membership, and OCR-plus-rule score.
- CLI: `poor-word evaluate real-seed`.

**TDD steps:**

1. Test that rare/out-of-catalog Chinese never auto-BLOCKs without visual anomaly evidence.
2. Test image-level OOF metrics, character metrics, raw counts, review rate, anomaly-type
   recall, Wilson intervals, and production-base-rate PPV.
3. Compare the glyph/MIL system against OCR/rule baselines on identical folds. Mark the MVP
   comparison inconclusive when real labels or normal replay are insufficient.
4. Include all dataset/model/OCR/source hashes, locked-test access audit, failures, exact L20
   commands, and a `Not a pilot approval` heading until the normal replay and locked test run.
5. Run the full CPU quality gate and an L20 command audit; commit
   `feature: report real seed adaptation metrics`.

## Phase completion gate

Run:

```bash
uv lock --check
uv run pytest -m "not gpu" --cov=poor_word --cov-report=term-missing --cov-fail-under=85
uv run ruff check .
uv run mypy src
git status --short
```

The phase is complete only when every real asset has validated source and label provenance,
all development predictions are out-of-fold, locked-test access is auditable, near duplicates
cannot cross folds, unreviewed pseudo-labels never enter gold data, and the report remains
inconclusive rather than guessing when the real/normal sample counts are below the approved
gate requirements.
