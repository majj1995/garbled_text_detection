# Malformed Chinese Glyph MVP Report

## Not a production claim

Synthetic results measure controlled corruptions. They do not establish production performance on AIGC traffic.

## Production prevalence

- Assumed anomaly prevalence: 0.100000%
- Precision below is recalculated from recall and FPR at that prevalence.
- Evaluation-set class balance is not used as the production base rate.

## Synthetic split metrics

- Positive samples: 39
- Negative samples: 10
- Synthetic AUROC: 1.000000
- Synthetic AUCPR: 1.000000

### Offline MVP: INCONCLUSIVE

- Threshold: `0.0013609529`
- Raw counts: TP=39, FP=0, TN=10, FN=0
- Recall: 100.000000%
- FPR: 0.000000%
- Production-base-rate precision: 100.000000%
- Negative sample support: 10/10000

### Pilot trial: INCONCLUSIVE

- Threshold: `0.0013609529`
- Raw counts: TP=39, FP=0, TN=10, FN=0
- Recall: 100.000000%
- FPR: 0.000000%
- Production-base-rate precision: 100.000000%
- Negative sample support: 10/40000

## Dataset hashes

- `generation_run`: `def6bc140c0f1ae283dc4049c81f441d36d0b64225824867c5b00173a720e97e`
- `synthetic_manifest`: `406a06d2815cb718e0af1bcc419e98098c41bf5f627f1b27a071e7c2f634757e`

## Asset hashes

- `common_chars_3500`: `4aac5efd4e149898e00829b16394e7c74aa2d78b7382ed2a069cbf69bf188858`
- `noto_sans_sc_regular`: `2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b`

## Model hashes

- `encoder.pt`: `81b2a52d7d6d871ceddad512f2023fda2a2754e361ccb9ce5750b7cc20dcf730`
- `metrics.json`: `a97e9355d660fa613dd828dd2345bfa98dac53ee38feb8a331fa0c66b04a8aba`
- `prototypes.json`: `afc13f3a551c63424ddc66b3468c3e274bc9ee293a0adc3d9fac234592798a9a`
- `prototypes.npz`: `cc2f9f1d9c0d7c58429157534353861530e682926a1434dcf5a3b40b89749f84`

## Source license decisions

- `common_chars_3500`: `Apache-2.0; production_allowed=true`
- `noto_sans_sc_regular`: `OFL-1.1; production_allowed=true`

## PP-OCRv5 L20 capabilities

- `audit_status`: `not_supplied`

## Latency

- `offline_cpu_ms_per_glyph`: `14.199958121099947`

## Known failures and capability gaps

- PP-OCRv5 L20 audit was not supplied to this report
- Only 10 synthetic normal samples: insufficient empirical resolution for the 0.01% MVP FPR gate
- Only 10 synthetic normal samples: insufficient empirical resolution for the 0.0025% pilot FPR gate
- This command evaluates the supplied synthetic manifest; use a separately versioned holdout and real seed set before making deployment claims

## Reproduction commands

```bash
uv run poor-word evaluate glyph --manifest data/generated/smoke-a/manifest.parquet --artifacts artifacts/train-smoke --prevalence 0.001 --device cpu --batch-size 16 --output-dir artifacts/report-smoke
```
