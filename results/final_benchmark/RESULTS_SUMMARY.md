# Reliability analysis results

Generated on 2026-09-10 from the nine retained model--dataset JSONL files.
The prediction files were read only. All nine runs contain their full expected
record counts and have zero final non-OK records.

## Recovery-path audit

The previous selective-repair count did not include format retries performed
during initial inference. The revised audit includes every stage.

| Dataset | Model | Native-valid records | Recovery-exposed records | Changed label set among exposed |
|---|---|---:|---:|---:|
| SmartBugs-Curated | Qwen | 85.31% | 14.69% | 71.43% |
| SmartBugs-Curated | CodeLlama | 32.87% | 67.13% | 97.92% |
| SmartBugs-Curated | Mistral | 80.42% | 19.58% | 100.00% |
| ScrawlD | Qwen | 64.67% | 35.33% | 19.64% |
| ScrawlD | CodeLlama | 13.51% | 86.49% | 90.73% |
| ScrawlD | Mistral | 3.71% | 96.29% | 72.24% |
| BCCC | Qwen | 65.82% | 34.18% | 16.49% |
| BCCC | CodeLlama | 9.70% | 90.30% | 96.19% |
| BCCC | Mistral | 3.92% | 96.08% | 84.54% |

The pre-specified recovery-sensitivity criterion fails for all three datasets.
The final model results must therefore be described as conditional on the
documented recovery pipeline, not as recovery-independent native performance.

## TACE and independent recovery baselines

| Dataset | Metric | TACE-F1 | Logistic stacker | Prevalence only | Family-grouped TACE |
|---|---|---:|---:|---:|---:|
| SmartBugs-Curated | Micro-F1 | 0.9458 | 0.9496 | 0.3869 | 0.9493 |
| ScrawlD | Micro-F1 | 0.5985 | 0.5986 | 0.5978 | 0.6018 |
| BCCC | Balanced accuracy | 0.5200 | 0.5202 | 0.5000 | 0.5200 |

TACE is clearly above the prevalence-only control on SmartBugs-Curated and
BCCC. On ScrawlD, the TACE--prevalence difference is 0.0007 with a conditional
95% paired interval of [-0.0075, 0.0088]. TACE does not significantly
outperform logistic stacking on any dataset under the conditional paired
analysis. Consequently, TACE should be presented as an interpretable
class-aware reliability layer, not as a universally superior ensemble.

## Shrinkage sensitivity

ScrawlD is unchanged across pseudo-count values 2, 4, 8, 16, and 32. BCCC
changes by at most 0.0002. SmartBugs-Curated is more sensitive: relative to
the configured value 8, Micro-F1 changes by +0.0104 at 4, -0.0170 at 16, and
-0.1119 at 32. This dataset-specific sensitivity must be reported.

## Selective-decision utility

At confidence 0.80, SmartBugs-Curated retains 95.87% decision coverage with
0.88% accepted-decision risk; its 76 accepted positive alerts have precision
1.00 and recover 53.15% of ground-truth positive class decisions. ScrawlD
retains 72.79% decision coverage but accepts only five positive alerts. BCCC
accepts no decisions at confidence 0.80 or above. The selective policy is
therefore useful for high-confidence SmartBugs decisions but does not provide
general positive-alert utility across all benchmarks.

## Source-family audit

Within-dataset exact duplication is small: none in SmartBugs-Curated, two
multi-record families in ScrawlD, and four in BCCC. Across datasets, however,
ScrawlD and BCCC share 1,996 exact-source families; SmartBugs-Curated has no
exact-source overlap with either dataset. ScrawlD and BCCC must not be
described as fully independent benchmark replications.

The lexical-normalized grouping is a conservative stress test, not a semantic
clone detector. Grouped outer and inner folds keep each lexical family intact.
The grouped results in the table above are close to the record-stratified
results, which supports stability of the TACE operating point under this
conservative grouping.

## Uncertainty scope

The paired bootstrap intervals condition on the retained OOF predictions.
They do not refit calibration tables, thresholds, and folds inside every
bootstrap sample. Files record this scope explicitly and the manuscript must
not call these intervals full-pipeline uncertainty estimates.

## Output map

- `robustness_analysis/recovery_path_summary.csv`: aggregate recovery-path
  exposure and prediction-change audit. The row-level audit is reserved for
  the DOI-backed research-data deposit.
- `robustness_analysis/`: recovery sensitivity, paired model comparisons, and
  support-aware metrics.
- `tace_analysis/`: OOF metrics, calibration, selective decisions, BCCC confusion
  matrix, threshold selection, shrinkage sensitivity, and conditional CIs.
- `recovery_baseline_analysis/`: prevalence-only and logistic-stacking comparisons,
  including per-class decisions and calibration.
- `source_family_analysis/`: exact/lexical overlap audit and source-family-grouped TACE.

Prompt-variant inference and genuine manual validation are not part of this
CPU-only result set and must be completed separately if retained in the
revision plan.
