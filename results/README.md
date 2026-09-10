# Result organization

`final_benchmark/` is the result root used by the analysis programs.

- `smartbugs/`, `scrawld/`, and `bccc/` contain the nine frozen JSONL model
  prediction files and their metadata.
- `benchmark_evaluation/` contains aggregate, per-class, and coverage tables.
- `robustness_analysis/` contains repair-sensitivity, uncertainty, and
  support-aware analyses.
- `tace_analysis/` contains summary metrics, calibration tables, and selective
  decision results.
- `recovery_baseline_analysis/` contains recovery-baseline comparisons.
- `source_family_analysis/` contains source-overlap and grouped-fold summaries.

Raw prediction files are excluded from ordinary Git because some exceed
GitHub's single-file limit. Publish them in a DOI-backed research repository
and record the archive URL, release identifier, and checksums in the associated
GitHub release. Compact aggregate and per-class tables remain under version
control so that the values reported in the paper can be inspected directly.
Row-level out-of-fold predictions, fold assignments, and recovery-path records
are intentionally excluded from GitHub and should be distributed with the
DOI-backed research-data deposit.

`RAW_PREDICTION_SHA256SUMS.txt` identifies the nine immutable prediction files
used for the reported analyses.
