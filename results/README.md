# Result organization

`final_benchmark/` is the result root expected by the analysis programs. The
lightweight GitHub release retains only the smaller `tace_analysis/` tables.
The remaining directories must be restored from the associated research
archive before reproducing every analysis.

- `smartbugs/`, `scrawld/`, and `bccc/` contain the nine frozen JSONL model
  prediction files and their metadata when restored locally.
- `benchmark_evaluation/` contains aggregate, per-class, and coverage tables.
- `robustness_analysis/` contains repair-sensitivity, uncertainty, and
  support-aware analyses.
- `tace_analysis/` contains cross-fitted TACE predictions and decisions.
- `recovery_baseline_analysis/` contains recovery-baseline comparisons.

Raw prediction files are excluded from Git because some exceed
GitHub's single-file limit. Publish them in a DOI-backed research repository
and record the archive URL, release identifier, and checksums in the associated
GitHub release. The smaller derived result tables can remain under version
control. Their presence in a local working copy does not mean Git will stage
them; verify this with `git status` before committing.

`RAW_PREDICTION_SHA256SUMS.txt` identifies the nine immutable prediction files
used for the reported analyses.
