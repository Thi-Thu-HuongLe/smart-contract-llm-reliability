# Smart-Contract LLM Reliability Benchmark

This lightweight repository contains the inference, validation, selective
output recovery, evaluation, and statistical analysis programs used in a
reliability study of frozen code language models for smart-contract
vulnerability detection.

## Study scope

- Benchmarks: SmartBugs Curated, ScrawlD, and BCCC.
- Frozen models: Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct, and
  Mistral-7B-Instruct-v0.3.
- Controls: pinned model snapshots, native chat templates, deterministic
  decoding, closed vulnerability taxonomies, JSON validation, and ordered
  record manifests.
- Analyses: aggregate and per-class evaluation, repair sensitivity, paired
  uncertainty, cross-fitted TACE decisions, and recovery baselines.

Figure-generation programs, manuscript sources, analysis-ready datasets, and
large raw predictions are intentionally excluded from the GitHub release.

## Programs

### Inference and validation

- `run_frozen_llm_inference.py`: deterministic frozen-model inference.
- `inspect_prediction_progress.py`: model-dataset coverage and resume audit.
- `repair_invalid_outputs.py`: selective recovery of invalid or incomplete
  structured outputs without modifying valid records.
- `recover_qwen_bccc_predictions.py`: sharded Qwen-BCCC recovery and merge.

### Evaluation and analysis

- `evaluate_predictions.py`: aggregate, per-class, and coverage tables.
- `analyze_prediction_robustness.py`: repair sensitivity, paired uncertainty,
  and support-aware results.
- `analyze_tace_crossfit.py`: leakage-controlled cross-fitted TACE analysis.
- `analyze_recovery_baselines.py`: recovery-baseline comparison.
- `summarize_repair_exposure.py`: record-level repair exposure summary.

## Environment setup

For CPU-only statistical analysis:

```bash
python -m pip install -r requirements-analysis.txt
```

For inference, retain the server's CUDA-enabled PyTorch installation and add:

```bash
python -m pip install -r requirements-inference.txt
```

## Required external files

The analysis-ready datasets and nine raw JSONL prediction files are not
tracked by Git. Obtain them from the associated research archive, verify the
published SHA-256 checksums, and place them in the layout documented by
`data/README.md` and `results/README.md`.

The GitHub release includes the smaller cross-fitted TACE result tables under
`results/final_benchmark/tace_analysis/`.

## Reproduce the analyses

After restoring the external files, run from the repository root:

```bash
python inspect_prediction_progress.py \
  --prediction-root results/final_benchmark

python evaluate_predictions.py \
  --prediction-root results/final_benchmark

python analyze_prediction_robustness.py
python analyze_tace_crossfit.py
python analyze_recovery_baselines.py
```

Use `python run_frozen_llm_inference.py --help` to inspect GPU inference
options. A complete inference run requires access to the pinned model
snapshots and sufficient GPU memory.

## Source portability

All Python source files in this repository use ASCII characters only and have
no UTF-8 byte-order mark. The raw JSONL files remain immutable and are
identified by `results/RAW_PREDICTION_SHA256SUMS.txt`.

No synthetic oversampling, smoke-run prediction, abandoned model, or simulated
human review is part of the retained results.
