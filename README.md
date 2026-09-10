# Smart-Contract LLM Reliability Benchmark

This repository supports a reliability study of frozen code language models
for smart-contract vulnerability detection. It contains the inference,
validation, selective-output recovery, evaluation, and statistical-analysis
programs used in the study.

## Study scope

- Benchmarks: SmartBugs Curated, ScrawlD, and BCCC.
- Frozen models: Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct, and
  Mistral-7B-Instruct-v0.3.
- Main controls: pinned model snapshots, native chat templates, deterministic
  decoding, a closed vulnerability taxonomy, JSON-schema validation, and
  record-level manifests.
- Analyses: per-class and aggregate evaluation, repair-sensitivity analysis,
  paired uncertainty estimates, cross-fitted TACE decisions, and recovery
  baselines.

## Repository layout

```text
smart-contract-llm-reliability/
|-- data/                         Input-file documentation and local datasets
|-- results/
|   `-- final_benchmark/
|       |-- smartbugs/            Frozen SmartBugs predictions
|       |-- scrawld/              Frozen ScrawlD predictions
|       |-- bccc/                 Frozen BCCC predictions
|       |-- benchmark_evaluation/ Aggregate and per-class metrics
|       |-- robustness_analysis/  Sensitivity and paired analyses
|       |-- tace_analysis/        Cross-fitted TACE results
|       `-- recovery_baseline_analysis/
|-- run_frozen_llm_inference.py
|-- repair_invalid_outputs.py
|-- evaluate_predictions.py
|-- analyze_tace_crossfit.py
|-- analyze_source_family_robustness.py
`-- analyze_recovery_baselines.py
```


## Main programs

### Inference and validation

- `run_frozen_llm_inference.py`: run deterministic frozen-model inference.
- `inspect_prediction_progress.py`: audit model--dataset coverage and resume
  safety.
- `repair_invalid_outputs.py`: selectively recover invalid or incomplete
  structured outputs without modifying already-valid records.
- `recover_qwen_bccc_predictions.py`: shard and merge the dedicated
  Qwen--BCCC recovery run.

### Evaluation and statistical analysis

- `evaluate_predictions.py`: produce aggregate, per-class, and coverage
  tables.
- `analyze_prediction_robustness.py`: quantify repair exposure, sensitivity,
  paired uncertainty, and support-aware results.
- `analyze_tace_crossfit.py`: run leakage-controlled cross-fitted TACE
  calibration and decision analysis.
- `analyze_recovery_baselines.py`: compare recovery strategies.
- `summarize_repair_exposure.py`: summarize which records required recovery.
- `analyze_source_family_robustness.py`: audit exact/lexical source overlap and
  rerun TACE with source-family-grouped outer and inner folds.


## Environment setup

Keep the server's CUDA-enabled PyTorch installation, then install the matching
dependency group:

```bash
python -m pip install -r requirements-inference.txt
python -m pip install -r requirements-analysis.txt
```

## Reproduce the retained analyses

Run these commands from this directory after placing the documented input and
prediction files in their expected locations:

```bash
python inspect_prediction_progress.py \
  --prediction-root results/final_benchmark

python evaluate_predictions.py \
  --prediction-root results/final_benchmark

python analyze_prediction_robustness.py
python analyze_tace_crossfit.py
python analyze_recovery_baselines.py
python summarize_repair_exposure.py \
  --prediction-root results/final_benchmark \
  --output results/final_benchmark/robustness_analysis/recovery_path_summary.csv \
  --record-output results/final_benchmark/robustness_analysis/recovery_path_records.csv
python analyze_source_family_robustness.py
```

Run the regression tests before publishing changes:

```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
```

The recovery-path audit distinguishes first response, format retry,
deterministic structural normalization, and targeted re-inference. The TACE
analysis additionally writes per-class calibration, positive-alert selective
metrics, BCCC confusion matrices, and a nested shrinkage sensitivity table.
Paired intervals based on retained OOF predictions are explicitly marked as
conditional; they must not be described as full-pipeline bootstrap intervals.

## Controlled prompt sensitivity

The inference program supports `canonical`, `concise`, and
`evidence_checklist` prompts. Run each variant into a separate output root and
on the same predeclared record subset. Non-canonical filenames include the
variant name, preventing accidental overwrite of canonical predictions.

```bash
python run_frozen_llm_inference.py \
  --models <repository@full_commit_sha> \
  --datasets smartbugs scrawld bccc \
  --max-records 100 \
  --prompt-variant evidence_checklist \
  --output-root results/prompt_sensitivity/evidence_checklist
```

The default inference command is intentionally omitted because a complete run
requires selecting available GPUs and confirming access to the pinned model
snapshots. Use `python run_frozen_llm_inference.py --help` to inspect all
runtime options.
