#!/usr/bin/env python3
"""Leakage-controlled recovery baselines for the TACE analysis.

This script uses the existing frozen model predictions and the exact outer-fold
assignments from the TACE evaluation.  It performs no model inference.  Two
label-informed comparators are evaluated:

1. Prevalence-only: a Beta(1, 1)-smoothed class prevalence estimated from the
   training partition, with no model-output features.
2. Logistic stacking: one L2-regularized logistic model per taxonomy class,
   using the three binary model votes as features.

For both systems, operating thresholds are selected using inner out-of-fold
predictions only.  For logistic stacking, the regularization strength is
selected jointly with the threshold in the same nested inner folds.  The
reported predictions are therefore aligned, out of fold, and directly
comparable with the existing TACE-F1 predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from analyze_tace_crossfit import (
    DATASET_DIRS,
    DATASET_NAMES,
    DEFAULT_PREDICTION_ROOT,
    METHOD_F1_NAME,
    MODEL_ORDER,
    Record,
    append_system_summary,
    brier_and_bins,
    binary_counts,
    classwise_binary_counts,
    contribution_arrays,
    load_records,
    metrics_from_counts,
    multilabel_counts,
    predict_from_scores,
    primary_metric,
    repository_path,
    safe_divide,
    stratified_folds,
    supported_macro_f1,
    write_csv,
)


PREVALENCE_NAME = "Prevalence-only (nested cross-fitted)"
STACKER_NAME = "Logistic stacker (nested cross-fitted)"
DEFAULT_TACE_ROOT = DEFAULT_PREDICTION_ROOT / "tace_analysis"
DEFAULT_OUTPUT_ROOT = DEFAULT_PREDICTION_ROOT / "recovery_baseline_analysis"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--tace-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--regularization-c",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
        help="Candidate inverse L2 strengths for class-wise logistic stacking.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--permutation-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args(argv)
    if any(value <= 0 for value in args.regularization_c):
        parser.error("--regularization-c values must be positive")
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")
    if args.permutation_repetitions < 100:
        parser.error("--permutation-repetitions must be at least 100")
    args.regularization_c = sorted(set(args.regularization_c))
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_labels(value: str) -> frozenset[str]:
    return frozenset(item for item in value.split(";") if item)


def inverse_dataset_names() -> dict[str, str]:
    return {display: internal for internal, display in DATASET_NAMES.items()}


def load_protocol(tace_root: Path) -> dict[str, Any]:
    manifest_path = tace_root / "tace_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing TACE manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest.get("parameters", {})
    required = {
        "outer_folds", "inner_folds", "candidate_thresholds", "seed",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(f"TACE manifest is missing protocol parameters: {missing}")
    return {
        "manifest": manifest,
        "outer_folds": int(parameters["outer_folds"]),
        "inner_folds": int(parameters["inner_folds"]),
        "thresholds": sorted(float(value) for value in parameters["candidate_thresholds"]),
        "tace_seed": int(parameters["seed"]),
    }


def load_exact_outer_folds(
    tace_root: Path,
    records_by_dataset: dict[str, dict[str, Record]],
    expected_folds: int,
) -> tuple[dict[str, dict[str, int]], Path]:
    path = tace_root / "tace_fold_assignments.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TACE fold assignments: {path}")
    inverse = inverse_dataset_names()
    fold_by_dataset: dict[str, dict[str, int]] = {dataset: {} for dataset in DATASET_DIRS}
    for row in read_csv(path):
        if row["dataset"] not in inverse:
            raise ValueError(f"Unknown dataset in fold assignments: {row['dataset']}")
        dataset = inverse[row["dataset"]]
        key = row["record_key"]
        fold = int(row["outer_fold"])
        if not 0 <= fold < expected_folds:
            raise ValueError(f"Invalid outer fold {fold} for {dataset}/{key}")
        if key in fold_by_dataset[dataset]:
            raise ValueError(f"Duplicate fold assignment for {dataset}/{key}")
        fold_by_dataset[dataset][key] = fold
    for dataset, records in records_by_dataset.items():
        if set(fold_by_dataset[dataset]) != set(records):
            missing = sorted(set(records) - set(fold_by_dataset[dataset]))[:5]
            extra = sorted(set(fold_by_dataset[dataset]) - set(records))[:5]
            raise ValueError(
                f"Fold assignment mismatch for {dataset}: missing={missing}, extra={extra}"
            )
        observed = set(fold_by_dataset[dataset].values())
        if observed != set(range(expected_folds)):
            raise ValueError(f"Incomplete outer folds for {dataset}: {sorted(observed)}")
    return fold_by_dataset, path


def load_tace_outputs(
    tace_root: Path,
    records_by_dataset: dict[str, dict[str, Record]],
    fold_by_dataset: dict[str, dict[str, int]],
) -> tuple[
    dict[str, dict[str, frozenset[str]]],
    dict[str, dict[str, dict[str, float]]],
    Path,
]:
    path = tace_root / "tace_oof_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TACE OOF predictions: {path}")
    inverse = inverse_dataset_names()
    predictions: dict[str, dict[str, frozenset[str]]] = {
        dataset: {} for dataset in DATASET_DIRS
    }
    scores: dict[str, dict[str, dict[str, float]]] = {
        dataset: {} for dataset in DATASET_DIRS
    }
    for row in read_csv(path):
        dataset = inverse[row["dataset"]]
        key = row["record_key"]
        if key in predictions[dataset]:
            raise ValueError(f"Duplicate TACE OOF prediction for {dataset}/{key}")
        if int(row["outer_fold"]) != fold_by_dataset[dataset][key]:
            raise ValueError(f"TACE/fold mismatch for {dataset}/{key}")
        truth = parse_labels(row["truth"])
        if truth != records_by_dataset[dataset][key].truth:
            raise ValueError(f"TACE/ground-truth mismatch for {dataset}/{key}")
        predictions[dataset][key] = parse_labels(row["tace_f1_prediction"])
        score_row = json.loads(row["posterior_by_class"])
        scores[dataset][key] = {str(label): float(value) for label, value in score_row.items()}
    for dataset, records in records_by_dataset.items():
        if set(predictions[dataset]) != set(records) or set(scores[dataset]) != set(records):
            raise ValueError(f"Incomplete TACE OOF coverage for {dataset}")
    return predictions, scores, path


def prevalence_scores(
    train_keys: Iterable[str],
    score_keys: Iterable[str],
    records: dict[str, Record],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    train = list(train_keys)
    if not train:
        raise ValueError("Cannot estimate prevalence without training records")
    prevalence = {
        label: (sum(label in records[key].truth for key in train) + 1.0) / (len(train) + 2.0)
        for label in labels
    }
    return {key: dict(prevalence) for key in score_keys}


def sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def design_matrix(keys: Sequence[str], records: dict[str, Record], label: str) -> np.ndarray:
    matrix = np.ones((len(keys), len(MODEL_ORDER) + 1), dtype=np.float64)
    for column, model in enumerate(MODEL_ORDER, start=1):
        matrix[:, column] = [float(label in records[key].votes[model]) for key in keys]
    return matrix


def fit_logistic_coefficients(
    train_keys: Sequence[str],
    records: dict[str, Record],
    label: str,
    inverse_l2: float,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> np.ndarray:
    if not train_keys:
        raise ValueError("Cannot fit a logistic stacker without training records")
    x = design_matrix(train_keys, records, label)
    y = np.asarray([float(label in records[key].truth) for key in train_keys], dtype=np.float64)
    prior = (float(y.sum()) + 1.0) / (len(y) + 2.0)
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prior / (1.0 - prior))
    # The zero-support ScrawlD class, or a rare class absent from one inner
    # training split, has no estimable vote effect.  Use the smoothed constant
    # probability instead of allowing an unpenalized intercept to diverge.
    if float(y.sum()) in (0.0, float(len(y))):
        return coefficients
    penalty = 1.0 / inverse_l2
    penalty_diagonal = np.diag([0.0] + [penalty] * (x.shape[1] - 1))

    for _ in range(max_iterations):
        probability = sigmoid(x @ coefficients)
        weights = np.maximum(probability * (1.0 - probability), 1e-9)
        gradient = x.T @ (probability - y) + penalty_diagonal @ coefficients
        hessian = x.T @ (weights[:, None] * x) + penalty_diagonal
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficients -= step
        if float(np.max(np.abs(step))) < tolerance:
            break
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError(f"Non-finite logistic coefficients for class {label}")
    return coefficients


def fit_logistic_models(
    train_keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    inverse_l2: float,
) -> dict[str, np.ndarray]:
    return {
        label: fit_logistic_coefficients(train_keys, records, label, inverse_l2)
        for label in labels
    }


def logistic_scores(
    score_keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    models: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    output = {key: {} for key in score_keys}
    for label in labels:
        probabilities = sigmoid(design_matrix(score_keys, records, label) @ models[label])
        for key, probability in zip(score_keys, probabilities.tolist()):
            output[key][label] = min(1.0, max(0.0, float(probability)))
    return output


def merge_scores(target: dict[str, dict[str, float]], source: dict[str, dict[str, float]]) -> None:
    overlap = set(target) & set(source)
    if overlap:
        raise ValueError(f"Duplicate OOF scores: {sorted(overlap)[:3]}")
    target.update(source)


def objective_value(
    dataset: str,
    keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    scores: dict[str, dict[str, float]],
    threshold: float,
) -> tuple[float, dict[str, float | int], float | str]:
    predictions = predict_from_scores(scores, threshold)
    counts = (
        binary_counts(keys, records, predictions)
        if dataset == "bccc"
        else multilabel_counts(keys, records, predictions, labels)
    )
    metrics = metrics_from_counts(counts, dataset)
    macro: float | str = ""
    if dataset != "bccc":
        macro = supported_macro_f1(classwise_binary_counts(keys, records, predictions, labels))
    return primary_metric(metrics, dataset), metrics, macro


def inner_oof_prevalence(
    train_keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    dataset: str,
    inner_folds: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for validation_keys in stratified_folds(train_keys, records, dataset, inner_folds, seed):
        validation_set = set(validation_keys)
        inner_train = [key for key in train_keys if key not in validation_set]
        merge_scores(output, prevalence_scores(inner_train, validation_keys, records, labels))
    if set(output) != set(train_keys):
        raise RuntimeError("Inner prevalence OOF coverage failure")
    return output


def inner_oof_logistic(
    train_keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    dataset: str,
    inner_folds: int,
    seed: int,
    inverse_l2: float,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for validation_keys in stratified_folds(train_keys, records, dataset, inner_folds, seed):
        validation_set = set(validation_keys)
        inner_train = [key for key in train_keys if key not in validation_set]
        models = fit_logistic_models(inner_train, records, labels, inverse_l2)
        merge_scores(output, logistic_scores(validation_keys, records, labels, models))
    if set(output) != set(train_keys):
        raise RuntimeError("Inner logistic OOF coverage failure")
    return output


def select_threshold(
    dataset: str,
    train_keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    scores: dict[str, dict[str, float]],
    thresholds: Sequence[float],
    system: str,
    outer_fold: int,
    inverse_l2: float | str = "",
) -> tuple[float, float, list[dict[str, Any]]]:
    selected = thresholds[0]
    selected_value = -1.0
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        value, metrics, macro = objective_value(
            dataset, train_keys, records, labels, scores, threshold
        )
        rows.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "outer_fold": outer_fold,
                "system": system,
                "inverse_l2_c": inverse_l2,
                "threshold": threshold,
                "inner_primary_metric": "balanced_accuracy" if dataset == "bccc" else "micro_f1",
                "inner_primary_value": value,
                "inner_macro_f1_supported": macro,
                **metrics,
            }
        )
        if value > selected_value + 1e-12 or (
            abs(value - selected_value) <= 1e-12 and threshold > selected
        ):
            selected, selected_value = threshold, value
    return selected, selected_value, rows


def crossfit_recovery_baselines(
    dataset: str,
    records: dict[str, Record],
    labels: Sequence[str],
    fold_by_key: dict[str, int],
    inner_folds: int,
    thresholds: Sequence[float],
    regularization_grid: Sequence[float],
    tace_seed: int,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, frozenset[str]],
    dict[str, dict[str, float]],
    dict[str, frozenset[str]],
    list[dict[str, Any]],
    dict[str, dict[str, float | int]],
]:
    keys = sorted(records)
    outer_count = max(fold_by_key.values()) + 1
    prevalence_oof: dict[str, dict[str, float]] = {}
    stacker_oof: dict[str, dict[str, float]] = {}
    prevalence_predictions: dict[str, frozenset[str]] = {}
    stacker_predictions: dict[str, frozenset[str]] = {}
    selection_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[str, dict[str, float | int]] = {}

    for outer_fold in range(outer_count):
        test_keys = [key for key in keys if fold_by_key[key] == outer_fold]
        test_set = set(test_keys)
        train_keys = [key for key in keys if key not in test_set]
        inner_seed = tace_seed + 10_000 * (outer_fold + 1)

        prevalence_inner = inner_oof_prevalence(
            train_keys, records, labels, dataset, inner_folds, inner_seed
        )
        prevalence_threshold, prevalence_value, rows = select_threshold(
            dataset, train_keys, records, labels, prevalence_inner, thresholds,
            PREVALENCE_NAME, outer_fold,
        )
        for row in rows:
            row["selected"] = row["threshold"] == prevalence_threshold
        selection_rows.extend(rows)
        prevalence_test = prevalence_scores(train_keys, test_keys, records, labels)
        merge_scores(prevalence_oof, prevalence_test)
        prevalence_predictions.update(
            predict_from_scores(prevalence_test, prevalence_threshold)
        )

        best_c = regularization_grid[0]
        best_threshold = thresholds[0]
        best_value = -1.0
        best_rows: list[dict[str, Any]] = []
        all_c_rows: list[dict[str, Any]] = []
        for inverse_l2 in regularization_grid:
            logistic_inner = inner_oof_logistic(
                train_keys, records, labels, dataset, inner_folds,
                inner_seed, inverse_l2,
            )
            threshold, value, candidate_rows = select_threshold(
                dataset, train_keys, records, labels, logistic_inner, thresholds,
                STACKER_NAME, outer_fold, inverse_l2,
            )
            all_c_rows.extend(candidate_rows)
            if (
                value > best_value + 1e-12
                or (
                    abs(value - best_value) <= 1e-12
                    and (
                        threshold > best_threshold
                        or (threshold == best_threshold and inverse_l2 < best_c)
                    )
                )
            ):
                best_c, best_threshold, best_value = inverse_l2, threshold, value
                best_rows = candidate_rows
        for row in all_c_rows:
            row["selected"] = bool(
                float(row["inverse_l2_c"]) == best_c
                and float(row["threshold"]) == best_threshold
            )
        selection_rows.extend(all_c_rows)
        if not any(row["selected"] for row in best_rows):
            raise RuntimeError(f"No selected logistic configuration for {dataset}/fold {outer_fold}")
        models = fit_logistic_models(train_keys, records, labels, best_c)
        stacker_test = logistic_scores(test_keys, records, labels, models)
        merge_scores(stacker_oof, stacker_test)
        stacker_predictions.update(predict_from_scores(stacker_test, best_threshold))

        selected_by_fold[str(outer_fold)] = {
            "prevalence_threshold": prevalence_threshold,
            "prevalence_inner_primary": prevalence_value,
            "stacker_threshold": best_threshold,
            "stacker_inverse_l2_c": best_c,
            "stacker_inner_primary": best_value,
        }
        print(
            f"  {dataset} fold {outer_fold}: prevalence t={prevalence_threshold:.2f}; "
            f"stacker C={best_c:g}, t={best_threshold:.2f}",
            flush=True,
        )

    for name, values in (
        ("prevalence scores", prevalence_oof),
        ("prevalence predictions", prevalence_predictions),
        ("stacker scores", stacker_oof),
        ("stacker predictions", stacker_predictions),
    ):
        if set(values) != set(keys):
            raise RuntimeError(f"Incomplete {name} for {dataset}")
    return (
        prevalence_oof,
        prevalence_predictions,
        stacker_oof,
        stacker_predictions,
        selection_rows,
        selected_by_fold,
    )


def metric_from_batch_counts(counts: np.ndarray, dataset: str) -> np.ndarray:
    values = counts.astype(np.float64)
    tp, fp, fn = values[:, 0], values[:, 1], values[:, 2]
    if dataset != "bccc":
        denominator = 2.0 * tp + fp + fn
        return np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    tn = values[:, 3]
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) != 0)
    return (recall + specificity) / 2.0


def paired_statistics(
    dataset: str,
    keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    candidate_name: str,
    candidate_predictions: dict[str, frozenset[str]],
    baseline_name: str,
    baseline_predictions: dict[str, frozenset[str]],
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    candidate = np.asarray(
        contribution_arrays(dataset, records, keys, candidate_predictions, labels),
        dtype=np.int32,
    )
    baseline = np.asarray(
        contribution_arrays(dataset, records, keys, baseline_predictions, labels),
        dtype=np.int32,
    )
    candidate_value = float(metric_from_batch_counts(candidate.sum(axis=0, keepdims=True), dataset)[0])
    baseline_value = float(metric_from_batch_counts(baseline.sum(axis=0, keepdims=True), dataset)[0])
    observed = candidate_value - baseline_value
    n = len(keys)
    rng = np.random.default_rng(seed)
    batch_size = min(32, bootstrap_repetitions)
    differences: list[float] = []
    completed = 0
    while completed < bootstrap_repetitions:
        current = min(batch_size, bootstrap_repetitions - completed)
        indices = rng.integers(0, n, size=(current, n))
        candidate_counts = candidate[indices].sum(axis=1)
        baseline_counts = baseline[indices].sum(axis=1)
        differences.extend(
            (metric_from_batch_counts(candidate_counts, dataset)
             - metric_from_batch_counts(baseline_counts, dataset)).tolist()
        )
        completed += current
    lower, upper = np.percentile(np.asarray(differences), [2.5, 97.5]).tolist()

    total = candidate + baseline
    extreme = 0
    completed = 0
    batch_size = min(32, permutation_repetitions)
    while completed < permutation_repetitions:
        current = min(batch_size, permutation_repetitions - completed)
        choose_candidate = rng.integers(0, 2, size=(current, n, 1), dtype=np.int8)
        permuted_candidate = (
            choose_candidate * candidate[None, :, :]
            + (1 - choose_candidate) * baseline[None, :, :]
        ).sum(axis=1)
        permuted_baseline = total.sum(axis=0, keepdims=True) - permuted_candidate
        permuted_difference = (
            metric_from_batch_counts(permuted_candidate, dataset)
            - metric_from_batch_counts(permuted_baseline, dataset)
        )
        extreme += int(np.count_nonzero(np.abs(permuted_difference) >= abs(observed) - 1e-15))
        completed += current
    permutation_p = (extreme + 1.0) / (permutation_repetitions + 1.0)
    return {
        "dataset": DATASET_NAMES[dataset],
        "candidate": candidate_name,
        "baseline": baseline_name,
        "metric": "balanced_accuracy" if dataset == "bccc" else "micro_f1",
        "candidate_value": candidate_value,
        "baseline_value": baseline_value,
        "observed_difference": observed,
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
        "bootstrap_repetitions": bootstrap_repetitions,
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "paired_random_swap_p": permutation_p,
        "permutation_repetitions": permutation_repetitions,
    }


def add_holm_adjustment(rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for group in by_dataset.values():
        ordered = sorted(group, key=lambda row: float(row["paired_random_swap_p"]))
        running = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, float(row["paired_random_swap_p"]) * (total - rank))
            running = max(running, adjusted)
            row["holm_adjusted_p"] = running


def auc_binary(scores: Sequence[float], truth: Sequence[int]) -> float:
    positives = sum(truth)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = 0.0
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        rank_sum += average_rank * sum(truth[order[index]] for index in range(position, end))
        position = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def discrimination_summary(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    labels: Sequence[str],
    scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    aucs: list[float] = []
    for label in labels:
        truth = [int(label in records[key].truth) for key in keys]
        value = auc_binary([scores[key][label] for key in keys], truth)
        if math.isfinite(value):
            aucs.append(value)
    return {
        "mean_supported_class_auc": safe_divide(sum(aucs), len(aucs)),
        "auc_supported_class_count": len(aucs),
    }


def mechanism_rows(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    labels: Sequence[str],
    tace_predictions: dict[str, frozenset[str]],
    prevalence_predictions: dict[str, frozenset[str]],
    stacker_predictions: dict[str, frozenset[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    systems = {
        "tace": tace_predictions,
        "prevalence": prevalence_predictions,
        "stacker": stacker_predictions,
    }
    pattern_rows: list[dict[str, Any]] = []
    total_decisions = len(keys) * len(labels)
    all_zero_tace_positive = all_zero_tace_tp = 0
    tace_positive = tace_tp = 0
    disagreement_prevalence = disagreement_stacker = 0
    different_record_prevalence: set[str] = set()
    different_record_stacker: set[str] = set()

    pattern_counts: dict[tuple[str, str], dict[str, int]] = {}
    for key in keys:
        record = records[key]
        for label in labels:
            pattern = "".join(str(int(label in record.votes[model])) for model in MODEL_ORDER)
            truth = label in record.truth
            row = pattern_counts.setdefault(
                (label, pattern),
                {
                    "decisions": 0, "truth_positive": 0,
                    "tace_positive": 0, "tace_tp": 0,
                    "prevalence_positive": 0, "prevalence_tp": 0,
                    "stacker_positive": 0, "stacker_tp": 0,
                },
            )
            row["decisions"] += 1
            row["truth_positive"] += int(truth)
            for short_name, predictions in systems.items():
                positive = label in predictions[key]
                row[f"{short_name}_positive"] += int(positive)
                row[f"{short_name}_tp"] += int(positive and truth)
            tace_is_positive = label in tace_predictions[key]
            tace_positive += int(tace_is_positive)
            tace_tp += int(tace_is_positive and truth)
            if pattern == "000":
                all_zero_tace_positive += int(tace_is_positive)
                all_zero_tace_tp += int(tace_is_positive and truth)
            if tace_is_positive != (label in prevalence_predictions[key]):
                disagreement_prevalence += 1
                different_record_prevalence.add(key)
            if tace_is_positive != (label in stacker_predictions[key]):
                disagreement_stacker += 1
                different_record_stacker.add(key)

    for (label, pattern), values in sorted(pattern_counts.items()):
        pattern_rows.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "taxonomy_class": label,
                "agreement_pattern": pattern,
                "empirical_positive_rate": safe_divide(
                    values["truth_positive"], values["decisions"]
                ),
                **values,
            }
        )
    summary = {
        "dataset": DATASET_NAMES[dataset],
        "class_decisions": total_decisions,
        "tace_positive_decisions": tace_positive,
        "tace_true_positive_decisions": tace_tp,
        "tace_positive_from_all_zero_pattern": all_zero_tace_positive,
        "fraction_tace_positive_from_all_zero_pattern": safe_divide(
            all_zero_tace_positive, tace_positive
        ),
        "tace_true_positive_from_all_zero_pattern": all_zero_tace_tp,
        "fraction_tace_true_positive_from_all_zero_pattern": safe_divide(
            all_zero_tace_tp, tace_tp
        ),
        "tace_vs_prevalence_disagreement_decisions": disagreement_prevalence,
        "tace_vs_prevalence_disagreement_fraction": safe_divide(
            disagreement_prevalence, total_decisions
        ),
        "tace_vs_prevalence_different_records": len(different_record_prevalence),
        "tace_vs_stacker_disagreement_decisions": disagreement_stacker,
        "tace_vs_stacker_disagreement_fraction": safe_divide(
            disagreement_stacker, total_decisions
        ),
        "tace_vs_stacker_different_records": len(different_record_stacker),
    }
    return summary, pattern_rows


def validate_tace_summary(
    tace_root: Path,
    generated_rows: list[dict[str, Any]],
) -> Path:
    path = tace_root / "tace_oof_summary.csv"
    rows = read_csv(path)
    expected = {
        row["dataset"]: float(row["primary_value"])
        for row in rows if row["system"] == METHOD_F1_NAME
    }
    observed = {
        str(row["dataset"]): float(row["primary_value"])
        for row in generated_rows if row["system"] == METHOD_F1_NAME
    }
    if set(expected) != set(observed):
        raise ValueError("TACE summary dataset mismatch")
    for dataset in expected:
        if abs(expected[dataset] - observed[dataset]) > 1e-12:
            raise ValueError(
                f"TACE reproduction mismatch for {dataset}: "
                f"expected={expected[dataset]}, observed={observed[dataset]}"
            )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_root = args.prediction_root.resolve()
    tace_root = (args.tace_root or (prediction_root / "tace_analysis")).resolve()
    output_root = (args.output_root or DEFAULT_OUTPUT_ROOT).resolve()
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Prediction root does not exist: {prediction_root}")
    if not tace_root.is_dir():
        raise FileNotFoundError(f"TACE root does not exist: {tace_root}")
    if output_root in (prediction_root, tace_root):
        raise ValueError("Recovery output root must be separate from input roots")

    protocol = load_protocol(tace_root)
    records_by_dataset, labels_by_dataset, truth_paths, source_paths = load_records(
        prediction_root
    )
    fold_by_dataset, fold_path = load_exact_outer_folds(
        tace_root, records_by_dataset, protocol["outer_folds"]
    )
    tace_predictions, tace_scores, tace_oof_path = load_tace_outputs(
        tace_root, records_by_dataset, fold_by_dataset
    )
    print("Validated frozen predictions, TACE OOF outputs, and exact outer folds.", flush=True)

    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    mechanism_summary_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, Any] = {}

    for dataset_index, dataset in enumerate(DATASET_DIRS):
        records = records_by_dataset[dataset]
        labels = labels_by_dataset[dataset]
        keys = sorted(records)
        print(
            f"Recovery cross-fitting {dataset}: {len(keys)} records, {len(labels)} classes",
            flush=True,
        )
        (
            prevalence_oof,
            prevalence_predictions,
            stacker_oof,
            stacker_predictions,
            dataset_selection_rows,
            dataset_selected_configs,
        ) = crossfit_recovery_baselines(
            dataset=dataset,
            records=records,
            labels=labels,
            fold_by_key=fold_by_dataset[dataset],
            inner_folds=protocol["inner_folds"],
            thresholds=protocol["thresholds"],
            regularization_grid=args.regularization_c,
            tace_seed=protocol["tace_seed"],
        )
        selection_rows.extend(dataset_selection_rows)
        selected_configs[dataset] = dataset_selected_configs

        append_system_summary(
            summary_rows, dataset, METHOD_F1_NAME, keys, records,
            tace_predictions[dataset], labels,
            "existing nested cross-fitted TACE-F1 OOF predictions",
        )
        append_system_summary(
            summary_rows, dataset, PREVALENCE_NAME, keys, records,
            prevalence_predictions, labels,
            "no model features; nested class-prevalence estimation and threshold selection",
        )
        append_system_summary(
            summary_rows, dataset, STACKER_NAME, keys, records,
            stacker_predictions, labels,
            "class-wise L2 logistic stacking; nested regularization and threshold selection",
        )

        for system, scores in (
            (METHOD_F1_NAME, tace_scores[dataset]),
            (PREVALENCE_NAME, prevalence_oof),
            (STACKER_NAME, stacker_oof),
        ):
            calibration, _ = brier_and_bins(dataset, records, keys, labels, scores)
            calibration_rows.append(
                {
                    "dataset": DATASET_NAMES[dataset],
                    "system": system,
                    **calibration,
                    **discrimination_summary(dataset, records, keys, labels, scores),
                }
            )

        comparisons = (
            (METHOD_F1_NAME, tace_predictions[dataset], PREVALENCE_NAME, prevalence_predictions),
            (METHOD_F1_NAME, tace_predictions[dataset], STACKER_NAME, stacker_predictions),
            (STACKER_NAME, stacker_predictions, PREVALENCE_NAME, prevalence_predictions),
        )
        for comparison_index, (
            candidate_name, candidate_predictions, baseline_name, baseline_predictions
        ) in enumerate(comparisons):
            print(
                f"  paired analysis: {candidate_name} vs {baseline_name}", flush=True
            )
            comparison_rows.append(
                paired_statistics(
                    dataset=dataset,
                    keys=keys,
                    records=records,
                    labels=labels,
                    candidate_name=candidate_name,
                    candidate_predictions=candidate_predictions,
                    baseline_name=baseline_name,
                    baseline_predictions=baseline_predictions,
                    bootstrap_repetitions=args.bootstrap_repetitions,
                    permutation_repetitions=args.permutation_repetitions,
                    seed=args.seed + 1000 * dataset_index + 100 * comparison_index,
                )
            )

        mechanism_summary, dataset_pattern_rows = mechanism_rows(
            dataset, records, keys, labels, tace_predictions[dataset],
            prevalence_predictions, stacker_predictions,
        )
        mechanism_summary_rows.append(mechanism_summary)
        pattern_rows.extend(dataset_pattern_rows)

        for key in keys:
            fold = fold_by_dataset[dataset][key]
            config = dataset_selected_configs[str(fold)]
            oof_rows.append(
                {
                    "dataset": DATASET_NAMES[dataset],
                    "record_key": key,
                    "record_index": records[key].record_index,
                    "outer_fold": fold,
                    "truth": ";".join(sorted(records[key].truth)),
                    "tace_prediction": ";".join(sorted(tace_predictions[dataset][key])),
                    "prevalence_prediction": ";".join(sorted(prevalence_predictions[key])),
                    "stacker_prediction": ";".join(sorted(stacker_predictions[key])),
                    "prevalence_scores": json.dumps(prevalence_oof[key], sort_keys=True),
                    "stacker_scores": json.dumps(stacker_oof[key], sort_keys=True),
                    **config,
                }
            )

    add_holm_adjustment(comparison_rows)
    tace_summary_path = validate_tace_summary(tace_root, summary_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "recovery_baseline_summary.csv": summary_rows,
        "recovery_nested_selection.csv": selection_rows,
        "recovery_calibration_discrimination.csv": calibration_rows,
        "recovery_paired_comparisons.csv": comparison_rows,
        "recovery_mechanism_summary.csv": mechanism_summary_rows,
        "recovery_pattern_diagnostics.csv": pattern_rows,
        "recovery_oof_predictions.csv": oof_rows,
    }
    for filename, rows in outputs.items():
        write_csv(output_root / filename, rows)

    generated = sorted(output_root.glob("recovery_*.csv"))
    manifest = {
        "schema_version": "recovery-baselines-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": repository_path(Path(__file__)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "scope": "analysis-only; reuses fixed model predictions and exact TACE outer folds",
        "methods": {
            "prevalence_only": (
                "Beta(1,1)-smoothed class prevalence; no model-output features; "
                "nested threshold selection"
            ),
            "logistic_stacker": (
                "one L2 logistic model per class using three binary model votes; "
                "nested joint regularization and threshold selection"
            ),
        },
        "protocol": {
            "outer_folds": protocol["outer_folds"],
            "inner_folds": protocol["inner_folds"],
            "candidate_thresholds": protocol["thresholds"],
            "regularization_c": args.regularization_c,
            "tace_seed_for_inner_folds": protocol["tace_seed"],
            "recovery_seed": args.seed,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "permutation_repetitions": args.permutation_repetitions,
            "selected_configs": selected_configs,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "inputs": [
            {"path": repository_path(fold_path), "sha256": sha256(fold_path)},
            {"path": repository_path(tace_oof_path), "sha256": sha256(tace_oof_path)},
            {"path": repository_path(tace_summary_path), "sha256": sha256(tace_summary_path)},
            *[
                {"path": repository_path(path), "sha256": sha256(path)}
                for path in sorted(set(truth_paths.values()))
            ],
            *[
                {"path": repository_path(path), "sha256": sha256(path)}
                for path in sorted(set(source_paths.values()))
            ],
        ],
        "generated_files": [
            {"path": repository_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        ],
    }
    manifest_path = output_root / "recovery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "README.md").write_text(
        "# Recovery baselines\n\n"
        "This directory compares the existing out-of-fold TACE-F1 decisions with "
        "a no-model prevalence-only rule and a conventional class-wise logistic "
        "stacker. All systems use the same outer record partitions. Hyperparameters "
        "and thresholds for the new baselines are selected only within nested inner "
        "folds. No LLM inference is performed.\n",
        encoding="utf-8",
    )
    print(f"Wrote recovery analysis to {output_root}", flush=True)
    print(f"Summary: {output_root / 'recovery_baseline_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
