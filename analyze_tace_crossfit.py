#!/usr/bin/env python3
"""Cross-fitted taxonomy-aware calibration for frozen-model predictions.

TACE (taxonomy-aware calibrated ensemble) is an analysis-only, post-inference
method.  It combines the *already generated* outputs of Qwen, CodeLlama, and
Mistral.  It does not import Transformers, load a checkpoint, or generate a
new prediction.

For a dataset ``d``, taxonomy class ``c``, and three-model agreement pattern
``z in {0,1}^3``, the method estimates the posterior probability

    P(y_c=1 | z,d,c) = (n_{z,c,+} + lambda * pi_c) / (n_{z,c} + lambda),

where ``pi_c`` is a Beta-smoothed class prior.  Thus each class has its own
calibration table and sparse agreement patterns shrink to that class prior.
This directly accommodates correlated model outputs; unlike a naive-Bayes
fusion rule, it does *not* assume the three LLMs are conditionally
independent.

All calibration tables and the decision threshold are fitted inside the
training partition of each outer fold.  Threshold selection uses only nested
inner folds.  The final report therefore contains one out-of-fold (OOF)
prediction for every benchmark record and avoids tuning on the test fold.

TACE-F1 selects a global operating threshold for micro-F1 (or BCCC balanced
accuracy).  For the multilabel benchmarks, TACE-Macro selects a separate
global threshold for supported-class macro-F1.  The script also reports a
fixed-threshold selective-decision curve.  A class decision is automatically
accepted when p >= q (positive) or p <= 1-q (negative); otherwise it is
deferred.  ``q`` is never selected on the test data.  This makes coverage and
decision risk explicit rather than silently discarding uncertain cases.

Example
-------
python analyze_tace_crossfit.py \
  --prediction-root results/final_benchmark

The output directory is self-contained and is suitable for manuscript tables.
It deliberately leaves the original JSONL and robustness results untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from analyze_prediction_robustness import (
    DATASET_DIRS,
    DATASET_NAMES,
    DEFAULT_PREDICTION_ROOT,
    MODEL_ORDER,
    CompactRecord,
    load_all_runs,
    load_multilabel_ground_truth,
    repository_path,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = DEFAULT_PREDICTION_ROOT / "tace_analysis"
METHOD_NAME = "TACE (cross-fitted empirical-Bayes ensemble)"
METHOD_F1_NAME = "TACE-F1 (cross-fitted empirical-Bayes ensemble)"
METHOD_MACRO_NAME = "TACE-Macro (cross-fitted empirical-Bayes ensemble)"
BASELINE_SYSTEMS = ("OR-3 ensemble", "Majority-3 ensemble", "AND-3 ensemble")


@dataclass(frozen=True)
class Record:
    """Aligned immutable record used by the analysis-only ensemble."""

    key: str
    record_index: int
    truth: frozenset[str]
    votes: dict[str, frozenset[str]]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        help="Candidate positive-decision thresholds, selected in nested folds.",
    )
    parser.add_argument(
        "--selective-confidence",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95],
        help="Fixed confidence levels for the selective risk-coverage curve.",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=8.0,
        help="Fixed empirical-Bayes pseudo-count for an agreement pattern.",
    )
    parser.add_argument(
        "--shrinkage-grid",
        type=float,
        nargs="*",
        default=[2.0, 4.0, 8.0, 16.0, 32.0],
        help=(
            "Pseudo-count values for a nested cross-fitted sensitivity analysis. "
            "Pass the option without values to disable the sweep."
        ),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args(argv)
    if args.outer_folds < 2:
        parser.error("--outer-folds must be at least 2")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2")
    if args.shrinkage <= 0:
        parser.error("--shrinkage must be positive")
    if any(value <= 0 for value in args.shrinkage_grid):
        parser.error("--shrinkage-grid values must be positive")
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")
    for value in args.thresholds:
        if not 0.0 <= value <= 1.0:
            parser.error("--thresholds values must be in [0, 1]")
    for value in args.selective_confidence:
        if not 0.5 <= value <= 1.0:
            parser.error("--selective-confidence values must be in [0.5, 1.0]")
    args.thresholds = sorted(set(args.thresholds))
    args.selective_confidence = sorted(set(args.selective_confidence))
    args.shrinkage_grid = sorted(set(args.shrinkage_grid + [args.shrinkage]))
    return args


def record_stratum(record: Record, dataset: str) -> str:
    """Return a deterministic stratification stratum without using predictions."""

    if dataset == "bccc":
        return "vulnerable" if "vulnerable" in record.truth else "secure"
    return "|".join(sorted(record.truth)) or "clean"


def stratified_folds(
    keys: Sequence[str],
    records: dict[str, Record],
    dataset: str,
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    """Create exact-label-set stratified folds with deterministic shuffling."""

    if len(keys) < n_splits:
        raise ValueError(f"Cannot create {n_splits} folds from only {len(keys)} records")
    groups: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        groups[record_stratum(records[key], dataset)].append(key)
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for stratum in sorted(groups):
        members = sorted(groups[stratum])
        rng.shuffle(members)
        start = rng.randrange(n_splits)
        for offset, key in enumerate(members):
            folds[(start + offset) % n_splits].append(key)
    for fold in folds:
        fold.sort()
    if any(not fold for fold in folds):
        raise ValueError("Stratification unexpectedly created an empty fold")
    return folds


def load_records(
    prediction_root: Path,
) -> tuple[dict[str, dict[str, Record]], dict[str, list[str]], dict[str, Path], dict[str, Path]]:
    multilabel_truth, labels_by_dataset, truth_paths = load_multilabel_ground_truth()
    all_runs = load_all_runs(prediction_root, multilabel_truth)
    records_by_dataset: dict[str, dict[str, Record]] = {}
    source_paths: dict[str, Path] = {}
    labels_by_dataset = dict(labels_by_dataset)
    labels_by_dataset["bccc"] = ["vulnerable"]

    for dataset in DATASET_DIRS:
        reference = all_runs[dataset][MODEL_ORDER[0]].records
        aligned: dict[str, Record] = {}
        for key, reference_record in reference.items():
            if reference_record.status != "ok":
                raise ValueError(f"{dataset}/{MODEL_ORDER[0]}/{key} does not have status=ok")
            votes: dict[str, frozenset[str]] = {}
            for model in MODEL_ORDER:
                compact: CompactRecord = all_runs[dataset][model].records[key]
                if compact.status != "ok":
                    raise ValueError(f"{dataset}/{model}/{key} does not have status=ok")
                if dataset == "bccc":
                    if compact.expected_binary not in (0, 1):
                        raise ValueError(f"Invalid BCCC truth for {key}")
                    truth = frozenset({"vulnerable"}) if compact.expected_binary else frozenset()
                    predicted = frozenset({"vulnerable"}) if compact.predicted else frozenset()
                else:
                    if compact.expected_multilabel is None:
                        raise ValueError(f"Missing multilabel truth for {dataset}/{key}")
                    truth = compact.expected_multilabel
                    predicted = compact.predicted
                votes[model] = predicted
            aligned[key] = Record(
                key=key,
                record_index=reference_record.record_index,
                truth=truth,
                votes=votes,
            )
        records_by_dataset[dataset] = aligned
        for model in MODEL_ORDER:
            source_paths[f"{dataset}/{model}"] = all_runs[dataset][model].path
    return records_by_dataset, labels_by_dataset, truth_paths, source_paths


def agreement_pattern(record: Record, label: str) -> tuple[int, ...]:
    return tuple(int(label in record.votes[model]) for model in MODEL_ORDER)


def fit_calibrator(
    keys: Iterable[str],
    records: dict[str, Record],
    labels: Sequence[str],
    shrinkage: float,
) -> dict[str, dict[str, Any]]:
    """Fit class-wise empirical-Bayes agreement-pattern calibration tables."""

    key_list = list(keys)
    if not key_list:
        raise ValueError("Cannot fit a calibrator without training records")
    tables: dict[str, dict[str, Any]] = {}
    for label in labels:
        positives = sum(label in records[key].truth for key in key_list)
        prior = (positives + 1.0) / (len(key_list) + 2.0)  # Beta(1, 1)
        counts: Counter[tuple[int, ...]] = Counter()
        positive_counts: Counter[tuple[int, ...]] = Counter()
        for key in key_list:
            record = records[key]
            pattern = agreement_pattern(record, label)
            counts[pattern] += 1
            if label in record.truth:
                positive_counts[pattern] += 1
        tables[label] = {
            "prior": prior,
            "counts": counts,
            "positive_counts": positive_counts,
            "shrinkage": shrinkage,
        }
    return tables


def calibrated_score(table: dict[str, Any], pattern: tuple[int, ...]) -> float:
    count = int(table["counts"].get(pattern, 0))
    positives = int(table["positive_counts"].get(pattern, 0))
    score = (positives + float(table["shrinkage"]) * float(table["prior"])) / (
        count + float(table["shrinkage"])
    )
    return min(1.0, max(0.0, score))


def score_keys(
    keys: Iterable[str],
    records: dict[str, Record],
    labels: Sequence[str],
    calibrator: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for key in keys:
        record = records[key]
        scores[key] = {
            label: calibrated_score(calibrator[label], agreement_pattern(record, label))
            for label in labels
        }
    return scores


def predict_from_scores(scores: dict[str, dict[str, float]], threshold: float) -> dict[str, frozenset[str]]:
    return {
        key: frozenset(label for label, score in row.items() if score >= threshold)
        for key, row in scores.items()
    }


def multilabel_counts(
    keys: Iterable[str],
    records: dict[str, Record],
    predictions: dict[str, frozenset[str]],
    labels: Sequence[str],
) -> dict[str, int]:
    label_set = set(labels)
    tp = fp = fn = 0
    for key in keys:
        truth = records[key].truth & label_set
        predicted = predictions[key] & label_set
        tp += len(truth & predicted)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
    return {"tp": tp, "fp": fp, "fn": fn}


def binary_counts(
    keys: Iterable[str], records: dict[str, Record], predictions: dict[str, frozenset[str]]
) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for key in keys:
        truth = "vulnerable" in records[key].truth
        predicted = "vulnerable" in predictions[key]
        if truth and predicted:
            tp += 1
        elif not truth and predicted:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict[str, int], dataset: str) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    output: dict[str, float | int] = {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
    }
    if dataset == "bccc":
        tn = counts["tn"]
        specificity = safe_divide(tn, tn + fp)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        output.update(
            {
                "specificity": specificity,
                "false_positive_rate": safe_divide(fp, fp + tn),
                "accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
                "balanced_accuracy": (recall + specificity) / 2.0,
                "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
            }
        )
    return output


def primary_metric(metrics: dict[str, float | int], dataset: str) -> float:
    return float(metrics["balanced_accuracy"] if dataset == "bccc" else metrics["f1"])


def classwise_binary_counts(
    keys: Iterable[str],
    records: dict[str, Record],
    predictions: dict[str, frozenset[str]],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Return one-vs-rest confusion counts for every declared taxonomy class."""

    result: dict[str, dict[str, int]] = {}
    for label in labels:
        tp = fp = fn = tn = 0
        for key in keys:
            truth = label in records[key].truth
            predicted = label in predictions[key]
            if truth and predicted:
                tp += 1
            elif not truth and predicted:
                fp += 1
            elif truth:
                fn += 1
            else:
                tn += 1
        result[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return result


def supported_macro_f1(class_counts: dict[str, dict[str, int]]) -> float:
    values: list[float] = []
    for counts in class_counts.values():
        support = counts["tp"] + counts["fn"]
        if support:
            values.append(safe_divide(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]))
    return safe_divide(sum(values), len(values))


def tune_threshold(
    train_keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    dataset: str,
    inner_folds: int,
    thresholds: Sequence[float],
    shrinkage: float,
    seed: int,
    objective: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose a threshold with inner OOF predictions only."""

    folds = stratified_folds(train_keys, records, dataset, inner_folds, seed)
    aggregate: dict[float, dict[str, Any]] = {}
    for threshold in thresholds:
        aggregate[threshold] = {
            "overall": {"tp": 0, "fp": 0, "fn": 0, **({"tn": 0} if dataset == "bccc" else {})},
            "per_class": {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in labels},
        }
    for validation_keys in folds:
        validation_set = set(validation_keys)
        inner_train = [key for key in train_keys if key not in validation_set]
        calibrator = fit_calibrator(inner_train, records, labels, shrinkage)
        scores = score_keys(validation_keys, records, labels, calibrator)
        for threshold in thresholds:
            predictions = predict_from_scores(scores, threshold)
            counts = (
                binary_counts(validation_keys, records, predictions)
                if dataset == "bccc"
                else multilabel_counts(validation_keys, records, predictions, labels)
            )
            for name, value in counts.items():
                aggregate[threshold]["overall"][name] += value
            if dataset != "bccc":
                for label, class_counts in classwise_binary_counts(
                    validation_keys, records, predictions, labels
                ).items():
                    for name, value in class_counts.items():
                        aggregate[threshold]["per_class"][label][name] += value
    rows: list[dict[str, Any]] = []
    selected = thresholds[0]
    selected_value = -1.0
    for threshold in thresholds:
        metrics = metrics_from_counts(aggregate[threshold]["overall"], dataset)
        if objective == "micro_f1":
            value = float(metrics["f1"])
        elif objective == "macro_f1_supported":
            value = supported_macro_f1(aggregate[threshold]["per_class"])
        elif objective == "balanced_accuracy":
            value = float(metrics["balanced_accuracy"])
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        row = {
            "threshold": threshold,
            "inner_objective": objective,
            "inner_objective_value": value,
            "inner_micro_f1": metrics["f1"],
            "inner_macro_f1_supported": (
                supported_macro_f1(aggregate[threshold]["per_class"])
                if dataset != "bccc" else ""
            ),
            "inner_balanced_accuracy": metrics.get("balanced_accuracy", ""),
            **metrics,
        }
        rows.append(row)
        # A high threshold is the deterministic tie-breaker: it emits fewer
        # low-confidence alerts when effectiveness is unchanged.
        if value > selected_value + 1e-12 or (abs(value - selected_value) <= 1e-12 and threshold > selected):
            selected, selected_value = threshold, value
    return selected, rows


def baseline_predictions(
    keys: Iterable[str], records: dict[str, Record], labels: Sequence[str], rule: str
) -> dict[str, frozenset[str]]:
    output: dict[str, frozenset[str]] = {}
    for key in keys:
        record = records[key]
        chosen: set[str] = set()
        for label in labels:
            votes = sum(label in record.votes[model] for model in MODEL_ORDER)
            if (rule == "or" and votes >= 1) or (rule == "majority" and votes >= 2) or (rule == "and" and votes == len(MODEL_ORDER)):
                chosen.add(label)
        output[key] = frozenset(chosen)
    return output


def original_model_predictions(keys: Iterable[str], records: dict[str, Record], model: str) -> dict[str, frozenset[str]]:
    return {key: records[key].votes[model] for key in keys}


def append_system_summary(
    rows: list[dict[str, Any]],
    dataset: str,
    system: str,
    keys: Sequence[str],
    records: dict[str, Record],
    predictions: dict[str, frozenset[str]],
    labels: Sequence[str],
    protocol: str,
) -> None:
    counts = (
        binary_counts(keys, records, predictions)
        if dataset == "bccc"
        else multilabel_counts(keys, records, predictions, labels)
    )
    metrics = metrics_from_counts(counts, dataset)
    macro_fields: dict[str, float | int | str] = {}
    if dataset != "bccc":
        per_class = classwise_binary_counts(keys, records, predictions, labels)
        supported = [
            safe_divide(2 * value["tp"], 2 * value["tp"] + value["fp"] + value["fn"])
            for value in per_class.values()
            if value["tp"] + value["fn"] > 0
        ]
        macro_fields = {
            "macro_f1_supported": safe_divide(sum(supported), len(supported)),
            "supported_class_count": len(supported),
        }
    rows.append(
        {
            "dataset": DATASET_NAMES[dataset],
            "system": system,
            "protocol": protocol,
            "records": len(keys),
            "primary_metric": "balanced_accuracy" if dataset == "bccc" else "micro_f1",
            "primary_value": primary_metric(metrics, dataset),
            **macro_fields,
            **metrics,
        }
    )


def per_class_rows(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    predictions: dict[str, frozenset[str]],
    labels: Sequence[str],
    system: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label in labels:
        tp = fp = fn = tn = 0
        for key in keys:
            truth = label in records[key].truth
            predicted = label in predictions[key]
            if truth and predicted:
                tp += 1
            elif not truth and predicted:
                fp += 1
            elif truth:
                fn += 1
            else:
                tn += 1
        metrics = metrics_from_counts(
            {"tp": tp, "fp": fp, "fn": fn, "tn": tn} if dataset == "bccc" else {"tp": tp, "fp": fp, "fn": fn},
            dataset,
        )
        output.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "system": system,
                "taxonomy_class": label,
                "support": tp + fn,
                **metrics,
            }
        )
    return output


def brier_and_bins(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    labels: Sequence[str],
    scores: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values: list[tuple[float, int]] = []
    for key in keys:
        for label in labels:
            values.append((scores[key][label], int(label in records[key].truth)))
    brier = safe_divide(sum((score - truth) ** 2 for score, truth in values), len(values))
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        members = [pair for pair in values if (low <= pair[0] < high) or (index == 9 and pair[0] == 1.0)]
        if members:
            mean_score = sum(pair[0] for pair in members) / len(members)
            empirical_rate = sum(pair[1] for pair in members) / len(members)
            ece += len(members) / len(values) * abs(mean_score - empirical_rate)
        else:
            mean_score = empirical_rate = 0.0
        bins.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "bin_index": index,
                "lower": low,
                "upper": high,
                "decisions": len(members),
                "mean_posterior": mean_score,
                "empirical_positive_rate": empirical_rate,
                "absolute_gap": abs(mean_score - empirical_rate) if members else 0.0,
            }
        )
    return {"brier_score": brier, "ece_10_bins": ece, "class_decisions": len(values)}, bins


def class_calibration_rows(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    labels: Sequence[str],
    scores: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Report calibration separately for every class, including rare classes."""

    rows: list[dict[str, Any]] = []
    for label in labels:
        values = [(scores[key][label], int(label in records[key].truth)) for key in keys]
        support = sum(truth for _, truth in values)
        brier = safe_divide(sum((score - truth) ** 2 for score, truth in values), len(values))
        ece = 0.0
        nonempty_bins = 0
        for index in range(10):
            low, high = index / 10.0, (index + 1) / 10.0
            members = [
                pair for pair in values
                if (low <= pair[0] < high) or (index == 9 and pair[0] == 1.0)
            ]
            if not members:
                continue
            nonempty_bins += 1
            mean_score = sum(score for score, _ in members) / len(members)
            empirical_rate = sum(truth for _, truth in members) / len(members)
            ece += len(members) / len(values) * abs(mean_score - empirical_rate)
        rows.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "system": METHOD_F1_NAME,
                "taxonomy_class": label,
                "records": len(values),
                "support": support,
                "prevalence": safe_divide(support, len(values)),
                "brier_score": brier,
                "ece_10_bins": ece,
                "nonempty_bins": nonempty_bins,
                "zero_support": support == 0,
            }
        )
    return rows


def selective_rows(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    labels: Sequence[str],
    scores: dict[str, dict[str, float]],
    confidences: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(keys) * len(labels)
    total_truth_positive = sum(
        label in records[key].truth for key in keys for label in labels
    )
    for confidence in confidences:
        accepted = errors = accepted_positive = positive_tp = 0
        accepted_negative = positive_fp = negative_fn = 0
        candidate_positive = sum(
            scores[key][label] >= 0.5 for key in keys for label in labels
        )
        for key in keys:
            for label in labels:
                score = scores[key][label]
                truth = label in records[key].truth
                if score >= confidence:
                    accepted += 1
                    accepted_positive += 1
                    if truth:
                        positive_tp += 1
                    else:
                        errors += 1
                        positive_fp += 1
                elif score <= 1.0 - confidence:
                    accepted += 1
                    accepted_negative += 1
                    if truth:
                        errors += 1
                        negative_fn += 1
        rows.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "system": METHOD_F1_NAME,
                "confidence": confidence,
                "class_decisions_total": total,
                "accepted_decisions": accepted,
                "abstained_decisions": total - accepted,
                "decision_coverage": safe_divide(accepted, total),
                "accepted_decision_risk": safe_divide(errors, accepted),
                "accepted_positive_alerts": accepted_positive,
                "accepted_positive_precision": safe_divide(positive_tp, accepted_positive),
                "accepted_positive_false_discovery_risk": safe_divide(positive_fp, accepted_positive),
                "accepted_positive_alert_coverage": safe_divide(accepted_positive, candidate_positive),
                "ground_truth_positive_recall": safe_divide(positive_tp, total_truth_positive),
                "accepted_negative_decisions": accepted_negative,
                "accepted_negative_error_risk": safe_divide(negative_fn, accepted_negative),
                "ground_truth_positive_decisions": total_truth_positive,
                "candidate_positive_decisions_at_0_5": candidate_positive,
            }
        )
    return rows


def crossfit_f1_for_shrinkage(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    labels: Sequence[str],
    outer_folds: Sequence[Sequence[str]],
    inner_folds: int,
    thresholds: Sequence[float],
    shrinkage: float,
    seed: int,
) -> tuple[dict[str, frozenset[str]], dict[int, float]]:
    """Repeat the full nested F1 operating-point protocol for one pseudo-count."""

    scores: dict[str, dict[str, float]] = {}
    threshold_by_fold: dict[int, float] = {}
    fold_by_key: dict[str, int] = {}
    objective = "balanced_accuracy" if dataset == "bccc" else "micro_f1"
    for fold_index, test_keys in enumerate(outer_folds):
        test_set = set(test_keys)
        train_keys = [key for key in keys if key not in test_set]
        threshold, _ = tune_threshold(
            train_keys=train_keys,
            records=records,
            labels=labels,
            dataset=dataset,
            inner_folds=inner_folds,
            thresholds=thresholds,
            shrinkage=shrinkage,
            seed=seed + 10_000 * (fold_index + 1),
            objective=objective,
        )
        threshold_by_fold[fold_index] = threshold
        calibrator = fit_calibrator(train_keys, records, labels, shrinkage)
        scores.update(score_keys(test_keys, records, labels, calibrator))
        for key in test_keys:
            fold_by_key[key] = fold_index
    if set(scores) != set(keys) or set(fold_by_key) != set(keys):
        raise RuntimeError(f"Shrinkage-sweep OOF coverage failure for {dataset}")
    predictions = {
        key: frozenset(
            label for label, score in scores[key].items()
            if score >= threshold_by_fold[fold_by_key[key]]
        )
        for key in keys
    }
    return predictions, threshold_by_fold


def contribution_arrays(
    dataset: str,
    records: dict[str, Record],
    keys: Sequence[str],
    predictions: dict[str, frozenset[str]],
    labels: Sequence[str],
) -> list[tuple[int, ...]]:
    values: list[tuple[int, ...]] = []
    if dataset == "bccc":
        for key in keys:
            truth = "vulnerable" in records[key].truth
            predicted = "vulnerable" in predictions[key]
            values.append((int(truth and predicted), int(not truth and predicted), int(truth and not predicted), int(not truth and not predicted)))
    else:
        label_set = set(labels)
        for key in keys:
            truth = records[key].truth & label_set
            predicted = predictions[key] & label_set
            values.append((len(truth & predicted), len(predicted - truth), len(truth - predicted)))
    return values


def paired_bootstrap_rows(
    np: Any,
    dataset: str,
    keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    tace_predictions: dict[str, frozenset[str]],
    candidate_name: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired percentile bootstrap CIs against the fixed-output baselines."""

    systems: dict[str, dict[str, frozenset[str]]] = {
        **{model: original_model_predictions(keys, records, model) for model in MODEL_ORDER},
        "OR-3 ensemble": baseline_predictions(keys, records, labels, "or"),
        "Majority-3 ensemble": baseline_predictions(keys, records, labels, "majority"),
        "AND-3 ensemble": baseline_predictions(keys, records, labels, "and"),
    }
    candidate = np.asarray(contribution_arrays(dataset, records, keys, tace_predictions, labels), dtype=np.int32)
    rng = np.random.default_rng(seed)
    n = len(keys)
    batch_size = min(64, repetitions)
    rows: list[dict[str, Any]] = []

    def metric(counts: Any) -> Any:
        counts = counts.astype(np.float64)
        tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
        if dataset != "bccc":
            denominator = 2 * tp + fp + fn
            return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
        tn = counts[:, 3]
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) != 0)
        return (recall + specificity) / 2.0

    for name, predictions in systems.items():
        baseline = np.asarray(contribution_arrays(dataset, records, keys, predictions, labels), dtype=np.int32)
        observed = float(metric(candidate.sum(axis=0, keepdims=True))[0] - metric(baseline.sum(axis=0, keepdims=True))[0])
        differences: list[float] = []
        completed = 0
        while completed < repetitions:
            current = min(batch_size, repetitions - completed)
            indices = rng.integers(0, n, size=(current, n))
            candidate_counts = candidate[indices].sum(axis=1)
            baseline_counts = baseline[indices].sum(axis=1)
            differences.extend((metric(candidate_counts) - metric(baseline_counts)).tolist())
            completed += current
        lower, upper = np.percentile(np.asarray(differences), [2.5, 97.5]).tolist()
        rows.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "candidate": candidate_name,
                "baseline": name,
                "metric": "balanced_accuracy" if dataset == "bccc" else "micro_f1",
                "observed_difference": observed,
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
                "bootstrap_repetitions": repetitions,
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "uncertainty_scope": "conditional_on_fixed_oof_predictions",
                "pipeline_refitted_within_bootstrap": False,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required. Install requirements-analysis.txt first.") from error

    prediction_root = args.prediction_root.resolve()
    output_root = (args.output_root or (prediction_root / "tace_analysis")).resolve()
    if output_root == prediction_root:
        raise ValueError("--output-root must not be the prediction root")
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Prediction root does not exist: {prediction_root}")

    print(f"Loading and validating all nine result files from {prediction_root}", flush=True)
    records_by_dataset, labels_by_dataset, truth_paths, source_paths = load_records(prediction_root)
    print("Identity, ground-truth, model-set, and status=ok checks passed.", flush=True)

    summaries: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_per_class: list[dict[str, Any]] = []
    all_selective: list[dict[str, Any]] = []
    all_reliability_bins: list[dict[str, Any]] = []
    all_class_calibration: list[dict[str, Any]] = []
    all_bootstrap: list[dict[str, Any]] = []
    all_shrinkage_sensitivity: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    calibration_summaries: list[dict[str, Any]] = []

    for dataset in DATASET_DIRS:
        records = records_by_dataset[dataset]
        labels = labels_by_dataset[dataset]
        keys = sorted(records)
        print(f"Cross-fitting {dataset}: {len(keys)} records, {len(labels)} taxonomy labels", flush=True)
        outer = stratified_folds(keys, records, dataset, args.outer_folds, args.seed)
        oof_scores: dict[str, dict[str, float]] = {}
        selected_thresholds_f1: dict[int, float] = {}
        selected_thresholds_macro: dict[int, float] = {}

        for fold_index, test_keys in enumerate(outer):
            test_set = set(test_keys)
            train_keys = [key for key in keys if key not in test_set]
            f1_objective = "balanced_accuracy" if dataset == "bccc" else "micro_f1"
            threshold, inner_rows = tune_threshold(
                train_keys=train_keys,
                records=records,
                labels=labels,
                dataset=dataset,
                inner_folds=args.inner_folds,
                thresholds=args.thresholds,
                shrinkage=args.shrinkage,
                seed=args.seed + 10_000 * (fold_index + 1),
                objective=f1_objective,
            )
            selected_thresholds_f1[fold_index] = threshold
            for row in inner_rows:
                threshold_rows.append(
                    {
                        "dataset": DATASET_NAMES[dataset],
                        "outer_fold": fold_index,
                        "operating_point": "TACE-F1",
                        "selected": row["threshold"] == threshold,
                        **row,
                    }
                )
            if dataset != "bccc":
                macro_threshold, macro_inner_rows = tune_threshold(
                    train_keys=train_keys,
                    records=records,
                    labels=labels,
                    dataset=dataset,
                    inner_folds=args.inner_folds,
                    thresholds=args.thresholds,
                    shrinkage=args.shrinkage,
                    seed=args.seed + 20_000 * (fold_index + 1),
                    objective="macro_f1_supported",
                )
                selected_thresholds_macro[fold_index] = macro_threshold
                for row in macro_inner_rows:
                    threshold_rows.append(
                        {
                            "dataset": DATASET_NAMES[dataset],
                            "outer_fold": fold_index,
                            "operating_point": "TACE-Macro",
                            "selected": row["threshold"] == macro_threshold,
                            **row,
                        }
                    )
            calibrator = fit_calibrator(train_keys, records, labels, args.shrinkage)
            oof_scores.update(score_keys(test_keys, records, labels, calibrator))
            for key in test_keys:
                fold_rows.append(
                    {
                        "dataset": DATASET_NAMES[dataset],
                        "record_key": key,
                        "record_index": records[key].record_index,
                        "outer_fold": fold_index,
                        "stratum": record_stratum(records[key], dataset),
                    }
                )

        if set(oof_scores) != set(keys):
            raise RuntimeError(f"OOF score coverage failure for {dataset}")
        # Materialise a checked key-to-fold map before applying the individual
        # outer-fold thresholds.  This keeps each threshold strictly separate
        # from the records used to assess its outer fold.
        fold_by_key = {
            row["record_key"]: int(row["outer_fold"])
            for row in fold_rows
            if row["dataset"] == DATASET_NAMES[dataset]
        }
        if set(fold_by_key) != set(keys):
            raise RuntimeError(f"Fold-assignment coverage failure for {dataset}")
        tace_f1_predictions = {
            key: frozenset(
                label for label, score in oof_scores[key].items()
                if score >= selected_thresholds_f1[fold_by_key[key]]
            )
            for key in keys
        }
        tace_macro_predictions = {
            key: frozenset(
                label for label, score in oof_scores[key].items()
                if score >= selected_thresholds_macro[fold_by_key[key]]
            )
            for key in keys
        } if dataset != "bccc" else None

        dataset_shrinkage_rows: list[dict[str, Any]] = []
        for candidate_shrinkage in args.shrinkage_grid:
            if abs(candidate_shrinkage - args.shrinkage) <= 1e-12:
                candidate_predictions = tace_f1_predictions
                candidate_thresholds = selected_thresholds_f1
            else:
                print(
                    f"  shrinkage sensitivity lambda={candidate_shrinkage:g}", flush=True
                )
                candidate_predictions, candidate_thresholds = crossfit_f1_for_shrinkage(
                    dataset=dataset,
                    records=records,
                    keys=keys,
                    labels=labels,
                    outer_folds=outer,
                    inner_folds=args.inner_folds,
                    thresholds=args.thresholds,
                    shrinkage=candidate_shrinkage,
                    seed=args.seed,
                )
            candidate_summary: list[dict[str, Any]] = []
            append_system_summary(
                candidate_summary,
                dataset,
                METHOD_F1_NAME,
                keys,
                records,
                candidate_predictions,
                labels,
                "nested cross-fitted shrinkage sensitivity",
            )
            row = candidate_summary[0]
            row.update(
                {
                    "shrinkage": candidate_shrinkage,
                    "configured_primary_shrinkage": args.shrinkage,
                    "selected_thresholds_by_fold": json.dumps(
                        candidate_thresholds, sort_keys=True
                    ),
                    "selected_threshold_min": min(candidate_thresholds.values()),
                    "selected_threshold_max": max(candidate_thresholds.values()),
                }
            )
            dataset_shrinkage_rows.append(row)
        reference_value = next(
            float(row["primary_value"])
            for row in dataset_shrinkage_rows
            if abs(float(row["shrinkage"]) - args.shrinkage) <= 1e-12
        )
        for row in dataset_shrinkage_rows:
            row["delta_primary_vs_configured"] = float(row["primary_value"]) - reference_value
        all_shrinkage_sensitivity.extend(dataset_shrinkage_rows)

        append_system_summary(
            summaries, dataset, METHOD_F1_NAME, keys, records, tace_f1_predictions, labels,
            "nested cross-fitted calibration + nested selection of micro-F1 (or BCCC balanced accuracy)",
        )
        if tace_macro_predictions is not None:
            append_system_summary(
                summaries, dataset, METHOD_MACRO_NAME, keys, records, tace_macro_predictions, labels,
                "nested cross-fitted calibration + nested selection of supported macro-F1",
            )
        for model in MODEL_ORDER:
            append_system_summary(
                summaries, dataset, model, keys, records,
                original_model_predictions(keys, records, model), labels, "frozen model output",
            )
        for name, rule in (("OR-3 ensemble", "or"), ("Majority-3 ensemble", "majority"), ("AND-3 ensemble", "and")):
            append_system_summary(
                summaries, dataset, name, keys, records, baseline_predictions(keys, records, labels, rule), labels,
                "fixed agreement rule; no fitted parameters",
            )
        all_per_class.extend(per_class_rows(
            dataset, records, keys, tace_f1_predictions, labels, METHOD_F1_NAME,
        ))
        if tace_macro_predictions is not None:
            all_per_class.extend(per_class_rows(
                dataset, records, keys, tace_macro_predictions, labels, METHOD_MACRO_NAME,
            ))
        all_selective.extend(selective_rows(
            dataset, records, keys, labels, oof_scores, args.selective_confidence,
        ))
        calibration, bins = brier_and_bins(dataset, records, keys, labels, oof_scores)
        calibration_summaries.append({"dataset": DATASET_NAMES[dataset], "system": METHOD_F1_NAME, **calibration})
        all_reliability_bins.extend(bins)
        all_class_calibration.extend(
            class_calibration_rows(dataset, records, keys, labels, oof_scores)
        )
        all_bootstrap.extend(paired_bootstrap_rows(
            np, dataset, keys, records, labels, tace_f1_predictions, METHOD_F1_NAME,
            args.bootstrap_repetitions,
            args.seed + {"smartbugs": 101, "scrawld": 202, "bccc": 303}[dataset],
        ))
        for key in keys:
            oof_rows.append(
                {
                    "dataset": DATASET_NAMES[dataset],
                    "record_key": key,
                    "record_index": records[key].record_index,
                    "outer_fold": fold_by_key[key],
                    "tace_f1_selected_threshold": selected_thresholds_f1[fold_by_key[key]],
                    "tace_macro_selected_threshold": (
                        selected_thresholds_macro[fold_by_key[key]] if dataset != "bccc" else ""
                    ),
                    "truth": ";".join(sorted(records[key].truth)),
                    "tace_f1_prediction": ";".join(sorted(tace_f1_predictions[key])),
                    "tace_macro_prediction": ";".join(sorted(tace_macro_predictions[key])) if tace_macro_predictions is not None else "",
                    "posterior_by_class": json.dumps(oof_scores[key], sort_keys=True),
                    "qwen_prediction": ";".join(sorted(records[key].votes[MODEL_ORDER[0]])),
                    "codellama_prediction": ";".join(sorted(records[key].votes[MODEL_ORDER[1]])),
                    "mistral_prediction": ";".join(sorted(records[key].votes[MODEL_ORDER[2]])),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "tace_oof_summary.csv", summaries)
    write_csv(output_root / "tace_oof_per_class.csv", all_per_class)
    write_csv(output_root / "tace_threshold_selection.csv", threshold_rows)
    write_csv(output_root / "tace_fold_assignments.csv", fold_rows)
    write_csv(output_root / "tace_selective_risk_coverage.csv", all_selective)
    write_csv(output_root / "tace_calibration_summary.csv", calibration_summaries)
    write_csv(output_root / "tace_calibration_per_class.csv", all_class_calibration)
    write_csv(output_root / "tace_reliability_bins.csv", all_reliability_bins)
    write_csv(output_root / "tace_paired_bootstrap.csv", all_bootstrap)
    write_csv(output_root / "tace_shrinkage_sensitivity.csv", all_shrinkage_sensitivity)
    write_csv(output_root / "tace_oof_predictions.csv", oof_rows)
    bccc_confusion = [
        {
            "dataset": row["dataset"],
            "system": row["system"],
            "tn": row["tn"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tp": row["tp"],
            "precision": row["precision"],
            "recall": row["recall"],
            "specificity": row["specificity"],
            "balanced_accuracy": row["balanced_accuracy"],
            "mcc": row["mcc"],
        }
        for row in summaries if row["dataset"] == DATASET_NAMES["bccc"]
    ]
    write_csv(output_root / "tace_bccc_confusion_matrix.csv", bccc_confusion)

    generated = sorted(output_root.glob("tace_*.csv"))
    manifest = {
        "schema_version": "tace-crossfit-analysis",
        "method": METHOD_NAME,
        "method_scope": "analysis-only post-inference fusion; no model generation or retraining",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": repository_path(Path(__file__)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "prediction_root": repository_path(prediction_root),
        "parameters": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "candidate_thresholds": args.thresholds,
            "selective_confidence": args.selective_confidence,
            "pattern_shrinkage": args.shrinkage,
            "shrinkage_sensitivity_grid": args.shrinkage_grid,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "method_notes": [
            "Class-wise calibration maps each three-model agreement pattern to an empirical-Bayes posterior.",
            "The estimator makes no conditional-independence claim about the three models.",
            "Every reported TACE posterior is fitted without that record's outer test fold.",
            "TACE-F1 and TACE-Macro thresholds are selected only with nested inner folds; fixed confidence values define the selective curve.",
            "BCCC is evaluated as binary vulnerable-versus-secure; SmartBugs-Curated and ScrawlD are closed-taxonomy multilabel tasks.",
            "Paired bootstrap intervals are conditional on the retained OOF predictions; calibration and threshold fitting are not repeated inside each bootstrap sample.",
        ],
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "ground_truth_files": [
            {"dataset": name, "path": repository_path(path), "sha256": sha256(path)}
            for name, path in sorted(truth_paths.items())
        ],
        "source_jsonl": [
            {"dataset_model": name, "path": repository_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in sorted(source_paths.items())
        ],
        "generated_files": [
            {"path": repository_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        ],
    }
    (output_root / "tace_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# TACE cross-fitted ensemble results\n\n"
        "These results implement a post-inference, taxonomy-aware empirical-Bayes "
        "ensemble over the three frozen model outputs. All TACE quantities are "
        "out-of-fold. TACE-F1 and TACE-Macro are distinct nested-selected operating "
        "points, not results tuned on the outer test folds. See `tace_manifest.json` "
        "for the exact inputs, hashes, and protocol.\n\n"
        "Do not describe this as a newly trained detector or as a new LLM inference run. "
        "It is a reproducible decision layer on fixed model outputs.\n",
        encoding="utf-8",
    )
    print(f"Wrote TACE results to {output_root}", flush=True)
    print(f"Primary results: {output_root / 'tace_oof_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
