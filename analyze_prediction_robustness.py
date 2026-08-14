#!/usr/bin/env python3
# Copyright 2026 Thi-Thu-Huong Le
# SPDX-License-Identifier: Apache-2.0

"""Robustness analysis for the selectively repaired benchmark predictions.

This is an analysis-only pipeline: it never imports Transformers, loads a
model, or performs inference.  It addresses three publication concerns:

1. sensitivity to selective repair;
2. paired model comparisons with bootstrap CIs and permutation tests; and
3. support-aware macro/per-class reporting for classes with zero positives.

The script reads the final JSONL files and writes a self-contained
``robustness_analysis`` directory suitable for manuscript tables and auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_PREDICTION_ROOT = SCRIPT_DIR / "results" / "final_benchmark"
DEFAULT_OUTPUT_ROOT = DEFAULT_PREDICTION_ROOT / "robustness_analysis"

DATASET_DIRS = ("smartbugs", "scrawld", "bccc")
DATASET_NAMES = {
    "smartbugs": "SmartBugs-Curated",
    "scrawld": "ScrawlD",
    "bccc": "BCCC",
}
MODEL_ORDER = (
    "Qwen2.5-Coder-7B-Instruct",
    "CodeLlama-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
)


def repository_path(path: Path) -> str:
    """Represent a file portably without exposing a machine-local path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        return resolved.name
SMARTBUGS_LABEL_MAP = {
    "access_control": "access_control",
    "arithmetic": "arithmetic",
    "bad_randomness": "bad_randomness",
    "denial_of_service": "denial_of_service",
    "front_running": "transaction_ordering",
    "other": "other",
    "reentrancy": "reentrancy",
    "short_addresses": "short_addresses",
    "time_manipulation": "time_manipulation",
    "unchecked_low_level_calls": "unchecked_low_level_calls",
}
SCRAWLD_LABEL_MAP = {
    "ARTHM": "arithmetic",
    "DOS": "denial_of_service",
    "LE": "locked_ether",
    "RENT": "reentrancy",
    "TimeM": "time_manipulation",
    "TimeO": "transaction_ordering",
    "TX-Origin": "tx_origin",
    "Tx-Origin": "tx_origin",
    "UE": "unchecked_low_level_calls",
}


@dataclass(frozen=True)
class CompactRecord:
    key: str
    record_index: int
    expected_multilabel: frozenset[str] | None
    expected_binary: int | None
    predicted: frozenset[str]
    touched: bool
    inference_actions: int
    structural_actions: int
    already_valid_actions: int
    status: str


@dataclass(frozen=True)
class CompactRun:
    dataset: str
    model: str
    path: Path
    records: dict[str, CompactRecord]


def import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for paired resampling. Install it with:\n"
            "  python -m pip install -r requirements-analysis.txt"
        ) from error
    return np


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def code_digest(code: str) -> str:
    normalized = code.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_prediction(category: object) -> str | None:
    text = " ".join("".join(char if char.isalnum() else " " for char in str(category).lower()).split())
    if text == "other":
        return "other"
    if "reentr" in text:
        return "reentrancy"
    if any(token in text for token in ("overflow", "underflow", "arithmetic", "arithm")):
        return "arithmetic"
    if any(
        token in text
        for token in (
            "unchecked low level", "unchecked ll", "unhandled exception",
            "unchecked call", "unchecked external call",
        )
    ):
        return "unchecked_low_level_calls"
    if "locked ether" in text:
        return "locked_ether"
    if any(
        token in text
        for token in ("bad randomness", "insecure randomness", "weak randomness", "randomness")
    ):
        return "bad_randomness"
    if "denial of service" in text or text == "dos":
        return "denial_of_service"
    if any(
        token in text
        for token in ("front running", "front run", "transaction order", "timestamp ordering")
    ):
        return "transaction_ordering"
    if any(
        token in text
        for token in (
            "time manipulation", "timestamp dependence", "timestamp dependency",
            "timing attack", "block timestamp",
        )
    ):
        return "time_manipulation"
    if "tx origin" in text or "txorigin" in text:
        return "tx_origin"
    if "short address" in text:
        return "short_addresses"
    if any(
        token in text
        for token in (
            "access control", "authorization", "unauthorized", "privilege escalation",
            "unprotected function", "unprotected ether withdrawal", "unprotected fallback",
        )
    ):
        return "access_control"
    return None


def model_name(path: Path) -> str:
    name = path.name.lower()
    if "qwen" in name:
        return "Qwen2.5-Coder-7B-Instruct"
    if "mistral" in name:
        return "Mistral-7B-Instruct-v0.3"
    if "codellama" in name:
        return "CodeLlama-7B-Instruct"
    raise ValueError(f"Cannot infer expected model name from {path.name}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_existing(candidates: Iterable[Path], label: str) -> Path:
    checked: list[str] = []
    for path in candidates:
        checked.append(str(path))
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Cannot locate {label}. Checked: {checked}")


def ground_truth_paths() -> dict[str, Path]:
    return {
        "smartbugs": first_existing(
            (DATA_ROOT / "smartbugs_curated.json",),
            "SmartBugs ground truth",
        ),
        "scrawld": first_existing(
            (DATA_ROOT / "scrawld_vulnerabilities.json",),
            "ScrawlD ground truth",
        ),
    }


def load_multilabel_ground_truth() -> tuple[dict[str, dict[str, frozenset[str]]], dict[str, list[str]], dict[str, Path]]:
    paths = ground_truth_paths()
    smart_records = read_json(paths["smartbugs"])
    smart = {
        code_digest(str(record.get("code", ""))): frozenset(
            SMARTBUGS_LABEL_MAP[finding["category"]]
            for finding in record.get("vulnerabilities", [])
            if finding.get("category") in SMARTBUGS_LABEL_MAP
        )
        for record in smart_records
    }
    scrawld_records = read_json(paths["scrawld"])
    scrawld = {
        str(record["address"]): frozenset(
            SCRAWLD_LABEL_MAP[key]
            for group in record.get("vulnerabilities", [])
            for key, value in group.items()
            if value and key in SCRAWLD_LABEL_MAP
        )
        for record in scrawld_records
    }
    labels = {
        "smartbugs": sorted(set(SMARTBUGS_LABEL_MAP.values())),
        "scrawld": sorted(set(SCRAWLD_LABEL_MAP.values())),
    }
    return {"smartbugs": smart, "scrawld": scrawld}, labels, paths


def accepted_predictions(row: dict[str, Any]) -> frozenset[str]:
    findings = row.get("vulnerabilities", [])
    if not isinstance(findings, list):
        raise ValueError("A JSONL vulnerabilities field is not a list")
    predicted: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("A JSONL vulnerability finding is not an object")
        canonical = canonicalize_prediction(finding.get("category", ""))
        if canonical is not None:
            predicted.add(canonical)
    return frozenset(predicted)


def action_count(actions: dict[str, Any], name: str) -> int:
    value = actions.get(name, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid repair action count {name}={value!r}")


def repair_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return current or legacy repair metadata without release-name coupling."""

    current = record.get("repair_details")
    if isinstance(current, dict):
        return current
    candidate: dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith("repair_") and isinstance(value, dict):
            # Historical passes appended their metadata in chronological
            # order, so the last mapping describes the retained prediction.
            candidate = value
    return candidate


def repair_actions(row: dict[str, Any]) -> tuple[bool, int, int, int]:
    repair = repair_metadata(row)
    actions = repair.get("actions") or {}
    inference = action_count(actions, "inference")
    structural = action_count(actions, "structural")
    already_valid = action_count(actions, "already_valid")

    # Defensive fallback for rows whose top-level action summary is absent.
    if inference == 0:
        inference = sum(
            1
            for chunk in row.get("chunks", [])
            if isinstance(chunk, dict)
            and repair_metadata(chunk).get("method")
            == "progressive_closed_choice_reinference"
        )
    touched = inference > 0 or structural > 0
    return touched, inference, structural, already_valid


def prediction_key(dataset: str, row: dict[str, Any]) -> str:
    field = "source_sha256" if dataset == "smartbugs" else "source_id"
    value = row.get(field)
    if not value:
        raise ValueError(f"Missing {field} in {dataset} JSONL row")
    return str(value)


def load_compact_run(
    path: Path,
    dataset: str,
    ground_truth: dict[str, frozenset[str]] | None,
) -> CompactRun:
    records: dict[str, CompactRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Corrupt JSONL at {path}:{line_number}") from error
            key = prediction_key(dataset, row)
            if key in records:
                raise ValueError(f"Duplicate record key {key!r} in {path}")
            touched, inference, structural, already_valid = repair_actions(row)
            if dataset == "bccc":
                expected_binary = row.get("ground_truth_binary")
                if expected_binary not in (0, 1):
                    raise ValueError(f"Invalid BCCC ground truth at {path}:{line_number}")
                expected_multilabel = None
            else:
                if ground_truth is None or key not in ground_truth:
                    raise ValueError(f"No {dataset} ground truth for key {key!r}")
                expected_multilabel = ground_truth[key]
                expected_binary = None
            records[key] = CompactRecord(
                key=key,
                record_index=int(row["record_index"]),
                expected_multilabel=expected_multilabel,
                expected_binary=expected_binary,
                predicted=accepted_predictions(row),
                touched=touched,
                inference_actions=inference,
                structural_actions=structural,
                already_valid_actions=already_valid,
                status=str(row.get("status", "missing_status")),
            )
    return CompactRun(dataset=dataset, model=model_name(path), path=path.resolve(), records=records)


def load_all_runs(
    prediction_root: Path,
    multilabel_truth: dict[str, dict[str, frozenset[str]]],
) -> dict[str, dict[str, CompactRun]]:
    all_runs: dict[str, dict[str, CompactRun]] = {}
    for dataset in DATASET_DIRS:
        paths = sorted((prediction_root / dataset).glob("*.jsonl"))
        if len(paths) != len(MODEL_ORDER):
            raise ValueError(f"Expected 3 JSONL files in {prediction_root / dataset}; found {len(paths)}")
        runs: dict[str, CompactRun] = {}
        for path in paths:
            run = load_compact_run(path, dataset, multilabel_truth.get(dataset))
            if run.model in runs:
                raise ValueError(f"Duplicate model {run.model} in {prediction_root / dataset}")
            runs[run.model] = run
        if set(runs) != set(MODEL_ORDER):
            raise ValueError(f"Unexpected model set for {dataset}: {sorted(runs)}")
        key_sets = {model: set(run.records) for model, run in runs.items()}
        reference = key_sets[MODEL_ORDER[0]]
        for model, keys in key_sets.items():
            if keys != reference:
                raise ValueError(
                    f"Record set mismatch for {dataset}/{model}: "
                    f"missing={len(reference - keys)}, extra={len(keys - reference)}"
                )
        # Validate identical ground truth across the three run files.
        for key in reference:
            expected_values = {
                (runs[model].records[key].expected_multilabel, runs[model].records[key].expected_binary)
                for model in MODEL_ORDER
            }
            if len(expected_values) != 1:
                raise ValueError(f"Ground-truth mismatch across models for {dataset}/{key}")
        all_runs[dataset] = runs
    return all_runs


def multi_record_counts(expected: frozenset[str], predicted: frozenset[str], labels: set[str]) -> tuple[int, int, int]:
    expected_set = set(expected) & labels
    predicted_set = set(predicted) & labels
    return (
        len(expected_set & predicted_set),
        len(predicted_set - expected_set),
        len(expected_set - predicted_set),
    )


def multilabel_metrics(
    records: Sequence[CompactRecord],
    labels: Sequence[str],
    prediction_policy: str = "final",
) -> dict[str, float | int]:
    label_set = set(labels)
    totals = [0, 0, 0]
    exact = 0
    sample_f1: list[float] = []
    per_class = {label: [0, 0, 0] for label in labels}
    for record in records:
        assert record.expected_multilabel is not None
        predicted = frozenset() if prediction_policy == "repair_as_empty" and record.touched else record.predicted
        tp, fp, fn = multi_record_counts(record.expected_multilabel, predicted, label_set)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        exact += (record.expected_multilabel & label_set) == (predicted & label_set)
        sample_f1.append(safe_divide(2 * tp, 2 * tp + fp + fn))
        for label in labels:
            expected_has = label in record.expected_multilabel
            predicted_has = label in predicted
            if expected_has and predicted_has:
                per_class[label][0] += 1
            elif not expected_has and predicted_has:
                per_class[label][1] += 1
            elif expected_has and not predicted_has:
                per_class[label][2] += 1
    tp, fp, fn = totals
    class_metrics: list[tuple[float, float, float, int]] = []
    for class_tp, class_fp, class_fn in per_class.values():
        support = class_tp + class_fn
        class_metrics.append(
            (
                safe_divide(class_tp, class_tp + class_fp),
                safe_divide(class_tp, support),
                safe_divide(2 * class_tp, 2 * class_tp + class_fp + class_fn),
                support,
            )
        )
    supported = [values for values in class_metrics if values[3] > 0]
    return {
        "n": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": safe_divide(tp, tp + fp),
        "micro_recall": safe_divide(tp, tp + fn),
        "micro_f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "macro_precision_all_declared_legacy": sum(values[0] for values in class_metrics) / len(class_metrics),
        "macro_recall_all_declared_legacy": sum(values[1] for values in class_metrics) / len(class_metrics),
        "macro_f1_all_declared_legacy": sum(values[2] for values in class_metrics) / len(class_metrics),
        "macro_precision_supported_classes": sum(values[0] for values in supported) / len(supported),
        "macro_recall_supported_classes": sum(values[1] for values in supported) / len(supported),
        "macro_f1_supported_classes": sum(values[2] for values in supported) / len(supported),
        "supported_class_count": len(supported),
        "declared_class_count": len(class_metrics),
        "exact_match_ratio": safe_divide(exact, len(records)),
        "sample_f1": safe_divide(sum(sample_f1), len(sample_f1)),
    }


def binary_metrics(records: Sequence[CompactRecord], prediction_policy: str = "final") -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for record in records:
        assert record.expected_binary in (0, 1)
        predicted = bool(record.predicted)
        if prediction_policy == "repair_as_empty" and record.touched:
            predicted = False
        if record.expected_binary == 1 and predicted:
            tp += 1
        elif record.expected_binary == 0 and predicted:
            fp += 1
        elif record.expected_binary == 0:
            tn += 1
        else:
            fn += 1
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "specificity": specificity,
        "false_positive_rate": safe_divide(fp, fp + tn),
        "accuracy": safe_divide(tp + tn, len(records)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
    }


def repair_audit_rows(all_runs: dict[str, dict[str, CompactRun]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_DIRS:
        for model in MODEL_ORDER:
            records = list(all_runs[dataset][model].records.values())
            touched = [record for record in records if record.touched]
            rows.append(
                {
                    "dataset": DATASET_NAMES[dataset],
                    "model": model,
                    "records_total": len(records),
                    "records_touched": len(touched),
                    "records_touched_rate": safe_divide(len(touched), len(records)),
                    "records_with_inference": sum(record.inference_actions > 0 for record in records),
                    "records_with_structural_repair": sum(record.structural_actions > 0 for record in records),
                    "chunk_inference_actions": sum(record.inference_actions for record in records),
                    "chunk_structural_actions": sum(record.structural_actions for record in records),
                    "chunk_already_valid_actions": sum(record.already_valid_actions for record in records),
                    "final_ok_records": sum(record.status == "ok" for record in records),
                    "source_jsonl": repository_path(all_runs[dataset][model].path),
                }
            )
    return rows


def sensitivity_rows(
    all_runs: dict[str, dict[str, CompactRun]],
    labels_by_dataset: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for dataset in DATASET_DIRS:
        runs = all_runs[dataset]
        keys = sorted(runs[MODEL_ORDER[0]].records)
        common_untouched = [
            key for key in keys
            if not any(runs[model].records[key].touched for model in MODEL_ORDER)
        ]
        dataset_metrics: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
        for model in MODEL_ORDER:
            run = runs[model]
            all_records = [run.records[key] for key in keys]
            common_records = [run.records[key] for key in common_untouched]
            policies = (
                ("final_all", all_records, "final"),
                ("repair_as_empty", all_records, "repair_as_empty"),
                ("common_untouched", common_records, "final"),
            )
            final_metrics: dict[str, float | int] | None = None
            for policy, records, prediction_policy in policies:
                metrics = (
                    binary_metrics(records, prediction_policy)
                    if dataset == "bccc"
                    else multilabel_metrics(records, labels_by_dataset[dataset], prediction_policy)
                )
                if policy == "final_all":
                    final_metrics = metrics
                assert final_metrics is not None
                primary_field = "balanced_accuracy" if dataset == "bccc" else "micro_f1"
                row = {
                    "dataset": DATASET_NAMES[dataset],
                    "model": model,
                    "policy": policy,
                    "n": metrics["n"],
                    "common_untouched_n": len(common_untouched),
                    "common_untouched_rate": safe_divide(len(common_untouched), len(keys)),
                    "primary_metric": primary_field,
                    "primary_value": metrics[primary_field],
                    "delta_primary_vs_final_all": float(metrics[primary_field]) - float(final_metrics[primary_field]),
                    **{key: value for key, value in metrics.items() if key != "n"},
                }
                rows.append(row)
                dataset_metrics[policy][model] = metrics

        rankings = {
            policy: sorted(
                MODEL_ORDER,
                key=lambda model: float(model_metrics[model]["balanced_accuracy" if dataset == "bccc" else "micro_f1"]),
                reverse=True,
            )
            for policy, model_metrics in dataset_metrics.items()
        }
        audit[DATASET_NAMES[dataset]] = {
            "records_total": len(keys),
            "common_untouched_records": len(common_untouched),
            "common_untouched_rate": safe_divide(len(common_untouched), len(keys)),
            "rankings": rankings,
        }
    return rows, audit


def contribution_arrays(
    np: Any,
    dataset: str,
    runs: dict[str, CompactRun],
    labels: Sequence[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    keys = sorted(runs[MODEL_ORDER[0]].records)
    arrays: dict[str, Any] = {}
    if dataset == "bccc":
        for model in MODEL_ORDER:
            values = []
            for key in keys:
                record = runs[model].records[key]
                expected = int(record.expected_binary)
                predicted = bool(record.predicted)
                values.append(
                    (
                        int(expected == 1 and predicted),
                        int(expected == 0 and predicted),
                        int(expected == 1 and not predicted),
                        int(expected == 0 and not predicted),
                    )
                )
            arrays[model] = np.asarray(values, dtype=np.int32)
    else:
        assert labels is not None
        label_set = set(labels)
        for model in MODEL_ORDER:
            values = []
            for key in keys:
                record = runs[model].records[key]
                assert record.expected_multilabel is not None
                values.append(multi_record_counts(record.expected_multilabel, record.predicted, label_set))
            arrays[model] = np.asarray(values, dtype=np.int32)
    return keys, arrays


def vector_metric(np: Any, counts: Any, metric: str) -> Any:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts.reshape(1, -1)
    tp = counts[:, 0]
    fp = counts[:, 1]
    fn = counts[:, 2]
    if metric == "micro_f1":
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    tn = counts[:, 3]
    if metric == "balanced_accuracy":
        tpr_den = tp + fn
        tnr_den = tn + fp
        tpr = np.divide(tp, tpr_den, out=np.zeros_like(tp), where=tpr_den != 0)
        tnr = np.divide(tn, tnr_den, out=np.zeros_like(tp), where=tnr_den != 0)
        return (tpr + tnr) / 2.0
    if metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return np.divide(numerator, denominator, out=np.zeros_like(tp), where=denominator != 0)
    raise ValueError(f"Unsupported paired metric: {metric}")


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32)


def paired_resampling(
    np: Any,
    dataset: str,
    model_a: str,
    model_b: str,
    contributions_a: Any,
    contributions_b: Any,
    metrics: Sequence[str],
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    n = int(contributions_a.shape[0])
    totals_a = contributions_a.sum(axis=0, dtype=np.int64)
    totals_b = contributions_b.sum(axis=0, dtype=np.int64)
    observed = {
        metric: float(vector_metric(np, totals_a, metric)[0] - vector_metric(np, totals_b, metric)[0])
        for metric in metrics
    }
    observed_values = {
        metric: (
            float(vector_metric(np, totals_a, metric)[0]),
            float(vector_metric(np, totals_b, metric)[0]),
        )
        for metric in metrics
    }

    bootstrap_values = {metric: np.empty(bootstrap_repetitions, dtype=np.float64) for metric in metrics}
    bootstrap_rng = np.random.default_rng(stable_seed(seed, dataset, model_a, model_b, "bootstrap"))
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    position = 0
    while position < bootstrap_repetitions:
        current = min(batch_size, bootstrap_repetitions - position)
        weights = bootstrap_rng.multinomial(n, probabilities, size=current)
        sampled_a = weights @ contributions_a
        sampled_b = weights @ contributions_b
        for metric in metrics:
            bootstrap_values[metric][position : position + current] = (
                vector_metric(np, sampled_a, metric) - vector_metric(np, sampled_b, metric)
            )
        position += current

    permutation_extreme = {metric: 0 for metric in metrics}
    permutation_rng = np.random.default_rng(stable_seed(seed, dataset, model_a, model_b, "permutation"))
    difference = contributions_b.astype(np.int64) - contributions_a.astype(np.int64)
    position = 0
    while position < permutation_repetitions:
        current = min(batch_size, permutation_repetitions - position)
        swap_mask = permutation_rng.integers(0, 2, size=(current, n), dtype=np.int8)
        swap_delta = swap_mask @ difference
        perm_a = totals_a + swap_delta
        perm_b = totals_b - swap_delta
        for metric in metrics:
            permuted_delta = vector_metric(np, perm_a, metric) - vector_metric(np, perm_b, metric)
            permutation_extreme[metric] += int(np.count_nonzero(np.abs(permuted_delta) >= abs(observed[metric])))
        position += current

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = bootstrap_values[metric]
        lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
        probability_a_greater = float(np.mean(values > 0) + 0.5 * np.mean(values == 0))
        p_value = (permutation_extreme[metric] + 1) / (permutation_repetitions + 1)
        rows.append(
            {
                "dataset": DATASET_NAMES[dataset],
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "n": n,
                "model_a_value": observed_values[metric][0],
                "model_b_value": observed_values[metric][1],
                "observed_difference_a_minus_b": observed[metric],
                "paired_bootstrap_ci95_lower": float(lower),
                "paired_bootstrap_ci95_upper": float(upper),
                "bootstrap_probability_a_greater": probability_a_greater,
                "paired_permutation_p_value": p_value,
                "bootstrap_repetitions": bootstrap_repetitions,
                "permutation_repetitions": permutation_repetitions,
                "seed": seed,
            }
        )
    return rows


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["metric"]))].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: float(row["paired_permutation_p_value"]))
        running = 0.0
        total = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (total - index) * float(row["paired_permutation_p_value"]))
            running = max(running, adjusted)
            row["holm_adjusted_p_value"] = running
            row["significant_after_holm_0_05"] = running < 0.05


def paired_rows(
    np: Any,
    all_runs: dict[str, dict[str, CompactRun]],
    labels_by_dataset: dict[str, list[str]],
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_DIRS:
        _, arrays = contribution_arrays(np, dataset, all_runs[dataset], labels_by_dataset.get(dataset))
        metrics = ("balanced_accuracy", "mcc") if dataset == "bccc" else ("micro_f1",)
        for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
            print(f"Paired resampling: {DATASET_NAMES[dataset]} | {model_a} vs {model_b}", flush=True)
            rows.extend(
                paired_resampling(
                    np=np,
                    dataset=dataset,
                    model_a=model_a,
                    model_b=model_b,
                    contributions_a=arrays[model_a],
                    contributions_b=arrays[model_b],
                    metrics=metrics,
                    bootstrap_repetitions=bootstrap_repetitions,
                    permutation_repetitions=permutation_repetitions,
                    seed=seed,
                    batch_size=batch_size,
                )
            )
    holm_adjust(rows)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def deduplicate_consistent(
    rows: Sequence[dict[str, str]], identity_fields: Sequence[str], label: str
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in identity_fields)].append(row)
    kept: list[dict[str, str]] = []
    for identity, group in grouped.items():
        fingerprints = {
            tuple(sorted((key, value or "") for key, value in row.items())) for row in group
        }
        if len(fingerprints) != 1:
            raise ValueError(f"Conflicting duplicate rows in {label} for {identity}")
        kept.append(group[0])
    return kept


def support_aware_results(evaluation_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = deduplicate_consistent(
        read_csv(evaluation_dir / "prediction_summary.csv"),
        ("dataset", "system_type", "system"),
        "prediction_summary.csv",
    )
    per_class = deduplicate_consistent(
        read_csv(evaluation_dir / "prediction_per_class.csv"),
        ("dataset", "system_type", "system", "class"),
        "prediction_per_class.csv",
    )
    by_system: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    support_rows: list[dict[str, Any]] = []
    for row in per_class:
        support = int(float(row["support"]))
        output: dict[str, Any] = dict(row)
        output["metric_defined"] = support > 0
        output["precision_report"] = row["precision"] if support > 0 else ""
        output["recall_report"] = row["recall"] if support > 0 else ""
        output["f1_report"] = row["f1"] if support > 0 else ""
        output["zero_support_false_positive_records"] = int(float(row["fp"])) if support == 0 else 0
        support_rows.append(output)
        by_system[(row["dataset"], row["system_type"], row["system"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for row in summary:
        output = dict(row)
        group = by_system.get((row["dataset"], row["system_type"], row["system"]), [])
        if group:
            supported = [class_row for class_row in group if int(float(class_row["support"])) > 0]
            zero_support = [class_row["class"] for class_row in group if int(float(class_row["support"])) == 0]
            output["macro_precision_all_declared_legacy"] = row.get("macro_precision", "")
            output["macro_recall_all_declared_legacy"] = row.get("macro_recall", "")
            output["macro_f1_all_declared_legacy"] = row.get("macro_f1", "")
            output["macro_precision_supported_classes"] = sum(float(item["precision"]) for item in supported) / len(supported)
            output["macro_recall_supported_classes"] = sum(float(item["recall"]) for item in supported) / len(supported)
            output["macro_f1_supported_classes"] = sum(float(item["f1"]) for item in supported) / len(supported)
            output["supported_class_count"] = len(supported)
            output["declared_class_count"] = len(group)
            output["zero_support_classes"] = json.dumps(sorted(zero_support))
        summary_rows.append(output)
    return summary_rows, support_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refuse to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def primary_value(row: dict[str, Any]) -> float:
    return float(row["primary_value"])


def sensitivity_assessment(
    sensitivity: list[dict[str, Any]],
    audit: dict[str, Any],
    margin: float,
) -> dict[str, Any]:
    dataset_results: dict[str, Any] = {}
    overall = True
    for dataset in DATASET_NAMES.values():
        rows = [row for row in sensitivity if row["dataset"] == dataset]
        deltas = [
            abs(float(row["delta_primary_vs_final_all"]))
            for row in rows if row["policy"] != "final_all"
        ]
        rankings = audit[dataset]["rankings"]
        ranking_stable = all(order == rankings["final_all"] for order in rankings.values())
        max_delta = max(deltas, default=0.0)
        passed = ranking_stable and max_delta <= margin
        overall = overall and passed
        dataset_results[dataset] = {
            "equivalence_margin": margin,
            "max_absolute_primary_delta": max_delta,
            "ranking_stable": ranking_stable,
            "passed": passed,
            **audit[dataset],
        }
    return {
        "criterion": "Ranking unchanged and max |primary metric delta| <= equivalence margin across repair_as_empty and common_untouched policies.",
        "overall_passed": overall,
        "datasets": dataset_results,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    path: Path,
    repair_audit: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    assessment: dict[str, Any],
    paired: list[dict[str, Any]],
    support_summary: list[dict[str, Any]],
) -> None:
    repair_table = markdown_table(
        ("Dataset", "Model", "Touched records", "Rate", "Inference chunks", "Structural chunks"),
        [
            (
                str(row["dataset"]), str(row["model"]),
                f'{row["records_touched"]}/{row["records_total"]}',
                f'{100 * float(row["records_touched_rate"]):.2f}%',
                str(row["chunk_inference_actions"]), str(row["chunk_structural_actions"]),
            )
            for row in repair_audit
        ],
    )
    sensitivity_table = markdown_table(
        ("Dataset", "Model", "Policy", "n", "Primary metric", "Value", "Delta vs final"),
        [
            (
                str(row["dataset"]), str(row["model"]), str(row["policy"]), str(row["n"]),
                str(row["primary_metric"]), f'{float(row["primary_value"]):.4f}',
                f'{float(row["delta_primary_vs_final_all"]):+.4f}',
            )
            for row in sensitivity
        ],
    )
    paired_table = markdown_table(
        ("Dataset", "Metric", "A", "B", "Delta A-B", "95% paired CI", "Holm p", "Significant"),
        [
            (
                str(row["dataset"]), str(row["metric"]), str(row["model_a"]), str(row["model_b"]),
                f'{float(row["observed_difference_a_minus_b"]):+.4f}',
                f'[{float(row["paired_bootstrap_ci95_lower"]):+.4f}, {float(row["paired_bootstrap_ci95_upper"]):+.4f}]',
                f'{float(row["holm_adjusted_p_value"]):.6f}',
                "yes" if row["significant_after_holm_0_05"] else "no",
            )
            for row in paired
        ],
    )
    support_rows = [
        row for row in support_summary
        if row.get("zero_support_classes") not in (None, "", "[]")
    ]
    support_table = markdown_table(
        ("Dataset", "System", "Zero-support classes", "Legacy macro-F1", "Supported macro-F1"),
        [
            (
                str(row["dataset"]), str(row["system"]), str(row["zero_support_classes"]),
                f'{float(row["macro_f1_all_declared_legacy"]):.4f}',
                f'{float(row["macro_f1_supported_classes"]):.4f}',
            )
            for row in support_rows
        ],
    )
    report = f"""# Prediction robustness analysis

Generated at {datetime.now(timezone.utc).isoformat()}.

## Overall sensitivity decision

**Passed: {assessment['overall_passed']}**

Pre-specified rule: {assessment['criterion']}

## Selective-repair exposure

Action counts are chunk-level; touched rates are record-level.

{repair_table}

## Sensitivity metrics

`repair_as_empty` applies the evaluator's conservative no-accepted-finding policy to every record touched by inference or structural repair. `common_untouched` evaluates the identical record subset untouched for all three models.

{sensitivity_table}

## Paired model comparisons

Confidence intervals use paired record bootstrap. P-values use paired random-swap permutation tests and are Holm-adjusted within each dataset/metric family.

{paired_table}

## Zero-support correction

Recall and F1 are reported as undefined (`N/A`) for classes with zero ground-truth positives. Supported-class macro metrics exclude those classes; false positives remain visible in the per-class CSV.

{support_table}

## Interpretation rule

- If sensitivity passes, the principal ranking is robust to the selective-repair policy under the declared 0.01 margin.
- A paired difference is supported when its CI excludes zero and the Holm-adjusted p-value is below 0.05.
- If sensitivity fails, describe the experiment as a selectively repaired evaluation or perform a homogeneous full re-execution.
"""
    path.write_text(report, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--permutation-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--equivalence-margin", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")
    if args.permutation_repetitions < 100:
        parser.error("--permutation-repetitions must be at least 100")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 < args.equivalence_margin < 1:
        parser.error("--equivalence-margin must be between 0 and 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_root = args.prediction_root.resolve()
    output_root = args.output_root.resolve()
    evaluation_dir = prediction_root / "benchmark_evaluation"
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Final benchmark evaluation directory is missing: {evaluation_dir}")

    np = import_numpy()
    multilabel_truth, labels_by_dataset, truth_paths = load_multilabel_ground_truth()
    print(f"Loading compact records from {prediction_root}", flush=True)
    all_runs = load_all_runs(prediction_root, multilabel_truth)
    print("All 9 model/dataset runs passed identity and ground-truth alignment checks.", flush=True)

    repair_audit = repair_audit_rows(all_runs)
    sensitivity, sensitivity_audit = sensitivity_rows(all_runs, labels_by_dataset)
    assessment = sensitivity_assessment(sensitivity, sensitivity_audit, args.equivalence_margin)
    support_summary, support_per_class = support_aware_results(evaluation_dir)
    paired = paired_rows(
        np=np,
        all_runs=all_runs,
        labels_by_dataset=labels_by_dataset,
        bootstrap_repetitions=args.bootstrap_repetitions,
        permutation_repetitions=args.permutation_repetitions,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "repair_exposure.csv", repair_audit)
    write_csv(output_root / "repair_sensitivity.csv", sensitivity)
    write_csv(output_root / "paired_model_comparisons.csv", paired)
    write_csv(output_root / "support_aware_summary.csv", support_summary)
    write_csv(output_root / "support_aware_per_class.csv", support_per_class)
    (output_root / "robustness_summary.json").write_text(
        json.dumps(
            {
                "sensitivity_assessment": assessment,
                "repair_exposure": repair_audit,
                "sensitivity": sensitivity,
                "paired_comparisons": paired,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(
        output_root / "ROBUSTNESS_REPORT.md",
        repair_audit,
        sensitivity,
        assessment,
        paired,
        support_summary,
    )

    source_jsonl = [all_runs[dataset][model].path for dataset in DATASET_DIRS for model in MODEL_ORDER]
    generated = [
        output_root / name
        for name in (
            "repair_exposure.csv",
            "repair_sensitivity.csv",
            "paired_model_comparisons.csv",
            "support_aware_summary.csv",
            "support_aware_per_class.csv",
            "robustness_summary.json",
            "ROBUSTNESS_REPORT.md",
        )
    ]
    manifest = {
        "schema_version": "robustness-analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": repository_path(Path(__file__)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "prediction_root": repository_path(prediction_root),
        "parameters": {
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "permutation_repetitions": args.permutation_repetitions,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "equivalence_margin": args.equivalence_margin,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "ground_truth_files": [
            {"dataset": dataset, "path": repository_path(path), "sha256": sha256(path)}
            for dataset, path in truth_paths.items()
        ],
        "source_jsonl": [
            {"path": repository_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_jsonl
        ],
        "generated_files": [
            {"path": repository_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        ],
        "sensitivity_passed": assessment["overall_passed"],
    }
    manifest_path = output_root / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote robustness results to {output_root}")
    print(f"Sensitivity passed: {assessment['overall_passed']}")
    print(f"Report: {output_root / 'ROBUSTNESS_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
