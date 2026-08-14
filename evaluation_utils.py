# Copyright 2026 Thi-Thu-Huong Le
# SPDX-License-Identifier: Apache-2.0

"""Shared metric and reference-loading utilities for benchmark evaluation."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = REPOSITORY_ROOT / "results"

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


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def code_digest(code: str) -> str:
    normalized = code.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_prediction(category: object) -> str | None:
    text = " ".join(
        "".join(char if char.isalnum() else " " for char in str(category).lower()).split()
    )
    if text == "other":
        return "other"
    if "reentr" in text:
        return "reentrancy"
    if any(token in text for token in ("overflow", "underflow", "arithmetic", "arithm")):
        return "arithmetic"
    if any(
        token in text
        for token in (
            "unchecked low level",
            "unchecked ll",
            "unhandled exception",
            "unchecked call",
            "unchecked external call",
        )
    ):
        return "unchecked_low_level_calls"
    if "locked ether" in text:
        return "locked_ether"
    if "randomness" in text:
        return "bad_randomness"
    if "denial of service" in text or text == "dos":
        return "denial_of_service"
    if any(token in text for token in ("front running", "front run", "transaction order")):
        return "transaction_ordering"
    if any(token in text for token in ("time manipulation", "timestamp dependence", "block timestamp")):
        return "time_manipulation"
    if "tx origin" in text or "txorigin" in text:
        return "tx_origin"
    if "short address" in text:
        return "short_addresses"
    if any(
        token in text
        for token in (
            "access control",
            "authorization",
            "unauthorized",
            "privilege escalation",
            "unprotected function",
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
    raise ValueError(f"Cannot infer model name from {path.name}")


def summarize_pairs(
    pairs: Sequence[tuple[set[str], set[str]]], labels: Sequence[str]
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    class_rows: list[dict[str, float | int | str]] = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(label in expected and label in predicted for expected, predicted in pairs)
        fp = sum(label not in expected and label in predicted for expected, predicted in pairs)
        fn = sum(label in expected and label not in predicted for expected, predicted in pairs)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        class_rows.append(
            {
                "class": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": safe_divide(tp, tp + fp),
                "recall": safe_divide(tp, tp + fn),
                "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
                "support": tp + fn,
            }
        )

    exact_match = safe_divide(
        sum(expected == predicted for expected, predicted in pairs), len(pairs)
    )
    sample_f1 = safe_divide(
        sum(
            1.0
            if not expected and not predicted
            else safe_divide(2 * len(expected & predicted), len(expected) + len(predicted))
            for expected, predicted in pairs
        ),
        len(pairs),
    )
    summary = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "micro_precision": safe_divide(total_tp, total_tp + total_fp),
        "micro_recall": safe_divide(total_tp, total_tp + total_fn),
        "micro_f1": safe_divide(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        "macro_precision": safe_divide(
            sum(float(row["precision"]) for row in class_rows), len(class_rows)
        ),
        "macro_recall": safe_divide(
            sum(float(row["recall"]) for row in class_rows), len(class_rows)
        ),
        "macro_f1": safe_divide(
            sum(float(row["f1"]) for row in class_rows), len(class_rows)
        ),
        "exact_match_ratio": exact_match,
        "sample_f1": sample_f1,
    }
    return summary, class_rows


def _percentile(values: Sequence[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = proportion * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_micro_f1(
    pairs: Sequence[tuple[set[str], set[str]]],
    labels: Sequence[str],
    *,
    iterations: int = 1000,
    seed: int = 2026,
) -> tuple[float, float]:
    if not pairs:
        return 0.0, 0.0
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        summary, _ = summarize_pairs(sample, labels)
        values.append(float(summary["micro_f1"]))
    return _percentile(values, 0.025), _percentile(values, 0.975)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _deduplicate(rows: Iterable[dict[str, str]], keys: Sequence[str]) -> list[dict[str, str]]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in keys)
        selected.setdefault(key, row)
    return list(selected.values())


def load_scrawld_static_baseline_references() -> tuple[list[dict], list[dict]]:
    """Load the frozen non-independent ScrawlD constituent-tool references.

    The retained analysis-ready ScrawlD file contains consensus labels but not
    every constituent tool's raw report, so these reference rows are loaded
    from the frozen evaluation output rather than reconstructed.
    """

    evaluation = RESULT_ROOT / "final_benchmark" / "benchmark_evaluation"
    summary_path = evaluation / "prediction_summary.csv"
    per_class_path = evaluation / "prediction_per_class.csv"
    if not summary_path.is_file() or not per_class_path.is_file():
        raise FileNotFoundError(
            "Frozen ScrawlD constituent-tool references are unavailable. "
            "Run without --include-static-baselines or provide the archived evaluation files."
        )
    summaries = _deduplicate(
        (
            row
            for row in _read_csv(summary_path)
            if row.get("dataset") == "ScrawlD" and row.get("system_type") == "static_baseline"
        ),
        ("dataset", "system_type", "system"),
    )
    per_class = _deduplicate(
        (
            row
            for row in _read_csv(per_class_path)
            if row.get("dataset") == "ScrawlD" and row.get("system_type") == "static_baseline"
        ),
        ("dataset", "system_type", "system", "class"),
    )
    return summaries, per_class
