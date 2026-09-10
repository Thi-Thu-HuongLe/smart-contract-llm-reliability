#!/usr/bin/env python3
"""Create publication-ready figures from the final benchmark evaluation.

The script intentionally reads only the four evaluator outputs.  It does not
recompute predictions and never scans the large JSONL files.  Before plotting,
it verifies that every expected model/dataset pair is complete and failure-free.

Examples
--------
Validate only::

    python plot_benchmark_results.py --validate-only

Render PDF, SVG, and 600-dpi PNG figures::

    python plot_benchmark_results.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASETS = ("SmartBugs-Curated", "ScrawlD", "BCCC")
MODELS = (
    "Qwen2.5-Coder-7B-Instruct",
    "CodeLlama-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
)
MODEL_SHORT = {
    "Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder",
    "CodeLlama-7B-Instruct": "CodeLlama",
    "Mistral-7B-Instruct-v0.3": "Mistral",
}
MODEL_COLORS = {
    "Qwen2.5-Coder-7B-Instruct": "#0072B2",  # Okabe-Ito blue
    "CodeLlama-7B-Instruct": "#D55E00",     # Okabe-Ito vermilion
    "Mistral-7B-Instruct-v0.3": "#009E73",  # Okabe-Ito green
}
MODEL_MARKERS = {
    "Qwen2.5-Coder-7B-Instruct": "o",
    "CodeLlama-7B-Instruct": "s",
    "Mistral-7B-Instruct-v0.3": "^",
}
DATASET_SHORT = {
    "SmartBugs-Curated": "SmartBugs-Curated",
    "ScrawlD": "ScrawlD",
    "BCCC": "BCCC",
}
CLASS_LABELS = {
    "access_control": "Access control",
    "arithmetic": "Arithmetic",
    "bad_randomness": "Bad randomness",
    "denial_of_service": "Denial of service",
    "locked_ether": "Locked Ether",
    "other": "Other",
    "reentrancy": "Reentrancy",
    "short_addresses": "Short addresses",
    "time_manipulation": "Time manipulation",
    "transaction_ordering": "Transaction ordering",
    "tx_origin": "tx.origin",
    "unchecked_low_level_calls": "Unchecked calls",
}

SUMMARY_FILE = "prediction_summary.csv"
PER_CLASS_FILE = "prediction_per_class.csv"
COVERAGE_FILE = "prediction_coverage.csv"
SUMMARY_JSON_FILE = "prediction_summary.json"


@dataclass(frozen=True)
class Inputs:
    evaluation_dir: Path
    summary: list[dict[str, str]]
    per_class: list[dict[str, str]]
    coverage: list[dict[str, str]]
    duplicate_rows_removed: dict[str, int]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluator output is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Evaluator output has no data rows: {path}")
    return rows


def require_columns(rows: Sequence[dict[str, str]], required: Iterable[str], label: str) -> None:
    present = set(rows[0]) if rows else set()
    missing = sorted(set(required) - present)
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def as_int(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError(f"Missing integer field {field!r} in row: {row}")
    return int(float(value))


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError(f"Missing numeric field {field!r} in row: {row}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value in {field!r}: {value}")
    return result


def _deduplicate_consistent(
    rows: Sequence[dict[str, str]], identity_fields: Sequence[str], label: str
) -> tuple[list[dict[str, str]], int]:
    """Drop byte-equivalent logical duplicates and reject conflicting ones."""
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in identity_fields)].append(row)

    kept: list[dict[str, str]] = []
    removed = 0
    for identity, group in grouped.items():
        normalized = {
            tuple(sorted((key, value or "") for key, value in row.items()))
            for row in group
        }
        if len(normalized) != 1:
            raise ValueError(
                f"Conflicting duplicate {label} rows for identity {identity}; "
                "refuse to choose one silently."
            )
        kept.append(group[0])
        removed += len(group) - 1
    return kept, removed


def load_inputs(evaluation_dir: Path) -> Inputs:
    evaluation_dir = evaluation_dir.resolve()
    summary_raw = read_csv(evaluation_dir / SUMMARY_FILE)
    per_class_raw = read_csv(evaluation_dir / PER_CLASS_FILE)
    coverage_raw = read_csv(evaluation_dir / COVERAGE_FILE)
    summary_json = evaluation_dir / SUMMARY_JSON_FILE
    if not summary_json.is_file():
        raise FileNotFoundError(f"Required evaluator output is missing: {summary_json}")

    require_columns(
        summary_raw,
        (
            "dataset", "system_type", "system", "ground_truth_records",
            "prediction_records", "matched_records", "missing_prediction_records",
            "out_of_taxonomy_findings", "ok_output_records",
            "partial_output_records", "failed_output_records", "other_status_records",
            "tp", "fp", "fn", "micro_precision", "micro_recall", "micro_f1",
            "micro_f1_ci95_lower", "micro_f1_ci95_upper", "tn", "precision",
            "recall", "f1", "specificity", "accuracy",
        ),
        SUMMARY_FILE,
    )
    require_columns(
        per_class_raw,
        ("class", "f1", "support", "dataset", "system_type", "system"),
        PER_CLASS_FILE,
    )
    require_columns(
        coverage_raw,
        (
            "dataset", "system", "records_expected", "records_written",
            "records_missing", "records_with_findings", "ok_output_records",
            "partial_output_records", "failed_output_records", "other_status_records",
        ),
        COVERAGE_FILE,
    )

    summary, summary_removed = _deduplicate_consistent(
        summary_raw, ("dataset", "system_type", "system"), SUMMARY_FILE
    )
    per_class, per_class_removed = _deduplicate_consistent(
        per_class_raw,
        ("dataset", "system_type", "system", "class"),
        PER_CLASS_FILE,
    )
    coverage, coverage_removed = _deduplicate_consistent(
        coverage_raw, ("dataset", "system"), COVERAGE_FILE
    )
    return Inputs(
        evaluation_dir=evaluation_dir,
        summary=summary,
        per_class=per_class,
        coverage=coverage,
        duplicate_rows_removed={
            SUMMARY_FILE: summary_removed,
            PER_CLASS_FILE: per_class_removed,
            COVERAGE_FILE: coverage_removed,
        },
    )


def validate_complete_predictions(inputs: Inputs) -> dict[str, Any]:
    """Apply the publication data-quality gate and return a compact audit."""
    llm_rows = [row for row in inputs.summary if row.get("system_type") == "code_llm"]
    by_pair = {(row["dataset"], row["system"]): row for row in llm_rows}
    expected_pairs = {(dataset, model) for dataset in DATASETS for model in MODELS}
    found_pairs = set(by_pair)
    if found_pairs != expected_pairs:
        raise ValueError(
            "Unexpected LLM summary coverage. "
            f"Missing={sorted(expected_pairs - found_pairs)}; "
            f"extra={sorted(found_pairs - expected_pairs)}"
        )

    coverage_by_pair = {(row["dataset"], row["system"]): row for row in inputs.coverage}
    if set(coverage_by_pair) != expected_pairs:
        raise ValueError(
            "Unexpected coverage rows. "
            f"Missing={sorted(expected_pairs - set(coverage_by_pair))}; "
            f"extra={sorted(set(coverage_by_pair) - expected_pairs)}"
        )

    problems: list[str] = []
    record_counts: dict[str, int] = {}
    for pair in sorted(expected_pairs):
        summary = by_pair[pair]
        coverage = coverage_by_pair[pair]
        dataset, model = pair
        expected = as_int(summary, "ground_truth_records")
        record_counts.setdefault(dataset, expected)
        if record_counts[dataset] != expected:
            problems.append(f"{pair}: inconsistent ground-truth record count")

        count_fields = {
            "prediction_records": as_int(summary, "prediction_records"),
            "matched_records": as_int(summary, "matched_records"),
            "ok_output_records": as_int(summary, "ok_output_records"),
            "coverage.records_expected": as_int(coverage, "records_expected"),
            "coverage.records_written": as_int(coverage, "records_written"),
            "coverage.ok_output_records": as_int(coverage, "ok_output_records"),
        }
        for field, value in count_fields.items():
            if value != expected:
                problems.append(f"{dataset}/{model}: {field}={value}, expected {expected}")

        zero_fields = {
            "missing_prediction_records": as_int(summary, "missing_prediction_records"),
            "out_of_taxonomy_findings": as_int(summary, "out_of_taxonomy_findings"),
            "partial_output_records": as_int(summary, "partial_output_records"),
            "failed_output_records": as_int(summary, "failed_output_records"),
            "other_status_records": as_int(summary, "other_status_records"),
            "coverage.records_missing": as_int(coverage, "records_missing"),
            "coverage.partial_output_records": as_int(coverage, "partial_output_records"),
            "coverage.failed_output_records": as_int(coverage, "failed_output_records"),
            "coverage.other_status_records": as_int(coverage, "other_status_records"),
        }
        for field, value in zero_fields.items():
            if value != 0:
                problems.append(f"{dataset}/{model}: {field}={value}, expected 0")

        if dataset == "BCCC":
            for field in ("precision", "recall", "f1", "specificity", "accuracy", "tn"):
                as_float(summary, field)
        else:
            for field in (
                "micro_precision", "micro_recall", "micro_f1",
                "micro_f1_ci95_lower", "micro_f1_ci95_upper",
            ):
                value = as_float(summary, field)
                if not 0.0 <= value <= 1.0:
                    problems.append(f"{dataset}/{model}: {field}={value} outside [0, 1]")

    if problems:
        formatted = "\n  - ".join(problems)
        raise ValueError(f"Publication quality gate failed:\n  - {formatted}")

    static_rows = [row for row in inputs.summary if row.get("system_type") == "static_baseline"]
    static_pairs = {(row["dataset"], row["system"]) for row in static_rows}
    expected_static = {
        ("ScrawlD", name) for name in ("mythril", "osiris", "oyente", "slither", "smartcheck")
    }
    if static_pairs != expected_static:
        raise ValueError(
            "Static-baseline set is incomplete or unexpected after de-duplication. "
            f"Found={sorted(static_pairs)}"
        )

    expected_per_class_pairs = {
        (dataset, model) for dataset in DATASETS[:2] for model in MODELS
    }
    per_class_pairs = {
        (row["dataset"], row["system"])
        for row in inputs.per_class
        if row.get("system_type") == "code_llm"
    }
    if per_class_pairs != expected_per_class_pairs:
        raise ValueError(
            "Per-class LLM rows are incomplete. "
            f"Missing={sorted(expected_per_class_pairs - per_class_pairs)}"
        )

    return {
        "quality_gate_passed": True,
        "llm_pairs": len(expected_pairs),
        "static_baselines": len(expected_static),
        "record_counts": record_counts,
        "duplicate_rows_removed": inputs.duplicate_rows_removed,
    }


def _import_plotting() -> tuple[Any, Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Visualization dependencies are missing. Install them with:\n"
            "  python -m pip install -r requirements-visualization.txt"
        ) from error
    return matplotlib, plt, np, mcolors


def configure_style(matplotlib: Any) -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7.4,
            "legend.frameon": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _llm_summary(inputs: Inputs) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["dataset"], row["system"]): row
        for row in inputs.summary
        if row.get("system_type") == "code_llm"
    }


def metric_triplet(row: dict[str, str], dataset: str) -> tuple[float, float, float]:
    prefix = "" if dataset == "BCCC" else "micro_"
    return tuple(as_float(row, f"{prefix}{field}") for field in ("precision", "recall", "f1"))


def add_panel_labels_below(
    fig: Any,
    axes: Sequence[Any],
    labels: Sequence[str],
    y_positions: float | Sequence[float],
) -> None:
    """Center parenthesized panel labels in a dedicated row below each panel.

    The y positions are figure coordinates chosen after ``subplots_adjust`` so
    panel labels cannot collide with axes titles.  A sequence is supported for
    vertically stacked panels, whose labels require separate rows.
    """
    if isinstance(y_positions, (int, float)):
        resolved_y = [float(y_positions)] * len(axes)
    else:
        resolved_y = [float(value) for value in y_positions]
    if len(axes) != len(labels) or len(axes) != len(resolved_y):
        raise ValueError("Panel axes, labels, and y positions must have equal lengths")

    for ax, label, y_position in zip(axes, labels, resolved_y):
        bounds = ax.get_position()
        fig.text(
            bounds.x0 + bounds.width / 2.0,
            y_position,
            f"({label})",
            fontsize=8.0,
            fontweight="semibold",
            va="bottom",
            ha="center",
            color="#222222",
        )


def _iso_f1_curve(np: Any, level: float) -> tuple[Any, Any]:
    precision = np.linspace(level / 2.0 + 0.006, 1.0, 400)
    recall = level * precision / (2.0 * precision - level)
    keep = (recall >= 0.0) & (recall <= 1.0)
    return precision[keep], recall[keep]


def figure_precision_recall(inputs: Inputs, plt: Any, np: Any) -> Any:
    rows = _llm_summary(inputs)
    counts = validate_record_counts(inputs)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharex=True, sharey=True)
    for panel_index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        for level in (0.2, 0.4, 0.6, 0.8):
            x, y = _iso_f1_curve(np, level)
            ax.plot(x, y, color="#C8C8C8", lw=0.6, ls=(0, (2, 2)), zorder=0)
            if panel_index == 2 and len(x):
                ax.text(x[-1] - 0.02, y[-1] + 0.02, f"F1={level:.1f}", color="#888888", fontsize=5.8, ha="right")

        for model in MODELS:
            precision, recall, _ = metric_triplet(rows[(dataset, model)], dataset)
            ax.scatter(
                precision, recall, s=43, marker=MODEL_MARKERS[model],
                facecolor=MODEL_COLORS[model], edgecolor="white", linewidth=0.7,
                zorder=3, label=MODEL_SHORT[model],
            )
        ax.set_title(f"{DATASET_SHORT[dataset]}  (n={counts[dataset]:,})")
        ax.set_xlim(-0.015, 1.035)
        ax.set_ylim(-0.015, 1.035)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.45)
        ax.set_xlabel("Precision")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
        ncol=3, columnspacing=1.4, handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.81, wspace=0.17)
    add_panel_labels_below(fig, axes, ("a", "b", "c"), 0.035)
    return fig


def validate_record_counts(inputs: Inputs) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in inputs.summary:
        if row.get("system_type") != "code_llm":
            continue
        count = as_int(row, "ground_truth_records")
        dataset = row["dataset"]
        if dataset in counts and counts[dataset] != count:
            raise ValueError(f"Inconsistent record counts for {dataset}")
        counts[dataset] = count
    return counts


def _point_interval(ax: Any, row: dict[str, str], y: float, color: str, marker: str) -> None:
    value = as_float(row, "micro_f1")
    lower = as_float(row, "micro_f1_ci95_lower")
    upper = as_float(row, "micro_f1_ci95_upper")
    ax.errorbar(
        value, y, xerr=[[value - lower], [upper - value]], fmt=marker,
        ms=5.3, mfc=color, mec="white", mew=0.6, ecolor=color,
        elinewidth=1.25, capsize=2.2, capthick=0.8, zorder=3,
    )
    ax.text(min(upper + 0.018, 0.955), y, f"{value:.3f}", va="center", fontsize=6.6, color="#333333")


def figure_f1_intervals(inputs: Inputs, plt: Any, np: Any) -> Any:
    smart = [
        row for row in inputs.summary
        if row["dataset"] == "SmartBugs-Curated" and row["system_type"] == "code_llm"
    ]
    scrawld = [row for row in inputs.summary if row["dataset"] == "ScrawlD"]
    smart.sort(key=lambda row: as_float(row, "micro_f1"), reverse=True)
    scrawld.sort(key=lambda row: as_float(row, "micro_f1"), reverse=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.60), sharex=True,
        gridspec_kw={"width_ratios": [0.78, 1.55]},
    )
    for panel_index, (ax, rows, title) in enumerate(
        zip(axes, (smart, scrawld), ("SmartBugs-Curated", "ScrawlD"))
    ):
        y_positions = np.arange(len(rows))[::-1]
        for y, row in zip(y_positions, rows):
            is_llm = row["system_type"] == "code_llm"
            color = MODEL_COLORS.get(row["system"], "#6F6F6F")
            marker = MODEL_MARKERS.get(row["system"], "D")
            _point_interval(ax, row, float(y), color, marker)
            if not is_llm:
                # Repaint static baselines as hollow diamonds for grayscale legibility.
                ax.plot(as_float(row, "micro_f1"), y, marker="D", ms=4.5, mfc="white", mec="#555555", mew=0.9, zorder=4)

        labels = [
            MODEL_SHORT.get(row["system"], row["system"].replace("smartcheck", "SmartCheck").title())
            for row in rows
        ]
        ax.set_yticks(y_positions, labels)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(True, axis="x")
        ax.set_xlabel("Micro-F1")
        ax.set_title(title)
        ax.set_ylim(-0.65, len(rows) - 0.35)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.20, top=0.93, wspace=0.48)
    add_panel_labels_below(fig, axes, ("a", "b"), 0.075)
    return fig


def _wrap_class_label(label: str) -> str:
    replacements = {
        "Access control": "Access\ncontrol",
        "Bad randomness": "Bad\nrandomness",
        "Denial of service": "Denial-of-\nservice",
        "Locked Ether": "Locked\nEther",
        "Short addresses": "Short\naddress",
        "Time manipulation": "Timestamp\nmanip.",
        "Transaction ordering": "Txn.\nordering",
        "Unchecked calls": "Unchecked\ncalls",
    }
    return replacements.get(label, label)


def figure_per_class_dataset(
    inputs: Inputs, plt: Any, np: Any, mcolors: Any, dataset: str
) -> Any:
    """Render one dataset heatmap for assembly with LaTeX subfigure."""
    if dataset not in DATASETS[:2]:
        raise ValueError(f"Per-class heatmap is unavailable for {dataset!r}")
    rows = [row for row in inputs.per_class if row.get("system_type") == "code_llm"]
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    classes = sorted(
        {row["class"] for row in dataset_rows},
        key=lambda name: CLASS_LABELS.get(name, name).lower(),
    )
    supports: dict[str, int] = {}
    matrix = np.full((len(MODELS), len(classes)), np.nan, dtype=float)
    for model_index, model in enumerate(MODELS):
        by_class = {
            row["class"]: row for row in dataset_rows if row["system"] == model
        }
        if set(by_class) != set(classes):
            raise ValueError(f"Incomplete per-class matrix for {dataset}/{model}")
        for class_index, class_name in enumerate(classes):
            support = as_int(by_class[class_name], "support")
            if class_name in supports and supports[class_name] != support:
                raise ValueError(f"Inconsistent support for {dataset}/{class_name}")
            supports[class_name] = support
            if support > 0:
                matrix[model_index, class_index] = as_float(by_class[class_name], "f1")

    fig, ax = plt.subplots(figsize=(7.2, 2.48))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "benchmark_f1", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    )
    cmap.set_bad("#E6E6E6")
    masked_matrix = np.ma.masked_invalid(matrix)
    image = ax.imshow(
        masked_matrix, cmap=cmap, vmin=0.0, vmax=1.0,
        aspect="auto", interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            undefined = not np.isfinite(value)
            ax.text(
                col_index, row_index, "N/A" if undefined else f"{value:.2f}",
                ha="center", va="center", fontsize=6.4,
                color="#555555" if undefined else ("white" if value >= 0.58 else "#222222"),
                fontweight="semibold" if not undefined and value >= 0.8 else "normal",
            )
    ax.set_yticks(np.arange(len(MODELS)), [MODEL_SHORT[model] for model in MODELS])
    labels = [
        f"{_wrap_class_label(CLASS_LABELS.get(name, name))}\n(n={supports[name]:,})"
        for name in classes
    ]
    ax.set_xticks(np.arange(len(classes)), labels, fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar_ax = fig.add_axes([0.925, 0.34, 0.012, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Per-class F1", rotation=90, labelpad=6)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.16, right=0.91, bottom=0.34, top=0.94)
    return fig


def figure_coverage(inputs: Inputs, plt: Any, np: Any) -> Any:
    coverage = {(row["dataset"], row["system"]): row for row in inputs.coverage}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.90), sharex=True, sharey=True)
    for panel_index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        y_positions = np.arange(len(MODELS))[::-1]
        for y, model in zip(y_positions, MODELS):
            row = coverage[(dataset, model)]
            written = as_int(row, "records_written")
            processed = 100.0 * as_int(row, "ok_output_records") / written
            findings = 100.0 * as_int(row, "records_with_findings") / written
            ax.plot([findings, processed], [y, y], color="#B8B8B8", lw=1.2, zorder=1)
            ax.scatter(
                findings, y, s=37, marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model], edgecolor="white", linewidth=0.6, zorder=3,
            )
            ax.scatter(processed, y, s=27, marker="D", facecolor="white", edgecolor="#444444", linewidth=0.8, zorder=2)
            dx = -4 if findings > 94 else 4
            ha = "right" if findings > 94 else "left"
            # Put the top-row annotation below its marker so it cannot collide
            # with the panel heading. Lower rows retain the above-marker label.
            dy = -8 if y == y_positions[0] else 6
            va = "top" if y == y_positions[0] else "bottom"
            ax.annotate(
                f"{findings:.1f}%", (findings, y), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va=va, fontsize=6.2,
                color=MODEL_COLORS[model], fontweight="semibold",
            )
        ax.set_title(DATASET_SHORT[dataset])
        ax.set_yticks(y_positions, [MODEL_SHORT[model] for model in MODELS])
        ax.set_xlim(-2, 103)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(True, axis="x")
        ax.set_xlabel("Records (%)")
        ax.tick_params(axis="y", length=0)
    axes[0].scatter([], [], s=36, marker="o", color="#666666", label="At least one finding")
    axes[0].scatter([], [], s=27, marker="D", facecolor="white", edgecolor="#444444", label="Valid processed output")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.96), ncol=2, handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.25, top=0.78, wspace=0.18)
    add_panel_labels_below(fig, axes, ("a", "b", "c"), 0.035)
    return fig


def matthews_correlation(tp: int, fp: int, fn: int, tn: int) -> float:
    numerator = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator else 0.0


def bccc_discrimination_metrics(inputs: Inputs) -> list[dict[str, float | str]]:
    rows = _llm_summary(inputs)
    metrics: list[dict[str, float | str]] = []
    for model in MODELS:
        row = rows[("BCCC", model)]
        tp, fp, fn, tn = (as_int(row, field) for field in ("tp", "fp", "fn", "tn"))
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        youden_j = tpr - fpr
        mcc = matthews_correlation(tp, fp, fn, tn)
        balanced_accuracy = (tpr + (tn / (tn + fp))) / 2.0
        metrics.append(
            {
                "model": model, "tpr": tpr, "fpr": fpr,
                "youden_j": youden_j, "mcc": mcc,
                "balanced_accuracy": balanced_accuracy,
            }
        )
    return metrics


def figure_bccc_discrimination(
    metrics: Sequence[dict[str, float | str]], plt: Any, np: Any
) -> Any:
    """Render BCCC rate separation and scalar summaries without an image title."""
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    y_positions = np.arange(len(MODELS))[::-1]
    by_model = {str(row["model"]): row for row in metrics}
    for y, model in zip(y_positions, MODELS):
        row = by_model[model]
        tpr = float(row["tpr"])
        fpr = float(row["fpr"])
        ax.plot([fpr, tpr], [y, y], color=MODEL_COLORS[model], lw=2.2, alpha=0.55, zorder=1)
        ax.scatter(fpr, y, marker="s", s=46, facecolor="white", edgecolor=MODEL_COLORS[model], linewidth=1.2, zorder=3)
        ax.scatter(tpr, y, marker="o", s=46, facecolor=MODEL_COLORS[model], edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(
            1.045, y,
            f"J={float(row['youden_j']):+.3f}   "
            f"MCC={float(row['mcc']):+.3f}   "
            f"BA={float(row['balanced_accuracy']):.3f}",
            transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=6.7, color="#333333", clip_on=False,
        )

    ax.axvline(0.5, color="#A0A0A0", lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.set_yticks(y_positions, [MODEL_SHORT[model] for model in MODELS])
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.65, len(MODELS) - 0.35)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Rate")
    ax.grid(True, axis="x")
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], marker="o", s=42, color="#666666", label="True-positive rate (recall)")
    ax.scatter([], [], marker="s", s=42, facecolor="white", edgecolor="#666666", label="False-positive rate")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.50, 1.08), ncol=2,
        columnspacing=1.4, handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.16, right=0.68, bottom=0.22, top=0.78)
    return fig


def save_figure(fig: Any, output_base: Path, formats: Sequence[str], dpi: int, plt: Any) -> list[Path]:
    written: list[Path] = []
    for extension in formats:
        path = output_base.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.04}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_clean_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_bccc_derived(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    fields = ("model", "tpr", "fpr", "youden_j", "mcc", "balanced_accuracy")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_captions(path: Path) -> None:
    path.write_text(
        """# Suggested captions for the benchmark figures

## LaTeX placement note

Panel labels are embedded below the multi-panel assets for precision--recall, confidence intervals, and coverage semantics. The per-class heatmaps are exported as two independent assets and must be assembled with LaTeX `subfigure`; their `(a)` and `(b)` labels therefore come only from the subfigure captions. Place each `\\label{...}` immediately after its caption.

## Figure 1 - Precision-recall landscape

Failure-aware precision and recall for the three code LLMs on (a) SmartBugs-Curated, (b) ScrawlD, and (c) BCCC. Curves show equal-F1 contours. SmartBugs-Curated and ScrawlD use closed-taxonomy micro-averaged metrics; BCCC uses contract-level binary metrics over the balanced vulnerable and secure splits. All nine model-dataset pairs completed the full manifest with no missing records, no partial or failed outputs, and no out-of-taxonomy findings. The results include targeted selective repair; see the repair-sensitivity analysis.

## Figure 2 - F1 confidence intervals

Micro-averaged F1 and 95% bootstrap confidence intervals on (a) SmartBugs-Curated and (b) ScrawlD. Filled markers denote code LLMs. Hollow diamonds denote reconstructed ScrawlD constituent-tool baselines; these comparisons are non-independent because the tools contributed to the ScrawlD consensus labels.

## Figure 3 - Per-class F1 (two LaTeX subfigures)

Per-vulnerability F1 for the three code LLMs. (a) SmartBugs-Curated contains ten supported vulnerability classes. (b) ScrawlD contains seven supported classes and one zero-support class. Parenthetical values report class support. Gray N/A cells denote classes with no ground-truth positives, for which recall and F1 are undefined. Both independently rendered panels use the same 0--1 color scale.

## Figure 4 - Coverage semantics

Valid processed-output coverage versus the proportion of contracts receiving at least one predicted finding on (a) SmartBugs-Curated, (b) ScrawlD, and (c) BCCC. Every model produced a valid output for every manifest record (hollow diamonds at 100%), while positive-finding coverage varied substantially. The distinction prevents predicted-positive counts from being misreported as analysis coverage.

## Figure 5 - BCCC discrimination

BCCC true-positive and false-positive rates over 9,756 vulnerable and 9,756 secure contracts. Youden's J, Matthews correlation coefficient (MCC), and balanced accuracy summarize separation between the two classes; larger TPR--FPR separation is preferable.
""",
        encoding="utf-8",
    )


def write_latex_includes(path: Path) -> None:
    """Write safe whole-figure LaTeX blocks with caption-before-label order."""
    path.write_text(
        r"""% Generated by plot_benchmark_results.py.
% Per-class heatmaps are separate assets and require subcaption.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig01_precision_recall_landscape.pdf}
  \caption{Failure-aware precision and recall for the three code LLMs on (a) SmartBugs-Curated, (b) ScrawlD, and (c) BCCC. Curves show equal-F1 contours. SmartBugs-Curated and ScrawlD use closed-taxonomy micro-averaged metrics; BCCC uses contract-level binary metrics over the balanced vulnerable and secure splits.}
  \label{fig:precision-recall}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig02_f1_confidence_intervals.pdf}
  \caption{Micro-averaged F1 and 95\% bootstrap confidence intervals on (a) SmartBugs-Curated and (b) ScrawlD. Filled markers denote code LLMs. Hollow diamonds denote reconstructed ScrawlD constituent-tool baselines; these comparisons are non-independent because the tools contributed to the ScrawlD consensus labels.}
  \label{fig:f1-confidence-intervals}
\end{figure*}

\begin{figure*}[t]
  \centering
  \begin{subfigure}[t]{\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig03a_per_class_f1_smartbugs.pdf}
    \caption{SmartBugs-Curated.}
    \label{fig:per-class-smartbugs}
  \end{subfigure}
  \par\medskip
  \begin{subfigure}[t]{\textwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig03b_per_class_f1_scrawld.pdf}
    \caption{ScrawlD.}
    \label{fig:per-class-scrawld}
  \end{subfigure}
  \caption{Per-vulnerability F1 for the three code LLMs. (a) SmartBugs-Curated contains ten supported vulnerability classes. (b) ScrawlD contains seven supported classes and one zero-support class. Parenthetical values report class support. Gray \texttt{N/A} cells denote classes with no ground-truth positives, for which recall and F1 are undefined. Both independently rendered panels use the same 0--1 color scale.}
  \label{fig:per-class-f1}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig04_coverage_semantics.pdf}
  \caption{Valid processed-output coverage versus the proportion of contracts receiving at least one predicted finding on (a) SmartBugs-Curated, (b) ScrawlD, and (c) BCCC. Every model produced a valid output for every manifest record (hollow diamonds at 100\%), while positive-finding coverage varied substantially. The distinction prevents predicted-positive counts from being misreported as analysis coverage.}
  \label{fig:coverage-semantics}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig05_bccc_discrimination.pdf}
  \caption{BCCC binary discrimination over 9,756 vulnerable and 9,756 secure contracts. Larger TPR--FPR separation, Youden's J, Matthews correlation coefficient (MCC), and balanced accuracy indicate stronger discrimination.}
  \label{fig:bccc-discrimination}
\end{figure*}
""",
        encoding="utf-8",
    )


def default_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    evaluation_dir = script_dir / "results" / "final_benchmark" / "benchmark_evaluation"
    output_dir = script_dir / "outputs" / "figures" / "benchmark"
    return evaluation_dir, output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_evaluation, default_output = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path, default=default_evaluation,
        help=f"Directory containing evaluator CSV/JSON outputs (default: {default_evaluation})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_output,
        help=f"Figure output directory (default: {default_output})",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "svg", "png"),
        default=("pdf", "svg", "png"), help="Output formats (default: pdf svg png)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600)")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Run the quality gate without importing Matplotlib or writing files",
    )
    args = parser.parse_args(argv)
    if args.dpi < 150:
        parser.error("--dpi must be at least 150 for publication output")
    args.formats = tuple(dict.fromkeys(args.formats))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = load_inputs(args.evaluation_dir)
    audit = validate_complete_predictions(inputs)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.validate_only:
        print("VALIDATION PASSED; no figures were written.")
        return 0

    matplotlib, plt, np, mcolors = _import_plotting()
    configure_style(matplotlib)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    figures = (
        ("fig01_precision_recall_landscape", figure_precision_recall(inputs, plt, np)),
        ("fig02_f1_confidence_intervals", figure_f1_intervals(inputs, plt, np)),
        (
            "fig03a_per_class_f1_smartbugs",
            figure_per_class_dataset(inputs, plt, np, mcolors, "SmartBugs-Curated"),
        ),
        (
            "fig03b_per_class_f1_scrawld",
            figure_per_class_dataset(inputs, plt, np, mcolors, "ScrawlD"),
        ),
        ("fig04_coverage_semantics", figure_coverage(inputs, plt, np)),
    )
    for name, figure in figures:
        generated.extend(save_figure(figure, output_dir / name, args.formats, args.dpi, plt))

    bccc_metrics = bccc_discrimination_metrics(inputs)
    generated.extend(
        save_figure(
            figure_bccc_discrimination(bccc_metrics, plt, np),
            output_dir / "fig05_bccc_discrimination",
            args.formats, args.dpi, plt,
        )
    )

    write_clean_csv(output_dir / "source_prediction_summary_deduplicated.csv", inputs.summary)
    write_clean_csv(output_dir / "source_prediction_per_class_deduplicated.csv", inputs.per_class)
    write_clean_csv(output_dir / "source_prediction_coverage.csv", inputs.coverage)
    write_bccc_derived(output_dir / "derived_bccc_discrimination.csv", bccc_metrics)
    write_captions(output_dir / "FIGURE_CAPTIONS.md")
    write_latex_includes(output_dir / "FIGURE_INCLUDES.tex")

    source_paths = [
        inputs.evaluation_dir / name
        for name in (SUMMARY_FILE, PER_CLASS_FILE, COVERAGE_FILE, SUMMARY_JSON_FILE)
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "evaluation_dir": str(inputs.evaluation_dir),
        "quality_audit": audit,
        "source_files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        "settings": {"formats": list(args.formats), "png_dpi": args.dpi},
        "generated_figures": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in generated
        ],
        "derived_files": [
            "source_prediction_summary_deduplicated.csv",
            "source_prediction_per_class_deduplicated.csv",
            "source_prediction_coverage.csv",
            "derived_bccc_discrimination.csv",
            "FIGURE_CAPTIONS.md",
            "FIGURE_INCLUDES.tex",
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(generated)} figure files to: {output_dir}")
    for path in generated:
        print(path)
    print(f"Provenance manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
