"""Evaluate JSONL produced by :mod:`run_frozen_llm_inference` against labels.

Unlike the retrospective evaluator, this script consumes the new failure-aware
JSONL format and its ordered manifest.  Missing or failed generations remain
visible in coverage statistics and cannot be mistaken for a complete negative
prediction set.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from evaluation_utils import (
    SCRAWLD_LABEL_MAP,
    SMARTBUGS_LABEL_MAP,
    bootstrap_micro_f1,
    canonicalize_prediction,
    code_digest,
    load_scrawld_static_baseline_references,
    model_name,
    safe_divide,
    summarize_pairs,
    write_csv,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_PREDICTION_ROOT = REPOSITORY_ROOT / "results" / "final_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate manifest-backed JSONL model predictions."
    )
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <prediction-root>/benchmark_evaluation.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("smartbugs", "scrawld", "bccc"),
        default=("smartbugs", "scrawld", "bccc"),
    )
    parser.add_argument(
        "--include-static-baselines",
        action="store_true",
        help="Also reconstruct ScrawlD analyzer outputs on the exact evaluation subset.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Corrupt JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object.")
            rows.append(row)
    return rows


def metadata_path_for(jsonl_path: Path) -> Path:
    return jsonl_path.parent / f"{jsonl_path.stem}.metadata.json"


def prediction_key(dataset: str, record: dict) -> str:
    if dataset == "smartbugs":
        key = record.get("source_sha256")
    else:
        key = record.get("source_id")
    if not key:
        raise ValueError(f"Missing identity field for {dataset} prediction record.")
    return str(key)


def manifest_key(dataset: str, entry: dict) -> str:
    if dataset == "smartbugs":
        key = entry.get("source_sha256")
    else:
        key = entry.get("source_id")
    if not key:
        raise ValueError(f"Missing identity field for {dataset} manifest record.")
    return str(key)


def load_run(jsonl_path: Path, dataset: str) -> tuple[dict, dict[str, dict]]:
    metadata_path = metadata_path_for(jsonl_path)
    if not metadata_path.exists():
        raise ValueError(f"Missing run metadata for {jsonl_path.name}.")
    metadata = read_json(metadata_path)
    entries = metadata.get("records")
    if not isinstance(entries, list) or len(entries) != metadata.get("record_count"):
        raise ValueError(f"Invalid or incomplete record manifest in {metadata_path}.")
    rows = read_jsonl(jsonl_path)
    manifest_by_index = {entry.get("record_index"): entry for entry in entries}
    if len(manifest_by_index) != len(entries):
        raise ValueError(f"Duplicate record_index in {metadata_path}.")

    results: dict[str, dict] = {}
    seen_indices: set[int] = set()
    for row in rows:
        try:
            record_index = int(row["record_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"A result row in {jsonl_path} has no valid record_index.") from error
        if record_index in seen_indices:
            raise ValueError(f"Duplicate record_index {record_index} in {jsonl_path}.")
        seen_indices.add(record_index)
        expected = manifest_by_index.get(record_index)
        if expected is None:
            raise ValueError(f"Unknown record_index {record_index} in {jsonl_path}.")
        for field in ("record_id", "source_id", "source_sha256", "ground_truth_binary"):
            if row.get(field) != expected.get(field):
                raise ValueError(
                    f"Result row {record_index} in {jsonl_path} does not match manifest field {field}."
                )
        key = prediction_key(dataset, row)
        if key in results:
            raise ValueError(f"Duplicate record identity {key!r} in {jsonl_path}.")
        results[key] = row
    return metadata, results


def predictions_for_record(record: dict) -> tuple[set[str], int, int]:
    predicted: set[str] = set()
    unknown = 0
    findings = record.get("vulnerabilities", [])
    if not isinstance(findings, list):
        raise ValueError("The vulnerabilities field must be a list.")
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("A vulnerability finding is not an object.")
        category = canonicalize_prediction(finding.get("category", ""))
        if category is None:
            unknown += 1
        else:
            predicted.add(category)
    return predicted, unknown, len(findings)


def scrawld_ground_truth() -> dict[str, set[str]]:
    records = read_json(DATA_ROOT / "scrawld_vulnerabilities.json")
    return {
        str(record["address"]): {
            SCRAWLD_LABEL_MAP[key]
            for group in record.get("vulnerabilities", [])
            for key, value in group.items()
            if value and key in SCRAWLD_LABEL_MAP
        }
        for record in records
    }


def smartbugs_ground_truth() -> dict[str, set[str]]:
    records = read_json(DATA_ROOT / "smartbugs_curated.json")
    return {
        code_digest(str(record.get("code", ""))): {
            SMARTBUGS_LABEL_MAP[finding["category"]]
            for finding in record.get("vulnerabilities", [])
            if finding.get("category") in SMARTBUGS_LABEL_MAP
        }
        for record in records
    }


def bccc_ground_truth() -> dict[str, int]:
    paths = (
        DATA_ROOT / "bccc_vulnerable.json",
        DATA_ROOT / "bccc_secure.json",
    )
    labels: dict[str, int] = {}
    for path in paths:
        for record in read_json(path):
            source_id = str(record["address"])
            label = record.get("vulnerablility")
            if label not in (0, 1):
                raise ValueError(f"Invalid BCCC label in {path}: {label!r}")
            if source_id in labels and labels[source_id] != label:
                raise ValueError(f"Conflicting BCCC labels for {source_id}.")
            labels[source_id] = label
    return labels


def status_counts(results: dict[str, dict]) -> dict[str, int]:
    counts = Counter(str(row.get("status", "missing_status")) for row in results.values())
    return dict(sorted(counts.items()))


def status_fields(statuses: dict[str, int]) -> dict[str, int]:
    """Expose valid, partial, and failed output counts as first-class columns."""

    known = {"ok", "partial", "failed"}
    return {
        "ok_output_records": statuses.get("ok", 0),
        "partial_output_records": statuses.get("partial", 0),
        "failed_output_records": statuses.get("failed", 0),
        "other_status_records": sum(
            count for status, count in statuses.items() if status not in known
        ),
    }


def evaluate_multilabel_run(
    dataset: str,
    jsonl_path: Path,
    ground_truth: dict[str, set[str]],
    labels: list[str],
) -> tuple[dict, list[dict], dict, dict[str, set[str]]]:
    metadata, results = load_run(jsonl_path, dataset)
    entries = metadata["records"]
    pairs: list[tuple[set[str], set[str]]] = []
    unknown_findings = total_findings = missing = 0
    selected_truth: dict[str, set[str]] = {}

    for entry in entries:
        key = manifest_key(dataset, entry)
        if key not in ground_truth:
            raise ValueError(f"Manifest record {key!r} is not present in {dataset} ground truth.")
        expected = ground_truth[key]
        selected_truth[key] = expected
        prediction = results.get(key)
        if prediction is None:
            missing += 1
            predicted, unknown, finding_count = set(), 0, 0
        else:
            predicted, unknown, finding_count = predictions_for_record(prediction)
        pairs.append((expected, predicted))
        unknown_findings += unknown
        total_findings += finding_count

    summary, class_rows = summarize_pairs(pairs, labels)
    ci_lower, ci_upper = bootstrap_micro_f1(pairs, labels)
    system = model_name(jsonl_path)
    statuses = status_counts(results)
    summary_row = {
        "dataset": "SmartBugs-Curated" if dataset == "smartbugs" else "ScrawlD",
        "system_type": "code_llm",
        "system": system,
        "evaluation_source": "manifest_backed_jsonl",
        "metric_scope": "closed_taxonomy_all_manifest_records",
        "ground_truth_records": len(entries),
        "prediction_records": len(results),
        "matched_records": len(entries) - missing,
        "missing_prediction_records": missing,
        "total_findings": total_findings,
        "out_of_taxonomy_findings": unknown_findings,
        "out_of_taxonomy_rate": safe_divide(unknown_findings, total_findings),
        "micro_f1_ci95_lower": ci_lower,
        "micro_f1_ci95_upper": ci_upper,
        "all_predictions_empty": total_findings == 0,
        "status_counts": json.dumps(statuses, sort_keys=True),
        **status_fields(statuses),
        **summary,
    }
    for row in class_rows:
        row.update(
            {
                "dataset": summary_row["dataset"],
                "system_type": "code_llm",
                "system": system,
                "evaluation_source": "manifest_backed_jsonl",
            }
        )
    coverage_row = {
        "dataset": summary_row["dataset"],
        "system": system,
        "records_expected": len(entries),
        "records_written": len(results),
        "records_missing": missing,
        "records_with_findings": sum(
            bool(row.get("vulnerabilities")) for row in results.values()
        ),
        "total_findings": total_findings,
        "status_counts": json.dumps(statuses, sort_keys=True),
        **status_fields(statuses),
    }
    return summary_row, class_rows, coverage_row, selected_truth


def binary_metrics(pairs: list[tuple[int, bool]]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for expected, predicted in pairs:
        if expected == 1 and predicted:
            tp += 1
        elif expected == 0 and predicted:
            fp += 1
        elif expected == 0 and not predicted:
            tn += 1
        else:
            fn += 1
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "accuracy": safe_divide(tp + tn, len(pairs)),
    }


def evaluate_bccc_run(jsonl_path: Path, ground_truth: dict[str, int]) -> tuple[dict, dict]:
    metadata, results = load_run(jsonl_path, "bccc")
    pairs: list[tuple[int, bool]] = []
    missing = unknown_findings = total_findings = 0
    for entry in metadata["records"]:
        key = manifest_key("bccc", entry)
        if key not in ground_truth:
            raise ValueError(f"Manifest BCCC record {key!r} is not in the benchmark.")
        result = results.get(key)
        if result is None:
            missing += 1
            predicted, unknown, finding_count = set(), 0, 0
        else:
            predicted, unknown, finding_count = predictions_for_record(result)
        pairs.append((ground_truth[key], bool(predicted)))
        unknown_findings += unknown
        total_findings += finding_count

    metrics = binary_metrics(pairs)
    statuses = status_counts(results)
    system = model_name(jsonl_path)
    summary = {
        "dataset": "BCCC",
        "system_type": "code_llm",
        "system": system,
        "evaluation_source": "manifest_backed_jsonl",
        "metric_scope": "binary_all_manifest_records",
        "ground_truth_records": len(pairs),
        "prediction_records": len(results),
        "matched_records": len(pairs) - missing,
        "missing_prediction_records": missing,
        "total_findings": total_findings,
        "out_of_taxonomy_findings": unknown_findings,
        "out_of_taxonomy_rate": safe_divide(unknown_findings, total_findings),
        "status_counts": json.dumps(statuses, sort_keys=True),
        **status_fields(statuses),
        **metrics,
    }
    coverage = {
        "dataset": "BCCC",
        "system": system,
        "records_expected": len(pairs),
        "records_written": len(results),
        "records_missing": missing,
        "records_with_findings": sum(
            bool(row.get("vulnerabilities")) for row in results.values()
        ),
        "total_findings": total_findings,
        "status_counts": json.dumps(statuses, sort_keys=True),
        **status_fields(statuses),
    }
    return summary, coverage


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.prediction_root / "benchmark_evaluation"
    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    coverage_rows: list[dict] = []

    if "smartbugs" in args.datasets:
        ground_truth = smartbugs_ground_truth()
        labels = sorted(set(SMARTBUGS_LABEL_MAP.values()))
        for path in sorted((args.prediction_root / "smartbugs").glob("*.jsonl")):
            summary, classes, coverage, _ = evaluate_multilabel_run(
                "smartbugs", path, ground_truth, labels
            )
            summary_rows.append(summary)
            class_rows.extend(classes)
            coverage_rows.append(coverage)

    if "scrawld" in args.datasets:
        ground_truth = scrawld_ground_truth()
        labels = sorted(set(SCRAWLD_LABEL_MAP.values()))
        for path in sorted((args.prediction_root / "scrawld").glob("*.jsonl")):
            summary, classes, coverage, selected_truth = evaluate_multilabel_run(
                "scrawld", path, ground_truth, labels
            )
            summary_rows.append(summary)
            class_rows.extend(classes)
            coverage_rows.append(coverage)
        if args.include_static_baselines:
            baselines, baseline_classes = load_scrawld_static_baseline_references()
            summary_rows.extend(baselines)
            class_rows.extend(baseline_classes)

    if "bccc" in args.datasets:
        ground_truth = bccc_ground_truth()
        for path in sorted((args.prediction_root / "bccc").glob("*.jsonl")):
            summary, coverage = evaluate_bccc_run(path, ground_truth)
            summary_rows.append(summary)
            coverage_rows.append(coverage)

    if not summary_rows:
        raise FileNotFoundError(
            f"No JSONL run files found beneath {args.prediction_root}."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "prediction_summary.csv"
    per_class_path = output_root / "prediction_per_class.csv"
    coverage_path = output_root / "prediction_coverage.csv"
    write_csv(summary_path, summary_rows)
    if class_rows:
        write_csv(per_class_path, class_rows)
    else:
        per_class_path.write_text(
            "dataset,system_type,system,evaluation_source,class,tp,fp,fn,precision,recall,f1,support\n",
            encoding="utf-8",
        )
    write_csv(coverage_path, coverage_rows)
    (output_root / "prediction_summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "input": "Manifest-backed JSONL from run_frozen_llm_inference.py",
                    "missing_output_policy": "Counted as missing in coverage and as an empty prediction in metrics.",
                    "failed_output_policy": "Retained in status columns. All-manifest metrics conservatively treat an absent accepted finding as negative, so coverage and status must accompany every score.",
                    "out_of_taxonomy_policy": "Reported explicitly. The strict generation pipeline rejects such categories before acceptance.",
                    "scrawld_baseline_limitation": "Constituent tools are not independent human ground truth.",
                },
                "summary": summary_rows,
                "per_class": class_rows,
                "coverage": coverage_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_class_path}")
    print(f"Wrote {coverage_path}")
    print(f"Wrote {output_root / 'prediction_summary.json'}")


if __name__ == "__main__":
    main()
