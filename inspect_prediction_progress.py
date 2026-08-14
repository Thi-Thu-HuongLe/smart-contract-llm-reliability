# Copyright 2026 Thi-Thu-Huong Le
# SPDX-License-Identifier: Apache-2.0

"""Read-only integrity and progress report for an interrupted inference run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


METADATA_SUFFIX = ".metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect benchmark metadata and JSONL files without changing them. Reports exact "
            "completion, status counts, approximate ETA, and whether resume is safe."
        )
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete report as JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "--expected-models",
        nargs="+",
        help=(
            "Expected model repositories (optionally repository@revision). Missing "
            "model/dataset files are reported as not started."
        ),
    )
    parser.add_argument(
        "--expected-datasets",
        nargs="+",
        help="Expected dataset names. Missing model/dataset files are reported as not started.",
    )
    return parser.parse_args()


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    rounded = int(round(seconds))
    days, remainder = divmod(rounded, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def jsonl_for_metadata(metadata_path: Path) -> Path:
    name = metadata_path.name
    if not name.endswith(METADATA_SUFFIX):
        raise ValueError(f"Unexpected metadata filename: {metadata_path}")
    return metadata_path.with_name(name[: -len(METADATA_SUFFIX)] + ".jsonl")


def inspect_pair(metadata_path: Path) -> dict:
    jsonl_path = jsonl_for_metadata(metadata_path)
    problems: list[str] = []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "metadata_path": str(metadata_path),
            "jsonl_path": str(jsonl_path),
            "dataset": metadata_path.parent.name,
            "model": "unknown",
            "expected": 0,
            "completed": 0,
            "percent": 0.0,
            "status_counts": {},
            "average_runtime_seconds": None,
            "estimated_remaining_seconds": None,
            "last_record_index": None,
            "last_modified_utc": None,
            "resumable": False,
            "complete": False,
            "problems": [f"corrupt metadata: {type(error).__name__}: {error}"],
        }

    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    records = metadata.get("records") if isinstance(metadata.get("records"), list) else []
    expected_count = metadata.get("record_count")
    if not isinstance(expected_count, int) or expected_count < 0:
        problems.append("record_count is missing or invalid")
        expected_count = len(records)
    if len(records) != expected_count:
        problems.append(
            f"metadata records length {len(records)} does not equal record_count {expected_count}"
        )
    if not metadata.get("run_fingerprint"):
        problems.append("run_fingerprint is missing")

    expected_by_index: dict[int, dict] = {}
    for entry in records:
        if not isinstance(entry, dict) or not isinstance(entry.get("record_index"), int):
            problems.append("metadata contains a record without an integer record_index")
            continue
        record_index = entry["record_index"]
        if record_index in expected_by_index:
            problems.append(f"metadata has duplicate record_index {record_index}")
        expected_by_index[record_index] = entry

    seen: dict[int, dict] = {}
    status_counts: Counter[str] = Counter()
    runtimes: list[float] = []
    corrupt_lines = 0
    duplicate_rows = 0
    unknown_indices = 0
    manifest_mismatches = 0
    last_modified_utc = None

    if jsonl_path.exists():
        last_modified_utc = datetime.fromtimestamp(
            jsonl_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    record_index = int(row["record_index"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    corrupt_lines += 1
                    continue
                if record_index in seen:
                    duplicate_rows += 1
                    continue
                seen[record_index] = row
                expected_entry = expected_by_index.get(record_index)
                if expected_entry is None:
                    unknown_indices += 1
                else:
                    for field in (
                        "record_id",
                        "source_id",
                        "source_sha256",
                        "ground_truth_binary",
                    ):
                        if row.get(field) != expected_entry.get(field):
                            manifest_mismatches += 1
                            break
                status_counts[str(row.get("status", "missing"))] += 1
                runtime = row.get("runtime_seconds")
                if isinstance(runtime, (int, float)) and math.isfinite(runtime) and runtime >= 0:
                    runtimes.append(float(runtime))

    if corrupt_lines:
        problems.append(f"{corrupt_lines} corrupt JSONL line(s)")
    if duplicate_rows:
        problems.append(f"{duplicate_rows} duplicate JSONL record_index value(s)")
    if unknown_indices:
        problems.append(f"{unknown_indices} JSONL record_index value(s) absent from metadata")
    if manifest_mismatches:
        problems.append(f"{manifest_mismatches} JSONL row(s) mismatch the input manifest")

    completed_count = len(seen)
    average_runtime = sum(runtimes) / len(runtimes) if runtimes else None
    remaining_count = max(expected_count - completed_count, 0)
    eta = average_runtime * remaining_count if average_runtime is not None else None
    percent = 100.0 if expected_count == 0 else 100.0 * completed_count / expected_count

    return {
        "metadata_path": str(metadata_path),
        "jsonl_path": str(jsonl_path),
        "dataset": str(config.get("dataset") or metadata_path.parent.name),
        "model": str(config.get("model") or "unknown"),
        "model_revision": config.get("model_revision"),
        "schema_version": metadata.get("schema_version"),
        "prompt_version": config.get("prompt_version"),
        "expected": expected_count,
        "completed": completed_count,
        "percent": percent,
        "status_counts": dict(sorted(status_counts.items())),
        "average_runtime_seconds": average_runtime,
        "estimated_remaining_seconds": eta,
        "last_record_index": max(seen) if seen else None,
        "last_modified_utc": last_modified_utc,
        "resumable": not problems,
        "complete": not problems and completed_count == expected_count,
        "problems": problems,
    }


def inspect_root(
    root: Path,
    expected_models: list[str] | None = None,
    expected_datasets: list[str] | None = None,
) -> dict:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Prediction root does not exist: {root}")

    metadata_paths = sorted(root.rglob(f"*{METADATA_SUFFIX}"))
    if not metadata_paths:
        raise FileNotFoundError(f"No *{METADATA_SUFFIX} files found beneath {root}")

    runs = [inspect_pair(path) for path in metadata_paths]
    paired_jsonl = {Path(run["jsonl_path"]).resolve() for run in runs}
    orphan_jsonl = [
        str(path)
        for path in sorted(root.rglob("*.jsonl"))
        if path.resolve() not in paired_jsonl
    ]
    observed_models = sorted({run["model"] for run in runs if run["model"] != "unknown"})
    observed_datasets = sorted({run["dataset"] for run in runs})
    models = [
        value.split("@", 1)[0] for value in (expected_models or observed_models)
    ]
    datasets = list(expected_datasets or observed_datasets)
    dataset_sizes: dict[str, int] = {}
    matrix_problems: list[str] = []
    for dataset in datasets:
        sizes = {run["expected"] for run in runs if run["dataset"] == dataset}
        if len(sizes) == 1:
            dataset_sizes[dataset] = sizes.pop()
        elif len(sizes) > 1:
            matrix_problems.append(
                f"inconsistent record_count values for {dataset}: {sorted(sizes)}"
            )
        else:
            matrix_problems.append(
                f"cannot infer record_count for expected dataset {dataset}"
            )

    observed_pairs = {(run["model"], run["dataset"]) for run in runs}
    missing_run_pairs = [
        {"model": model, "dataset": dataset}
        for model in models
        for dataset in datasets
        if (model, dataset) not in observed_pairs
    ]
    if all(dataset in dataset_sizes for dataset in datasets):
        total_expected = len(models) * sum(dataset_sizes[dataset] for dataset in datasets)
    else:
        total_expected = sum(run["expected"] for run in runs)
    total_completed = sum(run["completed"] for run in runs)
    return {
        "prediction_root": str(root),
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "expected_run_count": len(models) * len(datasets),
        "expected_models": models,
        "expected_datasets": datasets,
        "dataset_sizes": dataset_sizes,
        "missing_run_pairs": missing_run_pairs,
        "matrix_problems": matrix_problems,
        "total_expected": total_expected,
        "total_completed": total_completed,
        "total_percent": (
            100.0 if total_expected == 0 else 100.0 * total_completed / total_expected
        ),
        "safe_to_resume_all": (
            all(run["resumable"] for run in runs)
            and not orphan_jsonl
            and not matrix_problems
        ),
        "orphan_jsonl": orphan_jsonl,
        "runs": runs,
    }


def print_human(report: dict) -> None:
    print(f"Prediction root: {report['prediction_root']}")
    print(
        "Overall progress: "
        f"{report['total_completed']}/{report['total_expected']} "
        f"({report['total_percent']:.2f}%) across "
        f"{report['run_count']}/{report['expected_run_count']} run files."
    )
    print(f"Safe to resume all: {'YES' if report['safe_to_resume_all'] else 'NO'}")
    if report["missing_run_pairs"]:
        print("Not started yet:")
        for pair in report["missing_run_pairs"]:
            print(f"  {pair['dataset']} | {pair['model']}")
    if report["matrix_problems"]:
        print("Experiment-matrix problems:")
        for problem in report["matrix_problems"]:
            print(f"  {problem}")
    for run in report["runs"]:
        state = "COMPLETE" if run["complete"] else "INCOMPLETE"
        safety = "RESUMABLE" if run["resumable"] else "DO NOT RESUME"
        print(f"\n[{state}; {safety}] {run['dataset']} | {run['model']}")
        print(
            f"  progress: {run['completed']}/{run['expected']} "
            f"({run['percent']:.2f}%), last_index={run['last_record_index']}"
        )
        print(
            f"  statuses: {run['status_counts']}; "
            f"average_saved_runtime={format_duration(run['average_runtime_seconds'])}; "
            f"rough_ETA={format_duration(run['estimated_remaining_seconds'])}"
        )
        print(f"  last_write_UTC: {run['last_modified_utc'] or 'n/a'}")
        print(f"  JSONL: {run['jsonl_path']}")
        if run["problems"]:
            print("  integrity problems: " + "; ".join(run["problems"]))
    if report["orphan_jsonl"]:
        print("\nOrphan JSONL files without paired metadata:")
        for path in report["orphan_jsonl"]:
            print(f"  {path}")


def main() -> None:
    args = parse_args()
    report = inspect_root(
        args.prediction_root,
        expected_models=args.expected_models,
        expected_datasets=args.expected_datasets,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human(report)
    if not report["safe_to_resume_all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
