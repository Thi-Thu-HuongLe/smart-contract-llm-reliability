"""Report first-attempt failures and recovery exposure from frozen JSONL runs.

This is a CPU-only audit helper. It does not run model inference or change any
prediction file. The output is suitable for the manuscript's pre-repair table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASET_ORDER = ("smartbugs", "scrawld", "bccc")
MODEL_ORDER = (
    "Qwen2.5-Coder-7B-Instruct",
    "CodeLlama-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
)
MODEL_ALIASES = {
    "qwen": "Qwen2.5-Coder-7B-Instruct",
    "codellama": "CodeLlama-7B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def model_label(metadata_path: Path, jsonl_path: Path) -> str:
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = str(metadata.get("config", {}).get("model", ""))
        lower = model.lower()
        for key, label in MODEL_ALIASES.items():
            if key in lower:
                return label
    lower = jsonl_path.name.lower()
    for key, label in MODEL_ALIASES.items():
        if key in lower:
            return label
    return jsonl_path.stem


def first_attempt_status(chunk: dict) -> str:
    attempts = chunk.get("attempts") or []
    if attempts:
        return str(attempts[0].get("parse_status", ""))
    return ""


def inspect_jsonl(path: Path) -> dict:
    result = {
        "records": 0,
        "chunks": 0,
        "first_attempt_error_chunks": 0,
        "format_recovered_chunks": 0,
        "structural_actions": 0,
        "reinference_actions": 0,
        "final_failed_records": 0,
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result["records"] += 1
            if row.get("status") != "ok":
                result["final_failed_records"] += 1
            for chunk in row.get("chunks", []):
                result["chunks"] += 1
                initial = first_attempt_status(chunk).lower()
                if initial in {"parse_or_validation_error", "error"} or "error" in initial:
                    result["first_attempt_error_chunks"] += 1
                status = str(chunk.get("parse_status", "")).lower()
                if status == "valid_after_format_retry":
                    result["format_recovered_chunks"] += 1
                if "structural_repair" in status:
                    result["structural_actions"] += 1
                if "targeted_repair" in status or "closed_choice_repair" in status:
                    result["reinference_actions"] += 1
    return result


def touched_lookup(prediction_root: Path) -> dict[tuple[str, str], int]:
    exposure = prediction_root / "robustness_analysis" / "repair_exposure.csv"
    if not exposure.is_file():
        return {}
    lookup: dict[tuple[str, str], int] = {}
    with exposure.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"].lower().replace("-curated", "")
            lookup[(dataset, row["model"])] = int(row["records_touched"])
    return lookup


def dataset_label(name: str) -> str:
    return {"smartbugs": "SmartBugs-Curated", "scrawld": "ScrawlD", "bccc": "BCCC"}[name]


def collect(prediction_root: Path) -> list[dict]:
    touched = touched_lookup(prediction_root)
    rows: list[dict] = []
    for dataset in DATASET_ORDER:
        paths = sorted((prediction_root / dataset).glob("*.jsonl"))
        for path in paths:
            label = model_label(path.with_suffix(".metadata.json"), path)
            stats = inspect_jsonl(path)
            rows.append(
                {
                    "dataset": dataset_label(dataset),
                    "model": label,
                    **stats,
                    "touched_records": touched.get((dataset, label), 0),
                    "source_jsonl": str(path),
                }
            )
    rows.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"].lower().replace("-curated", ""))
            if row["dataset"].lower().replace("-curated", "") in DATASET_ORDER
            else 99,
            MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else 99,
        )
    )
    return rows


def main() -> None:
    args = parse_args()
    root = args.prediction_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = collect(root)
    if len(rows) != 9:
        raise RuntimeError(f"Expected nine model-dataset pairs, found {len(rows)}")
    fields = [
        "dataset",
        "model",
        "records",
        "chunks",
        "first_attempt_error_chunks",
        "format_recovered_chunks",
        "structural_actions",
        "reinference_actions",
        "touched_records",
        "final_failed_records",
        "source_jsonl",
    ]
    target = args.output
    if target is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
