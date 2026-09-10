#!/usr/bin/env python3
"""Audit native validity and every stage of structured-output recovery.

This CPU-only program never changes the frozen JSONL predictions. It reports
first-attempt validity and failure type, format retries, deterministic
structural normalization, targeted re-inference, and changes between the
strict first-attempt prediction and the retained final prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("smartbugs", "scrawld", "bccc")
DATASET_LABELS = {
    "smartbugs": "SmartBugs-Curated",
    "scrawld": "ScrawlD",
    "bccc": "BCCC",
}
MODELS = (
    "Qwen2.5-Coder-7B-Instruct",
    "CodeLlama-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
)
MODEL_ALIASES = {
    "qwen": MODELS[0],
    "codellama": MODELS[1],
    "mistral": MODELS[2],
}
TAXONOMIES = {
    "smartbugs": frozenset({
        "access_control", "arithmetic", "bad_randomness", "denial_of_service",
        "other", "reentrancy", "short_addresses", "time_manipulation",
        "transaction_ordering", "unchecked_low_level_calls",
    }),
    "scrawld": frozenset({
        "arithmetic", "denial_of_service", "locked_ether", "reentrancy",
        "time_manipulation", "transaction_ordering", "tx_origin",
        "unchecked_low_level_calls",
    }),
    "bccc": frozenset({
        "access_control", "arithmetic", "bad_randomness", "denial_of_service",
        "locked_ether", "reentrancy", "short_addresses", "time_manipulation",
        "transaction_ordering", "tx_origin", "unchecked_low_level_calls",
    }),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--record-output", type=Path, default=None,
        help="Optional record-level CSV for sensitivity analyses.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_label(metadata_path: Path, jsonl_path: Path) -> str:
    candidates = [jsonl_path.name]
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        candidates.insert(0, str(metadata.get("config", {}).get("model", "")))
    combined = " ".join(candidates).lower()
    for alias, label in MODEL_ALIASES.items():
        if alias in combined:
            return label
    raise ValueError(f"Cannot infer model from {jsonl_path.name}")


def normalize_category(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def extract_result(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(
        item.strip() for item in re.findall(
            r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
        )
    )
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and "vulnerabilities" in value:
            return value
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "vulnerabilities" in value:
                return value
    raise ValueError("no JSON result object with a vulnerabilities field")


def strict_categories(raw: object, taxonomy: frozenset[str]) -> frozenset[str]:
    findings = extract_result(str(raw or "")).get("vulnerabilities")
    if not isinstance(findings, list):
        raise ValueError("vulnerabilities is not a list")
    categories: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("finding is not an object")
        category = normalize_category(finding.get("category", ""))
        if category not in taxonomy:
            raise ValueError("finding is outside the taxonomy")
        categories.add(category)
    return frozenset(categories)


def final_categories(row: dict[str, Any], taxonomy: frozenset[str]) -> frozenset[str]:
    findings = row.get("vulnerabilities", [])
    if not isinstance(findings, list):
        raise ValueError("final vulnerabilities value is not a list")
    categories: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("final finding is not an object")
        category = normalize_category(finding.get("category", ""))
        if category not in taxonomy:
            raise ValueError(f"final category {category!r} is outside the taxonomy")
        categories.add(category)
    return frozenset(categories)


def flattened_text(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from flattened_text(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from flattened_text(nested)


def classify_attempt(attempt: dict[str, Any]) -> str:
    status = str(attempt.get("parse_status", "")).lower()
    text = " ".join(flattened_text(attempt)).lower()
    if status == "valid":
        return "valid"
    if any(token in status for token in ("inference_error", "runtime_error")) or any(
        token in text for token in ("cuda", "out of memory", "traceback", "generation failed")
    ):
        return "runtime_failure"
    if "outside the allowed taxonomy" in text or "outside the taxonomy" in text:
        return "taxonomy_violation"
    if any(token in text for token in (
        "is not an object", "is not a list", "must be a list", "missing vulnerabilities",
    )):
        return "schema_violation"
    if any(token in text for token in (
        "no valid json result object", "jsondecodeerror", "unterminated",
        "expecting value", "extra data",
    )):
        return "json_syntax_or_missing_result"
    if "valid_with_rejected_items" in status:
        return "other_validation_failure"
    return "other_failure"


def latest_repair(row: dict[str, Any]) -> dict[str, Any]:
    current = row.get("repair_details")
    if isinstance(current, dict):
        return current
    candidate: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("repair_") and isinstance(value, dict):
            candidate = value
    return candidate


def action_count(row: dict[str, Any], name: str) -> int:
    value = (latest_repair(row).get("actions") or {}).get(name, 0)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid repair action count {name}={value!r}") from error


def identity(dataset: str, row: dict[str, Any]) -> str:
    field = "source_sha256" if dataset == "smartbugs" else "source_id"
    value = row.get(field)
    if not value:
        raise ValueError(f"Missing {field} in prediction row")
    return str(value)


def inspect_record(
    dataset: str, model: str, row: dict[str, Any], source_name: str
) -> tuple[dict[str, Any], Counter[str]]:
    chunks = row.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Record {row.get('record_index')} has no chunks")
    taxonomy = TAXONOMIES[dataset]
    first_predictions: set[str] = set()
    outcomes: Counter[str] = Counter()
    native_valid_chunks = retry_chunks = retry_recovered = 0
    structural_chunks = targeted_chunks = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("chunk is not an object")
        attempts = chunk.get("attempts") or []
        if not isinstance(attempts, list) or not attempts:
            outcomes["missing_first_attempt"] += 1
        else:
            first = attempts[0] if isinstance(attempts[0], dict) else {}
            outcome = classify_attempt(first)
            outcomes[outcome] += 1
            if outcome == "valid":
                native_valid_chunks += 1
                try:
                    first_predictions.update(strict_categories(first.get("raw_response", ""), taxonomy))
                except ValueError:
                    outcomes["stored_status_content_mismatch"] += 1
            if len(attempts) > 1:
                retry_chunks += 1
                retry_recovered += int(any(
                    isinstance(attempt, dict)
                    and str(attempt.get("parse_status", "")).lower() == "valid"
                    for attempt in attempts[1:]
                ))
        final_status = str(chunk.get("parse_status", "")).lower()
        structural_chunks += int("structural_repair" in final_status)
        targeted_chunks += int(
            "targeted_repair" in final_status or "closed_choice_repair" in final_status
        )
    structural_chunks = max(structural_chunks, action_count(row, "structural"))
    targeted_chunks = max(targeted_chunks, action_count(row, "inference"))
    first = frozenset(first_predictions)
    final = final_categories(row, taxonomy)
    native_valid = native_valid_chunks == len(chunks)
    exposed = not native_valid or retry_chunks > 0 or structural_chunks > 0 or targeted_chunks > 0
    detail = {
        "dataset": DATASET_LABELS[dataset],
        "model": model,
        "record_key": identity(dataset, row),
        "record_index": int(row["record_index"]),
        "chunks": len(chunks),
        "first_attempt_valid_chunks": native_valid_chunks,
        "first_attempt_native_valid": native_valid,
        "format_retry_chunks": retry_chunks,
        "format_recovered_chunks": retry_recovered,
        "structural_repair_chunks": structural_chunks,
        "targeted_reinference_chunks": targeted_chunks,
        "any_recovery_exposure": exposed,
        "first_attempt_categories": ";".join(sorted(first)),
        "final_categories": ";".join(sorted(final)),
        "first_attempt_category_count": len(first),
        "final_category_count": len(final),
        "category_count_changed": len(first) != len(final),
        "category_set_changed": first != final,
        "binary_decision_changed": bool(first) != bool(final),
        "final_record_status": str(row.get("status", "missing_status")),
        "source_jsonl": source_name,
    }
    return detail, outcomes


def inspect_run(dataset: str, model: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Corrupt JSONL at {path}:{line_number}") from error
            detail, record_outcomes = inspect_record(dataset, model, row, path.name)
            details.append(detail)
            outcomes.update(record_outcomes)
    chunks = sum(int(row["chunks"]) for row in details)
    exposed = [row for row in details if row["any_recovery_exposure"]]
    changed_sets = sum(bool(row["category_set_changed"]) for row in exposed)
    changed_binary = sum(bool(row["binary_decision_changed"]) for row in exposed)
    native_records = sum(bool(row["first_attempt_native_valid"]) for row in details)
    summary = {
        "dataset": DATASET_LABELS[dataset],
        "model": model,
        "records": len(details),
        "chunks": chunks,
        "first_attempt_valid_chunks": outcomes["valid"],
        "first_attempt_valid_chunk_rate": safe_divide(outcomes["valid"], chunks),
        "first_attempt_json_failures": outcomes["json_syntax_or_missing_result"],
        "first_attempt_schema_violations": outcomes["schema_violation"],
        "first_attempt_taxonomy_violations": outcomes["taxonomy_violation"],
        "first_attempt_runtime_failures": outcomes["runtime_failure"],
        "first_attempt_other_failures": sum(
            outcomes[name] for name in (
                "other_validation_failure", "other_failure", "missing_first_attempt",
                "stored_status_content_mismatch",
            )
        ),
        "first_attempt_native_valid_records": native_records,
        "first_attempt_native_valid_record_rate": safe_divide(native_records, len(details)),
        "format_retry_chunks": sum(int(row["format_retry_chunks"]) for row in details),
        "format_recovered_chunks": sum(int(row["format_recovered_chunks"]) for row in details),
        "structural_repair_chunks": sum(int(row["structural_repair_chunks"]) for row in details),
        "targeted_reinference_chunks": sum(int(row["targeted_reinference_chunks"]) for row in details),
        "recovery_exposed_records": len(exposed),
        "recovery_exposed_record_rate": safe_divide(len(exposed), len(details)),
        "changed_category_count_records": sum(bool(row["category_count_changed"]) for row in exposed),
        "changed_category_set_records": changed_sets,
        "changed_binary_decision_records": changed_binary,
        "changed_category_set_rate_among_exposed": safe_divide(changed_sets, len(exposed)),
        "changed_binary_rate_among_exposed": safe_divide(changed_binary, len(exposed)),
        "final_non_ok_records": sum(row["final_record_status"] != "ok" for row in details),
        "source_jsonl": path.name,
    }
    return summary, details


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for dataset in DATASETS:
        paths = sorted((root / dataset).glob("*.jsonl"))
        if len(paths) != 3:
            raise RuntimeError(f"Expected three JSONL files for {dataset}; found {len(paths)}")
        for path in paths:
            model = model_label(path.with_suffix(".metadata.json"), path)
            summary, records = inspect_run(dataset, model, path)
            summaries.append(summary)
            details.extend(records)
    dataset_order = tuple(DATASET_LABELS.values())
    summaries.sort(key=lambda row: (dataset_order.index(row["dataset"]), MODELS.index(row["model"])))
    details.sort(key=lambda row: (
        dataset_order.index(row["dataset"]), MODELS.index(row["model"]), row["record_index"]
    ))
    return summaries, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refuse to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.prediction_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    summaries, details = collect(root)
    if len(summaries) != 9:
        raise RuntimeError(f"Expected nine model-dataset pairs; found {len(summaries)}")
    if args.output is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    else:
        write_csv(args.output.resolve(), summaries)
        print(f"Wrote {args.output.resolve()}")
    if args.record_output is not None:
        write_csv(args.record_output.resolve(), details)
        print(f"Wrote {args.record_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
