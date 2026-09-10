#!/usr/bin/env python3
"""Audit source overlap and rerun TACE with source-family-grouped folds.

Source families are defined by a conservative lexical fingerprint: comments
and formatting are removed, while string, hexadecimal, and numeric literals
are normalized. This catches exact and literal-renamed lexical clones without
claiming semantic clone detection. Every member of a family is assigned to the
same outer and inner fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from analyze_prediction_robustness import (
    DATASET_DIRS,
    DATASET_NAMES,
    DEFAULT_PREDICTION_ROOT,
    MODEL_ORDER,
)
from analyze_tace_crossfit import (
    METHOD_F1_NAME,
    Record,
    append_system_summary,
    binary_counts,
    classwise_binary_counts,
    fit_calibrator,
    load_records,
    metrics_from_counts,
    multilabel_counts,
    per_class_rows,
    record_stratum,
    score_keys,
    supported_macro_f1,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_PREDICTION_ROOT / "source_family_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[value / 20 for value in range(1, 20)],
    )
    parser.add_argument("--shrinkage", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("Both fold counts must be at least 2")
    if args.shrinkage <= 0:
        parser.error("--shrinkage must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        parser.error("--thresholds values must be in [0, 1]")
    args.thresholds = sorted(set(args.thresholds))
    return args


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_key(dataset: str, row: dict[str, Any], code: str) -> str:
    if dataset == "smartbugs":
        return hashlib.sha256(code.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    value = row.get("address")
    if not value:
        raise ValueError(f"Missing address in {dataset} source row")
    return str(value)


def lexical_family(code: str) -> str:
    """Hash normalized Solidity tokens; this is not a semantic-clone claim."""

    text = code.replace("\r\n", "\n")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', " <string> ", text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", " <string> ", text)
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", " <hex> ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", " <number> ", text, flags=re.I)
    tokens = re.findall(
        r"[A-Za-z_$][A-Za-z0-9_$]*|==|!=|<=|>=|=>|&&|\|\||\+\+|--|\S",
        text.lower(),
    )
    normalized = " ".join(tokens)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exact_family(code: str) -> str:
    normalized = "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").strip().split("\n"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_source_families(
    evaluated_keys: dict[str, set[str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[Path]]:
    paths = {
        "smartbugs": [DATA_ROOT / "smartbugs_curated.json"],
        "scrawld": [DATA_ROOT / "scrawld_vulnerabilities.json"],
        "bccc": [DATA_ROOT / "bccc_vulnerable.json", DATA_ROOT / "bccc_secure.json"],
    }
    exact: dict[str, dict[str, str]] = {dataset: {} for dataset in DATASET_DIRS}
    lexical: dict[str, dict[str, str]] = {dataset: {} for dataset in DATASET_DIRS}
    used_paths: list[Path] = []
    for dataset in DATASET_DIRS:
        for path in paths[dataset]:
            if not path.is_file():
                raise FileNotFoundError(path)
            used_paths.append(path)
            for row in read_json(path):
                code = str(row.get("code", ""))
                key = source_key(dataset, row, code)
                if key not in evaluated_keys[dataset]:
                    continue
                current_exact = exact_family(code)
                current_lexical = lexical_family(code)
                if key in exact[dataset] and exact[dataset][key] != current_exact:
                    raise ValueError(f"Conflicting source text for {dataset}/{key}")
                exact[dataset][key] = current_exact
                lexical[dataset][key] = current_lexical
        missing = evaluated_keys[dataset] - set(lexical[dataset])
        if missing:
            raise ValueError(f"Missing source code for {dataset}: {sorted(missing)[:3]}")
    return exact, lexical, used_paths


def family_audit_rows(
    records: dict[str, dict[str, Record]],
    exact: dict[str, dict[str, str]],
    lexical: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_DIRS:
        for definition, mapping in (("exact_source", exact[dataset]), ("lexical_normalized", lexical[dataset])):
            families: dict[str, list[str]] = defaultdict(list)
            for key, family in mapping.items():
                families[family].append(key)
            multi = [members for members in families.values() if len(members) > 1]
            conflicting = sum(
                len({records[dataset][key].truth for key in members}) > 1
                for members in families.values()
            )
            rows.append(
                {
                    "dataset": DATASET_NAMES[dataset],
                    "family_definition": definition,
                    "records": len(mapping),
                    "families": len(families),
                    "multi_record_families": len(multi),
                    "records_in_multi_record_families": sum(len(members) for members in multi),
                    "largest_family": max(map(len, families.values())),
                    "families_with_conflicting_truth": conflicting,
                }
            )
    return rows


def cross_dataset_rows(
    exact: dict[str, dict[str, str]],
    lexical: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for definition, mappings in (("exact_source", exact), ("lexical_normalized", lexical)):
        for index, dataset_a in enumerate(DATASET_DIRS):
            by_family_a: dict[str, int] = Counter(mappings[dataset_a].values())
            for dataset_b in DATASET_DIRS[index + 1:]:
                by_family_b: dict[str, int] = Counter(mappings[dataset_b].values())
                shared = set(by_family_a) & set(by_family_b)
                rows.append(
                    {
                        "family_definition": definition,
                        "dataset_a": DATASET_NAMES[dataset_a],
                        "dataset_b": DATASET_NAMES[dataset_b],
                        "shared_families": len(shared),
                        "dataset_a_records_in_shared_families": sum(
                            by_family_a[item] for item in shared
                        ),
                        "dataset_b_records_in_shared_families": sum(
                            by_family_b[item] for item in shared
                        ),
                    }
                )
    return rows


def grouped_folds(
    keys: Sequence[str],
    records: dict[str, Record],
    family_by_key: dict[str, str],
    dataset: str,
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    families: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        families[family_by_key[key]].append(key)
    if len(families) < n_splits:
        raise ValueError(f"Only {len(families)} families for {n_splits} folds")
    rng = random.Random(seed)
    items = list(families.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    stratum_counts: list[Counter[str]] = [Counter() for _ in range(n_splits)]
    for _, members in items:
        family_strata = Counter(record_stratum(records[key], dataset) for key in members)
        target = min(
            range(n_splits),
            key=lambda fold: (
                sum(stratum_counts[fold][name] * count for name, count in family_strata.items()),
                fold_sizes[fold],
                fold,
            ),
        )
        folds[target].extend(members)
        fold_sizes[target] += len(members)
        stratum_counts[target].update(family_strata)
    for fold in folds:
        fold.sort()
    if any(not fold for fold in folds):
        raise RuntimeError("Grouped stratification created an empty fold")
    assigned = [key for fold in folds for key in fold]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(keys):
        raise RuntimeError("Grouped fold assignment is not a partition")
    return folds


def tune_grouped_threshold(
    train_keys: Sequence[str],
    records: dict[str, Record],
    labels: Sequence[str],
    family_by_key: dict[str, str],
    dataset: str,
    inner_folds: int,
    thresholds: Sequence[float],
    shrinkage: float,
    seed: int,
) -> float:
    folds = grouped_folds(train_keys, records, family_by_key, dataset, inner_folds, seed)
    totals: dict[float, dict[str, int]] = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, **({"tn": 0} if dataset == "bccc" else {})}
        for threshold in thresholds
    }
    for validation_keys in folds:
        validation_set = set(validation_keys)
        inner_train = [key for key in train_keys if key not in validation_set]
        calibrator = fit_calibrator(inner_train, records, labels, shrinkage)
        scores = score_keys(validation_keys, records, labels, calibrator)
        for threshold in thresholds:
            predictions = {
                key: frozenset(label for label, score in scores[key].items() if score >= threshold)
                for key in validation_keys
            }
            counts = (
                binary_counts(validation_keys, records, predictions)
                if dataset == "bccc"
                else multilabel_counts(validation_keys, records, predictions, labels)
            )
            for name, value in counts.items():
                totals[threshold][name] += value
    selected = thresholds[0]
    selected_value = -1.0
    for threshold in thresholds:
        metrics = metrics_from_counts(totals[threshold], dataset)
        value = float(metrics["balanced_accuracy"] if dataset == "bccc" else metrics["f1"])
        if value > selected_value + 1e-12 or (
            abs(value - selected_value) <= 1e-12 and threshold > selected
        ):
            selected, selected_value = threshold, value
    return selected


def grouped_crossfit(
    dataset: str,
    records: dict[str, Record],
    labels: Sequence[str],
    family_by_key: dict[str, str],
    outer_folds: int,
    inner_folds: int,
    thresholds: Sequence[float],
    shrinkage: float,
    seed: int,
) -> tuple[dict[str, frozenset[str]], list[dict[str, Any]], dict[int, float]]:
    keys = sorted(records)
    outer = grouped_folds(keys, records, family_by_key, dataset, outer_folds, seed)
    predictions: dict[str, frozenset[str]] = {}
    assignments: list[dict[str, Any]] = []
    selected_thresholds: dict[int, float] = {}
    for fold_index, test_keys in enumerate(outer):
        test_set = set(test_keys)
        train_keys = [key for key in keys if key not in test_set]
        threshold = tune_grouped_threshold(
            train_keys, records, labels, family_by_key, dataset, inner_folds,
            thresholds, shrinkage, seed + 10_000 * (fold_index + 1),
        )
        selected_thresholds[fold_index] = threshold
        calibrator = fit_calibrator(train_keys, records, labels, shrinkage)
        scores = score_keys(test_keys, records, labels, calibrator)
        for key in test_keys:
            predictions[key] = frozenset(
                label for label, score in scores[key].items() if score >= threshold
            )
            assignments.append(
                {
                    "dataset": DATASET_NAMES[dataset],
                    "record_key": key,
                    "record_index": records[key].record_index,
                    "lexical_family": family_by_key[key],
                    "outer_fold": fold_index,
                    "selected_threshold": threshold,
                }
            )
    if set(predictions) != set(keys):
        raise RuntimeError(f"Grouped OOF coverage failure for {dataset}")
    family_folds: dict[str, set[int]] = defaultdict(set)
    for row in assignments:
        family_folds[str(row["lexical_family"])].add(int(row["outer_fold"]))
    if any(len(folds) != 1 for folds in family_folds.values()):
        raise RuntimeError(f"A source family crossed outer folds in {dataset}")
    return predictions, assignments, selected_thresholds


def main() -> int:
    args = parse_args()
    prediction_root = args.prediction_root.resolve()
    output_root = (args.output_root or prediction_root / "source_family_analysis").resolve()
    records_by_dataset, labels_by_dataset, _, _ = load_records(prediction_root)
    evaluated = {dataset: set(records_by_dataset[dataset]) for dataset in DATASET_DIRS}
    exact, lexical, source_paths = load_source_families(evaluated)

    audit = family_audit_rows(records_by_dataset, exact, lexical)
    overlap = cross_dataset_rows(exact, lexical)
    summaries: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset in DATASET_DIRS:
        print(f"Grouped cross-fitting {DATASET_NAMES[dataset]}", flush=True)
        records = records_by_dataset[dataset]
        labels = labels_by_dataset[dataset]
        predictions, dataset_assignments, selected = grouped_crossfit(
            dataset, records, labels, lexical[dataset], args.outer_folds,
            args.inner_folds, args.thresholds, args.shrinkage, args.seed,
        )
        assignments.extend(dataset_assignments)
        before = len(summaries)
        append_system_summary(
            summaries, dataset, f"{METHOD_F1_NAME} [source-family grouped]",
            sorted(records), records, predictions, labels,
            "nested source-family-grouped cross-fitting",
        )
        summaries[before]["family_definition"] = "lexical_normalized"
        summaries[before]["selected_thresholds_by_fold"] = json.dumps(selected, sort_keys=True)
        per_class.extend(per_class_rows(
            dataset, records, sorted(records), predictions, labels,
            f"{METHOD_F1_NAME} [source-family grouped]",
        ))
        for key in sorted(records):
            prediction_rows.append(
                {
                    "dataset": DATASET_NAMES[dataset],
                    "record_key": key,
                    "record_index": records[key].record_index,
                    "lexical_family": lexical[dataset][key],
                    "truth": ";".join(sorted(records[key].truth)),
                    "prediction": ";".join(sorted(predictions[key])),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "source_family_summary.csv", audit)
    write_csv(output_root / "cross_dataset_source_overlap.csv", overlap)
    write_csv(output_root / "grouped_tace_summary.csv", summaries)
    write_csv(output_root / "grouped_tace_per_class.csv", per_class)
    write_csv(output_root / "grouped_fold_assignments.csv", assignments)
    write_csv(output_root / "grouped_tace_oof_predictions.csv", prediction_rows)
    manifest = {
        "schema_version": "source-family-robustness-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "family_definition": (
            "SHA-256 of comment-stripped, lower-cased Solidity tokens with string, "
            "hexadecimal, and numeric literals normalized"
        ),
        "family_scope_note": "lexical clone proxy; not semantic clone detection",
        "parameters": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "thresholds": args.thresholds,
            "shrinkage": args.shrinkage,
            "seed": args.seed,
        },
        "source_data_files": [path.name for path in source_paths],
    }
    (output_root / "source_family_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote source-family results to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
