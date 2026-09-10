from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from analyze_source_family_robustness import grouped_folds
from analyze_tace_crossfit import MODEL_ORDER, Record, selective_rows
from run_frozen_llm_inference import build_system_prompt
from summarize_repair_exposure import (
    TAXONOMIES,
    classify_attempt,
    inspect_record,
    strict_categories,
)


class RecoveryAuditTests(unittest.TestCase):
    def test_fenced_json_is_strictly_parsed(self) -> None:
        raw = '```json\n{"vulnerabilities":[{"category":"reentrancy"}]}\n```'
        self.assertEqual(
            strict_categories(raw, TAXONOMIES["smartbugs"]),
            frozenset({"reentrancy"}),
        )

    def test_out_of_taxonomy_is_rejected(self) -> None:
        raw = '{"vulnerabilities":[{"category":"imaginary_bug"}]}'
        with self.assertRaises(ValueError):
            strict_categories(raw, TAXONOMIES["smartbugs"])

    def test_failure_taxonomy_separates_schema_and_taxonomy(self) -> None:
        schema = {
            "parse_status": "valid_with_rejected_items",
            "validation_errors": ["finding[0] is not an object"],
        }
        taxonomy = {
            "parse_status": "valid_with_rejected_items",
            "validation_errors": ["outside the allowed taxonomy"],
        }
        self.assertEqual(classify_attempt(schema), "schema_violation")
        self.assertEqual(classify_attempt(taxonomy), "taxonomy_violation")

    def test_early_retry_is_recovery_exposure(self) -> None:
        row = {
            "record_index": 0,
            "source_sha256": "abc",
            "status": "ok",
            "vulnerabilities": [],
            "chunks": [
                {
                    "parse_status": "valid_after_format_retry",
                    "attempts": [
                        {
                            "parse_status": "parse_or_validation_error",
                            "error": "No valid JSON result object",
                            "raw_response": "not json",
                        },
                        {
                            "parse_status": "valid",
                            "raw_response": '{"vulnerabilities":[]}',
                        },
                    ],
                }
            ],
        }
        detail, _ = inspect_record("smartbugs", MODEL_ORDER[0], row, "test.jsonl")
        self.assertTrue(detail["any_recovery_exposure"])
        self.assertFalse(detail["first_attempt_native_valid"])
        self.assertEqual(detail["format_retry_chunks"], 1)
        self.assertEqual(detail["format_recovered_chunks"], 1)


class GroupedFoldTests(unittest.TestCase):
    def test_family_never_crosses_folds(self) -> None:
        records = {}
        families = {}
        for index in range(12):
            key = f"record-{index}"
            family = f"family-{index // 2}"
            records[key] = Record(
                key=key,
                record_index=index,
                truth=frozenset({"vulnerable"}) if index % 2 else frozenset(),
                votes={model: frozenset() for model in MODEL_ORDER},
            )
            families[key] = family
        folds = grouped_folds(
            sorted(records), records, families, "bccc", n_splits=3, seed=7
        )
        fold_by_key = {
            key: fold_index for fold_index, fold in enumerate(folds) for key in fold
        }
        for family in set(families.values()):
            assigned = {
                fold_by_key[key] for key, value in families.items() if value == family
            }
            self.assertEqual(len(assigned), 1)


class PromptAndSelectiveMetricTests(unittest.TestCase):
    def test_prompt_variants_keep_closed_schema(self) -> None:
        categories = ("reentrancy", "arithmetic")
        prompts = {
            variant: build_system_prompt(categories, variant)
            for variant in ("canonical", "concise", "evidence_checklist")
        }
        self.assertEqual(len(set(prompts.values())), 3)
        for prompt in prompts.values():
            self.assertIn("vulnerabilities", prompt)
            self.assertIn("reentrancy", prompt)
            self.assertIn("arithmetic", prompt)

    def test_positive_alert_metrics_use_true_positive_denominator(self) -> None:
        records = {
            "a": Record("a", 0, frozenset({"x"}), {model: frozenset() for model in MODEL_ORDER}),
            "b": Record("b", 1, frozenset(), {model: frozenset() for model in MODEL_ORDER}),
        }
        rows = selective_rows(
            "smartbugs", records, ["a", "b"], ["x"],
            {"a": {"x": 0.9}, "b": {"x": 0.8}}, [0.8],
        )
        self.assertEqual(rows[0]["accepted_positive_alerts"], 2)
        self.assertEqual(rows[0]["ground_truth_positive_decisions"], 1)
        self.assertAlmostEqual(rows[0]["ground_truth_positive_recall"], 1.0)
        self.assertAlmostEqual(rows[0]["accepted_positive_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
