# Copyright 2026 Thi-Thu-Huong Le
# SPDX-License-Identifier: Apache-2.0

"""Repair invalid chunks without re-running already valid predictions.

The initial run exposed two deterministic model behaviours that strict JSON
retries did not resolve: Qwen repeated ``{}``, and CodeLlama continued
Solidity/prose. This pass retains every accepted parent chunk and re-runs only
unresolved chunks
with three progressively simpler closed-choice response protocols.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import run_frozen_llm_inference as base


SCHEMA_VERSION = "selective-repair"
PROMPT_VERSION = "selective-repair-closed-choice"
PARENT_ACCEPTED_STATUSES = frozenset(
    {
        "valid",
        "valid_after_format_retry",
        "valid_after_structural_repair",
        "valid_after_targeted_repair",
    }
)
ACCEPTED = frozenset(set(PARENT_ACCEPTED_STATUSES) | {"valid_after_closed_choice_repair"})
LEGACY_ACCEPTED_STATUS = re.compile(
    r"^valid_after_(?:structural|targeted|closed_choice)_repair_[a-z0-9.]+$"
)
METADATA_SUFFIX = ".metadata.json"
FULL_RUN_RECORD_COUNTS = {
    "smartbugs": 143,
    "scrawld": 5664,
    "bccc": 19512,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair only unresolved chunks from complete parent runs."
    )
    parser.add_argument("--preferred-source-root", type=Path, required=True)
    parser.add_argument("--fallback-source-root", type=Path, required=True)
    parser.add_argument(
        "--fallback-search-root",
        type=Path,
        help="Search this tree for a complete full-size pair missing from the fallback root.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=base.DATASET_NAMES, default=list(base.DATASET_NAMES)
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def is_accepted(status: object) -> bool:
    value = str(status)
    return value in ACCEPTED or LEGACY_ACCEPTED_STATUS.fullmatch(value) is not None


def repair_metadata(record: dict) -> dict:
    """Return current or legacy repair metadata without naming old releases."""

    current = record.get("repair_details")
    if isinstance(current, dict):
        return current
    candidate = {}
    for key, value in record.items():
        if key.startswith("repair_") and isinstance(value, dict):
            # Historical passes appended their metadata in chronological
            # order, so the last mapping describes the retained prediction.
            candidate = value
    return candidate


def normalize_category(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def raw_candidates(chunk: dict):
    attempts = chunk.get("attempts")
    if isinstance(attempts, list):
        for position in range(len(attempts) - 1, -1, -1):
            attempt = attempts[position]
            if isinstance(attempt, dict) and str(attempt.get("raw_response", "")).strip():
                yield f"attempts[{position}]", str(attempt["raw_response"])
    raw = str(chunk.get("raw_response", ""))
    if raw.strip():
        yield "chunk.raw_response", raw


def safe_structural_repair(chunk: dict, allowed_categories: set[str]) -> dict | None:
    """Normalize only an unambiguous list of exact taxonomy strings."""

    for origin, raw in raw_candidates(chunk):
        try:
            parsed = base.extract_first_json_object(raw)
        except Exception:
            continue
        values = parsed.get("vulnerabilities")
        if not isinstance(values, list) or not values:
            continue
        if not all(isinstance(value, str) and value.strip() for value in values):
            continue
        normalized = [normalize_category(value) for value in values]
        if not all(value in allowed_categories for value in normalized):
            continue
        payload = {"vulnerabilities": [{"category": value} for value in normalized]}
        findings, errors = base.validate_response(payload, allowed_categories)
        if errors:
            continue
        return {
            "findings": base.deduplicate_findings(findings),
            "origin": origin,
            "source_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "policy": "exact_taxonomy_string_list_to_category_objects",
        }
    return None


def structural_repair_chunk(old_chunk: dict, repair: dict) -> dict:
    repaired = copy.deepcopy(old_chunk)
    repaired.update(
        {
            "parse_status": "valid_after_structural_repair",
            "validation_errors": [],
            "error": "",
            "traceback": "",
            "accepted_findings": repair["findings"],
            "repair_details": {
                "method": "deterministic_structural_normalization",
                "old_parse_status": old_chunk.get("parse_status"),
                "origin": repair["origin"],
                "source_sha256": repair["source_sha256"],
                "policy": repair["policy"],
                "resolved": True,
            },
        }
    )
    return repaired


def expected_paths(root: Path, spec: base.ModelSpec, dataset: str) -> tuple[Path, Path]:
    assert spec.revision is not None
    stem = (
        f"results_{base.safe_filename(spec.repository)}_"
        f"{spec.revision[:12]}_{dataset}"
    )
    directory = root / dataset
    return directory / f"{stem}.jsonl", directory / f"{stem}.metadata.json"


def prepare_target(
    output_path: Path,
    metadata_path: Path,
    target_metadata: dict,
    source_rows: list[dict],
) -> dict[int, dict]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        existing = base.load_json(metadata_path)
        if existing.get("run_fingerprint") != target_metadata["run_fingerprint"]:
            raise ValueError(
                f"Existing repair metadata fingerprint differs: {metadata_path}. "
                "Use a new output root."
            )
    else:
        if output_path.exists() and output_path.stat().st_size:
            raise ValueError(f"Target JSONL exists without matching metadata: {output_path}")
        base.write_metadata(metadata_path, target_metadata)
    completed = base.load_completed_records(output_path)
    source_by_index = {int(row["record_index"]): row for row in source_rows}
    for index, row in completed.items():
        source = source_by_index.get(index)
        if source is None:
            raise ValueError(f"Target contains unknown record_index {index}: {output_path}")
        for field in ("record_id", "source_id", "source_sha256", "ground_truth_binary"):
            if row.get(field) != source.get(field):
                raise ValueError(f"Target record {index} mismatches source field {field}")
    return completed


def load_matching_records(source_metadata: dict, dataset: str) -> list[dict]:
    config = source_metadata["config"]
    seed = int(config.get("seed", 2026))
    expected_count = len(source_metadata["records"])
    records = base.load_dataset(dataset, seed, None, None)
    _, digest = base.record_manifest(records)
    if digest != source_metadata.get("records_sha256"):
        records = base.load_dataset(dataset, seed, expected_count, None)
        _, digest = base.record_manifest(records)
    if digest != source_metadata.get("records_sha256"):
        raise ValueError(
            f"Local {dataset} dataset does not match the immutable parent manifest."
        )
    return records


def paired_jsonl(metadata_path: Path) -> Path:
    text = str(metadata_path)
    if not text.endswith(METADATA_SUFFIX):
        raise ValueError(f"Not a benchmark metadata filename: {metadata_path}")
    return Path(text[: -len(METADATA_SUFFIX)] + ".jsonl")


def canonical_dataset_name(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    aliases = {
        "smartbugs": "smartbugs",
        "smartbugs_curated": "smartbugs",
        "scrawld": "scrawld",
        "bccc": "bccc",
    }
    return aliases.get(normalized, normalized)


def metadata_matches(metadata: dict, spec: base.ModelSpec, dataset: str) -> bool:
    config = metadata.get("config", {})
    return (
        config.get("model") == spec.repository
        and config.get("model_revision") == spec.revision
        and canonical_dataset_name(config.get("dataset")) == dataset
    )


def infer_expected_counts(fallback_root: Path) -> dict[str, int]:
    observed: dict[str, list[int]] = defaultdict(list)
    for path in fallback_root.rglob(f"*{METADATA_SUFFIX}") if fallback_root.is_dir() else []:
        try:
            metadata = base.load_json(path)
            dataset = canonical_dataset_name(
                metadata.get("config", {}).get("dataset", "")
            )
            count = int(metadata.get("record_count", len(metadata.get("records", []))))
        except Exception:
            continue
        if dataset in base.DATASET_NAMES and count > 0:
            observed[dataset].append(count)
    counts = {}
    for dataset, values in observed.items():
        frequencies = Counter(values)
        counts[dataset] = max(frequencies, key=lambda value: (frequencies[value], value))
    return counts


def candidate_pairs(
    root: Path,
    spec: base.ModelSpec,
    dataset: str,
    expected_count: int | None,
) -> list[tuple[Path, Path, dict]]:
    candidates = []
    if not root.is_dir():
        return candidates
    for metadata_path in root.rglob(f"*{METADATA_SUFFIX}"):
        try:
            metadata = base.load_json(metadata_path)
        except Exception:
            continue
        if not metadata_matches(metadata, spec, dataset):
            continue
        count = int(metadata.get("record_count", len(metadata.get("records", []))))
        if expected_count is not None and count != expected_count:
            continue
        jsonl_path = paired_jsonl(metadata_path)
        if not jsonl_path.is_file():
            continue
        candidates.append((jsonl_path, metadata_path, metadata))
    return candidates


def describe_near_candidates(
    roots: list[Path], spec: base.ModelSpec, dataset: str
) -> list[dict]:
    descriptions = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for metadata_path in root.rglob(f"*{METADATA_SUFFIX}"):
            resolved = metadata_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                metadata = base.load_json(metadata_path)
            except Exception:
                continue
            if not metadata_matches(metadata, spec, dataset):
                continue
            jsonl_path = paired_jsonl(metadata_path)
            descriptions.append(
                {
                    "metadata": str(resolved),
                    "jsonl": str(jsonl_path.resolve()),
                    "jsonl_exists": jsonl_path.is_file(),
                    "record_count": int(
                        metadata.get("record_count", len(metadata.get("records", [])))
                    ),
                    "records_sha256": metadata.get("records_sha256"),
                }
            )
    return descriptions


def choose_pair(
    roots: list[Path],
    spec: base.ModelSpec,
    dataset: str,
    expected_count: int | None,
) -> tuple[Path, Path, dict, dict] | None:
    all_candidates = []
    seen = set()
    for root_rank, root in enumerate(roots):
        for jsonl_path, metadata_path, metadata in candidate_pairs(
            root, spec, dataset, expected_count
        ):
            resolved = metadata_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            all_candidates.append((root_rank, jsonl_path, metadata_path, metadata))
    if not all_candidates:
        return None

    fingerprints = {
        metadata.get("records_sha256") for _, _, _, metadata in all_candidates
    }
    if len(fingerprints) > 1:
        paths = [str(item[2]) for item in all_candidates]
        raise ValueError(
            f"Ambiguous complete sources for {spec.reference} on {dataset}; "
            f"record manifests differ: {paths}"
        )
    all_candidates.sort(key=lambda item: (item[0], str(item[2])))
    root_rank, jsonl_path, metadata_path, metadata = all_candidates[0]
    return jsonl_path, metadata_path, metadata, {
        "resolution": "expected_root" if root_rank == 0 else "fallback_search",
        "selected_jsonl": str(jsonl_path.resolve()),
        "selected_metadata": str(metadata_path.resolve()),
        "equivalent_candidate_count": len(all_candidates),
    }


def load_complete_pair(
    selection: tuple[Path, Path, dict, dict],
) -> tuple[list[dict], dict, dict]:
    jsonl_path, metadata_path, metadata, resolution = selection
    rows_by_index = base.load_completed_records(jsonl_path)
    manifest = metadata.get("records")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Missing ordered record manifest: {metadata_path}")
    if len(rows_by_index) != len(manifest):
        raise ValueError(
            f"Candidate is incomplete: {jsonl_path} has {len(rows_by_index)}/{len(manifest)} rows"
        )
    rows = []
    for index, expected in enumerate(manifest):
        row = rows_by_index.get(index)
        if row is None:
            raise ValueError(f"Missing record_index {index}: {jsonl_path}")
        for field in ("record_id", "source_id", "source_sha256", "ground_truth_binary"):
            if row.get(field) != expected.get(field):
                raise ValueError(f"Manifest mismatch at record {index}, field {field}: {jsonl_path}")
        rows.append(row)
    return rows, metadata, resolution


def find_parent_pair(
    args: argparse.Namespace,
    spec: base.ModelSpec,
    dataset: str,
    expected_count: int,
) -> tuple[list[dict], dict, dict]:
    preferred_selection = choose_pair(
        [args.preferred_source_root], spec, dataset, expected_count
    )
    if preferred_selection is not None:
        rows, metadata, resolution = load_complete_pair(preferred_selection)
        resolution["parent_stage"] = "preferred"
        return rows, metadata, resolution

    search_roots = [args.fallback_source_root]
    if args.fallback_search_root is not None:
        search_roots.append(args.fallback_search_root)
    fallback_selection = choose_pair(search_roots, spec, dataset, expected_count)
    if fallback_selection is None:
        searched = [str(path.resolve()) for path in search_roots]
        near = describe_near_candidates(search_roots, spec, dataset)
        raise FileNotFoundError(
            f"No complete {expected_count}-record source pair for {spec.reference} on {dataset}. "
            f"Searched: {searched}. Matching metadata candidates: "
            f"{json.dumps(near, ensure_ascii=False)}"
        )
    rows, metadata, resolution = load_complete_pair(fallback_selection)
    resolution["parent_stage"] = "fallback"
    return rows, metadata, resolution


def parse_category_marker(response: str, categories: tuple[str, ...]) -> list[dict]:
    allowed = set(categories)
    matches = re.findall(r"(?im)\bCATEGORIES\s*=\s*([^\r\n]+)", response)
    if not matches:
        raise ValueError("No CATEGORIES= marker found")
    parsed_results = []
    for match in matches:
        value = match.strip().strip("`* .")
        if value.upper() == "NONE":
            parsed_results.append(())
            continue
        names = tuple(normalize_category(part) for part in value.split(",") if part.strip())
        if not names or any(name not in allowed for name in names):
            raise ValueError(f"Invalid CATEGORIES marker: {match!r}")
        parsed_results.append(tuple(dict.fromkeys(names)))
    if len(set(parsed_results)) != 1:
        raise ValueError("Conflicting CATEGORIES markers")
    return findings_for_categories(parsed_results[0], allowed)


def parse_id_marker(response: str, categories: tuple[str, ...]) -> list[dict]:
    matches = re.findall(r"(?im)\bLABEL_IDS\s*=\s*([^\r\n]+)", response)
    if not matches:
        raise ValueError("No LABEL_IDS= marker found")
    parsed_results = []
    for match in matches:
        value = match.strip().strip("`* .")
        if value.upper() == "NONE":
            parsed_results.append(())
            continue
        if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", value):
            raise ValueError(f"Invalid LABEL_IDS marker: {match!r}")
        ids = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
        if any(index < 1 or index > len(categories) for index in ids):
            raise ValueError(f"LABEL_IDS outside range: {ids}")
        parsed_results.append(tuple(categories[index - 1] for index in ids))
    if len(set(parsed_results)) != 1:
        raise ValueError("Conflicting LABEL_IDS markers")
    return findings_for_categories(parsed_results[0], set(categories))


def parse_bitvector_marker(response: str, categories: tuple[str, ...]) -> list[dict]:
    matches = re.findall(r"(?im)\bBITVECTOR\s*=\s*([01](?:\s*,\s*[01])*)", response)
    if not matches:
        raise ValueError("No BITVECTOR= marker found")
    parsed_results = []
    for match in matches:
        bits = tuple(int(value.strip()) for value in match.split(","))
        if len(bits) != len(categories):
            raise ValueError(
                f"BITVECTOR has {len(bits)} entries; expected {len(categories)}"
            )
        parsed_results.append(tuple(category for category, bit in zip(categories, bits) if bit))
    if len(set(parsed_results)) != 1:
        raise ValueError("Conflicting BITVECTOR markers")
    return findings_for_categories(parsed_results[0], set(categories))


def findings_for_categories(names, allowed: set[str]) -> list[dict]:
    payload = {"vulnerabilities": [{"category": name} for name in names]}
    findings, errors = base.validate_response(payload, allowed)
    if errors:
        raise ValueError("; ".join(errors))
    return base.deduplicate_findings(findings)


def choice_messages(
    categories: tuple[str, ...],
    code_chunk: str,
    chunk_index: int,
    chunk_count: int,
    attempt_index: int,
    model_repository: str,
) -> tuple[list[dict[str, str]], str]:
    category_text = ", ".join(categories)
    if attempt_index == 0:
        protocol = "categories_marker"
        output_rule = (
            "On the final line return exactly CATEGORIES=NONE if no listed vulnerability is "
            "present, or CATEGORIES= followed only by one or more comma-separated exact "
            "category names from the allowed list."
        )
    elif attempt_index == 1:
        protocol = "label_ids_marker"
        mapping = "; ".join(f"{index}={name}" for index, name in enumerate(categories, 1))
        output_rule = (
            f"Category IDs are: {mapping}. On the final line return exactly LABEL_IDS=NONE "
            "or LABEL_IDS=<comma-separated numeric IDs>."
        )
    else:
        protocol = "bitvector_marker"
        output_rule = (
            f"Use category order: {category_text}. On the final line return exactly BITVECTOR="
            f" followed by {len(categories)} comma-separated 0/1 values. A 1 means the category "
            "is supported by the code; 0 means it is not."
        )
    system = (
        "You are a closed-taxonomy Solidity vulnerability classifier. Base labels only on code "
        "evidence. Do not invent or rename categories. Follow the final-line response rule."
    )
    # The output instruction deliberately follows the code. CodeLlama otherwise
    # tends to continue the Solidity source instead of answering.
    user = (
        f"Analyze chunk {chunk_index + 1} of {chunk_count}. Allowed categories: {category_text}.\n"
        f"<solidity>\n{code_chunk}\n</solidity>\n\n{output_rule}\n"
        "Do not output JSON, Markdown, Solidity, explanations, or any additional line."
    )
    if base.is_codegemma(model_repository) or base.is_deepseek_coder(model_repository):
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return messages, protocol


def targeted_choice_repair(
    old_chunk: dict,
    code_chunk: str,
    chunk_index: int,
    chunk_count: int,
    tokenizer,
    model,
    config: base.RunConfig,
) -> dict:
    repaired = copy.deepcopy(old_chunk)
    categories = base.TAXONOMIES[config.dataset]
    parsers = (parse_category_marker, parse_id_marker, parse_bitvector_marker)
    input_device = base.first_model_device(model)
    attempts = []
    started = time.perf_counter()
    repaired["parse_status"] = "repair_failed"
    repaired["accepted_findings"] = []

    for attempt_index, parser in enumerate(parsers):
        messages, protocol = choice_messages(
            categories, code_chunk, chunk_index, chunk_count, attempt_index, config.model
        )
        attempt = {
            "attempt_index": attempt_index,
            "protocol": protocol,
            "prompt_sha256": base.json_hash(messages),
            "input_tokens": None,
            "raw_response": "",
            "raw_response_with_special_tokens": "",
            "generated_token_count": 0,
            "generated_token_ids_preview": [],
            "parse_status": "not_run",
            "error": "",
            "traceback": "",
        }
        try:
            model_inputs, input_tokens = base.prepare_chat_inputs(
                tokenizer, messages, input_device, config.model
            )
            attempt["input_tokens"] = input_tokens
            if input_tokens > config.max_input_tokens:
                raise ValueError(
                    f"Closed-choice prompt length {input_tokens} exceeds "
                    f"max_input_tokens {config.max_input_tokens}"
                )
            with base.torch.inference_mode():
                output_ids = base.generate_output(model, model_inputs, tokenizer, config)
            generated_ids = base.token_ids_as_list(output_ids[0][input_tokens:])
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            response_special = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            attempt.update(
                {
                    "raw_response": response,
                    "raw_response_with_special_tokens": response_special,
                    "generated_token_count": len(generated_ids),
                    "generated_token_ids_preview": generated_ids[:64],
                }
            )
            findings = parser(response, categories)
            attempt["parse_status"] = "valid"
        except Exception as error:
            findings = []
            attempt["parse_status"] = (
                "inference_error" if not attempt["raw_response"] else "parse_or_validation_error"
            )
            attempt["error"] = f"{type(error).__name__}: {error}"
            attempt["traceback"] = traceback.format_exc(limit=16)
            if base.is_cuda_oom(error):
                base.torch.cuda.empty_cache()
        attempts.append(attempt)
        if attempt["parse_status"] == "valid":
            repaired.update(
                {
                    "input_tokens": attempt["input_tokens"],
                    "raw_response": attempt["raw_response"],
                    "raw_response_with_special_tokens": attempt["raw_response_with_special_tokens"],
                    "generated_token_count": attempt["generated_token_count"],
                    "generated_token_ids_preview": attempt["generated_token_ids_preview"],
                    "parse_status": "valid_after_closed_choice_repair",
                    "validation_errors": [],
                    "error": "",
                    "traceback": "",
                    "accepted_findings": findings,
                }
            )
            break

    repaired["repair_attempts"] = attempts
    resolved = repaired["parse_status"] == "valid_after_closed_choice_repair"
    repaired["repair_details"] = {
        "method": "progressive_closed_choice_reinference",
        "parent_parse_status": old_chunk.get("parse_status"),
        "prompt_version": PROMPT_VERSION,
        "resolved": resolved,
        "runtime_seconds": time.perf_counter() - started,
    }
    if not resolved and attempts:
        final = attempts[-1]
        repaired.update(
            {
                "input_tokens": final.get("input_tokens"),
                "raw_response": final.get("raw_response", ""),
                "raw_response_with_special_tokens": final.get("raw_response_with_special_tokens", ""),
                "generated_token_count": final.get("generated_token_count", 0),
                "generated_token_ids_preview": final.get("generated_token_ids_preview", []),
                "error": final.get("error", ""),
                "traceback": final.get("traceback", ""),
            }
        )
    return repaired


def audit_parent(rows: list[dict], dataset: str) -> Counter:
    allowed = set(base.TAXONOMIES[dataset])
    counts = Counter(records=len(rows))
    for row in rows:
        actions = set()
        for chunk in row.get("chunks", []):
            counts["chunks"] += 1
            if is_accepted(chunk.get("parse_status")):
                action = "already_valid"
            elif safe_structural_repair(chunk, allowed) is not None:
                action = "structural"
            else:
                action = "inference"
            counts[f"chunks_{action}"] += 1
            actions.add(action)
        if "inference" in actions:
            counts["records_requiring_inference"] += 1
        elif "structural" in actions:
            counts["records_structural_only"] += 1
        else:
            counts["records_already_valid"] += 1
    return counts


def make_config(parent_metadata: dict, args: argparse.Namespace) -> base.RunConfig:
    source = parent_metadata["config"]
    return base.RunConfig(
        model=str(source["model"]),
        model_revision=str(source["model_revision"]),
        dataset=str(source["dataset"]),
        seed=int(source.get("seed", 2026)),
        max_input_tokens=int(source.get("max_input_tokens", 4096)),
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=1,
        format_retries=2,
        code_chunk_tokens=int(source.get("code_chunk_tokens", 2800)),
        chunk_overlap_tokens=int(source.get("chunk_overlap_tokens", 128)),
        dtype=args.dtype,
        device_map=args.device_map,
        prompt_version=PROMPT_VERSION,
    )


def recompute_row(parent_row, record, tokenizer, model, config):
    row = copy.deepcopy(parent_row)
    allowed = set(base.TAXONOMIES[config.dataset])
    parent_chunks = row.get("chunks", [])
    code_chunks = None
    new_chunks = []
    actions = []
    started = time.perf_counter()
    for index, chunk in enumerate(parent_chunks):
        if is_accepted(chunk.get("parse_status")):
            new_chunks.append(copy.deepcopy(chunk))
            actions.append("already_valid")
            continue
        structural = safe_structural_repair(chunk, allowed)
        if structural is not None:
            new_chunks.append(structural_repair_chunk(chunk, structural))
            actions.append("structural")
            continue
        if record is None or tokenizer is None or model is None:
            raise RuntimeError("Unresolved chunk requires source code and model")
        if code_chunks is None:
            code_chunks = base.split_code(
                tokenizer,
                str(record.get("code", "")),
                config.code_chunk_tokens,
                config.chunk_overlap_tokens,
            )
            if len(code_chunks) != len(parent_chunks):
                raise ValueError(
                    f"Chunk count changed for record {row.get('record_index')}: "
                    f"parent={len(parent_chunks)}, current={len(code_chunks)}"
                )
        new_chunks.append(
            targeted_choice_repair(
                chunk, code_chunks[index], index, len(code_chunks), tokenizer, model, config
            )
        )
        actions.append("inference")

    accepted = [is_accepted(chunk.get("parse_status")) for chunk in new_chunks]
    status = "ok" if all(accepted) else ("partial" if any(accepted) else "failed")
    findings = []
    for chunk, valid in zip(new_chunks, accepted):
        if valid:
            findings.extend(chunk.get("accepted_findings", []))
    runtime = time.perf_counter() - started
    row.update(
        {
            "status": status,
            "chunk_count": len(new_chunks),
            "valid_chunk_count": sum(accepted),
            "runtime_seconds": float(row.get("runtime_seconds", 0.0)) + runtime,
            "vulnerabilities": base.deduplicate_findings(findings),
            "chunks": new_chunks,
            "repair_details": {
                "parent_status": parent_row.get("status"),
                "actions": dict(sorted(Counter(actions).items())),
                "resolved": status == "ok",
                "runtime_seconds": runtime,
                "prompt_version": PROMPT_VERSION,
            },
        }
    )
    return row


def build_metadata(parent_metadata, resolution, config):
    metadata = copy.deepcopy(parent_metadata)
    protocol = {
        "version": "selective-repair",
        "prompt_version": PROMPT_VERSION,
        "parent_run_fingerprint": parent_metadata.get("run_fingerprint"),
        "parent_resolution": resolution,
        "accepted_statuses": sorted(ACCEPTED),
        "accepted_legacy_status_pattern": LEGACY_ACCEPTED_STATUS.pattern,
        "response_protocols": ["CATEGORIES", "LABEL_IDS", "BITVECTOR"],
        "old_raw_response_in_prompt": False,
        "instruction_after_code": True,
        "max_new_tokens": config.max_new_tokens,
    }
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["config"] = {**metadata.get("config", {}), **vars(config)}
    metadata["derived_from_parent"] = resolution
    metadata["repair_protocol"] = protocol
    metadata.pop("runtime", None)
    metadata.pop("completion", None)
    metadata["run_fingerprint"] = base.json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "records_sha256": metadata.get("records_sha256"),
            "config": metadata["config"],
            "repair_protocol": protocol,
        }
    )
    return metadata


def completion(metadata_path: Path, output_path: Path) -> dict:
    rows = base.load_completed_records(output_path)
    statuses = Counter(str(row.get("status")) for row in rows.values())
    actions = Counter()
    outcomes = Counter()
    for row in rows.values():
        repair = repair_metadata(row)
        actions.update(repair.get("actions", {}))
        for chunk in row.get("chunks", []):
            repair = repair_metadata(chunk)
            if repair:
                outcomes[f"{repair.get('method')}:{repair.get('resolved')}"] += 1
    value = {
        "record_count": len(rows),
        "record_statuses": dict(sorted(statuses.items())),
        "record_actions": dict(sorted(actions.items())),
        "repair_outcomes": dict(sorted(outcomes.items())),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata = base.load_json(metadata_path)
    metadata["completion"] = value
    base.write_metadata(metadata_path, metadata)
    return value


def run_pair(spec, dataset, expected_count, args, tokenizer, model):
    parent_rows, parent_metadata, resolution = find_parent_pair(
        args, spec, dataset, expected_count
    )
    audit = audit_parent(parent_rows, dataset)
    report = {
        "model": spec.repository,
        "revision": spec.revision,
        "dataset": dataset,
        "parent_stage": resolution["parent_stage"],
        "source_resolution": resolution["resolution"],
        "selected_source": resolution["selected_jsonl"],
        **dict(audit),
    }
    if args.dry_run:
        return tokenizer, model, report

    config = make_config(parent_metadata, args)
    output_path, metadata_path = expected_paths(args.output_root, spec, dataset)
    target_metadata = build_metadata(parent_metadata, resolution, config)
    completed = prepare_target(
        output_path, metadata_path, target_metadata, parent_rows
    )
    pending = [row for row in parent_rows if int(row["record_index"]) not in completed]
    pending_audit = audit_parent(pending, dataset) if pending else Counter()
    print(
        f"Resume selective repair for {spec.reference} on {dataset}: {len(completed)}/{len(parent_rows)}; "
        f"pending inference records={pending_audit.get('records_requiring_inference', 0)}; "
        f"parent={resolution['parent_stage']} ({resolution['selected_jsonl']}).",
        flush=True,
    )
    if not pending:
        return tokenizer, model, completion(metadata_path, output_path)

    records = None
    if pending_audit.get("records_requiring_inference", 0):
        records = load_matching_records(parent_metadata, dataset)
        if tokenizer is None or model is None:
            tokenizer, model = base.load_model(spec, os.environ.get("HF_TOKEN"), config)

    started = time.perf_counter()
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for parent_row in parent_rows:
            index = int(parent_row["record_index"])
            if index in completed:
                continue
            row = recompute_row(
                parent_row,
                records[index] if records is not None else None,
                tokenizer,
                model,
                config,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            current = len(completed) + written
            if written % args.progress_every == 0 or current == len(parent_rows):
                elapsed = time.perf_counter() - started
                rate = written / elapsed if elapsed else 0.0
                eta = (len(parent_rows) - current) / rate if rate else -1
                print(
                    f"Repair progress {spec.reference} on {dataset}: "
                    f"{current}/{len(parent_rows)} ({100 * current / len(parent_rows):.1f}%), "
                    f"rate={rate:.3f} records/s, ETA={base.format_duration(eta)}.",
                    flush=True,
                )
    finished = base.load_completed_records(output_path)
    if len(finished) != len(parent_rows):
        raise RuntimeError(
            f"Incomplete repaired output: {len(finished)}/{len(parent_rows)} at {output_path}"
        )
    result = completion(metadata_path, output_path)
    print(f"Completed repair for {spec.reference} on {dataset}: {json.dumps(result, sort_keys=True)}")
    return tokenizer, model, result


def validate_args(args):
    if not args.fallback_source_root.is_dir():
        raise FileNotFoundError(f"Missing fallback source root: {args.fallback_source_root}")
    if not args.preferred_source_root.is_dir():
        raise FileNotFoundError(f"Missing preferred source root: {args.preferred_source_root}")
    if args.max_new_tokens <= 0 or args.progress_every <= 0:
        raise ValueError("max_new_tokens and progress_every must be positive")
    roots = [args.fallback_source_root.resolve(), args.preferred_source_root.resolve()]
    if args.output_root.resolve() in roots:
        raise ValueError("Repair output must use a new root; parent roots are immutable")
    specs = [base.parse_model_spec(value) for value in args.models]
    base.require_pinned_models(specs)
    return specs


def main():
    args = parse_args()
    specs = validate_args(args)
    expected_counts = dict(FULL_RUN_RECORD_COUNTS)
    observed_counts = infer_expected_counts(args.fallback_source_root)
    conflicts = {
        dataset: {"expected": expected_counts[dataset], "observed": observed}
        for dataset, observed in observed_counts.items()
        if dataset in expected_counts and observed != expected_counts[dataset]
    }
    if conflicts:
        raise ValueError(
            "Fallback metadata record counts conflict with the fixed full-run protocol: "
            f"{conflicts}"
        )
    reports = []
    for spec in specs:
        tokenizer = None
        model = None
        try:
            for dataset in args.datasets:
                tokenizer, model, report = run_pair(
                    spec, dataset, expected_counts[dataset], args, tokenizer, model
                )
                reports.append(report)
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if base.torch is not None and base.torch.cuda.is_available():
                base.torch.cuda.empty_cache()
    payload = {
        "schema_version": "selective-repair",
        "fallback_source_root": str(args.fallback_source_root.resolve()),
        "preferred_source_root": str(args.preferred_source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "expected_counts": expected_counts,
        "observed_fallback_counts": observed_counts,
        "dry_run": args.dry_run,
        "runs": reports,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif args.dry_run:
        for report in reports:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
