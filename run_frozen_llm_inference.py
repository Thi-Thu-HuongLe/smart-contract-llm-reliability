"""Failure-aware, resumable smart-contract vulnerability detection.

This is the GPU inference pipeline for the reliability benchmark. It deliberately
refuses an unpinned model revision and refuses to resume when the input
manifest, prompt, or decoding configuration differs from the previous run.
Each result is stored as JSONL with raw model output and parse status so a
negative prediction is distinguishable from a failed generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Keep pre-flight checks and unit tests usable on a machine without the H100
# runtime.  PyTorch and Transformers are imported only for an actual run.
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
set_seed = None


REPOSITORY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "model_predictions"
DEFAULT_MODELS = (
    "codellama/CodeLlama-7b-Instruct-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
)
DATASET_NAMES = ("smartbugs", "scrawld", "bccc")
MANIFEST_SCHEMA_VERSION = 13
REVISION_PATTERN = re.compile(r"[0-9a-f]{7,64}$")
ACCEPTED_CHUNK_STATUSES = frozenset({"valid", "valid_after_format_retry"})

TAXONOMIES = {
    "smartbugs": (
        "access_control",
        "arithmetic",
        "bad_randomness",
        "denial_of_service",
        "other",
        "reentrancy",
        "short_addresses",
        "time_manipulation",
        "transaction_ordering",
        "unchecked_low_level_calls",
    ),
    "scrawld": (
        "arithmetic",
        "denial_of_service",
        "locked_ether",
        "reentrancy",
        "time_manipulation",
        "transaction_ordering",
        "tx_origin",
        "unchecked_low_level_calls",
    ),
    "bccc": (
        "access_control",
        "arithmetic",
        "bad_randomness",
        "denial_of_service",
        "locked_ether",
        "reentrancy",
        "short_addresses",
        "time_manipulation",
        "transaction_ordering",
        "tx_origin",
        "unchecked_low_level_calls",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    revision: str | None

    @property
    def reference(self) -> str:
        return f"{self.repository}@{self.revision}" if self.revision else self.repository


@dataclass(frozen=True)
class RunConfig:
    model: str
    model_revision: str
    dataset: str
    seed: int
    max_input_tokens: int
    max_new_tokens: int
    code_chunk_tokens: int
    chunk_overlap_tokens: int
    dtype: str
    device_map: str
    # Defaults keep third-party or legacy pre-flight tests able to construct
    # this configuration while the command-line path records the values
    # explicitly in every new manifest.
    min_new_tokens: int = 1
    format_retries: int = 1
    prompt_variant: str = "canonical"
    prompt_version: str = "closed-taxonomy-json-with-chunk-status"
    decoding: str = "greedy"
    trust_remote_code: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pinned, failure-aware smart-contract detection experiment. "
            "Use repository@full_commit_sha for every actual model run."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES)
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per attempt; compact JSON needs far fewer than prose.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=1,
        help="Require at least this many generated tokens before EOS.",
    )
    parser.add_argument(
        "--format-retries",
        type=int,
        default=1,
        help="Additional deterministic retries for malformed or out-of-taxonomy JSON.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=("canonical", "concise", "evidence_checklist"),
        default="canonical",
        help=(
            "Controlled prompt variant for a predeclared sensitivity subset. "
            "All variants retain the same taxonomy and JSON schema."
        ),
    )
    parser.add_argument("--code-chunk-tokens", type=int, default=2800)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=128)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--bccc-secure-count",
        type=int,
        help="Secure BCCC contracts to sample; defaults to the vulnerable count.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help=(
            "Print resumable progress, throughput, and an ETA after this many newly "
            "written records. This is logging-only and does not change the run manifest."
        ),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate datasets and print deterministic run manifests without GPU access or writes.",
    )
    return parser.parse_args()


def parse_model_spec(value: str) -> ModelSpec:
    repository, separator, revision = value.rpartition("@")
    if not separator:
        return ModelSpec(repository=value, revision=None)
    if not repository or not REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            "Model specifications must use repository@full_commit_sha; "
            f"received {value!r}."
        )
    return ModelSpec(repository=repository, revision=revision)


def require_pinned_models(specs: Iterable[ModelSpec]) -> None:
    unpinned = [spec.repository for spec in specs if spec.revision is None]
    if unpinned:
        raise ValueError(
            "Refusing an unpinned experiment. Supply each model as "
            "repository@full_commit_sha. Unpinned repositories: "
            + ", ".join(unpinned)
        )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_identifier(record: dict, index: int) -> str:
    return str(
        record.get("address")
        or record.get("name")
        or record.get("contract_index")
        or f"contract_{index}"
    )


def source_digest(code: str) -> str:
    return hashlib.sha256(code.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def stable_id(record: dict, index: int) -> str:
    """Return a unique run identity even when an input dataset repeats an address."""

    digest = source_digest(str(record.get("code", "")))[:16]
    return f"{source_identifier(record, index)}::{digest}::{index:08d}"


def validate_records(dataset_name: str, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"{dataset_name} has no records after sampling.")
    for index, record in enumerate(records):
        if not str(record.get("code", "")).strip():
            raise ValueError(f"{dataset_name} record {index} has empty source code.")
        if dataset_name == "scrawld" and not record.get("address"):
            raise ValueError(f"ScrawlD record {index} has no address.")
        if dataset_name == "bccc" and record.get("vulnerablility") not in (0, 1):
            raise ValueError(
                f"BCCC record {index} has invalid binary label "
                f"{record.get('vulnerablility')!r}."
            )


def deduplicate_bccc_split(records: list[dict], split_name: str) -> list[dict]:
    """Remove exact BCCC duplicates and fail on conflicting reuse of an address."""

    by_address: dict[str, tuple[str, object]] = {}
    unique: list[dict] = []
    seen_exact: set[tuple[str, str, object]] = set()
    for index, record in enumerate(records):
        address = str(record.get("address", ""))
        digest = source_digest(str(record.get("code", "")))
        label = record.get("vulnerablility")
        if address in by_address and by_address[address] != (digest, label):
            raise ValueError(
                f"Conflicting BCCC {split_name} records share address {address!r}."
            )
        by_address[address] = (digest, label)
        exact_key = (address, digest, label)
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        unique.append(record)
    return unique


def select_bccc_records(
    vulnerable: list[dict],
    secure: list[dict],
    seed: int,
    secure_count: int | None,
    max_records: int | None,
) -> list[dict]:
    """Create a deterministic balanced BCCC set, including for small smoke runs."""

    rng = random.Random(seed)
    requested_secure = secure_count if secure_count is not None else len(vulnerable)
    if requested_secure > len(secure):
        raise ValueError(
            f"Requested {requested_secure} secure BCCC contracts, but only {len(secure)} remain after deduplication."
        )
    selected_secure = rng.sample(secure, requested_secure)
    selected_vulnerable = list(vulnerable)

    if max_records is not None:
        positive_target = min(len(selected_vulnerable), (max_records + 1) // 2)
        negative_target = min(len(selected_secure), max_records - positive_target)
        remaining = max_records - positive_target - negative_target
        if remaining:
            add_positive = min(remaining, len(selected_vulnerable) - positive_target)
            positive_target += add_positive
            remaining -= add_positive
        if remaining:
            negative_target += min(remaining, len(selected_secure) - negative_target)
        selected_vulnerable = rng.sample(selected_vulnerable, positive_target)
        selected_secure = rng.sample(selected_secure, negative_target)

    records = selected_vulnerable + selected_secure
    rng.shuffle(records)
    return records


def load_dataset(
    name: str,
    seed: int,
    max_records: int | None,
    bccc_secure_count: int | None,
) -> list[dict]:
    if name == "smartbugs":
        records = load_json(DATA_ROOT / "smartbugs_curated.json")
        if max_records is not None:
            records = records[:max_records]
    elif name == "scrawld":
        records = load_json(
            DATA_ROOT / "scrawld_vulnerabilities.json"
        )
        if max_records is not None:
            records = records[:max_records]
    elif name == "bccc":
        vulnerable = deduplicate_bccc_split(
            load_json(DATA_ROOT / "bccc_vulnerable.json"),
            "vulnerable",
        )
        secure = deduplicate_bccc_split(
            load_json(DATA_ROOT / "bccc_secure.json"), "secure"
        )
        records = select_bccc_records(
            vulnerable,
            secure,
            seed=seed,
            secure_count=bccc_secure_count,
            max_records=max_records,
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    validate_records(name, records)
    return records


def build_system_prompt(categories: Iterable[str], variant: str = "canonical") -> str:
    category_text = ", ".join(categories)
    if variant == "concise":
        return (
            "Classify only vulnerabilities supported by the Solidity code. "
            f"Allowed categories: {category_text}. "
            "Return only one compact JSON object using "
            "{\"vulnerabilities\":[{\"category\":\"<allowed category>\"}]}. "
            "Use {\"vulnerabilities\":[]} when none is supported; do not use Markdown."
        )
    checklist = (
        " Before answering, check every allowed category against explicit code evidence, "
        "reject findings based only on names, and verify the schema. Do not reveal this "
        "check; output only the JSON object."
        if variant == "evidence_checklist" else ""
    )
    if variant not in {"canonical", "evidence_checklist"}:
        raise ValueError(f"Unsupported prompt variant: {variant}")
    return f"""You are a closed-taxonomy smart-contract vulnerability classifier.
Identify only vulnerabilities supported by the supplied Solidity code chunk.

Allowed categories (use these exact strings only):
{category_text}

Return exactly one compact JSON object. Its first character must be {{ and its
last character must be }}. Do not return Markdown, explanations, code fences,
or fields other than category.

Schema: {{"vulnerabilities":[{{"category":"<allowed category>"}}]}}
If no allowed vulnerability is present, return {{"vulnerabilities":[]}}.
List each category at most once. Do not infer a vulnerability from naming alone;
base every category on code evidence.{checklist}"""


def build_messages(
    system_prompt: str,
    chunk: str,
    chunk_index: int,
    chunk_count: int,
    is_format_retry: bool,
    model_repository: str,
) -> tuple[list[dict[str, str]], str]:
    retry_notice = (
        " Your preceding answer was not valid closed-taxonomy JSON. Reply with the "
        "compact JSON schema only."
        if is_format_retry
        else ""
    )
    analysis_request = (
        f"Analyze chunk {chunk_index + 1} of {chunk_count} from the same "
        f"contract:\n```solidity\n{chunk}\n```{retry_notice}"
    )
    # CodeGemma's documented Gemma chat template supports user/model turns,
    # not a separate system role. Consolidate the instructions into one user
    # turn, then let prepare_chat_inputs apply the tokenizer's native template.
    if is_codegemma(model_repository):
        return (
            [{"role": "user", "content": f"{system_prompt}\n\n{analysis_request}"}],
            "codegemma_single_user",
        )
    # Retain the legacy DeepSeek path for reproducibility of diagnostic runs,
    # although that checkpoint is no longer part of the default experiment.
    if model_repository.lower().startswith("deepseek-ai/deepseek-coder"):
        return (
            [{"role": "user", "content": f"{system_prompt}\n\n{analysis_request}"}],
            "deepseek_manual_instruction_response",
        )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": analysis_request,
        },
    ], "system_plus_user"


def split_code(
    tokenizer,
    code: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    token_ids = tokenizer.encode(code, add_special_tokens=False)
    if len(token_ids) <= chunk_tokens:
        return [code]
    if overlap_tokens >= chunk_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than code_chunk_tokens")
    step = chunk_tokens - overlap_tokens
    return [
        tokenizer.decode(token_ids[start : start + chunk_tokens], skip_special_tokens=True)
        for start in range(0, len(token_ids), step)
    ]


def extract_first_json_object(text: str) -> dict:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced)
    decoder = json.JSONDecoder()

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict) and "vulnerabilities" in value:
                return value
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            # Do not accept a nested finding object from an incomplete outer
            # response.  In particular, {"category": ...} is not a result.
            if isinstance(value, dict) and "vulnerabilities" in value:
                return value
    raise ValueError("No valid JSON result object with a 'vulnerabilities' field found")


def validate_response(value: dict, allowed_categories: set[str]) -> tuple[list[dict], list[str]]:
    vulnerabilities = value.get("vulnerabilities")
    if not isinstance(vulnerabilities, list):
        raise ValueError("The 'vulnerabilities' field is not a list")

    valid: list[dict] = []
    validation_errors: list[str] = []
    for index, finding in enumerate(vulnerabilities):
        if not isinstance(finding, dict):
            validation_errors.append(f"finding[{index}] is not an object")
            continue
        raw_category = finding.get("category")
        category = re.sub(r"[^a-z0-9]+", "_", str(raw_category).lower()).strip("_")
        if category not in allowed_categories:
            validation_errors.append(
                f"finding[{index}] category {raw_category!r} is outside the allowed taxonomy"
            )
            continue
        valid.append(
            {
                "category": category,
                "function": str(finding.get("function", "")).strip(),
                "severity": str(finding.get("severity", "")).lower().strip(),
                "description": str(finding.get("description", "")).strip(),
                "recommendation": str(finding.get("recommendation", "")).strip(),
            }
        )
    return valid, validation_errors


def deduplicate_findings(findings: Iterable[dict]) -> list[dict]:
    unique: dict[tuple[str, str, str], dict] = {}
    for finding in findings:
        key = (
            finding["category"],
            normalize_for_key(finding["function"]),
            normalize_for_key(finding["description"]),
        )
        unique.setdefault(key, finding)
    return list(unique.values())


def is_accepted_chunk_status(status: object) -> bool:
    """Return True only for fully validated chunks.

    ``valid_with_rejected_items`` deliberately is not accepted.  Using a
    prefix check here would silently turn a schema/taxonomy failure into a
    successful negative prediction.
    """

    return status in ACCEPTED_CHUNK_STATUSES


def normalize_for_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_manifest(records: list[dict]) -> tuple[list[dict], str]:
    entries = [
        {
            "record_index": index,
            "record_id": stable_id(record, index),
            "source_id": source_identifier(record, index),
            "source_sha256": source_digest(str(record.get("code", ""))),
            "ground_truth_binary": record.get("vulnerablility"),
        }
        for index, record in enumerate(records)
    ]
    return entries, json_hash(entries)


def build_run_manifest(config: RunConfig, records: list[dict]) -> dict:
    entries, records_sha256 = record_manifest(records)
    labels = Counter(
        str(entry["ground_truth_binary"])
        for entry in entries
        if entry["ground_truth_binary"] in (0, 1)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "prompt_sha256": hashlib.sha256(
            build_system_prompt(
                TAXONOMIES[config.dataset], config.prompt_variant
            ).encode("utf-8")
        ).hexdigest(),
        "record_count": len(entries),
        "records_sha256": records_sha256,
        # Keep the exact, ordered input manifest.  This lets the separate
        # evaluator count missing JSONL rows as failures rather than quietly
        # scoring only the rows that happened to be written.
        "records": entries,
        "binary_label_counts": dict(sorted(labels.items())),
    }
    manifest["run_fingerprint"] = json_hash(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at_utc", "run_fingerprint", "records"}
        }
    )
    return manifest


def write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if not path.is_file():
        raise RuntimeError(f"Metadata write did not create {path}.")


def load_completed_records(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    completed: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_index = int(record["record_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Corrupt JSONL at {path}:{line_number}") from error
            if record_index in completed:
                raise ValueError(f"Duplicate record_index {record_index} in {path}.")
            completed[record_index] = record
    return completed


def validate_completed_records(records: list[dict], completed: dict[int, dict]) -> set[int]:
    entries, _ = record_manifest(records)
    expected = {entry["record_index"]: entry for entry in entries}
    for record_index, result in completed.items():
        if record_index not in expected:
            raise ValueError(f"Result contains unknown record_index {record_index}.")
        for field in ("record_id", "source_id", "source_sha256", "ground_truth_binary"):
            if result.get(field) != expected[record_index][field]:
                raise ValueError(
                    f"Result record {record_index} does not match the current input manifest "
                    f"for {field}. Start a new output root."
                )
    return set(completed)


def prepare_run(
    metadata_path: Path,
    output_path: Path,
    manifest: dict,
    records: list[dict],
) -> set[int]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        try:
            existing = load_json(metadata_path)
        except Exception as error:
            raise ValueError(f"Corrupt run metadata: {metadata_path}") from error
        if existing.get("run_fingerprint") != manifest["run_fingerprint"]:
            raise ValueError(
                "Existing run metadata has a different configuration or input manifest. "
                "Use a new output root; do not resume this JSONL file."
            )
    else:
        if output_path.exists() and output_path.stat().st_size:
            raise ValueError(
                f"{output_path} exists without matching metadata. Refusing unsafe resume."
            )
        write_metadata(metadata_path, manifest)
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Run metadata is missing after preparation: {metadata_path}. "
            "Refusing to load a model without a manifest."
        )
    return validate_completed_records(records, load_completed_records(output_path))


def require_inference_dependencies() -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer, set_seed
    if torch is not None:
        return
    try:
        import torch as imported_torch
        from transformers import (
            AutoModelForCausalLM as imported_model,
            AutoTokenizer as imported_tokenizer,
            set_seed as imported_set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "GPU inference requires PyTorch and Transformers. Run this command in the H100 environment. "
            "Use --dry-run on a machine without those dependencies."
        ) from error
    torch = imported_torch
    AutoModelForCausalLM = imported_model
    AutoTokenizer = imported_tokenizer
    set_seed = imported_set_seed


def load_model(spec: ModelSpec, hf_token: str | None, config: RunConfig):
    require_inference_dependencies()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repository,
        revision=spec.revision,
        token=hf_token,
        trust_remote_code=False,
        use_fast=should_use_fast_tokenizer(spec.repository),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.repository,
        revision=spec.revision,
        token=hf_token,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map=config.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def first_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEEPSEEK_DEFAULT_SYSTEM_PROMPT = (
    "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
    "developed by DeepSeek Company, and you only answer questions related to "
    "computer science."
)


def is_deepseek_coder(model_repository: str) -> bool:
    return model_repository.lower().startswith("deepseek-ai/deepseek-coder")


def is_codegemma(model_repository: str) -> bool:
    return model_repository.lower().startswith("google/codegemma-")


def should_use_fast_tokenizer(model_repository: str) -> bool:
    """Avoid a tokenizers-backend incompatibility in Mistral v0.3 JSON files."""

    return (
        model_repository.lower()
        != "mistralai/mistral-7b-instruct-v0.3"
    )


def build_deepseek_prompt(tokenizer, user_content: str) -> str:
    """Build the documented Instruction/Response prompt without a chat-template API."""

    bos_token = tokenizer.bos_token or ""
    return (
        f"{bos_token}{DEEPSEEK_DEFAULT_SYSTEM_PROMPT}\n"
        f"### Instruction:\n{user_content}\n"
        "### Response:"
    )


def prepare_chat_inputs(
    tokenizer,
    messages: list[dict[str, str]],
    input_device,
    model_repository: str,
):
    """Build a model input mapping compatible with modern and older tokenizers.

    DeepSeek-Coder uses its documented Instruction/Response text template to
    avoid a broken chat-template rendering in some older Transformers stacks.
    Other models use mapping-style chat inputs, with a tensor fallback for
    older Transformers releases.
    """

    if is_deepseek_coder(model_repository):
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("DeepSeek-Coder must receive one consolidated user message.")
        encoded = tokenizer(
            build_deepseek_prompt(tokenizer, messages[0]["content"]),
            add_special_tokens=False,
            return_tensors="pt",
        )
        return normalize_chat_encoding(encoded, input_device)
    if not should_use_fast_tokenizer(model_repository):
        # Mistral v0.3 deliberately uses the slow tokenizer.  That tokenizer
        # returns a tensor and treats return_dict=True as an unknown tokenizer
        # kwarg, otherwise emitting one warning per prompt.  Omitting only that
        # unsupported logging-related kwarg preserves the encoded token IDs.
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            truncation=False,
        )
        return normalize_chat_encoding(encoded, input_device)
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
        )
    except (TypeError, KeyError, AttributeError):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=False,
        )
    return normalize_chat_encoding(encoded, input_device)


def normalize_chat_encoding(encoded, input_device):
    """Normalize BatchEncoding or a directly returned rank-2 tensor."""

    if hasattr(encoded, "keys"):
        if "input_ids" not in encoded:
            raise KeyError("Chat-template mapping does not contain input_ids.")
        input_ids = encoded["input_ids"]
        if hasattr(encoded, "to"):
            model_inputs = encoded.to(input_device)
        else:
            model_inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
    else:
        # Older/slow tokenizer paths can return a tensor even when
        # return_dict=True was requested. Never index that tensor with a string.
        input_ids = encoded
        if not hasattr(input_ids, "to") or not hasattr(input_ids, "shape"):
            raise TypeError(
                "Chat template returned neither a mapping nor a tensor-like value."
            )
        model_inputs = {"input_ids": input_ids.to(input_device)}

    if len(input_ids.shape) != 2:
        raise ValueError(
            f"Expected rank-2 input_ids, received shape {tuple(input_ids.shape)}."
        )
    return model_inputs, int(input_ids.shape[-1])


def token_ids_as_list(token_ids) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def is_cuda_oom(error: Exception) -> bool:
    cuda = getattr(torch, "cuda", None)
    error_type = getattr(cuda, "OutOfMemoryError", None)
    return error_type is not None and isinstance(error, error_type)


def generate_output(model, model_inputs: dict, tokenizer, config: RunConfig):
    """Generate with modern mapping inputs and a narrow legacy-test fallback."""

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "min_new_tokens": effective_min_new_tokens(config),
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.0 if is_deepseek_coder(config.model) else 1.1,
    }
    try:
        return model.generate(**model_inputs, **generation_kwargs)
    except TypeError as error:
        # Earlier CPU mock tests expected input_ids as a positional argument.
        # Real Transformers models use the mapping-style call above.  Do not
        # mask unrelated TypeErrors from a real inference runtime.
        if (
            set(model_inputs) == {"input_ids"}
            and "required positional argument" in str(error)
        ):
            return model.generate(model_inputs["input_ids"], **generation_kwargs)
        raise


def effective_min_new_tokens(config: RunConfig) -> int:
    """Return the actual EOS floor used for the model-native generation call."""

    return (
        0
        if is_deepseek_coder(config.model)
        else config.min_new_tokens
    )


def evaluate_record(
    record: dict,
    record_index: int,
    tokenizer,
    model,
    config: RunConfig,
) -> dict:
    code = str(record.get("code", ""))
    categories = TAXONOMIES[config.dataset]
    allowed_categories = set(categories)
    chunks = split_code(
        tokenizer,
        code,
        chunk_tokens=config.code_chunk_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    system_prompt = build_system_prompt(categories, config.prompt_variant)
    chunk_results: list[dict] = []
    aggregate_findings: list[dict] = []
    started = time.perf_counter()
    input_device = first_model_device(model)

    for chunk_index, chunk in enumerate(chunks):
        chunk_result = {
            "chunk_index": chunk_index,
            "input_tokens": None,
            "raw_response": "",
            "raw_response_with_special_tokens": "",
            "generated_token_count": 0,
            "generated_token_ids_preview": [],
            "parse_status": "not_run",
            "validation_errors": [],
            "error": "",
            "traceback": "",
            "attempts": [],
        }
        for attempt_index in range(config.format_retries + 1):
            messages, chat_protocol = build_messages(
                system_prompt,
                chunk,
                chunk_index,
                len(chunks),
                is_format_retry=attempt_index > 0,
                model_repository=config.model,
            )
            attempt = {
                "attempt_index": attempt_index,
                "chat_protocol": chat_protocol,
                "generation_min_new_tokens": effective_min_new_tokens(config),
                "generation_repetition_penalty": (
                    1.0 if is_deepseek_coder(config.model) else 1.1
                ),
                "input_tokens": None,
                "raw_response": "",
                "raw_response_with_special_tokens": "",
                "generated_token_count": 0,
                "generated_token_ids_preview": [],
                "parse_status": "not_run",
                "validation_errors": [],
                "error": "",
                "traceback": "",
            }
            try:
                model_inputs, input_tokens = prepare_chat_inputs(
                    tokenizer, messages, input_device, config.model
                )
                attempt["input_tokens"] = input_tokens
                chunk_result["input_tokens"] = input_tokens
                if input_tokens > config.max_input_tokens:
                    raise ValueError(
                        f"Prompt length {input_tokens} exceeds max_input_tokens "
                        f"{config.max_input_tokens}"
                    )
                with torch.inference_mode():
                    output_ids = generate_output(model, model_inputs, tokenizer, config)
                generated_ids = token_ids_as_list(output_ids[0][input_tokens:])
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                response_with_special_tokens = tokenizer.decode(
                    generated_ids, skip_special_tokens=False
                ).strip()
                attempt.update(
                    {
                        "raw_response": response,
                        "raw_response_with_special_tokens": response_with_special_tokens,
                        "generated_token_count": len(generated_ids),
                        "generated_token_ids_preview": generated_ids[:64],
                    }
                )
            except Exception as error:  # Inference failures must be observable, not fatal.
                attempt["parse_status"] = "inference_error"
                attempt["error"] = f"{type(error).__name__}: {error}"
                attempt["traceback"] = traceback.format_exc(limit=16)
                chunk_result.update(attempt)
                chunk_result["attempts"].append(attempt)
                if is_cuda_oom(error):
                    torch.cuda.empty_cache()
                break

            try:
                parsed = extract_first_json_object(response)
                findings, validation_errors = validate_response(parsed, allowed_categories)
                attempt["validation_errors"] = validation_errors
                if validation_errors:
                    attempt["parse_status"] = "valid_with_rejected_items"
                    attempt["error"] = "; ".join(validation_errors)
                else:
                    attempt["parse_status"] = "valid"
            except Exception as error:
                findings = []
                attempt["parse_status"] = "parse_or_validation_error"
                attempt["error"] = f"{type(error).__name__}: {error}"

            chunk_result.update(attempt)
            chunk_result["attempts"].append(attempt)
            if attempt["parse_status"] == "valid":
                chunk_result["parse_status"] = (
                    "valid" if attempt_index == 0 else "valid_after_format_retry"
                )
                chunk_result["error"] = ""
                chunk_result["accepted_findings"] = findings
                aggregate_findings.extend(findings)
                break
        chunk_results.append(chunk_result)

    valid_chunks = sum(
        is_accepted_chunk_status(chunk.get("parse_status")) for chunk in chunk_results
    )
    if all(is_accepted_chunk_status(chunk.get("parse_status")) for chunk in chunk_results):
        overall_status = "ok"
    elif any(is_accepted_chunk_status(chunk.get("parse_status")) for chunk in chunk_results):
        overall_status = "partial"
    else:
        overall_status = "failed"

    return {
        "record_index": record_index,
        "record_id": stable_id(record, record_index),
        "source_id": source_identifier(record, record_index),
        "source_sha256": source_digest(code),
        "ground_truth_binary": record.get("vulnerablility"),
        "status": overall_status,
        "chunk_count": len(chunks),
        "valid_chunk_count": valid_chunks,
        "runtime_seconds": time.perf_counter() - started,
        "vulnerabilities": deduplicate_findings(aggregate_findings),
        "chunks": chunk_results,
    }


def attach_runtime_metadata(path: Path, tokenizer, model, manifest: dict) -> None:
    """Attach runtime facts without losing the input manifest.

    A network filesystem or an interrupted pre-load setup should never cause a
    successfully loaded model to fail merely because the small JSON manifest is
    temporarily absent.  The manifest is already held in memory and is written
    back verbatim before runtime data are added.
    """

    if path.is_file():
        metadata = load_json(path)
        if metadata.get("run_fingerprint") != manifest.get("run_fingerprint"):
            raise ValueError(
                "Run metadata changed after preparation. Refusing to attach runtime facts "
                "to a different experiment."
            )
    else:
        metadata = dict(manifest)
        metadata["metadata_recovered_before_runtime"] = True
    metadata["runtime"] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_commit_reported": getattr(model.config, "_commit_hash", None),
        "tokenizer_commit_reported": tokenizer.init_kwargs.get("_commit_hash"),
        "torch_version": torch.__version__,
        "transformers_model_class": type(model).__name__,
        "tokenizer_special_tokens": {
            "bos_token": getattr(tokenizer, "bos_token", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        },
        "model_generation_config": {
            "eos_token_id": getattr(
                getattr(model, "generation_config", None), "eos_token_id", None
            ),
            "pad_token_id": getattr(
                getattr(model, "generation_config", None), "pad_token_id", None
            ),
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    write_metadata(path, metadata)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def run_model_dataset(
    spec: ModelSpec,
    dataset_name: str,
    records: list[dict],
    args: argparse.Namespace,
    hf_token: str | None,
) -> None:
    assert spec.revision is not None
    config = RunConfig(
        model=spec.repository,
        model_revision=spec.revision,
        dataset=dataset_name,
        seed=args.seed,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        format_retries=args.format_retries,
        prompt_variant=args.prompt_variant,
        code_chunk_tokens=args.code_chunk_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    run_dir = args.output_root / dataset_name
    run_dir.mkdir(parents=True, exist_ok=True)
    variant_suffix = "" if args.prompt_variant == "canonical" else f"_{args.prompt_variant}"
    stem = (
        f"results_{safe_filename(spec.repository)}_{spec.revision[:12]}_"
        f"{dataset_name}{variant_suffix}"
    )
    output_path = run_dir / f"{stem}.jsonl"
    metadata_path = run_dir / f"{stem}.metadata.json"
    manifest = build_run_manifest(config, records)
    completed = prepare_run(metadata_path, output_path, manifest, records)
    existing_count = len(completed)
    total_count = len(records)
    print(
        f"Resume check {spec.reference} on {dataset_name}: "
        f"{existing_count}/{total_count} records already valid.",
        flush=True,
    )
    if existing_count == total_count:
        print(
            f"Already complete {spec.reference} on {dataset_name}; model loading skipped.",
            flush=True,
        )
        return

    tokenizer, model = load_model(spec, hf_token, config)
    attach_runtime_metadata(metadata_path, tokenizer, model, manifest)
    invocation_started = time.perf_counter()
    newly_written = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for index, record in enumerate(records):
            if index in completed:
                continue
            result = evaluate_record(record, index, tokenizer, model, config)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            newly_written += 1
            current_count = existing_count + newly_written
            if newly_written % args.progress_every == 0 or current_count == total_count:
                elapsed = time.perf_counter() - invocation_started
                rate = newly_written / elapsed if elapsed > 0 else 0.0
                remaining = total_count - current_count
                eta = remaining / rate if rate > 0 else -1.0
                print(
                    f"Progress {spec.reference} on {dataset_name}: "
                    f"{current_count}/{total_count} "
                    f"({100.0 * current_count / total_count:.1f}%), "
                    f"new={newly_written}, rate={rate:.3f} records/s, "
                    f"ETA={format_duration(eta)}.",
                    flush=True,
                )
            # Do not call torch.cuda.empty_cache() after every record. It forces
            # allocator churn and synchronization in this long inference loop.
            # Cached blocks are reusable; explicit cleanup remains at dataset
            # boundaries and on caught CUDA OOM exceptions.

    finished = load_completed_records(output_path)
    if len(finished) != len(records):
        raise RuntimeError(
            f"Run stopped before all records were written: {len(finished)}/{len(records)}."
        )
    validate_completed_records(records, finished)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Completed {spec.reference} on {dataset_name}: {len(records)} records")


def print_dry_run(specs: list[ModelSpec], datasets: dict[str, list[dict]], args: argparse.Namespace) -> None:
    print("Dry run: no model is loaded and no files are written.")
    for dataset_name, records in datasets.items():
        labels = Counter(
            str(record.get("vulnerablility"))
            for record in records
            if record.get("vulnerablility") in (0, 1)
        )
        _, records_sha256 = record_manifest(records)
        label_note = f", binary labels={dict(sorted(labels.items()))}" if labels else ""
        print(
            f"{dataset_name}: records={len(records)}, records_sha256={records_sha256}{label_note}"
        )
    for spec in specs:
        state = "PINNED" if spec.revision else "UNPINNED (will be rejected for inference)"
        print(f"model: {spec.reference} [{state}]")
    print(
        "Validated settings: "
        f"seed={args.seed}, max_input_tokens={args.max_input_tokens}, "
        f"max_new_tokens={args.max_new_tokens}, min_new_tokens={args.min_new_tokens}, "
        f"format_retries={args.format_retries}, prompt_variant={args.prompt_variant}."
    )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if args.min_new_tokens < 0 or args.min_new_tokens > args.max_new_tokens:
        raise ValueError("min_new_tokens must be between 0 and max_new_tokens.")
    if args.format_retries < 0:
        raise ValueError("format_retries must be non-negative.")
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive.")
    if args.code_chunk_tokens + args.max_new_tokens >= args.max_input_tokens:
        raise ValueError(
            "code_chunk_tokens + max_new_tokens must be smaller than max_input_tokens "
            "to leave room for the chat template and taxonomy."
        )
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("max_records must be positive when supplied.")
    specs = [parse_model_spec(value) for value in args.models]
    datasets = {
        dataset_name: load_dataset(
            dataset_name,
            seed=args.seed,
            max_records=args.max_records,
            bccc_secure_count=args.bccc_secure_count,
        )
        for dataset_name in args.datasets
    }
    if args.dry_run:
        print_dry_run(specs, datasets, args)
        return

    require_pinned_models(specs)
    require_inference_dependencies()
    hf_token = os.environ.get("HF_TOKEN")
    set_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for dataset_name, records in datasets.items():
        for spec in specs:
            run_model_dataset(spec, dataset_name, records, args, hf_token)


if __name__ == "__main__":
    main()
