# Copyright 2026 Thi-Thu-Huong Le
# SPDX-License-Identifier: Apache-2.0

"""Recover the missing full Qwen-BCCC prediction pair with serial GPU shards.

Each shard runs ordinary one-record-at-a-time greedy inference. There is no
micro-batching, so splitting record indices across GPUs does not introduce the
batch-dependent prediction differences observed in the earlier benchmark.
The merge step requires every expected record exactly once and validates every
row against the deterministic full BCCC manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import run_frozen_llm_inference as base


MODEL_REPOSITORY = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
EXPECTED_RECORD_COUNT = 19512
RECOVERY_PROTOCOL = "qwen-bccc-serial-shard-recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover the missing full Qwen-BCCC run with resumable serial shards."
    )
    parser.add_argument("--mode", choices=("shard", "merge", "dry-run"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=f"{MODEL_REPOSITORY}@{MODEL_REVISION}",
        help="Pinned Qwen repository@commit; the recovery refuses any other checkpoint.",
    )
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> base.ModelSpec:
    spec = base.parse_model_spec(args.model)
    base.require_pinned_models([spec])
    if spec.repository != MODEL_REPOSITORY or spec.revision != MODEL_REVISION:
        raise ValueError(
            f"Recovery is fixed to {MODEL_REPOSITORY}@{MODEL_REVISION}; received {spec.reference}"
        )
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if args.mode == "shard":
        if args.shard_index is None or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard mode requires 0 <= shard-index < num-shards")
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    return spec


def load_records(seed: int) -> tuple[list[dict], list[dict], str]:
    records = base.load_dataset("bccc", seed, None, None)
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"BCCC full protocol requires {EXPECTED_RECORD_COUNT} records; found {len(records)}"
        )
    entries, digest = base.record_manifest(records)
    return records, entries, digest


def make_config(spec: base.ModelSpec, args: argparse.Namespace) -> base.RunConfig:
    assert spec.revision is not None
    return base.RunConfig(
        model=spec.repository,
        model_revision=spec.revision,
        dataset="bccc",
        seed=args.seed,
        max_input_tokens=4096,
        max_new_tokens=256,
        min_new_tokens=1,
        format_retries=1,
        code_chunk_tokens=2800,
        chunk_overlap_tokens=128,
        dtype=args.dtype,
        device_map=args.device_map,
        prompt_version="closed-taxonomy-json-with-chunk-status",
    )


def canonical_stem(spec: base.ModelSpec) -> str:
    assert spec.revision is not None
    return (
        f"results_{base.safe_filename(spec.repository)}_"
        f"{spec.revision[:12]}_bccc"
    )


def shard_paths(root: Path, shard_index: int, num_shards: int) -> tuple[Path, Path]:
    directory = root / "shards"
    label = f"qwen_bccc_shard_{shard_index:02d}_of_{num_shards:02d}"
    return directory / f"{label}.jsonl", directory / f"{label}.manifest.json"


def merged_paths(root: Path, spec: base.ModelSpec) -> tuple[Path, Path]:
    directory = root / "bccc"
    stem = canonical_stem(spec)
    return directory / f"{stem}.jsonl", directory / f"{stem}.metadata.json"


def shard_indices(record_count: int, shard_index: int, num_shards: int) -> list[int]:
    return list(range(shard_index, record_count, num_shards))


def make_shard_manifest(
    config: base.RunConfig,
    records: list[dict],
    shard_index: int,
    num_shards: int,
) -> dict:
    manifest = base.build_run_manifest(config, records)
    indices = shard_indices(len(records), shard_index, num_shards)
    recovery = {
        "protocol": RECOVERY_PROTOCOL,
        "mode": "serial_single_record_no_batching",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "expected_shard_records": len(indices),
        "index_rule": "record_index % num_shards == shard_index",
        "full_records_sha256": manifest["records_sha256"],
    }
    manifest["recovery_shard"] = recovery
    manifest["run_fingerprint"] = base.json_hash(
        {
            "base_run_fingerprint": manifest["run_fingerprint"],
            "recovery_shard": recovery,
        }
    )
    return manifest


def validate_shard_rows(
    records: list[dict],
    completed: dict[int, dict],
    expected_indices: set[int],
    path: Path,
) -> None:
    base.validate_completed_records(records, completed)
    unexpected = sorted(set(completed) - expected_indices)
    if unexpected:
        raise ValueError(f"Shard contains indices assigned elsewhere: {unexpected[:20]} at {path}")


def update_shard_completion(
    metadata_path: Path,
    output_path: Path,
    expected_count: int,
) -> None:
    metadata = base.load_json(metadata_path)
    rows = base.load_completed_records(output_path)
    metadata["shard_completion"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_records": len(rows),
        "expected_records": expected_count,
        "status_counts": dict(
            sorted(
                Counter(str(row.get("status")) for row in rows.values()).items()
            )
        ),
    }
    base.write_metadata(metadata_path, metadata)


def run_shard(args: argparse.Namespace, spec: base.ModelSpec) -> None:
    assert args.shard_index is not None
    records, _, records_sha256 = load_records(args.seed)
    config = make_config(spec, args)
    output_path, metadata_path = shard_paths(
        args.output_root, args.shard_index, args.num_shards
    )
    manifest = make_shard_manifest(config, records, args.shard_index, args.num_shards)
    if manifest["records_sha256"] != records_sha256:
        raise RuntimeError("Internal BCCC manifest digest disagreement")
    expected = set(shard_indices(len(records), args.shard_index, args.num_shards))
    completed_indices = base.prepare_run(metadata_path, output_path, manifest, records)
    completed = base.load_completed_records(output_path)
    validate_shard_rows(records, completed, expected, output_path)
    pending = sorted(expected - completed_indices)
    print(
        f"Shard {args.shard_index}/{args.num_shards}: "
        f"{len(completed_indices)}/{len(expected)} complete, {len(pending)} pending.",
        flush=True,
    )
    if not pending:
        update_shard_completion(metadata_path, output_path, len(expected))
        print(f"Shard already complete: {output_path}", flush=True)
        return

    base.require_inference_dependencies()
    base.set_seed(args.seed)
    tokenizer, model = base.load_model(spec, os.environ.get("HF_TOKEN"), config)
    base.attach_runtime_metadata(metadata_path, tokenizer, model, manifest)
    started = time.perf_counter()
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for record_index in pending:
            result = base.evaluate_record(
                records[record_index], record_index, tokenizer, model, config
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            current = len(completed_indices) + written
            if written % args.progress_every == 0 or current == len(expected):
                elapsed = time.perf_counter() - started
                rate = written / elapsed if elapsed else 0.0
                remaining = len(expected) - current
                eta = remaining / rate if rate else -1.0
                print(
                    f"Shard {args.shard_index}/{args.num_shards}: "
                    f"{current}/{len(expected)} ({100 * current / len(expected):.1f}%), "
                    f"rate={rate:.4f} records/s, ETA={base.format_duration(eta)}.",
                    flush=True,
                )

    finished = base.load_completed_records(output_path)
    validate_shard_rows(records, finished, expected, output_path)
    if set(finished) != expected:
        raise RuntimeError(
            f"Shard incomplete after run: {len(finished)}/{len(expected)} at {output_path}"
        )
    update_shard_completion(metadata_path, output_path, len(expected))
    del model, tokenizer
    if base.torch.cuda.is_available():
        base.torch.cuda.empty_cache()
    print(f"SHARD COMPLETE: {output_path}", flush=True)


def make_merged_manifest(
    config: base.RunConfig,
    records: list[dict],
    shard_manifests: list[dict],
    num_shards: int,
) -> dict:
    manifest = base.build_run_manifest(config, records)
    recovery = {
        "protocol": RECOVERY_PROTOCOL,
        "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_shards": num_shards,
        "merge_order": "ascending_record_index",
        "inference_mode": "serial_single_record_no_batching",
        "shard_run_fingerprints": [item["run_fingerprint"] for item in shard_manifests],
    }
    manifest["recovery_merge"] = recovery
    manifest["run_fingerprint"] = base.json_hash(
        {
            "base_run_fingerprint": manifest["run_fingerprint"],
            "recovery_merge": {
                key: value for key, value in recovery.items() if key != "recovered_at_utc"
            },
        }
    )
    return manifest


def merge_shards(args: argparse.Namespace, spec: base.ModelSpec) -> None:
    records, _, records_sha256 = load_records(args.seed)
    config = make_config(spec, args)
    merged: dict[int, dict] = {}
    shard_manifests = []
    for shard_index in range(args.num_shards):
        jsonl_path, metadata_path = shard_paths(
            args.output_root, shard_index, args.num_shards
        )
        if not jsonl_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing shard {shard_index}: {jsonl_path}, {metadata_path}"
            )
        metadata = base.load_json(metadata_path)
        recovery = metadata.get("recovery_shard", {})
        if (
            recovery.get("protocol") != RECOVERY_PROTOCOL
            or recovery.get("shard_index") != shard_index
            or recovery.get("num_shards") != args.num_shards
            or metadata.get("records_sha256") != records_sha256
        ):
            raise ValueError(f"Shard metadata mismatch: {metadata_path}")
        rows = base.load_completed_records(jsonl_path)
        expected = set(shard_indices(len(records), shard_index, args.num_shards))
        validate_shard_rows(records, rows, expected, jsonl_path)
        if set(rows) != expected:
            raise ValueError(
                f"Shard {shard_index} incomplete: {len(rows)}/{len(expected)}"
            )
        overlap = set(merged) & set(rows)
        if overlap:
            raise ValueError(f"Duplicate record indices across shards: {sorted(overlap)[:20]}")
        merged.update(rows)
        shard_manifests.append(metadata)

    expected_all = set(range(len(records)))
    if set(merged) != expected_all:
        missing = sorted(expected_all - set(merged))
        raise ValueError(f"Merged shards are missing records: {missing[:20]}")
    base.validate_completed_records(records, merged)
    final_jsonl, final_metadata = merged_paths(args.output_root, spec)
    final_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest = make_merged_manifest(config, records, shard_manifests, args.num_shards)
    temporary_jsonl = final_jsonl.with_name(final_jsonl.name + ".tmp")
    temporary_metadata = final_metadata.with_name(final_metadata.name + ".tmp")
    with temporary_jsonl.open("w", encoding="utf-8") as handle:
        for record_index in range(len(records)):
            handle.write(json.dumps(merged[record_index], ensure_ascii=False) + "\n")
    temporary_rows = base.load_completed_records(temporary_jsonl)
    if len(temporary_rows) != len(records):
        raise RuntimeError("Temporary merged JSONL failed record-count verification")
    base.validate_completed_records(records, temporary_rows)
    base.write_metadata(temporary_metadata, manifest)
    temporary_jsonl.replace(final_jsonl)
    temporary_metadata.replace(final_metadata)
    print(f"MERGED JSONL: {final_jsonl}")
    print(f"MERGED METADATA: {final_metadata}")
    print(f"MERGED RECORDS: {len(temporary_rows)}/{len(records)}")
    print(f"RECORDS SHA256: {records_sha256}")


def dry_run(args: argparse.Namespace, spec: base.ModelSpec) -> None:
    records, _, digest = load_records(args.seed)
    print(
        json.dumps(
            {
                "model": spec.reference,
                "dataset": "bccc",
                "record_count": len(records),
                "records_sha256": digest,
                "num_shards": args.num_shards,
                "shard_record_counts": [
                    len(shard_indices(len(records), index, args.num_shards))
                    for index in range(args.num_shards)
                ],
                "inference_batch_size": 1,
                "decoding": "greedy",
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    spec = validate_args(args)
    if args.mode == "shard":
        run_shard(args, spec)
    elif args.mode == "merge":
        merge_shards(args, spec)
    else:
        dry_run(args, spec)


if __name__ == "__main__":
    main()
