#!/usr/bin/env python3
"""Build a compact, identity-safe public download release for the review site.

The input data are intentionally *not* copied verbatim.  This builder keeps only
the fields required to reproduce the public visualizations, removes absolute or
source-relative paths, raw timestamps, storage identifiers, checksum manifests,
and free-text provenance, then creates a deterministic ZIP with a SHA-256
manifest.  It is designed for the 949 compact training traces and 51 inference
traces used by the anonymous review deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUN_FIELDS = (
    "run_id", "workload_type", "model", "model_family", "method",
    "inference_engine", "tensor_parallel_size", "kv_cache_quantization",
    "model_weight_quantization", "gpu_frequency_mhz", "in_flight_requests",
    "concurrency", "arrival_pattern", "arrival_rate_rps", "gpu_type",
    "gpu_count", "precision", "compute_dtype", "quantization_bits",
    "parallelism", "sequence_length", "microbatch_size", "grad_accum_steps",
    "global_batch_size", "checkpoint_interval", "duration_declared_min",
    "duration_observed_s", "sampling_interval_declared_s",
    "sampling_interval_observed_median_s", "sampling_interval_observed_p95_s",
    "has_stage_labels", "has_clock_telemetry", "has_utilization_telemetry",
    "has_temperature_telemetry", "quality_status", "mean_total_power_w",
    "p95_total_power_w", "p99_total_power_w", "max_total_power_w",
    "total_energy_wh", "mean_power_per_gpu_w", "ramp_up_p95_1s_w_per_s",
    "ramp_up_p99_1s_w_per_s", "ramp_down_p99_1s_w_per_s",
    "ramp_event_frequency_1s", "num_samples", "num_gpus_observed",
    "logging_method", "power_aggregation", "missing_fields",
    "gpu_count_mismatch", "duplicate_warning", "request_timeline_method",
    "requests_completed", "requests_arrived_derived",
)

SAMPLE_FIELDS = (
    "time_relative_s", "gpu_id", "power_w", "sm_clock_mhz", "gpu_util_pct",
    "memory_util_pct", "memory_used_mb", "memory_total_mb", "temperature_c",
    "stage",
)

TIMELINE_FIELDS = (
    "time_relative_s", "window_s", "requests_arrived", "active_requests",
    "mean_prompt_tokens", "mean_output_tokens", "mean_request_tokens",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_scalar(value: Any) -> Any:
    """Keep only JSON scalars and short lists of JSON scalars."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, list):
        return [clean_scalar(item) for item in value if item is None or isinstance(item, (bool, int, float, str))]
    return None


def scrub_run(raw: dict[str, Any], workload_type: str) -> dict[str, Any]:
    """Whitelist experimental metadata and replace provenance with generic labels."""
    clean: dict[str, Any] = {
        "run_id": raw.get("run_id"),
        "workload_type": workload_type,
        "source_family": f"Reviewed {workload_type.lower()} corpus",
        "data_release": "Anonymous compact public release",
    }
    for field in RUN_FIELDS:
        if field in raw:
            value = clean_scalar(raw[field])
            if value is not None:
                clean[field] = value
    # Remove potentially identifying free-text messages while preserving a
    # machine-readable indication that source validation took place.
    if raw.get("quality_flags"):
        clean["quality_checks_present"] = True
    return {key: value for key, value in clean.items() if value is not None}


def scrub_samples(samples: list[dict[str, Any]], max_per_gpu: int = 600) -> list[dict[str, Any]]:
    """Remove wall-clock timestamps and uniformly retain a compact trace."""
    by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        row = {key: sample.get(key) for key in SAMPLE_FIELDS if key in sample}
        if "time_relative_s" not in row or "power_w" not in row:
            continue
        by_gpu[str(row.get("gpu_id", "0"))].append(row)

    compact: list[dict[str, Any]] = []
    for gpu_id in sorted(by_gpu):
        rows = by_gpu[gpu_id]
        if len(rows) <= max_per_gpu:
            compact.extend(rows)
            continue
        indices = sorted({round(index * (len(rows) - 1) / (max_per_gpu - 1)) for index in range(max_per_gpu)})
        compact.extend(rows[index] for index in indices)
    return compact


def scrub_timeline(rows: list[dict[str, Any]], maximum: int = 600) -> list[dict[str, Any]]:
    clean = [
        {key: row.get(key) for key in TIMELINE_FIELDS if key in row}
        for row in rows if isinstance(row, dict) and "time_relative_s" in row
    ]
    if len(clean) <= maximum:
        return clean
    indices = sorted({round(index * (len(clean) - 1) / (maximum - 1)) for index in range(maximum)})
    return [clean[index] for index in indices]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    training_out = args.out / "training-bundles"
    inference_out = args.out / "inference-bundles"

    training_catalog = read_json(args.training_root / "catalog.pending.json")
    source_runs = training_catalog["runs"] if isinstance(training_catalog, dict) else training_catalog
    public_runs: list[dict[str, Any]] = []
    catalog_by_id: dict[str, dict[str, Any]] = {}

    # Rewrite every training display bundle rather than copying it, so no
    # source paths or wall-clock timestamps slip through.
    training_ids: set[str] = set()
    for bundle_path in sorted((args.training_root / "bundles").glob("training-display-*.json")):
        source_bundle = read_json(bundle_path)
        compact_bundle: dict[str, Any] = {}
        for run_id, entry in source_bundle.items():
            run = scrub_run(entry.get("run", {}), "Training")
            run["bundle_path"] = f"training-bundles/{bundle_path.name}"
            compact_bundle[run_id] = {
                "run": run,
                "samples": scrub_samples(entry.get("samples", [])),
            }
            catalog_by_id[run_id] = run
            training_ids.add(run_id)
        write_json(training_out / bundle_path.name, compact_bundle)

    # Use catalog values where compact bundle metadata is incomplete, but only
    # after passing them through the same whitelist.
    for source_run in source_runs:
        run_id = source_run.get("run_id")
        if run_id in training_ids:
            run = catalog_by_id[run_id]
            for key, value in scrub_run(source_run, "Training").items():
                if key not in run or run[key] in (None, "Not reported"):
                    run[key] = value
            public_runs.append(run)

    inference_source = sorted(args.inference_root.glob("*.json"))
    inference_entries: list[tuple[str, dict[str, Any]]] = []
    for path in inference_source:
        raw = read_json(path)
        raw_run = raw.get("run", raw)
        run = scrub_run(raw_run, "Inference")
        run_id = run.get("run_id")
        if not isinstance(run_id, str):
            raise ValueError(f"Missing run_id in {path}")
        inference_entries.append((run_id, {
            "run": run,
            "samples": scrub_samples(raw.get("samples", [])),
            "inference_timeline": scrub_timeline(raw.get("inference_timeline", [])),
        }))

    for bundle_index, offset in enumerate(range(0, len(inference_entries), 20)):
        bundle_name = f"inference-display-{bundle_index:03d}.json"
        compact_bundle: dict[str, Any] = {}
        for run_id, entry in inference_entries[offset:offset + 20]:
            entry["run"]["bundle_path"] = f"inference-bundles/{bundle_name}"
            compact_bundle[run_id] = entry
            public_runs.append(entry["run"])
        write_json(inference_out / bundle_name, compact_bundle)

    counts = Counter(run["workload_type"] for run in public_runs)
    if counts != Counter({"Training": 949, "Inference": 51}):
        raise ValueError(f"Expected 949 training and 51 inference traces; received {dict(counts)}")

    public_runs.sort(key=lambda run: (run["workload_type"], run["run_id"]))
    catalog = {
        "schema_version": "anonymous-compact-power-trace-v1",
        "release": "Anonymous compact public release for double-blind review",
        "record_count": len(public_runs),
        "workload_counts": dict(sorted(counts.items())),
        "contains_synthetic_records": False,
        "notes": [
            "This release contains visualization-ready compact telemetry only.",
            "Wall-clock timestamps, paths, logs, storage identifiers, checksums, and free-text provenance are excluded.",
            "Each telemetry series is uniformly downsampled to at most 600 points per GPU.",
        ],
        "runs": public_runs,
    }
    write_json(args.out / "catalog.json", catalog)
    (args.out / "README.md").write_text(
        "# Anonymous LLM Power Trace Dataset\n\n"
        "This compact public release contains 1,000 reviewed LLM power-trace records: 949 training and 51 inference. "
        "It is prepared for anonymous double-blind review. Each telemetry sample contains relative time and visualization fields only; "
        "the package intentionally excludes wall-clock timestamps, file paths, logs, storage identifiers, checksum source manifests, "
        "and free-text provenance. No synthetic records are included.\n\n"
        "`catalog.json` describes every run. `training-bundles/` and `inference-bundles/` contain JSON maps keyed by `run_id`. "
        "Samples are uniformly capped at 600 points per GPU; inference request telemetry is capped at 600 windows per run.\n",
        encoding="utf-8",
    )
    (args.out / "LICENSE.txt").write_text(
        "Research-use release. Do not attempt to re-identify contributors or link this release to an individual or institution.\n",
        encoding="utf-8",
    )

    manifest_files = []
    for path in sorted(item for item in args.out.rglob("*") if item.is_file() and item.name != "dataset-manifest.json"):
        manifest_files.append({
            "path": path.relative_to(args.out).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema_version": "anonymous-compact-power-trace-v1",
        "record_count": len(public_runs),
        "workload_counts": dict(sorted(counts.items())),
        "contains_synthetic_records": False,
        "files": manifest_files,
    }
    write_json(args.out / "dataset-manifest.json", manifest)

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in args.out.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(args.out.parent).as_posix())
    print(json.dumps({
        "zip": str(args.zip),
        "zip_bytes": args.zip.stat().st_size,
        "records": len(public_runs),
        "counts": dict(counts),
        "files": len(manifest_files),
    }, indent=2))


if __name__ == "__main__":
    main()
