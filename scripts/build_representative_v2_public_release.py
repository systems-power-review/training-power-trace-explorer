#!/usr/bin/env python3
"""Create an identity-safe anonymous public release from Representative-v2.

The input package is never uploaded verbatim. This tool rewrites every run and
telemetry record through a strict allow-list, preserves bounded per-GPU traces,
and produces both a portable ZIP and a Drive-catalog template for the website.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
FORBIDDEN = re.compile(r"/(?:Users|scratch)/|@[A-Za-z0-9.-]+|(?:haoch|hx2493|leo\.hxu)", re.I)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, list):
        return [clean_scalar(item) for item in value if item is None or isinstance(item, (bool, int, float, str))]
    return None


def scrub_run(raw: dict[str, Any], workload_type: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_id": raw.get("run_id"),
        "workload_type": workload_type,
        "source_family": f"Reviewed {workload_type.lower()} corpus",
        "data_release": "Anonymous representative release",
    }
    for field in RUN_FIELDS:
        if field in raw:
            clean = clean_scalar(raw[field])
            if clean is not None:
                result[field] = clean
    if raw.get("quality_flags"):
        result["quality_checks_present"] = True
    return {key: value for key, value in result.items() if value is not None}


def compact(rows: list[dict[str, Any]], maximum: int, fields: tuple[str, ...], required: tuple[str, ...]) -> list[dict[str, Any]]:
    result = [{key: row.get(key) for key in fields if key in row} for row in rows if isinstance(row, dict) and all(key in row for key in required)]
    if len(result) <= maximum:
        return result
    indexes = sorted({round(i * (len(result) - 1) / (maximum - 1)) for i in range(maximum)})
    return [result[i] for i in indexes]


def scrub_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or "time_relative_s" not in row or "power_w" not in row:
            continue
        cleaned = {key: row.get(key) for key in SAMPLE_FIELDS if key in row}
        gpu_id = str(cleaned.get("gpu_id", "0"))
        if gpu_id.lower() == "total":
            raise ValueError("Representative-v2 must contain individual GPU samples, not a total-only series")
        by_gpu[gpu_id].append(cleaned)
    output: list[dict[str, Any]] = []
    for gpu_id in sorted(by_gpu, key=lambda value: int(value) if value.isdigit() else value):
        output.extend(compact(by_gpu[gpu_id], 600, SAMPLE_FIELDS, ("time_relative_s", "power_w")))
    return output


def validate_entry(run: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    if not isinstance(run.get("run_id"), str):
        raise ValueError("run_id is required")
    observed = {str(row.get("gpu_id")) for row in samples}
    expected = run.get("gpu_count")
    if isinstance(expected, int) and expected > 0 and len(observed) != expected:
        raise ValueError(f"{run['run_id']}: expected {expected} GPUs but found {sorted(observed)}")
    if len(samples) < max(2, len(observed) * 2):
        raise ValueError(f"{run['run_id']}: insufficient public telemetry samples")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_anonymous(root: Path) -> None:
    for path in root.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        match = FORBIDDEN.search(text)
        if match:
            raise ValueError(f"Anonymity scan found {match.group()!r} in {path.relative_to(root)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--bundle-id-map-json", help="JSON object mapping public bundle names to Google Drive file IDs")
    parser.add_argument("--drive-catalog", type=Path, help="Write the browser catalog after resolving public Drive IDs")
    args = parser.parse_args()

    source_catalog = read_json(args.source / "catalog.json")
    source_runs = source_catalog["runs"]
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    catalog_by_id = {run["run_id"]: run for run in source_runs}
    public_runs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    bundle_count = 0
    for bundle_path in sorted(args.source.glob("*-bundles/*.json")):
        raw_bundle = read_json(bundle_path)
        if not isinstance(raw_bundle, dict):
            raise ValueError(f"{bundle_path} is not a run-id map")
        first_id = next(iter(raw_bundle))
        workload = raw_bundle[first_id]["run"]["workload_type"]
        prefix = "training" if workload == "Training" else "inference"
        target_name = f"representative-v2-{prefix}-{bundle_count:03d}.json"
        bundle_count += 1
        cleaned_bundle: dict[str, Any] = {}
        for run_id, entry in raw_bundle.items():
            raw_run = {**catalog_by_id.get(run_id, {}), **entry.get("run", {})}
            run = scrub_run(raw_run, workload)
            samples = scrub_samples(entry.get("samples", []))
            validate_entry(run, samples)
            detail: dict[str, Any] = {"run": run, "samples": samples}
            if workload == "Inference":
                detail["inference_timeline"] = compact(entry.get("inference_timeline", []), 600, TIMELINE_FIELDS, ("time_relative_s",))
            cleaned_bundle[run_id] = detail
            catalog_run = dict(run)
            catalog_run["display_bundle_filename"] = target_name
            catalog_run["display_bundle_entry"] = run_id
            public_runs.append(catalog_run)
            counts[workload] += 1
        write_json(args.out / "bundles" / target_name, cleaned_bundle)

    expected_counts = Counter(source_catalog.get("workload_counts", {}))
    if counts != expected_counts or len(public_runs) != source_catalog.get("record_count"):
        raise ValueError(f"Record count mismatch: expected {dict(expected_counts)}, got {dict(counts)}")
    public_runs.sort(key=lambda run: (run["workload_type"], run["run_id"]))

    portable_runs = []
    for run in public_runs:
        portable = dict(run)
        portable["bundle_path"] = f"bundles/{portable.pop('display_bundle_filename')}"
        portable_runs.append(portable)
    portable_catalog = {
        "schema_version": "anonymous-representative-power-trace-v2",
        "release": "Anonymous representative public release for double-blind review",
        "record_count": len(portable_runs),
        "workload_counts": dict(sorted(counts.items())),
        "contains_synthetic_records": False,
        "notes": [
            "All training records include bounded individual-GPU display telemetry.",
            "Wall-clock timestamps, file paths, storage identifiers, logs, checksums, and free-text provenance are excluded.",
            "Each GPU series and inference request timeline is capped at 600 uniformly selected points.",
        ],
        "runs": portable_runs,
    }
    write_json(args.out / "catalog.json", portable_catalog)
    write_json(args.out.parent / "catalog-drive-template.json", public_runs)
    if args.drive_catalog:
        if not args.bundle_id_map_json:
            raise ValueError("--drive-catalog requires --bundle-id-map-json")
        bundle_ids = json.loads(args.bundle_id_map_json)
        drive_runs = []
        for run in public_runs:
            filename = run["display_bundle_filename"]
            if filename not in bundle_ids:
                raise ValueError(f"No public Drive file ID for {filename}")
            drive_run = {key: value for key, value in run.items() if key != "display_bundle_filename"}
            drive_run["display_bundle_file_id"] = bundle_ids[filename]
            drive_runs.append(drive_run)
        write_json(args.drive_catalog, drive_runs)
    (args.out / "README.md").write_text(
        "# Anonymous LLM Power Trace Dataset — Representative v2\n\n"
        f"This anonymous double-blind review release contains {len(portable_runs)} representative records: "
        f"{counts['Training']} training and {counts['Inference']} inference. Training records provide compact individual-GPU power "
        "traces rather than fabricated per-GPU estimates. The package excludes wall-clock timestamps, paths, logs, storage "
        "identifiers, checksums from source packages, and free-text provenance. No synthetic records are included.\n\n"
        "`catalog.json` lists runs. Each JSON file under `bundles/` is a map from `run_id` to the corresponding run metadata and telemetry.\n",
        encoding="utf-8",
    )
    (args.out / "LICENSE.txt").write_text(
        "Research-use release for anonymous review. Do not attempt to re-identify contributors or link the release to an individual or institution.\n",
        encoding="utf-8",
    )
    manifest_files = [
        {"path": path.relative_to(args.out).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.out.rglob("*")) if path.is_file() and path.name != "dataset-manifest.json"
    ]
    write_json(args.out / "dataset-manifest.json", {
        "schema_version": "anonymous-representative-power-trace-v2",
        "record_count": len(portable_runs),
        "workload_counts": dict(sorted(counts.items())),
        "contains_synthetic_records": False,
        "files": manifest_files,
    })
    assert_anonymous(args.out)
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in args.out.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(args.out.parent).as_posix())
    print(json.dumps({"records": len(public_runs), "workloads": dict(counts), "bundles": bundle_count, "zip": str(args.zip), "zip_bytes": args.zip.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
