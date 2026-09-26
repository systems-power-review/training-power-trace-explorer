#!/usr/bin/env python3
"""Build an anonymous, compact browser-display package for training traces.

The source corpus remains untouched.  This script keeps only reviewed run-level
metadata and an evenly binned aggregate-power series for each selected run.
It intentionally excludes source paths, logs, per-GPU raw measurements, and
high-frequency telemetry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_value(value: Any, default: Any = "Not reported") -> Any:
    """Convert pandas/numpy values to JSON without leaking NaNs."""
    if value is None:
        return default
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    return value


def numeric(value: Any, default: float = 0.0) -> float:
    value = json_value(value, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def label(value: Any, default: str = "Not reported") -> str:
    value = json_value(value, default)
    if value in (None, "", "Unknown", "Not found"):
        return default
    return str(value)


def trace_key(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]


def usable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Use complete aggregate-support rows when the source contains them."""
    columns = ["time_s", "aggregate_power_w", "complete_gpu_support"]
    frame = frame[columns].dropna(subset=["time_s", "aggregate_power_w"]).copy()
    complete = frame[frame.complete_gpu_support.astype(bool)]
    if len(complete) >= 2:
        frame = complete
    return frame.sort_values("time_s").drop_duplicates("time_s", keep="last")


def aggregate_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Compute physical-time metrics from the aggregate trace, not raw log ticks."""
    frame = usable_frame(frame)
    time = frame.time_s.to_numpy(dtype=float)
    power = frame.aggregate_power_w.to_numpy(dtype=float)
    if len(time) < 2:
        raise ValueError("aggregate trace must contain at least two valid points")
    intervals = np.diff(time)
    positive = intervals[intervals > 0]
    duration = float(time[-1] - time[0])
    integral_ws = float(np.trapezoid(power, time))
    anchors = time[time >= time[0] + 1.0]
    ramps = np.interp(anchors, time, power) - np.interp(anchors - 1.0, time, power)
    upward = ramps[ramps >= 0]
    downward = -ramps[ramps < 0]
    p95_up = float(np.quantile(upward, 0.95)) if len(upward) else 0.0
    p99_up = float(np.quantile(upward, 0.99)) if len(upward) else 0.0
    p99_down = float(np.quantile(downward, 0.99)) if len(downward) else 0.0
    events = int(np.count_nonzero(upward >= p95_up)) if len(upward) else 0
    return {
        "duration_s": duration,
        "median_interval_s": float(np.median(positive)) if len(positive) else 0.0,
        "p95_interval_s": float(np.quantile(positive, 0.95)) if len(positive) else 0.0,
        "mean_total_power_w": integral_ws / duration if duration > 0 else float(np.mean(power)),
        "p95_total_power_w": float(np.quantile(power, 0.95)),
        "p99_total_power_w": float(np.quantile(power, 0.99)),
        "max_total_power_w": float(np.max(power)),
        "total_energy_wh": integral_ws / 3600,
        "ramp_up_p95_1s_w_per_s": p95_up,
        "ramp_up_p99_1s_w_per_s": p99_up,
        "ramp_down_p99_1s_w_per_s": p99_down,
        "ramp_event_frequency_1s": events / max(duration / 60, 1e-9),
        "source_sample_count": int(len(frame)),
    }


def compact_samples(frame: pd.DataFrame, max_points: int) -> list[dict[str, Any]]:
    """Uniformly bin aggregate power, preserving time order and endpoint range."""
    frame = usable_frame(frame)
    if len(frame) > max_points:
        start, end = float(frame.time_s.iloc[0]), float(frame.time_s.iloc[-1])
        width = max((end - start) / max_points, 1e-9)
        frame["_bin"] = np.minimum(max_points - 1, ((frame.time_s - start) / width).astype(int))
        frame = frame.groupby("_bin", as_index=False).agg(time_s=("time_s", "mean"), aggregate_power_w=("aggregate_power_w", "mean"))
    return [
        {
            "timestamp": f"relative:{float(row.time_s):.6f}s",
            "time_relative_s": round(float(row.time_s), 6),
            "gpu_id": "Total",
            "power_w": round(float(row.aggregate_power_w), 4),
            "sm_clock_mhz": None,
            "gpu_util_pct": None,
            "memory_util_pct": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "temperature_c": None,
            "stage": None,
        }
        for row in frame.itertuples(index=False)
    ]


def training_record(row: pd.Series, public_id: str, statistics: dict[str, float | int]) -> dict[str, Any]:
    gpu_count = max(1, int(numeric(row.get("gpu_count"), 1)))
    dp = label(row.get("data_parallel_degree"))
    tp = label(row.get("tensor_parallel_degree"))
    pp = label(row.get("pipeline_parallel_degree"))
    model = label(row.get("model_repo"), label(row.get("model_variant")))
    return {
        "run_id": public_id,
        "workload_type": "Training",
        "source_family": "Reviewed training corpus",
        "source_directory": "Anonymous compact display payload",
        "trace_path": "Embedded aggregate-power display series",
        "stdout_path": None,
        "stderr_path": None,
        "plot_path": None,
        "meta_path": "Embedded reviewed metadata",
        "model": model,
        "model_family": label(row.get("model_family")),
        "model_source_label": label(row.get("model_family")),
        "model_metadata_status": "reported" if model != "Not reported" else "not_reported",
        "method": label(row.get("training_method")),
        "gpu_type": label(row.get("gpu_type")),
        "gpu_count": gpu_count,
        "precision": label(row.get("launcher_mixed_precision"), label(row.get("compute_dtype"))),
        "compute_dtype": label(row.get("compute_dtype")),
        "quantization_bits": label(row.get("quantization_bits")),
        "parallelism": f"DP={dp}, TP={tp}, PP={pp}" if "Not reported" not in (dp, tp, pp) else "Not reported",
        "sequence_length": json_value(row.get("sequence_length")),
        "microbatch_size": json_value(row.get("microbatch_size")),
        "grad_accum_steps": json_value(row.get("grad_accum_steps")),
        "global_batch_size": json_value(row.get("global_batch_size")),
        "checkpoint_interval": json_value(row.get("checkpoint_interval")),
        "dataset_name": label(row.get("dataset_name")),
        "duration_declared_min": round(float(statistics["duration_s"]) / 60, 4),
        "duration_observed_s": round(float(statistics["duration_s"]), 6),
        "sampling_interval_declared_s": json_value(row.get("sampling_interval_declared_s")),
        "sampling_interval_observed_median_s": round(float(statistics["median_interval_s"]), 6),
        "sampling_interval_observed_p95_s": round(float(statistics["p95_interval_s"]), 6),
        "has_stage_labels": False,
        "has_clock_telemetry": False,
        "has_utilization_telemetry": False,
        "has_temperature_telemetry": False,
        "quality_status": label(row.get("quality_status")),
        "mean_total_power_w": round(float(statistics["mean_total_power_w"]), 4),
        "p95_total_power_w": round(float(statistics["p95_total_power_w"]), 4),
        "p99_total_power_w": round(float(statistics["p99_total_power_w"]), 4),
        "max_total_power_w": round(float(statistics["max_total_power_w"]), 4),
        "total_energy_wh": round(float(statistics["total_energy_wh"]), 6),
        "mean_power_per_gpu_w": round(float(statistics["mean_total_power_w"]) / gpu_count, 4),
        "ramp_up_p95_1s_w_per_s": round(float(statistics["ramp_up_p95_1s_w_per_s"]), 4),
        "ramp_up_p99_1s_w_per_s": round(float(statistics["ramp_up_p99_1s_w_per_s"]), 4),
        "ramp_down_p99_1s_w_per_s": round(float(statistics["ramp_down_p99_1s_w_per_s"]), 4),
        "ramp_event_frequency_1s": round(float(statistics["ramp_event_frequency_1s"]), 6),
        "num_samples": int(statistics["source_sample_count"]),
        "num_gpus_observed": gpu_count,
        "logging_method": label(row.get("logger_type")),
        "power_aggregation": "aggregate_total_compact_display",
        "quality_flags": [
            {
                "code": "compact_aggregate_display",
                "severity": "info",
                "message": "Browser display uses a bounded, time-binned total-power series; raw per-GPU telemetry is not published here.",
            }
        ],
        "missing_fields": [
            "per_gpu_power", "sm_clock_mhz", "gpu_util_pct", "memory_used_mb",
            "memory_total_mb", "temperature_c", "stage",
        ],
        "timestamp_issues": [],
        "gpu_count_mismatch": False,
        "duplicate_warning": False,
        "display_telemetry": "aggregate total power (compact)",
    }


def select_training(runs: pd.DataFrame, partitions: Path, target: int) -> pd.DataFrame:
    runs = runs[runs.source_family.eq("PowerTraces")].copy()
    runs["_key"] = runs.run_id.map(trace_key)
    runs = runs[runs._key.map(lambda key: (partitions / f"{key}.parquet").exists())].copy()
    runs = runs[runs.quality_status.isin(["PASS_MAIN", "PASS_LIMITED"])].copy()
    runs["_quality_rank"] = runs.quality_status.map({"PASS_MAIN": 0, "PASS_LIMITED": 1})
    return runs.sort_values(
        ["_quality_rank", "large_gap_count", "maximum_gap_s", "sampling_jitter_cv", "run_id"],
        kind="stable",
    ).head(target).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inference-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-count", type=int, default=949)
    parser.add_argument("--max-points", type=int, default=600)
    parser.add_argument("--bundle-size", type=int, default=25)
    parser.add_argument("--bundle-id-map", type=Path, help="JSON object mapping bundle filenames to public Drive file IDs")
    args = parser.parse_args()

    source = args.source_root.resolve()
    partitions = source / "training_analysis_v6/data/aggregate_power_partitions"
    runs = pd.read_parquet(source / "trace_catalog/run_catalog.parquet")
    selected = select_training(runs, partitions, args.training_count)
    if len(selected) != args.training_count:
        raise SystemExit(f"Expected {args.training_count} quality-screened traces, found {len(selected)}")

    input_catalog = json.loads(args.inference_catalog.read_text())
    jask = [
        run for run in input_catalog
        if str(run.get("workload_type", "")).lower() == "inference"
        and run.get("source_family") != "Synthetic showcase"
    ]
    if len({run["run_id"] for run in jask}) != len(jask):
        raise SystemExit("Inference catalog has duplicate run IDs")

    output = args.output.resolve()
    bundles = output / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    bundle_ids = json.loads(args.bundle_id_map.read_text()) if args.bundle_id_map else {}
    training: list[dict[str, Any]] = []
    bundle_payloads: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(selected.iterrows()):
        public_id = f"training-{row._key}"
        aggregate = pd.read_parquet(partitions / f"{row._key}.parquet")
        statistics = aggregate_statistics(aggregate)
        samples = compact_samples(aggregate, args.max_points)
        record = training_record(row, public_id, statistics)
        record["display_bundle_filename"] = f"training-display-{index // args.bundle_size:03d}.json"
        record["display_bundle_entry"] = public_id
        if bundle_ids:
            record["display_bundle_file_id"] = bundle_ids[record["display_bundle_filename"]]
        training.append(record)
        bundle_payloads.append({"run": record, "samples": samples})

    for group_start in range(0, len(bundle_payloads), args.bundle_size):
        number = group_start // args.bundle_size
        filename = f"training-display-{number:03d}.json"
        payload = {entry["run"]["run_id"]: entry for entry in bundle_payloads[group_start:group_start + args.bundle_size]}
        (bundles / filename).write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    catalog = jask + training
    if len(catalog) != len(jask) + args.training_count:
        raise SystemExit("Unexpected combined catalog count")
    (output / "catalog.pending.json").write_text(json.dumps(catalog, indent=2) + "\n")
    audit = {
        "schema_version": 1,
        "catalog_count": len(catalog),
        "inference_count": len(jask),
        "training_count": len(training),
        "synthetic_count": 0,
        "training_source": "PowerTraces",
        "training_selection": {
            "target": args.training_count,
            "eligible_pass_main": int((selected.quality_status == "PASS_MAIN").sum()),
            "selected_pass_limited": int((selected.quality_status == "PASS_LIMITED").sum()),
            "sort": "PASS_MAIN first; then fewer time gaps, lower maximum gap, lower sampling jitter, run ID",
        },
        "display_payload": {
            "type": "aggregate_total_power",
            "max_points_per_training_trace": args.max_points,
            "bundle_size": args.bundle_size,
            "raw_hpc_logs_published": False,
            "per_gpu_raw_telemetry_published": False,
        },
    }
    (output / "display-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
