"""Read-only analysis of the 20 official ArUco benchmark records."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = ROOT / "benchmarks" / "aruco" / "records"
COMMITTED_SUMMARY = (
    ROOT
    / "benchmarks"
    / "aruco"
    / "summary"
    / "aruco_final_benchmark_v3_summary.json"
)

LOCATION_FOLDERS = {
    "location_1_lower_left": "L1_lower_left",
    "location_2_upper_left": "L2_upper_left",
    "location_3_center": "L3_center",
    "location_4_upper_right": "L4_upper_right",
    "location_5_lower_right": "L5_lower_right",
}
SUCCESS_FIELDS = (
    "grip_contact_accepted",
    "release_open_verified",
    "cycle_completed",
    "observed_success",
)


def load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for folder_name, location_name in LOCATION_FOLDERS.items():
        folder = RECORDS_DIR / folder_name
        paths = sorted(folder.glob("aruco_trial_*.json"))
        if len(paths) != 4:
            raise RuntimeError(
                f"{folder}: found {len(paths)} records; expected 4."
            )
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("benchmark_location") != location_name:
                raise RuntimeError(f"Location mismatch in {path}.")
            if not str(record.get("mode", "")).startswith(
                "final_aruco_benchmark_v3_location_"
            ):
                raise RuntimeError(f"Non-official ArUco mode in {path}.")
            for field in SUCCESS_FIELDS:
                if not isinstance(record.get(field), bool):
                    raise RuntimeError(f"{path}: {field} is not boolean.")
            record["_source"] = str(path.relative_to(ROOT))
            records.append(record)
    if len(records) != 20:
        raise RuntimeError(f"Found {len(records)} ArUco records; expected 20.")
    return records


def descriptive(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "sample_stdev": round(statistics.stdev(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def empirical_repeatability(
    records: list[dict[str, Any]],
    coordinate_field: str,
    location_field: str,
) -> dict[str, Any]:
    distances: list[float] = []
    per_location: dict[str, Any] = {}
    for location_name in LOCATION_FOLDERS.values():
        points = [
            tuple(map(float, record[coordinate_field][:2]))
            for record in records
            if record[location_field] == location_name
        ]
        mean_x = statistics.mean(point[0] for point in points)
        mean_y = statistics.mean(point[1] for point in points)
        local = [
            math.hypot(point[0] - mean_x, point[1] - mean_y)
            for point in points
        ]
        distances.extend(local)
        per_location[location_name] = {
            "empirical_center_px": [round(mean_x, 3), round(mean_y, 3)],
            "mean_distance_px": round(statistics.mean(local), 3),
            "max_distance_px": round(max(local), 3),
        }
    return {
        "mean_distance_px": round(statistics.mean(distances), 3),
        "max_distance_px": round(max(distances), 3),
        "per_location": per_location,
    }


def nominal_center_error(records: list[dict[str, Any]]) -> dict[str, float]:
    distances = []
    for record in records:
        center = record["center_px"]
        nominal = record["benchmark_nominal_center_px"]
        distances.append(
            math.hypot(
                float(center[0]) - float(nominal[0]),
                float(center[1]) - float(nominal[1]),
            )
        )
    return {
        "mean_distance_px": round(statistics.mean(distances), 3),
        "max_distance_px": round(max(distances), 3),
    }


def analyze() -> dict[str, Any]:
    records = load_records()
    cycle_times = [float(record["cycle_time_seconds"]) for record in records]
    successes = {
        field: sum(record[field] is True for record in records)
        for field in SUCCESS_FIELDS
    }
    successes["harmonized_full_cycle"] = sum(
        all(record[field] is True for field in SUCCESS_FIELDS)
        for record in records
    )
    per_location = {}
    for location_name in LOCATION_FOLDERS.values():
        local = [
            record for record in records
            if record["benchmark_location"] == location_name
        ]
        per_location[location_name] = {
            "trials": len(local),
            "successful_full_cycles": sum(
                all(record[field] is True for field in SUCCESS_FIELDS)
                for record in local
            ),
            "mean_cycle_time_seconds": round(
                statistics.mean(
                    float(record["cycle_time_seconds"])
                    for record in local
                ),
                3,
            ),
        }
    return {
        "method": "ArUco",
        "records": len(records),
        "successes": successes,
        "cycle_time_seconds": descriptive(cycle_times),
        "nominal_center_error": nominal_center_error(records),
        "empirical_repeatability": empirical_repeatability(
            records, "center_px", "benchmark_location"
        ),
        "per_location": per_location,
    }


def verify(report: dict[str, Any]) -> None:
    expected = {
        "records": 20,
        "successes": 20,
        "cycle": (133.489, 133.276, 0.752, 132.599, 135.402),
        "nominal": (2.362, 5.805),
        "repeatability": (1.269, 3.973),
    }
    cycle = report["cycle_time_seconds"]
    actual_cycle = (
        cycle["mean"], cycle["median"], cycle["sample_stdev"],
        cycle["min"], cycle["max"],
    )
    if report["records"] != expected["records"]:
        raise RuntimeError("Unexpected ArUco trial count.")
    if report["successes"]["harmonized_full_cycle"] != expected["successes"]:
        raise RuntimeError("Unexpected ArUco success count.")
    if actual_cycle != expected["cycle"]:
        raise RuntimeError(f"Unexpected ArUco cycle statistics: {actual_cycle}")
    nominal = report["nominal_center_error"]
    if (nominal["mean_distance_px"], nominal["max_distance_px"]) != expected["nominal"]:
        raise RuntimeError("Unexpected ArUco nominal-center statistics.")
    repeatability = report["empirical_repeatability"]
    if (
        repeatability["mean_distance_px"],
        repeatability["max_distance_px"],
    ) != expected["repeatability"]:
        raise RuntimeError("Unexpected ArUco empirical repeatability.")

    committed = json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8"))
    if committed["total_trials_found"] != report["records"]:
        raise RuntimeError("Committed ArUco summary trial count differs.")
    if committed["cycle_time_seconds"]["mean"] != cycle["mean"]:
        raise RuntimeError("Committed ArUco mean cycle time differs.")


def main() -> None:
    report = analyze()
    verify(report)
    print(json.dumps(report, indent=2))
    print("ArUco analysis matches the committed official results.")


if __name__ == "__main__":
    main()
