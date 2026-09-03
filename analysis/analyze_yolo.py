"""Read-only analysis of the 20 official YOLO benchmark records."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = ROOT / "benchmarks" / "yolo" / "records"
COMMITTED_SUMMARY = (
    ROOT
    / "benchmarks"
    / "yolo"
    / "summary"
    / "yolo_final_benchmark_summary.json"
)

LOCATION_SPECS = {
    "location_1_lower_left": (1, "L1_lower_left"),
    "location_2_upper_left": (2, "L2_upper_left"),
    "location_3_center": (3, "L3_center"),
    "location_4_upper_right": (4, "L4_upper_right"),
    "location_5_lower_right": (5, "L5_lower_right"),
}
SUCCESS_FIELDS = (
    "grip_contact_accepted",
    "release_open_verified",
    "cycle_completed",
    "observed_success",
)


def load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for folder_name, (location_number, location_name) in LOCATION_SPECS.items():
        folder = RECORDS_DIR / folder_name
        paths = sorted(folder.glob("*.json"))
        if len(paths) != 4:
            raise RuntimeError(
                f"{folder}: found {len(paths)} records; expected 4."
            )
        for trial_number, path in enumerate(paths, start=1):
            expected_name = f"yolo_L{location_number}_trial_{trial_number:02d}.json"
            if path.name != expected_name:
                raise RuntimeError(
                    f"Unexpected official YOLO filename: {path.name}."
                )
            record = json.loads(path.read_text(encoding="utf-8"))
            metadata = (
                record.get("benchmark_method"),
                record.get("benchmark_location_number"),
                record.get("benchmark_location_name"),
                record.get("benchmark_trial_number"),
                record.get("benchmark_official"),
            )
            expected = ("YOLO", location_number, location_name, trial_number, True)
            if metadata != expected:
                raise RuntimeError(f"Official metadata mismatch in {path}.")
            for field in SUCCESS_FIELDS:
                if not isinstance(record.get(field), bool):
                    raise RuntimeError(f"{path}: {field} is not boolean.")
            record["_source"] = str(path.relative_to(ROOT))
            records.append(record)
    if len(records) != 20:
        raise RuntimeError(f"Found {len(records)} YOLO records; expected 20.")
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
    records: list[dict[str, Any]], coordinate_field: str
) -> dict[str, Any]:
    distances: list[float] = []
    per_location: dict[str, Any] = {}
    for _, location_name in LOCATION_SPECS.values():
        points = [
            tuple(map(float, record[coordinate_field][:2]))
            for record in records
            if record["benchmark_location_name"] == location_name
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
    for _, location_name in LOCATION_SPECS.values():
        local = [
            record for record in records
            if record["benchmark_location_name"] == location_name
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
        "method": "YOLO",
        "records": len(records),
        "successes": successes,
        "cycle_time_seconds": descriptive(cycle_times),
        "bbox_center_repeatability": empirical_repeatability(
            records, "bbox_center_px"
        ),
        "grasp_point_repeatability": empirical_repeatability(
            records, "grasp_point_px"
        ),
        "per_location": per_location,
    }


def verify(report: dict[str, Any]) -> None:
    expected_cycle = (133.392, 133.170, 0.623, 132.722, 134.551)
    cycle = report["cycle_time_seconds"]
    actual_cycle = (
        cycle["mean"], cycle["median"], cycle["sample_stdev"],
        cycle["min"], cycle["max"],
    )
    if report["records"] != 20:
        raise RuntimeError("Unexpected YOLO trial count.")
    if report["successes"]["harmonized_full_cycle"] != 20:
        raise RuntimeError("Unexpected YOLO success count.")
    if actual_cycle != expected_cycle:
        raise RuntimeError(f"Unexpected YOLO cycle statistics: {actual_cycle}")
    bbox = report["bbox_center_repeatability"]
    if (bbox["mean_distance_px"], bbox["max_distance_px"]) != (1.485, 5.026):
        raise RuntimeError("Unexpected YOLO bbox-center repeatability.")
    grasp = report["grasp_point_repeatability"]
    if (grasp["mean_distance_px"], grasp["max_distance_px"]) != (1.503, 5.057):
        raise RuntimeError("Unexpected YOLO grasp-point repeatability.")

    committed = json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8"))
    if committed["benchmark"]["files_analyzed"] != report["records"]:
        raise RuntimeError("Committed YOLO summary trial count differs.")
    if committed["overall_cycle_time_seconds"]["mean"] != cycle["mean"]:
        raise RuntimeError("Committed YOLO mean cycle time differs.")


def main() -> None:
    report = analyze()
    verify(report)
    print(json.dumps(report, indent=2))
    print("YOLO analysis matches the committed official results.")


if __name__ == "__main__":
    main()
