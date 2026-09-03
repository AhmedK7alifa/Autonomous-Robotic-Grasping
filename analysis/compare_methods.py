"""Read-only comparison of the official ArUco and YOLO benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analyze_aruco import analyze as analyze_aruco
from analyze_aruco import verify as verify_aruco
from analyze_yolo import analyze as analyze_yolo
from analyze_yolo import verify as verify_yolo


ROOT = Path(__file__).resolve().parent.parent
COMMITTED_COMPARISON = (
    ROOT / "results" / "comparison" / "aruco_vs_yolo_comparison.json"
)


def compare() -> dict[str, Any]:
    aruco = analyze_aruco()
    yolo = analyze_yolo()
    verify_aruco(aruco)
    verify_yolo(yolo)

    aruco_cycle = aruco["cycle_time_seconds"]
    yolo_cycle = yolo["cycle_time_seconds"]
    return {
        "benchmark_design": {
            "locations": 5,
            "trials_per_location": 4,
            "trials_per_method": 20,
        },
        "harmonized_success_definition": (
            "grip_contact_accepted, release_open_verified, "
            "cycle_completed, and observed_success are all true"
        ),
        "full_cycle_successes": {
            "ArUco": aruco["successes"]["harmonized_full_cycle"],
            "YOLO": yolo["successes"]["harmonized_full_cycle"],
        },
        "cycle_time_seconds": {
            "ArUco": aruco_cycle,
            "YOLO": yolo_cycle,
            "yolo_minus_aruco_mean": round(
                yolo_cycle["mean"] - aruco_cycle["mean"], 3
            ),
        },
        "harmonized_empirical_repeatability_px": {
            "ArUco_marker_center": {
                "mean": aruco["empirical_repeatability"]["mean_distance_px"],
                "max": aruco["empirical_repeatability"]["max_distance_px"],
            },
            "YOLO_bbox_center": {
                "mean": yolo["bbox_center_repeatability"]["mean_distance_px"],
                "max": yolo["bbox_center_repeatability"]["max_distance_px"],
            },
            "YOLO_upper_quarter_reference": {
                "mean": yolo["grasp_point_repeatability"]["mean_distance_px"],
                "max": yolo["grasp_point_repeatability"]["max_distance_px"],
            },
        },
        "separate_aruco_nominal_center_error_px": {
            "mean": aruco["nominal_center_error"]["mean_distance_px"],
            "max": aruco["nominal_center_error"]["max_distance_px"],
        },
    }


def verify_committed(report: dict[str, Any]) -> None:
    committed = json.loads(COMMITTED_COMPARISON.read_text(encoding="utf-8"))
    committed_times = committed["cycle_time_seconds"]
    for method in ("ArUco", "YOLO"):
        if committed_times[method] != report["cycle_time_seconds"][method]:
            raise RuntimeError(f"Committed {method} cycle statistics differ.")

    committed_repeatability = committed[
        "target_localization_repeatability"
    ]["harmonized_empirical_nominal_metrics"]
    pairs = {
        "ArUco_marker_center": "aruco_marker_center",
        "YOLO_bbox_center": "yolo_bbox_center",
        "YOLO_upper_quarter_reference": "yolo_upper_quarter_grasp_point",
    }
    for local_name, committed_name in pairs.items():
        local = report["harmonized_empirical_repeatability_px"][local_name]
        historical = committed_repeatability[committed_name]
        if local != {
            "mean": historical["overall_mean_distance_px"],
            "max": historical["overall_max_distance_px"],
        }:
            raise RuntimeError(f"Committed repeatability differs for {local_name}.")


def main() -> None:
    report = compare()
    verify_committed(report)
    print(json.dumps(report, indent=2))
    print("Method comparison matches the committed official results.")


if __name__ == "__main__":
    main()
