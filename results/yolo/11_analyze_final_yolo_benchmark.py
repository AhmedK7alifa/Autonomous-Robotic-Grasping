"""Analyze only the 20 official final YOLO benchmark trial records."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR / "yolo_final_benchmark"
OUTPUT_DIRECTORY = BENCHMARK_ROOT / "summary"

LOCATION_SPECS = (
    (1, "L1_lower_left", "location_1_lower_left"),
    (2, "L2_upper_left", "location_2_upper_left"),
    (3, "L3_center", "location_3_center"),
    (4, "L4_upper_right", "location_4_upper_right"),
    (5, "L5_lower_right", "location_5_lower_right"),
)

EXPECTED_TRIALS_PER_LOCATION = 4
EXPECTED_TOTAL_TRIALS = 20
BENCHMARK_METADATA_FIELDS = {
    "benchmark_method",
    "benchmark_location_number",
    "benchmark_location_name",
    "benchmark_trial_number",
    "benchmark_official",
}
ANALYSIS_FIELDS = (
    "grip_contact_accepted",
    "release_open_verified",
    "cycle_completed",
    "observed_success",
    "cycle_time_seconds",
    "bbox_center_px",
    "grasp_point_px",
    "triangle_ids",
    "correction_mode",
    "dynamic_hold_margin",
    "dynamic_hold_cap",
    "retention_policy",
)
FULL_SUCCESS_FIELDS = (
    "grip_contact_accepted",
    "release_open_verified",
    "cycle_completed",
    "observed_success",
)

TXT_OUTPUT = OUTPUT_DIRECTORY / "yolo_final_benchmark_summary.txt"
JSON_OUTPUT = OUTPUT_DIRECTORY / "yolo_final_benchmark_summary.json"
CSV_OUTPUT = OUTPUT_DIRECTORY / "yolo_final_benchmark_trials.csv"


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def valid_point(value: Any) -> tuple[float, float] | None:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and finite_number(value[0])
        and finite_number(value[1])
    ):
        return float(value[0]), float(value[1])

    return None


def full_cycle_success(trial: dict[str, Any]) -> bool:
    """Require every primary successful-completion field to be true."""
    return all(trial.get(field) is True for field in FULL_SUCCESS_FIELDS)


def percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def descriptive_statistics(values: list[float]) -> dict[str, Any]:
    """Use sample standard deviation, matching the final ArUco analyzer."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_stdev": None,
            "min": None,
            "max": None,
        }

    sample_stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
    return {
        "count": len(values),
        "mean": rounded(statistics.mean(values)),
        "median": rounded(statistics.median(values)),
        "sample_stdev": rounded(sample_stdev),
        "min": rounded(min(values)),
        "max": rounded(max(values)),
    }


def duplicate_payload_key(record: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key not in BENCHMARK_METADATA_FIELDS and not key.startswith("_")
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_and_validate_trials() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    issues: list[str] = []
    location_file_counts: dict[str, int] = {}
    payload_paths: dict[str, list[str]] = defaultdict(list)
    timestamp_paths: dict[str, list[str]] = defaultdict(list)

    for location_number, location_name, folder_name in LOCATION_SPECS:
        folder = BENCHMARK_ROOT / folder_name

        if not folder.is_dir():
            issues.append(f"Missing official input folder: {folder}")
            location_file_counts[location_name] = 0
            continue

        paths = sorted(folder.glob("*.json"))
        location_file_counts[location_name] = len(paths)

        if len(paths) != EXPECTED_TRIALS_PER_LOCATION:
            issues.append(
                f"{location_name}: found {len(paths)} JSON files; "
                f"expected {EXPECTED_TRIALS_PER_LOCATION}."
            )

        embedded_trial_numbers: list[int] = []

        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                issues.append(f"{path}: malformed or unreadable JSON: {error}")
                continue

            if not isinstance(record, dict):
                issues.append(f"{path}: top-level JSON value is not an object.")
                continue

            missing_metadata = sorted(
                BENCHMARK_METADATA_FIELDS.difference(record)
            )

            if missing_metadata:
                issues.append(
                    f"{path.name}: missing benchmark metadata: "
                    + ", ".join(missing_metadata)
                )

            embedded_location = record.get("benchmark_location_number")
            embedded_name = record.get("benchmark_location_name")
            embedded_trial = record.get("benchmark_trial_number")

            if embedded_location != location_number:
                issues.append(
                    f"{path.name}: location number {embedded_location!r}; "
                    f"expected {location_number}."
                )

            if embedded_name != location_name:
                issues.append(
                    f"{path.name}: location name {embedded_name!r}; "
                    f"expected {location_name!r}."
                )

            if record.get("benchmark_method") != "YOLO":
                issues.append(
                    f"{path.name}: benchmark_method must be 'YOLO'."
                )

            if record.get("benchmark_official") is not True:
                issues.append(
                    f"{path.name}: benchmark_official must be true."
                )

            if (
                not isinstance(embedded_trial, int)
                or isinstance(embedded_trial, bool)
                or embedded_trial not in range(1, 5)
            ):
                issues.append(
                    f"{path.name}: invalid benchmark trial number "
                    f"{embedded_trial!r}."
                )
            else:
                embedded_trial_numbers.append(embedded_trial)
                expected_filename = (
                    f"yolo_L{location_number}_trial_"
                    f"{embedded_trial:02d}.json"
                )

                if path.name != expected_filename:
                    issues.append(
                        f"{path.name}: expected filename "
                        f"{expected_filename}."
                    )

            for field in FULL_SUCCESS_FIELDS:
                if field in record and not isinstance(record[field], bool):
                    issues.append(
                        f"{path.name}: {field} must be boolean when present."
                    )

            if (
                "cycle_time_seconds" in record
                and not finite_number(record["cycle_time_seconds"])
            ):
                issues.append(
                    f"{path.name}: cycle_time_seconds must be finite numeric "
                    "when present."
                )

            record["_source_path"] = str(path.resolve())
            record["_folder_name"] = folder_name
            trials.append(record)
            payload_paths[duplicate_payload_key(record)].append(str(path))

            timestamp = record.get("timestamp_start")

            if timestamp is not None:
                timestamp_paths[str(timestamp)].append(str(path))

        expected_trials = [1, 2, 3, 4]

        if sorted(embedded_trial_numbers) != expected_trials:
            issues.append(
                f"{location_name}: embedded trial numbers are "
                f"{sorted(embedded_trial_numbers)}; expected {expected_trials}."
            )

        duplicate_numbers = sorted(
            number
            for number, count in Counter(embedded_trial_numbers).items()
            if count > 1
        )

        if duplicate_numbers:
            issues.append(
                f"{location_name}: duplicate trial numbers "
                f"{duplicate_numbers}."
            )

    if len(trials) != EXPECTED_TOTAL_TRIALS:
        issues.append(
            f"Parsed {len(trials)} official records; "
            f"expected {EXPECTED_TOTAL_TRIALS}."
        )

    for paths in payload_paths.values():
        if len(paths) > 1:
            issues.append(
                "Duplicate underlying trial payload: " + " | ".join(paths)
            )

    for timestamp, paths in timestamp_paths.items():
        if len(paths) > 1:
            issues.append(
                f"Duplicate timestamp_start {timestamp!r}: "
                + " | ".join(paths)
            )

    field_availability = {
        field: sum(field in trial for trial in trials)
        for field in ANALYSIS_FIELDS
    }
    integrity = {
        "passed": not issues,
        "expected_total_trials": EXPECTED_TOTAL_TRIALS,
        "files_parsed": len(trials),
        "expected_trials_per_location": EXPECTED_TRIALS_PER_LOCATION,
        "files_per_location": location_file_counts,
        "duplicate_payload_groups": sum(
            len(paths) > 1 for paths in payload_paths.values()
        ),
        "duplicate_start_timestamp_groups": sum(
            len(paths) > 1 for paths in timestamp_paths.values()
        ),
        "field_availability": field_availability,
        "issues": issues,
    }

    if issues:
        print("OFFICIAL YOLO BENCHMARK INTEGRITY: FAIL")

        for issue in issues:
            print(f"- {issue}")

        raise SystemExit(1)

    trials.sort(
        key=lambda trial: (
            trial["benchmark_location_number"],
            trial["benchmark_trial_number"],
        )
    )
    return trials, integrity


def completed_cycle_times(trials: list[dict[str, Any]]) -> list[float]:
    return [
        float(trial["cycle_time_seconds"])
        for trial in trials
        if (
            trial.get("cycle_completed") is True
            and finite_number(trial.get("cycle_time_seconds"))
        )
    ]


def repeatability_summary(
    trials: list[dict[str, Any]],
    coordinate_field: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    per_location: dict[str, Any] = {}
    distances_by_path: dict[str, float] = {}
    overall_distances: list[float] = []

    for location_number, location_name, _ in LOCATION_SPECS:
        location_trials = [
            trial
            for trial in trials
            if trial["benchmark_location_number"] == location_number
        ]
        points = [
            valid_point(trial.get(coordinate_field))
            for trial in location_trials
        ]

        if any(point is None for point in points) or not points:
            per_location[location_name] = {
                "available": False,
                "coordinate_count": sum(point is not None for point in points),
                "nominal_center_px": None,
                "mean_distance_px": None,
                "max_distance_px": None,
            }
            continue

        valid_points = [point for point in points if point is not None]
        nominal_x = statistics.mean(point[0] for point in valid_points)
        nominal_y = statistics.mean(point[1] for point in valid_points)
        location_distances = []

        for trial, point in zip(location_trials, valid_points):
            distance = math.hypot(
                point[0] - nominal_x,
                point[1] - nominal_y,
            )
            location_distances.append(distance)
            overall_distances.append(distance)
            distances_by_path[trial["_source_path"]] = distance

        per_location[location_name] = {
            "available": True,
            "coordinate_count": len(valid_points),
            "nominal_center_px": [
                rounded(nominal_x),
                rounded(nominal_y),
            ],
            "mean_distance_px": rounded(
                statistics.mean(location_distances)
            ),
            "max_distance_px": rounded(max(location_distances)),
        }

    available_locations = sum(
        result["available"] for result in per_location.values()
    )
    summary = {
        "available": bool(overall_distances),
        "coordinate_field": coordinate_field,
        "nominal_method": (
            "Per-location arithmetic mean of the four official trial "
            "coordinates; Euclidean pixel distance from that empirical "
            "nominal center."
        ),
        "available_locations": available_locations,
        "overall": {
            "count": len(overall_distances),
            "mean_distance_px": (
                rounded(statistics.mean(overall_distances))
                if overall_distances
                else None
            ),
            "max_distance_px": (
                rounded(max(overall_distances))
                if overall_distances
                else None
            ),
        },
        "per_location": per_location,
    }
    return summary, distances_by_path


def summarize_counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def optional_text(value: Any) -> str:
    if value is None:
        return "unavailable"

    return str(value)


def bool_count(trials: list[dict[str, Any]], field: str) -> dict[str, Any]:
    successful = sum(trial.get(field) is True for trial in trials)
    return {
        "count": successful,
        "percentage": rounded(percentage(successful, len(trials)), 2),
        "available_records": sum(field in trial for trial in trials),
    }


def build_analysis(
    trials: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = len(trials)
    full_success_count = sum(full_cycle_success(trial) for trial in trials)
    bbox_repeatability, bbox_distances = repeatability_summary(
        trials,
        "bbox_center_px",
    )
    grasp_repeatability, grasp_distances = repeatability_summary(
        trials,
        "grasp_point_px",
    )
    per_location: dict[str, Any] = {}

    for location_number, location_name, folder_name in LOCATION_SPECS:
        location_trials = [
            trial
            for trial in trials
            if trial["benchmark_location_number"] == location_number
        ]
        location_full_success = sum(
            full_cycle_success(trial) for trial in location_trials
        )
        per_location[location_name] = {
            "location_number": location_number,
            "input_folder": str((BENCHMARK_ROOT / folder_name).resolve()),
            "trials": len(location_trials),
            "successful_full_cycles": location_full_success,
            "success_percentage": rounded(
                percentage(location_full_success, len(location_trials)),
                2,
            ),
            "cycle_time_seconds": descriptive_statistics(
                completed_cycle_times(location_trials)
            ),
            "bbox_center_repeatability": bbox_repeatability[
                "per_location"
            ][location_name],
            "grasp_point_repeatability": grasp_repeatability[
                "per_location"
            ][location_name],
        }

    triangle_values = [
        " / ".join(str(item) for item in trial["triangle_ids"])
        for trial in trials
        if isinstance(trial.get("triangle_ids"), list)
    ]
    correction_values = [
        str(trial["correction_mode"])
        for trial in trials
        if "correction_mode" in trial
    ]
    retention_values = [
        str(trial["retention_policy"])
        for trial in trials
        if "retention_policy" in trial
    ]
    hold_margins = [
        float(trial["dynamic_hold_margin"])
        for trial in trials
        if finite_number(trial.get("dynamic_hold_margin"))
    ]
    hold_cap_values = [
        (
            "null"
            if trial.get("dynamic_hold_cap") is None
            else str(trial.get("dynamic_hold_cap"))
        )
        for trial in trials
        if "dynamic_hold_cap" in trial
    ]
    unavailable_fields = [
        field
        for field, count in integrity["field_availability"].items()
        if count == 0
    ]
    partially_unavailable_fields = {
        field: count
        for field, count in integrity["field_availability"].items()
        if 0 < count < total
    }

    rows = []

    for trial in trials:
        bbox = valid_point(trial.get("bbox_center_px"))
        grasp = valid_point(trial.get("grasp_point_px"))
        bbox_location = bbox_repeatability["per_location"][
            trial["benchmark_location_name"]
        ]
        grasp_location = grasp_repeatability["per_location"][
            trial["benchmark_location_name"]
        ]
        rows.append(
            {
                "benchmark_method": trial.get("benchmark_method"),
                "benchmark_location_number": trial.get(
                    "benchmark_location_number"
                ),
                "benchmark_location_name": trial.get(
                    "benchmark_location_name"
                ),
                "benchmark_trial_number": trial.get(
                    "benchmark_trial_number"
                ),
                "timestamp_start": trial.get("timestamp_start"),
                "timestamp_end": trial.get("timestamp_end"),
                "bbox_center_x_px": bbox[0] if bbox else None,
                "bbox_center_y_px": bbox[1] if bbox else None,
                "bbox_nominal_x_px": (
                    bbox_location["nominal_center_px"][0]
                    if bbox_location["available"]
                    else None
                ),
                "bbox_nominal_y_px": (
                    bbox_location["nominal_center_px"][1]
                    if bbox_location["available"]
                    else None
                ),
                "bbox_distance_from_nominal_px": rounded(
                    bbox_distances.get(trial["_source_path"])
                ),
                "grasp_point_x_px": grasp[0] if grasp else None,
                "grasp_point_y_px": grasp[1] if grasp else None,
                "grasp_nominal_x_px": (
                    grasp_location["nominal_center_px"][0]
                    if grasp_location["available"]
                    else None
                ),
                "grasp_nominal_y_px": (
                    grasp_location["nominal_center_px"][1]
                    if grasp_location["available"]
                    else None
                ),
                "grasp_distance_from_nominal_px": rounded(
                    grasp_distances.get(trial["_source_path"])
                ),
                "mapping_triangle": " / ".join(
                    str(item) for item in trial.get("triangle_ids", [])
                ),
                "correction_mode": trial.get("correction_mode"),
                "dynamic_hold_margin": trial.get("dynamic_hold_margin"),
                "dynamic_hold_cap": trial.get("dynamic_hold_cap"),
                "retention_policy": trial.get("retention_policy"),
                "grip_contact_accepted": trial.get(
                    "grip_contact_accepted"
                ),
                "grip_requested_command": trial.get(
                    "grip_requested_command"
                ),
                "grip_actual_position": trial.get("grip_actual_position"),
                "grip_hold_command": trial.get("grip_hold_command"),
                "release_open_verified": trial.get(
                    "release_open_verified"
                ),
                "release_gripper_before": trial.get(
                    "release_gripper_before"
                ),
                "release_gripper_after": trial.get(
                    "release_gripper_after"
                ),
                "cycle_completed": trial.get("cycle_completed"),
                "observed_success": trial.get("observed_success"),
                "full_cycle_success": full_cycle_success(trial),
                "cycle_time_seconds": trial.get("cycle_time_seconds"),
                "failure_reason": trial.get("failure_reason"),
                "source_json": trial["_source_path"],
            }
        )

    summary = {
        "benchmark": {
            "name": "Final YOLO Benchmark",
            "method": "YOLO",
            "design": "5 locations x 4 trials = 20 controlled trials",
            "benchmark_root": str(BENCHMARK_ROOT.resolve()),
            "analyzed_input_folders": [
                str((BENCHMARK_ROOT / folder_name).resolve())
                for _, _, folder_name in LOCATION_SPECS
            ],
            "files_analyzed": total,
            "development_and_debug_trials_excluded": True,
            "full_cycle_success_definition": (
                "grip_contact_accepted, release_open_verified, "
                "cycle_completed, and observed_success must all be true"
            ),
            "standard_deviation_convention": (
                "Sample standard deviation (statistics.stdev, n-1), "
                "matching the existing final ArUco analyzer."
            ),
        },
        "integrity": integrity,
        "total_counts": {
            "total_trials": total,
            "grip_contact_accepted": bool_count(
                trials, "grip_contact_accepted"
            ),
            "release_verified": bool_count(
                trials, "release_open_verified"
            ),
            "cycle_completed": bool_count(trials, "cycle_completed"),
            "observed_placement_success": bool_count(
                trials, "observed_success"
            ),
            "full_cycle_success": {
                "count": full_success_count,
                "percentage": rounded(
                    percentage(full_success_count, total), 2
                ),
            },
        },
        "per_location": per_location,
        "overall_cycle_time_seconds": descriptive_statistics(
            completed_cycle_times(trials)
        ),
        "perception_repeatability": {
            "bbox_center_px": bbox_repeatability,
            "grasp_point_px": grasp_repeatability,
        },
        "supplementary": {
            "mapping_triangle_usage": summarize_counter(triangle_values),
            "correction_mode_usage": summarize_counter(correction_values),
            "retention_policy_usage": summarize_counter(retention_values),
            "dynamic_hold_margin": descriptive_statistics(hold_margins),
            "dynamic_hold_cap_usage": summarize_counter(hold_cap_values),
        },
        "field_availability": {
            "available_counts": integrity["field_availability"],
            "unavailable_fields": unavailable_fields,
            "partially_unavailable_fields": partially_unavailable_fields,
        },
        "failures": [
            {
                "benchmark_location_name": trial.get(
                    "benchmark_location_name"
                ),
                "benchmark_trial_number": trial.get(
                    "benchmark_trial_number"
                ),
                "failure_reason": trial.get("failure_reason"),
                "source_json": trial["_source_path"],
            }
            for trial in trials
            if not full_cycle_success(trial)
        ],
    }
    return summary, rows


def format_time_statistics(values: dict[str, Any]) -> str:
    return (
        f"n={values['count']}, mean={optional_text(values['mean'])} s, "
        f"median={optional_text(values['median'])} s, "
        f"sample SD={optional_text(values['sample_stdev'])} s, "
        f"min={optional_text(values['min'])} s, "
        f"max={optional_text(values['max'])} s"
    )


def format_scalar_statistics(values: dict[str, Any]) -> str:
    return (
        f"n={values['count']}, mean={optional_text(values['mean'])}, "
        f"median={optional_text(values['median'])}, "
        f"sample SD={optional_text(values['sample_stdev'])}, "
        f"min={optional_text(values['min'])}, "
        f"max={optional_text(values['max'])}"
    )


def build_text_summary(summary: dict[str, Any]) -> str:
    counts = summary["total_counts"]
    integrity = summary["integrity"]
    repeatability = summary["perception_repeatability"]
    lines = [
        "FINAL YOLO BENCHMARK - OFFICIAL SUMMARY",
        "=" * 60,
        "",
        "BENCHMARK SCOPE",
        "-" * 60,
        "Design: 5 locations x 4 trials = 20 controlled trials",
        f"Official JSON files analyzed: {summary['benchmark']['files_analyzed']}",
        "Development/debug trials, dry runs, perception validations, "
        "dataset results, and all non-official JSON files were excluded.",
        "Analyzed input folders:",
    ]

    for folder in summary["benchmark"]["analyzed_input_folders"]:
        lines.append(f"- {folder}")

    lines.extend(
        [
            "",
            "INTEGRITY",
            "-" * 60,
            "Integrity result: PASS",
            f"Records parsed: {integrity['files_parsed']}/20",
            "Files per location: "
            + ", ".join(
                f"{name}={count}"
                for name, count in integrity["files_per_location"].items()
            ),
            "Duplicate underlying trial payloads: 0",
            "Duplicate trial start timestamps: 0",
            "Malformed records or benchmark-metadata inconsistencies: 0",
            "",
            "SUCCESS DEFINITION AND STATISTICAL CONVENTION",
            "-" * 60,
            "Full-cycle success requires grip_contact_accepted, "
            "release_open_verified, cycle_completed, and observed_success "
            "to all be true.",
            "Cycle-time standard deviations are sample standard deviations "
            "(n-1), matching the existing final ArUco analyzer.",
            "",
            "TOTAL COUNTS",
            "-" * 60,
            (
                "Grip/contact accepted: "
                f"{counts['grip_contact_accepted']['count']}/"
                f"{counts['total_trials']} "
                f"({counts['grip_contact_accepted']['percentage']:.2f}%)"
            ),
            (
                "Release verified: "
                f"{counts['release_verified']['count']}/"
                f"{counts['total_trials']} "
                f"({counts['release_verified']['percentage']:.2f}%)"
            ),
            (
                "Cycle completed: "
                f"{counts['cycle_completed']['count']}/"
                f"{counts['total_trials']} "
                f"({counts['cycle_completed']['percentage']:.2f}%)"
            ),
            (
                "Observed placement success: "
                f"{counts['observed_placement_success']['count']}/"
                f"{counts['total_trials']} "
                f"({counts['observed_placement_success']['percentage']:.2f}%)"
            ),
            (
                "Full-cycle success: "
                f"{counts['full_cycle_success']['count']}/"
                f"{counts['total_trials']} "
                f"({counts['full_cycle_success']['percentage']:.2f}%)"
            ),
            "",
        ]
    )

    if (
        counts["total_trials"] == 20
        and counts["full_cycle_success"]["count"] == 20
    ):
        lines.append(
            "The YOLO-based system achieved a 100% observed full-cycle "
            "success rate across 20 controlled benchmark trials distributed "
            "over five workspace locations."
        )
    else:
        lines.append(
            "Within this controlled benchmark, the observed full-cycle "
            f"success rate was {counts['full_cycle_success']['percentage']:.2f}% "
            f"({counts['full_cycle_success']['count']}/"
            f"{counts['total_trials']})."
        )

    lines.extend(["", "PER-LOCATION RESULTS", "-" * 60])

    for _, location_name, _ in LOCATION_SPECS:
        result = summary["per_location"][location_name]
        lines.append(
            f"{location_name}: "
            f"{result['successful_full_cycles']}/{result['trials']} "
            f"({result['success_percentage']:.2f}%); "
            + format_time_statistics(result["cycle_time_seconds"])
        )

    lines.extend(
        [
            "",
            "OVERALL CYCLE TIME",
            "-" * 60,
            format_time_statistics(summary["overall_cycle_time_seconds"]),
            "",
            "PERCEPTION / TARGET REPEATABILITY",
            "-" * 60,
            (
                "Method: for each location, the empirical nominal point is "
                "the arithmetic mean of the four official trial coordinates; "
                "distances are Euclidean pixel distances from that point."
            ),
        ]
    )

    for label, key in (
        ("YOLO bbox center", "bbox_center_px"),
        ("Final grasp point", "grasp_point_px"),
    ):
        result = repeatability[key]
        lines.append("")
        lines.append(label + ":")

        if not result["available"]:
            lines.append("- unavailable")
            continue

        lines.append(
            "- Overall mean distance from nominal: "
            f"{result['overall']['mean_distance_px']} px"
        )
        lines.append(
            "- Overall maximum distance from nominal: "
            f"{result['overall']['max_distance_px']} px"
        )

        for _, location_name, _ in LOCATION_SPECS:
            location_result = result["per_location"][location_name]
            lines.append(
                f"- {location_name}: nominal="
                f"{location_result['nominal_center_px']} px, "
                f"mean distance={location_result['mean_distance_px']} px, "
                f"max distance={location_result['max_distance_px']} px"
            )

    supplementary = summary["supplementary"]
    lines.extend(
        [
            "",
            "SUPPLEMENTARY YOLO-SPECIFIC STATISTICS",
            "-" * 60,
            "Mapping triangle usage:",
        ]
    )

    for name, count in supplementary["mapping_triangle_usage"].items():
        lines.append(f"- {name}: {count}")

    lines.append("Correction mode usage:")

    for name, count in supplementary["correction_mode_usage"].items():
        lines.append(f"- {name}: {count}")

    lines.append("Retention policy usage:")

    for name, count in supplementary["retention_policy_usage"].items():
        lines.append(f"- {name}: {count}")

    lines.append(
        "Dynamic hold margin: "
        + format_scalar_statistics(supplementary["dynamic_hold_margin"])
    )
    lines.append(
        "Dynamic hold-cap value usage: "
        + ", ".join(
            f"{name}={count}"
            for name, count in supplementary[
                "dynamic_hold_cap_usage"
            ].items()
        )
    )
    unavailable = summary["field_availability"]["unavailable_fields"]
    partial = summary["field_availability"][
        "partially_unavailable_fields"
    ]
    lines.extend(["", "UNAVAILABLE FIELDS", "-" * 60])
    lines.append(
        "Completely unavailable fields: "
        + (", ".join(unavailable) if unavailable else "None")
    )
    lines.append(
        "Partially unavailable fields: "
        + (
            ", ".join(f"{name}={count}/20" for name, count in partial.items())
            if partial
            else "None"
        )
    )
    lines.extend(
        [
            "",
            "OUTPUT FILES",
            "-" * 60,
            f"TXT: {TXT_OUTPUT.resolve()}",
            f"JSON: {JSON_OUTPUT.resolve()}",
            f"CSV: {CSV_OUTPUT.resolve()}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    TXT_OUTPUT.write_text(build_text_summary(summary), encoding="utf-8")


def main() -> None:
    trials, integrity = load_and_validate_trials()
    summary, rows = build_analysis(trials, integrity)
    write_outputs(summary, rows)
    print(build_text_summary(summary), end="")
    print(f"Saved summary TXT: {TXT_OUTPUT.resolve()}")
    print(f"Saved summary JSON: {JSON_OUTPUT.resolve()}")
    print(f"Saved detailed CSV: {CSV_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
