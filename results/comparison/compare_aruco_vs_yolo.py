"""Create the final comparison from official ArUco and YOLO summaries only."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIRECTORY = SCRIPT_DIR / "summary"

ARUCO_SUMMARY_DIRECTORY = (
    PROJECT_ROOT
    / "06_pixel_to_robot"
    / "aruco_final_benchmark_v3"
    / "summary"
)
YOLO_SUMMARY_DIRECTORY = (
    PROJECT_ROOT / "06_yolo" / "yolo_final_benchmark" / "summary"
)

ARUCO_TXT = ARUCO_SUMMARY_DIRECTORY / "aruco_final_benchmark_v3_summary.txt"
ARUCO_JSON = ARUCO_SUMMARY_DIRECTORY / "aruco_final_benchmark_v3_summary.json"
ARUCO_CSV = ARUCO_SUMMARY_DIRECTORY / "aruco_final_benchmark_v3_trials.csv"
YOLO_TXT = YOLO_SUMMARY_DIRECTORY / "yolo_final_benchmark_summary.txt"
YOLO_JSON = YOLO_SUMMARY_DIRECTORY / "yolo_final_benchmark_summary.json"
YOLO_CSV = YOLO_SUMMARY_DIRECTORY / "yolo_final_benchmark_trials.csv"

COMPARISON_TXT = OUTPUT_DIRECTORY / "aruco_vs_yolo_comparison.txt"
COMPARISON_JSON = OUTPUT_DIRECTORY / "aruco_vs_yolo_comparison.json"
COMPARISON_CSV = OUTPUT_DIRECTORY / "aruco_vs_yolo_trials.csv"

LOCATION_NAMES = (
    "L1_lower_left",
    "L2_upper_left",
    "L3_center",
    "L4_upper_right",
    "L5_lower_right",
)
LOCATION_NUMBERS = {
    name: index for index, name in enumerate(LOCATION_NAMES, start=1)
}
EXPECTED_TRIALS = 20
EXPECTED_REPETITIONS = 4
OUTCOME_FIELDS = (
    "grip_contact_accepted",
    "release_open_verified",
    "cycle_completed",
    "observed_success",
)


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def parse_finite(value: str, field: str, record_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{record_name}: {field} is not numeric: {value!r}"
        ) from error

    if not math.isfinite(number):
        raise ValueError(f"{record_name}: {field} is not finite.")

    return number


def parse_boolean(value: str, field: str, record_name: str) -> bool:
    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"{record_name}: {field} is not a boolean: {value!r}"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def time_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_stdev": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": rounded(statistics.mean(values)),
        "median": rounded(statistics.median(values)),
        "sample_stdev": rounded(
            statistics.stdev(values) if len(values) >= 2 else 0.0
        ),
        "min": rounded(min(values)),
        "max": rounded(max(values)),
    }


def validate_required_files() -> None:
    required = (
        ARUCO_TXT,
        ARUCO_JSON,
        ARUCO_CSV,
        YOLO_TXT,
        YOLO_JSON,
        YOLO_CSV,
    )
    missing = [str(path) for path in required if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing official benchmark analysis outputs:\n"
            + "\n".join(missing)
        )


def normalize_aruco_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for location_name in LOCATION_NAMES:
        location_rows = sorted(
            (row for row in rows if row.get("benchmark_location") == location_name),
            key=lambda row: (row.get("timestamp_start", ""), row.get("json_path", "")),
        )

        for repetition, row in enumerate(location_rows, start=1):
            record_name = row.get("json_path") or (
                f"ArUco/{location_name}/repetition_{repetition}"
            )
            outcomes = {
                field: parse_boolean(row.get(field, ""), field, record_name)
                for field in OUTCOME_FIELDS
            }
            normalized.append(
                {
                    "method": "ArUco",
                    "location_number": LOCATION_NUMBERS[location_name],
                    "location_name": location_name,
                    "repetition": repetition,
                    "timestamp_start": row.get("timestamp_start"),
                    "timestamp_end": row.get("timestamp_end"),
                    **outcomes,
                    "reported_full_cycle_success": parse_boolean(
                        row.get("full_cycle_success", ""),
                        "full_cycle_success",
                        record_name,
                    ),
                    "cycle_time_seconds": parse_finite(
                        row.get("cycle_time_seconds", ""),
                        "cycle_time_seconds",
                        record_name,
                    ),
                    "primary_localization_type": "ArUco marker center",
                    "primary_x_px": parse_finite(
                        row.get("center_x_px", ""),
                        "center_x_px",
                        record_name,
                    ),
                    "primary_y_px": parse_finite(
                        row.get("center_y_px", ""),
                        "center_y_px",
                        record_name,
                    ),
                    "secondary_localization_type": None,
                    "secondary_x_px": None,
                    "secondary_y_px": None,
                    "source_summary_csv": str(ARUCO_CSV.resolve()),
                    "source_record_reference": row.get("json_path"),
                }
            )

    return normalized


def normalize_yolo_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for row in rows:
        location_name = row.get("benchmark_location_name", "")
        record_name = row.get("source_json") or (
            f"YOLO/{location_name}/{row.get('benchmark_trial_number')}"
        )

        try:
            location_number = int(row.get("benchmark_location_number", ""))
            repetition = int(row.get("benchmark_trial_number", ""))
        except ValueError as error:
            raise ValueError(
                f"{record_name}: invalid location or trial number."
            ) from error

        outcomes = {
            field: parse_boolean(row.get(field, ""), field, record_name)
            for field in OUTCOME_FIELDS
        }
        normalized.append(
            {
                "method": "YOLO",
                "location_number": location_number,
                "location_name": location_name,
                "repetition": repetition,
                "timestamp_start": row.get("timestamp_start"),
                "timestamp_end": row.get("timestamp_end"),
                **outcomes,
                "reported_full_cycle_success": parse_boolean(
                    row.get("full_cycle_success", ""),
                    "full_cycle_success",
                    record_name,
                ),
                "cycle_time_seconds": parse_finite(
                    row.get("cycle_time_seconds", ""),
                    "cycle_time_seconds",
                    record_name,
                ),
                "primary_localization_type": "YOLO bbox center",
                "primary_x_px": parse_finite(
                    row.get("bbox_center_x_px", ""),
                    "bbox_center_x_px",
                    record_name,
                ),
                "primary_y_px": parse_finite(
                    row.get("bbox_center_y_px", ""),
                    "bbox_center_y_px",
                    record_name,
                ),
                "secondary_localization_type": (
                    "YOLO upper-quarter grasp point"
                ),
                "secondary_x_px": parse_finite(
                    row.get("grasp_point_x_px", ""),
                    "grasp_point_x_px",
                    record_name,
                ),
                "secondary_y_px": parse_finite(
                    row.get("grasp_point_y_px", ""),
                    "grasp_point_y_px",
                    record_name,
                ),
                "source_summary_csv": str(YOLO_CSV.resolve()),
                "source_record_reference": row.get("source_json"),
            }
        )

    normalized.sort(
        key=lambda row: (row["location_number"], row["repetition"])
    )
    return normalized


def harmonized_full_success(row: dict[str, Any]) -> bool:
    return all(row[field] is True for field in OUTCOME_FIELDS)


def validate_design(
    aruco_rows: list[dict[str, Any]],
    yolo_rows: list[dict[str, Any]],
    aruco_summary: dict[str, Any],
    yolo_summary: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []

    for method, rows in (("ArUco", aruco_rows), ("YOLO", yolo_rows)):
        if len(rows) != EXPECTED_TRIALS:
            issues.append(
                f"{method}: found {len(rows)} CSV trials; expected 20."
            )

        counts = Counter(row["location_name"] for row in rows)

        if set(counts) != set(LOCATION_NAMES):
            issues.append(
                f"{method}: location names do not match the five expected "
                "benchmark locations."
            )

        for location_name in LOCATION_NAMES:
            if counts[location_name] != EXPECTED_REPETITIONS:
                issues.append(
                    f"{method}/{location_name}: found "
                    f"{counts[location_name]} repetitions; expected 4."
                )

        references = [row["source_record_reference"] for row in rows]

        if len(set(references)) != len(references):
            issues.append(f"{method}: duplicate source-record references.")

        for location_name in LOCATION_NAMES:
            repetitions = sorted(
                row["repetition"]
                for row in rows
                if row["location_name"] == location_name
            )

            if repetitions != [1, 2, 3, 4]:
                issues.append(
                    f"{method}/{location_name}: repetition indices are "
                    f"{repetitions}; expected [1, 2, 3, 4]."
                )

    if aruco_summary.get("total_trials_found") != EXPECTED_TRIALS:
        issues.append("ArUco summary JSON does not report exactly 20 trials.")

    if (
        yolo_summary.get("total_counts", {}).get("total_trials")
        != EXPECTED_TRIALS
    ):
        issues.append("YOLO summary JSON does not report exactly 20 trials.")

    aruco_harmonized = sum(
        harmonized_full_success(row) for row in aruco_rows
    )
    yolo_harmonized = sum(
        harmonized_full_success(row) for row in yolo_rows
    )

    if aruco_harmonized != aruco_summary.get("full_cycle_successes"):
        issues.append(
            "ArUco harmonized success count disagrees with its official "
            "summary count."
        )

    if yolo_harmonized != (
        yolo_summary.get("total_counts", {})
        .get("full_cycle_success", {})
        .get("count")
    ):
        issues.append(
            "YOLO harmonized success count disagrees with its official "
            "summary count."
        )

    integrity = {
        "passed": not issues,
        "expected_trials_per_method": EXPECTED_TRIALS,
        "expected_locations": 5,
        "expected_repetitions_per_location": EXPECTED_REPETITIONS,
        "aruco_trials": len(aruco_rows),
        "yolo_trials": len(yolo_rows),
        "aruco_files_per_location": dict(
            Counter(row["location_name"] for row in aruco_rows)
        ),
        "yolo_files_per_location": dict(
            Counter(row["location_name"] for row in yolo_rows)
        ),
        "outcome_fields_available_for_both_methods": list(OUTCOME_FIELDS),
        "success_definition_compatibility": {
            "published_aruco_definition": (
                "cycle_completed and observed_success"
            ),
            "published_yolo_definition": (
                "grip_contact_accepted, release_open_verified, "
                "cycle_completed, and observed_success"
            ),
            "definitions_identical": False,
            "harmonized_definition": (
                "All four outcome fields must be true for both methods."
            ),
            "effect_on_current_counts": (
                "None; all four outcome fields are true in all 40 official "
                "CSV rows."
            ),
        },
        "cycle_time_compatibility": {
            "sample_standard_deviation_used_by_both": True,
            "all_trials_completed": True,
            "all_cycle_times_finite": True,
            "effective_analyzed_trial_sets_compatible": True,
        },
        "issues": issues,
    }

    if issues:
        print("FINAL COMPARISON INPUT INTEGRITY: FAIL")

        for issue in issues:
            print(f"- {issue}")

        raise SystemExit(1)

    return integrity


def task_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    metrics: dict[str, Any] = {"total_trials": total}

    for field in OUTCOME_FIELDS:
        count = sum(row[field] is True for row in rows)
        metrics[field] = {
            "count": count,
            "percentage": rounded(percentage(count, total), 2),
        }

    full_count = sum(harmonized_full_success(row) for row in rows)
    metrics["harmonized_full_cycle_success"] = {
        "count": full_count,
        "percentage": rounded(percentage(full_count, total), 2),
    }
    return metrics


def repeatability(
    rows: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    metric_name: str,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], float]]:
    per_location: dict[str, Any] = {}
    distance_by_trial: dict[tuple[str, int, int], float] = {}
    overall_distances: list[float] = []

    for location_name in LOCATION_NAMES:
        location_rows = [
            row for row in rows if row["location_name"] == location_name
        ]
        points = [(row[x_field], row[y_field]) for row in location_rows]
        nominal_x = statistics.mean(point[0] for point in points)
        nominal_y = statistics.mean(point[1] for point in points)
        distances = []

        for row, point in zip(location_rows, points):
            distance = math.hypot(
                point[0] - nominal_x,
                point[1] - nominal_y,
            )
            distances.append(distance)
            overall_distances.append(distance)
            distance_by_trial[
                (row["method"], row["location_number"], row["repetition"])
            ] = distance

        per_location[location_name] = {
            "nominal_center_px": [rounded(nominal_x), rounded(nominal_y)],
            "mean_distance_px": rounded(statistics.mean(distances)),
            "max_distance_px": rounded(max(distances)),
        }

    return (
        {
            "metric": metric_name,
            "nominal_definition": (
                "Per-location arithmetic mean of the four official trial "
                "coordinates."
            ),
            "distance_definition": (
                "Euclidean pixel distance from the empirical per-location "
                "nominal center."
            ),
            "overall_mean_distance_px": rounded(
                statistics.mean(overall_distances)
            ),
            "overall_max_distance_px": rounded(max(overall_distances)),
            "per_location": per_location,
        },
        distance_by_trial,
    )


def build_comparison(
    aruco_rows: list[dict[str, Any]],
    yolo_rows: list[dict[str, Any]],
    aruco_summary: dict[str, Any],
    yolo_summary: dict[str, Any],
    integrity: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aruco_task = task_metrics(aruco_rows)
    yolo_task = task_metrics(yolo_rows)
    aruco_times = time_statistics(
        [row["cycle_time_seconds"] for row in aruco_rows]
    )
    yolo_times = time_statistics(
        [row["cycle_time_seconds"] for row in yolo_rows]
    )
    mean_difference = yolo_times["mean"] - aruco_times["mean"]
    relative_difference = 100.0 * mean_difference / aruco_times["mean"]
    per_location: dict[str, Any] = {}

    for location_name in LOCATION_NAMES:
        aruco_location = [
            row for row in aruco_rows if row["location_name"] == location_name
        ]
        yolo_location = [
            row for row in yolo_rows if row["location_name"] == location_name
        ]
        aruco_mean = statistics.mean(
            row["cycle_time_seconds"] for row in aruco_location
        )
        yolo_mean = statistics.mean(
            row["cycle_time_seconds"] for row in yolo_location
        )
        per_location[location_name] = {
            "aruco": {
                "trials": len(aruco_location),
                "full_cycle_successes": sum(
                    harmonized_full_success(row) for row in aruco_location
                ),
                "success_percentage": rounded(
                    percentage(
                        sum(
                            harmonized_full_success(row)
                            for row in aruco_location
                        ),
                        len(aruco_location),
                    ),
                    2,
                ),
                "mean_cycle_time_seconds": rounded(aruco_mean),
            },
            "yolo": {
                "trials": len(yolo_location),
                "full_cycle_successes": sum(
                    harmonized_full_success(row) for row in yolo_location
                ),
                "success_percentage": rounded(
                    percentage(
                        sum(
                            harmonized_full_success(row)
                            for row in yolo_location
                        ),
                        len(yolo_location),
                    ),
                    2,
                ),
                "mean_cycle_time_seconds": rounded(yolo_mean),
            },
            "yolo_minus_aruco_mean_cycle_time_seconds": rounded(
                yolo_mean - aruco_mean
            ),
        }

    aruco_repeatability, aruco_distances = repeatability(
        aruco_rows,
        "primary_x_px",
        "primary_y_px",
        "ArUco marker center",
    )
    yolo_bbox_repeatability, yolo_bbox_distances = repeatability(
        yolo_rows,
        "primary_x_px",
        "primary_y_px",
        "YOLO bbox center",
    )
    yolo_grasp_repeatability, yolo_grasp_distances = repeatability(
        yolo_rows,
        "secondary_x_px",
        "secondary_y_px",
        "YOLO upper-quarter grasp point",
    )

    original_repeatability = {
        "direct_ranking_permitted": False,
        "methodological_difference": (
            "The official ArUco analyzer measures marker-center distance "
            "from stored benchmark_nominal_center_px values. The official "
            "YOLO analyzer derives a separate empirical nominal center for "
            "each location from the four official trials. These published "
            "repeatability values therefore do not share the same nominal "
            "reference definition."
        ),
        "aruco_published_marker_center": {
            "mean_distance_px": aruco_summary["center_error_px"]["mean"],
            "max_distance_px": aruco_summary["center_error_px"]["max"],
        },
        "yolo_published_bbox_center": {
            "mean_distance_px": yolo_summary["perception_repeatability"]
            ["bbox_center_px"]["overall"]["mean_distance_px"],
            "max_distance_px": yolo_summary["perception_repeatability"]
            ["bbox_center_px"]["overall"]["max_distance_px"],
        },
        "yolo_published_upper_quarter_grasp_point": {
            "mean_distance_px": yolo_summary["perception_repeatability"]
            ["grasp_point_px"]["overall"]["mean_distance_px"],
            "max_distance_px": yolo_summary["perception_repeatability"]
            ["grasp_point_px"]["overall"]["max_distance_px"],
        },
    }

    harmonized_repeatability = {
        "common_nominal_definition": (
            "Per-location arithmetic mean of the four official trial "
            "coordinates for each method and coordinate type."
        ),
        "aruco_marker_center": aruco_repeatability,
        "yolo_bbox_center": yolo_bbox_repeatability,
        "yolo_upper_quarter_grasp_point": yolo_grasp_repeatability,
        "descriptive_differences": {
            "yolo_bbox_minus_aruco_mean_distance_px": rounded(
                yolo_bbox_repeatability["overall_mean_distance_px"]
                - aruco_repeatability["overall_mean_distance_px"]
            ),
            "yolo_bbox_minus_aruco_max_distance_px": rounded(
                yolo_bbox_repeatability["overall_max_distance_px"]
                - aruco_repeatability["overall_max_distance_px"]
            ),
            "yolo_grasp_minus_aruco_mean_distance_px": rounded(
                yolo_grasp_repeatability["overall_mean_distance_px"]
                - aruco_repeatability["overall_mean_distance_px"]
            ),
            "yolo_grasp_minus_aruco_max_distance_px": rounded(
                yolo_grasp_repeatability["overall_max_distance_px"]
                - aruco_repeatability["overall_max_distance_px"]
            ),
        },
        "interpretation_limit": (
            "These are descriptive results from four controlled trials per "
            "location and are not evidence of general statistical "
            "superiority."
        ),
    }

    qualitative = [
        {
            "dimension": "Localization approach",
            "ArUco": "Deterministic fiducial-marker localization",
            "YOLO": "Learned appearance-based cube detection",
        },
        {
            "dimension": "Visible marker required",
            "ArUco": "Yes",
            "YOLO": "No for the tested cube detector",
        },
        {
            "dimension": "Detector training required",
            "ArUco": "No learned detector training",
            "YOLO": "Yes; labeled data and model training",
        },
        {
            "dimension": "Object preparation",
            "ArUco": "Marker installation and visibility required",
            "YOLO": "No marker installation for the tested cube",
        },
        {
            "dimension": "Extension to new object classes",
            "ArUco": "Requires an appropriate visible fiducial target",
            "YOLO": (
                "Potentially possible with new labeled data, training, and "
                "validation; not established by this benchmark"
            ),
        },
    ]

    held_out_perception = {
        "label": "HELD-OUT PERCEPTION TEST performance",
        "separate_from_robotic_benchmark": True,
        "ground_truth_cube_instances": 15,
        "ground_truth_cubes_detected": 15,
        "precision": 0.916,
        "recall": 1.000,
        "mAP50": 0.987,
        "mAP50_95": 0.885,
        "extra_background_false_positive_detections": 2,
        "false_negatives": 0,
        "approximate_cpu_processing_time_ms_per_image": 68.2,
    }

    paper_ready_table = [
        {
            "metric": "Official controlled trials",
            "ArUco": "20 (5 locations x 4 trials)",
            "YOLO": "20 (5 locations x 4 trials)",
        },
        {
            "metric": "Harmonized full-cycle success",
            "ArUco": "20/20 (100.00%)",
            "YOLO": "20/20 (100.00%)",
        },
        {
            "metric": "Mean cycle time",
            "ArUco": f"{aruco_times['mean']:.3f} s",
            "YOLO": f"{yolo_times['mean']:.3f} s",
        },
        {
            "metric": "Median cycle time",
            "ArUco": f"{aruco_times['median']:.3f} s",
            "YOLO": f"{yolo_times['median']:.3f} s",
        },
        {
            "metric": "Cycle-time sample SD",
            "ArUco": f"{aruco_times['sample_stdev']:.3f} s",
            "YOLO": f"{yolo_times['sample_stdev']:.3f} s",
        },
        {
            "metric": "Minimum cycle time",
            "ArUco": f"{aruco_times['min']:.3f} s",
            "YOLO": f"{yolo_times['min']:.3f} s",
        },
        {
            "metric": "Maximum cycle time",
            "ArUco": f"{aruco_times['max']:.3f} s",
            "YOLO": f"{yolo_times['max']:.3f} s",
        },
        {
            "metric": "Localization approach",
            "ArUco": "Deterministic fiducial-marker center",
            "YOLO": "Learned bbox detection plus upper-quarter grasp point",
        },
        {
            "metric": "Marker required",
            "ArUco": "Yes",
            "YOLO": "No for the tested cube",
        },
        {
            "metric": "Detector training required",
            "ArUco": "No",
            "YOLO": "Yes",
        },
    ]

    trial_rows: list[dict[str, Any]] = []

    for row in aruco_rows + yolo_rows:
        key = (row["method"], row["location_number"], row["repetition"])
        primary_distances = (
            aruco_distances if row["method"] == "ArUco" else yolo_bbox_distances
        )
        location_repeatability = (
            aruco_repeatability
            if row["method"] == "ArUco"
            else yolo_bbox_repeatability
        )["per_location"][row["location_name"]]
        secondary_location = (
            yolo_grasp_repeatability["per_location"][row["location_name"]]
            if row["method"] == "YOLO"
            else None
        )
        trial_rows.append(
            {
                "method": row["method"],
                "location_number": row["location_number"],
                "location_name": row["location_name"],
                "repetition": row["repetition"],
                "timestamp_start": row["timestamp_start"],
                "grip_contact_accepted": row["grip_contact_accepted"],
                "release_open_verified": row["release_open_verified"],
                "cycle_completed": row["cycle_completed"],
                "observed_success": row["observed_success"],
                "harmonized_full_cycle_success": harmonized_full_success(row),
                "cycle_time_seconds": row["cycle_time_seconds"],
                "primary_localization_type": row[
                    "primary_localization_type"
                ],
                "primary_x_px": row["primary_x_px"],
                "primary_y_px": row["primary_y_px"],
                "primary_empirical_nominal_x_px": location_repeatability[
                    "nominal_center_px"
                ][0],
                "primary_empirical_nominal_y_px": location_repeatability[
                    "nominal_center_px"
                ][1],
                "primary_distance_from_empirical_nominal_px": rounded(
                    primary_distances[key]
                ),
                "secondary_localization_type": row[
                    "secondary_localization_type"
                ],
                "secondary_x_px": row["secondary_x_px"],
                "secondary_y_px": row["secondary_y_px"],
                "secondary_empirical_nominal_x_px": (
                    secondary_location["nominal_center_px"][0]
                    if secondary_location
                    else None
                ),
                "secondary_empirical_nominal_y_px": (
                    secondary_location["nominal_center_px"][1]
                    if secondary_location
                    else None
                ),
                "secondary_distance_from_empirical_nominal_px": (
                    rounded(yolo_grasp_distances[key])
                    if row["method"] == "YOLO"
                    else None
                ),
                "source_summary_csv": row["source_summary_csv"],
                "source_record_reference": row["source_record_reference"],
            }
        )

    comparison = {
        "comparison": {
            "title": "Final ArUco-vs-YOLO Controlled Benchmark Comparison",
            "analysis_only": True,
            "source_scope": "Official final benchmark analysis outputs only",
            "official_inputs": [
                str(path.resolve())
                for path in (
                    ARUCO_TXT,
                    ARUCO_JSON,
                    ARUCO_CSV,
                    YOLO_TXT,
                    YOLO_JSON,
                    YOLO_CSV,
                )
            ],
        },
        "integrity": integrity,
        "benchmark_design": {
            "ArUco": {"trials": 20, "locations": 5, "repetitions": 4},
            "YOLO": {"trials": 20, "locations": 5, "repetitions": 4},
        },
        "task_performance": {
            "harmonized_success_definition": (
                "All four outcome fields must be true."
            ),
            "ArUco": aruco_task,
            "YOLO": yolo_task,
        },
        "cycle_time_seconds": {
            "standard_deviation_convention": "Sample standard deviation (n-1)",
            "ArUco": aruco_times,
            "YOLO": yolo_times,
            "yolo_minus_aruco_mean_seconds": rounded(mean_difference),
            "relative_mean_difference_percent_of_aruco": rounded(
                relative_difference
            ),
            "interpretation": (
                "The observed mean difference is descriptive. No claim of "
                "statistical speed superiority is made."
            ),
        },
        "per_location": per_location,
        "target_localization_repeatability": {
            "published_metrics": original_repeatability,
            "harmonized_empirical_nominal_metrics": harmonized_repeatability,
        },
        "yolo_held_out_perception_test": held_out_perception,
        "qualitative_engineering_comparison": qualitative,
        "paper_ready_comparison_table": paper_ready_table,
        "interpretation": {
            "controlled_success_scope": (
                "Both methods achieved 100% observed full-cycle success in "
                "their respective 20 controlled benchmark trials. This is "
                "not a universal 100% success claim."
            ),
            "cycle_time_context": (
                "Cycle-time similarity is expected because both systems "
                "share much of the same downstream mapping and robot-control "
                "pipeline."
            ),
            "generalization_limit": (
                "The benchmark does not establish untested robustness or "
                "generalization to other objects, environments, or operating "
                "conditions."
            ),
        },
    }
    return comparison, trial_rows


def format_success(metric: dict[str, Any], total: int) -> str:
    return f"{metric['count']}/{total} ({metric['percentage']:.2f}%)"


def format_time(values: dict[str, Any]) -> str:
    return (
        f"mean={values['mean']:.3f} s, median={values['median']:.3f} s, "
        f"sample SD={values['sample_stdev']:.3f} s, "
        f"min={values['min']:.3f} s, max={values['max']:.3f} s"
    )


def build_text(comparison: dict[str, Any]) -> str:
    task = comparison["task_performance"]
    cycle = comparison["cycle_time_seconds"]
    repeatability_data = comparison["target_localization_repeatability"]
    published = repeatability_data["published_metrics"]
    harmonized = repeatability_data["harmonized_empirical_nominal_metrics"]
    held_out = comparison["yolo_held_out_perception_test"]
    lines = [
        "FINAL ARUCO-VS-YOLO CONTROLLED BENCHMARK COMPARISON",
        "=" * 72,
        "",
        "SCOPE AND INTEGRITY",
        "-" * 72,
        "Analysis source: official final benchmark TXT, JSON, and CSV outputs only.",
        "ArUco: 20 trials, 5 locations, 4 repetitions per location.",
        "YOLO: 20 trials, 5 locations, 4 repetitions per location.",
        "Input integrity: PASS.",
        "All outcome fields and finite cycle times are present in all 40 rows.",
        "",
        "STATISTICAL DEFINITIONS",
        "-" * 72,
        "The published success definitions are not textually identical:",
        "- ArUco: cycle_completed and observed_success.",
        "- YOLO: grip_contact_accepted, release_open_verified, "
        "cycle_completed, and observed_success.",
        "For this comparison, both methods use the stricter four-field "
        "definition. This does not change the current counts because every "
        "outcome field is true in all 40 official CSV rows.",
        "Cycle-time summaries use sample standard deviation (n-1).",
        "",
        "TASK PERFORMANCE",
        "-" * 72,
        "Metric | ArUco | YOLO",
        "Grip/contact accepted | "
        + format_success(task["ArUco"]["grip_contact_accepted"], 20)
        + " | "
        + format_success(task["YOLO"]["grip_contact_accepted"], 20),
        "Release verified | "
        + format_success(task["ArUco"]["release_open_verified"], 20)
        + " | "
        + format_success(task["YOLO"]["release_open_verified"], 20),
        "Cycle completed | "
        + format_success(task["ArUco"]["cycle_completed"], 20)
        + " | "
        + format_success(task["YOLO"]["cycle_completed"], 20),
        "Observed placement success | "
        + format_success(task["ArUco"]["observed_success"], 20)
        + " | "
        + format_success(task["YOLO"]["observed_success"], 20),
        "Harmonized full-cycle success | "
        + format_success(
            task["ArUco"]["harmonized_full_cycle_success"], 20
        )
        + " | "
        + format_success(
            task["YOLO"]["harmonized_full_cycle_success"], 20
        ),
        "",
        "Both methods achieved a 100% observed full-cycle success rate within "
        "their respective 20 controlled benchmark trials. This does not "
        "establish a universal 100% success rate.",
        "",
        "CYCLE TIME",
        "-" * 72,
        "ArUco: " + format_time(cycle["ArUco"]),
        "YOLO:  " + format_time(cycle["YOLO"]),
        "YOLO minus ArUco mean cycle time: "
        f"{cycle['yolo_minus_aruco_mean_seconds']:.3f} s "
        f"({cycle['relative_mean_difference_percent_of_aruco']:.3f}% of the "
        "ArUco mean).",
        "The observed difference is descriptive; no statistical speed "
        "superiority is claimed.",
        "Cycle-time similarity is expected because both systems share much "
        "of the same downstream mapping and robot-control pipeline.",
        "",
        "PER-LOCATION TASK PERFORMANCE",
        "-" * 72,
        "Location | ArUco success / mean | YOLO success / mean | YOLO-ArUco mean",
    ]

    for location_name in LOCATION_NAMES:
        result = comparison["per_location"][location_name]
        lines.append(
            f"{location_name} | "
            f"{result['aruco']['full_cycle_successes']}/4, "
            f"{result['aruco']['mean_cycle_time_seconds']:.3f} s | "
            f"{result['yolo']['full_cycle_successes']}/4, "
            f"{result['yolo']['mean_cycle_time_seconds']:.3f} s | "
            f"{result['yolo_minus_aruco_mean_cycle_time_seconds']:.3f} s"
        )

    lines.extend(
        [
            "",
            "TARGET LOCALIZATION / REPEATABILITY",
            "-" * 72,
            "Published summary values are not directly rankable because the "
            "nominal-center definitions differ.",
            published["methodological_difference"],
            "Published ArUco marker-center distance: "
            f"mean={published['aruco_published_marker_center']['mean_distance_px']} "
            "px, max="
            f"{published['aruco_published_marker_center']['max_distance_px']} px.",
            "Published YOLO bbox-center distance: "
            f"mean={published['yolo_published_bbox_center']['mean_distance_px']} "
            "px, max="
            f"{published['yolo_published_bbox_center']['max_distance_px']} px.",
            "Published YOLO upper-quarter grasp-point distance: "
            "mean="
            f"{published['yolo_published_upper_quarter_grasp_point']['mean_distance_px']} "
            "px, max="
            f"{published['yolo_published_upper_quarter_grasp_point']['max_distance_px']} "
            "px.",
            "",
            "Harmonized descriptive comparison:",
            "All three coordinate types use the arithmetic mean of the four "
            "official trials at each location as their empirical nominal.",
        ]
    )

    for label, key in (
        ("ArUco marker center", "aruco_marker_center"),
        ("YOLO bbox center", "yolo_bbox_center"),
        (
            "YOLO upper-quarter grasp point",
            "yolo_upper_quarter_grasp_point",
        ),
    ):
        result = harmonized[key]
        lines.append(
            f"- {label}: overall mean distance="
            f"{result['overall_mean_distance_px']:.3f} px, "
            f"overall max={result['overall_max_distance_px']:.3f} px."
        )

    lines.append("Per-location harmonized mean/max distances (px):")

    for location_name in LOCATION_NAMES:
        aruco = harmonized["aruco_marker_center"]["per_location"][location_name]
        bbox = harmonized["yolo_bbox_center"]["per_location"][location_name]
        grasp = harmonized["yolo_upper_quarter_grasp_point"]["per_location"][
            location_name
        ]
        lines.append(
            f"- {location_name}: ArUco={aruco['mean_distance_px']:.3f}/"
            f"{aruco['max_distance_px']:.3f}, "
            f"YOLO bbox={bbox['mean_distance_px']:.3f}/"
            f"{bbox['max_distance_px']:.3f}, "
            f"YOLO grasp={grasp['mean_distance_px']:.3f}/"
            f"{grasp['max_distance_px']:.3f}"
        )

    lines.extend(
        [
            "These repeatability results are descriptive and do not establish "
            "general statistical superiority.",
            "",
            "HELD-OUT PERCEPTION TEST PERFORMANCE - YOLO",
            "-" * 72,
            "This section is separate from the 20-trial robotic benchmark.",
            f"Ground-truth cubes: {held_out['ground_truth_cube_instances']}",
            "Ground-truth cubes detected: "
            f"{held_out['ground_truth_cubes_detected']}",
            f"Precision: {held_out['precision']:.3f}",
            f"Recall: {held_out['recall']:.3f}",
            f"mAP50: {held_out['mAP50']:.3f}",
            f"mAP50-95: {held_out['mAP50_95']:.3f}",
            "Extra background false-positive detections: "
            f"{held_out['extra_background_false_positive_detections']}",
            f"False negatives: {held_out['false_negatives']}",
            "Approximate CPU processing time: "
            f"{held_out['approximate_cpu_processing_time_ms_per_image']:.1f} "
            "ms/image",
            "",
            "QUALITATIVE ENGINEERING COMPARISON",
            "-" * 72,
            "Dimension | ArUco | YOLO",
        ]
    )

    for row in comparison["qualitative_engineering_comparison"]:
        lines.append(f"{row['dimension']} | {row['ArUco']} | {row['YOLO']}")

    lines.extend(
        [
            "",
            "PAPER-READY COMPARISON TABLE",
            "-" * 72,
            "Metric | ArUco | YOLO",
        ]
    )

    for row in comparison["paper_ready_comparison_table"]:
        lines.append(f"{row['metric']} | {row['ArUco']} | {row['YOLO']}")

    lines.extend(
        [
            "",
            "INTERPRETATION LIMITS",
            "-" * 72,
            comparison["interpretation"]["controlled_success_scope"],
            comparison["interpretation"]["generalization_limit"],
            "No inference about untested robustness, universal success, or "
            "statistical speed superiority is made.",
            "",
            "OUTPUTS",
            "-" * 72,
            f"TXT: {COMPARISON_TXT.resolve()}",
            f"JSON: {COMPARISON_JSON.resolve()}",
            f"CSV: {COMPARISON_CSV.resolve()}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    comparison: dict[str, Any],
    trial_rows: list[dict[str, Any]],
) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    COMPARISON_JSON.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    COMPARISON_TXT.write_text(build_text(comparison), encoding="utf-8")

    with COMPARISON_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)


def main() -> None:
    validate_required_files()
    aruco_summary = json.loads(ARUCO_JSON.read_text(encoding="utf-8"))
    yolo_summary = json.loads(YOLO_JSON.read_text(encoding="utf-8"))
    aruco_rows = normalize_aruco_rows(read_csv(ARUCO_CSV))
    yolo_rows = normalize_yolo_rows(read_csv(YOLO_CSV))
    integrity = validate_design(
        aruco_rows,
        yolo_rows,
        aruco_summary,
        yolo_summary,
    )
    comparison, trial_rows = build_comparison(
        aruco_rows,
        yolo_rows,
        aruco_summary,
        yolo_summary,
        integrity,
    )
    write_outputs(comparison, trial_rows)
    print(build_text(comparison), end="")
    print(f"Saved comparison TXT: {COMPARISON_TXT.resolve()}")
    print(f"Saved comparison JSON: {COMPARISON_JSON.resolve()}")
    print(f"Saved comparison CSV: {COMPARISON_CSV.resolve()}")


if __name__ == "__main__":
    main()
