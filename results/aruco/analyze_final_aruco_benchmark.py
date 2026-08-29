
from pathlib import Path
import csv
import json
import math
import statistics

PROJECT_ROOT = Path.cwd()
BENCHMARK_ROOT = PROJECT_ROOT / "06_pixel_to_robot" / "aruco_final_benchmark_v3"
OUTPUT_DIR = BENCHMARK_ROOT / "summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_LOCATIONS = [
    "L1_lower_left",
    "L2_upper_left",
    "L3_center",
    "L4_upper_right",
    "L5_lower_right",
]

def load_trials():
    trials = []
    for path in sorted(BENCHMARK_ROOT.rglob("aruco_trial_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: Could not read {path}: {exc}")
            continue

        mode = str(data.get("mode", ""))
        if not mode.startswith("final_aruco_benchmark_v3_location_"):
            continue

        data["_path"] = str(path)
        trials.append(data)
    return trials

def center_error_px(trial):
    center = trial.get("center_px")
    nominal = trial.get("benchmark_nominal_center_px")
    if (
        isinstance(center, list) and len(center) >= 2
        and isinstance(nominal, list) and len(nominal) >= 2
    ):
        return math.hypot(
            float(center[0]) - float(nominal[0]),
            float(center[1]) - float(nominal[1]),
        )
    return None

def full_success(trial):
    return bool(trial.get("cycle_completed")) and bool(trial.get("observed_success"))

def pct(num, den):
    return 100.0 * num / den if den else 0.0

def safe_mean(values):
    return statistics.mean(values) if values else None

def safe_stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else None

trials = load_trials()

if not trials:
    raise SystemExit(
        f"No Final Benchmark V3 JSON logs found under:\n{BENCHMARK_ROOT}"
    )

rows = []
for t in trials:
    err = center_error_px(t)
    rows.append({
        "timestamp_start": t.get("timestamp_start"),
        "timestamp_end": t.get("timestamp_end"),
        "benchmark_location": t.get("benchmark_location"),
        "center_x_px": (t.get("center_px") or [None, None])[0],
        "center_y_px": (t.get("center_px") or [None, None])[1],
        "nominal_x_px": (t.get("benchmark_nominal_center_px") or [None, None])[0],
        "nominal_y_px": (t.get("benchmark_nominal_center_px") or [None, None])[1],
        "center_error_px": round(err, 3) if err is not None else None,
        "triangle_ids": " / ".join(t.get("triangle_ids") or []),
        "grip_contact_accepted": t.get("grip_contact_accepted"),
        "grip_actual_position": t.get("grip_actual_position"),
        "grip_hold_command": t.get("grip_hold_command"),
        "release_open_verified": t.get("release_open_verified"),
        "cycle_completed": t.get("cycle_completed"),
        "observed_success": t.get("observed_success"),
        "full_cycle_success": full_success(t),
        "cycle_time_seconds": t.get("cycle_time_seconds"),
        "failure_reason": t.get("failure_reason"),
        "json_path": t.get("_path"),
    })

csv_path = OUTPUT_DIR / "aruco_final_benchmark_v3_trials.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

total = len(trials)
grasp_ok = sum(bool(t.get("grip_contact_accepted")) for t in trials)
release_ok = sum(bool(t.get("release_open_verified")) for t in trials)
cycle_ok = sum(bool(t.get("cycle_completed")) for t in trials)
observed_ok = sum(bool(t.get("observed_success")) for t in trials)
full_ok = sum(full_success(t) for t in trials)

cycle_times = [
    float(t["cycle_time_seconds"])
    for t in trials
    if isinstance(t.get("cycle_time_seconds"), (int, float))
]
center_errors = [
    e for e in (center_error_px(t) for t in trials)
    if e is not None
]

by_location = {}
for loc in EXPECTED_LOCATIONS:
    loc_trials = [t for t in trials if t.get("benchmark_location") == loc]
    loc_times = [
        float(t["cycle_time_seconds"])
        for t in loc_trials
        if isinstance(t.get("cycle_time_seconds"), (int, float))
    ]
    by_location[loc] = {
        "trials": len(loc_trials),
        "full_cycle_successes": sum(full_success(t) for t in loc_trials),
        "full_cycle_success_rate_percent": round(
            pct(sum(full_success(t) for t in loc_trials), len(loc_trials)), 2
        ),
        "mean_cycle_time_seconds": (
            round(safe_mean(loc_times), 3) if loc_times else None
        ),
    }

summary = {
    "benchmark": "Final ArUco Benchmark V3",
    "benchmark_root": str(BENCHMARK_ROOT),
    "total_trials_found": total,
    "grasp_successes": grasp_ok,
    "grasp_success_rate_percent": round(pct(grasp_ok, total), 2),
    "release_verified_successes": release_ok,
    "release_verified_rate_percent": round(pct(release_ok, total), 2),
    "cycle_completed": cycle_ok,
    "cycle_completion_rate_percent": round(pct(cycle_ok, total), 2),
    "observed_placement_successes": observed_ok,
    "observed_placement_success_rate_percent": round(pct(observed_ok, total), 2),
    "full_cycle_successes": full_ok,
    "full_cycle_success_rate_percent": round(pct(full_ok, total), 2),
    "cycle_time_seconds": {
        "count": len(cycle_times),
        "mean": round(safe_mean(cycle_times), 3) if cycle_times else None,
        "median": round(statistics.median(cycle_times), 3) if cycle_times else None,
        "stdev": round(safe_stdev(cycle_times), 3) if cycle_times else None,
        "min": round(min(cycle_times), 3) if cycle_times else None,
        "max": round(max(cycle_times), 3) if cycle_times else None,
    },
    "center_error_px": {
        "count": len(center_errors),
        "mean": round(safe_mean(center_errors), 3) if center_errors else None,
        "max": round(max(center_errors), 3) if center_errors else None,
    },
    "per_location": by_location,
    "failures": [
        {
            "benchmark_location": t.get("benchmark_location"),
            "timestamp_start": t.get("timestamp_start"),
            "failure_reason": t.get("failure_reason"),
            "json_path": t.get("_path"),
        }
        for t in trials
        if not full_success(t)
    ],
}

json_path = OUTPUT_DIR / "aruco_final_benchmark_v3_summary.json"
json_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

txt_lines = [
    "FINAL ARUCO BENCHMARK V3 - SUMMARY",
    "=" * 48,
    f"Trials found: {total}",
    f"Grasp success: {grasp_ok}/{total} ({pct(grasp_ok, total):.2f}%)",
    f"Release verified: {release_ok}/{total} ({pct(release_ok, total):.2f}%)",
    f"Cycle completed: {cycle_ok}/{total} ({pct(cycle_ok, total):.2f}%)",
    f"Observed placement success: {observed_ok}/{total} ({pct(observed_ok, total):.2f}%)",
    f"Full-cycle success: {full_ok}/{total} ({pct(full_ok, total):.2f}%)",
    "",
    "PER LOCATION",
    "-" * 48,
]

for loc in EXPECTED_LOCATIONS:
    s = by_location[loc]
    txt_lines.append(
        f"{loc}: {s['full_cycle_successes']}/{s['trials']} "
        f"({s['full_cycle_success_rate_percent']:.2f}%), "
        f"mean cycle time={s['mean_cycle_time_seconds']} s"
    )

txt_lines += [
    "",
    "CYCLE TIME",
    "-" * 48,
    f"Mean: {summary['cycle_time_seconds']['mean']} s",
    f"Median: {summary['cycle_time_seconds']['median']} s",
    f"Std dev: {summary['cycle_time_seconds']['stdev']} s",
    f"Min: {summary['cycle_time_seconds']['min']} s",
    f"Max: {summary['cycle_time_seconds']['max']} s",
    "",
    "TARGET CENTER REPEATABILITY",
    "-" * 48,
    f"Mean distance from nominal center: {summary['center_error_px']['mean']} px",
    f"Maximum distance from nominal center: {summary['center_error_px']['max']} px",
    "",
    f"Detailed CSV: {csv_path}",
    f"Machine-readable summary: {json_path}",
]

txt_path = OUTPUT_DIR / "aruco_final_benchmark_v3_summary.txt"
txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

print("\n".join(txt_lines))
print(f"\nSaved summary TXT: {txt_path}")
print(f"Saved summary JSON: {json_path}")
print(f"Saved detailed CSV: {csv_path}")

if total != 20:
    print(
        f"\nWARNING: Expected 20 official trials, but found {total}. "
        "Review the CSV before freezing the benchmark."
    )
