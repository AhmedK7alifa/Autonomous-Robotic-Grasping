"""Promote one completed YOLO cycle record into the official benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIRECTORY = SCRIPT_DIR / "yolo_full_pick_and_place_trials"
BENCHMARK_DIRECTORY = SCRIPT_DIR / "yolo_final_benchmark"

LOCATION_CONFIG = {
    1: ("L1_lower_left", "location_1_lower_left"),
    2: ("L2_upper_left", "location_2_upper_left"),
    3: ("L3_center", "location_3_center"),
    4: ("L4_upper_right", "location_4_upper_right"),
    5: ("L5_lower_right", "location_5_lower_right"),
}

BENCHMARK_FIELDS = {
    "benchmark_method",
    "benchmark_location_number",
    "benchmark_location_name",
    "benchmark_trial_number",
    "benchmark_official",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing YOLO cycle JSON into the official benchmark "
            "and add benchmark metadata to the copy only."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to an existing yolo_full_cycle JSON record.",
    )
    parser.add_argument(
        "--location",
        required=True,
        type=int,
        choices=range(1, 6),
        metavar="1-5",
    )
    parser.add_argument(
        "--trial",
        required=True,
        type=int,
        choices=range(1, 5),
        metavar="1-4",
    )
    return parser.parse_args()


def load_source_record(source_argument: Path) -> tuple[Path, dict[str, Any]]:
    try:
        source_path = source_argument.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Source JSON does not exist or cannot be resolved: "
            f"{source_argument}"
        ) from error

    if not source_path.is_file():
        raise ValueError(f"Source is not a file: {source_path}")

    expected_directory = SOURCE_DIRECTORY.resolve()

    if source_path.parent != expected_directory:
        raise ValueError(
            "Source must be directly inside the cycle-record directory: "
            f"{expected_directory}"
        )

    if not (
        source_path.name.startswith("yolo_full_cycle_")
        and source_path.suffix.lower() == ".json"
    ):
        raise ValueError(
            "Source filename must match yolo_full_cycle_*.json: "
            f"{source_path.name}"
        )

    try:
        with source_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Source JSON is not readable and valid: {source_path}"
        ) from error

    if not isinstance(record, dict):
        raise ValueError("Source JSON must contain one top-level object.")

    collisions = sorted(BENCHMARK_FIELDS.intersection(record))

    if collisions:
        raise ValueError(
            "Source already contains official benchmark fields; refusing "
            "to replace original values: "
            + ", ".join(collisions)
        )

    return source_path, record


def destination_path(location: int, trial: int) -> tuple[Path, str]:
    location_name, folder_name = LOCATION_CONFIG[location]
    folder = BENCHMARK_DIRECTORY / folder_name

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Official benchmark folder is missing: {folder}")

    destination = folder / f"yolo_L{location}_trial_{trial:02d}.json"
    return destination, location_name


def promoted_record(
    original: dict[str, Any],
    location: int,
    location_name: str,
    trial: int,
) -> dict[str, Any]:
    promoted = dict(original)
    promoted.update(
        {
            "benchmark_method": "YOLO",
            "benchmark_location_number": location,
            "benchmark_location_name": location_name,
            "benchmark_trial_number": trial,
            "benchmark_official": True,
        }
    )
    return promoted


def display_value(record: dict[str, Any], key: str) -> Any:
    return record[key] if key in record else "<absent>"


def save_official_record(destination: Path, record: dict[str, Any]) -> None:
    payload = json.dumps(record, indent=2, ensure_ascii=False) + "\n"

    try:
        with destination.open("x", encoding="utf-8", newline="\n") as file:
            file.write(payload)
    except FileExistsError as error:
        raise ValueError(
            f"Official benchmark trial already exists; refusing overwrite: "
            f"{destination}"
        ) from error
    except OSError as error:
        raise ValueError(
            f"Could not create official benchmark record: {destination}"
        ) from error


def main() -> None:
    arguments = parse_arguments()
    source, original = load_source_record(arguments.source)
    destination, location_name = destination_path(
        arguments.location,
        arguments.trial,
    )

    if destination.exists():
        raise ValueError(
            f"Official benchmark trial already exists; refusing overwrite: "
            f"{destination}"
        )

    official_record = promoted_record(
        original,
        arguments.location,
        location_name,
        arguments.trial,
    )
    save_official_record(destination, official_record)

    print(f"Source: {source}")
    print(f"Destination: {destination.resolve()}")
    print(
        f"Location: {arguments.location} ({location_name})"
    )
    print(f"Trial number: {arguments.trial}")
    print(
        "cycle_completed: "
        f"{display_value(original, 'cycle_completed')}"
    )
    print(
        "observed_success: "
        f"{display_value(original, 'observed_success')}"
    )
    print(
        "cycle_time_seconds: "
        f"{display_value(original, 'cycle_time_seconds')}"
    )


if __name__ == "__main__":
    main()
