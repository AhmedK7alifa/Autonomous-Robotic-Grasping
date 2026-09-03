"""Static equivalence checks for the maintained ArUco controller.

The historical benchmark controllers remain the authoritative executable
record. This check verifies their hashes, confirms that their non-entry-point
functions are identical, and compares those functions and active constants
with the maintained controller. The maintained ``main`` differs intentionally
because repository-relative paths and location configuration were introduced.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

ARUCO_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = ARUCO_DIR / "benchmark_controllers"
MAINTAINED_FILE = ARUCO_DIR / "controller.py"
LOCATIONS_FILE = ARUCO_DIR / "locations.yaml"

EXPECTED_HASHES = {
    "location_01.py": "209cbe4b0def1eb2b569475dc3f9e3a4aad81bd7bc56617a1f9f056c607966ac",
    "location_02.py": "582b178f6ebd8d8e7354afc59c7dac27d7fdbcbfa8292ceddcf2165baa010710",
    "location_03.py": "8ecf4588939e655d8cf99534e39ec4d0af2f63b25edafe459b1085b9026b1358",
    "location_04.py": "462d99a1727aefd7aba4604c69030edefc90ffbe10dce6c0c5bc29d3d750c4a4",
    "location_05.py": "92a942b2a12c6d6c40f4cf34d1c0e38fa1cb75aa8af6bccd478d4adc6aa4d3b6",
}

ACTIVE_CONSTANTS = {
    "CAMERA_INDEX",
    "WIDTH",
    "HEIGHT",
    "FPS",
    "TARGET_ID",
    "SAMPLE_COUNT",
    "MIN_SAMPLES",
    "MAX_STD_PX",
    "START_TOL",
    "TO_ABOVE_STEPS",
    "VERTICAL_DESCENT_STEPS",
    "LIFT_BACK_TO_ABOVE_STEPS",
    "MIN_CLOSE",
    "CONTACT_ACTUAL_MAX",
    "CONTACT_REQUEST_MAX",
    "CONTACT_GAP_MIN",
    "GRIP_STEP",
    "GRIP_DELAY",
    "STALL_THRESHOLD",
    "STALL_COUNT",
    "RETRACT_STEPS",
    "BOX_HIGH_STEPS",
    "RELEASE_STEPS",
    "PLACEMENT_STAGE1_FACTOR",
    "PLACEMENT_STAGE1_STEPS",
    "PLACEMENT_STAGE2_STEPS",
    "PLACEMENT_SETTLE_SECONDS",
    "FINAL_RELEASE_OPEN_GRIPPER",
    "RELEASE_OPEN_STEPS",
    "RELEASE_OPEN_DELAY",
    "RELEASE_OPEN_MIN",
    "RELEASE_MIN_OPEN_DELTA",
    "RELEASE_SETTLE_SECONDS",
    "UPPER_MID_RETENTION_WEIGHT_MIN",
    "UPPER_MID_HOLD_MARGIN",
    "UPPER_MID_HOLD_CAP",
    "RETURN_STEPS",
    "FINAL_GRIP_STEPS",
    "MOVE_DELAY",
    "HOLD_SECONDS",
    "START_DELAY",
    "DYNAMIC_K_NEAREST",
    "EXACT_ANCHOR_PX",
    "SPARSE_SUPPORT_WARNING_PX",
    "RIGHT_PAIR_A",
    "RIGHT_PAIR_B",
    "RIGHT_PAIR_MAX_PERP_PX",
    "RIGHT_PAIR_T_MIN",
    "RIGHT_PAIR_T_MAX",
}

EXPECTED_LOCATION_METADATA = {
    "location_01": ("location_1_lower_left", "final_aruco_benchmark_v3_location_1", "L1_lower_left", [247.8, 271.0]),
    "location_02": ("location_2_upper_left", "final_aruco_benchmark_v3_location_2", "L2_upper_left", [237.0, 134.5]),
    "location_03": ("location_3_center", "final_aruco_benchmark_v3_location_3", "L3_center", [326.2, 195.5]),
    "location_04": ("location_4_upper_right", "final_aruco_benchmark_v3_location_4", "L4_upper_right", [428.2, 134.5]),
    "location_05": ("location_5_lower_right", "final_aruco_benchmark_v3_location_5", "L5_lower_right", [411.2, 290.8]),
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def function_map(tree: ast.Module) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def literal_assignments(tree: ast.Module) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return values


def load_location_metadata() -> dict[str, dict[str, Any]]:
    """Read the fixed, flat location schema without importing controller deps."""
    locations: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in LOCATIONS_FILE.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.strip() == "locations:":
            continue
        indentation = len(raw_line) - len(raw_line.lstrip())
        if indentation == 2 and raw_line.strip().endswith(":"):
            key = raw_line.strip()[:-1]
            current = locations.setdefault(key, {})
            continue
        if indentation == 4 and current is not None:
            name, value_text = raw_line.strip().split(":", 1)
            value_text = value_text.strip()
            if value_text.startswith(("[", "'", '"')):
                current[name] = ast.literal_eval(value_text)
            else:
                current[name] = value_text
            continue
        raise RuntimeError(f"Unexpected locations.yaml line: {raw_line!r}")
    return locations


def verify() -> None:
    historical = [HISTORICAL_DIR / name for name in EXPECTED_HASHES]
    for path in historical:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_HASHES[path.name]:
            raise RuntimeError(f"Historical controller hash mismatch: {path}")

    reference_tree = parse(historical[0])
    reference_functions = function_map(reference_tree)
    comparable_functions = {
        name: body
        for name, body in reference_functions.items()
        if name != "main"
    }
    reference_constants = literal_assignments(reference_tree)

    for path in historical[1:]:
        functions = function_map(parse(path))
        for name, expected in comparable_functions.items():
            if functions.get(name) != expected:
                raise RuntimeError(
                    f"Historical controller function differs: {path.name}:{name}"
                )

    maintained_tree = parse(MAINTAINED_FILE)
    maintained_functions = function_map(maintained_tree)
    for name, expected in comparable_functions.items():
        if maintained_functions.get(name) != expected:
            raise RuntimeError(
                f"Maintained controller function differs from history: {name}"
            )

    maintained_constants = literal_assignments(maintained_tree)
    for name in sorted(ACTIVE_CONSTANTS):
        if reference_constants.get(name) != maintained_constants.get(name):
            raise RuntimeError(
                f"Active constant mismatch for {name}: "
                f"{reference_constants.get(name)!r} != "
                f"{maintained_constants.get(name)!r}"
            )

    location_data = load_location_metadata()
    for key, expected in EXPECTED_LOCATION_METADATA.items():
        item = location_data[key]
        actual = (
            item["output_directory"],
            item["mode"],
            item["benchmark_location"],
            item["nominal_center_px"],
        )
        if actual != expected:
            raise RuntimeError(f"Location metadata mismatch for {key}: {actual!r}")


def main() -> None:
    verify()
    print("ArUco controller equivalence checks passed.")
    print("- five historical SHA-256 hashes match")
    print("- non-main function ASTs match across historical controllers")
    print("- maintained non-main function ASTs match location_01.py")
    print("- active constants and five location metadata sets match")
    print("The historical controllers remain authoritative for their main entry points.")


if __name__ == "__main__":
    main()
