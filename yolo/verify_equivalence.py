"""Static equivalence checks for the maintained YOLO controller chain."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


YOLO_DIR = Path(__file__).resolve().parent
HISTORICAL_CONTROLLER = YOLO_DIR / "benchmark_controller.py"
MAINTAINED_CONTROLLER = YOLO_DIR / "controller.py"
PERCEPTION_FILE = YOLO_DIR / "perception.py"
PERCEPTION_MAPPING_FILE = YOLO_DIR / "perception_mapping.py"
CONTROL_INTERFACE_FILE = YOLO_DIR / "control_interface.py"

EXPECTED_CONTROLLER_SHA256 = (
    "8eb4fa72003b5af379aa443fdf6f74bfa72ab06d10cbb7ee5bca39d6ac380ac0"
)

EXPECTED_PERCEPTION_CONSTANTS = {
    "CAMERA_INDEX": 0,
    "WIDTH": 1280,
    "HEIGHT": 720,
    "FPS": 30,
    "CUBE_CLASS_ID": 0,
    "INFERENCE_IMAGE_SIZE": 640,
    "CONFIDENCE_THRESHOLD": 0.25,
    "DUPLICATE_IOU_THRESHOLD": 0.50,
    "ROLLING_BUFFER_SIZE": 30,
    "MIN_SAMPLES": 20,
    "MAX_STD_PX": 2.0,
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


def verify() -> None:
    historical_digest = hashlib.sha256(
        HISTORICAL_CONTROLLER.read_bytes()
    ).hexdigest()
    if historical_digest != EXPECTED_CONTROLLER_SHA256:
        raise RuntimeError("Historical YOLO controller hash mismatch.")

    historical_functions = function_map(parse(HISTORICAL_CONTROLLER))
    maintained_functions = function_map(parse(MAINTAINED_CONTROLLER))
    if historical_functions != maintained_functions:
        changed = sorted(
            name
            for name in historical_functions.keys() | maintained_functions.keys()
            if historical_functions.get(name) != maintained_functions.get(name)
        )
        raise RuntimeError(
            "Maintained YOLO controller function AST mismatch: "
            + ", ".join(changed)
        )

    perception_constants = literal_assignments(parse(PERCEPTION_FILE))
    for name, expected in EXPECTED_PERCEPTION_CONSTANTS.items():
        if perception_constants.get(name) != expected:
            raise RuntimeError(
                f"YOLO perception constant mismatch: {name}="
                f"{perception_constants.get(name)!r}, expected {expected!r}"
            )

    mapping_constants = literal_assignments(parse(PERCEPTION_MAPPING_FILE))
    if mapping_constants.get("FINAL_GRASP_HEIGHT_FRACTION") != 0.25:
        raise RuntimeError("Upper-quarter grasp fraction is not 0.25.")

    control_constants = literal_assignments(parse(CONTROL_INTERFACE_FILE))
    if control_constants.get("EXPECTED_MAX_RELATIVE_TARGET") != 8.0:
        raise RuntimeError("Control-interface max-relative-target check changed.")
    if control_constants.get("EXPECTED_CONTROL_AST_SHA256") != (
        "6187e7aa74c19aca20a72a1d14cb0bb855c36143ec0e3d58d10723cded1f82a3"
    ):
        raise RuntimeError("Control-function AST digest check changed.")


def main() -> None:
    verify()
    print("YOLO controller equivalence checks passed.")
    print("- historical full-cycle controller SHA-256 matches")
    print("- all maintained controller function ASTs match the historical file")
    print("- perception, upper-quarter, and shared-control checks match")


if __name__ == "__main__":
    main()
