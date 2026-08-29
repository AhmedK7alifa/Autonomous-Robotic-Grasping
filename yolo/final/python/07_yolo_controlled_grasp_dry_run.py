from collections import deque
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FINAL_PIPELINE_FILE = SCRIPT_DIR / "06_yolo_final_perception_mapping.py"
FINAL_REFERENCE_DIR = (
    PROJECT_ROOT
    / "06_pixel_to_robot"
    / "FINAL_ARUCO_V3_REFERENCE"
)
CANONICAL_CONTROL_FILE = (
    FINAL_REFERENCE_DIR
    / "aruco_FINAL_BENCHMARK_V3_LOCATION1.py"
)
FINAL_CONTROL_PATTERN = "aruco_FINAL_BENCHMARK_V3_LOCATION*.py"

SAFE_REST_FILE = (
    PROJECT_ROOT / "03_robot_control" / "poses" / "safe_rest_final.json"
)
BOX_HIGH_FILE = (
    PROJECT_ROOT
    / "03_robot_control"
    / "poses"
    / "box_high_clearance_final.json"
)
BOX_LOW_RELEASE_FILE = (
    PROJECT_ROOT
    / "03_robot_control"
    / "poses"
    / "box_low_release_final.json"
)

WINDOW_NAME = "YOLO Controlled-Grasp Interface Dry Run"

CONTROL_FUNCTION_NAMES = {
    "load_pose",
    "read_pose",
    "print_pose",
    "smooth",
    "move_smoothly",
    "move_smoothly_fixed_grip",
    "log_grip_retention",
    "start_problems",
    "adaptive_close",
    "hold_grip",
    "move_gripper",
    "open_gripper_and_verify",
}

EXPECTED_CONTROL_AST_SHA256 = (
    "6187e7aa74c19aca20a72a1d14cb0bb855c36143ec0e3d58d10723cded1f82a3"
)

EXPECTED_MOTION_CONSTANTS = {
    "START_TOL": 8.0,
    "TO_ABOVE_STEPS": 240,
    "VERTICAL_DESCENT_STEPS": 180,
    "LIFT_BACK_TO_ABOVE_STEPS": 180,
    "MIN_CLOSE": 48.0,
    "CONTACT_ACTUAL_MAX": 70.5,
    "CONTACT_REQUEST_MAX": 64.5,
    "CONTACT_GAP_MIN": 2.5,
    "GRIP_STEP": 1.0,
    "GRIP_DELAY": 0.25,
    "STALL_THRESHOLD": 0.12,
    "STALL_COUNT": 3,
    "RETRACT_STEPS": 240,
    "BOX_HIGH_STEPS": 220,
    "RELEASE_STEPS": 80,
    "PLACEMENT_STAGE1_FACTOR": 0.60,
    "PLACEMENT_STAGE1_STEPS": 160,
    "PLACEMENT_STAGE2_STEPS": 180,
    "PLACEMENT_SETTLE_SECONDS": 0.8,
    "FINAL_RELEASE_OPEN_GRIPPER": 72.00,
    "RELEASE_OPEN_STEPS": 80,
    "RELEASE_OPEN_DELAY": 0.05,
    "RELEASE_OPEN_MIN": 71.0,
    "RELEASE_MIN_OPEN_DELTA": 0.8,
    "RELEASE_SETTLE_SECONDS": 2.0,
    "UPPER_MID_RETENTION_WEIGHT_MIN": 0.50,
    "UPPER_MID_HOLD_MARGIN": 3.5,
    "UPPER_MID_HOLD_CAP": 63.0,
    "RETURN_STEPS": 220,
    "FINAL_GRIP_STEPS": 60,
    "MOVE_DELAY": 0.05,
    "HOLD_SECONDS": 1.5,
    "START_DELAY": 3.0,
}

EXPECTED_MAX_RELATIVE_TARGET = 8.0

FORBIDDEN_ACTIVE_NAMES = {
    "SO101Follower",
    "SO101FollowerConfig",
    "send_action",
    "get_observation",
    "connect",
    "disconnect",
    "create_detector",
    "detect",
    "target_marker",
    "acquire_target",
}

LOCATION_BRANCH_NAMES = {
    "location",
    "location_number",
    "benchmark_location",
}


def load_local_module(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")

    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load required script: {path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def named_literal_assignments(tree: ast.Module) -> dict:
    assignments = {}

    for node in tree.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            continue

        name = node.targets[0].id

        if name not in EXPECTED_MOTION_CONSTANTS:
            continue

        try:
            assignments[name] = ast.literal_eval(node.value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Motion constant is not a literal: {name}"
            ) from error

    return assignments


def control_function_digest(tree: ast.Module) -> tuple[str, set[str]]:
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in CONTROL_FUNCTION_NAMES
        )
    ]
    found_names = {node.name for node in selected}
    serialized = "".join(
        ast.dump(node, include_attributes=False) for node in selected
    )
    return hashlib.sha256(serialized.encode()).hexdigest(), found_names


def extract_max_relative_target(tree: ast.Module) -> float:
    values = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function_name = (
            node.func.id if isinstance(node.func, ast.Name) else None
        )

        if function_name != "SO101FollowerConfig":
            continue

        for keyword in node.keywords:
            if keyword.arg == "max_relative_target":
                values.append(float(ast.literal_eval(keyword.value)))

    if values != [EXPECTED_MAX_RELATIVE_TARGET]:
        raise RuntimeError(
            "Unexpected max_relative_target configuration: "
            f"{values}"
        )

    return values[0]


def audit_active_dependencies(script_path: Path) -> None:
    tree = ast.parse(
        script_path.read_text(encoding="utf-8"),
        filename=str(script_path),
    )
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(str(node.module))

    if any(
        module == "serial"
        or module.startswith("serial.")
        or module == "lerobot"
        or module.startswith("lerobot.")
        for module in imported_modules
    ):
        raise RuntimeError("Hardware library import found in dry-run script.")

    active_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    active_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    forbidden_found = sorted(
        FORBIDDEN_ACTIVE_NAMES & (active_names | active_attributes)
    )

    if forbidden_found:
        raise RuntimeError(
            "Forbidden hardware or ArUco perception dependency found: "
            f"{forbidden_found}"
        )

    location_found = sorted(LOCATION_BRANCH_NAMES & active_names)

    if location_found:
        raise RuntimeError(
            "Location-specific name found in global YOLO handoff: "
            f"{location_found}"
        )


def audit_final_v3_parity() -> dict:
    audit_active_dependencies(Path(__file__).resolve())
    reference_files = sorted(
        FINAL_REFERENCE_DIR.glob(FINAL_CONTROL_PATTERN)
    )

    if len(reference_files) != 5:
        raise RuntimeError(
            "Expected five FINAL V3 benchmark references; found "
            f"{len(reference_files)}."
        )

    file_results = []

    for path in reference_files:
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        constants = named_literal_assignments(tree)

        if constants != EXPECTED_MOTION_CONSTANTS:
            differing = sorted(
                name
                for name, expected in EXPECTED_MOTION_CONSTANTS.items()
                if constants.get(name) != expected
            )
            raise RuntimeError(
                f"Motion constant parity failed for {path.name}: "
                f"{differing}"
            )

        digest, function_names = control_function_digest(tree)

        if function_names != CONTROL_FUNCTION_NAMES:
            missing = sorted(CONTROL_FUNCTION_NAMES - function_names)
            raise RuntimeError(
                f"Reusable control functions missing from {path.name}: "
                f"{missing}"
            )

        if digest != EXPECTED_CONTROL_AST_SHA256:
            raise RuntimeError(
                f"Control-function AST parity failed for {path.name}: "
                f"{digest}"
            )

        max_relative_target = extract_max_relative_target(tree)
        file_results.append(
            {
                "file": path.name,
                "control_ast_sha256": digest,
                "max_relative_target": max_relative_target,
            }
        )

    return {
        "reference_files": file_results,
        "motion_constants": dict(EXPECTED_MOTION_CONSTANTS),
        "control_function_count": len(CONTROL_FUNCTION_NAMES),
        "control_ast_sha256": EXPECTED_CONTROL_AST_SHA256,
        "max_relative_target": EXPECTED_MAX_RELATIVE_TARGET,
        "aruco_perception_dependency": False,
        "location_specific_branching": False,
    }


def acquire_one_locked_target(live, yolo_model) -> dict:
    camera_matrix, distortion = live.load_camera_calibration()
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (live.WIDTH, live.HEIGHT),
        cv2.CV_32FC1,
    )
    center_samples = deque(maxlen=live.ROLLING_BUFFER_SIZE)
    box_samples = deque(maxlen=live.ROLLING_BUFFER_SIZE)
    locked = None
    camera = cv2.VideoCapture(live.CAMERA_INDEX, cv2.CAP_DSHOW)

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, live.WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, live.HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, live.FPS)

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open camera index "
                f"{live.CAMERA_INDEX} with DirectShow."
            )

        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(camera.get(cv2.CAP_PROP_FPS))

        if actual_width != live.WIDTH or actual_height != live.HEIGHT:
            raise RuntimeError(
                "Unexpected camera resolution: "
                f"{actual_width}x{actual_height}."
            )

        if (
            not np.isfinite(actual_fps)
            or abs(actual_fps - live.FPS) > 0.5
        ):
            raise RuntimeError(
                f"Unexpected camera FPS: {actual_fps:.1f}."
            )

        print(
            "Camera-only dry run ready. "
            "Press S once after LOCKED; Q quits.",
            flush=True,
        )

        while True:
            success, raw_frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a camera frame.")

            if raw_frame.shape[:2] != (live.HEIGHT, live.WIDTH):
                raise RuntimeError(
                    "Captured frame has unexpected size: "
                    f"{raw_frame.shape[1]}x{raw_frame.shape[0]}."
                )

            frame = cv2.remap(
                raw_frame,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
            )
            result = yolo_model.predict(
                source=frame,
                imgsz=live.INFERENCE_IMAGE_SIZE,
                device="cpu",
                classes=[live.CUBE_CLASS_ID],
                conf=live.CONFIDENCE_THRESHOLD,
                verbose=False,
            )[0]
            detections = live.extract_cube_detections(result)
            selected = detections[0] if detections else None

            if selected is None:
                center_samples.clear()
                box_samples.clear()
                median_center = None
                median_box = None
                standard_deviation = None
                stable = False
            else:
                center_samples.append(selected.center.copy())
                box_samples.append(
                    np.asarray(selected.box, dtype=np.float64)
                )
                center_array = np.asarray(
                    center_samples,
                    dtype=np.float64,
                )
                box_array = np.asarray(box_samples, dtype=np.float64)
                median_center = np.median(center_array, axis=0)
                median_box = np.median(box_array, axis=0)
                standard_deviation = center_array.std(axis=0)
                stable = (
                    len(center_samples) >= live.MIN_SAMPLES
                    and float(standard_deviation.max())
                    <= live.MAX_STD_PX
                )

            display_frame = frame.copy()

            for detection in detections:
                live.draw_detection(
                    display_frame,
                    detection,
                    selected=detection is selected,
                )

            live.draw_status(
                display_frame,
                detections,
                selected,
                center_samples,
                median_center,
                standard_deviation,
                stable,
            )
            cv2.putText(
                display_frame,
                "DRY RUN ONLY - S: lock once - Q: quit - NO HARDWARE PATH",
                (18, live.HEIGHT - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "DRY RUN ONLY - S: lock once - Q: quit - NO HARDWARE PATH",
                (18, live.HEIGHT - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                raise KeyboardInterrupt("Dry run cancelled before lock.")

            if key not in (ord("s"), ord("S")):
                continue

            if (
                not stable
                or selected is None
                or median_center is None
                or median_box is None
                or standard_deviation is None
            ):
                print("Target is not LOCKED; nothing captured.", flush=True)
                continue

            locked = {
                "median_bbox_center": median_center.copy(),
                "median_bbox": median_box.copy(),
                "standard_deviation": standard_deviation.copy(),
            }
            break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.", flush=True)

    if locked is None:
        raise RuntimeError("No LOCKED YOLO target was captured.")

    return locked


def assert_pose_finite(name: str, pose: dict, joint_names: list[str]) -> None:
    missing = [joint for joint in joint_names if joint not in pose]

    if missing:
        raise RuntimeError(f"{name} is missing joints: {missing}")

    values = np.asarray(
        [float(pose[joint]) for joint in joint_names],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite values.")


def build_dry_run_plan(
    locked: dict,
    final_pipeline,
    mapping,
    mapping_model: dict,
    source_anchors: list[dict],
    safe_rest: dict,
    box_high: dict,
    box_low_release: dict,
) -> dict:
    bbox_center = np.asarray(
        locked["median_bbox_center"],
        dtype=np.float64,
    )
    grasp_point = final_pipeline.final_grasp_point(
        bbox_center,
        locked["median_bbox"],
    )
    prediction = mapping.predict(grasp_point, mapping_model)

    if prediction is None:
        raise RuntimeError(
            "Frozen grasp reference lies outside the validated mapping mesh."
        )

    dynamic = mapping.dynamic_source_poses(
        grasp_point,
        prediction,
        source_anchors,
    )
    frozen_result = final_pipeline.map_grasp_point(
        grasp_point,
        mapping,
        mapping_model,
        source_anchors,
    )

    if not frozen_result["inside_workspace"]:
        raise RuntimeError("Frozen pipeline unexpectedly rejected grasp point.")

    for pose_name in ("above", "grasp"):
        for joint in mapping.JOINTS:
            if not np.isclose(
                float(frozen_result[pose_name][joint]),
                float(dynamic[pose_name][joint]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "Frozen pipeline/direct FINAL mapping parity failed: "
                    f"{pose_name}/{joint}"
                )

    finite_arrays = [
        bbox_center,
        grasp_point,
        np.asarray(prediction["weights"], dtype=np.float64),
        np.asarray(
            [
                float(item["distance_px"])
                for item in dynamic["support"]
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [float(item["weight"]) for item in dynamic["support"]],
            dtype=np.float64,
        ),
        np.asarray([float(dynamic["hold_margin"])], dtype=np.float64),
    ]

    if dynamic["hold_cap"] is not None:
        finite_arrays.append(
            np.asarray([float(dynamic["hold_cap"])], dtype=np.float64)
        )

    if not all(np.isfinite(values).all() for values in finite_arrays):
        raise RuntimeError("Dry-run handoff contains non-finite values.")

    for name, pose in (
        ("source_above", dynamic["above"]),
        ("source_grasp", dynamic["grasp"]),
        ("safe_rest", safe_rest),
        ("box_high", box_high),
        ("box_low_release", box_low_release),
    ):
        assert_pose_finite(name, pose, mapping.JOINTS)

    return {
        "bbox_center": bbox_center,
        "grasp_point": grasp_point,
        "triangle_index": int(prediction["triangle_index"]) + 1,
        "triangle_ids": list(prediction["triangle_ids"]),
        "correction_mode": str(dynamic["source_mode"]),
        "support": list(dynamic["support"]),
        "source_above": dict(dynamic["above"]),
        "source_grasp": dict(dynamic["grasp"]),
        "hold_margin": float(dynamic["hold_margin"]),
        "hold_cap": (
            None
            if dynamic["hold_cap"] is None
            else float(dynamic["hold_cap"])
        ),
        "safe_rest": dict(safe_rest),
        "box_high": dict(box_high),
        "box_low_release": dict(box_low_release),
        "all_values_finite": True,
        "inside_mapping_mesh": True,
    }


def print_pose(title: str, pose: dict, joint_names: list[str]) -> None:
    print(title, flush=True)

    for joint in joint_names:
        print(f"  {joint}: {float(pose[joint]):.4f}", flush=True)


def print_dry_run_plan(
    plan: dict,
    parity: dict,
    joint_names: list[str],
) -> None:
    center = plan["bbox_center"]
    point = plan["grasp_point"]
    print("\nCONTROLLED-GRASP SOFTWARE HANDOFF DRY RUN", flush=True)
    print(
        f"1. YOLO bbox center: ({center[0]:.3f}, {center[1]:.3f}) px",
        flush=True,
    )
    print(
        "2. Frozen upper-quarter grasp reference: "
        f"({point[0]:.3f}, {point[1]:.3f}) px",
        flush=True,
    )
    print(
        f"3. Mapping triangle: {plan['triangle_index']}/8 - "
        + " / ".join(plan["triangle_ids"]),
        flush=True,
    )
    print(f"4. Correction mode: {plan['correction_mode']}", flush=True)
    print("5. Support anchors and weights:", flush=True)

    for item in plan["support"]:
        print(
            f"  {item['name']}: weight={item['weight']:.6f}, "
            f"distance={item['distance_px']:.3f}px",
            flush=True,
        )

    print_pose("6. source_above:", plan["source_above"], joint_names)
    print_pose(
        "7. source_grasp / open-grasp target:",
        plan["source_grasp"],
        joint_names,
    )
    print(f"8. hold_margin: {plan['hold_margin']:.4f}", flush=True)
    print(f"9. hold_cap: {plan['hold_cap']}", flush=True)
    print("10. Expected fixed-gripper retention logic inputs:", flush=True)
    print("  raw_contact_request: runtime contact input", flush=True)
    print("  contact_actual: runtime contact input", flush=True)
    print(f"  dynamic hold_margin: {plan['hold_margin']:.4f}", flush=True)
    print(f"  dynamic hold_cap: {plan['hold_cap']}", flush=True)
    print(
        "  pre_cap_hold = max(raw_contact_request, "
        "contact_actual - hold_margin)",
        flush=True,
    )
    print(
        "  fixed_hold = min(hold_cap, pre_cap_hold) when capped; "
        "otherwise pre_cap_hold",
        flush=True,
    )
    print_pose("11. safe-rest target:", plan["safe_rest"], joint_names)
    print_pose("12. box-high target:", plan["box_high"], joint_names)
    print_pose(
        "13. box-low-release target:",
        plan["box_low_release"],
        joint_names,
    )
    print("\nFINAL V3 PARITY AUDIT", flush=True)
    print(
        "Motion constants: PASS - "
        f"{len(parity['motion_constants'])} active named constants "
        "match all five FINAL V3 references",
        flush=True,
    )
    print(
        "Reusable control functions: PASS - "
        f"{parity['control_function_count']} functions AST-identical; "
        f"SHA256={parity['control_ast_sha256']}",
        flush=True,
    )
    print(
        "Driver max_relative_target parity: PASS - "
        f"{parity['max_relative_target']}",
        flush=True,
    )
    print("ArUco perception dependency: NONE", flush=True)
    print("Location-specific YOLO branching: NONE", flush=True)
    print("Calculated values finite: PASS", flush=True)
    print("Grasp reference inside validated mapping mesh: PASS", flush=True)
    print("Frozen YOLO/direct FINAL mapping target parity: PASS", flush=True)
    print("DRY RUN COMPLETE - NO HARDWARE OR MOTION PATH EXISTS", flush=True)


def main() -> None:
    parity = audit_final_v3_parity()
    final_pipeline = load_local_module(
        "final_yolo_perception_mapping",
        FINAL_PIPELINE_FILE,
    )
    live = final_pipeline.load_local_module(
        "cube_live_detector",
        final_pipeline.LIVE_DETECTOR_FILE,
    )
    mapping_dry_run = final_pipeline.load_local_module(
        "yolo_mapping_dry_run",
        final_pipeline.MAPPING_DRY_RUN_FILE,
    )
    mapping = mapping_dry_run.load_final_mapping_functions()
    mapping_model = mapping.load_model()
    source_anchors = mapping.load_source_anchors(mapping_model)
    safe_rest = mapping.load_pose(SAFE_REST_FILE)
    box_high = mapping.load_pose(BOX_HIGH_FILE)
    box_low_release = mapping.load_pose(BOX_LOW_RELEASE_FILE)
    yolo_model = live.load_frozen_model()
    locked = acquire_one_locked_target(live, yolo_model)
    plan = build_dry_run_plan(
        locked,
        final_pipeline,
        mapping,
        mapping_model,
        source_anchors,
        safe_rest,
        box_high,
        box_low_release,
    )
    print_dry_run_plan(plan, parity, mapping.JOINTS)


if __name__ == "__main__":
    main()
