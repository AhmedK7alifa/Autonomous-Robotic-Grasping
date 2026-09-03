from collections import deque
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np


LOCATION_COUNT = 5

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LIVE_DETECTOR_FILE = SCRIPT_DIR / "perception.py"

FINAL_REFERENCE_DIR = (
    PROJECT_ROOT
    / "aruco"
    / "benchmark_controllers"
)
FINAL_MAPPING_SOURCE = (
    FINAL_REFERENCE_DIR
    / "location_01.py"
)
FINAL_MAPPING_MODEL_FILE = (
    PROJECT_ROOT
    / "mapping"
    / "pixel_to_joint_model.json"
)
FINAL_POSES_DIR = (
    PROJECT_ROOT / "robot_control" / "poses" / "source_anchors"
)

WINDOW_NAME = "YOLO Mapping Dry Run - Camera Only"

SAFE_MAPPING_ASSIGNMENTS = {
    "JOINTS",
    "ARM_JOINTS",
    "UPPER_MID_RETENTION_WEIGHT_MIN",
    "UPPER_MID_HOLD_MARGIN",
    "UPPER_MID_HOLD_CAP",
    "SOURCE_ANCHOR_SPECS",
    "DYNAMIC_K_NEAREST",
    "EXACT_ANCHOR_PX",
    "SPARSE_SUPPORT_WARNING_PX",
    "RIGHT_PAIR_A",
    "RIGHT_PAIR_B",
    "RIGHT_PAIR_MAX_PERP_PX",
    "RIGHT_PAIR_T_MIN",
    "RIGHT_PAIR_T_MAX",
}

SAFE_MAPPING_FUNCTIONS = {
    "load_pose",
    "load_model",
    "barycentric",
    "predict",
    "load_source_anchors",
    "dynamic_source_poses",
}

FORBIDDEN_MAPPING_NAMES = {
    "SO101Follower",
    "SO101FollowerConfig",
    "PORT",
    "ROBOT_ID",
    "connect",
    "disconnect",
    "send_action",
    "move_smoothly",
    "move_smoothly_fixed_grip",
    "move_gripper",
    "adaptive_close",
    "main",
}


def assignment_name(node: ast.Assign) -> str | None:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None

    return node.targets[0].id


def load_final_mapping_functions() -> SimpleNamespace:
    """Load only the frozen V3 mapping math, never its hardware code."""
    if not FINAL_MAPPING_SOURCE.exists():
        raise FileNotFoundError(
            f"Final mapping reference not found: {FINAL_MAPPING_SOURCE}"
        )

    if not FINAL_MAPPING_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Final mapping model not found: {FINAL_MAPPING_MODEL_FILE}"
        )

    source = FINAL_MAPPING_SOURCE.read_text(encoding="utf-8")
    source_tree = ast.parse(source, filename=str(FINAL_MAPPING_SOURCE))
    selected_nodes = []
    found_assignments = set()
    found_functions = set()

    for node in source_tree.body:
        if isinstance(node, ast.Assign):
            name = assignment_name(node)

            if name in SAFE_MAPPING_ASSIGNMENTS:
                if any(isinstance(child, ast.Call) for child in ast.walk(node)):
                    raise RuntimeError(
                        f"Unsafe call found in mapping assignment: {name}"
                    )

                selected_nodes.append(node)
                found_assignments.add(name)

        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in SAFE_MAPPING_FUNCTIONS
        ):
            selected_nodes.append(node)
            found_functions.add(node.name)

    if found_assignments != SAFE_MAPPING_ASSIGNMENTS:
        missing = sorted(SAFE_MAPPING_ASSIGNMENTS - found_assignments)
        raise RuntimeError(
            "Final mapping reference is missing safe assignments: "
            f"{missing}"
        )

    if found_functions != SAFE_MAPPING_FUNCTIONS:
        missing = sorted(SAFE_MAPPING_FUNCTIONS - found_functions)
        raise RuntimeError(
            "Final mapping reference is missing safe functions: "
            f"{missing}"
        )

    referenced_names = {
        child.id
        for node in selected_nodes
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }
    forbidden_found = sorted(FORBIDDEN_MAPPING_NAMES & referenced_names)

    if forbidden_found:
        raise RuntimeError(
            "Hardware or motion name found in selected mapping code: "
            f"{forbidden_found}"
        )

    safe_tree = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(safe_tree)
    namespace = {
        "__builtins__": __builtins__,
        "json": json,
        "np": np,
        "MODEL_FILE": FINAL_MAPPING_MODEL_FILE,
        "POSES_DIR": FINAL_POSES_DIR,
    }
    exec(
        compile(safe_tree, str(FINAL_MAPPING_SOURCE), "exec"),
        namespace,
    )

    exported_names = SAFE_MAPPING_ASSIGNMENTS | SAFE_MAPPING_FUNCTIONS
    return SimpleNamespace(
        **{name: namespace[name] for name in exported_names}
    )


def load_live_detector_module():
    if not LIVE_DETECTOR_FILE.exists():
        raise FileNotFoundError(
            f"Live detector script not found: {LIVE_DETECTOR_FILE}"
        )

    specification = importlib.util.spec_from_file_location(
        "cube_live_detector",
        LIVE_DETECTOR_FILE,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            f"Could not load live detector: {LIVE_DETECTOR_FILE}"
        )

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def print_joint_pose(title: str, pose: dict, joint_names: list[str]) -> None:
    print(title, flush=True)

    for joint in joint_names:
        print(f"  {joint}: {float(pose[joint]):.4f}", flush=True)


def map_locked_center(
    location_number: int,
    median_center: np.ndarray,
    standard_deviation: np.ndarray,
    selected_confidence: float,
    mapping,
    model: dict,
    anchors: list[dict],
) -> dict:
    center = np.asarray(median_center, dtype=np.float64)
    prediction = mapping.predict(center, model)
    inside_workspace = prediction is not None

    record = {
        "location": location_number,
        "center": center.copy(),
        "standard_deviation": np.asarray(
            standard_deviation,
            dtype=np.float64,
        ).copy(),
        "confidence": float(selected_confidence),
        "inside_workspace": inside_workspace,
        "triangle": None,
        "source_mode": None,
    }

    print("\n" + "=" * 72, flush=True)
    print(f"YOLO MAPPING DRY RUN - LOCATION {location_number}", flush=True)
    print("=" * 72, flush=True)
    print(
        "YOLO median pixel center: "
        f"({center[0]:.3f}, {center[1]:.3f}) px",
        flush=True,
    )
    print(
        "YOLO center stability: "
        f"std_x={standard_deviation[0]:.3f}px, "
        f"std_y={standard_deviation[1]:.3f}px",
        flush=True,
    )
    print(
        f"Selected YOLO confidence: {selected_confidence:.3f}",
        flush=True,
    )
    print(
        "Inside validated mapping workspace: "
        f"{'YES' if inside_workspace else 'NO'}",
        flush=True,
    )

    if not inside_workspace:
        print("Mapping triangle: NONE", flush=True)
        print("Special right-side corridor used: NO", flush=True)
        print(
            "Mapped joint target: NOT GENERATED (point is outside "
            "the validated interpolation triangles)",
            flush=True,
        )
        return record

    dynamic = mapping.dynamic_source_poses(center, prediction, anchors)
    triangle_number = int(prediction["triangle_index"]) + 1
    triangle_ids = list(prediction["triangle_ids"])
    triangle_weights = np.asarray(
        prediction["weights"],
        dtype=np.float64,
    )
    source_mode = str(dynamic["source_mode"])
    corridor_used = source_mode == "right_pairwise_linear"

    record["triangle"] = triangle_ids
    record["source_mode"] = source_mode

    print(
        f"Mapping triangle: {triangle_number}/8 - "
        + " / ".join(triangle_ids),
        flush=True,
    )
    print(
        "Barycentric weights: "
        + ", ".join(
            f"{point_id}={weight:.6f}"
            for point_id, weight in zip(triangle_ids, triangle_weights)
        ),
        flush=True,
    )
    print(f"Correction region/mode: {source_mode}", flush=True)
    print(
        "Special right-side corridor used: "
        f"{'YES' if corridor_used else 'NO'}",
        flush=True,
    )

    if corridor_used:
        print(
            "Right-pair corridor projection: "
            f"t={dynamic['right_pairwise_t']:.6f}, "
            f"perpendicular_distance="
            f"{dynamic['right_pairwise_perp_px']:.3f}px",
            flush=True,
        )

    print("Dynamic correction support:", flush=True)

    for item in dynamic["support"]:
        print(
            f"  {item['name']}: "
            f"distance={item['distance_px']:.3f}px, "
            f"weight={item['weight']:.6f}",
            flush=True,
        )

    print_joint_pose(
        "Day 07 barycentric mapped joint target (reference):",
        dynamic["base_mapping"],
        mapping.JOINTS,
    )
    print_joint_pose(
        "Final corrected pre-grasp ABOVE target:",
        dynamic["above"],
        mapping.JOINTS,
    )
    print_joint_pose(
        "Final corrected OPEN-GRASP target:",
        dynamic["grasp"],
        mapping.JOINTS,
    )

    if (
        float(dynamic["nearest_anchor_distance_px"])
        > float(mapping.SPARSE_SUPPORT_WARNING_PX)
    ):
        print(
            "Mapping warning: sparse local correction support.",
            flush=True,
        )

    return record


def draw_dry_run_status(
    frame: np.ndarray,
    location_number: int,
    last_mapping_status: str,
) -> None:
    lines = (
        f"Dry-run location: {location_number}/{LOCATION_COUNT}",
        last_mapping_status,
        "S: map median only when LOCKED   Q: quit",
        "CAMERA + MAPPING ONLY - NO ROBOT OUTPUT",
    )

    for index, line in enumerate(lines):
        y_position = frame.shape[0] - 103 + index * 27
        color = (0, 255, 255) if index == 3 else (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    live = load_live_detector_module()
    mapping = load_final_mapping_functions()
    mapping_model = mapping.load_model()
    source_anchors = mapping.load_source_anchors(mapping_model)
    yolo_model = live.load_frozen_model()
    camera_matrix, distortion = live.load_camera_calibration()
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (live.WIDTH, live.HEIGHT),
        cv2.CV_32FC1,
    )
    samples = deque(maxlen=live.ROLLING_BUFFER_SIZE)
    location_number = 1
    records = []
    last_mapping_status = "Mapping: waiting for LOCKED target"
    completed = False
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
                f"{actual_width}x{actual_height}; "
                f"required {live.WIDTH}x{live.HEIGHT}."
            )

        if (
            not np.isfinite(actual_fps)
            or abs(actual_fps - live.FPS) > 0.5
        ):
            raise RuntimeError(
                f"Unexpected camera FPS: {actual_fps:.1f}; "
                f"required {live.FPS}."
            )

        print(
            f"Camera opened at {actual_width}x{actual_height}, "
            f"reported {actual_fps:.1f} FPS.",
            flush=True,
        )
        print(f"Frozen YOLO model: {live.MODEL_FILE}", flush=True)
        print(
            f"Frozen mapping source: {FINAL_MAPPING_SOURCE}",
            flush=True,
        )
        print(
            f"Frozen mapping model: {FINAL_MAPPING_MODEL_FILE}",
            flush=True,
        )
        print(
            "Place the cube at benchmark location 1. "
            "Press S only after LOCKED.",
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
                samples.clear()
                median_center = None
                standard_deviation = None
                locked = False
            else:
                samples.append(selected.center.copy())
                sample_array = np.asarray(samples, dtype=np.float64)
                median_center = np.median(sample_array, axis=0)
                standard_deviation = sample_array.std(axis=0)
                locked = (
                    len(samples) >= live.MIN_SAMPLES
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
                samples,
                median_center,
                standard_deviation,
                locked,
            )
            draw_dry_run_status(
                display_frame,
                location_number,
                last_mapping_status,
            )
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key not in (ord("s"), ord("S")):
                continue

            if (
                not locked
                or selected is None
                or median_center is None
                or standard_deviation is None
            ):
                print(
                    f"Location {location_number} is not LOCKED; "
                    "mapping not performed.",
                    flush=True,
                )
                continue

            record = map_locked_center(
                location_number,
                median_center,
                standard_deviation,
                selected.confidence,
                mapping,
                mapping_model,
                source_anchors,
            )
            records.append(record)
            last_mapping_status = (
                "Mapping: INSIDE validated workspace"
                if record["inside_workspace"]
                else "Mapping: OUTSIDE validated workspace"
            )

            if location_number >= LOCATION_COUNT:
                completed = True
                break

            location_number += 1
            samples.clear()
            last_mapping_status = "Mapping: waiting for LOCKED target"
            print(
                f"Move the cube to benchmark location {location_number}. "
                "Press S only after LOCKED.",
                flush=True,
            )

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.", flush=True)

    print("YOLO_MAPPING_DRY_RUN_RESULTS", flush=True)

    for record in records:
        center = record["center"]
        standard_deviation = record["standard_deviation"]
        print(
            f"location={record['location']} "
            f"median=({center[0]:.3f}, {center[1]:.3f}) "
            f"std=({standard_deviation[0]:.3f}, "
            f"{standard_deviation[1]:.3f}) "
            f"confidence={record['confidence']:.3f} "
            f"inside_workspace="
            f"{'YES' if record['inside_workspace'] else 'NO'} "
            f"triangle={record['triangle']} "
            f"source_mode={record['source_mode']}",
            flush=True,
        )

    if completed:
        print("YOLO_MAPPING_DRY_RUN_COMPLETE", flush=True)
    else:
        print(
            f"YOLO_MAPPING_DRY_RUN_INCOMPLETE: {len(records)}/"
            f"{LOCATION_COUNT} locations recorded.",
            flush=True,
        )


if __name__ == "__main__":
    main()
