from collections import deque
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


LOCATION_COUNT = 5
FINAL_GRASP_HEIGHT_FRACTION = 0.25

SCRIPT_DIR = Path(__file__).resolve().parent
LIVE_DETECTOR_FILE = SCRIPT_DIR / "perception.py"
MAPPING_DRY_RUN_FILE = SCRIPT_DIR / "mapping_interface.py"
WINDOW_NAME = "FINAL YOLO Perception-to-Mapping - Camera Only"


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


def final_grasp_point(
    median_bbox_center: np.ndarray,
    median_bbox: np.ndarray,
) -> np.ndarray:
    """Apply the frozen global Candidate-B rule."""
    center = np.asarray(median_bbox_center, dtype=np.float64)
    box = np.asarray(median_bbox, dtype=np.float64)

    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Median bbox center must contain two finite values.")

    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Median bbox must contain four finite values.")

    x1, y1, x2, y2 = box
    bbox_height = float(y2 - y1)

    if x2 <= x1 or bbox_height <= 0.0:
        raise ValueError("Median bbox has invalid geometry.")

    return np.array(
        [
            float(center[0]),
            float(y1 + FINAL_GRASP_HEIGHT_FRACTION * bbox_height),
        ],
        dtype=np.float64,
    )


def map_grasp_point(
    point: np.ndarray,
    mapping,
    mapping_model: dict,
    source_anchors: list[dict],
) -> dict:
    prediction = mapping.predict(point, mapping_model)

    if prediction is None:
        return {
            "inside_workspace": False,
            "triangle_index": None,
            "triangle_ids": None,
            "source_mode": None,
            "above": None,
            "grasp": None,
            "support": None,
            "corridor_t": None,
            "corridor_perpendicular_px": None,
        }

    dynamic = mapping.dynamic_source_poses(
        point,
        prediction,
        source_anchors,
    )
    return {
        "inside_workspace": True,
        "triangle_index": int(prediction["triangle_index"]) + 1,
        "triangle_ids": list(prediction["triangle_ids"]),
        "source_mode": str(dynamic["source_mode"]),
        "above": dict(dynamic["above"]),
        "grasp": dict(dynamic["grasp"]),
        "support": list(dynamic["support"]),
        "corridor_t": dynamic["right_pairwise_t"],
        "corridor_perpendicular_px": dynamic[
            "right_pairwise_perp_px"
        ],
    }


def print_joint_pose(title: str, pose: dict, joint_names: list[str]) -> None:
    print(title, flush=True)

    for joint in joint_names:
        print(f"  {joint}: {float(pose[joint]):.4f}", flush=True)


def print_location_record(record: dict, joint_names: list[str]) -> None:
    center = record["median_bbox_center"]
    point = record["grasp_point"]
    standard_deviation = record["standard_deviation"]
    result = record["mapping_result"]
    print("\n" + "=" * 72, flush=True)
    print(
        f"FINAL YOLO PERCEPTION-TO-MAPPING - LOCATION "
        f"{record['location']}",
        flush=True,
    )
    print("=" * 72, flush=True)
    print(
        "Locked bbox center: "
        f"({center[0]:.3f}, {center[1]:.3f}) px",
        flush=True,
    )
    print(
        "Final grasp reference point: "
        f"({point[0]:.3f}, {point[1]:.3f}) px",
        flush=True,
    )
    print(
        "Grasp-point vertical difference: "
        f"{point[1] - center[1]:.3f} px",
        flush=True,
    )
    print(
        "Center stability: "
        f"std_x={standard_deviation[0]:.3f}px, "
        f"std_y={standard_deviation[1]:.3f}px",
        flush=True,
    )
    print(
        "Inside validated mapping mesh: "
        f"{'YES' if result['inside_workspace'] else 'NO'}",
        flush=True,
    )

    if not result["inside_workspace"]:
        print("Mapping triangle: NONE", flush=True)
        print("Correction mode: NONE", flush=True)
        print("Mapped targets: NOT GENERATED", flush=True)
        return

    print(
        f"Mapping triangle: {result['triangle_index']}/8 - "
        + " / ".join(result["triangle_ids"]),
        flush=True,
    )
    print(f"Correction mode: {result['source_mode']}", flush=True)

    if result["source_mode"] == "right_pairwise_linear":
        print(
            "Right-pair corridor: "
            f"t={result['corridor_t']:.6f}, "
            "perpendicular_distance="
            f"{result['corridor_perpendicular_px']:.3f}px",
            flush=True,
        )

    print("Dynamic correction support:", flush=True)

    for item in result["support"]:
        print(
            f"  {item['name']}: "
            f"distance={item['distance_px']:.3f}px, "
            f"weight={item['weight']:.6f}",
            flush=True,
        )

    print_joint_pose(
        "Final mapped ABOVE target:",
        result["above"],
        joint_names,
    )
    print_joint_pose(
        "Final mapped OPEN-GRASP target:",
        result["grasp"],
        joint_names,
    )


def compact_pose(pose: dict | None) -> str:
    if pose is None:
        return "NOT GENERATED"

    return (
        f"pan={pose['shoulder_pan.pos']:.2f} "
        f"lift={pose['shoulder_lift.pos']:.2f} "
        f"elbow={pose['elbow_flex.pos']:.2f} "
        f"wrist={pose['wrist_flex.pos']:.2f} "
        f"roll={pose['wrist_roll.pos']:.2f} "
        f"grip={pose['gripper.pos']:.2f}"
    )


def draw_final_grasp_point(
    frame: np.ndarray,
    median_bbox_center: np.ndarray,
    grasp_point: np.ndarray,
) -> None:
    center_pixel = tuple(
        int(value) for value in np.rint(median_bbox_center)
    )
    grasp_pixel = tuple(int(value) for value in np.rint(grasp_point))
    cv2.drawMarker(
        frame,
        center_pixel,
        (0, 255, 255),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        frame,
        grasp_pixel,
        (255, 0, 255),
        cv2.MARKER_CROSS,
        24,
        3,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        center_pixel,
        grasp_pixel,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "FINAL GRASP POINT",
        (grasp_pixel[0] + 8, grasp_pixel[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "FINAL GRASP POINT",
        (grasp_pixel[0] + 8, grasp_pixel[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_pipeline_status(
    frame: np.ndarray,
    location_number: int,
    median_bbox_center: np.ndarray | None,
    grasp_point: np.ndarray | None,
    mapping_result: dict | None,
) -> None:
    lines = [
        f"Verification location: {location_number}/{LOCATION_COUNT}",
    ]

    if (
        median_bbox_center is None
        or grasp_point is None
        or mapping_result is None
    ):
        lines.append("Final grasp point: waiting for LOCKED median bbox")
    else:
        lines.append(
            "BBox center: "
            f"({median_bbox_center[0]:.1f}, "
            f"{median_bbox_center[1]:.1f})"
        )
        lines.append(
            "Final grasp point: "
            f"({grasp_point[0]:.1f}, {grasp_point[1]:.1f})"
        )

        if mapping_result["inside_workspace"]:
            lines.append(
                f"Mapping: INSIDE T{mapping_result['triangle_index']} "
                f"{mapping_result['source_mode']}"
            )
        else:
            lines.append("Mapping: OUTSIDE - NO TARGET GENERATED")

        lines.append(
            "ABOVE: " + compact_pose(mapping_result["above"])
        )
        lines.append(
            "OPEN-GRASP: " + compact_pose(mapping_result["grasp"])
        )

    lines.extend(
        [
            "S: record only when LOCKED   Q: quit",
            "CAMERA + MAPPING ONLY - NO ROBOT OUTPUT",
        ]
    )
    start_y = frame.shape[0] - 25 * len(lines) - 8

    for index, line in enumerate(lines):
        y_position = start_y + index * 25
        color = (
            (0, 255, 255)
            if index == len(lines) - 1
            else (255, 255, 255)
        )
        cv2.putText(
            frame,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def print_final_summary(records: list[dict]) -> None:
    print("FINAL_PERCEPTION_MAPPING_RESULTS", flush=True)

    for record in records:
        center = record["median_bbox_center"]
        point = record["grasp_point"]
        result = record["mapping_result"]
        print(
            f"location={record['location']} "
            f"locked=YES "
            f"bbox_center=({center[0]:.3f}, {center[1]:.3f}) "
            f"grasp_point=({point[0]:.3f}, {point[1]:.3f}) "
            f"inside_mesh="
            f"{'YES' if result['inside_workspace'] else 'NO'} "
            f"triangle={result['triangle_ids']} "
            f"correction_mode={result['source_mode']}",
            flush=True,
        )


def main() -> None:
    live = load_local_module("cube_live_detector", LIVE_DETECTOR_FILE)
    dry_run = load_local_module("yolo_mapping_dry_run", MAPPING_DRY_RUN_FILE)
    mapping = dry_run.load_final_mapping_functions()
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
    center_samples = deque(maxlen=live.ROLLING_BUFFER_SIZE)
    box_samples = deque(maxlen=live.ROLLING_BUFFER_SIZE)
    records = []
    location_number = 1
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
            "Frozen global grasp rule: "
            "x=median_bbox_center_x, "
            "y=median_bbox_top+0.25*median_bbox_height",
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
                center_samples.clear()
                box_samples.clear()
                median_bbox_center = None
                median_bbox = None
                standard_deviation = None
                grasp_point = None
                mapping_result = None
                locked = False
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
                median_bbox_center = np.median(center_array, axis=0)
                median_bbox = np.median(box_array, axis=0)
                standard_deviation = center_array.std(axis=0)
                locked = (
                    len(center_samples) >= live.MIN_SAMPLES
                    and float(standard_deviation.max())
                    <= live.MAX_STD_PX
                )

                if locked:
                    grasp_point = final_grasp_point(
                        median_bbox_center,
                        median_bbox,
                    )
                    mapping_result = map_grasp_point(
                        grasp_point,
                        mapping,
                        mapping_model,
                        source_anchors,
                    )
                else:
                    grasp_point = None
                    mapping_result = None

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
                median_bbox_center,
                standard_deviation,
                locked,
            )

            if (
                median_bbox_center is not None
                and grasp_point is not None
            ):
                draw_final_grasp_point(
                    display_frame,
                    median_bbox_center,
                    grasp_point,
                )

            draw_pipeline_status(
                display_frame,
                location_number,
                median_bbox_center if locked else None,
                grasp_point,
                mapping_result,
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
                or median_bbox_center is None
                or median_bbox is None
                or standard_deviation is None
                or grasp_point is None
                or mapping_result is None
            ):
                print(
                    f"Location {location_number} is not LOCKED; "
                    "nothing recorded.",
                    flush=True,
                )
                continue

            record = {
                "location": location_number,
                "median_bbox_center": median_bbox_center.copy(),
                "median_bbox": median_bbox.copy(),
                "standard_deviation": standard_deviation.copy(),
                "confidence": float(selected.confidence),
                "grasp_point": grasp_point.copy(),
                "mapping_result": mapping_result,
            }
            records.append(record)
            print_location_record(record, mapping.JOINTS)

            if location_number >= LOCATION_COUNT:
                completed = True
                break

            location_number += 1
            center_samples.clear()
            box_samples.clear()
            print(
                f"Move the cube to benchmark location {location_number}. "
                "Press S only after LOCKED.",
                flush=True,
            )

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.", flush=True)

    print_final_summary(records)

    if completed:
        print("FINAL_PERCEPTION_MAPPING_CHECK_COMPLETE", flush=True)
    else:
        print(
            f"FINAL_PERCEPTION_MAPPING_CHECK_INCOMPLETE: {len(records)}/"
            f"{LOCATION_COUNT} locations recorded.",
            flush=True,
        )


if __name__ == "__main__":
    main()
