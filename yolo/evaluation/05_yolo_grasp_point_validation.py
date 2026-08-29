from collections import deque
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


LOCATION_COUNT = 5

SCRIPT_DIR = Path(__file__).resolve().parent
LIVE_DETECTOR_FILE = SCRIPT_DIR / "03_live_cube_detection.py"
MAPPING_DRY_RUN_FILE = SCRIPT_DIR / "04_yolo_mapping_dry_run.py"
WINDOW_NAME = "YOLO Grasp-Point Validation - Camera Only"

# These are validation candidates, not frozen grasp-point settings.
# Fractions are measured down from the median bounding-box top edge.
CANDIDATE_RULES = (
    {
        "key": "A",
        "name": "bbox_center",
        "height_fraction": None,
        "color": (0, 255, 255),
    },
    {
        "key": "B",
        "name": "upper_quarter",
        "height_fraction": 0.25,
        "color": (255, 0, 255),
    },
    {
        "key": "C",
        "name": "upper_third",
        "height_fraction": 1.0 / 3.0,
        "color": (255, 255, 0),
    },
)


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


def candidate_points(
    median_center: np.ndarray,
    median_box: np.ndarray,
) -> list[dict]:
    center = np.asarray(median_center, dtype=np.float64)
    box = np.asarray(median_box, dtype=np.float64)

    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Median center must contain two finite values.")

    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Median box must contain four finite values.")

    x1, y1, x2, y2 = box
    box_height = float(y2 - y1)

    if x2 <= x1 or box_height <= 0.0:
        raise ValueError("Median bounding box has invalid geometry.")

    points = []

    for rule in CANDIDATE_RULES:
        fraction = rule["height_fraction"]

        if fraction is None:
            y_coordinate = float(center[1])
        else:
            y_coordinate = float(y1 + float(fraction) * box_height)

        point = np.array(
            [float(center[0]), y_coordinate],
            dtype=np.float64,
        )
        points.append(
            {
                **rule,
                "point": point,
                "vertical_difference": float(point[1] - center[1]),
                "box_height": box_height,
            }
        )

    return points


def evaluate_candidates(
    median_center: np.ndarray,
    median_box: np.ndarray,
    mapping,
    mapping_model: dict,
    source_anchors: list[dict],
) -> list[dict]:
    evaluations = []

    for candidate in candidate_points(median_center, median_box):
        prediction = mapping.predict(candidate["point"], mapping_model)
        evaluation = dict(candidate)
        evaluation["inside_workspace"] = prediction is not None
        evaluation["triangle_index"] = None
        evaluation["triangle_ids"] = None
        evaluation["source_mode"] = None
        evaluation["corridor_t"] = None
        evaluation["corridor_perpendicular_px"] = None

        if prediction is not None:
            dynamic = mapping.dynamic_source_poses(
                candidate["point"],
                prediction,
                source_anchors,
            )
            evaluation["triangle_index"] = (
                int(prediction["triangle_index"]) + 1
            )
            evaluation["triangle_ids"] = list(
                prediction["triangle_ids"]
            )
            evaluation["source_mode"] = str(dynamic["source_mode"])
            evaluation["corridor_t"] = dynamic["right_pairwise_t"]
            evaluation["corridor_perpendicular_px"] = dynamic[
                "right_pairwise_perp_px"
            ]

        evaluations.append(evaluation)

    return evaluations


def print_location_record(record: dict) -> None:
    print("\n" + "=" * 72, flush=True)
    print(
        f"YOLO GRASP-POINT VALIDATION - LOCATION {record['location']}",
        flush=True,
    )
    print("=" * 72, flush=True)
    center = record["median_center"]
    median_box = record["median_box"]
    standard_deviation = record["standard_deviation"]
    print(
        "Locked bbox center: "
        f"({center[0]:.3f}, {center[1]:.3f}) px",
        flush=True,
    )
    print(
        "Median bbox: "
        f"({median_box[0]:.3f}, {median_box[1]:.3f}, "
        f"{median_box[2]:.3f}, {median_box[3]:.3f})",
        flush=True,
    )
    print(
        "Center stability: "
        f"std_x={standard_deviation[0]:.3f}px, "
        f"std_y={standard_deviation[1]:.3f}px",
        flush=True,
    )

    for evaluation in record["evaluations"]:
        point = evaluation["point"]
        print(
            f"{evaluation['key']} {evaluation['name']}: "
            f"point=({point[0]:.3f}, {point[1]:.3f})px, "
            f"delta_y={evaluation['vertical_difference']:.3f}px, "
            "inside_workspace="
            f"{'YES' if evaluation['inside_workspace'] else 'NO'}",
            flush=True,
        )

        if not evaluation["inside_workspace"]:
            print("  triangle=NONE, correction_mode=NONE", flush=True)
            continue

        print(
            f"  triangle={evaluation['triangle_index']}/8 "
            + " / ".join(evaluation["triangle_ids"]),
            flush=True,
        )
        print(
            f"  correction_mode={evaluation['source_mode']}",
            flush=True,
        )

        if evaluation["source_mode"] == "right_pairwise_linear":
            print(
                "  right_pair_corridor="
                f"t={evaluation['corridor_t']:.6f}, "
                "perpendicular_distance="
                f"{evaluation['corridor_perpendicular_px']:.3f}px",
                flush=True,
            )


def draw_candidate_points(
    frame: np.ndarray,
    evaluations: list[dict],
) -> None:
    for evaluation in evaluations:
        point = tuple(
            int(value) for value in np.rint(evaluation["point"])
        )
        color = evaluation["color"]
        cv2.drawMarker(
            frame,
            point,
            color,
            cv2.MARKER_CROSS,
            20,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            evaluation["key"],
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            evaluation["key"],
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_validation_status(
    frame: np.ndarray,
    location_number: int,
    evaluations: list[dict],
) -> None:
    lines = [
        f"Grasp-point location: {location_number}/{LOCATION_COUNT}",
    ]

    if evaluations:
        for evaluation in evaluations:
            point = evaluation["point"]
            mapping_status = (
                f"IN T{evaluation['triangle_index']} "
                f"{evaluation['source_mode']}"
                if evaluation["inside_workspace"]
                else "OUTSIDE"
            )
            lines.append(
                f"{evaluation['key']} {evaluation['name']}: "
                f"({point[0]:.1f}, {point[1]:.1f}) {mapping_status}"
            )
    else:
        lines.append("Candidates: waiting for LOCKED median box")

    lines.extend(
        [
            "S: record candidates only when LOCKED   Q: quit",
            "VALIDATION ONLY - NO RULE FROZEN - NO ROBOT OUTPUT",
        ]
    )
    start_y = frame.shape[0] - 27 * len(lines) - 8

    for index, line in enumerate(lines):
        y_position = start_y + index * 27
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
            0.56,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color,
            2,
            cv2.LINE_AA,
        )


def print_final_summary(records: list[dict]) -> None:
    print("GRASP_POINT_VALIDATION_RESULTS", flush=True)

    for rule in CANDIDATE_RULES:
        key = rule["key"]
        evaluations = [
            next(
                evaluation
                for evaluation in record["evaluations"]
                if evaluation["key"] == key
            )
            for record in records
        ]
        successful = sum(
            evaluation["inside_workspace"]
            for evaluation in evaluations
        )
        differences = [
            evaluation["vertical_difference"]
            for evaluation in evaluations
        ]
        print(
            f"rule={key}_{rule['name']} "
            f"mapped={successful}/{len(records)} "
            "vertical_differences_px="
            + str([round(value, 3) for value in differences]),
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
            "Candidate A: locked bbox geometric center.",
            flush=True,
        )
        print(
            "Candidate B: horizontal center, 1/4 box height from top.",
            flush=True,
        )
        print(
            "Candidate C: horizontal center, 1/3 box height from top.",
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
                median_center = None
                median_box = None
                standard_deviation = None
                locked = False
                evaluations = []
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
                locked = (
                    len(center_samples) >= live.MIN_SAMPLES
                    and float(standard_deviation.max())
                    <= live.MAX_STD_PX
                )
                evaluations = (
                    evaluate_candidates(
                        median_center,
                        median_box,
                        mapping,
                        mapping_model,
                        source_anchors,
                    )
                    if locked
                    else []
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
                locked,
            )

            if evaluations:
                draw_candidate_points(display_frame, evaluations)

            draw_validation_status(
                display_frame,
                location_number,
                evaluations,
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
                or median_box is None
                or standard_deviation is None
                or not evaluations
            ):
                print(
                    f"Location {location_number} is not LOCKED; "
                    "candidate points not recorded.",
                    flush=True,
                )
                continue

            record = {
                "location": location_number,
                "median_center": median_center.copy(),
                "median_box": median_box.copy(),
                "standard_deviation": standard_deviation.copy(),
                "confidence": float(selected.confidence),
                "evaluations": evaluations,
            }
            records.append(record)
            print_location_record(record)

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
        print("GRASP_POINT_VALIDATION_COMPLETE", flush=True)
    else:
        print(
            f"GRASP_POINT_VALIDATION_INCOMPLETE: {len(records)}/"
            f"{LOCATION_COUNT} locations recorded.",
            flush=True,
        )


if __name__ == "__main__":
    main()
