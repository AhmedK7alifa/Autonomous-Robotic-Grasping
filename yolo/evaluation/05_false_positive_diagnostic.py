from collections import deque
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np


LOCATION_COUNT = 5
DUPLICATE_IOU_THRESHOLD = 0.50

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LIVE_DETECTOR_FILE = SCRIPT_DIR / "03_live_cube_detection.py"
WORKSPACE_BOUNDS_FILE = (
    PROJECT_ROOT
    / "06_pixel_to_robot"
    / "workspace_bounds_day07.json"
)
WINDOW_NAME = "Cube False-Positive Diagnostic"


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


def load_workspace_polygon() -> np.ndarray:
    if not WORKSPACE_BOUNDS_FILE.exists():
        raise FileNotFoundError(
            f"Workspace bounds file not found: {WORKSPACE_BOUNDS_FILE}"
        )

    with WORKSPACE_BOUNDS_FILE.open("r", encoding="utf-8") as stream:
        workspace_data = json.load(stream)

    polygon = np.asarray(
        workspace_data["safe_workspace_polygon_px"],
        dtype=np.float32,
    )

    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError(
            "Safe workspace polygon must contain four finite 2D vertices."
        )

    return polygon


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection_area = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return float(intersection_area / union_area)


def center_inside_workspace(
    center: np.ndarray,
    workspace_polygon: np.ndarray,
) -> bool:
    result = cv2.pointPolygonTest(
        workspace_polygon,
        (float(center[0]), float(center[1])),
        False,
    )
    return result >= 0.0


def diagnostic_classification(iou: float, inside_workspace: bool) -> str:
    if iou >= DUPLICATE_IOU_THRESHOLD:
        return "A_DUPLICATE_OVERLAP"

    if inside_workspace:
        return "B_SEPARATE_INSIDE"

    return "C_SEPARATE_OUTSIDE"


def detection_payload(
    detection,
    workspace_polygon: np.ndarray,
) -> dict:
    return {
        "box": [round(float(value), 3) for value in detection.box],
        "center": [
            round(float(detection.center[0]), 3),
            round(float(detection.center[1]), 3),
        ],
        "confidence": round(float(detection.confidence), 4),
        "inside_workspace": center_inside_workspace(
            detection.center,
            workspace_polygon,
        ),
    }


def log_multi_detection_frame(
    location_number: int,
    frame_number: int,
    episode_number: int,
    consecutive_frame_count: int,
    detections,
    workspace_polygon: np.ndarray,
) -> list[dict]:
    target = detections[0]
    target_payload = detection_payload(target, workspace_polygon)
    extras = []

    for extra_index, extra in enumerate(detections[1:], start=1):
        iou = box_iou(target.box, extra.box)
        extra_payload = detection_payload(extra, workspace_polygon)
        extra_payload.update(
            {
                "extra_index": extra_index,
                "iou_with_target": round(iou, 4),
                "substantial_overlap": iou >= DUPLICATE_IOU_THRESHOLD,
                "spatial_relation": (
                    "overlapping"
                    if iou >= DUPLICATE_IOU_THRESHOLD
                    else "separate"
                ),
                "classification": diagnostic_classification(
                    iou,
                    extra_payload["inside_workspace"],
                ),
            }
        )
        extras.append(extra_payload)

    frame_payload = {
        "location": location_number,
        "frame": frame_number,
        "episode": episode_number,
        "consecutive_multi_frame": consecutive_frame_count,
        "target": target_payload,
        "extras": extras,
    }
    print(
        "MULTI_FRAME "
        + json.dumps(frame_payload, separators=(",", ":")),
        flush=True,
    )
    return extras


def draw_diagnostic_status(
    frame: np.ndarray,
    location_number: int,
    multi_frame_count: int,
    current_streak: int,
    classification_counts: dict,
    workspace_polygon: np.ndarray,
) -> None:
    polygon = np.rint(workspace_polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        frame,
        [polygon],
        True,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    lines = (
        f"Diagnostic location: {location_number}/{LOCATION_COUNT}",
        (
            f"Multi frames: {multi_frame_count}  "
            f"Current consecutive streak: {current_streak}"
        ),
        (
            "Extras A/B/C: "
            f"{classification_counts['A']}/"
            f"{classification_counts['B']}/"
            f"{classification_counts['C']}"
        ),
        "Magenta: fixed safe workspace polygon",
        "S: finish location only when LOCKED   Q: quit",
    )

    for index, line in enumerate(lines):
        y_position = frame.shape[0] - 130 + index * 27
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
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def finalize_episode(
    location_number: int,
    episode_number: int,
    frame_count: int,
    episode_lengths: list[int],
) -> None:
    if frame_count <= 0:
        return

    episode_lengths.append(frame_count)
    persistence = (
        "single-frame"
        if frame_count == 1
        else "repeated-consecutive"
    )
    print(
        "EPISODE_END "
        f"location={location_number} "
        f"episode={episode_number} "
        f"frames={frame_count} "
        f"persistence={persistence}",
        flush=True,
    )


def new_location_counts() -> dict:
    return {
        "A": 0,
        "B": 0,
        "C": 0,
    }


def main() -> None:
    live = load_live_detector_module()
    workspace_polygon = load_workspace_polygon()
    model = live.load_frozen_model()
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
    frame_number = 0
    multi_frame_count = 0
    current_streak = 0
    episode_number = 0
    episode_lengths = []
    classification_counts = new_location_counts()
    location_summaries = []
    all_multi_frame_records = []
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
        print(f"Frozen model: {live.MODEL_FILE}", flush=True)
        print(
            f"Confidence threshold: {live.CONFIDENCE_THRESHOLD:.2f}",
            flush=True,
        )
        print(
            "Workspace polygon: "
            + json.dumps(workspace_polygon.tolist()),
            flush=True,
        )
        print(
            "Place the cube at diagnostic location 1. "
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

            frame_number += 1
            frame = cv2.remap(
                raw_frame,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
            )
            result = model.predict(
                source=frame,
                imgsz=live.INFERENCE_IMAGE_SIZE,
                device="cpu",
                classes=[live.CUBE_CLASS_ID],
                conf=live.CONFIDENCE_THRESHOLD,
                verbose=False,
            )[0]
            detections = live.extract_cube_detections(result)
            selected = detections[0] if detections else None

            if len(detections) > 1:
                if current_streak == 0:
                    episode_number += 1

                current_streak += 1
                multi_frame_count += 1
                extras = log_multi_detection_frame(
                    location_number,
                    frame_number,
                    episode_number,
                    current_streak,
                    detections,
                    workspace_polygon,
                )
                frame_record = {
                    "location": location_number,
                    "frame": frame_number,
                    "episode": episode_number,
                    "streak": current_streak,
                    "extras": extras,
                }
                all_multi_frame_records.append(frame_record)

                for extra in extras:
                    classification = extra["classification"]

                    if classification == "A_DUPLICATE_OVERLAP":
                        classification_counts["A"] += 1
                    elif classification == "B_SEPARATE_INSIDE":
                        classification_counts["B"] += 1
                    else:
                        classification_counts["C"] += 1
            elif current_streak > 0:
                finalize_episode(
                    location_number,
                    episode_number,
                    current_streak,
                    episode_lengths,
                )
                current_streak = 0

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
            draw_diagnostic_status(
                display_frame,
                location_number,
                multi_frame_count,
                current_streak,
                classification_counts,
                workspace_polygon,
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
                    "nothing recorded.",
                    flush=True,
                )
                continue

            if current_streak > 0:
                finalize_episode(
                    location_number,
                    episode_number,
                    current_streak,
                    episode_lengths,
                )
                current_streak = 0

            location_summary = {
                "location": location_number,
                "median": [
                    round(float(median_center[0]), 3),
                    round(float(median_center[1]), 3),
                ],
                "std": [
                    round(float(standard_deviation[0]), 3),
                    round(float(standard_deviation[1]), 3),
                ],
                "target_confidence": round(
                    float(selected.confidence),
                    4,
                ),
                "multi_detection_frames": multi_frame_count,
                "episode_lengths": list(episode_lengths),
                "max_consecutive_frames": max(
                    episode_lengths,
                    default=0,
                ),
                "classification_counts": dict(classification_counts),
            }
            location_summaries.append(location_summary)
            print(
                "LOCATION_SUMMARY "
                + json.dumps(location_summary, separators=(",", ":")),
                flush=True,
            )

            if location_number >= LOCATION_COUNT:
                completed = True
                break

            location_number += 1
            samples.clear()
            multi_frame_count = 0
            current_streak = 0
            episode_number = 0
            episode_lengths = []
            classification_counts = new_location_counts()
            print(
                f"Move the cube to diagnostic location {location_number}. "
                "Press S only after LOCKED.",
                flush=True,
            )

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.", flush=True)

    total_classification_counts = new_location_counts()

    for summary in location_summaries:
        for category in total_classification_counts:
            total_classification_counts[category] += summary[
                "classification_counts"
            ][category]

    final_summary = {
        "complete": completed,
        "locations_recorded": len(location_summaries),
        "locations": location_summaries,
        "total_multi_detection_frames": len(all_multi_frame_records),
        "total_classification_counts": total_classification_counts,
    }
    print(
        "DIAGNOSTIC_RESULTS "
        + json.dumps(final_summary, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    main()
