from collections import deque
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


LOCATION_COUNT = 5
SCRIPT_DIR = Path(__file__).resolve().parent
LIVE_DETECTOR_FILE = SCRIPT_DIR / "03_live_cube_detection.py"
WINDOW_NAME = "Five-Location Cube Stability Check"


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


def draw_benchmark_status(
    frame: np.ndarray,
    location_number: int,
    false_positive_seen: bool,
) -> None:
    lines = (
        f"Benchmark location: {location_number}/{LOCATION_COUNT}",
        (
            "Extra detection seen at this location: "
            f"{'YES' if false_positive_seen else 'NO'}"
        ),
        "S: record only when LOCKED   Q: quit",
    )

    for index, line in enumerate(lines):
        y_position = frame.shape[0] - 76 + index * 27
        cv2.putText(
            frame,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (18, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def print_record(record: dict) -> None:
    print(
        "RECORDED "
        f"location={record['location']} "
        f"median=({record['median_x']:.3f}, "
        f"{record['median_y']:.3f}) "
        f"std=({record['std_x']:.3f}, "
        f"{record['std_y']:.3f}) "
        f"confidence={record['confidence']:.3f} "
        "false_positive="
        f"{'YES' if record['false_positive'] else 'NO'}",
        flush=True,
    )


def main() -> None:
    live = load_live_detector_module()
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
    records = []
    location_number = 1
    false_positive_seen = False
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
                false_positive_seen = True

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
            draw_benchmark_status(
                display_frame,
                location_number,
                false_positive_seen,
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

            record = {
                "location": location_number,
                "median_x": float(median_center[0]),
                "median_y": float(median_center[1]),
                "std_x": float(standard_deviation[0]),
                "std_y": float(standard_deviation[1]),
                "confidence": float(selected.confidence),
                "false_positive": bool(false_positive_seen),
            }
            records.append(record)
            print_record(record)

            if location_number >= LOCATION_COUNT:
                completed = True
                break

            location_number += 1
            samples.clear()
            false_positive_seen = False
            print(
                f"Move the cube to benchmark location {location_number}. "
                "Press S only after LOCKED.",
                flush=True,
            )

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.", flush=True)

    print("FIVE_LOCATION_RESULTS", flush=True)

    for record in records:
        print_record(record)

    if completed:
        print("FIVE_LOCATION_CHECK_COMPLETE", flush=True)
    else:
        print(
            f"FIVE_LOCATION_CHECK_INCOMPLETE: {len(records)}/"
            f"{LOCATION_COUNT} locations recorded.",
            flush=True,
        )


if __name__ == "__main__":
    main()
