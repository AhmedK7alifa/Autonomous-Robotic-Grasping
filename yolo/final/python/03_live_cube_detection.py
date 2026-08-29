from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30

CUBE_CLASS_ID = 0
INFERENCE_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
DUPLICATE_IOU_THRESHOLD = 0.50

ROLLING_BUFFER_SIZE = 30
MIN_SAMPLES = 20
MAX_STD_PX = 2.0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_FILE = (
    SCRIPT_DIR
    / "runs"
    / "cube_yolo26n_run1"
    / "weights"
    / "best.pt"
)

CALIBRATION_FILE = (
    PROJECT_ROOT
    / "04_calibration"
    / "camera_intrinsics"
    / "camera_calibration.npz"
)

WINDOW_NAME = "Live Cube Detection - Undistorted 1280x720"


@dataclass(frozen=True)
class CubeDetection:
    box: tuple[float, float, float, float]
    confidence: float
    center: np.ndarray


def box_iou(
    first_box: tuple[float, float, float, float],
    second_box: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_width = max(
        0.0,
        min(first_x2, second_x2) - max(first_x1, second_x1),
    )
    intersection_height = max(
        0.0,
        min(first_y2, second_y2) - max(first_y1, second_y1),
    )
    intersection_area = intersection_width * intersection_height
    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def suppress_duplicate_detections(
    detections: list[CubeDetection],
) -> list[CubeDetection]:
    kept_detections = []

    for detection in sorted(
        detections,
        key=lambda candidate: candidate.confidence,
        reverse=True,
    ):
        if all(
            box_iou(detection.box, kept.box)
            < DUPLICATE_IOU_THRESHOLD
            for kept in kept_detections
        ):
            kept_detections.append(detection)

    return kept_detections


def load_camera_calibration() -> tuple[np.ndarray, np.ndarray]:
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Camera calibration file not found: {CALIBRATION_FILE}"
        )

    calibration = np.load(CALIBRATION_FILE, allow_pickle=False)
    required_keys = {
        "camera_matrix",
        "distortion_coefficients",
        "image_width",
        "image_height",
    }
    missing_keys = required_keys.difference(calibration.files)

    if missing_keys:
        raise KeyError(
            "Camera calibration is missing keys: "
            f"{sorted(missing_keys)}"
        )

    calibration_width = int(calibration["image_width"].item())
    calibration_height = int(calibration["image_height"].item())

    if calibration_width != WIDTH or calibration_height != HEIGHT:
        raise RuntimeError(
            "Calibration resolution does not match the required "
            f"{WIDTH}x{HEIGHT}: "
            f"{calibration_width}x{calibration_height}"
        )

    camera_matrix = np.asarray(
        calibration["camera_matrix"],
        dtype=np.float64,
    )
    distortion = np.asarray(
        calibration["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(-1)

    return camera_matrix, distortion


def load_frozen_model() -> YOLO:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Frozen YOLO model not found: {MODEL_FILE}")

    model = YOLO(str(MODEL_FILE))
    class_name = model.names[CUBE_CLASS_ID]

    if class_name != "cube":
        raise RuntimeError(
            "Frozen model class 0 is not named 'cube': "
            f"{class_name!r}"
        )

    return model


def extract_cube_detections(result) -> list[CubeDetection]:
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy()
    detections = []

    for coordinates_row, confidence, class_id in zip(
        coordinates,
        confidences,
        class_ids,
    ):
        if int(round(float(class_id))) != CUBE_CLASS_ID:
            continue

        values = np.asarray(coordinates_row, dtype=np.float64)

        if values.shape != (4,) or not np.isfinite(values).all():
            continue

        x1, y1, x2, y2 = values
        x1 = float(np.clip(x1, 0.0, WIDTH - 1.0))
        y1 = float(np.clip(y1, 0.0, HEIGHT - 1.0))
        x2 = float(np.clip(x2, 0.0, WIDTH - 1.0))
        y2 = float(np.clip(y2, 0.0, HEIGHT - 1.0))
        confidence = float(confidence)

        if (
            x2 <= x1
            or y2 <= y1
            or not np.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            continue

        center = np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            dtype=np.float64,
        )
        detections.append(
            CubeDetection(
                box=(x1, y1, x2, y2),
                confidence=confidence,
                center=center,
            )
        )

    return suppress_duplicate_detections(detections)


def draw_detection(
    frame: np.ndarray,
    detection: CubeDetection,
    selected: bool,
) -> None:
    x1, y1, x2, y2 = detection.box
    top_left = (int(round(x1)), int(round(y1)))
    bottom_right = (int(round(x2)), int(round(y2)))
    center_point = tuple(
        int(value) for value in np.rint(detection.center)
    )
    color = (0, 255, 0) if selected else (0, 200, 255)
    thickness = 3 if selected else 2
    prefix = "TARGET" if selected else "cube"
    label = (
        f"{prefix} {detection.confidence:.2f}  "
        f"({detection.center[0]:.1f}, {detection.center[1]:.1f})"
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        center_point,
        5,
        color,
        -1,
        cv2.LINE_AA,
    )

    label_y = max(24, top_left[1] - 8)
    cv2.putText(
        frame,
        label,
        (top_left[0], label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (top_left[0], label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status(
    frame: np.ndarray,
    detections: list[CubeDetection],
    selected: CubeDetection | None,
    samples: deque,
    median_center: np.ndarray | None,
    standard_deviation: np.ndarray | None,
    locked: bool,
) -> None:
    state = "LOCKED" if locked else "UNSTABLE"
    state_color = (0, 255, 0) if locked else (0, 128, 255)

    if selected is None:
        target_line = "Target: no cube detected (buffer cleared)"
    else:
        target_line = (
            "Target center: "
            f"({selected.center[0]:.1f}, {selected.center[1]:.1f}) px  "
            f"confidence: {selected.confidence:.2f}"
        )

    if median_center is None or standard_deviation is None:
        stability_line = (
            f"Samples: 0/{ROLLING_BUFFER_SIZE}  "
            "Median: --  Std: --"
        )
    else:
        stability_line = (
            f"Samples: {len(samples)}/{ROLLING_BUFFER_SIZE}  "
            f"Median: ({median_center[0]:.1f}, {median_center[1]:.1f})  "
            "Std: "
            f"({standard_deviation[0]:.2f}, "
            f"{standard_deviation[1]:.2f}) px"
        )

    lines = [
        (f"State: {state}", state_color),
        (target_line, (255, 255, 255)),
        (stability_line, (255, 255, 255)),
        (
            f"Cube detections: {len(detections)}  "
            "Selected: highest confidence",
            (255, 255, 255),
        ),
        ("Q: quit", (255, 255, 255)),
    ]

    for index, (line, color) in enumerate(lines):
        y_position = 32 + index * 29
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
            color,
            2,
            cv2.LINE_AA,
        )

    if median_center is not None:
        median_point = tuple(
            int(value) for value in np.rint(median_center)
        )
        cv2.drawMarker(
            frame,
            median_point,
            state_color,
            cv2.MARKER_CROSS,
            22,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    model = load_frozen_model()
    camera_matrix, distortion = load_camera_calibration()
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (WIDTH, HEIGHT),
        cv2.CV_32FC1,
    )
    samples = deque(maxlen=ROLLING_BUFFER_SIZE)
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, FPS)

        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index {CAMERA_INDEX} with DirectShow."
            )

        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(camera.get(cv2.CAP_PROP_FPS))

        if actual_width != WIDTH or actual_height != HEIGHT:
            raise RuntimeError(
                "Unexpected camera resolution: "
                f"{actual_width}x{actual_height}; "
                f"required {WIDTH}x{HEIGHT}."
            )

        if not np.isfinite(actual_fps) or abs(actual_fps - FPS) > 0.5:
            raise RuntimeError(
                f"Unexpected camera FPS: {actual_fps:.1f}; required {FPS}."
            )

        print(
            f"Camera opened at {actual_width}x{actual_height}, "
            f"reported {actual_fps:.1f} FPS."
        )
        print(f"Frozen model: {MODEL_FILE}")
        print("Camera-only inference started. Press Q to quit.")

        while True:
            success, raw_frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a camera frame.")

            if raw_frame.shape[:2] != (HEIGHT, WIDTH):
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
                imgsz=INFERENCE_IMAGE_SIZE,
                device="cpu",
                classes=[CUBE_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )[0]
            detections = extract_cube_detections(result)
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
                    len(samples) >= MIN_SAMPLES
                    and float(standard_deviation.max()) <= MAX_STD_PX
                )

            display_frame = frame.copy()

            for detection in detections:
                draw_detection(
                    display_frame,
                    detection,
                    selected=detection is selected,
                )

            draw_status(
                display_frame,
                detections,
                selected,
                samples,
                median_center,
                standard_deviation,
                locked,
            )
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.")


if __name__ == "__main__":
    main()
