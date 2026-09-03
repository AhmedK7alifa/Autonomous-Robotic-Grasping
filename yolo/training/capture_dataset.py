from pathlib import Path

import cv2
import numpy as np


CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CALIBRATION_FILE = (
    PROJECT_ROOT
    / "calibration"
    / "camera"
    / "camera_calibration.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_capture" / "images"
WINDOW_NAME = "Cube Dataset Capture - Undistorted 1280x720"


def load_camera_calibration():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Camera calibration file not found: {CALIBRATION_FILE}"
        )

    calibration = np.load(
        CALIBRATION_FILE,
        allow_pickle=False,
    )

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


def next_image_number(output_dir: Path) -> int:
    image_numbers = []

    for image_path in output_dir.glob("cube_*.jpg"):
        suffix = image_path.stem.removeprefix("cube_")

        if suffix.isdigit():
            image_numbers.append(int(suffix))

    return max(image_numbers, default=0) + 1


def count_saved_images(output_dir: Path) -> int:
    return sum(
        1
        for image_path in output_dir.glob("cube_*.jpg")
        if image_path.stem.removeprefix("cube_").isdigit()
    )


def draw_status(frame, saved_count: int) -> None:
    lines = (
        f"Saved images: {saved_count}",
        "S: save full undistorted frame",
        "Q: quit",
    )

    for line_number, line in enumerate(lines):
        y_position = 35 + line_number * 32

        cv2.putText(
            frame,
            line,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    camera_matrix, distortion = load_camera_calibration()

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (WIDTH, HEIGHT),
        cv2.CV_32FC1,
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, FPS)

        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index {CAMERA_INDEX}."
            )

        actual_width = int(
            camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        actual_height = int(
            camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        actual_fps = camera.get(cv2.CAP_PROP_FPS)

        if actual_width != WIDTH or actual_height != HEIGHT:
            raise RuntimeError(
                "Unexpected camera resolution: "
                f"{actual_width}x{actual_height}; "
                f"required {WIDTH}x{HEIGHT}."
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        image_number = next_image_number(OUTPUT_DIR)
        saved_count = count_saved_images(OUTPUT_DIR)

        print(
            f"Camera opened at {actual_width}x{actual_height}, "
            f"reported {actual_fps:.1f} FPS."
        )
        print(f"Images will be saved in: {OUTPUT_DIR}")
        print("Press S to save an image. Press Q to quit.")

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

            display_frame = frame.copy()
            draw_status(display_frame, saved_count)

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")):
                image_path = OUTPUT_DIR / f"cube_{image_number:04d}.jpg"

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(
                        f"Could not save image: {image_path}"
                    )

                print(f"Saved: {image_path}")
                image_number += 1
                saved_count += 1

            elif key in (ord("q"), ord("Q")):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed safely.")


if __name__ == "__main__":
    main()
