import json
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    Path(__file__).resolve().parent
    / "pixel_to_joint_model.json"
)

CAMERA_CALIBRATION_FILE = (
    PROJECT_ROOT
    / "calibration"
    / "camera"
    / "camera_calibration.npz"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "validation_results"
)

CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30

TARGET_MARKER_ID = 0

CENTER_SAMPLE_COUNT = 30
MINIMUM_VALID_SAMPLES = 15
MAX_CENTER_STD_PX = 2.0

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ============================================================
# File loading
# ============================================================

def load_mapping_model() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Mapping model not found: {MODEL_FILE}"
        )

    with MODEL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        model = json.load(file)

    expected_type = (
        "piecewise_linear_barycentric_pixel_to_joint"
    )

    if model.get("model_type") != expected_type:
        raise ValueError(
            "Unexpected mapping model type: "
            f"{model.get('model_type')}"
        )

    points = model.get("points")
    triangles = model.get("triangles")

    if not isinstance(points, dict):
        raise ValueError(
            "The mapping model does not contain "
            "valid points."
        )

    if not isinstance(triangles, list):
        raise ValueError(
            "The mapping model does not contain "
            "valid triangles."
        )

    if len(points) != 9:
        raise ValueError(
            f"Expected 9 Day 07 calibration points, "
            f"found {len(points)}."
        )

    if len(triangles) != 8:
        raise ValueError(
            f"Expected 8 interpolation triangles, "
            f"found {len(triangles)}."
        )

    return model


def load_camera_calibration() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if not CAMERA_CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            "Camera calibration file not found: "
            f"{CAMERA_CALIBRATION_FILE}"
        )

    calibration = np.load(
        CAMERA_CALIBRATION_FILE,
        allow_pickle=False,
    )

    if "camera_matrix" not in calibration:
        raise KeyError(
            "camera_matrix was not found in "
            "the camera calibration file."
        )

    if "distortion_coefficients" in calibration:
        distortion_key = (
            "distortion_coefficients"
        )
    elif "dist_coeffs" in calibration:
        distortion_key = "dist_coeffs"
    else:
        raise KeyError(
            "Distortion coefficients were not "
            "found in the calibration file."
        )

    camera_matrix = np.asarray(
        calibration["camera_matrix"],
        dtype=np.float64,
    )

    distortion_coefficients = np.asarray(
        calibration[distortion_key],
        dtype=np.float64,
    ).reshape(-1)

    return (
        camera_matrix,
        distortion_coefficients,
    )


# ============================================================
# ArUco detection
# ============================================================

def create_aruco_detector():
    dictionary = (
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )

    parameters = (
        cv2.aruco.DetectorParameters()
    )

    if hasattr(
        cv2.aruco,
        "ArucoDetector",
    ):
        return cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

    return (
        dictionary,
        parameters,
    )


def detect_markers(
    detector,
    gray_frame: np.ndarray,
):
    if hasattr(
        detector,
        "detectMarkers",
    ):
        return detector.detectMarkers(
            gray_frame
        )

    dictionary, parameters = detector

    return cv2.aruco.detectMarkers(
        gray_frame,
        dictionary,
        parameters=parameters,
    )


def find_target_marker(
    corners,
    ids,
):
    if ids is None:
        return None

    for index, marker_id in enumerate(
        ids.flatten()
    ):
        if (
            int(marker_id)
            != TARGET_MARKER_ID
        ):
            continue

        marker_corners = np.asarray(
            corners[index],
            dtype=np.float64,
        ).reshape(4, 2)

        center = marker_corners.mean(
            axis=0
        )

        return {
            "corners": marker_corners,
            "center": center,
        }

    return None


# ============================================================
# Piecewise barycentric interpolation
# ============================================================

def barycentric_weights(
    point: np.ndarray,
    vertex_a: np.ndarray,
    vertex_b: np.ndarray,
    vertex_c: np.ndarray,
) -> np.ndarray | None:
    matrix = np.column_stack(
        (
            vertex_a - vertex_c,
            vertex_b - vertex_c,
        )
    )

    determinant = float(
        np.linalg.det(matrix)
    )

    if abs(determinant) < 1e-9:
        return None

    first_two = np.linalg.solve(
        matrix,
        point - vertex_c,
    )

    weight_a = float(
        first_two[0]
    )

    weight_b = float(
        first_two[1]
    )

    weight_c = (
        1.0
        - weight_a
        - weight_b
    )

    return np.array(
        [
            weight_a,
            weight_b,
            weight_c,
        ],
        dtype=np.float64,
    )


def locate_triangle_and_predict(
    center_px: np.ndarray,
    model: dict,
) -> dict | None:
    points = model["points"]

    tolerance = 1e-6

    for (
        triangle_index,
        triangle_ids,
    ) in enumerate(
        model["triangles"]
    ):
        if len(triangle_ids) != 3:
            raise ValueError(
                "Every triangle must contain "
                "exactly three point IDs."
            )

        vertices = [
            np.asarray(
                points[
                    point_id
                ]["center_px"],
                dtype=np.float64,
            )
            for point_id
            in triangle_ids
        ]

        weights = (
            barycentric_weights(
                center_px,
                vertices[0],
                vertices[1],
                vertices[2],
            )
        )

        if weights is None:
            continue

        inside_triangle = (
            np.all(
                weights
                >= -tolerance
            )
            and np.all(
                weights
                <= 1.0 + tolerance
            )
        )

        if not inside_triangle:
            continue

        predicted_joints = {}

        for joint_name in JOINT_NAMES:
            joint_values = np.asarray(
                [
                    float(
                        points[
                            point_id
                        ][
                            "robot_joint_positions"
                        ][joint_name]
                    )
                    for point_id
                    in triangle_ids
                ],
                dtype=np.float64,
            )

            predicted_joints[
                joint_name
            ] = float(
                np.dot(
                    weights,
                    joint_values,
                )
            )

        return {
            "triangle_index": (
                triangle_index
            ),
            "triangle_ids": (
                triangle_ids
            ),
            "weights": weights,
            "predicted_joints": (
                predicted_joints
            ),
        }

    return None


# ============================================================
# Drawing
# ============================================================

def point_pixels_from_model(
    model: dict,
) -> dict[str, tuple[int, int]]:
    return {
        point_id: (
            int(
                round(
                    float(
                        point[
                            "center_px"
                        ][0]
                    )
                )
            ),
            int(
                round(
                    float(
                        point[
                            "center_px"
                        ][1]
                    )
                )
            ),
        )
        for point_id, point
        in model["points"].items()
    }


def draw_mapping_mesh(
    frame: np.ndarray,
    model: dict,
    active_triangle_index: int | None,
) -> None:
    point_pixels = (
        point_pixels_from_model(
            model
        )
    )

    for (
        triangle_index,
        triangle_ids,
    ) in enumerate(
        model["triangles"]
    ):
        polygon = np.asarray(
            [
                point_pixels[
                    point_id
                ]
                for point_id
                in triangle_ids
            ],
            dtype=np.int32,
        ).reshape(
            (-1, 1, 2)
        )

        if (
            triangle_index
            == active_triangle_index
        ):
            thickness = 4
            color = (
                0,
                255,
                0,
            )
        else:
            thickness = 1
            color = (
                255,
                140,
                0,
            )

        cv2.polylines(
            frame,
            [polygon],
            True,
            color,
            thickness,
            cv2.LINE_AA,
        )

    for (
        point_id,
        center,
    ) in point_pixels.items():
        cv2.circle(
            frame,
            center,
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            point_id,
            (
                center[0] + 5,
                center[1] - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_text_lines(
    frame: np.ndarray,
    lines: list[str],
    start_y: int,
) -> None:
    for (
        line_index,
        line,
    ) in enumerate(lines):
        y_position = (
            start_y
            + line_index * 27
        )

        cv2.putText(
            frame,
            line,
            (
                20,
                y_position,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            line,
            (
                20,
                y_position,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# Save validation result
# ============================================================

def save_validation_result(
    frame: np.ndarray,
    center_px: np.ndarray,
    center_std_px: np.ndarray,
    prediction: dict,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    image_path = (
        OUTPUT_DIR
        / (
            "day07_validation_"
            f"{timestamp}.jpg"
        )
    )

    json_path = (
        OUTPUT_DIR
        / (
            "day07_validation_"
            f"{timestamp}.json"
        )
    )

    if not cv2.imwrite(
        str(image_path),
        frame,
    ):
        raise RuntimeError(
            "Could not save "
            f"image: {image_path}"
        )

    result = {
        "timestamp": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "phase": (
            "day07_full_workspace"
        ),
        "mode": (
            "calculation_only_"
            "no_robot_connection"
        ),
        "mapping_model": str(
            MODEL_FILE
        ),
        "target_marker_id": (
            TARGET_MARKER_ID
        ),
        "center_px": [
            round(
                float(
                    center_px[0]
                ),
                3,
            ),
            round(
                float(
                    center_px[1]
                ),
                3,
            ),
        ],
        "center_std_px": [
            round(
                float(
                    center_std_px[0]
                ),
                3,
            ),
            round(
                float(
                    center_std_px[1]
                ),
                3,
            ),
        ],
        "triangle_index": int(
            prediction[
                "triangle_index"
            ]
        ),
        "triangle_ids": list(
            prediction[
                "triangle_ids"
            ]
        ),
        "barycentric_weights": [
            round(
                float(value),
                6,
            )
            for value
            in prediction[
                "weights"
            ]
        ],
        "predicted_joint_positions": {
            joint_name: round(
                float(value),
                4,
            )
            for joint_name, value
            in prediction[
                "predicted_joints"
            ].items()
        },
        "robot_connection_performed": (
            False
        ),
        "robot_movement_performed": (
            False
        ),
        "evidence_image": str(
            image_path
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        "\nDay 07 validation "
        "result saved."
    )

    print(
        f"Image: {image_path}"
    )

    print(
        f"Data:  {json_path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    model = (
        load_mapping_model()
    )

    (
        camera_matrix,
        distortion_coefficients,
    ) = (
        load_camera_calibration()
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        FRAME_RATE,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open "
            f"camera index "
            f"{CAMERA_INDEX}."
        )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if (
        actual_width
        != FRAME_WIDTH
        or actual_height
        != FRAME_HEIGHT
    ):
        camera.release()

        raise RuntimeError(
            "Unexpected camera "
            "resolution: "
            f"{actual_width}x"
            f"{actual_height}. "
            "Expected "
            f"{FRAME_WIDTH}x"
            f"{FRAME_HEIGHT}."
        )

    map_x, map_y = (
        cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion_coefficients,
            None,
            camera_matrix,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT,
            ),
            cv2.CV_32FC1,
        )
    )

    detector = (
        create_aruco_detector()
    )

    center_samples = deque(
        maxlen=CENTER_SAMPLE_COUNT
    )

    print(
        "\n"
        + "=" * 62
    )

    print(
        "DAY 07 - FULL-WORKSPACE "
        "MAPPING VALIDATION"
    )

    print(
        "=" * 62
    )

    print(
        "Calculation only."
    )

    print(
        "The robot will NOT "
        "connect or move."
    )

    print(
        "Place ArUco ID 0 at "
        "a NEW position inside "
        "the mapped region."
    )

    print(
        "Press S only when the "
        "prediction is stable."
    )

    print(
        "Press Q to close."
    )

    try:
        while True:
            success, raw_frame = (
                camera.read()
            )

            if not success:
                raise RuntimeError(
                    "Could not read "
                    "a camera frame."
                )

            corrected_frame = (
                cv2.remap(
                    raw_frame,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                )
            )

            display_frame = (
                corrected_frame.copy()
            )

            gray_frame = (
                cv2.cvtColor(
                    corrected_frame,
                    cv2.COLOR_BGR2GRAY,
                )
            )

            corners, ids, _ = (
                detect_markers(
                    detector,
                    gray_frame,
                )
            )

            marker = (
                find_target_marker(
                    corners,
                    ids,
                )
            )

            prediction = None
            stable_center = None

            center_std = np.array(
                [
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

            if marker is None:
                center_samples.clear()

                status_lines = [
                    "ID 0 not detected",
                    (
                        "NO ROBOT CONNECTION "
                        "OR MOVEMENT"
                    ),
                ]

                active_triangle_index = (
                    None
                )

            else:
                center_samples.append(
                    marker[
                        "center"
                    ].copy()
                )

                center_array = (
                    np.asarray(
                        center_samples,
                        dtype=np.float64,
                    )
                )

                stable_center = (
                    np.median(
                        center_array,
                        axis=0,
                    )
                )

                center_std = (
                    center_array.std(
                        axis=0,
                    )
                )

                prediction = (
                    locate_triangle_and_predict(
                        stable_center,
                        model,
                    )
                )

                integer_corners = (
                    marker[
                        "corners"
                    ]
                    .round()
                    .astype(
                        np.int32
                    )
                    .reshape(
                        (-1, 1, 2)
                    )
                )

                cv2.polylines(
                    display_frame,
                    [
                        integer_corners
                    ],
                    True,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                center_tuple = (
                    int(
                        round(
                            float(
                                stable_center[
                                    0
                                ]
                            )
                        )
                    ),
                    int(
                        round(
                            float(
                                stable_center[
                                    1
                                ]
                            )
                        )
                    ),
                )

                cv2.circle(
                    display_frame,
                    center_tuple,
                    7,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )

                stable_enough = (
                    len(
                        center_samples
                    )
                    >= MINIMUM_VALID_SAMPLES
                    and float(
                        center_std.max()
                    )
                    <= MAX_CENTER_STD_PX
                )

                if prediction is None:
                    active_triangle_index = (
                        None
                    )

                    status_lines = [
                        (
                            "ID 0 center: "
                            f"("
                            f"{stable_center[0]:.1f}, "
                            f"{stable_center[1]:.1f}"
                            f")"
                        ),
                        (
                            "OUTSIDE DAY 07 "
                            "MAPPED REGION"
                        ),
                        (
                            "Samples: "
                            f"{len(center_samples)}/"
                            f"{CENTER_SAMPLE_COUNT}  "
                            "Std: "
                            f"("
                            f"{center_std[0]:.2f}, "
                            f"{center_std[1]:.2f}"
                            f") px"
                        ),
                        (
                            "NO ROBOT CONNECTION "
                            "OR MOVEMENT"
                        ),
                    ]

                else:
                    active_triangle_index = (
                        int(
                            prediction[
                                "triangle_index"
                            ]
                        )
                    )

                    triangle_text = (
                        " / ".join(
                            prediction[
                                "triangle_ids"
                            ]
                        )
                    )

                    status_lines = [
                        (
                            "ID 0 center: "
                            f"("
                            f"{stable_center[0]:.1f}, "
                            f"{stable_center[1]:.1f}"
                            f")"
                        ),
                        (
                            "VALID INSIDE "
                            "DAY 07 REGION"
                        ),
                        (
                            "Triangle: "
                            f"{triangle_text}"
                        ),
                        (
                            "Samples: "
                            f"{len(center_samples)}/"
                            f"{CENTER_SAMPLE_COUNT}  "
                            "Std: "
                            f"("
                            f"{center_std[0]:.2f}, "
                            f"{center_std[1]:.2f}"
                            f") px"
                        ),
                        (
                            "Stable prediction: "
                            f"{'YES' if stable_enough else 'WAIT'}"
                        ),
                        (
                            "NO ROBOT CONNECTION "
                            "OR MOVEMENT"
                        ),
                    ]

                    joint_start_y = 205

                    for (
                        joint_index,
                        joint_name,
                    ) in enumerate(
                        JOINT_NAMES
                    ):
                        predicted_value = (
                            prediction[
                                "predicted_joints"
                            ][joint_name]
                        )

                        line = (
                            f"{joint_name}: "
                            f"{predicted_value:.2f}"
                        )

                        draw_text_lines(
                            display_frame,
                            [line],
                            (
                                joint_start_y
                                + joint_index
                                * 25
                            ),
                        )

            draw_mapping_mesh(
                display_frame,
                model,
                active_triangle_index,
            )

            draw_text_lines(
                display_frame,
                status_lines,
                30,
            )

            draw_text_lines(
                display_frame,
                [
                    (
                        "S: save stable "
                        "prediction   "
                        "Q: close"
                    ),
                ],
                FRAME_HEIGHT - 25,
            )

            cv2.imshow(
                (
                    "Day 07 Mapping Validation "
                    "- NO ROBOT MOVEMENT"
                ),
                display_frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == ord("s"):
                if marker is None:
                    print(
                        "Marker ID 0 "
                        "is not detected."
                    )
                    continue

                if prediction is None:
                    print(
                        "Target is outside "
                        "the Day 07 mapped "
                        "region."
                    )
                    continue

                if (
                    len(
                        center_samples
                    )
                    < MINIMUM_VALID_SAMPLES
                ):
                    print(
                        "Not enough stable "
                        "samples yet."
                    )
                    continue

                if (
                    float(
                        center_std.max()
                    )
                    > MAX_CENTER_STD_PX
                ):
                    print(
                        "Marker center is "
                        "not stable enough."
                    )
                    continue

                save_validation_result(
                    display_frame,
                    stable_center,
                    center_std,
                    prediction,
                )

                print(
                    "\nPredicted "
                    "joint positions:"
                )

                print(
                    "-" * 48
                )

                for joint_name in (
                    JOINT_NAMES
                ):
                    print(
                        f"{joint_name:18s}: "
                        f"{prediction['predicted_joints'][joint_name]:8.2f}"
                    )

                print(
                    "-" * 48
                )

                print(
                    "No robot connection "
                    "or movement was "
                    "performed."
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print(
            "Camera closed safely."
        )


if __name__ == "__main__":
    main()
