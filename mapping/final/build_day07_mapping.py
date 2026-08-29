import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Projects\Autonomous_Robotic_Grasping"
)

DATA_FILE = (
    PROJECT_ROOT
    / "06_pixel_to_robot"
    / "workspace_calibration_points.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "06_pixel_to_robot"
    / "pixel_to_joint_model_day07.json"
)

VISUALIZATION_FILE = (
    PROJECT_ROOT
    / "06_pixel_to_robot"
    / "day07_mapping_visualization.jpg"
)


# ============================================================
# Final Day 07 calibration points
# ============================================================

REQUIRED_POINT_IDS = [
    "P01_UL",
    "P02_TOP_MID_LEFT",
    "P03_TOP_RIGHT",
    "P04_LEFT_MID",
    "P05_CENTER",
    "P06_RIGHT_MID",
    "P07_BOTTOM_LEFT",
    "P08_BOTTOM_CENTER",
    "P09_BOTTOM_RIGHT",
]


# ============================================================
# Piecewise interpolation mesh
#
#  P01 -------- P02 -------- P03
#   | \           | \          |
#   |   \         |   \        |
#  P04 -------- P05 -------- P06
#   | \           | \          |
#   |   \         |   \        |
#  P07 -------- P08 -------- P09
#
#  Four cells, each divided into two triangles.
#  Total = 8 interpolation triangles.
#
#  No extrapolation outside this mesh.
# ============================================================

TRIANGLES = [
    # Upper-left cell
    [
        "P01_UL",
        "P02_TOP_MID_LEFT",
        "P05_CENTER",
    ],
    [
        "P01_UL",
        "P05_CENTER",
        "P04_LEFT_MID",
    ],

    # Upper-right cell
    [
        "P02_TOP_MID_LEFT",
        "P03_TOP_RIGHT",
        "P06_RIGHT_MID",
    ],
    [
        "P02_TOP_MID_LEFT",
        "P06_RIGHT_MID",
        "P05_CENTER",
    ],

    # Lower-left cell
    [
        "P04_LEFT_MID",
        "P05_CENTER",
        "P08_BOTTOM_CENTER",
    ],
    [
        "P04_LEFT_MID",
        "P08_BOTTOM_CENTER",
        "P07_BOTTOM_LEFT",
    ],

    # Lower-right cell
    [
        "P05_CENTER",
        "P06_RIGHT_MID",
        "P09_BOTTOM_RIGHT",
    ],
    [
        "P05_CENTER",
        "P09_BOTTOM_RIGHT",
        "P08_BOTTOM_CENTER",
    ],
]


# ============================================================
# Geometry
# ============================================================

def triangle_area(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    vector_ab = point_b - point_a
    vector_ac = point_c - point_a

    cross_value = (
        vector_ab[0] * vector_ac[1]
        - vector_ab[1] * vector_ac[0]
    )

    return abs(float(cross_value)) / 2.0


# ============================================================
# Load and validate calibration dataset
# ============================================================

def load_points() -> tuple[dict, dict[str, dict]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {DATA_FILE}"
        )

    with DATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    raw_points = data.get("points")

    if not isinstance(raw_points, list):
        raise ValueError(
            "The calibration file does not contain "
            "a valid points list."
        )

    points_by_id = {}

    for point in raw_points:
        point_id = point.get("point_id")

        if not point_id:
            raise ValueError(
                "A calibration point is missing "
                "its point_id."
            )

        if point_id in points_by_id:
            raise ValueError(
                f"Duplicate calibration point: "
                f"{point_id}"
            )

        center_px = (
            point.get("marker", {})
            .get("center_px")
        )

        joints = point.get(
            "robot_joint_positions"
        )

        if (
            not isinstance(center_px, list)
            or len(center_px) != 2
        ):
            raise ValueError(
                f"Invalid pixel center for "
                f"{point_id}."
            )

        if not isinstance(joints, dict):
            raise ValueError(
                f"Invalid robot joint data for "
                f"{point_id}."
            )

        points_by_id[point_id] = point

    missing = [
        point_id
        for point_id in REQUIRED_POINT_IDS
        if point_id not in points_by_id
    ]

    if missing:
        raise KeyError(
            "Missing required calibration points: "
            + ", ".join(missing)
        )

    if len(points_by_id) != 9:
        print(
            "\nWARNING:"
            f" Dataset contains {len(points_by_id)} "
            "points."
        )
        print(
            "The Day 07 model will use only the "
            "nine required final points."
        )

    return data, points_by_id


# ============================================================
# Build Day 07 model
# ============================================================

def build_model(
    source_data: dict,
    points_by_id: dict[str, dict],
) -> dict:
    simplified_points = {}

    x_values = []
    y_values = []

    for point_id in REQUIRED_POINT_IDS:
        point = points_by_id[point_id]

        center = [
            float(
                point["marker"]["center_px"][0]
            ),
            float(
                point["marker"]["center_px"][1]
            ),
        ]

        x_values.append(center[0])
        y_values.append(center[1])

        simplified_points[point_id] = {
            "center_px": center,
            "robot_joint_positions": {
                joint_name: float(value)
                for joint_name, value
                in point[
                    "robot_joint_positions"
                ].items()
            },
        }

    triangle_areas = []

    for triangle in TRIANGLES:
        point_a = np.array(
            simplified_points[
                triangle[0]
            ]["center_px"],
            dtype=np.float64,
        )

        point_b = np.array(
            simplified_points[
                triangle[1]
            ]["center_px"],
            dtype=np.float64,
        )

        point_c = np.array(
            simplified_points[
                triangle[2]
            ]["center_px"],
            dtype=np.float64,
        )

        area = triangle_area(
            point_a,
            point_b,
            point_c,
        )

        if area < 50.0:
            raise ValueError(
                "A mapping triangle is too small "
                "or degenerate: "
                + " -> ".join(triangle)
            )

        triangle_areas.append(area)

    return {
        "schema_version": 1,
        "model_type": (
            "piecewise_linear_barycentric_"
            "pixel_to_joint"
        ),
        "phase": (
            "day07_final_full_workspace_9pts"
        ),
        "source_data_file": str(
            DATA_FILE
        ),
        "camera": source_data.get(
            "camera",
            {},
        ),
        "marker": source_data.get(
            "marker",
            {},
        ),
        "robot": source_data.get(
            "robot",
            {},
        ),
        "safety": {
            "allow_extrapolation": False,
            "movement_enabled_by_this_script": (
                False
            ),
            "note": (
                "Targets are valid only when their "
                "pixel center lies inside one of "
                "the saved Day 07 interpolation "
                "triangles."
            ),
        },
        "pixel_bounds": {
            "min_u": min(x_values),
            "max_u": max(x_values),
            "min_v": min(y_values),
            "max_v": max(y_values),
        },
        "points": simplified_points,
        "triangles": TRIANGLES,
        "triangle_areas_px2": (
            triangle_areas
        ),
    }


# ============================================================
# Visualization
# ============================================================

def save_visualization(
    points_by_id: dict[str, dict],
) -> None:
    reference_image_path = Path(
        points_by_id[
            "P05_CENTER"
        ]["marker"]["evidence_image"]
    )

    image = cv2.imread(
        str(reference_image_path)
    )

    if image is None:
        image = np.full(
            (720, 1280, 3),
            235,
            dtype=np.uint8,
        )

    point_pixels = {
        point_id: tuple(
            int(round(value))
            for value
            in points_by_id[
                point_id
            ]["marker"]["center_px"]
        )
        for point_id
        in REQUIRED_POINT_IDS
    }

    # Draw interpolation mesh.
    for triangle in TRIANGLES:
        polygon = np.array(
            [
                point_pixels[point_id]
                for point_id in triangle
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            image,
            [polygon],
            True,
            (255, 140, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw calibration points.
    for point_id, center in (
        point_pixels.items()
    ):
        cv2.circle(
            image,
            center,
            7,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            point_id,
            (
                center[0] + 10,
                center[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            point_id,
            (
                center[0] + 10,
                center[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    title = (
        "Day 07 full-workspace mapping "
        "(9 points / 8 triangles)"
    )

    cv2.putText(
        image,
        title,
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        title,
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    subtitle = (
        "Piecewise barycentric interpolation "
        "- extrapolation disabled"
    )

    cv2.putText(
        image,
        subtitle,
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        subtitle,
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(
        str(VISUALIZATION_FILE),
        image,
    ):
        raise RuntimeError(
            "Could not save visualization: "
            f"{VISUALIZATION_FILE}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 62)
    print(
        "DAY 07 - FINAL FULL-WORKSPACE "
        "PIXEL-TO-JOINT MAPPING"
    )
    print("=" * 62)

    source_data, points_by_id = (
        load_points()
    )

    model = build_model(
        source_data,
        points_by_id,
    )

    with MODEL_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model,
            file,
            indent=2,
        )

    save_visualization(
        points_by_id
    )

    bounds = model["pixel_bounds"]

    print(
        "\nDay 07 mapping built "
        "successfully."
    )

    print(
        f"Calibration points: "
        f"{len(model['points'])}"
    )

    print(
        f"Interpolation triangles: "
        f"{len(model['triangles'])}"
    )

    print(
        "Pixel bounds: "
        f"u={bounds['min_u']:.2f} "
        f"to {bounds['max_u']:.2f}, "
        f"v={bounds['min_v']:.2f} "
        f"to {bounds['max_v']:.2f}"
    )

    print(
        f"Model file: "
        f"{MODEL_FILE}"
    )

    print(
        f"Visualization: "
        f"{VISUALIZATION_FILE}"
    )

    print(
        "\nSafety:"
    )
    print(
        "- Extrapolation outside the "
        "8-triangle mesh is disabled."
    )
    print(
        "- This script does NOT connect "
        "to the robot."
    )
    print(
        "- This script does NOT move "
        "the robot."
    )


if __name__ == "__main__":
    main()