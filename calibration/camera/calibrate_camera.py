from pathlib import Path
import json

import cv2
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "generated"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# ChArUco board parameters
# Must match the printed board
# --------------------------------------------------
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.022


# --------------------------------------------------
# Create the ChArUco board and detector
# --------------------------------------------------
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH_M,
    MARKER_LENGTH_M,
    dictionary,
)

detector = cv2.aruco.CharucoDetector(board)

board_points = np.asarray(
    board.getChessboardCorners(),
    dtype=np.float32,
).reshape(-1, 3)


# --------------------------------------------------
# Read and detect all calibration images
# --------------------------------------------------
object_points = []
image_points = []
used_files = []

image_size = None

image_files = sorted(
    IMAGE_DIR.glob("charuco_*.jpg")
)

if not image_files:
    raise SystemExit(
        f"No calibration images found in: {IMAGE_DIR}"
    )


for image_path in image_files:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"SKIP unreadable: {image_path.name}")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    current_size = (
        gray.shape[1],
        gray.shape[0],
    )

    if image_size is None:
        image_size = current_size

    elif current_size != image_size:
        print(
            f"SKIP different resolution: "
            f"{image_path.name}"
        )
        continue

    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ) = detector.detectBoard(gray)

    if charuco_ids is None or charuco_corners is None:
        print(f"SKIP no board: {image_path.name}")
        continue

    ids = charuco_ids.reshape(-1).astype(np.int32)

    corners = charuco_corners.reshape(
        -1, 2
    ).astype(np.float32)

    if len(ids) < 6:
        print(
            f"SKIP weak detection: "
            f"{image_path.name} "
            f"({len(ids)} corners)"
        )
        continue

    object_points.append(
        board_points[ids]
    )

    image_points.append(
        corners
    )

    used_files.append(
        image_path.name
    )

    print(
        f"USE  {image_path.name}: "
        f"{len(ids)} corners"
    )


if len(used_files) < 10:
    raise SystemExit(
        f"Only {len(used_files)} usable images found. "
        f"At least 10 are required."
    )


# --------------------------------------------------
# Camera calibration
# --------------------------------------------------
(
    rms_error,
    camera_matrix,
    distortion_coefficients,
    rotation_vectors,
    translation_vectors,
) = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None,
)


# --------------------------------------------------
# Calculate error for each image
# --------------------------------------------------
per_view_errors = {}

for name, obj, img, rvec, tvec in zip(
    used_files,
    object_points,
    image_points,
    rotation_vectors,
    translation_vectors,
):
    projected_points, _ = cv2.projectPoints(
        obj,
        rvec,
        tvec,
        camera_matrix,
        distortion_coefficients,
    )

    projected_points = projected_points.reshape(-1, 2)

    difference = projected_points - img

    image_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    difference * difference,
                    axis=1,
                )
            )
        )
    )

    per_view_errors[name] = image_rmse


# --------------------------------------------------
# Save NumPy calibration file
# --------------------------------------------------
np.savez(
    str(OUTPUT_DIR / "camera_calibration.npz"),
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
    image_width=image_size[0],
    image_height=image_size[1],
    rms_reprojection_error=rms_error,
)


# --------------------------------------------------
# Save readable JSON file
# --------------------------------------------------
results = {
    "camera": "Logitech C920 Pro",
    "resolution": [
        image_size[0],
        image_size[1],
    ],
    "board": {
        "dictionary": "DICT_4X4_50",
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_m": SQUARE_LENGTH_M,
        "marker_length_m": MARKER_LENGTH_M,
    },
    "images_found": len(image_files),
    "images_used": len(used_files),
    "used_files": used_files,
    "rms_reprojection_error_px": float(rms_error),
    "camera_matrix": camera_matrix.tolist(),
    "distortion_coefficients": (
        distortion_coefficients.reshape(-1).tolist()
    ),
    "per_view_rmse_px": per_view_errors,
}

with open(
    OUTPUT_DIR / "camera_calibration.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )


# --------------------------------------------------
# Print results
# --------------------------------------------------
worst_images = sorted(
    per_view_errors.items(),
    key=lambda item: item[1],
    reverse=True,
)

print("\n=== CAMERA CALIBRATION COMPLETE ===")
print(f"Images found: {len(image_files)}")
print(f"Images used : {len(used_files)}")
print(
    f"Resolution  : "
    f"{image_size[0]}x{image_size[1]}"
)
print(
    f"RMS error   : "
    f"{rms_error:.4f} pixels"
)

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(
    distortion_coefficients.reshape(-1)
)

print("\nWorst five images:")
for name, error in worst_images[:5]:
    print(
        f"{name}: {error:.4f} pixels"
    )

print(
    f"\nResults saved in: {OUTPUT_DIR}"
)
