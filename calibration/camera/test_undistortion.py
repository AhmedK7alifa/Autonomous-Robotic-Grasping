from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

CALIBRATION_FILE = (
    ROOT
    / "camera_intrinsics"
    / "camera_calibration.npz"
)

IMAGE_FILE = (
    ROOT
    / "charuco_images"
    / "charuco_20.jpg"
)

OUTPUT_FILE = (
    ROOT
    / "camera_intrinsics"
    / "undistortion_test.jpg"
)


# Load calibration results
calibration = np.load(CALIBRATION_FILE)

camera_matrix = calibration["camera_matrix"]
distortion_coefficients = calibration[
    "distortion_coefficients"
]


# Load the test image
image = cv2.imread(str(IMAGE_FILE))

if image is None:
    raise SystemExit(
        f"Could not read image: {IMAGE_FILE}"
    )


height, width = image.shape[:2]


# Calculate an adjusted camera matrix
new_camera_matrix, roi = (
    cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (width, height),
        1,
        (width, height),
    )
)


# Remove lens distortion
undistorted = cv2.undistort(
    image,
    camera_matrix,
    distortion_coefficients,
    None,
    new_camera_matrix,
)


# Add labels
original_labeled = image.copy()
undistorted_labeled = undistorted.copy()

cv2.putText(
    original_labeled,
    "ORIGINAL",
    (30, 55),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.4,
    (0, 0, 255),
    3,
)

cv2.putText(
    undistorted_labeled,
    "UNDISTORTED",
    (30, 55),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.4,
    (0, 180, 0),
    3,
)


# Join the two images side by side
comparison = np.hstack(
    (
        original_labeled,
        undistorted_labeled,
    )
)


# Save the full-resolution comparison
cv2.imwrite(
    str(OUTPUT_FILE),
    comparison,
)


# Resize only for displaying on the screen
preview = cv2.resize(
    comparison,
    (1280, 360),
)

cv2.imshow(
    "Original vs Undistorted",
    preview,
)

print("Undistortion test completed.")
print(f"Test image: {IMAGE_FILE}")
print(f"Saved result: {OUTPUT_FILE}")
print("Press any key inside the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()