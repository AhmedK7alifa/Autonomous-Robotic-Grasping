# Calibration

The [camera](camera) directory contains the publication-safe camera calibration program, undistortion test, and frozen calibration artifacts. Raw ChArUco capture images are excluded to keep the repository compact.

Calibration used 20 accepted images out of 20 found and produced an RMS reprojection error of 0.6122 px at 1280 × 720. The camera was fixed eye-to-hand. Any motion of the camera or robot base invalidates the saved pixel-to-joint mapping.

The robot’s per-unit servo calibration file is not distributed because it should not be applied to another physical unit. Reproduction requires a fresh device-specific calibration using the supported LeRobot/Hiwonder procedure.
