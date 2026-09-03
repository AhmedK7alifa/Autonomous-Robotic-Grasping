# Experimental setup

## Hardware

- Hiwonder SO-ARM101 follower robot
- Logitech C920 Pro HD Webcam in a fixed eye-to-hand arrangement
- 5 cm wooden cube
- Fixed open placement platform
- Windows workstation running the LeRobot SO-101 follower interface

The physical camera and mounting position were unchanged between methods.
Windows/OpenCV enumerated the camera as index 1 during the ArUco benchmark and
as DirectShow index 0 during the later YOLO benchmark. Both pipelines requested
1280 × 720 at 30 FPS and undistorted the full frame without cropping, resizing,
flipping, or rotation.

## Camera calibration

Intrinsic calibration used a 5 × 7 ChArUco board with 30 mm squares and 22 mm
markers. All 20 recorded calibration images were accepted. The saved result
reports an RMS reprojection error of 0.6122 px.

The calibration inputs, script, human-readable result, and runtime NPZ are in
`calibration/camera`. Re-running `calibrate_camera.py` writes to
`calibration/camera/generated` so the experimental files are not overwritten.

## Robot calibration and poses

`calibration/robot/white_follower_calibration_final_2026-07-26.json` records the
servo calibration for the physical SO-ARM101 unit used in the experiment.
Shared waypoint and source-anchor poses are stored under `robot_control/poses`.

The fixed waypoints are safe rest, box-high clearance, and box-low release. The
sixteen source-anchor pose files provide above and grasp poses for eight image
locations. Pose values are stored in the LeRobot joint-coordinate convention
used by the controllers.

## Workspace

The cube was placed at five benchmark locations spanning lower-left,
upper-left, center, upper-right, and lower-right workspace regions. Blue tape
marked the source region. The fixed open placement platform remained to the
right of the source region throughout both benchmarks.
