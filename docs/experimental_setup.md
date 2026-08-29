# Experimental setup

## Physical system

| Item | Frozen configuration |
|---|---|
| Robot | Hiwonder SO-ARM101 follower |
| Robot role | Follower arm |
| Camera | Logitech C920 Pro HD Webcam |
| Camera geometry | Fixed eye-to-hand |
| Image stream | 1280 × 720 at 30 FPS |
| Object | 5 cm wooden cube |
| Workspace | Fixed calibrated region on a gray work mat |
| Destination | Fixed platform/box |

The robot base, camera mount, pick region, and destination remained fixed across the controlled comparison. Camera or base motion after calibration invalidates the pixel-to-joint mapping.

## Camera enumeration

The earlier ArUco controllers record OpenCV camera index 1. During YOLO development, Windows device enumeration changed and the same physical camera was verified at DirectShow index 0. The camera hardware and mounting geometry did not change. Camera indices are host-specific and must never be assumed on another machine.

## Intrinsic calibration

The Logitech camera was calibrated with a 5 × 7 ChArUco board using DICT_4X4_50, 30 mm squares, and 22 mm board markers. All 20 captured calibration images were accepted. The RMS reprojection error was 0.6122209679 px.

Both final pipelines use the saved camera matrix and distortion coefficients to remap the full 1280 × 720 frame. No crop, resize, flip, or rotation is applied before localization.

## ArUco condition

The cube carried a visible 40 mm ArUco marker with ID 0 from DICT_4X4_50. The image-space handoff was the stable marker center.

## YOLO condition

The same 5 cm wooden cube was used without a visible fiducial marker. A single-class yolo26n model detected the cube. Duplicate, highly overlapping detections were suppressed before the temporal stability test. The frozen upper-quarter grasp reference was derived from median bounding-box geometry.

## Shared downstream system

Both methods handed an image coordinate to the same Day-7 9-point, 8-triangle piecewise barycentric mapping basis. The physical benchmark then used the same downstream system for both perception methods. The experiment did not measure gripping force, torque, or motor current.

## Public safety boundary

This public repository does not distribute executable robot-actuation controllers or instructions for direct physical robot motion. The physical benchmark used internal, hardware-specific controllers that remain archived outside the public release; their provenance and exclusion status are recorded in `SOURCE_FILE_MAP.tsv` and `PUBLIC_REPOSITORY_REVIEW.md`.

The included poses and calibration records are research artifacts from one experimental unit and geometry. They are not operating instructions and must not be treated as transferable robot commands. Physical robot operation is outside the scope of this repository and requires the robot manufacturer’s documentation, qualified supervision, and an institutionally approved safety process.
