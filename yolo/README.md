# YOLO learning-based pipeline

This directory preserves publication-safe YOLO26 training metadata, detector evaluation utilities, perception/mapping modules, dry-run/audit references, calibration and pose artifacts, and the project-specific best.pt checkpoint.

## Frozen perception handoff

The detector used class 0 for the cube, confidence threshold 0.25, duplicate suppression at IoU 0.50, a rolling buffer of 30 detections, at least 20 stable samples, and per-axis standard deviation at most 2.0 px. The final upper-quarter grasp reference was global and deterministic.

## Contents

- [training](training) contains dataset capture/annotation utilities, split metadata, saved training arguments, and epoch history.
- [evaluation](evaluation) contains camera/live validation utilities and provenance notes for the held-out detector result.
- [final](final) contains the publication-safe frozen perception/mapping and dry-run/reference subset, calibration, poses, and best.pt.

The executable full pick-and-place controller and the duplicated hardware-enabled ArUco source used by the internal integration are excluded from this public release. Their provenance remains in the freeze manifest and source-file audit.

## Model attribution

best.pt is the project-specific trained YOLO26 checkpoint. Ultralytics and its contributors created and maintain YOLO26 and the Ultralytics toolchain. See [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Official result

YOLO achieved 20/20 observed successful full cycles under the controlled benchmark conditions. Mean cycle time was 133.392 s. Harmonized repeatability was 1.485 px for the bounding-box center and 1.503 px for the upper-quarter grasp reference.
