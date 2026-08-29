# YOLO evaluation and validation

This directory contains the preserved live/stability, false-positive diagnostic, and grasp-reference validation utilities used during pipeline development. PRE_PHYSICAL_VALIDATION_CHECKPOINT.txt records the frozen pre-motion validation state.

## Official held-out detector test

The one-time held-out test used 15 images and 15 ground-truth cube instances. The frozen reported results were:

| Metric | Value |
|---|---:|
| Precision | 0.916 |
| Recall | 1.000 |
| mAP@0.5 | 0.987 |
| mAP@0.5:0.95 | 0.885 |
| Ground-truth cubes detected | 15/15 |
| Additional background false positives | 2 |
| False negatives | 0 |
| Approximate CPU time | 68.2 ms/image |

These are detector-only metrics and are separate from physical robot success.

## Provenance limitation

No dedicated held-out-test runner or saved Ultralytics test-run directory was present in the authoritative frozen project. The numerical test result is preserved in the final experiment log and in [../../results/comparison/aruco_vs_yolo_comparison.json](../../results/comparison/aruco_vs_yolo_comparison.json). The split manifest and checkpoint are present, but the excluded test images prevent an independent test rerun from this repository alone.

The included diagnostic scripts are not represented as substitutes for the missing one-off test runner.
