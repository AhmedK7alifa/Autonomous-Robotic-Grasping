# Benchmark protocol

## Scope

The official comparison consists only of the ArUco cube pipeline and the YOLO cube pipeline. Each method completed 20 physical trials under a matched controlled design.

## Locations and repetitions

| ID | Location | Trials per method |
|---|---|---:|
| L1 | lower left | 4 |
| L2 | upper left | 4 |
| L3 | center | 4 |
| L4 | upper right | 4 |
| L5 | lower right | 4 |

This yields 5 locations × 4 repetitions × 2 methods = 40 official robotic trials.

## Frozen inputs

Before official trials, the following were frozen:

- camera calibration and full-frame undistortion;
- Day-7 9-point/8-triangle mapping;
- source-anchor corrections and final waypoints;
- ArUco V3 controller variants, retained in the internal archive but withheld from the public release;
- YOLO best.pt checkpoint and perception thresholds;
- global upper-quarter YOLO grasp-reference rule; and
- outcome-record structure and analysis definitions.

Development tests, diagnostics, dry runs, and non-official records are excluded from the benchmark summaries.

The physical controllers are historical benchmark inputs, not public operating materials. This repository contains no executable robot-actuation controller or direct physical-motion instructions.

## Task outcome

The published method records were not textually identical, so the final comparison applies a common stricter definition. A harmonized successful full cycle requires:

1. grip/contact accepted;
2. release open verified;
3. cycle completed; and
4. observed placement success.

All four fields are true in every official ArUco and YOLO row, so harmonization does not alter the reported 20/20 counts.

The correct interpretation is: **100% observed success under the controlled benchmark conditions.** It is not a universal success probability.

## Cycle time

Cycle time is the recorded full-cycle duration in seconds. Summaries report mean, median, minimum, maximum, and sample standard deviation with denominator n − 1.

No inferential test is used to claim a speed difference. Most downstream robot motion is shared, so the observed mean difference of −0.097 s for YOLO relative to ArUco is descriptive only.

## Harmonized image-space repeatability

Published method-specific summaries originally used different nominal-center definitions. For the cross-method comparison, each location and coordinate type uses the arithmetic mean of its four official observations as an empirical nominal center.

For observation p at a location with empirical center c, distance is the Euclidean image-space norm of p − c in pixels. Overall mean and maximum distances pool the 20 per-trial distances for that coordinate type.

Coordinate types are:

- ArUco marker center;
- YOLO bounding-box center; and
- YOLO upper-quarter grasp reference.

These are descriptive repeatability values from four trials per location and do not establish statistical superiority.

## Detector-only held-out test

The 15-image YOLO held-out test is separate from the physical benchmark. It reports detector precision, recall, mAP, detected ground-truth cubes, extra background detections, and false negatives. Detector performance must not be presented as robotic task success.

## Integrity checks

The frozen analyzers require exactly four official records in each of five location directories, finite cycle times, consistent trial identifiers, complete outcome fields, and no unexpected official rows. The included official outputs report integrity PASS.
