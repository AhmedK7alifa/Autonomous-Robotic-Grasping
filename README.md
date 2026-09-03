# Autonomous Robotic Grasping

*A controlled comparison of marker-based ArUco and learning-based YOLO perception for pick-and-place*

**Author:** AHMED KHALIFA IDRISS ELHAJ (叶鹏)<br>
**Affiliation:** College of Computer Science, Chengdu Normal University, Chengdu, Sichuan, China

## Overview

This repository contains the actual full-cycle ArUco and YOLO controllers,
calibration, trained detector, dataset, benchmark records, and analysis used
to compare the two perception methods in a physical robotic pick-and-place
task. Both methods used the same robot, workspace, pixel-to-joint mapping,
source-anchor correction, and manipulation sequence. They differed in how the
target image coordinate was obtained.

ArUco used the center of marker ID 0. YOLO used the horizontal center and upper
quarter of a stable cube bounding box. Both coordinates entered the same
nine-point, eight-triangle barycentric mapping and the same dynamic correction
and robot-control pipeline.

Both methods completed 20/20 full pick-and-place trials in the controlled
benchmark.

## Demonstration

[Experimental demonstration on Bilibili](https://www.bilibili.com/video/BV13Xtg6REZJ/)

## Experimental System

| Component | Experimental configuration |
|---|---|
| Robot | Hiwonder SO-ARM101 follower |
| Camera | Logitech C920 Pro HD Webcam, fixed eye-to-hand view |
| Image stream | 1280 × 720 at 30 FPS |
| Object | 5 cm wooden cube |
| Destination | Fixed open placement platform |
| Robot interface | LeRobot SO-101 follower interface |

The camera, robot calibration, and pose values in this repository correspond
to the physical unit and arrangement used in the experiment. See
[experimental setup](docs/experimental_setup.md) for details.

## Methods

### ArUco

The ArUco pipeline detects marker ID 0 from `DICT_4X4_50`, undistorts the full
frame, and accepts the marker center after a 30-sample rolling buffer contains
at least 20 samples with no more than 2.0 px standard deviation on either
axis. [aruco/controller.py](aruco/controller.py) is the main repository entry point.
The five exact controller files used in the benchmark are retained in
[aruco/benchmark_controllers](aruco/benchmark_controllers).

### YOLO

The YOLO pipeline detects class 0 at image size 640 on CPU with confidence
threshold 0.25 and duplicate-suppression IoU threshold 0.50. It uses the same
30/20-sample temporal lock and 2.0 px stability threshold. The grasp reference
is

```text
x = median bounding-box center x
y = median bounding-box top + 0.25 × median bounding-box height
```

[yolo/controller.py](yolo/controller.py) is the main full-cycle entry
point. [yolo/benchmark_controller.py](yolo/benchmark_controller.py) preserves
the exact controller used for the reported experiment.

### Pixel-to-joint mapping

The mapping uses nine calibrated image-space points and eight piecewise
barycentric triangles. The containing triangle is selected automatically and
the benchmark controller does not extrapolate outside the calibrated mesh.
The experimental data and model are in [mapping](mapping).

### Manipulation controller

The mapping prediction is corrected with eight source anchors. Most targets
use reliability-weighted three-nearest interpolation; targets within the
mid-right to lower-right corridor use pairwise interpolation when their
perpendicular distance is at most 45 px. The correction also interpolates the
above-to-grasp geometry, open-gripper command, and retention parameters.

The resulting sequence is above-target approach, controlled descent, contact
detection from position response, adaptive retention, lift, safe transfer,
two-stage placement, release verification, retreat, and return to safe rest.
The complete implementation is described in [methods](docs/methods.md).

## Experimental Protocol

Each method was evaluated at five workspace locations with four trials per
location. A trial was counted as a full-cycle success when
`grip_contact_accepted`, `release_open_verified`, `cycle_completed`, and
`observed_success` were all true. Cycle-time dispersion is reported as the
sample standard deviation. See [benchmark protocol](docs/benchmark_protocol.md).

## Results

| Metric | ArUco | YOLO |
|---|---:|---:|
| Successful full cycles | 20/20 | 20/20 |
| Mean cycle time | 133.489 s | 133.392 s |
| Median cycle time | 133.276 s | 133.170 s |
| Cycle-time sample SD | 0.752 s | 0.623 s |
| Minimum cycle time | 132.599 s | 132.722 s |
| Maximum cycle time | 135.402 s | 134.551 s |
| Empirical coordinate repeatability, mean | 1.269 px | 1.485 px bbox / 1.503 px grasp reference |
| Empirical coordinate repeatability, maximum | 3.973 px | 5.026 px bbox / 5.057 px grasp reference |

The separate ArUco distance from the stored nominal benchmark centers was
2.362 px mean and 5.805 px maximum. It uses a different reference definition
from the harmonized empirical repeatability values.

## Repository Structure

| Directory | Contents |
|---|---|
| `calibration/` | Camera images and intrinsic results; robot-unit calibration |
| `mapping/` | Nine calibration points, experimental model, builder, validator |
| `robot_control/` | Shared waypoints and source-anchor poses |
| `aruco/` | Portable and exact benchmark ArUco controllers |
| `yolo/` | Portable YOLO controller chain, model, and training metadata |
| `data/yolo/` | 90/15/15 image and label split |
| `benchmarks/` | Forty official trial records, ArUco evidence images, summaries |
| `analysis/` | Read-only result recomputation and method comparison |
| `results/` | Committed comparison outputs and figures |
| `docs/` | Setup, methods, protocol, and reproducibility notes |

## Reproducing the Analysis

Create the environment, then run:

```bash
python analysis/analyze_aruco.py
python analysis/analyze_yolo.py
python analysis/compare_methods.py
```

These commands read the official JSON records and verify the recomputed values
against the committed summaries without modifying either. Further instructions
are in [docs/reproducibility.md](docs/reproducibility.md).

## Running the Experimental Controllers

The main controller entry points are:

```bash
python aruco/controller.py --location 1
python yolo/controller.py
```

The ArUco location number selects only benchmark metadata and the new-trial
output directory. Both programs retain the operator prompts used in the
experiment. The controllers are configured for the SO-ARM101 used in the
experiment. Inspect the camera index, robot connection, calibration, poses,
workspace, and clearances before using them on hardware.

Static equivalence checks are available without importing the hardware stack:

```bash
python aruco/verify_equivalence.py
python yolo/verify_equivalence.py
```

## Dataset and Model

The dataset contains 120 paired images and YOLO labels: 90 training,
15 validation, and 15 test pairs. Training configuration and metrics are in
[yolo/training](yolo/training), and the selected weights are
[yolo/model/best.pt](yolo/model/best.pt). The recorded Ultralytics version is
8.4.121. The original training command and LeRobot Git revision were not
retained.

## Paper

The associated manuscript is in preparation. Citation metadata will be updated
when a persistent paper identifier is available.

## Citation

Use [CITATION.cff](CITATION.cff) for software citation metadata.

## License

This repository is licensed under the [GNU Affero General Public License v3.0](LICENSE).
