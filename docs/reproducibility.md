# Reproducibility

## Environment

The experiment used Python 3.10.18. Create the recorded direct dependency
environment with either:

```bash
conda env create -f environment.yml
conda activate autonomous-robotic-grasping
```

or install `requirements.txt` in a Python 3.10 environment. The recorded
LeRobot package version is 0.3.4; the exact commit of the original editable
checkout was not recorded.

## Analysis

The analysis scripts are read-only:

```bash
python analysis/analyze_aruco.py
python analysis/analyze_yolo.py
python analysis/compare_methods.py
```

Each command validates the official record count and metadata, recomputes the
statistics, and checks them against committed outputs.

## Mapping and calibration

The mapping inputs used in the experiment are `mapping/calibration_points.json`
and `mapping/pixel_to_joint_model.json`. To rebuild without replacing the experimental
model:

```bash
python mapping/build_mapping.py
```

Generated files are written under `mapping/generated`. The camera calibration
script similarly writes reconstructed output under
`calibration/camera/generated`.

`mapping/validate_mapping.py` is an interactive camera validation program and
does not command the robot.

## Dataset and model

`data/yolo` contains 90 training, 15 validation, and 15 test image/label pairs.
`yolo/training/cube_dataset.yaml` points to this repository-relative dataset.
The split manifest records the dataset membership and hashes. Training
configuration, epoch metrics, and selected plots are stored beside it.

The inference weights are `yolo/model/best.pt`, SHA-256
`32c32fcef42b92a5ed3fd37f0023281251fbe50d94342af65ad6614592bff7c6`.
The original training command was not retained. The saved training
configuration is available in `args.yaml`.

## Controller entry points

Static equivalence checks:

```bash
python aruco/verify_equivalence.py
python yolo/verify_equivalence.py
```

Physical controller entry points:

```bash
python aruco/controller.py --location 1
python yolo/controller.py
```

The portable ArUco entry point differs from the five benchmark controller files
only in repository-relative paths and location metadata. The portable YOLO
controller uses the benchmark controller functions and resolves its support
chain from the repository. New records are written under
`benchmarks/<method>/new_trials`, separate from the official data.
