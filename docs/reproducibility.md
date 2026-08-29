# Reproducibility

## Reproducibility level

This repository supports artifact inspection, hash verification, trial-level result auditing, and regeneration of method summaries. It does not provide physical robot reproduction or detector retraining: robot-actuation controllers and operating instructions are excluded for public safety, hardware calibration is unit-specific, and the raw YOLO images/labels are excluded.

## Original software environment

The experiment was Windows-based. The preserved records report:

| Package/runtime | Recorded version |
|---|---|
| Python | 3.10.18 |
| OpenCV | 5.0.0, package opencv-contrib-python 5.0.0.93 |
| NumPy | 2.2.6 |
| PyTorch | 2.7.1+cpu |
| TorchVision | 0.22.1 |
| Ultralytics | 8.4.121 |
| LeRobot | 0.3.4 reported by environment export |
| Feetech servo SDK | 1.0.0 |

The original LeRobot installation was an editable local checkout. Its exact Git commit was not recorded, so package-version recreation is approximate. requirements.txt and environment.yml intentionally contain only documented direct dependencies rather than an invented full lockfile.

## Environment creation

From an Anaconda/Miniforge shell on Windows:

    conda env create -f environment.yml
    conda activate autonomous-robotic-grasping

Or, from an existing Python 3.10 environment:

    python -m pip install -r requirements.txt

Dependency installation alone does not configure the robot, camera, serial access, or safety system.

## Verify frozen artifacts

The original YOLO freeze manifest includes SHA-256 values for the archived experimental inputs, including the checkpoint, mapping, calibration, poses, support modules, and an internal controller that is not part of this public release. For example, the included checkpoint can be verified in PowerShell:

    Get-FileHash yolo\final\model\best.pt -Algorithm SHA256

Expected checkpoint SHA-256:

    32C32FCEF42B92A5ED3FD37F0023281251FBE50D94342AF65AD6614592BFF7C6

The final public review records source-to-public hash equality for every included copied artifact and separately records safety-withheld controller paths.

## Regenerate method summaries

The 40 official trial JSON files are included. The YOLO method analyzer resolves data relative to its own file. The ArUco analyzer instead expects 06_pixel_to_robot/aruco_final_benchmark_v3 beneath the process working directory. Run analyzers only in a disposable staging tree because they overwrite summary TXT/JSON/CSV files.

For an ArUco audit, copy results/aruco into a temporary staging directory named 06_pixel_to_robot, change the process working directory to the staging root, and run 06_pixel_to_robot/analyze_final_aruco_benchmark.py. For YOLO, copy results/yolo into a temporary 06_yolo directory and run 06_yolo/11_analyze_final_yolo_benchmark.py.

The frozen comparison script also expects the original numbered experimental topology, not this publication layout. It is preserved unchanged in results/comparison for provenance. To regenerate the comparison, populate the expected 06_pixel_to_robot and 06_yolo summary paths plus 07_comparison in the same disposable staging tree. Record the staging procedure and verify the regenerated metrics against the committed comparison files.

## Public safety boundary

The five ArUco benchmark controllers, the YOLO full pick-and-place candidate, and a duplicated controller in the YOLO reference snapshot are retained only in the read-only authoritative research archive. They are deliberately absent from this public repository. `SOURCE_FILE_MAP.tsv` preserves their source and former publication paths with status `removed_public_safety_boundary`, so the provenance record remains auditable without distributing executable actuation code.

The included dry-run scripts stop at perception, coordinate mapping, validation, or plan description and do not connect to or command a robot. This release cannot be used to actuate the experimental robot and provides no robot-enable or physical-motion procedure.

## Camera and mapping reproduction

Raw ChArUco images are excluded, but the calibration script, calibration JSON/NPZ, board definition, accepted-image count, and reprojection error are preserved. A changed camera or mount requires new images and a new calibration.

The mapping directory contains the frozen 9-point model and archived 9-point observation record. These research records are hardware-specific and are not distributed as transferable robot commands. The archived benchmark disabled extrapolation outside its validated triangle mesh.

## YOLO training and test reproduction

The repository includes capture/annotation utilities, deterministic split manifest, original dataset YAML, saved training arguments, epoch history, and best.pt. The raw 120 images and labels are excluded, so training and held-out evaluation cannot be independently repeated from this repository alone.

No standalone one-off held-out-test runner/output directory was found. The official detector metrics are preserved in the final comparison output. This is a provenance gap, not a reason to infer or reconstruct undocumented commands.

## Expected boundaries

A reproducible audit should recover the committed summary values from the included trial records. A physical rerun may differ because of hardware tolerances, calibration, lighting, object placement, software/hardware revisions, and operator procedures. The repository makes no claim of zero-shot performance, unseen-object generalization, universal reliability, or statistical method superiority.
