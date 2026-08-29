# Pixel-to-joint mapping

The [final](final) directory preserves the Day-7 mapping used by both official pipelines:

- 9 calibrated image-space points;
- 8 interpolation triangles;
- piecewise linear barycentric interpolation;
- validated workspace only; and
- no benchmark extrapolation.

[pixel_to_joint_model_day07.json](final/pixel_to_joint_model_day07.json) is the canonical frozen model. [workspace_calibration_points_day07_9pts.json](final/workspace_calibration_points_day07_9pts.json) is a byte-identical copy of the archived Day-7 9-point observation record, renamed only for clarity. The builder and validator are unchanged historical scripts and retain their original filenames/path assumptions.

The mapping model contains the original lab root, camera index, robot ID, and COM port as provenance. These fields are experimentally relevant configuration, not credentials. Do not reuse saved joint positions on different or uncalibrated hardware.
