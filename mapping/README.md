# Pixel-to-joint mapping

The experimental mapping uses `calibration_points.json`, the Day-07
nine-point snapshot, and the eight triangles defined in `build_mapping.py`.
`pixel_to_joint_model.json` is the model loaded by both full-cycle controllers.
It disables extrapolation outside the calibrated mesh.

Running `build_mapping.py` writes a reconstructed model and visualization to
`mapping/generated`; it does not overwrite the experimental model. The interactive
`validate_mapping.py` checks image-space locking and mapping output without
connecting to the robot.
