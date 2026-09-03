# Methods

## ArUco perception

The ArUco controller detects marker ID 0 from OpenCV `DICT_4X4_50`. It keeps a
30-frame center buffer, requires at least 20 valid samples, and accepts the
target when the standard deviation is no greater than 2.0 px on either image
axis. The stable coordinate is the component-wise median marker center.

## YOLO perception

The detector uses class 0, image size 640, CPU inference, confidence threshold
0.25, and greedy duplicate suppression at IoU 0.50. Detections are ordered by
confidence before suppression. The temporal lock uses 30 samples, requires 20,
and applies the same 2.0 px per-axis stability criterion.

For the median bounding box `(x1, y1, x2, y2)`, the grasp reference is:

```text
x = median bounding-box center x
y = y1 + 0.25 × (y2 - y1)
```

## Pixel-to-joint mapping

The Day-07 mapping contains nine calibrated image-space points and eight
piecewise-linear barycentric triangles:

```text
[P01, P02, P05]  [P01, P05, P04]
[P02, P03, P06]  [P02, P06, P05]
[P04, P05, P08]  [P04, P08, P07]
[P05, P06, P09]  [P05, P09, P08]
```

The containing triangle is selected automatically. Barycentric weights
interpolate the recorded joint coordinates, and points outside the mesh return
no prediction. Benchmark extrapolation is disabled.

The nine-point snapshot in `mapping/calibration_points.json` is the input that
produced `mapping/pixel_to_joint_model.json`. A later sixteen-point development
dataset was not used in the reported experiment.

## Dynamic source-anchor correction

The barycentric prediction is the global baseline. Eight anchors supply local
grasp-pose correction, above-to-grasp geometry, open-gripper position,
reliability, and retention parameters:

- `upper_left_raw`
- `upper_mid_validated`
- `mid_left_raw`
- `mid_right_validated`
- `mid_lower_validated`
- `lower_left_raw`
- `lower_center_validated`
- `lower_right_validated`

The normal rule uses the three nearest anchors with inverse-square distance and
anchor-reliability weighting. A target within 3.0 px of one anchor uses that
anchor directly. Targets more than 120 px from their nearest anchor are
reported as sparsely supported.

The right-side corridor joins `mid_right_validated` and
`lower_right_validated`. A target whose projection lies on that segment and
whose perpendicular distance is at most 45 px uses linear pairwise weights
instead of the three-nearest rule. This corridor caps retained grip at 62.5.
When `upper_mid_validated` contributes at least 0.50 weight, the hold margin is
at least 3.5 and the hold cap is 63.0.

## Manipulation sequence

After target locking and operator confirmation, the controller:

1. moves from safe rest to the corrected above-target pose;
2. descends to the corrected grasp pose;
3. closes incrementally and detects contact from commanded-versus-observed
   gripper response and stall behavior;
4. applies the interpolated positional-retention command;
5. lifts while keeping the retained gripper command fixed;
6. transfers through safe rest and box-high clearance;
7. performs two-stage descent to the release pose;
8. opens and verifies release from measured gripper position;
9. retreats with the gripper open and returns to safe rest.

YOLO extracts the same mapping and motion functions from the ArUco
benchmark controller after static parity checks. There is no benchmark-location
specific motion branch in either method.
