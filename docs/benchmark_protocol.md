# Benchmark protocol

## Design

The controlled benchmark used five source locations:

1. lower left;
2. upper left;
3. center;
4. upper right;
5. lower right.

Each method completed four trials at each location, giving 20 ArUco and 20 YOLO
trials. The robot, camera mounting, cube, fixed open placement platform, mapping,
correction layer, manipulation sequence, and outcome fields were held fixed.

## Official records

Official records are stored under `benchmarks/aruco/records` and
`benchmarks/yolo/records`. Each location directory contains exactly four JSON
files. ArUco also stores one detection-evidence image per official trial under
`benchmarks/aruco/images`.

Development and diagnostic runs are not part of these directories. In
particular, the earlier YOLO candidate record timestamped `004605` was not one
of the 20 official trials.

## Outcomes and timing

The harmonized full-cycle success definition requires all four fields to be
true:

- `grip_contact_accepted`
- `release_open_verified`
- `cycle_completed`
- `observed_success`

`observed_success` records the operator's observation that the cube was resting
on the fixed open placement platform after the cycle. Cycle time is read from
`cycle_time_seconds`; standard deviation is the sample standard deviation
using `n - 1`.

## Repeatability

Harmonized image-space repeatability is computed separately for each method
and coordinate type. For every location, the arithmetic mean of its four
official coordinates is the empirical center. Each trial contributes its
Euclidean distance from that center, and the reported summary gives the mean
and maximum over all 20 distances.

The ArUco summary also reports distance from the nominal location centers
stored in its records. That nominal-center metric is distinct from the
harmonized empirical repeatability calculation.
