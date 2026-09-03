import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


ROOT = Path(r"D:\Projects\Autonomous_Robotic_Grasping")
PIXEL_DIR = ROOT / "06_pixel_to_robot"
POSES_DIR = ROOT / "03_robot_control" / "poses"

MODEL_FILE = PIXEL_DIR / "pixel_to_joint_model_day07.json"
CAM_CAL_FILE = (
    ROOT
    / "04_calibration"
    / "camera_intrinsics"
    / "camera_calibration.npz"
)

SAFE_REST_FILE = (
    POSES_DIR
    / "safe_rest_final.json"
)

LOWER_LEFT_ABOVE_FILE = (
    POSES_DIR
    / "upper_mid_interp_above_v1.json"
)

LOWER_LEFT_GRASP_FILE = (
    POSES_DIR
    / "upper_mid_interp_grasp_v1.json"
)

BOX_HIGH_FILE = (
    POSES_DIR
    / "box_high_clearance_final.json"
)

BOX_LOW_FILE = (
    POSES_DIR
    / "box_low_release_final.json"
)

BOX_RELEASE_FILE = (
    POSES_DIR
    / "box_low_release_final.json"
)

BOX_EXIT_FILE = (
    POSES_DIR
    / "box_high_exit_final.json"
)

TRIAL_DIR = (
    PIXEL_DIR
    / "aruco_final_benchmark_v3"
    / "location_2_upper_left"
)


PORT = "COM5"
ROBOT_ID = "white_follower"

CAMERA_INDEX = 1
WIDTH = 1280
HEIGHT = 720
FPS = 30

TARGET_ID = 0


JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

ARM_JOINTS = [
    joint
    for joint in JOINTS
    if joint != "gripper.pos"
]


SAMPLE_COUNT = 30
MIN_SAMPLES = 20
MAX_STD_PX = 2.0


START_TOL = 8.0

# This full-cycle file is ONLY for validating the newly recorded
# lower-left vertical approach at the same source location where
# the two poses were recorded.
LOWER_LEFT_REFERENCE_PX = np.array(
    [220.0, 144.5],
    dtype=np.float64,
)
LOWER_LEFT_MAX_DISTANCE_PX = 8.0

# Saved lower-left approach uses a wider open gripper.
VERTICAL_OPEN_GRIPPER = 73.17
TO_ABOVE_STEPS = 240
VERTICAL_DESCENT_STEPS = 180
LIFT_BACK_TO_ABOVE_STEPS = 180

# Final approach offset measured during Day 07 alignment.
# Applied relative to the dynamically predicted grasp pose across the valid mapped region.

PREGRASP_OFFSETS = {
    "shoulder_pan.pos": 0.37,
    "shoulder_lift.pos": 14.14,
    "elbow_flex.pos": -22.34,
    "wrist_flex.pos": 20.72,
    "wrist_roll.pos": -0.86,
}

FINAL_APPROACH_OPEN_GRIPPER = 69.75
PREGRASP_STEPS = 220
FINAL_APPROACH_STEPS = 140

MIN_CLOSE = 48.0

CONTACT_ACTUAL_MAX = 70.5
CONTACT_REQUEST_MAX = 64.5
CONTACT_GAP_MIN = 2.5

GRIP_STEP = 1.0
GRIP_DELAY = 0.25

STALL_THRESHOLD = 0.12
STALL_COUNT = 3


APPROACH_STEPS = 200
PRELIFT_STEPS = 120
RETRACT_STEPS = 240

BOX_HIGH_STEPS = 220
BOX_LOW_STEPS = 160
RELEASE_STEPS = 80

PLACEMENT_STAGE1_FACTOR = 0.60
PLACEMENT_STAGE1_STEPS = 160
PLACEMENT_STAGE2_STEPS = 180
PLACEMENT_SETTLE_SECONDS = 0.8
BOX_EXIT_STEPS = 220

# Explicit release control.
# 69.75 is the already-tested open gripper value used during approach.
FINAL_RELEASE_OPEN_GRIPPER = 72.00
RELEASE_OPEN_STEPS = 80
RELEASE_OPEN_DELAY = 0.05
RELEASE_OPEN_MIN = 71.0
RELEASE_MIN_OPEN_DELTA = 0.8
RELEASE_SETTLE_SECONDS = 2.0

# Gentle holding pressure after contact. The gripper closes toward
# smaller position values. Candidate V3 keeps the global behavior,
# but strengthens retention only when the validated upper-mid anchor
# has dominant geometric support. This is a region-level policy, not
# a benchmark-location special case.
HOLD_PRESSURE_MARGIN = 2.5
UPPER_MID_RETENTION_WEIGHT_MIN = 0.50
UPPER_MID_HOLD_MARGIN = 3.5
UPPER_MID_HOLD_CAP = 63.0


RETURN_STEPS = 220
FINAL_GRIP_STEPS = 60

MOVE_DELAY = 0.05

PRELIFT_FACTOR = 0.15
HOLD_SECONDS = 1.5
START_DELAY = 3.0


def load_pose(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required pose file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    positions = data.get(
        "positions"
    )

    if not isinstance(
        positions,
        dict,
    ):
        raise ValueError(
            f"Invalid pose file: {path}"
        )

    missing = [
        joint
        for joint in JOINTS
        if joint not in positions
    ]

    if missing:
        raise KeyError(
            f"Missing joints in {path.name}: "
            + ", ".join(missing)
        )

    return {
        joint: float(
            positions[joint]
        )
        for joint in JOINTS
    }


def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Mapping model not found: {MODEL_FILE}"
        )

    with MODEL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        model = json.load(file)

    expected = (
        "piecewise_linear_"
        "barycentric_"
        "pixel_to_joint"
    )

    if model.get(
        "model_type"
    ) != expected:

        raise ValueError(
            "Unexpected mapping model type."
        )

    if (
        len(
            model.get(
                "points",
                {},
            )
        )
        != 9
    ):
        raise ValueError(
            "Day 07 model must contain "
            "9 calibration points."
        )

    if (
        len(
            model.get(
                "triangles",
                [],
            )
        )
        != 8
    ):
        raise ValueError(
            "Day 07 model must contain "
            "8 interpolation triangles."
        )

    return model


def load_camera_calibration():

    if not CAM_CAL_FILE.exists():
        raise FileNotFoundError(
            "Camera calibration file "
            f"not found: {CAM_CAL_FILE}"
        )

    calibration = np.load(
        CAM_CAL_FILE,
        allow_pickle=False,
    )

    camera_matrix = np.asarray(
        calibration[
            "camera_matrix"
        ],
        dtype=np.float64,
    )

    if (
        "distortion_coefficients"
        in calibration
    ):
        distortion_key = (
            "distortion_coefficients"
        )

    elif (
        "dist_coeffs"
        in calibration
    ):
        distortion_key = (
            "dist_coeffs"
        )

    else:
        raise KeyError(
            "Distortion coefficients "
            "were not found."
        )

    distortion = np.asarray(
        calibration[
            distortion_key
        ],
        dtype=np.float64,
    ).reshape(-1)

    return (
        camera_matrix,
        distortion,
    )


def read_pose(robot):

    observation = (
        robot.get_observation()
    )

    return {
        joint: float(
            observation[joint]
        )
        for joint in JOINTS
    }


def print_pose(
    title,
    pose,
):

    print(
        f"\n{title}"
    )

    print(
        "-" * 50
    )

    for joint in JOINTS:

        print(
            f"{joint:18s}: "
            f"{pose[joint]:8.2f}"
        )

    print(
        "-" * 50
    )


def smooth(progress):

    return (
        3.0
        * progress
        * progress
        -
        2.0
        * progress
        * progress
        * progress
    )


def move_smoothly(
    robot,
    start,
    target,
    steps,
    delay=MOVE_DELAY,
):

    for step in range(
        1,
        steps + 1,
    ):

        progress = smooth(
            step / steps
        )

        action = {
            joint: (
                start[joint]
                +
                progress
                * (
                    target[joint]
                    -
                    start[joint]
                )
            )
            for joint in JOINTS
        }

        robot.send_action(
            action
        )

        time.sleep(
            delay
        )



def move_smoothly_fixed_grip(
    robot,
    start,
    target,
    steps,
    hold_command,
    delay=MOVE_DELAY,
):
    """Move arm joints smoothly while commanding one fixed grip value."""

    for step in range(1, steps + 1):
        progress = smooth(step / steps)

        action = {
            joint: (
                start[joint]
                + progress * (target[joint] - start[joint])
            )
            for joint in ARM_JOINTS
        }
        action["gripper.pos"] = float(hold_command)

        robot.send_action(action)
        time.sleep(delay)


def log_grip_retention(record, robot, label, hold_command):
    actual = float(read_pose(robot)["gripper.pos"])
    gap = actual - float(hold_command)

    print(
        f"Grip retention check [{label}]: "
        f"command={hold_command:.2f}, "
        f"actual={actual:.2f}, "
        f"gap={gap:.2f}"
    )

    record.setdefault("grip_retention_checks", []).append(
        {
            "stage": str(label),
            "command": round(float(hold_command), 4),
            "actual": round(actual, 4),
            "gap": round(gap, 4),
        }
    )

    return actual


def start_problems(
    current,
    safe_rest,
):

    problems = []

    for joint in ARM_JOINTS:

        difference = abs(
            current[joint]
            -
            safe_rest[joint]
        )

        if (
            difference
            > START_TOL
        ):

            problems.append(
                f"{joint}: "
                f"difference="
                f"{difference:.2f} "
                "degrees"
            )

    return problems


def create_detector():

    dictionary = (
        cv2.aruco
        .getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )

    parameters = (
        cv2.aruco
        .DetectorParameters()
    )

    if hasattr(
        cv2.aruco,
        "ArucoDetector",
    ):

        return (
            cv2.aruco
            .ArucoDetector(
                dictionary,
                parameters,
            )
        )

    return (
        dictionary,
        parameters,
    )


def detect(
    detector,
    gray,
):

    if hasattr(
        detector,
        "detectMarkers",
    ):

        return (
            detector
            .detectMarkers(
                gray
            )
        )

    dictionary, parameters = (
        detector
    )

    return (
        cv2.aruco
        .detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )
    )


def target_marker(
    corners,
    ids,
):

    if ids is None:
        return None

    for (
        index,
        marker_id,
    ) in enumerate(
        ids.flatten()
    ):

        if (
            int(marker_id)
            != TARGET_ID
        ):
            continue

        points = np.asarray(
            corners[index],
            dtype=np.float64,
        ).reshape(
            4,
            2,
        )

        center = points.mean(
            axis=0
        )

        return (
            points,
            center,
        )

    return None


def barycentric(
    point,
    point_a,
    point_b,
    point_c,
):

    matrix = (
        np.column_stack(
            (
                point_a
                -
                point_c,

                point_b
                -
                point_c,
            )
        )
    )

    determinant = float(
        np.linalg.det(
            matrix
        )
    )

    if (
        abs(
            determinant
        )
        < 1e-9
    ):
        return None

    result = np.linalg.solve(
        matrix,
        point
        -
        point_c,
    )

    weight_a = float(
        result[0]
    )

    weight_b = float(
        result[1]
    )

    weight_c = (
        1.0
        -
        weight_a
        -
        weight_b
    )

    return np.array(
        [
            weight_a,
            weight_b,
            weight_c,
        ],
        dtype=np.float64,
    )


def predict(
    center,
    model,
):

    points = (
        model[
            "points"
        ]
    )

    tolerance = 1e-6

    for (
        triangle_index,
        triangle_ids,
    ) in enumerate(
        model[
            "triangles"
        ]
    ):

        vertices = [
            np.asarray(
                points[
                    point_id
                ][
                    "center_px"
                ],
                dtype=np.float64,
            )
            for point_id
            in triangle_ids
        ]

        weights = (
            barycentric(
                center,
                vertices[0],
                vertices[1],
                vertices[2],
            )
        )

        if weights is None:
            continue

        inside = (
            np.all(
                weights
                >= -tolerance
            )
            and
            np.all(
                weights
                <= 1.0
                + tolerance
            )
        )

        if not inside:
            continue

        predicted_joints = {}

        for joint in JOINTS:

            values = np.asarray(
                [
                    float(
                        points[
                            point_id
                        ][
                            "robot_joint_positions"
                        ][joint]
                    )
                    for point_id
                    in triangle_ids
                ],
                dtype=np.float64,
            )

            predicted_joints[
                joint
            ] = float(
                np.dot(
                    weights,
                    values,
                )
            )

        return {
            "triangle_index":
                triangle_index,

            "triangle_ids":
                triangle_ids,

            "weights":
                weights,

            "predicted_joints":
                predicted_joints,
        }

    return None


def draw_status(
    frame,
    center,
    standard_deviation,
    sample_count,
    prediction,
    stable,
):

    if center is None:

        lines = [
            "ArUco ID 0 not detected",
            (
                "Place cube inside "
                "Day 07 mapped region"
            ),
        ]

    else:

        lines = [
            (
                "ID 0 center: "
                f"({center[0]:.1f}, "
                f"{center[1]:.1f})"
            ),
            (
                "Samples: "
                f"{sample_count}/"
                f"{SAMPLE_COUNT}  "
                "Std: "
                f"("
                f"{standard_deviation[0]:.2f}, "
                f"{standard_deviation[1]:.2f}"
                ") px"
            ),
        ]

        if prediction is None:

            lines.append(
                "OUTSIDE DAY 07 "
                "MAPPED REGION"
            )

        else:

            lines.extend(
                [
                    (
                        "VALID INSIDE "
                        "DAY 07 REGION"
                    ),
                    (
                        "Triangle: "
                        +
                        " / ".join(
                            prediction[
                                "triangle_ids"
                            ]
                        )
                    ),
                    (
                        "Stable: "
                        +
                        (
                            "YES"
                            if stable
                            else "WAIT"
                        )
                    ),
                ]
            )

    lines.append(
        "S: lock target   "
        "Q: cancel"
    )

    for (
        index,
        line,
    ) in enumerate(
        lines
    ):

        y_position = (
            32
            +
            index
            * 28
        )

        cv2.putText(
            frame,
            line,
            (
                20,
                y_position,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (
                0,
                0,
                0,
            ),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            line,
            (
                20,
                y_position,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )


def acquire_target(
    model,
):

    (
        camera_matrix,
        distortion,
    ) = (
        load_camera_calibration()
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        FPS,
    )

    if not camera.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"index {CAMERA_INDEX}."
        )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if (
        actual_width
        != WIDTH
        or
        actual_height
        != HEIGHT
    ):

        camera.release()

        raise RuntimeError(
            "Unexpected camera resolution: "
            f"{actual_width}"
            "x"
            f"{actual_height}"
        )

    (
        map_x,
        map_y,
    ) = (
        cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            None,
            camera_matrix,
            (
                WIDTH,
                HEIGHT,
            ),
            cv2.CV_32FC1,
        )
    )

    detector = (
        create_detector()
    )

    samples = deque(
        maxlen=SAMPLE_COUNT
    )

    locked = None

    print(
        "\nLive ArUco "
        "acquisition started."
    )

    print(
        "Place the 5 cm cube "
        "with ArUco ID 0 "
        "inside the mapped region."
    )

    try:

        while True:

            success, raw_frame = (
                camera.read()
            )

            if not success:

                raise RuntimeError(
                    "Could not read "
                    "camera frame."
                )

            frame = cv2.remap(
                raw_frame,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
            )

            display_frame = (
                frame.copy()
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            corners, ids, _ = (
                detect(
                    detector,
                    gray,
                )
            )

            marker = (
                target_marker(
                    corners,
                    ids,
                )
            )

            center = None

            prediction = None

            standard_deviation = (
                np.array(
                    [
                        0.0,
                        0.0,
                    ],
                    dtype=np.float64,
                )
            )

            stable = False

            if marker is None:

                samples.clear()

            else:

                (
                    marker_points,
                    raw_center,
                ) = marker

                samples.append(
                    raw_center.copy()
                )

                sample_array = (
                    np.asarray(
                        samples,
                        dtype=np.float64,
                    )
                )

                center = np.median(
                    sample_array,
                    axis=0,
                )

                standard_deviation = (
                    sample_array.std(
                        axis=0
                    )
                )

                prediction = predict(
                    center,
                    model,
                )

                stable = (
                    len(samples)
                    >= MIN_SAMPLES
                    and
                    float(
                        standard_deviation.max()
                    )
                    <= MAX_STD_PX
                    and
                    prediction is not None
                )

                polygon = (
                    marker_points
                    .round()
                    .astype(
                        np.int32
                    )
                    .reshape(
                        (
                            -1,
                            1,
                            2,
                        )
                    )
                )

                cv2.polylines(
                    display_frame,
                    [
                        polygon
                    ],
                    True,
                    (
                        0,
                        255,
                        0,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                center_point = (
                    int(
                        round(
                            center[0]
                        )
                    ),
                    int(
                        round(
                            center[1]
                        )
                    ),
                )

                cv2.circle(
                    display_frame,
                    center_point,
                    7,
                    (
                        0,
                        0,
                        255,
                    ),
                    -1,
                    cv2.LINE_AA,
                )

            draw_status(
                display_frame,
                center,
                standard_deviation,
                len(samples),
                prediction,
                stable,
            )

            cv2.imshow(
                (
                    "DAY 07 LIVE ARUCO "
                    "PICK-AND-PLACE"
                ),
                display_frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord(
                "q"
            ):

                raise KeyboardInterrupt

            if key == ord(
                "s"
            ):

                if not stable:

                    print(
                        "Target is not "
                        "stable and valid yet."
                    )

                    continue

                locked = (
                    center.copy(),
                    standard_deviation.copy(),
                    prediction,
                    display_frame.copy(),
                )

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print(
            "Camera closed safely."
        )

    if locked is None:

        raise RuntimeError(
            "No valid target "
            "was locked."
        )

    return locked


def adaptive_close(
    robot,
):

    current = read_pose(
        robot
    )

    previous_actual = (
        current[
            "gripper.pos"
        ]
    )

    requested_target = (
        previous_actual
    )

    stall_count = 0

    print(
        "\nClosing gradually "
        "until object contact "
        "is detected..."
    )

    while (
        requested_target
        > MIN_CLOSE
    ):

        requested_target = max(
            MIN_CLOSE,
            requested_target
            -
            GRIP_STEP,
        )

        current = read_pose(
            robot
        )

        action = dict(
            current
        )

        action[
            "gripper.pos"
        ] = requested_target

        robot.send_action(
            action
        )

        time.sleep(
            GRIP_DELAY
        )

        actual_gripper = (
            read_pose(
                robot
            )[
                "gripper.pos"
            ]
        )

        movement = abs(
            actual_gripper
            -
            previous_actual
        )

        print(
            f"requested="
            f"{requested_target:6.2f}  "
            f"actual="
            f"{actual_gripper:6.2f}  "
            f"movement="
            f"{movement:5.2f}"
        )

        if (
            movement
            < STALL_THRESHOLD
            and
            requested_target
            <
            actual_gripper
            - 0.4
        ):

            stall_count += 1

        else:

            stall_count = 0

        previous_actual = (
            actual_gripper
        )

        if (
            stall_count
            >= STALL_COUNT
        ):

            print(
                "\nObject contact "
                "detected."
            )

            break

    actual_gripper = (
        read_pose(
            robot
        )[
            "gripper.pos"
        ]
    )

    contact_gap = (
        actual_gripper
        -
        requested_target
    )

    grip_ok = (
        actual_gripper
        <= CONTACT_ACTUAL_MAX
        and
        requested_target
        <= CONTACT_REQUEST_MAX
        and
        contact_gap
        >= CONTACT_GAP_MIN
    )

    print(
        "\nFinal requested "
        "grip command: "
        f"{requested_target:.2f}"
    )

    print(
        "Final measured "
        "gripper position: "
        f"{actual_gripper:.2f}"
    )

    print(
        "Contact gap: "
        f"{contact_gap:.2f}"
    )

    if grip_ok:

        print(
            "Grip contact accepted."
        )

    else:

        print(
            "Grip contact "
            "was not accepted."
        )

    return (
        requested_target,
        actual_gripper,
        grip_ok,
    )


def hold_grip(
    robot,
    command,
    seconds,
):

    end_time = (
        time.time()
        +
        seconds
    )

    while (
        time.time()
        <
        end_time
    ):

        current = read_pose(
            robot
        )

        action = dict(
            current
        )

        action[
            "gripper.pos"
        ] = command

        robot.send_action(
            action
        )

        time.sleep(
            0.08
        )


def move_gripper(
    robot,
    target,
    steps=60,
    delay=0.05,
):

    start = read_pose(
        robot
    )

    end = dict(
        start
    )

    end[
        "gripper.pos"
    ] = target

    move_smoothly(
        robot,
        start,
        end,
        steps,
        delay,
    )


def open_gripper_and_verify(
    robot,
):
    """
    Open the gripper explicitly while keeping the arm pose fixed.
    The function verifies measured gripper motion before allowing
    the robot to retract from the target placement platform.
    """

    before = read_pose(
        robot
    )[
        "gripper.pos"
    ]

    print(
        "\nOpening gripper explicitly "
        "for release..."
    )

    move_gripper(
        robot,
        FINAL_RELEASE_OPEN_GRIPPER,
        RELEASE_OPEN_STEPS,
        RELEASE_OPEN_DELAY,
    )

    time.sleep(
        RELEASE_SETTLE_SECONDS
    )

    after = read_pose(
        robot
    )[
        "gripper.pos"
    ]

    delta = (
        after
        -
        before
    )

    print(
        "Release gripper: "
        f"before={before:.2f}, "
        f"after={after:.2f}, "
        f"opened_by={delta:.2f}"
    )

    release_ok = (
        after
        >= RELEASE_OPEN_MIN
        and
        delta
        >= RELEASE_MIN_OPEN_DELTA
    )

    if not release_ok:

        print(
            "Release opening was not "
            "confirmed. Retrying once..."
        )

        retry_before = after

        move_gripper(
            robot,
            FINAL_RELEASE_OPEN_GRIPPER,
            RELEASE_OPEN_STEPS,
            RELEASE_OPEN_DELAY,
        )

        time.sleep(
            RELEASE_SETTLE_SECONDS
        )

        after = read_pose(
            robot
        )[
            "gripper.pos"
        ]

        total_delta = (
            after
            -
            before
        )

        retry_delta = (
            after
            -
            retry_before
        )

        print(
            "Release retry: "
            f"after={after:.2f}, "
            f"retry_opened_by={retry_delta:.2f}, "
            f"total_opened_by={total_delta:.2f}"
        )

        release_ok = (
            after
            >= RELEASE_OPEN_MIN
            and
            total_delta
            >= RELEASE_MIN_OPEN_DELTA
        )

    if release_ok:

        print(
            "Release opening verified."
        )

    else:

        print(
            "RELEASE OPENING NOT VERIFIED."
        )

    return (
        before,
        after,
        release_ok,
    )


def save_trial(
    record,
    evidence,
):

    TRIAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    json_path = (
        TRIAL_DIR
        /
        (
            "aruco_trial_"
            f"{timestamp}.json"
        )
    )

    if evidence is not None:

        image_path = (
            TRIAL_DIR
            /
            (
                "aruco_trial_"
                f"{timestamp}.jpg"
            )
        )

        if cv2.imwrite(
            str(
                image_path
            ),
            evidence,
        ):

            record[
                "detection_evidence_image"
            ] = str(
                image_path
            )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            record,
            file,
            indent=2,
        )

    return json_path



# ------------------------------------------------------------------
# Unified workspace source-pose field
# ------------------------------------------------------------------
#
# The Day 07 9-point barycentric mapping remains the global baseline.
# These local anchors contribute only correction terms and staged
# above-to-grasp geometry. The robot never stores a unique pose for
# the unseen test point itself.
#
# Mandatory anchors are the four currently validated development
# locations used by the supplied scripts. Additional raw anchors are
# loaded when their pose files are present, improving coverage without
# making them mandatory for startup.

SOURCE_ANCHOR_SPECS = [
    {
        "name": "upper_left_raw",
        "px": [240.5, 97.0],
        "above_file": "upper_left_above_cube_v1.json",
        "grasp_file": "upper_left_grasp_v2.json",
        "open_gripper": 73.17,
        "reliability": 0.75,
        "hold_margin": 2.5,
        "hold_cap": None,
        "optional": True,
    },
    {
        "name": "upper_mid_validated",
        "px": [220.0, 144.5],
        "above_file": "upper_mid_interp_above_v1.json",
        "grasp_file": "upper_mid_interp_grasp_v1.json",
        "open_gripper": 73.17,
        "reliability": 1.00,
        "hold_margin": 2.5,
        "hold_cap": 65.0,
        "optional": False,
    },
    {
        "name": "mid_left_raw",
        "px": [197.8, 196.2],
        "above_file": "mid_left_above_cube_v2.json",
        "grasp_file": "mid_left_grasp_v2.json",
        "open_gripper": 73.17,
        "reliability": 1.00,
        "hold_margin": 2.5,
        "hold_cap": None,
        "optional": True,
    },
    {
        "name": "mid_right_validated",
        "px": [350.25, 196.50],
        "above_file": "mid_right_above_cube_v1.json",
        "grasp_file": "mid_right_grasp_v1.json",
        "open_gripper": 73.55,
        "reliability": 1.00,
        "hold_margin": 1.2,
        "hold_cap": None,
        "optional": False,
    },
    {
        "name": "mid_lower_validated",
        "px": [172.0, 251.0],
        "above_file": "left_interp_above_v1.json",
        "grasp_file": "left_interp_grasp_v1.json",
        "open_gripper": 74.60,
        "reliability": 1.00,
        "hold_margin": 2.5,
        "hold_cap": None,
        "optional": False,
    },
    {
        "name": "lower_left_raw",
        "px": [146.0, 305.5],
        "above_file": "lower_left_above_cube_v2.json",
        "grasp_file": "lower_left_grasp_v3.json",
        "open_gripper": 75.67,
        "reliability": 0.85,
        "hold_margin": 2.5,
        "hold_cap": None,
        "optional": True,
    },
    {
        "name": "lower_center_validated",
        "px": [296.5, 301.2],
        "above_file": "lower_center_above_cube_v1.json",
        "grasp_file": "lower_center_grasp_v1.json",
        "open_gripper": 71.99,
        "reliability": 1.00,
        "hold_margin": 1.2,
        "hold_cap": None,
        "optional": False,
    },
    {
        "name": "lower_right_validated",
        "px": [428.2, 299.0],
        "above_file": "lower_right_above_cube_v1.json",
        "grasp_file": "lower_right_grasp_v1.json",
        "open_gripper": 71.84,
        "reliability": 1.00,
        "hold_margin": 1.2,
        "hold_cap": None,
        "optional": False,
    },
]

DYNAMIC_K_NEAREST = 3
EXACT_ANCHOR_PX = 3.0
SPARSE_SUPPORT_WARNING_PX = 120.0

# Geometric right-side transition corridor. Targets close to the line
# segment joining the validated mid-right and lower-right anchors use
# only that pair. This is a region-level rule, not a memorized test point.
RIGHT_PAIR_A = "mid_right_validated"
RIGHT_PAIR_B = "lower_right_validated"
RIGHT_PAIR_MAX_PERP_PX = 45.0
RIGHT_PAIR_T_MIN = 0.0
RIGHT_PAIR_T_MAX = 1.0


def load_source_anchors(model):
    anchors = []

    for spec in SOURCE_ANCHOR_SPECS:
        above_path = POSES_DIR / spec["above_file"]
        grasp_path = POSES_DIR / spec["grasp_file"]

        if not above_path.exists() or not grasp_path.exists():
            if spec["optional"]:
                print(
                    "Optional source anchor skipped because pose file "
                    f"is missing: {spec['name']}"
                )
                continue

            missing = []
            if not above_path.exists():
                missing.append(str(above_path))
            if not grasp_path.exists():
                missing.append(str(grasp_path))

            raise FileNotFoundError(
                "Required unified-workspace anchor file missing: "
                + " | ".join(missing)
            )

        above = load_pose(above_path)
        grasp = load_pose(grasp_path)
        px = np.asarray(spec["px"], dtype=np.float64)

        mapping = predict(px, model)
        if mapping is None:
            raise ValueError(
                f"Anchor {spec['name']} lies outside the Day 07 model."
            )

        mapping_pose = {
            joint: float(mapping["predicted_joints"][joint])
            for joint in JOINTS
        }

        grasp_correction = {
            joint: float(grasp[joint] - mapping_pose[joint])
            for joint in ARM_JOINTS
        }

        above_delta = {
            joint: float(above[joint] - grasp[joint])
            for joint in ARM_JOINTS
        }

        anchors.append(
            {
                "name": spec["name"],
                "px": px,
                "above": above,
                "grasp": grasp,
                "open_gripper": float(spec["open_gripper"]),
                "reliability": float(spec["reliability"]),
                "hold_margin": float(spec["hold_margin"]),
                "hold_cap": spec["hold_cap"],
                "grasp_correction": grasp_correction,
                "above_delta": above_delta,
            }
        )

    if len(anchors) < 4:
        raise RuntimeError(
            "Unified workspace requires at least four source anchors."
        )

    print("\nUnified source anchors loaded:")
    for anchor in anchors:
        print(
            f"- {anchor['name']}: "
            f"({anchor['px'][0]:.1f}, {anchor['px'][1]:.1f})"
        )

    return anchors


def dynamic_source_poses(center, prediction, anchors):
    """
    Build a source pose from the Day 07 global mapping plus local
    correction terms.

    V4 changes only the interpolation rule inside one geometric
    right-side corridor. Targets close to the segment between the
    validated mid-right and lower-right anchors use linear pairwise
    interpolation between those two anchors. Elsewhere the V3
    three-nearest correction field is unchanged.
    """

    center = np.asarray(center, dtype=np.float64)

    distances = np.asarray(
        [
            float(np.linalg.norm(center - anchor["px"]))
            for anchor in anchors
        ],
        dtype=np.float64,
    )

    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])

    name_to_index = {
        anchor["name"]: index
        for index, anchor in enumerate(anchors)
    }

    pairwise_mode = False
    pairwise_t = None
    pairwise_perp = None

    if (
        RIGHT_PAIR_A in name_to_index
        and RIGHT_PAIR_B in name_to_index
        and nearest_distance > EXACT_ANCHOR_PX
    ):
        ia = name_to_index[RIGHT_PAIR_A]
        ib = name_to_index[RIGHT_PAIR_B]
        a = anchors[ia]["px"]
        b = anchors[ib]["px"]
        segment = b - a
        denom = float(np.dot(segment, segment))

        if denom > 1e-9:
            t_raw = float(np.dot(center - a, segment) / denom)
            t_clamped = float(np.clip(t_raw, 0.0, 1.0))
            projected = a + t_clamped * segment
            perp = float(np.linalg.norm(center - projected))

            if (
                RIGHT_PAIR_T_MIN <= t_raw <= RIGHT_PAIR_T_MAX
                and perp <= RIGHT_PAIR_MAX_PERP_PX
            ):
                pairwise_mode = True
                pairwise_t = t_raw
                pairwise_perp = perp
                chosen_indices = [ia, ib]
                normalized_weights = np.asarray(
                    [1.0 - t_raw, t_raw],
                    dtype=np.float64,
                )

    if not pairwise_mode:
        if nearest_distance <= EXACT_ANCHOR_PX:
            chosen_indices = [nearest_index]
            normalized_weights = np.asarray([1.0], dtype=np.float64)
        else:
            order = np.argsort(distances)
            chosen_indices = [
                int(index)
                for index in order[: min(DYNAMIC_K_NEAREST, len(anchors))]
            ]

            raw_weights = np.asarray(
                [
                    anchors[index]["reliability"]
                    / max(distances[index], 1.0) ** 2
                    for index in chosen_indices
                ],
                dtype=np.float64,
            )

            normalized_weights = raw_weights / raw_weights.sum()

    base_mapping = {
        joint: float(prediction["predicted_joints"][joint])
        for joint in JOINTS
    }

    grasp = dict(base_mapping)

    for joint in ARM_JOINTS:
        correction = sum(
            float(weight) * anchors[index]["grasp_correction"][joint]
            for index, weight in zip(chosen_indices, normalized_weights)
        )
        grasp[joint] = base_mapping[joint] + correction

    above = dict(grasp)

    for joint in ARM_JOINTS:
        delta = sum(
            float(weight) * anchors[index]["above_delta"][joint]
            for index, weight in zip(chosen_indices, normalized_weights)
        )
        above[joint] = grasp[joint] + delta

    open_gripper = sum(
        float(weight) * anchors[index]["open_gripper"]
        for index, weight in zip(chosen_indices, normalized_weights)
    )

    above["gripper.pos"] = float(open_gripper)
    grasp["gripper.pos"] = float(open_gripper)

    support = [
        {
            "name": anchors[index]["name"],
            "distance_px": float(distances[index]),
            "weight": float(weight),
        }
        for index, weight in zip(chosen_indices, normalized_weights)
    ]

    hold_margin = sum(
        float(weight) * anchors[index]["hold_margin"]
        for index, weight in zip(chosen_indices, normalized_weights)
    )

    upper_mid_weight = sum(
        float(weight)
        for index, weight in zip(chosen_indices, normalized_weights)
        if anchors[index]["name"] == "upper_mid_validated"
    )

    # Candidate V3 adaptive retention policy for the upper-mid region.
    # Smaller position values mean tighter closure. When upper-mid has
    # dominant geometric support, increase the positional hold margin
    # and limit the hold command to 63.0 deg. This is driven by source
    # support weight, not by benchmark location name or trial number.
    if upper_mid_weight >= UPPER_MID_RETENTION_WEIGHT_MIN:
        hold_margin = max(float(hold_margin), UPPER_MID_HOLD_MARGIN)
        hold_cap = UPPER_MID_HOLD_CAP
    else:
        hold_cap = None

    # Candidate V2 regional retention policy:
    # the right-pairwise corridor reached/grasped reliably, but a
    # development trial slipped during target-platform descent while
    # holding at ~65.7 deg. Use the already-tested 62.5 deg positional
    # hold cap only in this general corridor. Smaller gripper position
    # means tighter closure. No torque/current/safety limit is changed.
    if pairwise_mode:
        if hold_cap is None:
            hold_cap = 62.5
        else:
            hold_cap = min(float(hold_cap), 62.5)

    return {
        "above": above,
        "grasp": grasp,
        "open_gripper": float(open_gripper),
        "support": support,
        "nearest_anchor_distance_px": nearest_distance,
        "base_mapping": base_mapping,
        "hold_margin": float(hold_margin),
        "hold_cap": hold_cap,
        "upper_mid_support_weight": float(upper_mid_weight),
        "retention_policy": (
            "upper_mid_adaptive_stronger_hold"
            if upper_mid_weight >= UPPER_MID_RETENTION_WEIGHT_MIN
            else (
                "right_pairwise_hold_cap"
                if pairwise_mode
                else "default_dynamic_hold"
            )
        ),
        "source_mode": (
            "right_pairwise_linear"
            if pairwise_mode
            else "three_nearest_correction"
        ),
        "right_pairwise_t": pairwise_t,
        "right_pairwise_perp_px": pairwise_perp,
    }


def main():

    model = load_model()

    safe_rest = load_pose(
        SAFE_REST_FILE
    )

    source_anchors = load_source_anchors(
        model
    )

    box_high = load_pose(
        BOX_HIGH_FILE
    )

    box_low = load_pose(
        BOX_LOW_FILE
    )

    # Unified development uses the low release pose that fixed the
    # intermittent drop during the unseen interpolation tests.
    box_release = load_pose(
        BOX_RELEASE_FILE
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "FINAL ARUCO BENCHMARK V3 - LOCATION 2 - UPPER LEFT"
    )

    print(
        "=" * 72
    )

    (
        center,
        standard_deviation,
        prediction,
        evidence,
    ) = acquire_target(
        model
    )

    dynamic = dynamic_source_poses(
        center,
        prediction,
        source_anchors,
    )

    mapping_grasp_pose = dynamic[
        "base_mapping"
    ]

    source_above = dynamic[
        "above"
    ]

    source_grasp = dynamic[
        "grasp"
    ]

    vertical_open_gripper = dynamic[
        "open_gripper"
    ]

    print(
        "\nLive target locked successfully."
    )

    print(
        "Center: "
        f"({center[0]:.2f}, "
        f"{center[1]:.2f})"
    )

    print(
        "Triangle: "
        +
        " / ".join(
            prediction[
                "triangle_ids"
            ]
        )
    )

    print(
        "\nCandidate V3 policy:"
    )
    print(
        "- General dynamic mapping / right-pairwise interpolation only"
    )
    print(
        "- No centered-v2 point-specific grasp pose"
    )
    print(
        "- Upper-mid dominant support: stronger adaptive hold (cap 63.0 deg)"
    )
    print(
        "- Right-pairwise corridor hold cap: 62.5 deg"
    )
    print(
        "- Grip command held fixed during lift and transfer"
    )
    print(
        "- Direct post-grasp lift"
    )
    print(
        "- Slow two-stage target-platform placement"
    )

    print(
        "\nDynamic correction support:"
    )
    print(
        f"Source interpolation mode: {dynamic.get('source_mode', 'unknown')}"
    )
    if dynamic.get("source_mode") == "right_pairwise_linear":
        print(
            "Right-pair projection: "
            f"t={dynamic['right_pairwise_t']:.3f}, "
            f"perp={dynamic['right_pairwise_perp_px']:.2f}px"
        )

    for item in dynamic[
        "support"
    ]:
        print(
            f"- {item['name']}: "
            f"distance={item['distance_px']:.2f}px, "
            f"weight={item['weight']:.3f}"
        )

    print(
        "Nearest anchor distance: "
        f"{dynamic['nearest_anchor_distance_px']:.2f} px"
    )

    print(
        "Dynamic hold policy: "
        f"margin={dynamic['hold_margin']:.2f}, "
        f"cap={dynamic['hold_cap']}"
    )
    print(
        "Retention policy: "
        f"{dynamic['retention_policy']} "
        f"(upper_mid_weight={dynamic['upper_mid_support_weight']:.3f})"
    )

    if (
        dynamic[
            "nearest_anchor_distance_px"
        ]
        >
        SPARSE_SUPPORT_WARNING_PX
    ):
        print(
            "DEVELOPMENT WARNING: target is in a sparsely supported "
            "part of the workspace. Inspect the computed poses before "
            "pressing ENTER."
        )

    print_pose(
        "Day 07 mapping prediction (REFERENCE ONLY)",
        mapping_grasp_pose,
    )

    print_pose(
        "Computed unified ABOVE-CUBE pose",
        source_above,
    )

    print_pose(
        "Computed unified OPEN-GRASP pose",
        source_grasp,
    )

    record = {
        "timestamp_start":
            datetime.now()
            .isoformat(
                timespec="seconds"
            ),

        "mode":
            "final_aruco_benchmark_v3_location_2",

        "benchmark_location":
            "L2_upper_left",

        "benchmark_nominal_center_px":
            [237.0, 134.5],

        "benchmark_code_version":
            "V3_adaptive_retention_frozen",

        "marker_id":
            TARGET_ID,

        "center_px":
            [
                round(float(center[0]), 3),
                round(float(center[1]), 3),
            ],

        "center_std_px":
            [
                round(float(standard_deviation[0]), 3),
                round(float(standard_deviation[1]), 3),
            ],

        "triangle_ids":
            list(
                prediction[
                    "triangle_ids"
                ]
            ),

        "dynamic_support":
            [
                {
                    "name": item["name"],
                    "distance_px": round(
                        float(item["distance_px"]),
                        4,
                    ),
                    "weight": round(
                        float(item["weight"]),
                        6,
                    ),
                }
                for item in dynamic[
                    "support"
                ]
            ],

        "nearest_anchor_distance_px":
            round(
                float(
                    dynamic[
                        "nearest_anchor_distance_px"
                    ]
                ),
                4,
            ),

        "dynamic_hold_margin":
            round(
                float(dynamic["hold_margin"]),
                4,
            ),

        "dynamic_hold_cap":
            dynamic["hold_cap"],

        "upper_mid_support_weight":
            round(float(dynamic["upper_mid_support_weight"]), 6),

        "retention_policy":
            dynamic["retention_policy"],

        "mapping_grasp_pose_reference_only":
            {
                joint:
                    round(
                        float(mapping_grasp_pose[joint]),
                        4,
                    )
                for joint
                in JOINTS
            },

        "computed_above_pose":
            {
                joint:
                    round(
                        float(source_above[joint]),
                        4,
                    )
                for joint
                in JOINTS
            },

        "computed_grasp_pose":
            {
                joint:
                    round(
                        float(source_grasp[joint]),
                        4,
                    )
                for joint
                in JOINTS
            },

        "cycle_completed":
            False,

        "grip_contact_accepted":
            False,

        "observed_success":
            None,

        "failure_reason":
            None,
    }

    robot = SO101Follower(
        SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            max_relative_target=8.0,
        )
    )

    connected = False
    cycle_start = None

    try:

        robot.connect()

        connected = True

        current = read_pose(
            robot
        )

        problems = (
            start_problems(
                current,
                safe_rest,
            )
        )

        if problems:

            record[
                "failure_reason"
            ] = (
                "Robot was not near safe_rest."
            )

            print(
                "\nCycle cancelled: "
                "robot is not near safe rest."
            )

            for problem in problems:
                print(
                    f"- {problem}"
                )

            return

        input(
            "\nComputed target is ready. "
            "Inspect the values above. "
            "Press ENTER to start the autonomous cycle..."
        )

        print(
            f"Starting in "
            f"{START_DELAY:.0f} seconds..."
        )

        time.sleep(
            START_DELAY
        )

        cycle_start = (
            time.time()
        )

        print(
            "\n1/8 Moving to computed "
            "ABOVE-CUBE pose..."
        )

        move_smoothly(
            robot,
            current,
            source_above,
            TO_ABOVE_STEPS,
        )

        print_pose(
            "Reached computed ABOVE-CUBE pose",
            read_pose(robot),
        )

        print(
            "Controlled staged descent to computed "
            "OPEN-GRASP pose..."
        )

        move_smoothly(
            robot,
            read_pose(robot),
            source_grasp,
            VERTICAL_DESCENT_STEPS,
        )

        print_pose(
            "Reached computed OPEN-GRASP pose",
            read_pose(robot),
        )

        print(
            "\n2/8 Grasping cube..."
        )

        (
            raw_contact_request,
            contact_actual,
            grip_ok,
        ) = adaptive_close(
            robot
        )

        record[
            "grip_requested_command"
        ] = round(
            float(raw_contact_request),
            4,
        )

        record[
            "grip_actual_position"
        ] = round(
            float(contact_actual),
            4,
        )

        record[
            "grip_contact_accepted"
        ] = bool(
            grip_ok
        )

        if grip_ok:
            # Preserve region-adaptive holding. Candidate V3 uses a
            # stronger upper-mid policy when that anchor has dominant
            # geometric support; the rule is independent of benchmark
            # location names.
            hold_command = max(
                raw_contact_request,
                contact_actual
                -
                dynamic["hold_margin"],
            )

            if dynamic["hold_cap"] is not None:
                hold_command = min(
                    float(dynamic["hold_cap"]),
                    hold_command,
                )

            record[
                "grip_hold_command"
            ] = round(
                float(hold_command),
                4,
            )

            print(
                "Gentle hold command: "
                f"{hold_command:.2f} "
                f"(contact actual={contact_actual:.2f})"
            )

        if not grip_ok:

            record[
                "failure_reason"
            ] = (
                "Grip contact was not accepted."
            )

            print(
                "\nGrip failed. "
                "Returning to safe rest "
                "without transfer."
            )

            move_gripper(
                robot,
                vertical_open_gripper,
                50,
            )

            above_open = dict(
                source_above
            )

            above_open[
                "gripper.pos"
            ] = vertical_open_gripper

            move_smoothly(
                robot,
                read_pose(robot),
                above_open,
                LIFT_BACK_TO_ABOVE_STEPS,
            )

            safe_open = dict(
                safe_rest
            )

            safe_open[
                "gripper.pos"
            ] = vertical_open_gripper

            move_smoothly(
                robot,
                read_pose(robot),
                safe_open,
                RETURN_STEPS,
            )

            move_gripper(
                robot,
                safe_rest[
                    "gripper.pos"
                ],
                FINAL_GRIP_STEPS,
            )

            return

        hold_grip(
            robot,
            hold_command,
            HOLD_SECONDS,
        )

        print(
            "\n3/8 Direct post-grasp lift to computed ABOVE-CUBE pose..."
        )

        grasp_hold = read_pose(
            robot
        )

        grasp_hold[
            "gripper.pos"
        ] = hold_command

        above_hold = dict(
            source_above
        )

        above_hold[
            "gripper.pos"
        ] = hold_command

        move_smoothly_fixed_grip(
            robot,
            grasp_hold,
            above_hold,
            LIFT_BACK_TO_ABOVE_STEPS,
            hold_command,
        )
        log_grip_retention(record, robot, "post_grasp_lift", hold_command)

        hold_grip(
            robot,
            hold_command,
            0.8,
        )

        print(
            "\n4/8 Retracting "
            "to safe-rest waypoint..."
        )

        safe_hold = dict(
            safe_rest
        )

        safe_hold[
            "gripper.pos"
        ] = hold_command

        move_smoothly_fixed_grip(
            robot,
            read_pose(robot),
            safe_hold,
            RETRACT_STEPS,
            hold_command,
        )
        log_grip_retention(record, robot, "safe_rest_transfer", hold_command)

        hold_grip(
            robot,
            hold_command,
            0.8,
        )

        print(
            "\n5/8 Moving above target platform..."
        )

        high_hold = dict(
            box_high
        )

        high_hold[
            "gripper.pos"
        ] = hold_command

        move_smoothly_fixed_grip(
            robot,
            read_pose(robot),
            high_hold,
            BOX_HIGH_STEPS,
            hold_command,
        )
        log_grip_retention(record, robot, "box_high_transfer", hold_command)

        hold_grip(
            robot,
            hold_command,
            0.8,
        )

        print(
            "\n6/8 Lowering cube to target platform "
            "with slow two-stage placement..."
        )

        low_hold = dict(
            box_low
        )

        low_hold[
            "gripper.pos"
        ] = hold_command

        placement_start = read_pose(
            robot
        )

        placement_mid = {
            joint: (
                placement_start[joint]
                + PLACEMENT_STAGE1_FACTOR
                * (
                    low_hold[joint]
                    - placement_start[joint]
                )
            )
            for joint in JOINTS
        }

        placement_mid[
            "gripper.pos"
        ] = hold_command

        print(
            "Placement stage 1/2: descending "
            f"{PLACEMENT_STAGE1_FACTOR * 100:.0f}% toward low placement pose..."
        )

        move_smoothly_fixed_grip(
            robot,
            placement_start,
            placement_mid,
            PLACEMENT_STAGE1_STEPS,
            hold_command,
        )
        log_grip_retention(record, robot, "placement_stage_1", hold_command)

        hold_grip(
            robot,
            hold_command,
            0.5,
        )

        print(
            "Placement stage 2/2: slow final descent "
            "to low placement pose..."
        )

        move_smoothly_fixed_grip(
            robot,
            read_pose(robot),
            low_hold,
            PLACEMENT_STAGE2_STEPS,
            hold_command,
        )
        log_grip_retention(record, robot, "placement_stage_2", hold_command)

        hold_grip(
            robot,
            hold_command,
            PLACEMENT_SETTLE_SECONDS,
        )

        print(
            "\n7/8 Releasing cube on target platform..."
        )

        release_hold = dict(
            box_release
        )

        release_hold[
            "gripper.pos"
        ] = hold_command

        move_smoothly_fixed_grip(
            robot,
            read_pose(robot),
            release_hold,
            RELEASE_STEPS,
            hold_command,
        )
        log_grip_retention(record, robot, "pre_release", hold_command)

        (
            release_before,
            release_after,
            release_ok,
        ) = open_gripper_and_verify(
            robot
        )

        record[
            "release_gripper_before"
        ] = round(
            float(release_before),
            4,
        )

        record[
            "release_gripper_after"
        ] = round(
            float(release_after),
            4,
        )

        record[
            "release_open_verified"
        ] = bool(
            release_ok
        )

        if not release_ok:

            record[
                "failure_reason"
            ] = (
                "Gripper release opening was not verified. "
                "Automatic platform retreat was cancelled."
            )

            raise RuntimeError(
                "Release opening was not verified; "
                "automatic platform retreat was cancelled."
            )

        time.sleep(
            RELEASE_SETTLE_SECONDS
        )

        print(
            "Release confirmed. "
            "Beginning open-platform retreat "
            "with the gripper held open..."
        )

        platform_retract = dict(
            box_low
        )

        platform_retract[
            "gripper.pos"
        ] = FINAL_RELEASE_OPEN_GRIPPER

        print(
            "\nOpen-platform retract 1/2: "
            "clearing the released cube..."
        )

        move_smoothly(
            robot,
            read_pose(robot),
            platform_retract,
            RELEASE_STEPS,
        )

        print_pose(
            "Reached open-platform local retract waypoint",
            read_pose(robot),
        )

        print(
            "\nOpen-platform retract 2/2: "
            "returning toward safe rest..."
        )

        safe_open = dict(
            safe_rest
        )

        safe_open[
            "gripper.pos"
        ] = FINAL_RELEASE_OPEN_GRIPPER

        move_smoothly(
            robot,
            read_pose(robot),
            safe_open,
            RETURN_STEPS,
        )

        print(
            "\n8/8 Safe-rest retreat completed."
        )

        print(
            "Closing gripper at safe rest..."
        )

        move_gripper(
            robot,
            safe_rest[
                "gripper.pos"
            ],
            FINAL_GRIP_STEPS,
        )

        final_pose = read_pose(
            robot
        )

        record[
            "cycle_completed"
        ] = True

        record[
            "final_robot_pose"
        ] = {
            joint:
                round(
                    float(final_pose[joint]),
                    4,
                )
            for joint
            in JOINTS
        }

        record[
            "cycle_time_seconds"
        ] = round(
            time.time()
            -
            cycle_start,
            3,
        )

        print_pose(
            "Final safe-rest position",
            final_pose,
        )

        print(
            "\nAutonomous unified ArUco "
            "cycle completed."
        )

        outcome = input(
            "Is the cube resting successfully "
            "on the target platform? [Y/N]: "
        ).strip().lower()

        record[
            "observed_success"
        ] = (
            outcome
            == "y"
        )

        if outcome != "y":

            record[
                "failure_reason"
            ] = (
                "Cycle completed, but cube was not "
                "successfully placed on the target platform."
            )

    except KeyboardInterrupt:

        record[
            "failure_reason"
        ] = (
            "Trial interrupted by user."
        )

        print(
            "\nTrial interrupted."
        )

    except Exception as error:

        record[
            "failure_reason"
        ] = str(
            error
        )

        print(
            f"\nERROR: {error}"
        )

    finally:

        record[
            "timestamp_end"
        ] = (
            datetime.now()
            .isoformat(
                timespec="seconds"
            )
        )

        record_path = (
            save_trial(
                record,
                evidence,
            )
        )

        print(
            "\nTrial record "
            "saved to: "
            f"{record_path}"
        )

        if connected:

            input(
                "\nPress ENTER "
                "to disconnect "
                "the robot..."
            )

            try:

                robot.disconnect()

                print(
                    "Robot disconnected safely."
                )

            except Exception as error:

                print(
                    "WARNING: disconnect failed: "
                    f"{error}"
                )


if __name__ == "__main__":
    main()
