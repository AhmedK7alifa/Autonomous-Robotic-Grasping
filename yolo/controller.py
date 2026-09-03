"""First full YOLO pick-and-place candidate.

This file is intentionally not imported by the frozen perception scripts.  When
it is deliberately run in the laboratory, it first completes the frozen
camera-only YOLO handoff, closes the camera, prints the complete mapped plan,
and requires an exact activation phrase before the hardware library is even
imported.

The proven FINAL ArUco V3 control functions are not copied or edited here.
They are extracted from the canonical FINAL V3 source with a strict AST
allowlist after the five-reference parity audit in
    control_interface.py has passed. No ArUco perception code,
canonical main function, hardware import, or hardware construction statement
is extracted.
"""

from __future__ import annotations

import ast
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DRY_RUN_FILE = SCRIPT_DIR / "control_interface.py"
FINAL_PIPELINE_FILE = SCRIPT_DIR / "perception_mapping.py"
CANONICAL_CONTROL_FILE = (
    PROJECT_ROOT
    / "aruco"
    / "benchmark_controllers"
    / "location_01.py"
)

SAFE_REST_FILE = (
    PROJECT_ROOT
    / "robot_control"
    / "poses"
    / "waypoints"
    / "safe_rest.json"
)
BOX_HIGH_FILE = (
    PROJECT_ROOT
    / "robot_control"
    / "poses"
    / "waypoints"
    / "box_high_clearance.json"
)
BOX_LOW_RELEASE_FILE = (
    PROJECT_ROOT
    / "robot_control"
    / "poses"
    / "waypoints"
    / "box_low_release.json"
)

TRIAL_DIR = PROJECT_ROOT / "benchmarks" / "yolo" / "new_trials"
ROBOT_PORT = "COM5"
ROBOT_ID = "white_follower"
HARDWARE_ACTIVATION_PHRASE = "RUN FULL YOLO CYCLE"

CONTROL_SUPPORT_ASSIGNMENTS = {"JOINTS", "ARM_JOINTS"}


def load_canonical_control_layer(dry_run) -> SimpleNamespace:
    """Load only parity-verified FINAL V3 constants and control functions."""
    parity = dry_run.audit_final_v3_parity()
    tree = ast.parse(
        CANONICAL_CONTROL_FILE.read_text(encoding="utf-8"),
        filename=str(CANONICAL_CONTROL_FILE),
    )
    wanted_assignments = (
        set(dry_run.EXPECTED_MOTION_CONSTANTS)
        | CONTROL_SUPPORT_ASSIGNMENTS
    )
    wanted_functions = set(dry_run.CONTROL_FUNCTION_NAMES)
    selected_nodes = []
    found_assignments = set()
    found_functions = set()

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted_assignments
        ):
            selected_nodes.append(node)
            found_assignments.add(node.targets[0].id)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in wanted_functions
        ):
            selected_nodes.append(node)
            found_functions.add(node.name)

    if found_assignments != wanted_assignments:
        raise RuntimeError(
            "Canonical control assignments are incomplete: "
            f"{sorted(wanted_assignments - found_assignments)}"
        )

    if found_functions != wanted_functions:
        raise RuntimeError(
            "Canonical control functions are incomplete: "
            f"{sorted(wanted_functions - found_functions)}"
        )

    filtered_tree = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(filtered_tree)
    namespace = {
        "json": json,
        "np": np,
        "time": time,
        "__builtins__": __builtins__,
    }
    exec(
        compile(filtered_tree, str(CANONICAL_CONTROL_FILE), "exec"),
        namespace,
    )

    constants = {
        name: namespace[name]
        for name in dry_run.EXPECTED_MOTION_CONSTANTS
    }

    if constants != dry_run.EXPECTED_MOTION_CONSTANTS:
        raise RuntimeError("Loaded FINAL V3 motion constants failed parity.")

    loaded_functions = {
        name: namespace[name] for name in wanted_functions
    }
    return SimpleNamespace(
        **constants,
        JOINTS=namespace["JOINTS"],
        ARM_JOINTS=namespace["ARM_JOINTS"],
        parity=parity,
        **loaded_functions,
    )


def assert_finite_pose(name: str, pose: dict, joint_names: list[str]) -> None:
    missing = [joint for joint in joint_names if joint not in pose]

    if missing:
        raise RuntimeError(f"{name} is missing joints: {missing}")

    values = np.asarray(
        [float(pose[joint]) for joint in joint_names],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite values.")


def build_candidate_plan(
    locked: dict,
    final_pipeline,
    mapping,
    mapping_model: dict,
    source_anchors: list[dict],
    safe_rest: dict,
    box_high: dict,
    box_low_release: dict,
) -> dict:
    """Convert one frozen YOLO lock into perception-neutral motion inputs."""
    bbox_center = np.asarray(
        locked["median_bbox_center"],
        dtype=np.float64,
    )
    bbox = np.asarray(locked["median_bbox"], dtype=np.float64)
    grasp_point = final_pipeline.final_grasp_point(bbox_center, bbox)
    prediction = mapping.predict(grasp_point, mapping_model)

    if prediction is None:
        raise RuntimeError(
            "Frozen upper-quarter grasp point is outside the validated mesh."
        )

    dynamic = mapping.dynamic_source_poses(
        grasp_point,
        prediction,
        source_anchors,
    )
    frozen_result = final_pipeline.map_grasp_point(
        grasp_point,
        mapping,
        mapping_model,
        source_anchors,
    )

    if not frozen_result["inside_workspace"]:
        raise RuntimeError("Frozen perception pipeline rejected the grasp point.")

    for pose_name, direct_pose in (
        ("above", dynamic["above"]),
        ("grasp", dynamic["grasp"]),
    ):
        frozen_pose = frozen_result[pose_name]

        for joint in mapping.JOINTS:
            if not np.isclose(
                float(frozen_pose[joint]),
                float(direct_pose[joint]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "Frozen/direct mapping parity failed: "
                    f"{pose_name}/{joint}"
                )

    for name, pose in (
        ("source_above", dynamic["above"]),
        ("source_grasp", dynamic["grasp"]),
        ("safe_rest", safe_rest),
        ("box_high", box_high),
        ("box_low_release", box_low_release),
    ):
        assert_finite_pose(name, pose, mapping.JOINTS)

    scalar_values = [
        *bbox_center,
        *bbox,
        *grasp_point,
        *np.asarray(prediction["weights"], dtype=np.float64),
        float(dynamic["open_gripper"]),
        float(dynamic["hold_margin"]),
    ]

    if dynamic["hold_cap"] is not None:
        scalar_values.append(float(dynamic["hold_cap"]))

    for item in dynamic["support"]:
        scalar_values.extend(
            (float(item["distance_px"]), float(item["weight"]))
        )

    if not np.isfinite(np.asarray(scalar_values, dtype=np.float64)).all():
        raise RuntimeError("YOLO-to-motion handoff contains non-finite values.")

    return {
        "bbox_center": bbox_center,
        "bbox": bbox,
        "grasp_point": grasp_point,
        "standard_deviation": np.asarray(
            locked["standard_deviation"],
            dtype=np.float64,
        ),
        "triangle_index": int(prediction["triangle_index"]) + 1,
        "triangle_ids": list(prediction["triangle_ids"]),
        "correction_mode": str(dynamic["source_mode"]),
        "dynamic": dynamic,
        "safe_rest": dict(safe_rest),
        "box_high": dict(box_high),
        "box_low_release": dict(box_low_release),
        "inside_mapping_mesh": True,
    }


def print_candidate_plan(plan: dict, control) -> None:
    center = plan["bbox_center"]
    point = plan["grasp_point"]
    dynamic = plan["dynamic"]
    print("\n" + "=" * 72, flush=True)
    print("FULL YOLO PICK-AND-PLACE CANDIDATE", flush=True)
    print("=" * 72, flush=True)
    print(
        f"YOLO bbox center: ({center[0]:.3f}, {center[1]:.3f}) px",
        flush=True,
    )
    print(
        "Frozen upper-quarter grasp point: "
        f"({point[0]:.3f}, {point[1]:.3f}) px",
        flush=True,
    )
    print(
        f"Mapping triangle: {plan['triangle_index']}/8 - "
        + " / ".join(plan["triangle_ids"]),
        flush=True,
    )
    print(f"Correction mode: {plan['correction_mode']}", flush=True)
    print("Support anchors:", flush=True)

    for item in dynamic["support"]:
        print(
            f"  {item['name']}: weight={item['weight']:.6f}, "
            f"distance={item['distance_px']:.3f}px",
            flush=True,
        )

    print(
        f"Hold policy: margin={dynamic['hold_margin']:.3f}, "
        f"cap={dynamic['hold_cap']}",
        flush=True,
    )
    print(
        f"Retention policy: {dynamic['retention_policy']}",
        flush=True,
    )
    control.print_pose("Computed source ABOVE pose", dynamic["above"])
    control.print_pose("Computed source OPEN-GRASP pose", dynamic["grasp"])
    control.print_pose("Safe-rest pose", plan["safe_rest"])
    control.print_pose("Box-high pose", plan["box_high"])
    control.print_pose("Box-low/release pose", plan["box_low_release"])
    print("Validated mapping mesh: INSIDE", flush=True)


def new_trial_record(plan: dict) -> dict:
    dynamic = plan["dynamic"]
    return {
        "timestamp_start": datetime.now().isoformat(timespec="seconds"),
        "mode": "full_yolo_pick_and_place_candidate",
        "pipeline": (
            "frozen_yolo_nms_lock_upper_quarter_final_v3_mapping_control"
        ),
        "bbox_center_px": [
            round(float(value), 3) for value in plan["bbox_center"]
        ],
        "grasp_point_px": [
            round(float(value), 3) for value in plan["grasp_point"]
        ],
        "center_std_px": [
            round(float(value), 3) for value in plan["standard_deviation"]
        ],
        "triangle_ids": list(plan["triangle_ids"]),
        "correction_mode": plan["correction_mode"],
        "dynamic_support": [
            {
                "name": str(item["name"]),
                "distance_px": round(float(item["distance_px"]), 4),
                "weight": round(float(item["weight"]), 6),
            }
            for item in dynamic["support"]
        ],
        "dynamic_hold_margin": round(float(dynamic["hold_margin"]), 4),
        "dynamic_hold_cap": (
            None
            if dynamic["hold_cap"] is None
            else round(float(dynamic["hold_cap"]), 4)
        ),
        "retention_policy": str(dynamic["retention_policy"]),
        "computed_above_pose": {
            key: round(float(value), 4)
            for key, value in dynamic["above"].items()
        },
        "computed_grasp_pose": {
            key: round(float(value), 4)
            for key, value in dynamic["grasp"].items()
        },
        "cycle_completed": False,
        "grip_contact_accepted": False,
        "observed_success": None,
        "failure_reason": None,
    }


def save_trial_record(record: dict) -> Path:
    """Persist a record only when a future hardware run reaches its finally."""
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = TRIAL_DIR / f"yolo_full_cycle_{timestamp}.json"

    with destination.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")

    return destination


def execute_proven_motion_cycle(
    robot,
    control,
    dynamic: dict,
    safe_rest: dict,
    box_high: dict,
    box_low_release: dict,
    record: dict,
) -> None:
    """Execute only the perception-neutral FINAL V3 motion sequence."""
    source_above = dict(dynamic["above"])
    source_grasp = dict(dynamic["grasp"])
    vertical_open_gripper = float(dynamic["open_gripper"])
    current = control.read_pose(robot)
    problems = control.start_problems(current, safe_rest)

    if problems:
        record["failure_reason"] = "Robot was not near safe_rest."
        print("\nCycle cancelled: robot is not near safe rest.", flush=True)

        for problem in problems:
            print(f"- {problem}", flush=True)

        return

    input(
        "\nComputed target is ready. Inspect the values above. "
        "Press ENTER to start the autonomous cycle..."
    )
    print(f"Starting in {control.START_DELAY:.0f} seconds...", flush=True)
    time.sleep(control.START_DELAY)
    cycle_start = time.time()

    print("\n1/8 Moving to computed ABOVE-CUBE pose...", flush=True)
    control.move_smoothly(
        robot,
        current,
        source_above,
        control.TO_ABOVE_STEPS,
    )
    control.print_pose(
        "Reached computed ABOVE-CUBE pose",
        control.read_pose(robot),
    )
    print("Controlled descent to computed OPEN-GRASP pose...", flush=True)
    control.move_smoothly(
        robot,
        control.read_pose(robot),
        source_grasp,
        control.VERTICAL_DESCENT_STEPS,
    )
    control.print_pose(
        "Reached computed OPEN-GRASP pose",
        control.read_pose(robot),
    )

    print("\n2/8 Grasping cube...", flush=True)
    raw_contact_request, contact_actual, grip_ok = control.adaptive_close(robot)
    record["grip_requested_command"] = round(
        float(raw_contact_request), 4
    )
    record["grip_actual_position"] = round(float(contact_actual), 4)
    record["grip_contact_accepted"] = bool(grip_ok)

    if grip_ok:
        hold_command = max(
            raw_contact_request,
            contact_actual - dynamic["hold_margin"],
        )

        if dynamic["hold_cap"] is not None:
            hold_command = min(float(dynamic["hold_cap"]), hold_command)

        record["grip_hold_command"] = round(float(hold_command), 4)
        print(
            "Gentle hold command: "
            f"{hold_command:.2f} (contact actual={contact_actual:.2f})",
            flush=True,
        )

    if not grip_ok:
        record["failure_reason"] = "Grip contact was not accepted."
        print(
            "\nGrip failed. Returning to safe rest without transfer.",
            flush=True,
        )
        control.move_gripper(robot, vertical_open_gripper, 50)
        above_open = dict(source_above)
        above_open["gripper.pos"] = vertical_open_gripper
        control.move_smoothly(
            robot,
            control.read_pose(robot),
            above_open,
            control.LIFT_BACK_TO_ABOVE_STEPS,
        )
        safe_open = dict(safe_rest)
        safe_open["gripper.pos"] = vertical_open_gripper
        control.move_smoothly(
            robot,
            control.read_pose(robot),
            safe_open,
            control.RETURN_STEPS,
        )
        control.move_gripper(
            robot,
            safe_rest["gripper.pos"],
            control.FINAL_GRIP_STEPS,
        )
        return

    control.hold_grip(robot, hold_command, control.HOLD_SECONDS)

    print("\n3/8 Direct post-grasp lift...", flush=True)
    grasp_hold = control.read_pose(robot)
    grasp_hold["gripper.pos"] = hold_command
    above_hold = dict(source_above)
    above_hold["gripper.pos"] = hold_command
    control.move_smoothly_fixed_grip(
        robot,
        grasp_hold,
        above_hold,
        control.LIFT_BACK_TO_ABOVE_STEPS,
        hold_command,
    )
    control.log_grip_retention(
        record, robot, "post_grasp_lift", hold_command
    )
    control.hold_grip(robot, hold_command, 0.8)

    print("\n4/8 Retracting to safe-rest waypoint...", flush=True)
    safe_hold = dict(safe_rest)
    safe_hold["gripper.pos"] = hold_command
    control.move_smoothly_fixed_grip(
        robot,
        control.read_pose(robot),
        safe_hold,
        control.RETRACT_STEPS,
        hold_command,
    )
    control.log_grip_retention(
        record, robot, "safe_rest_transfer", hold_command
    )
    control.hold_grip(robot, hold_command, 0.8)

    print("\n5/8 Moving above target platform...", flush=True)
    high_hold = dict(box_high)
    high_hold["gripper.pos"] = hold_command
    control.move_smoothly_fixed_grip(
        robot,
        control.read_pose(robot),
        high_hold,
        control.BOX_HIGH_STEPS,
        hold_command,
    )
    control.log_grip_retention(
        record, robot, "box_high_transfer", hold_command
    )
    control.hold_grip(robot, hold_command, 0.8)

    print("\n6/8 Slow two-stage placement...", flush=True)
    low_hold = dict(box_low_release)
    low_hold["gripper.pos"] = hold_command
    placement_start = control.read_pose(robot)
    placement_mid = {
        joint: (
            placement_start[joint]
            + control.PLACEMENT_STAGE1_FACTOR
            * (low_hold[joint] - placement_start[joint])
        )
        for joint in control.JOINTS
    }
    placement_mid["gripper.pos"] = hold_command
    control.move_smoothly_fixed_grip(
        robot,
        placement_start,
        placement_mid,
        control.PLACEMENT_STAGE1_STEPS,
        hold_command,
    )
    control.log_grip_retention(
        record, robot, "placement_stage_1", hold_command
    )
    control.hold_grip(robot, hold_command, 0.5)
    control.move_smoothly_fixed_grip(
        robot,
        control.read_pose(robot),
        low_hold,
        control.PLACEMENT_STAGE2_STEPS,
        hold_command,
    )
    control.log_grip_retention(
        record, robot, "placement_stage_2", hold_command
    )
    control.hold_grip(
        robot,
        hold_command,
        control.PLACEMENT_SETTLE_SECONDS,
    )

    print("\n7/8 Releasing cube on target platform...", flush=True)
    release_hold = dict(box_low_release)
    release_hold["gripper.pos"] = hold_command
    control.move_smoothly_fixed_grip(
        robot,
        control.read_pose(robot),
        release_hold,
        control.RELEASE_STEPS,
        hold_command,
    )
    control.log_grip_retention(record, robot, "pre_release", hold_command)
    release_before, release_after, release_ok = (
        control.open_gripper_and_verify(robot)
    )
    record["release_gripper_before"] = round(float(release_before), 4)
    record["release_gripper_after"] = round(float(release_after), 4)
    record["release_open_verified"] = bool(release_ok)

    if not release_ok:
        record["failure_reason"] = (
            "Gripper release opening was not verified. "
            "Automatic platform retreat was cancelled."
        )
        raise RuntimeError(
            "Release opening was not verified; "
            "automatic platform retreat was cancelled."
        )

    time.sleep(control.RELEASE_SETTLE_SECONDS)
    print("Release confirmed; beginning open-platform retreat...", flush=True)
    platform_retract = dict(box_low_release)
    platform_retract["gripper.pos"] = control.FINAL_RELEASE_OPEN_GRIPPER
    control.move_smoothly(
        robot,
        control.read_pose(robot),
        platform_retract,
        control.RELEASE_STEPS,
    )
    control.print_pose(
        "Reached open-platform local retract waypoint",
        control.read_pose(robot),
    )
    safe_open = dict(safe_rest)
    safe_open["gripper.pos"] = control.FINAL_RELEASE_OPEN_GRIPPER
    control.move_smoothly(
        robot,
        control.read_pose(robot),
        safe_open,
        control.RETURN_STEPS,
    )

    print("\n8/8 Safe-rest retreat completed.", flush=True)
    control.move_gripper(
        robot,
        safe_rest["gripper.pos"],
        control.FINAL_GRIP_STEPS,
    )
    final_pose = control.read_pose(robot)
    record["cycle_completed"] = True
    record["final_robot_pose"] = {
        joint: round(float(final_pose[joint]), 4)
        for joint in control.JOINTS
    }
    record["cycle_time_seconds"] = round(time.time() - cycle_start, 3)
    control.print_pose("Final safe-rest position", final_pose)
    outcome = input(
        "Is the cube resting successfully on the target platform? [Y/N]: "
    ).strip().lower()
    record["observed_success"] = outcome == "y"

    if outcome != "y":
        record["failure_reason"] = (
            "Cycle completed, but cube was not successfully placed "
            "on the target platform."
        )


def run_hardware_cycle(control, plan: dict) -> None:
    """Construct hardware only after the exact activation phrase is entered."""
    from lerobot.robots.so101_follower import (  # noqa: PLC0415
        SO101Follower,
        SO101FollowerConfig,
    )

    record = new_trial_record(plan)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=ROBOT_PORT,
            id=ROBOT_ID,
            max_relative_target=float(control.parity["max_relative_target"]),
        )
    )
    connected = False

    try:
        robot.connect()
        connected = True
        execute_proven_motion_cycle(
            robot,
            control,
            plan["dynamic"],
            plan["safe_rest"],
            plan["box_high"],
            plan["box_low_release"],
            record,
        )
    except KeyboardInterrupt:
        record["failure_reason"] = "Trial interrupted by user."
        print("\nTrial interrupted.", flush=True)
    except Exception as error:
        record["failure_reason"] = str(error)
        print(f"\nERROR: {error}", flush=True)
    finally:
        record["timestamp_end"] = datetime.now().isoformat(
            timespec="seconds"
        )
        record_path = save_trial_record(record)
        print(f"\nTrial record saved to: {record_path}", flush=True)

        if connected:
            input("\nPress ENTER to disconnect the robot...")

            try:
                robot.disconnect()
                print("Robot disconnected safely.", flush=True)
            except Exception as error:
                print(f"WARNING: disconnect failed: {error}", flush=True)


def main() -> None:
    dry_run = importlib.util.spec_from_file_location(
        "yolo_controlled_grasp_dry_run",
        DRY_RUN_FILE,
    )

    if dry_run is None or dry_run.loader is None:
        raise ImportError(f"Could not load required script: {DRY_RUN_FILE}")

    dry_module = importlib.util.module_from_spec(dry_run)
    sys.modules[dry_run.name] = dry_module
    dry_run.loader.exec_module(dry_module)
    control = load_canonical_control_layer(dry_module)
    final_pipeline = dry_module.load_local_module(
        "final_yolo_perception_mapping_candidate",
        FINAL_PIPELINE_FILE,
    )
    live = final_pipeline.load_local_module(
        "cube_live_detector_candidate",
        final_pipeline.LIVE_DETECTOR_FILE,
    )
    mapping_dry_run = final_pipeline.load_local_module(
        "yolo_mapping_dry_run_candidate",
        final_pipeline.MAPPING_DRY_RUN_FILE,
    )
    mapping = mapping_dry_run.load_final_mapping_functions()
    mapping_model = mapping.load_model()
    source_anchors = mapping.load_source_anchors(mapping_model)
    safe_rest = control.load_pose(SAFE_REST_FILE)
    box_high = control.load_pose(BOX_HIGH_FILE)
    box_low_release = control.load_pose(BOX_LOW_RELEASE_FILE)
    yolo_model = live.load_frozen_model()
    locked = dry_module.acquire_one_locked_target(live, yolo_model)
    plan = build_candidate_plan(
        locked,
        final_pipeline,
        mapping,
        mapping_model,
        source_anchors,
        safe_rest,
        box_high,
        box_low_release,
    )
    print_candidate_plan(plan, control)
    print(
        "\nCamera is closed. No hardware has been imported or constructed.",
        flush=True,
    )
    entered = input(
        "To enable the future hardware phase, type exactly "
        f"'{HARDWARE_ACTIVATION_PHRASE}': "
    ).strip()

    if entered != HARDWARE_ACTIVATION_PHRASE:
        print("Hardware phase cancelled; no connection attempted.", flush=True)
        return

    run_hardware_cycle(control, plan)


if __name__ == "__main__":
    main()
