"""V77: actor-depth coverage of independently feasible exact-stop routes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from audit_stop_state_v51 import true_route
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
STOP_RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
LOOK_RUN = ROOT / "artifacts/ape-closed-loop-cove50-v76"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v77"
CONFIG = "chest_rgbd_range_guard_goal_map"


def route_coverage(robot: RobotGeometry, actor_qpos: np.ndarray,
                   voxel: DepthVoxelMap, path: np.ndarray,
                   time_s: float) -> dict:
    segments = []
    for start, end in zip(path[:-1], path[1:]):
        segments.append(robot.swept_exterior_coverage(
            voxel, actor_qpos, start, end, time_s))
    def aggregate(rows: list[dict]) -> dict:
        counts = {key: sum(row[key] for row in rows)
                  for key in ("free", "occupied", "unknown", "samples")}
        counts["unknown_fraction"] = (round(counts["unknown"] /
                                            counts["samples"], 4)
                                      if counts["samples"] else None)
        counts["segments"] = len(rows)
        return counts
    return {"first_eight": aggregate(segments[:8]),
            "full": aggregate(segments)}


def replay(seed: int, room: Room, robot: RobotGeometry) -> dict:
    episode = STOP_RUN / (f"furnished_table_corridor-seed{seed}-0-"
                          f"{CONFIG}-none")
    stop_path = episode / "first-stop-actor-state.npz"
    with np.load(stop_path) as stop:
        actual = stop["actual_pose_xyyaw_evaluator_only"].copy()
        estimated = stop["estimated_pose_xyyaw"].copy()
        target = stop["target_true_xy_m_evaluator_only"].copy()
        actor_qpos = stop["actor_qpos_estimated"].copy()

    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = actor_qpos
    room.actual[:] = actual
    room._forward()
    # Simulator truth is used only after a route has already been chosen, to
    # diagnose what the actor's recorded depth did or did not observe.
    reference = true_route(room, actual, target)
    if not reference["found"]:
        raise AssertionError(f"No 5cm-clear reference route for seed {seed}")
    path = np.asarray(reference["path_xy_m"], dtype=float)
    actor_path = np.column_stack((estimated[:2] + path - actual[:2],
                                  np.full(len(path), estimated[2])))

    voxel = DepthVoxelMap()
    neutral = None
    input_hashes = []
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    for index in range(3):
        frame_path = LOOK_RUN / f"seed{seed}-look{index}-actor-input.npz"
        input_hashes.append(sha(frame_path))
        with np.load(frame_path) as frame:
            qpos = frame["actor_qpos"].copy()
            robot.set_state(qpos)
            solids = robot.primitives()
            time_s = float(frame["capture_time_s"])
            voxel.observe(
                frame["depth_m"],
                (frame["camera_origin_m"], frame["camera_rotation"]),
                time_s, fovy_deg=fovy,
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, solids) < 0.)
        if index == 0:
            neutral = copy.deepcopy(voxel)
            neutral_time = time_s
    assert neutral is not None
    neutral_coverage = route_coverage(
        robot, actor_qpos, neutral, actor_path, neutral_time)
    sweep_coverage = route_coverage(
        robot, actor_qpos, voxel, actor_path, time_s)
    first = sweep_coverage["first_eight"]
    gate = (first["occupied"] == 0 and
            first["unknown_fraction"] is not None and
            first["unknown_fraction"] <= .25 and
            first["unknown"] < neutral_coverage["first_eight"]["unknown"])
    return {"seed": seed, "stop_state_sha256": sha(stop_path),
            "look_actor_input_sha256": input_hashes,
            "reference_start_actual_xyyaw": actual.tolist(),
            "actor_estimated_xyyaw": estimated.tolist(),
            "reference_route": reference,
            "neutral_upper_shell_coverage": neutral_coverage,
            "after_sweep_upper_shell_coverage": sweep_coverage,
            "first_eight_observability_gate_pass": gate}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot) for seed in (85, 88)]
    finally:
        room.close()
    result = {
        "scope": "V76 saved calibrated actor-depth replay; 18-43cm leading selected-body shell coverage along independently solved exact-V72-stop 5cm-clear evaluator routes. Evaluator route never enters actor choice or motion.",
        "source_sha256": sha(SOURCE),
        "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "rows": rows,
        "upper_shell_route_observability_gate_pass": all(
            row["first_eight_observability_gate_pass"] for row in rows)}
    (OUT / "active-route-coverage-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_active_route_coverage_v77_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "route", row["reference_route"]["path_steps"],
              "neutral", row["neutral_upper_shell_coverage"]["first_eight"],
              "sweep", row["after_sweep_upper_shell_coverage"]["first_eight"])
    print("upper-shell route observability gate:",
          result["upper_shell_route_observability_gate_pass"])


if __name__ == "__main__":
    main()
