"""V83: actor-source and evaluator-edge attribution of V82 stops."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
V82 = ROOT / "artifacts/ape-closed-loop-cove50-v82"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v83"


def replay(seed: int, room: Room, robot: RobotGeometry) -> dict:
    branch = V82 / f"seed{seed}-oracle"
    trace_path = branch / "trace.json"
    trace = json.loads(trace_path.read_text())
    stop = next(event for event in trace["events"]
                if event["action"] == "actor_point_clearance_stop")
    last_frame_number = sum("actor_input_sha256" in event
                            for event in trace["events"])
    frame_paths = [branch / f"frame-{number}.npz"
                   for number in range(1, last_frame_number + 1)]
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    voxel = DepthVoxelMap()
    for frame_path in frame_paths:
        with np.load(frame_path) as frame:
            actor_qpos = frame["actor_qpos"].copy()
            robot.set_state(actor_qpos)
            solids = robot.primitives()
            voxel.observe(
                frame["depth_m"],
                (frame["camera_origin_m"], frame["camera_rotation"]),
                float(frame["capture_time_s"]), fovy_deg=fovy,
                self_hit=lambda hit: RobotGeometry.solid_distance(
                    hit, solids) < 0.)
            time_s = float(frame["capture_time_s"])
    estimated = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(trace["final_actual_pose_xyyaw_evaluator_only"],
                        dtype=float)
    desired_xy = np.asarray(stop["desired_xy_m"], dtype=float)
    scan11 = stop["range_11_m"]
    scan8 = stop["range_8_m"]
    scan4 = stop["range_4_m"]
    actor_state_path = OUT / f"seed{seed}-actor-stop-input.npz"
    np.savez_compressed(
        actor_state_path,
        voxel_free=np.asarray([[*key, stamp]
                               for key, stamp in voxel.free.items()],
                              dtype=float).reshape(-1, 4),
        voxel_occupied=np.asarray([[*key, stamp]
                                   for key, stamp in voxel.occupied.items()],
                                  dtype=float).reshape(-1, 4),
        range_11_m=np.asarray([np.nan if value is None else value
                               for value in scan11]),
        range_8_m=np.asarray([np.nan if value is None else value
                              for value in scan8]),
        range_4_m=np.asarray([np.nan if value is None else value
                              for value in scan4]),
        actor_qpos=actor_qpos,
        estimated_pose_xyyaw=estimated,
        capture_time_s=time_s,
        blocked_next_xy_m=np.asarray(stop["desired_xy_m"], dtype=float))
    point = TwoHeightClearance(
        robot, actor_qpos, estimated, scan11, scan4, voxel, time_s,
        range_8=scan8)
    replay_gap = point.edge_gap(estimated[:2], desired_xy, estimated[2])
    source_gaps = {}
    for source, hits in point.sources.items():
        gap = math.inf
        for fraction in (.25, .5, .75, 1.):
            future = estimated.copy()
            future[:2] += fraction * (desired_xy - estimated[:2])
            robot.set_state(robot.pose_qpos(actor_qpos, future))
            if len(hits):
                gap = min(gap, float(np.min(RobotGeometry.solid_distances(
                    hits, robot.primitives()))))
        source_gaps[source] = (None if math.isinf(gap) else round(gap, 4))
    if abs(replay_gap - stop["predicted_gap_m"]) > .001:
        raise AssertionError("V83 actor point gap does not replay V82 stop")

    room.data.qpos[:] = actor_qpos
    room.actual[:] = actual
    room.estimated[:] = estimated
    room.head = float(actor_qpos[room.head_addr])
    room.time_s = time_s
    command_delta = desired_xy - estimated[:2]
    true_gaps = []
    for substep in range(21):
        room.actual[:] = actual
        room.actual[:2] += .88 * command_delta * substep / 20
        room._forward()
        true_gaps.append(float(room._current_obstacle_clearance()[0]))
    evaluator_gap = min(true_gaps)
    return {"seed": seed,
            "v82_trace_sha256": sha(trace_path),
            "actor_stop_input_sha256": sha(actor_state_path),
            "depth_frame_sha256": [sha(path) for path in frame_paths],
            "replayed_actor_gap_m": round(replay_gap, 4),
            "recorded_actor_gap_m": stop["predicted_gap_m"],
            "actor_gap_reproduced": True,
            "actor_source_counts": point.source_counts(),
            "actor_min_gap_by_sensor_m": source_gaps,
            "actor_estimated_pose_xyyaw": estimated.tolist(),
            "actor_desired_next_xy_m": desired_xy.tolist(),
            "evaluator_actual_start_xyyaw": actual.tolist(),
            "evaluator_rejected_edge_20ms_min_clearance_m": round(
                evaluator_gap, 4),
            "evaluator_clearance_by_20ms_sample_m": [
                round(gap, 4) for gap in true_gaps],
            "rejected_edge_model_clear_5cm": evaluator_gap >= .05}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot) for seed in (85, 88)]
    finally:
        room.close()
    result = {"scope": "Exact V82 point-stop replay by actor-only 4/8/11cm and head-depth source, followed by evaluator-only 20ms signed selected-model distance on the rejected next command. No guard threshold changes or movement authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "both_rejected_edges_model_clear_5cm": all(
                  row["rejected_edge_model_clear_5cm"] for row in rows)}
    (OUT / "oracle-stop-attribution-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_oracle_stop_v83_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "actor", row["replayed_actor_gap_m"],
              "sources", row["actor_min_gap_by_sensor_m"],
              "true rejected edge", row[
                  "evaluator_rejected_edge_20ms_min_clearance_m"])


if __name__ == "__main__":
    main()
