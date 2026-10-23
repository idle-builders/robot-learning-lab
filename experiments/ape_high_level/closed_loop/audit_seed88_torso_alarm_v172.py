"""V172: read-only attribution of the V171 seed-88 11 cm range stop."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from fast_range_guard import scan_plane_gap
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v172"
TRACE = ROOT / "artifacts/ape-closed-loop-cove50-v171/scene88/seed88/trace.json"
FRAME = ROOT / "artifacts/ape-closed-loop-cove50-v171/scene88/seed88/frame-31.npz"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"


def main() -> None:
    trace = json.loads(TRACE.read_text())
    before, stopped = trace["events"][-2:]
    alarm = stopped["detail"]["move_range_alarm"]
    assert alarm["source"] == "torso11" and alarm["action_fraction"] == .5
    points_xy = np.asarray(alarm["upper"]["coherent_points_xy_m"], dtype=float)
    points = np.c_[points_xy, np.full(len(points_xy), .11)]
    delta = np.asarray(stopped["detail"]["command_delta_xy_m"], dtype=float)
    start_est = np.asarray(before["estimated_pose_xyyaw"], dtype=float)
    start_actual = np.asarray(before["evaluator_actual_pose_xyyaw"], dtype=float)
    half_est = start_est.copy()
    half_est[:2] += .5 * delta
    half_actual = start_actual.copy()
    half_actual[:2] += .88 * .5 * delta
    with np.load(FRAME) as saved:
        qpos = saved["actor_qpos"].copy()
        frame_time = float(saved["capture_time_s"])
    assert abs(frame_time - before["time_s"]) < 1e-6

    scenes = json.loads(SCENES.read_text())
    variant = next(v for row in scenes["rows"] if row["seed"] == 88
                   for v in row["passable_variants"] if v["variant"].endswith("-2"))
    room = MovedPropRoom({"CreamCheese": variant["cream_cheese_xy_m"],
                          "Parmesan": variant["parmesan_xy_m"]})
    try:
        room.reset()
        room.layout(88, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, -.46, 0.)
        room.set_table_height(.37)
        room.data.qpos[:] = qpos
        room.head = float(qpos[room.head_addr])
        robot = RobotGeometry(room.model)
        robot.set_state(robot.pose_qpos(qpos, half_est))
        exact = robot.solid_distances(points, robot.primitives())
        rectangle = scan_plane_gap(points_xy, half_est)
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        samples = []
        ray_hits = []
        for fraction in np.linspace(0., 1., 21):
            actual = start_actual.copy()
            actual[:2] += .88 * fraction * delta
            room.actual[:] = actual
            room._forward()
            gap, body, scene = full_body_clearance(room, obstacles, bodies)
            samples.append({"fraction": round(float(fraction), 2),
                            "gap_m": round(gap, 5), "body_geom": body,
                            "scene_geom": scene})
            if abs(fraction - .5) < 1e-6:
                for point in points:
                    direction = point - np.r_[actual[:2], .11]
                    direction[2] = 0.
                    direction /= np.linalg.norm(direction)
                    gid = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(
                        room.model, room.data, np.r_[actual[:2], .11], direction,
                        np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True,
                        room.model.body("torso").id, gid)
                    ray_hits.append({"geom": room.model.geom(int(gid[0])).name
                                     if gid[0] >= 0 else None,
                                     "range_m": round(float(distance), 5)})
    finally:
        room.close()
    result = {
        "scope": "Read-only V171 seed88 stop attribution; 11 cm changed points, neutral rectangle and selected-body solids at actor half-step; independent full-body scene geometry sampled along executed and remaining edge.",
        "source_sha256": sha(SOURCE), "trace_sha256": sha(TRACE),
        "frame_sha256": sha(FRAME), "scene_source_sha256": sha(SCENES),
        "actor_halfstep_pose_xyyaw": half_est.tolist(),
        "evaluator_halfstep_pose_xyyaw": half_actual.tolist(),
        "alarm": alarm,
        "point_rectangle_gap_m": np.round(rectangle, 5).tolist(),
        "point_selected_body_gap_m": np.round(exact, 5).tolist(),
        "point_ray_hits": ray_hits,
        "scene_path_samples": samples,
        "min_full_body_gap_m": min(s["gap_m"] for s in samples),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seed88-torso-alarm-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("rectangle", result["point_rectangle_gap_m"],
          "articulated", result["point_selected_body_gap_m"],
          "ray hits", [h["geom"] for h in ray_hits],
          "full body minimum", result["min_full_body_gap_m"], flush=True)


if __name__ == "__main__":
    main()
