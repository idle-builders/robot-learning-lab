"""Actor-only alternate-route screen from V252p's stopped sensor packets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_blocked_range_mounts_v319 import add_packet
from depth_voxel import VOXEL_M
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, model_with_room, sha
from run_medium_depth_recovery_v246p import MediumWindowMap


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v252p"
OPTIONS = {"forward": (.08, 0.), "back": (-.08, 0.),
           "left": (0., .08), "right": (0., -.08)}
POINT_MARGIN_M = .165


def point_gap(robot: RobotGeometry, qpos: np.ndarray, pose: np.ndarray,
              offset: tuple[float, float], points: np.ndarray) -> float:
    if not len(points):
        return float("inf")
    smallest = float("inf")
    for fraction in np.linspace(0., 1., 21):
        candidate = pose.copy()
        candidate[:2] += fraction * np.asarray(offset)
        robot.set_state(robot.pose_qpos(qpos, candidate))
        smallest = min(smallest, float(np.min(
            RobotGeometry.solid_distances(points, robot.primitives()))))
    return smallest


def replay_stage(stage: dict, model: mujoco.MjModel,
                 robot: RobotGeometry) -> dict:
    voxel_map = MediumWindowMap()
    for entry in stage["head_inputs"]:
        path = ROOT / entry["actor_frame_path"]
        assert sha(path) == entry["actor_frame_sha256"]
        with np.load(path) as packet:
            qpos = packet["actor_qpos"].copy()
            pose = packet["estimated_pose_xyyaw"].copy()
            time_s = float(packet["capture_time_s"])
            camera_pose = (packet["camera_origin_estimated_m"].copy(),
                           packet["camera_rotation_estimated"].copy())
            fovy = float(model.cam_fovy[model.camera("lower_rgb").id])
            voxel_map.observe(packet["map_depth_m"], camera_pose, time_s,
                              fovy_deg=fovy, stride=4, max_depth_m=1.,
                              self_masked=True)
    gate = stage["actor_gate"]
    range_path = ROOT / gate["packet_path"]
    assert sha(range_path) == gate["packet_sha256"]
    near = json.loads(range_path.read_text())
    time_s = float(near["capture_time_s"])
    pose = np.asarray(near["estimated_pose_xyyaw"], dtype=float)
    for rays in near["ray_packets"].values():
        add_packet(voxel_map, rays, time_s)
    shin_path = ROOT / gate["shin_frame_path"]
    assert sha(shin_path) == gate["shin_frame_sha256"]
    with np.load(shin_path) as packet:
        shin = packet["estimated_environment_points_xyz_m"].copy()
        qpos = packet["actor_qpos"].copy()
        np.testing.assert_allclose(packet["estimated_pose_xyyaw"], pose)
        assert np.isclose(float(packet["capture_time_s"]), time_s)
    occupied = np.asarray([[(x + .5) * VOXEL_M,
                            (y + .5) * VOXEL_M,
                            (z + .5) * VOXEL_M]
                           for (x, y, z), stamp in voxel_map.occupied.items()
                           if stamp >= time_s - .4 and
                           stamp >= voxel_map.free.get((x, y, z), -math.inf)
                           and (z + .5) * VOXEL_M >= .04],
                          dtype=float).reshape(-1, 3)
    result = {}
    for name, offset in OPTIONS.items():
        future = pose.copy()
        future[:2] += offset
        shell = robot.swept_exterior_coverage(
            voxel_map, qpos, pose, future, time_s, margin_m=.05)
        upper_gap = point_gap(robot, qpos, pose, offset, occupied)
        shin_gap = point_gap(robot, qpos, pose, offset, shin)
        result[name] = {"status": "free" if (
            shell["status"] == "free" and
            min(upper_gap, shin_gap) >= POINT_MARGIN_M) else "blocked",
            "shell_plus_5cm": shell,
            "upper_occupied_point_gap_m": round(upper_gap, 5),
            "shin_point_gap_m": round(shin_gap, 5),
            "point_motion_margin_m": POINT_MARGIN_M}
    assert result["forward"]["status"] == gate["status"]
    assert np.isclose(result["forward"]["upper_occupied_point_gap_m"],
                      gate["actor_upper_occupied_swept_gap_m"], atol=1e-5)
    assert np.isclose(result["forward"]["shin_point_gap_m"],
                      gate["actor_shin_swept_gap_m"], atol=1e-5)
    return {"estimated_pose_xyyaw": pose.tolist(),
            "capture_time_s": time_s,
            "source_head_frames": [entry["actor_frame_path"]
                                   for entry in stage["head_inputs"]],
            "source_range_path": gate["packet_path"],
            "source_shin_path": gate["shin_frame_path"],
            "options": result}


def main() -> None:
    report_path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(report_path.read_text())
    model = model_with_room(visual=selected_visual())
    robot = RobotGeometry(model)
    rows = []
    for episode in report["rows"]:
        trace_path = ROOT / episode["trace_path"]
        assert sha(trace_path) == episode["trace_sha256"]
        trace = json.loads(trace_path.read_text())
        stage = trace["recovery_inputs"][0]["continuation"]["stages"][1]
        row = replay_stage(stage, model, robot)
        row["seed"] = episode["case"][0]
        rows.append(row)
    output = {"schema": "robot-learning-lab.stopped-alternate-route-screen.v1",
              "source_sha256": sha(SOURCE),
              "episode_report_sha256": sha(report_path),
              "scope": "Actor-only replay of saved head/range/shin rays at the "
                       "stopped pose; no motion executed or evaluator route "
                       "truth used. Same historical kinematic body and 5cm "
                       "upper shell plus 16.5cm point envelope.",
              "rows": rows}
    (OUT / "stopped-alternate-route-screen.json").write_text(
        json.dumps(output, indent=2) + "\n")
    print(json.dumps({r["seed"]: {name: value["status"] for name, value
                                 in r["options"].items()} for r in rows},
                     indent=2))


if __name__ == "__main__":
    main()
