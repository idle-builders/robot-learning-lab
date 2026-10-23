"""Replay all 16 near-range mount subsets at the frozen V268p route gate.

This is an actor-packet ablation, not a new owner rollout or sensor-packaging
decision. The full and empty subsets are checked against live V268/V270 gates.
"""

from __future__ import annotations

import itertools
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
from screen_stopped_alternate_routes_v255p import point_gap


SOURCE = Path(__file__)
HEALTHY = ROOT / "artifacts/ape-articulated-depth-route-v268p"
EMPTY = ROOT / "artifacts/ape-articulated-depth-route-v270p"
MOUNTS = ("front_chest", "left_flank", "right_flank", "low_front")
POINT_MARGIN_M = .165


def replay(recovery: dict, enabled: tuple[str, ...],
           model: mujoco.MjModel, robot: RobotGeometry) -> dict:
    voxel_map = MediumWindowMap()
    for entry in recovery["head_inputs"]:
        path = ROOT / entry["actor_frame_path"]
        assert sha(path) == entry["actor_frame_sha256"]
        with np.load(path) as packet:
            pose = (packet["camera_origin_estimated_m"].copy(),
                    packet["camera_rotation_estimated"].copy())
            fovy = float(model.cam_fovy[model.camera("lower_rgb").id])
            voxel_map.observe(packet["map_depth_m"], pose,
                              float(packet["capture_time_s"]),
                              fovy_deg=fovy, stride=4, max_depth_m=1.,
                              self_masked=True)
    gate = recovery["actor_gate"]
    near_path = ROOT / gate["packet_path"]
    assert sha(near_path) == gate["packet_sha256"]
    near = json.loads(near_path.read_text())
    time_s = float(near["capture_time_s"])
    pose = np.asarray(near["estimated_pose_xyyaw"], dtype=float)
    for name in enabled:
        add_packet(voxel_map, near["ray_packets"][name], time_s)
    shin_path = ROOT / gate["shin_frame_path"]
    assert sha(shin_path) == gate["shin_frame_sha256"]
    with np.load(shin_path) as packet:
        shin = packet["estimated_environment_points_xyz_m"].copy()
        qpos = packet["actor_qpos"].copy()
    future = pose.copy()
    future[0] += .08
    shell = robot.swept_exterior_coverage(
        voxel_map, qpos, pose, future, time_s, margin_m=.05)
    occupied = np.asarray([[(x + .5) * VOXEL_M,
                            (y + .5) * VOXEL_M,
                            (z + .5) * VOXEL_M]
                           for (x, y, z), stamp in voxel_map.occupied.items()
                           if stamp >= time_s - .4 and
                           stamp >= voxel_map.free.get((x, y, z), -math.inf)
                           and (z + .5) * VOXEL_M >= .04],
                          dtype=float).reshape(-1, 3)
    upper_gap = point_gap(robot, qpos, pose, (.08, 0.), occupied)
    shin_gap = point_gap(robot, qpos, pose, (.08, 0.), shin)
    status = ("free" if shell["status"] == "free" and
              min(upper_gap, shin_gap) >= POINT_MARGIN_M else "blocked")
    return {"enabled_mounts": enabled, "status": status,
            "upper_shell": shell,
            "fresh_occupied_points": len(occupied),
            "upper_point_gap_m": round(upper_gap, 5),
            "shin_point_gap_m": round(shin_gap, 5)}


def main() -> None:
    report_path = HEALTHY / "blocked-forward-viewpoint-audit.json"
    empty_path = EMPTY / "blocked-forward-viewpoint-audit.json"
    report = json.loads(report_path.read_text())
    empty = json.loads(empty_path.read_text())
    model = model_with_room(visual=selected_visual())
    robot = RobotGeometry(model)
    rows = []
    for good, bad in zip(report["rows"], empty["rows"], strict=True):
        assert good["case"] == bad["case"]
        if good["recovery_gate"] is None:
            assert bad["recovery_gate"] is None
            continue
        trace = json.loads((ROOT / good["trace_path"]).read_text())
        recovery = trace["recovery_inputs"][0]
        subsets = [replay(recovery, tuple(combination), model, robot)
                   for length in range(5)
                   for combination in itertools.combinations(MOUNTS, length)]
        full = next(row for row in subsets if len(row["enabled_mounts"]) == 4)
        none = next(row for row in subsets if not row["enabled_mounts"])
        for replayed, live in ((full, good["recovery_gate"]),
                               (none, bad["recovery_gate"])):
            assert replayed["status"] == live["status"]
            assert replayed["upper_shell"] == live["shell_plus_5cm"]
            assert np.isclose(replayed["upper_point_gap_m"],
                              live["actor_upper_occupied_swept_gap_m"],
                              atol=1e-5)
            assert np.isclose(replayed["shin_point_gap_m"],
                              live["actor_shin_swept_gap_m"], atol=1e-5)
        free_sets = [r["enabled_mounts"] for r in subsets
                     if r["status"] == "free"]
        minimum_size = min((len(s) for s in free_sets), default=None)
        rows.append({"seed": good["case"][0], "mode": good["case"][1],
                     "full_live_gate": good["recovery_gate"]["status"],
                     "empty_live_gate": bad["recovery_gate"]["status"],
                     "smallest_authorizing_sets": (
                         [] if minimum_size is None else
                         [s for s in free_sets if len(s) == minimum_size]),
                     "all_subsets": subsets})
    result = {"schema": "robot-learning-lab.near-mount-subset-screen.v1",
              "source_sha256": sha(SOURCE),
              "healthy_report_sha256": sha(report_path),
              "empty_report_sha256": sha(empty_path),
              "mounts": MOUNTS,
              "scope": "Actor-packet replay of 16 virtual near-range subsets "
                       "at each V268p recovery gate. Full and empty "
                       "subsets reproduce the paired live V268/V270 gate. "
                       "No new movement, physical mount qualification or "
                       "held-out sensor selection.",
              "rows": rows}
    (HEALTHY / "near-mount-subset-screen.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({row["seed"]: {"minimum": row[
        "smallest_authorizing_sets"], "full": row["full_live_gate"],
        "none": row["empty_live_gate"]} for row in rows}, indent=2))


if __name__ == "__main__":
    main()
