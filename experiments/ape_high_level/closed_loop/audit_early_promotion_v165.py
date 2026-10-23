"""V165: actor/evaluator clearance of early target-route promotion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_origin_route_v152 import score_path
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v165"
V151 = ROOT / "artifacts/ape-closed-loop-cove50-v151"
V152 = ROOT / "artifacts/ape-closed-loop-cove50-v152/origin-route-audit.json"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
LIGHT = 1.35


def audit(seed: int, variant: dict, info_cells: list[list[int]]) -> dict:
    source_version = 163 if seed == 85 else 155
    trace_path = ROOT / (f"artifacts/ape-closed-loop-cove50-v{source_version}"
                         f"/scene{seed}/seed{seed}/trace.json")
    trace = json.loads(trace_path.read_text())
    transition = next(e for e in trace["events"]
                      if e["action"] == "information_to_annulus_transition")
    prior = next(e for e in reversed(trace["events"][:trace["events"].index(transition)])
                 if e.get("estimated_pose_xyyaw") is not None)
    target_cells = transition["fresh_path_cells"]
    frame_count = sum(e.get("estimated_pose_xyyaw") is not None and
                      e.get("actor_input_sha256") is not None
                      for e in trace["events"][:trace["events"].index(transition)])
    positions = {"CreamCheese": tuple(variant["cream_cheese_xy_m"]),
                 "Parmesan": tuple(variant["parmesan_xy_m"])}
    room = MovedPropRoom(positions)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.model.light_diffuse[0] *= LIGHT
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        voxel = DepthVoxelMap()
        paths = list(sorted((V151 / f"seed{seed}-light{LIGHT:.2f}-present")
                            .glob("lower-angle-*.npz")))
        paths += [trace_path.parent / f"frame-{i}.npz"
                  for i in range(1, frame_count + 1)]
        latest = None
        for path in paths:
            with np.load(path) as z:
                qpos = z["actor_qpos"].copy()
                filtered, _ = mask.apply(z["depth_m"], qpos)
                stamp = float(z["capture_time_s"])
                voxel.observe(filtered, (z["camera_origin_m"],
                                         z["camera_rotation"]), stamp,
                              fovy_deg=fovy, self_masked=True)
                latest = qpos, stamp
        assert latest is not None
        qpos, stamp = latest
        pose = np.asarray(prior["estimated_pose_xyyaw"], dtype=float)
        actual = np.asarray(prior["evaluator_actual_pose_xyyaw"], dtype=float)
        room.data.qpos[:] = qpos
        room.actual[:] = actual
        room.estimated[:] = pose
        room.head = float(qpos[room.head_addr])
        room.time_s = stamp
        room._forward()
        scans = {h: sensed_range(room, seed, 1650000 + seed * 100 + int(h * 100),
                                 height_m=h) for h in (.04, .08, .11)}
        robot = RobotGeometry(room.model)
        point = TwoHeightClearance(robot, qpos, pose, scans[.11], scans[.04],
                                   voxel, stamp, range_8=scans[.08])

        def score(cells: list[list[int]]) -> dict:
            points = [pose[:2].copy()] + [.04 * np.asarray(cell, dtype=float)
                                          for cell in cells]
            gaps = [round(point.edge_gap(start, end, pose[2]), 4)
                    for start, end in zip(points, points[1:])]
            evaluated = score_path(room, actual.copy(),
                                   [[0, 0]] + cells)
            room.actual[:] = actual
            room._forward()
            return {"actor_edge_point_gaps_m": gaps,
                    "actor_min_point_gap_m": min(gaps) if gaps else None,
                    "actor_first_8_edges_min_m": min(gaps[:8]) if gaps else None,
                    "evaluator_route": evaluated}

        info_tail = [cell for cell in info_cells[1:]
                     if .04 * cell[1] > pose[1] + .015] if seed == 88 else [
                     cell for cell in info_cells[1:]
                     if .04 * cell[1] < pose[1] - .015]
        return {"seed": seed, "source_version": source_version,
                "transition_time_s": transition["time_s"],
                "actor_pose_xyyaw": pose.tolist(),
                "actor_target_xy_m": prior["actor_target_xy_m"],
                "target_route_cells": target_cells,
                "target_route": score(target_cells),
                "remaining_information_cells": info_tail,
                "remaining_information_route": score(info_tail),
                "actor_input_sha256": [sha(path) for path in paths],
                "trace_sha256": sha(trace_path)}
    finally:
        mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(SCENES.read_text())
    selected = json.loads(V152.read_text())
    rows = []
    for seed in (85, 88):
        variant = next(v for row in scenes["rows"] if row["seed"] == seed
                       for v in row["passable_variants"]
                       if v["variant"].endswith("-2"))
        info = next(row["selected_path_cells"] for row in selected["rows"]
                    if row["seed"] == seed)
        row = audit(seed, variant, info)
        rows.append(row)
        print(seed, "target", row["target_route"]["actor_min_point_gap_m"],
              row["target_route"]["evaluator_route"]["min_full_scene_m"],
              "info", row["remaining_information_route"]["actor_min_point_gap_m"],
              row["remaining_information_route"]["evaluator_route"]["min_full_scene_m"],
              flush=True)
    result = {"scope": "V165 read-only route margin screen at the exact early information-to-annulus promotion observations in seed85 V163 and seed88 V155. Actor RGB-D/4/8/11cm point gaps on the recorded fresh target and remaining information paths; evaluator full-scene path clearance afterward only. No controller action.",
              "source_sha256": sha(SOURCE), "rows": rows}
    (OUT / "early-promotion-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
