"""V54: same-pose near-floor range ablation against saved V53 actor inputs.

The simulated room is used only to generate the new ray measurements and
independent evaluator labels. The clearance hypotheses use actor scans,
head depth, pose belief, and selected robot geometry only.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_3d_clearance_layout_v53 import (DELTAS, FRACTIONS,
                                             FROZEN_3D_MARGIN_M, point_gap)
from depth_voxel import DepthVoxelMap, VOXEL_M
from footprint_route import scan_points
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
BASE = ROOT / "artifacts/ape-closed-loop-cove50-v53"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v54"
PLANE_HEIGHT_M = .04
DROP_PROBABILITY = .025
NOISE_STD_M = .004


def near_floor_scan(room: Room, seed: int, pose_index: int) -> list[float | None]:
    rng = np.random.default_rng(seed * 1009 + 50000 + pose_index)
    origin = np.array([room.actual[0], room.actual[1], PLANE_HEIGHT_M])
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    ranges = []
    for angle in np.linspace(-math.pi, math.pi, 72, endpoint=False):
        direction = np.array([math.cos(room.actual[2] + angle),
                              math.sin(room.actual[2] + angle), 0.])
        geom_id = np.array([-1], dtype=np.int32)
        distance = float(mujoco.mj_ray(
            room.model, room.data, origin, direction, group, True,
            room.model.body("torso").id, geom_id))
        if distance < 0 or rng.random() < DROP_PROBABILITY:
            ranges.append(None)
        else:
            ranges.append(round(max(.02, distance + rng.normal(0., NOISE_STD_M)), 3))
    return ranges


def xyz_points(ranges: list[float | None], pose: np.ndarray,
               height: float, solids: tuple) -> np.ndarray:
    xy = scan_points(ranges, pose)
    xyz = np.column_stack((xy, np.full(len(xy), height)))
    return np.asarray([
        point for point in xyz
        if RobotGeometry.solid_distance(point, solids) >= -.005
    ], dtype=float).reshape(-1, 3)


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((BASE / "layout-transfer-audit.json").read_text())
    indexed = {(row["case"], tuple(row["direction_cell"])): row
               for row in baseline["rows"]}
    rows = []
    cases = []
    room = Room()
    try:
        for case in baseline["cases"]:
            if not case.get("valid_start", False):
                continue
            case_id = case["case"]
            seed = case["seed"]
            pose_index = int(case_id.split("pose")[-1])
            side = 1 if seed % 2 else -1
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (
                case["table_center_xy_m"][0],
                case["table_center_xy_m"][1], 0.)
            room.set_table_height(case["table_height_m"])
            pose = np.asarray(case["pose_xyyaw"], dtype=float)
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()

            saved_path = BASE / f"{case_id}-actor-input.npz"
            with np.load(saved_path) as saved:
                ranges_11 = [None if np.isnan(item) else float(item)
                             for item in saved["range_72_beams_m"]]
                depth = saved["depth_m"].copy()
                camera_pose = (saved["camera_origin_m"].copy(),
                               saved["camera_rotation"].copy())
                actor_qpos = saved["actor_qpos"].copy()
                assert np.allclose(saved["estimated_pose_xyyaw"], pose)
            ranges_4 = near_floor_scan(room, seed, pose_index)
            np.savez_compressed(
                OUT / f"{case_id}-near-floor-scan.npz",
                range_72_beams_m=np.asarray([
                    np.nan if item is None else item for item in ranges_4]),
                estimated_pose_xyyaw=pose)

            robot = RobotGeometry(room.model)
            robot.set_state(actor_qpos)
            current_solids = robot.primitives()
            voxel = DepthVoxelMap()
            voxel.observe(
                depth, camera_pose, 0.,
                fovy_deg=float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id]),
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, current_solids) < 0.)
            points_11 = xyz_points(ranges_11, pose, .11, current_solids)
            points_4 = xyz_points(ranges_4, pose, PLANE_HEIGHT_M,
                                  current_solids)
            upper_xyz = np.asarray([
                [(ix + .5) * VOXEL_M, (iy + .5) * VOXEL_M,
                 (iz + .5) * VOXEL_M]
                for (ix, iy, iz), hit_s in voxel.occupied.items()
                if hit_s >= voxel.free.get((ix, iy, iz), -math.inf) and
                .18 <= (iz + .5) * VOXEL_M <= .43
            ], dtype=float).reshape(-1, 3)
            cases.append({"case": case_id, "saved_input_sha256": sha(saved_path),
                          "range_11_points": len(points_11),
                          "range_4_points": len(points_4),
                          "head_upper_points": len(upper_xyz)})

            for direction in DELTAS:
                old = indexed[(case_id, direction)]
                candidate = np.asarray(old["candidate_xy_m"], dtype=float)
                delta = candidate - pose[:2]
                gaps = {"range_11": math.inf, "range_4": math.inf,
                        "head_upper": math.inf}
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(actor_qpos, future))
                    solids = robot.primitives()
                    gaps["range_11"] = min(
                        gaps["range_11"], point_gap(points_11, solids))
                    gaps["range_4"] = min(
                        gaps["range_4"], point_gap(points_4, solids))
                    gaps["head_upper"] = min(
                        gaps["head_upper"], point_gap(upper_xyz, solids))
                replay_pass = min(gaps["range_11"],
                                  gaps["head_upper"]) >= FROZEN_3D_MARGIN_M
                if replay_pass != old["frozen_3d_point_pass"]:
                    raise AssertionError(
                        f"V53 baseline failed exact pass replay: {case_id} {direction}: "
                        f"{gaps} vs {old['frozen_3d_point_pass']}")
                row = {"case": case_id, "seed": seed,
                       "direction_cell": list(direction),
                       "candidate_xy_m": candidate.tolist(),
                       "evaluator_model_clearance_m": old["evaluator_model_clearance_m"],
                       "evaluator_safe_5cm": old["evaluator_safe_5cm"],
                       "flat_2d_pass": old["flat_2d_pass"],
                       "range_11_head_pass": replay_pass,
                       "range_4_head_pass": min(gaps["range_4"], gaps["head_upper"]) >= FROZEN_3D_MARGIN_M,
                       "both_head_pass": min(gaps.values()) >= FROZEN_3D_MARGIN_M,
                       "gaps_m": {key: round(value, 4) for key, value in gaps.items()}}
                rows.append(row)
    finally:
        room.close()

    assert len(rows) == baseline["valid_moves"] == 128
    prior_false_safe = [row for row in rows if row["range_11_head_pass"]
                        and not row["evaluator_safe_5cm"]]
    scores = {name: score(rows, key) for name, key in (
        ("flat_2d", "flat_2d_pass"),
        ("range_11_head", "range_11_head_pass"),
        ("range_4_head", "range_4_head_pass"),
        ("both_head", "both_head_pass"))}
    assert scores["flat_2d"] == baseline["scores"]["flat_2d"]
    assert scores["range_11_head"] == baseline["scores"]["frozen_3d_point"]
    assert len(prior_false_safe) == 5
    criterion = (all(not row["both_head_pass"] for row in prior_false_safe)
                 and scores["both_head"]["false_block"] <=
                 scores["flat_2d"]["false_block"])
    output = {"scope": "V54 same-pose diagnostic near-floor range plane; actor clearance from saved V53 11cm/head observations plus newly ray-cast noisy 4cm scan; evaluator labels inherited unchanged from V53. No live motion authority.",
              "source_sha256": sha(SOURCE),
              "v53_audit_sha256": sha(BASE / "layout-transfer-audit.json"),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "near_floor_height_m": PLANE_HEIGHT_M,
              "near_floor_dropout": DROP_PROBABILITY,
              "near_floor_noise_std_m": NOISE_STD_M,
              "frozen_3d_margin_m": FROZEN_3D_MARGIN_M,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "scores": scores,
              "v53_false_safe_cases": prior_false_safe,
              "diagnostic_followup_criterion_pass": criterion,
              "rows": rows}
    (OUT / "near-floor-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_near_floor_range_v54_source.py").write_bytes(SOURCE.read_bytes())
    print("cases", len(cases), "moves", len(rows),
          "safe", output["evaluator_safe_moves"])
    print(scores)
    print("five prior false-safe now passed:",
          sum(row["both_head_pass"] for row in prior_false_safe))
    print("diagnostic followup criterion:", criterion)


if __name__ == "__main__":
    main()
