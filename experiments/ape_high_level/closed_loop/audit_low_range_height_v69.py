"""V69: test an additional virtual 8 cm range plane on rotated crate scenes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_3d_clearance_layout_v53 import FRACTIONS
from audit_low_obstacle_transfer_v56 import SCENES as CRATE_SCENES
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
PRIOR = ROOT / "artifacts/ape-closed-loop-cove50-v64"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v69"
BEAMS = 72
HEIGHT_M = .08
DROPOUT = .025
RANGE_NOISE_SIGMA_M = .004
POINT_MARGIN_M = .065
CRATES = {scene[0]: scene for scene in CRATE_SCENES}


def scan(room: Room, seed: int, case_id: str) -> list[float | None]:
    pose_index = int(case_id.split("-pose")[1].split("-")[0])
    yaw_deg = int(case_id.split("yaw")[1])
    scan_step = 100 + pose_index + 100 * (yaw_deg // 90)
    rng = np.random.default_rng(seed * 1009 + 50000 + scan_step + 2000000)
    origin = np.array((room.actual[0], room.actual[1], HEIGHT_M), dtype=float)
    group = np.array((1, 0, 0, 0, 0, 0), dtype=np.uint8)
    readings: list[float | None] = []
    for angle in np.linspace(-math.pi, math.pi, BEAMS, endpoint=False):
        direction = np.array((math.cos(room.actual[2] + angle),
                              math.sin(room.actual[2] + angle), 0.), dtype=float)
        gid = np.array((-1,), dtype=np.int32)
        distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                 group, True, room.model.body("torso").id, gid)
        readings.append(None if distance < 0 or rng.random() < DROPOUT else
                        round(max(.02, float(distance) +
                                  float(rng.normal(0., RANGE_NOISE_SIGMA_M))), 3))
    return readings


def points(ranges: list[float | None], pose: np.ndarray,
           solids: tuple) -> np.ndarray:
    result = []
    for beam, distance in enumerate(ranges):
        if distance is None or distance >= 1.3:
            continue
        angle = float(pose[2]) - math.pi + 2 * math.pi * beam / len(ranges)
        hit = np.array((pose[0] + distance * math.cos(angle),
                        pose[1] + distance * math.sin(angle), HEIGHT_M))
        if RobotGeometry.solid_distance(hit, solids) >= -.005:
            result.append(hit)
    return np.asarray(result, dtype=float).reshape(-1, 3)


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior_path = PRIOR / "low-range-side-back-audit.json"
    prior = json.loads(prior_path.read_text())
    by_case: dict[str, list[dict]] = {}
    for row in prior["rows"]:
        by_case.setdefault(row["case"], []).append(row)
    rows = []
    cases = []
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        for case_id, originals in by_case.items():
            seed = int(originals[0]["seed"])
            _, crate_x, crate_y_rel, half_x, half_y, half_height = CRATES[seed]
            side = 1 if seed % 2 else -1
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            crate_id = room.model.geom("loop_crate_geom").id
            room.model.geom_size[crate_id] = (half_x, half_y, half_height)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, side * crate_y_rel, half_height)
            actor_path = PRIOR / f"{case_id}-actor-input.npz"
            with np.load(actor_path) as saved:
                pose = saved["estimated_pose_xyyaw"].copy()
                qpos = saved["actor_qpos"].copy()
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()
            ranges_8 = scan(room, seed, case_id)
            np.savez_compressed(OUT / f"{case_id}-range-8cm.npz",
                                ranges_m=np.asarray([
                                    np.nan if value is None else value
                                    for value in ranges_8]))
            robot.set_state(qpos)
            hits_8 = points(ranges_8, pose, robot.primitives())
            cases.append({"case": case_id,
                          "saved_actor_input_sha256": sha(actor_path),
                          "range_8_input_sha256": sha(OUT / f"{case_id}-range-8cm.npz"),
                          "nonself_8cm_hits": len(hits_8)})
            for original in originals:
                candidate = np.asarray(original["candidate_xy_m"])
                delta = candidate - pose[:2]
                gap_8 = math.inf
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(qpos, future))
                    if len(hits_8):
                        gap_8 = min(gap_8, float(np.min(
                            RobotGeometry.solid_distances(hits_8, robot.primitives()))))
                saved = original["point_gaps_m"]
                other = min(float(saved["range_11"]), float(saved["head_upper"]))
                combined_8 = min(other, gap_8)
                combined_4_8 = min(combined_8, float(saved["360"]))
                rows.append({"case": case_id,
                             "direction_cell": original["direction_cell"],
                             "evaluator_safe_5cm": original["evaluator_safe_5cm"],
                             "evaluator_model_clearance_m": original["evaluator_model_clearance_m"],
                             "flat_11_pass": original["flat_2d_pass"],
                             "point_4_11_pass": original["field_pass"]["360"],
                             "point_8_11_pass": combined_8 >= POINT_MARGIN_M,
                             "point_4_8_11_pass": combined_4_8 >= POINT_MARGIN_M,
                             "gap_8_m": round(gap_8, 4),
                             "gap_4_8_11_m": round(combined_4_8, 4)})
    finally:
        room.close()
    scores = {name: score(rows, f"{name}_pass") for name in
              ("flat_11", "point_4_11", "point_8_11", "point_4_8_11")}
    assert len(rows) == prior["valid_moves"]
    assert scores["flat_11"] == prior["scores"]["flat_2d"]
    assert scores["point_4_11"] == prior["scores"]["360"]
    gate = (scores["point_4_8_11"]["false_safe"] == 0 and
            scores["point_4_8_11"]["false_block"] <= 32)
    output = {"scope": "One new virtual 8cm 72-beam range plane on exact V64 rotated poses, combined with saved 4/11cm/head actor point gaps at frozen .065m margin; evaluator clearance offline only.",
              "source_sha256": sha(SOURCE),
              "prior_audit_sha256": sha(prior_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "new_height_m": HEIGHT_M, "beams": BEAMS,
              "dropout_probability": DROPOUT,
              "noise_sigma_m": RANGE_NOISE_SIGMA_M,
              "point_margin_m": POINT_MARGIN_M,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "scores": scores, "three_plane_diagnostic_gate_pass": gate,
              "rows": rows}
    (OUT / "low-range-height-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_range_height_v69_source.py").write_bytes(SOURCE.read_bytes())
    print("moves", len(rows), "safe", output["evaluator_safe_moves"])
    print(scores)
    print("three-plane diagnostic gate:", gate)


if __name__ == "__main__":
    main()
