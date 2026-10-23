"""V68: 360-beam near-floor range versus saved 72-beam side/rear cohort."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_3d_clearance_layout_v53 import FRACTIONS
from audit_low_obstacle_transfer_v56 import SCENES as CRATE_SCENES
from audit_low_range_column_v67 import HEIGHTS_M, RANGE_NOISE_SIGMA_M
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
PRIOR = ROOT / "artifacts/ape-closed-loop-cove50-v64"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v68"
BEAMS = 360
DROPOUT = .025
POINT_MARGIN_M = .065
CRATES = {scene[0]: scene for scene in CRATE_SCENES}


def dense_scan(room: Room, seed: int, case_id: str) -> list[float | None]:
    # A separate reproducible stream keeps this new sensor paired with the
    # saved scene without silently altering its 72-beam noise realization.
    pose_index = int(case_id.split("-pose")[1].split("-")[0])
    yaw_deg = int(case_id.split("yaw")[1])
    scan_step = 100 + pose_index + 100 * (yaw_deg // 90)
    rng = np.random.default_rng(seed * 1009 + 50000 + scan_step + 1000000)
    origin = np.array((room.actual[0], room.actual[1], .04), dtype=float)
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


def source_points(ranges: list[float | None], pose: np.ndarray,
                  current_solids: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    hits = []
    columns = []
    radii = []
    kept = 0
    for beam, raw in enumerate(ranges):
        if raw is None or raw >= 1.3:
            continue
        angle = float(pose[2]) - math.pi + 2 * math.pi * beam / len(ranges)
        xy = pose[:2] + raw * np.array((math.cos(angle), math.sin(angle)))
        hit = np.array((xy[0], xy[1], .04))
        if RobotGeometry.solid_distance(hit, current_solids) < -.005:
            continue
        hits.append(hit)
        radius = raw * math.tan(math.pi / len(ranges)) + RANGE_NOISE_SIGMA_M
        for height in HEIGHTS_M:
            columns.append((xy[0], xy[1], height))
            radii.append(radius)
        kept += 1
    return (np.asarray(hits, dtype=float).reshape(-1, 3),
            np.asarray(columns, dtype=float).reshape(-1, 3),
            np.asarray(radii, dtype=float), kept)


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
    saved_column_path = ROOT / "artifacts/ape-closed-loop-cove50-v67/low-range-column-audit.json"
    saved_column = json.loads(saved_column_path.read_text())["cohorts"]["side_rear"]
    old_column = {(row["case"], tuple(row["direction_cell"])): row
                  for row in saved_column["rows"]}
    rows = []
    cases = []
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        for case_id, original_rows in by_case.items():
            seed = int(original_rows[0]["seed"])
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
                range_72 = saved["range_4_72_beams_m"].copy()
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()
            dense = dense_scan(room, seed, case_id)
            robot.set_state(qpos)
            hits, cloud, radii, kept = source_points(dense, pose, robot.primitives())
            raw_5deg_difference = [
                abs(float(dense[5 * beam]) - float(old))
                for beam, old in enumerate(range_72)
                if np.isfinite(old) and dense[5 * beam] is not None]
            cases.append({"case": case_id, "saved_actor_input_sha256": sha(actor_path),
                          "dense_nonself_hits": kept,
                          "range_5deg_median_abs_difference_m": (
                              None if not raw_5deg_difference else
                              round(float(np.median(raw_5deg_difference)), 4))})
            for original in original_rows:
                candidate = np.asarray(original["candidate_xy_m"])
                delta = candidate - pose[:2]
                point_gap = math.inf
                column_gap = math.inf
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(qpos, future))
                    solids = robot.primitives()
                    if len(hits):
                        point_gap = min(point_gap, float(np.min(
                            RobotGeometry.solid_distances(hits, solids))))
                    if len(cloud):
                        column_gap = min(column_gap, float(np.min(
                            RobotGeometry.solid_distances(cloud, solids) - radii)))
                other = original["point_gaps_m"]
                point_all = min(point_gap, float(other["range_11"]),
                                float(other["head_upper"]))
                column_all = min(column_gap, float(other["range_11"]),
                                 float(other["head_upper"]))
                old = old_column[(case_id, tuple(original["direction_cell"]))]
                rows.append({"case": case_id,
                             "direction_cell": original["direction_cell"],
                             "evaluator_safe_5cm": original["evaluator_safe_5cm"],
                             "evaluator_model_clearance_m": original["evaluator_model_clearance_m"],
                             "flat_11_pass": original["flat_2d_pass"],
                             "point_72_pass": original["field_pass"]["360"],
                             "column_72_pass": old["column_3d_pass"],
                             "point_360_pass": point_all >= POINT_MARGIN_M,
                             "column_360_pass": column_all >= POINT_MARGIN_M,
                             "point_360_gap_m": round(point_gap, 4),
                             "column_360_gap_m": round(column_gap, 4)})
    finally:
        room.close()
    scores = {name: score(rows, f"{name}_pass") for name in
              ("flat_11", "point_72", "column_72", "point_360", "column_360")}
    assert len(rows) == prior["valid_moves"]
    assert scores["flat_11"] == prior["scores"]["flat_2d"]
    assert scores["point_72"] == prior["scores"]["360"]
    assert scores["column_72"] == saved_column["scores"]["column_3d"]
    gate = (scores["column_360"]["false_safe"] == 0 and
            scores["column_360"]["false_block"] <= 32)
    output = {"scope": "One-degree virtual low 4cm range re-rendered on exact saved V64 side/rear poses and scene specs; same 72-beam 11cm/head upper points and offline evaluator labels; no live motion authority.",
              "source_sha256": sha(SOURCE),
              "prior_audit_sha256": sha(prior_path),
              "prior_column_audit_sha256": sha(saved_column_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "beams": BEAMS, "dropout_probability": DROPOUT,
              "noise_sigma_m": RANGE_NOISE_SIGMA_M,
              "point_margin_m": POINT_MARGIN_M,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "scores": scores, "column_360_diagnostic_gate_pass": gate,
              "rows": rows}
    (OUT / "low-range-density-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_range_density_v68_source.py").write_bytes(SOURCE.read_bytes())
    print("moves",len(rows),"safe",output["evaluator_safe_moves"])
    print(scores)
    print("dense column diagnostic gate:", gate)


if __name__ == "__main__":
    main()
