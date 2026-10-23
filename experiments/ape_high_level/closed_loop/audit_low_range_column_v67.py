"""V67: replay saved low scans as height-limited uncertain occupied columns."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_3d_clearance_layout_v53 import FRACTIONS
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v67"
POINT_MARGIN_M = .065
HEIGHTS_M = (0., .04, .08, .11)
BEAM_HALF_ANGLE_RAD = math.pi / 72.  # 72 beams around 2pi, half a bin.
RANGE_NOISE_SIGMA_M = .004
COHORTS = (
    ("varied_crate", ROOT / "artifacts/ape-closed-loop-cove50-v56",
     "low-obstacle-transfer.json"),
    ("side_rear", ROOT / "artifacts/ape-closed-loop-cove50-v64",
     "low-range-side-back-audit.json"),
)


def columns(ranges: np.ndarray, pose: np.ndarray,
            current_solids: tuple) -> tuple[np.ndarray, np.ndarray, int]:
    samples = []
    radii = []
    kept = 0
    for beam, raw in enumerate(ranges):
        if not np.isfinite(raw):
            continue
        distance = float(raw)
        angle = float(pose[2]) - math.pi + 2 * math.pi * beam / len(ranges)
        xy = pose[:2] + distance * np.array((math.cos(angle), math.sin(angle)))
        hit = np.array((xy[0], xy[1], .04))
        if RobotGeometry.solid_distance(hit, current_solids) < -.005:
            continue
        radius = distance * math.tan(BEAM_HALF_ANGLE_RAD) + RANGE_NOISE_SIGMA_M
        for height in HEIGHTS_M:
            samples.append((xy[0], xy[1], height))
            radii.append(radius)
        kept += 1
    return (np.asarray(samples, dtype=float).reshape(-1, 3),
            np.asarray(radii, dtype=float), kept)


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def evaluate(label: str, directory: Path, audit_name: str,
             robot: RobotGeometry) -> dict:
    audit_path = directory / audit_name
    previous = json.loads(audit_path.read_text())
    by_case: dict[str, list[dict]] = {}
    for row in previous["rows"]:
        by_case.setdefault(row["case"], []).append(row)
    cases = []
    rows = []
    for case_id, original_rows in by_case.items():
        actor_path = directory / f"{case_id}-actor-input.npz"
        with np.load(actor_path) as data:
            ranges = data["range_4_72_beams_m"].copy()
            pose = data["estimated_pose_xyyaw"].copy()
            qpos = data["actor_qpos"].copy()
        robot.set_state(qpos)
        cloud, radii, kept = columns(ranges, pose, robot.primitives())
        cases.append({"case": case_id, "actor_input_sha256": sha(actor_path),
                      "self_filtered_low_returns": kept,
                      "column_samples": len(cloud),
                      "radius_min_m": (None if not len(radii) else round(float(min(radii)), 4)),
                      "radius_max_m": (None if not len(radii) else round(float(max(radii)), 4))})
        for original in original_rows:
            candidate = np.asarray(original["candidate_xy_m"])
            delta = candidate - pose[:2]
            column_gap = math.inf
            for fraction in FRACTIONS:
                future = pose.copy()
                future[:2] += fraction * delta
                robot.set_state(robot.pose_qpos(qpos, future))
                if len(cloud):
                    gap = RobotGeometry.solid_distances(cloud, robot.primitives()) - radii
                    column_gap = min(column_gap, float(np.min(gap)))
            other_gaps = original["point_gaps_m"]
            candidate_gap = min(column_gap, float(other_gaps["range_11"]),
                                float(other_gaps["head_upper"]))
            baseline = (original["both_head_pass"] if label == "varied_crate"
                        else original["field_pass"]["360"])
            rows.append({"case": case_id,
                         "direction_cell": original["direction_cell"],
                         "evaluator_safe_5cm": original["evaluator_safe_5cm"],
                         "evaluator_model_clearance_m": original["evaluator_model_clearance_m"],
                         "flat_11_pass": original["flat_2d_pass"],
                         "point_3d_360_pass": baseline,
                         "column_3d_pass": candidate_gap >= POINT_MARGIN_M,
                         "column_gap_m": round(column_gap, 4),
                         "combined_gap_m": round(candidate_gap, 4)})
    scores = {name: score(rows, key) for name, key in (
        ("flat_11", "flat_11_pass"),
        ("point_3d_360", "point_3d_360_pass"),
        ("column_3d", "column_3d_pass"))}
    assert len(rows) == previous["valid_moves"]
    assert scores["flat_11"] == previous["scores"]["flat_2d"]
    assert scores["point_3d_360"] == (
        previous["scores"]["both_head"] if label == "varied_crate"
        else previous["scores"]["360"])
    return {"label": label, "prior_audit_sha256": sha(audit_path),
            "cases": cases, "valid_moves": len(rows),
            "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
            "scores": scores, "rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        cohorts = {label: evaluate(label, directory, name, robot)
                   for label, directory, name in COHORTS}
    finally:
        room.close()
    gate = (all(cohorts[label]["scores"]["column_3d"]["false_safe"] == 0
                for label in cohorts) and
            cohorts["varied_crate"]["scores"]["column_3d"]["false_block"] <= 13 and
            cohorts["side_rear"]["scores"]["column_3d"]["false_block"] <= 32)
    output = {"scope": "Same saved V56/V64 actor 4cm scans and joint poses, virtual vertical occupancy to 11cm with one-half beam width plus one noise sigma, frozen .065m selected-body point gate; evaluator clearance offline only.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "column_heights_m": HEIGHTS_M,
              "beam_half_angle_rad": BEAM_HALF_ANGLE_RAD,
              "range_noise_sigma_m": RANGE_NOISE_SIGMA_M,
              "point_margin_m": POINT_MARGIN_M,
              "cohorts": cohorts,
              "column_diagnostic_gate_pass": gate}
    (OUT / "low-range-column-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_range_column_v67_source.py").write_bytes(SOURCE.read_bytes())
    for label, cohort in cohorts.items():
        print(label, cohort["valid_moves"], cohort["scores"])
    print("column diagnostic gate:", gate)


if __name__ == "__main__":
    main()
