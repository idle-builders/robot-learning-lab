"""Exact skin-only sensitivity to an outboard 50 cm raised hip pivot.

The source thigh and offset-knee motor are not translated at stand. Only the
instantaneous pivot changes, representing an unbuilt short offset hip carrier.
This is not an actuator or supported-joint placement claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as b

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_hip_yaw_rethink.hip_yaw_rethink import Study
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import integration_shapes
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.hip_height_motion_proxy import rotate
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.check_common_volume import pair_report

OUT = ROOT / "artifacts/hip-pelvis-closure-20261005/lateral-axis-sensitivity.json"
BASE = (26.5 * 500 / 350, 0, 160 * 500 / 350 + 16)
POSES = (("yaw_out", -20, 0), ("yaw_in", 20, 0), ("stride", 20, -35), ("squat", 0, -45))


def main():
    study = Study(500, 2, .65)
    study.build()
    parts = {row["name"]: row["shape"] for row in study.parts}
    pelvis = [(n, parts[n]) for n in ("pelvis_front", "pelvis_back")]
    knee = integration_shapes(500)
    moving = [("thigh_front", parts["R_thigh_front"]),
              ("thigh_back", parts["R_thigh_back"]),
              ("knee_motor", b.Pos(0, 10, 12) * knee["fixed_knee_hardware"]["motor"])]
    rows = []
    for outboard in (0, 5, 10, 15):
        pivot = (BASE[0] + outboard, BASE[1], BASE[2])
        for name, yaw, pitch in POSES:
            posed = [(part, rotate(rotate(shape, "z", yaw, pivot), "x", pitch, pivot))
                     for part, shape in moving]
            check = pair_report(posed, pelvis)
            rows.append({"outboard_axis_mm": outboard, "pivot_mm": pivot,
                         "pose": name, "hits": check["intersections"],
                         "total_mm3": round(sum(x["common_mm3"] for x in check["intersections"]), 3)})
            print(outboard, name, rows[-1]["total_mm3"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"height_mm": 500, "wall_mm": 2,
                               "scope": "moving skin/motor proxy, unbuilt outboard carrier; no support or motor placement",
                               "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
