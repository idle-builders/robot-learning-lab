"""Small moving-skin sensitivity for a raised 50 cm hip-yaw axis.

The existing thigh skin and offset-knee motor rotate as rigid passengers about
the candidate hip pivot. This checks whether static room remains useful at a
few poses. It does not design the missing hip roll/pitch load path or closures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as b

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_hip_yaw_rethink.hip_yaw_rethink import Study, assembly_solids
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import integration_shapes
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.check_common_volume import pair_report

OUT = ROOT / "artifacts/shared-torso-integration/hip-height-motion-proxy.json"
HIP_X = 26.5 * 500 / 350
HIP_Z = 160 * 500 / 350
POSES = (("neutral", 0, 0), ("yaw_left", 20, 0),
         ("yaw_right", -20, 0), ("stride", 20, -35), ("squat", 0, -45))


def rotate(shape, axis, angle, pivot):
    r = {"x": b.Rot(angle, 0, 0), "z": b.Rot(0, 0, angle)}[axis]
    return b.Pos(*pivot) * r * b.Pos(*(-v for v in pivot)) * shape


def main():
    study = Study(500, 2, .65)
    study.build()
    parts = {row["name"]: row["shape"] for row in study.parts}
    fixed_skin = [(n, parts[n]) for n in ("torso_front", "torso_back", "pelvis_front", "pelvis_back")]
    hip = assembly_solids(500, "remote", 200, 225)
    motors = [(n, shape) for n, shape in hip["fixed"] if "RSF" in n]
    knee = integration_shapes(500)
    right_knee_motor = b.Pos(0, 10, 12) * knee["fixed_knee_hardware"]["motor"]
    rows = []
    for dz in (0, 16):
        raised_motors = [(n, b.Pos(0, 0, dz) * shape) for n, shape in motors]
        for side, label in ((-1, "L"), (1, "R")):
            pivot = (side * HIP_X, 0, HIP_Z + dz)
            source = [(label + "_thigh_front", parts[label + "_thigh_front"]),
                      (label + "_thigh_back", parts[label + "_thigh_back"]),
                      (label + "_knee_motor", right_knee_motor if side == 1
                       else right_knee_motor.mirror(b.Plane.YZ))]
            for pose, yaw, pitch in POSES:
                moving = [(name, rotate(rotate(shape, "z", yaw, pivot), "x", pitch, pivot))
                          for name, shape in source]
                skin_hit = pair_report(moving, fixed_skin)
                motor_hit = pair_report(moving, raised_motors)
                rows.append({"hip_shift_up_mm": dz, "side": label, "pose": pose,
                             "yaw_deg": yaw, "pitch_deg": pitch,
                             "fixed_skin_intersections": skin_hit["intersections"],
                             "hip_motor_intersections": motor_hit["intersections"]})
                print(dz, label, pose, len(skin_hit["intersections"]),
                      len(motor_hit["intersections"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"height_mm": 500, "wall_mm": 2,
                               "scope": "rigid thigh/offset-knee motor proxy; no complete hip or gaiter",
                               "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
