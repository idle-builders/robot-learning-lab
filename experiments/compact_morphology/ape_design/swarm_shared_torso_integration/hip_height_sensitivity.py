"""Bounded 50 cm hip-yaw height screen against the offset knee and shared reserves.

Moving the entire existing hip-yaw coupon is not a mechanism design: the femur
link, moving skin, bearing mounts, and torque path would all need rebuilding.
This only checks whether an axial relocation is geometrically promising before
spending time on another hip coupon.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as b

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_hip_yaw_rethink.hip_yaw_rethink import assembly_solids, shell
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import integration_shapes
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.check_common_volume import pair_report

OUT = ROOT / "artifacts/shared-torso-integration/hip-height-sensitivity.json"


def main():
    knee = integration_shapes(500)
    knees = []
    for frame in ("fixed_knee_hardware", "moving_knee_hardware"):
        for name, source in knee[frame].items():
            shape = source if name == "shin_cross_pad" else b.Pos(0, 10, 12) * source
            knees.extend((("R " + name, shape), ("L " + name, shape.mirror(b.Plane.YZ))))
    hip = assembly_solids(500, "remote", 200, 225)
    power = [(name, shape) for name, shape in hip["fixed"]
             if "battery" in name or "service" in name]
    hardware = [(name, shape) for name, shape in hip["fixed"] + hip["moving"]
                if "battery" not in name and "service" not in name]
    _, skins, _ = shell(500, 2)
    rows = []
    for dz in (0, 8, 16, 24, 32):
        shifted = [(name, b.Pos(0, 0, dz) * shape) for name, shape in hardware]
        knee_hits = pair_report(knees, shifted)
        power_hits = pair_report(shifted, power)
        shell_hits = pair_report(shifted, list(skins.items()))
        rows.append({
            "hip_shift_up_mm": dz,
            "knee_intersections": knee_hits["intersections"],
            "power_intersections": power_hits["intersections"],
            "skin_intersections": shell_hits["intersections"],
            "sub_0p5mm_gaps": {"knee": knee_hits["less_than_0p5mm_gap"],
                                "power": power_hits["less_than_0p5mm_gap"],
                                "skin": shell_hits["less_than_0p5mm_gap"]},
        })
        print(dz, len(knee_hits["intersections"]), len(power_hits["intersections"]),
              len(shell_hits["intersections"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"height_mm": 500, "wall_mm": 2,
                               "scope": "translated source hip hardware only; no moving femur or load-path qualification",
                               "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
