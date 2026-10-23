"""Screen a modest virtual knee-axis relocation against the 50 cm source skin.

This is a mechanism search coupon, not a connected assembly. It preserves the
standing shin silhouette, moves the drive/output/support datums as one rigid
group, and exposes the missing output-to-shin connection explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import build123d as b

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import (
    integration_shapes,
    turn,
)

OUT = ROOT / "artifacts/knee-cover-v2-20261005"
ANGLES = (0, 2, 15, 30, 45, 60, 75, 90, 105, 113, 115)


def overlap(a, c):
    ab, cb = a.bounding_box(), c.bounding_box()
    if any(getattr(ab.min, axis) > getattr(cb.max, axis)
           or getattr(cb.min, axis) > getattr(ab.max, axis)
           for axis in "XYZ"):
        return 0.0
    cut = a & c
    return 0.0 if cut is None else sum(s.volume for s in cut.solids())


def screen(dy, dz):
    case = integration_shapes(500)
    source_axis = case["knee_axis_mm"]
    axis = (source_axis[0], source_axis[1] + dy, source_axis[2] + dz)
    shift = b.Pos(0, dy, dz)
    fixed_skin = case["fixed_knee_skin"]
    shifted_fixed = {n: shift * s for n, s in case["fixed_knee_hardware"].items()}
    # The current cross-pad is physically attached to the original shin, so
    # it stays in its standing frame. The yoke must be redesigned to bridge
    # the new axis; reporting it as connected would be dishonest.
    moving = {n: case["moving_shin_skin"][n]
              for n in ("R_shin_front", "R_shin_back")}
    moving["shin_cross_pad"] = case["moving_knee_hardware"]["shin_cross_pad"]
    static_hits = []
    for hardware_name, hardware in shifted_fixed.items():
        for skin_name, skin in fixed_skin.items():
            volume = overlap(hardware, skin)
            if volume > 0.05:
                static_hits.append(dict(hardware=hardware_name, skin=skin_name,
                                        overlap_mm3=round(volume, 3)))
    poses = []
    for angle in ANGLES:
        hits = []
        for moving_name, solid in moving.items():
            posed = turn(solid, axis, angle)
            for skin_name, skin in fixed_skin.items():
                volume = overlap(posed, skin)
                if volume > 0.05:
                    hits.append(dict(moving=moving_name, fixed=skin_name,
                                     overlap_mm3=round(volume, 3)))
        poses.append(dict(angle_deg=angle, hits=hits))
    return dict(height_mm=500, wall_mm=2, source_axis_mm=source_axis,
                candidate_axis_mm=axis, displacement_mm=(dy, dz),
                fixed_hardware_skin_hits=static_hits, poses=poses,
                not_modeled="output-to-shin offset yoke, shaft bearings after relocation, moving hardware-to-skin clearance, 0.5 mm margin, material strength, continuous-angle clearance, full leg",
                verdict="axis-location sensitivity only; neither package nor moving assembly qualified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dy", type=float, default=10)
    ap.add_argument("--dz", type=float, default=10)
    args = ap.parse_args()
    report = screen(args.dy, args.dz)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pivot-y{args.dy:g}-z{args.dz:g}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
