"""Bounded swept-rigid subtraction for a recessed 50 cm knee gaiter trial.

Only the cosmetic thigh/cuff skin may change. Hardware and moving shin stay
where they were in the rated-knee coupon. The result is a cut proposal, not a
fabric or printability qualification.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import build123d as b
import trimesh

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import (
    integration_shapes,
    turn,
)
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh

OUT = ROOT / "artifacts/knee-gaiter-20261005"
ANGLES = (0, 2, 15, 30, 45, 60, 75, 90, 105, 113, 115)


def volume(shape):
    return 0.0 if shape is None else sum(s.volume for s in shape.solids())


def overlap(a, c):
    x, y = a.bounding_box(), c.bounding_box()
    if any(getattr(x.min, axis) > getattr(y.max, axis)
           or getattr(y.min, axis) > getattr(x.max, axis)
           for axis in "XYZ"):
        return 0.0
    return volume(a & c)


def main():
    case = integration_shapes(500)
    fixed_skin = case["fixed_knee_skin"]
    moving = {**case["moving_shin_skin"],
              "yoke": case["moving_knee_hardware"]["yoke"],
              "shin_cross_pad": case["moving_knee_hardware"]["shin_cross_pad"]}
    # Only geometry that actually touches a fixed skin in the named-contact
    # screen needs a swept clearance envelope.
    names = ("R_shin_front", "R_shin_back", "yoke", "shin_cross_pad")
    envelopes = {}
    for name in names:
        try:
            # offset_3d with no openings yields a Shell in build123d; a
            # Shell is not a cutting solid, so use the high-level solid
            # offset operation instead.
            envelopes[name] = b.offset(moving[name], amount=0.5)
        except Exception:
            envelopes[name] = moving[name]
    revised = {}
    rows = []
    for fixed_name in ("R_thigh_front", "R_thigh_back", "R_knee_cuff"):
        source = fixed_skin[fixed_name]
        cut = source
        for angle in ANGLES:
            for moving_name in names:
                posed = turn(envelopes[moving_name], case["knee_axis_mm"], angle)
                if overlap(cut, posed) > 0.05:
                    cut = cut - posed
        revised[fixed_name] = cut
        rows.append({"fixed": fixed_name,
                     "original_mm3": round(volume(source), 3),
                     "remaining_mm3": round(volume(cut), 3),
                     "removed_mm3": round(volume(source) - volume(cut), 3),
                     "remaining_solids": len(cut.solids()) if cut is not None else 0})
    # Verify the sampled rigid shell and yoke sweep against the revised skins.
    contacts = []
    for angle in ANGLES:
        for moving_name in names:
            posed = turn(moving[moving_name], case["knee_axis_mm"], angle)
            for fixed_name, skin in revised.items():
                common = overlap(posed, skin)
                if common > 0.05:
                    contacts.append({"angle": angle, "moving": moving_name,
                                     "fixed": fixed_name, "common_mm3": round(common, 3)})
    report = {"height_mm": 500, "wall_mm": 2,
              "source": "T-4cmz88 importable 50 cm knee coupon",
              "angles_deg": ANGLES,
              "cut_margin_mm": 0.5,
              "offset_fallback_names": [name for name in names if envelopes[name] is moving[name]],
              "cut_summary": rows,
              "remaining_sampled_intersections": contacts,
              "verdict": "numeric cut proposal only; gaiter stretch, shell appearance, hardware cover and continuous-angle clearance pending"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep-cut-report.json").write_text(json.dumps(report, indent=2))
    for label, skins in (("source", fixed_skin), ("sweep-cut", revised)):
        for angle in (0, 60, 115):
            scene = trimesh.Scene()
            for name in ("R_thigh_front", "R_thigh_back", "R_knee_cuff"):
                mesh(scene, name, skins[name], (225, 218, 203, 255))
            for name in ("R_shin_front", "R_shin_back"):
                mesh(scene, name, turn(moving[name], case["knee_axis_mm"], angle),
                     (226, 219, 205, 255))
            scene.export(OUT / f"{label}-{angle}.glb")
    shutil.copyfile(Path(__file__).with_name("viewer.html"), OUT / "index.html")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
