"""First rigid skin cut for the 50 cm offset-pivot knee candidate.

The standing shin skin stays unchanged. A shaft/yoke moves +10 mm rearward and
+12 mm upward, and a short bridge joins that yoke to the original shin pad.
Only the thigh/cuff skin may be locally relieved. This is still a geometry
coupon; a passing result would not establish rated durability or printability.
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

OUT = ROOT / "artifacts/knee-cover-v2-20261005"
ANGLES = (0, 15, 30, 60, 90, 115)
MARGIN = 0.5


def volume(shape):
    return 0.0 if shape is None else sum(x.volume for x in shape.solids())


def overlap(a, c):
    ab, cb = a.bounding_box(), c.bounding_box()
    if any(getattr(ab.min, axis) > getattr(cb.max, axis)
           or getattr(cb.min, axis) > getattr(ab.max, axis)
           for axis in "XYZ"):
        return 0.0
    return volume(a & c)


def main():
    case = integration_shapes(500)
    shift = b.Pos(0, 10, 12)
    axis = (66, 12, 158)
    fixed = {n: shift * shape for n, shape in case["fixed_knee_hardware"].items()}
    moving = {
        "yoke": shift * case["moving_knee_hardware"]["yoke"],
        "bridge": b.Pos(61, 11, 116) * b.Box(6, 12, 12),
        "shin_cross_pad": case["moving_knee_hardware"]["shin_cross_pad"],
        **{n: case["moving_shin_skin"][n]
           for n in ("R_shin_front", "R_shin_back")},
    }
    # Only the yoke and added bridge need a local skin aperture. The shin
    # shells already have >0.8 mm at the closest 115° thigh approach.
    moving_cutters = {n: b.offset(moving[n], amount=MARGIN)
                      for n in ("yoke", "bridge")}
    fixed_cutters = {n: b.offset(fixed[n], amount=MARGIN)
                     for n in ("motor", "support_B")}
    revised = {}
    rows = []
    for name, source in case["fixed_knee_skin"].items():
        cut = source
        for cutter in fixed_cutters.values():
            if overlap(cut, cutter) > 0.05:
                cut = cut - cutter
        for angle in ANGLES:
            for cutter in moving_cutters.values():
                posed = turn(cutter, axis, angle)
                if overlap(cut, posed) > 0.05:
                    cut = cut - posed
        revised[name] = cut
        rows.append(dict(skin=name,original_mm3=round(volume(source),3),
                         removed_mm3=round(volume(source)-volume(cut),3),
                         remaining_solids=len(cut.solids())))
    hits = []
    for angle in ANGLES:
        for name, shape in moving.items():
            posed = turn(shape, axis, angle)
            for fixed_name, skin in revised.items():
                common = overlap(posed, skin)
                if common > 0.05:
                    hits.append(dict(angle_deg=angle,moving=name,fixed=fixed_name,
                                     overlap_mm3=round(common,3)))
    fixed_hits = []
    for name, shape in fixed.items():
        for fixed_name, skin in revised.items():
            common = overlap(shape, skin)
            if common > 0.05:
                fixed_hits.append(dict(hardware=name,skin=fixed_name,
                                       overlap_mm3=round(common,3)))
    connectivity = dict(
        shaft_to_yoke_mm3=round(overlap(shift * case["moving_knee_hardware"]["shaft"],moving["yoke"]),3),
        yoke_to_bridge_mm3=round(overlap(moving["yoke"],moving["bridge"]),3),
        bridge_to_pad_mm3=round(overlap(moving["bridge"],moving["shin_cross_pad"]),3),
    )
    report = dict(height_mm=500,wall_mm=2,pivot_axis_mm=axis,
                  angles_deg=ANGLES,margin_mm=MARGIN,skin=rows,
                  moving_skin_hits=hits,fixed_skin_hits=fixed_hits,
                  nominal_connection_overlaps=connectivity,
                  unqualified="continuous-angle clearance, material strength, exact bearing mounts, belt rating, skin fold/pinch, printability, independent aesthetic review, full robot",
                  verdict="bounded 50 cm offset-pivot coupon; evaluate connectivity and opaque appearance before promotion")
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"offset-pivot-cut.json").write_text(json.dumps(report,indent=2)+"\n")
    for label, skins in (("original", case["fixed_knee_skin"]),
                         ("offset", revised)):
        for angle in (0, 60, 115):
            scene = trimesh.Scene()
            for name, skin in skins.items():
                mesh(scene, name, skin, (230, 222, 207, 255))
            for name in ("R_shin_front", "R_shin_back"):
                mesh(scene, name, turn(moving[name], axis, angle),
                     (230, 222, 207, 255))
            scene.export(OUT / f"{label}-{angle}.glb")
            if label == "offset":
                diagnostic = trimesh.Scene()
                for name, skin in skins.items():
                    mesh(diagnostic, name, skin, (230, 222, 207, 105))
                for name in ("R_shin_front", "R_shin_back"):
                    mesh(diagnostic, name, turn(moving[name], axis, angle),
                         (230, 222, 207, 105))
                for name, hardware in fixed.items():
                    mesh(diagnostic, name, hardware, (58, 104, 148, 255))
                for name in ("yoke", "bridge", "shin_cross_pad"):
                    mesh(diagnostic, name, turn(moving[name], axis, angle),
                         (196, 129, 74, 255))
                diagnostic.export(OUT / f"mechanism-{angle}.glb")
    shutil.copyfile(Path(__file__).with_name("viewer.html"), OUT / "index.html")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
