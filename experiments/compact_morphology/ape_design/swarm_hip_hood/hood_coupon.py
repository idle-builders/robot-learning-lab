"""One bounded 50 cm hip-hood trial on the sketch-06 source shell.

The moving thigh receives a rounded root cap. A smooth V-like side hem lifts
the fixed pelvis shell above the leg root rather than cutting each sampled
thigh pose into it. Actual thigh, offset-knee-motor, torso and shifted-battery
clearance is checked separately. This is no rated hip support design.
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
from experiments.compact_morphology.ape_design.swarm_hip_yaw_rethink.hip_yaw_rethink import Study
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import integration_shapes
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.hip_height_motion_proxy import rotate
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.check_common_volume import volume
from experiments.sketch_ape_shell.build import loft

OUT = ROOT / "artifacts/hip-v-hood-20261006"
PIVOT = (26.5 * 500 / 350, 0, 160 * 500 / 350 + 16)
POSES = (("stand", 0, 0), ("yaw_out", -20, 0), ("yaw_in", 20, 0),
         ("stride", 20, -35), ("squat", 0, -45))
WALL = 2.0
MARGIN = 0.5
CAP_CENTER = (42.0, 0.0, 215.0)
CAP_RADII = (22.0, 20.0, 20.0)


def ellipsoid(center, radii):
    return b.Solid.make_sphere(1).scale(radii).translate(center)


def v_hem_cutter():
    """One smooth right-side hem, low at centre and high above the leg root."""
    station = [(0, 185), (8, 189), (16, 213), (23, 232), (30, 239),
               (40, 239), (55, 234), (75, 226), (100, 226)]
    front_y = -100
    spline = b.Edge.make_spline([(x, front_y, z) for x, z in station])
    edges = [spline,
             b.Edge.make_line((100, front_y, 226), (100, front_y, -100)),
             b.Edge.make_line((100, front_y, -100), (0, front_y, -100)),
             b.Edge.make_line((0, front_y, -100), (0, front_y, 185))]
    return b.Solid.extrude(b.Face(b.Wire(edges)), (0, 200, 0))


def common_detail(a, c):
    A, C = a.bounding_box(), c.bounding_box()
    if any(getattr(A.min, axis) > getattr(C.max, axis)
           or getattr(C.min, axis) > getattr(A.max, axis) for axis in "XYZ"):
        return {"max_mm3": 0.0, "a_and_c_mm3": 0.0, "c_and_a_mm3": 0.0}
    forward, reverse = volume(a & c), volume(c & a)
    return {"max_mm3": max(forward, reverse),
            "a_and_c_mm3": forward, "c_and_a_mm3": reverse}


def common(a, c):
    return common_detail(a, c)["max_mm3"]


def posed(shape, yaw, pitch):
    return rotate(rotate(shape, "z", yaw, PIVOT), "x", pitch, PIVOT)


def bounds(shape):
    q = shape.bounding_box()
    return [[round(v, 3) for v in q.min], [round(v, 3) for v in q.max]]


def source_outer_lobes():
    """The exact 350 mm sketch stations scaled to the current 500 mm height."""
    k = 500 / 350
    return [loft(stations).scale(k) for stations in (
        [(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),
         (229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),
         (274,0,5,35,22),(278,0,5,23,18)],
        [(130,0,5,9,11),(135,0,5,20,17),(145,0,5,34,23),
         (155,0,4,43,26),(164,0,3,46,26),(178,0,3,49,29)],
        [(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
         (145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)]
    )]


def shapes():
    source = Study(500, WALL, .65)
    source.build()
    parts = {row["name"]: row["shape"] for row in source.parts}
    knee = integration_shapes(500)
    cap = ellipsoid(CAP_CENTER, CAP_RADII) - ellipsoid(
        CAP_CENTER, tuple(r-WALL for r in CAP_RADII))
    cut = v_hem_cutter()
    raw_fixed = {name: parts[name] - cut for name in ("pelvis_front", "pelvis_back")}
    # The smooth hem leaves tiny isolated tips of the *old* thigh-cut shell.
    # Remove those unbuildable islands rather than counting them as a hood.
    fixed = {name: max(shape.solids(), key=lambda s: s.volume)
             for name, shape in raw_fixed.items()}
    moving = {"thigh_front": parts["R_thigh_front"],
              "thigh_back": parts["R_thigh_back"],
              "cap": cap,
              "knee_motor": b.Pos(0, 10, 12) * knee["fixed_knee_hardware"]["motor"]}
    obstacles = {"torso_front": parts["torso_front"],
                 "torso_back": parts["torso_back"],
                 "rear_shifted_4S_pack": b.Pos(0, 14, 200 * 500 / 350) * b.Box(60, 30, 30.5)}
    return parts, fixed, moving, raw_fixed, obstacles


def main():
    source, fixed, moving, raw_fixed, obstacles = shapes()
    rows = []
    operand_order_disagreements = []
    for pose, yaw, pitch in POSES:
        intersections = []
        near = []
        for mn, ms in moving.items():
            for fn, fs in {**fixed, **obstacles}.items():
                moved = posed(ms, yaw, pitch)
                detail = common_detail(moved, fs)
                v = detail["max_mm3"]
                if v > .05:
                    intersections.append({"moving": mn, "fixed": fn, "mm3": round(v, 3)})
                if abs(detail["a_and_c_mm3"]-detail["c_and_a_mm3"]) > .05:
                    operand_order_disagreements.append({"pose": pose, "moving": mn,
                        "fixed": fn, **{k: round(v, 3) for k, v in detail.items()}})
                d = fs.distance_to(moved)
                if d < MARGIN:
                    near.append({"moving": mn, "fixed": fn, "gap_mm": round(d, 4)})
        rows.append({"pose": pose, "yaw_deg": yaw, "pitch_deg": pitch,
                     "intersections": intersections, "under_margin": near})
    cap_outside = moving["cap"]
    for lobe in source_outer_lobes():
        cap_outside = cap_outside - lobe
    report = {"height_mm": 500, "wall_mm": WALL, "pivot_mm": PIVOT,
              "cap_center_mm": CAP_CENTER, "cap_radii_mm": CAP_RADII,
              "cap_outside_original_torso_pelvis_thigh_outer_mm3": round(volume(cap_outside), 3),
              "hood_hem_xz_mm": [[0, 185], [8, 189], [16, 213], [23, 232],
                                 [30, 239], [40, 239], [55, 234], [75, 226], [100, 226]],
              "fixed": [{"name": name, "source_mm3": round(volume(source[name]), 3),
                         "candidate_mm3": round(volume(shape), 3),
                         "remaining_solids": len(shape.solids()),
                         "discarded_old_skin_islands_mm3": [round(s.volume, 3)
                             for s in raw_fixed[name].solids() if s.volume < shape.volume],
                         "bounds_mm": bounds(shape)} for name, shape in fixed.items()],
              "static_obstacles_mm": {name: bounds(shape) for name, shape in obstacles.items()},
              "moving": [{"name": name, "solids": len(shape.solids()),
                          "bounds_mm": bounds(shape)} for name, shape in moving.items()],
              "rows": rows,
              "intersection_operand_order_disagreements": operand_order_disagreements,
              "cap_to_source_thigh_mm3": {
                  name: round(common(moving["cap"], moving[name]), 3)
                  for name in ("thigh_front", "thigh_back")},
              "qualification": "single smooth V-hem and ellipsoid root-cap skin trial; no hip bearings, load path, fasteners or continuous-angle guarantee"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for pose, yaw, pitch in POSES:
        for service in (False, True):
            scene = trimesh.Scene()
            for name, shape in {**fixed, **obstacles}.items():
                if name == "rear_shifted_4S_pack" and not service:
                    continue
                color = (230, 223, 209, 255) if name != "rear_shifted_4S_pack" else (112, 91, 72, 255)
                mesh(scene, name, shape, color)
            for name, shape in moving.items():
                if name == "knee_motor" and not service:
                    continue
                color = (230, 223, 209, 255) if name != "knee_motor" else (65, 101, 140, 255)
                mesh(scene, name, posed(shape, yaw, pitch), color)
            scene.export(OUT / f"{'service' if service else 'opaque'}-{pose}.glb")
    shutil.copyfile(Path(__file__).with_name("viewer.html"), OUT / "index.html")
    print(json.dumps({"fixed": report["fixed"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
