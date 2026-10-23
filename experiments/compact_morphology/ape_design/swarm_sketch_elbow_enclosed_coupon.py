"""Bounded 35 cm sketch-shell elbow revision with a compact rear belt rib.

Uses the independent sculpted shell BReps. The outer pivot shift and 16:32
belt are tested with a carrier, a forearm yoke and a curved removable rear
cover. Envelopes and CAD contacts are evidence; tooth geometry, fasteners,
printed tolerances, cable flex and motor thermal duty are still open.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh
from shapely.geometry import Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study  # noqa: E402

OUT = ROOT / "artifacts/swarm-sketch-elbow-enclosed-20261004"
AXIS = (108, 2, 194.5)
CASE = (90, 1, 232.55)
DRIVER = (90, 22, 236.55)
OUTPUT = (108, 22, 194.5)
R1, R2 = 16 / math.pi, 32 / math.pi
POSES = (0, 30, 60, 90, 115)


def volume(shape):
    return sum(s.volume for s in shape) if isinstance(shape, bd.ShapeList) else (shape.volume if shape else 0.0)


def cyl_y(r, depth, x, y, z):
    return bd.Pos(x, y, z) * bd.Rot(90, 0, 0) * bd.Cylinder(r, depth)


def flex(shape, deg):
    return bd.Pos(*AXIS) * bd.Rot(0, -deg, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def profile(poly, y, depth):
    wire = bd.Wire.make_polygon([(x, y, z) for x, z in poly.exterior.coords])
    return bd.extrude(bd.Face(wire), amount=depth, dir=(0, 1, 0))


def belt(clearance=0):
    centers = ((DRIVER[0], DRIVER[2], R1), (OUTPUT[0], OUTPUT[2], R2))
    outside = unary_union([Point(x, z).buffer(r + .76 + clearance, 32)
                           for x, z, r in centers]).convex_hull
    inside = unary_union([Point(x, z).buffer(r - .76 - clearance, 32)
                          for x, z, r in centers]).convex_hull
    wire = lambda poly: bd.Wire.make_polygon([(x, 20 - clearance, z)
                                                for x, z in poly.exterior.coords])
    return bd.extrude(bd.Face(wire(outside), [wire(inside)]),
                      amount=4 + 2*clearance, dir=(0, 1, 0))


def render_mesh(shape, rgba):
    vertices, faces = shape.tessellate(.25, .25)
    mesh = trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in vertices]),
                           np.asarray(faces), process=False)
    mesh.merge_vertices()
    mesh.visual.vertex_colors = rgba
    return mesh


def add(scene, name, shape, rgba):
    scene.add_geometry(render_mesh(shape, rgba), geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    study = Study(height=350)
    study.build()
    parts = {p["name"]: p["shape"] for p in study.parts}
    original_upper = [parts[n] for n in ("R_upper_arm_front", "R_upper_arm_back")]
    original_fore = [parts[n] for n in ("R_forearm_front", "R_forearm_back",
                                       "R_palm_front", "R_palm_back")]
    cuff = parts["R_elbow_cuff"]
    motor = bd.Pos(*CASE) * bd.Box(20, 26, 34)
    boss = cyl_y(3, 6, DRIVER[0], 15.5, DRIVER[2])
    flange = cyl_y(8, 2.5, DRIVER[0], 17.25, DRIVER[2])
    neck = cyl_y(3, 3, DRIVER[0], 19.5, DRIVER[2])
    input_pulley = cyl_y(R1, 4, *DRIVER)
    output_pulley = cyl_y(R2, 4, *OUTPUT)
    belt_loop = belt()
    hardware = {"XL330_case": motor, "XL330_output_boss_envelope": boss,
                "stepped_4hole_motor_flange": flange, "stepped_hub_neck": neck,
                "16T_driver_envelope": input_pulley,
                "32T_output_envelope": output_pulley,
                "Gates_140_2MGT_04_belt_envelope": belt_loop}
    # Source shell halves remain removable. Relieve only the static motor and
    # drive route, with 0.6 mm nominal modeling allowance, not print fit.
    clearances = [bd.Pos(*CASE) * bd.Box(21.2, 27.2, 35.2),
                  cyl_y(8.6, 3.7, DRIVER[0], 17.25, DRIVER[2]),
                  cyl_y(3.6, 4.2, DRIVER[0], 19.5, DRIVER[2]),
                  cyl_y(R1+.6, 5.2, *DRIVER),
                  cyl_y(R2+.6, 5.2, *OUTPUT), belt(.6)]
    upper = []
    for part in original_upper:
        cut = part
        for tool in clearances:
            cut = cut - tool
        upper.append(cut)
    # A one-piece upper carrier routes load from shoulder-side rails into two
    # 688-size seats. A two-rail motor cradle surrounds (does not fuse to) the
    # source-size motor body. Its exact ROBOTIS fixing-hole map is still open.
    side_spine = bd.Pos(106, 1, 224) * bd.Box(7, 9, 58)
    inner_rail = bd.Pos(77, 1, 232.5) * bd.Box(5, 9, 41)
    top_bridge = bd.Pos(91, 1, 253.5) * bd.Box(32, 9, 5)
    lower_bridge = bd.Pos(91, 1, 211) * bd.Box(32, 9, 5)
    carrier = side_spine.fuse(inner_rail, top_bridge, lower_bridge)
    seats, bearings = [], []
    for y in (-9, 11):
        seat = cyl_y(10.5, 7, AXIS[0], y, AXIS[2])
        # Span the side seats into the central spine; a 7 mm web left both
        # bearing sockets floating across a 2 mm axial gap in the first pass.
        web = bd.Pos(107, y, 202) * bd.Box(10, 13, 16)
        carrier = carrier.fuse(seat, web)
        bearings.append(cyl_y(8, 5, AXIS[0], y, AXIS[2]) -
                        cyl_y(4.05, 6, AXIS[0], y, AXIS[2]))
    for y in (-9, 11):
        carrier -= cyl_y(8.15, 7.2, AXIS[0], y, AXIS[2])
    carrier -= cyl_y(4.1, 47, *AXIS)
    # Internal forearm spine, crossbar and opposed 4-mm shaft clamps. These
    # are intentionally united; the shell is not used as the sole load path.
    fore_spine = bd.Pos(101, 1, 145) * bd.Box(8, 8, 80)
    fore_crossbar = bd.Pos(103, 1, 178.5) * bd.Box(10, 36, 8)
    moving_yoke = fore_spine.fuse(fore_crossbar)
    for y in (-17, 17):
        web = bd.Pos(105, y, 186) * bd.Box(8, 4, 19)
        cheek = cyl_y(9.5, 4, AXIS[0], y, AXIS[2])
        moving_yoke = moving_yoke.fuse(web, cheek)
        moving_yoke -= cyl_y(4.05, 4.2, AXIS[0], y, AXIS[2])
    shaft = cyl_y(4, 50, *AXIS)
    # Sculpted, 2 mm-ish removable cap: narrow by the motor, wider only over
    # the 32T output. It is a rear rib rather than a rectangular pod. Three
    # screw/insert locations are reserved; retention is not yet verified.
    outer_profile = unary_union([Point(DRIVER[0], DRIVER[2]).buffer(12, 48),
                                 Point(OUTPUT[0], OUTPUT[2]).buffer(13.2, 48)]).convex_hull
    inner_profile = outer_profile.buffer(-2)
    cap = profile(outer_profile, 26, 2)  # broad smooth outside face
    cap = cap.fuse(profile(outer_profile, 20, 6) -
                   profile(inner_profile, 19.9, 6.2))
    for x, z in ((80.7, 236.5), (113.8, 208.0), (108, 183.7)):
        cap -= cyl_y(1.1, 3, x, 27, z)
    shell_removed = sum(a.volume-b.volume for a,b in zip(original_upper, upper))
    cap_vs_front = volume(cap.intersect(upper[0]))
    cap_vs_back = volume(cap.intersect(upper[1]))
    # This is molded as a curved extension of the detachable upper-back
    # shell; the back half comes away for belt assembly/service.
    upper[1] = upper[1].fuse(cap)
    report = {
        "stage": "bounded source-shell carrier and curved-cover CAD coupon, not a finished elbow",
        "source_shell": "experiments/sketch_ape_shell/build.py Study(height=350)",
        "source_urls": {
            "XL330": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
            "Gates_catalog": "https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/power-transmission-catalog.pdf",
            "Gates_design_manual": "https://www.gates.com/content/dam/documents-library/catalogs/light-power-and-precision-manual.pdf",
            "igus_688": "https://www.igus.com/product/xiros_radial_B180_ES_PA_C"},
        "motor_to_elbow": "16T:32T, elbow=motor/2; ideal torque 2x motor before losses",
        "pivot_axis_xyz_mm": AXIS, "source_cuff_center_xyz_mm": [96, 2, 194.5],
        "outboard_shift_mm": 12,
        "motor_center_xyz_mm": CASE,
        "belt_center_distance_mm": round(math.dist((DRIVER[0],DRIVER[2]),
                                                    (OUTPUT[0],OUTPUT[2])),3),
        "upper_source_shell_removed_mm3": round(shell_removed, 2),
        "upper_source_shell_substantial_solids_after_relief": [len([s for s in part.solids() if s.volume>1]) for part in upper],
        "upper_carrier_substantial_solids": len([s for s in carrier.solids() if s.volume>1]),
        "moving_yoke_substantial_solids": len([s for s in moving_yoke.solids() if s.volume>1]),
        "rear_cap_substantial_solids": len([s for s in cap.solids() if s.volume>1]),
        "hardware_vs_relived_upper_shell_mm3": {name: round(sum(volume(h.intersect(s)) for s in upper), 2)
                                                  for name,h in hardware.items()},
        "carrier_vs_motor_mm3": round(volume(carrier.intersect(motor)),2),
        "carrier_vs_source_shell_mm3": round(sum(volume(carrier.intersect(s)) for s in upper),2),
        "cap_vs_drive_hardware_mm3": round(sum(volume(cap.intersect(s)) for s in hardware.values()),2),
        "cap_vs_original_relived_front_skin_mm3": round(cap_vs_front,2),
        "cap_vs_original_relived_back_skin_intentional_join_mm3": round(cap_vs_back,2),
        "upper_back_with_cap_substantial_solids": len([s for s in upper[1].solids() if s.volume>1]),
        "cap_outboard_max_x_mm": round(cap.bounding_box().max.X,2),
        "cap_rear_max_y_mm": round(cap.bounding_box().max.Y,2),
        "unmodified_upper_outboard_max_x_mm": round(max(s.bounding_box().max.X for s in original_upper),2),
        "unmodified_upper_rear_max_y_mm": round(max(s.bounding_box().max.Y for s in original_upper),2),
        "Gates_corrected_16T_100rpm_4mm_140mm_input_rating_Nm": round(1.58*.6*.8,3),
        "XL330_5V_stall_not_continuous_Nm": .52,
        "poses": []}
    for deg in range(0,116,5):
        moved_shell = [flex(s,deg) for s in original_fore]
        moved_yoke = flex(moving_yoke,deg)
        row = {"angle":deg,
               "forearm_vs_relived_upper_mm3":round(sum(volume(a.intersect(b)) for a in moved_shell for b in upper),2),
               "forearm_vs_drive_mm3":round(sum(volume(a.intersect(h)) for a in moved_shell for h in hardware.values()),2),
               "forearm_vs_carrier_mm3":round(sum(volume(a.intersect(carrier)) for a in moved_shell),2),
               "forearm_vs_cap_mm3":round(sum(volume(a.intersect(cap)) for a in moved_shell),2),
               "yoke_vs_carrier_mm3":round(volume(moved_yoke.intersect(carrier)),2),
               "yoke_vs_drive_mm3":round(sum(volume(moved_yoke.intersect(h)) for h in hardware.values()),2),
               "yoke_vs_cuff_mm3":round(volume(moved_yoke.intersect(cuff)),2),
               "yoke_vs_upper_shell_mm3":round(sum(volume(moved_yoke.intersect(s)) for s in upper),2)}
        report["poses"].append(row)
        print(row, flush=True)
        if deg in POSES:
            scene = trimesh.Scene()
            for i,s in enumerate(upper):add(scene,f"source_upper_relieved_{i}",s,(230,224,209,255))
            for i,s in enumerate(moved_shell):add(scene,f"source_forearm_palm_{i}",s,(239,232,217,255))
            add(scene,"source_dark_elbow_cuff",cuff,(104,103,97,255))
            add(scene,"fixed_supported_carrier",carrier,(112,112,107,255))
            add(scene,"moving_forearm_yoke",moved_yoke,(139,139,132,255))
            # The curved rear rib is now geometrically joined to the upper
            # back shell; no second floating cover part is rendered.
            add(scene,"XL330_motor_case",motor,(77,112,144,85))
            for name,h in hardware.items():
                if name!="XL330_case":add(scene,name,h,(195,123,72,120))
            for i,b in enumerate(bearings):add(scene,f"688_bearing_envelope_{i}",b,(95,96,92,255))
            add(scene,"8mm_output_shaft",shaft,(105,105,101,190))
            scene.export(OUT/f"elbow-{deg:03d}.glb")
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"index.html").write_text(Path(__file__).with_name("swarm_sketch_elbow_enclosed_viewer.html").read_text())
    for name,shape in (("relieved-upper-front",upper[0]),("relieved-upper-back-with-cover",upper[1]),
                       ("upper-carrier",carrier),("forearm-yoke",moving_yoke),
                       ("rear-cover",cap)):
        bd.export_step(shape,OUT/f"{name}.step")
    return report


if __name__=="__main__":
    result=main()
    print("result",{k:v for k,v in result.items() if k!="poses"},flush=True)
