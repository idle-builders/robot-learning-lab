"""Bounded fixed-axis offset-knee coupon for the exact 350 mm ape shell.

X points right, Y rear, Z up; dimensions are millimetres. This models an
XL330-size case, a paired bearing path and a rigid shin yoke. It is an
appearance and interference screen, not a released actuator mount.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/lightweight-ape-20261004/source-shell-1mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/lightweight-ape-offset-knee-20261005"
AXIS = (42.0, 14.0, 106.0)
POSES = (0, 60, 115)
DENSITY_G_PER_MM3 = 1.24e-3  # Same provisional PLA density as shell ledger.


def box(x0, x1, y0, y1, z0, z1):
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0, bd.Plane(origin=(x0, y0, z0)))


def cylinder(x0, x1, y=14, z=106, r=3):
    return bd.Solid.make_cylinder(r, x1-x0,
                                  bd.Plane(origin=(x0, y, z), z_dir=(1, 0, 0)))


def cheek(x0, x1):
    # Narrow link from the offset pivot to a rib engaging both source shin skins.
    yz = [(10,107),(18,107),(14,101),(8,83),(18,79),(18,77),
          (-1,77),(-1,81),(5,84)]
    wire = bd.Wire.make_polygon([(x0,y,z) for y,z in yz], close=True)
    return bd.Solid.extrude(bd.Face(wire), (x1-x0,0,0)).fuse(cylinder(x0,x1,r=5.5))


def compound(items):
    return bd.Compound(children=[s for item in items for s in item.solids()
                                 if s.volume > 1e-6])


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def turn(shape, degrees):
    return bd.Pos(*AXIS) * bd.Rot(degrees,0,0) * bd.Pos(*(-v for v in AXIS)) * shape


def overlap(a,b):
    return sum(volume(sa & sb) for sa in a.solids() for sb in b.solids())


def gap(a,b):
    return min(sa.distance_to(sb) for sa in a.solids() for sb in b.solids())


def mesh(shape, rgba):
    verts, faces = shape.tessellate(.45,.35)
    result = trimesh.Trimesh(np.asarray([tuple(v) for v in verts]),
                             np.asarray(faces), process=False)
    result.visual.face_colors = rgba
    return result


def geometry():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Build source shell first: {SOURCE}")
    parts = {s.label:s for s in bd.import_step(SOURCE).children}
    thigh = compound(parts[n] for n in ("R_thigh_front", "R_thigh_back"))
    shin = compound(parts[n] for n in ("R_shin_front", "R_shin_back"))
    cuff = parts["R_knee_cuff"]

    # XL330 manual: 20 x 34 x 26 mm overall, 18 g. The case/boss split and
    # 9.5 mm output offset use the provisional distributor drawing datum from
    # the earlier coupon. Output lies at the *lower* end of the inverted case.
    motor_body = box(31,54,4,24,96.5,130.5)
    output_boss = cylinder(54,57,r=8)
    idler_boss = cylinder(28,31,r=8)
    motor = compound((motor_body, output_boss, idler_boss))

    # Paired support brackets join over the top of the case. The thin source
    # skin is only an appearance/contact datum, never the bearing itself.
    left_support = cylinder(21.5,25.0,r=7).fuse(box(21.5,25.0,10,18,105,135))
    right_support = cylinder(59.5,63.1,r=7).fuse(box(59.5,63.1,10,18,105,135))
    top_bridge = box(21.5,63.1,10,25,132.5,136)
    carrier = left_support.fuse(right_support,top_bridge)
    carrier -= cylinder(21.4,25.1,r=4.55)
    carrier -= cylinder(59.4,63.2,r=4.55)
    left_bearing = cylinder(22,25,r=4.5) - cylinder(21.9,25.1,r=2.55)
    right_bearing = cylinder(59.6,62.6,r=4.5) - cylinder(59.5,62.7,r=2.55)

    # Right cheek receives the nominal output boss; left cheek supplies a
    # journal through the opposed support. Geometry does not certify the horn.
    left_cheek = cheek(25.5,27.5)
    right_cheek = cheek(57.1,59.1)
    root_rib = box(25.5,59.1,-1,18,77,80.5)
    left_journal = cylinder(22.0,25.5,r=2.5)
    right_journal = cylinder(59.1,62.6,r=2.5)
    yoke = left_cheek.fuse(right_cheek,root_rib,left_journal,right_journal)

    return locals()


def make_report(g):
    source_thigh, source_shin = g["thigh"], g["shin"]
    cuff = g["cuff"]
    fixed_structure = compound((g["motor"],g["carrier"],
                                g["left_bearing"],g["right_bearing"]))
    report = {
        "purpose":"bounded offset fixed-axis and continuous-cover screen; no drive selection",
        "axis_xyz_mm":AXIS,
        "source_step":str(SOURCE.relative_to(ROOT)),
        "source_wall_mm":1,
        "xl330_body_mm":[20,34,26],
        "xl330_motor_mass_g":18,
        "geometry_datum_note":"23 mm case plus 3 mm boss and 9.5 mm output offset provisional distributor drawing, as prior XL330 coupon",
        "source_shell_volume_mm3":{"thigh":round(volume(source_thigh),2),
                                  "shin":round(volume(source_shin),2),
                                  "cuff":round(volume(cuff),2)},
        "mechanism_volume_mm3":{"carrier":round(volume(g["carrier"]),2),
                                "yoke":round(volume(g["yoke"]),2)},
        "provisional_pla_mass_g":{"carrier":round(volume(g["carrier"])*DENSITY_G_PER_MM3,2),
                                  "moving_yoke":round(volume(g["yoke"])*DENSITY_G_PER_MM3,2)},
        "neutral_motor_vs_source_thigh_mm3":round(overlap(g["motor"],source_thigh),2),
        "neutral_carrier_vs_source_thigh_mm3":round(overlap(g["carrier"],source_thigh),2),
        "neutral_yoke_vs_source_shin_mm3":round(overlap(g["yoke"],source_shin),2),
        "poses":[]}
    # Rear contour at relevant axis height, from the exact generated loft.
    # This is a reference for the side silhouette, not a replacement for B-rep.
    source_rear_at_axis = 3 + 22 + (28-22)*(106-98)/(124-98)
    report["source_rear_at_axis_mm_y_approx"] = round(source_rear_at_axis,2)
    for degrees in range(0,121,5):
        moving_shin = turn(source_shin,degrees)
        moving_cuff = turn(cuff,degrees)
        moving_yoke = turn(g["yoke"],degrees)
        moving_skin = compound((moving_shin,moving_cuff))
        # The original cosmetic cuff remains assigned to the shin; at 0° it
        # exactly recreates the source exterior. It must move to cover a bend.
        def point(y,z):
            a=math.radians(degrees)
            return (AXIS[1]+math.cos(a)*(y-AXIS[1])-math.sin(a)*(z-AXIS[2]),
                    AXIS[2]+math.sin(a)*(y-AXIS[1])+math.cos(a)*(z-AXIS[2]))
        shin_top = point(3,83)
        cuff_front = point(-14,86)
        row = {
            "flexion_deg":degrees,
            "source_thigh_vs_moving_shin_mm3":round(overlap(source_thigh,moving_shin),3),
            "source_thigh_vs_moving_cuff_mm3":round(overlap(source_thigh,moving_cuff),3),
            "source_thigh_vs_moving_yoke_mm3":round(overlap(source_thigh,moving_yoke),3),
            "fixed_structure_vs_moving_shin_mm3":round(overlap(fixed_structure,moving_shin),3),
            "fixed_structure_vs_moving_cuff_mm3":round(overlap(fixed_structure,moving_cuff),3),
            "fixed_structure_vs_moving_yoke_mm3":round(overlap(fixed_structure,moving_yoke),3),
            "source_thigh_to_moving_skin_gap_mm":round(gap(source_thigh,moving_skin),3),
            "source_thigh_to_moving_yoke_gap_mm":round(gap(source_thigh,moving_yoke),3),
            "shin_top_center_yz_mm":[round(v,2) for v in shin_top],
            "moving_cuff_front_edge_yz_mm":[round(v,2) for v in cuff_front],
            "moving_cuff_rear_extent_y_mm":round(moving_cuff.bounding_box().max.Y,2),
            "moving_yoke_rear_extent_y_mm":round(moving_yoke.bounding_box().max.Y,2),
            "skin_rear_excursion_past_source_thigh_at_axis_mm":round(
                max(moving_cuff.bounding_box().max.Y,shin_top[0])-source_rear_at_axis,2),
        }
        report["poses"].append(row)
    return report


def export(g,report):
    OUT.mkdir(parents=True,exist_ok=True)
    colors = {"thigh":(225,215,194,255),"shin":(230,211,179,255),
              "cuff":(68,70,69,255),"carrier":(81,92,91,255),
              "motor":(94,136,165,255),"bearing":(159,160,153,255),
              "yoke":(185,113,70,255)}
    fixed = [("thigh",g["thigh"]),("carrier",g["carrier"]),
             ("motor",g["motor"]),("bearing",g["left_bearing"]),
             ("bearing",g["right_bearing"])]
    for degrees in POSES:
        moving = [("shin",turn(g["shin"],degrees)),
                  ("cuff",turn(g["cuff"],degrees)),
                  ("yoke",turn(g["yoke"],degrees))]
        scene = trimesh.Scene()
        for i,(name,shape) in enumerate(fixed+moving):
            for j,solid in enumerate(shape.solids()):
                if solid.volume < .01: continue
                node=f"{name}-{i}-{j}"
                scene.add_geometry(mesh(solid,colors[name]),geom_name=node,node_name=node)
        scene.export(OUT/f"offset-knee-{degrees:03d}.glb")
        labeled=[]
        for name,shape in fixed+moving:
            shape.label=name
            labeled.append(shape)
        bd.export_step(compound(labeled),OUT/f"offset-knee-{degrees:03d}.step")
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")


def main():
    g=geometry()
    report=make_report(g)
    export(g,report)
    print(json.dumps({"axis":AXIS,"mass_g":report["provisional_pla_mass_g"],
                      "poses":[r for r in report["poses"] if r["flexion_deg"] in POSES]},indent=2))


if __name__=="__main__":
    main()
