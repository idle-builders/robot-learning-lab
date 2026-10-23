"""Round-actuator countercase for shoulder/hip yaw in the 350 mm sketch shell.

Run from the repository root with .venv/bin/python
experiments/compact_morphology/ape_design/torso_wildcard_case.py

This is a static geometry screen. It uses true cylinders for the sourced round
packages, exact torso/pelvis source B-reps, and explicit conceptual support
hardware. Axes, attachment details, thermal paths and moving-pose clearance
remain unvalidated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
import build as shell

OUT = ROOT / "artifacts/torso-wildcard-round-350"
BATTERY = {"name": "Tattu_4S850_candidate", "shape": (30, 60, 30.5),
           "center": (0, 5, 221), "mass_kg": .107}
# Body dimensions/mass/torque from maker product pages and drawings.
FAMILIES = {
    "Harmonic_RS F-5B".replace(" ", ""): {
        "diameter_mm": 25, "length_mm": 35.8, "mass_kg": .077,
        "torque_nm": .44, "rating": "allowable continuous",
        "source": "https://www.harmonicdrive.net/products/rotary-actuators/actuators-without-hollow-shaft/rsf; https://www.harmonicdrive.net/downloads/pdf-drawings/rsf-supermini-actuators; https://www.harmonicdrive.net/downloads/stp-files/rsf-actuators",
    },
    "CubeMars_AK40-10": {
        "diameter_mm": 53, "length_mm": 37, "mass_kg": .185,
        "torque_nm": 1.3, "rating": "rated (maker-listed)",
        "source": "https://www.cubemars.com/product/ak40-10-robotic-actuator.html",
    },
    "Motionlayer_Axion_R1": {
        "diameter_mm": 53, "length_mm": 26, "mass_kg": .160,
        "torque_nm": 1.0, "rating": "continuous torque",
        "source": "https://motionlayer.company/products/axion-r1-actuator",
    },
    "MyActuator_HM-50-05-80": {
        "diameter_mm": 52, "length_mm": 69.5, "mass_kg": .38,
        "torque_nm": 5, "rating": "rated torque",
        "source": "https://www.myactuator.com/hm-50-details (official product and 2D/3D files; drawing 1MT.055.031)",
    },
    "MyActuator_RMD-X2-P28-7": {
        "diameter_mm": 44, "length_mm": 63.5, "mass_kg": .26,
        "torque_nm": 2.5, "rating": "rated torque",
        "source": "https://www.myactuator.com/x2-7details (official product and 2D/3D files; drawing dated 2026-09-20)",
    },
}


def cylinder(center, diameter, length, axis="z"):
    # build123d cylinders are centred on Z; rotate to the requested shaft axis.
    obj = bd.Cylinder(diameter / 2, length)
    if axis == "x":
        obj = bd.Rot(0, 90, 0) * obj
    elif axis == "y":
        obj = bd.Rot(90, 0, 0) * obj
    return bd.Pos(*center) * obj


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def intersection(a, b):
    common = a.intersect(b)
    return round(sum(s.volume for s in common), 2) if common else 0.0


def exact_shell(wall):
    study = shell.Study(350, wall=wall, gap=.65).build()
    parts = [p for p in study.parts if p["name"].startswith(("torso_", "pelvis_"))]
    return study.parts, parts


def placements(family):
    # One vertical-axis yaw output on each side at both girdle levels.
    # Axial cases are vertical; coordinates are a deliberately documented
    # placement hypothesis. Shoulder: 29 mm lateral, z=244. Hip: 23 mm,
    # z=164. The battery is central and tilted only in the horizontal plane.
    rows = []
    d, length = family["diameter_mm"], family["length_mm"]
    for region, lateral, z in (("shoulder", 28, 244), ("hip", 28, 164)):
        for side in (-1, 1):
            rows.append({"name": f"{region}_{'L' if side < 0 else 'R'}_round_actuator",
                         "region": region, "center": (side*lateral, 3, z),
                         "shape": cylinder((side*lateral, 3, z), d, length),
                         "mass_kg": family["mass_kg"]})
    return rows


def support_hardware(rows):
    # A 6 mm supported output shaft and two assumed 12x4 mm bush seats per
    # actuator. Actual maker output interface and support stack are unverified.
    supports = []
    for r in rows:
        x, y, z = r["center"]
        shaft = cylinder((x, y, z), 6, 42)
        rings = []
        for zz in (z-20, z+20):
            outer = cylinder((x, y, zz), 12, 4)
            inner = cylinder((x, y, zz), 6.2, 4.2)
            rings.append(outer-inner)
        supports.append({"name": r["name"].replace("round_actuator", "supported_output_shaft"),
                         "shape": shaft, "mass_kg": .018, "kind": "support"})
        for i, ring in enumerate(rings):
            supports.append({"name": r["name"].replace("round_actuator", f"assumed_bearing_seat_{i+1}"),
                             "shape": ring, "mass_kg": .004, "kind": "support"})
    return supports


def mass_com(items):
    mass = sum(i["mass_kg"] for i in items)
    com = [sum(i["mass_kg"] * i["center"][j] for i in items)/mass for j in range(3)]
    # Point-mass diagonal inertia about the package COM; intrinsic actuator
    # inertia is unavailable and omitted. Results are kg mm^2.
    inertia = [0.0, 0.0, 0.0]
    for i in items:
        x, y, z = [i["center"][k]-com[k] for k in range(3)]
        m = i["mass_kg"]
        for j, value in enumerate((y*y+z*z, x*x+z*z, x*x+y*y)):
            inertia[j] += m*value
    return {"component_mass_kg": round(mass, 4), "component_COM_mm": [round(x, 2) for x in com],
            "point_mass_diagonal_inertia_kg_mm2": [round(x, 1) for x in inertia],
            "inertia_limit": "point masses only; package intrinsic inertia and unmodelled structure/electronics omitted"}


def evaluate(family_name, family, wall, source_parts):
    modules = placements(family)
    supports = support_hardware(modules)
    battery_shape = box(BATTERY["center"], BATTERY["shape"])
    skin = [p["shape"] for p in source_parts]
    all_items = [{**r, "kind": "actuator"} for r in modules]
    all_items += supports
    all_items += [{"name": BATTERY["name"], "shape": battery_shape, "kind": "battery",
                   "center": BATTERY["center"], "mass_kg": BATTERY["mass_kg"]}]
    collisions = []
    for i, item in enumerate(all_items):
        # Supports are intended coaxial with their motor and therefore excluded
        # from motor/support pair testing; all other pairs are tested exactly.
        for other in all_items[i+1:]:
            if item["kind"] == "support" and other["kind"] == "actuator":
                continue
            if item["kind"] == "actuator" and other["kind"] == "support":
                continue
            v = intersection(item["shape"], other["shape"])
            if v > .05:
                collisions.append({"a": item["name"], "b": other["name"], "intersection_mm3": v})
    shell_hits = {}
    for item in all_items:
        hits = [intersection(item["shape"], s) for s in skin]
        shell_hits[item["name"]] = round(sum(hits), 2)
    module_masses = [{"mass_kg": m["mass_kg"], "center": m["center"]} for m in modules]
    module_masses += [{"mass_kg": s["mass_kg"], "center": tuple(s["shape"].center())} for s in supports]
    module_masses += [{"mass_kg": BATTERY["mass_kg"], "center": BATTERY["center"]}]
    mass = mass_com(module_masses)
    mass.update({"actuator_count": len(modules), "actuator_only_mass_kg": round(sum(x["mass_kg"] for x in modules), 3),
                 "each_region_two_sided_actuator_mass_kg": round(2*family["mass_kg"], 3),
                 "torque_note": f"{family['torque_nm']} Nm per unit, maker basis: {family['rating']}; no gait duty inference"})
    return {"family": family_name, "source": family["source"], "wall_mm": wall,
            "package": {k: family[k] for k in ("diameter_mm", "length_mm", "mass_kg", "torque_nm", "rating")},
            "layouts": [{"name": m["name"], "region": m["region"], "center_mm": m["center"],
                         "axis": "Z (yaw)"} for m in modules],
            "fixed_component_intersections_mm3": collisions,
            "exact_torso_pelvis_brep_intersections_mm3": shell_hits,
            "mass_and_COM": mass,
            "appearance": {"actuator_shell_overlap_mm3": sum(v for k,v in shell_hits.items() if "round_actuator" in k),
                           "note": "positive shell overlap marks likely visible/protruding hardware in the intact shell; zero does not establish hidden appearance where source inner-volume corners fail"}}


def add_mesh(scene, name, shape, color):
    vertices, faces = shape.tessellate(.15, .15)
    mesh = trimesh.Trimesh(vertices=np.array([[v.X,v.Y,v.Z] for v in vertices]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def render(parts, family, rows, supports):
    scene = trimesh.Scene()
    for p in parts:
        if p["name"].startswith(("torso_", "pelvis_")):
            add_mesh(scene, p["name"], p["shape"], [225,216,196,90])
    for r in rows:
        add_mesh(scene, r["name"], r["shape"], [62,117,154,255])
    for s in supports:
        add_mesh(scene, s["name"], s["shape"], [213,153,74,255])
    add_mesh(scene, BATTERY["name"], box(BATTERY["center"], BATTERY["shape"]), [104,170,124,255])
    scene.export(OUT / f"{family}-intact-shell.glb")
    opened = trimesh.Scene()
    for r in rows:
        add_mesh(opened, r["name"], r["shape"], [62,117,154,255])
    for s in supports:
        add_mesh(opened, s["name"], s["shape"], [213,153,74,255])
    add_mesh(opened, BATTERY["name"], box(BATTERY["center"], BATTERY["shape"]), [104,170,124,255])
    opened.export(OUT / f"{family}-service-view.glb")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts2, source2 = exact_shell(2)
    _, source1 = exact_shell(1)
    report = {"height_mm": 350,
              "frame": "source sketch X lateral, Y rear, Z up",
              "shell": "Study(350, wall=1 or 2, gap=.65) exact torso/pelvis front and back B-reps",
              "battery": {"source": "Tattu 4S850 candidate; existing torso studies use 60x30x30.5 mm, 107 g", "center_mm": BATTERY["center"], "limits": "candidate only; battery cavity, straps, connectors and conversion hardware omitted"},
              "baseline": "This script computes a matched two-sided yaw-only conventional stack: published 2XL430 case envelopes at existing shoulder centers and XL330 yaw boxes at existing hip centers, plus the same rotated Tattu candidate. Baseline 2XL430 mass 102 g/package and XL330 mass 18 g/unit are source-label assumptions from the current joint-swarm records; remaining roll/pitch axes are excluded on both sides.",
              "method": "True cylindrical actuator bodies at sourced diameters/lengths. Vertical Z yaw modules at each shoulder and hip. 6 mm shaft with two conceptual 12x4 mm support rings per package. All fixed pairwise collisions and actuator-to-exact shell intersections computed as B-rep booleans.",
              "limitations": ["Harmonic STEP is listed by maker but this screen uses the published nominal envelope dimensions, not downloaded STEP face topology.", "CubeMars lists official STEP/2D drawing; primitive cylinder matches the published round envelope, connectors omitted.", "MyActuator official PDFs and STEP ZIPs were retrieved from its product pages; the screen uses the drawing's maximum diameter and axial length as cylinders.", "Support shaft/ring geometry and placements are hypotheses; maker integrated output flange bearings are not credited as robot structural supports.", "Only yaw axis at each side of shoulder and hip is screened; roll/pitch and limb attachment remain absent.", "No motion, cable, connector, heat, battery restraint, thermal duty cycle, price or production quote evaluation."],
              "families": {}}
    for name, family in FAMILIES.items():
        rows = placements(family)
        supports = support_hardware(rows)
        by_wall = {str(w): evaluate(name, family, w, source2 if w == 2 else source1) for w in (2,1)}
        report["families"][name] = by_wall
        render(parts2, name, rows, supports)
    baseline_items = [
        {"name": "shoulder_L_2XL430", "center": (-75,4,244), "size": (36,36,46.5), "mass_kg": .102},
        {"name": "shoulder_R_2XL430", "center": (75,4,244), "size": (36,36,46.5), "mass_kg": .102},
        {"name": "hip_L_XL330_yaw", "center": (-20,0,206), "size": (34,20,26), "mass_kg": .018},
        {"name": "hip_R_XL330_yaw", "center": (20,0,206), "size": (34,20,26), "mass_kg": .018},
    ]
    report["conventional_yaw_baseline"] = {}
    for wall, skins in ((2, source2), (1, source1)):
        shell_shapes = [p["shape"] for p in skins]
        shapes = [box(BATTERY["center"], BATTERY["shape"])] + [box(x["center"],x["size"]) for x in baseline_items]
        names = [BATTERY["name"]] + [x["name"] for x in baseline_items]
        report["conventional_yaw_baseline"][str(wall)] = {
            "wall_mm": wall,
            "case_to_shell_brep_intersections_mm3": {n: round(sum(intersection(sh,s) for s in shell_shapes),2) for n,sh in zip(names,shapes)},
            "fixed_component_intersections_mm3": [{"a":names[i],"b":names[j],"intersection_mm3":intersection(shapes[i],shapes[j])} for i in range(len(names)) for j in range(i+1,len(names)) if intersection(shapes[i],shapes[j])>.05],
            "component_mass_kg_including_candidate_pack": round(BATTERY["mass_kg"]+sum(x["mass_kg"] for x in baseline_items),3),
            "mass_note": "no bearings/frame/service mass; 2XL430 shoulder package has paired-axis function while each round package is one yaw axis; mass is not a full BOM comparison",
        }
    (OUT/"round-torso-screen.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:{w:{"pairwise_intersections":len(v["fixed_component_intersections_mm3"]),
                            "shell_hits_mm3":v["appearance"]["actuator_shell_overlap_mm3"],
                            "mass_kg":v["mass_and_COM"]["component_mass_kg"]}
                       for w,v in val.items()} for k,val in report["families"].items()}, indent=2))


if __name__ == "__main__":
    main()
