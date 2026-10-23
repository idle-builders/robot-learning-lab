"""Bounded XL330 torso-packaging competitor against the exact 350 mm shell.

Run from repo root with .venv/bin/python. This is a static occupancy coupon:
XL330 cases are source-sized, while output shafts, opposed supports and load
spine are explicit hypotheses. It does not infer continuous servo torque.
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
import build as shell_source

OUT = ROOT / "artifacts/torso-compact-servo-case-350"
XL_SIZE = (20.0, 34.0, 26.0)  # ROBOTIS XL330-M288-T case, source axes X/Y/Z
XL_MASS = 0.018
PACK_SIZE = (30.0, 60.0, 30.5)  # Tattu 4S850 rotated in X/Y; 107 g
PACK_MASS = 0.107
COLORS = {
    "shell": [228, 221, 208, 58], "servo": [53, 112, 157, 255],
    "pack": [104, 164, 116, 255], "support": [164, 177, 185, 255],
    "wire": [225, 169, 61, 140], "spine": [121, 128, 139, 255],
}


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def volume(a, b):
    common = a.intersect(b)
    return float(sum(s.volume for s in common)) if common else 0.0


def candidates():
    """Three purposeful source-coordinate layouts; no blind dense grid."""
    rows = []
    # Each side has a compact three-case bank in the anterior/posterior depth.
    # Fixed shell planes are X lateral, Y rearward, Z up. Outputs are assumed
    # to face laterally and couple through short shafts to separate supports.
    for label, hip_z, shoulder_z, pack_y, pack_z, bank_x in [
        ("central-pack", 153.0, 238.0, 0.0, 187.0, 39.0),
        ("low-pack", 153.0, 238.0, 0.0, 177.0, 39.0),
        ("narrow-banks", 153.0, 238.0, 0.0, 187.0, 34.0),
    ]:
        items = []
        for side, suffix in [(-1, "L"), (1, "R")]:
            # Hypothetical remote drive allocation: yaw/roll/pitch share a
            # three-case bank in depth; this is not an articulated mechanism.
            for i, (axis, yoff) in enumerate(zip(("yaw", "roll", "pitch"), (-21.0, 0.0, 21.0))):
                items.append({"name": f"hip_{suffix}_{axis}_XL330", "kind": "servo",
                              "center": (side * bank_x, yoff, hip_z), "size": XL_SIZE,
                              "mass_kg": XL_MASS, "layout": label,
                              "evidence": "XL330-M288-T case 20x34x26 mm, 18 g; axis/output coupling is a hypothesis"})
                items.append({"name": f"shoulder_{suffix}_{axis}_XL330", "kind": "servo",
                              "center": (side * bank_x, yoff, shoulder_z), "size": XL_SIZE,
                              "mass_kg": XL_MASS, "layout": label,
                              "evidence": "XL330-M288-T case 20x34x26 mm, 18 g; axis/output coupling is a hypothesis"})
        items.append({"name": "Tattu_4S850_candidate", "kind": "pack",
                      "center": (0.0, pack_y, pack_z), "size": PACK_SIZE,
                      "mass_kg": PACK_MASS, "layout": label,
                      "evidence": "Tattu 4S850 candidate, 60x30x30.5 mm, 107 g; rotated X/Y"})
        # Paired reaction cheeks and a simple common structural spine. These
        # are reserves that must remain non-overlapping; no load rating implied.
        for side, suffix in [(-1, "L"), (1, "R")]:
            for region, z in [("hip", hip_z), ("shoulder", shoulder_z)]:
                items.append({"name": f"{region}_{suffix}_shaft_support_reserve", "kind": "support",
                              "center": (side * (bank_x - 11.0), 0.0, z), "size": (4.0, 54.0, 32.0),
                              "mass_kg": 0.012, "layout": label,
                              "evidence": "assumption: paired 4 mm cheeks over 54 mm depth, 12 g allowance"})
        items.extend([
            {"name": "central_structural_spine_reserve", "kind": "spine",
             "center": (0.0, 0.0, 206.0), "size": (8.0, 20.0, 92.0),
             "mass_kg": 0.035, "layout": label,
             "evidence": "assumption: 8x20 mm continuous spine, 35 g allowance"},
            {"name": "shared_power_bus_and_connector_access", "kind": "wire",
             "center": (0.0, -29.0, 213.0), "size": (12.0, 12.0, 64.0),
             "mass_kg": 0.012, "layout": label,
             "evidence": "assumption: 12x12 mm accessible shared cable lane, 12 g harness allowance"},
            {"name": "shared_signal_cable_lane", "kind": "wire",
             "center": (0.0, 29.0, 213.0), "size": (12.0, 12.0, 64.0),
             "mass_kg": 0.006, "layout": label,
             "evidence": "assumption: 12x12 mm second bus/service lane, 6 g harness allowance"},
        ])
        rows.append((label, items))
    return rows


def report_layout(items, shell_parts):
    shapes = {i["name"]: box(i["center"], i["size"]) for i in items}
    skin = [p["shape"] for p in shell_parts if p["name"].startswith(("torso_", "pelvis_"))]
    intersections = []
    for n, a in enumerate(items):
        for b in items[n + 1:]:
            v = volume(shapes[a["name"]], shapes[b["name"]])
            if v > .05:
                intersections.append({"a": a["name"], "b": b["name"], "intersection_mm3": round(v, 2)})
    clipping = []
    for item in items:
        v = sum(volume(shapes[item["name"]], s) for s in skin)
        if v > .05:
            clipping.append({"name": item["name"], "shell_intersection_mm3": round(v, 2)})
    total = sum(i["mass_kg"] for i in items)
    com = [sum(i["mass_kg"] * i["center"][k] for i in items) / total for k in range(3)]
    return {"reservation_mass_kg": round(total, 4), "reservation_com_mm": [round(v, 2) for v in com],
            "positive_pair_intersections": intersections, "shell_clipping": clipping,
            "intersection_pair_count": len(intersections),
            "pair_intersection_sum_mm3_nonexclusive": round(sum(x["intersection_mm3"] for x in intersections), 2)}


def add_mesh(scene, name, shape, kind):
    verts, faces = shape.tessellate(.2, .2)
    mesh = trimesh.Trimesh(vertices=np.asarray([[v.X, v.Y, v.Z] for v in verts]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = COLORS[kind]
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell_parts = shell_source.Study(350, wall=2.0, gap=.65).build().parts
    thin_parts = shell_source.Study(350, wall=1.0, gap=.65).build().parts
    reports, layouts = {}, {}
    for name, items in candidates():
        report = report_layout(items, shell_parts)
        thin = report_layout(items, thin_parts)
        reports[name] = {"wall_2mm": report, "wall_1mm": thin,
                         "items": items}
        layouts[name] = items
    # one readable neutral viewer per candidate plus JSON; exact source shell
    # parts remain as BRep context, while exported viewers are triangulated.
    for name, items in layouts.items():
        scene = trimesh.Scene()
        for p in shell_parts:
            if p["name"].startswith(("torso_", "pelvis_")):
                add_mesh(scene, p["name"], p["shape"], "shell")
        for item in items:
            add_mesh(scene, item["name"], box(item["center"], item["size"]), item["kind"])
        scene.export(OUT / f"{name}.glb")
    result = {
        "source_frame": "X lateral, Y rearward, Z up; millimetres",
        "source_shell": "experiments/sketch_ape_shell/build.py Study(350, wall=1 or 2, gap=.65); exact torso/pelvis front/back BReps",
        "scope": "three bounded static occupancies of 12 XL330 cases, source-sized battery and stated structure/service reserves; no moving-axis, torque, thermal, assembly or print qualification",
        "evidence": {
            "XL330": "ROBOTIS XL330-M288-T published outer case 20x34x26 mm and 18 g; 0.52 N m is stall torque at 5 V, not continuous rating",
            "battery": "Tattu 4S850 candidate 60x30x30.5 mm and 107 g per prior source screen",
            "reference": "Microban is a roughly 300 mm, 19-XL330 printed humanoid; its opposed shim supports and connector-before-insertion assembly guide are relevant patterns, not proof for this 350 mm ape",
        },
        "layouts": reports,
        "inertia_limit": "No CAD-resolved internal servo rotor or support tensors; report provides only point-mass COM for listed reservations.",
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("torso_compact_servo_viewer.html").read_text())
    print(json.dumps({k: {w: v[w]["intersection_pair_count"] for w in ("wall_1mm", "wall_2mm")} for k,v in reports.items()}))


if __name__ == "__main__":
    main()
