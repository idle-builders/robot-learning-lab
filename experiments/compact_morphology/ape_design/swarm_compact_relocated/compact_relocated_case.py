"""Compact remote-drive architecture screen on exact sketch-06 shell.

Run from the repository root: .venv/bin/python
experiments/compact_morphology/ape_design/swarm_compact_relocated/compact_relocated_case.py

All dimensions are millimetres and the source shell frame is X lateral,
Y rearward, Z up. Case/support/transmission blocks are source-sized or marked
assumptions. This is a CAD concept gate, not an assembled load-rated robot.
"""
from __future__ import annotations

import json
import sys
import copy
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
import build as sketch

OUT = ROOT / "artifacts/compact-relocated-350"
CASE = (20.0, 34.0, 26.0)  # XL330 published outer case, not service envelope
MOTOR_KG = .018
BATTERY = (30.0, 60.0, 30.5)
COLORS = {"skin": [225, 219, 205, 65], "drive": [48, 105, 153, 255],
          "support": [150, 166, 176, 255], "belt": [205, 128, 64, 255],
          "pack": [100, 159, 112, 255], "service": [227, 177, 61, 170]}


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def item_key(item):
    return f"{item['side']}_{item['joint']}"


def vol(a, b):
    aa, bb = a.bounding_box(), b.bounding_box()
    if (aa.max.X <= bb.min.X or bb.max.X <= aa.min.X or
        aa.max.Y <= bb.min.Y or bb.max.Y <= aa.min.Y or
        aa.max.Z <= bb.min.Z or bb.max.Z <= aa.min.Z):
        return 0.0
    c = a.intersect(b)
    return float(sum(s.volume for s in c)) if c else 0.0


def build_architecture():
    """21 one-axis servos on proximal parent links; short remote-belt outputs."""
    joints = []
    # Geometry is a first-placement hypothesis at source body landmarks. Each
    # output has a separate support reserve; the cosmetic shell is not a bearing.
    for s, side in [(-1, "L"), (1, "R")]:
        for name, link, pos, ratio, axis in [
            ("hip_yaw", "pelvis", (s*13, 0, 161), 4, "Z"),
            ("hip_roll", "pelvis", (s*31, 0, 158), 4, "X"),
            ("hip_pitch", "thigh", (s*31, -2, 146), 4, "X"),
            ("knee_pitch", "thigh", (s*34, 0, 111), 4, "X"),
            ("ankle_pitch", "shin", (s*32, 0, 62), 3, "X"),
            ("shoulder_yaw", "torso", (s*14, 0, 251), 4, "Z"),
            ("shoulder_roll", "upper_arm", (s*55, 0, 239), 4, "Y"),
            ("shoulder_pitch", "upper_arm", (s*55, 0, 222), 3, "X"),
            ("elbow_pitch", "upper_arm", (s*54, 0, 202), 2, "X"),
            ("opening_paw", "forearm", (s*59, 0, 181), 1, "X"),
        ]:
            joints.append({"joint": name, "side": side, "parent_link": link,
                           "center_mm": pos, "case_mm": CASE, "motor_mass_kg": MOTOR_KG,
                           "bearing_pair_mass_kg": .004, "belt_pulley_reserve_mass_kg": .006,
                           "ratio_motor_to_output": f"{ratio}:1", "axis": axis,
                           "belt_assumption": "open 6 mm wide synchronous belt, center distance 16-24 mm; pulley tooth counts and tension unresolved",
                           "output_support": "two separate opposed 4x8x5 mm bearing envelope blocks at each output; shaft/retention unresolved"})
    joints.append({"joint": "head_yaw", "side": "C", "parent_link": "head_neck",
                   "center_mm": (0, 0, 273), "case_mm": CASE, "motor_mass_kg": MOTOR_KG,
                   "ratio_motor_to_output": "1:1", "axis": "Z",
                   "belt_assumption": "direct compact neck output; travel and cable loop unresolved",
                   "output_support": "two separate opposed 4x8x5 mm bearing envelope blocks; shaft/retention unresolved"})
    # Shared battery and central power reserves are deliberately explicit.
    joints.extend([
        {"joint": "battery_4S850", "side": "C", "parent_link": "pelvis", "center_mm": (0, 0, 190),
         "case_mm": BATTERY, "motor_mass_kg": .107, "ratio_motor_to_output": "n/a",
         "axis": "n/a", "belt_assumption": "Tattu candidate; strap/retention assumed, no service clearance", "output_support": "n/a"},
        {"joint": "power_converter_reserve", "side": "C", "parent_link": "torso", "center_mm": (0, 14, 217),
         "case_mm": (24, 28, 12), "motor_mass_kg": .025, "ratio_motor_to_output": "n/a",
         "axis": "n/a", "belt_assumption": "assumed 25 g buck/converter reserve for 4S to 5 V servo bus", "output_support": "n/a"},
        {"joint": "central_spine_reserve", "side": "C", "parent_link": "torso", "center_mm": (0, -8, 214),
         "case_mm": (8, 16, 96), "motor_mass_kg": .035, "ratio_motor_to_output": "n/a",
         "axis": "n/a", "belt_assumption": "assumed 35 g inner load spine; material and anchors unresolved", "output_support": "n/a"},
        {"joint": "shared_bus_lane", "side": "C", "parent_link": "torso", "center_mm": (0, 25, 215),
         "case_mm": (5, 8, 88), "motor_mass_kg": .018, "ratio_motor_to_output": "n/a",
         "axis": "n/a", "belt_assumption": "assumed shared 5x8 mm cable trunk; branch connectors/bend radii unresolved", "output_support": "n/a"},
    ])
    # Add a 12 mm driven pulley envelope near each output. Pair center derives
    # from a short belt hypothesis, not a verified servo horn or shaft drawing.
    for item in joints[:21]:
        x, y, z = item["center_mm"]
        axis = item["axis"]
        off = 18.0
        output = (x, y + off, z) if axis in ("X", "Z") else (x, y, z + off)
        item["output_center_mm"] = output
        item["pulley_mm"] = (12.0, 6.0, 12.0)
        item["bearing_pair_center_mm"] = output
    return joints


def build_vertical_phase_alternative(items):
    """Separate stacked actuator pairs in Z, accepting possible shell growth."""
    alt = copy.deepcopy(items)
    proposed = {
        ("hip_roll", "L"): (-34, 0, 158), ("hip_roll", "R"): (34, 0, 158),
        ("hip_pitch", "L"): (-34, -2, 126), ("hip_pitch", "R"): (34, -2, 126),
        ("shoulder_pitch", "L"): (-55, 0, 205), ("shoulder_pitch", "R"): (55, 0, 205),
        ("elbow_pitch", "L"): (-54, 0, 175), ("elbow_pitch", "R"): (54, 0, 175),
        ("opening_paw", "L"): (-59, 0, 145), ("opening_paw", "R"): (59, 0, 145),
    }
    for item in alt[:21]:
        key = (item["joint"], item["side"])
        if key in proposed:
            item["center_mm"] = proposed[key]
            x, y, z = item["center_mm"]
            item["output_center_mm"] = (x, y + 22, z) if item["axis"] in ("X", "Z") else (x, y, z + 22)
            item["pulley_mm"] = (12.0, 6.0, 12.0)
            item["bearing_pair_center_mm"] = item["output_center_mm"]
    return alt


def shape_item(item):
    return box(item["center_mm"], item["case_mm"])


def source_skin(parts, pose="neutral"):
    names = ("torso_", "pelvis_", "L_upper_arm", "R_upper_arm", "L_forearm", "R_forearm",
             "L_thigh", "R_thigh", "L_shin", "R_shin", "head_cranium", "neck_collar")
    return [p for p in parts if p["name"].startswith(names)]


def add_mesh(scene, name, shape, category):
    verts, faces = shape.tessellate(.2, .2)
    mesh = trimesh.Trimesh(vertices=np.asarray([[v.X, v.Y, v.Z] for v in verts]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = COLORS[category]
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def pose_shapes(joints):
    """Apply source-relevant isolated B-rep output sweeps around test axes."""
    sampled = {"hip_pitch", "knee_pitch", "ankle_pitch", "shoulder_pitch",
               "elbow_pitch", "opening_paw", "head_yaw"}
    angles = {"hip_pitch": (-54, 0, 40), "knee_pitch": (0, 60, 115),
              "ankle_pitch": (-35, 0, 35), "shoulder_pitch": (-60, 0, 60),
              "elbow_pitch": (0, 60, 115), "opening_paw": (0, 45),
              "head_yaw": (-60, 0, 60)}
    checks = []
    for item in joints[:21]:
        if item["joint"] not in sampled or item["side"] not in ("L", "C"):
            continue
        pivot = item["output_center_mm"]
        axis_vec = {"X": (1,0,0), "Y": (0,1,0), "Z": (0,0,1)}[item["axis"]]
        axis = bd.Axis(pivot, axis_vec)
        base = shape_item(item)
        for angle in angles[item["joint"]]:
            moved = base.rotate(axis, angle)
            checks.append({"joint": item["joint"], "side": item["side"], "angle_deg": angle,
                           "shape": moved})
    return checks


def compute(joints, skin_parts, sample_motion=True):
    shapes = {item_key(x): shape_item(x) for x in joints}
    positive = []
    for n, item in enumerate(joints):
        a = shapes[item_key(item)]
        for other in joints[n+1:]:
            v = vol(a, shapes[item_key(other)])
            if v > .05:
                positive.append({"a": item_key(item), "b": item_key(other), "intersection_mm3": round(v, 2)})
    clipping = []
    for item in joints:
        shape = shapes[item_key(item)]
        v = sum(vol(shape, part["shape"]) for part in skin_parts)
        if v > .05:
            clipping.append({"name": item_key(item), "shell_intersection_mm3": round(v, 2)})
    poses = []
    if sample_motion:
        for row in pose_shapes(joints):
            # Static shell pose is intentionally held fixed: this is a conservative
            # motor-vs-neutral-skin sweep, not a full articulated motion proof.
            v = sum(vol(row["shape"], p["shape"]) for p in skin_parts)
            if v > .05:
                poses.append({"joint": item_key(row), "side": row["side"],
                              "case_angle_deg": row["angle_deg"], "skin_intersection_mm3": round(v, 2)})
    mass_at = [x["motor_mass_kg"] + x.get("bearing_pair_mass_kg", 0) + x.get("belt_pulley_reserve_mass_kg", 0) for x in joints]
    mass = sum(mass_at)
    com = [sum(mass_at[i]*x["center_mm"][k] for i,x in enumerate(joints))/mass for k in range(3)]
    return {"reserved_mass_kg": round(mass, 4), "point_mass_com_mm": [round(v,2) for v in com],
            "fixed_pair_intersections": positive, "shell_clipping": clipping,
            "isolated_motor_rotation_vs_neutral_skin": poses,
            "isolated_rotation_sampled": sample_motion,
            "pair_intersection_count": len(positive), "pose_collision_count": len(poses)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    architecture = build_architecture()
    wall_reports = {}
    shell_data = {}
    phased = build_vertical_phase_alternative(architecture)
    for wall in (1.0, 2.0):
        parts = sketch.Study(350, wall=wall, gap=.65).build().parts
        shell_data[wall] = parts
        skin = source_skin(parts)
        wall_reports[str(int(wall))] = {
            "distributed_parent_link_drive": compute(architecture, skin),
            "vertical_phase_alt": compute(phased, skin, sample_motion=False),
        }
    # Three-quarter neutral source shell and every envelope are inspectable.
    for wall in (1.0, 2.0):
        scene = trimesh.Scene()
        parts = source_skin(shell_data[wall])
        for p in parts:
            add_mesh(scene, p["name"], p["shape"], "skin")
        for item in architecture:
            add_mesh(scene, item_key(item), shape_item(item),
                     "pack" if item["joint"] == "battery_4S850" else
                     "service" if item["joint"].endswith(("reserve", "lane")) else "drive")
        scene.export(OUT / f"neutral-wall-{int(wall)}mm.glb")
        phased_scene = trimesh.Scene()
        for p in parts:
            add_mesh(phased_scene, p["name"], p["shape"], "skin")
        for item in phased:
            add_mesh(phased_scene, item_key(item), shape_item(item),
                     "pack" if item["joint"] == "battery_4S850" else
                     "service" if item["joint"].endswith(("reserve", "lane")) else "drive")
        phased_scene.export(OUT / f"vertical-phase-wall-{int(wall)}mm.glb")
    # Representative isolated joint stress samples rotate cases against the
    # neutral shell; link transforms are omitted and results are not valid
    # coupled-motion acceptance evidence.
    mid = trimesh.Scene()
    for p in source_skin(shell_data[2.0]):
        add_mesh(mid, p["name"], p["shape"], "skin")
    for row in pose_shapes(architecture):
        if row["angle_deg"] != 0:
            add_mesh(mid, f"{row['joint']}_{row['side']}_{row['angle_deg']}", row["shape"], "drive")
    mid.export(OUT / "isolated-motor-rotation-vs-neutral-shell.glb")
    report = {
        "status": "concept / static exact-BRep probe; not an assembly candidate",
        "source_shell": "experiments/sketch_ape_shell/build.py Study(350, wall=1 or 2, gap=.65), exact source torso/pelvis/upper-arm/forearm/thigh/shin/head skin solids",
        "coordinates": "X lateral, Y rearward, Z up; mm",
        "motors": "21 source-sized XL330-M288-T cases, 20x34x26 mm, 18 g, 5 V, 0.52 N m stall only; no continuous torso/limb rating",
        "ratio_method": "estimated output stall ceilings ratio*0.52N m*0.65 assumed belt efficiency; not usable torque ratings: 4:1=1.35, 3:1=1.01, 2:1=0.68, 1:1=0.34 N m. No load-speed/thermal qualification.",
        "battery_and_power": "Tattu 4S850 107 g, 30x60x30.5 mm rotated; 4S to 5V converter 25 g reserve and 18 g harness lane assumptions",
        "wall_sensitivity": wall_reports,
        "axes": architecture,
        "vertical_phase_alt_axes": phased,
        "limitations": ["bearing pairs and belts are text/mass assumptions only; their solids are not geometry-exported or collision-checked",
                        "opposed bearing blocks are not fastener-designed supports or selected bearings",
                        "belt center distances, pulley tooth counts, wrap angle and tension are not resolved",
                        "pose sweep rotates each motor box alone about a hypothesized output and holds all shell links neutral; not articulated motion clearance",
                        "skin BRep collisions use exact source parts, but no shell is transformed with articulated joints",
                        "mass includes component/support/electrical reserves only; no CAD inertia tensor or learned-policy claim",
                        "XL330 stall torque is not continuous capability; larger loads may reject this architecture"]
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("compact_relocated_viewer.html").read_text())
    print(json.dumps({wall: {layout: {"pair_intersections": v["pair_intersection_count"],
                                      "shell_clips": len(v["shell_clipping"]),
                                      "isolated_rotations_intersecting": v["pose_collision_count"]}
                              for layout,v in row.items()}
                      for wall,row in wall_reports.items()}))


if __name__ == "__main__":
    main()
