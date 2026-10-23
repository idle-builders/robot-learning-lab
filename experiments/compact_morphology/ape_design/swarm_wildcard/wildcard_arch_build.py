"""Build the tendon-loom wildcard against the actual 350 mm sketch shell.

Run from the repository root with .venv/bin/python. Generated STEP, source
mesh, reports and the interactive viewer are written under ignored artifacts/.
This is a packaging concept and sampled moving visualization, not a validated
assembly or a print-ready mechanism.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from build123d import Compound, Plane, Solid, export_step

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "experiments" / "sketch_ape_shell"
OUT = ROOT / "artifacts" / "ape-wildcard-tendon-loom"
sys.path.insert(0, str(SOURCE))
from build import Study  # noqa: E402

XL_CASE = (20.0, 34.0, 26.0)  # source-sized XL330 body, mm
XL_MASS_G = 18.0
XL_STALL_NM = 0.52
XL_PRICE_USD = 27.49
PACK_DIMS = (45.0, 15.0, 24.0)  # Tattu 450 mAh 2S, official brochure nominal
PACK_MASS_G = 28.0
PLA_G_CM3 = 1.24
IVORY = "#dfd8c6"
JOINT = "#6b675c"
FRAME = "#4d5856"
MOTOR = "#bd624b"
TENDON = "#bd965c"
METAL = "#838b88"
BATTERY = "#657b8c"


def box_shape(center, dims):
    x, y, z = center
    dx, dy, dz = dims
    return Solid.make_box(dx, dy, dz, Plane(origin=(x-dx/2, y-dy/2, z-dz/2)))


def annular_support(center, axis, outer_radius, inner_radius, depth):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    origin = np.asarray(center, dtype=float)-axis*depth/2
    plane = Plane(origin=tuple(origin), z_dir=tuple(axis))
    outer = Solid.make_cylinder(outer_radius, depth, plane)
    inner = Solid.make_cylinder(inner_radius, depth, plane)
    return outer-inner


def link_rod(a, b, radius):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    direction = b-a
    length = float(np.linalg.norm(direction))
    return Solid.make_cylinder(radius, length,
        Plane(origin=tuple(a), z_dir=tuple(direction/length)))


def aabb_overlap(a, b):
    return bool(np.all(a[0] < b[1]) and np.all(b[0] < a[1]))


def ident():
    return np.eye(4)


def trans(p):
    m = ident()
    m[:3, 3] = p
    return m


def rot(axis, angle):
    x, y, z = np.asarray(axis, dtype=float)
    n = np.linalg.norm((x, y, z))
    if n == 0 or angle == 0:
        return ident()
    x, y, z = x/n, y/n, z/n
    c, s, q = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    r = np.array([[c+x*x*q, x*y*q-z*s, x*z*q+y*s],
                  [y*x*q+z*s, c+y*y*q, y*z*q-x*s],
                  [z*x*q-y*s, z*y*q+x*s, c+z*z*q]])
    m = ident(); m[:3, :3] = r
    return m


def around(p, axis, degrees):
    return trans(p) @ rot(axis, math.radians(degrees)) @ trans(-np.asarray(p))


def owner_for_shell(name):
    if name.startswith("head_") or "eye_" in name or name in {
        "face_mask", "muzzle_shell", "mouth_inlay", "L_nostril_inlay", "R_nostril_inlay"
    }:
        return "head"
    for side in ("L", "R"):
        prefix = side + "_"
        if not name.startswith(prefix):
            continue
        item = name[len(prefix):]
        if item.startswith("upper_arm") or item in {"shoulder_trim", "elbow_cuff"}:
            return f"upper_{side}"
        if item.startswith("forearm") or item in {"wrist_cuff", "palm_front", "palm_back", "finger_1_cap", "finger_2_cap", "finger_3_cap", "finger_4_cap"}:
            return f"forearm_{side}"
        if item == "thumb_cap":
            return f"thumb_{side}"
        if item.startswith("thigh") or item == "knee_cuff":
            return f"thigh_{side}"
        if item.startswith("shin") or item == "ankle_cuff":
            return f"shin_{side}"
        if item.startswith("foot_") or item.startswith("toe_") or item == "sole":
            return f"foot_{side}"
    return "root"


def pose_transforms(pose):
    """World transforms for shell/link groups; matrices are 4x4 column-vector."""
    body = ident()
    neck_yaw = 0
    shoulder = {s: [0.0, 0.0, 0.0] for s in ("L", "R")}  # yaw, roll, pitch
    elbow = {s: 0.0 for s in ("L", "R")}
    jaw = {s: 0.0 for s in ("L", "R")}
    hip = {s: [0.0, 0.0, 0.0] for s in ("L", "R")}  # yaw, roll, pitch
    knee = {s: 0.0 for s in ("L", "R")}
    ankle = {s: 0.0 for s in ("L", "R")}
    if pose == "arms-out":
        shoulder["L"][1], shoulder["R"][1] = -50, 50
    elif pose == "arms-forward":
        shoulder["L"][2], shoulder["R"][2] = -55, -55
    elif pose == "arm-floor-reach":
        shoulder["R"][2], elbow["R"] = -35, -105
    elif pose == "elbow-60":
        elbow["L"], elbow["R"] = -60, -60
    elif pose == "elbow-115":
        elbow["L"], elbow["R"] = -115, -115
    elif pose == "walk-swing":
        hip["L"][2], knee["L"], ankle["L"] = -22, -42, 12
    elif pose == "knee-60":
        knee["L"], knee["R"] = -60, -60
    elif pose == "knee-115":
        knee["L"], knee["R"] = -115, -115
    elif pose == "ankle-plus-35":
        ankle["L"], ankle["R"] = 35, 35
    elif pose == "ankle-minus-35":
        ankle["L"], ankle["R"] = -35, -35
    elif pose == "squat-reset":
        hip["L"][2], hip["R"][2] = -54, -54
        knee["L"], knee["R"] = 109, 109
    elif pose == "crawl":
        body = around((0, 4, 155), (1, 0, 0), -16)
        for s in ("L", "R"):
            shoulder[s][2], elbow[s] = -62, -65
            hip[s][2], knee[s], ankle[s] = 28, -82, -20
    out = {"root": body}
    neck_p = (0, 4, 276)
    out["head"] = body @ around(neck_p, (0, 0, 1), neck_yaw)
    for s, sign in (("L", -1), ("R", 1)):
        sh = (sign*67, 4, 245)
        sh_m = body @ around(sh, (0, 0, 1), shoulder[s][0]) \
            @ around(sh, (1, 0, 0), shoulder[s][1]) \
            @ around(sh, (0, 1, 0), shoulder[s][2])
        el = (sign*96, 2, 194.5)
        elbow_m = sh_m @ around(el, (1, 0, 0), elbow[s])
        th = (sign*39, 3, 155)
        hip_m = body @ around(th, (0, 0, 1), hip[s][0]) \
            @ around(th, (0, 1, 0), hip[s][1]) \
            @ around(th, (1, 0, 0), hip[s][2])
        kn = (sign*42, 3, 86)
        knee_m = hip_m @ around(kn, (1, 0, 0), knee[s])
        an = (sign*46, 3, 24)
        ankle_m = knee_m @ around(an, (1, 0, 0), ankle[s])
        thumb_p = (sign*84, -6, 103)
        thumb_m = elbow_m @ around(thumb_p, (1, 0, 0), jaw[s])
        out.update({f"upper_{s}": sh_m, f"forearm_{s}": elbow_m,
                    f"thumb_{s}": thumb_m, f"thigh_{s}": hip_m,
                    f"shin_{s}": knee_m, f"foot_{s}": ankle_m})
    return out


def add_box_component(items, name, center, dims, group, owner, color):
    mesh = trimesh.creation.box(extents=dims)
    mesh.apply_translation(center)
    items.append(dict(name=name, group=group, owner=owner, color=color,
                      positions=np.asarray(mesh.vertices).round(4).ravel().tolist(),
                      normals=np.asarray(mesh.vertex_normals).round(5).ravel().tolist(),
                      indices=np.asarray(mesh.faces).ravel().tolist()))


def add_cylinder_component(items, name, a, b, radius, group, owner, color, sections=12):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b-a
    h = float(np.linalg.norm(d))
    if h < 0.1:
        return
    mesh = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], d/h))
    mesh.apply_translation((a+b)/2)
    items.append(dict(name=name, group=group, owner=owner, color=color,
                      positions=np.asarray(mesh.vertices).round(4).ravel().tolist(),
                      normals=np.asarray(mesh.vertex_normals).round(5).ravel().tolist(),
                      indices=np.asarray(mesh.faces).ravel().tolist()))


def add_ring(items, name, center, radius, tube, axis, group, owner):
    mesh = trimesh.creation.torus(major_radius=radius, minor_radius=tube,
                                  major_sections=24, minor_sections=8)
    align = trimesh.geometry.align_vectors([0, 0, 1], axis)
    mesh.apply_transform(align)
    mesh.apply_translation(center)
    items.append(dict(name=name, group=group, owner=owner, color=METAL,
                      positions=np.asarray(mesh.vertices).round(4).ravel().tolist(),
                      normals=np.asarray(mesh.vertex_normals).round(5).ravel().tolist(),
                      indices=np.asarray(mesh.faces).ravel().tolist()))


def joint_specs():
    specs = []
    for s, sign in (("L", -1), ("R", 1)):
        hip = (sign*39, 3, 155)
        sh = (sign*67, 4, 245)
        for i, axis in enumerate(((0, 0, 1), (0, 1, 0), (1, 0, 0))):
            specs.append((f"hip_{s}_{('yaw','roll','pitch')[i]}", hip, axis,
                          f"thigh_{s}", "root", 8))
            specs.append((f"shoulder_{s}_{('yaw','roll','pitch')[i]}", sh, axis,
                          f"upper_{s}", "root", 8))
        specs.extend([
            (f"knee_{s}_pitch", (sign*42, 3, 86), (1, 0, 0), f"shin_{s}", f"thigh_{s}", 8),
            (f"ankle_{s}_pitch", (sign*46, 3, 24), (1, 0, 0), f"foot_{s}", f"shin_{s}", 8),
            (f"elbow_{s}_pitch", (sign*96, 2, 194.5), (1, 0, 0), f"forearm_{s}", f"upper_{s}", 8),
            (f"paw_{s}_opening", (sign*84, -6, 103), (1, 0, 0), f"thumb_{s}", f"forearm_{s}", 8),
        ])
    specs.append(("neck_yaw", (0, 4, 276), (0, 0, 1), "head", "root", 8))
    assert len(specs) == 21
    return specs


def design_components():
    items, motors, batteries = [], [], []
    # Twelve remote shoulder and hip winches: paired across X, layered in Z.
    # Distinct front/rear looms use small capstans and twin cable paths.
    for family, y, levels in (("hip", -15, (174, 196, 218)),
                              ("shoulder", 15, (174, 196, 218))):
        for side, x in (("L", -18), ("R", 18)):
            for axis, z in zip(("yaw", "roll", "pitch"), levels):
                name = f"{family}_{side}_{axis}"
                center = (x, y, z)
                add_box_component(items, "XL330_"+name, center, (34, 26, 20), "motors", "root", MOTOR)
                motors.append(dict(name=name, center=center, dims=(34,26,20), owner="root"))
    # Remaining local winches sit in the parent link, close to the outgoing joint.
    for s, sign in (("L", -1), ("R", 1)):
        local = [
            (f"knee_{s}_pitch", (sign*40, 2, 126), (20,26,34), f"thigh_{s}"),
            (f"ankle_{s}_pitch", (sign*46, 4, 52), (20,34,26), f"shin_{s}"),
            (f"elbow_{s}_pitch", (sign*88, 3, 234), (20,26,34), f"upper_{s}"),
            (f"paw_{s}_opening", (sign*99, 4, 160), (20,26,34), f"forearm_{s}"),
        ]
        for name, center, dims, owner in local:
            add_box_component(items, "XL330_"+name, center, dims, "motors", owner, MOTOR)
            motors.append(dict(name=name, center=center, dims=dims, owner=owner))
    center = (0, 4, 307)
    add_box_component(items, "XL330_neck_yaw", center, (20,26,34), "motors", "head", MOTOR)
    motors.append(dict(name="neck_yaw", center=center, dims=(20,26,34), owner="head"))

    # Two removable 2S cells in the dorsal chest shelf. Voltage conversion and
    # protection are absent; this is a dimensional envelope only.
    for i, y in enumerate((-8, 8), 1):
        c = (0, y, 245)
        add_box_component(items, f"Tattu_450mAh_2S_{i}", c, PACK_DIMS, "battery", "root", BATTERY)
        batteries.append(dict(name=f"Tattu_450mAh_2S_{i}", center=c, dims=PACK_DIMS))

    # Conceptual load-bearing spine, girdles, links and paired output supports.
    add_box_component(items, "torso_keel", (0, 4, 212), (12, 12, 118), "structure", "root", FRAME)
    add_box_component(items, "pelvis_crossbeam", (0, 3, 155), (84, 12, 10), "structure", "root", FRAME)
    add_box_component(items, "shoulder_crossbeam", (0, 4, 245), (118, 12, 10), "structure", "root", FRAME)
    for s, sign in (("L", -1), ("R", 1)):
        add_cylinder_component(items, f"upper_arm_rail_{s}", (sign*67,4,245), (sign*96,2,194.5), 3.2, "structure", f"upper_{s}", FRAME)
        add_cylinder_component(items, f"forearm_rail_{s}", (sign*96,2,194.5), (sign*99,1,106), 3.0, "structure", f"forearm_{s}", FRAME)
        add_cylinder_component(items, f"thigh_rail_{s}", (sign*39,3,155), (sign*42,3,86), 3.5, "structure", f"thigh_{s}", FRAME)
        add_cylinder_component(items, f"shin_rail_{s}", (sign*42,3,86), (sign*46,3,24), 3.2, "structure", f"shin_{s}", FRAME)

    specs = joint_specs()
    motor_by_name = {m["name"]: m for m in motors}
    for motor in motors:
        c = np.asarray(motor["center"], dtype=float)
        add_cylinder_component(items, "4mm_capstan_"+motor["name"], c-(0,0,1.5), c+(0,0,1.5),
                               4.0, "outputs", motor["owner"], TENDON, sections=16)
    for name, center, axis, child_owner, parent_owner, radius in specs:
        add_ring(items, name+"_output_sector", center, radius, 1.15, axis, "outputs", child_owner)
        # Two opposed support envelopes per axis. Exact bearing model/retention
        # remains open; the hoops make the reaction path visible in the viewer.
        axis = np.asarray(axis, dtype=float)
        for offset in (-4.0, 4.0):
            add_ring(items, name+f"_support_{offset:+g}", np.asarray(center)+axis*offset,
                     radius+3.0, 1.2, axis, "supports", parent_owner)
        # A paired agonist/antagonist route is shown as two 1.2 mm conceptual
        # cables between the local winch and its output sector.
        if name.startswith("hip_") or name.startswith("shoulder_"):
            motor_name = name
        elif name.startswith("neck_"):
            motor_name = "neck_yaw"
        else:
            motor_name = name
        motor = motor_by_name.get(motor_name)
        if motor is None:
            continue
        src = np.asarray(motor["center"], dtype=float)
        dst = np.asarray(center, dtype=float)
        for j, sign in enumerate((-1,1)):
            off = np.array([0, 0, sign*1.8])
            bend = (src+dst)/2 + np.array([0, sign*4.0, sign*2.0])
            add_cylinder_component(items, f"{name}_tendon_{j}_a", src+off, bend+off,
                                   .6, "tendons", "root", TENDON, sections=8)
            add_cylinder_component(items, f"{name}_tendon_{j}_b", bend+off, dst+off,
                                   .6, "tendons", "root", TENDON, sections=8)
    return items, motors, batteries


def make_html(scene_path, viewer_path):
    template = (Path(__file__).with_name("viewer.html")).read_text()
    vendor = SOURCE / "vendor"
    target_vendor = viewer_path.parent / "vendor"
    target_vendor.mkdir(parents=True, exist_ok=True)
    for f in ("three.module.js", "three.core.js", "OrbitControls.js", "LICENSE.three"):
        shutil.copy2(vendor/f, target_vendor/f)
    viewer_path.write_text(template.replace("__SCENE__", scene_path.name))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell_mesh_wall2 = None
    studies = {}
    for wall in (1.0, 2.0):
        destination = OUT / f"source-shell-{wall:g}mm"
        study = Study(350, wall=wall, gap=.65).build()
        # Exact STEP/B-rep is the authority here. The upstream shell's STL
        # boundary patcher can fail on an occasional tiny open torso seam, so
        # do not let optional tessellation block the geometry screen.
        manifest = study.export(destination, mesh=False)
        studies[wall] = study
        if wall == 2.0:
            base_mesh = ROOT/"artifacts"/"sketch-ape-shell"/"exports"/"350mm"/"mesh.json"
            if not base_mesh.exists():
                raise FileNotFoundError(f"Build the source shell viewer mesh first: {base_mesh}")
            shell_mesh_wall2 = json.loads(base_mesh.read_text())

    items, motors, batteries = design_components()

    # A fixed-pose exact STEP package per source-wall option. Custom carriers,
    # frame members and drive parts are proxies and explicitly mass-unknown.
    custom = []
    for m in motors:
        custom.append(box_shape(m["center"], m["dims"]))
        custom.append(Solid.make_cylinder(4, 3,
            Plane(origin=(m["center"][0],m["center"][1],m["center"][2]-1.5))))
    for b in batteries:
        custom.append(box_shape(b["center"], b["dims"]))
    custom.extend([
        box_shape((0,4,212),(12,12,118)),
        box_shape((0,3,155),(84,12,10)),
        box_shape((0,4,245),(118,12,10)),
    ])
    for sign in (-1, 1):
        custom.extend([
            link_rod((sign*67,4,245),(sign*96,2,194.5),3.2),
            link_rod((sign*96,2,194.5),(sign*99,1,106),3.0),
            link_rod((sign*39,3,155),(sign*42,3,86),3.5),
            link_rod((sign*42,3,86),(sign*46,3,24),3.2),
            link_rod((sign*46,3,24),(sign*46,-8,10),3.2),
        ])
    for name, center, axis, child_owner, parent_owner, radius in joint_specs():
        axis_vec = np.asarray(axis, dtype=float)
        custom.append(Solid.make_cylinder(radius, 2.3,
            Plane(origin=tuple(np.asarray(center)-axis_vec*1.15), z_dir=tuple(axis_vec))))
        for offset in (-4.0, 4.0):
            custom.append(annular_support(np.asarray(center)+axis_vec*offset, axis_vec,
                                           radius+3.0, 5.0, 2.4))
    # Dynamic cable/sector and rod visualization is in the moving viewer; it is
    # not included as fabricated STEP because spool teeth and fasteners are open.
    for wall, study in studies.items():
        export_step(Compound(label=f"Tendon-loom concept / shell {wall:g} mm",
                             children=[p["shape"] for p in study.parts]+custom),
                    OUT/f"wildcard-{wall:g}mm-neutral.step")

    # Exact static motor-case/shell-wall intersections, per shell thickness.
    interference = {}
    battery_interference = {}
    for wall, study in studies.items():
        source_parts = []
        for part in study.parts:
            bb = part["shape"].bounding_box()
            source_parts.append((part["name"], part["shape"],
                                 (np.array(tuple(bb.min)), np.array(tuple(bb.max)))))
        entries = []
        for m in motors:
            case = box_shape(m["center"], m["dims"])
            bbox = (np.array(tuple(case.bounding_box().min)), np.array(tuple(case.bounding_box().max)))
            overlaps = []
            for n, shape, bb in source_parts:
                if not aabb_overlap(bbox, bb):
                    continue
                try:
                    vol = (case & shape).volume
                except Exception:
                    vol = None
                if vol is not None and vol > .01:
                    overlaps.append({"shell_part": n, "overlap_mm3": round(vol, 3)})
            entries.append({"motor": m["name"], "bbox_mm": [bbox[0].tolist(), bbox[1].tolist()],
                            "shell_intersections": overlaps,
                            "classification": "shell-intersecting" if overlaps else "no sampled static shell intersection"})
        interference[f"{wall:g}mm"] = entries
        battery_entries = []
        for b in batteries:
            pack = box_shape(b["center"], b["dims"])
            bbox = (np.array(tuple(pack.bounding_box().min)), np.array(tuple(pack.bounding_box().max)))
            overlaps = []
            for n, shape, bb in source_parts:
                if not aabb_overlap(bbox, bb):
                    continue
                try:
                    vol = (pack & shape).volume
                except Exception:
                    vol = None
                if vol is not None and vol > .01:
                    overlaps.append({"shell_part": n, "overlap_mm3": round(vol, 3)})
            battery_entries.append({"pack": b["name"], "bbox_mm": [bbox[0].tolist(), bbox[1].tolist()],
                                    "shell_intersections": overlaps,
                                    "classification": "shell-intersecting" if overlaps else "no sampled static shell intersection"})
        battery_interference[f"{wall:g}mm"] = battery_entries

    # Assign actual shell tessellation and conceptual components to moving links.
    for part in shell_mesh_wall2["parts"]:
        part["owner"] = owner_for_shell(part["name"])
        part["group"] = "shell"
        part["color"] = part.get("color", IVORY)
        items.insert(0, part)

    pose_names = ["neutral", "arms-out", "arms-forward", "arm-floor-reach", "elbow-60",
                  "elbow-115", "walk-swing", "knee-60", "knee-115", "ankle-plus-35",
                  "ankle-minus-35", "squat-reset", "crawl"]
    poses = {}
    for pose in pose_names:
        owner_mats = pose_transforms(pose)
        poses[pose] = {k: v.T.flatten().round(9).tolist() for k, v in owner_mats.items()}

    (OUT/"scene.json").write_text(json.dumps({"height_mm":350,"shell_wall_mm":2,
        "parts":items,"poses":poses,"pose_names":pose_names},separators=(",",":")))
    make_html(OUT/"scene.json", OUT/"index.html")

    shell_mass = {}
    for w, study in studies.items():
        volume = sum(p["shape"].volume for p in study.parts)
        shell_mass[f"{w:g}mm"] = {"volume_cm3":round(volume/1000,3),
                                  "mass_g_at_1.24g_cm3":round(volume/1000*PLA_G_CM3,1)}
    motor_mass = len(motors)*XL_MASS_G
    pack_mass = len(batteries)*PACK_MASS_G
    report = {
        "stage":"concept",
        "datum":"experiments/sketch_ape_shell Study(350, wall=1 or 2, gap=.65); X right, Y rear, Z up",
        "architecture":"21 XL330-M288 servos drive 21 independently supported tendon sectors. Twelve yaw/roll/pitch hip/shoulder winches occupy paired front/rear torso rails; local knee/ankle/elbow/jaw/head winches live in parent links. Twin tendons and 4 mm motor spool / 8 mm output sector imply 2:1 reduction.",
        "axes":21,
        "actuator_source":{"model":"ROBOTIS XL330-M288-T","case_mm":[20,34,26],"mass_g_each":18,
            "stall_Nm_at_5V":XL_STALL_NM,"vendor_continuous_Nm":None,
            "comparison_heuristic_Nm_at_motor":0.10,"heuristic_status":"20%-of-stall screen only, not a vendor rating",
            "ideal_2_to_1_stall_Nm":round(XL_STALL_NM*2,2),
            "ideal_2_to_1_heuristic_Nm":0.20,"price_each_USD_observed":XL_PRICE_USD,
            "price_21_USD_observed":round(XL_PRICE_USD*21,2),"production_quote":"unknown"},
        "battery_source":{"model":"Tattu FPV 450mAh 2S1P","count":2,"case_each_mm":PACK_DIMS,
            "mass_each_g_nominal":PACK_MASS_G,"nominal_voltage":7.4,"combined_capacity_mAh":900,
            "connection":"parallel proposal; requires protected matching packs and 5 V high-current regulator, both unresolved",
            "price":"unquoted"},
        "mass_lower_bound":{"appearance_shell":shell_mass,"actuators_g":motor_mass,
            "battery_g":pack_mass,"subtotal_g_with_1mm_shell":round(shell_mass["1mm"]["mass_g_at_1.24g_cm3"]+motor_mass+pack_mass,1),
            "subtotal_g_with_2mm_shell":round(shell_mass["2mm"]["mass_g_at_1.24g_cm3"]+motor_mass+pack_mass,1),
            "omitted":"tendon/fairlead/friction liners, 21 capstans, output shafts, bearings, carriers, fasteners, driver/controller, regulator, wiring, camera, print reinforcement and local covers; all require CAD or measurement"},
        "power":{"voltage":"XL330 recommended 5.0 V, max 6.0 V; proposed 2S battery requires converter",
            "simultaneous_stall_current_upper_bound_A":round(21*1.47,2),
            "simultaneous_stall_power_upper_bound_W":round(21*1.47*5,1),
            "status":"not a duty point; battery discharge rating and converter current capacity unverified"},
        "motion_map":{"motor_angle":"2 × output joint angle in ideal 4 mm spool / 8 mm sector geometry",
            "ideal_output_torque":"2 × motor torque before tendon, guide and bearing losses",
            "status":"pretension, friction, compliance, backlash and moving-pose routing are not modeled; retain independent simulated joints with per-axis map and uncertainty"},
        "exact_static_case_shell_intersections":interference,
        "exact_static_battery_shell_intersections":battery_interference,
        "dynamic_geometry":"viewer poses transform source shell meshes and drive proxies; exact dynamic B-rep clash, minimum running gap and part-outside-silhouette are not certified",
        "source_shell_mass":shell_mass,
        "view_file":"artifacts/ape-wildcard-tendon-loom/index.html",
        "poses":pose_names,
        "motor_count":len(motors),"battery_count":len(batteries),
    }
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"output":str(OUT),"motors":len(motors),"poses":pose_names,
                      "mass_lower_bound_g":report["mass_lower_bound"],
                      "shell_intersections_by_wall":{w:sum(bool(x["shell_intersections"]) for x in v)
                                                       for w,v in interference.items()}},indent=2))


if __name__ == "__main__":
    main()
