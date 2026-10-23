"""Bounded supported-shoulder falsification coupon on the committed 500 mm Cove shell.

Millimetres. X right, Y rear, Z up. The source is rebuilt as a BRep; the
rendered skin is never used as a collision operand. This is one topology trial,
not a selected mechanism or an assertion of continuous torque capacity.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import build123d as b
from build123d import SkipClean
import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/ape_design/swarm_shell_v2"))
from build import VARIANTS, build, ellipsoid, skin  # noqa: E402

TOL = .5
PACK_CENTER = (0., 14., 285.7)
PACK_SIZE = (60., 30., 30.5)
SERVICE_CENTER = (0., -20., 365.)
SERVICE_SIZE = (25., 20., 14.)
YAW = {"R": (78., 1., 335.), "L": (-78., 1., 335.)}
ROLL = {"R": (108., 1., 326.), "L": (-108., 1., 326.)}
PITCH = {"R": (121., 1., 312.), "L": (-121., 1., 312.)}
POSES = {
    "arm_down": {"R": (0., 0., 0.), "L": (0., 0., 0.)},
    "yaw_forward_30": {"R": (-30., 0., 0.), "L": (30., 0., 0.)},
    "yaw_rear_30": {"R": (30., 0., 0.), "L": (-30., 0., 0.)},
    "arm_out_35": {"R": (0., 0., 35.), "L": (0., 0., -35.)},
    "forward_reach": {"R": (0., -45., 0.), "L": (0., -45., 0.)},
    # Upright torso, arm-root configuration only. No hand/floor constraint.
    "floor_reach_proxy": {"R": (-15., -55., 10.), "L": (15., -55., -10.)},
}


def box(center, size):
    return b.Pos(*center) * b.Box(*size)


def cylinder(axis, radius, length, center):
    turn = {"x": b.Rot(0, 90, 0), "y": b.Rot(90, 0, 0), "z": b.Rot(0, 0, 0)}[axis]
    return b.Pos(*center) * turn * b.Cylinder(radius, length)


def ring(axis, center, outer=6.5, inner=3., width=3.5):
    return cylinder(axis, outer, width, center) - cylinder(axis, inner, width+.2, center)


def rod(start, end, radius):
    a, z = np.asarray(start, float), np.asarray(end, float)
    d = z-a
    return b.Solid.make_cylinder(radius, float(np.linalg.norm(d)),
                                 b.Plane(origin=tuple(a), z_dir=tuple(d/np.linalg.norm(d))))


def volume(sh):
    return round(sum(s.volume for s in sh.solids()), 5) if sh else 0.


def checked(name, a, z):
    if os.getenv("COVE_TRACE"):
        print(name, flush=True)
    aa, zz = a.bounding_box(), z.bounding_box()
    separated = any(max((aa.min.X, aa.min.Y, aa.min.Z)[i], (zz.min.X, zz.min.Y, zz.min.Z)[i]) >
                    min((aa.max.X, aa.max.Y, aa.max.Z)[i], (zz.max.X, zz.max.Y, zz.max.Z)[i])
                    for i in range(3))
    if separated:
        hits = (0., 0.)
    else:
        with SkipClean():
            hits = (volume(a.intersect(z)), volume(z.intersect(a)))
    try:
        dist = round(a.distance(z), 5)
    except Exception:
        dist = None
    return {"pair": name, "intersection_mm3_a_b": hits[0], "intersection_mm3_b_a": hits[1],
            "gap_mm": dist, "pass_0p5_mm": hits == (0., 0.) and dist is not None and dist >= TOL}


def rotate(sh, axis, degrees, origin):
    p = np.asarray(origin, float)
    turn = {"x": b.Rot(degrees, 0, 0), "y": b.Rot(0, degrees, 0),
            "z": b.Rot(0, 0, degrees)}[axis]
    return b.Pos(*p) * turn * b.Pos(*(-p)) * sh


def pose(sh, side, angles, level="pitch"):
    yaw, roll, pitch = angles[side]
    if level == "pitch":
        sh = rotate(sh, "y", pitch, PITCH[side])
    if level in ("pitch", "roll"):
        sh = rotate(sh, "x", roll, ROLL[side])
    return rotate(sh, "z", yaw, YAW[side])


def geometry(source):
    """One lateral bay, two bearing stations per axis, and an internal swing link."""
    torso = source["torso"]
    skin_parts = {}
    hardware = {}
    for side in ("R", "L"):
        sign = 1 if side == "R" else -1
        # A local bay under the source roof is the deliberate clearance route.
        # The loft above it stays the single fixed sloping crown.
        bay = ellipsoid((sign*91, 1, 329), (23, 23, 18))
        torso = torso - bay
        root_outer = ellipsoid((sign*119, 2, 341), (17, 21, 16))
        root_inner = ellipsoid((sign*119, 2, 341), (15, 19, 14))
        root = root_outer-root_inner
        # Preserve Cove's arm outside the short, rounded moving upper root.
        skin_parts[f"{side}_upper_arm"] = source[f"{side}_upper_arm"].fuse(root)
        mx = sign*53
        hardware[f"{side}_RSF5B50_yaw_case"] = ("fixed", cylinder("z", 12.5, 35.8, (mx, -19, 326)))
        hardware[f"{side}_PT1_connector_reserve"] = ("fixed", box((mx+sign*16.5, -19, 311), (8, 8, 8)))
        hardware[f"{side}_yaw_inner_bearing"] = ("fixed", ring("z", (sign*78, 1, 324)))
        hardware[f"{side}_yaw_outer_bearing"] = ("fixed", ring("z", (sign*78, 1, 344)))
        hardware[f"{side}_yaw_support_bridge"] = ("fixed", box((sign*76, 1, 334), (16, 16, 27))-
                                                    cylinder("z", 6.7, 28, (sign*78, 1, 334)))
        hardware[f"{side}_yaw_belt_path"] = ("fixed", box((sign*65.5, -9, 341), (29, 28, 3)))
        hardware[f"{side}_yaw_shaft"] = ("yaw", cylinder("z", 3, 30, (sign*78, 1, 334)))
        hardware[f"{side}_swing_link"] = ("yaw", rod((sign*78, 1, 331), (sign*108, 1, 326), 3.5))
        hardware[f"{side}_roll_inboard_bearing"] = ("yaw", ring("x", (sign*99, 1, 326)))
        hardware[f"{side}_roll_outboard_bearing"] = ("yaw", ring("x", (sign*117, 1, 326)))
        hardware[f"{side}_XL330_roll_case"] = ("yaw", box((sign*114, 1, 326), (29, 20, 34)))
        hardware[f"{side}_roll_shaft"] = ("roll", cylinder("x", 3, 29, ROLL[side]))
        hardware[f"{side}_pitch_front_bearing"] = ("roll", ring("y", (sign*121, -15, 312)))
        hardware[f"{side}_pitch_rear_bearing"] = ("roll", ring("y", (sign*121, 17, 312)))
        hardware[f"{side}_XL330_pitch_case"] = ("roll", box((sign*121, 1, 301), (20, 29, 34)))
        hardware[f"{side}_pitch_shaft"] = ("pitch", cylinder("y", 3, 38, PITCH[side]))
    hardware["single_shared_4S850_pack"] = ("fixed", box(PACK_CENTER, PACK_SIZE))
    hardware["CAN_5V_service_reserve"] = ("fixed", box(SERVICE_CENTER, SERVICE_SIZE))
    return torso, skin_parts, hardware


def scene(source, torso, skin_parts, hardware, angles, candidate):
    out = {}
    for name, sh in source.items():
        if name == "torso" or name in ("R_upper_arm", "L_upper_arm"):
            continue
        # Cove's forearm/paw are carried rigidly for the appearance proxy.
        if name.startswith(("R_forearm", "R_paw", "L_forearm", "L_paw")):
            sh = pose(sh, name[0], angles)
        out[name] = (sh, "skin")
    out["torso"] = (torso if candidate else source["torso"], "skin")
    for side in ("R", "L"):
        sh = skin_parts[f"{side}_upper_arm"] if candidate else source[f"{side}_upper_arm"]
        out[f"{side}_upper_arm"] = (pose(sh, side, angles), "skin")
    if candidate:
        for name, (level, sh) in hardware.items():
            out[name] = (pose(sh, name[0], angles, level) if level != "fixed" else sh, "hardware")
    return out


def evaluate(source, torso, skin_parts, hardware, angles):
    checks = []
    for side in ("R", "L"):
        arm = pose(skin_parts[f"{side}_upper_arm"], side, angles)
        link = pose(hardware[f"{side}_swing_link"][1], side, angles, "yaw")
        roll_case = pose(hardware[f"{side}_XL330_roll_case"][1], side, angles, "yaw")
        pitch_case = pose(hardware[f"{side}_XL330_pitch_case"][1], side, angles, "roll")
        checks += [checked(f"{side} moving rounded arm / fixed Cove torso roof", arm, torso),
                   checked(f"{side} swing link / fixed Cove torso", link, torso),
                   checked(f"{side} roll case / fixed Cove torso", roll_case, torso),
                   checked(f"{side} pitch case / moving arm skin", pitch_case, arm)]
        for key in ("roll_inboard_bearing", "roll_outboard_bearing",
                    "pitch_front_bearing", "pitch_rear_bearing"):
            level, sh = hardware[f"{side}_{key}"]
            sh = pose(sh, side, angles, level)
            checks.append(checked(f"{side} {key} / moving arm skin", sh, arm))
    return checks


def fixed_checks(torso, hardware):
    checks = []
    for side in ("R", "L"):
        # Distinct serial axes do not excuse intersecting full case envelopes.
        checks.append(checked(f"{side} XL330 roll case / XL330 pitch case",
                              hardware[f"{side}_XL330_roll_case"][1],
                              hardware[f"{side}_XL330_pitch_case"][1]))
        for key in ("RSF5B50_yaw_case", "yaw_support_bridge", "yaw_belt_path",
                    "yaw_inner_bearing", "yaw_outer_bearing"):
            checks.append(checked(f"{side} {key} / fixed Cove torso",
                                  hardware[f"{side}_{key}"][1], torso))
        motor = hardware[f"{side}_RSF5B50_yaw_case"][1]
        for key in ("single_shared_4S850_pack", "CAN_5V_service_reserve"):
            checks.append(checked(f"{side} yaw case / {key}", motor, hardware[key][1]))
    return checks


VIEWS = {"front": (0., -1., .16), "side": (1., 0., .1),
         "rear": (0., 1., .16), "three-quarter": (1., -1., .32)}


def mesh(sh):
    vertices, faces = sh.tessellate(.7, .5)
    return trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
                           faces=np.asarray(faces), process=False)


def render(scene_parts, view, crop=False, size=(300, 390)):
    direction = np.asarray(VIEWS[view], float)
    direction /= np.linalg.norm(direction)
    up = np.array((0., 0., 1.))
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    center = np.array((0., 0., 340. if crop else 250.))
    scale = min(size) * .81 / (230 if crop else 510)
    image = Image.new("RGB", size, (237, 236, 231))
    drawer = ImageDraw.Draw(image)
    polys = []
    light = np.array((.4, -.5, .75))
    light /= np.linalg.norm(light)
    for name, (sh, kind) in scene_parts.items():
        if kind != "skin" or not sh.solids():
            continue
        m = mesh(sh)
        q = np.stack(((m.vertices-center) @ right, (m.vertices-center) @ up), axis=1)
        q = q*scale + np.array((size[0]/2, size[1]/2))
        q[:, 1] = size[1]-q[:, 1]
        depth = (m.vertices-center) @ direction
        for face, normal in zip(m.faces, m.face_normals):
            shade = .79 + .18*abs(float(normal @ light))
            color = tuple(np.clip(np.array((226, 218, 202))*shade, 0, 255).astype(int))
            polys.append((float(np.mean(depth[face])), tuple(map(tuple, q[face])), color))
    for _, poly, color in sorted(polys, key=lambda row: row[0]):
        drawer.polygon(poly, fill=color)
    return image


def contact_sheet(source, candidate, filename, poses, crop=False):
    width, height = 300, 390
    sheet = Image.new("RGB", (width*8, height*len(poses)), "white")
    for i, pose_name in enumerate(poses):
        for j, view in enumerate(VIEWS):
            for k, scenes in enumerate((source, candidate)):
                tile = render(scenes[pose_name], view, crop=crop)
                ImageDraw.Draw(tile).text((9, 9),
                    f"{pose_name} · {view} · {'source' if k == 0 else 'candidate'}",
                    fill=(34, 39, 37))
                sheet.paste(tile, ((j*2+k)*width, i*height))
    sheet.save(filename, optimize=True)


def export_glb(scene_parts, filename):
    out = trimesh.Scene()
    for name, (sh, kind) in scene_parts.items():
        if kind != "skin" or not sh.solids():
            continue
        part = mesh(sh)
        part.visual.vertex_colors = (226, 218, 202, 255)
        out.add_geometry(part, geom_name=name, node_name=name)
    out.export(filename)


def main():
    parts, _ = build(VARIANTS[0])
    source = {p.name: p.shape for p in parts}
    torso, skin_parts, hardware = geometry(source)
    if "--render-only" in sys.argv:
        make_views(source, torso, skin_parts, hardware)
        return
    static_mechanism = fixed_checks(torso, hardware)
    checks = {name: evaluate(source, torso, skin_parts, hardware, angles)
              for name, angles in POSES.items()}
    pack = hardware["single_shared_4S850_pack"][1]
    service = hardware["CAN_5V_service_reserve"][1]
    static = static_mechanism + [checked("shared pack / fixed Cove torso", pack, torso),
              checked("service reserve / fixed Cove torso", service, torso),
              checked("shared pack / service reserve", pack, service)]
    withdrawal = box((0, PACK_CENTER[1]+50, PACK_CENTER[2]),
                     (PACK_SIZE[0], PACK_SIZE[1]+100, PACK_SIZE[2]))
    static.append(checked("100 mm rear pack withdrawal / fixed Cove torso", withdrawal, torso))
    source_reference = [checked("uncut Cove right arm / uncut Cove torso",
                                source["R_upper_arm"], source["torso"]),
                        checked("uncut Cove right arm / trial bay torso",
                                source["R_upper_arm"], torso)]
    static_failures = [{"phase": "static", **row} for row in static if not row["pass_0p5_mm"]]
    pose_failures = [{"phase": "pose", "pose": name, **row} for name, rows in checks.items()
                     for row in rows if not row["pass_0p5_mm"]]
    report = {
        "source": "7e14993, swarm_shell_v2.build(VARIANTS[0]), 500 mm Cove 2 mm BRep",
        "topology": "Torso-side bay under Cove's fixed loft roof; inboard RSF yaw belt drive, opposed bearings, compact offset link, supported XL330 roll and pitch trunnions, rounded moving arm-root shell",
        "dimensions_mm": {"yaw": YAW, "roll": ROLL, "pitch": PITCH,
                          "RSF_case": "diameter 25 x 35.8; source-sized RSF-5B-50-IDT",
                          "XL330_case": "20 x 34 x 29 conservative case, ROBOTIS XL330-M288-T",
                          "bearing_envelope": "W618-like OD13 ID6 width3.5, two stations each axis",
                          "pack_center": PACK_CENTER, "pack_size": PACK_SIZE,
                          "service_center": SERVICE_CENTER, "service_size": SERVICE_SIZE},
        "method": "Exact 2 mm-wall BRep intersections in both operand orders and BRep minimum distance; 0.5 mm clearance gate; opaque renders are appearance only",
        "clearance_gate_mm": TOL, "pose_angles_deg": POSES, "pose_checks": checks,
        "source_reference_checks": source_reference,
        "static_pack_service_checks": static,
        "first_failed_check": (static_failures+pose_failures)[0] if static_failures or pose_failures else None,
        "first_failed_pose_check": pose_failures[0] if pose_failures else None,
        "all_named_checks_pass": not static_failures and not pose_failures,
        "limitations": ["A bearing envelope and path solid locate support and transfer, but fasteners, belt teeth, shaft retention, wiring and thermal duty are not designed.",
                        "Floor reach is an upright-torso arm configuration; no grounded hand-floor contact is established.",
                        "Whole-arm, bilateral hardware-to-hardware, wall stiffness and manufacturing tolerance are not qualified."],
    }
    (HERE/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    if "--measure-only" not in sys.argv:
        make_views(source, torso, skin_parts, hardware)
    print(json.dumps({"first_failed_check": report["first_failed_check"],
                      "all_named_checks_pass": report["all_named_checks_pass"]}, indent=2))


def make_views(source, torso, skin_parts, hardware):
    source_scenes = {n: scene(source, torso, skin_parts, hardware, p, False) for n, p in POSES.items()}
    candidate = {n: scene(source, torso, skin_parts, hardware, p, True) for n, p in POSES.items()}
    contact_sheet(source_scenes, candidate, HERE/"matched-opaque.png",
                  poses=("arm_down", "yaw_forward_30", "yaw_rear_30"))
    contact_sheet(source_scenes, candidate, HERE/"shoulder-crop-opaque.png",
                  poses=("arm_down", "yaw_forward_30", "yaw_rear_30"), crop=True)
    contact_sheet(source_scenes, candidate, HERE/"reach-opaque.png",
                  poses=("arm_out_35", "forward_reach", "floor_reach_proxy"), crop=True)
    export_glb(candidate["arm_down"], HERE/"candidate-skin.glb")


if __name__ == "__main__":
    main()
