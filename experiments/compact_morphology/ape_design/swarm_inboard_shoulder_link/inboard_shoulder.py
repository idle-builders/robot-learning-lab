"""Exact 500 mm inboard-yaw / offset-link shoulder rejection probe.

All dimensions are millimetres.  The source shell is rebuilt, not imported as a
mesh.  The local roof is fixed to the torso; the lower arm and the compact
link turn under it.  The second trial changes only the proximal arm neck cut.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b
from build123d import SkipClean
import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/inboard-shoulder-link-500"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study  # noqa: E402

SCALE = 500 / 350
PIVOT = {"R": (64., 6., 335.), "L": (-64., 6., 335.)}
ROLL = {"R": (102., 2., 337.), "L": (-102., 2., 337.)}
PITCH = {"R": (112., 2., 318.), "L": (-112., 2., 318.)}
MOTOR = {"R": (30., -22., 335.), "L": (-30., -22., 335.)}
PACK = (0., 14., 285.7)
PACK_SIZE = (60., 30., 30.5)
SERVICE = (0., -20., 365.)
SERVICE_SIZE = (25., 20., 14.)
TOL = 0.5


def box(center, size):
    return b.Pos(*center) * b.Box(*size)


def shape(value):
    return b.Compound(children=list(value)) if isinstance(value, (list, b.ShapeList)) else value


def cylinder(axis, radius, length, center):
    rotation = {"x": b.Rot(0, 90, 0), "y": b.Rot(90, 0, 0), "z": b.Rot(0, 0, 0)}[axis]
    return b.Pos(*center) * rotation * b.Cylinder(radius, length)


def ring(axis, center, outer_radius=6.5, inner_radius=3., width=3.5):
    return cylinder(axis, outer_radius, width, center) - cylinder(axis, inner_radius, width + .2, center)


def rod(a, c, radius):
    a, c = np.asarray(a, float), np.asarray(c, float)
    vector = c - a
    return b.Solid.make_cylinder(radius, float(np.linalg.norm(vector)),
                                 b.Plane(origin=tuple(a), z_dir=tuple(vector / np.linalg.norm(vector))))


def volume(shape):
    return round(sum(s.volume for s in shape.solids()), 5) if shape else 0.


def common(a, c):
    """Exact BRep intersection in both OCC operand orders."""
    aa, cc = a.bounding_box(), c.bounding_box()
    if any(max((aa.min.X, aa.min.Y, aa.min.Z)[i], (cc.min.X, cc.min.Y, cc.min.Z)[i]) >
           min((aa.max.X, aa.max.Y, aa.max.Z)[i], (cc.max.X, cc.max.Y, cc.max.Z)[i]) for i in range(3)):
        return [0., 0.]
    with SkipClean():
        ac = a.intersect(c)
        ca = c.intersect(a)
    return [volume(ac), volume(ca)]


def gap(a, c):
    try:
        return round(a.distance(c), 5)
    except Exception:
        return None


def checked(name, a, c):
    hit = common(a, c)
    return {"pair": name, "intersection_mm3_a_b": hit[0], "intersection_mm3_b_a": hit[1],
            "gap_mm": gap(a, c), "pass_0p5_mm": max(hit) < .00001 and (gap(a, c) or 0) >= TOL}


def rotate(shape, axis, degrees, point):
    p = np.asarray(point, float)
    rotation = {"x": b.Rot(degrees, 0, 0), "y": b.Rot(0, degrees, 0),
                "z": b.Rot(0, 0, degrees)}[axis]
    return b.Pos(*p) * rotation * b.Pos(*(-p)) * shape


def pose_shape(shape, side, pose):
    yaw, roll, pitch = pose[side]
    shape = rotate(shape, "y", pitch, PITCH[side])
    shape = rotate(shape, "x", roll, ROLL[side])
    return rotate(shape, "z", yaw, PIVOT[side])


def yaw_shape(shape, side, pose):
    return rotate(shape, "z", pose[side][0], PIVOT[side])


def slab(side, upper, local_gap=0.):
    """A sloping split: its high medial end is hidden in the axilla."""
    theta = math.degrees(math.atan(.40)) * (1 if side == "R" else -1)
    x = 100. if side == "R" else -100.
    z = 344.
    offset = 50. + local_gap
    return b.Pos(x, 0, z) * b.Rot(0, theta, 0) * b.Pos(0, 0, offset if upper else -offset) * b.Box(300, 300, 100)


def local_skin(parts, neck_cut):
    fixed = {"torso_front": parts["torso_front"], "torso_back": parts["torso_back"]}
    moving = {}
    for side in ("R", "L"):
        roof_half = slab(side, True)
        arm_half = slab(side, False, .8)  # 1.6 mm nominal normal seam
        for n in (f"{side}_upper_arm_front", f"{side}_upper_arm_back"):
            fixed[f"{side}_fixed_shoulder_roof_{n.rsplit('_', 1)[-1]}"] = shape(parts[n].intersect(roof_half))
            moving[n] = shape(parts[n].intersect(arm_half))
            if neck_cut:
                # One bounded topology correction: narrow the entire proximal
                # moving root rather than reshaping or widening the roof lip.
                x0 = 101. if side == "R" else -101.
                outer = box((x0 + (150 if side == "R" else -150), 0, 345), (300, 250, 60))
                lower = box((0, 0, 249), (500, 250, 140))
                moving[n] = shape(moving[n].intersect(outer.fuse(lower)))
        # The old exposed trim is deleted. The roof is source upper-arm surface,
        # fixed to the torso, while the root below it is the moving skin.
    return fixed, moving


def mechanism():
    fixed, yaw, roll, pitch = {}, {}, {}, {}
    for side in ("R", "L"):
        sign = 1 if side == "R" else -1
        p, m = PIVOT[side], MOTOR[side]
        fixed[f"{side}_RSF5B50_yaw_case"] = cylinder("z", 12.5, 35.8, (m[0], m[1], m[2] - 17.9))
        fixed[f"{side}_PT1_connector_reserve"] = box((m[0] + sign * 16.5, m[1], m[2] - 20), (8, 8, 8))
        fixed[f"{side}_yaw_lower_W618"] = ring("z", (p[0], p[1], p[2] - 13))
        fixed[f"{side}_yaw_upper_W618"] = ring("z", p)
        fixed[f"{side}_yaw_bored_support"] = box((sign * 57, 6, 326), (28, 11, 28)).cut(cylinder("z", 6.65, 29, (p[0], p[1], 326)))
        fixed[f"{side}_yaw_drive_belt_keepout"] = box((sign * 47, -8, 340.5), (39, 34, 4))
        yaw[f"{side}_yaw_supported_shaft"] = cylinder("z", 3, 34, (p[0], p[1], 336))
        yaw[f"{side}_offset_swing_link"] = rod((p[0], p[1], 345), (sign * 99, 2, 337), 3.5)
        yaw[f"{side}_roll_inner_W618"] = ring("x", (sign * 95, 2, 337))
        yaw[f"{side}_roll_outer_W618"] = ring("x", (sign * 109, 2, 337))
        yaw[f"{side}_XL330_roll_case"] = box((sign * 113, 2, 337), (29, 20, 34))
        roll[f"{side}_roll_supported_shaft"] = cylinder("x", 3, 31, ROLL[side])
        roll[f"{side}_pitch_inner_W618"] = ring("y", (sign * 112, -13, 318))
        roll[f"{side}_pitch_outer_W618"] = ring("y", (sign * 112, 17, 318))
        roll[f"{side}_XL330_pitch_case"] = box((sign * 119, 2, 303), (20, 29, 34))
        pitch[f"{side}_pitch_supported_shaft"] = cylinder("y", 3, 38, PITCH[side])
    fixed["shared_rear_shifted_4S850_pack"] = box(PACK, PACK_SIZE)
    fixed["CAN_5V_service_reserve"] = box(SERVICE, SERVICE_SIZE)
    return fixed, yaw, roll, pitch


POSES = {
    "arm_down": {"R": (0., 0., 0.), "L": (0., 0., 0.)},
    "yaw_forward_30": {"R": (-30., 0., 0.), "L": (30., 0., 0.)},
    "yaw_rear_30": {"R": (30., 0., 0.), "L": (-30., 0., 0.)},
    "forward_reach": {"R": (0., -45., 0.), "L": (0., -45., 0.)},
    # An arm-root proxy for floor grasp while standing; a squat lowering the
    # torso is required for hand-floor contact and is outside this coupon.
    "floor_reach": {"R": (-15., -15., 0.), "L": (15., -15., 0.)},
}
ELBOW = {"arm_down": 0., "yaw_forward_30": 0., "yaw_rear_30": 0.,
         "forward_reach": -35., "floor_reach": 0.}
ELBOW_PIVOT = {"R": (96*SCALE, 2*SCALE, 194.5*SCALE),
               "L": (-96*SCALE, 2*SCALE, 194.5*SCALE)}


def lower_arm_name(name):
    return name.startswith(("R_", "L_")) and any(token in name for token in
        ("_forearm_", "_wrist_cuff", "_palm_", "_finger_", "_thumb_cap"))


def posed_lower_arm(sh, name, pose, pose_name):
    side = name[0]
    sh = rotate(sh, "y", ELBOW[pose_name] * (1 if side == "R" else -1), ELBOW_PIVOT[side])
    return pose_shape(sh, side, pose)


def solids_for_pose(parts, fixed_skin, moving_skin, hw, pose, pose_name, candidate):
    scene = {}
    for n, sh in parts.items():
        if n.startswith(("torso_", "R_upper_arm_", "L_upper_arm_", "R_shoulder_trim", "L_shoulder_trim")):
            continue
        scene[n] = (posed_lower_arm(sh, n, pose, pose_name) if lower_arm_name(n)
                    else pose_shape(sh, n[0], pose) if n.startswith(("R_elbow_cuff", "L_elbow_cuff"))
                    else sh, "skin")
    if candidate:
        scene.update((n, (sh, "skin")) for n, sh in fixed_skin.items())
        scene.update((n, (pose_shape(sh, n[0], pose), "skin")) for n, sh in moving_skin.items())
        for group, objects in zip(("fixed", "yaw", "roll", "pitch"), hw):
            for n, sh in objects.items():
                if group == "yaw": sh = yaw_shape(sh, n[0], pose)
                elif group == "roll":
                    sh = rotate(sh, "x", pose[n[0]][1], ROLL[n[0]])
                    sh = yaw_shape(sh, n[0], pose)
                elif group == "pitch": sh = pose_shape(sh, n[0], pose)
                scene[n] = (sh, "hardware")
    else:
        scene.update((n, (sh, "skin")) for n, sh in parts.items() if n.startswith("torso_"))
        for side in ("R", "L"):
            for suffix in ("upper_arm_front", "upper_arm_back", "shoulder_trim"):
                n = f"{side}_{suffix}"
                scene[n] = (pose_shape(parts[n], side, pose), "skin")
    return scene


def mesh(sh, color):
    vertices, faces = sh.tessellate(.7, .5)
    m = trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
                        faces=np.asarray(faces), process=False)
    m.visual.vertex_colors = color
    return m


def export_glb(scene, filename):
    out = trimesh.Scene()
    for n, (sh, kind) in scene.items():
        if kind != "skin": continue
        if not sh.solids(): continue
        out.add_geometry(mesh(sh, (226, 218, 202, 255)), geom_name=n, node_name=n)
    out.export(filename)


VIEWS = {"front": (0., -1., .16), "side": (1., 0., .1),
         "rear": (0., 1., .16), "three-quarter": (1., -1., .32)}


def render(scene, view, size=(300, 390), crop=False):
    direction = np.array(VIEWS[view], float); direction /= np.linalg.norm(direction)
    up = np.array((0., 0., 1.)); right = np.cross(up, direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    center = np.array((0., 0., 340. if crop else 250.))
    scale = min(size) * .81 / (230 if crop else 510)
    image = Image.new("RGB", size, (237, 236, 231)); drawer = ImageDraw.Draw(image)
    polys = []
    light = np.array((.4, -.5, .75)); light /= np.linalg.norm(light)
    for n, (sh, kind) in scene.items():
        if kind != "skin" or not sh.solids(): continue
        m = mesh(sh, (226, 218, 202, 255))
        q = np.stack(((m.vertices - center) @ right, (m.vertices - center) @ up), axis=1)
        q = q * scale + np.array((size[0]/2, size[1]/2)); q[:, 1] = size[1] - q[:, 1]
        depth = (m.vertices-center) @ direction
        for i, face in enumerate(m.faces):
            shade = .79 + .18 * abs(float(m.face_normals[i] @ light))
            color = tuple(np.clip(np.array((226, 218, 202)) * shade, 0, 255).astype(int))
            polys.append((float(np.mean(depth[face])), tuple(map(tuple, q[face])), color))
    for _, polygon, color in sorted(polys, key=lambda row: row[0]): drawer.polygon(polygon, fill=color)
    return image


def contact_sheet(source, candidate, filename, poses=("arm_down", "yaw_forward_30", "yaw_rear_30"), crop=False):
    width, height = 300, 390
    sheet = Image.new("RGB", (width * 8, height * len(poses)), "white")
    for i, pose in enumerate(poses):
        for j, view in enumerate(VIEWS):
            for k, scenes in enumerate((source, candidate)):
                tile = render(scenes[pose], view, crop=crop)
                ImageDraw.Draw(tile).text((9, 9), f"{pose} · {view} · {'source' if k == 0 else 'candidate'}", fill=(34, 39, 37))
                sheet.paste(tile, ((j*2+k)*width, i*height))
    sheet.save(filename, optimize=True)


def evaluate(fixed_skin, moving_skin, hw, pose):
    torso = b.Compound(children=[fixed_skin[n] for n in ("torso_front", "torso_back")])
    roof = {s: b.Compound(children=[fixed_skin[f"{s}_fixed_shoulder_roof_front"],
                                          fixed_skin[f"{s}_fixed_shoulder_roof_back"]]) for s in ("R", "L")}
    fixed_hw, yaw_hw, roll_hw, pitch_hw = hw
    checks = []
    for side in ("R", "L"):
        arm = b.Compound(children=[pose_shape(moving_skin[f"{side}_upper_arm_front"], side, pose),
                                   pose_shape(moving_skin[f"{side}_upper_arm_back"], side, pose)])
        checks += [checked(f"{side} moving arm / torso", arm, torso),
                   checked(f"{side} moving arm / fixed shoulder roof", arm, roof[side])]
        link = yaw_shape(yaw_hw[f"{side}_offset_swing_link"], side, pose)
        checks += [checked(f"{side} swing link / torso", link, torso),
                   checked(f"{side} swing link / fixed shoulder roof", link, roof[side])]
        for key in (f"{side}_XL330_roll_case",):
            case = yaw_shape(yaw_hw[key], side, pose)
            checks.append(checked(f"{key} / fixed shoulder roof", case, roof[side]))
        case = rotate(roll_hw[f"{side}_XL330_pitch_case"], "x", pose[side][1], ROLL[side])
        case = yaw_shape(case, side, pose)
        checks.append(checked(f"{side} XL330 pitch case / moving arm skin", case, arm))
    pack = fixed_hw["shared_rear_shifted_4S850_pack"]
    service = fixed_hw["CAN_5V_service_reserve"]
    for side in ("R", "L"):
        for key in (f"{side}_RSF5B50_yaw_case", f"{side}_yaw_bored_support"):
            checks.append(checked(f"{key} / shared pack", fixed_hw[key], pack))
            checks.append(checked(f"{key} / service reserve", fixed_hw[key], service))
    return checks


def main():
    study = Study(500, 2., .65).build()
    parts = {p["name"]: p["shape"] for p in study.parts}
    hw = mechanism()
    source_scenes = {name: solids_for_pose(parts, {}, {}, hw, pose, name, False) for name, pose in POSES.items()}
    previous = HERE / "report.json"
    reach_only = "--reach-only" in sys.argv
    render_only = "--render-only" in sys.argv or reach_only
    results = json.loads(previous.read_text())["trials"] if render_only and previous.exists() else {}
    for trial, neck_cut in (("trial_01", False), ("trial_02_neck", True)):
        fixed, moving = local_skin(parts, neck_cut)
        candidate = {name: solids_for_pose(parts, fixed, moving, hw, pose, name, True) for name, pose in POSES.items()}
        if not render_only:
            poses = {name: evaluate(fixed, moving, hw, pose) for name, pose in POSES.items()}
            failures = [{"pose": name, **row} for name, rows in poses.items() for row in rows if not row["pass_0p5_mm"]]
            results[trial] = {"neck_cut": neck_cut, "pose_checks": poses, "first_failed_check": failures[0] if failures else None,
                              "all_required_checks_pass": not failures}
        if not reach_only:
            for name, scene in candidate.items(): export_glb(scene, OUT / f"{trial}-{name}.glb")
            contact_sheet(source_scenes, candidate, HERE / f"{trial}-matched-opaque.png")
            contact_sheet(source_scenes, candidate, HERE / f"{trial}-shoulder-crop.png", crop=True)
        if trial == "trial_02_neck":
            contact_sheet(source_scenes, candidate, HERE / "trial_02_neck-reach-opaque.png",
                          poses=("forward_reach", "floor_reach"), crop=True)
    static_pack = hw[0]["shared_rear_shifted_4S850_pack"]
    static_service = hw[0]["CAN_5V_service_reserve"]
    torso = b.Compound(children=[parts["torso_front"], parts["torso_back"]])
    rear_path = box((PACK[0], PACK[1] + 50, PACK[2]),
                    (PACK_SIZE[0], PACK_SIZE[1] + 100, PACK_SIZE[2]))
    static_checks = [checked("shared pack / torso", static_pack, torso),
                     checked("service reserve / torso", static_service, torso),
                     checked("shared pack / service reserve", static_pack, static_service),
                     checked("100 mm rear pack withdrawal / torso", rear_path, torso)]
    report = {"source": "Study(500, wall=2, gap=.65) exact BRep from experiments/sketch_ape_shell/build.py",
              "method": "Both exact Boolean operand orders plus BRep distance; skin and source/candidate opaque views matched; no mesh collision inference",
              "topology": "yaw pivot x=±64 z=335, below inboard of source x=±80 z=347; fixed source-surface shoulder roof; compact offset swing link; narrowed second-trial proximal neck; independent nominal yaw, roll and pitch shafts/bearings",
              "hardware": {"yaw": "RSF-5B-50-IDT 25 mm diameter x 35.8 mm, 77 g each; 0.29 Nm allowable continuous at 50:1",
                           "roll_pitch": "ROBOTIS XL330-M288-T conservative 20 x 34 x 29 mm cases, 18 g each; working torque unpublished",
                           "pack": {"center_mm": PACK, "size_mm": PACK_SIZE, "single_shared": True},
                           "service": {"center_mm": SERVICE, "size_mm": SERVICE_SIZE}},
              "pose_angles_deg": POSES, "elbow_angles_deg_for_visuals": ELBOW,
              "static_pack_and_service_checks": static_checks, "clearance_gate_mm": TOL,
              "limitations": ["Support, belt and cable paths are envelope hypotheses; no stiffness or thermal result.",
              "The split source arm lacks detailed flanges or fasteners; this is a geometric falsification coupon.",
                              "Floor-reach is an arm-root proxy with upright torso; hand-floor contact needs a squat model. Full arm-chain clearance is unproved because the shoulder-root gate fails."],
              "trials": results}
    (HERE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"all_pass": v["all_required_checks_pass"], "first": v["first_failed_check"]}
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__": main()
