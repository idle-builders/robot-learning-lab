"""Aesthetic-first, source-sized Tumble mechanical assembly.

Run from the repository root:
  .venv/bin/python experiments/compact_morphology/ape_design/tumble_cad.py --preview
  .venv/bin/python experiments/compact_morphology/ape_design/tumble_cad.py

The printed shells are design candidates. Motor mounting holes, horns, cable
exits, printed strength and full swept clearance remain explicit gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import build123d as bd
import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "ape_cad"))
import build as envelope  # noqa: E402
from screen_tumble_two_axis import build_candidate, four_point_pose, mapped_qpos  # noqa: E402

OUT = ROOT / "artifacts/ape-tumble-cad-20261001"
SOURCE = HERE.parent / "ape_cad/ape_spaced_v7_candidate.xml"
MODEL = OUT / "tumble-20-axis.xml"
PRINT_DENSITY_KG_MM3 = 1.23e-6  # PETG-like solid-equivalent estimate.

COLORS = {
    "skin": [229, 226, 215, 255],
    "back": [220, 222, 212, 255],
    "frame": [69, 77, 76, 255],
    "face": [244, 221, 191, 255],
    "lens": [34, 43, 46, 255],
    "accent": [226, 120, 73, 255],
    "sole": [68, 74, 72, 255],
    "component": [72, 128, 166, 255],
    "battery": [90, 168, 110, 255],
}


@dataclass
class Part:
    name: str
    body: str
    kind: str
    shape: bd.Shape
    role: str

    @property
    def mass_kg(self) -> float:
        return float(self.shape.volume) * PRINT_DENSITY_KG_MM3


PARTS: list[Part] = []


def add(name: str, body: str, kind: str, shape: bd.Shape, role: str) -> None:
    if shape is None or shape.volume <= 1:
        raise ValueError(f"Empty CAD part {name}")
    PARTS.append(Part(name, body, kind, shape, role))


def rounded(x: float, y: float, z: float, radius: float) -> bd.Shape:
    solid = bd.Box(x, y, z)
    return bd.fillet(solid.edges(), min(radius, x / 2 - .2, y / 2 - .2, z / 2 - .2))


def loft(sections: list[tuple[float, float, float, float]]) -> bd.Shape:
    return bd.loft([bd.Plane(origin=(0, 0, h)) * bd.RectangleRounded(x, y, r)
                    for h, x, y, r in sections])


def organic_loft(sections: list[tuple[float, float, float, float]]) -> bd.Shape:
    return bd.loft([bd.Plane(origin=(0, 0, h)) * bd.Ellipse(x / 2, y / 2)
                    for h, x, y, _ in sections])


def shell_halves(sections: list[tuple[float, float, float, float]],
                 wall: float = 1.8,
                 closed: bool = False,
                 cap_top: bool = False) -> tuple[bd.Shape, bd.Shape]:
    outer = organic_loft(sections)
    inner = organic_loft([(h + ((wall if closed else -2) if i == 0 else
                                (-wall if closed or cap_top else 2) if i == len(sections) - 1 else 0),
                           x - 2 * wall, y - 2 * wall, max(2, r - wall))
                          for i, (h, x, y, r) in enumerate(sections)])
    hollow = outer - inner
    if closed:
        # Neck service opening; the split front/rear skin gives access to the
        # remaining enclosed volume. Moving-link clearance is checked later.
        hollow -= bd.Pos(0, 0, sections[-1][0]) * bd.Cylinder(24, 12)
    front = hollow & (bd.Pos(100, 0, 0) * bd.Box(200, 400, 500))
    back = hollow & (bd.Pos(-100, 0, 0) * bd.Box(200, 400, 500))
    return front, back


def add_halves(stem: str, body: str,
               sections: list[tuple[float, float, float, float]],
               role: str, wall: float = 1.8,
               closed: bool = False, cap_top: bool = False) -> None:
    front, back = shell_halves(sections, wall, closed, cap_top)
    add(stem + "_front", body, "skin", front, role + "; front service half")
    add(stem + "_back", body, "back", back, role + "; rear service half")


def ring_shell(x: float, y: float, z: float, radius: float,
               wall: float = 2.0) -> bd.Shape:
    return rounded(x, y, z, radius) - rounded(
        x - 2 * wall, y - 2 * wall, z - 2 * wall, max(2, radius - wall))


def oval_motor_cover(sections: list[tuple[float, float, float]],
                     cavity_xyz: tuple[float, float, float]) -> bd.Shape:
    outside = bd.loft([bd.Plane(origin=(0, 0, z)) * bd.Ellipse(x / 2, y / 2)
                       for z, x, y in sections])
    inside = bd.loft([bd.Plane(origin=(0, 0, z + (2 if z < 0 else -2))) *
                      bd.Ellipse((x - 2.8) / 2, (y - 2.8) / 2)
                      for z, x, y in sections])
    return outside - inside - bd.Box(*cavity_xyz)


def body_tf(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    tf = np.eye(4)
    tf[:3, :3] = data.xmat[bid].reshape(3, 3)
    tf[:3, 3] = data.xpos[bid]
    return tf


def place(shape: bd.Shape, tf: np.ndarray) -> bd.Shape:
    # Compose extrinsic xyz in build123d's order, matching MuJoCo's matrix.
    x, y, z = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz", degrees=True)
    return (bd.Pos(*(tf[:3, 3] * 1000)) * bd.Rot(0, 0, z) *
            bd.Rot(0, y, 0) * bd.Rot(x, 0, 0) * shape)


def torso() -> None:
    # Core carapace stops below the face; separate high shoulders surround the
    # fixed roll motor cases. All sizes are local to pelvis/torso in mm.
    add_halves("torso", "torso", [
        (-5, 106, 91, 24), (10, 108, 107, 29),
        (37, 111, 134, 34), (67, 112, 156, 36),
        (88, 110, 165, 35), (103, 102, 150, 29),
        (110, 82, 110, 25),
    ], "Continuous belly/chest skin around battery and hip yaw cases", 2.0,
               closed=True)
    for side, sign in (("left", 1), ("right", -1)):
        # Fixed to the torso-side carrier: no motor case appears on the skin.
        add(f"{side}_fixed_shoulder_skin", f"{side}_shoulder_yaw_body",
            "skin", bd.Pos(-35, 0, -13) * oval_motor_cover(
                [(-35, 45, 34), (-24, 62, 46), (24, 62, 46), (35, 45, 34)],
                (38.6, 26.6, 46.6)),
            "Removable high-shoulder cover over fixed shoulder-roll drive")
        # The moving roll link carries the shoulder-pitch drive. Its shell is
        # visually a rounded upper-arm root with an intentional shadow seam.
        add(f"{side}_moving_shoulder_skin", f"{side}_shoulder_roll_body",
            "skin", bd.Pos(48, sign * 30, 0) * oval_motor_cover(
                [(-30, 46, 39), (-17, 70, 58), (17, 70, 58), (30, 46, 39)],
                (46.6, 38.6, 26.6)),
            "Moving shoulder cap around pitch drive; rear bridge inside")

    # Structural spine, hip deck and shoulder yoke. Motor case faces and shaft
    # attachment details remain clearance interfaces, not invented holes.
    spine = bd.Pos(-45, 0, 45) * rounded(7, 27, 160, 2)
    spine += bd.Pos(-45, 0, -13) * bd.Box(7, 36, 5)
    spine += bd.Pos(-45, 0, 91) * bd.Box(7, 56, 5)
    for sign in (-1, 1):
        spine += bd.Pos(-52, sign * 40, 123) * bd.Box(25, 70, 6)
    add("central_chassis", "pelvis", "frame", spine,
        "Printed hip deck, rear spine and shoulder motor mounting yoke")
    # Battery bay accepts the 60×30×30.5 mm source example without relying on
    # the cosmetic skin as a load path. Strap slot/connector still unverified.
    tray = (bd.Pos(0, 0, 39) * rounded(69, 40, 40, 4)) - (
        bd.Pos(0, 0, 43) * bd.Box(61.6, 31.6, 37))
    add("battery_cradle", "pelvis", "frame", tray,
        "Open-top 4S 850 mAh example pack bay; strap and XT60 exit reserved")


def head() -> None:
    add_halves("head_helmet", "head", [
        (-41, 39, 49, 15), (-28, 67, 73, 24),
        (0, 85, 86, 29), (25, 77, 78, 27),
        (42, 44, 46, 19),
    ], "Small removable head helmet with compact ape brow", 1.7,
               cap_top=True)
    # A shallow convex face mask, not a flat display. Optical windows are
    # provisional positions until actual camera board envelopes are chosen.
    face = bd.loft([
        bd.Plane.YZ.offset(36) * bd.Ellipse(37, 35),
        bd.Plane.YZ.offset(43) * bd.Ellipse(36, 34),
        bd.Plane.YZ.offset(49) * bd.Ellipse(28, 26),
    ])
    face = bd.Pos(0, 0, -4) * face
    face_cavity = bd.Pos(0, 0, -4) * bd.loft([
        bd.Plane.YZ.offset(34) * bd.Ellipse(34, 32),
        bd.Plane.YZ.offset(41) * bd.Ellipse(32, 30),
        bd.Plane.YZ.offset(45) * bd.Ellipse(25, 23),
    ])
    face -= face_cavity
    for sign in (-1, 1):
        face -= (bd.Pos(47, sign * 17, 10) * bd.Rot(0, 90, 0) *
                 bd.Cylinder(6.5, 25))
    add("face_panel", "head", "face", face,
        "Removable warm face insert with two provisional optical positions")
    for sign in (-1, 1):
        add(f"{'left' if sign == 1 else 'right'}_eye", "head", "lens",
            bd.Pos(50, sign * 17, 10) * bd.Rot(0, 90, 0) * bd.Cylinder(6.3, 3.5),
            "Provisional round optical window; sensor not selected")
    add("muzzle", "head", "skin", bd.Pos(49, 0, -23) * rounded(17, 50, 22, 8),
        "Quiet ape muzzle; no decorative moving beak")
    add("neck_guard", "head_yaw_body", "skin",
        bd.Pos(0, 0, 37) * ring_shell(43, 42, 42, 10, 2),
        "Guard around head pitch motor and neck pivot")


def limb_segment(stem: str, body: str, length: float,
                 xwidth: float, ywidth: float, role: str,
                 top_gap: float = 20) -> None:
    # A continuous fairing spans the fixed motor case at the distal end. The
    # opening is a moving joint seam, not exposed cross-brace geometry.
    sections = [(-top_gap, xwidth * .78, ywidth * .78, 13),
                (-length * .40, xwidth, ywidth, 20),
                (-length * .76, xwidth * .94, ywidth * .95, 18),
                (-length - 10, xwidth * .81, ywidth * .88, 13)]
    add_halves(stem, body, sections, role, 1.7)
    # Internal paired longitudinal webs; future mounting-hole patterns attach
    # these to source servo horns, not to the cosmetic shell.
    webs = None
    for sign in (-1, 1):
        web_z = -(length + 12) / 2
        web_length = max(14, length - 44)
        if "forearm" in stem:
            rib = bd.Pos(sign * 21, 0, web_z) * bd.Box(
                3.8, 3.5, web_length)
        else:
            rib = bd.Pos(-5, sign * (ywidth * .42), web_z) * bd.Box(
                3.5, 3.8, web_length)
        webs = rib if webs is None else webs + rib
    if "forearm" in stem:
        webs += bd.Pos(0, 0, -29) * bd.Box(46, 3.5, 3.8)
    else:
        bridge_z = -52 if "upper_arm" in stem else -29
        webs += bd.Pos(-5, 0, bridge_z) * bd.Box(3.5, ywidth * .84 + 5, 3.8)
    add(stem + "_webs", body, "frame", webs,
        "Twin internal load webs between adjacent motor output planes")


def limbs() -> None:
    for side, sign in (("left", 1), ("right", -1)):
        # Hip yaw/roll casings move independently from torso and thigh skins.
        add(f"{side}_hip_socket", f"{side}_hip_yaw", "skin",
            bd.Pos(0, 0, -72) * ring_shell(59, 54, 60, 17, 2.0),
            "Yaw-following hip socket over hip-roll motor")
        add(f"{side}_hip_pitch_cap", f"{side}_hip_roll", "skin",
            bd.Pos(25, sign * 43, 13) * ring_shell(62, 39, 52, 14, 2.0),
            "Roll-following cover of hip-pitch motor")
        limb_segment(f"{side}_thigh", f"{side}_thigh", 73, 57, 60,
                     "Smooth thigh enclosure over knee drive", 22)
        limb_segment(f"{side}_shin", f"{side}_shin", 76, 55, 57,
                     "Smooth shin enclosure over ankle drive", 22)
        foot_outer = bd.Pos(13, 0, 0) * loft([
            (-35, 93, 66, 28), (-27, 101, 70, 30),
            (-14, 78, 57, 25)])
        foot_void = bd.Pos(13, 0, -29) * bd.Box(85, 58, 23)
        add(f"{side}_foot_shell", f"{side}_foot", "skin",
            foot_outer - foot_void,
            "Broad planted foot with internal ankle-motor cavity")
        add(f"{side}_sole", f"{side}_foot", "sole",
            bd.Pos(13, 0, -35) * bd.extrude(
                bd.RectangleRounded(94, 67, 29), amount=4),
            "Replaceable friction/contact pad; material not selected")
        limb_segment(f"{side}_upper_arm", f"{side}_arm", 75, 55, 52,
                     "Short upper-arm enclosure over elbow drive", 22)
        limb_segment(f"{side}_forearm", f"{side}_forearm", 75, 52, 45,
                     "Forearm enclosure around small gripper drive", 22)
        add(f"{side}_palm", f"{side}_forearm", "skin",
            bd.Pos(0, 0, -79) * rounded(41, 43, 24, 11),
            "Broad palm/knuckle contact surface for pitched crawl")
        add(f"{side}_thumb", f"{side}_finger", "skin",
            bd.Pos(0, sign * 17, -27) * rounded(21, 15, 27, 7),
            "Single servo-driven opposed jaw hidden under palm silhouette")


def fit_own_motor_pods(model: mujoco.MjModel,
                       data: mujoco.MjData) -> None:
    """Grow smooth covers around every source-sized case fixed to that link.

    The rounded exterior grows 4 mm past each case face. A rectangular inner
    cavity grows 0.8 mm past each face, so source-sized square case corners fit.
    Other-link interference is screened separately and never auto-cut away.
    """
    for component in envelope.poses(model, data):
        if component["name"] in ("left_shoulder_roll", "right_shoulder_roll",
                                  "left_shoulder_pitch", "right_shoulder_pitch"):
            # The fitted oval housing carries the case. Cut the *transformed*
            # manufacturer envelope instead of growing a rectangular pod.
            name = component["name"]
            side = name.split("_")[0]
            shoulder = "fixed" if name.endswith("roll") else "moving"
            skin = next(p for p in PARTS if p.name == f"{side}_{shoulder}_shoulder_skin")
            own = body_tf(model, data, component["owner"])
            case_tf = np.eye(4)
            case_tf[:3, :3] = component["rot"]
            case_tf[:3, 3] = component["center"]
            cavity = place(bd.Box(*(component["half"] * 2000 + 1.6)),
                           np.linalg.inv(own) @ case_tf)
            skin.shape -= cavity
            continue
        owner = component["owner"]
        owner_tf = body_tf(model, data, owner)
        case_tf = np.eye(4)
        case_tf[:3, :3] = component["rot"]
        case_tf[:3, 3] = component["center"]
        local_tf = np.linalg.inv(owner_tf) @ case_tf
        dims = component["half"] * 2000
        outer = place(rounded(*(dims + 12.0), radius=10), local_tf)
        inner = place(bd.Box(*(dims + 1.6)), local_tf)
        attached = False
        for part in PARTS:
            if part.body != owner or part.kind not in ("skin", "back"):
                continue
            if part.name.endswith("_front"):
                half = bd.Pos(100, 0, 0) * bd.Box(200, 400, 500)
            elif part.name.endswith("_back"):
                half = bd.Pos(-100, 0, 0) * bd.Box(200, 400, 500)
            elif part.name.endswith(("_palm", "_thumb", "_muzzle")):
                continue
            else:
                half = None
            cover = outer if half is None else outer & half
            if cover is None or cover.volume <= 1:
                continue
            intersection = part.shape & cover
            if intersection is None or intersection.volume <= 5:
                continue
            candidate = (part.shape + cover) - inner
            if len(candidate.solids()) > len(part.shape.solids()):
                continue
            part.shape = candidate
            attached = True
        if not attached:
            add(component["name"] + "_cover", owner, "skin", outer - inner,
                "Separate removable case cover; mount interface still to be designed")


def relieve_foreign_motor_cases(model: mujoco.MjModel,
                                data: mujoco.MjData) -> list[dict]:
    """Cut case clearance from cosmetic shells and frame at one pose.

    This makes the assembled neutral CAD honest about occupied space. Joint
    sweeps, mounting strength and cable space are separate unresolved gates.
    """
    cuts: list[dict] = []
    for component in envelope.poses(model, data):
        case_tf = np.eye(4)
        case_tf[:3, :3] = component["rot"]
        case_tf[:3, 3] = component["center"]
        cutter = bd.Box(*(component["half"] * 2000 + 1.6))
        for part in PARTS:
            if part.kind not in ("skin", "back", "frame"):
                continue
            if part.body == component["owner"] and part.kind != "frame":
                continue
            local_tf = np.linalg.inv(body_tf(model, data, part.body)) @ case_tf
            local_cutter = place(cutter, local_tf)
            a, b = part.shape.bounding_box(optimal=False), local_cutter.bounding_box(optimal=False)
            if not all(getattr(a.min, axis) < getattr(b.max, axis) and
                       getattr(b.min, axis) < getattr(a.max, axis)
                       for axis in "XYZ"):
                continue
            before = part.shape.volume
            reduced = part.shape - local_cutter
            if reduced is None or reduced.volume <= 1:
                continue
            removed = before - reduced.volume
            if removed > 5:
                part.shape = reduced
                cuts.append({"part": part.name, "case": component["name"],
                             "removed_mm3": round(float(removed), 1)})
    return cuts


def apply_pose_case_reliefs(model: mujoco.MjModel,
                            data: mujoco.MjData) -> list[dict]:
    """Shape rigid skins against selected poses, retaining explicit provenance.

    The inputs are old traces mapped by joint name and a kinematic crawl pose;
    this is not a learned rollout or a continuous swept-volume calculation.
    """
    labeled: list[dict] = []
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    for cut in relieve_foreign_motor_cases(model, data):
        labeled.append({"pose": "neutral", **cut})
    old = mujoco.MjModel.from_xml_path(str(SOURCE))
    selected = {"forward:199", "squat:499", "acquire-place:874"}
    for label, old_qpos in envelope.recorded_poses(old):
        if label not in selected:
            continue
        data.qpos[:] = mapped_qpos(old, model, old_qpos)
        mujoco.mj_forward(model, data)
        for cut in relieve_foreign_motor_cases(model, data):
            labeled.append({"pose": label, **cut})
    data.qpos[:] = model.key_qpos[0]
    data.qpos[2] = .143
    x, y, z, w = Rotation.from_euler("y", 80, degrees=True).as_quat()
    data.qpos[3:7] = [w, x, y, z]
    for side in ("left", "right"):
        for joint, value in (("hip_pitch", -np.deg2rad(80)),
                             ("knee", 1.4), ("ankle_pitch", -1.4),
                             ("shoulder_pitch", -.9234768)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{side}_{joint}")
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    for cut in relieve_foreign_motor_cases(model, data):
        labeled.append({"pose": "illustrative_crawl_endpoint", **cut})
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    return labeled


def mesh_for(shape: bd.Shape) -> trimesh.Trimesh:
    vertices, triangles = shape.tessellate(0.15, 0.10)
    return trimesh.Trimesh(
        vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]) / 1000,
        faces=np.array(triangles), process=True)


def write_cad_mass_properties(model: mujoco.MjModel,
                              data: mujoco.MjData,
                              components: list[dict]) -> dict:
    """Replace inherited proxy inertials with actual CAD-volume mass moments.

    Printed density is a provisional material assumption; motor and battery
    masses are the sourced case assignment. Fasteners, wiring and electronics
    are not yet represented and must be added before physical procurement.
    """
    contributions: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}
    for part in PARTS:
        center_mm = part.shape.center(bd.CenterOf.MASS)
        center = np.array([center_mm.X, center_mm.Y, center_mm.Z]) / 1000
        inertia = (np.array(part.shape.matrix_of_inertia) *
                   PRINT_DENSITY_KG_MM3 * 1e-6)
        contributions.setdefault(part.body, []).append((part.mass_kg, center, inertia))
    for component in components:
        owner = component["owner"]
        case_tf = np.eye(4)
        case_tf[:3, :3] = component["rot"]
        case_tf[:3, 3] = component["center"]
        local = np.linalg.inv(body_tf(model, data, owner)) @ case_tf
        dims = component["half"] * 2
        mass = component["mass"]
        tensor = np.diag(mass / 12 * np.array([
            dims[1] ** 2 + dims[2] ** 2,
            dims[0] ** 2 + dims[2] ** 2,
            dims[0] ** 2 + dims[1] ** 2]))
        tensor = local[:3, :3] @ tensor @ local[:3, :3].T
        contributions.setdefault(owner, []).append((mass, local[:3, 3], tensor))
    tree = ET.parse(MODEL)
    root = tree.getroot()
    bodies = {}
    for owner, items in contributions.items():
        body = root.find(f".//body[@name='{owner}']")
        if body is None:
            raise ValueError(f"CAD mass owner missing from model: {owner}")
        mass = sum(item[0] for item in items)
        center = sum((m * p for m, p, _ in items), np.zeros(3)) / mass
        inertia = np.zeros((3, 3))
        for m, p, local_inertia in items:
            offset = p - center
            inertia += local_inertia + m * (np.dot(offset, offset) * np.eye(3) -
                                            np.outer(offset, offset))
        old = body.find("inertial")
        if old is not None:
            body.remove(old)
        ET.SubElement(body, "inertial", {
            "pos": " ".join(f"{x:.8g}" for x in center),
            "mass": f"{mass:.8g}",
            "fullinertia": " ".join(f"{x:.9g}" for x in (
                inertia[0, 0], inertia[1, 1], inertia[2, 2],
                inertia[0, 1], inertia[0, 2], inertia[1, 2]))})
        bodies[owner] = {"mass_kg": round(float(mass), 5),
                         "center_local_m": center.round(5).tolist(),
                         "inertia_body_kg_m2": inertia.round(8).tolist()}
    ET.indent(tree, space="  ")
    tree.write(MODEL, encoding="unicode", xml_declaration=True)
    checked = mujoco.MjModel.from_xml_path(str(MODEL))
    return {"body_mass_properties": bodies,
            "model_mass_kg": round(float(sum(checked.body_mass)), 5),
            "cad_and_component_mass_kg": round(sum(
                p.mass_kg for p in PARTS) + sum(c["mass"] for c in components), 5)}


def export(preview: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_candidate(SOURCE, MODEL)
    variant = ET.parse(MODEL)
    for side, sign in (("left", 1), ("right", -1)):
        arm = variant.find(f".//body[@name='{side}_arm']")
        forearm = variant.find(f".//body[@name='{side}_forearm']")
        finger = variant.find(f".//body[@name='{side}_finger']")
        assert arm is not None
        assert forearm is not None and finger is not None
        # Keep the 75 mm arm and 30 mm shoulder lateral offset from the
        # original source-sized compact envelope screen. The wider/longer
        # study added 79 mm of exterior span without proving reach benefits.
    neck = variant.find(".//body[@name='head_yaw_body']")
    assert neck is not None
    neck.set("pos", "0 0 .119")
    ET.indent(variant, space="  ")
    variant.write(MODEL, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    layout = copy.deepcopy(envelope.CFG)
    layout["tails"].update({"shoulder_roll_left": [0, 0, -1],
                            "shoulder_roll_right": [0, 0, -1],
                            "shoulder_pitch": [1, 0, 0]})
    envelope.CFG = layout
    PARTS.clear()
    torso()
    head()
    limbs()
    fit_own_motor_pods(model, data)
    sampled_case_reliefs = apply_pose_case_reliefs(model, data)
    scene = trimesh.Scene()
    mesh_parts: list[tuple[Part, trimesh.Trimesh]] = []
    neutral_solids = []
    records = []
    print_dir = OUT / "printed-parts"
    if not preview:
        print_dir.mkdir(exist_ok=True)
        current = {part.name for part in PARTS}
        for previous in print_dir.iterdir():
            if previous.suffix in (".step", ".stl") and previous.stem not in current:
                previous.unlink()
    for part in PARTS:
        tf = body_tf(model, data, part.body)
        placed = place(part.shape, tf)
        if not preview:
            stem = print_dir / part.name
            bd.export_step(part.shape, stem.with_suffix(".step"))
            bd.export_stl(part.shape, stem.with_suffix(".stl"), tolerance=.5)
            neutral_solids.append(placed)
        mesh = mesh_for(part.shape)
        mesh.visual.vertex_colors = COLORS[part.kind]
        mesh_parts.append((part, mesh))
        scene.add_geometry(mesh, node_name=part.name,
                           geom_name=part.name, transform=tf)
        # OCC's fast BSpline AABB can include far-away control points. The
        # actual STL tessellation is what must fit the printer's build volume.
        mesh_bounds_mm = mesh.bounds * 1000
        records.append({"name": part.name, "body": part.body,
                        "kind": part.kind, "role": part.role,
                        "volume_mm3": round(float(part.shape.volume), 2),
                        "mass_kg_assumed": round(part.mass_kg, 6),
                        "print_bounds_mm": (mesh_bounds_mm[1] - mesh_bounds_mm[0]).round(2).tolist(),
                        "step": f"printed-parts/{part.name}.step",
                        "stl": f"printed-parts/{part.name}.stl",
                        "cad_solid_valid": bool(part.shape.is_valid),
                        "solid_count": len(part.shape.solids()),
                        "mesh_watertight": bool(mesh.is_watertight)})
    if neutral_solids:
        bd.export_step(bd.Compound(children=neutral_solids), OUT / "tumble-assembled.step")
    components = envelope.poses(model, data) + [envelope.battery_pose(model, data)]
    component_records = []
    for component in components:
        component_records.append({"name": component["name"],
                                  "body": component["owner"],
                                  "source_part": component["part"],
                                  "center_m": component["center"].tolist(),
                                  "rotation": component["rot"].tolist(),
                                  "size_m": (component["half"] * 2).tolist(),
                                  "mass_kg": component["mass"]})
        box = trimesh.creation.box(extents=component["half"] * 2)
        box.visual.vertex_colors = COLORS[
            "battery" if component["family"] == "battery" else "component"]
        tf = np.eye(4)
        tf[:3, :3] = component["rot"]
        tf[:3, 3] = component["center"]
        scene.add_geometry(box, node_name="component_" + component["name"],
                           geom_name="component_" + component["name"],
                           transform=tf)
    scene.export(OUT / "tumble-assembled.glb")
    old = mujoco.MjModel.from_xml_path(str(SOURCE))
    saved_qpos = {label: mapped_qpos(old, model, old_qpos)
                  for label, old_qpos in envelope.recorded_poses(old)
                  if label in ("forward:199", "squat:499")}
    pose_qpos = {
        "tumble-forward.glb": saved_qpos["forward:199"],
        "tumble-squat.glb": saved_qpos["squat:499"],
    }
    home_qpos = np.array(model.key_qpos[0])
    for side in ("left", "right"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{side}_shoulder_pitch")
        home_qpos[model.jnt_qposadr[jid]] = .3
    pose_qpos["tumble-home-pitch-0.3.glb"] = home_qpos
    crawl_qpos = np.array(model.key_qpos[0])
    crawl_qpos[2] = .143
    x, y, z, w = Rotation.from_euler("y", 80, degrees=True).as_quat()
    crawl_qpos[3:7] = [w, x, y, z]
    for side in ("left", "right"):
        for joint, value in (("hip_pitch", -np.deg2rad(80)),
                             ("knee", 1.4), ("ankle_pitch", -1.4),
                             ("shoulder_pitch", -.9234768)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{side}_{joint}")
            crawl_qpos[model.jnt_qposadr[jid]] = value
    pose_qpos["tumble-crawl.glb"] = crawl_qpos
    for filename, qpos in pose_qpos.items():
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        posed = trimesh.Scene()
        for part, mesh in mesh_parts:
            posed.add_geometry(mesh, node_name=part.name, geom_name=part.name,
                               transform=body_tf(model, data, part.body))
        for component in envelope.poses(model, data) + [envelope.battery_pose(model, data)]:
            box = trimesh.creation.box(extents=component["half"] * 2)
            box.visual.vertex_colors = COLORS[
                "battery" if component["family"] == "battery" else "component"]
            transform = np.eye(4)
            transform[:3, :3] = component["rot"]
            transform[:3, 3] = component["center"]
            posed.add_geometry(box, node_name="component_" + component["name"],
                               geom_name="component_" + component["name"],
                               transform=transform)
        posed.export(OUT / filename)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    shutil.copy2(HERE / "tumble_viewer.html", OUT / "index.html")
    shutil.copy2(ROOT / "docs/design/ape-concepts/06-tumble-compact-shoulder.png",
                 OUT / "06-tumble-compact-shoulder.png")
    mass_properties = write_cad_mass_properties(model, data, components)
    joints = [{"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
               "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(model.jnt_bodyid[jid])),
               "axis_local": model.jnt_axis[jid].tolist(),
               "range_rad": model.jnt_range[jid].tolist()}
              for jid in range(1, model.njnt)]
    report = {"status": "CAD candidate; clearance and load paths not certified",
              "model": str(MODEL.relative_to(ROOT)),
              "model_sha256": envelope.sha256(MODEL),
              "source_sha256": envelope.sha256(Path(__file__)),
              "sketch": "docs/design/ape-concepts/06-tumble-compact-shoulder.png",
              "printed_material_density_g_cm3_assumed": 1.23,
              "printed_mass_kg_assumed": round(sum(p.mass_kg for p in PARTS), 4),
              "assembly_extents_mm": (scene.extents * 1000).round(2).tolist(),
              "assembly_bounds_mm": (scene.bounds * 1000).round(2).tolist(),
              "maximum_print_extent_mm": max(max(p["print_bounds_mm"]) for p in records),
              "parts": records, "joints": joints,
              "motor_and_battery_envelopes": component_records,
              "source_component_count": len(component_records),
              "sampled_case_reliefs": sampled_case_reliefs,
              "posed_views": {"tumble-home-pitch-0.3.glb": "Trial parked shoulder-pitch pose; only dominant neutral palm/thigh and forearm/hip intersections screened",
                              "tumble-forward.glb": "Old forward trace pose mapped by joint name",
                              "tumble-squat.glb": "Old squat trace pose mapped by joint name",
                              "tumble-crawl.glb": "Illustrative kinematic four-point pose"},
              "mass_properties": mass_properties,
              "unknown_interfaces": ["servo output horns and mounting hole positions",
                                     "shaft bearings", "strap and XT60 exit",
                                     "wire bend radii", "fastener bosses",
                                     "printed strength and wear",
                                     "moving shell-to-shell clearances"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("printed_mass_kg_assumed",
                                              "maximum_print_extent_mm",
                                              "source_component_count")}, indent=2))
    print(f"{len(records)} printed parts in {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Export GLB/report only for fast visual iteration")
    export(parser.parse_args().preview)
