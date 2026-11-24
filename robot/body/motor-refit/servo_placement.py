"""Servo placement for the motor-refit body builder: where an ST3025 / C047 actuator sits on an r1 joint axis.
Builder geometry (not a check); the body helpers it uses live in robot/body/checks/bodylib.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robot/body"))
from checks.bodylib import Body, src  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("motor-refit")   # configured values: robot/body/motor-refit/motor_refit.bom.yaml config:

ST3025_STEP = ROOT / "artifacts/cove50-engineering-competition/elbow-studio/direct-servo/r1/source/ST3025-MODEL/ST3025-MODEL.stp"

# ST3025 STEP solid classes (local frame: output axis = +Z through x=10, y=0; case spans x -20..20, y -10..10).
ST3025_OUTPUT = set(CFG["ST3025_OUTPUT"])
ST3025_REAR = set(CFG["ST3025_REAR"])
ST3025_AXIS_X = CFG["ST3025_AXIS_X"]


def joint_spec(side: str) -> dict:
    s = side
    return {
        f"{s}_hip_pitch": dict(case=f"{s}_pitch_servo_case_reference", horn=f"{s}_pitch_upper_horn_reference",
                               idler=f"{s}_pitch_passive_idler_reference", new="st3025"),
        f"{s}_knee": dict(case=f"{s}_C047_knee_case", horn=f"{s}_C047_knee_main_horn",
                          idler=f"{s}_C047_knee_rear_idler", new="st3025"),
        f"{s}_ankle_pitch": dict(case=f"{s}_B13R14_raised_C047_ankle_pitch_case",
                                 horn=f"{s}_B13R14_raised_C047_ankle_pitch_main_horn",
                                 idler=f"{s}_B13R14_raised_C047_ankle_pitch_rear_idler", new="st3025"),
    }


NECK = {
    "neck_yaw": dict(case="upper__neck_yaw_case_XM335", horn="upper__neck_yaw_front_horn", idler=None, new="c047"),
    "neck_pitch": dict(case="upper__neck_pitch_case_XM335", horn="upper__neck_pitch_front_horn", idler=None, new="c047"),
}


def st3025_local() -> dict:
    """Case (static) and output (moving) meshes of the official ST3025 STEP, local frame."""
    from build123d import import_step

    shape = import_step(str(src(ST3025_STEP)))
    out = {"case": [], "output": []}
    for i, so in enumerate(shape.solids()):
        verts, tris = so.tessellate(0.05, 0.3)
        m = trimesh.Trimesh(np.array([(v.X, v.Y, v.Z) for v in verts]), np.array(tris), process=True)
        out["output" if i in ST3025_OUTPUT | ST3025_REAR else "case"].append(m)
    return {k: trimesh.util.concatenate(v) for k, v in out.items()}


def frame_from_c047(body: Body, spec: dict) -> dict:
    """Axis point/direction, horn direction, long-dimension direction of an r1 servo, from its meshes."""
    case = body.parts[spec["case"]]
    jname = spec["joint"]
    j = body.joints[jname]
    o = np.array(j["origin_world_mm"], float)
    a = np.array(j["axis_world_neutral"], float)
    a /= np.linalg.norm(a)
    c = np.mean(np.array(case["bounds_mm"], float), axis=0)
    h = np.mean(np.array(body.parts[spec["horn"]]["bounds_mm"], float), axis=0)
    horn_dir = a if np.dot(h - c, a) >= 0 else -a
    radial = (c - o) - np.dot(c - o, a) * a
    long_dir = radial / (np.linalg.norm(radial) or 1.0)
    # Axial extents of the r1 actuator (case + horn + idler) along horn_dir, relative to the axis origin.
    pts = [np.array(case["bounds_mm"], float)]
    pts.append(np.array(body.parts[spec["horn"]]["bounds_mm"], float))
    if spec.get("idler"):
        pts.append(np.array(body.parts[spec["idler"]]["bounds_mm"], float))
    corners = []
    for b in pts:
        for xi in (0, 1):
            for yi in (0, 1):
                for zi in (0, 1):
                    corners.append([b[xi][0], b[yi][1], b[zi][2]])
    s = (np.array(corners) - o) @ horn_dir
    horn_face = float(np.dot(np.array(body.parts[spec["horn"]]["bounds_mm"], float) - o, horn_dir).max())
    return dict(origin=o, axis=a, horn_dir=horn_dir, long_dir=long_dir, horn_face=horn_face,
                axial_span=(float(s.min()), float(s.max())), radial_offset=float(np.linalg.norm(radial)))


def place_st3025(local: dict, fr: dict, axial: str, offset: float = 0.0) -> dict:
    """World transform mapping ST3025 local frame onto the r1 joint axis (neutral pose)."""
    z = fr["horn_dir"]
    x = -fr["long_dir"]            # local case spans x -20..+10 from the axis: the body lies toward -x
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    # local axis point (10, 0, 0); local wheel outer face z = 24.9, rear wheel outer face z = -26.05
    face_local, rear_local = 24.9, -26.05
    if axial == "horn":
        shift = fr["horn_face"] - face_local
    else:
        lo, hi = fr["axial_span"]
        shift = 0.5 * (lo + hi) - 0.5 * (face_local + rear_local)
    shift += offset
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = fr["origin"] - R @ np.array([ST3025_AXIS_X, 0.0, 0.0]) + z * shift
    new_span = (rear_local + shift, face_local + shift)
    return dict(T=T, axial_span_new=new_span)


def place_c047(body: Body, fr_ref: dict, ref_spec: dict, fr_dst: dict, axial: str, offset: float = 0.0) -> dict:
    """Use the r1 knee C047 (case + horn + idler) as a rigid model and move it onto the neck axis."""
    def basis(fr):
        z = fr["horn_dir"]; x = fr["long_dir"]; y = np.cross(z, x)
        return np.column_stack([x, y, z])
    Rs, Rd = basis(fr_ref), basis(fr_dst)
    R = Rd @ Rs.T
    lo_s, hi_s = fr_ref["axial_span"]
    if axial == "horn":
        shift = fr_dst["horn_face"] - fr_ref["horn_face"]
    else:
        lo_d, hi_d = fr_dst["axial_span"]
        shift = 0.5 * (lo_d + hi_d) - 0.5 * (lo_s + hi_s)
    shift += offset
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = fr_dst["origin"] - R @ fr_ref["origin"] + fr_dst["horn_dir"] * shift
    return dict(T=T, axial_span_new=(lo_s + shift, hi_s + shift))
