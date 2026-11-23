"""Refit the complete Cove-50 FDM body (r1, commit 48c38ceda) to the locked three-tier motor set M-44khpr.

Idle task T-jrnxfn. Decisions: M-jrrxya (owner: combine a small hip shift with a pelvis battery rework, keep the V body)
and M-jxjse2 (4 mm hip pitch shift per side, pelvis cells laid horizontally on the midline).

Legs (this file): hip pitch, knee and ankle pitch take the Waveshare ST3025 (official STEP). The r1 printed parts are
edited with small exact B-rep operations (cad_ops.py) instead of being regenerated:
  hip pitch   output face 4 mm outward; the whole leg below follows (hip pitch spacing 118 -> 126 mm); the unused
              rear output wheel is left off; flange gets the ST3025 wheel bolt circle.
  knee        ST3025 centred on the r1 joint (like the r1 C047). Shin clevis plates move out 6.6 / 7.2 mm (stretch),
              thigh case plate moves 4.6 mm onto the new case face, thigh end block extended 5 mm down to the shorter
              case; M2 holes for the case front-face screws and corner through-bolts.
  ankle pitch same centring; the shin (case side) gets pads to both case faces, M2 holes; the coupler stretches above
              z = 40 only, so the ankle roll interface below stays put, and a block reconnects its lateral leg.
  shells      thigh and shin hard covers widen by the same stretch planes.
Left parts are mirrors of the right ones (as in r1).

  .venv/bin/python -B robot/body/motor-refit/motor_refit.py legs
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Dir, gp_Pnt, gp_Trsf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import cad_ops as C  # noqa: E402
sys.path.insert(0, str(ROOT / "robot/body"))
sys.path.insert(0, str(ROOT / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402  (joint ranges from the body definitions)
from checks import bodylib as F  # noqa: E402
import servo_placement as SP  # noqa: E402

R1 = ROOT / "artifacts/robot/body/assembly"
OUT = ROOT / "artifacts/robot/body/motor-refit"
PARTS = OUT / "parts"

HIP_SHIFT = 4.0
DOWNSTREAM = ("pitch_link", "roll_link", "shin", "ankle_pitch", "foot")  # leg links below the hip pitch, per side
SHIFTED_JOINTS = ("hip_roll", "knee", "ankle_pitch", "ankle_roll")
# stretch planes (r1 world x, right leg) and moves for the knee/ankle widening
PLANE_MED, D_MED = 48.0, -6.6
PLANE_LAT, D_LAT = 72.0, +7.2
CLR = 0.15  # mm contact clearance to purchased cases

OLD_SERVO_PARTS = ("{s}_pitch_servo_case_reference", "{s}_pitch_upper_horn_reference", "{s}_pitch_passive_idler_reference",
                   "{s}_C047_knee_case", "{s}_C047_knee_main_horn", "{s}_C047_knee_rear_idler",
                   "{s}_B13R14_raised_C047_ankle_pitch_case", "{s}_B13R14_raised_C047_ankle_pitch_main_horn",
                   "{s}_B13R14_raised_C047_ankle_pitch_rear_idler")


def load_step(path):
    from build123d import import_step
    return import_step(str(F.src(path))).wrapped


def save(name, shape, kind="printed"):
    from build123d import Solid, Compound, export_step
    PARTS.mkdir(parents=True, exist_ok=True)
    sols = C.solids(shape)
    obj = Solid(sols[0]) if len(sols) == 1 else Compound(shape)
    step = PARTS / f"{name}.step"
    export_step(obj, str(step))
    tm = None
    try:  # closed, conforming surface (exact mesh booleans in the checks need it)
        from surface_mesh import step_to_closed_mesh
        tm = step_to_closed_mesh(step, 1.2 if kind == "printed" else 1.6)
        if F.to_mf(tm) is None:
            tm = None
    except Exception as exc:
        print(f"{name}: gmsh surface mesh failed ({str(exc)[:60]}); OCC tessellation used", flush=True)
    if tm is None:
        from build123d import export_stl
        stl = PARTS / f"{name}.stl"
        export_stl(obj, str(stl), tolerance=0.04, angular_tolerance=0.25)
        tm = trimesh.load(stl, force="mesh", process=True); stl.unlink()
    v = np.asarray(tm.vertices, float); f = np.asarray(tm.faces, np.int64)
    npz = PARTS / f"{name}.npz"
    np.savez_compressed(npz, vertices=v, faces=f)
    vol = C.volume(shape)
    return dict(native_step=str(step.relative_to(ROOT)), mesh=str(npz.relative_to(ROOT)), mesh_units="mm",
                native_step_sha256=hashlib.sha256(step.read_bytes()).hexdigest(),
                mesh_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                bounds_mm=[v.min(0).round(3).tolist(), v.max(0).round(3).tolist()],
                native_solid_count=len(sols), native_validity=[bool(C.valid(s)) for s in sols],
                volume_mm3=round(vol, 2), **({"mass_kg": BOM_CONFIG.mass_kg(name, vol),
                                              "mass_basis": f"Exact native STEP volume * {BOM_CONFIG.density_kg_mm3(name) * 1e6:g} g/cm3 ({BOM_CONFIG.material(name)}, declared in its definition)"} if kind == "printed" else {}))


def st3025_solids():
    from build123d import import_step
    sol = import_step(str(F.src(SP.ST3025_STEP))).solids()
    case = [s.wrapped for i, s in enumerate(sol) if i not in SP.ST3025_OUTPUT | SP.ST3025_REAR]
    front = [s.wrapped for i, s in enumerate(sol) if i in SP.ST3025_OUTPUT]
    rear = [s.wrapped for i, s in enumerate(sol) if i in SP.ST3025_REAR]
    return case, front, rear


def compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    c = TopoDS_Compound(); b = BRep_Builder(); b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def wheel_holes(ax_y, ax_z, x0, x1):
    """ST3025 output wheel: 4 x M3 clearance on a 14 mm circle, plus clearance for the centre screw head."""
    pts = [(ax_y, ax_z + 7.0), (ax_y, ax_z - 7.0), (ax_y + 7.0, ax_z), (ax_y - 7.0, ax_z)]
    return C.holes("x", pts, 1.65, x0, x1) + C.holes("x", [(ax_y, ax_z)], 3.25, x0, x1)


def case_holes(ax_y, ax_z, long_sign, x_front0, x_front1, x_thru0, x_thru1):
    """ST3025 front-face far screw group (4 x M2 on a 16 mm circle 20 mm from the axis) and the two far corner
    through-bolts (M2, 28 mm from the axis, +-8 mm)."""
    zc = ax_z + long_sign * 20.0
    grp = [(ax_y + 3.68, zc + long_sign * 7.17), (ax_y - 3.68, zc + long_sign * 7.17),
           (ax_y + 7.17, zc + long_sign * 3.68), (ax_y - 7.17, zc + long_sign * 3.68)]
    thru = [(ax_y + 8.0, ax_z + long_sign * 28.0), (ax_y - 8.0, ax_z + long_sign * 28.0)]
    return C.holes("x", grp, 1.2, x_front0, x_front1) + C.holes("x", thru, 1.2, x_thru0, x_thru1)


def build_legs(manifest):
    body = F.Body(manifest)
    st_mesh = SP.st3025_local()
    case_s, front_s, rear_s = st3025_solids()
    t0 = time.time()
    out = {}  # name -> edited shape
    sleeve_spec = {}  # name -> (stretch moves, cut tools that remove material) applied to the part's steel sleeves

    # ---- servo placements (right leg, r1 frames) ----
    place = {}
    for jn, mode, off in (("R_hip_pitch", "horn", HIP_SHIFT), ("R_knee", "center", 0.0), ("R_ankle_pitch", "center", 0.0)):
        sp = dict(SP.joint_spec("R")[jn], joint=jn)
        fr = SP.frame_from_c047(body, sp)
        place[jn] = (fr, SP.place_st3025(st_mesh, fr, mode, off))

    def placed(jn, which):
        fr, pl = place[jn]
        sh = compound(case_s if which == "case" else (front_s if jn == "R_hip_pitch" else front_s + rear_s))
        return C.transform(sh, pl["T"])

    def bbox(shape):
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        b = Bnd_Box(); BRepBndLib.Add_s(shape, b); v = b.Get()
        return np.array(v[:3]), np.array(v[3:])

    knee_case, knee_out = placed("R_knee", "case"), placed("R_knee", "output")
    ank_case, ank_out = placed("R_ankle_pitch", "case"), placed("R_ankle_pitch", "output")
    hip_case, hip_out = placed("R_hip_pitch", "case"), placed("R_hip_pitch", "output")
    kc_lo, kc_hi = bbox(knee_case); ac_lo, ac_hi = bbox(ank_case)
    print("knee case", kc_lo.round(2), kc_hi.round(2), "ankle case", ac_lo.round(2), ac_hi.round(2), flush=True)
    KY, KZ = 4.0, 123.0
    AY, AZ = 4.0, 61.0
    HY, HZ = 2.0, 219.0

    def main_faces(jn):
        """World x of the ST3025 case's main front face (local z 18.55) and main rear face (local z -21.45)."""
        T = place[jn][1]["T"]
        return float((T @ [10.0, 0.0, 18.55, 1.0])[0]), float((T @ [10.0, 0.0, -21.45, 1.0])[0])

    def pocket(lo, hi, xf, xr, ax_z, long_sign=+1):
        """Case envelope + CLR: main faces away from the output (|long| > 6 mm), bounding box near it."""
        xa, xb = sorted((xf, xr))
        far_z = (ax_z + 6.0, hi[2] + CLR) if long_sign > 0 else (lo[2] - CLR, ax_z - 6.0)
        near_z = (lo[2] - CLR, ax_z + 6.0) if long_sign > 0 else (ax_z - 6.0, hi[2] + CLR)
        return [C.box((xa - CLR, lo[1] - CLR, far_z[0]), (xb + CLR, hi[1] + CLR, far_z[1])),
                C.box((lo[0] - CLR, lo[1] - CLR, near_z[0]), (hi[0] + CLR, hi[1] + CLR, near_z[1]))]

    kf, kr = main_faces("R_knee"); af, ar = main_faces("R_ankle_pitch")
    print(f"knee case main faces front {kf:.2f} rear {kr:.2f}; ankle front {af:.2f} rear {ar:.2f}", flush=True)

    # ---- shin: widen at both stretch planes, then the ankle case interface ----
    s = load_step(manifest_part(manifest, "R_shin_combined")["native_step"])
    s = C.stretch(s, "x", PLANE_MED, D_MED)
    s = C.stretch(s, "x", PLANE_LAT, D_LAT)
    med_in, lat_in = 41.25 + D_MED, 78.75 + D_LAT  # inner faces of the moved side plates (r1 41.25 / 78.75)
    region = ((-1e3, ac_lo[1] - 0.5, AZ + 6.0), (1e3, ac_hi[1] + 0.5, ac_hi[2] + 0.5))
    s, v1 = C.pad(s, "x", med_in, ar - CLR, region)
    s, v2 = C.pad(s, "x", lat_in, af + CLR, region)
    sweep = C.cylinder((ac_lo[0] - 2.0, AY, AZ), (ac_hi[0] + 6.0, AY, AZ), 10.2)
    tools = pocket(ac_lo, ac_hi, af, ar, AZ) + [sweep] + wheel_holes(KY, KZ, 28.0, 36.0) + wheel_holes(KY, KZ, 84.5, 92.0)
    tools += case_holes(AY, AZ, +1, af - 1.0, af + 10.0, ar - 6.0, af + 10.0)
    s = C.cut_many(s, tools)
    print(f"shin done ({time.time() - t0:.0f}s) pads {v1:.0f}/{v2:.0f} mm3 solids {C.n_solids(s)}", flush=True)
    out["R_shin_combined"] = s
    sleeve_spec["R_shin_combined"] = ([("x", PLANE_MED, D_MED, None), ("x", PLANE_LAT, D_LAT, None)], pocket(ac_lo, ac_hi, af, ar, AZ) + [sweep])

    # ---- thigh: case plate onto the new medial case face, end block down to the shorter case ----
    s = load_step(manifest_part(manifest, "R_thigh_combined")["native_step"])
    s = C.stretch_region(s, "x", 46.0, (kf - CLR) - 43.4, ((-1e3, -1e3, 128.0), (1e3, 1e3, 166.0)))  # r1 plate face 43.4
    s, v3 = C.pad(s, "z", 158.2, kc_hi[2] + CLR, ((kc_lo[0] - 3, kc_lo[1] - 0.5, 0), (kc_hi[0] + 3, kc_hi[1] + 0.5, 0)))
    sweep = C.cylinder((kc_lo[0] - 6.0, KY, KZ), (kc_hi[0] + 3.0, KY, KZ), 10.2)
    tools = pocket(kc_lo, kc_hi, kf, kr, KZ) + [sweep] + case_holes(KY, KZ, +1, kf - 8.0, kf + 1.0, kf - 8.0, kr + 6.0)
    s = C.cut_many(s, tools)
    print(f"thigh done ({time.time() - t0:.0f}s) end pad {v3:.0f} mm3 solids {C.n_solids(s)}", flush=True)
    out["R_thigh_combined"] = s
    sleeve_spec["R_thigh_combined"] = ([("x", 46.0, (kf - CLR) - 43.4, ((-1e3, -1e3, 128.0), (1e3, 1e3, 166.0)))], pocket(kc_lo, kc_hi, kf, kr, KZ) + [sweep])

    # ---- ankle coupler: widen above z = 40 only, reconnect the lateral leg, wheel holes ----
    s = load_step(manifest_part(manifest, "R_ankle_coupler_combined")["native_step"])
    # reinforcement (first, on the clean r1 solid): the roll-horn plate (y 16.65-20.8) carries the ankle pitch stall
    # torque into the foot roll horn; ST3025 stall is 1.33x the C047's. Thicken it 3 mm at the back over the horn area.
    s, vpad = C.pad(s, "y", 20.8, 23.8, ((62.0, 0.0, 14.0), (88.0, 0.0, 38.0)))
    reg = ((-1e3, -1e3, 40.0), (1e3, 1e3, 1e3))
    s = C.stretch_region(s, "x", 48.0, -7.2, reg)
    s = C.stretch_region(s, "x", 70.0, +6.6, reg)
    s = C.fuse(s, C.box((75.0, 3.0, 37.0), (87.6, 20.0, 45.0)))
    s = C.cut_many(s, wheel_holes(AY, AZ, 28.0, 36.0) + wheel_holes(AY, AZ, 84.5, 92.0))
    print(f"coupler done ({time.time() - t0:.0f}s) roll plate +{vpad:.0f} mm3 solids {C.n_solids(s)}", flush=True)
    out["R_ankle_coupler_combined"] = s
    sleeve_spec["R_ankle_coupler_combined"] = ([("x", 48.0, -7.2, reg), ("x", 70.0, +6.6, reg)], [])

    # ---- hip yaw-to-pitch (hip pitch case side): the r1 U plate's legs bolted to the wider C047 face; the narrower
    # ST3025 sits in the U opening. Close the opening over the case's far end with a 2.5 mm plate on the main front
    # face (clear of the output wheel sweep), then the case pocket and the M2 face screws / corner through-bolts.
    hf, hr = main_faces("R_hip_pitch")
    hc_lo, hc_hi = bbox(hip_case)
    s = load_step(manifest_part(manifest, "R_hip_yaw_to_pitch_combined")["native_step"])
    s = C.fuse(s, C.box((hf + CLR, -9.5, HZ + 11.0), (hf + CLR + 2.5, 11.5, hc_hi[2] + 1.0)))
    tools = pocket(hc_lo, hc_hi, hf, hr, HZ) + case_holes(HY, HZ, +1, hf - 1.0, hf + 4.0, hr - 4.0, hf + 4.0)
    s = C.cut_many(s, tools)
    print(f"hip yaw-to-pitch done, front face {hf:.2f} rear {hr:.2f} solids {C.n_solids(s)}", flush=True)
    out["R_hip_yaw_to_pitch_combined"] = s
    sleeve_spec["R_hip_yaw_to_pitch_combined"] = ([], pocket(hc_lo, hc_hi, hf, hr, HZ))

    # ---- hip pitch flange: ST3025 front wheel bolt circle (shift applied with the leg) ----
    s = load_step(manifest_part(manifest, "R_hip_pitch_to_roll_combined")["native_step"])
    # reinforcement: the front plate around the upper hip-roll case bolt (x 47-63, z 195-207) is 2-3 mm and carries the
    # STS3250 hip-roll stall (1.67x the C047's). Thicken it forward to y -13.6, above the roll horn disc.
    s, vpad = C.pad(s, "y", -10.8, -13.6, ((47.0, 0.0, 195.5), (63.0, 0.0, 207.0)))
    print(f"hip flange front plate +{vpad:.0f} mm3", flush=True)
    s = C.cut_many(s, wheel_holes(HY, HZ, 57.0, 68.0))
    # the pad must stay clear of the thigh, which turns about the hip-roll axis (r1 frame: through (64, -12, 190), along y)
    from OCP.gp import gp_Ax1
    thigh_sweep = []
    for a in np.arange(-30.0, 25.1, 2.5):
        t = gp_Trsf(); t.SetRotation(gp_Ax1(gp_Pnt(64.0, -12.0, 190.0), gp_Dir(0, 1, 0)), np.radians(a))
        thigh_sweep.append(BRepBuilderAPI_Transform(C.translate(out["R_thigh_combined"], (0.0, 0.0, 0.0)), t, True).Shape())
    pad_zone = C.box((46.0, -14.5, 194.0), (64.0, -9.0, 208.0))
    clear = [C.common(t_, pad_zone) for t_ in thigh_sweep]
    s = C.cut_many(s, [c for c in clear if C.volume(c) > 1e-3], sequential=True)
    out["R_hip_pitch_to_roll_combined"] = s

    # ---- shells ----
    for n in ("R_thigh_front_hard_cover", "R_thigh_back_hard_cover", "R_shin_front_hard_cover", "R_shin_back_hard_cover"):
        s = load_step(manifest_part(manifest, n)["native_step"])
        s = C.stretch(s, "x", PLANE_MED, D_MED)
        s = C.stretch(s, "x", PLANE_LAT, D_LAT)
        out[n] = s
    # the coupler's back plate swings into the shin back cover's lower edge at ankle pitch > +5 deg (r1 too: 263 mm3;
    # refit 313 mm3 at +25). Notch the cover by the coupler's own sweep to +27.5 deg (the coupler is the weaker part).
    from OCP.gp import gp_Ax1
    cov = out["R_shin_back_hard_cover"]; v0 = C.volume(cov)
    swept = []
    for a in np.arange(5.0, 27.6, 2.5):
        t = gp_Trsf(); t.SetRotation(gp_Ax1(gp_Pnt(66.0, AY, AZ), gp_Dir(1, 0, 0)), np.radians(a))
        swept.append(BRepBuilderAPI_Transform(out["R_ankle_coupler_combined"], t, True).Shape())
    out["R_shin_back_hard_cover"] = C.cut_many(cov, swept, sequential=True)
    print(f"shells done ({time.time() - t0:.0f}s); shin back cover notched {v0 - C.volume(out['R_shin_back_hard_cover']):.0f} mm3", flush=True)

    servos = {"R_hip_pitch_ST3025_case": (hip_case, "R_yaw_link"), "R_hip_pitch_ST3025_output": (hip_out, "R_pitch_link"),
              "R_knee_ST3025_case": (knee_case, "R_roll_link"), "R_knee_ST3025_output": (knee_out, "R_shin"),
              "R_ankle_pitch_ST3025_case": (ank_case, "R_shin"), "R_ankle_pitch_ST3025_output": (ank_out, "R_ankle_pitch")}
    return out, servos, sleeve_spec


CELL_Y, CELL_Z = 6.0, (220.0, 242.7)   # M-jxjse2: two 21700 cells lying along y on the midline, stacked
CELL_R, CELL_L = 21.5 / 2, 70.0
WEB_Y = (CELL_Y - 9.0, CELL_Y + 9.0)
FIN_SEGMENTS = ((CELL_Y - 27.0, WEB_Y[0] - 1.0), (WEB_Y[1] + 1.0, CELL_Y + 20.0))  # clear of the shells
FRONT_POSTS = ((-6.5, 0.0), (6.5, 0.0))  # x of the two new front-shell posts
FRONT_POST_Z = 257.5                     # above the upper cell (top 253.45) and the hip pitch sweep (case top 249)
FRONT_POST_Y = (-37.5, -41.0)            # post face (tab front); skin inner face -39.96, outer -41.97 at z 257.5


def edit_front_shell(manifest):
    """Remove the two lower shelf channels that bolted to the r1 cradle front block (x +-12..20, z 221-227: now hip
    pitch sweep and cell sleeve space) and add two M2 posts from the skin to the new cradle tabs (FRONT_POSTS)."""
    s = load_step(manifest_part(manifest, "center_front_shell")["native_step"])
    v0 = C.volume(s)
    shelves = [C.box((sx * 10.5 if sx > 0 else -21.5, -38.6, 219.0), (21.5 if sx > 0 else -10.5, -10.0, 229.5)) for sx in (1, -1)]
    s = C.cut_many(s, shelves)
    posts = [C.cylinder((x, FRONT_POST_Y[1], FRONT_POST_Z), (x, FRONT_POST_Y[0], FRONT_POST_Z), 3.0) for x, _ in FRONT_POSTS]
    s = C.fuse(s, *posts)
    s = C.cut_many(s, C.holes("y", [(x, FRONT_POST_Z) for x, _ in FRONT_POSTS], 1.1, FRONT_POST_Y[1] - 2.0, FRONT_POST_Y[0] + 0.5))
    print(f"front shell: shelves removed, posts added, volume {v0:.0f} -> {C.volume(s):.0f} mm3, solids {C.n_solids(s)}", flush=True)
    return s


def hip_case_sweep(body, st_mesh, margin=1.0):
    """Box envelopes of both hip pitch ST3025 cases over hip yaw -30..30 (5 deg), in world (torso) coordinates."""
    out = []
    for side in ("R", "L"):
        jn = f"{side}_hip_pitch"
        sp = dict(SP.joint_spec(side)[jn], joint=jn)
        fr = SP.frame_from_c047(body, sp)
        T0 = SP.place_st3025(st_mesh, fr, "horn", HIP_SHIFT)["T"]
        local = C.box((-20.0 - margin, -10.0 - margin, -24.5 - margin), (20.0 + margin, 10.0 + margin, 19.3 + margin))
        for yaw in range(-30, 31, 5):
            out.append(C.transform(local, body.link_T(f"{side}_yaw_link", {f"{side}_hip_yaw": float(yaw)}) @ T0))
    return out


def build_pelvis(manifest):
    """Stage-2 lower pelvis cradle around the re-laid cells (M-jxjse2), clear of both hip pitch ST3025 sweeps."""
    body = F.Body(manifest)
    st_mesh = SP.st3025_local()
    sweeps = hip_case_sweep(body, st_mesh, 1.0)
    cells_env = [C.cylinder((0.0, CELL_Y - CELL_L / 2 - 1.0, z), (0.0, CELL_Y + CELL_L / 2 + 1.0, z), CELL_R + 0.85) for z in CELL_Z]
    s = load_step(manifest_part(manifest, "combined__stage2_lower_pelvis_battery_cradle")["native_step"])
    v_r1 = C.volume(s)
    s = C.cut_many(s, cells_env + sweeps)
    print("cradle after keep-outs:", [round(C.volume(x)) for x in C.solids(s)], flush=True)
    # figure-8 sleeve (1.6 mm wall) around the stacked cells, open ends for the terminals
    r_out = CELL_R + 0.85 + 1.6
    y0, y1 = CELL_Y - 33.0, CELL_Y + 26.0  # rear end kept clear of the back shell floor
    prism = C.fuse(C.cylinder((0, y0, CELL_Z[0]), (0, y1, CELL_Z[0]), r_out), C.cylinder((0, y0, CELL_Z[1]), (0, y1, CELL_Z[1]), r_out),
                   C.box((-r_out, y0, CELL_Z[0]), (r_out, y1, CELL_Z[1])))
    # slots for the aluminium fin strips that sit directly on the cells (two segments per fin, see cell_meshes)
    slots = [C.box((-3.3, a - 0.2, z - 1e2 if k == 0 else z), (3.3, b + 0.2, z if k == 0 else z + 1e2))
             for k, z in enumerate(CELL_Z) for a, b in FIN_SEGMENTS]
    floor = C.box((-r_out - 1.0, CELL_Y + 20.0, 190.0), (r_out + 1.0, y1 + 1.0, CELL_Z[0] - CELL_R - 0.3))  # back shell floor rises: open the sleeve bottom there
    sleeve = C.cut_many(prism, cells_env + hip_case_sweep(body, st_mesh, 1.5) + slots + [floor], min_keep=0.02)
    # webs tying the sleeve to the cradle lid above and the cradle floor below, between the fin segments
    web_top = C.box((-2.0, WEB_Y[0], CELL_Z[1] + r_out - 1.0), (2.0, WEB_Y[1], 265.2))  # spine bottom is z 265.6
    web_bot = C.box((-2.0, WEB_Y[0], 195.0), (2.0, WEB_Y[1], CELL_Z[0] - r_out + 1.0))
    # front-shell tabs: the r1 lower front-shell mounts (x +-16, z 224) sat where the hip pitch ST3025 now sweeps; the
    # shell's lower front is re-attached by two M2 posts screwing into these tabs above the upper cell (FRONT_POSTS)
    tabs = [C.box((x - 3.0, FRONT_POST_Y[0] + 0.15, FRONT_POST_Z - 3.0), (x + 3.0, y0 + 2.0, FRONT_POST_Z + 2.5)) for x, _ in FRONT_POSTS]
    tab_holes = C.holes("y", [(x, FRONT_POST_Z) for x, _ in FRONT_POSTS], 0.85, FRONT_POST_Y[0] - 1.0, y0 + 2.5)
    s = C.fuse(s, sleeve, web_top, web_bot, *tabs)
    fins = [C.box((-3.3, a - 0.3, (z - CELL_R - 2.8 - 0.3) if k == 0 else (z + CELL_R - 0.3)),
                  (3.3, b + 0.3, (z - CELL_R + 0.3) if k == 0 else (z + CELL_R + 2.8 + 0.3)))
            for k, z in enumerate(CELL_Z) for a, b in FIN_SEGMENTS]  # fin strips sit on the cell barrels (cell_meshes)
    s = C.cut_many(s, tab_holes + fins)
    sols = sorted(((C.volume(x), x) for x in C.solids(s)), key=lambda t: -t[0])
    print("cradle pieces after sleeve:", [round(v) for v, _ in sols], flush=True)
    return s, v_r1, cells_env + sweeps


def cell_meshes():
    """Cells, top insulators (now 0.25 mm end discs) and fin strips (now along y, above / below the stack)."""
    out = {}
    for i, (name, z) in enumerate((("Honeycomb_Cell_08_Lower_Pelvis_L", CELL_Z[0]), ("Honeycomb_Cell_09_Lower_Pelvis_R", CELL_Z[1]))):
        c = trimesh.creation.cylinder(radius=CELL_R, height=CELL_L, sections=96)
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])); c.apply_translation([0, CELL_Y, z])
        out[name] = c
        ins = trimesh.creation.cylinder(radius=CELL_R - 0.3, height=0.25, sections=96)
        ins.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        ins.apply_translation([0, CELL_Y - CELL_L / 2 - 0.125, z]); out[name + "_top_insulator"] = ins
        fin_z = (z - CELL_R - 1.4) if i == 0 else (z + CELL_R + 1.4)   # flat strip resting on the cell barrel
        segs = []
        for a, b in FIN_SEGMENTS:
            f = trimesh.creation.box(extents=(6.0, b - a, 2.8)); f.apply_translation([0, 0.5 * (a + b), fin_z]); segs.append(f)
        out[name.replace("Cell", "Fin")] = trimesh.util.concatenate(segs)
    return out


def overlap_box(mesh_a, mesh_b, grow=0.3):
    """Bounding box of the exact mesh intersection (None when below 0.05 mm3)."""
    i = F.to_mf(mesh_a) ^ F.to_mf(mesh_b)
    if i.volume() < 0.05:
        return None
    v = np.asarray(i.to_mesh().vert_properties)[:, :3]
    return v.min(0) - grow, v.max(0) + grow


def fix_static_defects(manifest, new):
    """Static overlaps found by static_audit.py in r1 and fixed here (T-jrnxfn, owner: fix all overlaps)."""
    P = {p["name"]: p for p in new["parts"]}
    B = F.Body(new)
    log = []
    # (a) the B13 short TPU pad and the v19 sensor pocket pad are two versions of one sole pad (same bounds; the v19 one
    #     has the sensor pockets): keep v19
    new["parts"] = [p for p in new["parts"] if p["name"] not in ("R_B13_short_TPU_pad", "L_B13_short_TPU_pad")]
    log.append("dropped R/L_B13_short_TPU_pad (duplicate of R/L_sensor_pocket_pad_v19)")
    P = {p["name"]: p for p in new["parts"]}
    # (b) r1 placed the left hip-roll servo, horn and idler flipped front to back, while the left printed parts are
    #     mirrors of the right ones (flange x case 324 mm3, horn x thigh 76 mm3): use mirrors of the right servo parts
    old_l = {n: np.array(P[n]["bounds_mm"], float) for n in ("L_roll_servo_case_reference", "L_roll_upper_horn_reference", "L_roll_passive_idler_reference")}
    for base in ("roll_servo_case_reference", "roll_upper_horn_reference", "roll_passive_idler_reference"):
        r, l = P["R_" + base], P["L_" + base]
        s = load_step(r["native_step"])
        if r.get("native_step_offset_mm"):
            s = C.translate(s, r["native_step_offset_mm"])
        rec = save("L_" + base, C.mirror_x(s), "purchased")
        rec.pop("volume_mm3", None); l.update(rec); l.pop("native_step_offset_mm", None)
    for p in list(new["parts"]):
        if not (p["name"].startswith("L_") and F.is_fastener(p["name"], p)):
            continue
        c = np.mean(np.array(p["bounds_mm"], float), axis=0)
        if any(np.all(c > b[0] - 3.0) and np.all(c < b[1] + 3.0) for b in old_l.values()):
            twin = P.get("R_" + p["name"][2:])
            if twin and twin.get("mesh"):
                m = F.load_mesh(twin["mesh"]); m.vertices[:, 0] *= -1; m.invert()
                npz = PARTS / f"{p['name']}.npz"; np.savez_compressed(npz, vertices=m.vertices, faces=m.faces)
                p.update(mesh=str(npz.relative_to(ROOT)), mesh_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                         bounds_mm=[m.bounds[0].round(3).tolist(), m.bounds[1].round(3).tolist()], mirrored_from=twin["name"])
                p.pop("native_step_offset_mm", None)
            else:
                new["parts"].remove(p)
    jr = next(j for j in new["joints"] if j["name"] == "R_hip_roll"); jl = next(j for j in new["joints"] if j["name"] == "L_hip_roll")
    jl["origin_world_mm"] = [jl["origin_world_mm"][0], jr["origin_world_mm"][1], jl["origin_world_mm"][2]]
    log.append("left hip-roll servo, horn, idler and their screws mirrored from the right; L_hip_roll origin y = right's")
    # (c) the forearm connector overlaps the elbow case (101.8 mm3 per side): pocket it
    for s_ in ("R", "L"):
        fa, case = f"{s_}_forearm_elbow_to_wrist_combined", f"upper__{s_}_elbow_C047_case"
        bx = overlap_box(B.mesh(fa), B.mesh(case))
        if bx is not None:
            rec = save(fa, C.cut_many(load_step(P[fa]["native_step"]), [C.box(*bx)]), "printed")
            P[fa].update(rec); log.append(f"{fa}: elbow case pocket {np.round(bx[0], 1).tolist()}..{np.round(bx[1], 1).tolist()}")
    # (d) upper cells: the top insulators were modelled inside the cell tops and the spine reaches into cells 02-04 and
    #     06-07: sit each insulator on its cell, then pocket the spine around every cell + insulator stack (+0.15 mm)
    tools = []
    for p in new["parts"]:
        n = p["name"]
        if not (n.startswith("Honeycomb_Cell_0") and "Upper" in n and not n.endswith("insulator")):
            continue
        cell = B.mesh(n); ins_name = n + "_top_insulator"; ins = B.mesh(ins_name)
        dz = cell.bounds[1][2] - ins.bounds[0][2]
        if dz > 0.01:
            ins = ins.copy(); ins.apply_translation((0, 0, dz))
            npz = PARTS / f"{ins_name}.npz"; np.savez_compressed(npz, vertices=ins.vertices, faces=ins.faces)
            P[ins_name].update(mesh=str(npz.relative_to(ROOT)), mesh_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                               bounds_mm=[ins.bounds[0].round(3).tolist(), ins.bounds[1].round(3).tolist()])
        cx, cy = 0.5 * (cell.bounds[0][:2] + cell.bounds[1][:2]); r = 0.5 * (cell.bounds[1][0] - cell.bounds[0][0]) + 0.15
        tools.append(C.cylinder((cx, cy, cell.bounds[0][2] - 0.15), (cx, cy, ins.bounds[1][2] + 0.15), r))
    sp = "combined__stage1_center_torso_spine_chassis"
    s = load_step(P[sp]["native_step"]); v0 = C.volume(s)
    s = C.cut_many(s, tools)
    P[sp].update(save(sp, s, "printed")); log.append(f"{sp}: cell + insulator pockets, {v0 - C.volume(s):.1f} mm3 removed")
    # (e) the edited front shell touches the front power carrier (0.56 mm3)
    fs, st3 = "center_front_shell", "combined__stage3_front_power_electronics_carrier"
    bx = overlap_box(B.mesh(fs), B.mesh(st3))
    if bx is not None:
        P[fs].update(save(fs, C.cut_many(load_step(P[fs]["native_step"]), [C.box(*bx)]), "shell"))
        log.append(f"{fs}: clearance to {st3}")
    for line in log:
        print("fix:", line, flush=True)
    return log


RANGE_NOTCHES = [  # (target shell, tool part, joints swept, grid step deg): contacts inside one limb's own range (r1)
    ("R_fore_front_hard_cover", "R_upper_arm_hard_cover", ("R_elbow", "R_shoulder_yaw"), 5.0),
    ("L_fore_front_hard_cover", "L_upper_arm_hard_cover", ("L_elbow", "L_shoulder_yaw"), 5.0),
    ("head_neck_collar", "original_profile_head_skin_v23", ("neck_pitch",), 2.5),
    ("head_neck_collar", "combined__neck_pitch_to_head_cradle", ("neck_pitch",), 2.5),
    ("original_profile_head_skin_v23", "combined__neck_yaw_to_pitch_link", ("neck_pitch",), 2.5),
    ("R_shin_front_hard_cover", "R_B13_C047_foot_roll_case", ("R_ankle_pitch", "R_ankle_roll"), 5.0),
    ("L_shin_front_hard_cover", "L_B13_C047_foot_roll_case", ("L_ankle_pitch", "L_ankle_roll"), 5.0),
]


def fix_range_contacts(new):
    """Notch the target shell's mesh by the tool's sweep (exact mesh booleans), over a grid of the given joints, at every
    pose where they overlap plus the neighbouring grid poses (one step of margin). Targets are shells; the forearm covers
    and the neck collar have no native STEP of their own in r1, so their mesh is the part."""
    import itertools
    import manifold3d as mf
    P = {p["name"]: p for p in new["parts"]}; J = {j["name"]: j for j in new["joints"]}
    B = F.Body(new); log = []; results = {}
    for target, tool, joints, step in RANGE_NOTCHES:
        grids = []
        for jn in joints:
            r = BOM_CONFIG.joint_range(jn)          # declared in the body definitions; undeclared raises (no invented range)
            grids.append(list(np.arange(r[0], r[1] + 1e-6, step)))
        res = results.get(target) or F.to_mf(B.mesh(target))
        tool_mf = F.to_mf(B.mesh(tool)); lt, lk = P[target]["owning_link"], P[tool]["owning_link"]
        hit_idx = []
        for idx in itertools.product(*[range(len(g)) for g in grids]):
            pose = {jn: float(grids[k][i]) for k, (jn, i) in enumerate(zip(joints, idx))}
            T = np.linalg.inv(B.link_T(lt, pose)) @ B.link_T(lk, pose)
            if (res ^ tool_mf.transform(T[:3, :4])).volume() > 0.05:
                hit_idx.append(idx)
        if not hit_idx:
            continue
        cut = set()
        for idx in hit_idx:  # dilate by one grid step in every swept joint
            for d in itertools.product(*[(-1, 0, 1)] * len(joints)):
                n = tuple(min(max(i + dd, 0), len(grids[k]) - 1) for k, (i, dd) in enumerate(zip(idx, d)))
                cut.add(n)
        v0 = res.volume()
        for idx in sorted(cut):
            pose = {jn: float(grids[k][i]) for k, (jn, i) in enumerate(zip(joints, idx))}
            T = np.linalg.inv(B.link_T(lt, pose)) @ B.link_T(lk, pose)
            res = res - tool_mf.transform(T[:3, :4])
        results[target] = res
        log.append(f"{target}: mesh notch by {tool} over {'+'.join(joints)} ({len(hit_idx)} overlapping poses, {len(cut)} cut), {v0 - res.volume():.0f} mm3 removed")
        print("fix:", log[-1], flush=True)
    for target, res in results.items():
        m = res.to_mesh(); v = np.asarray(m.vert_properties)[:, :3]; f = np.asarray(m.tri_verts)
        npz = PARTS / f"{target}.npz"; np.savez_compressed(npz, vertices=v, faces=f)
        P[target].update(mesh=str(npz.relative_to(ROOT)), mesh_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                         bounds_mm=[v.min(0).round(3).tolist(), v.max(0).round(3).tolist()],
                         mesh_note="range notch applied to the mesh (T-jrnxfn); the native STEP, where one exists, predates it")
    return log


def manifest_part(manifest, name):
    return next(p for p in manifest["parts"] if p["name"] == name)


def shifted(link):
    """Leg links below the hip pitch (exact names R_/L_ + DOWNSTREAM; not the shoulder links)."""
    return link[:2] in ("R_", "L_") and link[2:] in DOWNSTREAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["legs"])
    args = ap.parse_args()
    manifest = json.load(open(F.src(R1 / "assembly-parts.json")))
    new = copy.deepcopy(manifest)
    by_name = {p["name"]: p for p in new["parts"]}
    edited, servos, sleeve_spec = build_legs(manifest)

    records = {}
    for name, shape in edited.items():
        kind = "printed" if manifest_part(manifest, name)["role"].startswith("printed") else "shell"
        sols = C.solids(shape)
        if kind == "printed" and len(sols) > 1:
            keep = C.largest_solid(shape)
            print(f"{name}: dropped {len(sols) - 1} loose fragments, {C.volume(shape) - C.volume(keep):.1f} mm3", flush=True)
            shape = keep
        for side in ("R", "L"):
            nm = name.replace("R_", side + "_", 1)
            sh = shape if side == "R" else C.mirror_x(shape)
            link = by_name[nm]["owning_link"]
            if shifted(link):
                sh = C.translate(sh, ((HIP_SHIFT if side == "R" else -HIP_SHIFT), 0, 0))
            records[nm] = save(nm, sh, kind)
    for name, (shape, link) in servos.items():
        for side in ("R", "L"):
            nm = name.replace("R_", side + "_", 1); lk = link.replace("R_", side + "_", 1)
            sh = shape if side == "R" else C.mirror_x(shape)
            if shifted(lk) and "hip_pitch" not in nm:
                sh = C.translate(sh, ((HIP_SHIFT if side == "R" else -HIP_SHIFT), 0, 0))
            rec = save(nm, sh, "purchased")
            rec.update(name=nm, owning_link=lk, physical_part_id=nm, bom_group=nm,
                       role="motor" if nm.endswith("_case") else "output",
                       material="Waveshare ST3025 (Feetech ST-3025-C002) official STEP",
                       mass_kg=0.120 if nm.endswith("_case") else 0.004,
                       mass_basis="Catalogue 124 g per servo, split case 120 g / output wheels 4 g",
                       color=[38, 42, 48, 255] if nm.endswith("_case") else [150, 156, 166, 255],
                       display_group="mechanism", display_groups=["mechanism"], parent_assembly=by_name[f"{side}_shin_combined"]["parent_assembly"],
                       inspection_reference=False, source="Locked three-tier set M-44khpr, fast-leg tier")
            records[nm] = rec

    # ---- pelvis: cradle around the re-laid cells, cells / insulators / fins moved ----
    cradle, v_r1, cradle_cuts = build_pelvis(manifest)
    sleeve_spec["combined__stage2_lower_pelvis_battery_cradle"] = ([], cradle_cuts)
    pieces = sorted(C.solids(cradle), key=lambda x: -C.volume(x))
    loose = [C.volume(x) for x in pieces[1:]]
    if loose:
        print(f"cradle: {len(loose)} loose pieces dropped, {sum(loose):.1f} mm3 ({[round(v) for v in loose]})", flush=True)
    records["combined__stage2_lower_pelvis_battery_cradle"] = save("combined__stage2_lower_pelvis_battery_cradle", pieces[0], "printed")
    print(f"cradle volume r1 {v_r1:.0f} -> {C.volume(pieces[0]):.0f} mm3", flush=True)
    records["center_front_shell"] = save("center_front_shell", edit_front_shell(manifest), "shell")
    for nm, tm in cell_meshes().items():
        npz = PARTS / f"{nm}.npz"; np.savez_compressed(npz, vertices=tm.vertices, faces=tm.faces)
        records[nm] = dict(mesh=str(npz.relative_to(ROOT)), mesh_units="mm", mesh_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                           bounds_mm=[tm.bounds[0].round(3).tolist(), tm.bounds[1].round(3).tolist()],
                           placement_note="M-jxjse2: lower-pelvis cells laid along y on the midline, stacked")

    # ---- steel sleeves follow their part: same stretch moves by sleeve centre; sleeves inside a removed region drop ----
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    for name, (moves, cuts) in sleeve_spec.items():
        sname = name + "_steel_sleeves"
        sol = C.solids(load_step(manifest_part(manifest, sname)["native_step"]))
        kept, dropped = [], 0
        for so in sol:
            g = GProp_GProps(); BRepGProp.VolumeProperties_s(so, g); c = g.CentreOfMass(); c = np.array([c.X(), c.Y(), c.Z()])
            if any(C.volume(C.common(so, t)) > 0.05 for t in cuts):
                dropped += 1; continue
            d = np.zeros(3)
            for axis, at, delta, region in moves:
                if region is not None and not (np.all(c > np.array(region[0])) and np.all(c < np.array(region[1]))):
                    continue
                k = C.AX[axis]
                if (delta < 0 and c[k] < at) or (delta > 0 and c[k] > at):
                    d[k] += delta
            kept.append(C.translate(so, d) if np.any(d) else so)
        print(f"{sname}: {len(kept)} kept, {dropped} dropped (inside a cut)", flush=True)
        comp = compound(kept)
        sides = ("R", "L") if name.startswith("R_") else (None,)
        for side in sides:
            nm = sname if side is None else sname.replace("R_", side + "_", 1)
            sh = comp if side in (None, "R") else C.mirror_x(comp)
            if side and shifted(by_name[nm]["owning_link"]):
                sh = C.translate(sh, ((HIP_SHIFT if side == "R" else -HIP_SHIFT), 0, 0))
            rec = save(nm, sh, "purchased"); rec["native_solid_count"] = len(kept)
            rec["mass_kg"] = round(by_name[nm].get("mass_kg", 0) * len(kept) / max(1, len(sol)), 6)
            records[nm] = rec

    # ---- manifest: drop old servo references, apply edits, shift the lower legs ----
    drop = {t.format(s=s) for t in OLD_SERVO_PARTS for s in ("R", "L")}
    # r1 screws that held the removed horns / idlers (fasteners on the same link, centred inside the old part +3 mm)
    old_boxes = [(by_name[n]["owning_link"], np.array(by_name[n]["bounds_mm"], float)) for n in drop if n in by_name and "case" not in n]
    for p in new["parts"]:
        if F.is_fastener(p["name"], p) and p.get("bounds_mm"):
            c = np.mean(np.array(p["bounds_mm"], float), axis=0)
            if any(p["owning_link"] == lk and np.all(c > b[0] - 3.0) and np.all(c < b[1] + 3.0) for lk, b in old_boxes):
                drop.add(p["name"])
    print(f"dropping {len(drop)} r1 servo parts and their horn screws", flush=True)
    parts = []
    for p in new["parts"]:
        if p["name"] in drop:
            continue
        if p["name"] in records:
            p.update(records[p["name"]]); p.pop("mesh_units", None); p["mesh_units"] = "mm"
        elif shifted(p["owning_link"]):
            # unedited parts of the lower leg: a moved mesh copy, so viewers and checks read world-neutral meshes;
            # the native STEP stays r1's, with the rigid offset recorded
            d = HIP_SHIFT if p["owning_link"].startswith("R_") else -HIP_SHIFT
            if p.get("mesh"):
                m = F.load_mesh(p["mesh"]); m.apply_translation((d, 0.0, 0.0))
                npz = PARTS / "shifted" / f"{p['name']}.npz"; npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(npz, vertices=m.vertices, faces=m.faces)
                p["mesh"] = str(npz.relative_to(ROOT)); p["mesh_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
            p["native_step_offset_mm"] = [d, 0.0, 0.0]
            p["bounds_mm"] = [[b[0] + d, b[1], b[2]] for b in p["bounds_mm"]]
        parts.append(p)
    for nm, rec in records.items():
        if nm not in by_name:
            parts.append(rec)
    new["parts"] = parts
    for j in new["joints"]:
        if j["name"][:2] in ("R_", "L_") and j["name"][2:] in SHIFTED_JOINTS:
            d = HIP_SHIFT if j["name"].startswith("R_") else -HIP_SHIFT
            j["origin_world_mm"] = [j["origin_world_mm"][0] + d] + list(j["origin_world_mm"][1:])
    # motor identity per case part (M-44khpr); STS3250 and C047 share the r1 case geometry
    tiers = {"{s}_roll_servo_case_reference": ("Feetech STS3250 (strong tier)", 0.0745),
             "upper__{s}_pitch_C047_case": ("Feetech STS3250 (strong tier)", 0.0745),
             "upper__{s}_roll_C047_case": ("Feetech STS3250 (strong tier)", 0.0745),
             "upper__{s}_yaw_C047_case": ("Feetech STS3250 (strong tier)", 0.0745),
             "upper__{s}_elbow_C047_case": ("Feetech STS3250 (strong tier)", 0.0745),
             "lower__{s}_yaw_servo_case_reference": ("Feetech ST3215-C047 12 V (light tier)", 0.055),
             "{s}_B13_C047_foot_roll_case": ("Feetech ST3215-C047 12 V (light tier)", 0.055)}
    for p in new["parts"]:
        for pat, (motor, mass) in tiers.items():
            if p["name"] in (pat.format(s="R"), pat.format(s="L")):
                p["motor"] = motor; p["mass_kg"] = mass
                p["mass_basis"] = "Locked motor map M-44khpr catalogue mass (robot/model/motor-map/motor-map.json)"
        if p["name"].endswith("_ST3025_case"):
            p["motor"] = "Waveshare ST3025 (fast-leg tier)"
    new["static_fixes"] = fix_static_defects(manifest, new)
    new["range_fixes"] = fix_range_contacts(new)
    new["study_id"] = "cove50-complete-fdm-body-motor-refit"
    new["source_commit_r1"] = "48c38ceda"
    new["motor_map"] = "robot/model/motor-map/motor-map.json"
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(new, open(OUT / "assembly-parts.json", "w"), indent=1)
    print("wrote", OUT / "assembly-parts.json", len(records), "new/edited part files")


if __name__ == "__main__":
    main()
