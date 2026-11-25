"""Ape proportions on the motor-refit body: a longer arm and a larger foot, as exact B-rep edits.

Owner, 2026-11-13 (Idle T-gqwdrn): shoulder pivot to fingertip 267 mm, knuckles at about knee height standing
(upper arm x1.57 shoulder-yaw axis to elbow axis, forearm and gripper unchanged), feet x1.3 in length and width about the ankle-roll frame.

Method (robot/body/motor-refit/cad_ops.stretch): a part that crosses a cut plane is split there, the far side moves by
the stretch, and the gap is filled with the part's own cross-section at the plane, extruded. Joints, servo pockets,
horns, screw bosses and sleeves keep their exact shape; only the plain section between them gets longer. Parts wholly
beyond a plane move rigidly with it. Shell covers (meshes) stretch the same way on their vertices.

  Arm (both sides, the hanging arm runs along world -z):
    - forearm: plane z = 195 mm (between the elbow-case screws at z >= 217 and the wrist shell mounts at z <= 190),
      forearm links and both fingers;
    - upper arm: plane z = 270 mm (the single strap of the down-first Stage 3 connector), the upper-arm link and
      everything below it; the upper-arm hard cover (on the shoulder-roll link) stretches at z = 290 mm by the same
      amount, so the gap between the upper-arm and forearm covers stays as designed.
    The elbow moves straight down, so the upper arm hangs a little more vertically than in the r1 CAD.
  Foot (per side, about the ankle-roll joint origin, forward is -y):
    toe plane 16.5 mm ahead of the roll axis, heel plane 37.5 mm behind it, lateral plane 16 mm out, medial plane
    36.5 mm in: clear of the ankle-roll servo case, its screws and the monocoque sleeves. Each side grows by 0.3 x the
    monocoque's extent from the roll origin on that side, i.e. what training's scale_feet did to the collision sole.
    The boot shell takes the full stretch below z = 14 mm and none at its ankle collar (z >= 30 mm), blended between.

  .venv/bin/python -B robot/body/ape-proportions/build.py
Writes artifacts/robot/body/ape-proportions/ (assembly-parts.json, parts/, report.json).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFIT = HERE.parent / "motor-refit"
sys.path.insert(0, str(REFIT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402  (declared values: materials, joint ranges)
import cad_ops as C  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body"))
from checks import bodylib as F  # noqa: E402  (body model: Body, load_mesh, src, ...)

SRC = ROOT / "artifacts/robot/body/motor-refit/assembly-parts.json"
OUT = Path(os.environ.get("APE_OUT", ROOT / "artifacts/robot/body/ape-proportions"))  # APE_OUT: a variant build elsewhere
PARTS = OUT / "parts"

UPPER_ARM_SCALE = float(os.environ.get("APE_UPPER", 1.570))  # owner 2026-11-13; env only for comparison variants
FOREARM_SCALE = float(os.environ.get("APE_FOREARM", 1.0))
CFG = BOM_CONFIG.section("ape-proportions-build")   # configured values: robot/body/ape-proportions/build.bom.yaml config:
FOOT_SCALE = CFG["FOOT_SCALE"]
Z_FOREARM_CUT = CFG["Z_FOREARM_CUT"]
Z_UPPER_CUT = CFG["Z_UPPER_CUT"]
Z_UPPER_COVER_CUT = CFG["Z_UPPER_COVER_CUT"]
FOOT_PLANES = dict(toe=16.5, heel=37.5, lateral=16.0, medial=36.5)  # mm from the ankle-roll origin
BOOT_BLEND_Z = CFG["BOOT_BLEND_Z"]
# Upper-arm beam: the stage 3 connector is a 3 x 6.5 mm bar between the yaw and elbow horns; bending across its 3 mm
# side carried the 4 kg support case at ~160 MPa (fem_arm.py, retired 2026-11-25). The longer arm leaves free space between the elbow
# servo and the yaw horn, so the bar becomes a solid beam there (right side; the left side is its mirror in x).
SHELL_PASS = bool(int(os.environ.get("APE_SHELL_PASS", "0")))  # T-czv2d8 runs as its own step: shells.py MANIFEST OUT_DIR (memory)
NARROW_HIPS = CFG["NARROW_HIPS"]
ELBOW_FLIP = CFG["ELBOW_FLIP"]
UPPER_BEAM = (dict(x=(110.2, 126.0), y=(-6.0, 6.0), z_above_elbow_axis=36.7, z_top=283.5) if ELBOW_FLIP else  # above the turned case
              dict(x=(110.2, 119.0), y=(-6.0, 6.0), z_above_elbow_axis=11.7, z_top=283.5))

STRETCH_ROLES = CFG["STRETCH_ROLES"]
SLEEVES = CFG["SLEEVES"]


class Op:
    """Move what lies beyond `axis = at` (on delta's side) by delta; parts crossing the plane stretch."""

    def __init__(self, axis, at, delta, links=(), names=(), blend=None, label=""):
        self.axis, self.at, self.delta = axis, float(at), float(delta)
        self.k = C.AX[axis]
        self.links, self.names, self.blend, self.label = set(links), set(names), blend, label

    def applies(self, p):
        return p["owning_link"] in self.links or p["name"] in self.names

    def beyond(self, x):
        return x < self.at if self.delta < 0 else x > self.at

    def where(self, bounds):
        lo, hi = bounds[0][self.k], bounds[1][self.k]
        if self.beyond(lo) and self.beyond(hi):
            return "beyond"
        if not self.beyond(lo) and not self.beyond(hi):
            return "near"
        return "cross"

    def vec(self):
        d = np.zeros(3); d[self.k] = self.delta
        return d


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).relative_to(ROOT))


def load_step(p):
    from build123d import import_step
    s = import_step(str(F.src(p["native_step"]))).wrapped
    off = p.get("native_step_offset_mm")
    return C.translate(s, off) if off else s


def save_step(name, shape, kind):
    """STEP plus a closed surface mesh (same rules as the motor refit)."""
    from build123d import Solid, Compound, export_step
    PARTS.mkdir(parents=True, exist_ok=True)
    sols = C.solids(shape)
    obj = Solid(sols[0]) if len(sols) == 1 else Compound(shape)
    step = PARTS / f"{name}.step"
    export_step(obj, str(step))
    tm = None
    try:
        from surface_mesh import step_to_closed_mesh
        tm = step_to_closed_mesh(step, 1.2 if kind == "printed" else 1.6)
        if F.to_mf(tm) is None:
            tm = None
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: gmsh surface mesh failed ({str(exc)[:60]}); OCC tessellation used", flush=True)
    if tm is None:
        from build123d import export_stl
        stl = PARTS / f"{name}.stl"
        export_stl(obj, str(stl), tolerance=0.04, angular_tolerance=0.25)
        tm = trimesh.load(stl, force="mesh", process=True); stl.unlink()
    npz = PARTS / f"{name}.npz"
    np.savez_compressed(npz, vertices=np.asarray(tm.vertices, float), faces=np.asarray(tm.faces, np.int64))
    vol = C.volume(shape)
    rec = dict(native_step=rel(step), native_step_sha256=sha(step), mesh=rel(npz), mesh_sha256=sha(npz), mesh_units="mm",
               bounds_mm=[tm.bounds[0].round(3).tolist(), tm.bounds[1].round(3).tolist()],
               native_solid_count=len(sols), native_validity=[bool(C.valid(s)) for s in sols], volume_mm3=round(vol, 2))
    if kind == "printed":
        rec.update(BOM_CONFIG.mass_record(name, vol, builder="ape-proportions-build"))
    return rec


def move_part(p, d, sub="moved"):
    """Rigid translation of one part: a moved mesh copy, the STEP keeps its file with the offset recorded."""
    d = np.asarray(d, float)
    if p.get("mesh"):
        m = F.load_mesh(p["mesh"])
        if p.get("placement_offset_mm"):
            m.apply_translation(p.pop("placement_offset_mm"))
        m.apply_translation(d)
        p.update(save_mesh(p["name"], m, sub))
    else:
        p["bounds_mm"] = (np.array(p["bounds_mm"], float) + d).round(3).tolist()
    if p.get("native_step"):
        p["native_step_offset_mm"] = (np.array(p.get("native_step_offset_mm") or [0, 0, 0], float) + d).round(4).tolist()


def transform_part(p, T, sub="moved"):
    """Rigid 4x4 transform of one part's mesh (a rotated STEP is regenerated in the CAD pass)."""
    if p.get("mesh"):
        m = F.load_mesh(p["mesh"])
        if p.get("placement_offset_mm"):
            m.apply_translation(p.pop("placement_offset_mm"))
        m.apply_transform(T)
        p.update(save_mesh(p["name"], m, sub))
    else:
        c = np.array(p["bounds_mm"], float); corners = np.array([[x, y, z] for x in c[:, 0] for y in c[:, 1] for z in c[:, 2]])
        w = trimesh.transform_points(corners, T); p["bounds_mm"] = [w.min(0).round(3).tolist(), w.max(0).round(3).tolist()]
    if p.get("native_step"):
        p["native_step_transform"] = np.asarray(T).round(6).tolist(); p.pop("native_step_offset_mm", None)


def save_mesh(name, m, sub=""):
    d = PARTS / sub if sub else PARTS
    d.mkdir(parents=True, exist_ok=True)
    npz = d / f"{name}.npz"
    np.savez_compressed(npz, vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, np.int64))
    return dict(mesh=rel(npz), mesh_sha256=sha(npz), mesh_units="mm",
                bounds_mm=[m.bounds[0].round(3).tolist(), m.bounds[1].round(3).tolist()])


def mesh_stretch(m, ops):
    v = np.asarray(m.vertices, float).copy()
    for op in ops:
        w = np.where(op.beyond(v[:, op.k]), 1.0, 0.0)
        if op.blend is not None:
            z0, z1 = op.blend
            t = np.clip((v[:, 2] - z0) / (z1 - z0), 0.0, 1.0)
            w *= 1.0 - (3 * t ** 2 - 2 * t ** 3)
        v[:, op.k] += w * op.delta
    out = trimesh.Trimesh(v, np.asarray(m.faces), process=False)
    return out


def brep_apply(shape, ops):
    for op in ops:
        lo, hi = _bounds(shape)
        state = op.where((lo, hi))
        if state == "cross":
            shape = C.stretch(shape, op.axis, op.at, op.delta)
        elif state == "beyond":
            shape = C.translate(shape, op.vec())
    return shape


def _bounds(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box(); BRepBndLib.Add_s(shape, b)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return np.array([x0, y0, z0]), np.array([x1, y1, z1])


def centroid(shape):
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(shape, g); c = g.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()])


def reinforce_upper_arm(shape, side):
    """Fuse the upper-arm beam onto the lengthened stage 3 connector (z measured on the moved elbow axis)."""
    sgn = 1.0 if side == "R" else -1.0
    z0 = ELBOW_Z[side] + UPPER_BEAM["z_above_elbow_axis"]
    xs = sorted(sgn * x for x in UPPER_BEAM["x"])
    beam = C.box((xs[0], UPPER_BEAM["y"][0], z0), (xs[1], UPPER_BEAM["y"][1], UPPER_BEAM["z_top"]))
    return C.fuse(shape, beam)


ELBOW_Z = {}

# Upper-arm shell section (owner 2026-11-13: upper and lower arm should look proportional in section; the cover was
# 48 wide x 68 deep all the way down, the forearm about 50 x 51). Below the shoulder servos the sleeve blends to a
# rounder 56 x 58 mm section (a step up from the forearm), keeping room for the roll-to-yaw connector (y to 28.5 mm,
# down to z 321) and the yaw case. Right-side numbers; the left side mirrors x.
UPPER_COVER_SECTION = dict(center_xy=(131.0, 2.0), width=56.0, depth=58.0, width_blend_z=(338.0, 318.0),
                           depth_blend_z=(320.0, 300.0), now_width=48.4, now_depth=68.0)  # depth only below the connector


def morph_upper_cover(m, side):
    """Scale the sleeve's section about its centre, fully below blend_z[1], not at all above blend_z[0]."""
    c = UPPER_COVER_SECTION
    sgn = 1.0 if side == "R" else -1.0
    cx, cy = sgn * c["center_xy"][0], c["center_xy"][1]
    sx, sy = c["width"] / c["now_width"], c["depth"] / c["now_depth"]
    v = np.asarray(m.vertices, float).copy()
    def blend(z0, z1):
        t = np.clip((z0 - v[:, 2]) / (z0 - z1), 0.0, 1.0)
        return 3 * t ** 2 - 2 * t ** 3
    tx, ty = blend(*c["width_blend_z"]), blend(*c["depth_blend_z"])
    v[:, 0] = cx + (v[:, 0] - cx) * (1 + tx * (sx - 1))
    v[:, 1] = cy + (v[:, 1] - cy) * (1 + ty * (sy - 1))
    return trimesh.Trimesh(v, np.asarray(m.faces), process=False)


def leg_shell_meshes():
    """Boot and sole shells: the leg scene's meshes, slimmed as in the assembly, moved by the refit's offset."""
    sys.path.insert(0, str(ROOT / "robot/body/assembly"))
    from assemble_complete_fdm_body import slim_foot_shell_mesh
    from infra.storage.blob_store import ensure
    scene = trimesh.load(ensure(ROOT / "artifacts/robot/body/leg/complete_leg_hip_combined/assembly-with-full-shell.glb"),
                         force="scene")
    out = {}
    for side in "RL":
        for kind in ("boot_shell", "sole_shell"):
            name = f"{side}_foot_{kind}"
            T, key = scene.graph[name]
            m = scene.geometry[key].copy(); m.apply_transform(T); m.apply_scale(1000.0)
            out[name] = slim_foot_shell_mesh(m, side)
    return out


def arm_ops(man, side):
    J = {j["name"]: j for j in man["joints"]}
    yaw = np.array(J[f"{side}_shoulder_yaw"]["origin_world_mm"], float)
    elbow = np.array(J[f"{side}_elbow"]["origin_world_mm"], float)
    hand = [f"{side}_forearm", f"{side}_right_finger", f"{side}_left_finger"]
    tip_z = min(p["bounds_mm"][0][2] for p in man["parts"] if p["owning_link"] in hand and p.get("mesh"))
    dx, dy, dz = elbow - yaw
    upper0 = float(np.linalg.norm(elbow - yaw))
    du = np.sqrt((UPPER_ARM_SCALE * upper0) ** 2 - dx * dx - dy * dy) + dz  # elbow straight down to the new length
    fore0 = float(elbow[2] - tip_z)
    df = (FOREARM_SCALE - 1.0) * fore0
    lower = [f"{side}_upper_arm"] + hand
    ops = [Op("z", Z_FOREARM_CUT, -df, links=hand, label=f"{side} forearm"),
           Op("z", Z_UPPER_CUT, -du, links=lower, label=f"{side} upper arm"),
           Op("z", Z_UPPER_COVER_CUT, -du, names=[f"{side}_upper_arm_hard_cover"], label=f"{side} upper-arm cover")]
    geo = dict(upper_arm_mm=[round(upper0, 2), round(float(np.linalg.norm(elbow - yaw + [0, 0, -du])), 2)],
               forearm_elbow_to_hand_tip_mm=[round(fore0, 2), round(fore0 + df, 2)],
               elbow_drop_mm=round(float(du), 2), hand_drop_mm=round(float(du + df), 2),
               hand_tip_z_mm=[round(float(tip_z), 2), round(float(tip_z - du - df), 2)])
    return ops, geo


def foot_ops(man, side):
    J = {j["name"]: j for j in man["joints"]}
    o = np.array(J[f"{side}_ankle_roll"]["origin_world_mm"], float)
    # The roll axis runs along y, so its origin's y is arbitrary (R -13.5, L -19 in the refit manifest). Both feet use
    # the right side's, which is where the case and sleeves put the planes clear on both sides.
    o[1] = J["R_ankle_roll"]["origin_world_mm"][1]
    mono = next(p for p in man["parts"] if p["name"] == f"{side}_foot_monocoque_combined")
    lo, hi = np.array(mono["bounds_mm"], float)
    sx = 1.0 if o[0] > 0 else -1.0  # lateral direction
    k = FOOT_SCALE - 1.0
    ext = dict(toe=o[1] - lo[1], heel=hi[1] - o[1],
               lateral=(hi[0] - o[0]) if sx > 0 else (o[0] - lo[0]), medial=(o[0] - lo[0]) if sx > 0 else (hi[0] - o[0]))
    links = [f"{side}_foot"]
    ops = []
    for key, axis, at, d in (("toe", "y", o[1] - FOOT_PLANES["toe"], -k * ext["toe"]),
                             ("heel", "y", o[1] + FOOT_PLANES["heel"], k * ext["heel"]),
                             ("lateral", "x", o[0] + sx * FOOT_PLANES["lateral"], sx * k * ext["lateral"]),
                             ("medial", "x", o[0] - sx * FOOT_PLANES["medial"], -sx * k * ext["medial"])):
        ops.append(Op(axis, at, d, links=links, label=f"{side} foot {key}"))
    geo = dict(monocoque_mm=[[round(hi[0] - lo[0], 2), round(hi[1] - lo[1], 2)],
                             [round(hi[0] - lo[0] + k * (ext["lateral"] + ext["medial"]), 2),
                              round(hi[1] - lo[1] + k * (ext["toe"] + ext["heel"]), 2)]],
               grow_mm={key: round(k * v, 2) for key, v in ext.items()})
    return ops, geo


def main():
    man = json.load(open(F.src(SRC)))
    new = copy.deepcopy(man)
    ops, geometry = [], {}
    for side in "RL":
        a, ga = arm_ops(man, side); f, gf = foot_ops(man, side)
        ELBOW_Z[side] = next(j for j in man["joints"] if j["name"] == f"{side}_elbow")["origin_world_mm"][2] - ga["elbow_drop_mm"]
        ops += a + f
        geometry[f"{side}_arm"] = ga; geometry[f"{side}_foot"] = gf
    shells = leg_shell_meshes()
    log, crossings = [], []
    for p in new["parts"]:
        mine = [op for op in ops if op.applies(p)]
        if not mine and p["name"] not in shells:
            continue
        if p["name"] in shells:
            m = shells[p["name"]]
            boot = "boot" in p["name"]
            sops = [Op(op.axis, op.at, op.delta, blend=BOOT_BLEND_Z if boot else None) for op in mine]
            p.update(save_mesh(p["name"], mesh_stretch(m, sops)))
            p.pop("native_step_offset_mm", None)
            p["mesh_note"] = "leg-scene shell, slimmed as in the assembly, refit offset, stretched to the x1.3 foot"
            log.append((p["name"], "shell mesh stretch"))
            continue
        b = np.array(p["bounds_mm"], float)
        states = [op.where(b) for op in mine]
        if all(s == "near" for s in states):
            continue
        role = p.get("role") or ""
        crossing = any(s == "cross" for s in states)
        if role == SLEEVES and p.get("native_step"):
            sols = C.solids(load_step(p))
            moved = []
            for so in sols:
                c = centroid(so); d = np.zeros(3)
                for op in mine:
                    if op.beyond(c[op.k]):
                        d += op.vec()
                    lo, hi = _bounds(so)
                    if op.where((lo, hi)) == "cross":
                        crossings.append(f"{p['name']}: a sleeve crosses {op.label} plane")
                moved.append(C.translate(so, d) if np.any(d) else so)
            from OCP.BRep import BRep_Builder
            from OCP.TopoDS import TopoDS_Compound
            comp = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(comp)
            for so in moved:
                bb.Add(comp, so)
            p.update(save_step(p["name"], comp, "purchased")); p.pop("native_step_offset_mm", None)
            log.append((p["name"], "sleeves moved by centre"))
        elif crossing and role.startswith(STRETCH_ROLES) and p.get("native_step"):
            shape = brep_apply(load_step(p), mine)
            kind = "printed" if role.startswith("printed") else "shell"
            if kind == "printed" and len(C.solids(shape)) > 1:
                keep = C.largest_solid(shape)
                print(f"{p['name']}: dropped {len(C.solids(shape)) - 1} loose fragments", flush=True)
                shape = keep
            v0 = p.get("volume_mm3")
            if p["name"].endswith("_upper_arm_yaw_to_elbow_combined") and not ELBOW_FLIP:  # flip: centre column instead
                shape = reinforce_upper_arm(shape, p["name"][0])
            p.update(save_step(p["name"], shape, kind)); p.pop("native_step_offset_mm", None)
            log.append((p["name"], f"B-rep stretch, volume {v0} -> {p['volume_mm3']} mm3"))
        elif crossing and role.startswith(STRETCH_ROLES) and p.get("mesh"):
            m = F.load_mesh(p["mesh"])
            if p.get("placement_offset_mm"):
                m.apply_translation(p.pop("placement_offset_mm"))
            v0 = m.volume if m.is_watertight else None
            m2 = mesh_stretch(m, mine)
            p.update(save_mesh(p["name"], m2))
            if v0:
                p["volume_mm3"] = round(float(m2.volume), 2)
            log.append((p["name"], "mesh stretch"))
        else:
            # rigid: wholly beyond, or a non-structural part (screw, sensor film, purchased) moved by its centre
            c = b.mean(0); d = np.zeros(3)
            for op, s in zip(mine, states):
                if s == "beyond" or (s == "cross" and op.beyond(c[op.k])):
                    d += op.vec()
                if s == "cross" and not role.startswith(("candidate_sensor", "produced_backing", "adhesive")):
                    crossings.append(f"{p['name']} ({role}) crosses {op.label} plane; moved by its centre")
            if not np.any(d):
                continue
            if p.get("mesh"):
                m = F.load_mesh(p["mesh"])
                if p.get("placement_offset_mm"):
                    m.apply_translation(p.pop("placement_offset_mm"))
                m.apply_translation(d)
                p.update(save_mesh(p["name"], m, "moved"))
            else:
                p["bounds_mm"] = (b + d).round(3).tolist()
            if p.get("native_step"):
                p["native_step_offset_mm"] = (np.array(p.get("native_step_offset_mm") or [0, 0, 0], float) + d).round(4).tolist()
            log.append((p["name"], f"moved {d.round(2).tolist()}"))
    # joints: an origin beyond an op's plane on the moving links moves with it
    for j in new["joints"]:
        for op in ops:
            if op.names:
                continue
            if j["child"] in op.links and op.beyond(j["origin_world_mm"][op.k]):
                j["origin_world_mm"] = (np.array(j["origin_world_mm"], float) + op.vec()).round(4).tolist()
    for p in new["parts"]:
        if p["name"].endswith("_upper_arm_hard_cover"):
            p.update(save_mesh(p["name"], morph_upper_cover(F.load_mesh(p["mesh"]), p["name"][0])))
            p["native_step_unmorphed"] = p.pop("native_step", None); p.pop("native_step_sha256", None)
            p["mesh_note"] = "stretched B-rep, then the sleeve section morphed (UPPER_COVER_SECTION); the cover is mesh-defined until the shell pass (T-czv2d8)"
            log.append((p["name"], "sleeve section morph"))
    if ELBOW_FLIP:
        import elbow_flip
        helpers = dict(C=C, F=F, load_step=load_step, save_step=save_step, save_mesh=save_mesh, centroid=centroid, log=log)
        for side in "RL":
            for nt in elbow_flip.flip(new, side, helpers):
                print(f"{side} elbow flip: {nt}", flush=True)
    if NARROW_HIPS:
        import hips
        helpers = dict(C=C, F=F, load_step=load_step, save_step=save_step, save_mesh=save_mesh, centroid=centroid, log=log,
                       move=move_part, transform=transform_part)
        for nt in hips.apply(new, helpers):
            print(f"hips: {nt}", flush=True)
    if SHELL_PASS:
        import shells
        for nt in shells.apply(new, dict(F=F, save_mesh=save_mesh, log=log)):
            print(f"shells: {nt}", flush=True)
    new.update(study_id="cove50-ape-proportions", result_id="cove50-ape-proportions", version=None,
               source_body=rel(SRC), source_body_sha256=sha(F.src(SRC)),
               ape_proportions=dict(upper_arm_scale=UPPER_ARM_SCALE, forearm_scale=FOREARM_SCALE, foot_scale=FOOT_SCALE,
                                    owner_decision="Idle T-gqwdrn, owner 2026-11-13", geometry=geometry))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(new, open(OUT / "assembly-parts.json", "w"), indent=1)
    report = dict(schema="cove50.ape-proportions.report.v1", idle_task="T-gqwdrn", geometry=geometry,
                  ops=[dict(label=o.label, axis=o.axis, at_mm=o.at, delta_mm=round(o.delta, 3)) for o in ops],
                  edits=[dict(part=n, edit=e) for n, e in log], crossings=crossings)
    json.dump(report, open(OUT / "report.json", "w"), indent=1)
    print(json.dumps(geometry, indent=1))
    print(f"{len(log)} parts edited; {len(crossings)} crossing notes")
    for c in crossings:
        print("  ", c)


if __name__ == "__main__":
    main()
