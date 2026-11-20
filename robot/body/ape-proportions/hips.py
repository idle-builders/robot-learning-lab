"""Narrower hips toward the sketch's V torso (owner, 2026-11-13/14; Idle T-xdcjn9).

  - The legs move LEG_IN mm inward below the hip-pitch servo. Each hip-pitch servo keeps its original place: it turns
    about its own long axis, so sliding it along that axis back to where it was leaves the joint axis unchanged. The
    pitch-to-roll connector's wheel plate carries the LEG_IN offset (a stretched mesh until the CAD pass); the
    yaw-to-pitch connector and the hip-yaw servos stay as selected.
  - The two lower-pelvis cells keep their original place between the hip-pitch servos.
  - The pelvis cradle loses what the narrower hips sweep through (hip yaw over its range, hip pitch over its range,
    CLEARANCE mm around it): its floor under the lower cell goes, so that cell is held at its ends in the CAD pass.
  - The legs splay outward in the standing pose: hip roll -SPLAY_DEG, ankle roll +SPLAY_DEG (a zero offset, no part),
    so the feet sit FOOT_IN mm inward of the old stance while the hips sit LEG_IN mm in.
"""
from __future__ import annotations

import itertools

import numpy as np
import trimesh

LEG_IN = 12.0
FOOT_IN = 4.0
SPLAY_DEG = 2.68            # 171 mm hip-roll axis to ankle: sin(2.68 deg) * 171 = 8 mm outward at the foot
SERVO_OUT = 12.0          # owner 2026-11-14: the hip-pitch servos slide back to their original place along their own axis
STEP_Z = (250.0, 262.0)     # hip-yaw-link fasteners below STEP_Z[0] belong to the hip-pitch servo and move with it
MOVED_LINKS = ("pitch_link", "roll_link", "shin", "ankle_pitch", "foot")
MOVED_JOINTS = ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
CLEARANCE = 1.5
CRADLE_TOOL_FACES = 2500   # swept hip parts are decimated to this many faces before the cradle cut (the selected build's value)
CRADLE_BATCH = 150         # swept copies per subtraction; each batch is evaluated before the next (see cut_cradle)


_SLEEVES = {}


def apply(new, h):
    C, F = h["C"], h["F"]
    log = h["log"]
    P = {p["name"]: p for p in new["parts"]}
    moved = {f"{s}_{lk}" for s in "RL" for lk in MOVED_LINKS}
    for s in "RL":
        dx = -LEG_IN if s == "R" else LEG_IN
        yaw_link = f"{s}_yaw_link"
        for p in new["parts"]:
            lk = p["owning_link"]
            name = p["name"]
            if (lk[:2] == s + "_" and lk in moved) or (lk == yaw_link and name == f"{s}_hip_pitch_ST3025_case") or \
                    (lk == yaw_link and F.is_fastener(name, p) and np.mean(np.array(p["bounds_mm"])[:, 2]) < STEP_Z[0]):
                extra = 0.0
                if name == f"{s}_hip_pitch_ST3025_case" or name == f"{s}_hip_pitch_ST3025_output" \
                        or name.startswith(f"{s}_A13_pitch_horn_") or (lk == yaw_link and F.is_fastener(name, p)):
                    extra = -dx / LEG_IN * SERVO_OUT  # back outward along the pitch axis: the joint axis stays put
                h["move"](p, np.array([dx + extra, 0.0, 0.0])); log.append((name, f"hip offset {dx + extra:+.0f} mm"))
        # the yaw-to-pitch connector stays as selected (exact STEP kept): the hip-pitch servo keeps its original place
        P[f"{s}_hip_yaw_to_pitch_combined"]["mesh_note"] = "unchanged from the motor refit: the hip-pitch servo keeps its original place"
        # the pitch-to-roll connector's wheel plate follows the servo outward, blended to nothing at the roll-servo mount
        out = -dx / LEG_IN * SERVO_OUT; sg = 1.0 if s == "R" else -1.0
        pr = P[f"{s}_hip_pitch_to_roll_combined"]
        m = F.load_mesh(pr["mesh"])
        if pr.get("placement_offset_mm"):
            m.apply_translation(pr.pop("placement_offset_mm"))
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=0.8); v = np.asarray(v, float)
        jo = np.array([j for j in new["joints"] if j["name"] == f"{s}_hip_pitch"][0]["origin_world_mm"]) + [dx, 0, 0]
        wx = np.clip((sg * (v[:, 0] - jo[0]) - 1.0) / 2.0, 0.0, 1.0)  # outboard of the old wheel face (axis +2..+4 mm)
        wz = np.clip((v[:, 2] - 200.0) / 12.0, 0.0, 1.0)              # above the roll-servo mount
        before = trimesh.Trimesh(v.copy(), f, process=False)
        v[:, 0] += out * wx * wz  # x-dependent: stretches the 2 mm band outboard of the old wheel face to 2 + |out| mm
        mm = trimesh.Trimesh(v, f, process=True)
        pr.update(h["save_mesh"](pr["name"], mm))
        # the selected mesh is open (not watertight), so scale the native STEP volume by the mesh's signed-volume ratio
        # on the same topology (the open edges are the same before and after)
        ratio = float(trimesh.Trimesh(v, f, process=False).volume / before.volume)
        v_step = float(pr["volume_mm3"])
        pr["volume_mm3"] = round(v_step * ratio, 1); pr["mass_kg"] = round(v_step * ratio * 1.22e-6, 6)
        pr["mass_basis"] = (f"native STEP volume {v_step:.0f} mm3 x {ratio:.4f} (signed-volume ratio of the open mesh before and "
                            "after the wheel-plate stretch) * 1.22 g/cm3 (PAHT-CF); exact in the CAD pass")
        pr["native_step_unmodified"] = pr.pop("native_step", None); pr.pop("native_step_sha256", None); pr.pop("native_step_offset_mm", None)
        pr["mesh_note"] = f"wheel plate stays at the hip-pitch servo's original place ({out:+.0f} mm relative to the {LEG_IN:.0f} mm inward leg), stretched back to the roll-servo mount (mesh until the CAD pass)"
        log.append((pr["name"], f"wheel plate {out:+.0f} mm"))
        sl2 = P.get(pr["name"] + "_steel_sleeves")
        if sl2:
            from OCP.BRep import BRep_Builder
            from OCP.TopoDS import TopoDS_Compound
            comp = TopoDS_Compound(); b = BRep_Builder(); b.MakeCompound(comp)
            if s == "L" and "R" in _SLEEVES:  # the hips are mirror images: the left sleeves mirror the right ones
                comp = C.mirror_x(_SLEEVES["R"])
            else:
                for so in C.solids(h["load_step"](sl2)):  # load_step already applies the recorded offset
                    c = h["centroid"](so)
                    b.Add(comp, C.translate(so, (out, 0, 0)) if (sg * (c[0] - jo[0]) > 2.0 and c[2] > 205.0) else so)
                _SLEEVES[s] = comp
            sl2.pop("native_step_offset_mm", None)
            sl2.update(h["save_step"](sl2["name"], comp, "purchased"))
        for j in new["joints"]:
            if j["name"] in {f"{s}_{n}" for n in MOVED_JOINTS}:
                j["origin_world_mm"] = (np.array(j["origin_world_mm"], float) + [dx, 0, 0]).round(4).tolist()

    # pelvis cradle: clear what the narrower hips sweep through
    body = F.Body(new)
    cradle = P["combined__stage2_lower_pelvis_battery_cradle"]
    cm = F.load_mesh(cradle["mesh"])
    keep, parts = cut_cradle(body, F, cm)
    v0, v1 = cm.volume, keep.volume
    cradle.update(h["save_mesh"](cradle["name"], keep))
    cradle["native_step_unmodified"] = cradle.pop("native_step", None); cradle.pop("native_step_sha256", None)
    cradle["volume_mm3"] = round(float(v1), 1); cradle["mass_kg"] = round(float(v1) * 1.22e-6, 6)
    cradle["mesh_note"] = "selected cradle minus the swept narrower hips; the floor under the lower cell is gone, hold that cell at its ends (mesh until the CAD pass)"
    log.append((cradle["name"], f"cleared for the narrower hips: volume {v0:.0f} -> {v1:.0f} mm3, {len(parts)} pieces"))

    new["standing_pose_deg"] = {f"{s}_hip_roll": -SPLAY_DEG for s in "RL"} | {f"{s}_ankle_roll": SPLAY_DEG for s in "RL"}
    new["hips"] = dict(leg_in_mm=LEG_IN, foot_in_mm=FOOT_IN, splay_deg=SPLAY_DEG, owner="Idle T-xdcjn9, 2026-11-13/14",
                       note="the standing pose splays the legs; joint zeros stay straight")
    return [f"cradle volume {v0:.0f} -> {v1:.0f} mm3 ({len(parts)} pieces)"]


def cut_cradle(body, F, cm, faces=CRADLE_TOOL_FACES, batch_n=CRADLE_BATCH, stats=None):
    """The cradle mesh cm minus the hip parts swept over hip yaw x pitch, each enlarged about CLEARANCE mm.

    manifold3d records booleans lazily: without num_tri() after each batch, the whole tree of ~3,500 swept copies is
    evaluated in the final to_mesh(), one C++ call that never returns to Python (Idle T-be4dt7: 2026-11-20 chain rerun
    past 3600 s, no progress line, out of reach of the gmsh faulthandler limit). Evaluating each batch keeps memory and
    time bounded (on the selected body: 3,538 copies in 24 batches, 185 s, 4.9 GB) and prints progress.
    Returns (largest piece, all pieces)."""
    import time
    import manifold3d as mf

    def to_mf(m):
        return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    res = to_mf(cm); batch = []; ntools = 0; t0 = time.time(); st = stats if stats is not None else {}
    st.setdefault("batches", [])

    def flush():
        nonlocal res, batch
        if batch:
            tb = time.time()
            res = res - mf.Manifold.batch_boolean(batch, mf.OpType.Add)
            st["batches"].append(dict(tools=len(batch), tri=res.num_tri(), s=round(time.time() - tb, 2)))  # evaluates now
            print(f"cradle cut: batch {len(st['batches'])}, {ntools} swept copies, {st['batches'][-1]['tri']} triangles, "
                  f"{time.time() - t0:.0f} s", flush=True)
            batch = []
    cb = cm.bounds
    for s in "RL":
        sweeps = [(f"{s}_yaw_link", [(y, 0.0) for y in np.arange(-45.0, 45.1, 2.5)])]
        for lk in (f"{s}_pitch_link", f"{s}_roll_link"):
            sweeps.append((lk, list(itertools.product(np.arange(-45.0, 45.1, 7.5), np.arange(-65.0, 45.1, 5.0)))))
        for lk, grid in sweeps:
            names = [n for n, p in body.parts.items() if p["owning_link"] == lk and not F.is_fastener(n, p)
                     and p.get("role") not in ("cut_material", "adhesive_allocation")]
            base = []
            for n in names:
                m = body.mesh(n)
                if len(m.faces) > faces:
                    m = m.simplify_quadric_decimation(face_count=faces)
                c = m.centroid; e = np.ptp(m.vertices, axis=0); k = 1.0 + 2.0 * CLEARANCE / max(e.min(), 4.0)
                m = m.copy(); m.vertices = c + (m.vertices - c) * k  # about CLEARANCE mm bigger on its thinnest side
                t = to_mf(m)
                if t.status() == mf.Error.NoError and t.volume() > 0:
                    base.append((t, m.bounds))
            for yaw, pitch in grid:  # each swept copy that reaches the cradle's box goes into a batch
                T = body.link_T(lk, {f"{s}_hip_yaw": yaw, f"{s}_hip_pitch": pitch})
                for t, bnd in base:
                    w = trimesh.transform_points(bnd, T)
                    if np.any(w.min(0) > cb[1]) or np.any(w.max(0) < cb[0]):
                        continue
                    batch.append(t.transform(T[:3, :4])); ntools += 1
                    if len(batch) >= batch_n:
                        flush()
    flush()
    st["tools"] = ntools; st["build_s"] = round(time.time() - t0, 2)
    tm = time.time()
    out = res.to_mesh(); cm2 = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
    st["to_mesh_s"] = round(time.time() - tm, 2)
    parts = cm2.split(only_watertight=False)
    keep = max(parts, key=lambda q: q.volume if q.is_watertight else q.area)
    return keep, parts
