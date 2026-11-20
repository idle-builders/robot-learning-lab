"""Upper-arm cap trimmed by the torso shells it meets over the shoulder range (Idle T-czv2d8).

After the elbow-gap change the upper-arm cover's top half ('cap', R/L_upper_arm_hard_cover on the shoulder-roll link)
touches both torso shells at shoulder roll -12.5 and +100 deg (about 36 mm3, ~1 mm deep), which cut the measured roll
range from -15..120 to -10..97.5. The cap loses what the torso shells sweep through in the cap's own frame over
shoulder roll ROLL x shoulder pitch PITCH, with about CLEARANCE mm. The torso shells are first cropped to the region
the cap sweeps, so the clearance (a scale about the crop's centroid) stays close to CLEARANCE. Right side cut, left
side mirrored in x (the arm is mirror-symmetric within 0.03 mm). Mesh-defined until the CAD pass.

  python3 infra/tools/memguard.py --limit-gb 4 -- .venv/bin/python -B robot/body/ape-proportions/cap_trim.py MANIFEST OUT_DIR
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import trimesh

CLEARANCE = 1.5
ROLL = np.arange(-15.0, 120.1, 5.0)
PITCH = np.arange(-90.0, 180.1, 22.5)
CAP = "R_upper_arm_hard_cover"
TORSO = ("center_front_shell", "center_back_shell")
MARGIN = 6.0
TORSO_FACES = 20000


def _mf(m):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


def _tm(x):
    r = x.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=True)


def poses():
    return [{"R_shoulder_roll": r, "R_shoulder_pitch": p} for p, r in itertools.product(PITCH, ROLL)]


def torso_in_cap_frame(body, q):
    lk = body.parts[CAP]["owning_link"]
    return np.linalg.inv(body.link_T(lk, q)) @ body.link_T(body.parts[TORSO[0]]["owning_link"], q)


def trim(body, F):
    """The right cap minus the swept, cropped, enlarged torso shells; returns (mesh, notes, residual check)."""
    import manifold3d as mf
    cap = body.mesh(CAP)
    if not cap.is_watertight:
        raise ValueError(f"{CAP} is not watertight")
    Q = poses()
    Ts = [torso_in_cap_frame(body, q) for q in Q]
    # crop region: the cap's bounds, grown by MARGIN, seen in the torso frame at every pose
    lo, hi = cap.bounds[0] - MARGIN, cap.bounds[1] + MARGIN
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    res = _mf(cap); batch = []; ntools = 0
    for s in TORSO:
        tm = body.mesh(s)
        if len(tm.faces) > TORSO_FACES:
            tm = tm.simplify_quadric_decimation(face_count=TORSO_FACES)
        sm = _mf(tm)
        for T in Ts:
            w = trimesh.transform_points(corners, np.linalg.inv(T))
            box = mf.Manifold.cube(tuple(w.max(0) - w.min(0))).translate(tuple(w.min(0)))
            crop = sm ^ box
            if crop.is_empty() or crop.volume() < 1.0:
                continue
            cm = _tm(crop)
            c = cm.centroid; e = np.ptp(cm.vertices, axis=0)
            cm.vertices = c + (cm.vertices - c) * (1.0 + 2.0 * CLEARANCE / max(e.min(), 4.0))
            batch.append(_mf(cm).transform(T[:3, :4])); ntools += 1
            if len(batch) >= 100:
                res = res - mf.Manifold.batch_boolean(batch, mf.OpType.Add); batch = []
                res.num_tri()  # evaluate now: manifold3d is lazy, the whole tree would otherwise run in one call at the end (hips.cut_cradle)
    if batch:
        res = res - mf.Manifold.batch_boolean(batch, mf.OpType.Add)
    out = _tm(res)
    big = [q for q in out.split(only_watertight=False) if (q.volume if q.is_watertight else 0) > 20.0]
    out = trimesh.util.concatenate(big) if big else out
    # residual: un-enlarged torso shells against the trimmed cap at every pose
    rc = _mf(out); worst = (0.0, None)
    shells = [_mf(body.mesh(s)) for s in TORSO]
    for q, T in zip(Q, Ts):
        for sm in shells:
            v = (rc ^ sm.transform(T[:3, :4])).volume()
            if v > worst[0]:
                worst = (v, q)
    removed = cap.volume - out.volume
    return out, dict(cap_volume_mm3=round(cap.volume, 1), trimmed_volume_mm3=round(out.volume, 1),
                     removed_mm3=round(removed, 1), tools=ntools, residual_overlap_mm3=round(worst[0], 2),
                     residual_pose=worst[1])


def mirror_x(m):
    v = m.vertices.copy(); v[:, 0] *= -1
    return trimesh.Trimesh(v, m.faces[:, ::-1], process=True)


def apply(new, h):
    F = h["F"]; body = F.Body(new)
    P = {p["name"]: p for p in new["parts"]}
    out, info = trim(body, F)
    for name, m in ((CAP, out), ("L" + CAP[1:], mirror_x(out))):
        P[name].update(h["save_mesh"](name, m))
        P[name].pop("native_step", None); P[name].pop("native_step_sha256", None)
        P[name]["mesh_note"] = (f"cap trimmed by the torso shells swept over shoulder roll {ROLL[0]:.0f}..{ROLL[-1]:.0f} "
                                f"and pitch {PITCH[0]:.0f}..{PITCH[-1]:.0f} deg, {CLEARANCE} mm (cap_trim.py)")
        h["log"].append((name, "trimmed by the swept torso shells"))
    new.setdefault("shell_pass", {})["cap_trim"] = info
    return [f"upper-arm cap: {info}"]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "motor-refit"))
    import json
    import fit_check as F
    man = json.load(open(sys.argv[1])); outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    m, info = trim(F.Body(man), F)
    m.export(outdir / f"{CAP}.stl"); print(json.dumps(info, indent=1))
