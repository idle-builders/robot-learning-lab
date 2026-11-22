"""Standard-part instances of a body manifest, registered to their official definitions (robot/actuators/interfaces).

Each servo case / horn / idler part of the manifest is matched to its motor's official STEP by rigid registration
(PCA start over all 48 axis permutations incl. mirror, then ICP). The result per instance: the transform, the RMS
distance between the instance surface and the official surface both ways (mm; a large value means the body uses a
different CAD of that servo), whether it is mirrored, and every defined mount hole in the body frame.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFS = ROOT / "robot/actuators/interfaces/standard-parts.json"
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "robot/body"))
from checks import cachefs  # noqa: E402
CACHE = cachefs.work() / "mount-check/canon"   # keyed by STEP / mesh sha256: shared with the worktrees

MOTOR_ROLES = {"motor", "actuator", "purchased_servo_case_family_reference", "output", "output_horn", "horn",
               "purchased_servo_output_family_reference"}


def family_of(name: str, part: dict, by_joint: dict) -> tuple[str, str] | None:
    """(motor key, member) for a servo part of the manifest, or None."""
    n = name
    if "ST3025" in n:
        return ("ST3025", "case" if n.endswith("_case") else "output")
    if "XM335" in n:
        return ("XM335", "case")
    if re.search(r"neck_(pitch|yaw)_front_horn", n):
        return ("XM335", "horn")
    if "hand_OEM" in n:
        return None
    member = "idler" if "idler" in n else "horn" if "horn" in n else "case" if "case" in n else None
    if member is None:
        return None
    motor = part.get("motor") or by_joint.get(joint_key(n), "")
    key = "STS3250" if "STS3250" in motor else "ST3215" if ("ST3215" in motor or "C047" in motor) else None
    return (key, member) if key else None


def joint_key(n: str) -> str:
    """One key per servo joint. The upper__ (shoulder/arm) and lower__/unprefixed (leg) parts keep their tier in the key:
    both chains have R_yaw and R_roll, with different servos (STS3250 shoulder yaw, ST3215 hip yaw)."""
    tier = "upper" if n.startswith("upper__") else "lower"
    n = re.sub(r"^(upper__|lower__)", "", n)
    return tier + ":" + re.sub(r"_(C047_)?(main_horn|upper_horn_reference|passive_idler_reference|upper_horn|rear_idler|passive_idler|servo_case_reference|case_reference|case|"
                  r"foot_roll_main_horn|foot_roll_rear_idler|foot_roll_case)$", "", n).replace("_C047", "").replace("_foot_roll", "")


def canonical(defs: dict, key: str, member: str):
    """Official surface points of one member (case = all solids not in a moving member; output = horn + idler)."""
    import trimesh
    m = defs["motors"][key]
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}-{member}-{m['step_sha256'][:12]}-{'-'.join(map(str, m['case_exterior']))}.npz"
    z = cachefs.load_npz(f) if f.exists() else None
    if z is not None:
        return trimesh.Trimesh(z["v"], z["f"], process=False)
    from build123d import import_step
    shape = import_step(str(ROOT / m["step"]))
    solids = shape.solids()
    moving = {i for v in m["members"].values() for i in v}
    if member == "case":
        idx = m["case_exterior"]          # outer shells only: internals differ between CAD sources
    elif member == "output":
        idx = [i for k in ("horn", "idler", "shaft") for i in m["members"].get(k, [])]
    else:
        idx = m["members"][member]
    vs, fs, off = [], [], 0
    for i in idx:
        v, t = solids[i].tessellate(0.05, 0.3)
        vs.append(np.array([(p.X, p.Y, p.Z) for p in v])); fs.append(np.array(t) + off); off += len(v)
    V, Fc = np.vstack(vs), np.vstack(fs)
    cachefs.savez(f, v=V, f=Fc)
    return trimesh.Trimesh(V, Fc, process=False)


def _kabsch(A, B, reflect):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    want = -1.0 if reflect else 1.0
    if np.sign(np.linalg.det(Vt.T @ U.T)) != want:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def trimesh_sample(m, n, seed):
    """Area-weighted surface samples (deterministic)."""
    rng = np.random.default_rng(seed)
    tri = m.triangles
    area = m.area_faces
    k = rng.choice(len(tri), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[k]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def _sample_with_normals(m, n, seed):
    rng = np.random.default_rng(seed)
    area = m.area_faces
    k = rng.choice(len(m.faces), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = m.triangles[k]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]), m.face_normals[k]


def _surface_refine(A, inst, R, t, reflect, iters=25, n_dense=120000, seed=7):
    """Point-to-plane ICP against dense instance samples with their face normals (distance along the normal is
    sub-sample accurate). A mirrored fit is a proper rotation of the reflected canonical points."""
    Bd, Nd = _sample_with_normals(inst, n_dense, seed)
    tree = cKDTree(Bd)
    M = np.diag([1.0, 1.0, -1.0]) if reflect else np.eye(3)
    R0 = R @ M                       # proper rotation acting on the reflected points
    X = A @ M.T
    for _ in range(iters):
        Y = X @ R0.T + t
        _, j = tree.query(Y)
        n_, q = Nd[j], Bd[j]
        r = np.einsum("ij,ij->i", Y - q, n_)
        J = np.c_[np.cross(Y, n_), n_]
        x, *_ = np.linalg.lstsq(J, -r, rcond=None)
        w, dt = x[:3], x[3:]
        th = np.linalg.norm(w)
        dR = np.eye(3) if th < 1e-12 else trimesh.transformations.rotation_matrix(th, w / th)[:3, :3]
        R0 = dR @ R0; t = dR @ t + dt
        if th < 1e-7 and np.linalg.norm(dt) < 1e-6:
            break
    Y = X @ R0.T + t
    _, j = tree.query(Y)
    d = np.abs(np.einsum("ij,ij->i", Y - Bd[j], Nd[j]))
    return R0 @ M, t, d


def register(canon, inst, n=2500, seed=0):
    """Transform canon -> inst. Coarse: PCA start over the 48 signed axis permutations + sample ICP; fine: the best
    proper rotation and the best mirrored fit are each refined point-to-surface. Returns T, distances (official surface
    samples -> instance surface, mm) of the chosen fit, mirrored flag, and both fits' RMS."""
    rng = np.random.default_rng(seed)
    A = trimesh_sample(canon, n, seed)
    B = trimesh_sample(inst, n * 2, seed + 1)
    tB = cKDTree(B)
    ca, cb = A.mean(0), B.mean(0)
    _, _, Va = np.linalg.svd(A - ca, full_matrices=False)
    _, _, Vb = np.linalg.svd(B - cb, full_matrices=False)
    best = {False: None, True: None}
    X = A[rng.choice(len(A), min(800, len(A)), replace=False)]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                P[i, j] = s
            R = Vb.T @ P @ Va                      # canon pca axes -> inst pca axes
            reflect = bool(np.linalg.det(R) < 0)
            t = cb - R @ ca
            for _ in range(15):
                d, j = tB.query(X @ R.T + t)
                R, t = _kabsch(X, B[j], reflect)
            d, _ = tB.query(X @ R.T + t)
            score = float(np.sqrt(np.mean(d ** 2)))
            if best[reflect] is None or score < best[reflect][0]:
                best[reflect] = (score, R, t)
    fits = {}
    for reflect, (_, R, t) in best.items():
        R, t, d = _surface_refine(A, inst, R, t, reflect)
        fits[reflect] = (float(np.sqrt(np.mean(d ** 2))), R, t, d)
    mirrored = fits[True][0] < fits[False][0] - 0.02      # mirrored only when clearly the better fit
    rms, R, t, d = fits[mirrored]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return dict(T=T, rms=rms, p95=float(np.percentile(d, 95)), max=float(d.max()), mirrored=bool(mirrored),
                rms_proper=fits[False][0], rms_mirrored=fits[True][0])


def _hole_centre(rays, o, ax, depth, reach=2.6, step=0.1):
    """Centre of the through-opening of one hole in the instance mesh near the nominal opening point o: a grid of rays
    parallel to the hole axis, the open ones (free for the defined depth) are clustered, the cluster holding the open
    point nearest o is the hole; its centroid is the centre. Returns (centre, n_open) or None."""
    u = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u); v = np.cross(ax, u)
    g = np.arange(-reach, reach + 1e-9, step)
    gu, gv = np.meshgrid(g, g); keep = gu ** 2 + gv ** 2 <= reach ** 2
    off = gu[keep][:, None] * u + gv[keep][:, None] * v
    org = o + off + 0.3 * ax
    loc, idx, _ = rays.intersects_location(org, np.tile(-ax, (len(org), 1)), multiple_hits=False)
    free = np.full(len(org), 0.3 + depth)
    if len(idx):
        free[idx] = np.minimum(free[idx], np.linalg.norm(loc - org[idx], axis=1))
    op = free >= 0.3 + depth - 1e-6
    if not op.any():
        return None
    P = off[op]
    lab = cKDTree(P).query_ball_point(P, step * 1.5)
    seen = -np.ones(len(P), int); n = 0
    for i in range(len(P)):
        if seen[i] >= 0:
            continue
        stack = [i]; seen[i] = n
        while stack:
            k = stack.pop()
            for m in lab[k]:
                if seen[m] < 0:
                    seen[m] = n; stack.append(m)
        n += 1
    near = int(np.argmin(np.linalg.norm(P, axis=1)))
    sel = seen == seen[near]
    return o + P[sel].mean(0), int(sel.sum())


def snap_rotation(defs, key, member, T, inst):
    """Horns and idlers are near-round discs: shape fitting cannot fix their turn about the output axis. Locate the
    centre of every defined hole in the instance mesh (_hole_centre), then turn the defined pattern about the output
    axis to the angle minimising the distances between defined and located centres (closed-form 2D Kabsch, repeated
    as the search points move). hole_snap reports the residual RMS of the centre distances (mm), the poor-fit score."""
    m = defs["motors"][key]
    mem = ("horn", "idler") if member == "output" else (member,)
    hs = [h for h in m["mount_holes"] if h["member"] in mem and not h.get("centre")]
    if not hs:
        return T, None
    p0, d0 = (np.array(v, float) for v in m["output_axis"])
    R, t = T[:3, :3], T[:3, 3]
    c, a = R @ p0 + t, R @ d0; a /= np.linalg.norm(a)
    O = np.array([R @ np.array(h["opening_mm"]) + t for h in hs]); A = np.array([R @ np.array(h["axis_out"]) for h in hs])
    D = np.array([min(h["modelled_depth_mm"], 3.0) for h in hs])
    try:
        rays = trimesh.ray.ray_pyembree.RayMeshIntersector(inst)
    except Exception:  # noqa: BLE001
        rays = trimesh.ray.ray_triangle.RayMeshIntersector(inst)
    perp = lambda x: x - np.outer(x @ a, a)
    th = 0.0
    for _ in range(4):
        Rz = trimesh.transformations.rotation_matrix(math.radians(th), a, c)[:3, :3]
        o = (O - c) @ Rz.T + c; ax = A @ Rz.T
        found = [(i, _hole_centre(rays, o[i], ax[i] / np.linalg.norm(ax[i]), D[i])) for i in range(len(hs))]
        found = [(i, f[0]) for i, f in found if f is not None]
        if not found:
            return T, dict(turn_deg=0.0, fit_rms_mm=None, holes_located=0, holes_defined=len(hs))
        I = [i for i, _ in found]
        nom = perp(O[I] - c); got = perp(np.array([f for _, f in found]) - c)
        th = math.degrees(math.atan2(float((np.cross(nom, got) @ a).sum()), float((nom * got).sum())))
    Rz = trimesh.transformations.rotation_matrix(math.radians(th), a, c)
    res = got - perp((O[I] - c) @ Rz[:3, :3].T)
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    return Rz @ T, dict(turn_deg=round(((th + 180) % 360) - 180, 3), fit_rms_mm=round(rms, 3),
                        holes_located=len(I), holes_defined=len(hs))


def cached_fit(defs, key, member, body, n):
    """register() of one instance, cached by the sha256 of the instance mesh file, its offset and the official STEP."""
    p = body.parts[n]
    src = ROOT / p["mesh"]
    h = hashlib.sha256(src.read_bytes() if src.exists() else p["mesh"].encode()).hexdigest()[:16]
    tag = hashlib.sha256(json.dumps([h, p.get("placement_offset_mm"), defs["motors"][key]["step_sha256"], member,
                                     defs["motors"][key]["case_exterior"], 7]).encode()).hexdigest()[:16]
    f = CACHE / "fits" / f"{tag}.json"
    z = cachefs.read_json(f) if f.exists() else None
    if z is not None:
        z["T"] = np.array(z["T"]); return z
    tries = ("output", "horn", "idler") if member == "output" else (member,)
    # STS3250 and ST3215 share one case frame and hole pattern; the body may model one with the other's CAD
    shapes = (key, "ST3215") if key == "STS3250" else (key, "STS3250") if key == "ST3215" else (key,)
    fits = []
    for mem in tries:            # an 'output' part may be the front wheel alone (no rear idler)
        for shp in shapes:
            fit = register(canonical(defs, shp, mem), body.mesh(n))
            fit["member_fitted"] = mem
            fit["modelled_with"] = shp
            fits.append(fit)
    fit = min(fits, key=lambda x: x["rms"])
    if fit["member_fitted"] in ("horn", "idler", "output"):
        fit["T"], fit["hole_snap"] = snap_rotation(defs, key, fit["member_fitted"], fit["T"], body.mesh(n))
    cachefs.write_text(f, json.dumps({**fit, "T": fit["T"].tolist()}))
    return fit


def instances(body, man) -> dict:
    defs = json.loads(DEFS.read_text())
    by_joint: dict = {}
    for n, p in body.parts.items():
        if p.get("motor"):
            if by_joint.setdefault(joint_key(n), p["motor"]) != p["motor"]:
                raise ValueError(f"joint key {joint_key(n)!r} maps to two servos ({by_joint[joint_key(n)]!r}, {p['motor']!r} at {n})")
    rows, holes, missing = [], [], []
    for n, p in body.parts.items():
        if p["role"] not in MOTOR_ROLES:
            continue
        fam = family_of(n, p, by_joint)
        if not fam:
            missing.append(dict(part=n, role=p["role"], reason="no official definition (unknown servo)"))
            continue
        key, member = fam
        fit = cached_fit(defs, key, member, body, n)
        T = fit["T"]; R, t = T[:3, :3], T[:3, 3]
        fitted = fit.get("member_fitted", member)
        members = ("horn", "idler") if fitted == "output" else (fitted,)
        k = 0
        for h in defs["motors"][key]["mount_holes"]:
            if h["member"] not in members:
                continue
            o = R @ np.array(h["opening_mm"]) + t
            a = R @ np.array(h["axis_out"]); a /= np.linalg.norm(a)
            holes.append(dict(hole_id=f"{n}:{h['hole_id']}", part=n, standard=key, label=h["hole_id"], member=h["member"],
                              thread=h["thread"], opening=o, axis_out=a, depth_mm=h["modelled_depth_mm"],
                              drawing_max_tap_depth_mm=h.get("drawing_max_tap_depth_mm"),
                              reference_engagement_mm=h["reference_engagement_mm"], terminal_material=h["terminal_material"],
                              centre=bool(h.get("centre"))))
            k += 1
        rows.append(dict(part=n, standard=key, member=fit.get("member_fitted", member), owning_link=p.get("owning_link"), mount_holes=k,
                         fit_rms_mm=round(fit["rms"], 3), fit_p95_mm=round(fit["p95"], 3), fit_max_mm=round(fit["max"], 3),
                         modelled_with=fit.get("modelled_with", key), mirrored=fit["mirrored"], hole_snap=fit.get("hole_snap"), fit_rms_proper_mm=round(fit["rms_proper"], 3),
                         fit_rms_mirrored_mm=round(fit["rms_mirrored"], 3), transform=np.round(T, 5).tolist()))
    return dict(instances=rows, holes=holes, summary=dict(instances=len(rows), holes=len(holes), undefined=missing))


def match_screw(holes, head, d, shank, r_major=1.0, max_offset=1.5, max_angle_deg=12.0):
    """The defined tapped hole a screw goes into: offset of the hole axis from the screw axis at the opening (mm),
    angle (deg), and engaged length: shank (from the head underside to the tip less a 0.5 mm point) inside the hole
    depth (modelled depth, capped by the drawing's maximum tap depth)."""
    best = None
    for h in holes:
        cosang = float(-h["axis_out"] @ d)
        if cosang < math.cos(math.radians(max_angle_deg)):
            continue
        rel = h["opening"] - head
        t_open = float(rel @ d)
        off = float(np.linalg.norm(rel - t_open * d))
        if off > max_offset or not (shank[0] - 2.0 <= t_open <= shank[1] + 0.5):
            continue                      # the hole must open within the screw's own length, not anywhere on its line
        depth = h["depth_mm"] if not h["drawing_max_tap_depth_mm"] else min(h["depth_mm"], h["drawing_max_tap_depth_mm"])
        s0, s1 = shank[0], shank[1] - 0.5
        eng = max(0.0, min(s1, t_open + depth) - max(s0, t_open))
        cand = dict(hole_id=h["hole_id"], part=h["part"], standard=h["standard"], label=h["label"], thread=h["thread"],
                    axis_offset_mm=round(off, 3), angle_deg=round(math.degrees(math.acos(min(1.0, cosang))), 2),
                    opening_depth_along_screw_mm=round(t_open, 3), engagement_mm=round(eng, 3),
                    hole_depth_mm=round(depth, 2), reference_engagement_mm=h["reference_engagement_mm"],
                    terminal_material=h["terminal_material"],
                    tip_short_of_hole_bottom_mm=round(t_open + depth - s1, 3))
        if best is None or (off, -eng) < (best["axis_offset_mm"], -best["engagement_mm"]):
            best = cand
    return best
