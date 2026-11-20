"""Lens A (concentric surfaces) helpers: revolved sector shells about the hip axes, obstacle clouds swept over the
hip pose grid, maximal concentric covers, briefs, and the render/interference run.

Angles: about the PITCH axis (x through y 2, z 219) 0 = back (+y), 90 = up, 180 = front; about the ROLL axis
(y through x 56, z 190) 0 = outboard (+x), 90 = up, 180 = inboard. Right side; kit mirrors R_ parts to the left.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import manifold3d as M
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kit  # noqa: E402

HERE = Path(__file__).resolve().parent
BODY = kit.base()
C = kit.HIP_C.copy()
PITCH = dict(origin=np.array([0.0, 2.0, 219.0]), axis=np.array([1.0, 0, 0]), ref=np.array([0.0, 1, 0]))
ROLL = dict(origin=np.array([56.0, 0.0, 190.0]), axis=np.array([0.0, -1, 0]), ref=np.array([1.0, 0, 0]))
YAW_O = np.array([59.0, 2.0])
SHELLS = ("center_front_shell", "center_back_shell")
COVERS = ("R_thigh_front_hard_cover", "R_thigh_back_hard_cover")
HIDE_BASE = SHELLS + COVERS + ("L_thigh_front_hard_cover", "L_thigh_back_hard_cover")
IVORY, SAND, DARK = (0.90, 0.87, 0.80), (0.84, 0.80, 0.72), (0.30, 0.28, 0.27)


# ---------------------------------------------------------------- manifold basics
def mf(m):
    return kit.mf(m)


def tm(x):
    return kit.tm(x)


def frame(ax):
    a, e0 = ax["axis"], ax["ref"]
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = e0, np.cross(a, e0), a, ax["origin"]
    return T


def axis_coords(ax, p):
    d = p - ax["origin"]; a, e0 = ax["axis"], ax["ref"]; e1 = np.cross(a, e0)
    u, v = d @ e0, d @ e1
    return d @ a, np.hypot(u, v), np.degrees(np.arctan2(v, u))


def rev(profile, ax, a0=0.0, a1=360.0, segs=144):
    """Solid of revolution of a closed (rho, w) polygon about axis ax over angles a0..a1 (deg)."""
    pr = np.asarray(profile, float)
    x, y = pr[:, 0], pr[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        pr = pr[::-1]
    span = a1 - a0
    s = M.Manifold.revolve(M.CrossSection([pr.tolist()]), segs, float(span))
    if a0:
        s = s.rotate((0.0, 0.0, float(a0)))
    return s.transform(frame(ax)[:3, :4])


def union(parts):
    parts = [p for p in parts if p is not None and not p.is_empty()]
    return M.Manifold.batch_boolean(parts, M.OpType.Add) if parts else M.Manifold()


def half_space(n, p0, size=600.0):
    """Solid {x: (x - p0).n <= 0} clipped to a big cube."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    box = M.Manifold.cube((size, size, size), True).translate((0.0, 0.0, -size / 2))  # z <= 0
    R = trimesh.geometry.align_vectors([0, 0, 1], n)
    T = R.copy(); T[:3, 3] = p0
    return box.transform(T[:3, :4])


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return M.Manifold.cube(tuple(hi - lo), False).translate(tuple(lo))


def yaw_rot(psi_deg):
    T = np.eye(4); a = np.radians(psi_deg)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:2, 3] = YAW_O - T[:2, :2] @ YAW_O
    return T


# ---------------------------------------------------------------- obstacles
def solid(name, faces=1500):
    """The solid kit.interference uses for an existing part (watertight mesh, else its convex hull)."""
    m = kit._mesh(BODY, name, faces)
    return m if m.is_watertight else m.convex_hull


def cloud(m, step=2.5):
    """Surface and interior points of a closed mesh."""
    s, _ = trimesh.sample.sample_surface(m, int(m.area / step ** 2) + 200, seed=1)
    lo, hi = m.bounds
    g = np.stack(np.meshgrid(*[np.arange(lo[i] + step / 2, hi[i], step * 1.4) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    if len(g):
        off = np.einsum("fk,fk->f", m.face_normals, m.triangles[:, 0])
        inside = np.zeros(len(g), bool)
        for i in range(0, len(g), 4000):
            gi = g[i:i + 4000]
            inside[i:i + 4000] = np.all(gi @ m.face_normals.T - off[None] <= 1e-6, axis=1) if m.is_convex else m.contains(gi)
        g = g[inside]
    return np.vstack([s, g, m.vertices])


def real_cloud(name, step=2.5):
    """Points of the part's real geometry (2500-face mesh): full cloud if closed, else surface samples only. Used for
    design and for the true-geometry check (kit.interference uses convex hulls of open meshes, which fill e.g. the
    whole lower-pelvis cradle around the hip-pitch servo)."""
    m = kit._mesh(BODY, name, 2500)
    if m.is_watertight:
        return cloud(m, step)
    s, _ = trimesh.sample.sample_surface(m, int(m.area / step ** 2) + 200, seed=1)
    return np.vstack([s, m.vertices])


def obstacle_clouds(hide=(), extra=(), skip_links=(), near=130.0):
    """[(name, link, points)] for every hip part kit.interference would test (and extra Concepts) whose bounds come
    within 'near' mm of C (parts further away cannot reach the hip shells)."""
    out = []
    for n, p in kit._hip_parts(BODY, hide).items():
        if n in hide or p["owning_link"] in skip_links:
            continue
        b = np.array(p["bounds_mm"])
        if np.linalg.norm(np.clip(C, b[0], b[1]) - C) > near:
            continue
        out.append((n, p["owning_link"], real_cloud(n)))
    for c in extra:
        if c.link not in skip_links:
            out.append((c.name, c.link, cloud(c.mesh)))
    return out


def grid_poses(pitch_step=5.0, roll_vals=(-45, -35, -25, -15, -5, 5, 15, 25), yaw_step=10.0):
    for p, r, y in itertools.product(np.arange(-90.0, 40.1, pitch_step), roll_vals, np.arange(-40.0, 40.1, yaw_step)):
        yield {**BODY.standing, "R_hip_pitch": float(p), "R_hip_roll": float(r), "R_hip_yaw": float(y)}
    yield dict(BODY.standing)


def relative_sets(link, clouds, poses):
    """For each obstacle link: the unique transforms (obstacle neutral -> 'link' neutral frame) over the poses."""
    poses = list(poses)
    links = sorted({lk for _, lk, _ in clouds})
    rel = {}
    for lk in links:
        seen = {}
        for q in poses:
            T = np.linalg.inv(BODY.link_T(link, q)) @ BODY.link_T(lk, q)
            seen.setdefault(tuple(np.round(T[:3], 5).ravel()), T)
        rel[lk] = list(seen.values())
    return rel


# ---------------------------------------------------------------- maximal concentric sector shell
def max_sector_shell(ax, w_edges, rho_in, rho_out, link, clouds, poses, clear=2.5, bin_deg=2.0,
                     keep=lambda w, a: True, min_run_deg=6.0, segs=144, extra_forbid=None):
    """Shell of revolution about ax (slabs between w_edges, radii rho_in(w)..rho_out(w)) on 'link', keeping per slab
    only the angles that no obstacle ever reaches over the poses (with clearance). Returns (manifold, coverage)."""
    nb = int(round(360 / bin_deg)); ns = len(w_edges) - 1
    forb = np.zeros((ns, nb), bool)
    wc = 0.5 * (np.asarray(w_edges[:-1]) + np.asarray(w_edges[1:]))
    rel = relative_sets(link, clouds, poses)
    we = np.asarray(w_edges, float)
    RIN = np.array([rho_in(x) for x in wc]) - clear
    ROUT = np.array([rho_out(x) for x in wc]) + clear
    rmax = ROUT.max()
    for name, lk, pts in clouds:
        ph = np.c_[pts, np.ones(len(pts))]
        for T in rel[lk]:
            q = (ph @ T[:3].T)
            w, r, a = axis_coords(ax, q)
            m = (w >= we[0] - clear) & (w <= we[-1] + clear) & (r <= rmax)
            if not m.any():
                continue
            w, r, a = w[m], r[m], a[m]
            si = np.clip(np.searchsorted(we, w) - 1, 0, ns - 1)
            hit = (r >= RIN[si]) & (r <= ROUT[si])
            if hit.any():
                forb[si[hit], (np.floor((a[hit] % 360) / bin_deg).astype(int)) % nb] = True
    forb = forb | np.roll(forb, 1, 0) * (np.arange(ns) > 0)[:, None] | np.roll(forb, -1, 0) * (np.arange(ns) < ns - 1)[:, None]
    if extra_forbid is not None:
        forb |= extra_forbid
    # angular clearance: dilate by clear/rho
    dil = forb.copy()
    for i in range(ns):
        k = int(np.ceil(np.degrees(clear / max(rho_out(wc[i]), 5.0)) / bin_deg))
        for d in range(-k, k + 1):
            dil[i] |= np.roll(forb[i], d)
    pieces, cover = [], np.zeros((ns, nb), bool)
    for i in range(ns):
        ri0, ri1 = rho_in(w_edges[i]), rho_in(w_edges[i + 1])
        ro0, ro1 = rho_out(w_edges[i]), rho_out(w_edges[i + 1])
        if max(ro0, ro1) < 1.0:
            continue
        allowed = np.array([not dil[i, b] and keep(wc[i], (b + 0.5) * bin_deg) for b in range(nb)])
        cover[i] = allowed
        if allowed.all():
            runs = [(0, nb)]
        else:
            start = int(np.argmin(allowed)); runs, cur = [], None  # rotate so we start in a forbidden bin
            for j in range(nb + 1):
                b = (start + j) % nb
                if j < nb and allowed[b]:
                    cur = cur if cur is not None else start + j
                elif cur is not None:
                    runs.append((cur, start + j)); cur = None
        prof = [(max(ri0, 0.0), w_edges[i] - 0.01), (max(ro0, 0.6), w_edges[i] - 0.01),
                (max(ro1, 0.6), w_edges[i + 1] + 0.01), (max(ri1, 0.0), w_edges[i + 1] + 0.01)]
        for b0, b1 in runs:
            if (b1 - b0) * bin_deg < min_run_deg:
                continue
            pieces.append(rev(prof, ax, b0 * bin_deg, b1 * bin_deg, segs))
    return union(pieces), dict(w_edges=list(map(float, w_edges)), bin_deg=bin_deg, allowed=cover)


# ---------------------------------------------------------------- shared concept pieces
def roll_dome(R_eq=44.0, y0=6.5, a_y=34.5, t=2.5, alpha=(-35.0, 215.0)):
    """Thigh top: spheroid of revolution about the roll axis (axis along y), upper part only. Returns profile fns."""
    w0 = -y0  # roll axis coordinate w = -y

    def ro(w):
        u = (w - w0) / a_y
        return R_eq * np.sqrt(max(0.0, 1 - u * u))

    def ri(w):
        u = (w - w0) / (a_y - t)
        return (R_eq - t) * np.sqrt(max(0.0, 1 - u * u))
    edges = np.linspace(w0 - a_y, w0 + a_y, 24)
    keep = (lambda w, a: (alpha[0] <= a <= alpha[1]) or (alpha[0] <= a - 360 <= alpha[1]))
    return edges, ri, ro, keep


def spheroid_solid(ax, w0, a_w, R_eq, segs=96, n=40):
    """Filled spheroid of revolution about ax (for clearances)."""
    th = np.linspace(-np.pi / 2, np.pi / 2, n)
    prof = [(0.0, w0 - a_w)] + [(R_eq * np.cos(t), w0 + a_w * np.sin(t)) for t in th[1:-1]] + [(0.0, w0 + a_w)]
    return rev(prof, ax, 0, 360, segs)


def clipped_covers(z_max):
    """The existing thigh covers below z_max (roll link), as closed solids."""
    out = []
    for n in COVERS:
        m = kit._mesh(BODY, n, 2500)
        if not m.is_watertight:
            m = kit._mesh(BODY, n, 1500)
        s = mf(m) ^ box((-200, -200, -50), (200, 200, z_max))
        out.append(kit.Concept("R_" + n[2:].replace("hard_cover", "cover_low"), tm(s), "R_roll_link", IVORY))
    return out


def ellipsoid(c, s, segs=96):
    return M.Manifold.sphere(1.0, segs).scale(tuple(map(float, s))).translate(tuple(map(float, c)))


def briefs(socket_out, socket_in, leg_cut, z_w=255.0, t=2.5, E=((0.0, 3.0, 255.0), (80.0, 45.0, 78.0)), name="briefs"):
    """Fixed lower torso shell (both sides): an ellipsoid bowl below the waist z_w, merged with the right/left socket
    solids (socket_out/in are right-side manifolds; mirrored here), with the leg openings cut by leg_cut (right side)."""
    c, s = E
    mirror = lambda x: x.mirror((1.0, 0.0, 0.0))
    E_out = ellipsoid(c, s); E_in = ellipsoid(c, np.asarray(s) - t)
    outer = union([E_out, socket_out, mirror(socket_out)]) ^ box((-200, -200, 100), (200, 200, z_w + 2))
    inner = union([E_in, socket_in, mirror(socket_in)])
    cut = union([leg_cut, mirror(leg_cut)])
    return kit.Concept(name, tm(outer - inner - cut), "torso", IVORY)


def upper_shells_render(z_w):
    """Render-only: the original torso shells above the waist (open meshes, not used in interference)."""
    out = []
    for n in SHELLS:
        m = kit._mesh(BODY, n)
        s = trimesh.intersections.slice_mesh_plane(m, [0, 0, 1], [0, 0, z_w], cap=False)
        col = np.array(BODY.parts[n].get("color") or [225, 220, 205, 255], float)[:3] / 255
        out.append(kit.Concept(n + "_above_waist", s, "torso", tuple(col)))
    return out


CACHE = kit.ROOT / ".cache/work/hip-shell/concentric"


def real_check(parts, hide, grid=None, side="R"):
    """Auxiliary true-geometry measure in kit.interference's structure: real-surface points of every other part
    (2500-face meshes, no convex hulls) found inside the posed concept solids (1 mm voxels), growth over standing.
    Returns contact-free single-joint ranges and the count of combined grid poses with contact (points, not mm3)."""
    grid = grid or kit.GRID
    cps = [p for p in parts if p.name.startswith(side + "_") or p.name[:2] not in ("R_", "L_")]
    vox = []
    pitch = 1.5
    for p in cps:
        lo, hi = p.mesh.bounds
        axes = [np.arange(lo[i], hi[i] + pitch, pitch) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        inside = np.zeros(len(g), bool)
        for i in range(0, len(g), 200000):
            inside[i:i + 200000] = p.mesh.contains(g[i:i + 200000])
        mat = inside.reshape([len(a) for a in axes])
        S = np.eye(4); S[:3, :3] *= pitch; S[:3, 3] = lo
        vox.append((p, mat, np.linalg.inv(S)))
    near = lambda pp: np.linalg.norm(np.clip(C, *np.array(pp["bounds_mm"])) - C) <= 150.0
    others = [(n, pp["owning_link"], real_cloud(n, 2.0)) for n, pp in kit._hip_parts(BODY, hide).items() if n not in hide and near(pp)]
    others += [(p.name, p.link, cloud(p.mesh, 2.0)) for p in cps]

    def hits(q):
        out = {}
        for p, mat, inv in vox:
            Tp = np.linalg.inv(BODY.link_T(p.link, q))
            for n, lk, pts in others:
                if lk == p.link or n == p.name:
                    continue
                T = inv @ Tp @ BODY.link_T(lk, q)
                ij = np.round(pts @ T[:3, :3].T + T[:3, 3]).astype(int)
                ok = np.all((ij >= 0) & (ij < np.array(mat.shape)), axis=1)
                k = int(mat[tuple(ij[ok].T)].sum())
                if k:
                    out[(p.name, n)] = k
        return out
    base0 = hits(dict(BODY.standing))
    grow = lambda q: {k: v for k, v in hits({**BODY.standing, **q}).items() if v > base0.get(k, 0) + 2}
    ranges = {}
    for j, vals in (("pitch", grid["pitch"]), ("roll", grid["roll"]), ("yaw", grid["yaw"])):
        free, first = [0.0, 0.0], {}
        for sgn, k in ((-1, 0), (1, 1)):
            for v in sorted([x for x in vals if sgn * x > 0], key=abs):
                g = grow({f"{side}_hip_{j}": float(v)})
                if g:
                    first[str(k)] = [float(v), sorted([f"{a} x {b}: {c} pts" for (a, b), c in g.items()])[:4]]; break
                free[k] = float(v)
        ranges[j] = dict(contact_free_deg=free, first_contact=first)
    n_c, worst = 0, []
    for p_, r_, y_ in itertools.product(grid["pitch"][::2], grid["roll"], grid["yaw"]):
        g = grow({f"{side}_hip_pitch": float(p_), f"{side}_hip_roll": float(r_), f"{side}_hip_yaw": float(y_)})
        if g:
            n_c += 1; (a, b), c = max(g.items(), key=lambda t: t[1])
            worst.append(dict(pose_pitch_roll_yaw=[float(p_), float(r_), float(y_)], pair=f"{a} x {b}", points=c, total_points=sum(g.values())))
    worst.sort(key=lambda w: -w["total_points"])
    return dict(method="real-surface points (2 mm spacing) inside 1.5 mm voxels of the concept solids; growth over standing > 2 points",
                neutral_points={f"{a} x {b}": c for (a, b), c in base0.items()}, ranges=ranges,
                combined_poses_with_contact=n_c, worst_combined=worst[:8])


def save_parts(stem, parts):
    d = CACHE / stem; d.mkdir(parents=True, exist_ok=True)
    meta = []
    for p in parts:
        f = d / f"{p.name}.npz"
        np.savez_compressed(f, v=np.asarray(p.mesh.vertices), f=np.asarray(p.mesh.faces))
        meta.append(dict(name=p.name, link=p.link, color=list(p.color), file=str(f)))
    json.dump(meta, open(d / "parts.json", "w"))


def load_parts(stem):
    out = []
    for m in json.load(open(CACHE / stem / "parts.json")):
        d = np.load(m["file"])
        out.append(kit.Concept(m["name"], trimesh.Trimesh(d["v"], d["f"], process=False), m["link"], tuple(m["color"])))
    return out


def mem(tag):
    import resource
    print(f"[{tag}] maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB", flush=True)


def run(stem, parts, hide, title, z_w, notes=None):
    """Phase 1 (this process): kit.interference (default grid) + real_check, JSON next to the script, meshes cached.
    Phase 2 (fresh process via exec, so the render gets the memory): kit.render, default LOOK_POSES, hip region."""
    import os
    hide = tuple(hide)
    t0 = time.time()
    import gc; gc.collect(); mem("start run " + " ".join(f"{p.name}:{len(p.mesh.faces)}" for p in parts))
    card = kit.interference(BODY, parts, hide=hide)
    card["seconds"] = round(time.time() - t0, 1)
    mem("after kit.interference")
    card["real_geometry_check"] = real_check(parts, hide)
    mem("after real_check")
    card["concept_parts"] = {p.name: dict(link=p.link, volume_mm3=round(float(p.mesh.volume), 0), faces=len(p.mesh.faces),
                                          bounds_mm=np.round(p.mesh.bounds, 1).tolist()) for p in parts}
    card["hidden"] = list(hide)
    card["notes"] = notes or {}
    json.dump(card, open(HERE / f"{stem}-interference.json", "w"), indent=1, default=float)
    print(json.dumps({k: card[k] for k in ("neutral_overlap_mm3", "ranges", "combined_poses_with_contact")}, indent=1, default=float))
    print("worst", json.dumps(card["worst_combined"][:4], default=float))
    print("real", json.dumps(card["real_geometry_check"]["ranges"], default=float), card["real_geometry_check"]["combined_poses_with_contact"])
    save_parts(stem, parts)
    json.dump(dict(hide=hide, title=title, z_w=z_w), open(CACHE / stem / "render.json", "w"))
    print("phase 1", round(time.time() - t0, 1), "s", flush=True)
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()), "render", stem])


def render_phase(stem):
    cfg = json.load(open(CACHE / stem / "render.json"))
    parts = []
    for p in load_parts(stem):
        # kit's painter sort needs small triangles: keep the revolved shells near full resolution
        m = p.mesh if len(p.mesh.faces) <= 40000 else p.mesh.simplify_quadric_decimation(face_count=40000)
        parts.append(kit.Concept(p.name, m, p.link, p.color))
    parts += upper_shells_render(cfg["z_w"])
    # kit.render with the default LOOK_POSES, two poses per call (one call with all four peaks just above 6 GB here),
    # stacked into one image
    import gc
    import matplotlib.pyplot as plt
    halves = []
    for k, poses in enumerate((kit.LOOK_POSES[:2], kit.LOOK_POSES[2:])):
        f = CACHE / stem / f"render-{k}.png"
        kit.render(BODY, parts, str(f), hide=tuple(cfg["hide"]), poses=poses, title=cfg["title"] if k == 0 else "")
        halves.append(plt.imread(f)); gc.collect()
    w = min(h.shape[1] for h in halves)
    plt.imsave(HERE / f"{stem}.png", np.concatenate([h[:, :w] for h in halves], axis=0))
    print("rendered", HERE / f"{stem}.png")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "render":
    render_phase(sys.argv[2])


# ---------------------------------------------------------------- swept cut (voxel) for range-first briefs
def target_poses():
    """Hip poses the briefs must clear: kit grid values plus intermediate yaw/roll samples."""
    roll = sorted(set(kit.GRID["roll"]) | {-37.5, -22.5, -7.5, 7.5, 20.0})
    yaw = sorted(set(kit.GRID["yaw"]) | {-30.0, -10.0, 10.0, 30.0})
    for p, r, y in itertools.product(kit.GRID["pitch"], roll, yaw):
        yield {**BODY.standing, "R_hip_pitch": float(p), "R_hip_roll": float(r), "R_hip_yaw": float(y)}
    yield dict(BODY.standing)


def swept_cut(moving, poses, gap=3.0, res=1.5, lo=(-2.0, -140.0, 105.0), hi=(140.0, 120.0, 335.0)):
    """Closed solid around everything in 'moving' [(link, points)] over the poses, dilated by gap (voxel + marching
    cubes). Used to cut the fixed briefs so they clear the moving shells and mechanism."""
    from scipy import ndimage
    from skimage import measure
    lo, hi = np.asarray(lo), np.asarray(hi)
    shape = tuple(np.ceil((hi - lo) / res).astype(int) + 1)
    occ = np.zeros(shape, bool)
    poses = list(poses)
    for link, pts in moving:
        seen = {}
        for q in poses:
            T = BODY.link_T(link, q)
            seen.setdefault(tuple(np.round(T[:3], 5).ravel()), T)
        for T in seen.values():
            p = pts @ T[:3, :3].T + T[:3, 3]
            ij = np.floor((p - lo) / res).astype(int)
            ok = np.all((ij >= 0) & (ij < np.array(shape)), axis=1)
            occ[tuple(ij[ok].T)] = True
    k = int(np.ceil(gap / res))
    st = np.linalg.norm(np.stack(np.meshgrid(*[np.arange(-k, k + 1)] * 3, indexing="ij"), -1), axis=-1) <= gap / res + 0.5
    occ = ndimage.binary_closing(occ, iterations=2)
    occ = ndimage.binary_dilation(occ, structure=st)
    f = ndimage.gaussian_filter(np.pad(occ.astype(float), 2), 0.8)
    v, faces, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 2) * res + lo
    m = trimesh.Trimesh(v, faces[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    s = mf(m)
    if s.is_empty():
        m = trimesh.Trimesh(v, faces, process=True); s = mf(m)
    return s


def moving_clouds(hide, concepts, links=("R_yaw_link", "R_pitch_link", "R_roll_link", "R_shin")):
    out = []
    for n, p in kit._hip_parts(BODY, hide).items():
        if n not in hide and p["owning_link"] in links:
            out.append((p["owning_link"], real_cloud(n)))
    for c in concepts:
        if c.link in links:
            out.append((c.link, cloud(c.mesh, step=2.0)))
    return out
