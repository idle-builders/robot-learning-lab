"""Shared helpers for lens C (moving or compliant covers) on top of ../kit.py.

- VBody: wraps kit.base() so concept parts can sit on *virtual links* whose pose is computed from the hip angles
  (a plate that slides with a pin on the pitch link, a spring flap pushed by the thigh, ...). kit.render and
  kit.interference only call body.link_T(link, q), so they measure these parts unchanged.
- render(): kit.render plus per-pose extra items (fabric surfaces that deform with the pose).
- shell trimming, thigh-cover chamfer, side hip panel, mapped-cube block builder, per-joint contact sweep.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kit  # noqa: E402

PITCH_PT = np.array([59.0, 2.0, 219.0])


def rot(origin, axis, deg):
    return trimesh.transformations.rotation_matrix(math.radians(deg), np.asarray(axis, float), np.asarray(origin, float))


def tr(v):
    T = np.eye(4); T[:3, 3] = v
    return T


def apply(T, p):
    p = np.asarray(p, float)
    return p @ T[:3, :3].T + T[:3, 3]


class VBody:
    """kit body with virtual links. virtual[key](body, q, side) -> 4x4; links are named '<side>_<key>'."""

    def __init__(self, body):
        self._b = body
        self.virtual = {}

    def __getattr__(self, n):
        return getattr(self._b, n)

    def link_T(self, link, q):
        key = link[2:] if link[:2] in ("R_", "L_") else link
        if key in self.virtual:
            return self.virtual[key](self._b, q, link[0])
        return self._b.link_T(link, q)


def joint(b, s, j):
    return b.joints[f"{s}_hip_{j}"]


def frac_link(kp, kr=0.0):
    """Turns kp of the hip pitch (and kr of the roll) on top of the yaw link: an interpolated link."""
    def f(b, q, s):
        T = b.link_T(f"{s}_yaw_link", q)
        jp = joint(b, s, "pitch"); jr = joint(b, s, "roll")
        T = T @ rot(jp["origin_world_mm"], jp["axis_world_neutral"], kp * q.get(jp["name"], 0.0))
        if kr:
            T = T @ rot(jr["origin_world_mm"], jr["axis_world_neutral"], kr * q.get(jr["name"], 0.0))
        return T
    return f


def hip_key(q, s):
    return tuple(round(float(q.get(f"{s}_hip_{j}", 0.0)), 2) for j in ("yaw", "pitch", "roll"))


def follower_link(part_R, obstacles_R, motion, pmax, tol=0.3, iters=8, log=None):
    """A spring-returned cover pushed by the thigh: its pose is motion(p, side) with the smallest p in [0, pmax]
    for which the part overlaps the thigh obstacles by <= tol mm3 (below kit.interference's 0.5 mm3 contact floor) (binary search, monotone assumption).
    obstacles_R: [(mesh, link_suffix)] for the right side; the left side uses mirrored meshes.
    log: optional dict filled with key -> p (for reporting the travel)."""
    import manifold3d  # noqa: F401
    P = {"R": kit.mf(part_R), "L": kit.mf(kit.mirror_x(part_R))}
    O = {"R": [(kit.mf(m), lk) for m, lk in obstacles_R],
         "L": [(kit.mf(kit.mirror_x(m)), lk) for m, lk in obstacles_R]}
    cache = {}

    def f(b, q, s):
        key = (s,) + hip_key(q, s)
        if key not in cache:
            obs = [g.transform(b.link_T(f"{s}_{lk}", q)[:3, :4]) for g, lk in O[s]]

            def ov(p):
                g = P[s].transform(motion(p, s)[:3, :4])
                return sum((g ^ o).volume() for o in obs)
            if ov(0.0) <= tol:
                p = 0.0
            elif ov(pmax) > tol:
                p = pmax
            else:
                lo, hi = 0.0, pmax
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if ov(mid) <= tol else (mid, hi)
                p = hi
            cache[key] = p
            if log is not None:
                log[key] = p
        return motion(cache[key], s)
    return f


# ---------------------------------------------------------------- geometry blocks

def mapped_cube(fn, n=(24, 12, 1)):
    """Closed mesh of the image of the unit cube under fn(u, v, w) -> xyz (vectorised over arrays).
    n = cells along u, v, w. Faces of the cube are gridded and merged."""
    nu, nv, nw = n
    us, vs, ws = np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), np.linspace(0, 1, nw + 1)
    V, F = [], []

    def face(a, b, fixed_axis, fixed_val, flip):
        A, B = np.meshgrid(a, b, indexing="ij")
        uvw = [None, None, None]
        free = [i for i in range(3) if i != fixed_axis]
        uvw[free[0]] = A; uvw[free[1]] = B; uvw[fixed_axis] = np.full_like(A, fixed_val)
        P = np.stack(fn(*uvw), -1).reshape(-1, 3)
        base = sum(len(v) for v in V)
        na, nb = A.shape
        idx = np.arange(na * nb).reshape(na, nb) + base
        q0, q1, q2, q3 = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        t = np.concatenate([np.stack([q0, q1, q2], -1).reshape(-1, 3), np.stack([q0, q2, q3], -1).reshape(-1, 3)])
        if flip:
            t = t[:, ::-1]
        V.append(P); F.append(t)

    face(vs, ws, 0, 0.0, False); face(vs, ws, 0, 1.0, True)
    face(us, ws, 1, 0.0, True); face(us, ws, 1, 1.0, False)
    face(us, vs, 2, 0.0, False); face(us, vs, 2, 1.0, True)
    m = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=True)
    m.merge_vertices(digits_vertex=6)
    m.remove_unreferenced_vertices()
    if m.volume < 0:
        m.invert()
    return m


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=tr((lo + hi) / 2))


def boolean(a, b, op):
    A, B = kit.mf(a), kit.mf(b)
    r = A - B if op == "-" else (A + B if op == "+" else A ^ B)
    return kit.tm(r)


def section_curve(mesh, z, x_range, y_side="front", n=40):
    """Outer surface y(x) of a torso shell at height z: for each x the extreme y (front: min y, back: max y)."""
    s = mesh.section([0, 0, 1], [0, 0, z])
    P = np.asarray(s.vertices)[:, :2]
    xs = np.linspace(*x_range, n); ys = []
    for x in xs:
        sel = np.abs(P[:, 0] - x) < 3.0
        ys.append(P[sel, 1].min() if y_side == "front" else P[sel, 1].max())
    return xs, np.array(ys)


# ---------------------------------------------------------------- standard part edits

def trimmed_torso(body, z_front=250.0, z_back=None, x_keep=8.0, x_side=66.0, y_side=(-12.0, 14.0)):
    """Front (and optionally back) torso shell with the hip windows removed: below z_front (z_back) everything with
    x_keep <= |x| <= x_side is cut away, and further out only the part in front of y_side[0] (behind y_side[1]),
    so a side strip around the hip-yaw connector keeps the original rim. The centre strip |x| < x_keep stays down
    to the crotch (briefs gusset)."""
    out = []

    def cut(m, boxes):
        for lo, hi in boxes:
            m = boolean(m, box(lo, hi), "-")
            m = boolean(m, box((-hi[0], lo[1], lo[2]), (-lo[0], hi[1], hi[2])), "-")
        return m
    zf = z_front
    m = cut(kit._mesh(body, "center_front_shell"),
            [((x_keep, -120, 100), (x_side, 10, zf)), ((x_side - 1, -120, 100), (130, y_side[0], zf))])
    out.append(kit.Concept("torso_front_upper", m, "torso", (0.88, 0.85, 0.78)))
    if z_back is not None:
        zb = z_back
        m = cut(kit._mesh(body, "center_back_shell"),
                [((x_keep, 0, 100), (x_side, 120, zb)), ((x_side - 1, y_side[1], 100), (130, 120, zb))])
        out.append(kit.Concept("torso_back_upper", m, "torso", (0.86, 0.83, 0.76)))
    return out


def side_hip_panel(x0=93.0, t=2.5, yc=2.0, zc=212.0, ay=26.0, az=17.0, link="R_yaw_link"):
    """Oval plate outside the thigh's outer wall, on the yaw link: covers the side of the hip joint below the torso
    side rim. It does not pitch; the thigh pitches and rolls behind it."""
    from shapely.geometry import Polygon
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    poly = Polygon(np.c_[yc + ay * np.cos(a), zc + az * np.sin(a)])
    m = trimesh.creation.extrude_polygon(poly, t)           # in (y, z) plane, extruded along +x
    T = np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    return kit.Concept("R_side_hip_panel", m, link, (0.86, 0.84, 0.78))


def chamfered_thigh_covers(body, p0=(4.0, 207.0), p1=(-25.0, 190.0), color=(0.90, 0.87, 0.80)):
    """Right thigh covers with the front-top corner chamfered (plane through (y,z) p0 and p1, along x), so the
    cover clears the yaw-to-pitch connector at deep flexion. Returns Concepts on R_roll_link."""
    (y0, z0), (y1, z1) = p0, p1
    d = np.array([0.0, y1 - y0, z1 - z0]); d /= np.linalg.norm(d)
    nrm = np.array([0.0, -d[2], d[1]])          # perpendicular in the y-z plane
    if nrm[2] < 0:
        nrm = -nrm                               # points up(-forward): the removed side
    # half-space as a big box aligned to the plane
    big = trimesh.creation.box(extents=(400, 400, 200))
    R = np.eye(4); R[:3, 0] = [1, 0, 0]; R[:3, 1] = np.cross(nrm, [1, 0, 0]); R[:3, 2] = nrm
    R[:3, 3] = np.array([0.0, y0, z0]) + nrm * 100.0
    big.apply_transform(R)
    out = []
    for n in ("R_thigh_front_hard_cover", "R_thigh_back_hard_cover"):
        m = boolean(kit._mesh(body, n), big, "-")
        out.append(kit.Concept(n.replace("hard_cover", "cover_chamfered"), m, "R_roll_link", color))
    return out


# ---------------------------------------------------------------- render with per-pose extras

def render(body, concepts, out, hide=(), poses=None, extra=None, title="",
           views=((0, "front"), (-35, "3/4"), (90, "side"), (180, "back"))):
    """Same rows, columns, crop and sketch column as kit.render; extra(q) -> [(mesh, T, rgb)] adds pose-dependent
    meshes (already in world frame when T is identity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    poses = poses or kit.LOOK_POSES
    cs = list(concepts)
    have = {c.name for c in cs}
    for c in list(cs):
        if c.name.startswith("R_") and "L_" + c.name[2:] not in have:
            cs.append(kit.Concept("L_" + c.name[2:], kit.mirror_x(c.mesh), "L_" + c.link[2:], c.color, c.note))
    parts = kit._hip_parts(body, hide)
    xlim, zlim = (-140, 140), (90, 360)
    ncol = len(views) + 1
    fig, axs = plt.subplots(len(poses), ncol, figsize=(5.2 * ncol, 5.6 * len(poses)), squeeze=False)
    img = plt.imread(kit.SKETCH)
    for r, (lab, dq) in enumerate(poses):
        q = {**body.standing, **dq}
        items = []
        for n, p in parts.items():
            col = np.array(p.get("color") or [180, 180, 180, 255], float)[:3] / 255
            items.append((kit._mesh(body, n), body.link_T(p["owning_link"], q), col))
        for c in cs:
            items.append((c.mesh, body.link_T(c.link, q), np.array(c.color)))
        if extra:
            items.extend(extra(q))
        crop = img[300:620, 120:560] if r == 0 else img[300:620, 1150:1500] if r == 1 else img[300:620, 650:1120]
        axs[r, 0].imshow(crop); axs[r, 0].axis("off"); axs[r, 0].set_title("sketch", fontsize=11)
        for c, (yaw, v) in enumerate(views):
            _draw(axs[r, c + 1], items, yaw, f"{lab} - {v}", xlim, zlim)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(out, dpi=60); plt.close(fig)


def _draw(ax, items, yaw, title, xlim, zlim):
    """kit._draw with the polygon collection rasterised (much less memory at savefig) and polygon edges drawn in the
    face colour, so thin block-model facets do not render see-through."""
    from matplotlib.collections import PolyCollection
    R = kit._rot(yaw, 8); L = np.array([0.4, -0.7, 0.6]); L /= np.linalg.norm(L); P, C, D = [], [], []
    for m, T, c in items:
        v = (m.vertices @ T[:3, :3].T + T[:3, 3]) @ R.T; nn = m.face_normals @ T[:3, :3].T
        tri = v[m.faces]; vis = (nn @ R.T)[:, 1] < 0.05; sh = 0.35 + 0.65 * np.clip(np.abs(nn @ L), 0, 1)
        P.append(tri[vis][:, :, [0, 2]]); D.append(tri[vis][:, :, 1].mean(1)); C.append(np.clip(np.outer(sh[vis], c), 0, 1))
    P = np.concatenate(P); D = np.concatenate(D); C = np.concatenate(C)
    o = np.argsort(-D)
    ax.add_collection(PolyCollection(P[o], facecolors=C[o], edgecolors=C[o], linewidths=0.35, rasterized=True))
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=11)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


def sweep(body, concepts, hide, joint="R_hip_pitch", values=np.arange(-90.0, 40.1, 10.0), base=None, side="R"):
    """Per-value overlap growth (mm3, vs. standing) of every concept/part pair along one hip joint, others held at
    standing: the full contact table behind kit.interference's first-contact walk."""
    others = {n: p for n, p in kit._hip_parts(body, hide).items() if n not in hide}
    cm = {c.name: (kit.mf(c.mesh), c.link) for c in concepts if c.name.startswith(side + "_") or c.name[:2] not in ("R_", "L_")}
    om = {}
    for n, p in others.items():
        m = kit._mesh(body, n, 1500)
        om[n] = (kit.mf(m if m.is_watertight else m.convex_hull), p["owning_link"])

    def overlap(q):
        posed = {n: (g.transform(body.link_T(lk, q)[:3, :4]), lk) for n, (g, lk) in list(cm.items()) + list(om.items())}
        res = {}
        for a, (ga, la) in posed.items():
            if a not in cm:
                continue
            for b, (gb, lb) in posed.items():
                if b == a or lb == la or (b in cm and b < a):
                    continue
                v = (ga ^ gb).volume()
                if v > 0.5:
                    res[f"{a} x {b}"] = v
        return res
    q0 = {**body.standing, **(base or {})}
    b0 = overlap(q0)
    out = {}
    for v in values:
        o = overlap({**q0, joint: float(v)})
        out[float(v)] = {k: round(x - b0.get(k, 0.0), 1) for k, x in o.items() if x > b0.get(k, 0.0) + 0.5}
    return out
