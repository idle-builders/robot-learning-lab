"""Helpers for lens B (converging geometry): block primitives, the twisted axis wedge, the egg thigh, and a ray-cast
visible-gap measure. Shared by the concept scripts in this folder; does not modify kit.py.

Angles about the hip pitch axis: psi is measured in a plane x = const around O = (y 2, z 219), from straight down
(psi 0) toward the front (psi +90 = forward, -90 = backward, 180 = up). Hip pitch theta (negative = flexion) moves a
leg point from psi to psi - theta, so flexion -90 carries the thigh's front (psi ~ +30) to psi ~ +120.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kit  # noqa: E402
import manifold3d as M  # noqa: E402

O_YZ = np.array([2.0, 219.0])
HIDE_BASE = ("center_front_shell", "center_back_shell", "R_thigh_front_hard_cover", "R_thigh_back_hard_cover",
             "L_thigh_front_hard_cover", "L_thigh_back_hard_cover")
SHELL = (0.88, 0.85, 0.78)
THIGH = (0.86, 0.83, 0.76)
DARK = (0.30, 0.28, 0.27)


def mf(m):
    return kit.mf(m)


def tm(x):
    return kit.tm(x)


def ell(c, r, n=40):
    return M.Manifold.sphere(1.0, n).scale(tuple(map(float, r))).translate(tuple(map(float, c)))


def hull(parts):
    return M.Manifold.batch_hull(list(parts))


def _yzx(solid, x0):
    """Map an extrusion made in local (a, b, c) = (y, z, x - x0) to world."""
    return solid.transform([[0.0, 0.0, 1.0, float(x0)], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def prism_yz(poly_yz, x0, x1):
    """Polygon [(y, z), ...] (counter-clockwise in y-z) extruded along x from x0 to x1."""
    p = np.asarray(poly_yz, float)
    if _area(p) < 0:
        p = p[::-1]
    cs = M.CrossSection([p.tolist()])
    return _yzx(M.Manifold.extrude(cs, float(x1 - x0)), x0)


def _area(p):
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def ray_yz(psi_deg, r):
    a = np.radians(psi_deg)
    return O_YZ + r * np.array([-np.sin(a), -np.cos(a)])


def sector_poly(psi_a, psi_b, r_out=260.0, r_in=0.0, n=24):
    """Sector about the pitch axis between psi_a < psi_b (degrees), from r_in to r_out (y-z polygon)."""
    ps = np.linspace(psi_a, psi_b, n)
    outer = [ray_yz(p, r_out) for p in ps]
    inner = [ray_yz(p, r_in) for p in ps[::-1]] if r_in > 0 else [O_YZ]
    return np.array(outer + inner)


def twisted_wedge(xs, psi_back, psi_front, r_out=260.0):
    """Leg-opening wedge whose walls are ruled surfaces through the pitch axis: at each x in xs the opening is the
    sector psi_back(x)..psi_front(x). Consecutive sections are hulled, so every wall line passes through the axis
    (the owner's 'slope inward toward the turning point')."""
    out = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for half in ((lambda x: psi_back(x), lambda x: 0.0), (lambda x: 0.0, lambda x: psi_front(x))):
            # each half sector spans < 180 deg, so the hull of two sections stays a ruled wedge
            secs = [prism_yz(sector_poly(half[0](x), half[1](x), r_out), x - 0.01, x + 0.01) for x in (x0, x1)]
            out.append(hull(secs))
    return M.Manifold.batch_boolean(out, M.OpType.Add)


def lip(xs, psi_of_x, r0, r1, t=2.5):
    """A plate lying in the wedge wall (a half-plane through the pitch axis at psi(x)) from radius r0 to r1:
    the inward-sloping funnel lip that reads as the dark gap band. t grows toward the closed side of the wedge."""
    out = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        secs = []
        for x in (x0, x1):
            p = psi_of_x(x)
            s = np.sign(p) if p != 0 else 1.0
            dp = np.degrees(t / max(r0, 1.0)) * s
            a, b = ray_yz(p, r0), ray_yz(p, r1)
            c, d = ray_yz(p + dp, r1), ray_yz(p + dp * r0 / r1, r0)
            secs.append(prism_yz([a, b, c, d], x - 0.01, x + 0.01))
        out.append(hull(secs))
    return M.Manifold.batch_boolean(out, M.OpType.Add)


# ---------------------------------------------------------------- torso pieces

def upper_torso(body, zcut):
    """center_front_shell + center_back_shell trimmed to z >= zcut (one fixed torso part)."""
    f = mf(kit._mesh(body, "center_front_shell")).trim_by_plane((0.0, 0.0, 1.0), float(zcut))
    b = mf(kit._mesh(body, "center_back_shell")).trim_by_plane((0.0, 0.0, 1.0), float(zcut))
    return f + b


def waist_polygon(body, z, inset=0.0):
    """Convex outline (x, y) of the torso shells at height z."""
    pts = []
    for n in ("center_front_shell", "center_back_shell"):
        s = kit._mesh(body, n).section((0, 0, 1), (0, 0, z))
        if s is not None:
            pts.append(np.asarray(s.vertices)[:, :2])
    from scipy.spatial import ConvexHull
    p = np.concatenate(pts)
    h = p[ConvexHull(p).vertices]
    if inset:
        import shapely.geometry as G
        h = np.asarray(G.Polygon(h).buffer(-inset, join_style=1).exterior.coords)[:-1]
    return h


def prism_xy(poly_xy, z0, z1):
    p = np.asarray(poly_xy, float)
    if _area(p) < 0:
        p = p[::-1]
    return M.Manifold.extrude(M.CrossSection([p.tolist()]), float(z1 - z0)).translate((0.0, 0.0, float(z0)))


BOWL = (0.0, 4.0, 222.0, 86.0, 45.0, 42.0)


def briefs_solid(body, waist_z=238.0, bowl=BOWL, hips=None, d=0.0, top_extra=0.0):
    """Rounded briefs solid offset inward by d: hull of the waist outline (matching the torso shells) and a bowl
    ellipsoid (bottom = crotch at z ~180), optionally two hip ellipsoids that hang the sides lower."""
    w = waist_polygon(body, waist_z, inset=d)
    cx, cy, cz, rx, ry, rz = bowl
    es = [prism_xy(w, waist_z - 12.0, waist_z + top_extra), ell((cx, cy, cz), (rx - d, ry - d, rz - d), 64)]
    if hips is not None:
        hx, hy, hz, hrx, hry, hrz = hips
        es += [ell((s * hx, hy, hz), (hrx - d, hry - d, hrz - d)) for s in (1.0, -1.0)]
    return hull(es).trim_by_plane((0.0, 0.0, -1.0), -waist_z) if not top_extra else hull(es)


def briefs_blank(body, waist_z=238.0, bowl=BOWL, hips=None, t=2.5):
    """Hollow briefs (wall t), open at the top so it continues the trimmed torso shell."""
    return briefs_solid(body, waist_z, bowl, hips) - briefs_solid(body, waist_z, bowl, hips, d=t, top_extra=12.0)


def body_envelope(body, waist_z=238.0, bowl=BOWL, hips=None):
    """Outer envelope of briefs + torso up to z 300 (closed), as a trimesh, for edge and skin look-ups."""
    up = hull([prism_xy(waist_polygon(body, z), z - 1.0, z + 1.0) for z in (waist_z, 300.0)])
    return as_mesh(briefs_solid(body, waist_z, bowl, hips) + up)


def skin_y(env, x, z, side=-1.0):
    """y of the envelope skin at (x, z) seen from the front (side -1) or back (+1); nan if missed."""
    o = np.array([[x, 300.0 * side, z]]); d = np.array([[0.0, -side, 0.0]])
    loc, _, _ = env.ray.intersects_location(o, d)
    if not len(loc):
        return float("nan")
    return float(loc[np.argmax(loc[:, 1] * side), 1])


def edge_on_skin(env, x, psi):
    """Point where the wall half-plane psi at x leaves the envelope (the visible seam line), as (y, z)."""
    a = np.radians(psi)
    o = np.array([[x, O_YZ[0], O_YZ[1]]]); d = np.array([[0.0, -np.sin(a), -np.cos(a)]])
    loc, _, _ = env.ray.intersects_location(o, d)
    if not len(loc):
        return (float("nan"), float("nan"))
    k = np.argmax(np.linalg.norm(loc - o, axis=1))
    return (round(float(loc[k, 1]), 1), round(float(loc[k, 2]), 1))


# ---------------------------------------------------------------- thigh

def pitch_link_sweep(body, roll=(-45.0, 25.0), step=5.0, grow=2.0,
                     names=("R_hip_pitch_to_roll_combined", "R_roll_servo_case_reference", "R_hip_pitch_ST3025_output")):
    """Space the pitch-link mechanism takes in the thigh (roll-link) frame over the roll range (hulls of consecutive
    poses, grown by about `grow` mm). The egg thigh is cut by this so roll is clear by construction."""
    out = []
    ax_p, ax_d = kit.ROLL_AXIS[0].copy(), kit.ROLL_AXIS[1]
    ax_p[1] = 0.0
    angs = np.arange(roll[0], roll[1] + 0.1, step)
    for n in names:
        v = np.asarray(kit._mesh(body, n, 1500).vertices)
        h = trimesh.convex.convex_hull(v)
        c = h.centroid
        ext = np.ptp(h.vertices, axis=0)
        hv = c + (h.vertices - c) * (1.0 + 2.0 * grow / ext)
        prev = None
        for a in angs:
            # pitch-link part seen from the roll link rotated by -a
            T = trimesh.transformations.rotation_matrix(np.radians(-a), ax_d, ax_p)
            pv = hv @ T[:3, :3].T + T[:3, 3]
            cur = M.Manifold.hull_points(pv.tolist())
            out.append(cur if prev is None else hull([prev, cur]))
            prev = cur
    return M.Manifold.batch_boolean(out, M.OpType.Add)


def egg_thigh(body, lower=(50.0, 6.5, 142.0, 39.0, 32.5, 20.0), mid=(51.0, 6.5, 170.0, 38.0, 32.0, 22.0),
              top=(60.0, 6.0, 196.0, 25.0, 26.0, 16.0), t=2.5, extra=(), cut=None):
    """Egg-topped thigh shell (roll link): hull of ellipsoids whose top converges toward C; hollow, open at the
    bottom (shin), with the pitch-link sweep cut out (the egg's open top where the hip mechanism enters)."""
    def solid(d):
        es = [ell(c[:3], np.array(c[3:]) - d) for c in (lower, mid, top) + tuple(extra)]
        return hull(es)
    outer = solid(0.0)
    inner = solid(t) + M.Manifold.cube((200.0, 200.0, 40.0)).translate((-50.0, -100.0, 90.0))
    shell = outer - inner - pitch_link_sweep(body)
    if cut is not None:
        shell = shell - cut
    return shell


def as_mesh(x):
    m = tm(x)
    m.remove_unreferenced_vertices()
    return m


# ---------------------------------------------------------------- visible gap (ray cast)

def gap_measure(body, concepts, hide, q_extra, views=("front", "side", "back"), step=1.0):
    """Cast parallel rays at the right hip (x >= 0) and report, per view, the largest vertical run (mm) of
    non-shell hits (mechanism or see-through) lying between shell hits, and the visible mechanism area (mm2) in
    the hip window. Shell = concept parts + parts with role body_shell."""
    q = {**body.standing, **q_extra}
    meshes, shell_flag = [], []
    parts = kit._hip_parts(body, hide)
    for n, p in parts.items():
        lk = p["owning_link"]
        if n in hide or any(k in lk for k in ("shoulder", "arm", "forearm", "hand", "wrist")):
            continue  # arms hang beside the hip and would block the side view
        m = kit._mesh(body, n).copy()
        m.apply_transform(body.link_T(p["owning_link"], q))
        meshes.append(m); shell_flag.append(p.get("role") == "body_shell")
    cs = list(concepts)
    have = {c.name for c in cs}
    for c in list(cs):
        if c.name.startswith("R_") and "L_" + c.name[2:] not in have:
            cs.append(kit.Concept("L_" + c.name[2:], kit.mirror_x(c.mesh), "L_" + c.link[2:], c.color))
    for c in cs:
        m = c.mesh.copy(); m.apply_transform(body.link_T(c.link, q))
        meshes.append(m); shell_flag.append(True)
    lab = np.concatenate([np.full(len(m.faces), i) for i, m in enumerate(meshes)])
    big = trimesh.util.concatenate(meshes)
    shell_flag = np.array(shell_flag)
    out = {}
    zs = np.arange(150.0, 272.0, step)
    for v in views:
        if v == "front":
            us = np.arange(0.0, 104.0, 2.0); org = lambda u, z: (u, -300.0, z); d = (0.0, 1.0, 0.0)
        elif v == "back":
            us = np.arange(0.0, 104.0, 2.0); org = lambda u, z: (u, 300.0, z); d = (0.0, -1.0, 0.0)
        else:
            us = np.arange(-56.0, 64.0, 2.0); org = lambda u, z: (300.0, u, z); d = (-1.0, 0.0, 0.0)
        U, Z = np.meshgrid(us, zs, indexing="ij")
        o = np.array([org(u, z) for u, z in zip(U.ravel(), Z.ravel())])
        tri = big.ray.intersects_first(o, np.tile(d, (len(o), 1)))
        hit = np.where(tri >= 0, lab[np.maximum(tri, 0)], -1)
        is_shell = (hit >= 0) & shell_flag[np.maximum(hit, 0)]
        S = is_shell.reshape(U.shape)
        H = (hit >= 0).reshape(U.shape)
        best, best_u, best_z, area = 0.0, None, None, 0.0
        for i, u in enumerate(us):
            col = S[i]
            idx = np.where(col)[0]
            if len(idx) < 2:
                continue
            lo, hi = idx.min(), idx.max()
            run = 0
            for k in range(lo, hi + 1):
                if not col[k]:
                    run += 1
                    if H[i, k]:
                        area += step * 2.0
                    if run * step > best:
                        best, best_u, best_z = run * step, float(u), float(zs[k])
                else:
                    run = 0
        out[v] = dict(largest_gap_mm=round(best, 1), at_u_mm=best_u, at_z_mm=best_z,
                      visible_mechanism_mm2=round(area, 0))
    return out


def leg_points(body, thigh_mesh, rolls=(-45.0, -30.0, -15.0, 0.0, 15.0, 25.0), n_thigh=6000):
    """Neutral-frame points of the leg below the hip pitch joint (pitch-link parts as they are, roll-link parts and
    the concept thigh over the roll range), i.e. what hip pitch carries round the axis."""
    pts = []
    base = kit._hip_parts(body, HIDE_BASE)
    roll_pts = [np.asarray(thigh_mesh.sample(n_thigh))]
    for n, p in base.items():
        lk = p["owning_link"]
        if lk == "R_pitch_link":
            pts.append(np.asarray(kit._mesh(body, n, 1500).vertices))
        elif lk in ("R_roll_link",):
            roll_pts.append(np.asarray(kit._mesh(body, n, 1500).vertices))
    rp = np.concatenate(roll_pts)
    rp = rp[rp[:, 2] > 110.0]
    for a in rolls:
        T = body.link_T("R_roll_link", {"R_hip_roll": a})
        pts.append(rp @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts)


def psi_r(p):
    dy, dz = p[:, 1] - O_YZ[0], p[:, 2] - O_YZ[1]
    return np.degrees(np.arctan2(-dy, -dz)), np.hypot(dy, dz)


def wedge_from_leg(pts, xs, r0=28.0, flex=90.0, ext=40.0, margin=4.0, x_pad=4.0):
    """psi_front(x) = (largest psi of leg points with r >= r0 near x) + flex + margin; psi_back likewise. This is
    where a fixed wall through the axis must sit for the leg to reach -flex..+ext."""
    ps, r = psi_r(pts)
    keep = r >= r0
    f, b = [], []
    for x in xs:
        s = keep & (np.abs(pts[:, 0] - x) <= x_pad)
        if s.any():
            f.append(ps[s].max() + flex + margin); b.append(ps[s].min() - ext - margin)
        else:
            f.append(np.nan); b.append(np.nan)
    return np.array(f), np.array(b)


def yaw_sweep(body, yaw=(-40.0, 40.0), step=5.0, grow=2.0,
              names=("R_hip_pitch_ST3025_case", "R_hip_yaw_to_pitch_combined")):
    """Space the yaw-link parts (hip pitch servo, yaw-to-pitch connector) take in the torso frame over the yaw range,
    right side plus its mirror; fixed parts (briefs, lips) are kept out of it."""
    out = []
    ax_p, ax_d = kit.YAW_AXIS
    for n in names:
        h = trimesh.convex.convex_hull(np.asarray(kit._mesh(body, n, 1500).vertices))
        c = h.centroid
        hv = c + (h.vertices - c) * (1.0 + 2.0 * grow / np.ptp(h.vertices, axis=0))
        prev = None
        for a in np.arange(yaw[0], yaw[1] + 0.1, step):
            T = trimesh.transformations.rotation_matrix(np.radians(a), ax_d, ax_p)
            cur = M.Manifold.hull_points((hv @ T[:3, :3].T + T[:3, 3]).tolist())
            out.append(cur if prev is None else hull([prev, cur]))
            prev = cur
    r = M.Manifold.batch_boolean(out, M.OpType.Add)
    return r + r.mirror((1.0, 0.0, 0.0))
