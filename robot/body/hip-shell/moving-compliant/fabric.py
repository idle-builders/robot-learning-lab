"""Fabric zones for lens C: ruled stretch-knit panels between a top edge on rigid islands and a bottom edge on the
thigh. Per pose: taut length of each ruling (straight, or wrapped over body parts that poke across it, in the
ruling's own plane), strain against the fabric's natural length (mounted with PRESTRETCH at standing), shear of the
ruling against the top edge, length change along the mid-line, crease gap at the top edge, and a render mesh (slack
rulings drawn bulged outward by the excess length).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

import mc_common as mc
from mc_common import kit

PRESTRETCH = 0.10           # fabric mounted 10 % stretched at standing, so it does not sag
KNIT_WORKING_STRAIN = 0.40  # repeated-cycle working stretch used for 4-way nylon/elastane knit (owner range 30-60 %)
NEOPRENE_WORKING_STRAIN = 0.25
TPU_SKIN_WORKING_STRAIN = 0.15   # printed TPU 95A solid skin, repeated bending/stretch near layer lines (estimate)


@dataclass
class Zone:
    name: str
    A: np.ndarray          # (n, 3) top edge, neutral world
    A_links: list          # link per A point (side-generic suffix, e.g. 'torso' or 'yaw_link')
    B: np.ndarray          # (n, 3) bottom edge on the thigh (roll link)
    lateral: np.ndarray    # direction along the zone's edges (x for front/back, y for the inner zone)
    inside: np.ndarray     # a point on the body side of the fabric (wrap reference)
    color: tuple = (0.42, 0.42, 0.46)


def link_name(s, lk):
    return lk if lk == "torso" else f"{s}_{lk}"


class Field:
    """Surface sample points of the body parts in the hip region (local frames), posed on demand."""

    def __init__(self, body, concepts, hide, links=("torso", "R_yaw_link", "R_pitch_link", "R_roll_link"), n=1200):
        rng = np.random.default_rng(0)
        self.items = []
        for nm, p in kit._hip_parts(body, hide).items():
            if p["owning_link"] not in links:
                continue
            m = kit._mesh(body, nm, 1500)
            pts, _ = trimesh.sample.sample_surface(m, n, seed=int(rng.integers(1 << 30)))
            self.items.append((nm, p["owning_link"], np.asarray(pts)))
        for c in concepts:
            if c.link in links or c.link == "torso":
                pts, _ = trimesh.sample.sample_surface(c.mesh, 2 * n, seed=int(rng.integers(1 << 30)))
                self.items.append((c.name, c.link, np.asarray(pts)))

    def posed(self, body, q, exclude_links=()):
        out, lab = [], []
        for nm, lk, P in self.items:
            if lk in exclude_links:
                continue
            out.append(mc.apply(body.link_T(lk, q), P)); lab.append((nm, lk, len(P)))
        return np.concatenate(out), lab


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def taut(A, B, pts, inside, lateral, slab=4.0, res=1.0, dil=2.0, margin=45.0):
    """Taut path A->B in the plane through A and B whose normal is closest to 'lateral': the body points within the
    slab are rasterised (res mm, dilated, holes filled) and the shortest free path is found on the grid and then
    string-pulled, so the fabric wraps over whatever pokes across AB. Returns (length, polyline 3D, (d, e2)),
    e2 = in-plane normal pointing away from 'inside' (the outer side, used to bulge slack fabric)."""
    from scipy import ndimage
    from skimage.graph import MCP_Geometric
    d = B - A; L = np.linalg.norm(d); d /= L
    n = lateral - (lateral @ d) * d; n /= np.linalg.norm(n)
    e2 = np.cross(n, d)
    if (inside - A) @ e2 > 0:
        e2 = -e2
    rel = pts - A
    sel = np.abs(rel @ n) < slab
    u = rel[sel] @ d; v = rel[sel] @ e2
    u0, u1, v0, v1 = -margin, L + margin, -margin - 20, margin + 20
    ok = (u > u0) & (u < u1) & (v > v0) & (v < v1)
    nu, nv = int((u1 - u0) / res) + 1, int((v1 - v0) / res) + 1
    occ = np.zeros((nu, nv), bool)
    occ[((u[ok] - u0) / res).astype(int), ((v[ok] - v0) / res).astype(int)] = True
    r = int(np.ceil(dil / res)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    occ = ndimage.binary_dilation(occ, structure=(xx ** 2 + yy ** 2) <= r * r)
    occ = ndimage.binary_fill_holes(occ)
    ia = (int((0 - u0) / res), int((0 - v0) / res)); ib = (int((L - u0) / res), int((0 - v0) / res))
    gu, gv = np.mgrid[0:nu, 0:nv]
    for c in (ia, ib):
        occ[(gu - c[0]) ** 2 + (gv - c[1]) ** 2 <= (3.5 / res) ** 2] = False

    def free_seg(p, q):
        m = int(np.hypot(*(np.subtract(q, p)))) * 2 + 2
        ii = np.linspace(p[0], q[0], m).round().astype(int); jj = np.linspace(p[1], q[1], m).round().astype(int)
        return not occ[ii, jj].any()
    if free_seg(ia, ib):
        chain = np.array([[0.0, 0.0], [L, 0.0]])
    else:
        cost = np.where(occ, np.inf, 1.0)
        mcp = MCP_Geometric(cost)
        mcp.find_costs([ia], [ib])
        try:
            path = np.array(mcp.traceback(ib))
        except Exception:
            path = np.array([ia, ib])
        pulled = [path[0]]; i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not free_seg(path[i], path[j]):
                j -= 1
            pulled.append(path[j]); i = j
        P = np.array(pulled, float)
        chain = np.c_[P[:, 0] * res + u0, P[:, 1] * res + v0]
        chain[0] = (0.0, 0.0); chain[-1] = (L, 0.0)
    length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
    poly = A + chain[:, :1] * d + chain[:, 1:2] * e2
    return length, poly, (d, e2)


def zone_state(body, z, q, field_pts, s="R"):
    A = np.array([mc.apply(body.link_T(link_name(s, lk), q), a) for a, lk in zip(z.A, z.A_links)])
    B = mc.apply(body.link_T(f"{s}_roll_link", q), z.B)
    inside = mc.apply(body.link_T("torso", q), z.inside) if not z.name.startswith("inner") else \
        mc.apply(body.link_T(f"{s}_yaw_link", q), z.inside)
    res = []
    for a, b in zip(A, B):
        res.append(taut(a, b, field_pts, inside, z.lateral))
    return A, B, res


def measure(body, zones, poses, field, s="R"):
    """Strain table: per zone and pose, ruling strain (vs natural length) min/max, shear (deg) max, mid-line
    length change, crease gap (min distance from the top edge to leg parts), contact rulings count."""
    q0 = dict(body.standing)
    P0, _ = field.posed(body, q0)
    base = {}
    for z in zones:
        A, B, r = zone_state(body, z, q0, P0, s)
        L0 = np.array([x[0] for x in r])
        mid0 = np.array([x[1][len(x[1]) // 2] if len(x[1]) > 2 else 0.5 * (x[1][0] + x[1][-1]) for x in r])
        base[z.name] = (A, B, L0, mid0)
    table = {}
    for lab, dq in poses:
        q = {**q0, **dq}
        P, _ = field.posed(body, q)
        Pleg, _ = field.posed(body, q, exclude_links=("torso", f"{s}_yaw_link"))
        row = {}
        for z in zones:
            A, B, r = zone_state(body, z, q, P, s)
            A0, B0, L0, mid0 = base[z.name]
            L = np.array([x[0] for x in r]); Lnat = L0 / (1 + PRESTRETCH)
            strain = L / Lnat - 1.0
            straight = np.linalg.norm(B - A, axis=1)
            wrapped = int(np.sum(L > straight + 1.0))
            # shear: angle between ruling and top-edge tangent, change vs standing
            def ang(Ax, Bx):
                t = np.gradient(Ax, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
                r_ = Bx - Ax; r_ /= np.linalg.norm(r_, axis=1, keepdims=True)
                return np.degrees(np.arccos(np.clip(np.sum(t * r_, 1), -1, 1)))
            shear = ang(A, B) - ang(A0, B0)
            mid = 0.5 * (A + B); midn = 0.5 * (A0 + B0)
            ml = np.linalg.norm(np.diff(mid, axis=0), axis=1).sum(); ml0 = np.linalg.norm(np.diff(midn, axis=0), axis=1).sum()
            dmin = np.min(np.linalg.norm(A[:, None, :] - Pleg[None, ::3, :], axis=2), axis=1)
            row[z.name] = dict(strain_min=round(float(strain.min()), 3), strain_max=round(float(strain.max()), 3),
                               strain_per_ruling=[round(float(x), 3) for x in strain],
                               excess_fabric_mm_max=round(float(max(0.0, (Lnat - L).max())), 1),
                               shear_deg_max=round(float(np.abs(shear).max()), 1),
                               midline_change=round(float(ml / ml0 - 1.0), 3),
                               crease_gap_mm_min=round(float(dmin.min()), 1),
                               rulings_wrapped_over_parts=wrapped)
        table[lab] = row
    return table


def render_mesh(body, z, q, field, s="R", rings=7, thick_offset=0.6):
    """Two-sided surface of the zone at pose q (world frame), slack rulings bulged outward."""
    P, _ = field.posed(body, q)
    A, B, r = zone_state(body, z, q, P, s)
    A0, B0, r0 = zone_state(body, z, dict(body.standing), field.posed(body, dict(body.standing))[0], s)
    rows = []
    for (L, poly, (d, e2)), (L0, _, _) in zip(r, r0):
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
        ts = np.linspace(0, cum[-1], rings)
        pts = np.array([np.array([np.interp(t, cum, poly[:, k]) for k in range(3)]) for t in ts])
        Lnat = L0 / (1 + PRESTRETCH)
        if L < Lnat:
            h = min(18.0, np.sqrt(3 * L * (Lnat - L) / 8.0))
            w = ts / max(cum[-1], 1e-6)
            pts = pts + (4 * w * (1 - w) * h)[:, None] * e2[None, :]
        pts = pts + thick_offset * e2[None, :]
        rows.append(pts)
    G = np.array(rows)                          # stations x rings x 3
    ns, nr = G.shape[:2]
    idx = np.arange(ns * nr).reshape(ns, nr)
    q0_, q1_, q2_, q3_ = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    F = np.concatenate([np.stack([q0_, q1_, q2_], -1).reshape(-1, 3), np.stack([q0_, q2_, q3_], -1).reshape(-1, 3)])
    F = np.concatenate([F, F[:, ::-1]])
    m = trimesh.Trimesh(G.reshape(-1, 3), F, process=False)
    return m
