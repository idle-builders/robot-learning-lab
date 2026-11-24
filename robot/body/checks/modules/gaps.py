"""gaps: standing, every part (any link) within 6 mm of a scoped part: overlap (mm3) and distance (mm), and where it is set
(limiting_location; exact closest points and faces for the closest pairs). Raw function moved from leg/fast_check.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks.core import ROOT, R, kit, screen_solid, step_mesh, tick
from checks.registry import register


def check_gaps(body, changed, sol, mf):
    """Fasteners are left out, their sleeve sets kept (process.relations: definition, role class sleeve_set)."""
    from checks import process as P
    rel = P.relations(body)
    t0 = time.time()
    q = body.standing
    out, memo = {}, {}
    for a in changed:
        Ta = body.link_T(body.parts[a]["owning_link"], q)[:3, :4]
        ga = mf[a].transform(Ta); ba = np.array(ga.bounding_box()).reshape(2, 3)
        for n, p in body.parts.items():
            if n == a or rel.is_fastener(n) and not rel.is_sleeve_set(n):
                continue
            b = np.array(p["bounds_mm"], float)
            Tn = body.link_T(p["owning_link"], q)
            c = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ Tn[:3, :3].T + Tn[:3, 3]
            if np.linalg.norm(np.maximum(0.0, np.maximum(ba[0] - c.max(0), c.min(0) - ba[1]))) > 6.0:
                continue
            if n not in mf:
                t1 = time.time(); sol[n] = screen_solid(body, n, changed); mf[n] = kit.mf(sol[n][0]); tick("solids (others)", t1)
            key = (n, a) if n < a else (a, n)
            if key in memo:                     # both parts scoped (unscoped run): the pair was measured from the other side
                v, d = memo[key]
            else:
                gn = mf[n].transform(Tn[:3, :4])
                v = (ga ^ gn).volume()
                d = 0.0 if v > 1e-3 else ga.min_gap(gn, 6.0)
                memo[key] = (v, d)
            if d < 6.0:
                out[f"{a} x {n}"] = dict(same_link=p["owning_link"] == body.parts[a]["owning_link"], overlap_mm3=round(v, 3),
                                         distance_mm=round(min(d, 6.0), 3), solid=sol[n][1])
    tick("gaps (standing)", t0)
    return dict(sorted(out.items(), key=lambda t: (t[1]["distance_mm"], -t[1]["overlap_mm3"])))


EXACT_UNDER_MM = 1.0   # pairs closer than this get the exact closest points and the face they lie on (native STEP, BRepExtrema)
EXACT_MAX_PAIRS = 8
EXACT_CALL_S = 30.0


def mesh_closest(ga, gb, d0=6.0):
    """(point on a, point on b, distance) of two posed trimeshes: surface samples (0.2 mm apart, at most 40k per part) of the
    faces inside both boxes widened by the pair's distance + 1 mm, nearest pair by KD tree, then three alternating projections
    onto the surfaces (trimesh.proximity on one point at a time), which slide the pair to the local closest points."""
    import trimesh
    from scipy.spatial import cKDTree
    lo = np.maximum(ga.bounds[0], gb.bounds[0]) - (d0 + 1.0)
    hi = np.minimum(ga.bounds[1], gb.bounds[1]) + (d0 + 1.0)
    pts = []
    for m in (ga, gb):
        c = m.triangles_center
        keep = np.all((c >= lo - 2.0) & (c <= hi + 2.0), axis=1) | np.any(np.all((m.triangles >= lo[None]) & (m.triangles <= hi[None]), axis=2), axis=1)
        if not keep.any():
            return None, None, np.inf
        sub = m.submesh([np.flatnonzero(keep)], append=True)
        n = int(min(40000, max(2000, sub.area / 0.04)))
        pts.append(np.vstack([trimesh.sample.sample_surface(sub, n, seed=1)[0], sub.vertices]))
    d, j = cKDTree(pts[1]).query(pts[0])
    i = int(np.argmin(d))
    pa, pb = pts[0][i], pts[1][j[i]]
    for _ in range(3):
        pb = trimesh.proximity.closest_point(gb, pa[None])[0][0]
        pa = trimesh.proximity.closest_point(ga, pb[None])[0][0]
    return pa, pb, float(np.linalg.norm(pa - pb))


def limiting_location(ctx, a, n, g):
    """Where the pair's standing distance is set: the overlap centroid when they overlap, else the closest point on each part
    (mesh, mm, world frame at standing); for pairs under EXACT_UNDER_MM with native STEP on both, the exact closest points and the
    face each lies on (surface type, face index in the part's STEP, face area) from BRepExtrema."""
    q = ctx.body.standing
    m = {}
    for x in (a, n):
        t = ctx.solid(x)[0].copy()
        t.apply_transform(ctx.body.link_T(ctx.link(x), q))
        m[x] = t
    if g["overlap_mm3"] > 1e-3:
        try:
            inter = kit.mf(m[a]) ^ kit.mf(m[n])
            return dict(kind="overlap centroid (mesh)", point_mm=np.round(kit.tm(inter).center_mass, 2).tolist())
        except Exception as exc:  # noqa: BLE001
            return dict(kind=f"overlap centroid not found: {type(exc).__name__}")
    pa, pb, d = mesh_closest(m[a], m[n], g["distance_mm"])
    if pa is None:
        return None
    return dict(kind="closest points (mesh)", point_a_mm=np.round(pa, 2).tolist(), point_b_mm=np.round(pb, 2).tolist(), distance_mm=round(d, 3))


def exact_location(ctx, a, n, shapes):
    """Exact closest points and faces of a and n at standing (brep.closest_points on the native STEP solids), in the world frame."""
    from checks import brep as L
    from checks.modules.fit import _shape
    from checks.core import FINE_STEP_MAX_MB, _mb
    stp = {x: ctx.parts[x].get("native_step") or ctx.parts[x].get("native_step_unmodified") for x in (a, n)}
    big = [x for x in (a, n) if stp[x] and _mb(stp[x]) > FINE_STEP_MAX_MB]
    if big:                                   # supplier boards: tens of MB of component solids, as screen_solid's coarse path
        return dict(kind=f"exact closest points not measured: STEP over {FINE_STEP_MAX_MB:g} MB ({', '.join(big)})")
    sa, sn = _shape(ctx, a, shapes), _shape(ctx, n, shapes)
    if sa is None or sn is None:
        return None
    q = ctx.body.standing
    Ta, Tn = ctx.body.link_T(ctx.link(a), q), ctx.body.link_T(ctx.link(n), q)
    try:
        r = L.bounded(lambda: L.closest_points(L.transform_shape(sa, Ta), L.transform_shape(sn, Tn)), EXACT_CALL_S)
    except (L.CallTimeout, RuntimeError) as e:
        return dict(kind=f"exact closest points not found ({str(e)[:60]})")
    if r.get("distance_mm") is None:
        return None
    r["distance_mm"] = round(r["distance_mm"], 4)
    return dict(kind="closest points (exact B-rep)", **r)


@register("gaps", lambda ctx: bool(ctx.scope))
def run(ctx):
    """Standing: every part within 6 mm of a scoped part: overlap (mm3) and distance (mm)."""
    for n in ctx.scope:
        ctx.solid(n)
    raw = check_gaps(ctx.body, ctx.scope, ctx.sol, ctx.mf)
    per_part, cases = {}, []
    for k, g in raw.items():
        a, n = k.split(" x ")
        per_part.setdefault(a, {})[n] = g
        cases.append(dict(part=a, mating_part=n, pose="standing", value=g["distance_mm"], unit="mm distance",
                          overlap_mm3=g["overlap_mm3"], what="gap at standing", location_mm=None))
    for c in cases[:5]:
        c["location_mm"] = ctx.locate(c["part"], c["mating_part"], "standing") if "standing" in [p[0] for p in ctx.poses] else None
    t0, shapes, n_exact = time.time(), {}, 0
    for k, g in raw.items():                  # the limiting location of every listed pair; the exact one for the closest pairs
        a, n = k.split(" x ")
        g["limiting_location"] = limiting_location(ctx, a, n, g)
        if g["overlap_mm3"] <= 1e-3 and g["distance_mm"] < EXACT_UNDER_MM and n_exact < EXACT_MAX_PAIRS:
            ex = exact_location(ctx, a, n, shapes)
            if ex and ex["kind"].startswith("closest points"):
                n_exact += 1
                g["limiting_location_exact"] = ex
            elif ex:
                g["limiting_location_exact_note"] = ex["kind"]
    tick("gaps (locations)", t0)
    for c in cases:
        g = raw[f"{c['part']} x {c['mating_part']}"]
        c["limiting_location"] = g.get("limiting_location_exact") or g.get("limiting_location")
    return dict(per_part=per_part, worst_cases=cases[:20],
                bad_events=dict(parts_overlapping_at_standing=sum(1 for g in raw.values() if g["overlap_mm3"] > 0.1)),
                evidence=[], notes="standing pose, search radius 6 mm; overlap = mesh boolean volume; limiting_location per pair: overlap centroid, else the closest point on "
                      f"each part (mesh vertex to surface); pairs under {EXACT_UNDER_MM} mm with native STEP on both (the {EXACT_MAX_PAIRS} closest) also get "
                      "limiting_location_exact: exact closest points and the face each lies on (surface type, index in the part's STEP faces, area), BRepExtrema")
