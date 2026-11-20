"""walls: per scoped printed part, wall thickness and gap width by ray casting on its solid, against the line width of
its print setup (process: nozzle -> printer line width; `--process nozzle_mm=...` cross-evaluates another nozzle).

Points are sampled evenly over the surface (about one per `spacing_mm`^2, capped). Wall: a ray from each point inward
along the inner normal to the first hit, kept when the hit face looks back along the ray (|angle| < 60 deg), which drops
grazing hits at edges and corners. Gap: the same outward, to the part itself, up to GAP_MAX_MM. A wall thinner than one
line width prints as one line or drops out; under two line widths it has no infill and one perimeter per side at best; a
gap narrower than a line width fuses shut. Area figures are sample counts x area per sample. Measurements and counts only."""
from __future__ import annotations

import time

import numpy as np

from checks import process as P
from checks.core import tick
from checks.registry import register

SPACING_MM = 0.5
CAP = 60_000
GAP_MAX_MM = 3.0
COS_FACING = 0.5          # hit face normal within 60 deg of the ray (opposite wall, not a grazing edge)
EPS = 0.004               # ray origin offset off the surface (mm)


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n)]


def _cast(m, pts, dirs):
    """First hit distance per ray (inf = none) and the hit face index."""
    loc, ir, it = m.ray.intersects_location(pts, dirs, multiple_hits=False)
    d = np.full(len(pts), np.inf); f = np.full(len(pts), -1)
    d[ir] = np.linalg.norm(loc - pts[ir], axis=1); f[ir] = it
    return d, f


def _spots(pts, val, mask, k=5, sep=3.0):
    """Up to k worst (smallest val) sample points under mask, at least sep mm apart: [(value, point)]."""
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(val[idx])]
    out = []
    for i in idx:
        if all(np.linalg.norm(pts[i] - p) >= sep for _, p in out):
            out.append((float(val[i]), pts[i]))
            if len(out) == k:
                break
    return [dict(value_mm=round(v, 3), location_mm=np.round(p, 2).tolist()) for v, p in out]


def measure(m, lw):
    """Wall / gap measurements of closed mesh m (mm) against line width lw (mm, may be None)."""
    import trimesh
    cnt = int(min(CAP, max(2000, m.area / SPACING_MM ** 2)))
    pts, fi = trimesh.sample.sample_surface(m, cnt, seed=7)
    nrm = m.face_normals[fi]
    per = float(m.area) / cnt
    dw, fw = _cast(m, pts - EPS * nrm, -nrm)
    okw = (fw >= 0) & (np.einsum("ij,ij->i", m.face_normals[np.maximum(fw, 0)], -nrm) > COS_FACING)
    wall = np.where(okw, dw + EPS, np.inf)
    dg, fg = _cast(m, pts + EPS * nrm, nrm)
    okg = (fg >= 0) & (np.einsum("ij,ij->i", m.face_normals[np.maximum(fg, 0)], nrm) < -COS_FACING) & (dg + EPS <= GAP_MAX_MM)
    gap = np.where(okg, dg + EPS, np.inf)
    fin = np.isfinite(wall)
    q = (lambda a, p: round(float(np.percentile(a, p)), 3) if len(a) else None)
    out = dict(samples=cnt, area_per_sample_mm2=round(per, 3),
               wall_min_mm=q(wall[fin], 0), wall_p1_mm=q(wall[fin], 1), wall_p5_mm=q(wall[fin], 5), wall_median_mm=q(wall[fin], 50),
               gap_min_mm=q(gap[np.isfinite(gap)], 0), gap_samples=int(np.isfinite(gap).sum()))
    if lw:
        out.update(wall_under_1_line_mm2=round(float((wall < lw).sum() * per), 1),
                   wall_under_2_lines_mm2=round(float((wall < 2 * lw).sum() * per), 1),
                   gap_under_1_line_mm2=round(float((gap < lw).sum() * per), 1),
                   gap_under_2_lines_mm2=round(float((gap < 2 * lw).sum() * per), 1),
                   thinnest_walls=_spots(pts, wall, wall < 2 * lw), narrowest_gaps=_spots(pts, gap, gap < 2 * lw))
    return out


def check_part(ctx, n):
    t0 = time.time()
    e = P.effective(ctx, n)
    m, how = ctx.solid(n)
    out = dict(material=e["material"], nozzle_mm=e["nozzle_mm"], line_width_mm=e["line_width_mm"],
               line_width_source=e["source"]["line_width_mm"], solid=how, **measure(m, e["line_width_mm"]))
    out["seconds"] = round(time.time() - t0, 1)
    tick("walls", t0)
    return out


@register("walls", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Per printed part: wall thickness and gap width (ray casting) against its line width; thinnest spots with locations."""
    per, cases = {}, []
    ev = dict(parts=0, parts_without_line_width=0, parts_with_walls_under_1_line=0, parts_with_walls_under_2_lines=0, parts_with_gaps_under_1_line=0)
    for n in _targets(ctx):
        r = check_part(ctx, n)
        per[n] = r
        ev["parts"] += 1
        if not r["line_width_mm"]:
            ev["parts_without_line_width"] += 1
            continue
        lw = r["line_width_mm"]
        ev["parts_with_walls_under_1_line"] += r["wall_under_1_line_mm2"] > 0.5
        ev["parts_with_walls_under_2_lines"] += r["wall_under_2_lines_mm2"] > 0.5
        ev["parts_with_gaps_under_1_line"] += r["gap_under_1_line_mm2"] > 0.5
        if r["thinnest_walls"]:
            w = r["thinnest_walls"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=w["location_mm"], value=w["value_mm"], unit="mm wall",
                              what=f"{r['wall_under_1_line_mm2']} mm2 under 1 line ({lw} mm), {r['wall_under_2_lines_mm2']} mm2 under 2 lines; nozzle {r['nozzle_mm']} mm"))
        if r["narrowest_gaps"]:
            g = r["narrowest_gaps"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=g["location_mm"], value=g["value_mm"], unit="mm gap",
                              what=f"{r['gap_under_1_line_mm2']} mm2 of gap under 1 line ({lw} mm) fuses; nozzle {r['nozzle_mm']} mm"))
    cases.sort(key=lambda c: c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"ray casting (embree) from surface samples (~1 per {SPACING_MM} mm2, cap {CAP}) along the inner / outer normal; hits "
                      f"kept when the hit face looks back within 60 deg; gaps up to {GAP_MAX_MM} mm; locations in the manifest frame; "
                      f"line width from the part's print setup (printer default per nozzle unless set)")
