"""walls: per scoped printed part, wall thickness and gap width by ray casting on its solid, against the line width of
its print setup (process: nozzle -> printer line width; `--process nozzle_mm=...` cross-evaluates another nozzle).

Solid: the run's screening solid, except a mesh-only part's full closed mesh where the screening decimated it
(checks/print_solid.py). Samples are deterministic and do not depend on the order of the faces: every face is split into
k x k equal sub-triangles (k = ceil(sqrt(face area) / SPACING_MM), a pattern that is the same under any order of the
face's corners) and each sub-triangle's centre is one sample weighing its own area. So a part and its exact mirror read
alike, two runs read the same, and area figures are sums of exact sample areas (no random draw).

Wall: a ray from each sample inward along the inner normal to the first hit. The hit face says what the reading is:
facing back within FACING_DEG (30 deg) = a wall between two near-parallel faces, counted in the wall figures; 30-60 deg
= an edge reading (the ray ends on a fillet, a chamfer or the far face of a wedge / knife-edge tip: a sample 0.05 mm
above an r 0.4 fillet on the H-cage knee cheek reads 1.05 mm across a 1.26 mm wall), reported apart as the edge_*
figures, never dropped silently; over 60 deg = a grazing hit, dropped. Gap: the same outward to the part itself, up to
GAP_MAX_MM, with the same split. A wall thinner than one line width prints as one line or drops out; under two line
widths it has no infill and one perimeter per side at best; a gap narrower than a line width fuses shut.

A part whose line width does not resolve (no material / nozzle) is measured in mm but not against a line width:
`line_width_status` says why, its line-width figures are None, and the counts (parts_not_measured_no_line_width) and
worst cases name it. Measurements and counts only."""
from __future__ import annotations

import time

import numpy as np

from checks import print_solid as PS
from checks import process as P
from checks.core import tick
from checks.registry import register

SPACING_MM = 0.5
CAP = 600_000             # samples per part (the spacing grows when a part would take more)
K_MAX = 64                # sub-triangles per face edge
GAP_MAX_MM = 3.0
FACING_DEG = 30.0         # hit face within 30 deg of facing back: a wall / gap between near-parallel faces
GRAZE_DEG = 60.0          # 30-60 deg: an edge reading (fillet, chamfer, wedge tip); over 60: a grazing hit, dropped
EPS = 0.004               # ray origin offset off the surface (mm)


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n)]


def _cast(m, pts, dirs):
    """First hit distance per ray (inf = none) and the hit face index."""
    loc, ir, it = m.ray.intersects_location(pts, dirs, multiple_hits=False)
    d = np.full(len(pts), np.inf); f = np.full(len(pts), -1)
    d[ir] = np.linalg.norm(loc - pts[ir], axis=1); f[ir] = it
    return d, f


def samples(m, spacing=SPACING_MM, cap=CAP):
    """Deterministic surface samples of mesh m: (points, face index, area per sample). Each face is split into k x k equal
    sub-triangles, k = ceil(sqrt(area) / spacing) (at least 1, at most K_MAX); the sub-triangle centres are the samples,
    each weighing area / k^2. The pattern is the same for any order of a face's corners and any face order."""
    a = np.asarray(m.area_faces, float)
    while True:
        k = np.clip(np.ceil(np.sqrt(a) / spacing), 1, K_MAX).astype(int)
        if int((k * k).sum()) <= cap or spacing > 50:
            break
        spacing *= 1.25
    tri = m.triangles
    Ps, Fs, Ws = [], [], []
    for kk in np.unique(k):
        sel = np.flatnonzero(k == kk)
        uv = [((i + 1 / 3) / kk, (j + 1 / 3) / kk) for i in range(kk) for j in range(kk - i)]
        uv += [((i + 2 / 3) / kk, (j + 2 / 3) / kk) for i in range(kk) for j in range(kk - 1 - i)]
        uv = np.asarray(uv)
        t = tri[sel]
        e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        pts = t[:, None, 0] + uv[None, :, :1] * e1[:, None] + uv[None, :, 1:] * e2[:, None]
        Ps.append(pts.reshape(-1, 3)); Fs.append(np.repeat(sel, len(uv))); Ws.append(np.repeat(a[sel] / (kk * kk), len(uv)))
    return np.vstack(Ps), np.concatenate(Fs), np.concatenate(Ws)


def _wq(val, w, q):
    """Area-weighted quantile (q in 0..100) of the finite values; None when there are none."""
    ok = np.isfinite(val)
    if not ok.any():
        return None
    v, ww = val[ok], w[ok]
    o = np.argsort(v, kind="stable"); v, ww = v[o], ww[o]
    c = np.cumsum(ww)
    i = int(np.searchsorted(c, q / 100.0 * c[-1], side="left")) if q > 0 else 0
    return round(float(v[min(i, len(v) - 1)]), 3)


def _spots(pts, val, mask, k=5, sep=3.0):
    """Up to k worst (smallest val) sample points under mask, at least sep mm apart: [(value, point)]."""
    idx = np.flatnonzero(mask)
    idx = idx[np.lexsort((pts[idx, 2], pts[idx, 1], np.abs(pts[idx, 0]), val[idx]))]   # ties by position (|x|: mirror alike)
    out = []
    for i in idx:
        if all(np.linalg.norm(pts[i] - p) >= sep for _, p in out):
            out.append((float(val[i]), pts[i]))
            if len(out) == k:
                break
    return [dict(value_mm=round(v, 3), location_mm=np.round(p, 2).tolist()) for v, p in out]


def _split(m, fhit, dirs, d, limit=np.inf):
    """(face-to-face readings, edge readings) in mm, inf where none: the hit face within FACING_DEG of facing back along
    the ray, or between FACING_DEG and GRAZE_DEG. `dirs` is the direction the hit face's normal points for a face that
    looks straight back."""
    c = np.einsum("ij,ij->i", m.face_normals[np.maximum(fhit, 0)], dirs)
    hit = (fhit >= 0) & (d + EPS <= limit)
    face = hit & (c > np.cos(np.radians(FACING_DEG)))
    edge = hit & ~face & (c > np.cos(np.radians(GRAZE_DEG)))
    return np.where(face, d + EPS, np.inf), np.where(edge, d + EPS, np.inf)


def measure(m, lw):
    """Wall / gap measurements of closed mesh m (mm) against line width lw (mm, may be None)."""
    pts, fi, w = samples(m)
    nrm = m.face_normals[fi]
    dw, fw = _cast(m, pts - EPS * nrm, -nrm)
    wall, wedge = _split(m, fw, -nrm, dw)                 # the far wall's outward normal points along the inward ray
    dg, fg = _cast(m, pts + EPS * nrm, nrm)
    gap, gedge = _split(m, fg, -nrm, dg, GAP_MAX_MM)       # the far side of a gap looks back at the sample
    area = lambda mask: round(float(w[mask].sum()), 1)     # noqa: E731
    out = dict(samples=int(len(pts)), sampling="k x k sub-triangle centres per face, area-weighted (deterministic)",
               mean_area_per_sample_mm2=round(float(w.mean()), 4),
               wall_min_mm=_wq(wall, w, 0), wall_p1_mm=_wq(wall, w, 1), wall_p5_mm=_wq(wall, w, 5), wall_median_mm=_wq(wall, w, 50),
               edge_wall_min_mm=_wq(wedge, w, 0), edge_readings_mm2=area(np.isfinite(wedge)),
               gap_min_mm=_wq(gap, w, 0), gap_samples=int(np.isfinite(gap).sum()), edge_gap_min_mm=_wq(gedge, w, 0))
    if lw:
        out.update(wall_under_1_line_mm2=area(wall < lw), wall_under_2_lines_mm2=area(wall < 2 * lw),
                   edge_wall_under_1_line_mm2=area(wedge < lw), edge_wall_under_2_lines_mm2=area(wedge < 2 * lw),
                   gap_under_1_line_mm2=area(gap < lw), gap_under_2_lines_mm2=area(gap < 2 * lw),
                   edge_gap_under_1_line_mm2=area(gedge < lw),
                   thinnest_walls=_spots(pts, wall, wall < 2 * lw), narrowest_gaps=_spots(pts, gap, gap < 2 * lw),
                   thinnest_edge_readings=_spots(pts, wedge, wedge < 2 * lw, k=3))
    else:
        out.update(wall_under_1_line_mm2=None, wall_under_2_lines_mm2=None, gap_under_1_line_mm2=None, gap_under_2_lines_mm2=None,
                   thinnest_walls=_spots(pts, wall, np.isfinite(wall)))
    return out


def _lw_status(e):
    if e["line_width_mm"]:
        return f"resolved ({e['source']['line_width_mm']})"
    miss = [k for k in ("printer", "material", "nozzle_mm") if not e.get(k)]
    return ("not measured against a line width: " + ", ".join(miss) + " missing" if miss else
            f"not measured against a line width: printer {e['printer']} gives none for nozzle {e['nozzle_mm']}")


def check_part(ctx, n):
    t0 = time.time()
    e = P.effective(ctx, n)
    m, how = PS.solid(ctx, n)
    out = dict(material=e["material"], nozzle_mm=e["nozzle_mm"], line_width_mm=e["line_width_mm"],
               line_width_source=e["source"]["line_width_mm"], line_width_status=_lw_status(e), solid=how, **measure(m, e["line_width_mm"]))
    out["seconds"] = round(time.time() - t0, 1)
    tick("walls", t0)
    return out


@register("walls", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Per printed part: wall thickness and gap width (ray casting) against its line width; thinnest spots with locations."""
    per, cases, not_measured = {}, [], []
    ev = dict(parts=0, parts_without_line_width=0, parts_not_measured_no_line_width=0, parts_with_walls_under_1_line=0,
              parts_with_walls_under_2_lines=0, parts_with_gaps_under_1_line=0, parts_with_edge_readings_under_2_lines=0)
    for n in _targets(ctx):
        r = check_part(ctx, n)
        per[n] = r
        ev["parts"] += 1
        if not r["line_width_mm"]:
            ev["parts_without_line_width"] += 1
            ev["parts_not_measured_no_line_width"] += 1
            not_measured.append(n)
            w = (r.get("thinnest_walls") or [{}])[0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=w.get("location_mm"), value=r["wall_p1_mm"],
                              unit="mm wall p1 (NOT measured against a line width)",
                              what=f"{r['line_width_status']}; wall min {r['wall_min_mm']}, p1 {r['wall_p1_mm']}, p5 {r['wall_p5_mm']}, "
                                   f"median {r['wall_median_mm']} mm; set the part's process (material / nozzle) to read it in line widths"))
            continue
        lw = r["line_width_mm"]
        ev["parts_with_walls_under_1_line"] += r["wall_under_1_line_mm2"] > 0.5
        ev["parts_with_walls_under_2_lines"] += r["wall_under_2_lines_mm2"] > 0.5
        ev["parts_with_gaps_under_1_line"] += r["gap_under_1_line_mm2"] > 0.5
        ev["parts_with_edge_readings_under_2_lines"] += r["edge_wall_under_2_lines_mm2"] > 0.5
        if r["thinnest_walls"]:
            w = r["thinnest_walls"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=w["location_mm"], value=w["value_mm"], unit="mm wall",
                              what=f"{r['wall_under_1_line_mm2']} mm2 under 1 line ({lw} mm), {r['wall_under_2_lines_mm2']} mm2 under 2 lines; "
                                   f"edge readings (fillet / chamfer / wedge tip) apart: {r['edge_wall_under_2_lines_mm2']} mm2 under 2 lines; nozzle {r['nozzle_mm']} mm"))
        if r["narrowest_gaps"]:
            g = r["narrowest_gaps"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=g["location_mm"], value=g["value_mm"], unit="mm gap",
                              what=f"{r['gap_under_1_line_mm2']} mm2 of gap under 1 line ({lw} mm) fuses; nozzle {r['nozzle_mm']} mm"))
    cases.sort(key=lambda c: (c["value"] is None, c["value"] if c["value"] is not None else 0.0))
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[], parts_not_measured_no_line_width=not_measured,
                notes=f"ray casting (embree) from deterministic area-weighted samples (k x k sub-triangle centres per face, k = ceil(sqrt(area) / "
                      f"{SPACING_MM} mm), cap {CAP}) along the inner / outer normal; a reading counts as a wall / gap when the hit face looks back "
                      f"within {FACING_DEG:g} deg, as an edge reading (fillet, chamfer, wedge tip; edge_* figures) at {FACING_DEG:g}-{GRAZE_DEG:g} deg, "
                      f"dropped beyond; gaps up to {GAP_MAX_MM} mm; mesh-only parts on their full closed mesh; locations in the manifest frame; "
                      f"line width from the part's print setup (printer default per nozzle unless set); parts without one are listed in "
                      f"parts_not_measured_no_line_width")
