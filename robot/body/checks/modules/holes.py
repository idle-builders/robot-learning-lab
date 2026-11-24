"""holes: hole inventory of each scoped printed part from its native STEP, with the hardware through each hole.

Holes = concave cylindrical faces of the B-rep grouped by axis line and radius, kept when they wrap at least HOLE_SPAN_DEG
around the axis (fillets and slots are partial cylinders); a bore of 180-300 deg with a sleeve or screw in it is kept as a
partial bore (a teardrop, a slot or a breakout). Per hole: diameter, axis, depth, centre (manifest frame) and
the axis angle to the print direction (0 = vertical hole, printed as stacked rings; 90 = across the layers: the top sags,
the hole prints oval unless teardropped or reamed). Hardware matched by coaxiality (axis within 3 deg, lines within
0.3 mm): steel sleeves (press interference = sleeve outer diameter - hole diameter, also after the printer's hole offset
when a coupon has set it) and screws (hole diameter as % of the thread's major diameter: a pilot for self-tapping screws,
a clearance hole otherwise). Boss wall: around every sleeve and screw hole, the thinnest material from the hole surface
outward (24 radial rays x 5 depths into the part's solid; rays that start in air, in a teardrop tip, slot or breakout, are
the open share), on holes the hardware sits in; a coaxial hole at least 1.5 x the thread
major (or 90 % of the head) is the head's recess (counterbore / spot face), listed with no boss wall. A partial bore whose open rays
run into a roof close to the circle, pointing up in the print direction, is a teardrop (role "sleeve bore (partial: teardrop)",
`teardrop` on the row) and is not counted as a hole that sags across the layers; else it is a slot or breakout. A printed part
without a STEP is measured on its closed mesh: circle fits along the axes of the sleeves and screws in it (mesh_holes); an open
mesh is listed as not measured. Measurements and counts only."""
from __future__ import annotations

import json
import math
import time

import numpy as np

from checks import bodylib as BL
from checks import process as P
from checks.core import DATA, tick
from checks.registry import register

HOLE_SPAN_DEG = 300.0
PARTIAL_SPAN_DEG = 180.0      # bores wrapping 180-300 deg break out of the part: kept, flagged partial
AX_TOL_DEG, LINE_TOL_MM = 3.0, 0.3
BOSS_MIN_DEPTH_MM = 1.0       # shallower recesses (spot faces, counterbores) have no boss wall to speak of
RECESS_OVER_MAJOR = 1.5       # a screw's coaxial hole this much wider than its thread (or 90 % of its head) is a head recess, not a clearance hole
TIP_MM_OVER_R = 0.6           # an open boss-wall ray that meets material within 0.6 r of the hole surface is in a teardrop's tip (roof <= 0.41 r away)
TEARDROP_TIP_SHARE = 0.8      # ... and a partial bore is a teardrop when 80 % of its open rays are tip rays
TEARDROP_UP_DEG = 30.0        # ... pointing within 30 deg of the print direction (when the part has one)


def _printed(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n)]


def _closed_mesh(ctx, n):
    """Part n's own mesh, not decimated, when it is a closed manifold (core.full_closed_mesh, once per run); else None."""
    cache = ctx.__dict__.setdefault("_holes_closed", {})
    if n not in cache:
        from checks import core
        cache[n] = core.full_closed_mesh(ctx.body, n) if ctx.parts[n].get("mesh") else None
    return cache[n]


def _has_step(ctx, n):
    return bool(ctx.parts[n].get("native_step")) and BL.src(ctx.parts[n]["native_step"]).exists()


def _targets(ctx):
    """Printed parts with a native STEP (B-rep cylinders), and mesh-only printed parts whose own mesh is closed (circle fits)."""
    return [n for n in _printed(ctx) if _has_step(ctx, n) or (not ctx.parts[n].get("native_step") and _closed_mesh(ctx, n) is not None)]


def not_measured(ctx):
    """{part: why} for the printed parts in scope this check cannot measure: a STEP path that resolves nowhere, or no STEP and an
    open mesh (bores are fitted on a closed one)."""
    out = {}
    for n in _printed(ctx):
        st = ctx.parts[n].get("native_step")
        if st and not BL.src(st).exists():
            out[n] = f"native STEP resolves nowhere ({st})"
        elif not st and _closed_mesh(ctx, n) is None:
            out[n] = "no native STEP and its mesh is not closed: bores are fitted on a closed mesh"
    return out


def brep_holes(step, offset):
    """[{d_mm, axis, centre_mm, depth_mm, span_deg}] of the concave cylinders of a STEP (manifest frame = STEP + offset)."""
    from build123d import Location, import_step
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.gp import gp_Pnt, gp_Vec
    sh = import_step(str(step)).moved(Location(tuple(float(v) for v in (offset or (0, 0, 0)))))
    cyl = []
    for f in sh.faces():
        s = BRepAdaptor_Surface(f.wrapped, True)
        if s.GetType() != GeomAbs_Cylinder:
            continue
        c = s.Cylinder(); ax = c.Axis(); p0 = ax.Location(); dd = ax.Direction()
        a = np.array([dd.X(), dd.Y(), dd.Z()]); o = np.array([p0.X(), p0.Y(), p0.Z()])
        u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
        pm, vn = gp_Pnt(), gp_Vec()
        BRepGProp_Face(f.wrapped).Normal(0.5 * (u0 + u1), 0.5 * (v0 + v1), pm, vn)
        pmv = np.array([pm.X(), pm.Y(), pm.Z()]); rv = (pmv - o) - np.dot(pmv - o, a) * a
        if float(np.dot([vn.X(), vn.Y(), vn.Z()], rv)) >= 0:        # normal points away from the axis: a boss, not a hole
            continue
        bb = f.bounding_box()
        corners = np.array([[bb.min.X, bb.min.Y, bb.min.Z], [bb.max.X, bb.max.Y, bb.max.Z]])
        cyl.append(dict(r=float(c.Radius()), a=a, o=o, span=math.degrees(abs(u1 - u0)), t=sorted(((corners - o) @ a).tolist())))
    holes = []
    for c in cyl:
        for h in holes:
            if abs(h["r"] - c["r"]) < 1e-3 and abs(abs(float(h["a"] @ c["a"])) - 1) < 1e-4 and \
                    np.linalg.norm((c["o"] - h["o"]) - np.dot(c["o"] - h["o"], h["a"]) * h["a"]) < 1e-3:
                ct = sorted(((np.array(c["t"]) + float((c["o"] - h["o"]) @ h["a"]))).tolist())     # c's range on h's axis
                if ct[0] <= max(h["t"]) + 0.05 and ct[-1] >= min(h["t"]) - 0.05:              # same line AND touching: one hole
                    h["span"] += c["span"]; h["t"] += ct
                    break
        else:
            holes.append(dict(c, t=list(c["t"])))
    out = []
    for h in holes:
        if h["span"] < PARTIAL_SPAN_DEG:
            continue
        t0, t1 = min(h["t"]), max(h["t"])
        out.append(dict(d_mm=round(2 * h["r"], 3), axis=np.round(h["a"], 4).tolist(), centre_mm=np.round(h["o"] + 0.5 * (t0 + t1) * h["a"], 3).tolist(),
                        depth_mm=round(t1 - t0, 3), span_deg=round(min(h["span"], 360.0), 1),
                        partial=h["span"] < HOLE_SPAN_DEG))
    return out


FIT_STEP_MM, FIT_RMS_MM, FIT_MIN_RAYS = 0.2, 0.05, 12     # mesh bores: station step, circle-fit rms limit, rays on the circle (of 24)


def _fit_station(m, o, a, u, w, r_max):
    """Circle fitted to where 24 rays from o (on axis a) outward first enter material (within r_max): (centre, r, rms, rays kept)
    or None. Rays that start in material, miss, or sit more than 0.1 mm off the circle (a teardrop tip, a slot) are dropped and the
    circle refitted, so a partial bore keeps the arc it has."""
    ang = np.radians(np.arange(0, 360, 15))
    dirs = np.array([math.cos(t) * u + math.sin(t) * w for t in ang])
    loc, ir, it = m.ray.intersects_location(np.repeat(o[None], 24, 0), dirs, multiple_hits=False)
    ok = (np.einsum("ij,ij->i", m.face_normals[it], dirs[ir]) < 0) & (np.linalg.norm(loc - o, axis=1) <= r_max)
    q = loc[ok] - o
    xy = np.c_[q @ u, q @ w]
    keep = np.ones(len(xy), bool)
    for _ in range(3):
        if keep.sum() < FIT_MIN_RAYS:
            return None
        A = np.c_[2 * xy[keep], np.ones(keep.sum())]
        sol = np.linalg.lstsq(A, (xy[keep] ** 2).sum(1), rcond=None)[0]
        cc, rr = sol[:2], math.sqrt(max(sol[2] + float(sol[:2] @ sol[:2]), 0.0))
        res = np.abs(np.linalg.norm(xy - cc, axis=1) - rr)
        new = res < 0.1
        if (new == keep).all():
            break
        keep = new
    if keep.sum() < FIT_MIN_RAYS:
        return None
    rms = float(np.sqrt((res[keep] ** 2).mean()))
    return (o + cc[0] * u + cc[1] * w, rr, rms, int(keep.sum())) if rms <= FIT_RMS_MM else None


def mesh_holes(m, axes):
    """Bores of a part without a B-rep, fitted on its closed mesh along the axes of the hardware in it (sleeves, screws:
    [(centre, unit axis, length)]): stations every FIT_STEP_MM over the hardware's length + 3 mm each side, a circle fitted
    per station (_fit_station); contiguous stations of one radius (within 0.1 mm) are one hole. Same rows as brep_holes plus
    the fit's rms. Holes without hardware in them are not found (no axis to search along). Gaps of up to 2 stations without a fit
    stay inside a run."""
    out = []
    for c, a, L in axes:
        a = np.asarray(a, float) / np.linalg.norm(a); c = np.asarray(c, float)
        u = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(a, u)
        ss = np.arange(-L / 2 - 3.0, L / 2 + 3.0 + 1e-9, FIT_STEP_MM)
        st = [(t, _fit_station(m, c + t * a, a, u, w, 10.0)) for t in ss]
        runs, cur, gap = [], [], 0
        for t, f in st:      # a run of one radius goes on over up to 2 stations without a fit (a layer step, a vent, a mesh crease)
            if f is not None and (not cur or abs(f[1] - cur[-1][1][1]) < 0.1):
                cur.append((t, f)); gap = 0; continue
            if f is None and cur and gap < 2:
                gap += 1; continue
            if cur:
                runs.append(cur)
            cur, gap = ([(t, f)] if f is not None else []), 0
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) < 2:
                continue
            t0, t1 = run[0][0] - FIT_STEP_MM / 2, run[-1][0] + FIT_STEP_MM / 2
            cen = np.mean([f[0] for _, f in run], 0) - np.mean([t for t, _ in run]) * a + 0.5 * (t0 + t1) * a
            r = float(np.median([f[1] for _, f in run])); span = 15.0 * float(np.median([f[3] for _, f in run]))
            h = dict(d_mm=round(2 * r, 3), axis=np.round(a, 4).tolist(), centre_mm=np.round(cen, 3).tolist(), depth_mm=round(t1 - t0, 3),
                     span_deg=round(span, 1), partial=span < HOLE_SPAN_DEG, fit_rms_mm=round(float(np.max([f[2] for _, f in run])), 4), source="mesh circle fit")
            if not any(abs(o["d_mm"] - h["d_mm"]) < 0.1 and _coaxial(o["centre_mm"], np.asarray(o["axis"]), h["centre_mm"], a)
                       and abs(float((np.asarray(o["centre_mm"]) - cen) @ a)) < (o["depth_mm"] + h["depth_mm"]) / 2 for o in out):
                out.append(h)
    return out


def _coaxial(c1, a1, c2, a2):
    if abs(abs(float(np.dot(a1, a2))) - 1) > 1 - math.cos(math.radians(AX_TOL_DEG)):
        return False
    v = np.asarray(c2) - np.asarray(c1)
    return np.linalg.norm(v - np.dot(v, a1) * a1) < LINE_TOL_MM


def boss_wall(m, hole, window=None, up=None):
    """Thinnest material (mm) from the hole surface radially outward: 24 angles x 5 depths over 90 % of the length, first exit hit,
    and the share of those rays (%) that start in air: the open sector of a teardrop, a slot or a bore that breaks out. A ray
    whose first hit enters the part (or that hits nothing) started in air and measures no wall; the thinnest wall is over the
    rays that leave the material, so a breakout still reads the thin edge beside its opening. window (centre, length) limits
    the depths to an axial stretch (a sleeve bore: where the sleeve sits, not where the bore runs on past it into a slot).
    The open rays are told apart: a ray that enters material again within TIP_MM_OVER_R x r of the hole surface is in a
    teardrop's tip (its roof is at most (sqrt 2 - 1) r from the circle); one that hits nothing that close runs out of the
    part (a slot or a breakout). With the print direction `up`, the angle between up (across the axis) and the mean open
    direction is given too (a teardrop points up). Returns (wall mm or None, location, open %, open detail dict)."""
    a = np.asarray(hole["axis"], float); r = hole["d_mm"] / 2
    c, depth = (np.asarray(window[0], float), window[1]) if window else (np.asarray(hole["centre_mm"], float), hole["depth_mm"])
    c = np.asarray(hole["centre_mm"], float) + float((c - np.asarray(hole["centre_mm"], float)) @ a) * a   # on the hole's axis
    u = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(a, u)
    ang = np.radians(np.arange(0, 360, 15))
    dirs = np.array([math.cos(t) * u + math.sin(t) * w for t in ang])
    org = np.vstack([c + s * depth * a + (r + 0.02) * dirs for s in (-0.45, -0.225, 0.0, 0.225, 0.45)])
    dd = np.vstack([dirs] * 5)
    loc, ir, it = m.ray.intersects_location(org, dd, multiple_hits=False)
    leaves = np.zeros(len(org), bool)
    leaves[ir] = np.einsum("ij,ij->i", m.face_normals[it], dd[ir]) > 0      # the first hit exits the material: the ray started in it
    hitd = np.full(len(org), np.inf)
    hitd[ir] = np.linalg.norm(loc - org[ir], axis=1) + 0.02
    open_ = ~leaves
    tip = open_ & (hitd <= TIP_MM_OVER_R * r)
    open_pct = round(100.0 * float(open_.mean()), 1)
    detail = dict(open_tip_pct=round(100.0 * float(tip.mean()), 1), open_out_pct=round(100.0 * float((open_ & ~tip).mean()), 1), open_to_up_deg=None)
    if open_.any() and up is not None:
        upp = np.asarray(up, float) - float(np.asarray(up, float) @ a) * a
        mv = dd[open_].mean(0)
        if np.linalg.norm(upp) > 1e-6 and np.linalg.norm(mv) > 1e-6:
            detail["open_to_up_deg"] = round(math.degrees(math.acos(np.clip(float(mv @ upp) / np.linalg.norm(mv) / np.linalg.norm(upp), -1, 1))), 1)
    if not leaves.any():
        return None, None, open_pct, detail
    dist = np.where(leaves, hitd, np.inf)
    i = int(np.argmin(dist))
    return round(float(dist[i]), 3), np.round(org[i], 2).tolist(), open_pct, detail


def is_teardrop(detail, open_pct):
    """A partial bore is a teardrop when its open rays run into a roof close to the circle (most of them, TEARDROP_TIP_SHARE)
    and, with a print direction, their mean direction is within TEARDROP_UP_DEG of up; else a slot or a breakout."""
    if not open_pct or detail["open_tip_pct"] < TEARDROP_TIP_SHARE * open_pct:
        return False
    return detail["open_to_up_deg"] is None or detail["open_to_up_deg"] <= TEARDROP_UP_DEG


def check_part(ctx, n, hw, prn):
    import trimesh
    from checks.core import sha  # noqa: F401  (keeps import order with core)
    t0 = time.time()
    p = ctx.parts[n]
    e = P.effective(ctx, n)
    up = P.up_vector(e["up"]) if e["up"] else None
    from_step = _has_step(ctx, n)
    if from_step:
        m = ctx.solid(n)[0]
    else:
        m = _closed_mesh(ctx, n)                  # the full closed mesh (never the screening decimation): bores are fitted on it
    off = (prn.get(e["printer"] or "", {}) or {}).get("hole_offset_mm")
    # sleeves of this part: the sleeve sets pressed into it (ctx.rel: definition joined_to {how: pressed}, else the sets touching it)
    sets = ctx.rel.sleeve_sets_of(n)
    import sys
    from checks.core import ROOT
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from mount_check import screw_geometry, screw_spec, tubes as _tubes
    tubes = [t for sn in sets for t in _tubes(BL.load_mesh(ctx.parts[sn]["mesh"]))]
    # screws near the part: hardware its definition calls a screw (spec.kind) or a builder placed as one (screw_plan); the thread from
    # the screw's spec (mount_check.screw_spec: screw_plan, else the definition), never from its name; a screw without one is listed
    b = np.array(p["bounds_mm"], float)
    screws, no_thread, unknown = [], [], []
    for k, q in ctx.parts.items():
        if not ctx.rel.is_fastener(k) or ctx.rel.is_sleeve_set(k):
            continue
        qb = np.array(q["bounds_mm"], float)
        if np.any(qb[0] > b[1] + 1) or np.any(qb[1] < b[0] - 1):
            continue
        if not (ctx.rel.hardware_kind(k) == "screw" or q.get("screw_plan")):
            if ctx.rel.hardware_kind(k) is None:
                unknown.append(k)          # hardware whose definition does not say what it is: not matched to the holes
            continue
        try:
            g = screw_geometry(BL.load_mesh(q["mesh"]))
        except Exception:  # noqa: BLE001
            continue
        sp = screw_spec(q)
        th = sp.get("thread") if sp.get("spec_source") in ("screw_plan", "definition") else None
        screws.append((k, th, g))
    if from_step:
        holes = brep_holes(BL.src(p["native_step"]), p.get("native_step_offset_mm"))
    else:
        holes = mesh_holes(m, [(c, ta, L) for c, ta, L, od, idd in tubes]
                           + [(g["_head"] + 0.5 * g["length_mm"] * g["_d"], g["_d"], g["length_mm"]) for _, _, g in screws])
    rows = []
    for h in holes:
        a = np.asarray(h["axis"], float)
        r = dict(h)
        r["axis_to_print_up_deg"] = None if up is None else round(math.degrees(math.acos(min(1.0, abs(float(a @ up))))), 1)
        hc = np.asarray(h["centre_mm"], float)
        best = None                                   # the coaxial sleeve that overlaps this hole most along the axis
        for c, ta, L, od, idd in tubes:
            if _coaxial(hc, a, c, ta):
                ov = h["depth_mm"] / 2 + L / 2 - abs(float((np.asarray(c) - hc) @ a))
                if best is None or ov > best[0]:
                    best = (ov, od, idd, c, L)
        if best is not None:
            ov, od, idd, sc, sL = best
            if ov > 0.2 and abs(od - h["d_mm"]) < 1.0:
                r.update(role="sleeve bore" + (" (partial: teardrop, slot or breakout)" if h["partial"] else ""), sleeve_window=(sc, sL), sleeve_od_mm=round(od, 3), sleeve_id_mm=round(idd, 3), press_interference_mm=round(od - h["d_mm"], 3),
                         press_interference_printed_mm=None if off is None else round(od - (h["d_mm"] + off), 3))
            elif ov > 0.2:
                r["role"] = "around a sleeve (counterbore / access)"
        sbest = None                                  # the coaxial screw whose shank passes through this hole
        for k, th, g in screws:
            if _coaxial(hc, a, g["_head"], g["_d"]):
                tt = float((hc - g["_head"]) @ g["_d"])  # hole centre along the screw from its head
                ov = min(tt + h["depth_mm"] / 2, g["length_mm"]) - max(tt - h["depth_mm"] / 2, 0.0)
                if sbest is None or ov > sbest[0]:
                    sbest = (ov, k, th)
        if sbest is not None and sbest[0] > 0.1:
            _, k, th = sbest
            maj = hw["threads"].get(th)
            r.update(screw=k, thread=th, screw_major_mm=maj)
            if th is None and k not in no_thread:
                no_thread.append(k)
            head_d = 2 * next((g["r_head_mm"] for kk, _, g in screws if kk == k), 0.0)
            if maj and r.get("role") is None and h["d_mm"] >= min(RECESS_OVER_MAJOR * maj, 0.9 * head_d or 1e9):
                r.update(role="head recess (counterbore / spot face)" + (" (partial: breaks out of the surface)" if h["partial"] else ""),
                         screw_head_d_mm=round(head_d, 3) or None)
            elif maj and r.get("role") is None:
                if h["d_mm"] < maj:
                    r.update(role="pilot" if th in hw["self_tapping"] else "thread-forming hole", self_tapping=th in hw["self_tapping"],
                             hole_pct_of_major=round(100 * h["d_mm"] / maj, 1))
                else:
                    r.update(role="clearance hole", clearance_mm=round(h["d_mm"] - maj, 3))
        win = r.pop("sleeve_window", None)
        if win is not None:     # where the sleeve sits, clipped to the bore
            lo = max(float((np.asarray(win[0]) - hc) @ a) - win[1] / 2, -h["depth_mm"] / 2); hi = min(float((np.asarray(win[0]) - hc) @ a) + win[1] / 2, h["depth_mm"] / 2)
            win = (hc + 0.5 * (lo + hi) * a, max(hi - lo, 0.1))
        # boss wall only where the hardware itself sits (sleeve bore, pilot, clearance hole); a head recess or a counterbore around a
        # sleeve is a seat for the head, and one cut into a curved surface breaks out without that being a missing wall
        carries = "sleeve_od_mm" in r or r.get("role") in ("pilot", "thread-forming hole", "clearance hole")
        # a partial bore (teardrop, slot, breakout) is measured the same way: its open sector starts in air and reads as open_pct
        if carries and (win[1] if win else h["depth_mm"]) >= BOSS_MIN_DEPTH_MM:
            r["boss_wall_mm"], r["boss_wall_location_mm"], r["boss_wall_open_pct"], det = boss_wall(m, h, win, up)
            r.update(boss_wall_open_detail=det)
            if h["partial"]:
                r["teardrop"] = is_teardrop(det, r["boss_wall_open_pct"])
                if str(r.get("role", "")).startswith("sleeve bore"):
                    r["role"] = "sleeve bore (partial: " + ("teardrop)" if r["teardrop"] else "slot or breakout)")
        if h["partial"] and "sleeve_od_mm" not in r and "screw" not in r:
            continue                                                                          # an arc of a slot or fillet, not a hole
        rows.append(r)
    out = dict(source="B-rep cylinders of the native STEP" if from_step else f"circle fits on the closed mesh along the hardware axes (rms <= {FIT_RMS_MM} mm, "
               f">= {FIT_MIN_RAYS} of 24 rays, every {FIT_STEP_MM} mm; holes without hardware are not found)",
               up=e["up"], printer=e["printer"], hole_offset_mm=off, holes=len(rows),
               horizontal_holes=sum(1 for r in rows if (r["axis_to_print_up_deg"] or 0) > 60),
               horizontal_holes_with_hardware=sum(1 for r in rows if (r["axis_to_print_up_deg"] or 0) > 60 and ("sleeve_od_mm" in r or "screw" in r) and not r.get("teardrop")),
               horizontal_teardrop_holes_with_hardware=sum(1 for r in rows if (r["axis_to_print_up_deg"] or 0) > 60 and ("sleeve_od_mm" in r or "screw" in r) and r.get("teardrop")),
               sleeves_in_manifest=len(tubes), sleeve_holes=sum(1 for r in rows if str(r.get("role", "")).startswith("sleeve bore")),
               partial_sleeve_bores=sum(1 for r in rows if str(r.get("role", "")).startswith("sleeve bore (partial")),
               teardrop_sleeve_bores=sum(1 for r in rows if str(r.get("role", "")) == "sleeve bore (partial: teardrop)"),
               sleeves_without_bore=[np.round(c, 2).tolist() for c, ta, L, od, idd in tubes
                                     if not any(str(r.get("role", "")).startswith("sleeve bore") and _coaxial(r["centre_mm"], np.asarray(r["axis"]), c, ta) for r in rows)],
               screw_holes=sum(1 for r in rows if "screw" in r), screws_without_known_thread=no_thread,
               hardware_of_unknown_kind_nearby=sorted(unknown), hole_list=rows)
    out["seconds"] = round(time.time() - t0, 1); tick("holes", t0)
    return out


@register("holes", lambda ctx: bool(_printed(ctx)))
def run(ctx):
    """Per printed part: holes from the B-rep (diameter, depth, axis vs print direction), sleeve press interference, screw hole vs thread (pilot / clearance), boss wall around hardware holes."""
    hw = json.loads((DATA / "hardware.json").read_text())
    _, prn = P.tables()
    per, cases = {}, []
    ev = dict(parts=0, holes=0, horizontal_holes=0, horizontal_holes_with_hardware=0, horizontal_teardrop_holes_with_hardware=0, sleeves_without_matching_hole=0,
              partial_sleeve_bores=0, teardrop_sleeve_bores=0,
              pilots_outside_reference_band=0, boss_walls_under_2_lines=0, screws_without_known_thread=0, hardware_of_unknown_kind_nearby=0)
    band = hw["pilot_pct_of_major"]
    skipped = not_measured(ctx)
    ev["parts_not_measured"] = len(skipped)
    for n in _targets(ctx):
        r = check_part(ctx, n, hw, prn)
        per[n] = r
        lw = P.effective(ctx, n)["line_width_mm"] or 0.42
        ev["parts"] += 1; ev["holes"] += r["holes"]
        ev["horizontal_holes"] += r["horizontal_holes"]; ev["horizontal_holes_with_hardware"] += r["horizontal_holes_with_hardware"]
        ev["sleeves_without_matching_hole"] += len(r["sleeves_without_bore"])
        ev["partial_sleeve_bores"] += r["partial_sleeve_bores"]; ev["teardrop_sleeve_bores"] += r["teardrop_sleeve_bores"]
        ev["horizontal_teardrop_holes_with_hardware"] += r["horizontal_teardrop_holes_with_hardware"]
        ev["screws_without_known_thread"] += len(r["screws_without_known_thread"])
        ev["hardware_of_unknown_kind_nearby"] += len(r["hardware_of_unknown_kind_nearby"])
        if r["hardware_of_unknown_kind_nearby"]:
            cases.append(dict(part=n, mating_part=", ".join(r["hardware_of_unknown_kind_nearby"][:6]), pose=None, location_mm=None, value=float(len(r["hardware_of_unknown_kind_nearby"])),
                              unit="hardware of unknown kind near the part", context=True,
                              what="hardware within the part's box whose definition gives no spec.kind and no screw_plan: not matched to its holes (declare it in the definition)"))
        for k in r["screws_without_known_thread"]:
            cases.append(dict(part=n, mating_part=k, pose=None, location_mm=None, value=0.0, unit="screws with a known thread",
                              what="screw through a hole of the part without a thread in its screw_plan or definition spec: pilot / clearance not judged"))
        for c in r["sleeves_without_bore"][:3]:
            cases.append(dict(part=n, mating_part=", ".join(ctx.rel.sleeve_sets_of(n)) or None, pose=None, location_mm=c, value=0.0, unit="mm bore found for this sleeve",
                              what="no B-rep bore of 180 deg or more around this sleeve"))
        for h in r["hole_list"]:
            if h.get("role") == "pilot" and not band[0] <= h["hole_pct_of_major"] <= band[1]:
                ev["pilots_outside_reference_band"] += 1
            if h.get("boss_wall_mm") is not None and h["boss_wall_mm"] < 2 * lw:
                ev["boss_walls_under_2_lines"] += 1
        hw_holes = [h for h in r["hole_list"] if h.get("boss_wall_mm") is not None]
        if hw_holes:
            w = min(hw_holes, key=lambda h: h["boss_wall_mm"])
            cases.append(dict(part=n, mating_part=w.get("screw") or ", ".join(ctx.rel.sleeve_sets_of(n)) or None, pose=None, location_mm=w["boss_wall_location_mm"],
                              value=w["boss_wall_mm"], unit="mm thinnest boss wall",
                              what=f"around a {w['d_mm']} mm {w.get('role', 'hole')} ({w.get('thread') or 'sleeve'}); line width {lw} mm"
                                   + (f"; {w['boss_wall_open_pct']} % of the rays open ({w['boss_wall_open_detail']['open_tip_pct']} % in a teardrop tip, "
                                      f"{w['boss_wall_open_detail']['open_out_pct']} % out of the part)" if w.get("boss_wall_open_pct") else "")))
        sl = [h for h in r["hole_list"] if "press_interference_mm" in h]
        if sl:
            v = [h["press_interference_mm"] for h in sl]
            cases.append(dict(part=n, mating_part=", ".join(ctx.rel.sleeve_sets_of(n)) or None, pose=None, location_mm=None, value=round(min(v), 3), unit="mm sleeve press interference (min)",
                              what=f"{len(sl)} sleeve holes, interference {min(v)} to {max(v)} mm in CAD"
                                   + ("" if r["hole_offset_mm"] is not None else "; printer hole offset not measured (coupon)")))
        pil = [h for h in r["hole_list"] if h.get("role") == "pilot"]
        if pil:
            v = [h["hole_pct_of_major"] for h in pil]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=min(v), unit="% pilot of thread major (min)",
                              what=f"{len(pil)} self-tapping holes, {min(v)} to {max(v)} % (reference {band[0]}-{band[1]} %)"))
        if r["horizontal_holes_with_hardware"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=float(r["horizontal_holes_with_hardware"]),
                              unit="hardware holes across the layers", what=f"of {r['horizontal_holes']} horizontal holes (up {r['up']}): sag / oval unless teardropped or reamed"
                                                                                + (f"; {r['horizontal_teardrop_holes_with_hardware']} teardropped hardware holes not counted" if r["horizontal_teardrop_holes_with_hardware"] else "")))
    cases[:0] = [dict(part=n, mating_part=None, pose=None, location_mm=None, value=0.0, unit="holes measured", what=f"not measured: {why}")
                 for n, why in skipped.items()]
    return dict(per_part=per, worst_cases=cases, bad_events=ev, not_measured=skipped, evidence=[],
                notes=f"holes = concave B-rep cylinders wrapping >= {HOLE_SPAN_DEG:g} deg (180-300 deg kept as partial when hardware sits in them: teardrop, slot or breakout); "
                      f"a partial bore is a teardrop when >= {TEARDROP_TIP_SHARE:.0%} of its open boss-wall rays meet material within {TIP_MM_OVER_R} r of the circle and point within {TEARDROP_UP_DEG:g} deg of print up; "
                      "teardropped hardware holes are counted apart from the holes across the layers (horizontal_teardrop_holes_with_hardware); "
                      f"parts without a STEP: circle fits on the closed mesh along the hardware axes (rms <= {FIT_RMS_MM} mm, >= {FIT_MIN_RAYS} of 24 rays, every {FIT_STEP_MM} mm; holes without hardware not found); "
                      f"sleeve boss walls are measured over the sleeve's length only; head recesses (screw-coaxial holes >= {RECESS_OVER_MAJOR:g} x thread major or 90 % of the head) and counterbores around sleeves get no boss wall; coaxial = axis within {AX_TOL_DEG:g} deg and lines within {LINE_TOL_MM} mm; "
                      "horizontal = axis more than 60 deg from the print direction; boss wall = shortest of 120 radial rays from the hole surface to the part's "
                      f"outer surface over the rays that start in material (boss_wall_open_pct = the rays that start in air: the open sector of a teardrop, slot or breakout), holes at least {BOSS_MIN_DEPTH_MM:g} mm deep; screws = hardware whose definition spec.kind is screw or that carries a builder's screw_plan, thread from that spec (never the name); thread data and pilot reference band from data/hardware.json; printer hole offset from printers.json (null = not measured)")
