"""holes: hole inventory of each scoped printed part from its native STEP, with the hardware through each hole.

Holes = concave cylindrical faces of the B-rep grouped by axis line and radius, kept when they wrap at least HOLE_SPAN_DEG
around the axis (fillets and slots are partial cylinders); a bore of 180-300 deg with a sleeve or screw in it is kept as a
partial bore (it breaks out of the part: boss wall 0). Per hole: diameter, axis, depth, centre (manifest frame) and
the axis angle to the print direction (0 = vertical hole, printed as stacked rings; 90 = across the layers: the top sags,
the hole prints oval unless teardropped or reamed). Hardware matched by coaxiality (axis within 3 deg, lines within
0.3 mm): steel sleeves (press interference = sleeve outer diameter - hole diameter, also after the printer's hole offset
when a coupon has set it) and screws (hole diameter as % of the thread's major diameter: a pilot for self-tapping screws,
a clearance hole otherwise). Boss wall: around every sleeve and screw hole, the thinnest material from the hole surface
outward (24 radial rays x 3 depths into the part's solid). Measurements and counts only."""
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


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and ctx.parts[n].get("native_step")]


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


def _coaxial(c1, a1, c2, a2):
    if abs(abs(float(np.dot(a1, a2))) - 1) > 1 - math.cos(math.radians(AX_TOL_DEG)):
        return False
    v = np.asarray(c2) - np.asarray(c1)
    return np.linalg.norm(v - np.dot(v, a1) * a1) < LINE_TOL_MM


def boss_wall(m, hole, window=None):
    """Thinnest material (mm) from the hole surface radially outward: 24 angles x 5 depths over 90 % of the length, first exit hit. window
    (centre, length) limits the depths to an axial stretch (a sleeve bore: where the sleeve sits, not where the bore runs
    on past it into a slot)."""
    a = np.asarray(hole["axis"], float); r = hole["d_mm"] / 2
    c, depth = (np.asarray(window[0], float), window[1]) if window else (np.asarray(hole["centre_mm"], float), hole["depth_mm"])
    c = np.asarray(hole["centre_mm"], float) + float((c - np.asarray(hole["centre_mm"], float)) @ a) * a   # on the hole's axis
    u = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(a, u)
    ang = np.radians(np.arange(0, 360, 15))
    dirs = np.array([math.cos(t) * u + math.sin(t) * w for t in ang])
    org = np.vstack([c + s * depth * a + (r + 0.02) * dirs for s in (-0.45, -0.225, 0.0, 0.225, 0.45)])
    dd = np.vstack([dirs] * 5)
    loc, ir, _ = m.ray.intersects_location(org, dd, multiple_hits=False)
    if not len(ir):
        return None, None
    dist = np.full(len(org), np.inf); dist[ir] = np.linalg.norm(loc - org[ir], axis=1) + 0.02
    i = int(np.argmin(dist))
    return round(float(dist[i]), 3), np.round(org[i], 2).tolist()


def check_part(ctx, n, hw, prn):
    import trimesh
    from checks.core import sha  # noqa: F401  (keeps import order with core)
    t0 = time.time()
    p = ctx.parts[n]
    e = P.effective(ctx, n)
    up = P.up_vector(e["up"]) if e["up"] else None
    holes = brep_holes(BL.src(p["native_step"]), p.get("native_step_offset_mm"))
    m = ctx.solid(n)[0]
    off = (prn.get(e["printer"] or "", {}) or {}).get("hole_offset_mm")
    # sleeves of this part
    sl = ctx.parts.get(f"{n}_steel_sleeves")
    tubes = []
    import sys
    from checks.core import ROOT
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from mount_check import kind, nominal, screw_geometry, tubes as _tubes
    if sl:
        tubes = _tubes(BL.load_mesh(sl["mesh"]))
    # screws near the part
    b = np.array(p["bounds_mm"], float)
    screws = []
    for k, q in ctx.parts.items():
        if kind(q) != "screw":
            continue
        qb = np.array(q["bounds_mm"], float)
        if np.any(qb[0] > b[1] + 1) or np.any(qb[1] < b[0] - 1):
            continue
        try:
            g = screw_geometry(BL.load_mesh(q["mesh"]))
        except Exception:  # noqa: BLE001
            continue
        screws.append((k, nominal(k).get("thread"), g))
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
                r.update(role="sleeve bore" + (" (partial: breaks out of the part)" if h["partial"] else ""), sleeve_window=(sc, sL), sleeve_od_mm=round(od, 3), sleeve_id_mm=round(idd, 3), press_interference_mm=round(od - h["d_mm"], 3),
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
            if maj and r.get("role") is None:
                if h["d_mm"] < maj:
                    r.update(role="pilot" if th in hw["self_tapping"] else "thread-forming hole", self_tapping=th in hw["self_tapping"],
                             hole_pct_of_major=round(100 * h["d_mm"] / maj, 1))
                else:
                    r.update(role="clearance hole", clearance_mm=round(h["d_mm"] - maj, 3))
        win = r.pop("sleeve_window", None)
        if win is not None:     # where the sleeve sits, clipped to the bore
            lo = max(float((np.asarray(win[0]) - hc) @ a) - win[1] / 2, -h["depth_mm"] / 2); hi = min(float((np.asarray(win[0]) - hc) @ a) + win[1] / 2, h["depth_mm"] / 2)
            win = (hc + 0.5 * (lo + hi) * a, max(hi - lo, 0.1))
        if ("sleeve_od_mm" in r or "screw" in r) and (win[1] if win else h["depth_mm"]) >= BOSS_MIN_DEPTH_MM and not h["partial"]:
            r["boss_wall_mm"], r["boss_wall_location_mm"] = boss_wall(m, h, win)
        elif h["partial"] and ("sleeve_od_mm" in r or "screw" in r):
            r["boss_wall_mm"], r["boss_wall_location_mm"] = 0.0, np.round(hc, 2).tolist()      # the bore breaks out: no wall on one side
        if h["partial"] and "sleeve_od_mm" not in r and "screw" not in r:
            continue                                                                          # an arc of a slot or fillet, not a hole
        rows.append(r)
    out = dict(up=e["up"], printer=e["printer"], hole_offset_mm=off, holes=len(rows),
               horizontal_holes=sum(1 for r in rows if (r["axis_to_print_up_deg"] or 0) > 60),
               horizontal_holes_with_hardware=sum(1 for r in rows if (r["axis_to_print_up_deg"] or 0) > 60 and ("sleeve_od_mm" in r or "screw" in r)),
               sleeves_in_manifest=len(tubes), sleeve_holes=sum(1 for r in rows if str(r.get("role", "")).startswith("sleeve bore")),
               partial_sleeve_bores=sum(1 for r in rows if str(r.get("role", "")).startswith("sleeve bore (partial")),
               sleeves_without_bore=[np.round(c, 2).tolist() for c, ta, L, od, idd in tubes
                                     if not any(str(r.get("role", "")).startswith("sleeve bore") and _coaxial(r["centre_mm"], np.asarray(r["axis"]), c, ta) for r in rows)],
               screw_holes=sum(1 for r in rows if "screw" in r), hole_list=rows)
    out["seconds"] = round(time.time() - t0, 1); tick("holes", t0)
    return out


@register("holes", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Per printed part: holes from the B-rep (diameter, depth, axis vs print direction), sleeve press interference, screw hole vs thread (pilot / clearance), boss wall around hardware holes."""
    hw = json.loads((DATA / "hardware.json").read_text())
    _, prn = P.tables()
    per, cases = {}, []
    ev = dict(parts=0, holes=0, horizontal_holes=0, horizontal_holes_with_hardware=0, sleeves_without_matching_hole=0, partial_sleeve_bores=0,
              pilots_outside_reference_band=0, boss_walls_under_2_lines=0)
    band = hw["pilot_pct_of_major"]
    for n in _targets(ctx):
        r = check_part(ctx, n, hw, prn)
        per[n] = r
        lw = P.effective(ctx, n)["line_width_mm"] or 0.42
        ev["parts"] += 1; ev["holes"] += r["holes"]
        ev["horizontal_holes"] += r["horizontal_holes"]; ev["horizontal_holes_with_hardware"] += r["horizontal_holes_with_hardware"]
        ev["sleeves_without_matching_hole"] += len(r["sleeves_without_bore"])
        ev["partial_sleeve_bores"] += r["partial_sleeve_bores"]
        for c in r["sleeves_without_bore"][:3]:
            cases.append(dict(part=n, mating_part=f"{n}_steel_sleeves", pose=None, location_mm=c, value=0.0, unit="mm bore found for this sleeve",
                              what="no B-rep bore of 180 deg or more around this sleeve"))
        for h in r["hole_list"]:
            if h.get("role") == "pilot" and not band[0] <= h["hole_pct_of_major"] <= band[1]:
                ev["pilots_outside_reference_band"] += 1
            if h.get("boss_wall_mm") is not None and h["boss_wall_mm"] < 2 * lw:
                ev["boss_walls_under_2_lines"] += 1
        hw_holes = [h for h in r["hole_list"] if h.get("boss_wall_mm") is not None]
        if hw_holes:
            w = min(hw_holes, key=lambda h: h["boss_wall_mm"])
            cases.append(dict(part=n, mating_part=w.get("screw") or f"{n}_steel_sleeves", pose=None, location_mm=w["boss_wall_location_mm"],
                              value=w["boss_wall_mm"], unit="mm thinnest boss wall",
                              what=f"around a {w['d_mm']} mm {w.get('role', 'hole')} ({w.get('thread') or 'sleeve'}); line width {lw} mm"))
        sl = [h for h in r["hole_list"] if "press_interference_mm" in h]
        if sl:
            v = [h["press_interference_mm"] for h in sl]
            cases.append(dict(part=n, mating_part=f"{n}_steel_sleeves", pose=None, location_mm=None, value=round(min(v), 3), unit="mm sleeve press interference (min)",
                              what=f"{len(sl)} sleeve holes, interference {min(v)} to {max(v)} mm in CAD"
                                   + ("" if r["hole_offset_mm"] is not None else "; printer hole offset not measured (coupon)")))
        pil = [h for h in r["hole_list"] if h.get("role") == "pilot"]
        if pil:
            v = [h["hole_pct_of_major"] for h in pil]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=min(v), unit="% pilot of thread major (min)",
                              what=f"{len(pil)} self-tapping holes, {min(v)} to {max(v)} % (reference {band[0]}-{band[1]} %)"))
        if r["horizontal_holes_with_hardware"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=float(r["horizontal_holes_with_hardware"]),
                              unit="hardware holes across the layers", what=f"of {r['horizontal_holes']} horizontal holes (up {r['up']}): sag / oval unless teardropped or reamed"))
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"holes = concave B-rep cylinders wrapping >= {HOLE_SPAN_DEG:g} deg (180-300 deg kept as partial when hardware sits in them: the bore breaks out, wall 0); "
                      f"sleeve boss walls are measured over the sleeve's length only; coaxial = axis within {AX_TOL_DEG:g} deg and lines within {LINE_TOL_MM} mm; "
                      "horizontal = axis more than 60 deg from the print direction; boss wall = shortest of 72 radial rays from the hole surface to the part's "
                      f"outer surface, holes at least {BOSS_MIN_DEPTH_MM:g} mm deep; thread data and pilot reference band from data/hardware.json; printer hole offset from printers.json (null = not measured)")
