"""Pressability of the pressed-in 304 SS compression sleeves of FDM connectors (owner rule, Idle M-evmrv6).

Every sleeve needs (1) a flat support face on its counter (exit) side, so the part can rest on an anvil, and (2) a
straight, unobstructed access path along its axis for the heated press element on the entry side. Scored on the bare
part (that is how it is pressed), continuous values only (tool sizes from data/press_tools.json; the numbers below are its values):

  tool path     heated tip  dia 8 mm over the first 15 mm beyond the sleeve's entry end, tool body dia 20 mm beyond
                that, 60 mm long in total: part volume inside each (mm3)
  anvil path    dia 20 mm x 40 mm, from 0.2 mm beyond the exit plane outward: part volume inside (mm3); the first
                15 mm inside dia 10 is reported too
  counter face  dia 10 disc in the exit plane (the sleeve end's plane), outside the sleeve: rays along the axis find the
                part surface; flatness = max |deviation| from the plane of the surface found within 3 mm, coverage = % of
                the annulus (sleeve OD .. dia 10) with part material within 0.2 mm of the plane, tilt = angle between the
                axis and the normal of a plane fitted to the surface points within 1 mm
  spacer model  the counter side is a spacer block (any size) resting on the exit plane, not a fixed post. Each annulus point
                is classified: flat (air within 0.2 mm above the plane, surface within 0.2 mm of it), wall (material
                above the plane: an integral wall the spacer cannot enter), recess (surface more than 0.2 mm below the
                plane, within 10 mm), open (nothing within 10 mm below). Shares of the annulus area in *_pct; support_pct =
                flat + wall (material at the plane). free_depth_mm = how far a spacer can reach along the axis from the
                flat points before meeting the part (min over those rays, capped at 40 mm; 0 without flat points);
                axis_free_depth_mm = the same on the sleeve axis itself. The fixed dia 20 x 40 post numbers stay reported.
  heat-set      a sleeve-set item whose definition spec.kind is insert (brass heat-set insert) is melted in from its entry end
  insert        with a heated tip; nothing presses on its far end, so it has no exit-plane / spacer support. Per insert:
                face obstruction = part volume inside the insert's outer diameter (definition spec.outer_dia_mm) beyond the
                entry end (blocks the insert and the tip's face); shank = the catalogue tip shank of its thread
                (heat_set_insert_tips: 1/4 in = 6.35 mm, McMaster-Carr): part volume inside it beyond the entry end and
                shank_clearance_mm = the highest material inside it above the entry plane (the working-end length the tip
                needs; no length is assumed); entry face = the counter-face classification on the entry plane from the
                insert to the shank diameter (the face the insert ends flush in); exit side: bore_beyond_exit_mm = free
                distance on the axis past the far end (blind: the bore's melt room; null = through to air within 40 mm)
  entry side    a plain sleeve is symmetric: both ends are tried and the one with less tool-path obstruction is the entry
                (ties: less anvil-path obstruction)

  from checks.press_lib import score_part
  rows = score_part(part_shape, sleeves_shape)       # build123d shapes in the same frame
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

# the tool sizes are data: data/press_tools.json (LAB_CHECKS_DATA: another set of check tables)
TOOLS = json.loads((Path(os.environ.get("LAB_CHECKS_DATA") or Path(__file__).resolve().parent / "data") / "press_tools.json").read_text())
TIP_D, TIP_L, BODY_D, TOOL_L = TOOLS["tool_tip_dia_mm"], TOOLS["tool_tip_length_mm"], TOOLS["tool_body_dia_mm"], TOOLS["tool_total_length_mm"]
SPACER_CAP, RECESS_MAX = TOOLS["spacer_free_depth_cap_mm"], TOOLS["spacer_recess_search_mm"]
ANVIL_D, ANVIL_PATH_D, ANVIL_L, FACE_TOL = TOOLS["anvil_face_dia_mm"], TOOLS["anvil_path_dia_mm"], TOOLS["anvil_path_length_mm"], TOOLS["counter_face_tolerance_mm"]
EDGE_SKIP = TOOLS["counter_face_edge_skip_mm"]
INSERT_TIPS = {k: v for k, v in (TOOLS.get("heat_set_insert_tips") or {}).items() if not k.startswith("_")}
ASSUMPTIONS = dict(tool_tip_dia_mm=TIP_D, tool_tip_length_mm=TIP_L, tool_body_dia_mm=BODY_D, tool_total_length_mm=TOOL_L,
                   anvil_face_dia_mm=ANVIL_D, anvil_path_dia_mm=ANVIL_PATH_D, anvil_path_length_mm=ANVIL_L,
                   counter_face_tolerance_mm=FACE_TOL, counter_face_annulus_mm=f'sleeve OD/2 + {EDGE_SKIP} .. {ANVIL_D / 2}', anvil_path_start_mm_beyond_exit_plane=0.2,
                   tool_path_start_mm_beyond_entry_end=0.05,
                   spacer_free_depth_cap_mm=SPACER_CAP, spacer_recess_search_mm=RECESS_MAX)


def _cyl(p0, d, r, length):
    from build123d import Solid
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    ax = gp_Ax2(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, d)))
    return Solid(BRepPrimAPI_MakeCylinder(ax, float(r), float(length)).Shape())


def _common_volume(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    ba, bb = Bnd_Box(), Bnd_Box()
    BRepBndLib.Add_s(a.wrapped, ba); BRepBndLib.Add_s(b.wrapped, bb)
    if ba.IsOut(bb):
        return 0.0
    op = BRepAlgoAPI_Common(a.wrapped, b.wrapped); op.Build()
    if not op.IsDone():
        return float("nan")
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(op.Shape(), g)
    return abs(float(g.Mass()))


def sleeve_axis(sol):
    """(centre of end 0, centre of end 1, unit axis 0->1, outer r, inner r) of a tube from its cylindrical faces."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    cyls = []
    for f in sol.faces():
        s = BRepAdaptor_Surface(f.wrapped)
        if s.GetType() == GeomAbs_Cylinder:
            c = s.Cylinder(); a = c.Axis()
            cyls.append((c.Radius(), np.array([a.Location().X(), a.Location().Y(), a.Location().Z()]),
                         np.array([a.Direction().X(), a.Direction().Y(), a.Direction().Z()])))
    if not cyls:
        return None
    r_out = max(c[0] for c in cyls); r_in = min(c[0] for c in cyls)
    _, p, d = max(cyls, key=lambda c: c[0])
    d = d / np.linalg.norm(d)
    v = np.array([[x.X, x.Y, x.Z] for x in sol.vertices()])
    t = (v - p) @ d
    return p + t.min() * d, p + t.max() * d, d, float(r_out), float(r_in)


def _rays(mesh):
    import trimesh
    try:
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:  # noqa: BLE001
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


def spacer_contact(mesh, rays, pts, e, n_out):
    """Classify annulus points of the exit plane as flat / wall / recess / open and measure the spacer's free depth."""
    up = pts + FACE_TOL * n_out
    wall = np.asarray(mesh.contains(up))
    dn = np.repeat(-n_out[None], len(pts), 0)
    loc, idx, _ = rays.intersects_location(up, dn, multiple_hits=False)
    h = np.full(len(pts), np.inf)
    if len(idx):
        h[idx] = np.linalg.norm(loc - up[idx], axis=1)          # distance below the plane + FACE_TOL
    air = ~wall
    flat = air & (h <= 2 * FACE_TOL)
    recess = air & (h > 2 * FACE_TOL) & (h <= RECESS_MAX)
    open_ = air & ~flat & ~recess
    free = np.full(len(pts), np.inf)
    fi = np.where(flat)[0]
    if len(fi):
        loc, idx, _ = rays.intersects_location(up[fi], np.repeat(n_out[None], len(fi), 0), multiple_hits=False)
        free[fi[idx]] = np.linalg.norm(loc - up[fi[idx]], axis=1) + FACE_TOL      # rays start FACE_TOL off the plane: depth from the plane
    free_depth = float(min(free[fi].min(), SPACER_CAP)) if len(fi) else 0.0
    a0 = (e + FACE_TOL * n_out)[None]
    loc, idx, _ = rays.intersects_location(a0, n_out[None], multiple_hits=False)
    axis_free = float(min(np.linalg.norm(loc[0] - a0[0]) + FACE_TOL, SPACER_CAP)) if len(idx) else SPACER_CAP
    sh = lambda m: round(100.0 * float(m.mean()), 1)
    return dict(flat_pct=sh(flat), wall_pct=sh(wall), recess_pct=sh(recess), open_pct=sh(open_),
                support_pct=sh(flat | wall), free_depth_mm=round(free_depth, 2), axis_free_depth_mm=round(axis_free, 2))


def counter_face(mesh, rays, e, n_out, r_out, n_ring=10, n_ang=36, outer_d=None):
    """Exit-plane disc (dia ANVIL_D, or outer_d) outside the sleeve: flatness, coverage, tilt. n_out points away from the part."""
    u = np.cross(n_out, [1.0, 0.0, 0.0] if abs(n_out[0]) < 0.9 else [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
    w = np.cross(n_out, u)
    r0 = r_out + EDGE_SKIP          # the bore edge itself is not sampled (tessellation chords of the bore)
    rs = np.sqrt(np.linspace(r0 ** 2, ((outer_d or ANVIL_D) / 2) ** 2, n_ring))     # equal-area rings
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    pts = np.array([e + r * (math.cos(a) * u + math.sin(a) * w) for r in rs for a in ang])
    start = pts + 3.0 * n_out
    loc, idx, _ = rays.intersects_location(start, np.repeat(-n_out[None], len(pts), 0), multiple_hits=False)
    dev = np.full(len(pts), np.nan)
    if len(idx):
        dev[idx] = 3.0 - np.linalg.norm(loc - start[idx], axis=1)   # > 0: part surface proud of the plane (outward)
        dev[np.abs(dev) > 3.0] = np.nan
    covered = np.abs(dev) <= FACE_TOL
    found = ~np.isnan(dev)
    near = found & (np.abs(dev) <= 1.0)
    tilt = None
    if near.sum() >= 6:
        P = pts[near] + dev[near, None] * n_out
        c = P.mean(0); _, _, vt = np.linalg.svd(P - c)
        nn = vt[-1]
        tilt = round(float(math.degrees(math.acos(min(1.0, abs(float(nn @ n_out)))))), 2)
    spacer = spacer_contact(mesh, rays, pts, e, n_out)
    return dict(**spacer, coverage_pct=round(100.0 * covered.mean(), 1),
                flatness_mm=round(float(np.nanmax(np.abs(dev))), 3) if found.any() else None,
                proud_max_mm=round(float(np.nanmax(dev)), 3) if found.any() else None,
                no_surface_pct=round(100.0 * (~found).mean(), 1), tilt_deg=tilt)


def score_orientation(part, mesh, rays, e_in, e_out, d, r_out):
    """d points from the entry end to the exit end."""
    tip = _common_volume(part, _cyl(e_in - 0.05 * d, -d, TIP_D / 2, TIP_L))
    body = _common_volume(part, _cyl(e_in - (0.05 + TIP_L) * d, -d, BODY_D / 2, TOOL_L - TIP_L))
    anv = _common_volume(part, _cyl(e_out + 0.2 * d, d, ANVIL_PATH_D / 2, ANVIL_L))
    anv10 = _common_volume(part, _cyl(e_out + 0.2 * d, d, ANVIL_D / 2, TIP_L))
    face = counter_face(mesh, rays, e_out, d, r_out)
    return dict(tool_tip_obstruction_mm3=round(tip, 3), tool_body_obstruction_mm3=round(body, 3),
                anvil_path_obstruction_mm3=round(anv, 3), anvil_dia10_first15mm_obstruction_mm3=round(anv10, 3), counter_face=face)


ITEM_MATCH_MM = 1.0     # a tube is a definition item when |length| + |outer dia| (+ |inner dia| for a sleeve) differ by at most this


def item_of(tube, items):
    """The definition item ({kind, spec, line}) a sleeve-set tube is, from the set's contains: the item whose spec length and outer
    diameter (and, for a sleeve, inner diameter) are nearest the tube's, within ITEM_MATCH_MM (a STEP models an insert as a tube of
    its hole, 0.4-0.6 mm under the knurl); (item, score mm), or (None, score of the nearest) when none is that close."""
    L, od, idd = tube
    best = None
    for it in items:
        sp = it.get("spec") or {}
        if sp.get("length_mm") is None or sp.get("outer_dia_mm") is None:
            continue
        sc = abs(float(sp["length_mm"]) - L) + abs(float(sp["outer_dia_mm"]) - od)
        if sp.get("kind") != "insert" and sp.get("inner_dia_mm") is not None:
            sc += abs(float(sp["inner_dia_mm"]) - idd)
        if best is None or sc < best[1]:
            best = (it, round(sc, 3))
    if best is None:
        return None, None
    return (best[0], best[1]) if best[1] <= ITEM_MATCH_MM else (None, best[1])


def _top_height(rays, e, n_out, r, length=None, n_ring=6, n_ang=24):
    """Highest part material (mm above the plane through e, along n_out) inside the cylinder of radius r over `length`: rays from the
    cylinder's far end back toward the plane, first hit; 0 when nothing is inside."""
    length = length or TOOL_L
    u = np.cross(n_out, [1.0, 0.0, 0.0] if abs(n_out[0]) < 0.9 else [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
    w = np.cross(n_out, u)
    pts = [e] + [e + rr * (math.cos(a) * u + math.sin(a) * w) for rr in np.linspace(r / n_ring, r, n_ring) for a in np.linspace(0, 2 * np.pi, n_ang, endpoint=False)]
    start = np.array(pts) + length * n_out
    loc, idx, _ = rays.intersects_location(start, np.repeat(-n_out[None], len(start), 0), multiple_hits=False)
    if not len(idx):
        return 0.0
    h = length - np.linalg.norm(loc - start[idx], axis=1)
    h = h[h > 0.05]
    return round(float(h.max()), 3) if len(h) else 0.0


def insert_orientation(vol, mesh, rays, e_in, e_out, d, r_out, spec):
    """Heat-set insert melted in from e_in (d points from the entry end to the far end). vol(p0, dir, r, L) = part volume inside a
    cylinder. See the module doc (heat-set insert)."""
    od = float(spec.get("outer_dia_mm") or 2 * r_out)
    tip = INSERT_TIPS.get(str(spec.get("thread")))
    face = vol(e_in - 0.05 * d, -d, od / 2, TOOL_L)
    out = dict(model="heat-set insert", thread=spec.get("thread"), insert_outer_dia_mm=od, tool_tip_obstruction_mm3=round(face, 3), tool_body_obstruction_mm3=0.0,
               tip_source=(tip or {}).get("part_number") or f"no tip in heat_set_insert_tips for thread {spec.get('thread')!r}")
    sd = (tip or {}).get("shank_dia_mm")
    if sd:
        out.update(tip_shank_dia_mm=sd, shank_obstruction_mm3=round(vol(e_in - 0.05 * d, -d, sd / 2, TOOL_L), 3),
                   shank_clearance_mm=_top_height(rays, e_in, -d, sd / 2))
    out["entry_face"] = counter_face(mesh, rays, e_in, -d, r_out, outer_d=max(sd or 0.0, od + 2 * EDGE_SKIP + 0.5))
    a0 = (e_out + 0.05 * d)[None]
    loc, idx, _ = rays.intersects_location(a0, d[None], multiple_hits=False)
    out["exit_side"] = dict(bore_beyond_exit_mm=round(float(np.linalg.norm(loc[0] - a0[0])) + 0.05, 2) if len(idx) and np.linalg.norm(loc[0] - a0[0]) < SPACER_CAP else None,
                            support="none needed: a heat-set insert is melted in, nothing presses on its far end")
    return out


def insert_row(vol, mesh, rays, i, e0, e1, d, r_out, r_in, spec, item_src):
    """Score both ends of an insert as the entry; the entry is the end with less face obstruction (ties: less shank clearance)."""
    a = insert_orientation(vol, mesh, rays, e0, e1, d, r_out, spec)
    b = insert_orientation(vol, mesh, rays, e1, e0, -d, r_out, spec)
    ka = (a["tool_tip_obstruction_mm3"], a.get("shank_clearance_mm") or 0.0)
    kb = (b["tool_tip_obstruction_mm3"], b.get("shank_clearance_mm") or 0.0)
    best, entry, other, dd = (a, e0, b, d) if ka <= kb else (b, e1, a, -d)
    return dict(index=i, kind="insert", kind_source=item_src, centre_mm=np.round((e0 + e1) / 2, 2).tolist(), axis=np.round(dd, 3).tolist(),
                entry_end_mm=np.round(entry, 2).tolist(), length_mm=round(float(np.linalg.norm(e1 - e0)), 2),
                od_mm=round(2 * r_out, 2), id_mm=round(2 * r_in, 2), **best,
                other_side=dict(tool_mm3=other["tool_tip_obstruction_mm3"], shank_clearance_mm=other.get("shank_clearance_mm"),
                                entry_face_support_pct=other["entry_face"]["support_pct"]))


def part_mesh(part):
    import trimesh
    v, f = part.tessellate(0.03, 0.2)
    return trimesh.Trimesh(np.array([(p.X, p.Y, p.Z) for p in v]), np.array(f), process=False)


def score_part(part, sleeves, items=()):
    """items: the sleeve set's definition contains (spec.kind insert -> the heat-set insert model)."""
    mesh = part_mesh(part); rays = _rays(mesh)
    rows = []
    vol = lambda p0, d, r, L: _common_volume(part, _cyl(p0, d, r, L))  # noqa: E731
    for i, s in enumerate(sleeves.solids()):
        ax = sleeve_axis(s)
        if ax is None:
            rows.append(dict(index=i, error="no cylindrical face")); continue
        e0, e1, d, r_out, r_in = ax
        it, sc = item_of((float(np.linalg.norm(e1 - e0)), 2 * r_out, 2 * r_in), items)
        if it is not None and (it.get("spec") or {}).get("kind") == "insert":
            rows.append(insert_row(vol, mesh, rays, i, e0, e1, d, r_out, r_in, it["spec"], f"definition item {it.get('line')} (nearest length / diameters, {sc} mm)"))
            continue
        a = score_orientation(part, mesh, rays, e0, e1, d, r_out)
        b = score_orientation(part, mesh, rays, e1, e0, -d, r_out)
        ka = (a["tool_tip_obstruction_mm3"] + a["tool_body_obstruction_mm3"], a["anvil_path_obstruction_mm3"])
        kb = (b["tool_tip_obstruction_mm3"] + b["tool_body_obstruction_mm3"], b["anvil_path_obstruction_mm3"])
        best, entry, other = (a, e0, b) if ka <= kb else (b, e1, a)
        d_in = d if best is a else -d
        rows.append(dict(index=i, kind="sleeve", kind_source=f"definition item {it.get('line')} ({sc} mm)" if it is not None else f"no definition item within {ITEM_MATCH_MM} mm (nearest {sc} mm)", centre_mm=np.round((e0 + e1) / 2, 2).tolist(), axis=np.round(d_in, 3).tolist(),
                         entry_end_mm=np.round(entry, 2).tolist(), length_mm=round(float(np.linalg.norm(e1 - e0)), 2),
                         od_mm=round(2 * r_out, 2), id_mm=round(2 * r_in, 2), **best,
                         other_side=dict(tool_mm3=round(other["tool_tip_obstruction_mm3"] + other["tool_body_obstruction_mm3"], 3),
                                         anvil_mm3=other["anvil_path_obstruction_mm3"],
                                         coverage_pct=other["counter_face"]["coverage_pct"],
                                         support_pct=other["counter_face"]["support_pct"],
                                         free_depth_mm=other["counter_face"]["free_depth_mm"])))
    return rows


def summary(rows, cover_min=90.0, eps=0.01):
    """Counts over the sleeves (exit-plane support, anvil) and, apart, the heat-set inserts (face obstruction, entry face, shank)."""
    ok = [r for r in rows if "error" not in r and r.get("kind") != "insert"]
    ins = [r for r in rows if "error" not in r and r.get("kind") == "insert"]
    tool = sum(1 for r in ok if r["tool_tip_obstruction_mm3"] + r["tool_body_obstruction_mm3"] > eps)
    anvil = sum(1 for r in ok if r["anvil_path_obstruction_mm3"] > eps)
    face = sum(1 for r in ok if r["counter_face"]["coverage_pct"] < cover_min)
    unsupported = sum(1 for r in ok if r["counter_face"]["support_pct"] < cover_min)
    return dict(sleeves=len(rows) - len(ins), inserts=len(ins), with_tool_obstruction=tool, with_anvil_obstruction=anvil,
                counter_face_coverage_below_90pct=face, counter_face_support_below_90pct=unsupported,
                inserts_with_face_obstruction=sum(1 for r in ins if r["tool_tip_obstruction_mm3"] > eps),
                inserts_entry_face_flat_below_90pct=sum(1 for r in ins if r["entry_face"]["flat_pct"] < cover_min),
                max_insert_shank_clearance_mm=max([r.get("shank_clearance_mm") or 0.0 for r in ins] or [None]),
                min_free_depth_mm=min([r["counter_face"]["free_depth_mm"] for r in ok] or [None]),
                max_tool_obstruction_mm3=round(max([r["tool_tip_obstruction_mm3"] + r["tool_body_obstruction_mm3"] for r in ok] or [0]), 2),
                max_anvil_obstruction_mm3=round(max([r["anvil_path_obstruction_mm3"] for r in ok] or [0]), 2),
                min_coverage_pct=min([r["counter_face"]["coverage_pct"] for r in ok] or [None]),
                max_flatness_mm=max([r["counter_face"]["flatness_mm"] or 0 for r in ok] or [0]))


def refresh_faces(part, rows):
    """Recompute only the counter-face scores of rows from score_part (same entry side), e.g. after a face-metric change."""
    mesh = part_mesh(part); rays = _rays(mesh)
    for r in rows:
        if "error" in r:
            continue
        d = np.array(r["axis"], float); e_in = np.array(r["entry_end_mm"], float); e_out = e_in + r["length_mm"] * d
        r["counter_face"] = counter_face(mesh, rays, e_out, d, r["od_mm"] / 2)
        r["other_side"]["coverage_pct"] = counter_face(mesh, rays, e_in, -d, r["od_mm"] / 2)["coverage_pct"]
    return rows
