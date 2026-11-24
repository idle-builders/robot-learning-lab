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


def counter_face(mesh, rays, e, n_out, r_out, n_ring=10, n_ang=36):
    """Exit-plane disc (dia ANVIL_D) outside the sleeve: flatness, coverage, tilt. n_out points away from the part."""
    u = np.cross(n_out, [1.0, 0.0, 0.0] if abs(n_out[0]) < 0.9 else [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
    w = np.cross(n_out, u)
    r0 = r_out + EDGE_SKIP          # the bore edge itself is not sampled (tessellation chords of the bore)
    rs = np.sqrt(np.linspace(r0 ** 2, (ANVIL_D / 2) ** 2, n_ring))     # equal-area rings
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


def part_mesh(part):
    import trimesh
    v, f = part.tessellate(0.03, 0.2)
    return trimesh.Trimesh(np.array([(p.X, p.Y, p.Z) for p in v]), np.array(f), process=False)


def score_part(part, sleeves):
    mesh = part_mesh(part); rays = _rays(mesh)
    rows = []
    for i, s in enumerate(sleeves.solids()):
        ax = sleeve_axis(s)
        if ax is None:
            rows.append(dict(index=i, error="no cylindrical face")); continue
        e0, e1, d, r_out, r_in = ax
        a = score_orientation(part, mesh, rays, e0, e1, d, r_out)
        b = score_orientation(part, mesh, rays, e1, e0, -d, r_out)
        ka = (a["tool_tip_obstruction_mm3"] + a["tool_body_obstruction_mm3"], a["anvil_path_obstruction_mm3"])
        kb = (b["tool_tip_obstruction_mm3"] + b["tool_body_obstruction_mm3"], b["anvil_path_obstruction_mm3"])
        best, entry, other = (a, e0, b) if ka <= kb else (b, e1, a)
        d_in = d if best is a else -d
        rows.append(dict(index=i, centre_mm=np.round((e0 + e1) / 2, 2).tolist(), axis=np.round(d_in, 3).tolist(),
                         entry_end_mm=np.round(entry, 2).tolist(), length_mm=round(float(np.linalg.norm(e1 - e0)), 2),
                         od_mm=round(2 * r_out, 2), id_mm=round(2 * r_in, 2), **best,
                         other_side=dict(tool_mm3=round(other["tool_tip_obstruction_mm3"] + other["tool_body_obstruction_mm3"], 3),
                                         anvil_mm3=other["anvil_path_obstruction_mm3"],
                                         coverage_pct=other["counter_face"]["coverage_pct"],
                                         support_pct=other["counter_face"]["support_pct"],
                                         free_depth_mm=other["counter_face"]["free_depth_mm"])))
    return rows


def summary(rows, cover_min=90.0, eps=0.01):
    ok = [r for r in rows if "error" not in r]
    tool = sum(1 for r in ok if r["tool_tip_obstruction_mm3"] + r["tool_body_obstruction_mm3"] > eps)
    anvil = sum(1 for r in ok if r["anvil_path_obstruction_mm3"] > eps)
    face = sum(1 for r in ok if r["counter_face"]["coverage_pct"] < cover_min)
    unsupported = sum(1 for r in ok if r["counter_face"]["support_pct"] < cover_min)
    return dict(sleeves=len(rows), with_tool_obstruction=tool, with_anvil_obstruction=anvil,
                counter_face_coverage_below_90pct=face, counter_face_support_below_90pct=unsupported,
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
