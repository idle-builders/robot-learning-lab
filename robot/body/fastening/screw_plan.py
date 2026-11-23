"""Screws generated from the defined tapped holes they land in (owner 2026-11-18, Idle T-cmn5fc).

One helper for builders (hole positions, no typed numbers) and for the manifest assembly (screw meshes):

  plan_connector(body, iface, connector, side)   per screw of a connector (robot/body/fastening/connector_screws.json):
      defined hole id, opening + axis in the body frame (registered instance of the standard part), clamped stack
      thickness along the axis measured in the CAD (from the hole opening to the outer face of the clamped parts, any
      gap included), thread family by the terminal material, the standard length (data/hardware.json screw_standards)
      that reaches the hole's reference engagement without the tip (a 0.5 mm point does not count, as
      mount_check.match_screw) passing the hole bottom, and the resulting engagement / tip margin. When no standard length
      satisfies both, the nearest one and its shortfall / overrun; for every case the stack change that would make the
      nearest standard length fit.
  hole_positions(iface, standard_part, labels)   opening + axis of defined holes, for builders
  screw_mesh(plan)                               revolved envelope of the chosen screw, head underside on the outer face

Engagement of a screw of nominal length L into a hole opening at stack s (mm from the clamped outer face), depth z:
    engagement = L - point - s        tip margin = (s + z) - (L - point)      (margin < 0: the tip passes the hole bottom)
Everything is a measurement in mm; nothing here accepts or rejects.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HARDWARE = Path(__import__("os").environ.get("LAB_CHECKS_DATA") or ROOT / "robot/body/checks/data") / "hardware.json"
CONNECTORS = HERE / "connector_screws.json"
RING_MM = (1.9, 2.3, 2.7)       # radii of the axial rays that find the clamped parts' outer face around the hole
REACH_MM = 40.0


def standards() -> dict:
    return json.loads(HARDWARE.read_text())["screw_standards"]


def connectors() -> dict:
    return json.loads(CONNECTORS.read_text())["connectors"]


def _fill(s: str, side: str) -> str:
    return s.replace("{S}", side)


def hole_positions(iface: dict, standard_part: str, labels: list[str]) -> list[dict]:
    """Opening and outward axis (body frame, mm) of defined holes of one registered standard-part instance."""
    by = {h["hole_id"]: h for h in iface["holes"]}
    out = []
    for lb in labels:
        h = by.get(f"{standard_part}:{lb}")
        if h is None:
            raise KeyError(f"defined hole {standard_part}:{lb} not in the registered instances")
        out.append(h)
    return out


def stack_along_axis(meshes: list, opening: np.ndarray, axis_out: np.ndarray, radii=RING_MM) -> dict:
    """Outer face of the clamped parts above a hole opening: along rays parallel to the hole axis on rings around it,
    the material segments of the clamped meshes are merged (touching within 0.05 mm) and the first merged run's end is
    the outer face. Returns the median distance from the opening (mm), the run's start (gap to the opening, mm), the spread
    over the rays and how many rays found material."""
    import mount_check as MC
    d = np.asarray(axis_out, float)
    ends, starts = [], []
    per_part = []
    for m in meshes:
        per_part.append(MC.axial_segments(m, MC.rays_for(m), np.asarray(opening, float), d, radii, -0.3, REACH_MM))
    for r in radii:
        for k in range(MC.N_ANG):
            segs = sorted(s for pp in per_part for s in pp[r][k])
            if not segs:
                continue
            b = segs[0][1]
            for s0, s1 in segs[1:]:
                if s0 <= b + 0.05:
                    b = max(b, s1)
            ends.append(b); starts.append(max(segs[0][0], 0.0))
    if not ends:
        return dict(stack_mm=None, gap_mm=None, spread_mm=None, rays=0)
    e = np.array(ends)
    return dict(stack_mm=round(float(np.median(e)), 3), gap_mm=round(float(np.median(starts)), 3),
                spread_mm=round(float(e.max() - e.min()), 3), rays=int(len(e)))


def choose_length(stack: float, depth: float, ref: float, lengths: list[float], point: float) -> dict:
    """Smallest standard length with ref <= engagement and tip inside the hole; else the nearest by violation. A blind hole
    shallower than the reference cannot give it: the length is chosen against min(ref, depth), the shortfall stays against ref."""
    rows = []
    want = min(ref, depth)
    for L in lengths:
        eng = L - point - stack
        margin = (stack + depth) - (L - point)
        short = max(0.0, ref - eng)
        over = max(0.0, -margin)
        pick_short = max(0.0, want - eng)
        rows.append(dict(length_mm=float(L), engagement_mm=round(eng, 3), tip_margin_mm=round(margin, 3),
                         shortfall_mm=round(short, 3), overrun_mm=round(over, 3), violation_mm=pick_short + over,
                         stack_window_mm=[round(L - point - depth, 3), round(L - point - want, 3)]))
    best = min(rows, key=lambda r: (round(r["violation_mm"], 6), r["length_mm"]))
    # stack change that puts the nearest length inside its window (negative = thinner stack)
    lo, hi = best["stack_window_mm"]
    best["stack_change_to_fit_mm"] = round(0.0 if lo <= stack <= hi else (lo - stack if stack < lo else hi - stack), 3)
    return best


def screw_spec(thread: str, family: str, head: str) -> dict:
    st = standards()
    h = st["head_types"][head][thread]
    return dict(dk=h["dk"], k=h["k"], lengths=h["lengths"], point_mm=st["point_mm"], standard=head,
                countersunk=bool(st["head_types"][head].get("countersunk")))


def sides_of(spec: dict) -> list[str]:
    """Side prefixes a connector exists on: its "sides" key, default both legs (["R_", "L_"]); [""] for the unprefixed neck parts."""
    return spec.get("sides", ["R_", "L_"])


def plan_connector(body, iface: dict, connector: str, side: str, mesh=None, spec: dict | None = None) -> list[dict]:
    """The screws of one connector on one side ('R_' or 'L_'). `mesh(name)` returns a body-frame mesh (default body.mesh)."""
    mesh = mesh or body.mesh
    spec = spec or connectors()[connector]
    if side not in sides_of(spec):
        return []
    st = standards()
    clamped = [_fill(n, side) for n in spec["clamped"]]
    cm = [mesh(n) for n in clamped]
    plans = []
    for gname, g in spec["groups"].items():
        sp = _fill(g["standard_part"], side)
        for i, h in enumerate(hole_positions(iface, sp, g["holes"])):
            thr = h["thread"]
            fam = st["families_by_terminal_material"][h["terminal_material"]]["family"]
            head = g.get("head", "ISO 7380-1")
            hs = screw_spec(thr, fam, head)
            stk = stack_along_axis(cm, h["opening"], h["axis_out"])
            pick = choose_length(stk["stack_mm"], h["depth_mm"], h["reference_engagement_mm"], hs["lengths"], hs["point_mm"])
            plans.append(dict(
                name=_fill(g["name"], side).replace("{i}", str(i)), connector=connector, group=gname, side=side,
                owning_link=_fill(g["owning_link"], side), hole_id=h["hole_id"], standard_part=sp, thread=thr, family=fam,
                thread_label=(thr if fam == "M" else "PA" + thr.lstrip("M")), head_standard=head,
                head_dia_mm=hs["dk"], head_height_mm=hs["k"], terminal_material=h["terminal_material"],
                opening=np.asarray(h["opening"], float), axis_out=np.asarray(h["axis_out"], float),
                hole_depth_mm=h["depth_mm"], through=bool(h.get("through", False)),
                reference_engagement_mm=h["reference_engagement_mm"], clamped=clamped,
                stack_mm=stk["stack_mm"], head_gap_mm=stk["gap_mm"], stack_spread_mm=stk["spread_mm"], stack_rays=stk["rays"],
                point_mm=hs["point_mm"], **pick))
            plans[-1]["bom_group"] = f"{side}{gname}_{plans[-1]['thread_label']}x{pick['length_mm']:g}"
    return plans


def screw_mesh(plan: dict) -> trimesh.Trimesh:
    """Revolved envelope (head disc dk x k, shank at the thread's major diameter, 0.5 mm point) in the body frame: the
    head underside on the clamped outer face (opening + stack along the hole axis), the shank into the hole.
    Countersunk heads (plan["countersunk"]): a 90 deg cone from dk at the head top to the shank, the head top at
    opening + stack (the stack then runs to the surface the head sits flush in) and the nominal length includes the head."""
    d = float(plan["thread"].lstrip("M").replace("PA", ""))
    r, hd, hh, L, pt = d / 2.0, plan["head_dia_mm"], plan["head_height_mm"], plan["length_mm"], plan["point_mm"]
    if plan.get("countersunk"):
        prof = [(0.0, 0.0), (hd / 2, 0.0), (r, hh), (r, L - pt), (0.0, L)]
        m = trimesh.creation.revolve(np.array(prof), sections=48)
        a = np.asarray(plan["axis_out"], float); a /= np.linalg.norm(a)
        z = -a
        u = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
        T = np.eye(4); T[:3, :3] = np.c_[u, np.cross(z, u), z]
        T[:3, 3] = np.asarray(plan["opening"], float) + a * plan["stack_mm"]      # head top, flush
        m.apply_transform(T)
        if m.volume < 0:
            m.invert()
        return m
    prof = [(0.0, 0.0), (hd / 2, 0.0), (hd / 2, hh), (r, hh), (r, hh + L - pt), (0.0, hh + L)]   # (radius, along the screw from the head top)
    m = trimesh.creation.revolve(np.array(prof), sections=48)
    a = np.asarray(plan["axis_out"], float); a /= np.linalg.norm(a)
    z = -a                                              # proto +z runs head -> tip = into the hole
    u = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    T = np.eye(4); T[:3, :3] = np.c_[u, np.cross(z, u), z]
    T[:3, 3] = np.asarray(plan["opening"], float) + a * (plan["stack_mm"] + hh)     # head top
    m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    return m
