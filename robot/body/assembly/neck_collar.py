"""head_neck_collar screwed to the neck yaw-to-pitch link (Idle T-a2mt33; connectivity triage 2026-11-20: the collar touched nothing),
and rebuilt as a B-rep with a native STEP (owner 2026-11-22: "fix the collar so that it is fully BOM compliant").

The collar (body shell on link neck_yaw, tapered ring z 394-408 around the neck) came from
assemble_complete_fdm_body.build_cute_ape_head_shell as a mesh: manifold ring (tapered ellipse cones, scaled 1.04 / 0.94, centre
(-0.8, 3.5)) minus the head's outer envelope (a smoothed point-cloud mesh), then notched by motor_refit RANGE_NOTCHES over
neck_pitch. Here the same ring is a B-rep (ruled elliptic lofts) and both mesh cuts are parametric:
  - head envelope: a ruled surface z = z28(theta) + slope(theta) * (rho - 28) between two periodic splines (rho 25 and 34), prism up;
  - neck-pitch notch: a ruled surface z = z(theta), flat across rho 25..33 (square corners with the walls), prism up;
  both read from neck_collar_envelope.json (the envelope measured on the mesh collar of a88fc4d78 by vertical rays; the old mesh is
  gone from the tree). T-6rr3sa: `neck_collar.py sweep` re-measures what the moving head parts reach over the head poses
  (neck_pitch range every 2.5 deg + the pose grid collision_sweep takes for neck_pitch / neck_yaw, + 0.5 mm, then re-measured against the collar
  made with the stored cutters until nothing overlaps): the notch floor (pitch_notch_swept: neck pitch case, cradle) and a small drop of the
  head cut where the skin tilts into the ring (sweep_drop_mm); no overlap over the range. A trim removes the footprint where the head
  cut leaves under 1.4 mm over the flange (zero-thickness sliver); the notch floor is the only link between the ring's two halves.

Mount (measured on the unedited collar and link, mm in the manifest frame):
  the link's left plate (x -24.5..-21.0, y -4.4..14.4) faces the collar's left inner wall; the wall stands 1.9-3.8 mm off the
  plate at y 8-13 (the ring curves in toward +y). ONE M2 screw along x at y 9.0, z 399.8 (the collar's mid height: the dia 10 press
  counter disc of the sleeve (checks/press_lib ANVIL_D) then lies on the flat tab face, support 98 %; two screws stacked in z (4.5 apart)
  or side by side in y (7 apart) put the disc off the collar top or into the lower, front part where the head cut is low):
    - the plate is 3.5 mm, the thread reference in plastic 4.0: a boss on the plate's outer face (1.1 mm, y 5-13.3, z 394.4-405.4,
      with a plug in the plate's lightening window next to the pilot) makes the thread 4.6 deep: M2x8 engages 4.25; the collar's inward
      tab grows to the boss face (x -25.6): the clamped faces meet at 0.00;
    - the hole in the tab / wall carries a pressed 304 steel sleeve 3.6 / 2.2 x 3.5 (sleeve end flush with a flat head seat); its roof is a
      45 deg teardrop (the collar prints standing on its flange, the bore is horizontal);
    - the outer wall is cut by a vertical flat channel (floor x -29.1, 8.1 wide, depth 0..1.8 mm; nothing stands out): the screw head
      bears flat on it and the heated press tip (dia 8) has a free path; the flat tab face is the counter support;
    - a locating ridge on the tab face (trapezoid, 45 deg flanks, 0.8 high, along y at z 395.6..397.6) stands in a groove of the link boss
      (0.25 mm clearance): it keeps the collar from turning about the screw (about 3 deg of play), and slides in along y like the collar;
    - the link takes a dia 1.6 pilot (M2 tap).
  Tab, ridge, bore, channel, boss and groove are B-rep cuts and fuses. The tab stays on the inside; the channel is the only change seen from
  outside. Edges: the tab and ridge bed edges 0.5 mm fillets, the channel flank's top edge 0.7 mm (the head cut meets it in a 38 deg knife edge).
  The tab is bounded by the ring's own outer surface (cut from a box far outside it) so no seam edge is left on the outer wall.

  recreate:  .venv/bin/python -B robot/body/assembly/neck_collar.py build   (parts + records.json in OUT)
             .venv/bin/python -B robot/body/assembly/neck_collar.py apply [--variant M] [--out O]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
sys.path.insert(0, str(ROOT / "robot/body"))
sys.path.insert(0, str(ROOT / "robot/body/fastening"))

import os
OUT = ROOT / os.environ.get("NECK_COLLAR_OUT", "artifacts/robot/body/assembly/neck-collar")   # NECK_COLLAR_OUT: scratch folder inside the repo for iterations
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"   # unedited sources
VARIANT = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
TASK = "T-a2mt33"

COLLAR, LINK = "head_neck_collar", "combined__neck_yaw_to_pitch_link"
SLEEVES = "head_neck_collar_steel_sleeves"
LINK_SRC = "artifacts/robot/body/neck-head/parts/combined__neck_yaw_to_pitch_link.step"   # neck_parts output, never edited in place
SCREW_YZ = [(9.0, 399.8)]                         # (y, z) of the screw axis (along x): the collar's mid height, so the dia 10 press counter disc stays on the tab face
TAB_Y, TAB_Z = (2.8, 14.8), (394.2, 412.0)          # tab runs into the ring wall at y 14.8 and out through the head cut at the top (no tab edge meets the cut at a grazing angle)
RIDGE_Y, RIDGE_Z, RIDGE_H = (4.6, 16.5), (395.6, 397.6), 0.8   # locating ridge on the tab face (trapezoid, 45 deg flanks, base z 395.6..397.6, 0.8 high, merges into the wall at its +y end) in a groove of the link boss
GROOVE_CLEAR = 0.25                                # groove = ridge + this much on every flank and at the tip (FDM slip fit)
ROOF = True                                        # teardrop roof on the horizontal sleeve bore (print up +z)
WEDGE_R = 0.7                                      # fillet on the channel flank's top edge
BED_R = 0.5                                        # bed-edge fillet of the tab and ridge
SLEEVE_OD, SLEEVE_ID = 3.6, 2.2                    # kind already in the body (bore_for M2 2.2..2.4)
CHANNEL_Y = (4.95, 13.05)                          # spot-face channel: the screw at y 9 +- 4.05 (the heated press tip is dia 8, checks/press_lib.TIP_D)
TRIM_Z = 395.6                                     # the head cut never leaves a wall under 1.4 mm over the flange (perpendicular >= 2 lines on the 50 deg slope)
SWEEP_MARGIN = 0.5                                 # mm around the swept overlap of the moving parts
FLOOR_X = -29.1                                    # flat head seat = sleeve outer end
BOSS_T = 1.1                                       # link boss on the plate's outer face (x -24.5 -> -25.5): thread depth 3.5 -> 4.6: engagement 4.1 with the 0.5 mm tip margin (reference 4.0)
BOSS_Y, BOSS_Z = (5.0, 13.3), (394.4, 405.4)       # solid plate zone (the plate has lightening windows at y 4..6, z < 402)
FILL_Y, FILL_Z = (5.0, 8.5), (394.4, 402.5)        # lightening window of the plate (y 2.5..7.5, z up to 400.5) filled from y 5.0 on
PILOT_D = 1.6                                      # M2 tap diameter (standard-parts screw_families)
SCREW_NAME = "head_neck_collar_M2x8_{i}"


def _src():
    return {p["name"]: p for p in json.loads(SHELL_PASS.read_text())["parts"]}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _plate_faces(link_mesh):
    """Link plate outer face (towards the collar), back face, along +x at the screw axes."""
    out = []
    for _ in SCREW_YZ:
        loc, _, _ = link_mesh.ray.intersects_location([[-40.0, 9.0, 399.8]], [[1.0, 0.0, 0.0]])     # the plate is flat there; a screw may sit by the lightening window's rounded edge
        xs = sorted(float(l[0]) for l in loc)
        assert len(xs) >= 2, "no plate at y 9 z 399.8"
        out.append((xs[0], xs[1]))
    return out


ENVELOPE = HERE / "neck_collar_envelope.json"
CENTRE = (-0.8, 3.5)
SCALE_XY = (1.04, 0.94)


def _loft(r0, r1, h, z0):
    from build123d import Location, Plane, Solid, Wire
    pl = lambda z: Plane.XY.offset(z).move(Location((CENTRE[0], CENTRE[1], 0.0)))
    w0 = Wire.make_ellipse(r0 * SCALE_XY[0], r0 * SCALE_XY[1], pl(z0))
    w1 = Wire.make_ellipse(r1 * SCALE_XY[0], r1 * SCALE_XY[1], pl(z0 + h))
    return Solid.make_loft([w0, w1], ruled=True).wrapped


def _ruled_cutter(theta, z_of_rho, rho0, rho1, periodic, height=40.0):
    """Everything above the ruled surface through the splines at rho0 and rho1 (z_of_rho(i, rho) at knot i), up by `height`."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepFill import BRepFill
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.TColgp import TColgp_HArray1OfPnt
    from OCP.gp import gp_Pnt, gp_Vec

    def edge(rho):
        h = TColgp_HArray1OfPnt(1, len(theta))
        for i, t in enumerate(theta):
            a = np.radians(t)
            h.SetValue(i + 1, gp_Pnt(CENTRE[0] + rho * SCALE_XY[0] * np.cos(a), CENTRE[1] + rho * SCALE_XY[1] * np.sin(a),
                                     float(z_of_rho(i, rho))))
        g = GeomAPI_Interpolate(h, periodic, 1e-6); g.Perform()
        return BRepBuilderAPI_MakeEdge(g.Curve()).Edge()
    return BRepPrimAPI_MakePrism(BRepFill.Face_s(edge(rho0), edge(rho1)), gp_Vec(0, 0, height)).Shape()


def collar_ring_cutters(base=False):
    """base=True: the head cut without the sweep drop and no notch (what sweep() measures against)."""
    d = json.loads(ENVELOPE.read_text())
    if base:
        d = {k: v for k, v in d.items() if k not in ("sweep_drop_mm", "pitch_notch_swept")}
    env = np.array(d["envelope_top"]["rows"], float)
    nch = np.array(d.get("pitch_notch_swept", d["pitch_notch"])["rows"], float)
    k1 = d.get("pitch_notch_swept", d["pitch_notch"]).get("slope", d["pitch_notch"]["slope"])
    drop = np.asarray(d.get("sweep_drop_mm", [0.0] * len(env)), float)        # head skin swept over the neck pitch range: lowers the cut (sweep())
    head = _ruled_cutter(env[:, 0], lambda i, r: env[i, 1] + env[i, 2] * (r - 28.0) - drop[i], 21.0, 34.0, True)    # rho 21: the cut reaches in over the tab too (the ring itself starts at 26.4)
    notch = None if base else _ruled_cutter(nch[:, 0], lambda i, r: nch[i, 1] + k1 * (r - 27.5), 25.0, 33.0, False)
    return head, notch


def _outer_solid():
    """The ring's outside: the tapered cone and the flange (its top radius is the cone's at z 397.4: no ledge)."""
    import cad_ops as C
    return C.fuse(_loft(29.0, 28.5, 18.5, 394.5), _loft(32.2, 28.5 + 0.5 * (18.5 - 2.9) / 18.5, 3.2, 394.2))


def collar_ring(base=False):
    """The pure parametric ring (before the head envelope and notch cuts) and the two cutters."""
    import cad_ops as C
    ring = C.cut(_outer_solid(), _loft(26.4, 26.8, 21.0, 393.0))
    return ring, collar_ring_cutters(base)


def _prism(pts, vec):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Pnt, gp_Vec
    w = BRepBuilderAPI_MakePolygon()
    for q in pts:
        w.Add(gp_Pnt(*q))
    w.Close()
    return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(w.Wire()).Face(), gp_Vec(*vec)).Shape()


def _ridge_profile(fx, grow=0.0):
    """Trapezoid in (x, z): base on the tab face (x = fx, embedded 0.5 mm into the tab), 45 deg flanks to the tip at x = fx + RIDGE_H; `grow` offsets every side."""
    z0, z1 = RIDGE_Z; t = (z1 - z0) / 2 - RIDGE_H               # half width of the flat tip
    zc = 0.5 * (z0 + z1); g = grow * 1.41421356
    xt = fx + RIDGE_H + grow
    top = lambda x: (z1 + g) - (x - fx)                          # upper flank
    bot = lambda x: (z0 - g) + (x - fx)                          # lower flank
    return [(fx - 0.5, bot(fx - 0.5)), (fx - 0.5, top(fx - 0.5)), (xt, top(xt)), (xt, bot(xt))]


def _ridge(face_x):
    """The locating ridge on the tab face: runs along y (the way the collar slides on), merges into the ring wall at its +y end."""
    pts = [(x, RIDGE_Y[0], z) for x, z in _ridge_profile(face_x)]
    return _prism(pts, (0, RIDGE_Y[1] - RIDGE_Y[0], 0))


def _groove(fx):
    """The ridge's groove in the link boss: through the boss along y, ridge + GROOVE_CLEAR on every side."""
    pts = [(x, BOSS_Y[0] - 0.5, z) for x, z in _ridge_profile(fx, GROOVE_CLEAR)]
    return _prism(pts, (0, BOSS_Y[1] + 1.0 - BOSS_Y[0] + 0.5, 0))


def _round_flank_top(col):
    """The channel's low flank (a vertical plane, y CHANNEL_Y[0]) meets the head cut, which rises towards +y, in a 38 deg wedge: a knife edge under the line width.
    A 0.5 mm fillet on that one edge blunts it (OCC cannot round the spline rims in general; this edge it can)."""
    from build123d import Solid
    s = Solid(col)
    es = [e for e in s.edges() if abs(e.center().Y - CHANNEL_Y[0]) < 1e-3 and e.center().X < FLOOR_X - 0.2 and e.center().Z > 400.0 and e.geom_type.name == "BSPLINE"]
    if not es:
        return col
    r = s.fillet(WEDGE_R, es)
    assert r.is_valid and len(r.solids()) == 1, "flank fillet made an invalid solid"
    return r.wrapped


def _break_bed_edges(col):
    """Round the bed edges of the tab and the rib, the walls that stand against the link (elephant foot: the first layers flare against the clamped face),
    with a 0.5 mm fillet in one operation. The ring's own foot is left sharp: it is 4.1 mm wide, any bed edge treatment takes its flat contact under the 4 mm
    strip width (bed check: 51 -> 250-370 mm2 in thin strips), and OCC cannot round the 1.4 mm high vertical corners of the foot ends."""
    from build123d import Solid
    s = Solid(col)
    bot = [f for f in s.faces() if abs(f.normal_at().Z + 1) < 1e-6 and abs(f.center().Z - TAB_Z[0]) < 1e-3]
    es = [e for f in bot for e in f.edges() if all(p.X > -28.4 and TAB_Y[0] - 0.1 < p.Y < TAB_Y[1] + 0.1 for p in (e.start_point(), e.end_point(), e.center()))]
    r = s.fillet(BED_R, es)
    assert r.is_valid and len(r.solids()) == 1, "bed fillet made an invalid solid"
    return r.wrapped


def _trim_thin(col):
    """Drop the footprint where the head cut leaves less than TRIM_Z - 394.2 over the flange (the zero-thickness sliver of the cut meeting the base)."""
    import cad_ops as C
    from build123d import Solid
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec
    slab = C.common(col, C.box((-60, -60, TRIM_Z), (60, 60, TRIM_Z + 0.05)))
    bottoms = [f for f in Solid(slab).faces() if abs(f.normal_at().Z + 1) < 1e-6 and abs(f.center().Z - TRIM_Z) < 1e-3]
    keep = None
    for f in bottoms:
        pr = BRepPrimAPI_MakePrism(f.wrapped, gp_Vec(0, 0, 30.0)).Shape()
        dn = BRepPrimAPI_MakePrism(f.wrapped, gp_Vec(0, 0, -TRIM_Z + 390.0)).Shape()
        keep = C.fuse(pr, dn) if keep is None else C.fuse(keep, pr, dn)
    return C.common(col, keep)


def _head_poses(B):
    """Poses of the head against the collar's link (neck_yaw): the neck_pitch range every 2.5 deg plus every pose collision_sweep takes for the
    neck_pitch and neck_yaw joints (step 5 deg: each joint over its range, the neighbour joints at their ends and middle, the range corners),
    reduced to the distinct head-in-neck_yaw-link transforms (the collar sits on neck_yaw: the yaw turns collar and head together, only the pitch matters)."""
    from checks import bodylib as BL
    lo, hi = B.joints["neck_pitch"]["range_deg"]
    poses = [{"neck_pitch": float(a)} for a in list(np.arange(lo, hi + 1e-6, 2.5)) + [hi]]
    for j in ("neck_pitch", "neck_yaw"):
        poses += BL.pose_set(B, j, 5.0)[0]
    out, seen = [], set()
    for pose in poses:
        T = np.linalg.inv(B.link_T("neck_yaw", pose)) @ B.link_T("head", pose)
        k = tuple(np.round(T, 3).ravel())
        if k not in seen:
            seen.add(k); out.append((pose, T))
    return out


def sweep(rounds=4):
    """Measure the overlap of the collar (before the tab and bores) with every moving head part over the head poses (_head_poses) and store what the
    cutters must additionally remove in neck_collar_envelope.json (pitch_notch_swept, sweep_drop_mm). Round 0 measures against the base cutters;
    each further round measures the collar made with the stored cutters and deepens them where an overlap is left (tessellation between the 2 deg
    knots), until none is left (or `rounds` are used)."""
    import trimesh
    from checks import bodylib as BL, core as CORE
    from build123d import Solid, export_stl
    import cad_ops as C
    man = json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]
    man = json.loads((ROOT / man).read_text())
    B = BL.Body(man)
    poses = _head_poses(B)
    moving = {}
    for n, p in B.parts.items():
        if p.get("owning_link") != "head" or p.get("role") in ("screw", "fastener"):
            continue
        try:
            moving[n] = CORE.screen_solid(B, n, set())[0]
        except Exception:
            continue
    env0 = np.array(json.loads(ENVELOPE.read_text())["envelope_top"]["rows"], float)
    lo, hi = B.joints["neck_pitch"]["range_deg"]
    for rd in range(rounds):
        ring, (head, notch) = collar_ring(base=(rd == 0))
        col = _trim_thin(C.cut(ring, head))
        if notch is not None:
            col = C.cut(col, notch)
        stl = ROOT / ".cache/work/neck_collar_sweep.stl"; stl.parent.mkdir(parents=True, exist_ok=True)
        export_stl(Solid(col), str(stl), tolerance=0.02, angular_tolerance=0.1)
        cm = trimesh.load(stl, force="mesh", process=True)
        cmf = BL.to_mf(cm)
        SAMPLES = np.asarray(trimesh.sample.sample_surface(cm, 150000)[0])
        d = json.loads(ENVELOPE.read_text())
        drop = np.asarray(d.get("sweep_drop_mm", [0.0] * len(env0)), float) if rd else np.zeros(len(env0))
        add = np.zeros(len(env0))                                        # depth of the collar's surface points inside the moved skin, per theta knot
        npts, found = [], False
        for n, m in moving.items():
            for pose, T in poses:
                mm = m.copy(); mm.apply_transform(T)
                if np.any(mm.bounds[0] > cm.bounds[1] + 1) or np.any(mm.bounds[1] < cm.bounds[0] - 1):
                    continue
                ov = BL.to_mf(mm) ^ cmf
                if ov.volume() <= 1e-4:
                    continue
                found = True
                inside = mm.contains(SAMPLES)
                P = SAMPLES[inside]
                if not len(P):
                    continue
                if n.startswith("original_profile_head_skin"):
                    # vertical depth the collar's surface points reach into the moved skin (to the skin's lower surface), per theta knot
                    loc, ray, _ = mm.ray.intersects_location(P, np.tile([0.0, 0.0, -1.0], (len(P), 1)), multiple_hits=True)
                    th = np.degrees(np.arctan2((P[:, 1] - CENTRE[1]) / SCALE_XY[1], (P[:, 0] - CENTRE[0]) / SCALE_XY[0])) % 360.0
                    depth = np.zeros(len(P))
                    for l, r in zip(loc, ray):
                        depth[r] = max(depth[r], P[r, 2] - l[2])
                    for i, t0 in enumerate(env0[:, 0]):
                        dth = np.abs((th - t0 + 180.0) % 360.0 - 180.0)
                        sel = (dth <= 1.0) & (depth > (0.05 if rd == 0 else 0.0))        # < 0.05 mm is the tessellation noise of touching surfaces (round 0)
                        if np.any(sel):
                            add[i] = max(add[i], float(depth[sel].max()))
                else:
                    npts.append(P)
        print(f"sweep round {rd}: overlap left {found}", flush=True)
        if rd and not found:
            break
        if npts:                                           # notch: the surface z = z275(theta) + k1 (rho - 27.5) lies below every collar point a moving part reaches
            P = np.vstack(npts)
            th = np.degrees(np.arctan2((P[:, 1] - CENTRE[1]) / SCALE_XY[1], (P[:, 0] - CENTRE[0]) / SCALE_XY[0])) % 360.0
            zreq = P[:, 2] - SWEEP_MARGIN                  # flat floor (slope 0): square corners with the walls, no acute tips
            old = np.array(d["pitch_notch"]["rows"], float)
            prev = np.array(d["pitch_notch_swept"]["rows"], float) if rd else None
            t0, t1 = np.floor(th.min() / 2) * 2 - 4.0, np.ceil(th.max() / 2) * 2 + 4.0
            knots = np.arange(min(t0, old[0, 0]), max(t1, old[-1, 0]) + 1e-6, 2.0)
            z = np.full(len(knots), 412.0)                                             # no cut where nothing is reached
            if prev is not None:
                knots = np.union1d(knots, prev[:, 0]); z = np.interp(knots, prev[:, 0], prev[:, 1], left=412.0, right=412.0)
            for i, t in enumerate(knots):
                sel = np.abs(th - t) <= 1.0
                if sel.any():
                    z[i] = min(z[i], float(zreq[sel].min()))
            for _ in range(3):                                                         # soften the flanks: a knot is at most 3 mm above its deeper neighbour
                z = np.minimum(z, np.minimum(np.roll(z, 1), np.roll(z, -1)) + 3.0)
            z = np.where((z > 396.7) & (z < 397.9), 396.7, z)                          # keep the floor off the flange ledge (z 397.4): no 0.3 mm slab
            z[0] = z[-1] = 412.0
            d["pitch_notch_swept"] = dict(slope=0.0, columns=["theta_deg", "z275_mm"], rows=[[round(float(t), 2), round(float(v), 3)] for t, v in zip(knots, z)],
                                          note=f"pitch_notch extended to every collar point the moving head parts reach over the head poses (neck_pitch {lo}..{hi} deg, "
                                               f"2.5 deg steps, and collision_sweep's pitch x yaw grid) + {SWEEP_MARGIN} mm (`neck_collar.py sweep`, T-6rr3sa)")
        if add.max() > 0:
            add = np.where(add > 0, add + (SWEEP_MARGIN if rd == 0 else 0.15), 0.0)
            add = np.maximum(add, np.maximum(np.roll(add, 1), np.roll(add, -1)) * 0.5)       # neighbours follow at half depth (no spikes)
            drop = drop + add if rd else add
        d["sweep_drop_mm"] = np.round(drop, 3).tolist()
        d.pop("swept_hulls", None); d.pop("swept_hulls_doc", None)
        ENVELOPE.write_text(json.dumps(d, indent=1) + "\n")


def collar_solid(face_x):
    """The collar B-rep before the edges are broken; also returns the ring volume."""
    import cad_ops as C
    ring, (head, notch) = collar_ring()
    x0 = -36.0                                      # past the ring outside: the tab is bounded by the ring's outer surface itself (no seam edges)
    rib = _ridge(face_x)
    tab = C.common(C.box((x0, TAB_Y[0], TAB_Z[0]), (face_x, TAB_Y[1], TAB_Z[1])), _trim_thin(_outer_solid()))     # never outside the ring's outer surface (the foot trimmed vertically like the ring's)
    tab = C.cut(C.fuse(tab, rib), head)                     # the tab (and rib) from the plate face back into the ring wall, cut by the head envelope
    col = C.fuse(_trim_thin(C.cut(ring, head)), tab)         # the trim keeps the ring in one piece (on the ring alone: the tab would split its faces)
    col = C.cut(col, notch)                                  # the notch floor (z 394.8..) is the only link between the ring's two halves
    ring = C.fuse(ring, tab)
    v_ring = C.volume(ring)
    tools = []
    tools.append(C.box((FLOOR_X - 8.0, CHANNEL_Y[0], 390.0), (FLOOR_X, CHANNEL_Y[1], 420.0)))      # flat spot face channel: heads and the press tip
    for y, z in SCREW_YZ:
        tools.append(C.cylinder((FLOOR_X - 1.0, y, z), (face_x + 1.0, y, z), SLEEVE_OD / 2))             # sleeve bore
        if ROOF:                                                                                        # 45 deg roof tangent to the bore: no flat ceiling for the printer to bridge
            r = SLEEVE_OD / 2; c = r * 0.70710678
            tools.append(_prism([(FLOOR_X - 1.0, y - c, z + c), (FLOOR_X - 1.0, y + c, z + c), (FLOOR_X - 1.0, y, z + r * 1.41421356)], (face_x - FLOOR_X + 2.0, 0, 0)))
    col = C.unify(C.cut(col, *tools))
    return col, v_ring


def build_collar(face_x):
    import cad_ops as C
    import motor_refit as MR
    col, v_ring = collar_solid(face_x)
    col = _break_bed_edges(_round_flank_top(col))
    assert C.valid(col) and C.n_solids(col) == 1, "collar is not one valid solid"
    rec = MR.save(COLLAR, col, "printed")
    vol = _volume(col)                                          # GProp's default integration is 8 % off on the spline faces
    dens = json.loads((ROOT / "robot/body/checks/data/materials.json").read_text())["Bambu PAHT-CF"]["density_g_cm3"]
    rec.update(volume_mm3=round(vol, 2), mass_kg=round(vol * dens * 1e-6, 6), mass_basis=f"native STEP volume x {dens} g/cm3 (PAHT-CF)",
               mesh_note="parametric B-rep (robot/body/assembly/neck_collar.py, " + TASK + "): elliptic ring cut by the head envelope and the "
                         "neck-pitch notch (ruled splines measured from the old mesh), screw tab, head recesses and sleeve bores",
               neck_collar=f"{TASK}: screwed to the neck yaw-to-pitch link, 2 x M2x8 through {SLEEVE_OD}/{SLEEVE_ID} steel sleeves")
    print(COLLAR, "ring", round(v_ring, 1), "-> collar", rec["volume_mm3"], flush=True)
    return rec


def _volume(s):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g, 1e-5, False, False)
    return float(g.Mass())


def build_sleeves(face_x):
    import cad_ops as C
    import motor_refit as MR
    shapes = [C.cut(C.cylinder((FLOOR_X, y, z), (face_x, y, z), SLEEVE_OD / 2),
                    C.cylinder((FLOOR_X - 0.5, y, z), (face_x + 0.5, y, z), SLEEVE_ID / 2)) for y, z in SCREW_YZ]
    rec = MR.save(SLEEVES, MR.compound(shapes), "purchased")
    rec["mass_kg"] = round(rec["volume_mm3"] * 7.93e-6, 6)
    rec["mass_basis"] = "native STEP volume x 7.93 g/cm3 (304 SS)"
    return rec


def link_solid(faces):
    """The link with the boss, the window plug, the ridge groove and the pilots (before saving); also its original volume."""
    import cad_ops as C
    import motor_refit as MR
    s = MR.load_step(LINK_SRC)
    v0 = C.volume(s)
    fx0 = faces[0][0]
    s = C.fuse(s, C.box((fx0 - BOSS_T, BOSS_Y[0], BOSS_Z[0]), (fx0 + 1.0, BOSS_Y[1], BOSS_Z[1])))        # boss, overlaps the plate by 1 mm
    s = C.fuse(s, C.box((fx0 + 0.1, FILL_Y[0], FILL_Z[0]), (faces[0][1], FILL_Y[1], FILL_Z[1])))                # plug the lightening window edge next to the pilots (wall 0.47 -> 3 mm)
    s = _round_groove(C.cut(s, _groove(fx0 - BOSS_T)))
    tools = [C.cylinder((fx - BOSS_T - 1.0, y, z), (bx + 1.0, y, z), PILOT_D / 2) for (y, z), (fx, bx) in zip(SCREW_YZ, faces)]
    s = C.cut_many(s, tools, min_keep=0.1)
    return s, v0


def _round_groove(s):
    """0.3 mm fillets on the four long edges of the ridge groove (mouth and bottom): no straight profile corner left."""
    from build123d import Solid
    sol = Solid(s)
    es = [e for e in sol.edges() if e.geom_type.name == "LINE" and abs(e.length - (BOSS_Y[1] - BOSS_Y[0])) < 0.3 and
          RIDGE_Z[0] - 0.6 < e.center().Z < RIDGE_Z[1] + 0.6 and -26.2 < e.center().X < -23.9]
    assert len(es) == 4, f"ridge groove: {len(es)} long edges, expected 4"
    r = sol.fillet(0.3, es)
    assert r.is_valid and len(r.solids()) == 1, "groove fillet made an invalid solid"
    return r.wrapped


def build_link(faces):
    import cad_ops as C
    import motor_refit as MR
    s, v0 = link_solid(faces)
    assert C.valid(s) and C.n_solids(s) == 1, "link is not one valid solid"
    rec = MR.save(LINK, s, "printed")
    dens = json.loads((ROOT / "robot/body/checks/data/materials.json").read_text())["Bambu PAHT-CF"]["density_g_cm3"]
    rec.update(mass_kg=round(rec["volume_mm3"] * dens * 1e-6, 6), mass_basis=f"native STEP volume x {dens} g/cm3 (Bambu PAHT-CF, materials.json)")
    print(LINK, "volume", round(v0, 1), "->", rec["volume_mm3"], flush=True)
    return rec


def screw_rows(by, faces):
    import screw_plan as SP
    st = SP.standards()
    hs = SP.screw_spec("M2", "M", "ISO 7380-1")
    rows = []
    tmpl = by["upper__neck_pitch_horn_M2x5_0"]
    for i, ((y, z), (fx, bx)) in enumerate(zip(SCREW_YZ, faces)):
        stack = fx - FLOOR_X
        depth = bx - fx
        ref = 4.0                                              # plastic: 2.0 x 2 x thread radius (mount_check thread_in_part)
        pick = SP.choose_length(stack, depth, ref, hs["lengths"], hs["point_mm"])
        pl = dict(thread="M2", head_dia_mm=hs["dk"], head_height_mm=hs["k"], length_mm=pick["length_mm"], point_mm=hs["point_mm"],
                  opening=np.array([fx, y, z]), axis_out=np.array([-1.0, 0.0, 0.0]), stack_mm=stack, countersunk=False)
        m = SP.screw_mesh(pl)
        name = SCREW_NAME.format(i=i)
        npz = OUT / "parts" / f"{name}.npz"
        np.savez_compressed(npz, vertices=np.asarray(m.vertices, np.float32), faces=np.asarray(m.faces, np.int32))
        q = {k: copy.deepcopy(v) for k, v in tmpl.items() if k in ("owning_link", "role", "color", "display_group", "display_groups",
                                                                   "overlay_id", "parent_assembly", "inspection_reference", "mesh_units")}
        lbl = f"M2x{pick['length_mm']:g}"
        q.update(name=name, physical_part_id=name, bom_group=f"collar_{lbl}", mesh=str(npz.relative_to(ROOT)), mesh_sha256=_sha(npz),
                 bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()],
                 volume_mm3=round(float(m.volume), 3), mass_kg=round(float(m.volume) * st["density_g_cm3"] / 1000.0, 7),
                 material=f"ISO 7380-1 {lbl} into the link plate (printed pilot, dia {PILOT_D}); head dia {hs['dk']} x {hs['k']}, shank dia 2, "
                          f"{hs['point_mm']} mm point (envelope)",
                 mass_basis="mesh volume x 7.85 g/cm3", neck_collar=f"generated by robot/body/assembly/neck_collar.py ({TASK})",
                 screw_plan=dict(hole_id=f"{LINK}:collar.{i}", thread_label="M2", head_standard="ISO 7380-1", hole_depth_mm=round(depth, 3),
                                 reference_engagement_mm=ref, stack_mm=round(stack, 3), head_gap_mm=0.0, length_mm=pick["length_mm"],
                                 engagement_mm=pick["engagement_mm"], tip_margin_mm=pick["tip_margin_mm"], shortfall_mm=pick["shortfall_mm"],
                                 overrun_mm=pick["overrun_mm"], stack_change_to_fit_mm=pick["stack_change_to_fit_mm"]))
        rows.append(q)
        print(name, lbl, "stack", round(stack, 3), "engagement", pick["engagement_mm"], "tip margin", pick["tip_margin_mm"], flush=True)
    return rows


def build():
    import motor_refit as MR
    from checks import bodylib as BL
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    MR.PARTS = OUT / "parts"
    by = _src()
    link_m = BL.load_mesh(LINK_SRC.replace(".step", ".npz"))
    faces = _plate_faces(link_m)
    assert all(abs(f[0] - faces[0][0]) < 0.01 for f in faces), "plate face is not flat across the screws"
    link_faces = faces                                   # plate faces: the boss is cut from these in build_link
    faces = [(fx - BOSS_T, bx) for fx, bx in faces]     # the clamped face is the boss face
    face_x = faces[0][0]
    collar = build_collar(face_x)
    sleeves = build_sleeves(face_x)
    link = build_link(link_faces)
    s0 = by[LINK]
    srows = screw_rows({**by, "upper__neck_pitch_horn_M2x5_0": _selected()["upper__neck_pitch_horn_M2x5_0"]}, faces)
    sl = _selected()["combined__neck_yaw_to_pitch_link_steel_sleeves"]
    sleeve_row = {k: copy.deepcopy(v) for k, v in sl.items() if k in ("owning_link", "role", "color", "display_group", "display_groups",
                                                                       "parent_assembly", "inspection_reference", "mesh_units", "material")}
    sleeve_row.update(name=SLEEVES, physical_part_id=SLEEVES, bom_group=SLEEVES, **sleeves, neck_collar=f"{TASK}: 2 sleeves {SLEEVE_OD}/{SLEEVE_ID}")
    (OUT / "records.json").write_text(json.dumps(dict(collar=collar, link=link, new_rows=[sleeve_row] + srows,
                                                      screws_yz=SCREW_YZ, plate_faces=faces), indent=1) + "\n")
    print("wrote", (OUT / "records.json").relative_to(ROOT), flush=True)


def _selected():
    """Template rows (the neck pitch horn screw, the link's sleeve row): the shell-pass manifest this generator already declares as input.
    It never reads a later step's output (assembly-parts-shoulder-u.json is written after this step)."""
    return _src()


def swap_parts(parts):
    """The part list with the collar mount applied (idempotent): collar mesh (no native STEP), link with pilots, sleeves, 2 screws."""
    if not (OUT / "records.json").exists():
        print(f"WARNING neck collar: {(OUT / 'records.json').relative_to(ROOT)} missing, head_neck_collar stays unfastened; "
              f"run `.venv/bin/python -B robot/body/assembly/neck_collar.py build` ({TASK})")
        return parts
    d = json.loads((OUT / "records.json").read_text())
    new = {r["name"] for r in d["new_rows"]}
    res = []
    for p in parts:
        if p["name"] in new or p["name"].startswith("head_neck_collar_M2x8_"):       # the screws of an earlier build (a screw count that changed) go too
            continue
        if p["name"] == COLLAR:
            q = dict(p)
            for k in ("native_step", "native_step_sha256", "native_step_offset_mm", "placement_offset_mm"):
                q.pop(k, None)
            q.update(d["collar"])
            p = q
        elif p["name"] == LINK:
            q = dict(p)
            for k in ("native_step_offset_mm", "placement_offset_mm"):
                q.pop(k, None)
            q.update(d["link"])
            q["neck_collar"] = f"{TASK}: two dia {PILOT_D} M2 pilots through the plate for the collar screws"
            p = q
        res.append(p)
    return res + [copy.deepcopy(r) for r in d["new_rows"]]


def apply(variant: Path = VARIANT, out: Path | None = None):
    man = json.loads(variant.read_text())
    n0 = len(man["parts"])
    man["parts"] = swap_parts(man["parts"])
    (out or variant).write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out or variant, n0, "->", len(man["parts"]), "parts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "apply", "sweep"])
    ap.add_argument("--variant", default=str(VARIANT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.phase == "sweep":
        sweep()
    else:
        build() if a.phase == "build" else apply(Path(a.variant), Path(a.out) if a.out else None)
