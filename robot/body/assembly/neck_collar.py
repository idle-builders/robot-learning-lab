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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("neck-collar")   # configured values: robot/body/assembly/neck_collar.bom.yaml config:
OUT = ROOT / os.environ.get("NECK_COLLAR_OUT", "artifacts/robot/body/assembly/neck-collar")   # NECK_COLLAR_OUT: scratch folder inside the repo for iterations
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"   # unedited sources
VARIANT = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
TASK = CFG["TASK"]

COLLAR, LINK = CFG["COLLAR"], CFG["LINK"]
SLEEVES = CFG["SLEEVES"]
LINK_SRC = "artifacts/robot/body/neck-head/parts/combined__neck_yaw_to_pitch_link.step"   # neck_parts output, never edited in place
SCREW_YZ = list(CFG["SCREW_YZ"])
BED_Z = CFG["BED_Z"]
TAB_Y, TAB_Z = (2.8, 14.8), (BED_Z, 412.0)          # tab runs into the ring wall at y 14.8 and out through the head cut at the top (no tab edge meets the cut at a grazing angle)
RIDGE_Y, RIDGE_Z, RIDGE_H = CFG["RIDGE_Y"], CFG["RIDGE_Z"], CFG["RIDGE_H"]
GROOVE_CLEAR = CFG["GROOVE_CLEAR"]
ROOF = CFG["ROOF"]
WEDGE_R = CFG["WEDGE_R"]
BED_R = CFG["BED_R"]
SLEEVE_OD, SLEEVE_ID = CFG["SLEEVE_OD"], CFG["SLEEVE_ID"]
CHANNEL_Y = CFG["CHANNEL_Y"]
TRIM_Z = CFG["TRIM_Z"]
SWEEP_MARGIN = CFG["SWEEP_MARGIN"]
FLOOR_X = CFG["FLOOR_X"]
BOSS_T = CFG["BOSS_T"]
BOSS_Y, BOSS_Z = CFG["BOSS_Y"], CFG["BOSS_Z"]
FILL_Y, FILL_Z = CFG["FILL_Y"], CFG["FILL_Z"]
PILOT_D = CFG["PILOT_D"]
SCREW_NAME = CFG["SCREW_NAME"]


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
CENTRE = CFG["CENTRE"]
SCALE_XY = CFG["SCALE_XY"]


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
        g = GeomAPI_Interpolate(h, periodic, 1e-6)
        g.Perform()
        return BRepBuilderAPI_MakeEdge(g.Curve()).Edge()
    return BRepPrimAPI_MakePrism(BRepFill.Face_s(edge(rho0), edge(rho1)), gp_Vec(0, 0, height)).Shape()


def _smooth_head_cut(env, drop):
    """The measured head cut (z and slope per theta knot) as a smooth periodic function on a uniform 2 deg grid, never above the measurement.
    The knots are measured by rays: their noise bends the rim (second difference up to 0.8 mm per knot, a radius of 0.3 mm) so no fillet rolls along it
    (OCC fails at every radius). A gaussian of one knot (2 deg) removes the noise; the surface is then lowered by the largest amount it stands above the
    measurement (over the wall's radii), itself smoothed, so the cut only ever removes more (<= 0.6 mm), never less."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    th = np.arange(0.0, 360.0, 2.0)
    zr = env[:, 1] - drop
    ok = env[:, 0] < 360.0 - 1e-6
    z0 = np.interp(th, env[ok, 0], zr[ok], period=360.0)
    k0 = np.interp(th, env[ok, 0], env[ok, 2], period=360.0)
    near = np.abs(((env[:, 0][None, :] - th[:, None] + 180.0) % 360.0) - 180.0) <= 1.0          # raw knots within one grid knot's reach
    rhos = (24.0, 26.4, 28.0, 30.0, 33.5)
    low = [np.array([(zr + env[:, 2] * (r - 28.0))[n].min() for n in near]) for r in rhos]       # the lowest measurement a grid knot stands for
    z, k = gaussian_filter1d(z0, 1.0, mode="wrap"), gaussian_filter1d(k0, 1.0, mode="wrap")
    lift = np.zeros(len(th))
    for _ in range(6):
        above = np.maximum(np.max([z - lift + k * (r - 28.0) - lo for r, lo in zip(rhos, low)], axis=0), 0.0)
        if above.max() < 1e-3:
            break
        lift = lift + gaussian_filter1d(maximum_filter1d(above, 5, mode="wrap"), 1.0, mode="wrap") + 0.01
    zc = z - lift
    # @lore the gaussian and the lift lower the two rim peaks (theta 6-12 and 168-176, tents of two knots) by 0.3-1.2 mm, which moves the FEM load patch (rim within
    # 2 mm of the head skin) off the places the skin comes closest; the peak's two knots and their neighbours stay at the measurement: a land two knots (1 mm) wide.
    for lo, hi in ((0.0, 100.0), (150.0, 200.0)):
        sel = np.where((th >= lo) & (th < hi))[0]
        top = sel[np.argsort(z0[sel])[-2:]]
        top = np.unique(np.concatenate([top - 1, top, top + 1]))
        zc[top] = np.maximum(zc[top], z0[top])
    return th, zc, k


def _smooth_floor(z):
    """The notch floor (z per 2 deg knot, 412 = no cut) as a smooth curve, never above the stored values: the stored floor has steps of 1.3 mm per knot
    and knots 0.2 mm apart, whose sharp edges and slivers no fillet rolls along. Gaussian of 1.5 knots over the values capped at the head cut's height, lowered by
    what it stands above the stored values (smoothed itself), and kept at or above NOTCH_MIN."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    zc = np.minimum(np.asarray(z, float), 406.0)
    f = gaussian_filter1d(zc, 1.5, mode="nearest")
    lift = np.zeros(len(zc))
    for _ in range(8):
        above = np.maximum(f - lift - zc, 0.0)
        if above.max() < 1e-3:
            break
        lift = lift + gaussian_filter1d(maximum_filter1d(above, 3), 1.0, mode="nearest") + 0.01
    return np.maximum(f - lift, NOTCH_MIN)                             # the foot band under the floor is at least two lines (0.84 mm) thick (walls check); sweep() measures the head clearance against it


PEAK_CAP_Z = CFG["PEAK_CAP_Z"]
PEAK_CAP_SECTOR = CFG["PEAK_CAP_SECTOR"]


def _peak_cap():
    """Everything above z PEAK_CAP_Z within a sector of theta (the head cutter's addition). @lore the swept rim peak (406.8) is as high as the head skin allows and
    round 2's was 406.56: cut flat at 12.4 over the bed (the slicer's top layer middle is 12.3: no sliver on top, the slice below is 0.9 mm wide, the rim slopes 0.33 up outwards)."""
    import cad_ops as C
    pts = [(CENTRE[0], CENTRE[1])] + [(CENTRE[0] + 40.0 * np.cos(np.radians(t)), CENTRE[1] + 40.0 * np.sin(np.radians(t)))
                                      for t in np.linspace(*PEAK_CAP_SECTOR, 12)]
    return _prism([(x, y, PEAK_CAP_Z) for x, y in pts], (0, 0, 30.0))


def collar_ring_cutters(base=False, lower=0.0):
    """base=True: the head cut without the sweep drop and no notch (what sweep() measures against). lower: the head cut moved down by this much (the tab's top)."""
    d = json.loads(ENVELOPE.read_text())
    if base:
        d = {k: v for k, v in d.items() if k not in ("sweep_drop_mm", "pitch_notch_swept")}
    env = np.array(d["envelope_top"]["rows"], float)
    nch = np.array(d.get("pitch_notch_swept", d["pitch_notch"])["rows"], float)
    k1 = d.get("pitch_notch_swept", d["pitch_notch"]).get("slope", d["pitch_notch"]["slope"])
    drop = np.asarray(d.get("sweep_drop_mm", [0.0] * len(env)), float)        # head skin swept over the neck pitch range: lowers the cut (sweep())
    th, zs, sl = _smooth_head_cut(env, drop)
    head = _ruled_cutter(th, lambda i, r: zs[i] - lower + sl[i] * (r - 28.0), 21.0, 34.0, True)    # rho 21: the cut reaches in over the tab too (the ring itself starts at 26.4)
    import cad_ops as C
    head = C.fuse(head, _peak_cap()) if PEAK_CAP_Z else head
    nz = _smooth_floor(nch[:, 1]) if not base else None
    notch = None if base else _ruled_cutter(nch[:, 0], lambda i, r: nz[i] + k1 * (r - 27.5), 25.0, 33.0, False)
    return head, notch


NOTCH_MIN = CFG["NOTCH_MIN"]
FOOT_R = CFG["FOOT_R"]
FOOT_H = 395.6 - BED_Z                             # height of the vertical foot (BED_Z .. 395.6)
ROUND_R = CFG["ROUND_R"]
TAB_LOWER = CFG["TAB_LOWER"]
TAB_EDGE_R = CFG["TAB_EDGE_R"]
TAB_BED_R = CFG["TAB_BED_R"]
TAB_R = CFG["TAB_R"]
PLAN_SKEW = CFG["PLAN_SKEW"]
END_OFF = CFG["END_OFF"]
PLAN_R_MOUTH = CFG["PLAN_R_MOUTH"]
PLAN_R_MOUTH_LOW = CFG["PLAN_R_MOUTH_LOW"]
PLAN_R = CFG["PLAN_R"]
PLAN_R_INNER = CFG["PLAN_R_INNER"]
BED_ROUND = CFG["BED_ROUND"]


R_IN0 = CFG["R_IN0"]


KEEP_OUT = CFG["KEEP_OUT"]


def _r_in(z):
    return R_IN0 + 0.4 * (z - 393.0) / 21.0           # inner wall: the old loft 26.4 (z 393) -> 26.8 (z 414)


def _r_out(z):
    return 29.0 - 0.5 * (z - 394.5) / 18.5          # outer cone: 29.0 (z 394.5) -> 28.5 (z 413)


def _revolved(pts, rounds):
    """Revolve the (r, z) polygon about z, round its corners in the profile (rounds: {vertex index: radius}) and scale / move it to the ring's ellipse
    (the scale is the one of the old ruled ellipse lofts, so a cone stays the same cone). Rounded by construction: no sharp edge across the section."""
    from build123d import Axis, Face, Polyline, Solid
    from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
    from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ
    f = Face(Polyline(*[(r, 0.0, z) for r, z in pts], close=True))
    for i, rad in rounds.items():
        r, z = pts[i]
        f = f.fillet_2d(rad, [v for v in f.vertices() if abs(v.X - r) < 1e-6 and abs(v.Z - z) < 1e-6])
    g = gp_GTrsf(); g.SetVectorialPart(gp_Mat(SCALE_XY[0], 0, 0, 0, SCALE_XY[1], 0, 0, 0, 1)); g.SetTranslationPart(gp_XYZ(CENTRE[0], CENTRE[1], 0.0))
    return BRepBuilderAPI_GTransform(Solid.revolve(f, 360, Axis.Z).wrapped, g, True).Shape()


def _profile(inner):
    """Section of the ring (r, z): inner wall, bed, a vertical foot, the 45 deg flange cone, the outer cone. inner=False: the same outside with the axis for the inner wall (the tab is bounded by it)."""
    zt = 414.0
    s = (1.025 - 0.5 / 18.5)                        # flange cone (slope 1.025 r per z) meets the outer cone at u above the foot top
    u = (FOOT_R - _r_out(BED_Z + FOOT_H)) / s
    zj = BED_Z + FOOT_H + u; rj = FOOT_R - 1.025 * u
    r0 = _r_in(BED_Z) if inner else 0.0
    pts = [(r0, BED_Z), (FOOT_R, BED_Z), (FOOT_R, BED_Z + FOOT_H), (rj, zj), (_r_out(zt), zt), (_r_in(zt) if inner else 0.0, zt)]
    rounds = {1: BED_ROUND, 2: 1.5, 3: 3.0}
    if inner:
        rounds[0] = BED_ROUND
    return pts, rounds


def _outer_solid():
    """The ring's outside (full, no bore): foot, flange and tapered cone with the roundings in the section."""
    pts, rounds = _profile(False)
    return _revolved(pts, rounds)


def collar_ring(base=False):
    """The pure parametric ring (before the head envelope and notch cuts) and the two cutters."""
    import cad_ops as C
    pts, rounds = _profile(True)
    ring = _revolved(pts, rounds)
    if R_IN0 < 26.4:                                  # the inward growth stays out of the sectors the neck pitch parts reach into (sweep()) and the link plate stands in
        cx, cy = CENTRE; sx, sy = SCALE_XY
        for a0, a1 in KEEP_OUT:
            arc = [(cx + 45.0 * sx * np.cos(np.radians(t)), cy + 45.0 * sy * np.sin(np.radians(t))) for t in np.linspace(a0, a1, 14)]
            wedge = _prism([(cx, cy, 380.0)] + [(x, y, 380.0) for x, y in arc], (0, 0, 50.0))
            ring = C.cut(ring, C.common(wedge, _loft(26.4, 26.8, 21.0, 393.0)))
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
    """The locating ridge on the tab face: runs along y (the way the collar slides on), merges into the ring wall at its +y end. Its start is a 45 deg ramp
    (a planar bevel, no square end), so every edge of the ridge is a bevel edge of 45-60 deg."""
    import cad_ops as C
    pts = [(x, RIDGE_Y[0], z) for x, z in _ridge_profile(face_x)]
    r = _prism(pts, (0, RIDGE_Y[1] - RIDGE_Y[0], 0))
    y0 = RIDGE_Y[0]; z0, z1 = RIDGE_Z[0] - 1.0, RIDGE_Z[1] + 1.0
    ramp = _prism([(face_x - 0.5, y0 - 0.5, z0), (face_x + 1.5, y0 + 1.5, z0), (face_x + 1.5, y0 - 1.0, z0)], (0, 0, z1 - z0))   # xy triangle above the 45 deg line through (face_x, y0)
    return C.cut(r, ramp)


def _groove(fx):
    """The ridge's groove in the link boss: through the boss along y, ridge + GROOVE_CLEAR on every side."""
    pts = [(x, BOSS_Y[0] - 0.5, z) for x, z in _ridge_profile(fx, GROOVE_CLEAR)]
    return _prism(pts, (0, BOSS_Y[1] + 1.0 - BOSS_Y[0] + 0.5, 0))


def _end_angles(col):
    """Elliptic angles (deg) of the ring's two ends at the foot: the most retracted point of each trimmed end face (the end faces cross the band obliquely,
    the tip of one side reaches further): (left end, right end), from the B-rep vertices of the foot."""
    from build123d import Solid
    th = []
    for v in Solid(col).vertices():
        if v.Z < 395.7:
            th.append(np.degrees(np.arctan2((v.Y - CENTRE[1]) / SCALE_XY[1], (v.X - CENTRE[0]) / SCALE_XY[0])) % 360.0)
    th = np.array(th)
    return float(th[(th > 190) & (th < 215)].min()), float(th[(th > 325) & (th < 350)].max())


def _fillet_corner(curve_a, curve_b, r, hint):
    """Centre, tangent points and arc midpoint of the circle of radius r tangent to an ellipse and a line (plan, mm). curve_a = ('ell', rho, material_inside),
    curve_b = ('line', p0, direction, normal into the material); hint: a point near the corner (the line meets the ellipse twice)."""
    from scipy.optimize import brentq
    _, rho, inside = curve_a
    _, p0, _d, nin = curve_b
    a, b = rho * SCALE_XY[0], rho * SCALE_XY[1]
    sg = -1.0 if inside else 1.0
    P = lambda t: np.array([CENTRE[0] + a * np.cos(t), CENTRE[1] + b * np.sin(t)])
    def nrm(t):
        n = np.array([np.cos(t) / a, np.sin(t) / b]); return n / np.linalg.norm(n)
    C = lambda t: P(t) + sg * r * nrm(t)
    g = lambda t: float((C(t) - p0) @ nin) - r
    ts = np.radians(np.arange(0.0, 360.0, 0.5)); gs = [g(t) for t in ts]
    roots = []
    for i in range(len(ts) - 1):
        if gs[i] * gs[i + 1] < 0:
            roots.append(brentq(g, ts[i], ts[i + 1], xtol=1e-12))
    t = min(roots, key=lambda t: np.linalg.norm(P(t) - hint))
    c = C(t); T1 = c - r * nin; T2 = P(t)
    m = (T1 - c) + (T2 - c); M = c + r * m / np.linalg.norm(m)
    return c, T1, T2, M, sg, nrm(t)


def _round_bed_plan(col):
    """The ring's bed outline rounded by construction: the ring ends cut radial (perpendicular to the band, at the more retracted tip of each trimmed end) and the
    four end corners and the two channel mouth corners cut with plan fillets of PLAN_R, as prisms of the exact circle arcs (tangent to the foot's outer / inner
    cylinders and to the end / flank planes) through the whole height. The vertical seams where an arc meets a cylinder or plane are smooth (G1)."""
    import cad_ops as C
    cx, cy = CENTRE; sx, sy = SCALE_XY
    tl, tr = _end_angles(col)
    far = 46.0
    ends = []
    for th, side in ((tl, -1.0), (tr, 1.0)):                          # the end plane is parallel to the radial direction, END_OFF mm inside the retracted tip (a meridian plane
        d = np.array([sx * np.cos(np.radians(th)), sy * np.sin(np.radians(th))]); d /= np.linalg.norm(d)    # through the axis makes OCC's fillets of its edges fail)
        nin = side * np.array([-d[1], d[0]])
        ends.append((d, nin, np.array([cx, cy]) + nin * END_OFF))
    (dl, nl, pl), (dr, nr, pr) = ends
    arcs = [np.array([cx, cy]) + far * np.array([sx * np.cos(np.radians(t)), sy * np.sin(np.radians(t))]) for t in np.linspace(tl, tr, 9)[1:-1]]
    pts = [pl, pl + dl * far] + arcs + [pr + dr * far, pr, np.array([cx, cy])]
    col = C.cut(col, _prism([(x, y, 390.0) for x, y in pts], (0, 0, 30.0)))
    ro, ri = FOOT_R, _r_in(BED_Z)
    corners = []
    for d, nin, p0 in ends:
        corners.append((("ell", ro, True), ("line", p0, d, nin), p0 + d * (ro * 0.5 * (sx + sy) + 1.0)))
        corners.append((("ell", ri, False), ("line", p0, d, nin), p0 + d * (ri * 0.5 * (sx + sy) - 1.0)))
    for y, nin in ((CHANNEL_Y[0], (0.0, -1.0)), (CHANNEL_Y[1], (0.0, 1.0))):
        corners.append((("ell", ro - PLAN_SKEW, True), ("line", np.array([-33.0, y]) + np.array(nin) * PLAN_SKEW, np.array([1.0, 0.0]), np.array(nin)), np.array([-32.5, y])))
    for ca, cb, hint in corners:
        c, T1, T2, M, sg, n2 = _fillet_corner(ca, cb, (PLAN_R_MOUTH if hint[1] > 10.0 else PLAN_R_MOUTH_LOW) if abs(hint[0] + 32.5) < 1e-6 else (PLAN_R if ca[2] else PLAN_R_INNER), hint)
        nin = cb[3]
        W1 = T1 - nin * 3.0
        W2 = T2 - sg * n2 * 3.0                                       # away from the material
        X = W1 + (W2 - T2)
        from build123d import Edge, Face, Wire, Vector, ThreePointArc
        z = 390.0
        v = lambda q: Vector(float(q[0]), float(q[1]), z)
        arc = Edge.make_three_point_arc(v(T1), v(M), v(T2))
        w = Wire([arc, Edge.make_line(v(T2), v(W2)), Edge.make_line(v(W2), v(X)), Edge.make_line(v(X), v(W1)), Edge.make_line(v(W1), v(T1))])
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Vec
        cut = BRepPrimAPI_MakePrism(Face(w).wrapped, gp_Vec(0, 0, 30.0)).Shape()
        col = C.cut(col, cut)
    return col


def _round_foot_corners(col, radii=(0.4,)):
    """The channel floor's two vertical foot corners (0.65 - 1.0 mm high, between the bed roundings and the flange) with 0.4 mm in one operation, before the generic
    rounding: without it the final rounding of the tab's edges fails."""
    from build123d import Solid
    s = Solid(col)
    es = []
    for e in s.edges():
        bb = e.bounding_box(); c = e.center()
        dz = bb.max.Z - bb.min.Z; dxy = max(bb.max.X - bb.min.X, bb.max.Y - bb.min.Y)
        seam = abs(c.Y - CENTRE[1]) < 0.05 and abs(c.X) > 20.0                       # the revolve's seam
        if c.Z < 396.0 and dz > 0.3 and dxy < dz * 1.5 and not seam and abs(c.X - FLOOR_X) < 0.5:       # the channel floor's corners: the ring ends and the mouth corners are cut away by _round_bed_plan
            es.append(e)
    for r in radii:
        try:
            t = s.fillet(r, es)
        except ValueError:
            continue
        if t.is_valid and len(t.solids()) == 1:
            return t.wrapped
    return col


def _round_flank_top(col):
    """Each flank of the channel meets the head cut, which rises towards it, in a wedge of 38-50 deg: a knife edge under the line width. A fillet on that one edge
    blunts it (0.7 mm, the largest the edge takes; OCC cannot round the long spline rims in general, these edges it can)."""
    from build123d import Solid
    for y in CHANNEL_Y:
        s = Solid(col)
        es = [e for e in s.edges() if abs(e.center().Y - y) < 1e-3 and e.center().X < FLOOR_X - 0.2 and e.center().Z > 400.0 and e.geom_type.name == "BSPLINE"]
        for r in (WEDGE_R, 0.5, 0.35):
            try:
                t = s.fillet(r, es)
            except ValueError:
                continue
            if es and t.is_valid and len(t.solids()) == 1:
                col = t.wrapped
                break
    return col


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


def _trim_thin(col, head):
    """Drop the footprint where the head cut leaves less than TRIM_Z - 394.2 over the flange (the zero-thickness sliver of the cut meeting the base):
    the plan region where the cut is above TRIM_Z over a plain wide annulus (not the ring's own section: its rounded foot stays), prism down and up."""
    import cad_ops as C
    from build123d import Solid
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec
    wide = C.cut(_loft(40.0, 40.0, 30.0, 385.0), _loft(15.0, 15.0, 30.0, 385.0))
    slab = C.common(C.cut(wide, head), C.box((-60, -60, TRIM_Z), (60, 60, TRIM_Z + 0.05)))
    bottoms = [f for f in Solid(slab).faces() if abs(f.normal_at().Z + 1) < 1e-6 and abs(f.center().Z - TRIM_Z) < 1e-3]
    keep = None
    for f in bottoms:
        pr = BRepPrimAPI_MakePrism(f.wrapped, gp_Vec(0, 0, 30.0)).Shape()
        dn = BRepPrimAPI_MakePrism(f.wrapped, gp_Vec(0, 0, -TRIM_Z + 380.0)).Shape()
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
        col = _trim_thin(C.cut(ring, head), head)
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


def _sharp_edges(shape, over_deg=20.0):
    """The shape's edges whose two faces meet at more than over_deg (the edges check's SHARP_DEG), as (edge, deviation deg)."""
    import math
    from build123d import Edge, Solid
    from OCP.BRep import BRep_Tool
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.gp import gp_Pnt, gp_Vec

    def normal(face, p):
        pr = GeomAPI_ProjectPointOnSurf(p, BRep_Tool.Surface_s(face))
        if pr.NbPoints() == 0:
            return None
        u, v = pr.LowerDistanceParameters(); q, n = gp_Pnt(), gp_Vec()
        BRepGProp_Face(face).Normal(u, v, q, n)
        a = np.array([n.X(), n.Y(), n.Z()]); L = np.linalg.norm(a)
        return a / L if L > 1e-12 else None
    sol = Solid(shape)
    ef = TopTools_IndexedDataMapOfShapeListOfShape(); TopExp.MapShapesAndAncestors_s(sol.wrapped, TopAbs_EDGE, TopAbs_FACE, ef)
    out = []
    for i in range(1, ef.Extent() + 1):
        lst = ef.FindFromIndex(i)
        if lst.Extent() != 2:
            continue
        f1, f2 = TopoDS.Face_s(lst.First()), TopoDS.Face_s(lst.Last())
        if f1.IsSame(f2):
            continue
        e = Edge(TopoDS.Edge_s(ef.FindKey(i))); m = e.position_at(0.5); p = gp_Pnt(m.X, m.Y, m.Z)
        n1, n2 = normal(f1, p), normal(f2, p)
        if n1 is None or n2 is None:
            continue
        d = math.degrees(math.acos(float(np.clip(n1 @ n2, -1, 1))))
        if d > over_deg:
            out.append((e, d))
    return sol, out


def _round_sharp(col, radius=ROUND_R, keep=lambda e: False):
    """Fillet every edge the edges check calls sharp (dihedral over 20 deg; `keep` names edges to leave): the rims of the head cut, the notch and the channel, the
    tab, bore and channel edges. The section of the ring and the tab block are already rounded by construction; this is the break on the cut edges, in one
    operation at the largest radius from `radius` down (0.25, 0.2, 0.15 mm) that OCC rolls along all of them at once (it fails on one edge at a larger radius);"""
    sol, es = _sharp_edges(col)
    edges = [e for e, _ in es if not keep(e)]
    if not edges:
        return col
    for r in (radius, radius * 0.8, radius * 0.6):
        try:
            res = sol.fillet(r, edges)
        except ValueError:
            continue
        if res.is_valid and len(res.solids()) == 1:
            return res.wrapped
    return _round_sharp_by_cluster(col, keep, radius)


def _round_sharp_by_cluster(col, keep, radius):
    """The fallback of _round_sharp when OCC cannot roll all the edges at once: one corner cluster (edges within 4 mm of each other) at a time with the largest radius from
    `radius` down to 0.4 x that OCC rolls, a single edge when the cluster fails; an edge no radius rolls is left sharp (the edges check reads it) and printed."""
    left = set()
    key = lambda e: tuple(round(v, 3) for v in (e.center().X, e.center().Y, e.center().Z))
    while True:
        sol, es = _sharp_edges(col)
        es = [e for e, _ in es if not keep(e) and key(e) not in left]
        if not es:
            break
        c0 = es[0].center()
        cl = [e for e in es if ((e.center().X - c0.X) ** 2 + (e.center().Y - c0.Y) ** 2 + (e.center().Z - c0.Z) ** 2) ** 0.5 < 4.0]
        for group in (cl, es[:1]):
            for r in (radius, radius * 0.8, radius * 0.6, radius * 0.4):
                try:
                    t = sol.fillet(r, group)
                except ValueError:
                    continue
                if t.is_valid and len(t.solids()) == 1:
                    col = t.wrapped
                    break
            else:
                continue
            break
        else:
            left.add(key(es[0]))
            print(f"edge left sharp at ({c0.X:.2f}, {c0.Y:.2f}, {c0.Z:.2f}): no radius rolls it", flush=True)
    return col


def _round_plan_cuts(col, keep, radii=(0.25, 0.2, 0.15, 0.1, 0.07, 0.05)):
    """Round the edges the plan cuts of _round_bed_plan left (all below z 398), one corner cluster at a time with the largest radius from `radii` OCC rolls: one
    operation over all of them fails at every radius, each cluster takes 0.1 - 0.25 mm. A cluster no radius rolls is left sharp (printed; the edges check reads it)."""
    left = set()
    key = lambda e: tuple(round(v, 3) for v in (e.center().X, e.center().Y, e.center().Z))
    while True:
        sol, es = _sharp_edges(col)
        es = [e for e, _ in es if e.center().Z < 398.0 and not keep(e) and key(e) not in left]
        if not es:
            return col
        c0 = es[0].center()
        cl = [e for e in es if ((e.center().X - c0.X) ** 2 + (e.center().Y - c0.Y) ** 2) ** 0.5 < 4.0]
        for r in radii:
            try:
                t = sol.fillet(r, cl)
            except ValueError:
                continue
            if t.is_valid and len(t.solids()) == 1:
                col = t.wrapped
                break
        else:
            left.update(key(e) for e in cl)
            print(f"plan edges left sharp: {len(cl)} at ({c0.X:.1f}, {c0.Y:.1f}): no radius rolls them", flush=True)


def _left_alone(e, face_x):
    """Edges the rounding leaves: (1) the sleeve bore's edges, on a mating face of the sleeve (within 0.3 mm of it: the edges check counts them apart; a
    rounded bore edge at the tab face is a recess in the press counter disc: support 98.6 -> 94.7 %); (2) the teardrop roof's apex line (a rounded apex is
    a flat ceiling: overhang beyond 70 deg); (3) the ridge's bevel edges (a planar bevel of 45-60 deg is not sharp for the edges check; the tip is 0.4 mm wide)."""
    y, z = SCREW_YZ[0]
    m = e.position_at(0.5)
    near_sleeve = FLOOR_X - 0.05 <= m.X <= face_x + 0.05 and ((m.Y - y) ** 2 + (m.Z - z) ** 2) ** 0.5 <= SLEEVE_OD / 2 + 0.3
    apex = e.geom_type.name == "LINE" and abs(m.Y - y) < 1e-3 and abs(m.Z - (z + SLEEVE_OD / 2 * 1.41421356)) < 1e-3
    ridge = RIDGE_Z[0] - 0.2 < m.Z < RIDGE_Z[1] + 0.2 and m.X > -26.3 and RIDGE_Y[0] - 0.1 < m.Y < RIDGE_Y[1] + 0.1
    return (near_sleeve and not os.environ.get("NOSLEEVE")) or apex or ridge


def _round_apex(col, radii=(0.2, 0.25, 0.3)):
    """The teardrop roof's apex line (3.5 mm, between the channel floor and the tab face, ends on the unrounded bore edges, so no fillet chain meets it) with the
    smallest radius the edges check reads as not sharp: a 0.2 mm round is a 0.15 mm flat bridge at the roof (about 0.5 mm2 beyond 70 deg)."""
    from build123d import Solid
    sol = Solid(col)
    y, z = SCREW_YZ[0]
    es = [e for e in sol.edges() if e.geom_type.name == "LINE" and abs(e.center().Y - y) < 1e-3 and abs(e.center().Z - (z + SLEEVE_OD / 2 * 1.41421356)) < 1e-3]
    assert len(es) == 1, f"roof apex: {len(es)} edges"
    for r in radii:
        try:
            t = sol.fillet(r, es)
        except ValueError:
            continue
        if t.is_valid and len(t.solids()) == 1:
            return t.wrapped
    raise AssertionError("roof apex rounding failed")


def _round_floor_bore(col, radii=(0.1, 0.15, 0.07)):
    """Round the teardrop opening in the channel floor, head side only (the edges of the planar face at x == FLOOR_X, normal -x, within 1.5 sleeve radii of the
    screw axis; the tab-face press disc stays sharp): the apex round of _round_apex meets the floor in a 0.31 mm edge that the edges check reads as sharp."""
    from build123d import Solid
    sol = Solid(col)
    y, z = SCREW_YZ[0]
    fs = [f for f in sol.faces() if f.geom_type.name == "PLANE" and abs(f.center().X - FLOOR_X) < 1e-3 and f.normal_at().X < -0.99]
    es = [e for f in fs for e in f.edges() if abs(e.center().X - FLOOR_X) < 1e-3 and ((e.center().Y - y) ** 2 + (e.center().Z - z) ** 2) ** 0.5 <= 1.5 * SLEEVE_OD / 2]
    assert es, "floor bore: no edges"
    for r in radii:
        try:
            t = sol.fillet(r, es)
        except ValueError:
            continue
        if t.is_valid and len(t.solids()) == 1:
            return t.wrapped
    raise AssertionError("floor bore rounding failed")


def _tab_box(face_x, x0):
    """The tab's block (plate face back to x0): its two vertical edges at the plate face rounded in plan (bed outline corners), then its face-side bed edges."""
    from build123d import Solid
    import cad_ops as C
    b = Solid(C.box((x0, TAB_Y[0], TAB_Z[0]), (face_x, TAB_Y[1], TAB_Z[1])))
    es = [e for e in b.edges() if abs(e.center().X - face_x) < 1e-6 and abs(e.length - (TAB_Z[1] - TAB_Z[0])) < 1e-6]
    b = b.fillet(TAB_R, es)
    bot = [e for f in b.faces() if abs(f.normal_at().Z + 1) < 1e-6 and abs(f.center().Z - TAB_Z[0]) < 1e-6 for e in f.edges() if e.center().X > x0 + 0.1]
    return b.fillet(BED_ROUND, bot).wrapped


def collar_solid(face_x):
    """The collar B-rep before the final rounding; also returns the ring volume."""
    import cad_ops as C
    ring, (head, notch) = collar_ring()
    x0 = FLOOR_X + 0.35                             # the tab starts inside the wall behind the channel floor: nothing of it shows outside, no face coplanar with the floor; 0.35: its bed face meets the ring foot's flat bed (0.5 left a 0.1 mm crevice and a 10 deg bed corner)
    rib = _ridge(face_x)
    channel = C.box((FLOOR_X - 8.0, CHANNEL_Y[0], 390.0), (FLOOR_X, CHANNEL_Y[1], 420.0))      # flat spot face channel: heads and the press tip
    tab = C.common(_tab_box(face_x, x0), _outer_solid())     # never outside the ring's outer surface
    tab = C.cut(tab, collar_ring_cutters(lower=TAB_LOWER)[0])  # cut TAB_LOWER under the head envelope: its top never meets the rounded rim of the ring
    tab = C.fuse(_round_sharp(tab, TAB_EDGE_R), rib)          # round on its own (a block), then the ridge
    body = C.cut(_trim_thin(C.cut(ring, head), head), notch, channel)       # the notch floor (z 394.8..) is the only link between the ring's two halves
    body = _round_sharp(_round_flank_top(_round_foot_corners(body)))              # the rims of the head cut, notch and channel, before the tab is fused on (they roll further than with the tab)
    col = C.fuse(body, tab)
    v_ring = C.volume(C.fuse(ring, tab))
    tools = []
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
    col = _round_sharp(col, keep=lambda e: _left_alone(e, face_x))
    col = _round_apex(col)
    col = _round_floor_bore(col)
    col = _round_plan_cuts(_round_bed_plan(col), keep=lambda e: _left_alone(e, face_x))    # the plan cuts last (before the tab a cut leaves the tab's rim edges un-rollable)
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
