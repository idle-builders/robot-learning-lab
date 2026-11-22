"""Shoulder shells: upper-arm covers and torso shells for the shoulder U connectors, covers mounted on the green U
(Idle T-tph7xb, parent T-ce88t9; owner M-s9tnc9: shells and cover redesigned once the connectors passed their checks).

Owner 2026-11-22: "The shell should go above the green U and also the shell currently follows the sweep of the motors to
closely" and "Make use of gently and simple parametric functions to define the shape of the shell". So every surface made
here is a few smooth parametric functions (the thigh-shell pattern, robot/body/leg/thigh_shell.py); nothing follows a swept
volume, and the openings are simple shapes about the joint axes:

  upper-arm cover (roll link)  outer skin of horizontal superellipse rings about a straight centre line (cx, cy linear in z):
                               per ring four half-widths (a_in / a_out in x, b_front / b_back in y) and two exponents
                               (front / back half), each ONE natural cubic spline in z through a few control values (COVER),
                               closed at the top by a superellipse dome (CAP); fitted (fit-cover) as the skin closest to
                               today's cover. Where the green U's inboard corners reach through it, two smooth C2 bumps
                               (BUMPS) grow the skin locally so the green U, its sleeves and screws sit inside with GAP + WALL:
                               the cover encloses the green U instead of being cut around it. Constant WALL inward along the
                               surface normal; the bottom keeps today's elbow cut (plane normal to the yaw axis, BOTTOM_S).
  cover top opening (TRIM)     the top end trimmed by two planes along y: one square to the arm (like the elbow cut) just
                               above the roll axis, one on the inboard side whose offset dips smoothly over the middle of the
                               width; their corner rounded. The blue U (stage 1, sleeves, roll horn / idler screws, the roll
                               idler) clears every cut face by BLUE_CLEAR + TRIM_MARGIN over shoulder roll -15..95, the inboard
                               edge stays TRIM_MARGIN above the green U's enclosure.
  torso shells (torso link)    one round opening about the shoulder pitch axis (TORSO_OPENING): a cylinder of radius R_mm
                               (the blue U's largest radius about the pitch axis + BLUE_CLEAR) that narrows inward in a cone
                               below x_mm, where the torso curves away from the blue U; outer edge rounded (round_mm).
  cover mount                  two pads on the green U at the cover's screw wells (positions as today's cover): a boss with
                               a pressed 304 sleeve (4.5 x 3.2, the M3 sleeve of the U) and a captive ISO 4032 M3 nut in a
                               slot (front open to +z, rear to +x); ISO 7380-1 M3 button head from outside through the
                               cover's well (front M3x8, rear M3x6). The wells are rebuilt analytically in the new cover.

Runs after `shoulder_clevis.py manifest`: reads the selected body it wrote (assembly-parts-shoulder-u.json, never written
here), takes the shell meshes from the roll-flip source manifest and the green U from the build output (cad-manifest.json),
and writes its own candidate shells/assembly-parts-shoulder-u-shells.json plus meshes / STEP under
artifacts/robot/body/arm/shoulder-clevis/shells/. Deterministic: a rerun gives the same files.

  .venv/bin/python -B robot/body/arm/shoulder_shells.py fit-cover      # re-fit COVER / CAP (prints the table), TRIM, R_mm
  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 1800 -- .venv/bin/python -B robot/body/arm/shoulder_shells.py apply
  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 900 -- .venv/bin/python -B robot/body/arm/shoulder_shells.py views
Check the candidate with robot/body/checks/run.py --manifest artifacts/robot/body/arm/shoulder-clevis/shells/assembly-parts-shoulder-u-shells.json
Every result is a measurement (mm, mm3); nothing accepts or rejects.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body"))
sys.path.insert(0, str(HERE))
from checks import bodylib as F  # noqa: E402
import shoulder_clevis as SC  # noqa: E402

OUT = SC.OUT / "shells"
PARTS = OUT / "parts"
REPORT = OUT / "report.json"
CANDIDATE = OUT / "assembly-parts-shoulder-u-shells.json"   # own candidate; the selected body (SC.CANDIDATE) is only read
VIEWS = SC.VIEWS
TORSO_SHELLS = ("center_front_shell", "center_back_shell")

# ---------------------------------------------------------------------------------------------- the parameters
# R arm, roll-link neutral frame (= body frame at standing), mm; the left cover is the exact mirror (x -> -x).
WALL = 0.9                 # cover wall along the surface normal (today's cover: 0.84-0.98, median 0.89)
GAP = 1.75                 # cover inner wall to the green U, its sleeves and screws
BLUE_CLEAR = 3.0           # blue U to every cut face, over the joint range
YAW_AXIS = np.array([-0.2236068, -0.1490712, 0.96308682])
BOTTOM_S = 246.75          # today's elbow cut, kept: the cover is the part with p . YAW_AXIS >= BOTTOM_S

# Outer skin: per height z a ring |dx / a|^n + |dy / b|^n = 1 about (cx, cy), a = a_out (dx > 0, outboard) or a_in, b = b_back
# (dy > 0) or b_front, n = n_back (dy > 0) or n_front; with n > 2 the four quarters meet with matching tangent and zero
# curvature. Each value is one natural cubic spline in z through the control values (2 heights = a straight line), constant
# beyond the end heights. fit-cover fits them (cx, cy: least squares through today's section centres).
COVER = {                  # parameter: (heights, control values), mm (exponents unitless)
    "cx": ((282.0, 366.0), (133.360, 114.569)),
    "cy": ((282.0, 366.0), (0.882, 5.886)),
    "a_in": ((282.0, 310.0, 338.0, 366.0), (29.889, 27.264, 26.796, 31.765)),
    "a_out": ((282.0, 310.0, 338.0, 366.0), (25.717, 29.102, 25.715, 18.062)),
    "b_front": ((282.0, 310.0, 338.0, 366.0), (26.933, 31.527, 34.422, 30.793)),
    "b_back": ((282.0, 310.0, 338.0, 366.0), (26.517, 31.700, 33.886, 32.317)),
    "n_front": ((282.0, 324.0, 366.0), (2.003, 2.005, 2.000)),
    "n_back": ((282.0, 324.0, 366.0), (2.099, 2.000, 2.000)),
}
CAP = dict(z0=364.103, z_top=377.0, p=1.855)   # dome above z0: the z0 ring scaled by g = (1 - ((z - z0) / (z_top - z0))^p)^(1/p)
RING = ("a_in", "a_out", "b_front", "b_back", "n_front", "n_back")
FIT_N = (2.0, 4.5)         # exponent range in the fit
CAP_START = (364.0, 377.0, 2.0)   # the fit's start for z0, z_top, p (fixed, so a re-fit is reproducible)
# Green U enclosure (owner 2026-11-22: "the shell should go above the green U"): the base skin above is today's cover; where
# the green U's inboard corners reach through it, smooth bumps are added to the ring radius:
#   dr = A * w((z_ring - z) / w_z) * w(angle(theta - th) / w_th),   w(t) = (1 - t^2)^3 for |t| < 1, else 0
# (value, slope and curvature vanish at the bump's edge: it blends into the skin with continuous curvature). theta / th in deg
# about the ring centre from +x (outboard) toward +y (back). fit-cover fits A, z, th; the half-widths are chosen gentle
# (slope A / w <= about 0.25).
BUMPS = (dict(A=4.688, z=327.88, w_z=24.0, th=-132.022, w_th=45.0),    # front-inboard corner of the green U
         dict(A=6.369, z=326.099, w_z=24.0, th=129.269, w_th=45.0))    # back-inboard corner
BUMP_W = dict(w_z=24.0, w_th=45.0)
GROW_MARGIN = 0.35         # extra radial margin in the fit (radial vs normal distance); the exact distance is reported

# Top opening: two planes along y, angles in deg from +z toward +x (outboard) in the x-z plane, offsets in mm from the
# shoulder roll axis toward the removed side (+ keeps more). The top plane is square to the arm, like the elbow cut (its
# normal is the yaw axis): `top` = the yaw axis' angle + 90, at `top_offset`. The inboard plane runs at `inboard`, at
# `inboard_offset`; over the middle of the width, where the blue U's plate passes the inboard wall lower than at the ends
# (there the green U's arms are enclosed), its offset dips smoothly:
#   offset(y) = inboard_offset - dip * w((y - dip_y) / dip_w),   w as for the bumps
# The cover keeps what lies below both planes; their corner is rounded with round_mm. fit-cover places them: the blue U
# clears every cut face by BLUE_CLEAR + TRIM_MARGIN over the roll range, the inboard edge keeps TRIM_MARGIN to the green U's
# enclosure, and the least cover is removed.
TRIM = dict(inboard=-144.5, inboard_offset=-7.75, dip=2.25, dip_y=9.0, dip_w=16.0, top=76.93, top_offset=4.0, round_mm=10.0)
TOP_OFFSET_LOOK = 4.0      # the top plane at most this far above the roll axis (look: a level rim over the outboard
                           # shoulder, obtuse corners; higher, the inboard cut meets the dome's crest in a peak)
TRIM_MARGIN = 1.0          # the trim edges' margin beyond the blue set (BLUE_CLEAR + 0.3) and beyond the green enclosure
NEAR_AXIS = 14.0           # mm from the roll axis: inside it the blue U (horn / idler screw heads) only turns in place
# Torso opening about the shoulder pitch axis (along x): cylinder radius R_mm for x >= x_mm, narrowing inward in a cone of
# cone_deg (from the axis) below x_mm; the outer edge rounded with a quarter circle of round_mm (along x).
TORSO_OPENING = dict(R_mm=34.63, x_mm=79.62, cone_deg=45.0, round_mm=1.5)


def _side(n, s):
    return n.replace("R_", s, 1) if n.startswith("R_") else n.replace("upper__R_", "upper__" + s, 1)


def blue_group(body, s):
    """The blue U and what moves with it on the pitch link near the roll servo."""
    names = [SC.S1, SC.S1 + "_steel_sleeves", SC.IDLER] + [n for n in body.parts if n.startswith("R_shoulder_u_roll_horn_")
                                                             or n.startswith("R_shoulder_u_roll_idler_")]
    return [_side(n, s) for n in names]


def green_group(body, s):
    names = [SC.S2, SC.S2 + "_steel_sleeves"] + [n for n in body.parts if n.startswith("R_shoulder_u_roll_case_")]
    return [_side(n, s) for n in names]


def _mf(m):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


def _tm(x):
    r = x.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=True)


def _pts(body, names, n=20000, seed=0):
    """Vertices + seeded surface samples of parts (body frame at standing)."""
    out = []
    for nm in names:
        m = body.mesh(nm)
        out.append(np.vstack([m.vertices, trimesh.sample.sample_surface(m, n, seed=seed)[0]]))
    return np.vstack(out)


# ---------------------------------------------------------------------------------------------- parametric cover skin
def _basis(zk, z):
    """B with B @ values = the natural cubic spline through (zk, values) at z (2 heights: straight line, 1: constant)."""
    from scipy.interpolate import CubicSpline
    zk = np.asarray(zk, float); z = np.clip(np.asarray(z, float), zk[0], zk[-1]); k = len(zk)
    if k == 1:
        return np.ones((len(z), 1))
    if k == 2:
        t = (z - zk[0]) / (zk[1] - zk[0])
        return np.c_[1 - t, t]
    return np.stack([CubicSpline(zk, np.eye(k)[i], bc_type="natural")(z) for i in range(k)], -1)


def ring(z, cov=None, cap=None):
    """Centre (cx, cy), ring parameters (columns RING) and dome scale g at heights z."""
    cov = cov or COVER; cap = cap or CAP
    z = np.atleast_1d(np.asarray(z, float)); zr = np.minimum(z, cap["z0"])
    ev = {k: _basis(cov[k][0], zr) @ np.asarray(cov[k][1], float) for k in cov}
    h = np.clip((z - cap["z0"]) / (cap["z_top"] - cap["z0"]), 0.0, 1.0)
    g = np.clip(1.0 - h ** cap["p"], 0.0, 1.0) ** (1.0 / cap["p"])
    return ev["cx"], ev["cy"], np.stack([ev[k] for k in RING], -1), g


def _w(t):
    return np.where(np.abs(t) < 1.0, (1.0 - np.minimum(t * t, 1.0)) ** 3, 0.0)


def bump(z, th, bumps=None):
    """Sum of the enclosure bumps (mm) at heights z and angles th (rad), broadcast."""
    out = 0.0
    for b in (BUMPS if bumps is None else bumps):
        d = np.degrees(np.angle(np.exp(1j * (th - np.radians(b["th"])))))
        out = out + b["A"] * _w((z - b["z"]) / b["w_z"]) * _w(d / b["w_th"])
    return out


def skin_radius(z, th, cov=None, cap=None, bumps=None):
    """Outer skin radius about the ring centre at height z, angle th (rad); z and th broadcast."""
    z, th = np.broadcast_arrays(np.asarray(z, float), np.asarray(th, float))
    _, _, prm, g = ring(z.ravel(), cov, cap)
    r = (radius(th.ravel(), prm, 1.0) + bump(np.minimum(z.ravel(), (cap or CAP)["z0"]), th.ravel(), bumps)) * g
    return r.reshape(z.shape)


def radius(th, prm, g):
    """Outer radius about the ring centre at angle th (from +x toward +y); prm (..., 6) broadcast against th."""
    a_in, a_out, bf, bb, nf, nb = (prm[..., i] for i in range(6))
    c, s = np.cos(th), np.sin(th)
    a = np.where(c > 0, a_out, a_in); b = np.where(s > 0, bb, bf); e = np.where(s > 0, nb, nf)
    return g * (np.abs(c / a) ** e + np.abs(s / b) ** e) ** (-1.0 / e)


def _grid(cov, cap, z_lo, nth=240, dz=0.75, ncap=60, bumps=None):
    """Outer skin as a grid of rings (nz, nth, 3) from z_lo up to the dome, the dome pole, and the outward normals."""
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    zs = np.arange(z_lo, cap["z0"], dz)
    phi = np.linspace(0.0, np.pi / 2, ncap)[:-1]                 # dome rows; the last one (phi = pi / 2) is the pole
    zc = cap["z0"] + (cap["z_top"] - cap["z0"]) * np.sin(phi) ** (2.0 / cap["p"])
    z = np.concatenate([zs, zc])
    cx, cy, prm, g = ring(z, cov, cap)
    g[len(zs):] = np.cos(phi) ** (2.0 / cap["p"])
    R = (radius(th[None, :], prm[:, None, :], 1.0) + bump(np.minimum(z, cap["z0"])[:, None], th[None, :], bumps)) * g[:, None]
    V = np.stack([cx[:, None] + R * np.cos(th), cy[:, None] + R * np.sin(th), np.broadcast_to(z[:, None], R.shape)], -1)
    pole = np.array([cx[-1], cy[-1], cap["z_top"]])
    dth = np.roll(V, -1, 1) - np.roll(V, 1, 1)
    Vz = np.concatenate([V[1:], np.broadcast_to(pole, (1,) + V.shape[1:])], 0)
    Vm = np.concatenate([V[:1] - (V[1:2] - V[:1]), V[:-1]], 0)
    N = np.cross(dth, Vz - Vm)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return V, pole, N


def _solid(V, pole, bottom):
    """Closed mesh of a ring grid with a pole on top and a fan to `bottom` below."""
    nz, nth = V.shape[:2]
    i, j = np.meshgrid(np.arange(nz - 1), np.arange(nth), indexing="ij")
    a, b = i * nth + j, i * nth + (j + 1) % nth
    F = np.r_[np.c_[a.ravel(), b.ravel(), (b + nth).ravel()], np.c_[a.ravel(), (b + nth).ravel(), (a + nth).ravel()]]
    P, Bc = nz * nth, nz * nth + 1
    j = np.arange(nth); top = (nz - 1) * nth
    F = np.r_[F, np.c_[top + j, top + (j + 1) % nth, np.full(nth, P)], np.c_[(j + 1) % nth, j, np.full(nth, Bc)]]
    m = trimesh.Trimesh(np.vstack([V.reshape(-1, 3), pole, bottom]), F, process=True)
    if m.volume < 0:
        m.invert()
    return m


def cover_outer(cov=None, cap=None, z_lo=255.0):
    cov = cov or COVER; cap = cap or CAP
    V, pole, _ = _grid(cov, cap, z_lo)
    return _solid(V, pole, V[0].mean(0))


def cover_skin(cov=None, cap=None):
    """Closed cover shell: outer skin minus the skin offset inward by WALL (open at the bottom), above the elbow cut."""
    cov = cov or COVER; cap = cap or CAP
    outer = cover_outer(cov, cap)
    V, pole, N = _grid(cov, cap, 245.0)
    inner = _solid(V - WALL * N, pole - np.array([0.0, 0.0, WALL]), V[0].mean(0) - np.array([0.0, 0.0, 5.0]))
    sh = (_mf(outer) - _mf(inner)).trim_by_plane(YAW_AXIS.tolist(), float(BOTTOM_S))
    return sh, outer


# ---------------------------------------------------------------------------------------------- wells, top trim, torso opening
WELL_R = (3.25, 4.75)      # today's cover wells: recess radius (button head 5.7 dia), tube outer radius
WELL_HOLE_R = 1.7          # M3 clearance through the pad


def _cyl_y(x, z, y0, y1, r, n=72):
    return _mf(trimesh.creation.cylinder(radius=r, segment=[[x, y0, z], [x, y1, z]], sections=n))


def add_wells(shell, outer, design):
    """Screw wells at the mount axes (R): a tube r 4.75 from the pad face out to the skin, recess r 3.25 from the seat
    outward, M3 clearance through the pad (pad thickness = |seat - pad face|, as today's cover)."""
    o = _mf(outer)
    for d in design.values():
        (x, z), h = d["axis_xz"], d["head_side"]
        pad, seat = d["pad_face_y"], d["seat_y"]
        shell = shell + (_cyl_y(x, z, pad, pad + h * 40.0, WELL_R[1]) ^ o)
        shell = shell - _cyl_y(x, z, seat, seat + h * 40.0, WELL_R[0]) - _cyl_y(x, z, pad - h * 2.0, seat + h * 40.0, WELL_HOLE_R)
    return shell


def _dir(psi):
    p = np.radians(psi)
    return np.stack([np.sin(p), np.cos(p)], -1)                 # (x, z): psi from +z toward +x


def _trim_frame(y, t):
    """Per y: unit normals n_a, n_b (toward the removed side), plane offsets from the roll axis, and the rounding centre O
    relative to the roll axis (the apex of both planes moved in by round_mm)."""
    y = np.atleast_1d(np.asarray(y, float))
    na, nb = _dir(t["inboard"] + 90.0), _dir(t["top"] - 90.0)
    da = t["inboard_offset"] - t["dip"] * _w((y - t["dip_y"]) / t["dip_w"])
    r = t["round_mm"]
    M = np.array([na, nb])
    O = np.linalg.solve(M, np.stack([da - r, np.full_like(da, t["top_offset"] - r)]))      # (2, len(y))
    return na, nb, da, O.T


def _edge_dirs(na, nb):
    """Directions along each plane away from the corner, into the kept side of the other plane."""
    ta, tb = np.array([-na[1], na[0]]), np.array([-nb[1], nb[0]])
    return (ta if ta @ nb < 0 else -ta), (tb if tb @ na < 0 else -tb)


def trim_kept(P, axis_xz, trim=None):
    """True for points P (x, y, z) the top trim keeps: within round_mm of the two planes' kept wedge moved in by round_mm
    (= the wedge with its corner rounded)."""
    t = trim or TRIM
    na, nb, da, O = _trim_frame(P[:, 1], t)
    ta, tb = _edge_dirs(na, nb)
    q = P[:, [0, 2]] - np.asarray(axis_xz, float) - O
    sa, sb = q @ na, q @ nb
    apex = (q @ ta <= 0) & (q @ tb <= 0)
    return np.where(apex, np.hypot(q[:, 0], q[:, 1]) <= t["round_mm"], (sa <= t["round_mm"]) & (sb <= t["round_mm"]))


def trim_removed(P, axis_xz, trim=None):
    return ~trim_kept(P, axis_xz, trim)


def _ang(v):
    return np.degrees(np.arctan2(v[..., 0], v[..., 1]))


def _arc(a0, a1, through):
    """Angles (deg) from a0 to a1 the way that passes `through`."""
    up = (a1 - a0) % 360.0
    return (a0, a0 + up) if (through - a0) % 360.0 <= up else (a0, a0 - (360.0 - up))


def _trim_slice(c, y, t, L=150.0, n_ray=24, n_out=72, n_arc=24):
    """Boundary (x, z) of the removed region in one y slice: out along the top plane, the outer arc over the removed side,
    in along the inboard plane, the rounded corner (fixed point counts, so slices loft)."""
    na, nb, da, O = _trim_frame(y, t)
    O = c + O[0]; r = t["round_mm"]
    ta, tb = _edge_dirs(na, nb)
    TA, TB = O + r * na, O + r * nb
    s = np.linspace(0.0, L, n_ray)[:, None]
    a0, a1 = _arc(_ang(TB + L * tb - O), _ang(TA + L * ta - O), _ang(nb))
    big = O + np.hypot(L, r) * _dir(np.linspace(a0, a1, n_out + 2)[1:-1])
    b0, b1 = _arc(_ang(na), _ang(nb), _ang(na + nb))
    corner = O + r * _dir(np.linspace(b0, b1, n_arc + 2)[1:-1])
    return np.vstack([TB + s * tb, big, TA + s[::-1] * ta, corner])


def trim_cutter(axis_xz, trim=None, y_lo=-70.0, y_hi=70.0, dy=0.5):
    """The removed region lofted over y (the inboard plane's offset follows the dip)."""
    import mapbox_earcut as ec
    t = trim or TRIM
    c = np.asarray(axis_xz, float)
    ys = np.arange(y_lo, y_hi + 1e-6, dy)
    rings = [_trim_slice(c, y, t) for y in ys]
    V = np.stack([np.c_[r[:, 0], np.full(len(r), y), r[:, 1]] for r, y in zip(rings, ys)])
    nz, n = V.shape[:2]
    i, j = np.meshgrid(np.arange(nz - 1), np.arange(n), indexing="ij")
    a, b = i * n + j, i * n + (j + 1) % n
    Fs = [np.c_[a.ravel(), b.ravel(), (b + n).ravel()], np.c_[a.ravel(), (b + n).ravel(), (a + n).ravel()]]
    for k, flip in ((0, True), (nz - 1, False)):
        tri = ec.triangulate_float64(rings[k], np.array([n], np.uint32)).reshape(-1, 3).astype(np.int64) + k * n
        Fs.append(tri[:, ::-1] if flip else tri)
    m = trimesh.Trimesh(V.reshape(-1, 3), np.vstack(Fs), process=True)
    if m.volume < 0:
        m.invert()
    return m


def torso_cutter(shells, axis_yz, op=None, sx=1.0, nphi=180, nx=240):
    """Opening cutter about the pitch axis (R side, x > 0; sx = -1 mirrors it): per angle phi a radius profile over x,
    cylinder R above x_mm, cone below, plus a quarter-circle flare of round_mm ending at the shell's outer surface (found by
    rays along the profile), so the outer edge is rounded."""
    op = op or TORSO_OPENING
    R, x1, rf = op["R_mm"], op["x_mm"], op["round_mm"]
    tb = np.tan(np.radians(op["cone_deg"]))
    xs = np.linspace(40.0, 112.0, nx); phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    base = np.where(xs >= x1, R, np.maximum(R - (x1 - xs) / tb, 4.0))
    cat = trimesh.util.concatenate(shells)
    rx = trimesh.ray.ray_pyembree.RayMeshIntersector(cat)
    xo = np.full(nphi, np.nan)
    for k, f in enumerate(phi):                                   # walk the profile from outside: first hit = outer surface
        d = np.array([0.0, np.cos(f), np.sin(f)])
        P = np.c_[xs[::-1], axis_yz[0] + base[::-1, None] * d[1], axis_yz[1] + base[::-1, None] * d[2]]
        P[:, 0] *= sx
        seg = P[1:] - P[:-1]; L = np.linalg.norm(seg, axis=1)
        loc, ir, _ = rx.intersects_location(P[:-1], seg / L[:, None], multiple_hits=False)
        ok = np.linalg.norm(loc - P[:-1][ir], axis=1) <= L[ir] + 1e-9
        if ok.any():
            xo[k] = abs(loc[ok][np.argmin(ir[ok])][0])
    good = np.isfinite(xo)
    xo = np.interp(phi, phi[good], xo[good], period=2 * np.pi) if good.any() else np.full(nphi, x1)
    u = xs[None, :] - (xo[:, None] - rf)
    fl = np.where(u <= 0, 0.0, np.where(u < rf, rf - np.sqrt(np.clip(rf ** 2 - u ** 2, 0, None)), rf + 4.0 * (u - rf)))
    rho = base[None, :] + fl
    V = np.stack([np.broadcast_to(xs, rho.shape) * sx, axis_yz[0] + rho * np.cos(phi)[:, None],
                  axis_yz[1] + rho * np.sin(phi)[:, None]], -1).transpose(1, 0, 2)        # (nx, nphi, 3): rings along x
    m = _solid(V, np.array([xs[-1] * sx, *axis_yz]), np.array([xs[0] * sx, *axis_yz]))
    return m, dict(outer_x_at_edge_mm=[round(float(xo.min()), 2), round(float(xo.max()), 2)])


# ---------------------------------------------------------------------------------------------- fitting (fit-cover)
def _roll_axis_xz(body, s="R_"):
    o = body.joints[f"{s}shoulder_roll"]["origin_world_mm"]
    return np.array([o[0], o[2]], float)


def _pitch_axis_yz(body, s="R_"):
    o = body.joints[f"{s}shoulder_pitch"]["origin_world_mm"]
    return np.array([o[1], o[2]], float)


def today_samples(c, design, n=90000):
    """Points on today's cover outer skin with their normals (seeded), without the screw wells and rims."""
    P, fi = trimesh.sample.sample_surface(c, n, seed=1)
    N = c.face_normals[fi]
    cx, cy, _, _ = ring(P[:, 2])
    C3 = np.c_[cx, cy, np.minimum(P[:, 2], 345.0)]
    out = np.einsum("ij,ij->i", N, P - C3) > 0
    rx = trimesh.ray.ray_pyembree.RayMeshIntersector(c)
    loc, ir, _ = rx.intersects_location(P - N * 0.02, -N, multiple_hits=False)
    t = np.full(len(P), np.inf); t[ir] = np.linalg.norm(loc - P[ir], axis=1)
    ok = out & (t > 0.4) & (t < 1.6)
    for d in design.values():
        (x, z), h = d["axis_xz"], d["head_side"]
        ok &= ~((np.hypot(P[:, 0] - x, P[:, 2] - z) < 8.0) & (P[:, 1] * h > abs(d["pad_face_y"]) - 4.0))
    return P[ok], N[ok]


def requirement(body, design):
    """Points the cover must hold with GAP + WALL: the green U (its bosses end at the wells' pads, left out), its sleeves,
    the roll-case screws."""
    Q = _pts(body, green_group(body, "R_"), n=30000)
    keep = np.ones(len(Q), bool)
    for d in design.values():
        (x, z), h = d["axis_xz"], d["head_side"]
        yb = d["boss_back_y"]
        keep &= ~((np.hypot(Q[:, 0] - x, Q[:, 2] - z) < 6.5) & ((Q[:, 1] - yb) * h > -0.5))
    return Q[keep]


def _polar(P):
    cx, cy, prm, g = ring(P[:, 2])
    dx, dy = P[:, 0] - cx, P[:, 1] - cy
    return np.arctan2(dy, dx), np.hypot(dx, dy), np.c_[dx, dy] / np.maximum(np.hypot(dx, dy), 1e-9)[:, None]


def skin_deviation(P, N, cov=None, cap=None, bumps=None):
    """Signed distance (mm, + = the new skin lies outside) of the new skin from today's skin points, along today's normal
    (radial difference x the radial share of the normal)."""
    th, rho, u = _polar(P)
    cosb = np.abs(np.einsum("ij,ij->i", N[:, :2], u))
    return (skin_radius(P[:, 2], th, cov, cap, bumps) - rho) * np.maximum(cosb, 0.05)


def fit_cover(body, src, design, verbose=True):
    """Two steps. (1) base skin: COVER (the four half-widths, two exponents) and CAP closest to today's cover (normal
    distance, least squares); cx, cy: straight lines through today's section centres. (2) enclosure: BUMPS (A, z, th each;
    half-widths BUMP_W) of the least squared height that hold the requirement (green U + GAP + WALL + GROW_MARGIN, radial).
    Returns (cov, cap, bumps, stats)."""
    from scipy.optimize import minimize
    c = src.mesh("R_upper_arm_hard_cover")
    zs, C = np.arange(288.0, 362.1, 2.0), []
    for z in zs:
        sec = c.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        lp = max(sec.discrete, key=lambda e: np.ptp(e[:, 0]) * np.ptp(e[:, 1]))
        C.append(0.5 * (lp[:, :2].min(0) + lp[:, :2].max(0)))
    C = np.array(C); zk = COVER["cx"][0]
    lin = [np.polyval(np.polyfit(zs, C[:, i], 1), zk) for i in (0, 1)]
    cov = {k: (v[0], tuple(v[1])) for k, v in COVER.items()}
    cov["cx"], cov["cy"] = (zk, tuple(np.round(lin[0], 3))), (zk, tuple(np.round(lin[1], 3)))
    COVER.update(cx=cov["cx"], cy=cov["cy"])           # the polar frame of the samples
    P, N = today_samples(c, design)
    thp, rhop, up = _polar(P)
    cosb = np.maximum(np.abs(np.einsum("ij,ij->i", N[:, :2], up)), 0.05)
    sizes = np.cumsum([0] + [len(cov[k][1]) for k in RING])

    def unpack(v):
        cv = dict(cov); cv.update({k: (cov[k][0], tuple(v[sizes[i]:sizes[i + 1]])) for i, k in enumerate(RING)})
        return cv, dict(z0=v[-3], z_top=v[-2], p=v[-1])

    def obj(v):
        cv, cp = unpack(v)
        return float(np.mean(((skin_radius(P[:, 2], thp, cv, cp, ()) - rhop) * cosb) ** 2))

    v0 = []                                             # start: today's radial extents at the knots
    for k in RING:
        for z in cov[k][0]:
            if k.startswith("n_"):
                v0.append(2.2); continue
            sel = np.abs(P[:, 2] - z) < 3.0
            ang = {"a_out": 0.0, "a_in": np.pi, "b_back": np.pi / 2, "b_front": -np.pi / 2}[k]
            w = sel & (np.abs(np.angle(np.exp(1j * (thp - ang)))) < 0.25)
            v0.append(float(np.median(rhop[w])) if w.any() else 25.0)
    v0 += list(CAP_START)
    bounds = [((FIT_N if k.startswith("n_") else (5.0, 80.0))) for k in RING for _ in cov[k][0]]
    bounds += [(330.0, 372.0), (373.0, 377.0), (1.6, 6.0)]
    r = minimize(obj, np.array(v0, float), method="SLSQP", bounds=bounds, options=dict(maxiter=500, ftol=1e-10),
                 constraints=[dict(type="ineq", fun=lambda v: v[-2] - v[-3] - 4.0)])
    cv, cp = unpack(r.x)
    cv = {k: (tuple(float(z) for z in cv[k][0]), tuple(round(float(x), 3) for x in cv[k][1])) for k in cv}
    cp = {k: round(float(x), 3) for k, x in cp.items()}
    if verbose:
        print(f"base skin: {r.message}; rms normal deviation from today {np.sqrt(r.fun):.3f} mm", flush=True)
    # (2) bumps
    Q = requirement(body, design)
    thq, rhoq, _ = _polar(Q)
    need = rhoq + GAP + WALL + GROW_MARGIN - skin_radius(Q[:, 2], thq, cv, cp, ())
    hit = need > -6.0
    Qz, Qt, Qn = Q[hit, 2], thq[hit], need[hit]
    seeds = [b for b in BUMPS]

    def bset(v):
        return tuple(dict(A=v[3 * i], z=v[3 * i + 1], th=v[3 * i + 2], **BUMP_W) for i in range(len(seeds)))

    def margin(v):
        return bump(Qz, Qt, bset(v)) - Qn

    def bobj(v):
        return float(sum(v[3 * i] ** 2 for i in range(len(seeds))))
    for i, b in enumerate(seeds):                        # seed each bump on the deepest point of its side
        side = (Qt < 0) if b["th"] < 0 else (Qt >= 0)
        j = np.argmax(np.where(side, Qn, -np.inf))
        seeds[i] = dict(b, A=max(float(Qn[j]), 0.5), z=float(Qz[j]), th=float(np.degrees(Qt[j])))
    vb = np.ravel([[b["A"], b["z"], b["th"]] for b in seeds])
    rb = minimize(bobj, vb, method="SLSQP", options=dict(maxiter=300, ftol=1e-9),
                  constraints=[dict(type="ineq", fun=margin)],
                  bounds=[(0.0, 15.0), (300.0, 350.0), (-180.0, 180.0)] * len(seeds))
    bumps = tuple({k: round(float(x), 3) if k in ("A", "z", "th") else x for k, x in b.items()} for b in bset(rb.x))
    mb = margin(np.ravel([[b["A"], b["z"], b["th"]] for b in bumps]))
    if verbose:
        print(f"bumps: {rb.message}; {bumps}; worst requirement margin {mb.min():.3f} mm ({int(hit.sum())} points near)", flush=True)
    dev = skin_deviation(P, N, cv, cp, bumps)
    return cv, cp, bumps, dict(rms_normal_deviation_base_mm=round(float(np.sqrt(r.fun)), 3),
                               rms_normal_deviation_with_bumps_mm=round(float(np.sqrt(np.mean(dev ** 2))), 3),
                               worst_requirement_margin_mm=round(float(mb.min()), 3), samples=int(len(P)),
                               requirement_points=int(len(Q)), deepest_need_mm=round(float(need.max()), 2))


def blue_roll_sweep(body, step=1.0, n=25000):
    """The blue group at every shoulder roll angle (step deg) in the roll-link neutral frame, as one point cloud."""
    B = _pts(body, blue_group(body, "R_"), n=n)
    lo, hi = body.joints["R_shoulder_roll"]["range_deg"]
    out = [trimesh.transform_points(B, np.linalg.inv(body.link_T("R_shoulder_roll_link", {"R_shoulder_roll": float(q)})))
           for q in np.arange(lo, hi + 1e-6, step)]
    return np.vstack(out)


def _psi(P, c):
    return np.degrees(np.arctan2(P[:, 0] - c[0], P[:, 2] - c[1]))


def fit_trim(body, skin_mesh, green_pts, sweep):
    """Place the trim: every skin point within BLUE_CLEAR (+0.3 sampling) of the swept blue U and beyond NEAR_AXIS of the
    roll axis removed (`rm`), the skin within GAP + WALL + 1 of the green U kept (`keep`, its enclosure). The top plane is square
    to the arm; (1) the inboard plane's direction and offset, with the top plane as high as the rm points it leaves allow,
    that remove the least cover with TRIM_MARGIN to rm (where the green U's arms are: keep within 3 mm in y) and to keep;
    (2) the smallest dip that removes the rest of rm with the same margin, without lowering the top plane; (3) the top plane
    no higher than TOP_OFFSET_LOOK."""
    from scipy.spatial import cKDTree
    c = _roll_axis_xz(body)
    S = np.vstack([skin_mesh.vertices, trimesh.sample.sample_surface(skin_mesh, 300000, seed=2)[0]])
    dB = cKDTree(sweep).query(S, distance_upper_bound=BLUE_CLEAR + 1.0)[0]
    dG = cKDTree(green_pts).query(S, distance_upper_bound=GAP + WALL + 2.0)[0]
    rho = np.hypot(S[:, 0] - c[0], S[:, 2] - c[1])
    # near the roll axis the side walls face the roll horn / idler screw heads, which turn in place: no trim plane reaches
    # them (apply reports their clearance), so the planes are placed against the blue set beyond NEAR_AXIS
    rm, keep = S[(dB < BLUE_CLEAR + 0.3) & (rho >= NEAR_AXIS)], S[dG < GAP + WALL + 1.0]
    top = round(float(np.degrees(np.arctan2(YAW_AXIS[0], YAW_AXIS[2]))) + 90.0, 2)     # square to the arm
    nb = _dir(top - 90.0)
    qr, qk = rm[:, [0, 2]] - c, keep[:, [0, 2]] - c
    sB, kB = qr @ nb, qk @ nb
    ends = np.zeros(len(rm), bool)
    for yk in np.unique(np.round(keep[:, 1])):
        ends |= np.abs(rm[:, 1] - yk) < 3.0
    Ssub = S[np.random.default_rng(5).choice(len(S), 30000, replace=False)]
    qs = Ssub[:, [0, 2]] - c
    ssB = qs @ nb

    def top_off(sr, off, sel):
        """The top plane's offset: as high as the rm points the inboard plane leaves (within TRIM_MARGIN) allow."""
        need = sel & (sr - off < TRIM_MARGIN)
        return float(np.min(sB[need]) - TRIM_MARGIN) if need.any() else 40.0
    # (1) inboard direction and offset (+ the top offset they leave): margins >= TRIM_MARGIN where the green U's arms are,
    #     least cover removed
    best = None
    for psi in np.arange(-178.0, -100.0, 0.5):
        na = _dir(psi + 90.0)
        sr, sk, ss = qr @ na, qk @ na, qs @ na
        for off in np.arange(-24.0, 16.01, 0.25):
            if float(np.min(off - sk)) < TRIM_MARGIN:
                continue
            db = top_off(sr, off, ends)
            if float(np.max(kB)) > db - TRIM_MARGIN:
                continue
            gone = int(np.count_nonzero((ss > off) | (ssB > db)))
            if best is None or gone < best[0]:
                best = (gone, float(psi), float(off), db)
    if best is None:
        raise SystemExit(f"fit_trim: no inboard plane keeps {TRIM_MARGIN} mm to both sets")
    _, psi, off, db = best
    na = _dir(psi + 90.0); sr, sk = qr @ na, qk @ na
    # (2) the dip: the smallest that removes the rest of rm with the same margin (the top plane as high as it then allows)
    dip_best = None
    for w in (16.0, 20.0, 24.0, 28.0, 32.0):
        for yc in np.arange(-8.0, 14.1, 1.0):
            for dip in np.arange(0.0, 14.01, 0.25):
                dr = off - dip * _w((rm[:, 1] - yc) / w); dk = off - dip * _w((keep[:, 1] - yc) / w)
                need = sr - dr < TRIM_MARGIN
                db2 = min(db, float(np.min(sB[need]) - TRIM_MARGIN)) if need.any() else db
                mr = float(np.min(np.maximum(sr - dr, sB - db2))); mk = float(min(np.min(dk - sk), db2 - np.max(kB)))
                if min(mr, mk) >= TRIM_MARGIN - 1e-9 and db2 >= db - 0.5 and (dip_best is None or dip * w < dip_best[0]):
                    dip_best = (dip * w, dict(dip=float(dip), dip_y=float(yc), dip_w=w, top_offset=round(db2, 2)), mr, mk)
                    break
    if dip_best is None:                               # no dip keeps the margin: the best compromise
        raise SystemExit(f"fit_trim: no dip keeps the {TRIM_MARGIN} mm margin")
    _, dp, mr, mk = dip_best
    # (3) the look: the top plane no higher than TOP_OFFSET_LOOK (clearance alone would leave a peak where the inboard cut
    #     meets the dome's crest)
    dp["top_offset"] = round(min(dp["top_offset"], TOP_OFFSET_LOOK), 2)
    trim = dict(inboard=psi, inboard_offset=off, dip=dp["dip"], dip_y=dp["dip_y"], dip_w=dp["dip_w"], top=top,
                top_offset=dp["top_offset"], round_mm=TRIM["round_mm"])
    kept = trim_kept(S, c, trim)
    sel_rm = (dB < BLUE_CLEAR + 0.3) & (rho >= NEAR_AXIS)
    return trim, dict(rm_points=int(len(rm)), keep_points=int(len(keep)),
                      margin_to_blue_set_mm=round(mr, 2), margin_to_green_enclosure_mm=round(mk, 2),
                      rm_left_by_trim=int(kept[sel_rm].sum()), keep_cut_by_trim=int((~kept[dG < GAP + WALL + 1.0]).sum()),
                      near_axis_kept_blue_min_distance_mm=round(float(np.min(dB[(rho < NEAR_AXIS) & kept], initial=99.0)), 2))


def torso_radius(body):
    """R_mm: largest distance of the blue U from the pitch axis where it is within BLUE_CLEAR of the shells' x, + BLUE_CLEAR;
    x_mm: the blue U's smallest x - BLUE_CLEAR (below it the cone narrows the cut)."""
    B = _pts(body, blue_group(body, "R_"), n=25000)
    yz = _pitch_axis_yz(body)
    xmax = max(body.mesh(n).bounds[1][0] for n in TORSO_SHELLS)
    near = B[:, 0] < xmax + BLUE_CLEAR
    r = np.hypot(B[near, 1] - yz[0], B[near, 2] - yz[1])
    return dict(R_mm=round(float(r.max() + BLUE_CLEAR), 2), x_mm=round(float(B[:, 0].min() - BLUE_CLEAR), 2),
                cone_deg=TORSO_OPENING["cone_deg"], round_mm=TORSO_OPENING["round_mm"])


def fit_all():
    """fit-cover: prints COVER, CAP, TRIM and TORSO_OPENING to paste above (the build uses the values in the script)."""
    t0 = time.time()
    man = json.loads(SC.CANDIDATE.read_text()); body = F.Body(man)
    src = F.Body(json.loads(SC.MANIFEST.read_text()))
    design = mount_design(src.mesh("R_upper_arm_hard_cover"))
    cov, cap, bumps, st = fit_cover(body, src, design)
    print(json.dumps(st), f"({time.time() - t0:.0f} s)", flush=True)
    COVER.update(cov); CAP.update(cap)
    global BUMPS
    BUMPS = bumps
    sh, outer = cover_skin()
    sh = add_wells(sh, outer, design)
    sweep = blue_roll_sweep(body)
    trim, tst = fit_trim(body, _tm(sh), requirement(body, design), sweep)
    print(json.dumps(tst), f"({time.time() - t0:.0f} s)", flush=True)
    top = torso_radius(body)
    print("COVER = {")
    for k in ("cx", "cy") + RING:
        print(f'    "{k}": ({tuple(cov[k][0])}, ({", ".join(f"{x:.3f}" for x in cov[k][1])},)),')
    print("}")
    print(f"CAP = dict(z0={cap['z0']}, z_top={cap['z_top']}, p={cap['p']})")
    print("BUMPS = (" + ", ".join(f"dict(A={b['A']}, z={b['z']}, w_z={b['w_z']}, th={b['th']}, w_th={b['w_th']})" for b in bumps) + ")")
    print("TRIM = dict(" + ", ".join(f"{k}={v}" for k, v in trim.items()) + ")")
    print(f"TORSO_OPENING = dict(R_mm={top['R_mm']}, x_mm={top['x_mm']}, cone_deg={top['cone_deg']}, round_mm={top['round_mm']})")


# ---------------------------------------------------------------------------------------------- cover mount (B-rep)
# The cover keeps its two screw wells (inward tubes r 3.25..4.75 ending in a flat pad with an M3 clearance hole, made
# for the old stage-2 posts); a boss on the green U meets each pad. R frame (body mm); L = mirror x, as the build.
#   axis: (x, z) of the well (along y); head: side of the screw head (-1 face / -y, +1 back / +y)
#   sleeve: pressed 304 sleeve length from the pad face (the clamp: head - cover pad - sleeve - nut)
#   back: boss end away from the pad; chamfer: 45 deg underside where the boss overhangs the arm (print up ~ +z)
#   bore_slot: the screw bore behind the nut runs out of the arm's top edge (front: a 0.004 mm skin), so it is
#   opened to +z like the nut slot
#   slot: nut insertion direction in (x, z), default +z; rear tilted 15 deg toward +x: straight up the slot passes 0.25 mm
#   from the +y arm's M2 sleeve bore, straight along +x it cuts the yaw-screw counterbore (sleeve support 82 -> 60 %)
#   pocket_extra: pocket depth past the nut (default 0.2); rear 0.6 so the pocket opens cleanly into the yaw-screw
#   driver channel (r 2.25 at x ~133.4) instead of leaving a 0.08 mm membrane against it
NUT = dict(s=5.5, m=2.4, clear=0.15, depth_clear=0.2)    # ISO 4032 M3; pocket: flats along the slot walls
NUT_HW = NUT["s"] / 2 + NUT["clear"]
MOUNTS = {
    "front": dict(axis=(116.5, 336.5), head=-1, sleeve=2.5, screw=8, back=-16.9, chamfer=True, bore_slot=True),
    "rear": dict(axis=(130.5, 329.8), head=+1, sleeve=1.5, screw=6, back=25.3, chamfer=False, bore_slot=False, pocket_extra=0.6,
                 slot=(0.258819, 0.965926)),
}
BOSS_R = 4.5                  # boss radius = the well's flat pad (r 4.75) less 0.25
PAD_GAP = 0.05                # boss face to pad face
PAD_KEEP_R = 5.5              # no static clearance cut of the cover inside this radius of a mount axis (the pads)
SCREW_BORE_R = 1.7            # M3 clearance behind the nut
SLOT_TOP = 2.0                # boss top over the nut slot (above the axis): slot walls >= 1.15 mm wide (0.62 mm line)
HEAD = "ISO 7380-1"           # button head as the old cover screws, seated in the well


def pad_faces(cover, x, z, head):
    """Pad inner face (toward the green U) and screw seat (outer face of the pad, in the well) along the well axis,
    from rays parallel to the axis at r 2.6 (seat ring) and r 4.2 (pad ring)."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    rx = RayMeshIntersector(cover)
    out = {}
    for key, r in (("seat", 2.6), ("pad", 4.2)):
        ys = []
        for a in np.radians(np.arange(0, 360, 30)):
            o = np.array([[x + r * np.cos(a), float(cover.bounds[:, 1].mean()), z + r * np.sin(a)]])   # from inside the cover
            loc, _, _ = rx.intersects_location(o, np.array([[0.0, float(head), 0.0]]), multiple_hits=True)
            h = np.sort(loc[:, 1] * head)
            ys.append(h[:2] * head)
        ys = np.array(ys)
        out[key] = ys
    pad = float(np.median(out["pad"][:, 0]))          # first wall met going outward = the pad's inner face
    seat = float(np.median(out["seat"][:, 1]))        # second hit at the seat ring = the seat (pad outer face)
    return pad, seat, dict(pad_spread=float(np.ptp(out["pad"][:, 0])), seat_spread=float(np.ptp(out["seat"][:, 1])))


def _ccw(pts):
    """Counter-clockwise order (build123d extrudes along the face normal, which flips for a clockwise outline)."""
    a = np.asarray(pts, float)
    area = 0.5 * float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]))
    return [tuple(map(float, q)) for q in (a if area > 0 else a[::-1])]


def _prism_y(pts_xz, y0, y1):
    """Prism of a polygon (x, z) between y0 < y1."""
    from build123d import Polygon, extrude
    s = extrude(Polygon(*_ccw(pts_xz), align=None), amount=y1 - y0)
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]; T[1, 3] = y1
    return SC.transformed(s, T)


def _prism_x(pts_yz, x0, x1):
    from build123d import Polygon, extrude
    s = extrude(Polygon(*_ccw(pts_yz), align=None), amount=x1 - x0)
    T = np.eye(4); T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]; T[0, 3] = x0
    return SC.transformed(s, T)


def mount_design(cover):
    """Per mount: the pad / seat from the cover, the boss, sleeve, nut and screw positions (R frame)."""
    out = {}
    for k, m in MOUNTS.items():
        (x, z), h = m["axis"], m["head"]
        pad, seat, spread = pad_faces(cover, x, z, h)
        face = pad - h * PAD_GAP
        s_end = face - h * m["sleeve"]                       # sleeve end = nut seat
        n_far = s_end - h * (NUT["m"] + m.get("pocket_extra", NUT["depth_clear"]))  # pocket far face
        nut_far = s_end - h * NUT["m"]
        tip = seat - h * m["screw"]
        out[k] = dict(axis_xz=[x, z], head_side=h, pad_face_y=round(pad, 3), seat_y=round(seat, 3), cover_pad_mm=round(abs(seat - pad), 3),
                      boss_face_y=round(face, 3), boss_back_y=m["back"], boss_r=BOSS_R, sleeve_y=[round(face, 3), round(s_end, 3)],
                      sleeve_len=m["sleeve"], nut_y=[round(s_end, 3), round(nut_far, 3)], pocket_y=[round(s_end, 3), round(n_far, 3)],
                      screw=f"{HEAD} M3x{m['screw']}", screw_tip_y=round(tip, 3),
                      thread_past_nut_mm=round((nut_far - tip) * h, 3), nut_engaged_mm=round(min(NUT["m"], (s_end - tip) * h - 0.5), 3),
                      stack_head_to_nut_mm=round(abs(seat - nut_far), 3), pad_ray_spread_mm=spread)
    return out


def build_mounts(design):
    """Green U (R, from the build output) with the two bosses, sleeve bores, nut pockets and screw bores; the 2 new
    sleeves. Returns (part, [sleeve solids])."""
    from build123d import import_step
    cad = json.loads((SC.OUT / "cad-manifest.json").read_text())["parts"]
    green = SC.largest(import_step(str(ROOT / cad[SC.S2]["native_step"])))
    add, cut, sleeves = [], [], []
    for k, d in design.items():
        (x, z), h, m = d["axis_xz"], d["head_side"], MOUNTS[k]
        face, back = d["boss_face_y"], m["back"]
        y0, y1 = sorted((face, back))
        boss = SC._bool("fuse", SC.occ_cyl((x, y0, z), (x, y1, z), BOSS_R), [SC.occ_box(x - BOSS_R, x + BOSS_R, y0, y1, z - 14.0, z)])
        if m["chamfer"]:          # underside: z_bottom = z - R at the pad face, falling 45 deg toward the back
            zb = z - BOSS_R
            yf, yb = face + h * 1.0, face - h * 16.0
            tri = [(yf, zb + 1.0), (yb, zb - 16.0), (yb, z - 40.0), (yf, z - 40.0)]
            boss = SC._bool("cut", SC.largest(boss), [_prism_x(tri, x - BOSS_R - 1, x + BOSS_R + 1)])
        else:
            boss = SC._bool("cut", SC.largest(boss), [SC.occ_box(x - 9, x + 9, y0 - 1, y1 + 1, z - 40.0, z - BOSS_R)])
        # over the nut slot the round boss top would leave knife-edged slot walls: flat top at z + SLOT_TOP
        s_end, back_ = d["sleeve_y"][1], m["back"]
        ya, yb = sorted((s_end, back_ - h * 0.5))
        boss = SC._bool("cut", SC.largest(boss), [SC.occ_box(x - BOSS_R - 1, x + BOSS_R + 1, ya, yb, z + m.get("slot_top", SLOT_TOP), z + 20.0)])
        add.append(SC.largest(boss))
        # green material beyond the pad face (the plate at the rear reaches 0.16 mm into the cover's pad)
        cut.append(SC.occ_cyl((x, face, z), (x, face + h * 4.0, z), PAD_KEEP_R))
        # ... and run that trim out through the plate's bottom edge (a disc alone leaves a 0.06 mm lip under it)
        cut.append(SC.occ_box(x - PAD_KEEP_R, x + PAD_KEEP_R, face, face + h * 4.0, z - 20.0, z))
        s0, s1 = d["sleeve_y"]
        cut.append(SC.occ_cyl((x, s0 + h * 0.5, z), (x, s1, z), SC.M3_SLEEVE[0]))
        p0, p1 = d["pocket_y"]
        hw, e = NUT["s"] / 2 + NUT["clear"], (NUT["s"] / np.sqrt(3) + NUT["clear"] / np.cos(np.pi / 6))
        su = np.array(m.get("slot", (0.0, 1.0)), float)          # slot direction in (x, z); flats face across it
        sv = np.array([-su[1], su[0]])
        prof = [tuple(np.array([x, z]) + u * su + v * sv) for u, v in ((-e, 0.0), (-e / 2, -hw), (20.0, -hw), (20.0, hw), (-e / 2, hw))]
        a, b = sorted((p0, p1))
        cut.append(_prism_y(prof, a, b))
        cut.append(SC.occ_cyl((x, p1 + h * 0.01, z), (x, d["screw_tip_y"] - h * 0.5, z), SCREW_BORE_R))
        if m["bore_slot"]:
            a2, b2 = sorted((p1 + h * 0.01, d["screw_tip_y"] - h * 0.5))
            cut.append(SC.occ_box(x - SCREW_BORE_R, x + SCREW_BORE_R, a2, b2, z, z + 20.0))
        sleeves.append(SC.sleeve((x, s1, z), (x, s0, z), *SC.M3_SLEEVE))
    part = SC.largest(SC._bool("cut", SC.largest(SC._bool("fuse", green, add)), cut))
    return part, sleeves


def nut_mesh(x, z, y_near, y_far, slot=(0.0, 1.0)):
    """ISO 4032 M3 nut (s 5.5, m 2.4, bore dia 3) between y_near and y_far, a corner toward the slot direction (x, z)
    so its flats lie along the slot walls."""
    a = np.radians(np.arange(6) * 60.0 + np.degrees(np.arctan2(slot[1], slot[0])))
    hexa = trimesh.creation.extrude_polygon(__import__("shapely.geometry", fromlist=["Polygon"]).Polygon(
        np.c_[np.cos(a), np.sin(a)] * NUT["s"] / np.sqrt(3)).difference(
        __import__("shapely.geometry", fromlist=["Point"]).Point(0, 0).buffer(1.5, 32)), NUT["m"])
    lo = min(y_near, y_far)
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]     # (u, v, w) -> (u, w, v): w along y
    m = hexa.copy(); m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    m.apply_translation([x, lo, z])
    return m


def screw_plan(k, d):
    import screw_plan as SP
    sp = SP.screw_spec("M3", "M", HEAD)
    (x, z), h = d["axis_xz"], d["head_side"]
    return dict(thread="M3", head_dia_mm=sp["dk"], head_height_mm=sp["k"], length_mm=float(MOUNTS[k]["screw"]),
                point_mm=sp["point_mm"], axis_out=np.array([0.0, float(h), 0.0]),
                opening=np.array([x, d["nut_y"][0], z]), stack_mm=abs(d["seat_y"] - d["nut_y"][0]))


def _save_npz(name, m):
    PARTS.mkdir(parents=True, exist_ok=True)
    f = PARTS / f"{name}.npz"
    np.savez_compressed(f, vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, np.int64))
    return dict(mesh=SC._rel(f), mesh_sha256=SC._sha(f), mesh_units="mm",
                bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()])


def openings(before, after):
    """The removed material as connected pieces: volume (mm3), box extents (mm), max extent, centre."""
    rem = _tm(_mf(before) - _mf(after))
    rows = []
    for q in rem.split(only_watertight=False):
        v = float(abs(q.volume))            # signed volume; approximate when the piece is not closed (flagged)
        if not v > 1.0:
            continue
        ext = q.bounds[1] - q.bounds[0]
        rows.append(dict(volume_mm3=round(v, 1), closed=bool(q.is_watertight), extent_xyz_mm=np.round(ext, 1).tolist(),
                         max_extent_mm=round(float(ext.max()), 1),
                         centre_mm=np.round(q.bounds.mean(0), 1).tolist()))
    rows.sort(key=lambda r: -r["volume_mm3"])
    return dict(removed_mm3=round(float(before.volume - after.volume), 1), before_mm3=round(float(before.volume), 1),
                after_mm3=round(float(after.volume), 1), pieces=rows)


def _export(name, shapes):
    from build123d import Compound, export_step, export_stl
    PARTS.mkdir(parents=True, exist_ok=True)
    comp = Compound(children=list(shapes))
    st, sl = PARTS / f"{name}.step", PARTS / f"{name}.stl"
    export_step(comp, str(st))
    for tol, ang in ((0.02, 0.08), (0.035, 0.09)):
        export_stl(comp, str(sl), tolerance=tol, angular_tolerance=ang)
        if trimesh.load(str(sl), force="mesh", process=True).is_watertight:
            break
    bb = comp.bounding_box()
    return dict(native_step=SC._rel(st), native_step_sha256=SC._sha(st), mesh=SC._rel(sl), mesh_sha256=SC._sha(sl), mesh_units="mm",
                bounds_mm=[[round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)], [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]],
                volume_mm3=round(float(sum(x.volume for x in shapes)), 2), native_solid_count=len(shapes))


def hardware_rows(design, templ):
    """Cover screws and nuts (R and L) as manifest rows; names as the old cover posts' hardware."""
    import screw_plan as SP
    rows = []
    for s, sx in (("R_", 1.0), ("L_", -1.0)):
        M = np.diag([sx, 1.0, 1.0, 1.0])
        for k, d in design.items():
            (x, z) = d["axis_xz"]
            pl = screw_plan(k, d)
            sm = SP.screw_mesh(pl); nm = nut_mesh(x, z, d["nut_y"][0], d["nut_y"][1], MOUNTS[k].get("slot", (0.0, 1.0)))
            for m in (sm, nm):
                m.apply_transform(M)
                if m.volume < 0:
                    m.invert()
            for kind, m, nmx in (("screw", sm, f"{s}upper_arm_shell_{k}_M3x{MOUNTS[k]['screw']}"), ("nut", nm, f"{s}upper_arm_shell_{k}_M3_nut")):
                t = copy.deepcopy(templ[kind]); 
                for key in ("native_step", "native_step_sha256"):
                    t.pop(key, None)
                t.update(name=nmx, physical_part_id=nmx, owning_link=f"{s}shoulder_roll_link", role="fastener", **_save_npz(nmx, m),
                         volume_mm3=round(float(m.volume), 3), mass_kg=round(float(m.volume) * 7.85e-6, 7),
                         mass_basis="mesh volume x 7.85 g/cm3",
                         material=(f"{HEAD} M3x{MOUNTS[k]['screw']} through the cover pad and the green U sleeve into the captive nut"
                                   if kind == "screw" else "ISO 4032 M3 nut, captive in the green U slot"),
                         shoulder_shells="cover mount on the green U, robot/body/arm/shoulder_shells.py")
                if kind == "screw":
                    t["screw_plan"] = dict(thread_label="M3", head_standard=HEAD, length_mm=float(MOUNTS[k]["screw"]),
                                           stack_mm=round(pl["stack_mm"], 3), terminal="ISO 4032 M3 nut",
                                           nut_engaged_mm=d["nut_engaged_mm"], thread_past_nut_mm=d["thread_past_nut_mm"])
                rows.append(t)
    return rows


def apply():
    """Cut the torso shells and the upper-arm covers, add the cover mounts, update the candidate manifest."""
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from build123d import Compound, import_step
    t0 = time.time()
    man = json.loads(SC.CANDIDATE.read_text())      # the selected body (shoulder_clevis.py manifest); never written here
    srcm = json.loads(SC.MANIFEST.read_text())
    src = F.Body(srcm)
    SP_ = {p["name"]: p for p in srcm["parts"]}
    P = {p["name"]: p for p in man["parts"]}
    rep = dict(idle_task="T-tph7xb (parent T-ce88t9)", owner="M-s9tnc9: shells and cover redesigned once the U connectors passed",
               candidate=SC._rel(CANDIDATE), input_manifest=SC._rel(SC.CANDIDATE),
               input_manifest_sha256=SC._sha(SC.CANDIDATE), source_manifest=SC._rel(SC.MANIFEST),
               source_manifest_sha256=SC._sha(SC.MANIFEST), cad_manifest=SC._rel(SC.OUT / "cad-manifest.json"),
               method=dict(wall_mm=WALL, gap_green_mm=GAP, blue_clearance_mm=BLUE_CLEAR, bottom_cut=dict(normal=YAW_AXIS.tolist(), s=BOTTOM_S),
                           cover=COVER, cap=CAP, trim=TRIM, torso_opening=TORSO_OPENING,
                           note="parametric skin (superellipse rings, one spline per ring value) + analytic openings; no sweeps"),
               source=dict(path=SC._rel(__file__), sha256=SC._sha(__file__)))
    # 1. mounts (R designed against the source cover's wells; L = mirror x)
    design = mount_design(src.mesh("R_upper_arm_hard_cover"))
    rep["mounts"] = design
    print("mounts:", json.dumps(design), flush=True)
    part, new_sl = build_mounts(design)
    cad = json.loads((SC.OUT / "cad-manifest.json").read_text())["parts"]
    old_sl = import_step(str(ROOT / cad[SC.S2 + "_steel_sleeves"]["native_step"])).solids()
    print(f"green U with mounts: {part.volume:.0f} mm3 (was {cad[SC.S2]['volume_mm3']}), valid {part.is_valid} ({time.time() - t0:.0f} s)", flush=True)
    for s, f in (("R_", lambda x: x), ("L_", SC.mirror_x)):
        nm = _side(SC.S2, s); r = _export(nm, [f(part)]); mat, rho = SC._density(nm)
        q = P[nm]; q.update(r, mass_kg=round(r["volume_mm3"] * rho * 1e-6, 6),
                            mass_basis=f"native STEP volume x {rho} g/cm3 ({mat}, part setting material)",
                            shoulder_shells="green U of the build + 2 cover-mount bosses (sleeve bore, captive M3 nut slot), shoulder_shells.py")
        q.pop("native_validity", None)
        nm2 = nm + "_steel_sleeves"
        sls = [f(x) for x in (list(old_sl) if s == "R_" else [])] + [f(x) for x in new_sl]
        if s == "L_":
            sls = [x for x in import_step(str(ROOT / cad[nm2]["native_step"])).solids()] + [SC.mirror_x(x) for x in new_sl]
        r = _export(nm2, sls)
        P[nm2].update(r, mass_kg=round(r["volume_mm3"] * 7.93e-6, 7), mass_basis="304 SS sleeves, native STEP volume x 7.93 g/cm3",
                      shoulder_shells=f"{len(sls) - 2} U sleeves of the build + 2 cover-mount sleeves (4.5 x 3.2), shoulder_shells.py")
    # 2. hardware rows (old names of the cover-post hardware)
    templ = dict(screw=SP_["R_upper_arm_shell_front_M3x8"], nut=SP_["R_upper_arm_shell_front_M3_nut"])
    hw = hardware_rows(design, templ)
    mine = {r["name"] for r in hw} | {f"{s}upper_arm_shell_{k}_{h}" for s in ("R_", "L_") for k in MOUNTS for h in ("M3x6", "M3x8", "M3_nut")}
    man["parts"] = [p for p in man["parts"] if p["name"] not in mine] + hw
    P = {p["name"]: p for p in man["parts"]}
    body = F.Body(man)
    print(f"hardware: {[r['name'] for r in hw]} ({time.time() - t0:.0f} s)", flush=True)
    # 3. covers: the parametric skin with the wells and the top trim on the right arm; the left cover is its exact mirror
    from scipy.spatial import cKDTree
    covers, torso = {}, {}
    sh, outer = cover_skin()
    sh = add_wells(sh, outer, design)
    pre = _tm(sh)
    cut = _mf(trim_cutter(_roll_axis_xz(body)))
    cov_R = _tm(sh - cut)
    cov_R = trimesh.util.concatenate([q for q in cov_R.split(only_watertight=False) if abs(q.volume) > 20.0])
    c_src = src.mesh("R_upper_arm_hard_cover")
    Ps, Ns = today_samples(c_src, design)
    dev = skin_deviation(Ps, Ns)
    kept = ~trim_removed(Ps, _roll_axis_xz(body)) & (Ps @ YAW_AXIS >= BOTTOM_S + 0.5)
    G = requirement(body, design)
    Sc = np.vstack([cov_R.vertices, trimesh.sample.sample_surface(cov_R, 800000, seed=4)[0]])   # ~0.1 mm spacing
    Gs = G[::3]
    dG = cKDTree(Sc).query(Gs)[0] * np.where(cov_R.contains(Gs), -1.0, 1.0)      # < 0: inside the cover material
    inside = cover_outer().contains(Gs); trimmed = trim_removed(Gs, _roll_axis_xz(body))
    sweep = blue_roll_sweep(body, n=12000)
    S_ = Sc[::4]
    dB = cKDTree(sweep).query(S_)[0]
    del sweep

    def where(mask, v, f):
        i = int(f(np.where(mask, v, np.nan)))
        return dict(mm=round(float(v[i]), 2), at_mm=np.round(Ps[i], 1).tolist())
    green_zone = kept & (Ps[:, 2] > 312.0) & (Ps[:, 2] < 350.0)
    skin = dict(today_points=int(kept.sum()), rms_mm=round(float(np.sqrt(np.mean(dev[kept] ** 2))), 3),
                p05_p50_p95_mm=np.round(np.percentile(dev[kept], [5, 50, 95]), 2).tolist(),
                max_outward=where(kept, dev, np.nanargmax), max_inward=where(kept, dev, np.nanargmin),
                max_outward_green_zone_z312_350=where(green_zone, dev, np.nanargmax),
                max_outward_elsewhere=where(kept & ~green_zone, dev, np.nanargmax))
    enc = dict(points=int(len(dG)), min_distance_to_cover_mm=round(float(dG.min()), 3),
               p01_distance_mm=round(float(np.percentile(dG, 1)), 3), inside_outer_skin_pct=round(100.0 * inside.mean(), 2),
               under_the_trim_pct=round(100.0 * trimmed.mean(), 2),
               note="green U + sleeves + roll-case screws (bosses at the wells left out) to the cover's material")
    trimrep = dict(removed_by_trim_mm3=round(float(pre.volume - cov_R.volume), 1), skin_with_wells_mm3=round(float(pre.volume), 1),
                   blue_swept_min_distance_mm=round(float(dB.min()), 2), blue_swept_min_at_mm=np.round(S_[int(dB.argmin())], 1).tolist(),
                   cover_points_within_3mm_of_blue=int((dB < BLUE_CLEAR).sum()))
    print(f"cover: {cov_R.volume:.0f} mm3 (today {c_src.volume:.0f}); skin vs today {json.dumps(skin)}", flush=True)
    print(f"green enclosure {json.dumps(enc)}; trim {json.dumps(trimrep)} ({time.time() - t0:.0f} s)", flush=True)
    for s in ("R_", "L_"):
        cn = f"{s}upper_arm_hard_cover"
        out = cov_R if s == "R_" else _mirror(cov_R)
        covers[cn] = dict(volume_mm3=round(float(out.volume), 1), today_volume_mm3=round(float(src.mesh(cn).volume), 1),
                          watertight=bool(out.is_watertight), skin_vs_today=skin, green_enclosure=enc, top_trim=trimrep,
                          built="parametric R cover" if s == "R_" else "mirror of the R cover")
        _update_shell(P[cn], SP_[cn], src.mesh(cn), out, cn,
                      "parametric upper-arm cover (superellipse rings, splines in z, dome; constant wall) enclosing the green U, "
                      "top trimmed by two planes through the roll axis, screw wells at the green U mounts")
    # 4. torso shells: one round opening about each shoulder pitch axis (the left one mirrors the right)
    sh_t = {n: src.mesh(n) for n in TORSO_SHELLS}
    cuts = {}
    for sx in (1.0, -1.0):
        cuts[sx] = torso_cutter(list(sh_t.values()), _pitch_axis_yz(body), sx=sx)
    for n in TORSO_SHELLS:
        cur = _tm(_mf(sh_t[n]) - _mf(cuts[1.0][0]) - _mf(cuts[-1.0][0]))
        cur = trimesh.util.concatenate([q for q in cur.split(only_watertight=False) if abs(q.volume) > 20.0])
        torso[n] = dict(opening=dict(TORSO_OPENING, axis_yz=_pitch_axis_yz(body).tolist(), **cuts[1.0][1]), **openings(sh_t[n], cur))
        _update_shell(P[n], SP_[n], sh_t[n], cur, n,
                      f"shoulder openings: round about each pitch axis, R {TORSO_OPENING['R_mm']} mm, cone below |x| {TORSO_OPENING['x_mm']}, "
                      f"outer edge rounded {TORSO_OPENING['round_mm']} mm")
        print(f"{n}: {torso[n]['before_mm3']} -> {torso[n]['after_mm3']} mm3, pieces "
              f"{[(r['volume_mm3'], r['extent_xyz_mm'], r['centre_mm']) for r in torso[n]['pieces']]}", flush=True)
    rep.update(covers=covers, torso_shells=torso, seconds=round(time.time() - t0, 1))
    man["shoulder_shells"] = dict(idle_task=rep["idle_task"], report=SC._rel(REPORT), source=rep["source"], method=rep["method"],
                                  changes=sorted([n for n in P if P[n].get("shoulder_shells")]),
                                  input_manifest=SC._rel(SC.CANDIDATE), input_manifest_sha256=SC._sha(SC.CANDIDATE),
                                  note="Candidate: the selected body + shoulder shell cuts + cover mounts. Reads the selected body, "
                                       "writes only its own files; rerun after shoulder_clevis.py manifest; idempotent.")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1, default=float) + "\n")
    man["study_id"] = man["result_id"] = "cove50-shoulder-u-shells"
    CANDIDATE.write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {SC._rel(CANDIDATE)} and {SC._rel(REPORT)} ({time.time() - t0:.0f} s)")


def _mirror(m):
    """Mirror x -> -x (face order flipped so the normals stay outward)."""
    v = np.asarray(m.vertices, float).copy(); v[:, 0] *= -1.0
    return trimesh.Trimesh(v, np.asarray(m.faces)[:, ::-1], process=True)


def _update_shell(q, src_row, before, after, name, note):
    """Point a shell row at its cut mesh (the source row's mesh stays the input)."""
    r = _save_npz(name, after)
    keep = {k: src_row[k] for k in ("mass_kg",) if k in src_row}
    q.update(r, volume_mm3=round(float(after.volume), 2), mesh_note=(src_row.get("mesh_note", "") + "; " + note).lstrip("; "),
             shoulder_shells=dict(input_mesh=src_row["mesh"], input_mesh_sha256=src_row.get("mesh_sha256"),
                                  removed_mm3=round(float(before.volume - after.volume), 1)))
    if "mass_kg" in keep:
        q["mass_kg"] = round(keep["mass_kg"] * float(after.volume / before.volume), 6)
        q["mass_basis"] = (src_row.get("mass_basis", "source mass") + f"; x mesh volume ratio {after.volume / before.volume:.4f} after the shoulder cut")



# ---------------------------------------------------------------------------------------------- owner images
LIGHT = dict(shell=(0.86, 0.86, 0.88), cover=(0.93, 0.90, 0.82), nut=(0.85, 0.72, 0.30))


def _scene_rows(man, s="R_"):
    """(name, colour, link) of the right shoulder scene: torso shells and cover light, U's coloured."""
    names = {p["name"]: p for p in man["parts"]}
    rows = [(n, LIGHT["shell"]) for n in TORSO_SHELLS] + [(f"{s}upper_arm_hard_cover", LIGHT["cover"])]
    rows += [(_side(SC.S1, s), SC.COL["blue"]), (_side(SC.S2, s), SC.COL["green"]),
             (_side(SC.S1, s) + "_steel_sleeves", SC.COL["sleeve"]), (_side(SC.S2, s) + "_steel_sleeves", SC.COL["sleeve"]),
             (f"upper__{s}roll_C047_case", SC.COL["case"]), (f"upper__{s}roll_C047_main_horn", SC.COL["horn"]),
             (_side(SC.IDLER, s), SC.COL["idler"]), (f"upper__{s}pitch_C047_main_horn", SC.COL["horn"]),
             (f"upper__{s}yaw_C047_case", SC.COL["pitchcase"])]
    rows += [(n, SC.COL["screw"]) for n in names if n.startswith(f"{s}shoulder_u_")]
    rows += [(n, LIGHT["nut"] if n.endswith("_nut") else SC.COL["screw"]) for n in names if n.startswith(f"{s}upper_arm_shell_")]
    return [(n, c) for n, c in rows if n in names and names[n].get("mesh")]


def _posed(body, rows, q):
    out = []
    for n, c in rows:
        m = body.mesh(n).copy(); m.apply_transform(body.link_T(body.parts[n]["owning_link"], q)); out.append((m, c))
    return out


def views():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    man = json.loads(CANDIDATE.read_text()); body = F.Body(man)
    old = json.loads(SC.CANDIDATE.read_text()); ob = F.Body(old)     # before = the selected body (input of apply)
    rows = _scene_rows(man)
    legend = [("torso shells", LIGHT["shell"]), ("upper-arm cover", LIGHT["cover"]), ("blue U (pitch link)", SC.COL["blue"]),
              ("green U (roll link)", SC.COL["green"]), ("roll servo case", SC.COL["case"]), ("horns", SC.COL["horn"]),
              ("roll idler", SC.COL["idler"]), ("yaw servo case", SC.COL["pitchcase"]), ("sleeves / screws", SC.COL["sleeve"]),
              ("cover-mount nuts", LIGHT["nut"])]
    lo, hi = np.array([62.0, -52.0, 296.0]), np.array([166.0, 60.0, 396.0])
    VIEWS.mkdir(parents=True, exist_ok=True)
    # (a) before / after, 4 views
    fig, axs = plt.subplots(2, 4, figsize=(28, 14.5), dpi=80)
    for r, (lab, b, rr) in enumerate((("BEFORE", ob, _scene_rows(old)), ("AFTER", body, rows))):
        for c, (t, d, up, xl, yl) in enumerate(SC.VIEWS4):
            img, ext = SC._render(_posed(b, rr, {}), d, up, lo, hi, px_mm=0.16)
            ax = axs[r, c]; ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"{lab}: {t}", fontsize=12)
            if xl:
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.tick_params(labelsize=8)
            else:
                ax.axis("off")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle("Right shoulder at standing, before / after: parametric upper-arm cover enclosing the green U, top trimmed about the roll axis, "
                 "round torso opening about the pitch axis", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); f1 = VIEWS / "shoulder-shells-before-after.png"; fig.savefig(f1, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f1)}")
    # (b) range poses
    poses = [("pitch -60", {"R_shoulder_pitch": -60.0}), ("pitch +150", {"R_shoulder_pitch": 150.0}),
             ("roll -15", {"R_shoulder_roll": -15.0}), ("roll +95", {"R_shoulder_roll": 95.0})]
    vv = [("3/4 (front, outside, above)", (-0.75, 0.9, -0.55), (0, 0, 1)), ("FRONT (-y)", (0, 1, 0), (0, 0, 1)),
          ("SIDE (+x)", (-1, 0, 0), (0, 0, 1))]
    lo2, hi2 = np.array([55.0, -75.0, 270.0]), np.array([200.0, 80.0, 430.0])
    fig, axs = plt.subplots(len(vv), len(poses), figsize=(28, 21), dpi=80)
    for c, (pl, q) in enumerate(poses):
        sc = _posed(body, rows, q)
        for r, (t, d, up) in enumerate(vv):
            img, ext = SC._render(sc, d, up, lo2, hi2, px_mm=0.2)
            ax = axs[r, c]; ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"{pl}: {t}", fontsize=12); ax.axis("off")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle(f"Right shoulder at the range ends (after): blue U >= {BLUE_CLEAR:g} mm from the cover trim (roll -15..95) and the torso opening (pitch -60..150)", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); f2 = VIEWS / "shoulder-shells-range-poses.png"; fig.savefig(f2, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f2)}")
    # (c) the cover mount: green U with bosses, sleeves, nuts, screws (cover off), and sections through both mount axes
    rep = json.loads(REPORT.read_text()); dsg = rep["mounts"]
    mrows = [(n, c) for n, c in rows if n not in TORSO_SHELLS and not n.endswith("hard_cover")]
    fig, axs = plt.subplots(2, 3, figsize=(24, 16), dpi=80)
    lo3, hi3 = np.array([96.0, -30.0, 312.0]), np.array([144.0, 38.0, 352.0])
    for c, (t, d, up) in enumerate((("FRONT (-y), cover off", (0, 1, 0), (0, 0, 1)), ("BACK (+y), cover off", (0, -1, 0), (0, 0, 1)),
                                    ("3/4 from front-above, cover off", (-0.6, 0.8, -0.6), (0, 0, 1)))):
        img, ext = SC._render(_posed(body, mrows, {}), d, up, lo3, hi3, px_mm=0.06)
        axs[0, c].imshow(img, extent=ext, interpolation="bilinear"); axs[0, c].set_title(t, fontsize=12); axs[0, c].axis("off")
    secs = [n for n, _ in rows if n in body.parts]
    colr = {n: c for n, c in rows}
    for c, (k, plane) in enumerate((("front", "x"), ("rear", "x"), ("front", "y"))):
        d = dsg[k]; (x, z) = d["axis_xz"]
        ax = axs[1, c]
        o, nrm, ij = ((x, 0, 0), (1, 0, 0), (1, 2)) if plane == "x" else ((0, float(np.mean(d["nut_y"])), 0), (0, 1, 0), (0, 2))
        for n in secs:
            sct = body.mesh(n).section(plane_origin=o, plane_normal=nrm)
            if sct is None:
                continue
            from shapely.geometry import Polygon as SPoly
            geo = None
            for e in sct.discrete:
                if len(e) < 4:
                    continue
                pg = SPoly(e[:, list(ij)]).buffer(0)
                geo = pg if geo is None else geo.symmetric_difference(pg)      # even-odd fill
            for poly in ([] if geo is None else getattr(geo, "geoms", [geo])):
                if poly.geom_type != "Polygon":
                    continue
                g = np.array(poly.exterior.coords)
                ax.fill(g[:, 0], g[:, 1], color=colr[n], ec="#333333", lw=0.6)
                for h in poly.interiors:
                    hh = np.array(h.coords); ax.fill(hh[:, 0], hh[:, 1], color="white", ec="#333333", lw=0.6)
        cx = (d["axis_xz"][1] if plane == "x" else d["axis_xz"][0])
        cy0 = (np.mean(d["sleeve_y"]) if plane == "x" else z)
        if plane == "x":
            ax.set_xlim(min(d["seat_y"], d["boss_back_y"]) - 8, max(d["seat_y"], d["boss_back_y"]) + 8); ax.set_ylim(z - 12, z + 12)
            ax.set_xlabel("y mm"); ax.set_ylabel("z mm")
        else:
            ax.set_xlim(x - 14, x + 14); ax.set_ylim(z - 12, z + 12); ax.set_xlabel("x mm"); ax.set_ylabel("z mm")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{k} mount, section {'x' if plane == 'x' else 'y'} = {o[0] if plane == 'x' else round(o[1], 2)}: "
                     f"cover pad {d['cover_pad_mm']} mm, sleeve {d['sleeve_len']} mm, nut slot open to {'+z' if not MOUNTS[k].get('slot', (0, 1))[0] else '+z tilted 15 deg to +x'}, {d['screw']}", fontsize=11)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle("Cover mount on the green U: boss + pressed 304 sleeve (4.5 x 3.2) + captive M3 nut; screw from outside through the cover's well", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); f3 = VIEWS / "shoulder-shells-cover-mount.png"; fig.savefig(f3, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f3)}")


if __name__ == "__main__":
    {"fit-cover": fit_all, "apply": apply, "views": views}[sys.argv[1] if len(sys.argv) > 1 else "apply"]()
