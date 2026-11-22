"""Shoulder shells: upper-arm covers and torso shells for the shoulder U connectors, covers mounted on the green U
(Idle T-tph7xb, parent T-ce88t9; owner M-s9tnc9: shells and cover redesigned once the connectors passed their checks).

Owner 2026-11-22: "The shell should go above the green U and also the shell currently follows the sweep of the motors to
closely" and "Make use of gently and simple parametric functions to define the shape of the shell". So every surface made
here is a few smooth parametric functions (the thigh-shell pattern, robot/body/leg/thigh_shell.py); nothing follows a swept
volume, and the openings are simple shapes about the joint axes:

  upper-arm cover (roll link)  outer skin of horizontal superellipse rings about a centre line (cx, cy): per ring four
                               half-widths (a_in / a_out in x, b_front / b_back in y) and two exponents (front / back half),
                               each ONE natural cubic spline in z through a few control values (COVER), continued smoothly
                               into a dome (CAP: x half-widths rounded with p about a line along y, y half-widths with p_y).
                               Below 352 the stations are today's fit (fit-cover); two smooth C2 bumps (BUMPS) grow the skin
                               where the green U's inboard corners reach through it, so the green U sits inside with GAP +
                               WALL. Owner 2026-11-22 ("scoping closely inward towards the rotation point", "smarted slopes
                               on the existing shell"): the upper stations converge on the roll servo and the roll axis, the
                               dome rounds over the roll axis, the front / back walls rise beside the blue U's arms (ears).
                               Constant WALL inward along the surface normal; the bottom keeps today's elbow cut (BOTTOM_S).
  cover rim (RIM)              the surface ends at a closed curve in the roll axis' own coordinates (psi about the axis, y
                               along it): a superellipse over the sector the blue U (plate, arms) sweeps between roll -15 and
                               95 and over its width between the arms' outer faces; nothing else is open.
  torso shells (torso link)    the skin slopes in toward each shoulder pitch axis as a dish of revolution about the axis
                               (TORSO_SLOPE: floor x, floor radius, rise angle, end radius): each skin point moves along x
                               (with the wall under it) to the dish where it lay outboard of it; the dish ends at R_end,
                               the opening, just around the pitch horn's coupling. Fitted for RUN_CLEAR to the whole arm.
  cover mount                  two pads on the green U at the cover's screw wells (positions as today's cover): a boss with
                               a pressed 304 sleeve (4.5 x 3.2, the M3 sleeve of the U) and a captive ISO 4032 M3 nut in a
                               slot (front open to +z, rear to +x); ISO 7380-1 M3 button head from outside through the
                               cover's well (front M3x8, rear M3x6). The wells are rebuilt analytically in the new cover.

Runs after `shoulder_clevis.py manifest`: reads the selected body it wrote (assembly-parts-shoulder-u.json, never written
here), takes the shell meshes from the roll-flip source manifest and the green U from the build output (cad-manifest.json),
and writes its own candidate shells/assembly-parts-shoulder-u-shells.json plus meshes / STEP under
artifacts/robot/body/arm/shoulder-clevis/shells/. Deterministic: a rerun gives the same files.

  .venv/bin/python -B robot/body/arm/shoulder_shells.py fit-cover [keep-upper]   # re-fit (prints COVER / CAP / BUMPS / RIM / TORSO_SLOPE)
  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 1800 -- .venv/bin/python -B robot/body/arm/shoulder_shells.py apply
  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 900 -- .venv/bin/python -B robot/body/arm/shoulder_shells.py views
Check the candidate with robot/body/checks/run.py --manifest artifacts/robot/body/arm/shoulder-clevis/shells/assembly-parts-shoulder-u-shells.json
  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 900 -- .venv/bin/python -B robot/body/arm/shoulder_shells.py mount-views
`views` also measures the visible open area of the right shoulder from outside (front, side, the owner's camera) before /
after (BEFORE: the previous candidate, if a copy is kept). Every result is a measurement (mm, mm2, mm3); nothing accepts or rejects.
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
    "cx": ((282.0, 310.0, 338.0, 352.0, 366.0), (133.360, 127.096, 120.833, 113.416, 106.000,)),
    "cy": ((282.0, 310.0, 338.0, 352.0, 366.0), (0.882, 2.550, 4.218, 4.997, 5.775,)),
    "a_in": ((282.0, 310.0, 338.0, 352.0, 366.0), (29.889, 27.264, 26.796, 20.898, 15.000,)),
    "a_out": ((282.0, 310.0, 338.0, 352.0, 366.0), (25.717, 29.102, 25.715, 23.358, 21.000,)),
    "b_front": ((282.0, 310.0, 338.0, 352.0, 366.0), (26.933, 31.527, 34.422, 32.214, 30.006,)),
    "b_back": ((282.0, 310.0, 338.0, 352.0, 366.0), (26.517, 31.700, 33.886, 31.946, 30.006,)),
    "n_front": ((282.0, 324.0, 366.0), (2.003, 2.005, 4.000,)),
    "n_back": ((282.0, 324.0, 366.0), (2.099, 2.000, 4.000,)),
}
CAP = dict(z0=360.0, z_top=379.0, p=2.0, p_y=6.0)   # dome above z0: the z0 ring's x half-widths scaled by
# g(p), its y half-widths by g(p_y), g(q) = (1 - ((z - z0) / (z_top - z0))^q)^(1/q): p ~ 2 rounds the top about a line
# along y (the roll axis' direction), a larger p_y keeps the front / back walls up to near the crest
RING = ("a_in", "a_out", "b_front", "b_back", "n_front", "n_back")
FIT_N = (2.0, 4.5)         # exponent range in the fit
CAP_START = (364.0, 377.0, 2.0)   # the fit's start for z0, z_top, p (fixed, so a re-fit is reproducible)
# Green U enclosure (owner 2026-11-22: "the shell should go above the green U"): the base skin above is today's cover; where
# the green U's inboard corners reach through it, smooth bumps are added to the ring radius:
#   dr = A * w((z_ring - z) / w_z) * w(angle(theta - th) / w_th),   w(t) = (1 - t^2)^3 for |t| < 1, else 0
# (value, slope and curvature vanish at the bump's edge: it blends into the skin with continuous curvature). theta / th in deg
# about the ring centre from +x (outboard) toward +y (back). fit-cover fits A, z, th; the half-widths are chosen gentle
# (slope A / w <= about 0.25).
BUMPS = (dict(A=3.914, z=324.545, w_z=24.0, th=-132.766, w_th=45.0),    # front-inboard corner of the green U
         dict(A=5.754, z=323.538, w_z=24.0, th=129.736, w_th=45.0))    # back-inboard corner
BUMP_W = dict(w_z=24.0, w_th=45.0)
GROW_MARGIN = 0.35         # extra radial margin in the fit (radial vs normal distance); the exact distance is reported

# Upper end of the cover (owner 2026-11-22: "The shell should be scoping closely inward towards the rotation point to leave
# as minimal open space as possible", "smarted slopes on the existing shell"). The upper stations of COVER / CAP converge on
# the roll servo and the roll axis (upper_stations(): the dome centred on the roll axis, just over the blue U's turning parts
# with RUN_CLEAR + WALL; TOP below); the front and back walls rise beside the blue U's arms and reach inboard to EAR_X_IN. The surface ends at its rim, a closed curve in the roll axis' own coordinates (psi = angle about the roll axis
# from +z toward +x, deg; y along the axis, mm): the skin is kept where
#   F(psi, y) = |(psi - psi_c) / psi_h|^m + |(y - y_c) / y_h|^m >= 1
# (a superellipse in (psi, y), corners rounded by m), measured radially from the roll axis between r0 and r1. Its sector
# psi_c +- psi_h is where the blue U (stage 1 plate and arms) passes over the roll servo between roll -15 and 95; its width
# y_c +- y_h spans the blue U between its arms' outer faces. fit-cover places it: the least skin removed, every skin point
# within RUN_CLEAR + 0.3 of the swept blue U beyond NEAR_AXIS inside the rim with RIM_MARGIN (in mm along the skin), every
# skin point of the green U's enclosure outside it with RIM_MARGIN.
RIM = dict(psi_c=-50.2, psi_h=110.9, y_c=6.18, y_h=26.04, m=16.0, r0=4.0, r1=38.0)
RIM_MARGIN = 0.6           # rim edge beyond the swept blue set and beyond the green U's enclosure, mm along the skin
NEAR_AXIS = 14.0           # mm from the roll axis: inside it the blue U (horn / idler, their screws) only turns in place
EAR_X_IN = 91.0            # the front / back walls ("ears") rise beside the opening and reach inboard to this x at the top
                           # station: they cover the blue U's arms, the roll horn and idler and their screws seen from the
                           # front and the back (further inboard they would swing into the torso at roll -15)
RUN_CLEAR = 1.75           # running clearance of a moving part to a shell over the joint ranges (owner: about 1.5-2 mm)
TOP = dict(crest=19.0, p=2.0, p_y=6.0, n=4.0, ear_over=2.6, psi_out=40.0)   # upper_stations(): the crest above the roll
                           # axis (the arm ends need 18.7 + RUN_CLEAR), the dome round about a line along y (p = 2) with
                           # its front / back walls kept up to near the crest (p_y), squarer top rings (n), the walls
                           # ear_over mm beyond the opening's y-edges; the outboard radius from the turning parts past
                           # psi_out (the blue plate reaches psi 40 at roll 95); the stations below 352 keep today's fit
# Torso shells: the skin slopes in toward each shoulder pitch axis (along x) as a dish of revolution about the axis, outer
# skin at x_dish(rho) where it lay outboard of it (rho = distance from the pitch axis in y-z):
#   x_dish(rho) = x_floor + tan(slope_deg) * ease(rho - R_floor),  ease(u) = 0 (u <= -k), (u + k)^2 / 4k (|u| < k), u (u >= k)
# with k = ease_mm; the skin follows min(own x, x_dish) blended over blend_mm (C1), each skin point and the wall under it
# moved along x by the same amount (constant wall along x). The dish ends at R_end: the opening, as deep and as small as the
# blue U's pitch coupling allows (inside R_end the pitch horn turns at the servo face). fit-cover: x_floor / R_floor /
# slope_deg the shallowest dish that keeps RUN_CLEAR to every part of the arm (the pitch-link, roll-link and upper-arm parts,
# the covers) over roll -15..95 and yaw -60..60 (pitch turns about the axis itself, so it leaves x and rho unchanged).
TORSO_SLOPE = dict(R_end=20.75, x_floor=84.24, R_floor=55.25, slope_deg=60.0, ease_mm=6.0, blend_mm=1.5)
TORSO_R_WIN = 50.0         # the slope is fitted against the arm out to this rho (beyond: the upper arm's lower end at roll
                           # -15 and the chest / back joint screws at rho 56-62; reported, not fitted)
TORSO_WALL_BLEND = (4.0, 22.0)   # the move fades from full (within wall + 4 mm under the outer skin) to none 22 mm deeper


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
_BASIS_CACHE = {}


def _basis(zk, z):
    """B with B @ values = the natural cubic spline through (zk, values) at z (2 heights: straight line, 1: constant).
    Cached by knots and heights (the fits evaluate the same heights many times)."""
    z = np.asarray(z, float)
    key = (tuple(float(v) for v in zk), z.shape, hash(z.tobytes()))
    if key not in _BASIS_CACHE:
        if len(_BASIS_CACHE) > 64:
            _BASIS_CACHE.clear()
        _BASIS_CACHE[key] = _basis_eval(zk, z)
    return _BASIS_CACHE[key]


def _basis_eval(zk, z):
    from scipy.interpolate import CubicSpline
    zk = np.asarray(zk, float); z = np.clip(np.asarray(z, float), zk[0], zk[-1]); k = len(zk)
    if k == 1:
        return np.ones((len(z), 1))
    if k == 2:
        t = (z - zk[0]) / (zk[1] - zk[0])
        return np.c_[1 - t, t]
    out = []
    for i in range(k):
        cs = CubicSpline(zk, np.eye(k)[i], bc_type="natural")
        out.append(cs(z))
    return np.stack(out, -1)


def _basis_up(zk, z):
    """As _basis, but continued above the top knot along the spline's end tangent (its curvature is zero there: C2)."""
    z = np.asarray(z, float)
    key = ("up", tuple(float(v) for v in zk), z.shape, hash(z.tobytes()))
    if key not in _BASIS_CACHE:
        if len(_BASIS_CACHE) > 64:
            _BASIS_CACHE.clear()
        _BASIS_CACHE[key] = _basis_up_eval(zk, z)
    return _BASIS_CACHE[key]


def _basis_up_eval(zk, z):
    from scipy.interpolate import CubicSpline
    zk = np.asarray(zk, float)
    B = _basis(zk, z)
    over = z > zk[-1]
    if over.any() and len(zk) > 2:
        d = np.array([CubicSpline(zk, np.eye(len(zk))[i], bc_type="natural")(zk[-1], 1) for i in range(len(zk))])
        B = B.copy(); B[over] = _basis(zk, np.full(int(over.sum()), zk[-1])) + (z[over] - zk[-1])[:, None] * d[None, :]
    elif over.any() and len(zk) == 2:
        t = (z[over] - zk[0]) / (zk[1] - zk[0]); B = B.copy(); B[over] = np.c_[1 - t, t]
    return B


def _dome(h, p):
    return np.clip(1.0 - np.clip(h, 0.0, 1.0) ** p, 0.0, 1.0) ** (1.0 / p)


def ring(z, cov=None, cap=None):
    """Centre (cx, cy), ring parameters (columns RING) and dome scales g (n, 2): (x half-widths, y half-widths) at heights z."""
    cov = cov or COVER; cap = cap or CAP
    z = np.atleast_1d(np.asarray(z, float))
    ev = {k: _basis_up(cov[k][0], z) @ np.asarray(cov[k][1], float) for k in cov}   # the rings continue into the dome
    h = (z - cap["z0"]) / (cap["z_top"] - cap["z0"])
    g = np.stack([_dome(h, cap["p"]), _dome(h, cap.get("p_y", cap["p"]))], -1)
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
    r = radius(th.ravel(), prm, g) + bump(np.minimum(z.ravel(), (cap or CAP)["z0"]), th.ravel(), bumps) * g[:, 0]
    return r.reshape(z.shape)


def radius(th, prm, g):
    """Outer radius about the ring centre at angle th (from +x toward +y); prm (..., 6) broadcast against th; g the dome
    scales (..., 2) of the x and the y half-widths (or a scalar)."""
    a_in, a_out, bf, bb, nf, nb = (prm[..., i] for i in range(6))
    g = np.asarray(g, float)
    gx, gy = (g[..., 0], g[..., 1]) if g.ndim and g.shape[-1] == 2 else (g, g)
    c, s = np.cos(th), np.sin(th)
    a = np.where(c > 0, a_out, a_in) * gx; b = np.where(s > 0, bb, bf) * gy; e = np.where(s > 0, nb, nf)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (np.abs(c / a) ** e + np.abs(s / b) ** e) ** (-1.0 / e)
    return np.nan_to_num(r, nan=0.0, posinf=0.0)


def _grid(cov, cap, z_lo, nth=240, dz=0.75, ncap=60, bumps=None):
    """Outer skin as a grid of rings (nz, nth, 3) from z_lo up to the dome, the dome pole, and the outward normals."""
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    zs = np.arange(z_lo, cap["z0"], dz)
    phi = np.linspace(0.0, np.pi / 2, ncap)[:-1]                 # dome rows; the last one (phi = pi / 2) is the pole
    phi = np.concatenate([phi, np.linspace(phi[-1], np.pi / 2, 12)[1:-1]])   # finer under the pole (the y-ridge closes there)
    zc = cap["z0"] + (cap["z_top"] - cap["z0"]) * np.sin(phi) ** (2.0 / cap["p"])
    z = np.concatenate([zs, zc])
    cx, cy, prm, g = ring(z, cov, cap)
    g[len(zs):, 0] = np.cos(phi) ** (2.0 / cap["p"])
    R = (radius(th[None, :], prm[:, None, :], g[:, None, :])
         + bump(np.minimum(z, cap["z0"])[:, None], th[None, :], bumps) * g[:, None, 0])
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


# ---------------------------------------------------------------------------------------------- wells, rim of the cover
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


def roll_polar(P, axis_xz):
    """psi (deg, about the roll axis from +z toward +x) and rho (mm from the roll axis) of points P (x, y, z)."""
    q = np.asarray(P, float)[:, [0, 2]] - np.asarray(axis_xz, float)
    return np.degrees(np.arctan2(q[:, 0], q[:, 1])), np.hypot(q[:, 0], q[:, 1])


def rim_F(psi, y, rim=None):
    """The rim function: < 1 inside the opening (psi wrapped about psi_c)."""
    r = rim or RIM
    u = np.degrees(np.angle(np.exp(1j * np.radians(np.asarray(psi, float) - r["psi_c"])))) / r["psi_h"]
    v = (np.asarray(y, float) - r["y_c"]) / r["y_h"]
    return np.abs(u) ** r["m"] + np.abs(v) ** r["m"]


def rim_open(P, axis_xz, rim=None):
    """True for points P inside the opening (the cover's surface does not reach there)."""
    r = rim or RIM
    psi, rho = roll_polar(P, axis_xz)
    return (rim_F(psi, P[:, 1], r) < 1.0) & (rho > r["r0"]) & (rho < r["r1"])


def rim_cutter(axis_xz, rim=None, nt=720, ns=40, nr=12):
    """The opening as a solid in the roll axis' coordinates: (psi, y) inside the rim curve, r0 < rho < r1. Rings of the
    superellipse (s = 0..1 of its size) at r0, its side from r0 to r1, rings back at r1; closed with _solid."""
    r = rim or RIM
    t = np.linspace(-np.pi, np.pi, nt, endpoint=False)
    cu = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / r["m"]); sv = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / r["m"])
    c = np.asarray(axis_xz, float)

    def ring_at(s, rho):
        psi = np.radians(r["psi_c"] + s * r["psi_h"] * cu); y = r["y_c"] + s * r["y_h"] * sv
        return np.c_[c[0] + rho * np.sin(psi), y, c[1] + rho * np.cos(psi)]
    ss = np.linspace(0.0, 1.0, ns + 1)[1:]
    rr = np.linspace(r["r0"], r["r1"], nr + 1)
    V = np.stack([ring_at(s, r["r0"]) for s in ss] + [ring_at(1.0, q) for q in rr[1:-1]] + [ring_at(s, r["r1"]) for s in ss[::-1]])
    ctr = lambda rho: np.array([c[0] + rho * np.sin(np.radians(r["psi_c"])), r["y_c"], c[1] + rho * np.cos(np.radians(r["psi_c"]))])  # noqa: E731
    return _solid(V, ctr(r["r1"]), ctr(r["r0"]))


# ---------------------------------------------------------------------------------------------- torso slope
def _ease(u, k):
    return np.where(u <= -k, 0.0, np.where(u >= k, u, (u + k) ** 2 / (4.0 * k)))


def dish_x(rho, ts=None):
    """Outer skin x of the dish at distance rho from the pitch axis (R side)."""
    t = ts or TORSO_SLOPE
    return t["x_floor"] + np.tan(np.radians(t["slope_deg"])) * _ease(np.asarray(rho, float) - t["R_floor"], t["ease_mm"])


def skin_x_map(shells, dy=0.5):
    """Today's outer skin x of the torso shells on the R side seen along -x, on a (y, z) grid: (ys, zs, X); NaN where the
    ray meets nothing outboard of x 40 (the old opening), filled from the nearest hit."""
    from scipy.interpolate import NearestNDInterpolator
    cat = trimesh.util.concatenate(shells)
    rx = trimesh.ray.ray_pyembree.RayMeshIntersector(cat)
    ys, zs = np.arange(-80.0, 100.0 + 1e-6, dy), np.arange(250.0, 420.0 + 1e-6, dy)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    O = np.c_[np.full(Y.size, 250.0), Y.ravel(), Z.ravel()]
    loc, ir, _ = rx.intersects_location(O, np.tile([-1.0, 0.0, 0.0], (len(O), 1)), multiple_hits=False)
    X = np.full(Y.size, np.nan); X[ir] = loc[:, 0]
    X[X < 40.0] = np.nan
    ok = np.isfinite(X)
    X[~ok] = NearestNDInterpolator(np.c_[Y.ravel()[ok], Z.ravel()[ok]], X[ok])(Y.ravel()[~ok], Z.ravel()[~ok])
    return ys, zs, X.reshape(Y.shape), ok.reshape(Y.shape)


def torso_slope_mesh(m, xmap, axis_yz, ts=None, wall=2.7):
    """The torso shell with its skin sloped in toward both pitch axes (R as given, L the mirror): every vertex near the
    outer skin moves along x by D = blend(x_skin - x_dish) (vertices deeper than the wall fade out), then the dish ends at
    R_end (vertices inside it removed by a cylinder about the pitch axis). A shear along x: one-to-one for any D."""
    from scipy.interpolate import RegularGridInterpolator
    t = ts or TORSO_SLOPE
    ys, zs, X, _ = xmap
    xo_at = RegularGridInterpolator((ys, zs), X, bounds_error=False, fill_value=None)
    V = np.asarray(m.vertices, float).copy()
    e, (f0, f1) = t["blend_mm"], TORSO_WALL_BLEND
    moved = np.zeros(len(V))
    for sx in (1.0, -1.0):
        side = sx * V[:, 0] > 0.0
        y, z, x = V[side, 1], V[side, 2], sx * V[side, 0]
        rho = np.hypot(y - axis_yz[0], z - axis_yz[1])
        xo = xo_at(np.c_[y, z])
        u = xo - dish_x(rho, t)
        D = np.where(u <= -e, 0.0, np.where(u >= e, u, (u + e) ** 2 / (4.0 * e)))
        depth = xo - x                                              # below the outer skin
        w = np.clip(1.0 - (depth - (wall + f0)) / (f1 - f0), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)
        V[side, 0] -= sx * D * w
        moved[side] = D * w
    out = trimesh.Trimesh(V, np.asarray(m.faces), process=False)
    for sx in (1.0, -1.0):
        cyl = trimesh.creation.cylinder(radius=t["R_end"], segment=[[sx * 40.0, *axis_yz], [sx * 110.0, *axis_yz]], sections=180)
        out = _tm(_mf(out) - _mf(cyl))
    return out, float(moved.max())


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
    bumps, mb, Q, need, hit = fit_bumps(body, design, cv, cp, verbose)
    dev = skin_deviation(P, N, cv, cp, bumps)
    return cv, cp, bumps, dict(rms_normal_deviation_base_mm=round(float(np.sqrt(r.fun)), 3),
                               rms_normal_deviation_with_bumps_mm=round(float(np.sqrt(np.mean(dev ** 2))), 3),
                               worst_requirement_margin_mm=round(float(mb.min()), 3), samples=int(len(P)),
                               requirement_points=int(len(Q)), deepest_need_mm=round(float(need.max()), 2))


def fit_bumps(body, design, cv, cp, verbose=True):
    """The enclosure bumps (A, z, th each; half-widths BUMP_W) of the least squared height that hold the requirement (green
    U + GAP + WALL + GROW_MARGIN, radial) on the base skin cv / cp."""
    from scipy.optimize import minimize
    Q = requirement(body, design)
    thq, rhoq = _polar_cov(Q, cv, cp)
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
    return bumps, mb, Q, need, hit


def turning_group(body, s="R_"):
    """Every part on the pitch link (the blue U, its sleeves, the roll horn and idler with their screws, the pitch horn):
    relative to the cover they turn about the roll axis."""
    return [_side(n, s) for n, p in body.parts.items() if p["owning_link"] == "R_shoulder_pitch_link"]


def blue_roll_sweep(body, step=1.0, n=25000, names=None):
    """The turning group at every shoulder roll angle (step deg) in the roll-link neutral frame, as one point cloud."""
    ax = _roll_axis_xz(body)
    B = _thin(_pts(body, names or turning_group(body, "R_"), n=n), 0.35)
    B = B[roll_polar(B, ax)[1] < 60.0]                         # the part near the shoulder (not the pitch servo's far idler)
    lo, hi = body.joints["R_shoulder_roll"]["range_deg"]
    out = [trimesh.transform_points(B, np.linalg.inv(body.link_T("R_shoulder_roll_link", {"R_shoulder_roll": float(q)})))
           for q in np.arange(lo, hi + 1e-6, step)]
    return np.vstack(out)


def enclosed_static(body, design):
    """Roll-link points the cover keeps inside with GAP + WALL: the green U set (requirement) and the roll servo case."""
    return np.vstack([requirement(body, design), _pts(body, ["upper__R_roll_C047_case"], n=30000)])


def _thin(P, cell=0.8):
    """One point per cell (the first), for the fits."""
    k = np.floor(P / cell).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[np.sort(i)]


def _polar_cov(P, cov, cap):
    cx, cy, _, _ = ring(P[:, 2], cov, cap)
    dx, dy = P[:, 0] - cx, P[:, 1] - cy
    return np.arctan2(dy, dx), np.hypot(dx, dy)


def upper_stations(body, sweep):
    """The cover's upper stations by rule (owner: converge on the rotation point): the dome centred on the shoulder roll
    axis at its height (cx, cy at the top = the axis and the walls' middle; z0 = the axis' z); its outboard half-width = the
    largest distance from the roll axis of a turning part between psi_out and 90 deg (the arm ends round the horn, past
    the plate's reach) + RUN_CLEAR + WALL, rounded up to 0.5; the crest TOP["crest"] above the axis; the walls TOP["ear_over"]
    beyond the opening's y-edges (the turning parts' y extent + RUN_CLEAR + RIM_MARGIN + WALL); the inboard reach EAR_X_IN;
    the 352 station halfway between 338 (today's fit) and the top. Returns (cov, cap, info)."""
    ax = _roll_axis_xz(body)
    psi, rho = roll_polar(sweep, ax)
    quad = (psi >= TOP["psi_out"]) & (psi <= 90.0) & (rho < 30.0)   # the upper outboard quadrant past the plate's reach
    r_out = float(np.ceil((rho[quad].max() + RUN_CLEAR + WALL) * 2.0) / 2.0)
    near = sweep[rho < 30.0]
    y_lo = float(near[:, 1].min()) - RUN_CLEAR - RIM_MARGIN - WALL - TOP["ear_over"]
    y_hi = float(near[:, 1].max()) + RUN_CLEAR + RIM_MARGIN + WALL + TOP["ear_over"]
    cy = round(0.5 * (y_lo + y_hi), 3); b = round(0.5 * (y_hi - y_lo), 3)
    top = dict(cx=float(ax[0]), cy=cy, a_in=round(float(ax[0]) - EAR_X_IN, 3), a_out=r_out, b_front=b, b_back=b)
    cov = {}
    for k, z in COVER.items():
        zz, vv = z[0], list(z[1])
        if k in top:
            i338, i352, i366 = zz.index(338.0), zz.index(352.0), zz.index(366.0)
            vv[i366] = top[k]; vv[i352] = round(0.5 * (vv[i338] + top[k]), 3)
        if k in ("n_front", "n_back"):
            vv[-1] = TOP["n"]
        cov[k] = (zz, tuple(vv))
    cap = dict(z0=float(ax[1]), z_top=float(ax[1]) + TOP["crest"], p=TOP["p"], p_y=TOP["p_y"])
    return cov, cap, dict(r_out_mm=r_out, walls_y_mm=[round(y_lo, 2), round(y_hi, 2)])


def body_src_cover():
    return F.Body(json.loads(SC.MANIFEST.read_text())).mesh("R_upper_arm_hard_cover")


def fit_rim(body, skin_mesh, design, sweep, m_set=(4.0, 6.0, 8.0)):
    """Place the rim: rm = skin points within RUN_CLEAR + 0.3 of the swept turning group beyond NEAR_AXIS (they must be
    inside, RIM_MARGIN along the skin), keep = skin points within GAP + WALL + 0.5 of the green U set (outside, RIM_MARGIN);
    the least skin inside. Nelder-Mead on (psi_c, psi_h, y_c, y_h) per exponent m, from the box around rm."""
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    ax = _roll_axis_xz(body)
    S = np.vstack([skin_mesh.vertices, trimesh.sample.sample_surface(skin_mesh, 300000, seed=2)[0]])
    psi, rho = roll_polar(S, ax)
    dB = cKDTree(sweep).query(S, distance_upper_bound=RUN_CLEAR + 2.0)[0]
    dG = cKDTree(requirement(body, design)).query(S, distance_upper_bound=GAP + WALL + 2.0)[0]    # the green U set
    band = (rho > RIM["r0"]) & (rho < RIM["r1"])
    rm = band & (dB < RUN_CLEAR + 0.3) & (rho >= NEAR_AXIS)
    keep = band & (dG < GAP + WALL + 0.5)
    cand = band & ~keep
    sub = np.random.default_rng(5).choice(np.flatnonzero(band), min(60000, int(band.sum())), replace=False)
    dps = np.degrees(RIM_MARGIN / np.maximum(rho, 5.0))

    def uv(r, idx):
        u = np.abs(np.degrees(np.angle(np.exp(1j * np.radians(psi[idx] - r["psi_c"]))))) / r["psi_h"]
        return u, np.abs(S[idx, 1] - r["y_c"]) / r["y_h"], dps[idx] / r["psi_h"], RIM_MARGIN / r["y_h"]

    def margins(r):
        u, v, du, dv = uv(r, np.flatnonzero(rm))
        mr = 1.0 - ((u + du) ** r["m"] + (v + dv) ** r["m"])                       # >= 0: inside with margin
        u, v, du, dv = uv(r, np.flatnonzero(keep))
        mk = (np.maximum(u - du, 0) ** r["m"] + np.maximum(v - dv, 0) ** r["m"]) - 1.0   # >= 0: outside with margin
        return mr, mk
    P_rm = np.c_[psi[rm], S[rm, 1]]
    best = None
    for m in m_set:
        v0 = [float(np.mean(P_rm[:, 0].min() + P_rm[:, 0].max()) / 1.0) / 2.0, float(np.ptp(P_rm[:, 0])) / 2.0 + 8.0,
              float(P_rm[:, 1].min() + P_rm[:, 1].max()) / 2.0, float(np.ptp(P_rm[:, 1])) / 2.0 + 2.0]

        bnd = [(-140.0, 20.0), (40.0, 150.0), (-10.0, 20.0), (12.0, 40.0)]

        def obj(v):
            v = [min(max(x, lo), hi) for x, (lo, hi) in zip(v, bnd)]
            r = dict(RIM, psi_c=v[0], psi_h=v[1], y_c=v[2], y_h=v[3], m=m)
            mr, mk = margins(r)
            area = float(np.mean(rim_F(psi[sub], S[sub, 1], r) < 1.0))
            return area * 100.0 + 1e3 * float(np.mean(np.minimum(mr - 0.01, 0) ** 2)) * 100 + 1e3 * float(np.mean(np.minimum(mk - 0.01, 0) ** 2)) * 100
        v0 = [min(max(x, lo), hi) for x, (lo, hi) in zip(v0, bnd)]
        res = minimize(obj, np.array(v0), method="Powell", bounds=bnd, options=dict(maxiter=6000, xtol=0.01, ftol=1e-7))
        r = dict(RIM, psi_c=round(float(res.x[0]), 2), psi_h=round(float(res.x[1]), 2), y_c=round(float(res.x[2]), 2),
                 y_h=round(float(res.x[3]), 2), m=m)
        mr, mk = margins(r)
        area = float(np.mean(rim_F(psi[sub], S[sub, 1], r) < 1.0))
        bad = int((mr < 0).sum() + (mk < 0).sum())
        print(f"  rim m={m}: {r}  skin share inside {area:.4f}, rm out {int((mr < 0).sum())}, keep in {int((mk < 0).sum())}", flush=True)
        if best is None or (bad, area) < (best[0], best[1]):
            best = (bad, area, r, dict(rm_points=int(rm.sum()), keep_points=int(keep.sum()), rm_outside=int((mr < 0).sum()),
                                       keep_inside=int((mk < 0).sum()), skin_share_open=round(area, 4)))
    return best[2], best[3]


def arm_envelope(body, cover_R, step=2.5, rho_max=64.0):
    """Smallest x of every part of the right arm at each distance rho from the pitch axis (1 mm bins), over shoulder roll
    -15..95 (step) and shoulder yaw -60..60 (pitch turns about the axis itself and leaves x and rho unchanged)."""
    ay, az = _pitch_axis_yz(body)
    by = {}
    for n, p in body.parts.items():
        lk = p["owning_link"]
        if lk in ("R_shoulder_pitch_link", "R_shoulder_roll_link", "R_upper_arm") and n != "R_upper_arm_hard_cover":
            by.setdefault(lk, []).append(n)
    pts = {lk: _pts(body, ns, n=1500) for lk, ns in by.items()}
    pts["R_shoulder_roll_link"] = np.vstack([pts["R_shoulder_roll_link"], cover_R.vertices,
                                             trimesh.sample.sample_surface(cover_R, 60000, seed=3)[0]])
    env = np.full(int(rho_max) + 1, np.inf)

    def add(X):
        r = np.hypot(X[:, 1] - ay, X[:, 2] - az); k = r < rho_max
        np.minimum.at(env, np.floor(r[k]).astype(int), X[k, 0])
    add(pts["R_shoulder_pitch_link"])
    lo, hi = body.joints["R_shoulder_roll"]["range_deg"]
    for q in np.arange(lo, hi + 1e-6, step):
        add(trimesh.transform_points(pts["R_shoulder_roll_link"], body.link_T("R_shoulder_roll_link", {"R_shoulder_roll": q})))
        for qy in (-60.0, -30.0, 0.0, 30.0, 60.0):
            add(trimesh.transform_points(pts["R_upper_arm"], body.link_T("R_upper_arm", {"R_shoulder_roll": q, "R_shoulder_yaw": qy})))
    return env


def torso_internals_xmax(body, rho_max=64.0):
    """Largest x of the torso's own parts (no shells, no shell nuts) at each rho from the R pitch axis, x > 40."""
    ay, az = _pitch_axis_yz(body)
    names = [n for n, p in body.parts.items() if p["owning_link"] == "torso" and n not in TORSO_SHELLS and p.get("role") != "body_shell"
             and body.mesh(n).bounds[1][0] > 40.0]
    X = _pts(body, names, n=2000)
    r = np.hypot(X[:, 1] - ay, X[:, 2] - az); k = (r < rho_max) & (X[:, 0] > 40.0)
    out = np.full(int(rho_max) + 1, -np.inf)
    np.maximum.at(out, np.floor(r[k]).astype(int), X[k, 0])
    return out


def fit_torso_slope(body, env, xmap, ts=None, tint=None, wall=2.7, gap=0.5):
    """R_end: the old opening's largest radius + 1.5 (its wall and inner flange then lie inside the new round one). x_floor, R_floor,
    slope_deg: the shallowest dish (least skin moved, weighted by rho) with x_dish <= env - RUN_CLEAR from R_end out to
    TORSO_R_WIN, and its inner face (x_dish - wall) at least `gap` outboard of the torso's own parts (tint, the pitch servo);
    where both cannot hold, the clearance to the arm gives way (reported as the margins)."""
    t = dict(ts or TORSO_SLOPE)
    ay, az = _pitch_axis_yz(body)
    ys, zs, X, ok = xmap
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    R = np.hypot(Y - ay, Z - az)
    from scipy import ndimage
    hole = ndimage.binary_opening((~ok) & (R < 28.0), iterations=3)   # the old opening (no seam slits, no rays above)
    t["R_end"] = round(float(np.ceil((R[hole].max() + 1.5) * 4.0) / 4.0), 2) if hole.any() else t["R_end"]
    rr = np.arange(len(env)) + 0.5
    use = (rr >= t["R_end"] - 1.0) & (rr <= TORSO_R_WIN) & np.isfinite(env)
    lim = env[use] - RUN_CLEAR
    sk = (R >= t["R_end"]) & (R < TORSO_R_WIN + 15.0) & ok
    Rs, Xs = R[sk], X[sk]
    best = None
    for sd in np.arange(10.0, 60.01, 1.0):
        tn = np.tan(np.radians(sd))
        for Rf in np.arange(t["R_end"], 62.0, 0.5):
            xf = float(np.min(lim - tn * _ease(rr[use] - Rf, t["ease_mm"])))
            short = 0.0
            if tint is not None:
                lo_ = np.where(np.isfinite(tint[use]), tint[use] + wall + gap - tn * _ease(rr[use] - Rf, t["ease_mm"]), -np.inf)
                short = max(float(np.max(lo_)) - xf, 0.0)                # > 0: both limits cannot hold
                xf = max(xf, float(np.max(lo_)))
            cand = dict(t, x_floor=round(xf - 0.005, 2), R_floor=float(Rf), slope_deg=float(sd))
            cost = float(np.sum(np.maximum(Xs - dish_x(Rs, cand), 0.0) * Rs)) + 1e9 * short
            if best is None or cost < best[0]:
                best = (cost, cand)
    t = best[1]
    g_arm = env - dish_x(rr, t)
    out = dict(moved_area_weighted=round(best[0], 1), min_clearance_to_arm_envelope_mm=round(float(np.min(g_arm[use])), 3),
               at_rho_mm=float(rr[use][int(np.argmin(g_arm[use]))]))
    if tint is not None:
        g_in = dish_x(rr, t) - wall - tint
        out["min_inner_face_to_torso_parts_mm"] = round(float(np.min(np.where(use & np.isfinite(g_in), g_in, np.inf))), 3)
    return t, out


def fit_all():
    """fit-cover: prints COVER, CAP, BUMPS, RIM and TORSO_SLOPE to paste above (the build uses the values in the script).
    (1) the upper stations by rule (upper_stations; `fit-cover keep-upper` keeps the table's), (2) the enclosure
    bumps on that skin, (3) the rim on the skin with its wells, (4) the torso slope against the arm with that cover."""
    global RIM, BUMPS
    t0 = time.time()
    man = json.loads(SC.CANDIDATE.read_text()); body = F.Body(man)
    src = F.Body(json.loads(SC.MANIFEST.read_text()))
    design = mount_design(src.mesh("R_upper_arm_hard_cover"))
    sweep = blue_roll_sweep(body, step=1.0, n=12000)
    if "keep-upper" not in sys.argv:
        cov, cap, st = upper_stations(body, sweep)
        COVER.update(cov); CAP.update(cap)
        print("upper stations:", json.dumps(st), flush=True)
    BUMPS, mb, _, _, _ = fit_bumps(body, design, COVER, CAP)
    sh, outer = cover_skin()
    sh = add_wells(sh, outer, design)
    RIM, rst = fit_rim(body, _tm(sh), design, sweep, m_set=(8.0, 12.0, 16.0))
    print(f"rim {RIM} {json.dumps(rst)} ({time.time() - t0:.0f} s)", flush=True)
    cov_R = _tm(sh - _mf(rim_cutter(_roll_axis_xz(body))))
    env = arm_envelope(body, cov_R)
    xmap = skin_x_map([src.mesh(n) for n in TORSO_SHELLS])
    ts, tst = fit_torso_slope(body, env, xmap, tint=torso_internals_xmax(body))
    print("torso slope:", ts, json.dumps(tst), f"({time.time() - t0:.0f} s)", flush=True)
    print("arm envelope x by rho:", {i: round(float(v), 2) for i, v in enumerate(env) if np.isfinite(v) and i % 2 == 0})
    print("COVER = {")
    for k in ("cx", "cy") + RING:
        print(f'    "{k}": ({tuple(COVER[k][0])}, ({", ".join(f"{x:.3f}" for x in COVER[k][1])},)),')
    print("}")
    print(f"CAP = dict(z0={CAP['z0']}, z_top={CAP['z_top']}, p={CAP['p']}, p_y={CAP['p_y']})")
    print("BUMPS = (" + ", ".join(f"dict(A={b['A']}, z={b['z']}, w_z={b['w_z']}, th={b['th']}, w_th={b['w_th']})" for b in BUMPS) + ")")
    print("RIM = dict(" + ", ".join(f"{k}={v}" for k, v in RIM.items()) + ")")
    print("TORSO_SLOPE = dict(" + ", ".join(f"{k}={v}" for k, v in ts.items()) + ")")


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
               method=dict(wall_mm=WALL, gap_green_mm=GAP, running_clearance_mm=RUN_CLEAR, bottom_cut=dict(normal=YAW_AXIS.tolist(), s=BOTTOM_S),
                           cover=COVER, cap=CAP, bumps=BUMPS, rim=RIM, ear_x_in=EAR_X_IN, torso_slope=TORSO_SLOPE,
                           note="parametric skin (superellipse rings, one spline per ring value) converging on the roll servo, ending at "
                                "a superellipse rim in the roll axis' coordinates; torso skin sloped in toward the pitch axis (dish of "
                                "revolution), ending at R_end; no sweeps"),
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
    # 3. covers: the parametric skin with the wells, ending at the rim; the left cover is its exact mirror
    from scipy.spatial import cKDTree
    covers, torso = {}, {}
    ax = _roll_axis_xz(body)
    sh, outer = cover_skin()
    sh = add_wells(sh, outer, design)
    pre = _tm(sh)
    cov_R = _tm(sh - _mf(rim_cutter(ax)))
    cov_R = trimesh.util.concatenate([q for q in cov_R.split(only_watertight=False) if abs(q.volume) > 20.0])
    c_src = src.mesh("R_upper_arm_hard_cover")
    Ps, Ns = today_samples(c_src, design)
    dev = skin_deviation(Ps, Ns)
    kept = ~rim_open(Ps, ax) & (Ps @ YAW_AXIS >= BOTTOM_S + 0.5)
    G = requirement(body, design)
    Sc = np.vstack([cov_R.vertices, trimesh.sample.sample_surface(cov_R, 800000, seed=4)[0]])   # ~0.1 mm spacing
    Gs = G[::3]
    dG = cKDTree(Sc).query(Gs)[0] * np.where(cov_R.contains(Gs), -1.0, 1.0)      # < 0: inside the cover material
    inside = cover_outer().contains(Gs); opened = rim_open(Gs, ax)
    case = _pts(body, ["upper__R_roll_C047_case"], n=40000)
    dC = cKDTree(Sc).query(case)[0]
    sweep = blue_roll_sweep(body, n=12000)
    S_ = Sc[::4]
    dB = cKDTree(sweep).query(S_)[0]
    del sweep
    psi_s, rho_s = roll_polar(S_, ax)
    edge = (np.abs(rim_F(psi_s, S_[:, 1]) - 1.0) < 0.08) & (rho_s < RIM["r1"]) & (rho_s > RIM["r0"])

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
               under_the_opening_pct=round(100.0 * opened.mean(), 2), roll_case_min_distance_to_cover_mm=round(float(dC.min()), 3),
               note="green U + sleeves + roll-case screws (bosses at the wells left out) to the cover's material")
    rimrep = dict(removed_by_rim_mm3=round(float(pre.volume - cov_R.volume), 1), skin_with_wells_mm3=round(float(pre.volume), 1),
                  turning_swept_min_distance_mm=round(float(dB.min()), 2), turning_swept_min_at_mm=np.round(S_[int(dB.argmin())], 1).tolist(),
                  cover_points_within_1p5mm_of_turning=int((dB < 1.5).sum()),
                  rim_edge_rho_from_roll_axis_mm=dict(min=round(float(rho_s[edge].min()), 2), p50=round(float(np.median(rho_s[edge])), 2),
                                                      max=round(float(rho_s[edge].max()), 2)) if edge.any() else None)
    print(f"cover: {cov_R.volume:.0f} mm3 (today {c_src.volume:.0f}); skin vs today {json.dumps(skin)}", flush=True)
    print(f"green enclosure {json.dumps(enc)}; rim {json.dumps(rimrep)} ({time.time() - t0:.0f} s)", flush=True)
    for s in ("R_", "L_"):
        cn = f"{s}upper_arm_hard_cover"
        out = cov_R if s == "R_" else _mirror(cov_R)
        covers[cn] = dict(volume_mm3=round(float(out.volume), 1), today_volume_mm3=round(float(src.mesh(cn).volume), 1),
                          watertight=bool(out.is_watertight), skin_vs_today=skin, green_enclosure=enc, rim=rimrep,
                          built="parametric R cover" if s == "R_" else "mirror of the R cover")
        _update_shell(P[cn], SP_[cn], src.mesh(cn), out, cn,
                      "parametric upper-arm cover (superellipse rings, splines in z, dome; constant wall) enclosing the green U and "
                      "converging on the roll servo, ending at a superellipse rim about the roll axis; screw wells at the green U mounts")
    # 4. torso shells: the skin sloped in toward each pitch axis, ending at R_end (the left side mirrors the right)
    sh_t = {n: src.mesh(n) for n in TORSO_SHELLS}
    xmap = skin_x_map(list(sh_t.values()))
    ayz = _pitch_axis_yz(body)
    inner_pts = []
    for n in TORSO_SHELLS:
        cur, dmax = torso_slope_mesh(sh_t[n], xmap, ayz)
        cur = trimesh.util.concatenate([q for q in cur.split(only_watertight=False) if abs(q.volume) > 20.0])
        torso[n] = dict(slope=dict(TORSO_SLOPE, axis_yz=ayz.tolist()), largest_move_mm=round(dmax, 2),
                        opening_area_per_side_mm2=round(float(np.pi * TORSO_SLOPE["R_end"] ** 2), 1),
                        watertight=bool(cur.is_watertight), **openings(sh_t[n], cur))
        _update_shell(P[n], SP_[n], sh_t[n], cur, n,
                      f"shoulder: skin sloped in toward each pitch axis (dish floor x {TORSO_SLOPE['x_floor']} to rho {TORSO_SLOPE['R_floor']}, "
                      f"{TORSO_SLOPE['slope_deg']} deg rise), ending at R {TORSO_SLOPE['R_end']} mm")
        inner_pts.append(trimesh.sample.sample_surface(cur, 400000, seed=6)[0])
        print(f"{n}: {torso[n]['before_mm3']} -> {torso[n]['after_mm3']} mm3, largest move {dmax:.2f} mm, watertight {cur.is_watertight}", flush=True)
    T_int = _pts(body, [n for n, p in body.parts.items() if p["owning_link"] == "torso" and n not in TORSO_SHELLS
                        and p.get("role") != "body_shell" and body.mesh(n).bounds[1][0] > 40.0], n=4000)
    Ti = T_int[np.hypot(T_int[:, 1] - ayz[0], T_int[:, 2] - ayz[1]) < 64.0]
    dT = cKDTree(np.vstack(inner_pts)).query(Ti)[0]
    torso["static_gap_to_torso_parts_near_shoulder_mm"] = dict(min=round(float(dT.min()), 3), at_mm=np.round(Ti[int(dT.argmin())], 1).tolist())
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


# before / after: the previous shoulder shells (the committed candidate, kept by hand as a copy for the comparison) when
# present, else the input body
BEFORE = ROOT / ".cache/work/shoulder-shells-before/assembly-parts-shoulder-u-shells.json"
OWNER_CAM = (np.array([-0.4637, 0.4637, -0.7547]), (0.0, 0.0, 1.0))   # view direction of the owner's screenshot: from the
                                                                        # front (-y) and outside (+x), 49 deg above
EXPO_LO, EXPO_HI = np.array([58.0, -45.0, 300.0]), np.array([165.0, 62.0, 402.0])   # right shoulder window (open-area measure)
EXPO_VIEWS = (("FRONT (-y)", np.array([0.0, 1.0, 0.0]), (0.0, 0.0, 1.0)), ("SIDE (+x)", np.array([-1.0, 0.0, 0.0]), (0.0, 0.0, 1.0)),
              ("OWNER CAMERA (front + outside, 49 deg above)",) + OWNER_CAM)
EXPO_LINKS = ("torso", "R_shoulder_pitch_link", "R_shoulder_roll_link", "R_upper_arm")


def _is_shell(n, p):
    """Shell outer skin for the open-area measure: the body shells and the cover's own screw heads in its wells."""
    return (p.get("role") == "body_shell" and not n.endswith("_nut")) or ("upper_arm_shell_" in n and "_M3x" in n)


def _near_scene(body, q, lo=EXPO_LO, hi=EXPO_HI, pad=30.0):
    """(name, posed mesh) of every part on the torso and the right shoulder / upper arm near the window at pose q."""
    out = []
    for n, p in body.parts.items():
        lk = p["owning_link"]
        if lk not in EXPO_LINKS:
            continue
        T = body.link_T(lk, q)
        b = body.mesh(n).bounds
        C = trimesh.transform_points(np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]]), T)
        if np.all(C.max(0) > lo - pad) and np.all(C.min(0) < hi + pad):
            m = body.mesh(n).copy(); m.apply_transform(T); out.append((n, m))
    return out


def _cast(ms, d, up, lo, hi, px):
    """First hit per pixel of an orthographic view of the box lo..hi: (part index or -1, hit points, face normals, grid)."""
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    u = np.cross(d, up); u /= np.linalg.norm(u); v = np.cross(u, d)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pu, pv = C @ u, C @ v
    W, H = int((pu.max() - pu.min()) / px), int((pv.max() - pv.min()) / px)
    GU, GV = np.meshgrid(pu.min() + (np.arange(W) + 0.5) * px, pv.max() - (np.arange(H) + 0.5) * px)
    c = 0.5 * (lo + hi); R = 0.5 * float(np.abs(d) @ (hi - lo)) + 60.0
    org = GU.reshape(-1, 1) * u + GV.reshape(-1, 1) * v + (float(c @ d) - R) * d
    cat = trimesh.util.concatenate(ms); fid = np.concatenate([np.full(len(m.faces), i) for i, m in enumerate(ms)])
    rx = trimesh.ray.ray_pyembree.RayMeshIntersector(cat)
    tri, ray, loc = rx.intersects_id(org, np.tile(d, (len(org), 1)), multiple_hits=False, return_locations=True)
    pid = -np.ones(H * W, int); pid[ray] = fid[tri]
    P = np.full((H * W, 3), np.nan); P[ray] = loc
    N = np.zeros((H * W, 3)); N[ray] = cat.face_normals[tri]
    # inside the window only (projected box of the window seen along d: points beyond it are not the shoulder)
    inwin = np.all((P >= lo - 1e-6) & (P <= hi + 1e-6), axis=1)
    pid[~inwin] = -1
    return pid.reshape(H, W), P.reshape(H, W, 3), N.reshape(H, W, 3), (pu.min(), pu.max(), pv.min(), pv.max()), (u, v, d)


def exposure(body, q, d, up, px=0.25):
    """Visible open area of the right shoulder from direction d at pose q: pixels whose first hit in the window is not a
    shell's outer skin (a part inside, or a shell's inner face seen through an opening). mm2 by class, the largest open gap
    (diameter of the largest circle inside the open region, mm), the pixel classes for the picture."""
    from scipy import ndimage
    sc = _near_scene(body, q)
    names = [n for n, _ in sc]; ms = [m for _, m in sc]
    pid, P, N, ext, frame = _cast(ms, d, up, EXPO_LO, EXPO_HI, px)
    shell = np.array([_is_shell(n, body.parts[n]) for n in names])
    hit = pid >= 0
    cls = np.zeros(pid.shape, int)                                   # 0 none, 1 shell outer, 2 shell inner, 3 turning, 4 roll servo, 5 other
    groups = {}
    for i, n in enumerate(names):
        g = "torso_shells" if n in TORSO_SHELLS else n
        groups.setdefault(g, []).append(i)
    # a shell hit is its outer skin when the ray along its normal leaves the shell's own group without meeting it again
    sh_px = hit & shell[np.maximum(pid, 0)]
    for g, idx in groups.items():
        if not shell[idx[0]]:
            continue
        sel = sh_px & np.isin(pid, idx)
        if not sel.any():
            continue
        grp = trimesh.util.concatenate([ms[i] for i in idx])
        rx = trimesh.ray.ray_pyembree.RayMeshIntersector(grp)
        o = P[sel] + N[sel] * 0.05
        again = rx.intersects_any(o, N[sel])
        c2 = np.where(again, 2, 1); cls[sel] = c2
    turn = {i for i, n in enumerate(names) if body.parts[n]["owning_link"] == "R_shoulder_pitch_link"}
    roll = {i for i, n in enumerate(names) if n.startswith("upper__R_roll_C047")}
    ns = hit & ~shell[np.maximum(pid, 0)]
    cls[ns & np.isin(pid, list(turn))] = 3
    cls[ns & np.isin(pid, list(roll))] = 4
    cls[ns & (cls == 0)] = 5
    openm = cls >= 2
    a = px * px
    edt = ndimage.distance_transform_edt(openm) * px
    lab, nl = ndimage.label(openm)
    big = float(np.bincount(lab.ravel())[1:].max() * a) if nl else 0.0
    out = dict(open_mm2=round(float(openm.sum() * a), 0), shell_inner_faces_mm2=round(float((cls == 2).sum() * a), 0),
               turning_parts_mm2=round(float((cls == 3).sum() * a), 0), roll_servo_mm2=round(float((cls == 4).sum() * a), 0),
               other_parts_mm2=round(float((cls == 5).sum() * a), 0), largest_open_region_mm2=round(big, 0),
               largest_open_gap_mm=round(float(2.0 * edt.max()), 2), regions=int(nl))
    return out, cls, ext


EXPO_COL = {0: (1.0, 1.0, 1.0), 1: (0.86, 0.86, 0.88), 2: (0.35, 0.33, 0.33), 3: SC.COL["blue"], 4: SC.COL["case"], 5: (0.55, 0.57, 0.60)}


def _expo_image(cls):
    img = np.array([EXPO_COL[k] for k in range(6)])[cls]
    edge = np.zeros(cls.shape, bool)
    o = cls >= 2
    edge[1:] |= o[1:] != o[:-1]; edge[:-1] |= o[1:] != o[:-1]; edge[:, 1:] |= o[:, 1:] != o[:, :-1]; edge[:, :-1] |= o[:, 1:] != o[:, :-1]
    img[edge] = (0.9, 0.05, 0.05)
    return img


def views():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    man = json.loads(CANDIDATE.read_text()); body = F.Body(man)
    old = json.loads((BEFORE if BEFORE.exists() else SC.CANDIDATE).read_text()); ob = F.Body(old)
    rows = _scene_rows(man)
    legend = [("torso shells", LIGHT["shell"]), ("upper-arm cover", LIGHT["cover"]), ("blue U (pitch link)", SC.COL["blue"]),
              ("green U (roll link)", SC.COL["green"]), ("roll servo case", SC.COL["case"]), ("horns", SC.COL["horn"]),
              ("roll idler", SC.COL["idler"]), ("yaw servo case", SC.COL["pitchcase"]), ("sleeves / screws", SC.COL["sleeve"]),
              ("cover-mount nuts", LIGHT["nut"])]
    lo, hi = np.array([62.0, -52.0, 296.0]), np.array([166.0, 60.0, 396.0])
    VIEWS.mkdir(parents=True, exist_ok=True)
    v4 = list(SC.VIEWS4[:3]) + [("OWNER CAMERA (front + outside, 49 deg above)", tuple(OWNER_CAM[0]), OWNER_CAM[1], None, None)]
    # (a) before / after, 4 views + the open-area measure
    expo = {}
    fig, axs = plt.subplots(2, 4, figsize=(28, 14.5), dpi=80)
    for r, (lab, b, rr) in enumerate((("BEFORE", ob, _scene_rows(old)), ("AFTER", body, rows))):
        for c, (t, d, up, xl, yl) in enumerate(v4):
            img, ext = SC._render(_posed(b, rr, {}), d, up, lo, hi, px_mm=0.16)
            ax = axs[r, c]; ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"{lab}: {t}", fontsize=12)
            if xl:
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.tick_params(labelsize=8)
            else:
                ax.axis("off")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle("Right shoulder at standing, before / after: upper-arm cover converging on the roll servo and ending at a rim about the roll "
                 "axis; torso skin sloped in toward the pitch axis, opening R %.1f mm" % TORSO_SLOPE["R_end"], fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); f1 = VIEWS / "shoulder-shells-before-after.png"; fig.savefig(f1, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f1)}", flush=True)
    # (b) open area from outside at standing: before / after x front / side / owner camera
    fig, axs = plt.subplots(2, 3, figsize=(27, 17), dpi=80)
    for r, (lab, b) in enumerate((("BEFORE", ob), ("AFTER", body))):
        for c, (t, d, up) in enumerate(EXPO_VIEWS):
            st, cls, ext = exposure(b, {}, d, up)
            expo[f"{lab.lower()} / {t}"] = st
            ax = axs[r, c]; ax.imshow(_expo_image(cls), extent=ext, interpolation="nearest"); ax.axis("off")
            ax.set_title(f"{lab}: {t}\nopen {st['open_mm2']:.0f} mm2 (blue U etc. {st['turning_parts_mm2']:.0f}, roll servo "
                         f"{st['roll_servo_mm2']:.0f}, shell insides {st['shell_inner_faces_mm2']:.0f}, other {st['other_parts_mm2']:.0f}); "
                         f"largest gap {st['largest_open_gap_mm']:.1f} mm", fontsize=11)
    lg = [("shell outer skin", EXPO_COL[1]), ("shell inner face (through an opening)", EXPO_COL[2]), ("pitch-link parts (blue U, horns)", EXPO_COL[3]),
          ("roll servo", EXPO_COL[4]), ("other inner parts", EXPO_COL[5]), ("edge of the open area", (0.9, 0.05, 0.05))]
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in lg], loc="lower center", ncol=6, fontsize=12, frameon=False)
    fig.suptitle("Visible open area at the right shoulder, standing (window x 58..165, y -45..62, z 300..402): every pixel whose first hit is not "
                 "a shell's outer skin", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); f4 = VIEWS / "shoulder-shells-open-area.png"; fig.savefig(f4, facecolor="white"); plt.close(fig)
    (VIEWS / "shoulder-shells-open-area.json").write_text(json.dumps(expo, indent=1) + "\n")
    print(f"wrote {SC._rel(f4)}", json.dumps(expo), flush=True)
    # (c) the owner's camera alone, before / after, opaque
    fig, axs = plt.subplots(1, 2, figsize=(26, 13), dpi=80)
    lo5, hi5 = np.array([20.0, -70.0, 250.0]), np.array([175.0, 75.0, 405.0])
    for c, (lab, b, rr) in enumerate((("BEFORE", ob, _scene_rows(old) + _torso_rows(old)), ("AFTER", body, rows + _torso_rows(man)))):
        img, ext = SC._render(_posed(b, rr, {}), OWNER_CAM[0], OWNER_CAM[1], lo5, hi5, px_mm=0.14)
        axs[c].imshow(img, extent=ext, interpolation="bilinear"); axs[c].axis("off"); axs[c].set_title(f"{lab}: owner camera (front + outside, 49 deg above), standing", fontsize=14)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); f5 = VIEWS / "shoulder-shells-owner-camera.png"; fig.savefig(f5, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f5)}", flush=True)
    # (d) range poses
    poses = [("pitch -60", {"R_shoulder_pitch": -60.0}), ("pitch +150", {"R_shoulder_pitch": 150.0}),
             ("roll -15", {"R_shoulder_roll": -15.0}), ("roll +95", {"R_shoulder_roll": 95.0}),
             ("pitch +150, roll +95", {"R_shoulder_pitch": 150.0, "R_shoulder_roll": 95.0})]
    vv = [("owner camera", tuple(OWNER_CAM[0]), OWNER_CAM[1]), ("FRONT (-y)", (0, 1, 0), (0, 0, 1)), ("SIDE (+x)", (-1, 0, 0), (0, 0, 1))]
    lo2, hi2 = np.array([45.0, -80.0, 265.0]), np.array([200.0, 85.0, 440.0])
    allrows = rows + _torso_rows(man)
    fig, axs = plt.subplots(len(vv), len(poses), figsize=(34, 21), dpi=80)
    for c, (pl, q) in enumerate(poses):
        sc = _posed(body, allrows, q)
        for r, (t, d, up) in enumerate(vv):
            img, ext = SC._render(sc, d, up, lo2, hi2, px_mm=0.2)
            ax = axs[r, c]; ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"{pl}: {t}", fontsize=12); ax.axis("off")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle(f"Right shoulder at the range ends (after): the cover and the torso slope fitted for >= {RUN_CLEAR:g} mm running clearance "
                 "(roll -15..95, pitch -60..150)", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); f2 = VIEWS / "shoulder-shells-range-poses.png"; fig.savefig(f2, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f2)}")


def _torso_rows(man):
    """Torso-link parts near the right shoulder that show through its openings (pitch servo, chassis), mid grey."""
    names = {p["name"]: p for p in man["parts"]}
    keep = ("upper__R_pitch_C047_case", "upper__R_pitch_C047_rear_idler", "combined__stage1_center_torso_spine_chassis",
            "combined__stage3_front_power_electronics_carrier")
    return [(n, SC.COL["pitchcase"]) for n in keep if n in names and names[n].get("mesh")]


def mount_views():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    man = json.loads(CANDIDATE.read_text()); body = F.Body(man)
    rows = _scene_rows(man)
    legend = [("torso shells", LIGHT["shell"]), ("upper-arm cover", LIGHT["cover"]), ("blue U (pitch link)", SC.COL["blue"]),
              ("green U (roll link)", SC.COL["green"]), ("roll servo case", SC.COL["case"]), ("horns", SC.COL["horn"]),
              ("roll idler", SC.COL["idler"]), ("yaw servo case", SC.COL["pitchcase"]), ("sleeves / screws", SC.COL["sleeve"]),
              ("cover-mount nuts", LIGHT["nut"])]
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
    {"fit-cover": fit_all, "apply": apply, "views": views, "mount-views": mount_views}[sys.argv[1] if len(sys.argv) > 1 else "apply"]()
