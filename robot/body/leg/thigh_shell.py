"""Blue thigh shell for the roll-flip hip (owner D-12 / D-13 step 2; Idle T-czv2d8 / T-fs5v6d).

One shell per leg on R_roll_link, in two halves (front / back, split at y 8 like today's covers), bolted to the four
Stage 4 posts. It encloses the hanging roll case and the new Stage 4 and covers the roll and knee joints; only the yellow
pitch-to-roll arms enter it from above.

Built from today's thigh hard covers (two halves split at y 8, PETG / PC, 2.1 mm wall, M3 into captive nuts in the
Stage 4 posts), re-fitted:
  - outer skin (owner 2026-11-17/19): superellipse rings, round from the knee to mid-thigh and closer to the parts around
    the roll case and Stage 4 at the top; each of the seven ring parameters is one natural cubic spline in z through 1-5
    control values (SURF, 25 values in all), fitted once as the surface closest to the 2026-11-18 skin that holds what the
    shell covers (Stage 4, the turned roll case, knee motor, the yellow arm hubs) + GAP + WALL and never widens toward the
    knee below mid-thigh (fit-surface re-fits it when a covered part moves); constant wall;
  - edges (owner 2026-11-17/18/20): the top edge is a low, calm periodic height around the shell (3 Fourier harmonics,
    peaks capped above its mean, dips capped below it, slope along the edge limited; edge_cut), below the moving
    neighbours: the green connector / hip-pitch motor rim region about the roll axis and the yellow pitch-to-roll part
    swept over ROLL_RANGE (fabric covers what is above it); the knee edge clears every lower-leg part swept
    over KNEE_RANGE about the knee axis (the shin hard covers slope in to clear it); the knee edge is one tilted plane, at
    the front just above the knee hub, and below the Stage 4 bracket the back slopes in to hug the knee motor;
  - holes with 90-deg countersinks (flat-head M3, flush with the outer skin, or sunk below it in a counterbore where no
    standard length gives a full nut with a tip margin; post_fit) at the Stage 4 posts, bosses from the post tips to the
    wall, a steel sleeve in each.
shin-cover cuts the lower leg's back hard cover clear of all this over KNEE_RANGE and re-mounts it (its old top M3s are
cut away): four PA2x8 per leg through bosses on its side wings into pilot holes drilled in the shin frame's STEP.
Nothing in the build or the check uses the core torso shell (cut at z 271 and redesigned next, D-12).
Mesh solid (manifold3d), mm, R_roll_link neutral frame; the left shell mirrors x.

  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/leg/thigh_shell.py build
  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/leg/thigh_shell.py shin-cover
  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/leg/thigh_shell.py check
  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/leg/thigh_shell.py fit-surface

Working on a copy (another agent owns the shared body): THIGH_SHELL_MANIFEST=<copy.json> reads and writes that manifest,
THIGH_SHELL_OUT=<folder> writes the part files, thigh_shell_edges.npz and (in its parent) report.json there. Both default to
today's paths (assembly-parts-roll-flip.json, artifacts/robot/body/leg/roll-flip/parts). shin-cover's screw proxies still
go to the shared parts/moved folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import roll_flip_checks as R  # noqa: E402  (kit, closed(), library poses, manifest paths)
import build_upper_hip_connectors as B  # noqa: E402

kit = R.kit
OUT = R.OUT / "parts"
if os.environ.get("THIGH_SHELL_MANIFEST"):
    R.VARIANT = Path(os.environ["THIGH_SHELL_MANIFEST"]).resolve()
if os.environ.get("THIGH_SHELL_OUT"):
    OUT = Path(os.environ["THIGH_SHELL_OUT"]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    R.OUT, R.REPORT = OUT.parent, OUT.parent / "report.json"     # R.merge writes the report there
GAP, WALL = 1.5, 2.1              # inner clearance to covered parts, wall thickness (today's covers: 2.1)
Z0, TOP_Z = 116.0, 206.5          # tube from just above the knee case's bottom (z 113) to the top; edges cut by edge_cut()
WINDOW = 5.0                      # each station hulls the sections within +- WINDOW mm (no waviness)
ROLL_RANGE, CLEAR = (-45.0, 25.0), 1.5
DENSITY = 1.27e-3                 # g/mm3, PETG / PC hard cover (as today's covers)
NTH = 144
CENTRE = np.array([55.0, 5.0])
COVERED = ("R_thigh_roll_flip", "R_roll_servo_case_reference", "R_knee_ST3025_case",
           "R_roll_upper_horn_reference", "R_roll_passive_idler_reference")
# (the knee output horn is on R_shin and ends at z 132.4, below Z0: it is not covered; hulling it made a lip at the knee end)
TODAY = ("R_thigh_front_hard_cover", "R_thigh_back_hard_cover")
NAMES = ("R_thigh_shell_front", "R_thigh_shell_back")


def sec_hull(body, names, z, extra=()):
    from shapely.geometry import MultiPoint
    pts = []
    for n in names:
        s = body.mesh(n).section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if s is not None:
            pts += [d[:, :2] for d in s.discrete]
    pts += list(extra)
    return MultiPoint(np.vstack(pts)).convex_hull if pts else None


def radial(poly, th):
    """Distance from CENTRE to a convex contour along each angle."""
    from shapely.geometry import LineString
    r = np.empty(len(th))
    for i, t in enumerate(th):
        far = CENTRE + 200.0 * np.array([np.cos(t), np.sin(t)])
        x = poly.exterior.intersection(LineString([CENTRE, far]))
        pts = np.array([[g.x, g.y] for g in getattr(x, "geoms", [x])]) if not x.is_empty else CENTRE[None]
        r[i] = np.linalg.norm(pts - CENTRE, axis=1).max()
    return r


def arm_hub_points(z):
    """Section at z of the yellow arm hubs (turning in place about the roll axis inside the shell)."""
    dz = z - B.RF_AZ
    if abs(dz) >= B.RF_ROLL_HUB_R:
        return []
    hw = np.sqrt(B.RF_ROLL_HUB_R ** 2 - dz ** 2)
    ys = (B.RF_ARM_F[0] - B.RF_ARM_THICKEN, B.RF_ARM_F[2], B.RF_ARM_R[0], B.RF_ARM_R[2] + B.RF_ARM_THICKEN)
    return [np.array([[B.RF_AX - hw, y], [B.RF_AX + hw, y]]) for y in ys]


# Outer surface (owner 2026-11-19: "simple smooth parametric courses, the less parameters the better"; keep the form of
# 2026-11-18, remove its small ridges and valleys, the outward flare at the knee end and the ridge across the back).
# Every ring is a two-depth superellipse |dx/a|^e + |dy/b|^e = 1 about (cx, cy): front half (y < cy) depth b_front and
# exponent n, back half depth b_back and exponent n_back; with e > 2 both halves meet at the sides with zero curvature (no
# crease). Each of the seven ring parameters is ONE natural cubic spline in z through a few control values at fixed heights
# (SURF: 2 values = a straight line, 1 = a constant), C2 along z, so the skin has continuous curvature. The control values
# were fitted (fit-surface) as the surface closest to the 2026-11-18 skin that holds every covered part + GAP + WALL at every
# 1-mm station between the edges, stays inside the top free space below the top edge (green rim region, yellow sweep; the
# top edge keeps its height), never widens toward the knee below SURF_Z_MID (extents x min / max, y min / max), keeps the
# front round from mid-thigh down (n <= 2.6) and the exponents within 2.05..3.6 (front) / 4.2 (back). 25 values in all
# (2026-11-18: about 165 per-station values plus 17 shaping constants).
SURF_Z = (133.0, 207.0)            # the splines run over this height (constant beyond; the edges cut at z 136..206.5)
SURF = {                           # parameter: (heights, control values), mm (exponents unitless)
    "cx": ((133.0, 207.0), (51.08, 54.289)),
    "cy": ((133.0,), (4.5,)),
    "a": ((133.0, 160.0, 178.0, 192.0, 207.0), (26.839, 29.897, 30.855, 19.971, 15.167)),
    "b_front": ((133.0, 160.0, 192.0, 207.0), (25.787, 34.306, 31.85, 25.875)),
    "b_back": ((133.0, 160.0, 178.0, 192.0, 207.0), (18.624, 34.293, 35.315, 35.533, 32.609)),
    "n": ((133.0, 160.0, 192.0, 207.0), (2.508, 2.533, 2.465, 3.6)),
    "n_back": ((133.0, 160.0, 192.0, 207.0), (4.2, 3.785, 2.847, 4.2)),
}
RING = ("cx", "cy", "a", "b_front", "b_back", "n", "n_back")
SURF_Z_MID = 172.0                 # below it the ring extents never grow toward the knee (no flare)
SURF_N = dict(min=2.05, round_max=2.6, front_max=3.6, back_max=4.2, cy=(4.5, 6.7))
DZ_ST = 1.0                        # ring spacing (and requirement stations) along z


def station_hull(body, z):
    """Convex hull of the covered parts' sections within +-WINDOW of z, plus the yellow arm hubs' outer faces over the whole
    roll-joint height (front and back walls run straight past the arms)."""
    from shapely.ops import unary_union
    zc = min(z, 199.5)
    samples = [zz for zz in np.linspace(zc - WINDOW, zc + WINDOW, 5) if Z0 - 4.0 <= zz <= 199.5]
    extra = [p for zz in samples for p in arm_hub_points(zz)]
    y_f = B.RF_ARM_F[0] - B.RF_ARM_THICKEN; y_r = B.RF_ARM_R[2] + B.RF_ARM_THICKEN
    if z >= 160.0:
        extra.append(np.array([[B.RF_AX - 6.0, y_f], [B.RF_AX + 6.0, y_f], [B.RF_AX - 6.0, y_r], [B.RF_AX + 6.0, y_r]]))
    return unary_union([g for zz in samples for g in [sec_hull(body, COVERED, zz, extra=extra)] if g is not None]).convex_hull


def _spline_basis(zk, z, nu=0):
    """Matrix B with B @ values = the spline through (zk, values) at z (derivative nu): natural cubic for 3+ heights,
    straight line for 2, constant for 1; clipped to SURF_Z."""
    from scipy.interpolate import CubicSpline
    z = np.clip(np.asarray(z, float), *SURF_Z); k = len(zk)
    if k == 1:
        return np.full((len(z), 1), 1.0 if nu == 0 else 0.0)
    if k == 2:
        t = (z - zk[0]) / (zk[1] - zk[0])
        return np.c_[1 - t, t] if nu == 0 else (np.tile([-1.0, 1.0], (len(z), 1)) / (zk[1] - zk[0]) if nu == 1 else np.zeros((len(z), 2)))
    return np.stack([CubicSpline(zk, np.eye(k)[i], bc_type="natural")(z, nu) for i in range(k)], -1)


def surface(z, surf=None, nu=0):
    """Ring parameters (cx, cy, a, b_front, b_back, n, n_back) of the outer skin at heights z (rows)."""
    surf = surf or SURF
    return np.stack([_spline_basis(surf[k][0], z, nu) @ np.asarray(surf[k][1], float) for k in RING], -1)


def ring_params():
    zs = np.arange(Z0 - 2.0, TOP_Z + 4.1, DZ_ST)
    return zs, surface(zs)


def level(prm, Q):
    """Superellipse level of points Q (x, y) under ring parameters prm (one row per point): <= 1 inside."""
    cx, cy, a, bf, bk, n, nk = prm.T
    dx, dy = Q[:, 0] - cx, Q[:, 1] - cy
    back = dy > 0; b = np.where(back, bk, bf); e = np.where(back, nk, n)
    return (np.abs(dx / a) ** e + np.abs(dy / b) ** e) ** (1 / e)


def requirement(body, edges=None):
    """Points (x, y, z) the outer skin must hold: per 1-mm station the hull of the covered parts (+-WINDOW) grown by
    GAP + WALL; only between the edges (knee edge - 3 mm .. top edge + 2 mm) when their heights per angle are given."""
    zf = np.arange(Z0 - 2.0, TOP_Z + 4.1, DZ_ST)
    Q = np.vstack([np.c_[p[:, :2], np.full(len(p), z)] for z in zf
                   for p in [np.array(station_hull(body, z).buffer(GAP + WALL, join_style=1, resolution=24).exterior.coords)]])
    if edges is not None:
        t = np.arctan2(Q[:, 1] - CENTRE[1], Q[:, 0] - CENTRE[0])
        Q = Q[(Q[:, 2] >= np.interp(t, edges["th"], edges["zb"], period=2 * np.pi) - 3.0)
              & (Q[:, 2] <= np.interp(t, edges["th"], edges["zt"], period=2 * np.pi) + 2.0)]
    return Q


def containment(Q, surf=None):
    """Per requirement point: level and radial margin (mm, from CENTRE along the point's angle; >= 0 holds GAP + WALL)."""
    P = surface(Q[:, 2], surf)
    cx, cy, a, bf, bk, n, nk = P.T
    d = Q[:, :2] - CENTRE; rq = np.hypot(d[:, 0], d[:, 1]); u = d / rq[:, None]
    lo, hi = np.zeros(len(Q)), np.full(len(Q), 200.0)
    for _ in range(50):
        t = (lo + hi) / 2; px, py = CENTRE[0] + t * u[:, 0], CENTRE[1] + t * u[:, 1]
        back = py > cy; b = np.where(back, bk, bf); e = np.where(back, nk, n)
        f = np.abs((px - cx) / a) ** e + np.abs((py - cy) / b) ** e
        lo, hi = np.where(f < 1, t, lo), np.where(f < 1, hi, t)
    return level(P, Q), (lo + hi) / 2 - rq


def se_radial(prm, th):
    """Distance from CENTRE to the superellipse prm = (cx, cy, a, b_front, b_back, n, n_back) along each angle (bisection)."""
    cx, cy, a, bf, bk, n, nk = prm
    d = np.c_[np.cos(th), np.sin(th)]
    lo, hi = np.zeros(len(th)), np.full(len(th), 200.0)
    for _ in range(50):
        t = (lo + hi) / 2; p = CENTRE + t[:, None] * d
        back = p[:, 1] > cy; b = np.where(back, bk, bf); e = np.where(back, nk, n)
        f = np.abs((p[:, 0] - cx) / a) ** e + np.abs((p[:, 1] - cy) / b) ** e
        lo, hi = np.where(f < 1, t, lo), np.where(f < 1, hi, t)
    return (lo + hi) / 2


def _radial_grid(P, th):
    """se_radial for many rings at once: P (m, 7), angles th -> (m, len(th))."""
    cx, cy, a, bf, bk, n, nk = [P[:, i:i + 1] for i in range(7)]
    lo, hi = np.zeros((len(P), len(th))), np.full((len(P), len(th)), 200.0)
    for _ in range(26):
        t = (lo + hi) / 2
        px, py = CENTRE[0] + t * np.cos(th), CENTRE[1] + t * np.sin(th)
        back = py > cy; b = np.where(back, bk, bf); e = np.where(back, nk, n)
        f = np.abs((px - cx) / a) ** e + np.abs((py - cy) / b) ** e
        lo, hi = np.where(f < 1, t, lo), np.where(f < 1, hi, t)
    return (lo + hi) / 2


def top_free_limit(body, th, zz, r_skin, z_top):
    """Per (z, angle) the smallest radius about CENTRE of the top free space (green rim region + yellow sweep, as edge_cut)
    at or outside the skin's inner wall (r_skin - WALL - 1), below the top edge z_top(angle); inf where none. The skin has
    to stay inside it so that the top edge keeps its height."""
    import trimesh
    clip = kit.mf(trimesh.creation.box(bounds=[[-100, -150, B.RF_AZ - B.RF_ROLL_HUB_R], [250, 150, 400]]))
    free = (green_rim(body)[0] ^ clip) + yellow_sweep(body)
    m = kit.tm(free ^ kit.mf(trimesh.creation.box(bounds=[[-10, -50, zz[0] - 1], [120, 60, zz[-1] + 1]])))
    P = np.vstack([m.vertices, trimesh.sample.sample_surface(m, 600000, seed=3)[0]])
    d = P[:, :2] - CENTRE; r = np.hypot(d[:, 0], d[:, 1])
    i = np.round((np.arctan2(d[:, 1], d[:, 0]) - th[0]) / (th[1] - th[0])).astype(int) % len(th)
    j = np.round((P[:, 2] - zz[0]) / (zz[1] - zz[0])).astype(int)
    ok = (j >= 0) & (j < len(zz)); i, j, r = i[ok], j[ok], r[ok]
    keep = r >= r_skin[j, i] - WALL - 1.0
    lim = np.full(r_skin.shape, np.inf)
    np.minimum.at(lim, (j[keep], i[keep]), r[keep])
    lim[zz[:, None] > z_top[None] + 0.5] = np.inf
    return lim


FIT_MAX_CHANGE = 8.0               # fit-surface refuses a result that moves any control value further (mm / exponent units)


def fit_surface():
    """Re-fit the SURF control values (same heights) after a covered or neighbouring part moved: the surface closest to
    today's SURF skin between the edges that holds the requirement, stays inside the top free space below the top edge
    (so the top edge keeps its height), and obeys the constraints named above SURF. Prints the new SURF values."""
    from scipy.optimize import minimize
    body = kit.base(R.VARIANT)
    E = np.load(OUT / "thigh_shell_edges.npz")
    Q = requirement(body, E)
    zz = np.arange(SURF_Z[0], SURF_Z[1] + 0.1, 1.0); th = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    zb = np.interp(th, E["th"], E["zb"], period=2 * np.pi); zt = np.interp(th, E["th"], E["zt"], period=2 * np.pi)
    vis = (zz[:, None] >= zb[None] - 3) & (zz[:, None] <= zt[None] + 1)
    R0 = _radial_grid(surface(zz), th)
    lim = np.maximum(top_free_limit(body, th, zz, R0, zt) - 0.3, R0)
    band = np.isfinite(lim); rows, cols = np.nonzero(band.any(1))[0], np.nonzero(band.any(0))[0]
    print(f"{len(Q)} requirement points, {band.sum()} free-space cells", flush=True)
    sizes = np.cumsum([0] + [len(SURF[k][1]) for k in RING])
    def unpack(v):
        return {k: (SURF[k][0], v[sizes[i]:sizes[i + 1]]) for i, k in enumerate(RING)}
    BQ = {k: _spline_basis(SURF[k][0], Q[:, 2]) for k in RING}
    def lev(v, m):
        s = unpack(v); P = np.stack([BQ[k][m] @ s[k][1] for k in RING], -1)
        return level(P, Q[m])
    zm = np.arange(SURF_Z[0], SURF_Z_MID + 0.1, 2.0)
    def mono(v):
        cx, cy, a, bf, bk = surface(zm, unpack(v), nu=1)[:, :5].T
        return 100.0 * np.concatenate([cx + a, a - cx, bf - cy, cy + bk])
    def bounds(v):
        P = surface(zz, unpack(v))
        return np.concatenate([P[:, 5:7].ravel() - SURF_N["min"], SURF_N["front_max"] - P[:, 5], SURF_N["back_max"] - P[:, 6],
                               SURF_N["round_max"] - P[zz <= SURF_Z_MID, 5], P[:, 1] - SURF_N["cy"][0], SURF_N["cy"][1] - P[:, 1]])
    def free(v):
        return (lim[np.ix_(rows, cols)] - _radial_grid(surface(zz[rows], unpack(v)), th[cols]))[band[np.ix_(rows, cols)]]
    def obj(v):
        return float(np.mean(((_radial_grid(surface(zz, unpack(v)), th) - R0)[vis]) ** 2))
    v = np.concatenate([np.asarray(SURF[k][1], float) for k in RING])
    v0 = v.copy()
    L0 = lev(v, slice(None)).max()
    if L0 > 1.0:     # a covered part reaches outside today's skin: SLSQP starts infeasible (2026-11-20: it then ran away)
        Lq = lev(v, slice(None)); jq = int(np.argmax(Lq))
        print(f"start: max level {L0:.4f} > 1 at {np.round(Q[jq], 1)} ({int((Lq > 1).sum())} requirement points outside)", flush=True)
    act = lev(v, slice(None)) > 0.85
    for _ in range(4):
        m = np.nonzero(act)[0]
        r = minimize(obj, v, method="SLSQP", options=dict(maxiter=300, ftol=1e-7),
                     constraints=[dict(type="ineq", fun=lambda v, m=m: 1.0 - lev(v, m)), dict(type="ineq", fun=mono),
                                  dict(type="ineq", fun=bounds), dict(type="ineq", fun=free)])
        v = r.x; L = lev(v, slice(None))
        print(f"{r.message}: mean sq radial change {r.fun:.3f} mm2, max level {L.max():.4f}", flush=True)
        if L.max() <= 1.0 + 1e-4:
            break
        act |= L > 0.97
    # fail loudly instead of printing a run-away surface (2026-11-20: from an infeasible start SLSQP ended at control values in
    # the thousands with "Positive directional derivative for linesearch")
    change = float(np.max(np.abs(v - v0)))
    if not (r.success and np.all(np.isfinite(v)) and L.max() <= 1.0 + 1e-4 and change <= FIT_MAX_CHANGE):
        raise SystemExit(f"fit-surface did not converge ({r.message}; max level {L.max():.4f}, largest control value change "
                         f"{change:.2f} > {FIT_MAX_CHANGE} allowed or solver failed): SURF left as committed")
    print(f"largest control value change {change:.3f} (mm, or exponent units)")
    s = unpack(v)
    print("SURF = {"); [print(f'    "{k}": ({tuple(s[k][0])}, ({", ".join(f"{x:.3f}" for x in s[k][1])},)),') for k in RING]; print("}")


def stations(body=None, dz=1.0):
    """Dense rings: angles th, heights zz, outer radial ro and inner radial ri (outer offset inward by WALL) from CENTRE."""
    from scipy.interpolate import PchipInterpolator
    from shapely.geometry import Polygon
    th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
    zs, prm = ring_params()
    zz = np.arange(zs[0], zs[-1] + 1e-6, dz)
    Pz = PchipInterpolator(zs, prm, axis=0)(zz)
    ro = np.array([se_radial(p, th) for p in Pz])
    ri = []
    for r in ro:
        poly = Polygon(np.c_[CENTRE[0] + r * np.cos(th), CENTRE[1] + r * np.sin(th)]).buffer(-WALL, join_style=1, resolution=24)
        ri.append(radial(poly, th))
    return th, zz, ro, np.array(ri), dict(z=zs, params=prm)


def skin(th, zz, Ro, Ri):
    """Closed tube between an outer and inner radial skin (dense rings)."""
    import trimesh
    n = len(th); c, s = np.cos(th), np.sin(th)

    def ring(R, z):
        return np.c_[CENTRE[0] + R * c, CENTRE[1] + R * s, np.full(n, z)]
    V = np.vstack([ring(Ro[k], zz[k]) for k in range(len(zz))] + [ring(Ri[k], zz[k]) for k in range(len(zz))])
    m = len(zz); off = m * n
    F = []
    for k in range(m - 1):
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]                                   # outer, outward
            F += [[off + a, off + b + n, off + b], [off + a, off + a + n, off + b + n]]  # inner, inward
    for k, sgn in ((0, -1), (m - 1, 1)):                                              # end annuli
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            if sgn < 0:                                                               # bottom: normal -z
                F += [[a, off + b, b], [a, off + a, off + b]]
            else:                                                                     # top: normal +z
                F += [[a, b, off + b], [a, off + b, off + a]]
    t = trimesh.Trimesh(V, np.array(F), process=True)
    if t.volume < 0:
        t.invert()
    return t


def prism_y(poly_xz, y0=-80.0, y1=80.0):
    import trimesh
    from shapely.geometry import Polygon
    m = trimesh.creation.extrude_polygon(Polygon(poly_xz), y1 - y0)       # along +z in (x, y) -> map to (x, z) / y
    v = m.vertices.copy(); m.vertices = np.c_[v[:, 0], v[:, 2] + y0, v[:, 1]]
    m.invert()                                                           # the axis swap mirrors the solid
    return m


def prism_x(poly_yz, x0=-20.0, x1=130.0):
    import trimesh
    from shapely.geometry import Polygon
    m = trimesh.creation.extrude_polygon(Polygon(poly_yz), x1 - x0)
    v = m.vertices.copy(); m.vertices = np.c_[v[:, 2] + x0, v[:, 0], v[:, 1]]
    return m


def dilated(m, vox=0.5, faces=None):
    """Closed mesh m grown by CLEAR (voxel dilation + marching cubes), optionally decimated, as a manifold."""
    import trimesh
    from scipy.ndimage import binary_dilation
    from skimage.measure import marching_cubes
    lo = m.bounds[0] - CLEAR - 2 * vox; shape = np.ceil((m.bounds[1] + CLEAR + 2 * vox - lo) / vox).astype(int)
    g = np.stack(np.meshgrid(*[lo[i] + vox * np.arange(shape[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    inside = m.contains(g).reshape(shape)
    k = int(np.ceil(CLEAR / vox))
    ball = np.linalg.norm(np.stack(np.meshgrid(*[np.arange(-k, k + 1)] * 3, indexing="ij"), -1), axis=-1) <= k
    d = binary_dilation(inside, structure=ball)
    v, f, _, _ = marching_cubes(np.pad(d, 1).astype(np.float32), 0.5)
    dm = trimesh.Trimesh((v - 1) * vox + lo, f, process=True)
    if dm.volume < 0:
        dm.invert()
    if faces and len(dm.faces) > faces:
        dd = dm.simplify_quadric_decimation(face_count=faces)
        if kit.mf(dd).status().name == "NoError":
            dm = dd
    return kit.mf(dm)


def yellow_sweep(body):
    """Yellow pitch-to-roll part dilated by CLEAR, turned about the roll axis over ROLL_RANGE, in the thigh frame."""
    import manifold3d as M
    base = dilated(R.closed(body, "R_hip_pitch_to_roll_roll_flip"))
    copies = []
    for q in np.arange(ROLL_RANGE[0], ROLL_RANGE[1] + 0.1, 2.5):
        T = np.linalg.inv(B_roll(q))          # thigh frame: pitch-link parts turn by -q about the roll axis
        copies.append(base.transform(T[:3, :4]))
    return M.Manifold.batch_boolean(copies, M.OpType.Add)


# Joint-axis reliefs (owner D-12: shell edges are tight arcs about the joint axes). The neighbours that reach the shell in the
# realistic poses are cut out of the shell, grown by CLEAR:
#   - knee pit: the polar envelope r(phi) about the knee axis of the shin parts (their top, z > 86) swept over KNEE_RANGE,
#     max-filtered over +-3 deg, convex hull (arc + one straight chord), as one prism across the width (a clean arc, ~R 37.5 about the knee axis, not the shin
#     cover's outline);
#   - top rim: a clean arc about the roll axis (owner 2026-11-17: the swept cut-out looked ragged). The green yaw-to-pitch
#     connector and hip-pitch motor turn about the hip pitch axis and, relative to the thigh, about the roll axis; turning about the roll axis
#     keeps both their distance rho from the roll axis and their y, so the pitch sweep gives the smallest rho per angle and y. Above the roll
#     axis everything farther than that (minus CLEAR) is cut: a surface of revolution about the roll axis whose radius R(y)
#     changes with y at most 1 mm per mm (front wall: the arc round the roll horn; back wall: a wider arc).
KNEE_RANGE = (0.0, 110.0)         # knee flexion the shell leaves room for (owner 2026-11-18: 110; with the back sloped in, 80 would keep the knee edge level at z 136)
PITCH_RANGE, ROLL_SWEEP = (-90.0, 45.0), (-50.0, 30.0)   # range limits -85..40 / -45..25 plus 5 deg
KNEE_PARTS = ("R_shin_combined", "R_shin_back_hard_cover", "R_shin_front_hard_cover")
TOP_PARTS = ("R_hip_yaw_to_pitch_combined", "R_hip_pitch_ST3025_case", "R_hip_pitch_ST3025_output")   # green parts above the thigh
RIM_R_MAX = 80.0                  # rim radius about the roll axis where the green connector never comes (beyond the shell)
RIM_DPSI = 0.6                    # the rim radius grows at most this many mm per degree about the roll axis (smooth ramps)


def _rel(body, link, dq):
    q = {**body.standing, **dict(zip(R.J, (0.0,) * 4)), **dq}
    return np.linalg.inv(body.link_T("R_roll_link", q)) @ body.link_T(link, q)


def knee_pit(body):
    """Prism (along x) of the shin's swept polar envelope about the knee axis, + CLEAR."""
    T1 = _rel(body, "R_shin", dict(R_knee=30.0))
    a = np.linalg.lstsq(np.eye(3) - T1[:3, :3], T1[:3, 3], rcond=None)[0]          # a point on the knee axis (x)
    P = []
    for n in KNEE_PARTS:
        m = R.closed(body, n); V = m.vertices[m.vertices[:, 2] > 86.0]
        for k in np.arange(KNEE_RANGE[0], KNEE_RANGE[1] + 0.1, 2.5):
            T = _rel(body, "R_shin", dict(R_knee=k)); P.append((V @ T[:3, :3].T + T[:3, 3])[:, 1:] - a[1:])
    P = np.vstack(P)
    phi = np.degrees(np.arctan2(P[:, 1], P[:, 0])); r = np.hypot(P[:, 0], P[:, 1])
    bins = np.arange(-180, 180); env = np.zeros(len(bins))
    np.maximum.at(env, np.clip((phi + 180).astype(int), 0, 359), r)
    up = bins >= -90                                   # from straight down (-90) over the back to straight up: never grows
    env[up] = np.maximum.accumulate(env[up][::-1])[::-1]   # going up -> a clean arc ending in one radial edge (no saw teeth)
    env = np.max([np.roll(env, s) for s in range(-3, 4)], axis=0)
    env = np.where(env > 0, env + CLEAR, 0.0)
    th = np.radians(bins + 0.5)
    from shapely.geometry import MultiPoint
    poly = MultiPoint([(a[1] + e * np.cos(t), a[2] + e * np.sin(t)) for e, t in zip(env, th)]).convex_hull   # clean arc + chord
    return kit.mf(prism_x(list(poly.exterior.coords)[:-1])), dict(axis_y_z=[round(float(a[1]), 2), round(float(a[2]), 2)],
                                      radius_mm=round(float(env.max()), 1),
                                      up_to_deg_above_back=int(bins[(env >= env.max() - 1.0)].max()))


def green_rim(body):
    """Top-rim cut {rho >= Rr(psi, y)} about the roll axis (psi from straight up, + outboard). Rr is the green connector's
    smallest distance from the roll axis at that angle and y over its pitch x roll sweep, less CLEAR, then lowered to a
    smooth surface (|dRr/dpsi| <= RIM_DPSI mm/deg, |dRr/dy| <= 1): arcs about the roll axis joined by smooth ramps, not a
    union of swept copies."""
    import trimesh
    from scipy.ndimage import minimum_filter, gaussian_filter
    ps = np.arange(-180.0, 180.0, 1.0); ys = np.arange(-70.0, 70.01, 1.0)
    rmin = np.full((len(ps), len(ys)), np.inf)
    for n in TOP_PARTS:
        m = R.closed(body, n)
        V = np.vstack([m.vertices, trimesh.sample.sample_surface(m, 20000, seed=1)[0]])
        for p in np.arange(PITCH_RANGE[0], PITCH_RANGE[1] + 0.1, 1.0):     # roll keeps rho and y; it only turns psi
            T = _rel(body, body.parts[n]["owning_link"], dict(R_hip_pitch=p))
            Q = V @ T[:3, :3].T + T[:3, 3]
            dx, dz = Q[:, 0] - B.RF_AX, Q[:, 2] - B.RF_AZ
            i = np.round(np.degrees(np.arctan2(dx, dz)) + 180.0).astype(int) % len(ps)
            j = np.clip(np.round(Q[:, 1] - ys[0]).astype(int), 0, len(ys) - 1)
            np.minimum.at(rmin, (i, j), np.hypot(dx, dz))
    # roll q turns psi by -q: q in ROLL_SWEEP spreads each angle over [psi - q_max, psi - q_min]; CLEAR spreads +-5 deg, +-CLEAR in y
    lo, hi = -int(ROLL_SWEEP[1]) - 5, -int(ROLL_SWEEP[0]) + 5
    big = np.where(np.isfinite(rmin), rmin, 1e3)
    spread = np.full_like(big, 1e3)
    for k in range(lo, hi + 1):
        spread = np.minimum(spread, np.roll(big, k, axis=0))
    w = int(np.ceil(CLEAR))
    Rr = np.minimum(minimum_filter(spread, size=(1, 2 * w + 1), mode="nearest") - CLEAR, RIM_R_MAX)
    for _ in range(4):                                             # slope limits (lower envelope): smooth ramps, no steps
        for i in range(1, len(ps)):
            Rr[i] = np.minimum(Rr[i], Rr[i - 1] + RIM_DPSI)
        for i in range(len(ps) - 2, -1, -1):
            Rr[i] = np.minimum(Rr[i], Rr[i + 1] + RIM_DPSI)
        for j in range(1, len(ys)):
            Rr[:, j] = np.minimum(Rr[:, j], Rr[:, j - 1] + 1.0)
        for j in range(len(ys) - 2, -1, -1):
            Rr[:, j] = np.minimum(Rr[:, j], Rr[:, j + 1] + 1.0)
    Rs = gaussian_filter(Rr, (3.0, 1.5), mode=("wrap", "nearest"))
    Rs = Rs - minimum_filter(np.maximum(Rs - Rr, 0.0) * -1.0, size=(9, 5), mode=("wrap", "nearest")) * -1.0   # stay <= Rr
    keep = (ps >= -150) & (ps <= 150)
    psi = np.radians(ps[keep]); RR = np.minimum(Rs[keep], RIM_R_MAX)
    R_OUT = RIM_R_MAX + 60.0
    rings = []
    for j, y in enumerate(ys):
        r = RR[:, j]; Mn = len(psi)
        inner = np.c_[B.RF_AX + r * np.sin(psi), np.full(Mn, y), B.RF_AZ + r * np.cos(psi)]
        outer = np.c_[B.RF_AX + R_OUT * np.sin(psi), np.full(Mn, y), B.RF_AZ + R_OUT * np.cos(psi)][::-1]
        rings.append(np.vstack([inner, outer]))
    Mn = len(psi); n = 2 * Mn; Vt = np.vstack(rings); F = []
    for k in range(len(rings) - 1):
        for jj in range(n):
            a_, b_ = k * n + jj, k * n + (jj + 1) % n
            F += [[a_, b_, b_ + n], [a_, b_ + n, a_ + n]]
    for k, flip in ((0, True), (len(rings) - 1, False)):
        for jj in range(Mn - 1):
            i0, i1, o0, o1 = k * n + jj, k * n + jj + 1, k * n + n - 1 - jj, k * n + n - 2 - jj
            tri = [[i0, i1, o0], [i1, o1, o0]]
            F += [t[::-1] for t in tri] if flip else tri
    t = trimesh.Trimesh(Vt, np.array(F), process=True)
    if t.volume < 0:
        t.invert()
    def at(pd, y):
        return round(float(Rs[int(pd + 180), int(y - ys[0])]), 1)
    info = {f"y {y:.0f}": {f"psi {pd}": at(pd, y) for pd in (-90, -60, -30, 0, 30, 60, 90)} for y in (-25, 0, 8, 20, 40)}
    return kit.mf(t), info


# Edges (owner 2026-11-17: "one continuous ridge that avoids collision and wraps the actuators closely; hills and valleys
# are fine when they are continuous, intentional and smooth, not random"). Both edges are heights z(theta) around the
# shell's axis (vertical through CENTRE). The space the shell must leave free is computed first: on top, the green
# connector / hip-pitch motor rim region about the roll axis (green_rim) and the yellow pitch-to-roll part swept over the
# roll range; at the knee, every part on the lower leg (R_shin) grown by CLEAR and swept over KNEE_RANGE about the knee
# axis. Their overlap with the tube gives, per angle, the lowest height the top edge may reach and the highest the bottom
# edge must clear.
# Top edge (owner 2026-11-20: "some of the peaks on the top of the thigh shell are too aggressively high ... there will
# still be fabric underneath, we don't need to aggressively cover ... smooth is aesthetic"): no longer as high as the
# neighbours allow (5 harmonics reached 206.5 back-inboard and 200 front-inboard, with the yellow arms' two 30-deg slots
# cut 17-23 mm deep between them, slopes up to 1.1 mm/mm). Now 3 harmonics (a calm crown: one valley over each yellow
# arm slot, front-outboard and back-outboard, low hills front-inboard, back-inboard and outboard) as high on average as a
# linear program allows with every peak at most TOP_PEAK_CAP above the edge's mean, every dip at most TOP_DIP_CAP below
# it and |dz/ds| <= TOP_SLOPE_MAX along the edge (s = arc length round the outer skin at TOP_RING_Z); still on the safe
# side of the free space at every angle (same CLEAR as before). The covered parts' tops (roll case 200.1, yellow arm hubs
# 202.4) stand up to 16 mm above it (as before at the slots; now all round); fabric covers them. Values chosen from the
# look (drawing-sheets/roll-flip/thigh-shell-top-before / -after.png, thigh_shell_views.py top).
EDGE_K = 3                        # top edge harmonics (was 5)
TOP_PEAK_CAP = 4.5                # mm a hill may rise above the edge's mean height
TOP_DIP_CAP = 5.0                 # mm a valley may sink below it (the yellow-arm valleys: wide and shallow)
TOP_SLOPE_MAX = 0.35              # mm of height per mm along the edge
TOP_RING_Z = 190.0                # the arc length round the outer skin is taken at this height (the roll axis, mid-band)
# owner 2026-11-18: no tongue down the front; the knee edge is one tilted plane (a straight line in the side view)
Z_KNEE_EDGE_MIN = 135.0           # ... ending just above the knee hub (output horn top z 132.4)
EDGE_NTH = 360


def shin_sweep(body):
    """Everything on the lower leg near the knee, grown by CLEAR, swept over KNEE_RANGE (thigh frame)."""
    import manifold3d as M
    # the shin hard covers are left out: they slope in toward the knee hinge to clear this shell (owner 2026-11-18, shin step)
    names = [n for n, p in body.parts.items() if n.startswith("R_") and p["owning_link"] == "R_shin"
             and np.array(p["bounds_mm"])[1][2] > 90.0 and "hard_cover" not in n]
    base = []
    for n in names:
        try:
            base.append(dilated(R.closed(body, n)))
        except Exception:  # noqa: BLE001  (screws and nuts without a closed mesh: their bounds as boxes)
            import trimesh
            b = np.array(body.parts[n]["bounds_mm"])
            base.append(kit.mf(trimesh.creation.box(bounds=[b[0] - CLEAR, b[1] + CLEAR])))
    base = M.Manifold.batch_boolean(base, M.OpType.Add)
    copies = []
    for k in np.arange(KNEE_RANGE[0], KNEE_RANGE[1] + 0.1, 2.5):
        T = _rel(body, "R_shin", dict(R_knee=float(k)))
        copies.append(base.transform(T[:3, :4]))
    return M.Manifold.batch_boolean(copies, M.OpType.Add), names


def _theta_extreme(inter, top):
    """Per angle about CENTRE: lowest (top=True) or highest z of the solid `inter` (NaN where it is empty)."""
    import trimesh
    m = kit.tm(inter)
    if len(m.faces) == 0:
        return np.full(EDGE_NTH, np.nan)
    P = np.vstack([m.vertices, trimesh.sample.sample_surface(m, 200000, seed=2)[0]])
    t = np.arctan2(P[:, 1] - CENTRE[1], P[:, 0] - CENTRE[0])
    i = np.floor((t + np.pi) / (2 * np.pi) * EDGE_NTH).astype(int) % EDGE_NTH
    out = np.full(EDGE_NTH, np.inf if top else -np.inf)
    (np.minimum if top else np.maximum).at(out, i, P[:, 2])
    out[~np.isfinite(out)] = np.nan
    w = 2                                                    # +-1 deg: bin edges never let a corner slip between angles
    stack = np.array([np.roll(out, s) for s in range(-w, w + 1)])
    with np.errstate(all="ignore"):
        return np.nanmin(stack, 0) if top else np.nanmax(stack, 0)


def fourier_edge(req, top, k=EDGE_K, dsdth=None, peak=None, dip=None, slope=None):
    """Smooth periodic edge z(theta) = c0 + sum_k (a_k cos k th + b_k sin k th) on the safe side of req at every angle,
    keeping as much shell as possible (linear program). Optional shape limits (top edge): z - c0 <= peak, c0 - z <= dip
    (c0 is the edge's mean height) and |dz/ds| <= slope with ds = dsdth(theta) dtheta (arc length along the edge)."""
    from scipy.optimize import linprog
    th = (np.arange(EDGE_NTH) + 0.5) / EDGE_NTH * 2 * np.pi - np.pi
    A = np.c_[np.ones(EDGE_NTH), *[f(j * th) for j in range(1, k + 1) for f in (np.cos, np.sin)]]
    dA = np.c_[np.zeros(EDGE_NTH), *[g for j in range(1, k + 1) for g in (-j * np.sin(j * th), j * np.cos(j * th))]]
    e0 = np.zeros(A.shape[1]); e0[0] = 1.0
    c = -A.sum(0) if top else A.sum(0)                      # top: as high as allowed; bottom: as low as allowed
    rows, rhs = [A if top else -A], [req if top else -req]
    if peak is not None:
        rows.append(A - e0); rhs.append(np.full(EDGE_NTH, float(peak)))
    if dip is not None:
        rows.append(e0 - A); rhs.append(np.full(EDGE_NTH, float(dip)))
    if slope is not None:
        rows += [dA / dsdth[:, None], -dA / dsdth[:, None]]; rhs += [np.full(EDGE_NTH, float(slope))] * 2
    res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), bounds=[(None, None)] * A.shape[1], method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return th, A @ res.x, res.x


def edge_shape(th, z, ring_z=TOP_RING_Z):
    """Shape measures of an edge z(theta) on the outer skin: heights, peaks above / dips below the mean and median,
    slope dz/ds and curvature d2z/ds2 along the edge (s = arc length round the skin at ring_z, mm), local hills / valleys."""
    r = se_radial(surface(np.array([ring_z]))[0], th)
    h = th[1] - th[0]
    dsdth = np.hypot(r, (np.roll(r, -1) - np.roll(r, 1)) / (2 * h))
    dz = (np.roll(z, -1) - np.roll(z, 1)) / (2 * h) / dsdth
    d2 = (np.roll(z, -1) - 2 * z + np.roll(z, 1)) / (h * dsdth) ** 2
    deg = np.degrees(th)
    ext = [dict(kind="hill" if z[i] > z[i - 1] else "valley", deg=round(float(deg[i]), 1), z=round(float(z[i]), 1))
           for i in range(len(z)) if (z[i] - z[i - 1]) * (z[(i + 1) % len(z)] - z[i]) < 0]
    return dict(z_min=round(float(z.min()), 1), z_max=round(float(z.max()), 1), z_mean=round(float(z.mean()), 1),
                z_median=round(float(np.median(z)), 1), peak_above_mean_mm=round(float(z.max() - z.mean()), 1),
                peak_above_median_mm=round(float(z.max() - np.median(z)), 1), dip_below_mean_mm=round(float(z.mean() - z.min()), 1),
                slope_abs_mm_per_mm=dict(max=round(float(abs(dz).max()), 3), p95=round(float(np.percentile(abs(dz), 95)), 3),
                                         mean=round(float(abs(dz).mean()), 3)),
                curvature_abs_per_mm=dict(max=round(float(abs(d2).max()), 4), p95=round(float(np.percentile(abs(d2), 95)), 4)),
                perimeter_mm=round(float(dsdth.sum() * h), 1), hills_valleys=ext), dsdth


def plane_edge(req, xy, margin=1.0):
    """Knee edge as one plane z = c + p x + q y, as low as the lower leg allows: on or above req (+ margin) at the outer
    skin points xy of each angle (linear program, least shell given up)."""
    from scipy.optimize import linprog
    A = np.c_[np.ones(len(xy)), xy]
    res = linprog(A.sum(0), A_ub=-A, b_ub=-(req + margin), bounds=[(None, None)] * 3, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x, A @ res.x


def plane_cutter(cpq, far=400.0):
    """Half-space below the plane z = c + p x + q y (a sheared box)."""
    import trimesh
    c, p, q = cpq
    box = kit.mf(trimesh.creation.box(bounds=[[-300, -300, -far], [400, 300, 0.0]]))
    return box.transform(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [p, q, 1, c]], float))


def edge_cutter(th, z_edge, top, far=400.0):
    """Closed solid on the far side of the edge height z_edge(theta): an annulus about CENTRE (r 1..160)."""
    import trimesh
    n = len(th); r0, r1 = 1.0, 160.0
    zf = far if top else -far
    c, s_ = np.cos(th), np.sin(th)
    rings = [np.c_[CENTRE[0] + r * c, CENTRE[1] + r * s_, z] for r, z in
             ((r0, z_edge), (r1, z_edge), (r1, np.full(n, zf)), (r0, np.full(n, zf)))]
    V = np.vstack(rings); F = []
    for q in range(4):
        a0, b0 = q * n, ((q + 1) % 4) * n
        for j in range(n):
            j1 = (j + 1) % n
            F += [[a0 + j, a0 + j1, b0 + j1], [a0 + j, b0 + j1, b0 + j]]
    t = trimesh.Trimesh(V, np.array(F), process=True)
    t.fix_normals()
    if t.volume < 0:
        t.invert()
    return kit.mf(t)


def edge_cut(body, tube, knee_ring):
    """Top and bottom cutters for the tube, and a report of what set them. knee_ring = (angles, outer radius) of the skin
    near the knee edge."""
    # the green rim region is a surface of revolution about the roll axis built for the top; below TOP_FREE_Z0 (the yellow
    # arm hubs' bottom) it says nothing about the shell, so it is clipped there (the full check tests the real green parts)
    import trimesh
    clip = kit.mf(trimesh.creation.box(bounds=[[-100, -150, B.RF_AZ - B.RF_ROLL_HUB_R], [250, 150, 400]]))
    top_free = (green_rim(body)[0] ^ clip) + yellow_sweep(body)
    knee, names = shin_sweep(body)
    req_t = _theta_extreme(tube ^ top_free, top=True)
    req_b = _theta_extreme(tube ^ knee, top=False)
    req_t = np.where(np.isnan(req_t), TOP_Z, np.minimum(req_t, TOP_Z))
    req_b = np.where(np.isnan(req_b), Z_KNEE_EDGE_MIN, np.maximum(req_b, Z_KNEE_EDGE_MIN))
    th0 = (np.arange(EDGE_NTH) + 0.5) / EDGE_NTH * 2 * np.pi - np.pi
    _, dsdth = edge_shape(th0, np.zeros(EDGE_NTH))
    th, zt, _ = fourier_edge(req_t, top=True, dsdth=dsdth, peak=TOP_PEAK_CAP, dip=TOP_DIP_CAP, slope=TOP_SLOPE_MAX)
    tr, rr = knee_ring
    r_at = np.interp(th % (2 * np.pi), np.r_[tr, tr[0] + 2 * np.pi], np.r_[rr, rr[0]])
    xy = CENTRE + r_at[:, None] * np.c_[np.cos(th), np.sin(th)]
    cpq, zb = plane_edge(req_b, xy)
    deg = np.degrees(th)
    def at(z):
        return {f"{d:+d}": round(float(np.interp(d, deg, z)), 1) for d in (-180, -135, -90, -45, 0, 45, 90, 135)}
    info = dict(harmonics=EDGE_K, top_edge_limits=dict(peak_above_mean_mm=TOP_PEAK_CAP, dip_below_mean_mm=TOP_DIP_CAP,
                slope_mm_per_mm=TOP_SLOPE_MAX, arc_at_z=TOP_RING_Z), top_edge_shape=edge_shape(th, zt)[0], knee_edge_plane_c_p_q=[round(float(v), 4) for v in cpq], knee_edge_min_z=Z_KNEE_EDGE_MIN, knee_range_deg=list(KNEE_RANGE), lower_leg_parts=len(names),
                top_edge_z=at(zt), top_required_z=at(req_t), bottom_edge_z=at(zb), bottom_required_z=at(req_b),
                top_edge_range_z=[round(float(zt.min()), 1), round(float(zt.max()), 1)],
                bottom_edge_range_z=[round(float(zb.min()), 1), round(float(zb.max()), 1)],
                top_shell_given_up_mm=round(float(np.mean(req_t - zt)), 2),
                bottom_shell_given_up_mm=round(float(np.mean(zb - req_b)), 2),
                angles="deg about the shell axis: 0 outboard (+x), -90 front (-y), +90 back (+y), 180 inboard")
    return edge_cutter(th, zt, True) + plane_cutter(cpq), info, dict(th=th, zt=zt, zb=zb, req_t=req_t, req_b=req_b)


def relief_sweep(body):
    rim, rinfo = green_rim(body)
    print("top rim (arc about the roll axis)", rinfo, flush=True)
    pit, info = knee_pit(body)
    print("knee pit", info, flush=True)
    return rim + pit


def B_roll(q):
    a = np.radians(q)
    Rm = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = np.array([B.RF_AX, 0.0, B.RF_AZ]); T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = p - Rm @ p
    return T


def build():
    import manifold3d as M
    import trimesh
    t0 = time.time()
    body = kit.base(R.VARIANT)
    th, zs, ro, ri, rings = stations(body)
    tube = kit.mf(skin(th, zs, ro, ri))
    # edges: smooth Fourier heights around the shell, on the safe side of the green / yellow (top) and lower-leg (knee) sweeps
    cut, edge_info, edge_curves = edge_cut(body, tube, (th, ro[int(np.argmin(abs(zs - 142.0)))]))
    print("edges", json.dumps(edge_info), flush=True)
    shell = tube - cut
    np.savez(OUT / "thigh_shell_edges.npz", **edge_curves)
    # how well the skin holds what it covers (GAP + WALL included), between the new edges
    L, margin = containment(requirement(body, edge_curves))
    edge_info["skin_containment"] = dict(points=len(L), max_level=round(float(L.max()), 4),
                                         radial_margin_mm=dict(min=round(float(margin.min()), 2), p10=round(float(np.percentile(margin, 10)), 2),
                                                               median=round(float(np.median(margin)), 2)),
                                         points_short_over_0_2mm=int((margin < -0.2).sum()))
    print("skin containment", edge_info["skin_containment"], flush=True)
    print(f"tube + smooth edges ({time.time() - t0:.0f}s)", flush=True)
    # bosses / holes at the Stage 4 posts: boss r 4.8 from the post tip out, a steel sleeve bore through it and a 90-deg
    # countersink for an ISO 10642 M3 head, flush with the outer surface or sunk below it (post_fit, post_screws)
    posts = shell_posts(th, zs, ro)
    s4 = body.mesh("R_thigh_roll_flip")
    bosses, holes = [], []
    for q in posts:
        x, z, tip, sg, y_out = q["x"], q["z"], q["tip_y"], q["sg"], q["skin_y"]
        q["fit"] = post_fit(q, s4)
        y0, y1 = sorted((tip, y_out + sg * 2.0))
        bosses.append(trimesh.creation.cylinder(4.8, y1 - y0, sections=48, transform=_ty(x, (y0 + y1) / 2, z)))
        holes.append(trimesh.creation.cylinder(SLEEVE_R, 40.0, sections=48, transform=_ty(x, tip + sg * 15.0, z)))
        holes += head_seat_cutters(x, y_out, z, sg, q["fit"]["recess_mm"])
    hull_out = kit.mf(_outer_hull(th, zs, ro))
    boss = M.Manifold.batch_boolean([kit.mf(b) for b in bosses], M.OpType.Add) ^ hull_out
    shell = (shell + boss) - M.Manifold.batch_boolean([kit.mf(h) for h in holes], M.OpType.Add)
    # halves, today's split (front y < 7.7, back y > 8.3) and today's knee cuts
    front = shell ^ kit.mf(trimesh.creation.box(bounds=[[-50, -100, Z0], [150, 7.7, 300]]))
    back = shell ^ kit.mf(trimesh.creation.box(bounds=[[-50, 8.3, Z0], [150, 100, 300]]))
    out = {}
    for name, part in zip(NAMES, (front, back)):
        mm = kit.tm(part)
        parts = mm.split(only_watertight=False)
        mm = max(parts, key=lambda p: abs(p.volume))
        dropped = round(float(sum(abs(p.volume) for p in parts) - abs(mm.volume)), 2)   # slivers of the boss / hole cuts
        for side, msh in (("R", mm), ("L", kit.mirror_x(mm))):
            nm = side + name[1:]
            f = OUT / f"{nm}.stl"; msh.export(f)
            out[nm] = dict(mesh=str(f.relative_to(R.ROOT)), mesh_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                           bounds_mm=[np.round(msh.bounds[0], 2).tolist(), np.round(msh.bounds[1], 2).tolist()],
                           volume_mm3=round(float(msh.volume), 1), mass_g=round(float(msh.volume) * DENSITY, 2),
                           dropped_fragments_mm3=dropped, watertight=bool(msh.is_watertight))
        print(name, out["R" + name[1:]], flush=True)
    # manifest: replace today's four thigh covers by the two shells per leg
    hw_rows, hw_report = post_screws(posts, body)
    man = json.loads(R.VARIANT.read_text())
    keep = [p for p in man["parts"] if p["name"][2:] not in ("thigh_front_hard_cover", "thigh_back_hard_cover", "thigh_shell_front", "thigh_shell_back")
            and not _shell_hardware(p["name"])]
    keep += hw_rows
    for nm, r in out.items():
        keep.append(dict(name=nm, owning_link=f"{nm[0]}_roll_link", role="body_shell", mesh=r["mesh"], mesh_sha256=r["mesh_sha256"],
                         mesh_units="mm", bounds_mm=r["bounds_mm"], mass_kg=round(r["mass_g"] / 1000, 5), color=[47, 111, 208, 255],
                         material="PETG / polycarbonate hard-shell cover (as today's thigh covers)",
                         mass_basis="closed mesh volume x 1.27 g/cm3", roll_flip="blue thigh shell (thigh_shell.py)"))
    man["parts"] = keep
    man["roll_flip"]["thigh_shell"] = dict(replaces=list(TODAY) + ["L" + n[1:] for n in TODAY], source=str(Path(__file__).relative_to(R.ROOT)),
                                           gap_mm=GAP, wall_mm=WALL, roll_range_deg=list(ROLL_RANGE), slot_clearance_mm=CLEAR)
    R.VARIANT.write_text(json.dumps(man, indent=1) + "\n")
    R.merge("thigh_shell_build", dict(parts=out, seconds=round(time.time() - t0, 1), post_screws=hw_report,
                                     edges=edge_info,
                                     surface=dict(spline_z=SURF_Z, control_values={k: dict(z=list(SURF[k][0]), value=list(SURF[k][1])) for k in RING},
                                                  shape_parameters=sum(len(SURF[k][1]) for k in RING)),
                                     settings=dict(gap_mm=GAP, wall_mm=WALL, bottom_z_mm=Z0, station_window_mm=WINDOW,
                                                   top_z_mm=TOP_Z, roll_range_deg=ROLL_RANGE, slot_clearance_mm=CLEAR)))


# ------------------------------------------------------------------------------------------------ shell screws
# The shell halves are bolted to the four Stage 4 posts (RF_POSTS / RF_POSTS_R): an ISO 10642 M3 countersunk screw, head
# flush with the outer skin at the post axis, through a 304 steel sleeve (dia 4.5 / 3.2) pressed into the boss from the
# post tip to the countersink (its outer end countersunk 90 deg like the shell, so the head bears on the steel and the clamp
# is head -> sleeve -> nut: the printed boss is not crushed and the preload does not creep away), into the ISO 4032 M3 nut
# in the post's captive-nut pocket, behind a solid wall at the post tip (B.RF_POST_NUT_SHIFT): under pull-off the nut bears
# on that wall, and the clamp is head -> sleeve -> post tip wall -> nut. post_fit picks the length and the head recess: the
# thread runs through the whole nut and the tip stays POST_TIP_MIN short of the post bore's bottom (the whole screw
# length, point included); flush heads first, then the shortest length; a head sinks below the skin (counterbore over the
# countersink) only when no standard length fits flush. stack = head top to the nut's near face, reference = nut height.
SLEEVE_R, SLEEVE_BORE_R = 2.25, 1.60
POST_TIP_MIN, POST_RECESS_MAX, POST_CBORE_CLEAR = 0.3, 2.5, 0.2   # mm: tip to bore bottom, head below the skin, counterbore
CS_HEAD, NUT_STD, THREAD = "ISO 10642", "ISO 4032", "M3"


def _hw():
    sys.path.insert(0, str(R.ROOT / "robot/body/fastening"))
    import screw_plan as SP
    st = SP.standards()
    return SP, st, st["head_types"][CS_HEAD][THREAD], st["nut_standards"][NUT_STD][THREAD]


def shell_posts(th, zs, ro):
    """The four post sites (right leg): post axis (x, z), tip y (shell boss / nut face), side sg (-1 front, +1 back), skin y
    at the axis, the shell half and the legacy screw name stem."""
    out = []
    for half, sg, posts in (("front", -1.0, B.RF_POSTS), ("back", 1.0, B.RF_POSTS_R)):
        for i, ((x, z), tip, _nut) in enumerate(posts):
            stem = f"Thigh_carrier_shell_thigh_front_{i + 1}" if half == "front" else f"Thigh_rear_carrier_shell_thigh_back_{i + 1}"
            out.append(dict(half=half, i=i + 1, stem=stem, x=x, z=z, tip_y=tip, sg=sg, skin_y=_outer_y(th, zs, ro, x, z, sg),
                            nut_face_y=B.rf_post_nut_face(tip, _nut),
                            skin_y_over_head=[_outer_y(th, zs, ro, x + 3.36 * np.cos(a), z + 3.36 * np.sin(a), sg)
                                              for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]))
    return out


def countersink(x, y_out, z, sg):
    """90-deg countersink cutter for the ISO 10642 head: radius dk/2 at the skin point (x, y_out, z), apex inward."""
    import trimesh
    rcs = _hw()[2]["dk"] / 2.0
    rb = rcs + 6.0
    cone = trimesh.creation.cone(radius=rb, height=rb, sections=64)        # apex up (local z), 45 deg
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, -sg], [0, sg, 0]]        # apex (local z) points inward
    T[:3, 3] = [x, y_out + sg * (rb - rcs), z]
    cone.apply_transform(T)
    return cone


def head_seat_cutters(x, y_out, z, sg, recess):
    """Countersink with its head top `recess` below the skin point (x, y_out, z), plus (recess > 0) a counterbore of the
    head's radius + POST_CBORE_CLEAR from that plane out through the skin."""
    import trimesh
    y_seat = y_out - sg * recess
    out = [countersink(x, y_seat, z, sg)]
    if recess > 1e-6:
        r = _hw()[2]["dk"] / 2.0 + POST_CBORE_CLEAR
        out.append(trimesh.creation.cylinder(r, 12.0, sections=64, transform=_ty(x, y_seat + sg * 6.0, z)))
    return out


def post_fit(q, s4):
    """Length and head recess for one post (see the comment above SLEEVE_R): dist = flush head top -> nut near face,
    free = nut near face -> post bore bottom."""
    SP, st, head, nut = _hw()
    pt, m = st["point_mm"], nut["m"]
    tip, sg, y_out, nf = q["tip_y"], q["sg"], q["skin_y"], q["nut_face_y"]
    bottom = _bore_bottom(s4, q["x"], q["z"], tip, sg)
    dist0, free = abs(y_out - nf), abs(bottom - nf)
    cands = []
    for L in head["lengths"]:
        s = max(0.0, dist0 + m - (L - pt))                 # recess that runs the thread through the whole nut
        if s <= POST_RECESS_MAX and (dist0 - s + free) - L >= POST_TIP_MIN:
            cands.append((round(s, 6), L, s))
    if cands:
        _, L, s = min(cands)
    else:                                                    # nothing fits: flush, nearest length by violation
        s, L = 0.0, SP.choose_length(dist0, free - pt, nut["reference_engagement_mm"], head["lengths"], pt)["length_mm"]
    return dict(length_mm=float(L), recess_mm=round(s, 3), dist_flush_mm=round(dist0, 3), dist_mm=round(dist0 - s, 3),
                free_mm=round(free, 3), bore_bottom_y=round(bottom, 3), nut_face_y=round(nf, 3),
                tip_wall_mm=round(abs(nf - tip), 3), fits=bool(cands))


def _shell_hardware(name):
    n = name[2:]
    return (n.startswith("Thigh_carrier_shell_thigh_front_") or n.startswith("Thigh_rear_carrier_shell_thigh_back_")
            or n in ("thigh_shell_front_steel_sleeves", "thigh_shell_back_steel_sleeves"))


def _bore_bottom(s4, x, z, tip, sg):
    """y of the first Stage 4 material on the post axis beyond the tip (inward): the bottom of the post's screw bore."""
    loc, _, _ = s4.ray.intersects_location([[x, tip + sg * 0.3, z]], [[0.0, -sg, 0.0]])
    ys = loc[:, 1][(loc[:, 1] - tip) * -sg > 0.3]
    return float(ys[np.argmin(np.abs(ys - tip))])


def post_screws(posts, body):
    """Screw, nut and sleeve proxies of both legs for the four posts per leg, and their measurements."""
    import manifold3d as M
    import trimesh
    SP, st, head, nut = _hw()
    pt = st["point_mm"]
    s4 = body.mesh("R_thigh_roll_flip")
    rows, rep, sleeves = [], [], {"front": [], "back": []}
    for q in posts:
        x, z, tip, sg, y_out = q["x"], q["z"], q["tip_y"], q["sg"], q["skin_y"]
        fit = q.get("fit") or post_fit(q, s4)
        nf, rec, free, L = fit["nut_face_y"], fit["recess_mm"], fit["free_mm"], fit["length_mm"]
        y_head = y_out - sg * rec                                    # head top (flush, or sunk by the recess)
        dist = abs(y_head - nf)                                      # head top -> nut near face
        pick = SP.choose_length(dist, free - pt, nut["reference_engagement_mm"], [L], pt)
        eng = pick["engagement_mm"]
        plan = dict(thread=THREAD, head_dia_mm=head["dk"], head_height_mm=head["k"], length_mm=L, point_mm=pt, countersunk=True,
                    axis_out=np.array([0.0, sg, 0.0]), opening=np.array([x, nf, z]), stack_mm=dist)
        sm = SP.screw_mesh(plan)
        nc = nf - sg * nut["m"] / 2                                  # nut against the tip wall (pull-off direction)
        nm = trimesh.creation.cylinder(nut["s"] / np.sqrt(3.0), nut["m"], sections=6, transform=_ty(x, nc, z))
        nm = kit.tm(kit.mf(nm) - kit.mf(trimesh.creation.cylinder(1.25, nut["m"] + 2.0, sections=32, transform=_ty(x, nc, z))))
        sl_len = abs(y_head - tip)                                   # sleeve: post tip -> head seat
        tube = trimesh.creation.cylinder(SLEEVE_R, sl_len, sections=48, transform=_ty(x, tip + sg * sl_len / 2, z))
        bore = trimesh.creation.cylinder(SLEEVE_BORE_R, sl_len + 2.0, sections=48, transform=_ty(x, tip + sg * sl_len / 2, z))
        sl = kit.mf(tube) - kit.mf(bore) - kit.mf(countersink(x, y_head, z, sg))
        sleeves[q["half"]].append(sl)
        over = np.array(q["skin_y_over_head"]) - y_head
        r = dict(site=q["stem"], post_xz_mm=[x, z], half=q["half"], head_standard=CS_HEAD, length_mm=L,
                 head_recess_below_skin_mm=rec, fits_full_nut_and_tip_margin=fit["fits"],
                 post_tip_wall_mm=fit["tip_wall_mm"], nut_face_y_mm=nf,
                 head_top_to_nut_mm=round(dist, 3), nut_to_bore_bottom_mm=round(free, 3),
                 thread_in_nut_mm=round(min(max(eng, 0.0), nut["m"]), 3), nut_height_mm=nut["m"],
                 thread_past_nut_mm=round(eng - nut["m"], 3), tip_to_bore_bottom_mm=round(dist + free - L, 3),
                 shortfall_mm=pick["shortfall_mm"], overrun_mm=pick["overrun_mm"],
                 skin_vs_head_top_over_head_disc_mm=[round(float((-sg * over).min()), 2), round(float((-sg * over).max()), 2)],
                 sleeve_len_axis_mm=round(sl_len - (head["dk"] / 2 - SLEEVE_R), 3))
        rep.append(r)
        lbl = f"M3x{L:g}"
        for side in ("R", "L"):
            mir = (lambda m: kit.mirror_x(m)) if side == "L" else (lambda m: m)
            sn = f"{side}_{q['stem']}_{lbl}_countersunk"
            ms = mir(sm)
            rows.append(dict(name=sn, physical_part_id=sn, bom_group=f"{side}_thigh_shell_{lbl}_countersunk", owning_link=f"{side}_roll_link",
                             role="fastener_proxy", **R.save_npz(sn, ms), volume_mm3=round(float(ms.volume), 3),
                             mass_kg=round(float(ms.volume) * R.STEEL / 1000.0, 7),
                             material=f"{CS_HEAD} {lbl} (hex socket countersunk, head flush with the shell skin) into the {NUT_STD} M3 nut in the "
                                      f"Stage 4 post; head dk {head['dk']} (theoretical) x k {head['k']}, shank dia 3, {pt} mm point (envelope)",
                             color=[195, 204, 214, 255], display_group="mechanism", display_groups=["mechanism"],
                             overlay_id="actuators_and_hardware", parent_assembly="complete_fdm_ape_body_assembly",
                             inspection_reference=False, mass_basis="mesh volume x 7.85 g/cm3",
                             roll_flip="generated from the shell's countersink and the post's nut by thigh_shell.py post_screws",
                             screw_plan=dict(hole_id=f"{q['stem']}: nut in the Stage 4 post", thread_label="M3", head_standard=CS_HEAD,
                                             length_mm=L, stack_mm=round(dist, 3), engagement_mm=round(eng, 3),
                                             reference_engagement_mm=nut["reference_engagement_mm"], hole_depth_mm=round(free, 3),
                                             tip_margin_mm=round(dist + free - L, 3), shortfall_mm=pick["shortfall_mm"],
                                             overrun_mm=pick["overrun_mm"], stack_change_to_fit_mm=pick["stack_change_to_fit_mm"])))
            nn = f"{side}_{q['stem']}_nut"
            mn = mir(nm)
            rows.append(dict(name=nn, physical_part_id=nn, bom_group=f"{side}_thigh_shell_nut_M3", owning_link=f"{side}_roll_link",
                             role="fastener_proxy", **R.save_npz(nn, mn), volume_mm3=round(float(mn.volume), 3),
                             mass_kg=round(float(mn.volume) * R.STEEL / 1000.0, 7),
                             material=f"{NUT_STD} M3 hex nut (s {nut['s']}, m {nut['m']}) in the Stage 4 post's captive-nut pocket, against its tip wall",
                             color=[220, 185, 75, 255], display_group="mechanism", display_groups=["mechanism"],
                             overlay_id="actuators_and_hardware", parent_assembly="complete_fdm_ape_body_assembly",
                             inspection_reference=False, mass_basis="mesh volume x 7.85 g/cm3",
                             roll_flip="generated at the post by thigh_shell.py post_screws"))
    for half in ("front", "back"):
        sm = kit.tm(M.Manifold.batch_boolean(sleeves[half], M.OpType.Add))
        for side in ("R", "L"):
            nm_ = f"{side}_thigh_shell_{half}_steel_sleeves"
            ms = kit.mirror_x(sm) if side == "L" else sm
            rows.append(dict(name=nm_, owning_link=f"{side}_roll_link", role="compression_limit_sleeves", **R.save_npz(nm_, ms),
                             volume_mm3=round(float(ms.volume), 3), mass_kg=round(float(ms.volume) * R.STEEL / 1000.0, 6),
                             color=[215, 222, 230, 255], mass_basis="304 SS sleeves, mesh volume x 7.85 g/cm3",
                             material=f"304 stainless tube dia {2 * SLEEVE_R:g} / {2 * SLEEVE_BORE_R:g}, pressed into the shell boss, outer end "
                                      "countersunk 90 deg (hand countersink to the OD) so the screw head bears on it",
                             roll_flip="generated at the posts by thigh_shell.py post_screws"))
    return rows, rep


def _outer_y(th, zz, Ro, x, z, sg):
    """y of the outer skin at (x, z) on the front (sg -1) or back (sg +1) side."""
    k = int(np.argmin(abs(zz - z))); X = CENTRE[0] + Ro[k] * np.cos(th); Y = CENTRE[1] + Ro[k] * np.sin(th)
    side = (Y - CENTRE[1]) * sg > 0
    o = np.argsort(X[side])
    return float(np.interp(x, X[side][o], Y[side][o]))


def _ty(x, y, z):
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]; T[:3, 3] = [x, y, z]   # cylinder axis z -> y
    return T


def _outer_hull(th, zz, Ro):
    """Closed outer solid of the shell (bosses are trimmed to it)."""
    pts = [np.c_[CENTRE[0] + Ro[k] * np.cos(th), CENTRE[1] + Ro[k] * np.sin(th), np.full(len(th), zz[k])] for k in range(len(zz))]
    return _loft_solid(pts)


def _loft_solid(rings):
    import trimesh
    V = np.vstack(rings); n = len(rings[0]); m = len(rings); F = []
    for k in range(m - 1):
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]
    c0 = len(V); c1 = c0 + 1
    V = np.vstack([V, rings[0].mean(0), rings[-1].mean(0)])
    for j in range(n):
        F += [[c0, (j + 1) % n, j], [c1, (m - 1) * n + j, (m - 1) * n + (j + 1) % n]]
    t = trimesh.Trimesh(V, np.array(F), process=True)
    if t.volume < 0:
        t.invert()
    return t


# ------------------------------------------------------------------------------------------------ shin back cover
# Owner 2026-11-18 (knee to 110 deg with the back of the thigh covered): the lower leg's back hard cover gives way instead.
# Everything fixed to the thigh that reaches behind the knee (this shell, the knee motor, the Stage 4 bracket and its knee
# screws), swept over KNEE_RANGE into the shin frame, is cleared from the cover by one plane along the knee axis
# z = c + q y (shin frame): its top-back edge becomes a straight line rising toward the hinge. q is chosen for the least
# cover removed; CLEAR along the plane normal.
SHIN_COVER = "R_shin_back_hard_cover"
THIGH_SIDE = ("R_thigh_shell_front", "R_thigh_shell_back", "R_knee_ST3025_case", "R_thigh_roll_flip",
              "R_knee_ST3025_PA2x8_0", "R_knee_ST3025_PA2x8_1", "R_knee_ST3025_PA2x8_lat0", "R_knee_ST3025_PA2x8_lat1")

# Owner 2026-11-18 ("move the mounting points elsewhere"): the cut takes the cover's two M3 screws (z 102, into nuts in the
# frame's back plate). New mounts: the cover's side wings sit ~6 mm outboard of the shin frame's side walls, which are
# solid 7 mm (medial) / 9 mm (lateral) at y 9.8..14.3, z 73..91, right behind the ankle motor. Per side two PA2x8
# self-tapping screws along x, head recessed in a counterbore on the cover's outer skin, through a boss that bridges the
# gap (0.2 mm short of the frame), biting into a pilot hole drilled in the frame wall. Clear of the ankle-motor screw hole
# at (y 10.9, z 84.7, lateral) and (y 10, z 89), and of the ankle coupler (x 24.8 .. 79.8, below z 71.5). (y, z) right leg, shin frame.
MOUNT_YZ = ((12.0, 74.5), (12.0, 81.0))
BOSS_D, HOLE_D, CB_D, CB_DEPTH = 6.5, 2.3, 4.0, 1.5     # cover boss, clearance hole, counterbore for the PA2 head (3.5 x 1.3)
PILOT_D, PILOT_DEPTH, BOSS_GAP = 1.7, 4.5, 0.2          # pilot hole in the frame wall for a PA2 self-tapper
PA2_L, PA2_HD, PA2_HH, PA2_D, PA2_PT = 8.0, 3.5, 1.3, 2.0, 0.6
FRAME = "R_shin_combined"


def _cyl_x(x0, x1, y, z, d):
    import manifold3d as M
    a, b = min(x0, x1), max(x0, x1)
    return M.Manifold.cylinder(b - a, d / 2, d / 2, 64).rotate([0.0, 90.0, 0.0]).translate([a, y, z])


def _hits_x(m, y, z, x0):
    """x of every surface crossing of the line (y, z) || x, sorted from the x0 side inward."""
    loc, _, _ = m.ray.intersects_location([[x0, y, z]], [[1.0 if x0 < 50 else -1.0, 0, 0]])
    return sorted(loc[:, 0], key=lambda v: abs(v - x0))


SINK = 0.4                        # the boss starts this far under the cover's outer skin (wall 1.25 along x at least)


def _skin_heightfield(cov, y, z, x0, d, rad, end, n=17):
    """Closed solid outboard of the cover's outer skin (shifted SINK inward) over a square of half-side rad about (y, z):
    subtracting it from a boss cylinder lets the boss start just under the slanted skin. Returns it and the outermost x."""
    import trimesh
    g = np.linspace(-rad, rad, n)
    def ox(a, b):
        h = _hits_x(cov, y + a, z + b, x0)
        return h[0] + d * SINK if h else np.nan
    X = np.array([[ox(a, b) for b in g] for a in g])
    if np.isnan(X).any():                 # beyond the wing's front edge (y 8.3): carry the nearest skin value out
        from scipy.ndimage import distance_transform_edt
        ii = distance_transform_edt(np.isnan(X), return_distances=False, return_indices=True)
        X = X[tuple(ii)]
    far = X.min() - 6.0 if d > 0 else X.max() + 6.0
    V, F = [], []
    idx = lambda i, j, k: k * n * n + i * n + j  # noqa: E731
    for k, xx in enumerate((X, np.full_like(X, far))):
        for i, a in enumerate(g):
            for j, b in enumerate(g):
                V.append([xx[i, j], y + a, z + b])
    for k in (0, 1):
        for i in range(n - 1):
            for j in range(n - 1):
                a_, b_, c_, e_ = idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k)
                F += [[a_, b_, c_], [a_, c_, e_]] if k == 0 else [[a_, c_, b_], [a_, e_, c_]]
    rim = [(i, 0) for i in range(n - 1)] + [(n - 1, j) for j in range(n - 1)] + [(i, n - 1) for i in range(n - 1, 0, -1)] + \
          [(0, j) for j in range(n - 1, 0, -1)]
    for t in range(len(rim)):
        (i0, j0), (i1, j1) = rim[t], rim[(t + 1) % len(rim)]
        F += [[idx(i0, j0, 0), idx(i1, j1, 1), idx(i1, j1, 0)], [idx(i0, j0, 0), idx(i0, j0, 1), idx(i1, j1, 1)]]
    tm = trimesh.Trimesh(np.array(V), np.array(F), process=True)
    trimesh.repair.fix_normals(tm)
    assert tm.is_watertight and tm.volume > 0, "skin heightfield not closed"
    xs = X.min() if d > 0 else X.max()
    return kit.mf(tm), xs


def cover_mounts(cov, c, q, body):
    """Bosses + counterbored holes on the cut cover (right leg); returns the cover and the mount sites."""
    import manifold3d as M
    fr = body.mesh(FRAME)
    sites = []
    m = kit.mf(cov)
    for y, z in MOUNT_YZ:
        assert z < c + q * (y + BOSS_D), "mount above the cut"
        for side, x0 in (("medial", 0.0), ("lateral", 120.0)):
            d = 1.0 if x0 < 50 else -1.0                      # screw direction, from the cover skin into the frame
            face = _hits_x(fr, y, z + 2.5, x0)                 # beside the pilot hole (the frame may already be drilled)
            ff, fin = face[0], face[1]
            end = ff - d * BOSS_GAP
            # the side wing is slanted (~20 deg about z): the boss starts SINK under the outer skin, following it
            hf, xs = _skin_heightfield(cov, y, z, x0, d, BOSS_D / 2 + 0.3, end)
            m = m + (_cyl_x(xs - d * 3.0, end, y, z, BOSS_D) - hf)
            wall = [d * (h[1] - h[0]) for h in (_hits_x(cov, y + r * np.cos(a), z + r * np.sin(a), x0)
                                                for r in (0.0, BOSS_D / 2) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False))]
            o_head = [_hits_x(cov, y + r * np.cos(a), z + r * np.sin(a), x0)[0]
                      for r in (0.0, CB_D / 2) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
            xo = _hits_x(cov, y, z, x0)[0]
            head_top = max(o_head, key=lambda v: d * v) + d * 0.2           # head wholly under the skin
            m = m - _cyl_x(xs - d * 4.0, end + d * 1.0, y, z, HOLE_D)
            m = m - _cyl_x(xs - d * 4.0, head_top + d * PA2_HH, y, z, CB_D)
            tip = head_top + d * (PA2_HH + PA2_L)
            ring_wall = []                                     # solid frame wall around the pilot (no other hole there)
            for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                h = np.array(_hits_x(fr, y + (PILOT_D / 2 + 1.0) * np.cos(a), z + (PILOT_D / 2 + 1.0) * np.sin(a), x0))
                k = [(d * (h[i] - ff), d * (h[i + 1] - ff)) for i in range(0, len(h) - 1, 2)]
                ring_wall.append(sum(max(0.0, min(b_, PILOT_DEPTH + 0.5) - max(a_, -0.1)) for a_, b_ in k))
            sites.append(dict(y=y, z=z, side=side, dir=d, cover_outer_x=round(float(xo), 2),
                              skin_slant_across_boss_mm=round(float(np.ptp([_hits_x(cov, y + BOSS_D / 2 * np.cos(a), z + BOSS_D / 2 * np.sin(a), x0)[0]
                                                                        for a in np.linspace(0, 2 * np.pi, 16)])), 2),
                              wall_thickness_along_x_min_mm=round(float(min(wall)), 2),
                              boss_to_x=round(float(end), 2), boss_sunk_under_skin_mm=SINK,
                              counterbore_depth_at_centre_mm=round(float(d * (head_top - xo)) + PA2_HH, 2),
                              head_top_x=round(float(head_top), 2), frame_face_x=round(float(ff), 2),
                              frame_wall_inner_x=round(float(fin), 2), screw_tip_x=round(float(tip), 2),
                              thread_engagement_mm=round(float(d * (tip - ff)), 2),
                              pilot_bottom_to_wall_inner_mm=round(float(d * (fin - ff) - PILOT_DEPTH), 2),
                              solid_around_pilot_min_mm=round(float(min(ring_wall)), 2),
                              screw_tip_to_pilot_bottom_mm=round(float(PILOT_DEPTH - d * (tip - ff)), 2)))
    out = kit.tm(m)
    return trimesh_clean(out), sites


def trimesh_clean(m):
    import trimesh
    parts = [p_ for p_ in m.split(only_watertight=False) if abs(p_.volume) > 1.0]
    return trimesh.util.concatenate(parts)


def frame_pilots(man, sites):
    """Pilot holes drilled into the shin frame's STEP (both legs; left = mirror x); manifest rows by name."""
    import trimesh
    from build123d import Location, Solid, Plane, export_step, export_stl, import_step, Compound
    rows, rep = {}, {}
    entry = {p["name"]: p for p in man["parts"]}
    for s, sx in (("R", 1.0), ("L", -1.0)):
        n = s + FRAME[1:]
        p = entry[n]
        src = p.get("pilot_source_step") or p["native_step"]
        off = p.get("pilot_source_offset_mm") or p.get("native_step_offset_mm") or [0.0, 0.0, 0.0]
        sh = import_step(str(R.F.src(src))).moved(Location(tuple(float(v) for v in off)))
        v0 = sh.volume
        for st in sites:
            d = st["dir"] * sx
            x0 = sx * st["frame_face_x"] - d * 0.5
            cyl = Solid.make_cylinder(PILOT_D / 2, PILOT_DEPTH + 0.5, Plane(origin=(x0, st["y"], st["z"]), z_dir=(d, 0, 0)))
            sh = sh - cyl
        sh = Compound(children=list(sh.solids())) if hasattr(sh, "solids") else sh
        f = OUT / f"{n}.step"; export_step(sh, str(f))
        fs = OUT / f"{n}.stl"; export_stl(sh, str(fs), tolerance=0.02, angular_tolerance=0.2)
        tm = trimesh.load(fs)
        bb = sh.bounding_box()
        rows[n] = dict(native_step=str(f.relative_to(R.ROOT)), native_step_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                       native_step_offset_mm=[0.0, 0.0, 0.0], pilot_source_step=src, pilot_source_offset_mm=off,
                       mesh=str(fs.relative_to(R.ROOT)), mesh_sha256=hashlib.sha256(fs.read_bytes()).hexdigest(), mesh_units="mm",
                       bounds_mm=[[round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)], [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]],
                       mass_kg=round(p["mass_kg"] * sh.volume / v0, 6) if not p.get("pilot_source_step") else round(p["mass_kg"], 6),
                       roll_flip=f"{len(sites)} pilot holes dia {PILOT_D} x {PILOT_DEPTH} for the shin back cover's PA2x8 side screws")
        rep[n] = dict(volume_before_mm3=round(v0, 1), volume_after_mm3=round(sh.volume, 1), mesh_watertight=bool(tm.is_watertight))
    return rows, rep


def mount_screws(sites):
    """PA2x8 proxies (PA2 family envelope as the knee screws), axis along x, right leg and its mirror."""
    import trimesh
    prof = [(0.0, 0.0), (PA2_HD / 2, 0.0), (PA2_HD / 2, PA2_HH), (PA2_D / 2, PA2_HH), (PA2_D / 2, PA2_HH + PA2_L - PA2_PT), (0.0, PA2_HH + PA2_L)]
    proto = trimesh.creation.revolve(np.array(prof), sections=48)
    rows = []
    for s, sx in (("R", 1.0), ("L", -1.0)):
        for i, st in enumerate(sites):
            d = st["dir"] * sx
            T = np.eye(4); T[:3, :3] = [[0, 0, d], [0, 1, 0], [-d, 0, 0]]     # axis z -> d x
            T[:3, 3] = [sx * st["head_top_x"], st["y"], st["z"]]
            m = proto.copy(); m.apply_transform(T)
            if m.volume < 0:
                m.invert()
            nm = f"{s}_shin_back_cover_PA2x8_{i}"
            rows.append(dict(name=nm, physical_part_id=nm, bom_group=f"{s}_shin_back_cover_PA2x8", owning_link=f"{s}_shin",
                             role="fastener_proxy", **R.save_npz(nm, m), volume_mm3=round(float(m.volume), 3),
                             mass_kg=round(float(m.volume) * R.STEEL / 1000.0, 7),
                             material="PA2x8 self-tapping, through the shin back cover's side boss into a dia 1.7 pilot hole in "
                                      "the shin frame's side wall (PA2 family envelope: head dia 3.5 x 1.3, shank dia 2, 0.6 mm point)",
                             color=[195, 204, 214, 255], display_group="mechanism", display_groups=["mechanism"],
                             overlay_id="actuators_and_hardware", parent_assembly="complete_fdm_ape_body_assembly",
                             inspection_reference=False, mass_basis="mesh volume x 7.85 g/cm3",
                             roll_flip=f"shin back cover side mount, {st['side']} wall (y {st['y']}, z {st['z']})"))
    return rows


def shin_cover():
    import manifold3d as M
    import trimesh
    t0 = time.time()
    man = json.loads(R.VARIANT.read_text())
    entry = {p["name"]: p for p in man["parts"]}
    body = kit.base(R.VARIANT)
    src = entry[SHIN_COVER].get("trim_source_mesh") or entry[SHIN_COVER]["mesh"]
    d = np.load(R.ROOT / src) if src.endswith(".npz") else None
    cov_tm = trimesh.Trimesh(d["vertices"], d["faces"], process=True) if d is not None else trimesh.load(R.ROOT / src)
    cov = kit.mf(cov_tm)
    side = M.Manifold.batch_boolean([kit.mf(body.mesh(n) if "shell" in n else R.closed(body, n)) for n in THIGH_SIDE
                                     if n in body.parts], M.OpType.Add)
    sweep = M.Manifold.batch_boolean([side.transform(np.linalg.inv(_rel(body, "R_shin", dict(R_knee=float(k))))[:3, :4])
                                      for k in np.arange(KNEE_RANGE[0], KNEE_RANGE[1] + 0.1, 2.5)], M.OpType.Add)
    hit = cov ^ sweep
    before = hit.volume()
    P = kit.tm(hit).vertices
    print(f"thigh side swept into the shin frame ({time.time() - t0:.0f}s): {before:.0f} mm3 of the cover in the way", flush=True)
    best = None
    for q in np.arange(-3.0, 0.01, 0.1):
        c = float(np.min(P[:, 2] - q * P[:, 1] - CLEAR * np.sqrt(1 + q * q)))
        above = plane_cutter((c, 0.0, q))                       # half-space BELOW z = c + q y ...
        kept = cov ^ above                                       # ... is what the cover keeps
        if best is None or kept.volume() > best[2]:
            best = (q, c, kept.volume(), kept)
    q, c, vol, kept = best
    mm = kit.tm(kept)
    parts = [p_ for p_ in mm.split(only_watertight=False) if abs(p_.volume) > 1.0]
    mm = trimesh.util.concatenate(parts)
    old = [n for n in body.parts if n.startswith("R_Shin_carrier_shell_shin_back") and n.endswith("M3x8")]
    old_mounts = {n: round(float(np.mean(np.array(body.parts[n]["bounds_mm"])[:, 2]) - (c + q * np.mean(np.array(body.parts[n]["bounds_mm"])[:, 1]))), 1)
                  for n in old}
    mm, sites = cover_mounts(mm, c, q, body)
    frame_rows, frame_rep = frame_pilots(man, sites)
    screw_rows = mount_screws(sites)
    out = {}
    for sd, msh in (("R", mm), ("L", kit.mirror_x(mm))):
        nm = sd + SHIN_COVER[1:]
        f = OUT / f"{nm}.stl"; msh.export(f)
        out[nm] = dict(mesh=str(f.relative_to(R.ROOT)), mesh_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                       bounds_mm=[np.round(msh.bounds[0], 2).tolist(), np.round(msh.bounds[1], 2).tolist()],
                       volume_mm3=round(float(msh.volume), 1))
    for p in man["parts"]:
        if p["name"] in out:
            p.setdefault("trim_source_mesh", p["mesh"]); p.setdefault("trim_source_sha256", p["mesh_sha256"])
            p.update(mesh=out[p["name"]]["mesh"], mesh_sha256=out[p["name"]]["mesh_sha256"], mesh_units="mm",
                     bounds_mm=out[p["name"]]["bounds_mm"], volume_mm3=out[p["name"]]["volume_mm3"],
                     roll_flip="top-back cut along the knee axis to clear the thigh over the knee range (thigh_shell.py shin-cover)")
            p.pop("native_step", None); p.pop("native_step_sha256", None)
            p["roll_flip"] += "; mounted by 4 PA2x8 through side bosses into the shin frame's side walls (shin_cover_mounts)"
        if p["name"] in frame_rows:
            p.update(frame_rows[p["name"]])
    gone = [p["name"] for p in man["parts"] if p["name"][2:].startswith("Shin_carrier_shell_shin_back")]
    new_names = {r["name"] for r in screw_rows}
    man["parts"] = [p for p in man["parts"] if p["name"] not in gone and p["name"] not in new_names
                    and not p["name"][2:].startswith("shin_back_cover_PA2")] + screw_rows
    R.VARIANT.write_text(json.dumps(man, indent=1) + "\n")
    rep = dict(plane_shin_frame=dict(c=round(c, 2), q=round(q, 2), note="cover kept below z = c + q y (shin frame, mm)"),
               knee_range_deg=list(KNEE_RANGE), in_the_way_before_mm3=round(before, 1),
               volume_before_mm3=round(cov.volume(), 1), volume_after_mm3=round(float(mm.volume), 1),
               top_edge_z_at_y={f"{y}": round(c + q * y, 1) for y in (8.3, 15, 20, 25, 29.4)},
               old_mount_screw_centre_above_cut_mm=old_mounts, removed_parts=gone,
               mounts=dict(sites=sites, frame=frame_rep, screws=[r["name"] for r in screw_rows]),
               fragments=len(parts), parts=out, seconds=round(time.time() - t0, 1))
    R.merge("shin_back_cover_trim", rep)
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------------------------------------ checks
def torso_shell(n, p):
    """Core (torso) shell parts: center_front_shell, center_back_shell and their fittings, chest indicator, cowl."""
    return n.startswith("center_") or (p.get("owning_link") == "torso" and p.get("role") == "body_shell")


def check():
    import itertools
    t0 = time.time()
    body = kit.base(R.VARIANT)
    shells = [n for n in body.parts if n[2:] in ("thigh_shell_front", "thigh_shell_back")]
    others = {n: p for n, p in kit._hip_parts(body).items()
              if p["owning_link"] not in ("R_roll_link",) and n not in shells and not n.startswith("L_thigh_shell")}
    others.update({n: body.parts[n] for n in body.parts if n.startswith("L_thigh_shell")})
    # the static core (torso) shell is cut at z 271 for now and redesigned next (owner D-12, 2026-11-17): left out of the check
    excluded = sorted(n for n, p in others.items() if torso_shell(n, p))
    others = {n: p for n, p in others.items() if n not in excluded}
    # exact pre-filter: the poses move only the right leg, so every other part is static; the shell stays inside the ball
    # about C (yaw / pitch / roll axes) that holds it at standing, so a static part wholly outside that ball can never touch
    # it (keeps the check light: closing far torso / arm / head shells cost > 6 GB)
    C = np.array([59.0, 2.0, 219.0])
    def bound(v):   # |q(p) - C| <= |p - A| + |A - C|, A = p's foot on the roll axis (y through x 56, z 190): roll turns about A
        A = np.c_[np.full(len(v), B.RF_AX), v[:, 1], np.full(len(v), B.RF_AZ)]
        return float((np.linalg.norm(v - A, axis=1) + np.linalg.norm(A - C, axis=1)).max())
    reach = max(bound(body.mesh(n).vertices) for n in shells if n.startswith("R_"))
    moving = {"R_yaw_link", "R_pitch_link", "R_shin", "R_ankle_pitch", "R_foot", "R_ankle_roll"}
    def far(p):
        b = np.array(p["bounds_mm"]); return float(np.linalg.norm(np.clip(C, b[0], b[1]) - C)) > reach + 8.0   # margin: left-leg standing roll -2.68 deg vs neutral bounds
    others = {n: p for n, p in others.items() if p["owning_link"] in moving or not far(p)}
    print(f"{len(others)} parts within reach {reach:.1f} mm of C or on the moving right leg", flush=True)
    quick = "--quick" in sys.argv          # owner waiting (2026-11-17): decimated meshes, the 151 realistic poses only

    def dec(m, faces):
        if not quick or len(m.faces) <= faces:
            return m
        d = m.simplify_quadric_decimation(face_count=faces)
        return d if kit.mf(d).status().name == "NoError" else m
    mfs = {n: kit.mf(dec(body.mesh(n), 8000)) for n in shells}
    om = {}
    for n in others:
        try:
            om[n] = kit.mf(dec(R.closed(body, n), 3000))
        except Exception:  # noqa: BLE001
            pass
    link = {n: body.parts[n]["owning_link"] for n in list(mfs) + list(om)}
    R_sh = [n for n in shells if n.startswith("R_")]

    def overlaps(q):
        T = {l: body.link_T(l, q)[:3, :4] for l in set(link.values())}
        res = {}
        for a in R_sh:
            pa = mfs[a].transform(T[link[a]]); ba = np.array(pa.bounding_box()).reshape(2, 3)
            for b, g in om.items():
                pb = g.transform(T[link[b]]); bb = np.array(pb.bounding_box()).reshape(2, 3)
                if np.any(ba[1] < bb[0]) or np.any(bb[1] < ba[0]):
                    continue
                v = (pa ^ pb).volume()
                if v > 0.5:
                    res[(a, b)] = v
        return res
    q0 = {**body.standing, **dict(zip(R.J, (0.0,) * 4))}
    base = overlaps(q0)
    fam = {}
    import resource
    for i, (f, p, _) in enumerate(list(R.library_poses()) + ([] if quick else [("all combinations", g, True) for g in R.GRID])):
        if i % 10 == 0:
            print(f"pose {i}: {time.time() - t0:.0f} s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB", flush=True)
        o = overlaps({**body.standing, **dict(zip(R.J, map(float, p)))})
        for k, v in o.items():
            d = v - base.get(k, 0.0)
            if d > 2.0:
                r = fam.setdefault(f, {}).setdefault(f"{k[0]} x {k[1]}", dict(max_new_mm3=0.0, poses_over_20mm3=0, worst_pose=None))
                r["poses_over_20mm3"] += d > 20.0
                if d > r["max_new_mm3"]:
                    r["max_new_mm3"], r["worst_pose"] = round(d, 1), [round(float(x), 1) for x in p]
    # wall offset from the covered parts: distance from inner-surface samples of the shell to the covered parts
    import trimesh
    cov = trimesh.util.concatenate([body.mesh(n) for n in COVERED if n in body.parts])
    # distances by KD-tree over dense surface samples of the covered parts (trimesh closest_point took > 6 GB);
    # error <= the sample spacing, reported as offset_resolution_mm
    from scipy.spatial import cKDTree
    S = np.vstack([cov.vertices, trimesh.sample.sample_surface(cov, 600000)[0]])
    tree = cKDTree(S)
    spacing = float(np.sqrt(cov.area / 600000))
    offs = {"offset_resolution_mm": round(spacing, 2)}
    for n in R_sh:
        m = body.mesh(n)
        P, fi = trimesh.sample.sample_surface(m, 20000)
        inward = -m.face_normals[fi]
        facing = np.einsum("ij,ij->i", inward, (np.c_[np.full(len(P), CENTRE[0]), np.full(len(P), CENTRE[1]), P[:, 2]] - P)) > 0
        P = P[facing & (P[:, 2] > 150.0)]                      # inner wall above the knee end
        d = tree.query(P)[0]
        offs[n] = dict(min_mm=round(float(d.min()), 2), p10_mm=round(float(np.percentile(d, 10)), 2),
                       median_mm=round(float(np.median(d)), 2), share_within_3mm_pct=round(100 * float((d <= 3.0).mean()), 1))
    rep = dict(excluded_torso_shell_parts=excluded, standing_overlap_mm3={f"{a} x {b}": round(v, 1) for (a, b), v in base.items()},
               new_overlap_by_family=fam, inner_wall_offset_from_covered_parts=offs,
               poses=dict(library_incl_paths=len(R.library_poses()), range_combinations=0 if quick else len(R.GRID)),
               quick=quick,
               seconds=round(time.time() - t0, 1),
               method="closed meshes (manifold3d exact mesh booleans); shells vs every hip-region part not on R_roll_link "
                      "(left leg, shin covers, battery cradle and cells, arms; the core torso shell is excluded, D-12); new overlap vs standing")
    R.merge("thigh_shell_check_quick" if quick else "thigh_shell_check", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "new_overlap_by_family"}, indent=1))
    for f, rows in fam.items():
        for k, r in rows.items():
            print(f"  {f:30s} {k:70s} {r['max_new_mm3']:8.1f} mm3 ({r['poses_over_20mm3']}) {r['worst_pose']}")


if __name__ == "__main__":
    {"build": build, "check": check, "shin-cover": shin_cover, "fit-surface": fit_surface}[sys.argv[1]]()
