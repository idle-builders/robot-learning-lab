"""Shell pass, step 1: openings cut by the motion they have to allow (Idle T-czv2d8).

The torso shells (belly and back) lose exactly what the legs sweep through over the target hip ranges, and the boot
shell loses what the shin and ankle sweep through over the target ankle ranges, each with CLEARANCE mm. The cut-outs
are therefore as small as the motion allows, and the shell that remains around the pelvis is the 'briefs'.

Targets (owner intent: ranges are what the body allows; hip internals allow flexion to -100 and roll to +30):
  hip pitch -90..+40, hip roll -45..+25, hip yaw -40..+40 (grid of combined poses);
  ankle pitch -35..+27.5, ankle roll -25..+25.

Method (voxel sweep; mesh-union sweeps fragment into millions of triangles on near-coplanar copies):
  - each moving link's parts are voxelized at PITCH mm in the link's neutral frame and grown by CLEARANCE with a
    distance transform;
  - a PITCH grid over the target shells' region is looked up in every link grid at every pose; any hit marks the
    point as swept;
  - the swept mask is smoothed and turned into one surface (marching cubes), which is subtracted from each shell once.
Shells stay mesh-defined until the CAD pass.

Boot shells (Idle T-6c2e93): the boot came from a closed skin whose floor and lower rim sat inside the foot monocoque's
sole plate (about 10.6 cm3 per side at standing) and dipped into the foot-roll servo case, so no order or direction could
put it on. The same subtraction therefore also takes the foot link's own parts (screws included, body shells left out)
grown by BOOT_STATIC_CLEARANCE, and the boot is then split at y = BOOT_SPLIT_Y (the rear edge of the ankle collar) into
the toe part ({s}_foot_boot_shell, slides on from the toe along +y) and the heel cap ({s}_foot_boot_heel_shell, slides on
from behind along -y). Both go on last, after the foot screws are driven; detached voxel crumbs are dropped.

Boot retention (Idle T-jh7fcm): each half is held by one PA2x8 along its slide direction at its far end (toe tip, heel), its
head on an ISO 7089 washer in a counterbore under the skin (countersunk mouth at the toe), through a plain clearance hole in an inward tab that seats on a
monocoque boss (BOOT_* site constants below; catalogue hardware only, no sleeve). The bosses join the boot's static keep-out here;
foot_screws.py builds them on the monocoque from manifest.shell_pass.boot.screws. Washer and screw rows are added to the manifest
({s}_foot_boot_{toe|heel}_M2_washer, {s}_foot_boot_{toe|heel}_PA2x8).
"""
from __future__ import annotations

import itertools

import numpy as np
import trimesh

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("ape-proportions-shells")   # configured values: robot/body/ape-proportions/shells.bom.yaml config:
HW_RHO = BOM_CONFIG.density_g_mm3(CFG["HARDWARE_MATERIAL"]) * 1e-3   # kg/mm3 of the steel screws and washers this pass adds
HW_BASIS = f"mesh volume x {HW_RHO * 1e6:g} g/cm3 ({CFG['HARDWARE_MATERIAL']}, robot/body/ape-proportions/shells.bom.yaml config HARDWARE_MATERIAL)"

CLEARANCE = CFG["CLEARANCE"]
PITCH = CFG["PITCH"]
HIP_FLEX = float(__import__("os").environ.get("APE_HIP_FLEX", -90.0))  # deepest hip flexion the openings allow
HIP = dict(pitch=np.arange(HIP_FLEX, 40.1, 5.0), roll=(-45.0, -30.0, -15.0, 0.0, 10.0, 20.0, 25.0),
           yaw=(-40.0, -20.0, 0.0, 20.0, 40.0))
HIP_LINKS = CFG["HIP_LINKS"]
TORSO_SHELLS = CFG["TORSO_SHELLS"]
ANKLE = dict(pitch=np.arange(-35.0, 27.6, 2.5), roll=np.arange(-25.0, 25.1, 5.0))
ANKLE_LINKS = CFG["ANKLE_LINKS"]
SKIP = ("cut_material", "adhesive_allocation", "authored_harness")
BOOT_STATIC_CLEARANCE = CFG["BOOT_STATIC_CLEARANCE"]
                              # eat part of it: measured 2026-11-23 (review rv-6c2e93) min 0.13 mm heel cap to monocoque, 0.36 mm toe part
BOOT_SPLIT_Y = CFG["BOOT_SPLIT_Y"]
                              # measured 2026-11-23 on the selected body: both halves free along their paths for a split between
                              # y 14.8 and 16.3 (below it the heel cap meets the ankle coupler, above it the toe part the monocoque)
BOOT_SPLIT_GAP = CFG["BOOT_SPLIT_GAP"]
                              # (toe face y -63.25, heel face y 44.35), so the seam takes the length error of the monocoque between the
                              # boss faces (107.6 mm) and of both halves (78.7 + 28.8 mm): about +-0.3 mm in FDM (review rv-jh7fcm); 0.2
                              # could close before a tab seats, 0.4 keeps both halves' ends inside the split window above
BOOT_MIN_BODY_MM3 = CFG["BOOT_MIN_BODY_MM3"]
SEALED_SPACING = 0.15 * PITCH  # surface sample spacing that leaves no voxel of a face unsampled, so the fill closes
# Boot retention (Idle T-jh7fcm): each half is held by ONE PA2x8 ISO 7380-1 (the shin back cover's screw, one BOM line) along its
# slide direction, at its far end: the toe part at the toe tip (screw +y, driven from the front), the heel cap at the heel
# (screw -y, driven from behind). Catalogue hardware only (owner 2026-11-22: no custom sleeves or rings): the head stands on an
# ISO 7089 M2 washer 2.2 / 5 x 0.3 (the body's line hw:M2-washer-2.2-5-x-0.3) in a dia BOOT_CB_D counterbore under the skin, on
# an inward printed tab with a plain BOOT_HOLE_D clearance hole (ISO 273 medium); the tab grows from the skin to the boss face
# (clamped faces meet at 0.00) and fills the toe tip / heel up to the skin above it (no slot between tab and skin). The screw
# threads into a dia 1.7 pilot in a small boss on the foot monocoque's plate (robot/body/leg/foot_screws.py builds it from
# manifest.shell_pass.boot.screws). Joint chosen by measurement (2026-11-23): a bare dk 3.5 head on the 2.4 hole bears on a
# 0.55 mm ring (5.1 mm2, under one 0.62 mm line); the washer puts 15.1 mm2 (2.96 x) on the tab, as on the neck cradle (T-acpw7j).
# Stack (point 0.5 as robot/body/checks/data/hardware.json): tab BOOT_TAB_T 2.9 + washer 0.3 -> 4.8 of the 8 mm screw in the
# boss, engagement 4.3 (4.1 at tab +0.2, reference 2 d = 4.0); the tip ends 4.8 in a 6.0 pilot, 5.3 at tab -0.2 without the
# washer (the joint does not need the 0.3 mm washer to close: 0.7 mm tip margin left). Sites: right foot, manifest frame,
# left = mirror x. Toe: axis z 9.7 so the tab's underside keeps 0.3 mm over the monocoque plate (top z 5.4 under the tab) with
# 1.3 mm of tab under the dia 5.4 counterbore; face_y -63.25 puts the head top 4.5 mm out from the face, under the toe skin all
# round the counterbore (the skin slopes back to y -68.1 at the counterbore's top rim). Heel: face_y = the plate's rear edge
# (y 44.35). The toe screw sits at x 52, not on the toe's frontmost point (x 54-55): there its boss stood in the driver path
# (blade dia 3) of the roll-case screw R_B13_rear_PA2x6_1 (axis x 58.7, driver_access 2026-11-23); at x 52 the boss ends 1.2 mm
# short of that path.
BOOT_SCREW = dict(label="PA2 x 8 ISO 7380-1", length=8.0, head_d=3.5, head_h=1.3, shank_d=2.0, point=0.6)
BOOT_SCREW_POINT = CFG["BOOT_SCREW_POINT"]
BOOT_WASHER = dict(label="M2 washer 2.2/5 x 0.3", standard="ISO 7089", d1=2.2, d2=5.0, h=0.3)
BOOT_HOLE_D = CFG["BOOT_HOLE_D"]
BOOT_TAB_T = CFG["BOOT_TAB_T"]
BOOT_TAB_TOL = CFG["BOOT_TAB_TOL"]
BOOT_CB_D = CFG["BOOT_CB_D"]
BOOT_SITES = (dict(half="toe", x=52.0, z=9.7, dir=1.0, face_y=-63.25),
              dict(half="heel", x=50.5, z=8.0, dir=-1.0, face_y=44.35))
BOOT_BOSS_HW = CFG["BOOT_BOSS_HW"]
BOOT_TAB_HW, BOOT_TAB_HH = CFG["BOOT_TAB_HW"], CFG["BOOT_TAB_HH"]
                                      # its side walls (at 4.0 the walls beside a 0.8-1.1 toe countersink came to 0.7-1.0 mm); tab depth below the axis (z):
                                      # r 2.7 + 1.3 wall; the tab runs up to the skin above it (fills the toe tip / heel, no slot)
# Countersink at the counterbore mouth (review rv-jh7fcm): the toe skin slopes back over the bore (about 50 deg in the x = axis section),
# so the plain dia 5.4 bore left a feather edge at the top of its rim (2.7 / 4.2 mm2 under one 0.62 mm line R / L, down to
# 0.003 mm). A 90 deg countersink whose axis is the normal of the skin round the rim (plane fitted to the skin crossings at the
# counterbore radius) meets the skin at 135 deg all round and the bore at an obtuse angle; it stays BOOT_CSK_SEAT_KEEP outward of the
# washer seat, so seat and stack do not change. Measured 2026-11-23 on the pre-tab halves (reviewer's local ray-cast, rim within
# r 6 mm): chamfer 0.5 / 0.7 / 0.9 / 1.0 / 1.1 -> thinnest rim wall R 0.61 / 0.80 / 1.02 / 0.98 / 0.90, L 0.48 / 0.68 / 0.82 / 0.92 /
# 1.03 mm (tab half width 4.6); the rest under 2 lines is the roof between the bore top and the skin above it, which rises at about
# 50 deg. Heels: nothing under one line without it, and a countersink adds nothing measurable there (left plain).
BOOT_CSK_HALF_ANGLE = CFG["BOOT_CSK_HALF_ANGLE"]
BOOT_CSK_W = dict(toe=1.0, heel=0.0)   # mm of chamfer in the skin plane beyond the bore's long axis there (dia ~8 at the skin); 0 = none
BOOT_CSK_SEAT_KEEP = CFG["BOOT_CSK_SEAT_KEEP"]
BOOT_BOSS_LEN, BOOT_BOSS_TOP, BOOT_BOSS_Z0 = CFG["BOOT_BOSS_LEN"], CFG["BOOT_BOSS_TOP"], CFG["BOOT_BOSS_Z0"]
BOOT_PILOT = CFG["BOOT_PILOT"]
BOOT_SKIN_SINK = CFG["BOOT_SKIN_SINK"]


def boot_site_geometry(st, sx=1.0):
    """Derived site numbers (manifest frame; sx = -1 mirrors to the left foot)."""
    d = st["dir"]
    seat = st["face_y"] - d * BOOT_TAB_T                       # the washer's tab side
    head_seat = seat - d * BOOT_WASHER["h"]
    head_top = head_seat - d * BOOT_SCREW["head_h"]
    yb = (st["face_y"], st["face_y"] + d * BOOT_BOSS_LEN)
    x = sx * st["x"]
    boss = ([x - BOOT_BOSS_HW, min(yb), BOOT_BOSS_Z0], [x + BOOT_BOSS_HW, max(yb), st["z"] + BOOT_BOSS_TOP])
    return dict(x=x, z=st["z"], dir=d, face_y=st["face_y"], seat_y=seat, head_seat_y=head_seat, head_top_y=head_top, boss_lo=boss[0],
                boss_hi=boss[1], tip_y=head_seat + d * BOOT_SCREW["length"])


def boot_stack():
    """Clamp stack of the retention screw over the tab tolerance (mm): engagement (thread past the point) and tip depth in the
    pilot, with and without the washer (the joint must not rely on the 0.3 mm washer to close)."""
    L, pt, w, T, tol, D = BOOT_SCREW["length"], BOOT_SCREW_POINT, BOOT_WASHER["h"], BOOT_TAB_T, BOOT_TAB_TOL, BOOT_PILOT[1]
    into = lambda t, ww: L - t - ww   # noqa: E731  screw length past the boss face
    return dict(tab_mm=T, tab_tol_mm=tol, washer_mm=w, screw_in_boss_mm=round(into(T, w), 3),
                engagement_mm=round(into(T, w) - pt, 3), engagement_min_mm=round(into(T + tol, w) - pt, 3),
                engagement_without_washer_mm=round(into(T, 0.0) - pt, 3), reference_engagement_mm=round(2 * BOOT_SCREW["shank_d"], 2),
                tip_depth_mm=round(into(T, w), 3), tip_depth_max_mm=round(into(T - tol, 0.0), 3), pilot_depth_mm=D,
                tip_margin_min_mm=round(D - into(T - tol, 0.0), 3))


def bearing_mm2():
    """Plastic ring the clamp load bears on (mm2) and its width (mm): washer d2 over the clearance hole."""
    ro = min(BOOT_WASHER["d2"], BOOT_CB_D) / 2
    ri = BOOT_HOLE_D / 2
    return round(float(np.pi * (ro ** 2 - ri ** 2)), 2), round(ro - ri, 3)


def _cyl_y(y0, y1, x, z, dia):
    """Cylinder along y (manifold)."""
    import manifold3d as mf
    a, b = min(y0, y1), max(y0, y1)
    return mf.Manifold.cylinder(b - a, dia / 2, dia / 2, 64).rotate([-90.0, 0.0, 0.0]).translate([x, a, z])


def _skin(m, p, axis, out):
    """Outer skin coordinate of mesh m on the line through p along `axis`, going `out` (+1 / -1): the farthest crossing within 20 mm."""
    v = np.zeros(3); v[axis] = out
    loc, _, _ = m.ray.intersects_location([list(p)], [v])
    if not len(loc):
        return np.nan
    t = out * (loc[:, axis] - p[axis])
    t = t[(t > 0) & (t < 20.0)]
    return p[axis] + out * t.max() if len(t) else np.nan


def _skin_y(m, x, z, y_in, d):
    """Outer skin y of mesh m on the line (x, z) || y: the farthest crossing out from y_in against d (the screw direction)."""
    return _skin(m, (x, y_in, z), 1, -d)


def _outside(m, axis, ga, gb, p_in, out, sink):
    """Closed solid outboard of m's outer skin (moved `sink` inward) along `axis` (going `out` from p_in) over the grid ga x gb of
    the other two axes: cut from a tab, the tab ends just under the skin, whatever its slope."""
    ia, ib = [k for k in range(3) if k != axis]
    def pt(a, b):
        q = [0.0, 0.0, 0.0]; q[ia] = a; q[ib] = b; q[axis] = p_in
        return q
    Y = np.array([[_skin(m, pt(a, b), axis, out) for b in gb] for a in ga])
    if np.isnan(Y).all():
        raise ValueError("no skin found over the boot tab")
    if np.isnan(Y).any():
        from scipy.ndimage import distance_transform_edt
        Y = Y[tuple(distance_transform_edt(np.isnan(Y), return_distances=False, return_indices=True))]
    Y = Y - out * sink
    far = (Y.max() + 6.0) if out > 0 else (Y.min() - 6.0)
    n, k = len(ga), len(gb)
    def vert(s, i, j):
        q = [0.0, 0.0, 0.0]; q[ia] = ga[i]; q[ib] = gb[j]; q[axis] = Y[i, j] if s == 0 else far
        return q
    V = [vert(s, i, j) for s in (0, 1) for i in range(n) for j in range(k)]
    idx = lambda s, i, j: s * n * k + i * k + j  # noqa: E731
    F = []
    for s in (0, 1):
        for i in range(n - 1):
            for j in range(k - 1):
                q = [idx(s, i, j), idx(s, i + 1, j), idx(s, i + 1, j + 1), idx(s, i, j + 1)]
                F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if s == 0 else [[q[0], q[2], q[1]], [q[0], q[3], q[2]]]
    rim = [(i, 0) for i in range(n - 1)] + [(n - 1, j) for j in range(k - 1)] + [(i, k - 1) for i in range(n - 1, 0, -1)] + \
          [(0, j) for j in range(k - 1, 0, -1)]
    for t in range(len(rim)):
        (i0, j0), (i1, j1) = rim[t], rim[(t + 1) % len(rim)]
        F += [[idx(0, i0, j0), idx(1, i1, j1), idx(0, i1, j1)], [idx(0, i0, j0), idx(1, i0, j0), idx(1, i1, j1)]]
    tm = trimesh.Trimesh(np.array(V, float), np.array(F), process=True)
    if tm.volume < 0:
        tm.invert()
    return _mf(tm)


def _countersink(half, g):
    """Countersink cutter at the counterbore mouth (manifold) and its record: a cone of half angle BOOT_CSK_HALF_ANGLE about the
    normal of the plane fitted to the outer skin crossings at the counterbore radius (pre-tab half), through the point where the
    screw axis meets that plane, BOOT_CSK_W wider than the bore's section in that plane; cut back to BOOT_CSK_SEAT_KEEP outward of
    the washer seat."""
    import manifold3d as mf
    x, z, d, fy = g["x"], g["z"], g["dir"], g["face_y"]
    R = BOOT_CB_D / 2
    P = []
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        y = _skin_y(half, x + R * np.cos(a), z + R * np.sin(a), fy, d)
        if np.isfinite(y):
            P.append([x + R * np.cos(a), y, z + R * np.sin(a)])
    P = np.array(P); c = P.mean(0)
    n = np.linalg.svd(P - c)[2][2]
    n = n if n[1] * d < 0 else -n                                  # outward: against the screw direction
    C = np.array([x, c[1] + ((c - [x, c[1], z]) @ n) / n[1], z])   # screw axis meets the fitted plane
    tilt = float(np.degrees(np.arccos(abs(n[1]))))
    rc = R / abs(n[1]) + BOOT_CSK_W[g["half"]]                     # bore section in that plane: ellipse, long semi-axis R / cos(tilt)
    ta = np.tan(np.radians(BOOT_CSK_HALF_ANGLE))
    h = rc / ta + 4.0                                               # apex .. 4 mm outside the skin plane
    zax = np.array([0.0, 0.0, 1.0]); v = np.cross(zax, n); sn = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rm = np.eye(3) + K + K @ K * ((1 - zax @ n) / sn ** 2) if sn > 1e-9 else np.eye(3) * np.sign(zax @ n)
    A = C - n * (rc / ta)
    cone = mf.Manifold.cylinder(h, 0.0, h * ta, 96).transform(np.hstack([Rm, A[:, None]]).tolist())
    lim = g["seat_y"] - d * BOOT_CSK_SEAT_KEEP
    keep = mf.Manifold.cube([60.0, 60.0, 60.0]).translate([x - 30.0, lim - 60.0 if d > 0 else lim, z - 30.0])
    rec = dict(countersink_axis_tilt_deg=round(tilt, 1), countersink_dia_at_skin_mm=round(2 * rc, 2),
               countersink_half_angle_deg=BOOT_CSK_HALF_ANGLE, countersink_chamfer_mm=BOOT_CSK_W[g["half"]],
               countersink_skin_point_y=round(float(C[1]), 2), countersink_apex_y=round(float(A[1]), 2))
    return cone ^ keep, rec


def _mouth_y(m, x, z, y_in, d, r):
    """Outermost material y round the bore mouth: per clock position at radius r (just outside the counterbore), the skin or
    countersink crossing seen from y_in outward."""
    v = [_skin_y(m, x + r * np.cos(a), z + r * np.sin(a), y_in, d) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    return np.array([q for q in v if np.isfinite(q)])


def boot_mount(half, st, sx):
    """Inward tab, clearance hole, washer counterbore and countersink on one boot half (mesh); returns the half, its washer and the
    site record."""
    import manifold3d as mf
    g = boot_site_geometry(st, sx)
    x, z, d, fy = g["x"], g["z"], g["dir"], g["face_y"]
    far = fy - d * 14.0
    z0, z1 = z - BOOT_TAB_HH, z + 14.0
    tab = mf.Manifold.cube([2 * BOOT_TAB_HW, abs(fy - far), z1 - z0]).translate([x - BOOT_TAB_HW, min(fy, far), z0])
    gx = np.linspace(x - BOOT_TAB_HW - 0.5, x + BOOT_TAB_HW + 0.5, 19)
    gz = np.linspace(z0 - 0.5, z1 + 0.5, max(15, int(np.ceil((z1 - z0 + 1.0) / 0.5)) + 1))
    gy = np.linspace(min(fy, far) - 0.5, max(fy, far) + 0.5, 31)
    tab = tab - _outside(half, 1, gx, gz, fy, -d, BOOT_SKIN_SINK)                 # the toe / heel skin out along the screw axis
    tab = tab - _outside(half, 2, gx, gy, z, 1.0, BOOT_SKIN_SINK)                 # and the skin above the tab
    # head under the skin: the skin's innermost crossing within the counterbore radius vs the head top
    rim = [_skin_y(half, x + r * np.cos(a), z + r * np.sin(a), fy, d) for r in (0.0, BOOT_CB_D / 2) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    rim = np.array([v for v in rim if np.isfinite(v)])
    head_under_skin = float(np.min(d * (g["head_top_y"] - rim)))   # >= 0: the head top lies inward of every skin point round it
    csk, csk_rec = _countersink(half, dict(g, half=st["half"])) if BOOT_CSK_W[st["half"]] > 0 else (None, {})
    m = _mf(half) + tab
    m = m - _cyl_y(g["seat_y"] - d * 0.01, fy + d * 1.0, x, z, BOOT_HOLE_D)   # no teardrop roof: the 2.4 / 5.4 bores bridge
    m = m - _cyl_y(g["seat_y"], g["seat_y"] - d * 12.0, x, z, BOOT_CB_D)
    if csk is not None:
        m = m - csk
    out = _tm(m)
    bodies = sorted(out.split(only_watertight=False), key=lambda q: -(q.volume if q.is_watertight else 0.0))
    out = trimesh.util.concatenate([bodies[0]] + [q for q in bodies[1:] if q.is_watertight and q.volume >= BOOT_MIN_BODY_MM3])
    # where the countersink meets the counterbore: the innermost mouth crossing just outside the bore vs the head top (< 0: the head's
    # top edge stands that far out of the plain bore into the countersink, still inward of the skin round it)
    mouth = _mouth_y(out, x, z, g["seat_y"], d, BOOT_CB_D / 2 + 0.05)
    w = BOOT_WASHER
    washer = _tm(_cyl_y(g["seat_y"], g["head_seat_y"], x, z, w["d2"]) - _cyl_y(g["seat_y"] + d, g["head_seat_y"] - d, x, z, w["d1"]))
    area, ring = bearing_mm2()
    stack = boot_stack()
    below = z - BOOT_CB_D / 2 - (z - BOOT_TAB_HH)
    rec = dict(g, side="R" if sx > 0 else "L", half=st["half"], tab_wall_below_counterbore_mm=round(below, 2),
               tab_wall_x_counterbore_mm=round(BOOT_TAB_HW - BOOT_CB_D / 2, 2), tab_wall_round_hole_mm=round(BOOT_TAB_HH - BOOT_HOLE_D / 2, 2),
               head_under_skin_min_mm=round(head_under_skin, 2), skin_crossings_round_counterbore=[round(float(rim.min()), 2), round(float(rim.max()), 2)],
               head_below_countersink_edge_min_mm=round(float(np.min(d * (g["head_top_y"] - mouth))), 2), **csk_rec,
               screw=BOOT_SCREW["label"], washer=f"{w['standard']} {w['label']}", hole_dia_mm=BOOT_HOLE_D, counterbore_dia_mm=BOOT_CB_D,
               bearing_area_mm2=area, bearing_ring_mm=ring, pilot_dia_mm=BOOT_PILOT[0], pilot_depth_mm=BOOT_PILOT[1], stack=stack,
               thread_engagement_mm=stack["engagement_mm"], tab_volume_mm3=round(float(out.volume - half.volume), 1))
    rec = {k: ([round(float(u), 3) for u in v] if isinstance(v, (list, tuple)) and v and isinstance(v[0], (float, int, np.floating)) else
               (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)) for k, v in rec.items()}
    return out, washer, rec


def boot_screw_mesh(g):
    """PA2x8 proxy (PA2 family envelope as the shin back cover's: head dia 3.5 x 1.3, shank dia 2, 0.6 mm point), axis d*y."""
    s = BOOT_SCREW
    prof = [(0.0, 0.0), (s["head_d"] / 2, 0.0), (s["head_d"] / 2, s["head_h"]), (s["shank_d"] / 2, s["head_h"]),
            (s["shank_d"] / 2, s["head_h"] + s["length"] - s["point"]), (0.0, s["head_h"] + s["length"])]
    m = trimesh.creation.revolve(np.array(prof), sections=48)
    d = g["dir"]
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, d], [0, -d, 0]]      # axis z -> d y
    T[:3, 3] = [g["x"], g["head_top_y"], g["z"]]
    m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    return m


def _mf(m):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


def _tm(x):
    r = x.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=True)


def _solid_voxels(m, lo, shape, spacing=0.4 * PITCH):
    """Voxels of a closed mesh on the grid (lo, shape): dense surface samples, then the enclosed cells filled.
    (trimesh's voxelizer subdivides detailed meshes down to the voxel size and needs several GB.)
    At the default spacing a large flat face keeps a few unsampled voxels, through which the fill leaks (the solid stays a
    hollow skin one voxel thick); the boot keep-out passes spacing=SEALED_SPACING (about 40 samples per voxel face)."""
    from scipy.ndimage import binary_fill_holes
    n = int(min(4e6, max(2e4, m.area / spacing ** 2)))
    pts = np.vstack([trimesh.sample.sample_surface(m, n, seed=0)[0], m.vertices])  # seeded: the same voxels every run
    idx = np.round((pts - lo) / PITCH).astype(np.int32)
    a = np.maximum(idx.min(0) - 1, 0); b = np.minimum(idx.max(0) + 2, shape)
    sub = np.zeros(b - a, bool)
    k = idx - a; ok = np.all((k >= 0) & (k < sub.shape), 1)
    sub[tuple(k[ok].T)] = True
    sub = binary_fill_holes(sub)
    out = np.zeros(shape, bool); out[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = sub
    return out


class LinkGrid:
    """A link's parts as an occupancy grid in its neutral frame, grown by `clearance` (default CLEARANCE).
    fasteners=True keeps the screws and sleeves in (a static keep-out: their heads stand proud); skip adds roles to leave out;
    spacing is the surface sample spacing of the fill (_solid_voxels)."""
    def __init__(self, body, F, link, clearance=CLEARANCE, fasteners=False, skip=(), spacing=0.4 * PITCH):
        from scipy.ndimage import distance_transform_edt
        meshes = []
        for n, p in body.parts.items():
            if p["owning_link"] != link or (F.is_fastener(n, p) and not fasteners) or p.get("role") in SKIP + tuple(skip):
                continue
            m = body.mesh(n)
            meshes.append(m if m.is_watertight else m.convex_hull)
        self.empty = not meshes
        if self.empty:
            return
        pad = clearance + 3 * PITCH
        lo = np.min([m.bounds[0] for m in meshes], 0) - pad; hi = np.max([m.bounds[1] for m in meshes], 0) + pad
        shape = np.ceil((hi - lo) / PITCH).astype(int) + 1
        occ = np.zeros(shape, bool)
        for m in meshes:
            occ |= _solid_voxels(m, lo, shape, spacing)
        self.hit = distance_transform_edt(~occ) * PITCH <= clearance
        self.lo, self.shape = lo, shape
        self.corners = np.array(list(itertools.product(*zip(lo, lo + (shape - 1) * PITCH))))


class Sweep:
    """A PITCH grid over a target region (in the target link's neutral frame), marked where any link grid reaches."""
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, float); self.shape = np.ceil((np.asarray(hi) - self.lo) / PITCH).astype(int) + 1
        self.mask = np.zeros(self.shape, bool)

    def add(self, g, T):
        """T maps target-neutral coordinates into the link grid's neutral frame."""
        if g.empty:
            return
        Ti = np.linalg.inv(T)
        w = trimesh.transform_points(g.corners, Ti)
        a = np.clip(np.floor((w.min(0) - self.lo) / PITCH).astype(int), 0, self.shape)
        b = np.clip(np.ceil((w.max(0) - self.lo) / PITCH).astype(int) + 1, 0, self.shape)
        if np.any(b <= a):
            return
        ii = np.stack(np.meshgrid(*[np.arange(a[k], b[k]) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        p = self.lo + ii * PITCH
        q = p @ T[:3, :3].T + T[:3, 3]
        j = np.round((q - g.lo) / PITCH).astype(np.int32)
        ok = np.all((j >= 0) & (j < g.shape), 1)
        hit = np.zeros(len(ii), bool); hit[ok] = g.hit[tuple(j[ok].T)]
        self.mask[tuple(ii[hit].T)] = True

    def add_box(self, lo, hi, grow=0.0):
        """Mark the grid points within `grow` of the axis-aligned box lo..hi (target frame)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        a = np.clip(np.floor((lo - grow - self.lo) / PITCH).astype(int), 0, self.shape)
        b = np.clip(np.ceil((hi + grow - self.lo) / PITCH).astype(int) + 1, 0, self.shape)
        if np.any(b <= a):
            return
        ii = np.stack(np.meshgrid(*[np.arange(a[k], b[k]) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        p = self.lo + ii * PITCH
        hit = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1) <= grow
        self.mask[tuple(ii[hit].T)] = True

    def surface(self):
        """The swept region as one closed, smoothed mesh."""
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        f = gaussian_filter(np.pad(self.mask, 2).astype(np.float32), 0.6)
        v, faces, _, _ = marching_cubes(f, 0.5)
        v = self.lo + (v - 2) * PITCH
        return trimesh.Trimesh(v, faces[:, ::-1], process=True)


def _subtract(target, env):
    import manifold3d as mf
    e = _mf(env)
    if e.status() != mf.Error.NoError:
        raise ValueError(f"swept surface is not a valid manifold: {e.status()}")
    m = _tm(_mf(target) - e)
    big = [q for q in m.split(only_watertight=False) if (q.volume if q.is_watertight else 0) > 20.0]
    return trimesh.util.concatenate(big) if big else m


def apply(new, h):
    F = h["F"]; log = h["log"]; notes = []
    body = F.Body(new)
    P = {p["name"]: p for p in new["parts"]}
    # torso shells minus the swept legs (torso frame = neutral world)
    sh = {s: F.load_mesh(P[s]["mesh"]) for s in TORSO_SHELLS}
    lo = np.min([m.bounds[0] for m in sh.values()], 0) - 1.0; hi = np.max([m.bounds[1] for m in sh.values()], 0) + 1.0
    sw = Sweep(lo, hi); npose = 0
    for s in "RL":
        grids = {lk: LinkGrid(body, F, f"{s}_{lk}") for lk in HIP_LINKS}
        for pitch, roll, yaw in itertools.product(HIP["pitch"], HIP["roll"], HIP["yaw"]):
            q = {f"{s}_hip_pitch": pitch, f"{s}_hip_roll": roll, f"{s}_hip_yaw": yaw}; npose += 1
            for lk, g in grids.items():
                sw.add(g, np.linalg.inv(body.link_T(f"{s}_{lk}", q)))
    env = sw.surface()
    notes.append(f"hip sweep: {npose} poses, swept region {sw.mask.sum() * PITCH ** 3 / 1000:.0f} cm3 inside the shell box")
    for shell, sm in sh.items():
        if not sm.is_watertight:
            notes.append(f"{shell}: not watertight, skipped"); continue
        out = _subtract(sm, env)
        P[shell].update(h["save_mesh"](shell, out))
        P[shell]["native_step_unmodified"] = P[shell].pop("native_step", P[shell].get("native_step_unmodified")); P[shell].pop("native_step_sha256", None)
        P[shell]["mesh_note"] = "leg openings cut by the legs swept over the target hip ranges (shells.py, voxel sweep)"
        notes.append(f"{shell}: {sm.volume:.0f} -> {out.volume:.0f} mm3")
        log.append((shell, "leg openings cut by the swept hips"))
    # boot shells minus the swept shin and ankle and minus the foot's own parts, in the foot's own frame; then split in two
    screws, extra = [], []
    for s in "RL":
        boot = P[f"{s}_foot_boot_shell"]
        bm = F.load_mesh(boot["mesh"])
        if not bm.is_watertight:
            notes.append(f"{boot['name']}: not watertight, skipped"); continue
        grids = {lk: LinkGrid(body, F, f"{s}_{lk}") for lk in ANKLE_LINKS}
        sw = Sweep(bm.bounds[0] - 1.0, bm.bounds[1] + 1.0)
        for pitch, roll in itertools.product(ANKLE["pitch"], ANKLE["roll"]):
            q = {f"{s}_ankle_pitch": pitch, f"{s}_ankle_roll": roll}
            Tf = body.link_T(f"{s}_foot", q)
            for lk, g in grids.items():
                sw.add(g, np.linalg.inv(body.link_T(f"{s}_{lk}", q)) @ Tf)
        # the foot's own parts (monocoque sole plate, roll servo case, screws) as a static keep-out (T-6c2e93)
        sw.add(LinkGrid(body, F, f"{s}_foot", clearance=BOOT_STATIC_CLEARANCE, fasteners=True, skip=("body_shell",),
                       spacing=SEALED_SPACING), np.eye(4))
        # the monocoque bosses of the retention screws (built later by foot_screws.py) as part of the same keep-out (T-jh7fcm)
        sx = 1.0 if s == "R" else -1.0
        for st in BOOT_SITES:
            g = boot_site_geometry(st, sx)
            sw.add_box(g["boss_lo"], g["boss_hi"], BOOT_STATIC_CLEARANCE)
        if not sw.mask.any():
            notes.append(f"{boot['name']}: nothing swept"); continue
        cut = _subtract(bm, sw.surface())
        heel_name = f"{s}_foot_boot_heel_shell"
        heel = dict(boot, name=heel_name)
        g = BOOT_SPLIT_GAP / 2.0
        for part, normal, off in ((boot, (0.0, -1.0, 0.0), -(BOOT_SPLIT_Y - g)), (heel, (0.0, 1.0, 0.0), BOOT_SPLIT_Y + g)):
            half = _tm(_mf(cut).trim_by_plane(normal, off))
            bodies = sorted(half.split(only_watertight=False), key=lambda q: -(q.volume if q.is_watertight else 0.0))
            keep = [q for q in bodies[:1]] + [q for q in bodies[1:] if q.is_watertight and q.volume >= BOOT_MIN_BODY_MM3]
            dropped = sum(abs(q.volume) for q in bodies if not any(q is k for k in keep))
            assert bodies[0].is_watertight, f"{part['name']}: the main body of the boot half is not watertight"
            half = trimesh.util.concatenate(keep)
            st = next(q for q in BOOT_SITES if q["half"] == ("heel" if part is heel else "toe"))
            half, washer, rec = boot_mount(half, st, sx)
            screws.append(rec)
            wn = f"{s}_foot_boot_{st['half']}_M2_washer"
            w = BOOT_WASHER
            extra.append(dict(name=wn, physical_part_id=wn, owning_link=boot["owning_link"], role="standard_nominal", hardware_kind="washer",
                              **h["save_mesh"](wn, washer), volume_mm3=round(float(washer.volume), 3),
                              mass_kg=round(float(washer.volume) * HW_RHO, 7), color=[128, 136, 146, 255], display_group="mechanism",
                              display_groups=["mechanism"], overlay_id="actuators_and_hardware",
                              parent_assembly=boot.get("parent_assembly", "complete_fdm_ape_body_assembly"), inspection_reference=False,
                              mass_basis=HW_BASIS,
                              material=(f"{w['standard']} M2 washer d1 {w['d1']:g} / d2 {w['d2']:g} x {w['h']:g} under the retention screw's head, "
                                        f"in the dia {BOOT_CB_D:g} counterbore of the {st['half']} of the boot, on the printed tab (shells.py, T-jh7fcm)")))
            cn = f"{s}_foot_boot_{st['half']}_PA2x8"
            sm = boot_screw_mesh(rec)
            extra.append(dict(name=cn, physical_part_id=cn, bom_group=cn, owning_link=boot["owning_link"], role="fastener_proxy",
                              **h["save_mesh"](cn, sm), volume_mm3=round(float(sm.volume), 3), mass_kg=round(float(sm.volume) * HW_RHO, 7),
                              material=(f"PA2x8 self-tapping, ISO 7380-1 button head, on an ISO 7089 M2 washer through the {st['half']} of the boot's "
                                        f"dia {BOOT_HOLE_D:g} clearance hole into a "
                                        f"dia {BOOT_PILOT[0]:g} pilot in the foot monocoque's boss (PA2 family envelope: head dia 3.5 x 1.3, "
                                        "shank dia 2, 0.6 mm point)"),
                              color=[195, 204, 214, 255], display_group="mechanism", display_groups=["mechanism"],
                              overlay_id="actuators_and_hardware", parent_assembly=boot.get("parent_assembly", "complete_fdm_ape_body_assembly"),
                              inspection_reference=False, mass_basis=HW_BASIS))
            notes.append(f"{part['name']}: retention tab {rec['tab_volume_mm3']} mm3, head under skin {rec['head_under_skin_min_mm']} mm")
            part.update(h["save_mesh"](part["name"], half))
            part["mesh_note"] = ("boot collar cut by the shin and ankle swept over the target ankle ranges, the foot's own parts cut out with a "
                                 f"{BOOT_STATIC_CLEARANCE:g} mm keep-out (voxel; measured clearance down to about 0.13 mm), split at y = {BOOT_SPLIT_Y:g} mm ({BOOT_SPLIT_GAP:g} mm seam) into toe part and heel cap "
                                 "(shells.py, voxel sweep, T-6c2e93)")
            notes.append(f"{part['name']}: {half.volume:.0f} mm3 ({len(keep)} bodies, {dropped:.0f} mm3 of crumbs dropped)")
        if heel_name not in P:
            new["parts"].insert(new["parts"].index(boot) + 1, heel); P[heel_name] = heel
        notes.append(f"{boot['name']}: {bm.volume:.0f} -> {cut.volume:.0f} mm3 before the split")
        log.append((boot["name"], "collar cut by the swept ankle, foot parts kept clear, split into toe part and heel cap"))
        log.append((heel_name, "heel cap split off the boot shell at y = %g mm" % BOOT_SPLIT_Y))
    names = {r["name"] for r in extra}
    new["parts"] = [p for p in new["parts"] if p["name"] not in names] + extra
    new["shell_pass"] = dict(hip_targets_deg=dict(pitch=[HIP_FLEX, 40], roll=[-45, 25], yaw=[-40, 40]),
                             ankle_targets_deg=dict(pitch=[-35, 27.5], roll=[-25, 25]), clearance_mm=CLEARANCE,
                             voxel_mm=PITCH, method="voxel sweep + marching cubes, one subtraction per shell",
                             boot=dict(static_clearance_mm=BOOT_STATIC_CLEARANCE, split_y_mm=BOOT_SPLIT_Y, split_gap_mm=BOOT_SPLIT_GAP,
                                       idle_task="T-6c2e93",
                                       screws=[r for r in screws if r["side"] == "R"],
                                       screws_doc=("retention screws of the two boot halves (T-jh7fcm), right foot, manifest frame (left = "
                                                   "mirror x): foot_screws.py builds the monocoque boss boss_lo..boss_hi with a pilot "
                                                   "pilot_dia_mm x pilot_depth_mm from face_y along dir"),
                                       screws_both_sides=screws))
    return notes


def main():
    """Run the shell pass as its own step on a built manifest (keeps the build's memory out of the cuts).

      python3 infra/tools/memguard.py --limit-gb 12 -- .venv/bin/python -B robot/body/ape-proportions/shells.py \\
          artifacts/robot/body/ape-proportions/assembly-parts.json artifacts/robot/body/ape-proportions/shell-pass
    """
    import hashlib
    import json
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent; root = here.parents[2]
    sys.path.insert(0, str(here.parent / "motor-refit"))
    import fit_check as F
    src, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    new = json.load(open(src)); log = []

    def save_mesh(name, m):
        npz = parts / f"{name}.npz"
        np.savez_compressed(npz, vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, np.int64))
        return dict(mesh=str(npz.relative_to(root)), mesh_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                    mesh_units="mm", bounds_mm=[m.bounds[0].round(3).tolist(), m.bounds[1].round(3).tolist()])
    notes = apply(new, dict(F=F, save_mesh=save_mesh, log=log))
    new["shell_pass"]["input_manifest"] = str(src)
    new["shell_pass"]["input_manifest_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    json.dump(new, open(out / "assembly-parts.json", "w"), indent=1)
    json.dump(dict(schema="cove50.ape-proportions.shell-pass.v1", idle_task="T-czv2d8", notes=notes,
                   edits=[dict(part=n, edit=e) for n, e in log]), open(out / "report.json", "w"), indent=1)
    for nt in notes:
        print(nt)


if __name__ == "__main__":
    main()
