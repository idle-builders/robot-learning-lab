"""Arm shell: the whole right arm's covers as ONE parametric surface, cut into four parts at the joints (Idle T-tph7xb,
decision M-f5gs7k; owner 2026-11-23: "the arm needs to go back to an aesthetic design of simple continuous and calm curves.
use simple parametric curves to design the shell and think of the entire arm as one continuous design").

The master surface is a sweep of planar sections:

  station i       centre C_i (mm, body frame at standing), section plane tilt w_i (0 = horizontal, 1 = normal to the
                  shoulder-yaw axis), half-widths a_out (outboard, +x), a_in, b_front (-y), b_back and one exponent n
  along the arm   every column is ONE natural cubic spline over the chord length of the centres (C2: no kinks, curvature
                  changes smoothly); a section is the superellipse |u / a|^n + |v / b|^n = 1 in its plane (u ~ +x, v ~ -y)
  top             route D's hood (kept, owner: "the top cap design is fine like in the pictures with this small cut in"):
                  above the first station the section shrinks as a dome, x half-widths by g(p), y half-widths by g(p_y),
                  g(q) = (1 - h^q)^(1/q), h = (z - z0) / (z_top - z0); the rim about the roll axis (shoulder_shells.RIM) is
                  the small cut-in where the blue hub passes
  bottom          open just above the hand (as today's forearm covers)

The joints are simple analytic seams, so the parts keep a constant gap when the arm moves:

  shoulder yaw    a plane normal to the yaw axis (s = p . YAW = SEAM["s"]); around it the master is a circle about the yaw
                  axis (two stations, n = 2), so the lower cover turns in the hard cover with no step; the lower cover
                  carries a thin lip (a tube about the yaw axis) up inside the hard cover, RUN_CLEAR from it
  elbow           seen along the elbow axis (x), about the axis: the upper-arm part keeps the sector phi_u .. 270 deg (phi
                  measured from straight down toward the front), outside radius R_s; the forearm keeps -90 + d_back ..
                  phi_f and the disc inside R_d (its side walls cover the clevis cheeks). Planes through the axis and a
                  cylinder about it: the forearm part turns in itself; phi_u - phi_f = 95 deg (range) + d_front
  forearm         front / back halves split at the plane y = FORE_SPLIT (as today's covers, screwed to the forearm)

Parts (R; L = exact mirror x -> -x): R_upper_arm_hard_cover (roll link: shoulder to the yaw seam, green U mount wells),
R_upper_arm_lower_cover (upper arm: yaw seam to the elbow; a tab with a pressed 304 sleeve under each of the two elbow-case
screws, which become PA2x10: the tab adds LOWER_TAB to their stack), R_fore_front_hard_cover / R_fore_back_hard_cover
(forearm: M3x8 into the forearm nuts, as today, through a pressed 304 sleeve dia 4.5 / 3.2 in each boss, owner 2026-11-18:
every clamped printed hole carries one), and the sleeve sets R_upper_arm_lower_cover_steel_sleeves,
R_fore_{front,back}_hard_cover_steel_sleeves.

  .venv/bin/python -B robot/body/arm/arm_shell.py margins     # requirement margins per region (fast, no meshes)
  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 1800 -- .venv/bin/python -B robot/body/arm/arm_shell.py apply
  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 1800 -- .venv/bin/python -B robot/body/arm/arm_shell.py views
`apply` is the generator arm-shell (robot/body/arm/arm_shell.bom.yaml, after shoulder-shells): it reads the shoulder-shells
body (shoulder_shells.CANDIDATE) and writes its own body, MANIFEST_OUT (the selected body, robot/design.json body_manifest):
the eight cover rows, the four elbow-case screws and six sleeve-set rows change. Joint ranges come from the body
definitions (robot/body/bom/bom_config.py). Measurements only; nothing accepts or rejects.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "robot/body/bom"))
from checks import bodylib as F  # noqa: E402
import bom_config as BOM_CONFIG  # noqa: E402  (joint ranges and part materials: the body definitions)
import shoulder_clevis as SC  # noqa: E402
import shoulder_shells as SS  # noqa: E402
from design_kit import shell as K  # noqa: E402  (docs/design/design-kit.md: the master surface and seam cutters)

SOURCE = SS.CANDIDATE                            # input: the shoulder-shells body (only read)
OUT = ROOT / "artifacts/robot/body/arm/arm-shell"
PARTS = OUT / "parts"
MANIFEST_OUT = OUT / "assembly-parts-arm-shell.json"   # output: the selected body
REPORT = OUT / "report.json"
CANDIDATE = SOURCE                               # fit / margins / fit-rim read the parts the covers must hold (covers excluded)
VIEWS = OUT / "renders"                          # views: owner renders of the arm shell (derived; recreated by `views`)
YAW = SS.YAW_AXIS
YAW_O = np.array([127.0, -10.0, 306.0])          # shoulder-yaw origin (manifest)
ELBOW = np.array([138.0, -3.0, 219.7638])        # elbow origin; axis along x
COVERS = ("R_upper_arm_hard_cover", "R_upper_arm_lower_cover", "R_fore_front_hard_cover", "R_fore_back_hard_cover")


def jrange(j):
    """Declared range (deg) of joint j (robot/body/arm/build_arm_connectors.bom.yaml joints:)."""
    return BOM_CONFIG.joint_range(j)


def jsteps(j, step):
    lo, hi = jrange(j)
    return np.unique(np.r_[np.arange(lo, hi, step), hi])


def yaw_point(s):
    """The point of the yaw axis at s = p . YAW."""
    return YAW_O + (s - float(YAW_O @ YAW)) * YAW


# ---------------------------------------------------------------------------------------------- the parameters
WALL = 0.9          # shell wall along the surface normal (today's covers 0.84-0.98)
GAP = SS.GAP        # static: shell inner face to the parts of its own link
RUN_CLEAR = 1.5     # moving pairs (owner: 1.5-2 mm)
SEAM = dict(s=226.0, R=26.0, gap=1.5, lip=5.0, lip_margin=0.5)   # yaw seam: plane s, circle radius about the yaw axis,
# slit, lip length, extra lip clearance (the skin is a circle only to within a few tenths over the lip)
ELB = dict(phi_u=134.0, phi_f=34.0, d_back=6.5, R_s=17.1, R_d=15.3)   # elbow split (deg, mm), see the docstring
FORE_SPLIT = dict(y=-2.85, gap=0.3)                # forearm front / back halves
_SP = yaw_point(SEAM["s"])
#                name        centre x, y, z                      tilt  a_out  a_in   b_fr   b_bk   n
STATIONS = (("hood",        (106.0, 5.775, 360.0),              0.0, 22.30, 22.30, 37.00, 37.00, 4.33),
            ("green U",     (119.5, 5.0, 330.0),                0.0, 22.25, 28.21, 33.96, 36.43, 3.29),
            ("yaw case",    (128.0, -0.5, 298.0),               0.4, 22.93, 23.59, 26.65, 35.55, 3.15),
            ("seam top",    tuple(yaw_point(SEAM["s"] + 7.0)),  1.0, SEAM["R"], SEAM["R"], SEAM["R"], SEAM["R"], 2.0),
            ("seam",        tuple(_SP),                         1.0, SEAM["R"], SEAM["R"], SEAM["R"], SEAM["R"], 2.0),
            ("elbow top",   (136.35, -3.0, 232.0),              0.0, 27.44, 27.24, 23.88, 24.02, 3.01),
            ("elbow",       (136.35, -3.0, 212.0),              0.0, 27.44, 27.24, 23.88, 24.02, 3.01),
            ("forearm",     (135.5, -2.0, 178.0),               0.0, 23.82, 24.06, 24.32, 21.80, 2.59),
            ("wrist",       (133.0, -2.5, 141.5),               0.0, 19.28, 25.43, 27.18, 29.41, 2.77))
# centres, tilts, the seam circle and the hood centre by rule (joint centres, the yaw axis, route D's hood); half-widths and
# exponents from `fit` (least area + smoothness + rounder exponents, every requirement inside)
CAP = dict(z_top=382.3, p=2.0, p_y=4.0)            # route D hood crest above the first station, dome exponents
HOOD_MARGIN = 0.6          # extra radial room over the blue U's arc (the need is radial in the section, the skin's normal is not)
RIM = None                 # the cut-in about the roll axis (shoulder_shells.RIM form); None = route D's; `fit-rim` re-places it on this skin
COLS = ("cx", "cy", "cz", "w", "a_out", "a_in", "b_front", "b_back", "n")


# ---------------------------------------------------------------------------------------------- the master surface
# design_kit.shell.Sweep: superellipse sections on natural cubic splines, tilt 0 = horizontal (up +z), tilt 1 = normal to
# the yaw axis, U from +x (outboard), V = U x N (~ -y, front), the route D hood as the dome above the first station
def sweep(st=None):
    return K.Sweep(st or STATIONS, tilt_axis=YAW, up=(0.0, 0.0, 1.0), u_ref=(1.0, 0.0, 0.0), cap=CAP, wall=WALL)


SWEEP = sweep()
_knots = lambda st=None: sweep(st).knots() if st else SWEEP.knots()          # noqa: E731
stations = lambda t, st=None: (sweep(st) if st else SWEEP).frames(t)          # noqa: E731
ring_r = K.ring_r
grid = lambda st=None, **kw: (sweep(st) if st else SWEEP).grid(**kw)          # noqa: E731
master = lambda st=None: (sweep(st) if st else SWEEP).master()                # noqa: E731
local = lambda P, st=None, dt=0.25: (sweep(st) if st else SWEEP).local(P, dt)  # noqa: E731


def margin(P, need, st=None, exclude=False):
    """Radial room (mm) between points P + need and the outer skin (< 0: the point with its need pokes out); for
    `exclude` sets (parts that must stay outside) the room outside the skin."""
    return (sweep(st) if st else SWEEP).margin(P, need, exclude=exclude)


# ---------------------------------------------------------------------------------------------- requirements
def _link_pts(body, link, n=2500, exclude=()):
    ns = [k for k, p in body.parts.items() if p["owning_link"] == link and p.get("role") != "body_shell" and p.get("mesh")
          and k not in exclude and not k.startswith("R_upper_arm_shell_")]
    return SS._thin(SS._pts(body, ns, n=n), 0.7)


def requirements(body):
    """Point sets the master must hold, each with its radial need (mm): static parts of each part's link (GAP + WALL),
    the parts turning inside it (RUN_CLEAR + WALL)."""
    design = SS.mount_design(F.Body(json.loads(SC.MANIFEST.read_text())).mesh("R_upper_arm_hard_cover"))
    s = lambda P: P @ YAW  # noqa: E731
    out = {}
    roll = np.vstack([SS.requirement(body, design), _link_pts(body, "R_shoulder_roll_link",
                                                              exclude=set(SS.green_group(body, "R_")))])
    out["hard cover: roll link (static)"] = (roll[s(roll) > SEAM["s"] + 0.5], GAP + WALL)
    pa2 = [k for k in body.parts if k.startswith("upper__R_elbow_case_PA2x7_")]
    up = _link_pts(body, "R_upper_arm", exclude=set(pa2))
    sw = np.vstack([trimesh.transform_points(up, body.link_T("R_upper_arm", {"R_shoulder_yaw": q})) for q in jsteps("R_shoulder_yaw", 10.0)])
    out["hard cover: upper arm turning in it (yaw range)"] = (sw[s(sw) > SEAM["s"] - 1.0], RUN_CLEAR + WALL)
    ax = SS._roll_axis_xz(body)
    bl = SS.blue_roll_sweep(body, step=2.0, n=8000)
    _, rho = SS.roll_polar(bl, ax)
    out["hood: blue U arc envelope (roll range)"] = (SS._thin(bl[rho <= SC.ARC_R + SC.ARC_T + 0.3], 1.0), RUN_CLEAR + WALL + HOOD_MARGIN)
    up_lo = up[(s(up) < SEAM["s"] - SEAM["gap"]) & (up[:, 2] > ELBOW[2] - 12.0)]
    out["lower cover: upper arm (static)"] = (up_lo, GAP + WALL)
    out["lower cover: PA2x7 heads in their wells"] = (SS._pts(body, pa2, n=3000), 0.3)
    # moving neighbours at the elbow, only where the other part's shell is (the sector it keeps)
    def phi_rho(P):
        d = P[:, 1:] - ELBOW[1:]
        return np.degrees(np.arctan2(-d[:, 0], -d[:, 1])), np.hypot(d[:, 0], d[:, 1])
    fo_all = _link_pts(body, "R_forearm")
    fo_all = fo_all[np.linalg.norm(fo_all - ELBOW, axis=1) < 60.0]
    sw = np.vstack([trimesh.transform_points(fo_all, body.link_T("R_forearm", {"R_elbow": q})) for q in jsteps("R_elbow", 5.0)])
    ph, rr = phi_rho(sw)
    keep = ((ph >= ELB["phi_u"] - 5.0) | (ph <= -85.0)) & (rr >= ELB["R_s"] - 2.0) & (sw @ YAW < SEAM["s"])
    out["lower cover: forearm turning under it (elbow range)"] = (sw[keep], RUN_CLEAR + WALL)
    up_e = up[np.linalg.norm(up - ELBOW, axis=1) < 60.0]
    sw = np.vstack([trimesh.transform_points(up_e, np.linalg.inv(body.link_T("R_forearm", {"R_elbow": q}))) for q in jsteps("R_elbow", 5.0)])
    ph, rr = phi_rho(sw)
    keep = (((ph >= -95.0 + ELB["d_back"]) & (ph <= ELB["phi_f"] + 5.0)) | (rr <= ELB["R_d"] + 2.0)) & (sw[:, 2] < ELBOW[2] + 30.0)
    out["forearm: upper arm turning over it (elbow range)"] = (sw[keep], RUN_CLEAR + WALL)
    tor = SS._pts(body, list(SS.TORSO_SHELLS), n=40000)
    tor = tor[tor[:, 0] > 50.0]
    ex = []
    r_lo = jrange("R_shoulder_roll")[0]
    for qp in jsteps("R_shoulder_pitch", 30.0):
        for qr in (r_lo, r_lo + 5.0, r_lo + 10.0):
            T = body.link_T("R_shoulder_roll_link", {"R_shoulder_pitch": qp, "R_shoulder_roll": qr})
            Q = trimesh.transform_points(tor, np.linalg.inv(T))
            ex.append(Q[(Q[:, 2] > ELBOW[2] + 22.0) & (np.abs(Q[:, 0] - 120.0) < 45.0)])
    out["EXCLUDE torso shells beside the arm (roll low end + 10, pitch range)"] = (SS._thin(np.vstack(ex), 1.0), RUN_CLEAR)
    heads = [k for k in body.parts if k.startswith("R_fore_") and "_M3x8_" in k]
    fo = _link_pts(body, "R_forearm", exclude=set(heads))
    fo = fo[(fo[:, 2] > 141.0)]
    out["forearm: forearm parts (static)"] = (fo, GAP + WALL)
    out["forearm: screw heads in their wells"] = (SS._pts(body, heads, n=3000), 0.3)
    palm = SS._pts(body, ["R_palm_lower", "R_palm_upper"], n=6000)
    out["forearm end: as wide as the hand's top (look)"] = (palm[palm[:, 2] > 132.0], 0.0)
    return out


def report_margins(body, st=None, verbose=True):
    return (sweep(st) if st else SWEEP).report_margins(requirements(body), verbose=verbose)


# ---------------------------------------------------------------------------------------------- the cut into parts
_mf, _tm = K.to_manifold, K.to_trimesh
_boss = K.boss


def _wedge_x(phi0, phi1, r=400.0, n=90, x0=60.0, x1=200.0):
    """Prism along x over the sector phi0..phi1 (deg, from straight down toward the front) about the elbow axis."""
    return K.seam_sector(ELBOW, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), phi0, phi1, x0, x1, r=r, n=n)


def _cyl_x(r, x0=60.0, x1=200.0):
    return K.seam_cylinder([x0, ELBOW[1], ELBOW[2]], [x1, ELBOW[1], ELBOW[2]], r)


def _tube_yaw(r_in, r_out, s0, s1):
    return K.tube(yaw_point(s0), yaw_point(s1), r_in, r_out, axis=YAW)


LOWER_TAB = 1.5     # lower-cover tab under each elbow-case screw head (on the washer): the screw's stack grows by this
LOWER_SCREW = dict(src="upper__{S}_elbow_case_PA2x7_{k}", name="upper__{S}_elbow_case_PA2x10_{k}", thread="PA2", length_mm=10.0,
                   head_dia_mm=3.5, head_height_mm=1.3, point_mm=0.5, standard="ISO 7380-1")
# the elbow-case screw with the tab in its stack: today's PA2x7 engages 1.8 mm of the STS3250 case hole's reference 3.0
# (aluminium, modelled depth 3.9 / 4.8); + LOWER_TAB the next standard M2 button-head length that reaches the reference is 10
# (data/hardware.json: 4, 5, 6, 8, 10): 3.3 mm engaged, tip 0.6 mm or more short of the hole bottom (PA2x8 would engage 1.3)
SLEEVE_PA2 = (3.6, 2.2)   # 304 sleeve (outer, bore) under a PA2 head (as the shin back cover's, thigh_shell.py SLV_D / SLV_BORE_D)
SLEEVE_M3 = (4.5, 3.2)    # 304 sleeve under an M3 head (owner M-pxbsxq: 4.5 x 3.2 tubes; thigh_shell.py SLEEVE_R / SLEEVE_BORE_R)


def lower_mounts(body):
    """The two elbow-case screws (head outboard, +x). Between spacer and washer sits the connector's own ear, so the
    cover's tab lies on the washer, LOWER_TAB thick, and the head on the tab: (y, z) axis, tab faces, head seat, the
    sleeve (washer face to head seat; the tab starts 0.05 off the washer, the steel bears on it)."""
    out = []
    for k in ("2", "4"):
        b = body.mesh(LOWER_SCREW["src"].format(S="R", k=k)).bounds
        wa = body.mesh(f"upper__R_elbow_case_washer{k}").bounds
        xw = float(wa[1, 0])
        x0 = xw + 0.05
        out.append(dict(k=k, y=float(b[:, 1].mean()), z=float(b[:, 2].mean()), x_tab=(x0, xw + LOWER_TAB),
                        x_head=xw + LOWER_TAB, head_r=float(b[1, 1] - b[0, 1]) / 2.0,
                        sleeve_x=(xw, xw + LOWER_TAB), sleeve_len_mm=LOWER_TAB))
    return out


def lower_screw_mesh(m):
    """The elbow-case screw of mount m: LOWER_SCREW's revolved envelope (screw_plan.screw_mesh), head underside on the tab."""
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    import screw_plan
    plan = dict(LOWER_SCREW, axis_out=[1.0, 0.0, 0.0], opening=[m["x_head"], m["y"], m["z"]], stack_mm=0.0)
    return screw_plan.screw_mesh(plan)


_tube = K.sleeve_tube   # sleeve solid on the axis a + t d, t0..t1 (manifold)


def fore_mounts(body):
    """Forearm cover screws (M3x8 along y): axis (x, z), head side, seat (head underside), the forearm's outermost face
    within the boss radius along the axis."""
    conn = [n for n, p in body.parts.items() if p["owning_link"] == "R_forearm" and p.get("role") != "body_shell"
            and not n.startswith("R_fore_") and p.get("mesh")]
    cat = trimesh.util.concatenate([body.mesh(n) for n in conn] + [body.mesh(n) for n in body.parts if n.startswith("R_fore_") and n.endswith("_nut")])
    rx = trimesh.ray.ray_pyembree.RayMeshIntersector(cat)
    out = {}
    for half, h in (("front", -1.0), ("back", 1.0)):
        for tag in ("178", "187"):
            b = body.mesh(f"R_fore_{half}_M3x8_{tag}").bounds
            x, z = float(b[:, 0].mean()), float(b[:, 2].mean())
            head = float(b[1, 1] if h > 0 else b[0, 1])
            seat = head - h * 1.65                                     # ISO 7380-1 M3 head height
            face = []
            for r in (0.0, 2.0, 3.0, 4.0, 4.6):
                for a in np.radians(np.arange(0, 360, 30)):
                    o = np.array([[x + r * np.cos(a), h * 80.0, z + r * np.sin(a)]])
                    loc, _, _ = rx.intersects_location(o, np.array([[0.0, -h, 0.0]]), multiple_hits=False)
                    if len(loc):
                        face.append(loc[0, 1])
            face = max(face) if h > 0 else min(face)
            out[f"{half}_{tag}"] = dict(x=x, z=z, h=h, head=head, seat=seat, face=float(face))
    return out


def build(body, st=None):
    """The four R covers as trimesh meshes, and facts about the cut."""
    t0 = time.time()
    outer, inner = master(st)
    sh = _mf(outer) - _mf(inner)
    o_mf = _mf(outer)
    facts = {}
    # 1. hard cover: above the seam plane, the rim (cut-in about the roll axis), the green U wells
    design = SS.mount_design(F.Body(json.loads(SC.MANIFEST.read_text())).mesh("R_upper_arm_hard_cover"))
    hard = sh.trim_by_plane(YAW.tolist(), float(SEAM["s"]))
    hard = SS.add_wells(hard, outer, design)
    hard = hard - _mf(SS.rim_cutter(SS._roll_axis_xz(body), RIM))
    # 2. lower cover: below the seam's slit, the elbow's upper sector outside R_s; lip; PA2 tabs
    lo = sh.trim_by_plane((-YAW).tolist(), -float(SEAM["s"] - SEAM["gap"]))
    upper_sector = _mf(_wedge_x(ELB["phi_u"], 270.0)) - _mf(_cyl_x(ELB["R_s"]))
    lo = lo ^ (upper_sector + _mf(trimesh.creation.box(bounds=[[60, -80, ELBOW[2] + 45.0], [200, 80, 400]])))
    R = SEAM["R"]
    lip_o = R - WALL - RUN_CLEAR - SEAM["lip_margin"]
    lo = lo + _tube_yaw(lip_o - WALL, lip_o, SEAM["s"] - SEAM["gap"] - 3.0, SEAM["s"] + SEAM["lip"])
    lo = lo + (_tube_yaw(lip_o - WALL, R - 0.3, SEAM["s"] - SEAM["gap"] - 3.0, SEAM["s"] - SEAM["gap"] - 1.2) ^ o_mf)
    sleeves = {"R_upper_arm_lower_cover": [], "R_fore_front_hard_cover": [], "R_fore_back_hard_cover": []}
    for m in lower_mounts(body):
        ax = np.array([0.0, m["y"], m["z"]])
        tab = _boss(ax, [1, 0, 0], 3.6, m["x_tab"][0], 200.0) ^ o_mf
        lo = lo + tab
        lo = lo - _boss(ax, [1, 0, 0], m["head_r"] + 0.45, m["x_head"], 200.0) - _boss(ax, [1, 0, 0], SLEEVE_PA2[0] / 2.0, m["x_tab"][0] - 1.0, m["x_head"] + 0.01)
        sleeves["R_upper_arm_lower_cover"].append(_tube(ax, np.array([1.0, 0.0, 0.0]), m["sleeve_x"][0], m["sleeve_x"][1], *SLEEVE_PA2))
    facts["lower_mounts"] = lower_mounts(body)
    # 3. forearm: the lower sector and the disc, split front / back, M3x8 bosses
    fore_sector = _mf(_wedge_x(-90.0 + ELB["d_back"], ELB["phi_f"])) + _mf(_cyl_x(ELB["R_d"]))
    fo = sh ^ fore_sector
    fo = fo.trim_by_plane((-YAW).tolist(), -float(SEAM["s"] - 20.0))   # nothing of the upper arm
    fm = fore_mounts(body)
    facts["fore_mounts"] = fm
    halves = {}
    for half, sgn in (("front", -1.0), ("back", 1.0)):
        y0 = FORE_SPLIT["y"] + sgn * FORE_SPLIT["gap"] / 2.0
        part = fo.trim_by_plane([0.0, sgn, 0.0], sgn * y0)
        for k, m in fm.items():
            if not k.startswith(half):
                continue
            ax = np.array([m["x"], 0.0, m["z"]]); d = np.array([0.0, m["h"], 0.0])
            boss = _boss(ax, d, 4.5, m["h"] * (m["face"]) + 0.1, 80.0) ^ o_mf
            part = part + boss
            part = part - _boss(ax, d, 3.25, m["h"] * m["seat"], 80.0) - _boss(ax, d, SLEEVE_M3[0] / 2.0, m["h"] * m["face"] - 1.0, m["h"] * m["seat"] + 0.01)
            sleeves[f"R_fore_{half}_hard_cover"].append(_tube(ax, d, m["h"] * m["face"], m["h"] * m["seat"], *SLEEVE_M3))
            m["sleeve_len_mm"] = round(abs(m["seat"] - m["face"]), 2)
        halves[half] = part
    out = {"R_upper_arm_hard_cover": _tm(hard), "R_upper_arm_lower_cover": _tm(lo),
           "R_fore_front_hard_cover": _tm(halves["front"]), "R_fore_back_hard_cover": _tm(halves["back"])}
    for k, m in list(out.items()):
        keep = [q for q in m.split(only_watertight=False) if abs(q.volume) > 50.0]
        out[k] = trimesh.util.concatenate(keep)
        facts[k] = dict(volume_mm3=round(float(out[k].volume), 1), watertight=bool(out[k].is_watertight), pieces=len(keep))
    for k, ss in sleeves.items():
        sm = ss[0]
        for q in ss[1:]:
            sm = sm + q
        out[k + "_steel_sleeves"] = _tm(sm)
    facts["sleeves"] = {k: dict(count=len(v), volume_mm3=round(float(out[k + "_steel_sleeves"].volume), 2)) for k, v in sleeves.items()}
    facts["seconds"] = round(time.time() - t0, 1)
    return out, outer, facts


# ---------------------------------------------------------------------------------------------- measures
def spine_smoothness(st=None, dt=0.5):
    """Spine curvature (1/mm) and its change per mm along the centre curve; ring-value slopes."""
    return (sweep(st) if st else SWEEP).smoothness(dt)


def growth(src, outer):
    """Outward growth of the new outer skin (the master's outer solid) against today's covers (source manifest), mm."""
    return K.growth(outer, {n: src.mesh(n) for n in COVERS})


# ---------------------------------------------------------------------------------------------- apply
def _mirror(m):
    return SS._mirror(m)


STEEL_KG_MM3 = 7.85e-6     # 304 stainless (sleeve mass: mesh volume x density, as the other sleeve sets)
SLEEVE_LINK = {"R_upper_arm_lower_cover": "R_upper_arm", "R_fore_front_hard_cover": "R_forearm", "R_fore_back_hard_cover": "R_forearm"}


def _save(name, m):
    f = PARTS / f"{name}.npz"
    np.savez_compressed(f, vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, np.int64))
    return dict(mesh=SC._rel(f), mesh_sha256=SC._sha(f), mesh_units="mm", volume_mm3=round(float(m.volume), 3),
                bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()])


def apply():
    t0 = time.time()
    man = json.loads(SOURCE.read_text()); body = F.Body(man)
    src = F.Body(json.loads(SC.MANIFEST.read_text()))
    margins = report_margins(body)
    meshes, outer, facts = build(body)
    print(json.dumps({k: v for k, v in facts.items() if k in COVERS}), f"({time.time() - t0:.0f} s)", flush=True)
    P = {p["name"]: p for p in man["parts"]}
    PARTS.mkdir(parents=True, exist_ok=True)
    for f in PARTS.glob("*.npz"):
        f.unlink()
    me = dict(source=SC._rel(__file__), sha256=SC._sha(__file__))
    for n in COVERS:
        for s in ("R_", "L_"):
            nm = n.replace("R_", s, 1)
            m = meshes[n] if s == "R_" else _mirror(meshes[n])
            q = P[nm]
            for k in ("native_step", "native_step_sha256", "native_step_unmorphed", "shoulder_shells"):
                q.pop(k, None)
            rho = BOM_CONFIG.density_kg_mm3(nm)
            q.update(_save(nm, m), mass_kg=round(float(m.volume) * rho, 6),
                     mass_basis=f"mesh volume x {rho * 1e6:g} g/cm3 ({BOM_CONFIG.material(nm)}, declared in robot/body/arm/arm_shell.bom.yaml)",
                     mesh_note="arm_shell.py: one parametric arm surface (superellipse sections, natural cubic splines along "
                               "the arm, route D hood) cut at the yaw seam and the elbow",
                     arm_shell=dict(me, mirror_of=n if s == "L_" else None))
    # the elbow-case screws: LOWER_SCREW in place of the PA2x7 (the tab is in their stack), same axis
    mounts = {m["k"]: m for m in lower_mounts(body)}
    rows = []
    for s in ("R", "L"):
        for k0, m in mounts.items():
            sm = lower_screw_mesh(m)
            sm = sm if s == "R" else _mirror(sm)
            # the L screws are numbered differently from their R mirrors: replace the one on the same axis (y, z)
            olds = [LOWER_SCREW["src"].format(S=s, k=k) for k in mounts]
            c = np.asarray(sm.bounds, float)[:, 1:].mean(axis=0)
            src_nm = min(olds, key=lambda n: float(np.linalg.norm(np.asarray(P[n]["bounds_mm"], float)[:, 1:].mean(axis=0) - c)) if n in P else 1e9)
            k = src_nm.rsplit("_", 1)[1]
            old = P.pop(src_nm)
            nm = LOWER_SCREW["name"].format(S=s, k=k)
            r = {kk: v for kk, v in old.items() if kk not in ("name", "mesh", "mesh_sha256", "bounds_mm", "volume_mm3", "native_step",
                                                               "native_step_sha256", "mass_kg", "mass_basis")}
            r.update(name=nm, **_save(nm, sm), mass_kg=round(float(sm.volume) * STEEL_KG_MM3, 6),
                     mass_basis="modelled envelope volume x 7.85 g/cm3 (data/hardware.json screw_standards density)",
                     material=f"{LOWER_SCREW['standard']} {LOWER_SCREW['thread']}x{LOWER_SCREW['length_mm']:g} into the elbow STS3250 case "
                              f"(through washer, the {LOWER_TAB:g} mm cover tab and its sleeve)",
                     arm_shell=dict(me, replaces=old["name"], note=f"{LOWER_TAB} mm cover tab with a pressed sleeve under the head"))
            rows.append(r)
    # sleeve sets
    for n, link in SLEEVE_LINK.items():
        sm = meshes[n + "_steel_sleeves"]
        for s in ("R_", "L_"):
            nm = n.replace("R_", s, 1) + "_steel_sleeves"
            ms = sm if s == "R_" else _mirror(sm)
            od, bore = SLEEVE_PA2 if n == "R_upper_arm_lower_cover" else SLEEVE_M3
            rows.append(dict(name=nm, owning_link=link.replace("R_", s, 1), role="compression_limit_sleeves", **_save(nm, ms),
                             mass_kg=round(float(ms.volume) * STEEL_KG_MM3, 6), color=[215, 222, 230, 255],
                             mass_basis="304 SS sleeves, mesh volume x 7.85 g/cm3",
                             material=f"304 stainless tube dia {od:g} / {bore:g}, pressed into the {n.replace('R_', s, 1)} bosses from the "
                                      "head seat to the boss end; the head bears on it, its far end on the part below",
                             arm_shell=dict(me, mirror_of=nm.replace("L_", "R_", 1) if s == "L_" else None)))
    man["parts"] = list(P.values()) + rows
    src_sha = SC._sha(SOURCE)
    rep = dict(idle_task="T-tph7xb", decision="M-eb6cxz", stations=[dict(zip(("name", "centre") + COLS[3:], s)) for s in STATIONS],
               cap=CAP, seam=SEAM, elbow=ELB, fore_split=FORE_SPLIT, wall_mm=WALL, gap_mm=GAP, run_clear_mm=RUN_CLEAR,
               lower_tab_mm=LOWER_TAB, lower_screw=LOWER_SCREW, sleeves=dict(pa2=SLEEVE_PA2, m3=SLEEVE_M3),
               margins=margins, facts=facts, smoothness=spine_smoothness(), growth_vs_today=growth(src, outer))
    man["arm_shell"] = dict(me, input_manifest=SC._rel(SOURCE), input_manifest_sha256=src_sha, report=SC._rel(REPORT),
                            replaced_screws=[r["arm_shell"]["replaces"] for r in rows if "replaces" in r["arm_shell"]])
    REPORT.write_text(json.dumps(rep, indent=1, default=float) + "\n")
    MANIFEST_OUT.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(rep["smoothness"]), json.dumps(rep["growth_vs_today"]), flush=True)
    print(f"wrote {SC._rel(MANIFEST_OUT)} ({time.time() - t0:.0f} s)")


FIT = dict(n_max=3.4, n_min=2.2, n_pref=2.4, n_weight=0.5, w_min=12.0, w_max=48.0, smooth=400.0,
           hood_a=((22.3, 26.0), (22.3, 22.3)), hood_n_max=4.5, hood_b_max=37.0)   # fit: exponent and half-width
# bounds, the preferred exponent (rounder) and its weight, smoothing weight; the hood's outboard half-width from route D's 22.3 up,
# its inboard one kept at route D's 22.3 (the torso dish was fitted to it); the hood's depth held near route D's (b <= 37,
# exponent up to 4.5 there: its sections must hold the blue U's arc envelope, a cylinder along y)


def fit(verbose=True):
    """The half-widths and exponents of every station (centres, tilts, the hood's x half-widths and the seam circle are
    set by rule): design_kit.shell.Sweep.fit, least section area + smoothing (weight FIT["smooth"]) + a pull of the
    exponents toward FIT["n_pref"], every requirement point inside with its need. The elbow's two stations are tied (a
    constant band about the axis)."""
    body = F.Body(json.loads(CANDIDATE.read_text()))
    names = SWEEP.names
    hood = names.index("hood")

    def bounds(i, c):
        if c == 4:
            return (FIT["n_min"], FIT["hood_n_max"] if i == hood else FIT["n_max"])
        if i == hood:
            return FIT["hood_a"][c] if c < 2 else (FIT["w_min"], FIT["hood_b_max"])
        return (FIT["w_min"], FIT["w_max"])
    V, r, m = SWEEP.fit(requirements(body), bounds, fixed=("seam top", "seam"), tied=(("elbow top", "elbow"),),
                        smooth=FIT["smooth"], n_pref=FIT["n_pref"], n_weight=FIT["n_weight"])
    if verbose:
        print(f"fit: {r.message}; worst margin {m.min():.3f} mm over {len(m)} points; objective {r.fun:.3f}")
        print(SWEEP.stations_text(V))
    return V, r


# ---------------------------------------------------------------------------------------------- images, open area
COL = dict(cover=(0.93, 0.90, 0.82), shell=(0.86, 0.86, 0.88), inner=(0.36, 0.36, 0.40), hand=(0.62, 0.60, 0.58),
           screw=(0.55, 0.57, 0.60))
ARM_LINKS = ("torso", "R_shoulder_pitch_link", "R_shoulder_roll_link", "R_upper_arm", "R_forearm")
ARM_LO, ARM_HI = np.array([40.0, -75.0, 92.0]), np.array([200.0, 80.0, 400.0])
RANGE_POSES = tuple((f"{j.replace('R_', '').replace('_', ' ')} {v:g}", {j: float(v)})
                    for j in ("R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow") for v in jrange(j))
CAMS = (("owner camera (front + outside, 49 deg above)",) + tuple(SS.OWNER_CAM), ("front (-y)", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ("side (+x)", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), ("top (-z)", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))


def _cls(n, p):
    if p.get("role") == "body_shell" and not n.endswith("_nut"):
        return "cover" if n in COVERS else "shell"
    if ("_M3x8_" in n and n.startswith("R_fore_")) or n.startswith("upper__R_elbow_case_PA2x") or ("upper_arm_shell_" in n and "_M3x" in n):
        return "screw"
    if n.startswith("R_palm") or "_hand" in n:
        return "hand"
    return "inner"


def arm_scene(body, q, links=ARM_LINKS, lo=ARM_LO, hi=ARM_HI, pad=30.0):
    """(name, class, posed mesh) of the right arm and the torso near it at pose q."""
    out = []
    for n, p in body.parts.items():
        lk = p["owning_link"]
        if lk not in links or not p.get("mesh"):
            continue
        T = body.link_T(lk, q)
        b = body.mesh(n).bounds
        C = trimesh.transform_points(np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]]), T)
        if np.all(C.max(0) > lo - pad) and np.all(C.min(0) < hi + pad):
            m = body.mesh(n).copy(); m.apply_transform(T); out.append((n, _cls(n, p), m))
    return out


def open_area(body, q, d, up, px=0.3, lo=ARM_LO, hi=ARM_HI):
    """Visible open area of the right arm from direction d at pose q (window lo..hi): pixels whose first hit is not a
    shell's outer skin and not the hand (a part inside, or a shell's inner face seen through an opening), mm2; the
    largest open region; the class image (0 none, 1 shell / cover skin, 2 shell inner face, 3 inner part, 4 hand)."""
    from scipy import ndimage
    sc = arm_scene(body, q)
    ms = [m for _, _, m in sc]
    pid, Pm, Nm, ext, _ = SS._cast(ms, d, up, lo, hi, px)
    cls = np.zeros(pid.shape, int)
    hit = pid >= 0
    kinds = np.array([c for _, c, _ in sc])
    k = np.where(hit, kinds[np.maximum(pid, 0)], "")
    shell = hit & np.isin(k, ["cover", "shell"])
    grp = {}
    for i, (n, c, _) in enumerate(sc):
        if c in ("cover", "shell"):
            grp.setdefault("torso" if c == "shell" else n, []).append(i)
    for g, idx in grp.items():
        sel = shell & np.isin(pid, idx)
        if not sel.any():
            continue
        rx = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh.util.concatenate([ms[i] for i in idx]))
        again = rx.intersects_any(Pm[sel] + Nm[sel] * 0.05, Nm[sel])
        cls[sel] = np.where(again, 2, 1)
    cls[hit & (k == "inner")] = 3
    cls[hit & (k == "hand")] = 4
    cls[hit & (k == "screw")] = 1                                     # cover screw heads seated in their wells: skin
    om = (cls == 2) | (cls == 3)
    lab, nl = ndimage.label(om)
    a = px * px
    big = float(np.bincount(lab.ravel())[1:].max() * a) if nl else 0.0
    return dict(open_mm2=round(float(om.sum() * a), 0), inner_parts_mm2=round(float((cls == 3).sum() * a), 0),
                shell_inner_faces_mm2=round(float((cls == 2).sum() * a), 0), largest_region_mm2=round(big, 0), regions=int(nl)), cls, ext


def _arm_window(body, q, pad=12.0):
    """Box around the right arm's covers and hand at pose q (the shoulder top included)."""
    B = []
    for n in COVERS + ("R_palm_lower", "R_palm_upper"):
        m = body.mesh(n).copy(); m.apply_transform(body.link_T(body.parts[n]["owning_link"], q)); B.append(m.bounds)
    B = np.array(B)
    return B[:, 0].min(0) - pad, B[:, 1].max(0) + pad


def _colour(sc):
    return [(m, COL[c]) for _, c, m in sc]


def views(tag="arm-shell"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    man = json.loads(MANIFEST_OUT.read_text()); body = F.Body(man)
    VIEWS.mkdir(parents=True, exist_ok=True)
    leg = [Patch(facecolor=COL["cover"], edgecolor="#444", label="arm covers (4 parts)"), Patch(facecolor=COL["shell"], edgecolor="#444", label="torso shells"),
           Patch(facecolor=COL["inner"], edgecolor="#444", label="mechanism"), Patch(facecolor=COL["hand"], edgecolor="#444", label="hand")]
    files = []
    # 1. standing, four cameras
    sc = arm_scene(body, {})
    fig, axs = plt.subplots(1, 4, figsize=(30, 13), dpi=80)
    for ax, (t, d, up) in zip(axs, CAMS):
        img, ext = SC._render(_colour(sc), d, up, ARM_LO, ARM_HI, px_mm=0.22)
        ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"standing: {t}", fontsize=14); ax.axis("off")
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); f = VIEWS / f"{tag}-standing.png"; fig.savefig(f, facecolor="white"); plt.close(fig); files.append(f)
    # 2. range ends: owner camera and front
    fig, axs = plt.subplots(2, len(RANGE_POSES), figsize=(5.2 * len(RANGE_POSES), 13), dpi=80)
    for c, (pl, q) in enumerate(RANGE_POSES):
        lo2, hi2 = _arm_window(body, q)
        scq = arm_scene(body, q, lo=lo2, hi=hi2)
        for r, (t, d, up) in enumerate(CAMS[:2]):
            img, ext = SC._render(_colour(scq), d, up, lo2, hi2, px_mm=0.3)
            axs[r, c].imshow(img, extent=ext, interpolation="bilinear"); axs[r, c].set_title(f"{pl}: {t.split(' (')[0]}", fontsize=12); axs[r, c].axis("off")
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); f = VIEWS / f"{tag}-range-ends.png"; fig.savefig(f, facecolor="white"); plt.close(fig); files.append(f)
    # 3. full body, front
    allp = [(n, _cls(n, p), body.mesh(n)) for n, p in body.parts.items() if p.get("mesh") and p["owning_link"] in body.joints.keys() | {"torso"}
            or p.get("mesh")]
    scene = []
    for n, p in body.parts.items():
        if not p.get("mesh"):
            continue
        m = body.mesh(n).copy(); m.apply_transform(body.link_T(p["owning_link"], {})); scene.append((m, COL[_cls(n, p)] if p.get("role") != "body_shell" else (COL["cover"] if n.replace("L_", "R_", 1) in COVERS else COL["shell"])))
    B = np.array([m.bounds for m, _ in scene]); lo3, hi3 = B[:, 0].min(0) - 5, B[:, 1].max(0) + 5
    fig, axs = plt.subplots(1, 2, figsize=(22, 14), dpi=80)
    for ax, (t, d, up) in zip(axs, (CAMS[1], ("3/4 front", (-0.5, 0.85, -0.15), (0, 0, 1)))):
        img, ext = SC._render(scene, d, up, lo3, hi3, px_mm=0.45)
        ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"full body, standing: {t}", fontsize=14); ax.axis("off")
    fig.tight_layout(); f = VIEWS / f"{tag}-full-body.png"; fig.savefig(f, facecolor="white"); plt.close(fig); files.append(f)
    del allp
    # 4. open area: standing x 3 cameras, the range ends from the owner camera
    rows = {}
    fig, axs = plt.subplots(2, 5, figsize=(30, 17), dpi=70)
    cases = [("standing", {}, CAMS[0]), ("standing", {}, CAMS[1]), ("standing", {}, CAMS[2])] + \
            [(pl, q, CAMS[0]) for pl, q in RANGE_POSES if pl in ("elbow 95", "shoulder yaw -60", "shoulder yaw 60", "shoulder roll 95")] + \
            [(pl, q, CAMS[1]) for pl, q in RANGE_POSES if pl in ("elbow 95", "shoulder pitch 150", "shoulder roll -15")]
    for ax, (pl, q, (t, d, up)) in zip(axs.ravel(), cases):
        st, cls, ext = open_area(body, q, d, up, lo=_arm_window(body, q)[0], hi=_arm_window(body, q)[1])
        rows[f"{pl} / {t}"] = st
        img = np.array([(1, 1, 1), COL["cover"], (0.25, 0.22, 0.22), (0.15, 0.40, 0.90), COL["hand"]])[cls]
        ax.imshow(img, extent=ext, interpolation="nearest"); ax.axis("off")
        ax.set_title(f"{pl}: {t.split(' (')[0]}\nopen {st['open_mm2']:.0f} mm2 (parts {st['inner_parts_mm2']:.0f}, shell insides {st['shell_inner_faces_mm2']:.0f})", fontsize=11)
    fig.tight_layout(); f = VIEWS / f"{tag}-open-area.png"; fig.savefig(f, facecolor="white"); plt.close(fig); files.append(f)
    (VIEWS / f"{tag}-open-area.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(json.dumps(rows))
    for f in files:
        print("wrote", SC._rel(f))


def fit_rim():
    """Re-place the cut-in on this hood (shoulder_shells.fit_rim: least skin open, the swept blue U stem inside with
    margin, the green U enclosure outside); prints RIM to paste above."""
    body = F.Body(json.loads(CANDIDATE.read_text()))
    design = SS.mount_design(F.Body(json.loads(SC.MANIFEST.read_text())).mesh("R_upper_arm_hard_cover"))
    outer, inner = master()
    hard = (_mf(outer) - _mf(inner)).trim_by_plane(YAW.tolist(), float(SEAM["s"]))
    hard = SS.add_wells(hard, outer, design)
    sweep = SS.blue_roll_sweep(body, step=1.0, n=12000)
    rim, st = SS.fit_rim(body, _tm(hard), design, sweep, m_set=(8.0, 12.0, 16.0))
    print("RIM =", rim, json.dumps(st))


def margins_cmd():
    body = F.Body(json.loads(CANDIDATE.read_text()))
    report_margins(body)
    print(json.dumps(spine_smoothness()))


if __name__ == "__main__":
    {"margins": margins_cmd, "apply": apply, "fit": fit, "fit-rim": fit_rim,
     "views": lambda: views(sys.argv[2] if len(sys.argv) > 2 else "arm-shell")}[sys.argv[1] if len(sys.argv) > 1 else "margins"]()
