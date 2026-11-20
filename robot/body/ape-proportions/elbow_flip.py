"""Elbow servo flipped: its case sits in the upper arm, the forearm is driven from the horn (owner, 2026-11-13).

Before: the elbow case hung in the forearm (35 mm below the elbow axis) and the upper arm gripped only the horn,
from one side, through a 3 mm plate and a thin bar. After: the case and its screws turn 180 deg about the elbow axis,
so the case points up into the upper arm. The pieces of the two connectors swap with it:

  new upper-arm connector (stage 3) = stage 3 above Z3 (the yaw-horn bracket and the beam that now starts above the
                                      case) + stage 4's case clamp (above Z4), turned 180 deg about the elbow axis
  new forearm connector (stage 4)   = stage 4 below Z4 (the forearm rails to the wrist) + stage 3's horn plate (below
                                      Z3), turned 180 deg, + a forearm beam from the horn plate down to the rails

Turning both members of every pair (case and clamp, horn and plate) by the same 180 deg keeps every contact and
clearance of the selected design at the neutral pose. The elbow joint itself (origin, axis) is unchanged; only the
servo's mounting direction reverses (a sign in the servo calibration).

Shells: the upper-arm cover is split at the shoulder-yaw joint, perpendicular to the yaw axis. The cap above stays on
the shoulder-roll link; the sleeve below goes onto the upper-arm link, so it turns with the case inside it. The sleeve
is extended down over the case; the forearm covers lose their top, which wrapped the case.
"""
from __future__ import annotations

import numpy as np
import trimesh


def flip_motor_sign(j):
    """The case turns 180 deg about the elbow axis itself, so the servo axis keeps its direction, but the horn moves from
    the parent to the child: horn-relative-to-case goes from -q to +q. The command sign is negated here, once."""
    if j.get("servo_mount_reversed"):
        return
    j["motor_command_sign_refit"] = j["motor_command_sign"]
    j["motor_command_sign"] = -j["motor_command_sign"]
    j["servo_mount_reversed"] = True
    j["motor_allocation"] = ("case fixed in the upper arm; forearm driven from the main horn (medial clevis cheek), with the "
                             "rear idler as the opposed support (lateral cheek). motor_command_sign already includes the "
                             f"reversal (refit value {j['motor_command_sign_refit']:+d}); do not apply servo_mount_reversed again")

# Heights are measured from the elbow axis (mm), so the flip follows any upper-arm length.
Z3 = 36.17          # stage 3 split: the beam starts 0.5 mm above it, 1.6 mm above the turned case top
Z4 = -41.33         # stage 4 split: clean two-rail section below the case clamp
FOREARM_BEAM = dict(x=(109.0, 117.0), y=(-12.5, 4.0), z=(-45.33, -10.83))   # medial, clear of the horn and its screws
FOREARM_BRIDGE = dict(x=(109.0, 135.0), y=(-12.5, 4.0), z=(-46.33, -37.33))  # beam foot onto the rails
SLEEVE_BOTTOM = -6.0         # the lower cover's rim is first carried down to here, then cut by the slanted edge below
UPPER_EDGE = (3.0, float(__import__("os").environ.get("APE_UPPER_FRONT", 18.0)))     # lower cover bottom edge above the elbow axis: (back, front); the front clears the forearm swing
LATERAL_WINDOW_R = 18.0      # round window in the lower cover's outer wall around the idler-side cheek (radius 15)
FOREARM_COVER_TOP = -15.33   # (superseded by the continuous skin below)
# Forearm skin (owner 2026-11-13: upper and lower arm shells should read as one form with a gap at the elbow): the
# forearm shell starts at the upper-arm cover's own bottom section and blends to the old forearm section at the wrist.
SKIN_TOP = -4.0              # forearm shell top edge at the back, from the elbow axis (the front drops to SKIN_FRONT_DROP)
SKIN_TOP_SHIFT_X = 4.0       # lateral, so the idler-side cheek (outer face x 158.5) fits inside
SKIN_TOP_SHIFT_Y = -2.0      # forward: the clevis sits 5 mm forward of the upper cover's centre
SKIN_WALL = 1.5
SKIN_N, SKIN_RINGS = 160, 48
SKIN_HOLD = 30.0            # mm below the gap with exactly the upper-arm section, then the blend
SKIN_FRONT_DROP = float(__import__('os').environ.get('APE_SKIN_FRONT_DROP', -28.0))  # top edge height at the front, from the elbow axis
SKIN_WRIST_Z = 160.0         # the old forearm section here (it wrapped the wrist block's shell-mount bosses) and below
CAP_CUT_Z = 285.0           # where the yaw-axis-normal cut crosses the yaw axis (just below the yaw horn)
CAP_GAP = 1.5               # mm between the turning sleeve and the fixed cap


def r180(o):
    """Half turn about the world x axis through o (both elbows turn about +-x)."""
    return np.array([[1, 0, 0, 0], [0, -1, 0, 2 * o[1]], [0, 0, -1, 2 * o[2]], [0, 0, 0, 1.0]])


_IDLER_R = {}


def rear_idler(F, P, side, o):
    """The C047 rear idler on the elbow servo: shoulder-pitch servo (case + horn) registered onto the elbow servo."""
    import itertools
    if side == "L" and "R" in _IDLER_R:
        m = _IDLER_R["R"].copy(); m.vertices[:, 0] *= -1; m.invert(); return m
    L = lambda n: F.load_mesh(P[n]["mesh"])  # noqa: E731
    sc, sh, si = L(f"upper__{side}_pitch_C047_case"), L(f"upper__{side}_pitch_C047_main_horn"), L(f"upper__{side}_pitch_C047_rear_idler")
    dc, dh = L(f"upper__{side}_elbow_C047_case"), L(f"upper__{side}_elbow_C047_main_horn")
    def frame(m):
        c = m.vertices.mean(0); return c, np.linalg.svd(m.vertices - c, full_matrices=False)[2].T
    cs, Rs = frame(sc); cd, Rd = frame(dc)
    src = np.vstack([sc.vertices[::4], sh.vertices[::2]]); dst = np.vstack([dc.vertices[::4], dh.vertices[::2]])
    best = None
    for sg in itertools.product([1, -1], [1, -1], [1, -1]):
        R0 = Rd @ np.diag(sg) @ Rs.T
        if np.linalg.det(R0) < 0:
            continue
        T0 = np.eye(4); T0[:3, :3] = R0; T0[:3, 3] = cd - R0 @ cs
        T, _, cost = trimesh.registration.icp(src, dst, initial=T0, threshold=1e-7, max_iterations=80, reflection=False, scale=False)
        err = np.linalg.norm(trimesh.transform_points(sh.vertices, T).mean(0) - dh.vertices.mean(0))
        if best is None or cost + err < best[0]:
            best = (cost + err, T, err)
    if best[2] > 0.1:
        raise RuntimeError(f"idler registration off by {best[2]:.3f} mm")
    si.apply_transform(best[1]); _IDLER_R[side] = si
    return si


def _outline(mesh_or_meshes, z):
    """Outer boundary (shapely polygon) of shell walls cut at height z."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    ms = mesh_or_meshes if isinstance(mesh_or_meshes, (list, tuple)) else [mesh_or_meshes]
    polys = []
    for m in ms:
        s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        for loop in (s.discrete if s is not None else []):
            if len(loop) > 3:
                pg = Polygon(np.asarray(loop)[:, :2]).buffer(0)
                if pg.area > 1:
                    polys.append(pg)
    u = unary_union(polys).buffer(0.8).buffer(-0.8)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    return Polygon(u.exterior)


def _resample(poly, n):
    """n points along the boundary by arc length, counter-clockwise, starting at the point due +x of the centroid."""
    from shapely.geometry import LineString
    from shapely.geometry.polygon import orient
    ring = orient(poly, 1.0).exterior
    c = np.array(poly.centroid.coords[0])
    start = ring.project(ring.intersection(LineString([c, c + [500, 0]])).representative_point()) if \
        not ring.intersection(LineString([c, c + [500, 0]])).is_empty else 0.0
    L = ring.length
    return np.array([ring.interpolate((start + L * i / n) % L).coords[0] for i in range(n)])


def forearm_skin(F, P, side, sgn, ez, upper_cover):
    """Closed shell from the upper cover's bottom section (below the gap) to the old forearm section at the wrist."""
    old = [F.load_mesh(P[n]["mesh"]) for n in (f"{side}_fore_front_hard_cover", f"{side}_fore_back_hard_cover")]
    z_bot = min(m.bounds[0][2] for m in old) + 1.0
    z_top = ez + SKIN_TOP
    top = _outline(upper_cover, ez + 8.0)  # the full upper-arm section just above the elbow
    from shapely.affinity import translate
    top = translate(top, xoff=sgn * SKIN_TOP_SHIFT_X, yoff=SKIN_TOP_SHIFT_Y)
    bot = _outline(old, SKIN_WRIST_Z)
    to, bo = _resample(top, SKIN_N), _resample(bot, SKIN_N)
    ti, bi = _resample(top.buffer(-SKIN_WALL, join_style=2), SKIN_N), _resample(bot.buffer(-SKIN_WALL, join_style=2), SKIN_N)
    if sgn < 0:  # keep the left side a true mirror of the right: start points mirror too
        pass
    rings_o, rings_i, zs = [], [], np.linspace(z_top, z_bot, SKIN_RINGS)
    for z in zs:
        t = float(np.clip((z_top - SKIN_HOLD - z) / (z_top - SKIN_HOLD - SKIN_WRIST_Z), 0.0, 1.0)); s = 3 * t ** 2 - 2 * t ** 3
        rings_o.append(np.c_[(1 - s) * to + s * bo, np.full(SKIN_N, z)])
        rings_i.append(np.c_[(1 - s) * ti + s * bi, np.full(SKIN_N, z)])
    V = np.vstack(rings_o + rings_i); K, N = SKIN_RINGS, SKIN_N
    oi = lambda k, j: k * N + j % N          # noqa: E731
    ii = lambda k, j: (K + k) * N + j % N    # noqa: E731
    Fc = []
    for k in range(K - 1):
        for j in range(N):
            Fc += [[oi(k, j), oi(k + 1, j), oi(k + 1, j + 1)], [oi(k, j), oi(k + 1, j + 1), oi(k, j + 1)]]
            Fc += [[ii(k, j), ii(k + 1, j + 1), ii(k + 1, j)], [ii(k, j), ii(k, j + 1), ii(k + 1, j + 1)]]
    for k in (0, K - 1):
        for j in range(N):
            Fc += [[oi(k, j), oi(k, j + 1), ii(k, j + 1)], [oi(k, j), ii(k, j + 1), ii(k, j)]]
    skin = trimesh.Trimesh(V, np.array(Fc), process=True)
    trimesh.repair.fix_normals(skin)
    # slanted top edge: SKIN_TOP at the back, SKIN_FRONT_DROP at the front (the inside of the elbow)
    yb, yf = skin.bounds[1][1], skin.bounds[0][1]
    d = np.array([0.0, yf - yb, SKIN_FRONT_DROP - SKIN_TOP]); n = np.array([0.0, d[2], -d[1]]); n /= np.linalg.norm(n)
    if n[2] > 0:
        n = -n
    skin = trimesh.intersections.slice_mesh_plane(skin, n, [0.0, yb, ez + SKIN_TOP], cap=True)
    split_y = -3.0  # the old covers' front/back seam
    front = trimesh.intersections.slice_mesh_plane(skin, [0, -1, 0], [0, split_y, 0], cap=True)
    back = trimesh.intersections.slice_mesh_plane(skin, [0, 1, 0], [0, split_y + 0.3, 0], cap=True)
    return front, back


def box(C, spec, sgn, ez):
    xs = sorted(sgn * x for x in spec["x"])
    return C.box((xs[0], spec["y"][0], ez + spec["z"][0]), (xs[1], spec["y"][1], ez + spec["z"][1]))


def flip(new, side, h):
    """h: helpers from build.py (C, F, load_step, save_step, save_mesh, centroid, log)."""
    C, F = h["C"], h["F"]
    sgn = 1.0 if side == "R" else -1.0
    P = {p["name"]: p for p in new["parts"]}
    J = {j["name"]: j for j in new["joints"]}
    o = np.array(J[f"{side}_elbow"]["origin_world_mm"], float)
    T = r180(o)
    ez = float(o[2])
    up, fore = f"{side}_upper_arm", f"{side}_forearm"
    notes = []

    def move_mesh_part(p, link, sub="flipped"):
        m = F.load_mesh(p["mesh"])
        if p.get("placement_offset_mm"):
            m.apply_translation(p.pop("placement_offset_mm"))
        m.apply_transform(T)
        p.update(h["save_mesh"](p["name"], m, sub))
        p["owning_link"] = link
        if p.get("native_step"):
            p["native_step_transform_note"] = "native STEP is pre-flip; mesh is turned 180 deg about the elbow axis"

    # purchased parts and hardware
    for p in new["parts"]:
        n = p["name"]
        if n == f"upper__{side}_elbow_C047_case" or n.startswith(f"upper__{side}_elbow_case_"):
            move_mesh_part(p, up); notes.append(f"{n} -> {up}")
        elif n == f"upper__{side}_elbow_C047_main_horn" or n.startswith(f"upper__{side}_elbow_horn_") \
                or n == f"upper__{side}_elbow_central_rocker_M3x6":
            move_mesh_part(p, fore); notes.append(f"{n} -> {fore}")

    drop = [p["name"] for p in new["parts"] if p["name"].startswith(f"upper__{side}_elbow_case_")
            and np.mean(np.array(p["bounds_mm"])[:, 2]) - o[2] < 20.0]
    new["parts"][:] = [p for p in new["parts"] if p["name"] not in drop]
    notes.append(f"dropped the lower rear-face case screws inside the idler cheek's circle: {sorted(drop)}")
    P = {p["name"]: p for p in new["parts"]}
    idler = rear_idler(F, P, side, o)
    rec = dict(name=f"upper__{side}_elbow_C047_rear_idler", owning_link=fore, role="output", parent_assembly=P[f"upper__{side}_elbow_C047_main_horn"].get("parent_assembly"),
               color=P[f"upper__{side}_elbow_C047_main_horn"].get("color"), display_group="mechanism", display_groups=["mechanism"],
               source="C047 rear idler (shoulder-pitch servo model) registered onto the elbow servo", mass_kg=0.003)
    rec.update(h["save_mesh"](rec["name"], idler, "flipped")); new["parts"].append(rec)
    flip_motor_sign(J[f"{side}_elbow"])
    # connectors
    s3, s4 = P[f"{side}_upper_arm_yaw_to_elbow_combined"], P[f"{side}_forearm_elbow_to_wrist_combined"]
    a3, a4 = h["load_step"](s3), h["load_step"](s4)
    top3 = C.common(a3, C._halfspace("z", ez + Z3, +1)); bot3 = C.common(a3, C._halfspace("z", ez + Z3, -1))
    top4 = C.common(a4, C._halfspace("z", ez + Z4, +1)); bot4 = C.common(a4, C._halfspace("z", ez + Z4, -1))
    new3 = C.fuse(top3, C.transform(top4, T))
    import forearm_clevis as FC  # owner 2026-11-13: forearm grips the servo on both sides, one tapered body to the wrist
    bracket = C.common(a3, C._halfspace("z", FC.BRACKET_KEEP_Z, +1))
    col_out, col_in = FC.column(sgn)
    new3 = C.cut(C.fuse(bracket, C.transform(top4, T), col_out), col_in)  # the old side strap and beam are gone
    new3 = C.cut(new3, FC.cradle_relief(C, o, sgn))
    new4 = FC.build(C, o, sgn, C.transform(bot3, T), bot4)
    for name, shape, part in ((s3["name"], new3, s3), (s4["name"], new4, s4)):
        sols = C.solids(shape)
        if len(sols) > 1:
            notes.append(f"{name}: {len(sols) - 1} loose fragments dropped "
                         f"({sum(C.volume(x) for x in sols) - C.volume(C.largest_solid(shape)):.1f} mm3)")
            shape = C.largest_solid(shape)
        part.update(h["save_step"](name, shape, "printed")); part.pop("native_step_offset_mm", None)

    # steel sleeves follow their piece
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    sl3, sl4 = P[s3["name"] + "_steel_sleeves"], P[s4["name"] + "_steel_sleeves"]
    keep3, keep4 = [], []
    for so in C.solids(h["load_step"](sl3)):
        (keep3 if h["centroid"](so)[2] > ez + Z3 else keep4).append(so if h["centroid"](so)[2] > ez + Z3 else C.transform(so, T))
    for so in C.solids(h["load_step"](sl4)):
        (keep3 if h["centroid"](so)[2] > ez + Z4 else keep4).append(C.transform(so, T) if h["centroid"](so)[2] > ez + Z4 else so)
    for part, sols in ((sl3, keep3), (sl4, keep4)):
        comp = TopoDS_Compound(); b = BRep_Builder(); b.MakeCompound(comp)
        for so in sols:
            b.Add(comp, so)
        part.update(h["save_step"](part["name"], comp, "purchased")); part.pop("native_step_offset_mm", None)
        part["native_solid_count"] = len(sols)
    notes.append(f"sleeves: stage 3 {len(keep3)}, stage 4 {len(keep4)}")

    # shells: split the upper-arm cover at the yaw joint, extend the sleeve, trim the forearm covers
    cover = P[f"{side}_upper_arm_hard_cover"]
    m = F.load_mesh(cover["mesh"])
    yaw = J[f"{side}_shoulder_yaw"]
    ax = np.array(yaw["axis_world_neutral"], float); ax /= np.linalg.norm(ax)
    if ax[2] < 0:
        ax = -ax
    p0 = np.array(yaw["origin_world_mm"], float); p0 = p0 + ax * (CAP_CUT_Z - p0[2]) / ax[2]
    cap = trimesh.intersections.slice_mesh_plane(m, ax, p0 + ax * CAP_GAP / 2, cap=m.is_watertight)
    sleeve = trimesh.intersections.slice_mesh_plane(m, -ax, p0 - ax * CAP_GAP / 2, cap=m.is_watertight)
    bottom = sleeve.bounds[0][2]
    z_cut = bottom + 8.0  # above the rim's inward lip: a straight section
    sleeve = trimesh.intersections.slice_mesh_plane(sleeve, [0, 0, 1], [0, 0, z_cut], cap=sleeve.is_watertight)
    v = np.asarray(sleeve.vertices, float).copy()
    low = v[:, 2] < z_cut + 0.01  # the cut rim and its cap drop straight down: an extrusion of that section
    v[low, 2] = ez + SLEEVE_BOTTOM
    sleeve = trimesh.Trimesh(v, sleeve.faces, process=True)
    template = sleeve.copy()  # full section, the forearm shell continues it
    yb, yf = sleeve.bounds[1][1], sleeve.bounds[0][1]
    d = np.array([0.0, yf - yb, UPPER_EDGE[1] - UPPER_EDGE[0]]); nrm = np.array([0.0, d[2], -d[1]]); nrm /= np.linalg.norm(nrm)
    if nrm[2] < 0:
        nrm = -nrm  # keep above the slanted edge
    sleeve = trimesh.intersections.slice_mesh_plane(sleeve, nrm, [0.0, yb, ez + UPPER_EDGE[0]], cap=sleeve.is_watertight)
    win_x = sorted([sgn * 150.0, sgn * 175.0])
    window = trimesh.creation.cylinder(radius=LATERAL_WINDOW_R, segment=[[win_x[0], o[1], o[2]], [win_x[1], o[1], o[2]]], sections=64)
    if sleeve.is_watertight:
        sleeve = trimesh.boolean.difference([sleeve, window], engine="manifold")
    else:
        keep = ~((np.abs(sleeve.triangles_center[:, 0]) > 150.0) & (np.hypot(sleeve.triangles_center[:, 1] - o[1], sleeve.triangles_center[:, 2] - o[2]) < LATERAL_WINDOW_R))
        sleeve.update_faces(keep); sleeve.remove_unreferenced_vertices()
    cover.update(h["save_mesh"](cover["name"], cap))
    cover["mesh_note"] = "shoulder cap: upper-arm cover above the yaw-axis-normal cut (elbow flip)"
    sleeve_part = dict(cover, name=f"{side}_upper_arm_lower_cover", owning_link=up)
    for k in ("native_step", "native_step_unmorphed", "native_step_sha256", "volume_mm3"):
        sleeve_part.pop(k, None)
    sleeve_part.update(h["save_mesh"](sleeve_part["name"], sleeve))
    sleeve_part["mesh_note"] = "upper-arm sleeve: below the yaw-axis-normal cut, on the upper-arm link, extended over the elbow case"
    new["parts"].append(sleeve_part)
    front, back = forearm_skin(F, P, side, sgn, ez, template)
    for n, fm in ((f"{side}_fore_front_hard_cover", front), (f"{side}_fore_back_hard_cover", back)):
        P[n].update(h["save_mesh"](n, fm))
        P[n]["mesh_note"] = "forearm skin: the upper-arm cover's bottom section continued below the elbow gap, blended to the wrist section"
    notes.append(f"cover split at the yaw joint; lower cover edge {UPPER_EDGE} mm above the elbow axis (back, front) with a {LATERAL_WINDOW_R} mm window; watertight {sleeve.is_watertight}")
    for nt in notes:
        h["log"].append((f"{side} elbow flip", nt))
    return notes
