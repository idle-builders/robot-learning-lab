"""Phase 3: Build clean, watertight OpenCascade BRep STEP/STL/NPZ models for all 4 Neck & Head
connector stages across Family 1 (`truss_box` — Parametric 3D Space-Truss & Ribbed Box-Cradle)
and Family 2 (`topopt` — Blank-State 3D SIMP Topology-Optimized Monocoque), with asymmetric
3D beveling (`r_inner > r_outer`); stages 1-2 with pressed-in 304 SS compression sleeves, stages 3-4 with brass heat-set inserts
and plain clearance holes (stage 3 case screws on ISO 7089 washers, T-acpw7j).
"""
from __future__ import annotations

from pathlib import Path
import shutil
import time
import numpy as np
from build123d import (
    Align,
    Axis,
    Box,
    Circle,
    Cylinder,
    Face,
    Plane,
    Polygon,
    Pos,
    Rectangle,
    SlotOverall,
    Solid,
    Wire,
    export_step,
    export_stl,
)

from common import (
    ART,
    MATERIALS,
    STAGE_LINKS,
    STAGE_PART_BASENAMES,
    bevel_3d_edges,
    build_stage_sleeves,
    cyl_along,
    pitch_case_joint,
    extrude_xy_to_x,
    extrude_xy_to_y,
    extrude_xy_to_z,
    fillet_2d_face,
    rel,
    save_solid_npz,
    stage_interface_specs,
    write_json,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("build-neck-head-connectors")   # configured values: robot/body/neck-head/build_neck_head_connectors.bom.yaml config:


RIB_HEIGHT = CFG["RIB_HEIGHT"]


def _seat(connector: str, group: str) -> float:
    """Coordinate along the hole axis of a group's seat plane (the standard part's outer face), from the definitions."""
    import defined_holes as DH
    xs = [float(np.dot(x["centre"], x["axis_out"])) for x in DH.seat_sites(connector) if x["group"] == group]
    return float(np.mean(xs))


def teardrop_along(pt, direction, r: float, length: float, apex_dir, flat_at: float | None = None):
    """A cylinder of radius r along `direction` from `pt` with a 45 deg teardrop point toward `apex_dir` (perpendicular to the axis): the
    horizontal hole / boss profile FDM prints without support (apex up for a hole, down for a boss). flat_at: the point cut flat at that
    distance from the axis (a boss's apex would otherwise stand as a knife edge past the circle)."""
    from build123d import Plane
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    a = np.asarray(apex_dir, float); a = a - np.dot(a, d) * d; a = a / np.linalg.norm(a)
    pl = Plane(origin=tuple(float(x) for x in pt), x_dir=tuple(float(x) for x in a), z_dir=tuple(float(x) for x in d))
    q = r / np.sqrt(2.0)
    tri = Polygon((q, -q), (r * np.sqrt(2.0), 0.0), (q, q), align=None)
    face = (Circle(r) + tri).faces()[0]
    if flat_at is not None and flat_at < r * np.sqrt(2.0):
        face = (face - Pos(flat_at + r, 0.0) * Rectangle(2 * r, 4 * r)).faces()[0]
    return Solid.extrude(pl.location * face, tuple(float(x) * length for x in d))


def revolved_column(c, segments):
    """A printed column along +y at centre (x, z) of c: cylinders and 45 deg cones (kind, y0, y1, r0, r1) fused."""
    sol = None
    for kind, y0, y1, r0, r1 in segments:
        if y1 - y0 <= 1e-6:
            continue
        if kind == "cyl":
            piece = cyl_along(np.array([float(c[0]), float(y0), float(c[2])]), np.array([0.0, 1.0, 0.0]), float(r0), float(y1 - y0))
        else:
            from build123d import Plane
            piece = Solid.make_cone(float(r0), float(r1), float(y1 - y0), Plane(origin=(float(c[0]), float(y0), float(c[2])), z_dir=(0.0, 1.0, 0.0)))
        sol = piece if sol is None else sol + piece
    return Solid(sol.wrapped) if hasattr(sol, "wrapped") else sol


# (stage, role) sites on a plate whose thickness equals the sleeve length (screws into the XM335 holes, generated from the
# definitions by defined_holes.py / neck_screws.py)
FLUSH_SITES = {("stage1_neck_root_cradle", "load"), ("stage2_neck_yaw_to_pitch_link", "support"),
               ("stage2_neck_yaw_to_pitch_link", "load"), ("stage3_neck_pitch_to_head_cradle", "support")}


def _cut_stage_bores(sol: Solid, stage_key: str) -> Solid:
    """Cut all sleeved bolt bores (`ro = sleeve_ro`), spotfaces, rocker center bores, and auxiliary pilot bores."""
    specs = stage_interface_specs()[stage_key]
    cut_tools = []
    keep_out = []
    for role_key in [r for r in ("support", "load", "camera", "mic") if r in specs]:
        grp = specs[role_key]
        centres = grp["centres"]
        normals = grp.get("normals", [grp["normal"]] * len(centres))
        ro = float(grp["sleeve_ro"])
        h = float(grp["sleeve_len"])
        for k, (pt, n_raw) in enumerate(zip(centres, normals)):
            n = np.asarray(n_raw, dtype=float)
            n = n / np.linalg.norm(n)
            if grp.get("no_sleeve"):
                # T-5bqc3m K2: plain clearance hole through the plate (and the printed column of a compute post), no sleeve
                pt = np.asarray(pt, float)
                if stage_key == "stage1_neck_root_cradle":
                    # torso anchor (T-jxshhv): clearance hole through the spine plate; the ISO 7380-1 head on the rear face keeps a ring
                    # of head_clear_r free behind the plate (head, hex key)
                    cut_tools.append(cyl_along(pt - 1.0 * n, n, ro, h + 2.0))
                    keep_out.append(cyl_along(pt + h * n, n, float(grp["head_clear_r"]), 6.0))
                    continue
                if stage_key == "stage3_neck_pitch_to_head_cradle" and role_key == "support":
                    # pitch case (T-acpw7j): horizontal clearance hole through the rear plate as a 45 deg teardrop pointing up in the print
                    # direction, flat at teardrop_flat_mm; the washer and head on the rear face keep a ring of head_clear_r free behind it
                    cut_tools.append(teardrop_along(pt - 1.0 * n, n, ro, h + 2.0, grp["teardrop_up"], flat_at=float(grp["teardrop_flat_mm"])))
                    keep_out.append(cyl_along(pt + (h + 0.02) * n, n, float(grp["head_clear_r"]), 6.0))   # starts past the face: no skim
                    continue
                if role_key == "support":        # neck root: ISO 7380-1 head on the rear face, a clearance cut for it through the dorsal ribs
                    cut_tools.append(cyl_along(pt - 1.0 * n, n, ro, h + 2.0))
                    w = max(float(grp["rib_window_half"]), float(grp["head_clear_r"]))
                    keep_out.append(Pos(float(pt[0]), float(np.dot(pt, n)) + h + 3.0, float(pt[2])) * Box(2 * w, 6.0, 2 * w))
                else:                            # compute: through the column, counterbore for the washer where the post has a column
                    post = grp["posts"][k]
                    top = float(post["column_top"]) if post["kind"] == "column" else float(np.dot(pt, n)) + h
                    cut_tools.append(cyl_along(pt - 1.0 * n, n, ro, top - float(np.dot(pt, n)) + 2.0))
                    if post["kind"] == "column":
                        cut_tools.append(cyl_along(pt - 1.0 * n, n, float(grp["counterbore_r"]), float(post["floor"]) - float(np.dot(pt, n)) + 1.0))
                continue
            if role_key in ("camera", "mic"):
                # heat-set insert (hole = 2 ro, length h) pushed in from the far end (camera: the plate's rear face; mic: the boss end) up to a ledge,
                # a clearance bore through the column / the wing's floor in front of it (T-5bqc3m)
                start = pt + (float(grp["sleeve_starts"][k]) - float(np.dot(pt, n))) * n
                bore = float(grp["column" if role_key == "camera" else "wing"]["bore_dia"]) / 2.0
                if role_key == "mic":            # horizontal holes: 45 deg teardrops pointing up in the print direction (no support)
                    up = -np.asarray(grp["boss"]["teardrop_down"], float)
                    cut_tools.append(teardrop_along(start, n, ro, h + 3.0, up))
                    cut_tools.append(teardrop_along(start - 8.0 * n, n, bore, 8.2, up))
                    continue
                cut_tools.append(cyl_along(start, n, ro, h + 3.0))
                cut_tools.append(cyl_along(start - 8.0 * n, n, bore, 8.2))
                if role_key == "camera":
                    # the heated tip reaches the insert from the plate's rear face: the ribs behind it keep a square window of half-width
                    # press_clear_mm round the axis (rib ends cut square across their width: a round window left 0.01 mm wedges where it
                    # grazed a rib side)
                    w = float(grp["press_clear_mm"])
                    y0 = float(np.dot(start + h * n, n))
                    rear = max(y0, float(grp["plate_rear"]))
                    keep_out.append(Pos(float(pt[0]), rear + 2.5, float(pt[2])) * Box(2 * w, 5.0, 2 * w))
                continue
            # Cut through bore + outer spotface so dorsal ribs/gussets never obstruct sleeve or driver access
            cut_tools.append(cyl_along(pt - 1.2 * n, n, ro + 0.005, h + 5.2))
            if (stage_key, role_key) in FLUSH_SITES:
                # The sleeve is as long as the plate is thick (flush, no counterbore, M-evmrv6): ribs and walls that rise above the
                # plate keep a ring of 0.75 mm around the sleeve clear for the head, the driver and the heated press tool.
                keep_out.append(cyl_along(pt + h * n, n, ro + 0.75, 5.0))
        if "rocker_center" in grp:
            rc = grp["rocker_center"]
            rr = float(grp["rocker_r"]) + 0.15
            n = np.asarray(grp["normal"], dtype=float)
            n = n / np.linalg.norm(n)
            cut_tools.append(cyl_along(rc - 1.5 * n, n, rr, h + 6.5))

    for ab in specs.get("aux_bores", []):
        pt = ab["pt"]
        n = ab["normal"] / np.linalg.norm(ab["normal"])
        ri = float(ab["ri"])
        h = float(ab["length"])
        cut_tools.append(cyl_along(pt - 0.8 * n, n, ri, h + 2.0))

    res = sol
    for ct in cut_tools:
        res = Solid((res - ct).wrapped)
    for ko in keep_out:     # only where the ring really removes material (a cut that only grazes the plate face leaves sliver facets)
        hit = res & ko
        if hit is not None and float(hit.volume) > 0.05:
            res = Solid((res - ko).wrapped)
    return res


def _skip_near_stage_bores(stage_key: str, margin: float = 0.75):
    specs = stage_interface_specs()[stage_key]
    cyls = []
    for role_key in [r for r in ("support", "load", "camera", "mic") if r in specs]:
        grp = specs[role_key]
        centres = grp["centres"]
        normals = grp.get("normals", [grp["normal"]] * len(centres))
        ro = float(grp["sleeve_ro"]) + margin
        h = float(grp["sleeve_len"])
        for k, (pt, n_raw) in enumerate(zip(centres, normals)):
            n = np.asarray(n_raw, dtype=float) / np.linalg.norm(n_raw)
            if grp.get("no_sleeve"):        # plain holes: the head / washer seat, the counterbore, the column
                if stage_key == "stage1_neck_root_cradle" or (stage_key == "stage3_neck_pitch_to_head_cradle" and role_key == "support"):
                    # torso anchor / pitch case: the plate's hole and the head (washer) seat round it
                    cyls.append((pt, n, float(grp["head_clear_r"]) + margin, h))
                    continue
                if role_key == "support":
                    cyls.append((pt, n, max(float(grp["rib_window_half"]), float(grp["head_clear_r"])) * 1.42 + margin, h + 6.0))
                else:
                    post = grp["posts"][k]
                    r_lim = max(float(grp["counterbore_r"]), float(grp["column"]["base_ro"]) if post["kind"] == "column" else 0.0) + margin
                    cyls.append((pt, n, r_lim, float(post["column_top"]) - float(np.dot(pt, n)) + 1.0))
                continue
            if role_key == "camera":        # the printed column in front of the plate carries the bore and its edges
                cyls.append((pt - float(grp["column"]["height"]) * n, n, ro, h + float(grp["column"]["height"])))
                continue
            if role_key == "mic":           # floor, insert hole and boss end
                f = float(grp["wing_face"])
                cyls.append((pt + (f - 0.5 - float(np.dot(pt, n))) * n, n, ro, float(grp["sleeve_starts"][0]) + h + 1.0 - f))
                continue
            cyls.append((pt, n, ro, h))
        if "rocker_center" in grp:
            n = np.asarray(grp["normal"], dtype=float) / np.linalg.norm(grp["normal"])
            cyls.append((grp["rocker_center"], n, float(grp["rocker_r"]) + margin, h))
    for ab in specs.get("aux_bores", []):
        n = ab["normal"] / np.linalg.norm(ab["normal"])
        cyls.append((ab["pt"], n, float(ab["ri"]) + margin, float(ab["length"])))

    def _fn(x: float, y: float, z: float) -> bool:
        p = np.array([x, y, z], dtype=float)
        for pt, n, r_lim, h in cyls:
            rel_v = p - pt
            ax = float(np.dot(rel_v, n))
            rad = float(np.linalg.norm(rel_v - ax * n))
            if -1.0 <= ax <= h + 1.0 and rad <= r_lim:
                return True
        return False

    return _fn


# ==============================================================================
# STAGE 1: `stage1_neck_root_cradle` (on `torso`)
# ==============================================================================
def build_stage1(family: str) -> Solid:
    """Build Stage 1 (`neck_root_cradle`): connects the 2x M3 torso anchor screws (neck_anchor.json: x -30 / -17, z 347.5, either side
    of the `X in [-26.6, -20.3]` riser column; T-jxshhv) through the riser column onto the `Z = 359.0` shelf and wraps around the `-X`
    cheek and `+Y` rear face (`4x M2`) of `upper__neck_yaw_case_XM335`. The spine plate (y face_y .. plate_rear_y of neck_anchor.json,
    x / z from its cradle_plate box) lies flat on the chassis pad; its 2 holes are plain clearance holes (no sleeves).
    """
    case_face_y = _seat("neck_root_cradle", "yaw_case_rear")    # the yaw case's rear face; the sleeves are 2.5 mm long
    from common import neck_anchor
    na = neck_anchor()
    (px0, px1), (pz0, pz1) = na["cradle_plate"]["x_mm"], na["cradle_plate"]["z_mm"]
    py0, py1 = float(na["face_y_mm"]), float(na["plate_rear_y_mm"])
    plate_c, plate_wh = ((px0 + px1) / 2.0, (pz0 + pz1) / 2.0), (px1 - px0, pz1 - pz0)
    if family == "truss_box":
        # 1. Lower spine wing plate in XZ plane (neck_anchor.json cradle_plate box, y face .. plate rear; T-jxshhv)
        f_spine = fillet_2d_face(Pos(*plate_c) * Rectangle(*plate_wh), r_default=1.5, r_inner=1.9)
        s_spine_plate = extrude_xy_to_y(f_spine, py0, py1)

        # 2. Pure vertical riser column in XZ plane (Y in [25.0, 33.50], X in [-26.6, -20.3], Z in [343.6, 365.6])
        # Strictly X <= -20.3 so it clears Common_power_carrier_mounting_r33 (X >= -20.0) with 0.0000 mm^3 overlap
        f_riser = fillet_2d_face(
            Pos(-23.45, 354.60) * Rectangle(6.3, 22.0),
            r_default=1.4,
            r_inner=1.8,
            skip_xy=[(-26.6, 343.6), (-20.3, 343.6)],
        )
        s_riser = extrude_xy_to_y(f_riser, 25.0, 33.50)

        # 3. Rear motor cradle + upper bridge (Y in [30.5, 33.0], X in [-24.0, 9.2], Z in [353.6, 365.6])
        pts_motor = [
            (-24.0, 359.2),
            (-9.55, 359.2),
            (-9.55, 353.6),
            (9.2, 353.6),
            (9.2, 365.6),
            (-24.0, 365.6),
        ]
        f_motor = fillet_2d_face(Polygon(*pts_motor, align=None), r_default=1.6, r_inner=2.2)
        win_m = fillet_2d_face(Pos(0.0, 359.5) * Rectangle(5.6, 6.0), r_default=1.4, r_inner=1.8)
        f_motor = (f_motor - win_m).faces()[0]
        s_motor = extrude_xy_to_y(f_motor, case_face_y, case_face_y + 2.5)

        # 4. Dorsal stiffening rib across rear cradle (X in [-24.0, 3.6], Y in [33.0, 34.6], Z in [361.4, 364.2])
        f_rib = fillet_2d_face(Pos(-10.2, 362.8) * Rectangle(27.6, 2.8), r_default=1.0, r_inner=1.4)
        s_rib = extrude_xy_to_y(f_rib, 33.0, 34.60)

        # 5. Left motor cheek (X in [-11.8, -9.55], Y in [20.5, 33.5], Z in [359.2, 364.6])
        f_cheek = fillet_2d_face(
            Pos(27.0, 361.9) * Rectangle(13.0, 5.4),
            r_default=1.2,
            r_inner=1.6,
            skip_xy=[(33.5, 359.2), (33.5, 364.6)],
        )
        s_cheek = extrude_xy_to_x(f_cheek, -11.8, -9.55)

        raw = Solid((s_spine_plate + s_riser + s_motor + s_rib + s_cheek).wrapped)

    else:
        # Family 2 (`topopt`): Organic SIMP arch with twin-lobe motor yoke & sculpted teardrop window
        f_spine = fillet_2d_face(Pos(*plate_c) * Rectangle(*plate_wh), r_default=1.6, r_inner=2.0)
        s_spine_plate = extrude_xy_to_y(f_spine, py0, py1)

        f_riser = fillet_2d_face(
            Pos(-23.45, 354.60) * Rectangle(6.3, 22.0),
            r_default=1.5,
            r_inner=1.9,
            skip_xy=[(-26.6, 343.6), (-20.3, 343.6)],
        )
        win_r = Pos(-23.45, 355.0) * Circle(1.40)
        f_riser = (f_riser - win_r).faces()[0]
        s_riser = extrude_xy_to_y(f_riser, 25.0, 33.50)

        pts_motor = [
            (-24.0, 359.2),
            (-9.55, 359.2),
            (-9.55, 353.6),
            (9.2, 353.6),
            (9.2, 365.6),
            (-24.0, 365.6),
        ]
        f_motor = fillet_2d_face(Polygon(*pts_motor, align=None), r_default=1.7, r_inner=2.3)
        win_m = Pos(0.0, 359.5) * SlotOverall(6.2, 4.4, rotation=90)
        f_motor = (f_motor - win_m).faces()[0]
        s_motor = extrude_xy_to_y(f_motor, case_face_y, case_face_y + 2.5)

        f_bridge = fillet_2d_face(Pos(-10.2, 362.8) * Rectangle(27.6, 2.8), r_default=1.0, r_inner=1.4)
        s_bridge = extrude_xy_to_y(f_bridge, case_face_y + 2.5, case_face_y + 2.5 + RIB_HEIGHT)

        f_cheek = fillet_2d_face(
            Pos(27.0, 361.9) * Rectangle(13.0, 5.4),
            r_default=1.2,
            r_inner=1.6,
            skip_xy=[(33.5, 359.2), (33.5, 364.6)],
        )
        s_cheek = extrude_xy_to_x(f_cheek, -11.8, -9.55)

        raw = Solid((s_spine_plate + s_riser + s_motor + s_bridge + s_cheek).wrapped)

    cut_sol = _cut_stage_bores(raw, "stage1_neck_root_cradle")
    return bevel_3d_edges(cut_sol, r_outer=0.38, r_inner=0.72, skip_fn=_skip_near_stage_bores("stage1_neck_root_cradle"))


# ==============================================================================
# STAGE 2: `stage2_neck_yaw_to_pitch_link` (on `neck_yaw`)
# ==============================================================================
def build_stage2(family: str) -> Solid:
    """Build Stage 2 (`neck_yaw_to_pitch_link`): connects all 4x M2 holes on `upper__neck_yaw_front_horn`
    (`Z = 372.5`, center `(0, 5, 372.5)`) to all 4x M2 holes on `upper__neck_pitch_front_horn`
    (`X = -21.0`, center `(-21, 5, 414.0)`), with a stepped underside (`Z >= 374.5` for `X < -9.2`)
    that clears `Power_supervisor_PA12_open_tray` (`Z_max = 374.0`) across all yaw angles, deep
    triangulated corner gussets, and a circular `R = 9.4 mm` upper dome.
    """
    z_horn = _seat("neck_yaw_to_pitch_link", "yaw_horn")             # yaw horn face (definition); the sleeves are 3.5 mm long
    x_horn = -_seat("neck_yaw_to_pitch_link", "pitch_horn")          # pitch horn face (axis -X)
    # 1. Yaw horn mounting pad (X in [-9.2, 8.8], Y in [-4.4, 14.4], Z in [372.5, 376.0]): as thick as the sleeves are long
    f_pad = fillet_2d_face(
        Pos(-0.2, 5.0) * Rectangle(18.0, 18.8),
        r_default=2.4,
        r_inner=3.0,
        skip_xy=[(-9.2, -4.4), (-9.2, 14.4)],
    )
    s_pad = extrude_xy_to_z(f_pad, z_horn, z_horn + 3.5)

    # 2. Stepped horizontal bridge deck (X in [-24.5, 8.8], Y in [-4.4, 14.4], Z in [374.5, 377.5])
    # Clears Power_supervisor_PA12_open_tray (Z_max = 374.0 mm) across all ±60 deg yaw angles!
    # It ends at X = -8.0, 0.25 mm short of the sleeve of the yaw hole at X = -6 (the pad carries the heads there).
    pts_base = [
        (x_horn - 3.5, -4.4),
        (-8.0, -4.4),
        (-8.0, 14.4),
        (x_horn - 3.5, 14.4),
    ]
    f_base = fillet_2d_face(Polygon(*pts_base, align=None), r_default=2.4, r_inner=3.2, skip_xy=[(x_horn - 3.5, -4.4), (x_horn - 3.5, 14.4), (-8.0, -4.4), (-8.0, 14.4)])
    s_base = extrude_xy_to_z(f_base, 374.5, 377.5)

    # 3. Vertical left arm in YZ plane (X in [-24.5, -21.0]), capped by tangent circular R = 9.4 mm dome at (Y=5, Z=414)
    f_rect = Pos(5.0, 394.25) * Rectangle(18.8, 39.5)  # Y in [-4.4, 14.4], Z in [374.5, 414.0]
    f_dome = Pos(5.0, 414.0) * Circle(9.4)             # Y in [-4.4, 14.4], R = 9.4 mm <= 10.61 mm Wuerth clearance!
    f_vert = (f_rect + f_dome).faces()[0]
    if family == "truss_box":
        win_v = fillet_2d_face(Pos(5.0, 395.0) * Rectangle(5.4, 11.5), r_default=1.8, r_inner=2.4)
    else:
        win_v = Pos(5.0, 395.0) * SlotOverall(12.0, 5.6, rotation=90)
    f_vert = (f_vert - win_v).faces()[0]
    s_vert = extrude_xy_to_x(f_vert, x_horn - 3.5, x_horn)

    # 4. Deep triangulated anterior & posterior 3D corner gusset webs (Y in [-4.4, -1.3] and Y in [11.3, 14.4])
    # spanning X in [-23.0, -7.5] at the base (Z in [375.5, 381.4]) and up the column (X in [-23.0, -19.5], Z up to 396.0)
    pts_gus = [
        (-23.0, 375.5),
        (-7.5, 375.5),
        (-7.5, 379.2),
        (-19.5, 381.8),
        (-19.5, 392.0),
        (-21.0, 397.0),
        (-23.0, 397.0),
    ]
    f_gus = Polygon(*pts_gus, align=None).faces()[0]
    s_gus_front = extrude_xy_to_y(f_gus, -4.4, -1.3)
    s_gus_rear = extrude_xy_to_y(f_gus, 11.3, 14.4)

    # 5. Central lower base spine rib (Y in [2.6, 7.4], X in [-23.0, -7.5], Z in [375.5, 383.5])
    # ends at X = -9.0: the 0.75 mm ring around the sleeve of the yaw hole at X = -6 stays clear for head, driver and press tool
    pts_spine = [
        (-23.0, 375.5),
        (-9.0, 375.5),
        (-9.0, 378.8),
        (-19.5, 381.8),
        (-21.0, 385.0),
        (-23.0, 385.0),
    ]
    f_spine = Polygon(*pts_spine, align=None).faces()[0]
    s_spine = extrude_xy_to_y(f_spine, 2.6, 7.4 if family == "truss_box" else 7.2)

    raw = Solid((s_pad + s_base + s_vert + s_gus_front + s_gus_rear + s_spine).wrapped)
    cut_sol = _cut_stage_bores(raw, "stage2_neck_yaw_to_pitch_link")
    return bevel_3d_edges(cut_sol, r_outer=0.38, r_inner=0.72, skip_fn=_skip_near_stage_bores("stage2_neck_yaw_to_pitch_link", margin=1.05))


# ==============================================================================
# STAGE 3: `stage3_neck_pitch_to_head_cradle` (on `head`)
# ==============================================================================
# @lore Stage 3 tongue (T-9wjdqc): the 2 brass M3 x 4 heat-set inserts fill the tongue from its rear face (the carrier's seat, y -12.92) to
# its front face, so the tongue is exactly one insert long; its -x edge at -5.0 keeps 2.15 mm round the 4.0 mm holes at x -0.85 (the front
# corner there is rounded 0.8, not 1.6: a 1.6 corner left 0.55 mm at the front lip); its top at 423.0 keeps 2.0 mm over the upper hole
# (z 419): the carrier is all behind y -12.9 there and the bridge behind the tongue stays at 421.2 under the carrier plate (z >= 422).
TONGUE_X_MIN, TONGUE_TOP, TONGUE_CORNER_R = CFG["TONGUE_X_MIN"], CFG["TONGUE_TOP"], CFG["TONGUE_CORNER_R"]
PLATE_Z_BOTTOM = CFG["PLATE_Z_BOTTOM"]
X_BRIDGE = CFG["X_BRIDGE"]


def _fillet_vertex(face, xy, r: float):
    """Round one vertex of a planar face (the corner at xy, within 0.2 mm) with radius r."""
    from build123d import fillet
    v = [v for v in face.vertices() if (v.X - xy[0]) ** 2 + (v.Y - xy[1]) ** 2 < 0.04]
    return fillet(v, r).faces()[0] if v else face


def _plate_bottom_relief(x_min: float, x_max: float, y_back: float, z_bot: float = PLATE_Z_BOTTOM):
    """Cutter for the cradle rear plate's bottom (pitch_case_joint.json bottom_relief, T-acpw7j): a chamfer along the bottom rear edge
    (edge_chamfer_y_mm into the bottom face, edge_chamfer_z_mm up the rear face, over the plate's whole length) and a flat cut of each
    bottom rear corner (corner_cuts: the -x end at x_min, the +x end at x_max; the plane with that outward normal lies depth_mm below
    the sharp corner (x, y_back, z_bot) along it), where the neck collar passes over neck pitch."""
    rl = pitch_case_joint()["bottom_relief"]
    cy, cz = float(rl["edge_chamfer_y_mm"]), float(rl["edge_chamfer_z_mm"])
    pts = [(y_back - cy, z_bot), (y_back, z_bot + cz), (y_back + 2.0, z_bot + cz), (y_back + 2.0, z_bot - 2.0), (y_back - cy, z_bot - 2.0)]
    cutter = extrude_xy_to_x(Polygon(*pts, align=None), x_min - 1.0, x_max + 1.0)
    near = {"minus_x": Pos(x_min + 1.5, y_back - 1.0, z_bot + 0.5) * Box(5.0, 4.0, 3.0),     # x_min - 1 .. + 4, y_back - 3 .. + 1
            "plus_x": Pos(x_max - 2.5, y_back - 1.5, z_bot + 0.5) * Box(7.0, 5.0, 3.0)}      # x_max - 6 .. + 1, y_back - 4 .. + 1
    for c in rl["corner_cuts"]:
        n = np.asarray(c["normal"], float)
        n /= np.linalg.norm(n)
        p0 = np.array([x_min if c["end"] == "minus_x" else x_max, y_back, z_bot]) - float(c["depth_mm"]) * n
        half = Plane(origin=tuple(float(v) for v in p0 + 5.0 * n), z_dir=tuple(float(v) for v in n)) * Box(10.0, 10.0, 10.0)
        cutter = cutter + (half & near[c["end"]])
    return Solid(cutter.wrapped)


def build_stage3(family: str) -> Solid:
    """Build Stage 3 (`neck_pitch_to_head_cradle`): connects the 4 rear M2 tapped holes on
    `upper__neck_pitch_case_XM335` (`X in {-10.75, -5.25}, Y = 14.5..17.5, Z in {391.5, 401.0}`)
    around the `+X` right side (`X in [3.2, 8.8]`) to the Stage 4 carrier tongue: 2 brass M3 heat-set inserts in the cradle's front
    tongue (head_carrier_posts.json neck_root: x -0.85, z 413 / 419; tongue y from -12.92 forward by the insert's length; T-9wjdqc).
    """
    # 1. U-profile in XY plane (rear plate + right bridge) extruded across Z in [388.5, 421.2], the front tongue on it up to TONGUE_TOP,
    # then sculpted in YZ and XZ
    y_case = _seat("neck_pitch_to_head_cradle", "pitch_case_rear")   # pitch case rear face (definition)
    specs3 = stage_interface_specs()["stage3_neck_pitch_to_head_cradle"]
    g3 = specs3["load"]
    sup3 = specs3["support"]
    # rear plate's rear face: the washer seat, each site's centre + the plate thickness (common.pitch_case_joint, T-acpw7j) along its
    # axis; the case-face coordinate y_case is a dot product with the slightly tilted hole axes and lies 0.006 mm behind it
    y_plate_rear = float(np.mean([(np.asarray(c, float) + float(sup3["sleeve_len"]) * np.asarray(a, float))[1]
                                  for c, a in zip(sup3["centres"], sup3["normals"])]))
    t_plate = y_plate_rear - y_case
    edge = pitch_case_joint()["minus_x_edge"]
    x_min, r_end = float(edge["x_mm"]), float(edge["corner_r_mm"])
    y_rear = float(g3["centres"][0][1])                               # tongue rear face (the carrier's seat)
    y_front = y_rear - float(g3["sleeve_len"])                        # tongue front face: the inserts are flush with both faces
    pts_u = [
        (x_min, y_case + t_plate),
        (X_BRIDGE, y_case + t_plate),
        (X_BRIDGE, y_rear),
        (3.25, y_rear),
        (3.25, y_case),
        (x_min, y_case),
    ]
    # the inner corner at the pitch case's rear edge stays sharp: its 2.4 mm fillet stood 7.63 mm3 inside the case (rv4-5bqc3m); the
    # plate's -x end has its own smaller rounds (T-acpw7j: at 1.6 they left 0.63 .. 0.91 mm of wall beside the case-screw holes)
    f_u = fillet_2d_face(Polygon(*pts_u, align=None), r_default=1.6, r_inner=2.4,
                         skip_xy=[(8.8, y_rear), (3.25, y_rear), (3.25, y_case), (x_min, y_case), (x_min, y_case + t_plate)])
    f_u = _fillet_vertex(_fillet_vertex(f_u, (x_min, y_case), r_end), (x_min, y_case + t_plate), r_end)
    s_u = extrude_xy_to_z(f_u, PLATE_Z_BOTTOM, 421.2)
    pts_t = [(TONGUE_X_MIN, y_front), (8.8, y_front), (8.8, y_rear), (TONGUE_X_MIN, y_rear)]
    f_t = fillet_2d_face(Polygon(*pts_t, align=None), r_default=1.6, skip_xy=[(8.8, y_rear), (TONGUE_X_MIN, y_rear), (TONGUE_X_MIN, y_front)])
    f_t = _fillet_vertex(f_t, (TONGUE_X_MIN, y_front), TONGUE_CORNER_R)
    s_t = extrude_xy_to_z(f_t, 388.5, TONGUE_TOP)
    s_u_full = Solid((s_u + s_t).wrapped)

    # Trim front carrier tongue (Y < -4.5) below Z = 408.8 so the front below Z = 408.8 stays completely open
    cut_front_low = Pos(-6.0, -11.5, 398.0) * Box(24.0, 14.0, 21.6)  # Y in [-18.5, -4.5], Z in [387.2, 408.8]
    # Also cut the rear plate (Y > 13.5, X < 2.5) above Z = 409.0, 6.8 mm over the upper M2 holes (Z 401.0 + 1.2): it frames the 4 rear
    # M2 holes, and the bench driver of the head root screws (z 413 / 419, along +y through the U) passes over it (at 413.0 the plate stood
    # in the lower screw's blade path 24.3 mm from the head, 8.6 mm3, nr-v1-asm; T-9wjdqc)
    cut_rear_high = Pos(-6.5, 16.0, 416.5) * Box(18.0, 5.0, 15.0)    # X in [-15.5, 2.5], Z in [409.0, 424.0]

    s_body = Solid((s_u_full - cut_front_low - cut_rear_high).wrapped)

    # Cut family-specific structural lightening windows through the +X right side bridge (X in [2.5, 9.5])
    if family == "truss_box":
        w1 = fillet_2d_face(Pos(4.5, 398.0) * Rectangle(11.0, 9.5), r_default=1.8, r_inner=2.4)
        w2 = fillet_2d_face(Pos(-3.5, 414.5) * Rectangle(9.5, 7.0), r_default=1.6, r_inner=2.2)
        c1 = extrude_xy_to_x(w1, 2.5, 9.5)
        c2 = extrude_xy_to_x(w2, 2.5, 9.5)
        s_body = Solid((s_body - c1 - c2).wrapped)
    else:
        # Family 2 (`topopt`): Organic SIMP diagonal oval pocket + lower front taper
        # the oval's front end at y -3.18, 1.32 mm behind the front cut's face at y -4.5 (at y 1.5 it crossed that face and left
        # 0.006 / 0.064 mm walls at z 407.35 / 400.69, rv4-5bqc3m)
        w_oval = Pos(4.2, 405.5) * SlotOverall(15.5, 9.2, rotation=28)
        c_oval = extrude_xy_to_x(w_oval, 2.5, 9.5)
        # the taper (same slope) runs on to the tongue's front face, flat at the tongue's underside (408.8)
        pts_taper = [
            (-18.5, 388.0),
            (0.5, 388.0),
            (-16.5, 408.8),
            (-18.5, 408.8),
        ]
        f_tap = Polygon(*pts_taper, align=None)
        c_tap = extrude_xy_to_x(f_tap, 2.5, 9.5)
        s_body = Solid((s_body - c_oval - c_tap).wrapped)

    cut_sol = _cut_stage_bores(s_body, "stage3_neck_pitch_to_head_cradle")
    s_bev = bevel_3d_edges(cut_sol, r_outer=0.38, r_inner=0.72, skip_fn=_skip_near_stage_bores("stage3_neck_pitch_to_head_cradle"))
    # the plate's bottom rear edge and both bottom rear corners give way to the neck collar over neck pitch (T-acpw7j, pitch_case_joint.json
    # bottom_relief), cut after the bevel: cut before it, the chamfer's run-out on the corner rounds broke the bevel's tangent chain
    # round the plate bottom and every 0.38 round on it was dropped; the cut's own edges stay as cut (obtuse)
    return Solid((s_bev - _plate_bottom_relief(x_min, X_BRIDGE, y_case + t_plate)).wrapped)


# ==============================================================================
# STAGE 4: `stage4_head_sensor_compute_carrier` (on `head`)
# ==============================================================================
def build_stage4(family: str) -> Solid:
    """Build Stage 4 (`head_sensor_compute_carrier`): consolidates `single_shared_carrier_v23`
    and `microphone_compact_tab_v23` into a single one-piece FDM carrier with 9 brass M2 heat-set inserts (8 camera columns,
    1 mic boss, T-5bqc3m) and plain catalogue-screw holes elsewhere (2x M3 neck root, 4x M2.5 ROCK4D: ISO 7092 washer seats,
    printed columns at the fan-bridge posts; no steel rings, T-5bqc3m K2), integrated right microphone side wing, radial
    head-skin lugs, and dorsal stiffening ribs in the collision-free `Y in [-11.40, -10.15]` corridor.
    """
    # 1. Main carrier XZ profile (Y in [-12.90, -11.40]); lower neck tongue X in [-4.2, 2.5] clears Stage 3 X >= 3.25
    pts_main = [
        (-4.2, 409.2),
        (2.5, 409.2),
        (2.5, 422.0),
        (9.2, 422.0),
        (41.5, 422.0),
        (44.0, 427.5),
        (44.0, 471.5),
        (39.2, 477.5),
        (-39.2, 477.5),
        (-44.0, 471.5),
        (-44.0, 427.5),
        (-41.5, 420.2),
        (-4.2, 420.2),
    ]
    f_main = fillet_2d_face(Polygon(*pts_main, align=None), r_default=2.0, r_inner=2.8, skip_xy=[(2.5, 422.0)])

    # Cut family-specific lightening windows in non-structural zones between camera & compute bosses
    if family == "truss_box":
        wins = [
            fillet_2d_face(Pos(-21.0, 433.0) * Rectangle(15.0, 8.0), r_default=2.0, r_inner=2.6),
            fillet_2d_face(Pos(-21.0, 456.0) * Rectangle(18.0, 16.0), r_default=2.4, r_inner=3.0),
            fillet_2d_face(Pos(21.0, 458.0) * Rectangle(16.0, 16.0), r_default=2.4, r_inner=3.0),
            fillet_2d_face(Pos(27.5, 433.0) * Rectangle(11.0, 9.5), r_default=2.0, r_inner=2.6),
        ]
    else:
        wins = [
            Pos(-21.0, 433.0) * SlotOverall(15.5, 8.2, rotation=10),
            Pos(-21.0, 456.0) * SlotOverall(18.5, 15.5, rotation=-8),
            Pos(21.0, 458.0) * SlotOverall(16.5, 15.5, rotation=8),
            Pos(27.5, 433.0) * Circle(4.8),
        ]
    for w in wins:
        f_main = (f_main - w).faces()[0]

    import head_holes as HH
    y_front, y_rear = HH.plate_faces()      # where the camera / compute spacers rest (supplier parts)
    specs4 = stage_interface_specs()["stage4_head_sensor_compute_carrier"]
    for role in ("load", "camera"):         # material all round each insert / washer seat: an ear where the profile ends too close
        g = specs4[role]
        for k, c in enumerate(g["centres"]):
            # @lore 0.7 mm rim: 0.6 left a ring thinner than one 0.62 mm print line (walls under one line 45 -> 107 mm2)
            # compute: the ISO 7092 washer seat with 2 lines (1.24 mm) round it (3.74; the old 6.1 mm collar's ear was 3.75), and under a
            # printed column's base (base_ro: a base past the plate's edge left a 0.8 mm ledge)
            if role == "load":
                r_ear = float(g["washer_r"]) + 1.24
                if g["posts"][k]["kind"] == "column":
                    r_ear = max(r_ear, float(g["column"]["base_ro"]))
            else:
                r_ear = max(float(g["sleeve_ro"]) + 0.7, float(g["column"]["ro"]))
            # an ear arc passing within 0.3 mm of an outline corner left a sliver face and micro-edges (compute post 2 by the corner
            # (-44, 427.5), rv3-5bqc3m): grow it past the corner
            for (px, pz) in pts_main:
                dv = float(np.hypot(px - float(c[0]), pz - float(c[2])))
                if abs(dv - r_ear) < 0.3:
                    r_ear = dv + 0.3
            ear = Pos(float(c[0]), float(c[2])) * Circle(r_ear)
            if (ear - f_main).area > 0.05:
                f_main = (f_main + ear).faces()[0]
    s_plate = extrude_xy_to_y(f_main, y_front, y_rear)

    # 2. Dorsal stiffening rib network in the collision-free rear zone Y in [-11.40, -10.15]
    # staying strictly outside the 4 compute_FF4_spacer seats and clear of lightening windows:
    rib1 = extrude_xy_to_y(fillet_2d_face(Pos(-0.7, 442.0) * Rectangle(5.6, 52.0), r_default=1.5), y_rear, y_rear + 1.25)
    rib2 = extrude_xy_to_y(fillet_2d_face(Pos(-1.5, 425.0) * Rectangle(21.0, 3.6), r_default=1.2), y_rear, y_rear + 1.25)
    rib3 = extrude_xy_to_y(fillet_2d_face(Pos(-1.0, 442.5) * Rectangle(60.0, 4.4), r_default=1.2), y_rear, y_rear + 1.25)
    # rib4's +x end stops at the press window of the camera insert at (35, 472.17) (x 35 - press_clear_mm = 30.1): ending at 31 it ran 0.9 mm
    # into the window, and how the bevel met the window's cut face there changed with unrelated edits (a 0.005 mm one-layer sliver and
    # 0.04 mm3 in the iron's path in nr-v1-a; T-9wjdqc)
    x4 = 35.0 - float(HH.POSTS["camera"]["press_clear_mm"])
    rib4 = extrude_xy_to_y(fillet_2d_face(Pos((x4 - 31.0) / 2, 471.5) * Rectangle(x4 + 31.0, 4.2), r_default=1.2), y_rear, y_rear + 1.25)

    # 3. Radial head-skin mounting lugs at X in [-45.5, -43.2] and [43.2, 45.5], Y in [-17.2, -11.4], Z in [447.2, 454.8]
    f_lug = fillet_2d_face(
        Pos(-14.3, 451.0) * Rectangle(5.8, 7.6),
        r_default=1.2,
        r_inner=1.6,
        skip_xy=[(-11.4, 447.2), (-11.4, 454.8)],
    )
    s_lug_l = extrude_xy_to_x(f_lug, -45.5, -43.2)
    s_lug_r = extrude_xy_to_x(f_lug, 43.2, 45.5)

    # 4. Right microphone wing (T-5bqc3m review: mic, insert and cable fitted on the bench before the skin goes on): a front bridge from the
    # plate and a side wing in the board's back plane, thinned where the head skin closes in; one boss on its skin side carries the insert
    # (pushed from the boss end up to the floor), a printed pin on its face goes into the board's second mounting hole, a relief clears the
    # board's acoustic port. Sizes from head_carrier_posts.json mic, positions from the supplier shims' bores (head_holes.py).
    gm = specs4["mic"]
    w = gm["wing"]
    xf = float(gm["wing_face"])
    xt = xf + float(w["t_mm"])
    z_top = float(w["top_z_mm"])
    # bottom edge below the boss (r 3.3 round z 458.3) and into the skin lug's top (454.8): the old 455.2 left the boss 0.2 mm past the
    # edge (sliver faces) and a 0.4 mm slot over the lug
    z_bot = float(w["bottom_z_mm"])
    f_mic_br = fillet_2d_face(Pos((43.0 + xt) / 2, (z_bot + 462.4) / 2) * Rectangle(xt - 43.0, 462.4 - z_bot), r_default=1.2, r_inner=1.6)
    s_mic_br = extrude_xy_to_y(f_mic_br, y_front, y_rear)
    y_diag = -7.80 + (z_top - 462.4) * (6.3 / 10.0)        # the wing's upper-left edge (parallel to the skin's taper, as before)
    pts_mic_side = [(y_front, z_bot), (4.50, z_bot), (4.50, z_top), (y_diag, z_top), (-7.80, 462.4), (y_front, 462.4)]
    f_mic_side = fillet_2d_face(Polygon(*pts_mic_side, align=None), r_default=1.2, r_inner=1.8, skip_xy=[(y_front, z_bot), (y_front, 462.4)])
    s_mic_side = extrude_xy_to_x(f_mic_side, xf, xt)
    zt0 = float(w["taper_z_mm"])
    s_mic_side = Solid((s_mic_side - Pos(xt, 0.0, (zt0 + z_top + 2.0) / 2) * Box(2 * (xt - xf - float(w["taper_t_mm"])), 30.0, z_top + 2.0 - zt0)).wrapped)
    c_mic = np.asarray(gm["centres"][0], float)
    boss_end = float(gm["sleeve_starts"][0]) + float(gm["sleeve_lens"][0])      # the insert's rear end is flush with the boss end
    # horizontal boss: its underside a 45 deg teardrop pointing down in the print direction (prints without support, T-5bqc3m review)
    s_boss = teardrop_along(np.array([xf, c_mic[1], c_mic[2]]), np.array([1.0, 0.0, 0.0]), float(gm["boss"]["ro"]), boss_end - xf,
                            np.asarray(gm["boss"]["teardrop_down"], float), flat_at=float(gm["boss"]["ro"]))
    pin = gm["pin"]
    s_pin = cyl_along(np.array([xf, pin["centre"][1], pin["centre"][2]]), np.array([-1.0, 0.0, 0.0]), float(pin["dia"]) / 2.0, float(pin["length"]))
    port = gm["port"]
    c_port = cyl_along(np.array([xf - 1.0, float(port["y"]), float(port["z"])]), np.array([1.0, 0.0, 0.0]), float(port["dia"]) / 2.0, xt - xf + 2.0)
    s_mic = Solid((s_mic_br + s_mic_side + s_boss + s_pin - c_port).wrapped)
    # 5. Camera columns: the 8 supplier F-F standoffs become printed columns (height = the standoff) on the plate's front face, narrower next to
    # the components on the board's back (radius bands from the board's mesh, head_holes.column_profile, T-5bqc3m K1); the insert sits in their
    # rear end (_cut_stage_bores)
    cols = [revolved_column(np.asarray(p["centre"], float), p["segments"]) for p in HH.column_profile()]
    # 6. Compute columns (T-5bqc3m K2): where no ISO 4762 M2.5 length reaches the nut with the washer on the plate front (the fan-bridge posts),
    # the supplier F-F spacer becomes a printed column from the plate's rear face to the board (gap_mm short), with a base of base_ro round the
    # washer's counterbore and a 45 deg cone up to the column radius
    gl = specs4["load"]
    gc = gl["column"]
    for post in gl["posts"]:
        if post["kind"] != "column":
            continue
        c, ro_c, rb = post["centre"], float(gc["ro"]), float(gc["base_ro"])
        y_base = max(y_rear, float(post["floor"])) + float(gc["base_above_floor_mm"])
        segs = [("cyl", y_rear, y_base, rb, rb), ("cone", y_base, y_base + (rb - ro_c), rb, ro_c),
                ("cyl", y_base + (rb - ro_c), float(post["column_top"]), ro_c, ro_c)]
        cols.append(revolved_column(np.asarray(c, float), segs))
    raw = s_plate + rib1 + rib2 + rib3 + rib4 + s_lug_l + s_lug_r + s_mic
    for b in cols:
        raw = raw + b
    raw = Solid(raw.wrapped)
    cut_sol = _cut_stage_bores(raw, "stage4_head_sensor_compute_carrier")
    return bevel_3d_edges(cut_sol, r_outer=0.32, r_inner=0.60, skip_fn=_skip_near_stage_bores("stage4_head_sensor_compute_carrier"))


BUILDERS = {
    "stage1_neck_root_cradle": build_stage1,
    "stage2_neck_yaw_to_pitch_link": build_stage2,
    "stage3_neck_pitch_to_head_cradle": build_stage3,
    "stage4_head_sensor_compute_carrier": build_stage4,
}


def main():
    t0 = time.time()
    parts_dir = ART / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    candidate_steps: dict[str, dict[str, Path]] = {"truss_box": {}, "topopt": {}}
    manifest_parts = []

    # First export the pressed-in sleeves / inserts of every stage
    sleeves_by_stage = {}
    for stg_key in STAGE_PART_BASENAMES:
        slvs = build_stage_sleeves(stg_key)
        sleeves_by_stage[stg_key] = []
        for s_info in slvs:
            s_name = s_info["name"]
            sp = parts_dir / f"{s_name}.step"
            stlp = parts_dir / f"{s_name}.stl"
            npzp = parts_dir / f"{s_name}.npz"
            export_step(s_info["solid"], str(sp))
            export_stl(s_info["solid"], str(stlp))
            save_solid_npz(s_info["solid"], npzp, tol=0.04)
            sleeves_by_stage[stg_key].append({
                "name": s_name,
                "stage": stg_key,
                "interface": s_info["interface"],
                "owning_link": s_info["owning_link"],
                "od_mm": s_info["od_mm"],
                "id_mm": s_info["id_mm"],
                "length_mm": s_info["length_mm"],
                "volume_mm3": s_info["volume_mm3"],
                "mass_g": s_info["mass_g"],
                "step_path": rel(sp),
                "stl_path": rel(stlp),
                "npz_path": rel(npzp),
            })

    # Build and export all 4 stages for Family 1 (`truss_box`) and Family 2 (`topopt`)
    rho_paht = MATERIALS["PAHT_CF_dry"]["density_g_cm3"]
    for fam_key in ("truss_box", "topopt"):
        for stg_key, bname in STAGE_PART_BASENAMES.items():
            t_s = time.time()
            sol = BUILDERS[stg_key](fam_key)
            part_name = f"{fam_key}__{bname}"
            sp = parts_dir / f"{part_name}.step"
            stlp = parts_dir / f"{part_name}.stl"
            npzp = parts_dir / f"{part_name}.npz"
            export_step(sol, str(sp))
            export_stl(sol, str(stlp))
            save_solid_npz(sol, npzp, tol=0.05)

            vol = float(sol.volume)
            poly_mass = vol * (rho_paht * 1e-3)
            slv_mass = sum(s["mass_g"] for s in sleeves_by_stage[stg_key])
            candidate_steps[fam_key][stg_key] = sp
            manifest_parts.append({
                "name": part_name,
                "family": fam_key,
                "stage": stg_key,
                "part_basename": bname,
                "owning_link": STAGE_LINKS[stg_key],
                "volume_mm3": round(vol, 2),
                "polymer_mass_PAHT_CF_g": round(poly_mass, 3),
                "sleeve_count": len(sleeves_by_stage[stg_key]),
                "sleeve_mass_g": round(slv_mass, 3),
                "total_stage_mass_g": round(poly_mass + slv_mass, 3),
                "step_path": rel(sp),
                "stl_path": rel(stlp),
                "npz_path": rel(npzp),
                "build_elapsed_s": round(time.time() - t_s, 2),
            })
            print(f"Built {part_name:38s}: vol={vol:7.1f} mm^3 | poly={poly_mass:5.2f}g + sleeves={slv_mass:4.2f}g = {poly_mass+slv_mass:5.2f}g ({time.time()-t_s:.2f}s)")

    # Per-stage winners of the combined family: the stored selection of the selected body's combined manifest (it came from the
    # old FEM comparison, which is retired; robot/body/checks fem runs the per-part setups instead).
    import json
    stored = json.loads((ART / "combined" / "manifest.json").read_text())["stages"]
    fem_rep = {"combined_selection": {k: {"selected_family": v["selected_family"]} for k, v in stored.items()}}

    # Copy selected per-stage winners into `combined__<basename>.step/.stl/.npz`
    candidate_steps["combined"] = {}
    for stg_key, bname in STAGE_PART_BASENAMES.items():
        win_fam = fem_rep["combined_selection"][stg_key]["selected_family"]
        src_base = f"{win_fam}__{bname}"
        dst_base = f"combined__{bname}"
        for ext in (".step", ".stl", ".npz"):
            shutil.copy2(parts_dir / f"{src_base}{ext}", parts_dir / f"{dst_base}{ext}")
        candidate_steps["combined"][stg_key] = parts_dir / f"{dst_base}.step"

    summary = {
        "study": "cove50-neck-head-fdm-connectors-20261104",
        "elapsed_s": round(time.time() - t0, 2),
        "parts": manifest_parts,
        "sleeves_by_stage": sleeves_by_stage,
        "combined_selection": fem_rep["combined_selection"],
    }
    write_json(ART / "neck-head-build-summary.json", summary)
    print(f"Build finished in {time.time() - t0:.1f}s; checks: robot/body/checks/run.py (no gates here).")


if __name__ == "__main__":
    main()
