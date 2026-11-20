"""Phase 3: Build clean, watertight OpenCascade BRep STEP/STL/NPZ models for all 4 Neck & Head
connector stages across Family 1 (`truss_box` — Parametric 3D Space-Truss & Ribbed Box-Cradle)
and Family 2 (`topopt` — Blank-State 3D SIMP Topology-Optimized Monocoque), with asymmetric
3D beveling (`r_inner > r_outer`) and 34 pressed-in 304 SS compression sleeves.
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
    extrude_xy_to_x,
    extrude_xy_to_y,
    extrude_xy_to_z,
    fillet_2d_face,
    rel,
    save_solid_npz,
    stage_interface_specs,
    write_json,
)


def _cut_stage_bores(sol: Solid, stage_key: str) -> Solid:
    """Cut all sleeved bolt bores (`ro = sleeve_ro`), spotfaces, rocker center bores, and auxiliary pilot bores."""
    specs = stage_interface_specs()[stage_key]
    cut_tools = []
    for role_key in ("support", "load"):
        grp = specs[role_key]
        centres = grp["centres"]
        normals = grp.get("normals", [grp["normal"]] * len(centres))
        ro = float(grp["sleeve_ro"])
        h = float(grp["sleeve_len"])
        for pt, n_raw in zip(centres, normals):
            n = np.asarray(n_raw, dtype=float)
            n = n / np.linalg.norm(n)
            # Cut through bore + outer spotface so dorsal ribs/gussets never obstruct sleeve or driver access
            cut_tools.append(cyl_along(pt - 1.2 * n, n, ro + 0.005, h + 5.2))
            if stage_key == "stage2_neck_yaw_to_pitch_link":
                # Spotface (r = 2.40 mm) beyond the 3.0 mm sleeve seat for M2x5 pan-head screws
                cut_tools.append(cyl_along(pt + (h - 0.02) * n, n, 2.40, 4.5))
            if stage_key == "stage1_neck_root_cradle" and role_key == "support":
                cut_tools.append(cyl_along(pt + (h - 0.2) * n, n, 1.70, 9.0))
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
    return res


def _skip_near_stage_bores(stage_key: str, margin: float = 0.75):
    specs = stage_interface_specs()[stage_key]
    cyls = []
    for role_key in ("support", "load"):
        grp = specs[role_key]
        centres = grp["centres"]
        normals = grp.get("normals", [grp["normal"]] * len(centres))
        ro = float(grp["sleeve_ro"]) + margin
        h = float(grp["sleeve_len"])
        for pt, n_raw in zip(centres, normals):
            n = np.asarray(n_raw, dtype=float) / np.linalg.norm(n_raw)
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
    """Build Stage 1 (`neck_root_cradle`): connects the 2x M3 torso spine anchor bolts
    (`(-29, 25, 347.5)` & `(-21, 25, 347.5)`) through the `X in [-26.6, -20.3]` riser column
    onto the `Z = 359.0` shelf and wraps around the `-X` cheek and `+Y` rear face (`4x M2`)
    of `upper__neck_yaw_case_XM335`.
    """
    if family == "truss_box":
        # 1. Lower spine wing plate in XZ plane (Y in [25.0, 28.0], X in [-32.8, -17.2], Z in [343.6, 350.5])
        f_spine = fillet_2d_face(Pos(-25.0, 347.05) * Rectangle(15.6, 6.9), r_default=1.5, r_inner=1.9)
        s_spine_plate = extrude_xy_to_y(f_spine, 25.0, 28.0)

        # 2. Pure vertical riser column in XZ plane (Y in [25.0, 33.50], X in [-26.6, -20.3], Z in [343.6, 365.6])
        # Strictly X <= -20.3 so it clears Common_power_carrier_mounting_r33 (X >= -20.0) with 0.0000 mm^3 overlap
        f_riser = fillet_2d_face(
            Pos(-23.45, 354.60) * Rectangle(6.3, 22.0),
            r_default=1.4,
            r_inner=1.8,
            skip_xy=[(-26.6, 343.6), (-20.3, 343.6)],
        )
        s_riser = extrude_xy_to_y(f_riser, 25.0, 33.50)

        # 3. Rear motor cradle + upper bridge (Y in [30.52, 33.50], X in [-24.0, 9.2], Z in [353.6, 365.6])
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
        s_motor = extrude_xy_to_y(f_motor, 30.52, 33.50)

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
        f_spine = fillet_2d_face(Pos(-25.0, 347.05) * Rectangle(15.6, 6.9), r_default=1.6, r_inner=2.0)
        s_spine_plate = extrude_xy_to_y(f_spine, 25.0, 28.0)

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
        s_motor = extrude_xy_to_y(f_motor, 30.52, 33.50)

        f_bridge = fillet_2d_face(Pos(-10.2, 362.8) * Rectangle(27.6, 2.8), r_default=1.0, r_inner=1.4)
        s_bridge = extrude_xy_to_y(f_bridge, 33.0, 34.50)

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
    # 1. Yaw horn mounting pad (X in [-9.2, 8.8], Y in [-4.4, 14.4], Z in [372.5, 377.5])
    f_pad = fillet_2d_face(
        Pos(-0.2, 5.0) * Rectangle(18.0, 18.8),
        r_default=2.4,
        r_inner=3.0,
        skip_xy=[(-9.2, -4.4), (-9.2, 14.4)],
    )
    s_pad = extrude_xy_to_z(f_pad, 372.5, 377.5)

    # 2. Stepped horizontal bridge deck (X in [-24.5, 8.8], Y in [-4.4, 14.4], Z in [374.5, 377.5])
    # Clears Power_supervisor_PA12_open_tray (Z_max = 374.0 mm) across all ±60 deg yaw angles!
    pts_base = [
        (-24.5, -4.4),
        (8.8, -4.4),
        (8.8, 14.4),
        (-24.5, 14.4),
    ]
    f_base = fillet_2d_face(Polygon(*pts_base, align=None), r_default=2.4, r_inner=3.2, skip_xy=[(-24.5, -4.4), (-24.5, 14.4)])
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
    s_vert = extrude_xy_to_x(f_vert, -24.5, -21.0)

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
    pts_spine = [
        (-23.0, 375.5),
        (-7.5, 375.5),
        (-7.5, 378.8),
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
def build_stage3(family: str) -> Solid:
    """Build Stage 3 (`neck_pitch_to_head_cradle`): connects the 4 rear M2 tapped holes on
    `upper__neck_pitch_case_XM335` (`X in {-10.75, -5.25}, Y = 14.5..17.5, Z in {391.5, 401.0}`)
    around the `+X` right side (`X in [3.2, 8.8]`) to the Stage 4 carrier tongue
    (`2x M3` at `X = 0.0, Y in [-15.9, -12.9], Z in {413.0, 419.0}`).
    """
    # 1. U-profile in XY plane extruded across Z in [388.5, 421.2], then sculpted in YZ and XZ
    pts_u = [
        (-13.8, 17.5),
        (8.8, 17.5),
        (8.8, -15.9),
        (-4.2, -15.9),
        (-4.2, -12.92),
        (3.25, -12.92),
        (3.25, 14.52),
        (-13.8, 14.52),
    ]
    f_u = fillet_2d_face(Polygon(*pts_u, align=None), r_default=1.6, r_inner=2.4, skip_xy=[(3.25, -12.92), (-4.2, -12.92)])
    s_u_full = extrude_xy_to_z(f_u, 388.5, 421.2)

    # Trim front carrier tongue (Y < -4.5) below Z = 408.8 so the front below Z = 408.8 stays completely open
    cut_front_low = Pos(-6.0, -11.5, 398.0) * Box(24.0, 14.0, 21.6)  # Y in [-18.5, -4.5], Z in [387.2, 408.8]
    # Also taper rear plate (Y > 14.0, X < 2.5) above Z = 412.5 so it cleanly frames the 4 rear M2 holes (Z=391.5, 401.0)
    cut_rear_high = Pos(-6.5, 16.0, 417.5) * Box(18.0, 5.0, 9.0)     # X in [-15.5, 2.5], Z in [413.0, 422.0]

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
        w_oval = Pos(1.5, 405.5) * SlotOverall(15.5, 9.2, rotation=28)
        c_oval = extrude_xy_to_x(w_oval, 2.5, 9.5)
        pts_taper = [
            (-16.5, 388.0),
            (0.5, 388.0),
            (-16.5, 408.8),
        ]
        f_tap = Polygon(*pts_taper, align=None)
        c_tap = extrude_xy_to_x(f_tap, 2.5, 9.5)
        s_body = Solid((s_body - c_oval - c_tap).wrapped)

    cut_sol = _cut_stage_bores(s_body, "stage3_neck_pitch_to_head_cradle")
    return bevel_3d_edges(cut_sol, r_outer=0.38, r_inner=0.72, skip_fn=_skip_near_stage_bores("stage3_neck_pitch_to_head_cradle"))


# ==============================================================================
# STAGE 4: `stage4_head_sensor_compute_carrier` (on `head`)
# ==============================================================================
def build_stage4(family: str) -> Solid:
    """Build Stage 4 (`head_sensor_compute_carrier`): consolidates `single_shared_carrier_v23`
    and `microphone_compact_tab_v23` into a single one-piece FDM carrier with 14 pressed-in 304 SS
    compression sleeves (`2x M3` neck root + `4x M2.5` ROCK4D + `8x M2` cameras), integrated
    right microphone side wing, radial head-skin lugs, and dorsal stiffening ribs in the
    collision-free `Y in [-11.40, -10.15]` corridor.
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

    s_plate = extrude_xy_to_y(f_main, -12.90, -11.40)

    # 2. Dorsal stiffening rib network in the collision-free rear zone Y in [-11.40, -10.15]
    # staying strictly outside the 4 compute_FF4_spacer seats and clear of lightening windows:
    rib1 = extrude_xy_to_y(fillet_2d_face(Pos(-0.7, 442.0) * Rectangle(5.6, 52.0), r_default=1.5), -11.40, -10.15)
    rib2 = extrude_xy_to_y(fillet_2d_face(Pos(-1.5, 425.0) * Rectangle(21.0, 3.6), r_default=1.2), -11.40, -10.15)
    rib3 = extrude_xy_to_y(fillet_2d_face(Pos(-1.0, 442.5) * Rectangle(60.0, 4.4), r_default=1.2), -11.40, -10.15)
    rib4 = extrude_xy_to_y(fillet_2d_face(Pos(0.0, 471.5) * Rectangle(62.0, 4.2), r_default=1.2), -11.40, -10.15)

    # 3. Radial head-skin mounting lugs at X in [-45.5, -43.2] and [43.2, 45.5], Y in [-17.2, -11.4], Z in [447.2, 454.8]
    f_lug = fillet_2d_face(
        Pos(-14.3, 451.0) * Rectangle(5.8, 7.6),
        r_default=1.2,
        r_inner=1.6,
        skip_xy=[(-11.4, 447.2), (-11.4, 454.8)],
    )
    s_lug_l = extrude_xy_to_x(f_lug, -45.5, -43.2)
    s_lug_r = extrude_xy_to_x(f_lug, 43.2, 45.5)

    # 4. Integrated one-piece right microphone carrier wing (replaces microphone_compact_tab_v23 + mic_foot_M2x3_0/1)
    # Front bridge (X in [43.0, 48.97], Y in [-12.90, -11.40], Z in [455.2, 462.4] to clear head-skin upper curve)
    f_mic_br = fillet_2d_face(Pos(45.985, 458.8) * Rectangle(5.97, 7.2), r_default=1.2, r_inner=1.6)
    s_mic_br = extrude_xy_to_y(f_mic_br, -12.90, -11.40)

    # Side wing (X in [46.97, 48.97], stepped back to Y >= -7.8 above Z = 462.4 to clear original_profile_head_skin_v23)
    pts_mic_side = [
        (-12.90, 455.2),
        (4.50, 455.2),
        (4.50, 471.2),
        (-1.50, 471.2),
        (-7.80, 462.4),
        (-12.90, 462.4),
    ]
    f_mic_side = fillet_2d_face(
        Polygon(*pts_mic_side, align=None),
        r_default=1.2,
        r_inner=1.8,
        skip_xy=[(-12.90, 455.2), (-12.90, 462.4)],
    )
    s_mic_side = extrude_xy_to_x(f_mic_side, 46.97, 48.97)
    # Taper top outer corner (Z >= 467.2, X >= 47.82) to match original_profile_head_skin_v23 inner contour
    cut_skin_taper = Pos(48.55, 0.0, 469.5) * Box(1.5, 16.0, 4.6)
    s_mic_side = Solid((s_mic_side - cut_skin_taper).wrapped)

    raw = Solid((s_plate + rib1 + rib2 + rib3 + rib4 + s_lug_l + s_lug_r + s_mic_br + s_mic_side).wrapped)
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

    # First export all 34 pressed-in 304 SS sleeves
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
