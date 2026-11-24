"""Assemble, verify, render, and package the Complete Cove-50 FDM Ape Body (`cove50-complete-fdm-body-r1`).

Combines all four completed 3D-printable FDM structural subsystems (28 printed FDM parts + 316 pressed-in 304 SS sleeves):
  1. Lower Body (`cove50-complete-leg-hip-fdm-combined-r10`, T-f39h6b, M-y8acff):
     - 12 printed parts (6 stages x 2 legs: Shin, Ankle Coupler, Foot Monocoque, Thigh, Hip Pitch-to-Roll, Hip Yaw-to-Pitch)
     - 128 pressed-in 304 SS compression sleeves (64/leg)
     - 187.64 g bilateral (down from 332.78 g legacy V16 sheet-metal + carrier baseline, -43.61%)
  2. Upper Body / Arms (`cove50-arm-fdm-combined-r12`, T-zw7f7t, M-d34p89):
     - 8 printed parts (4 stages x 2 arms: Shoulder Pitch-to-Roll, Shoulder Roll-to-Yaw, Upper-Arm Yaw-to-Elbow, Forearm Elbow-to-Wrist)
     - 68 pressed-in 304 SS compression sleeves (34/arm)
     - 74.74 g bilateral (down from 136.05 g legacy sheet-metal + carrier baseline, -45.06%)
  3. Neck & Head (`cove50-neck-head-fdm-combined-r1`, T-zesmm6, M-8jkx67):
     - 4 printed parts (Neck Root Cradle, Yaw-to-Pitch Link, Pitch-to-Head U-Cradle, Head Shared Sensor/Compute/Mic Carrier)
     - 34 pressed-in 304 SS compression sleeves (6 + 8 + 6 + 14)
     - 28.56 g total (down from 52.67 g legacy sheet-metal + PA12 carrier baseline, -45.78%)
  4. Center-Body Torso, Pelvis, 9-Cell Honeycomb Battery & Electronics Carrier (`cove50-torso-fdm-combined-r1`, T-zdevbv, M-h3aaq6):
     - 4 printed parts (Stage 1 7-Cell Honeycomb Torso Spine, Stage 2 2-Cell Pelvis & 7-Cell Lid U-Clevis, Stage 3 Dual-Deck Front Power Frame, Stage 4 Dorsal IMU/ADC Carrier)
     - 86 pressed-in 304 SS compression sleeves (39 + 11 + 22 + 14)
     - 9 vertical honeycomb 21700 P50B battery cells + 9 top insulators + 9 direct-on-battery 6063-T5 aluminum 4-blade heat-sink fin strips
     - Smooth 2-piece continuous 12-station NURBS Center Shell (`center_front_shell` + `center_back_shell` + `chest_indicator`)
     - 178.21 g total (down from 251.82 g legacy AlSi10Mg LPBF + PA12 baseline, -29.23%)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_k] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from build123d import Box, Compound, Cylinder, Plane, Pos, Solid, export_brep, export_step, export_stl, import_brep, import_step
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt, gp_Trsf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.storage.blob_store import ensure

STUDY_DIR = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts/robot/body/assembly"
PARTS_DIR = ART_DIR / "parts"
RENDERS_DIR = ART_DIR / "renders"

CPP_RENDERER = ROOT / "robot/electronics/power/cad/zbuffer_renderer.cpp"
EXE_RENDERER = ART_DIR / "zbuffer_renderer_bin"

# NOTE: Do NOT import `solid_to_watertight_arrays` or `manifold3d` at module top-level before running
# OpenCASCADE (`OCP`) BRepFilletAPI_MakeFillet (`bevel_3d_edges` / `fillet_2d_face`) on macOS ARM64.
sys.path.insert(0, str(ROOT / "robot/body/leg"))
sys.path.insert(0, str(ROOT / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402  (joint ranges: robot/body/bom/bom_config.py, from the body definitions)
from build_upper_hip_connectors import (  # noqa: E402
    bevel_3d_edges,
    cut_all,
    extrude_xy_to_x,
    extrude_xy_to_y,
    extrude_xy_to_z,
    fillet_2d_face,
    fuse_all,
    largest_solid,
    make_captive_nut_pocket,
    mirror_x,
    occ_box,
    occ_cyl,
)
from build123d import Circle, Face, Polygon, RectangleRounded, Rot, Vector, Wire  # noqa: E402


def _get_solid_to_watertight_arrays():
    if str(ROOT / "robot/body/arm") not in sys.path:
        sys.path.insert(0, str(ROOT / "robot/body/arm"))
    sys.modules.pop("common", None)
    from common import solid_to_watertight_arrays
    return solid_to_watertight_arrays


def _intersect_solid(a: Solid | Compound, b: Solid | Compound) -> Solid:
    q = BRepAlgoAPI_Common(a.wrapped if hasattr(a, "wrapped") else a, b.wrapped if hasattr(b, "wrapped") else b)
    q.Build()
    return largest_solid(Compound(q.Shape()))


def _make_hex_nut(center_xyz, d_out, af: float = 5.5, h: float = 2.4, bore_r: float = 1.5) -> Solid:
    d = np.asarray(d_out, dtype=float)
    d /= np.linalg.norm(d)
    u = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.5 else np.array([0.0, 1.0, 0.0])
    v = np.cross(d, u)
    r = af / math.sqrt(3.0)
    p0 = np.asarray(center_xyz, dtype=float)
    pts = [Vector(*(p0 + r * (math.cos(a) * u + math.sin(a) * v))) for a in np.arange(6) * math.pi / 3.0]
    hex_sol = Solid.extrude(Face(Wire.make_polygon(pts, close=True)), Vector(*(d * h)))
    bore = occ_cyl(tuple(p0 - d * 0.2), tuple(p0 + d * (h + 0.4)), bore_r)
    return largest_solid(cut_all(hex_sol, [bore]))


def _make_m3x8_screw(seat_xyz, d_out) -> Solid:
    d = np.asarray(d_out, dtype=float)
    d /= np.linalg.norm(d)
    p = np.asarray(seat_xyz, dtype=float)
    shank = occ_cyl(tuple(p - d * 8.0), tuple(p), 1.50)
    head = occ_cyl(tuple(p), tuple(p + d * 1.65), 2.85)
    return largest_solid(fuse_all([shank, head]))


def build_contoured_thigh_and_covers() -> dict[str, Solid]:
    """Build inward-tapering anatomical thigh hard covers (`R/L_thigh_front_hard_cover`, `R/L_thigh_back_hard_cover`),
    matching Stage 4 Thigh (`R/L_thigh_combined`) with shortened symmetric Z=178.0 mm shell mounting posts,
    and matching M3x8 screws + captive M3 nuts."""
    y_f_in, y_f_clamp, y_f_out = -12.10, -14.10, -15.30
    y_r_in, y_r_clamp, y_r_out = 25.35, 27.35, 28.55

    # 1. Stage 4 Thigh (`R_thigh_combined`) with shortened symmetric Z=178.0 mm shell mounting posts
    ch_low_poly = Polygon(
        (-9.8, 128.4), (17.8, 128.4), (17.8, 153.0), (28.55, 158.50),
        (28.55, 162.80), (-15.30, 162.80), (-15.30, 149.5), (-9.8, 144.0),
        align=None,
    )
    for by_c, bz_c in [(-6.25, 131.3), (-6.25, 152.0), (14.25, 131.3), (14.25, 152.0)]:
        ch_low_poly = ch_low_poly + Pos(by_c, bz_c) * Circle(3.15)
    ch_low_face = fillet_2d_face(
        ch_low_poly.clean().faces()[0], 2.2, skip_xy=[(28.55, 158.50), (28.55, 162.80), (-15.30, 162.80)]
    )

    ch_hi_poly = Polygon(
        (-9.8, 135.5), (7.5, 135.5), (17.8, 144.5), (17.8, 153.0), (28.55, 158.50),
        (28.55, 162.80), (-15.30, 162.80), (-15.30, 149.5), (-9.8, 144.0),
        align=None,
    )
    for by_c, bz_c in [(-6.25, 152.0), (14.25, 152.0)]:
        ch_hi_poly = ch_hi_poly + Pos(by_c, bz_c) * Circle(3.15)
    ch_hi_face = fillet_2d_face(
        ch_hi_poly.clean().faces()[0], 2.2, skip_xy=[(28.55, 158.50), (28.55, 162.80), (-15.30, 162.80), (-9.8, 135.5)]
    )

    win_ch = Pos(4.0, 144.0) * Circle(4.8).faces()[0]
    ch_low_face = (ch_low_face - win_ch).clean().faces()[0]
    ch_hi_face = (ch_hi_face - win_ch).clean().faces()[0]

    cheek_inner = extrude_xy_to_x(ch_low_face, 42.575, 44.575, sx=1.0)
    cheek_outer = extrude_xy_to_x(ch_hi_face, 41.40, 44.575, sx=1.0)
    knee_low_bosses = [occ_cyl((41.40, by_c, bz_c), (44.575, by_c, bz_c), 3.15) for by_c, bz_c in [(-6.25, 131.3), (14.25, 131.3)]]

    arm_pts = [(41.40, 155.8), (56.5, 155.8), (74.8, 158.50), (74.8, 190.0), (53.2, 190.0), (41.40, 162.80)]
    arm_union = (Polygon(*arm_pts, align=None) + Pos(64.0, 190.0) * Circle(10.8)).clean().faces()[0]
    arm_face = fillet_2d_face(arm_union, 2.2, skip_xy=[(41.40, 155.8), (41.40, 162.80), (74.8, 158.50)], r_inner=3.2)
    arm_face = (arm_face - Pos(61.5, 171.8) * Circle(4.0).faces()[0]).clean().faces()[0]
    arm_f = extrude_xy_to_y(arm_face, y_f_out, y_f_in, sx=1.0)
    arm_r = extrude_xy_to_y(arm_face, y_r_in, y_r_out, sx=1.0)

    sad_outer = Polygon((41.40, y_f_out), (74.8, y_f_out), (74.8, y_r_out), (41.40, y_r_out), align=None).faces()[0]
    sad_face = (sad_outer - Pos(58.0, 0.5) * Circle(5.2).faces()[0] - Pos(58.0, 13.5) * Circle(5.2).faces()[0]).clean().faces()[0]
    saddle_deck = extrude_xy_to_z(sad_face, 158.50, 162.80)

    body = fuse_all([cheek_inner, cheek_outer, *knee_low_bosses, arm_f, arm_r, saddle_deck])
    cutters = [
        occ_box(44.575, 80.0, -8.50, 16.50, 112.0, 158.30),
        occ_box(43.40, 80.0, -3.05, 11.05, 131.8, 158.30),
        occ_cyl((39.0, 4.0, 123.0), (44.10, 4.0, 123.0), 10.0),
        occ_box(28.0, 76.0, y_f_in, y_r_in, 162.85, 205.00),
        occ_cyl((64.0, y_f_out - 2.0, 190.0), (64.0, y_f_in + 2.0, 190.0), 4.2),
        occ_cyl((64.0, y_r_in - 2.0, 190.0), (64.0, y_r_out + 2.0, 190.0), 4.2),
    ]
    for bx_c, bz_c in [(57.0, 190.0), (71.0, 190.0), (64.0, 183.0), (64.0, 197.0)]:
        cutters.append(occ_cyl((bx_c, y_f_out - 2.0, bz_c), (bx_c, y_f_in + 1.0, bz_c), 2.25))
        cutters.append(occ_cyl((bx_c, y_f_out - 25.0, bz_c), (bx_c, y_f_clamp, bz_c), 2.90))
        cutters.append(occ_cyl((bx_c, y_r_in - 1.0, bz_c), (bx_c, y_r_out + 2.0, bz_c), 2.25))
        cutters.append(occ_cyl((bx_c, y_r_clamp, bz_c), (bx_c, y_r_out + 25.0, bz_c), 2.90))
    for by_c, bz_c in [(-6.25, 131.3), (-6.25, 152.0), (14.25, 131.3), (14.25, 152.0)]:
        cutters.append(occ_cyl((40.0, by_c, bz_c), (46.0, by_c, bz_c), 1.80))
        cutters.append(occ_cyl((15.0, by_c, bz_c), (42.575, by_c, bz_c), 2.35))

    bev_sol = bevel_3d_edges(largest_solid(cut_all(body, cutters)), r_outer=0.40, r_inner=0.75)

    shell_posts = [
        bevel_3d_edges(occ_cyl((57.0, -20.50, 178.0), (57.0, y_f_in, 178.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
        bevel_3d_edges(occ_cyl((71.0, -17.00, 178.0), (71.0, y_f_in, 178.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
        bevel_3d_edges(occ_cyl((57.0, y_r_in, 178.0), (57.0, 32.50, 178.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
        bevel_3d_edges(occ_cyl((71.0, y_r_in, 178.0), (71.0, 29.00, 178.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
    ]
    shell_cutters = [
        occ_cyl((57.0, -22.5, 178.0), (57.0, y_f_in + 0.5, 178.0), 1.70),
        make_captive_nut_pocket((57.0, -17.82, 178.0), (0.0, 1.0, 0.0)),
        occ_cyl((71.0, -19.0, 178.0), (71.0, y_f_in + 0.5, 178.0), 1.70),
        make_captive_nut_pocket((71.0, -14.32, 178.0), (0.0, 1.0, 0.0)),
        occ_cyl((57.0, y_r_in - 0.5, 178.0), (57.0, 34.5, 178.0), 1.70),
        make_captive_nut_pocket((57.0, 29.82, 178.0), (0.0, -1.0, 0.0)),
        occ_cyl((71.0, y_r_in - 0.5, 178.0), (71.0, 31.0, 178.0), 1.70),
        make_captive_nut_pocket((71.0, 26.32, 178.0), (0.0, -1.0, 0.0)),
    ]
    r_thigh = largest_solid(cut_all(fuse_all([bev_sol, *shell_posts]), shell_cutters))

    def _make_ell_wire(z: float, x0: float, x1: float, y0: float, y1: float, inset: float = 0.0) -> Wire:
        xc = 0.5 * (x0 + x1)
        rx = max(2.0, 0.5 * (x1 - x0) - inset)
        yc = 0.5 * (y0 + y1)
        ry = max(2.0, 0.5 * (y1 - y0) - inset)
        return Wire.make_ellipse(rx, ry, Plane(origin=(xc, yc, z)))

    # 2. Pure-Ellipse Parabolic Barrel Thigh Hard Covers (`R_thigh_front_hard_cover`, `R_thigh_back_hard_cover`)
    #    4-station single-arc elliptical barrel (`53.0 mm` at the knee, `60.2-61.2 mm` round mid-thigh belly,
    #    and `56.0 mm` at the upper hip crease, down from 72.7-86.5 mm while keeping the round organic drawing shape).
    stations = [
        (123.5, 33.0, 86.0, -19.8, 26.5),
        (152.0, 28.0, 88.5, -24.2, 35.5),
        (180.0, 24.5, 88.8, -25.2, 38.4),
        (210.0, 23.0, 85.0, -23.5, 36.8),
    ]

    outer_loft = Solid.make_loft([_make_ell_wire(*s, inset=0.0) for s in stations])
    inner_loft = Solid.make_loft([_make_ell_wire(*s, inset=2.1) for s in stations])
    inner_thin = Solid.make_loft([_make_ell_wire(*s, inset=1.15) for s in stations])
    inner_corner_relief = _intersect_solid(occ_box(25.0, 86.5, -22.5, 36.2, 123.0, 210.0), inner_thin)
    shell_wall = largest_solid(cut_all(outer_loft, [inner_loft, inner_corner_relief]))

    xs_top = np.linspace(18.0, 98.0, 28)
    pts_top = []
    for x in xs_top:
        t = np.clip((x - 35.0) / (68.0 - 35.0), 0.0, 1.0)
        z = 168.5 + 38.8 * (3.0 * t**2 - 2.0 * t**3)
        pts_top.append((float(x), float(z)))
    poly_top_pts = pts_top + [(98.0, 225.0), (18.0, 225.0)]
    top_cutter = extrude_xy_to_y(Polygon(*poly_top_pts, align=None).faces()[0], -60.0, 60.0)
    bot_cutter = occ_box(15.0, 105.0, -60.0, 60.0, 110.0, 124.0)

    shell_trimmed = largest_solid(cut_all(shell_wall, [top_cutter, bot_cutter]))
    front_raw = largest_solid(cut_all(shell_trimmed, [occ_box(15.0, 105.0, 7.7, 60.0, 110.0, 225.0)]))

    ys_rear = np.linspace(8.0, 45.0, 20)
    pts_rear = []
    for y in ys_rear:
        t = np.clip((y - 8.3) / (26.0 - 8.3), 0.0, 1.0)
        z = 140.0 + 11.5 * (3.0 * t**2 - 2.0 * t**3)
        pts_rear.append((float(y), float(z)))
    poly_rear_pts = pts_rear + [(45.0, 115.0), (8.0, 115.0)]
    rear_knee_cutter = extrude_xy_to_x(Polygon(*poly_rear_pts, align=None).faces()[0], 15.0, 105.0)
    back_raw = largest_solid(cut_all(shell_trimmed, [occ_box(15.0, 105.0, -60.0, 8.3, 110.0, 225.0), rear_knee_cutter]))

    f_bosses = [
        _intersect_solid(occ_cyl((57.0, -20.50, 178.0), (57.0, -28.00, 178.0), 4.8), outer_loft),
        _intersect_solid(occ_cyl((71.0, -17.00, 178.0), (71.0, -26.00, 178.0), 4.8), outer_loft),
    ]
    f_cuts = [
        occ_cyl((57.0, -20.00, 178.0), (57.0, -28.00, 178.0), 1.70),
        occ_cyl((57.0, -22.25, 178.0), (57.0, -28.00, 178.0), 3.20),
        occ_cyl((71.0, -16.50, 178.0), (71.0, -26.00, 178.0), 1.70),
        occ_cyl((71.0, -18.55, 178.0), (71.0, -26.00, 178.0), 3.20),
    ]
    r_front_cover = largest_solid(cut_all(fuse_all([front_raw, *f_bosses]), f_cuts))

    r_bosses = [
        _intersect_solid(occ_cyl((57.0, 32.50, 178.0), (57.0, 40.00, 178.0), 4.8), outer_loft),
        _intersect_solid(occ_cyl((71.0, 29.00, 178.0), (71.0, 38.00, 178.0), 4.8), outer_loft),
    ]
    r_cuts = [
        occ_cyl((57.0, 32.00, 178.0), (57.0, 40.00, 178.0), 1.70),
        occ_cyl((57.0, 34.25, 178.0), (57.0, 40.00, 178.0), 3.20),
        occ_cyl((71.0, 28.50, 178.0), (71.0, 38.00, 178.0), 1.70),
        occ_cyl((71.0, 30.55, 178.0), (71.0, 38.00, 178.0), 3.20),
    ]
    r_back_cover = largest_solid(cut_all(fuse_all([back_raw, *r_bosses]), r_cuts))

    # 3. Fasteners & captive M3 nuts seated at the Z=178.0 mm thigh bosses
    r_f1_nut = _make_hex_nut((57.0, -17.95, 178.0), (0.0, -1.0, 0.0))
    r_f1_scr = _make_m3x8_screw((57.0, -22.50, 178.0), (0.0, -1.0, 0.0))
    r_f2_nut = _make_hex_nut((71.0, -14.45, 178.0), (0.0, -1.0, 0.0))
    r_f2_scr = _make_m3x8_screw((71.0, -18.80, 178.0), (0.0, -1.0, 0.0))

    r_b1_nut = _make_hex_nut((57.0, 29.95, 178.0), (0.0, 1.0, 0.0))
    r_b1_scr = _make_m3x8_screw((57.0, 34.50, 178.0), (0.0, 1.0, 0.0))
    r_b2_nut = _make_hex_nut((71.0, 26.45, 178.0), (0.0, 1.0, 0.0))
    r_b2_scr = _make_m3x8_screw((71.0, 30.80, 178.0), (0.0, 1.0, 0.0))

    # 4. Slimmed Stage 1 Shin (`R_shin_combined`) without horizontal side ears (`X in [39.25, 81.25]`, 42.0 mm wide)
    #    and with symmetric rear mounting bridge & side-by-side rear posts at X=53.0, 67.0 mm, Z=102.5 mm.
    shin_orig_step = ensure(ROOT / "artifacts/robot/body/leg/combined/parts/R_shin_combined.step")
    s_shin_orig = largest_solid(import_step(shin_orig_step))
    s_shin_trim = largest_solid(cut_all(s_shin_orig, [
        occ_box(20.0, 39.25, 0.0, 25.0, 95.0, 112.0),
        occ_box(81.25, 110.0, 0.0, 25.0, 95.0, 112.0),
    ]))
    shin_rear_bar = bevel_3d_edges(occ_box(39.25, 81.25, 12.80, 16.50, 97.70, 107.30), r_outer=0.35)
    shin_rear_posts = [
        bevel_3d_edges(occ_cyl((53.0, 12.80, 102.5), (53.0, 21.00, 102.5), 4.6), r_outer=0.35, min_cyl_r=4.3),
        bevel_3d_edges(occ_cyl((67.0, 12.80, 102.5), (67.0, 21.00, 102.5), 4.6), r_outer=0.35, min_cyl_r=4.3),
    ]
    shin_rear_cuts = [
        occ_cyl((53.0, 11.5, 102.5), (53.0, 22.5, 102.5), 1.70),
        make_captive_nut_pocket((53.0, 18.32, 102.5), (0.0, -1.0, 0.0)),
        occ_cyl((67.0, 11.5, 102.5), (67.0, 22.5, 102.5), 1.70),
        make_captive_nut_pocket((67.0, 18.32, 102.5), (0.0, -1.0, 0.0)),
    ]
    r_shin = largest_solid(cut_all(fuse_all([s_shin_trim, shin_rear_bar, *shin_rear_posts]), shin_rear_cuts))

    # 5. Pure-Ellipse Parabolic Barrel Shin Hard Covers (`R_shin_front_hard_cover`, `R_shin_back_hard_cover`)
    #    4-station single-arc elliptical barrel (`55.5 mm` at ankle, `58.8 mm` round mid-calf belly,
    #    and `52.6 mm` at knee, down from 76.8 mm while keeping the round calf curve of the drawing).
    shin_stations = [
        (43.5,  30.0, 90.5, -18.8, 29.8),
        (68.0,  30.6, 89.4, -20.5, 28.5),
        (95.0,  30.6, 89.4, -20.5, 27.5),
        (118.5, 33.2, 85.8, -18.0, 23.8),
    ]
    s_out = Solid.make_loft([_make_ell_wire(*s, inset=0.0) for s in shin_stations])
    s_in = Solid.make_loft([_make_ell_wire(*s, inset=2.1) for s in shin_stations])
    s_thin = Solid.make_loft([_make_ell_wire(*s, inset=1.15) for s in shin_stations])
    s_corner_relief = _intersect_solid(occ_box(31.5, 89.0, -17.5, 28.0, 43.0, 119.0), s_thin)
    s_wall = largest_solid(cut_all(s_out, [s_in, s_corner_relief]))

    s_trim = largest_solid(cut_all(s_wall, [
        occ_box(15.0, 105.0, -60.0, 60.0, 30.0, 46.5),
        occ_box(15.0, 105.0, -60.0, 60.0, 118.0, 130.0),
    ]))
    s_front_raw = largest_solid(cut_all(s_trim, [occ_box(15.0, 105.0, 7.7, 50.0, 30.0, 130.0)]))

    ys_rk = np.linspace(8.0, 35.0, 20)
    pts_rk = []
    for y in ys_rk:
        t = np.clip((y - 8.3) / (19.0 - 8.3), 0.0, 1.0)
        z = 118.0 - 10.8 * (3.0 * t**2 - 2.0 * t**3)
        pts_rk.append((float(y), float(z)))
    rear_shin_cutter = extrude_xy_to_x(Polygon(*(pts_rk + [(35.0, 130.0), (8.0, 130.0)]), align=None).faces()[0], 15.0, 105.0)
    s_back_raw = largest_solid(cut_all(s_trim, [occ_box(15.0, 105.0, -50.0, 8.3, 30.0, 130.0), rear_shin_cutter]))

    sf_bosses = [
        _intersect_solid(occ_cyl((53.0, -14.00, 102.5), (53.0, -22.00, 102.5), 4.6), s_out),
        _intersect_solid(occ_cyl((67.0, -14.00, 102.5), (67.0, -22.00, 102.5), 4.6), s_out),
    ]
    sf_cuts = [
        occ_cyl((53.0, -13.50, 102.5), (53.0, -22.00, 102.5), 1.70),
        occ_cyl((53.0, -17.20, 102.5), (53.0, -22.00, 102.5), 3.10),
        occ_cyl((67.0, -13.50, 102.5), (67.0, -22.00, 102.5), 1.70),
        occ_cyl((67.0, -17.20, 102.5), (67.0, -22.00, 102.5), 3.10),
    ]
    r_shin_front_cover = largest_solid(cut_all(fuse_all([s_front_raw, *sf_bosses]), sf_cuts))

    sb_bosses = [
        _intersect_solid(occ_cyl((53.0, 21.00, 102.5), (53.0, 28.00, 102.5), 4.6), s_out),
        _intersect_solid(occ_cyl((67.0, 21.00, 102.5), (67.0, 28.00, 102.5), 4.6), s_out),
    ]
    sb_cuts = [
        occ_cyl((53.0, 20.50, 102.5), (53.0, 28.00, 102.5), 1.70),
        occ_cyl((53.0, 23.80, 102.5), (53.0, 28.00, 102.5), 3.10),
        occ_cyl((67.0, 20.50, 102.5), (67.0, 28.00, 102.5), 1.70),
        occ_cyl((67.0, 23.80, 102.5), (67.0, 28.00, 102.5), 3.10),
    ]
    r_shin_back_cover = largest_solid(cut_all(fuse_all([s_back_raw, *sb_bosses]), sb_cuts))

    r_sf1_nut = _make_hex_nut((53.0, -11.47, 102.5), (0.0, -1.0, 0.0))
    r_sf1_scr = _make_m3x8_screw((53.0, -17.20, 102.5), (0.0, -1.0, 0.0))
    r_sf2_nut = _make_hex_nut((67.0, -11.47, 102.5), (0.0, -1.0, 0.0))
    r_sf2_scr = _make_m3x8_screw((67.0, -17.20, 102.5), (0.0, -1.0, 0.0))

    r_sb1_nut = _make_hex_nut((53.0, 18.45, 102.5), (0.0, 1.0, 0.0))
    r_sb1_scr = _make_m3x8_screw((53.0, 23.80, 102.5), (0.0, 1.0, 0.0))
    r_sb2_nut = _make_hex_nut((67.0, 18.45, 102.5), (0.0, 1.0, 0.0))
    r_sb2_scr = _make_m3x8_screw((67.0, 23.80, 102.5), (0.0, 1.0, 0.0))

    solids_r = {
        "R_thigh_combined": r_thigh,
        "R_thigh_front_hard_cover": r_front_cover,
        "R_thigh_back_hard_cover": r_back_cover,
        "R_Thigh_carrier_shell_thigh_front_1_nut": r_f1_nut,
        "R_Thigh_carrier_shell_thigh_front_1_M3x8": r_f1_scr,
        "R_Thigh_carrier_shell_thigh_front_2_nut": r_f2_nut,
        "R_Thigh_carrier_shell_thigh_front_2_M3x8": r_f2_scr,
        "R_Thigh_rear_carrier_shell_thigh_back_1_nut": r_b1_nut,
        "R_Thigh_rear_carrier_shell_thigh_back_1_M3x8": r_b1_scr,
        "R_Thigh_rear_carrier_shell_thigh_back_2_nut": r_b2_nut,
        "R_Thigh_rear_carrier_shell_thigh_back_2_M3x8": r_b2_scr,
        "R_shin_combined": r_shin,
        "R_shin_front_hard_cover": r_shin_front_cover,
        "R_shin_back_hard_cover": r_shin_back_cover,
        "R_Shin_carrier_shell_shin_front_1_nut": r_sf1_nut,
        "R_Shin_carrier_shell_shin_front_1_M3x8": r_sf1_scr,
        "R_Shin_carrier_shell_shin_front_2_nut": r_sf2_nut,
        "R_Shin_carrier_shell_shin_front_2_M3x8": r_sf2_scr,
        "R_Shin_carrier_shell_shin_back_1_nut": r_sb1_nut,
        "R_Shin_carrier_shell_shin_back_1_M3x8": r_sb1_scr,
        "R_Shin_carrier_shell_shin_back_2_nut": r_sb2_nut,
        "R_Shin_carrier_shell_shin_back_2_M3x8": r_sb2_scr,
    }
    out = dict(solids_r)
    for k, v in solids_r.items():
        out["L_" + k[2:]] = mirror_x(v)
    return out


def slim_foot_shell_mesh(m_mm: trimesh.Trimesh, side: str) -> trimesh.Trimesh:
    """Subdivide, Taubin-smooth, and slim `R/L_foot_boot_shell` and `R/L_foot_sole_shell` in X so the
    boot keeps its round dome shape at `72.0 mm` wide (`0.82x` of the original 88.0 mm) and tapers smoothly
    to `51.8 mm` wide at the `Z = 37.5 mm` ankle collar right under the `52.0 mm` lower calf shell."""
    m = m_mm.copy()
    m.merge_vertices()
    v_sub, f_sub = trimesh.remesh.subdivide(m.vertices, m.faces)
    v_sub, f_sub = trimesh.remesh.subdivide(v_sub, f_sub)
    m = trimesh.Trimesh(vertices=v_sub, faces=f_sub, process=True)
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=10)
    v = np.asarray(m.vertices, dtype=np.float64).copy()
    sgn = 1.0 if side == "R" else -1.0
    x_abs = v[:, 0] * sgn
    t = np.clip((v[:, 2] - 14.0) / (37.5 - 14.0), 0.0, 1.0)
    s = 3.0 * t**2 - 2.0 * t**3
    xc_old = (1.0 - s) * 66.08 + s * 65.03
    xc_new = (1.0 - s) * 65.00 + s * 60.20
    scale_x = (1.0 - s) * 0.82 + s * 0.96
    v[:, 0] = sgn * (xc_new + (x_abs - xc_old) * scale_x)
    m.vertices = v
    return m


def build_redesigned_shoulder_and_upper_arm() -> dict[str, Solid | Compound]:
    """Build the redesigned Shoulder-to-Elbow architecture requested by the user:
      1. Outward-shifted Torso Shoulder Pitch Mounting Cheeks (`combined__stage1_center_torso_spine_chassis` + 8 shifted PA2 sleeves at `|X| = 42.4 mm`, shifting `P_PITCH: (±52.0, 9.0, 352.0) -> (±64.0, 9.0, 352.0)` by `±12.0 mm`).
      2. Flipped Shoulder Roll Actuator (`P_ROLL: (±112.0, 3.0, 360.0) -> (±106.0, 5.05, 360.0)` with `ROT_FLIP = ROT_ROLL @ diag(-1, -1, 1)`):
         - The rotating horn (`upper__R/L_roll_C047_main_horn`) is now on the `+Y` (rear) side (`Y = +23.675 mm`) and bolts directly to the compact Blue Piece (`R/L_shoulder_pitch_to_roll_combined`) on `R/L_shoulder_pitch_link`.
         - The actuator housing (`upper__R/L_roll_C047_case`) now has its 4-boss mounting face on the `-Y` (front) side (`Y = -12.525 mm`) and bolts directly to the compact Green Piece (`R/L_shoulder_roll_to_yaw_combined`) on `R/L_shoulder_roll_link`, so the Roll actuator housing rotates with shoulder roll.
      3. Inward-shifted Upper-Arm Yaw Actuator (`P_YAW: (±134.0, -10.0, 306.0) -> (±127.0, -10.0, 306.0)`, `ΔX = ∓7.0 mm`) and Elbow/Forearm (`P_ELBOW: (±145.0, -3.0, 252.0) -> (±138.0, -3.0, 252.0)`).
      4. Single One-Piece Smooth B-Rep Shoulder-to-Elbow Shell (`R/L_upper_arm_hard_cover` on `R/L_shoulder_roll_link`, `Z = 270.0 -> 383.5 mm`) encapsulating both the flipped Roll actuator and the shifted Yaw actuator + upper-arm neck in one continuous shell."""
    P_ROLL_NEW = np.array([106.0, 5.05, 360.0], dtype=float)
    P_YAW_NEW = np.array([127.0, -10.0, 306.0], dtype=float)

    ROT_ROLL = np.array([
        [0.0, math.cos(math.radians(20.0)), math.sin(math.radians(20.0))],
        [1.0, 0.0, 0.0],
        [0.0, math.sin(math.radians(20.0)), -math.cos(math.radians(20.0))],
    ], dtype=float)
    ROT_FLIP = ROT_ROLL @ np.diag([-1.0, -1.0, 1.0])

    N_YAW = np.array([-0.1643989873053573, 0.0, 0.9863939238321437], dtype=float)
    V_YAW = np.array([0.9863939238321437, 0.0, 0.1643989873053573], dtype=float)
    W_YAW = np.array([0.0, 1.0, 0.0], dtype=float)

    def _extrude_yaw_local_x(face_vw, x0: float, x1: float) -> Solid:
        xa, xb = min(float(x0), float(x1)), max(float(x0), float(x1))
        p_orig = P_YAW_NEW + xa * N_YAW
        pl = Plane(origin=tuple(p_orig), x_dir=tuple(V_YAW), y_dir=tuple(W_YAW))
        f3d = (pl * face_vw).faces()[0]
        return Solid(Solid.extrude(f3d, Vector(*tuple((xb - xa) * N_YAW))).wrapped)

    def _extrude_yaw_local_y(face_wn, v0: float, v1: float) -> Solid:
        va, vb = min(float(v0), float(v1)), max(float(v0), float(v1))
        p_orig = P_YAW_NEW + va * V_YAW
        pl = Plane(origin=tuple(p_orig), x_dir=tuple(W_YAW), y_dir=tuple(N_YAW))
        f3d = (pl * face_wn).faces()[0]
        return Solid(Solid.extrude(f3d, Vector(*tuple((vb - va) * V_YAW))).wrapped)

    def _make_sleeve(pt, d, ro: float, ri: float, h: float) -> Solid:
        d_u = np.asarray(d, dtype=float)
        d_u = d_u / np.linalg.norm(d_u)
        p0 = np.asarray(pt, dtype=float)
        outer = occ_cyl(tuple(p0), tuple(p0 + d_u * h), ro)
        inner = occ_cyl(tuple(p0 - 0.1 * d_u), tuple(p0 + d_u * (h + 0.2)), ri)
        return largest_solid(cut_all(outer, [inner]))

    def _translate_x(sh: Solid | Compound, dx: float) -> Solid | Compound:
        t = gp_Trsf()
        t.SetTranslation(gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(float(dx), 0.0, 0.0))
        res = Compound(BRepBuilderAPI_Transform(sh.wrapped, t, True).Shape())
        return largest_solid(res) if isinstance(sh, Solid) else res

    # 1. Stage 1 Torso Spine Chassis (`combined__stage1_center_torso_spine_chassis` + 39 sleeves)
    torso_man = read_json(ROOT / "artifacts/robot/body/torso/combined/assembly-parts.json")
    p_s1 = next(p for p in torso_man["parts"] if p["name"] == "combined__stage1_center_torso_spine_chassis")
    p_s1_slv = next(p for p in torso_man["parts"] if p["name"] == "combined__stage1_center_torso_spine_chassis_steel_sleeves")
    s1_raw = import_step(ensure(ROOT / p_s1["native_step"])).solids()[0]

    s1_adds = [
        Pos(-30.4, -1.25, 360.3) * Rot(0, 90, 0) * Cylinder(2.5, 3.9),
        Pos( 30.4, -1.25, 360.3) * Rot(0, 90, 0) * Cylinder(2.5, 3.9),
        Pos(-30.4, 19.25, 360.3) * Rot(0, 90, 0) * Cylinder(2.5, 3.9),
        Pos( 30.4, 19.25, 360.3) * Rot(0, 90, 0) * Cylinder(2.5, 3.9),
        Pos(-42.4, 9.0, 363.25) * Box(3.9, 24.4, 48.5),
        Pos( 42.4, 9.0, 363.25) * Box(3.9, 24.4, 48.5),
        Pos(0.0, 15.90, 343.25) * Box(84.8, 4.6, 8.5),
        Pos(0.0, -1.20, 343.25) * Box(84.8, 3.8, 8.5),
        Pos(-37.5, 12.0, 352.5) * Rot(0, -12, 0) * Box(5.6, 6.0, 19.0),
        Pos( 37.5, 12.0, 352.5) * Rot(0,  12, 0) * Box(5.6, 6.0, 19.0),
        Pos(-41.175, -0.85, 347.25) * Box(6.35, 4.7, 16.5),
        Pos( 41.175, -0.85, 347.25) * Box(6.35, 4.7, 16.5),
        Pos(-41.175, 18.85, 347.25) * Box(6.35, 4.7, 16.5),
        Pos( 41.175, 18.85, 347.25) * Box(6.35, 4.7, 16.5),
    ]
    s1_cuts = [
        Pos(-23.5, 27.75, 346.5) * Box(20.0, 5.5, 12.0),
        Pos(-29.0, 9.0, 368.5) * Box(10.0, 26.0, 42.0),
        Pos( 29.0, 9.0, 368.5) * Box(10.0, 26.0, 42.0),
        Pos(-42.4, -3.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
        Pos( 42.4, -3.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
        Pos(-42.4, 21.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
        Pos( 42.4, 21.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
        Pos(-42.4, 9.0, 372.5) * Rot(0, 90, 0) * Cylinder(5.2, 8.0),
        Pos( 42.4, 9.0, 372.5) * Rot(0, 90, 0) * Cylinder(5.2, 8.0),
        Pos(-44.8, 25.6, 301.8) * Cylinder(10.95, 76.0),
        Pos( 44.8, 25.6, 301.8) * Cylinder(10.95, 76.0),
        Pos(-44.8, 27.6, 348.0) * Cylinder(6.80, 24.0),
        Pos( 44.8, 27.6, 348.0) * Cylinder(6.80, 24.0),
    ]
    for sx in (-1.0, 1.0):
        for by in (-1.25, 19.25):
            for bz in (360.3, 384.75):
                s1_cuts.append(Pos(sx * 42.4, by, bz) * Rot(0, 90, 0) * Cylinder(1.94, 12.0))
    s1_sol = largest_solid(s1_raw.fuse(*s1_adds).cut(*s1_cuts))

    s1_slv_orig = import_step(ensure(ROOT / p_s1_slv["native_step"])).solids()
    s1_slv_new = []
    for slv in s1_slv_orig:
        bb = slv.bounding_box()
        cx, cz = 0.5 * (bb.min.X + bb.max.X), 0.5 * (bb.min.Z + bb.max.Z)
        if cz > 350.0 and abs(cx) > 25.0:
            dx = 12.0 if cx > 0.0 else -12.0
            s1_slv_new.append(_translate_x(slv, dx))
        else:
            s1_slv_new.append(slv)
    s1_slv_comp = Compound(children=s1_slv_new)

    # 2. Compact Blue Piece (`R/L_shoulder_pitch_to_roll_combined` on `R/L_shoulder_pitch_link`)
    # Inside X <= 88.0 mm, stay within R <= 16.5 mm of (Y=9.0, Z=352.0); at Y in [22.0, 25.4], stay within R <= 21.5 mm of (X=106.0, Z=360.0)
    pitch_cheek = occ_cyl((82.625, 9.0, 352.0), (86.00, 9.0, 352.0), 11.0)
    pitch_neck_bridge = occ_box(85.40, 89.00, 8.0, 24.50, 348.0, 359.0)
    pitch_bosses = [
        occ_cyl((82.625, 9.0 + u, 352.0 + v), (86.00, 9.0 + u, 352.0 + v), 3.6)
        for u, v in [(-7.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0)]
    ]
    pitch_ring = occ_cyl((82.625, 9.0, 352.0), (86.00, 9.0, 352.0), 9.6)
    roll_horn_ring = occ_cyl((106.0, 23.675, 360.0), (106.0, 26.675, 360.0), 10.2)
    roll_rear_web = occ_box(88.00, 108.5, 22.0, 25.40, 348.0, 359.5)
    rib_top = occ_box(95.8, 108.5, 22.0, 25.40, 358.0, 365.5)
    rib_bot = occ_box(95.8, 106.0, 22.0, 25.40, 348.0, 351.0)
    blue_raw = fuse_all([pitch_cheek, pitch_neck_bridge, *pitch_bosses, pitch_ring, roll_horn_ring, roll_rear_web, rib_top, rib_bot])
    blue_cuts = [
        occ_cyl((80.0, 9.0, 352.0), (91.0, 9.0, 352.0), 3.10),
        occ_cyl((106.0, 21.0, 360.0), (106.0, 29.0, 360.0), 3.10),
        occ_cyl((106.0, 18.0, 360.0), (106.0, 23.675, 360.0), 9.85),
        occ_cyl((80.0, 18.8, 354.0), (88.0, 18.8, 354.0), 3.00),
        occ_cyl((94.0, 21.0, 356.5), (94.0, 29.0, 356.5), 3.20),
    ]
    r_blue_sleeves = []
    for u, v in [(-7.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0)]:
        blue_cuts.append(occ_cyl((81.0, 9.0 + u, 352.0 + v), (87.0, 9.0 + u, 352.0 + v), 2.29))
        blue_cuts.append(occ_cyl((85.00, 9.0 + u, 352.0 + v), (92.0, 9.0 + u, 352.0 + v), 3.35))
        r_blue_sleeves.append(_make_sleeve((82.625, 9.0 + u, 352.0 + v), (1.0, 0.0, 0.0), 2.25, 1.65, 2.375))

        pt_rh = P_ROLL_NEW + ROT_FLIP @ np.array([-18.625, u, v])
        blue_cuts.append(occ_cyl((float(pt_rh[0]), 22.0, float(pt_rh[2])), (float(pt_rh[0]), 28.0, float(pt_rh[2])), 2.29))
        r_blue_sleeves.append(_make_sleeve((float(pt_rh[0]), 23.675, float(pt_rh[2])), (0.0, 1.0, 0.0), 2.25, 1.65, 3.0))

    r_blue = bevel_3d_edges(largest_solid(cut_all(blue_raw, blue_cuts)), r_outer=0.35, r_inner=0.60)
    l_blue = mirror_x(r_blue)
    r_blue_slv_comp = Compound(children=r_blue_sleeves)
    l_blue_slv_comp = Compound(children=[mirror_x(s) for s in r_blue_sleeves])

    # 3. Compact Green Piece (`R/L_shoulder_roll_to_yaw_combined` on `R/L_shoulder_roll_link`)
    roll_boss_pts = [
        P_ROLL_NEW + ROT_FLIP @ np.array([17.575, u1, u2])
        for u1, u2 in [(-10.25, 8.30), (-10.25, 32.75), (10.25, 8.30), (10.25, 32.75)]
    ]
    b0, b1, b2, b3 = [(float(p[0]), float(p[2])) for p in roll_boss_pts]
    front_poly = Polygon(
        (96.5, 348.7),
        (99.2, 351.8),
        (118.5, 358.8),
        (122.0, 355.5),
        (132.5, 326.0),
        (132.5, 323.5),
        (118.0, 323.5),
        (107.6, 323.5),
        align=None,
    )
    for bx, bz in (b0, b2, b3):
        front_poly = front_poly + Pos(bx, bz) * Circle(2.85)
    front_poly = front_poly + Pos(b1[0], b1[1]) * Circle(2.30)
    front_face = fillet_2d_face(front_poly.clean().faces()[0], r_default=1.2, r_inner=2.0, skip_xy=[(132.5, 323.5), (107.6, 323.5)])
    w_f1 = Pos(111.5, 342.0) * Circle(4.2).faces()[0]
    w_f2 = Pos(118.5, 331.5) * Circle(3.2).faces()[0]
    front_face = (front_face - w_f1 - w_f2).clean().faces()[0]
    front_cheek_web = extrude_xy_to_y(front_face, -15.525, -14.20)
    front_boss_pads = [occ_cyl((bx, -15.525, bz), (bx, -12.525, bz), 2.30) for bx, bz in (b0, b1, b2, b3)]

    yaw_plate_vw = Polygon(
        (-12.95, -9.35),
        (13.20, -9.35),
        (13.20, 35.35),
        (6.40, 35.35),
        (6.40, 30.0),
        (-6.0, 30.0),
        (-7.2, 35.35),
        (-12.95, 35.35),
        align=None,
    ).faces()[0]
    yaw_plate_face = fillet_2d_face(yaw_plate_vw, r_default=1.4, r_inner=2.2, skip_xy=[(-12.95, -9.35), (-12.95, 35.35)])
    yaw_center_clear = Pos(0.0, 0.0) * Circle(3.45).faces()[0]
    yw1 = Pos(-4.2, 18.5) * Circle(3.4).faces()[0]
    yaw_plate_face = (yaw_plate_face - yw1 - yaw_center_clear).clean().faces()[0]
    yaw_saddle_full = _extrude_yaw_local_x(yaw_plate_face, 17.575, 20.575)

    dorsal_wn = Polygon(
        (-5.5, 19.0),
        (35.35, 19.0),
        (35.35, 22.40),
        (-2.5, 22.40),
        (-5.5, 25.5),
        align=None,
    ).faces()[0]
    dorsal_rib = _extrude_yaw_local_y(dorsal_wn, 3.55, 6.40)

    ant_gusset_wn = Polygon(
        (-9.35, 19.30),
        (-4.25, 19.30),
        (-4.25, 28.5),
        (-9.35, 28.5),
        align=None,
    ).faces()[0]
    ant_gusset = _extrude_yaw_local_y(ant_gusset_wn, -12.20, 6.40)

    sh_post_f = occ_cyl((116.5, -21.5, 336.5), (116.5, -14.8, 336.5), 4.6)
    sh_post_r = occ_cyl((130.5, 22.5, 329.8), (130.5, 28.5, 329.8), 4.6)

    boss_keep = (
        Pos(-10.25, 8.30) * Circle(2.60).faces()[0]
        + Pos(10.25, 8.30) * Circle(2.60).faces()[0]
        + Pos(-10.25, 32.75) * Circle(2.60).faces()[0]
        + Pos(10.25, 32.75) * Circle(2.60).faces()[0]
    )
    bot_relief_face = (((Pos(0.0, 13.50) * RectangleRounded(20.60, 35.20, 1.8)).faces()[0] - boss_keep).clean().faces()[0])
    bot_relief = _extrude_yaw_local_x(bot_relief_face, 15.5, 19.25)

    top_relief_face = (((Pos(-7.00, 13.00) * RectangleRounded(16.50, 34.00, 1.8)).faces()[0] - boss_keep).clean().faces()[0])
    top_relief = _extrude_yaw_local_x(top_relief_face, 19.80, 22.0)

    bev_frame = bevel_3d_edges(largest_solid(fuse_all([front_cheek_web, yaw_saddle_full, dorsal_rib, ant_gusset])), r_outer=0.35, r_inner=0.60)
    green_raw = fuse_all([bev_frame, *front_boss_pads, sh_post_f, sh_post_r])
    green_cuts = [
        bot_relief,
        top_relief,
        occ_cyl((116.5, -23.0, 336.5), (116.5, -14.5, 336.5), 1.70),
        make_captive_nut_pocket((116.5, -18.82, 336.5), (0.0, 1.0, 0.0)),
        occ_cyl((130.5, 22.2, 329.8), (130.5, 30.0, 329.8), 1.70),
        make_captive_nut_pocket((130.5, 25.82, 329.8), (0.0, -1.0, 0.0)),
    ]
    r_green_sleeves = []
    for bx, bz in (b0, b1, b2, b3):
        green_cuts.append(occ_cyl((bx, -16.0, bz), (bx, -11.0, bz), 1.94))
        green_cuts.append(occ_cyl((bx, -30.0, bz), (bx, -15.50, bz), 2.25))
        r_green_sleeves.append(_make_sleeve((bx, -12.525, bz), (0.0, -1.0, 0.0), 1.90, 1.20, 3.0))
    for u_y in (-10.25, 10.25):
        for v_y in (8.30, 32.75):
            pt_y = P_YAW_NEW + 17.575 * N_YAW + u_y * V_YAW + v_y * W_YAW
            p_a = pt_y - 1.0 * N_YAW
            p_b = pt_y + 3.1 * N_YAW
            p_c = pt_y + 25.0 * N_YAW
            green_cuts.append(occ_cyl(tuple(p_a), tuple(p_b), 1.79))
            green_cuts.append(occ_cyl(tuple(pt_y + 2.98 * N_YAW), tuple(p_c), 2.25))
            r_green_sleeves.append(_make_sleeve(tuple(pt_y), tuple(N_YAW), 1.75, 1.20, 3.0))

    r_green = largest_solid(cut_all(green_raw, green_cuts))
    l_green = mirror_x(r_green)
    r_green_slv_comp = Compound(children=r_green_sleeves)
    l_green_slv_comp = Compound(children=[mirror_x(s) for s in r_green_sleeves])

    def _extrude_yaw_local_z(face_vn, w0: float, w1: float) -> Solid:
        wa, wb = min(float(w0), float(w1)), max(float(w0), float(w1))
        p_orig = P_YAW_NEW + wa * W_YAW
        pl = Plane(origin=tuple(p_orig), x_dir=tuple(V_YAW), y_dir=tuple(N_YAW))
        f3d = (pl * face_vn).faces()[0]
        return Solid(Solid.extrude(f3d, Vector(*tuple((wb - wa) * W_YAW))).wrapped)

    def _vn_yaw(pt_xyz) -> tuple[float, float]:
        d = np.asarray(pt_xyz, dtype=float) - P_YAW_NEW
        return (round(float(np.dot(d, V_YAW)), 3), round(float(np.dot(d, N_YAW)), 3))

    # 4. Inside-Mounted Stage 3 Yellow Connector (`R/L_upper_arm_yaw_to_elbow_combined` + 8 sleeves) &
    #    Flipped-Cheek Stage 4 Forearm Connector (`R/L_forearm_elbow_to_wrist_combined` + 10 sleeves)
    yaw_hub_vw = Polygon(
        (-11.2, -5.0),
        (-11.2, 12.6),
        (5.0, 12.6),
        (10.8, 4.5),
        (10.8, -4.5),
        (4.5, -10.8),
        (-5.0, -10.8),
        align=None,
    ).faces()[0]
    yaw_hub_face = fillet_2d_face(yaw_hub_vw, r_default=1.4, r_inner=2.4, skip_xy=())
    yaw_hub_plate = _extrude_yaw_local_x(yaw_hub_face, -22.2, -18.625)

    elb_yoke_yz = Polygon(
        (-4.20, 260.0),
        (-4.20, 276.5),
        (2.80, 276.5),
        (6.25, 264.5),
        (6.25, 248.0),
        (1.20, 242.7),
        (-7.20, 242.7),
        (-12.35, 248.0),
        (-12.35, 255.8),
        align=None,
    ).faces()[0]
    elb_yoke_face = fillet_2d_face(elb_yoke_yz, r_default=1.4, r_inner=2.4, skip_xy=())
    ew = Pos(-0.8, 266.0) * Circle(1.8).faces()[0]
    elb_yoke_face = (elb_yoke_face - ew).clean().faces()[0]
    elb_yoke_plate = extrude_xy_to_x(elb_yoke_face, 114.225, 117.225)

    u_elb_top_out, n_elb_top_out = _vn_yaw((123.5, 0.0, 276.0))
    u_elb_top_in,  n_elb_top_in  = _vn_yaw((114.3, 0.0, 276.0))
    u_elb_bot_in,  n_elb_bot_in  = _vn_yaw((114.3, 0.0, 264.5))
    u_elb_bot_out, n_elb_bot_out = _vn_yaw((117.2, 0.0, 264.5))

    gusset_vn = Polygon(
        (-11.0, -20.4),
        (-1.8, -20.4),
        (u_elb_top_out, n_elb_top_out),
        (u_elb_bot_out, n_elb_bot_out),
        (u_elb_bot_in,  n_elb_bot_in),
        (u_elb_top_in,  n_elb_top_in),
        align=None,
    ).faces()[0]
    gusset_vn_face = fillet_2d_face(
        gusset_vn,
        r_default=1.4,
        r_inner=2.5,
        skip_xy=[
            (-11.0, -20.4),
            (-1.8, -20.4),
            (u_elb_top_in, n_elb_top_in),
            (u_elb_bot_in, n_elb_bot_in),
        ],
    )
    gusset_ant = _extrude_yaw_local_z(gusset_vn_face, 5.8, 8.8)
    gusset_post = _extrude_yaw_local_z(gusset_vn_face, 10.2, 12.60)

    yellow_raw = fuse_all([yaw_hub_plate, elb_yoke_plate, gusset_ant, gusset_post])
    yellow_bev = bevel_3d_edges(largest_solid(yellow_raw), r_outer=0.38, r_inner=0.65)

    s3_yaw_origin = P_YAW_NEW - 18.625 * N_YAW
    s3_yaw_pts = [s3_yaw_origin + u * V_YAW + v * W_YAW for u, v in [(-7.0, 0.0), (0.0, -7.0), (0.0, 7.0), (7.0, 0.0)]]
    s3_elb_pts = [np.array([117.225, -3.0 + u, 252.0 + v]) for u, v in [(-7.0, 0.0), (0.0, 7.0), (7.0, 0.0), (0.0, -7.0)]]

    yellow_cuts = [
        occ_cyl(tuple(s3_yaw_origin + 2.0 * N_YAW), tuple(s3_yaw_origin - 10.0 * N_YAW), 3.25),
        occ_cyl((120.0, -3.0, 252.0), (110.0, -3.0, 252.0), 3.25),
        occ_box(118.4, 160.0, -20.0, 15.0, 240.0, 263.0),
        occ_cyl((118.2, -3.0, 252.0), (160.0, -3.0, 252.0), 24.4),
        _intersect_solid(occ_cyl((113.5, -3.0, 252.0), (118.5, -3.0, 252.0), 23.2), occ_box(113.0, 119.0, -30.0, -0.80, 252.0, 276.0)),
    ]
    r_yellow_sleeves = []
    for pt in s3_yaw_pts:
        d = -N_YAW
        yellow_cuts.append(occ_cyl(tuple(pt - 1.0 * d), tuple(pt + 3.65 * d), 2.29))
        yellow_cuts.append(occ_cyl(tuple(pt + 3.50 * d), tuple(pt + 28.0 * d), 3.15))
        r_yellow_sleeves.append(_make_sleeve(tuple(pt), tuple(d), 2.25, 1.60, 3.575))
    for pt in s3_elb_pts:
        d = np.array([-1.0, 0.0, 0.0])
        yellow_cuts.append(occ_cyl(tuple(pt - 1.0 * d), tuple(pt + 3.15 * d), 2.29))
        yellow_cuts.append(occ_cyl(tuple(pt + 2.95 * d), tuple(pt + 25.0 * d), 3.15))
        r_yellow_sleeves.append(_make_sleeve(tuple(pt), tuple(d), 2.25, 1.60, 3.00))

    r_yellow = largest_solid(cut_all(yellow_bev, yellow_cuts))
    l_yellow = mirror_x(r_yellow)
    r_yellow_slv = Compound(children=r_yellow_sleeves)
    l_yellow_slv = Compound(children=[mirror_x(s) for s in r_yellow_sleeves])

    lat_tall_poly = Polygon((-15.95, 212.0), (9.95, 212.0), (9.95, 246.8), (-15.95, 246.8), align=None)
    lat_tall_face = fillet_2d_face(lat_tall_poly.clean().faces()[0], r_default=2.2, skip_xy=[(-15.95, 212.0), (9.95, 212.0)])
    lw_tall = Pos(-3.0, 231.5) * Circle(5.8).faces()[0]
    lat_tall_face = (lat_tall_face - lw_tall).clean().faces()[0]
    lateral_cheek_web = extrude_xy_to_x(lat_tall_face, 154.90, 156.425)
    lateral_boss_pads = [
        occ_cyl((153.425, y, z), (156.425, y, z), 2.65)
        for y in (-13.25, 7.25)
        for z in (219.25, 243.70)
    ]

    med_short_poly = Polygon((-15.95, 212.0), (9.95, 212.0), (9.95, 222.8), (-15.95, 222.8), align=None)
    med_short_face = fillet_2d_face(med_short_poly.clean().faces()[0], r_default=1.5, skip_xy=[(-15.95, 212.0), (9.95, 212.0)])
    mw_short = Pos(-3.0, 217.5) * Circle(2.6).faces()[0]
    med_short_face = (med_short_face - mw_short).clean().faces()[0]
    medial_cheek_web = extrude_xy_to_x(med_short_face, 119.20, 120.725)
    medial_boss_pads = [
        occ_cyl((119.20, y, 219.25), (122.20, y, 219.25), 2.65)
        for y in (-13.25, 7.25)
    ]

    aframe_xz_poly = Polygon(
        (119.20, 220.0),
        (120.725, 220.0),
        (120.725, 215.5),
        (154.90, 215.5),
        (154.90, 220.0),
        (156.425, 220.0),
        (156.425, 212.0),
        (141.20, 190.6),
        (141.20, 174.8),
        (138.4, 174.8),
        (138.4, 171.0),
        (124.6, 171.0),
        (124.6, 174.8),
        (119.20, 212.0),
        align=None,
    ).faces()[0]
    aframe_xz_face = fillet_2d_face(
        aframe_xz_poly,
        r_default=1.3,
        r_inner=1.8,
        skip_xy=[
            (119.20, 220.0),
            (120.725, 220.0),
            (154.90, 220.0),
            (156.425, 220.0),
            (141.20, 174.8),
            (138.4, 174.8),
            (138.4, 171.0),
            (124.6, 171.0),
            (124.6, 174.8),
        ],
    )
    af_w1 = Pos(134.5, 204.2) * Circle(5.5).faces()[0]
    af_w2 = Pos(129.0, 186.8) * Circle(3.4).faces()[0]
    aframe_xz_face = (aframe_xz_face - af_w1 - af_w2).clean().faces()[0]
    aframe_ant = extrude_xy_to_y(aframe_xz_face, -13.0, -9.2)
    aframe_post = extrude_xy_to_y(aframe_xz_face, -0.2, 3.30)

    ub_poly = fillet_2d_face(
        Polygon((119.20, -15.95), (156.425, -15.95), (156.425, 9.95), (119.20, 9.95), align=None).faces()[0],
        r_default=1.5,
        skip_xy=[(119.20, -15.95), (156.425, -15.95), (156.425, 9.95), (119.20, 9.95)],
    )
    ub_win = Pos(137.8, -3.0) * RectangleRounded(29.0, 14.0, 2.5).faces()[0]
    ub_deck = extrude_xy_to_z((ub_poly - ub_win).clean().faces()[0], 212.0, 215.5)

    wrist_xy = Polygon((124.6, -13.0), (138.4, -13.0), (138.4, 8.50), (124.6, 8.50), align=None)
    wrist_face = (
        fillet_2d_face(wrist_xy.clean().faces()[0], r_default=1.4, skip_xy=[(138.4, 8.50), (138.4, -13.0)])
        - Pos(131.5, -5.5) * Circle(2.0).faces()[0]
    ).clean().faces()[0]
    wrist_pad = extrude_xy_to_z(wrist_face, 171.0, 174.8)

    post_ff187 = extrude_xy_to_z(
        fillet_2d_face(Polygon((134.6, -22.25), (141.20, -22.25), (141.20, -10.5), (134.6, -10.5), align=None).faces()[0], r_default=1.2, skip_xy=[(134.6, -10.5), (141.20, -10.5)]),
        183.4, 190.6,
    )
    post_fb187 = extrude_xy_to_z(
        fillet_2d_face(Polygon((134.6, 1.0), (141.20, 1.0), (141.20, 15.65), (134.6, 15.65), align=None).faces()[0], r_default=1.2, skip_xy=[(134.6, 1.0), (141.20, 1.0)]),
        183.4, 190.6,
    )
    post_ff178 = extrude_xy_to_z(
        fillet_2d_face(Polygon((134.6, -18.15), (141.20, -18.15), (141.20, -10.5), (134.6, -10.5), align=None).faces()[0], r_default=1.2, skip_xy=[(134.6, -10.5), (141.20, -10.5)]),
        174.8, 181.6,
    )
    post_fb178 = extrude_xy_to_z(
        fillet_2d_face(Polygon((134.6, 1.0), (141.20, 1.0), (141.20, 11.65), (134.6, 11.65), align=None).faces()[0], r_default=1.2, skip_xy=[(134.6, 1.0), (141.20, 1.0)]),
        174.8, 181.6,
    )

    s4_raw = fuse_all([
        medial_cheek_web,
        lateral_cheek_web,
        aframe_ant,
        aframe_post,
        ub_deck,
        wrist_pad,
        post_ff187,
        post_fb187,
        post_ff178,
        post_fb178,
        *medial_boss_pads,
        *lateral_boss_pads,
    ])
    s4_cuts = []
    r_s4_sleeves = []
    for y in (-13.25, 7.25):
        for z in (219.25, 243.70):
            s4_cuts.append(occ_cyl((152.0, y, z), (156.6, y, z), 1.79))
            s4_cuts.append(occ_cyl((156.4, y, z), (175.0, y, z), 2.25))
            r_s4_sleeves.append(_make_sleeve((153.425, y, z), (1.0, 0.0, 0.0), 1.75, 1.20, 3.0))
    for y in (-13.25, 7.25):
        z = 219.25
        s4_cuts.append(occ_cyl((123.5, y, z), (119.0, y, z), 1.79))
        s4_cuts.append(occ_cyl((119.22, y, z), (100.0, y, z), 2.25))
        r_s4_sleeves.append(_make_sleeve((122.20, y, z), (-1.0, 0.0, 0.0), 1.75, 1.20, 3.0))
    for x in (128.0, 135.0):
        for y in (-3.0, 5.0):
            s4_cuts.append(occ_cyl((x, y, 170.0), (x, y, 175.0), 2.29))
            s4_cuts.append(occ_cyl((x, y, 174.75), (x, y, 198.0), 3.15))
            r_s4_sleeves.append(_make_sleeve((x, y, 171.0), (0.0, 0.0, 1.0), 2.25, 1.60, 3.8))
    for cy, cz, y0, y1, ax_y in [
        (-19.97, 187.0, -22.65, -10.5,  1.0),
        ( 13.37, 187.0,   1.00,  16.05, -1.0),
        (-15.87, 178.0, -18.55, -10.5,  1.0),
        (  9.37, 178.0,   1.00,  12.05, -1.0),
    ]:
        s4_cuts.append(occ_cyl((138.0, y0 - 4.0, cz), (138.0, y1 + 4.0, cz), 1.70))
        s4_cuts.append(make_captive_nut_pocket((138.0, cy, cz), (0.0, ax_y, 0.0)))

    r_s4 = largest_solid(cut_all(s4_raw, s4_cuts))
    l_s4 = mirror_x(r_s4)
    r_s4_slv = Compound(children=r_s4_sleeves)
    l_s4_slv = Compound(children=[mirror_x(s) for s in r_s4_sleeves])

    shifted_arm_solids: dict[str, Solid | Compound] = {
        "R_upper_arm_yaw_to_elbow_combined": r_yellow,
        "L_upper_arm_yaw_to_elbow_combined": l_yellow,
        "R_upper_arm_yaw_to_elbow_combined_steel_sleeves": r_yellow_slv,
        "L_upper_arm_yaw_to_elbow_combined_steel_sleeves": l_yellow_slv,
        "R_forearm_elbow_to_wrist_combined": r_s4,
        "L_forearm_elbow_to_wrist_combined": l_s4,
        "R_forearm_elbow_to_wrist_combined_steel_sleeves": r_s4_slv,
        "L_forearm_elbow_to_wrist_combined_steel_sleeves": l_s4_slv,
    }

    # 5. Smooth Continuous Core Body Shell (`center_front_shell` & `center_back_shell`) without neck cowl notch
    #    or shoulder steps/dents, enclosing `upper__R/L_pitch_C047_case`, `main_horn`, `pitch_cheek`, and pitch horn screws:
    scale_t = 500.0 / 350.0
    torso_stations = [
        [185.71, 0.0, 7.00, 22.00, 20.00],
        [193.00, 0.0, 6.50, 36.00, 29.00],
        [204.00, 0.0, 5.50, 64.00, 40.00],
        [218.00, 0.0, 4.60, 76.80, 45.40],
        [242.00, 0.0, 4.10, 78.60, 45.60],
        [270.00, 0.0, 4.09, 79.80, 47.80],
        [284.29, 0.0, 4.25, 82.85, 50.60],
        [327.14, 0.0, 5.61, 89.20, 54.00],
        [361.43, 0.0, 7.14, 89.20, 46.50],
        [380.00, 0.0, 7.14, 88.80, 43.00],
        [391.43, 0.0, 7.14, 66.00, 34.50],
        [397.14, 0.0, 7.14, 32.86, 25.71],
    ]
    torso_outer = Solid.make_loft([
        Wire.make_ellipse(rx, ry, Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in torso_stations
    ])
    torso_inner_rows = [[z, x, y, rx - 2.0, ry - 2.0] for z, x, y, rx, ry in torso_stations]
    torso_inner_rows[0][0] += 2.0
    torso_inner_rows[-3][0] -= 0.8
    torso_inner_rows[-2][0] -= 1.6
    torso_inner_rows[-1][0] -= 1.8
    torso_inner = Solid.make_loft([
        Wire.make_ellipse(rx, ry, Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in torso_inner_rows
    ])
    torso_thin_rows = [[z, x, y, rx - 0.85, ry - 0.85] for z, x, y, rx, ry in torso_stations]
    torso_thin_rows[0][0] += 0.85
    torso_thin_rows[4][3] = torso_stations[4][3] - 0.55
    torso_thin_rows[5][3] = torso_stations[5][3] - 0.55
    torso_thin_rows[-3][0] -= 0.4
    torso_thin_rows[-2][0] -= 0.8
    torso_thin_rows[-1][0] -= 0.9
    torso_thin = Solid.make_loft([
        Wire.make_ellipse(rx, ry, Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in torso_thin_rows
    ])
    pitch_boxes = (Pos(66.5, 9.0, 364.5) * Box(45.5, 36.0, 49.0)).fuse(Pos(-66.5, 9.0, 364.5) * Box(45.5, 36.0, 49.0))
    torso_shell = (torso_outer - torso_inner) - (pitch_boxes & torso_thin)

    neck_hole = Solid.make_cylinder(20.0 * scale_t, 40.0 * scale_t, Plane(origin=(0.0, 4.0 * scale_t, 260.0 * scale_t)))
    torso_shell = torso_shell - neck_hole

    for sg in (-1.0, 1.0):
        leg_tool_main = Solid.make_sphere(1.0).scale((47.5, 59.5, 54.5)).translate((sg * 54.0, 5.0, 185.0))
        leg_tool_flex = Solid.make_sphere(1.0).scale((33.0, 46.0, 57.5)).translate((sg * 64.0, -5.0, 182.0))
        sh_cut = Pos(sg * 86.5, 9.0, 352.0) * Rot(0, 90, 0) * Cylinder(18.8, 8.6)
        hip_yaw_relief = (Pos(sg * 53.0, 2.0, 261.0) * Cylinder(25.0, 14.0)) & torso_thin
        torso_shell = torso_shell - leg_tool_main - leg_tool_flex - sh_cut - hip_yaw_relief

    p_ind = next(p for p in torso_man["parts"] if p["name"] == "ape_body_shell__chest_indicator_relief4_r33")
    y_ind_mid = 0.5 * (float(p_ind["bounds_mm"][0][1]) + float(p_ind["bounds_mm"][1][1]))
    ind_tool = (
        Solid.make_sphere(1.0)
        .scale((1.95 * scale_t, 4.5 * scale_t, 5.8 * scale_t))
        .translate((0.0, y_ind_mid, 244.0 * scale_t))
    )
    torso_shell = torso_shell - ind_tool

    s_front = torso_shell & (Pos(0.0, -57.307, 290.0) * Box(240.0, 125.389, 250.0))
    s_back = torso_shell & (Pos(0.0, 66.038, 290.0) * Box(240.0, 120.000, 250.0))
    for z_slot in (324.0, 330.0, 336.0, 342.0):
        s_back = s_back.cut(Pos(0.0, 60.0, z_slot) * Box(28.0, 50.0, 2.4))

    s_t01 = import_step(str(ensure(ROOT / "artifacts/cove50-fitted-shell/r1-20261027/panels/T01_chest.step")))
    s_t02 = import_step(str(ensure(ROOT / "artifacts/cove50-fitted-shell/r1-20261027/panels/T02_back.step")))
    box_couplers = (Pos(-85.0, 5.7143, 293.0) * Box(22.0, 18.0, 16.0)).fuse(Pos(85.0, 5.7143, 293.0) * Box(22.0, 18.0, 16.0))
    cf_cpl = ((s_t01 & (Pos(-85.0, 5.7143, 293.0) * Box(22.0, 18.0, 16.0))) & torso_outer)
    cf_cpl = cf_cpl.fuse(cf_cpl.mirror(Plane.YZ))

    f_bosses = []
    f_cuts = []
    for sgn in (-1.0, 1.0):
        fb = (Pos(sgn * 16.0, -26.50, 224.0) * Rot(90, 0, 0) * Cylinder(3.50, 28.90)) & torso_outer
        f_bosses.append(fb)
        f_cuts.append(Pos(sgn * 16.0, -26.0, 224.0) * Rot(90, 0, 0) * Cylinder(1.15, 32.0))
        f_cuts.append(Pos(sgn * 16.0, -29.5, 224.0) * Rot(90, 0, 0) * Cylinder(2.30, 26.0))

    center_front_shell_sol = max(((s_front - box_couplers).fuse(cf_cpl, *f_bosses)).cut(*f_cuts).solids(), key=lambda s: s.volume)

    box_mount2 = Pos(10.10, 42.0, 377.457) * Box(14.0, 16.0, 14.0)
    box_m1_src = Pos(-49.90, 41.0, 275.457) * Box(12.0, 14.0, 12.0)
    box_m3_src = Pos(49.90, 41.0, 275.457) * Box(12.0, 14.0, 12.0)
    m1_piece = ((s_t02 & box_m1_src) - (Pos(-49.90, 34.80, 275.457) * Box(16.0, 9.60, 16.0))) & torso_outer
    m3_piece = m1_piece.mirror(Plane.YZ)
    m2_piece = (s_t02 & box_mount2) & torso_inner

    b_low_boss = (Pos(0.0, 37.025, 204.0) * Rot(90, 0, 0) * Cylinder(3.80, 17.95)) & torso_outer
    b_low_cuts = [
        Pos(0.0, 37.0, 204.0) * Rot(90, 0, 0) * Cylinder(1.65, 22.0),
        Pos(0.0, 40.5, 204.0) * Rot(90, 0, 0) * Cylinder(2.85, 16.0),
    ]
    center_back_shell_sol = max(
        (s_back - box_couplers - box_m1_src - box_m3_src)
        .fuse((s_t02 & box_couplers) & torso_outer, m2_piece, m1_piece, m3_piece, b_low_boss)
        .cut(*b_low_cuts)
        .solids(),
        key=lambda s: s.volume,
    )

    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

    def _make_ell_wire(z: float, x0: float, x1: float, y0: float, y1: float, inset: float = 0.0) -> Wire:
        xc = 0.5 * (x0 + x1)
        rx = max(2.0, 0.5 * (x1 - x0) - inset)
        yc = 0.5 * (y0 + y1)
        ry = max(2.0, 0.5 * (y1 - y0) - inset)
        return Wire.make_ellipse(rx, ry, Plane(origin=(xc, yc, z)))

    def _make_deg2_loft(st_rows: list[tuple[float, float, float, float, float]], inset: float = 0.0) -> Solid:
        builder = BRepOffsetAPI_ThruSections(True, False)
        builder.SetMaxDegree(2)
        for s in st_rows:
            builder.AddWire(_make_ell_wire(*s, inset=inset).wrapped)
        builder.Build()
        return Solid(builder.Shape())

    stations_out = [
        (275.0, 108.5, 155.5, -31.8, 35.0),
        (292.0, 106.5, 155.0, -32.0, 36.0),
        (312.0, 103.5, 152.5, -31.4, 36.8),
        (336.0,  96.0, 146.5, -29.8, 37.5),
        (354.0,  87.5, 140.5, -27.5, 38.5),
        (365.0,  88.0, 134.0, -25.5, 37.5),
        (371.0,  95.5, 128.5, -23.0, 35.0),
        (374.2, 103.5, 121.0, -15.5, 28.5),
    ]

    u_out = _make_deg2_loft(stations_out, inset=0.0)
    u_in = _intersect_solid(_make_deg2_loft(stations_out, inset=2.0), occ_box(70.0, 175.0, -45.0, 50.0, 275.2, 372.2))
    u_thin = _make_deg2_loft(stations_out, inset=0.90)

    internal_relief = _intersect_solid(occ_box(88.0, 156.0, -30.5, 36.0, 275.2, 371.8), u_thin)
    yaw_case_relief = fuse_all([
        occ_box(116.1, 124.0, -19.90, -14.0, 291.0, 297.6),
        occ_box(115.35, 124.0, -19.88, -14.0, 296.0, 301.5),
        occ_box(114.95, 124.0,  17.5,  25.1, 290.0, 303.5),
    ])
    roll_top_relief = occ_box(108.4, 114.5, -10.5, 19.5, 370.0, 372.9)
    green_seat_relief = occ_box(96.4, 106.0, -16.2, -13.5, 324.0, 349.5)
    bot_cut = occ_box(85.0, 175.0, -45.0, 50.0, 245.0, 279.9)
    med_cut_1 = occ_box(80.0, 91.2, -35.0, 40.0, 276.0, 385.0)
    med_cut_3 = _intersect_solid(
        occ_cyl((106.0, -5.0, 360.0), (106.0, 25.8, 360.0), 24.0),
        occ_box(80.0, 108.5, -4.5, 25.8, 335.0, 382.0),
    )
    green_post_clear = fuse_all([
        occ_cyl((116.5, -21.52, 336.5), (116.5, -14.0, 336.5), 4.70),
        occ_cyl((130.5, 22.0, 329.8), (130.5, 28.52, 329.8), 4.70),
    ])

    u_shell_raw = largest_solid(cut_all(u_out, [
        u_in,
        internal_relief,
        yaw_case_relief,
        roll_top_relief,
        green_seat_relief,
        bot_cut,
        med_cut_1,
        med_cut_3,
        green_post_clear,
    ]))
    sh_bosses = [
        _intersect_solid(occ_cyl((116.5, -21.5, 336.5), (116.5, -30.0, 336.5), 4.6), u_out),
        _intersect_solid(occ_cyl((130.5, 28.52, 329.8), (130.5, 36.0, 329.8), 4.6), u_out),
    ]
    sh_cuts = [
        occ_cyl((116.5, -21.0, 336.5), (116.5, -30.0, 336.5), 1.70),
        occ_cyl((116.5, -23.55, 336.5), (116.5, -30.0, 336.5), 3.15),
        occ_cyl((130.5, 28.0, 329.8), (130.5, 36.0, 329.8), 1.70),
        occ_cyl((130.5, 29.95, 329.8), (130.5, 36.0, 329.8), 3.15),
    ]
    r_upper_shell = largest_solid(cut_all(fuse_all([u_shell_raw, *sh_bosses]), sh_cuts))
    l_upper_shell = mirror_x(r_upper_shell)

    r_sh_f_nut = _make_hex_nut((116.5, -18.95, 336.5), (0.0, -1.0, 0.0))
    r_sh_f_scr = _make_m3x8_screw((116.5, -23.8, 336.5), (0.0, -1.0, 0.0))
    r_sh_r_nut = _make_hex_nut((130.5, 25.95, 329.8), (0.0, 1.0, 0.0))
    r_sh_r_scr = _make_m3x8_screw((130.5, 30.2, 329.8), (0.0, 1.0, 0.0))

    return {
        "combined__stage1_center_torso_spine_chassis": s1_sol,
        "combined__stage1_center_torso_spine_chassis_steel_sleeves": s1_slv_comp,
        "center_front_shell": center_front_shell_sol,
        "center_back_shell": center_back_shell_sol,
        "R_shoulder_pitch_to_roll_combined": r_blue,
        "L_shoulder_pitch_to_roll_combined": l_blue,
        "R_shoulder_pitch_to_roll_combined_steel_sleeves": r_blue_slv_comp,
        "L_shoulder_pitch_to_roll_combined_steel_sleeves": l_blue_slv_comp,
        "R_shoulder_roll_to_yaw_combined": r_green,
        "L_shoulder_roll_to_yaw_combined": l_green,
        "R_shoulder_roll_to_yaw_combined_steel_sleeves": r_green_slv_comp,
        "L_shoulder_roll_to_yaw_combined_steel_sleeves": l_green_slv_comp,
        **shifted_arm_solids,
        "R_upper_arm_hard_cover": r_upper_shell,
        "L_upper_arm_hard_cover": l_upper_shell,
        "R_upper_arm_shell_front_M3_nut": r_sh_f_nut,
        "L_upper_arm_shell_front_M3_nut": mirror_x(r_sh_f_nut),
        "R_upper_arm_shell_front_M3x8": r_sh_f_scr,
        "L_upper_arm_shell_front_M3x8": mirror_x(r_sh_f_scr),
        "R_upper_arm_shell_rear_M3_nut": r_sh_r_nut,
        "L_upper_arm_shell_rear_M3_nut": mirror_x(r_sh_r_nut),
        "R_upper_arm_shell_rear_M3x8": r_sh_r_scr,
        "L_upper_arm_shell_rear_M3x8": mirror_x(r_sh_r_scr),
    }


def build_smooth_forearm_shells() -> dict[str, trimesh.Trimesh]:
    """Build smooth, single-arc parabolic B-Rep barrel forearm shells (`R/L_fore_front_hard_cover` and
    `R/L_fore_back_hard_cover`) matching the smooth organic outside curves of the leg shells and shifted
    inward by `-7.0 mm` in X (`P_ELBOW = (±138.0, -3.0, 252.0)`):
      - 4-station `Wire.make_ellipse` loft (`Z = 172.5 -> 261.9 mm`) with zero external ribs or lateral cutouts
      - Encloses `upper__R/L_elbow_C047_case`, `PA2x7_1..4`, `R/L_forearm_elbow_to_wrist_combined`,
        `upper__R/L_elbow_C047_main_horn`, `elb_yoke_plate`, and all 4 elbow horn screws inside a smooth barrel
      - Exact smooth cylindrical M3 counterbores at `X = +-138.0 mm, Z = 178.0 & 187.0 mm`."""

    def _make_ell_wire(z: float, x0: float, x1: float, y0: float, y1: float, inset: float = 0.0) -> Wire:
        xc = 0.5 * (x0 + x1)
        rx = max(2.0, 0.5 * (x1 - x0) - inset)
        yc = 0.5 * (y0 + y1)
        ry = max(2.0, 0.5 * (y1 - y0) - inset)
        return Wire.make_ellipse(rx, ry, Plane(origin=(xc, yc, z)))

    fore_stations = [
        (171.5, 116.5, 154.5, -24.2, 17.5),
        (196.0, 114.5, 163.5, -28.6, 22.4),
        (220.0, 115.2, 165.5, -28.8, 22.8),
        (245.0, 108.0, 166.0, -24.5, 19.0),
        (263.0, 108.0, 165.5, -23.2, 17.8),
    ]

    f_out = Solid.make_loft([_make_ell_wire(*s, inset=0.0) for s in fore_stations])
    f_in = Solid.make_loft([_make_ell_wire(*s, inset=2.0) for s in fore_stations])
    f_thin = Solid.make_loft([_make_ell_wire(*s, inset=0.85) for s in fore_stations])

    f_in_main = _intersect_solid(f_in, occ_box(95.0, 175.0, -35.0, 35.0, 170.0, 261.0))
    f_in_neck_open = _intersect_solid(f_in, occ_box(112.0, 125.0, -18.0, 12.0, 255.0, 264.5))
    c047_relief = _intersect_solid(occ_box(109.5, 159.5, -23.0, 16.5, 173.5, 262.15), f_thin)
    s4_medial_relief = occ_box(119.1, 124.0, -16.1, 10.1, 211.8, 223.0)
    f_wall = largest_solid(cut_all(f_out, [f_in_main, f_in_neck_open, c047_relief, s4_medial_relief]))

    top_cutter = occ_box(95.0, 175.0, -40.0, 40.0, 261.9, 275.0)
    bot_wrist_cutter = occ_box(95.0, 175.0, -40.0, 40.0, 160.0, 172.5)
    med_flex_chamfer = occ_box(100.0, 118.5, -30.0, -13.5, 252.0, 265.0)

    f_trimmed = largest_solid(cut_all(f_wall, [top_cutter, bot_wrist_cutter, med_flex_chamfer]))

    front_raw = largest_solid(cut_all(f_trimmed, [occ_box(95.0, 175.0, -3.30, 40.0, 165.0, 270.0)]))
    back_raw = largest_solid(cut_all(f_trimmed, [occ_box(95.0, 175.0, -40.0, -2.70, 165.0, 270.0)]))

    ff_bosses = [
        _intersect_solid(occ_cyl((138.0, -22.65, 187.0), (138.0, -30.00, 187.0), 4.6), f_out),
        _intersect_solid(occ_cyl((138.0, -18.55, 178.0), (138.0, -27.00, 178.0), 4.6), f_out),
    ]
    ff_cuts = [
        occ_cyl((138.0, -22.00, 187.0), (138.0, -30.00, 187.0), 1.70),
        occ_cyl((138.0, -24.38, 187.0), (138.0, -30.00, 187.0), 3.15),
        occ_cyl((138.0, -18.00, 178.0), (138.0, -27.00, 178.0), 1.70),
        occ_cyl((138.0, -21.09, 178.0), (138.0, -27.00, 178.0), 3.15),
    ]
    r_fore_front = largest_solid(cut_all(fuse_all([front_raw, *ff_bosses]), ff_cuts))

    fb_bosses = [
        _intersect_solid(occ_cyl((138.0, 16.05, 187.0), (138.0, 24.00, 187.0), 4.6), f_out),
        _intersect_solid(occ_cyl((138.0, 12.05, 178.0), (138.0, 21.00, 178.0), 4.6), f_out),
    ]
    fb_cuts = [
        occ_cyl((138.0, 15.50, 187.0), (138.0, 24.00, 187.0), 1.70),
        occ_cyl((138.0, 18.38, 187.0), (138.0, 24.00, 187.0), 3.15),
        occ_cyl((138.0, 11.50, 178.0), (138.0, 21.00, 178.0), 1.70),
        occ_cyl((138.0, 15.09, 178.0), (138.0, 21.00, 178.0), 3.15),
    ]
    r_fore_back = largest_solid(cut_all(fuse_all([back_raw, *fb_bosses]), fb_cuts))

    breps = {
        "R_fore_front_hard_cover": r_fore_front,
        "R_fore_back_hard_cover": r_fore_back,
        "L_fore_front_hard_cover": mirror_x(r_fore_front),
        "L_fore_back_hard_cover": mirror_x(r_fore_back),
    }
    solid_to_watertight_arrays = _get_solid_to_watertight_arrays()
    meshes = {}
    for k, v in breps.items():
        v_arr, f_arr = solid_to_watertight_arrays(v, tol=0.05)
        meshes[k] = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False)
    return meshes


def build_cute_ape_head_shell() -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    """Build a non-spherical primate head shell (`original_profile_head_skin_v23`) + warm-dark glossy eyes +
    eye specular glints + compact sculpted neck collar that directly traces the 3 orthographic/perspective
    reference drawings:
      1. Side View Profile (`39d0854c.png`):
         - Upward-sloping bottom jawline from the low chin (`Z = 395.5 mm`) to the rear occiput/nape (`Z = 411.5 mm`),
           eliminating the downward-hanging spherical belly of the legacy head.
         - Elongated horizontal ovoid cranium (`Z_top = 500.0 mm`, rear occiput reaching `Y = +49.0 mm` down to `Z = 426 mm`).
         - Distinct vertical-ish recessed eye visor (`Z in [435, 475.5] mm, Y ~ -28.2 mm`) behind the brow ridge.
         - Plump droplet/pillow monkey muzzle (`Z in [395.5, 445.5] mm`) peaking at the upper lip (`Z = 419.5 mm, Y = -45.2 mm`)
           with a cute upturned button-nose step (`Z = 438.5 mm`) and a broad rounded chin (`Z = 395.5 mm`).
      2. Front View Profile (`d872a43a.png`):
         - Non-circular squircle/ovoid cranium silhouette (`width = 109.5 mm`) with flatter vertical temple/cheek sides.
         - Classic 2-lobe monkey face mask (`wide upper visor X in [-39.8, +39.8] mm` stepping inward at `Z = 435 mm`
           to the `56 mm` wide U-cushion muzzle `X in [-28.0, +28.0] mm`) and arched nose bridge (`Z = 445.5 mm` at `X = 0`).
         - Warm-dark glossy dome eyes (`R = 8.3 mm` at `X = +-21.8, Z = 455.8 mm`) with upper-left specular glints,
           crisp forehead helmet-cap seam, and horizontal smile crease."""
    import manifold3d as md

    def _sig(x: float) -> float:
        if x > 30.0:
            return 1.0
        if x < -30.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _smin(a: float, b: float, k: float = 0.10) -> float:
        h = max(k - abs(a - b), 0.0) / k
        return min(a, b) - h * h * k * 0.25

    n_u = 168
    n_v = 296
    u_grid = np.linspace(0.0, 1.0, n_u)
    v_grid = np.linspace(-np.pi, np.pi, n_v, endpoint=False)

    def _eval_surface(is_inner: bool, include_fine_grooves: bool = False) -> np.ndarray:
        pts = np.zeros((n_u, n_v, 3), dtype=np.float64)
        wall = 1.65 if is_inner else 0.0

        for iu, u in enumerate(u_grid):
            phi = u * np.pi
            c_phi = math.cos(phi)
            s_phi = math.sin(phi)

            for iv, v in enumerate(v_grid):
                cv = math.cos(v)
                sv = math.sin(v)

                pole_w = s_phi ** 0.65
                z_bot_v = 403.5 + pole_w * (8.0 * sv - 0.5 * (cv * cv))
                z_top_v = 500.0 + pole_w * (0.8 * sv - 0.4 * (cv * cv))
                if is_inner:
                    z_bot_v += wall
                    z_top_v -= wall

                zc_v = 0.5 * (z_top_v + z_bot_v)
                hz_v = 0.5 * (z_top_v - z_bot_v)
                vert_t = -math.copysign(abs(c_phi) ** 0.94, c_phi)
                z0 = zc_v + hz_v * vert_t
                h = vert_t

                if s_phi <= 1e-6:
                    rad_env = 0.0
                elif h >= 0.0:
                    rad_env = (max(0.0, 1.0 - (h ** 2.28))) ** 0.46
                else:
                    p_low = 3.60 + 0.55 * max(0.0, sv)
                    rad_env = (max(0.0, 1.0 - (abs(h) ** p_low))) ** (0.41 - 0.04 * max(0.0, sv))

                cheek_taper = 1.0 - 0.14 * _sig((416.0 - z0) / 6.5)
                rx = 54.4 * cheek_taper * rad_env

                occiput_swell = 1.0 + 0.09 * math.exp(-0.5 * ((z0 - 452.0) / 24.0) ** 2)
                ry_back = 45.6 * occiput_swell * rad_env

                forehead_swell = 1.0 + 0.10 * math.exp(-0.5 * ((z0 - 472.0) / 16.0) ** 2)
                jaw_front_tuck = 1.0 - 0.12 * _sig((410.0 - z0) / 6.5)
                ry_front = 36.2 * forehead_swell * jaw_front_tuck * rad_env

                corner_boost = 1.0 + (0.145 if sv < 0.0 else 0.095) * (cv * cv) * (sv * sv) * 4.0 * _sig((z0 - 416.0) / 4.5)

                if is_inner:
                    rx = max(0.0, rx - wall)
                    ry_back = max(0.0, ry_back - wall)
                    ry_front = max(0.0, ry_front - wall)

                x0 = rx * cv * corner_boost
                y_hemi = (ry_back * sv if sv >= 0.0 else ry_front * sv) * corner_boost
                yc_shift = 3.0 - 3.0 * _sig((434.0 - z0) / 10.0)
                y0 = yc_shift + y_hemi

                if sv < -0.04 and s_phi > 0.02:
                    front_w = max(0.0, -sv) ** 0.75

                    z_brow_x = 475.5 - 2.2 * math.exp(-0.5 * (x0 / 11.5) ** 2) - 0.0030 * (x0 ** 2)
                    z_cheek_x = 434.8 + 0.0020 * (x0 ** 2)
                    zc_eye = 0.5 * (z_brow_x + z_cheek_x)
                    hz_eye = max(6.0, 0.5 * (z_brow_x - z_cheek_x))
                    dx_e = abs(x0) / 39.8
                    dz_e = abs(z0 - zc_eye) / hz_eye
                    d_visor = (dx_e ** 3.5 + dz_e ** 3.3) ** (1.0 / 3.4)

                    z_muz_top = 436.5 + 9.0 * math.exp(-0.5 * (x0 / 15.5) ** 2) - 0.0035 * (x0 ** 2)
                    z_muz_bot = 395.5 + 1.5 * ((x0 / 28.0) ** 2)
                    zc_muz = 0.5 * (z_muz_top + z_muz_bot)
                    hz_muz = max(6.0, 0.5 * (z_muz_top - z_muz_bot))
                    wx_muz = 28.0 - 1.2 * _sig((z0 - 433.0) / 4.0)
                    dx_m = abs(x0) / wx_muz
                    dz_m = abs(z0 - zc_muz) / hz_muz
                    d_muzzle = (dx_m ** 3.1 + dz_m ** 3.0) ** (1.0 / 3.05)

                    d_mask = _smin(d_visor, d_muzzle, k=0.08)

                    in_visor = _sig((1.0 - d_visor) * 12.5) * _sig((z0 - z_muz_top + 0.8) / 1.6)
                    y_visor_target = -28.2 + 2.0 * ((x0 / 39.8) ** 2) + (wall if is_inner else 0.0)
                    y0 = y0 * (1.0 - 0.76 * in_visor * front_w) + y_visor_target * (0.76 * in_visor * front_w)

                    if d_muzzle < 1.35:
                        in_muz = _sig((1.01 - d_muzzle) * 10.0) * front_w
                        dz_lip = (z0 - 419.5) / 20.5
                        y_center_z = -45.2 + 7.4 * (dz_lip ** 2) + 2.0 * (dz_lip ** 3)
                        y_center_z -= 1.5 * math.exp(-0.5 * ((z0 - 438.5) / 3.6) ** 2) * math.exp(-0.5 * (x0 / 11.0) ** 2)
                        y_muz_target = y_center_z + 9.8 * (abs(x0 / (wx_muz + 1.2)) ** 2.2) + (wall if is_inner else 0.0)
                        y0 = y0 * (1.0 - in_muz) + min(y0, y_muz_target) * in_muz

                    if not is_inner and d_mask < 1.35:
                        brow_w = _sig((z0 - 433.0) / 3.0) * (1.0 + 0.45 * _sig((z0 - 462.0) / 5.0))
                        y0 -= 1.50 * brow_w * math.exp(-0.5 * ((d_mask - 1.05) / 0.08) ** 2) * front_w

                    d_sock = math.hypot(abs(x0) - 20.9, z0 - 456.0)
                    if not is_inner and d_sock < 12.5:
                        y0 -= 0.95 * math.exp(-0.5 * ((d_sock - 8.7) / 1.2) ** 2) * front_w

                    if not is_inner and include_fine_grooves:
                        if z0 < 437.5 and d_muzzle < 1.25:
                            crease = 1.40 * math.exp(-0.5 * ((d_muzzle - 1.00) / 0.052) ** 2) * _sig((436.5 - z0) / 1.8) * front_w
                            y0 += crease
                        if abs(x0) < 27.0 and abs(z0 - z_muz_top) < 2.2:
                            nb_groove = 0.85 * math.exp(-0.5 * ((z0 - z_muz_top) / 0.65) ** 2) * front_w
                            y0 += nb_groove

                        if abs(x0) < 19.8 and 407.5 < z0 < 418.0:
                            z_sm = 411.5 + 0.0042 * (x0 ** 2)
                            d_sm = abs(z0 - z_sm)
                            if d_sm < 1.5:
                                w_x = (1.0 - (abs(x0) / 19.8) ** 2) ** 0.75
                                y0 += 1.35 * math.exp(-0.5 * (d_sm / 0.42) ** 2) * w_x * front_w

                        if z0 > 444.0 and 1.05 < d_visor < 1.32:
                            cap_groove = 0.78 * math.exp(-0.5 * ((d_visor - 1.18) / 0.042) ** 2) * _sig((z0 - 445.0) / 2.0) * front_w
                            y0 += cap_groove
                        if z0 > 481.5 and abs(x0) < 1.4:
                            y0 += 0.62 * math.exp(-0.5 * (x0 / 0.38) ** 2) * _sig((z0 - 482.0) / 1.0) * front_w

                if not is_inner and include_fine_grooves and abs(cv) > 0.45:
                    side_w = ((abs(cv) - 0.45) / 0.55) ** 1.3
                    d_ear = math.hypot((y0 - 4.5) / 14.5, (z0 - 442.0) / 19.5)
                    if d_ear < 1.35:
                        ear_groove = -0.75 * math.exp(-0.5 * ((d_ear - 1.0) / 0.055) ** 2) * side_w
                        ear_flat = -0.35 * _sig((0.95 - d_ear) * 8.0) * side_w
                        x0 += math.copysign(ear_groove + ear_flat, x0)
                    if 404.0 < z0 < 423.0:
                        y_seam = -4.0 + 5.0 * ((z0 - 404.0) / 19.0)
                        d_jseam = abs(y0 - y_seam)
                        if d_jseam < 1.5:
                            x0 -= math.copysign(0.50 * math.exp(-0.5 * (d_jseam / 0.42) ** 2) * side_w, x0)

                pts[iu, iv] = [x0, y0, z0]
        return pts

    p_out_base = _eval_surface(is_inner=False, include_fine_grooves=False)
    p_out_grooves = _eval_surface(is_inner=False, include_fine_grooves=True)
    p_in = _eval_surface(is_inner=True, include_fine_grooves=False)

    def _pts_to_trimesh(pts: np.ndarray) -> trimesh.Trimesh:
        v_list = [pts[0, :, :].mean(axis=0)]
        for iu in range(1, n_u - 1):
            for iv in range(n_v):
                v_list.append(pts[iu, iv])
        north_idx = len(v_list)
        v_list.append(pts[-1, :, :].mean(axis=0))
        v_arr = np.asarray(v_list, dtype=np.float64)

        f_list = []
        for iv in range(n_v):
            iv_n = (iv + 1) % n_v
            f_list.append([0, 1 + iv_n, 1 + iv])
        for iu in range(1, n_u - 2):
            r0 = 1 + (iu - 1) * n_v
            r1 = 1 + iu * n_v
            for iv in range(n_v):
                iv_n = (iv + 1) % n_v
                f_list.append([r0 + iv, r0 + iv_n, r1 + iv_n])
                f_list.append([r0 + iv, r1 + iv_n, r1 + iv])
        r_last = 1 + (n_u - 3) * n_v
        for iv in range(n_v):
            iv_n = (iv + 1) % n_v
            f_list.append([r_last + iv, r_last + iv_n, north_idx])

        return trimesh.Trimesh(vertices=v_arr, faces=np.asarray(f_list, dtype=np.int64), process=False)

    tm_out_base = _pts_to_trimesh(p_out_base)
    tm_out_grooves = _pts_to_trimesh(p_out_grooves)
    groove_delta = np.asarray(tm_out_grooves.vertices, dtype=np.float64) - np.asarray(tm_out_base.vertices, dtype=np.float64)
    trimesh.smoothing.filter_taubin(tm_out_base, lamb=0.40, nu=-0.42, iterations=3)
    tm_out_base.vertices = np.asarray(tm_out_base.vertices, dtype=np.float64) + groove_delta
    trimesh.smoothing.filter_taubin(tm_out_base, lamb=0.32, nu=-0.34, iterations=1)
    mf_out = md.Manifold(md.Mesh64(np.asarray(tm_out_base.vertices, dtype=np.float64), np.asarray(tm_out_base.faces, dtype=np.uint64)))

    tm_in = _pts_to_trimesh(p_in)
    trimesh.smoothing.filter_taubin(tm_in, lamb=0.40, nu=-0.42, iterations=3)
    mf_in = md.Manifold(md.Mesh64(np.asarray(tm_in.vertices, dtype=np.float64), np.asarray(tm_in.faces, dtype=np.uint64)))

    mf_shell = mf_out - mf_in

    for sx in (-1.0, 1.0):
        eye_cyl = md.Manifold.cylinder(45.0, 7.8, 7.8, 64).rotate([-90.0, 0.0, 0.0]).translate([sx * 20.9, -60.0, 456.0])
        mf_shell = mf_shell - eye_cyl
        collar_out = md.Manifold.cylinder(11.0, 9.8, 9.8, 64).rotate([-90.0, 0.0, 0.0]).translate([sx * 20.9, -36.5, 456.0])
        collar_in = md.Manifold.cylinder(13.0, 7.8, 7.8, 64).rotate([-90.0, 0.0, 0.0]).translate([sx * 20.9, -37.5, 456.0])
        mf_shell = mf_shell + ((collar_out - collar_in) ^ mf_out)

    # Compact, drawing-matched tapered neck collar (`head_neck_collar`, Z in [394.5, 413.0] mm) with base flange
    nc_out = md.Manifold.cylinder(18.5, 29.0, 28.5, 64).scale([1.04, 0.94, 1.0]).translate([-0.8, 3.5, 394.5])
    nc_flange = md.Manifold.cylinder(3.2, 32.2, 29.0, 64).scale([1.04, 0.94, 1.0]).translate([-0.8, 3.5, 394.2])
    nc_in = md.Manifold.cylinder(21.0, 26.4, 26.8, 64).scale([1.04, 0.94, 1.0]).translate([-0.8, 3.5, 393.0])
    mf_collar = ((nc_out + nc_flange) - nc_in) - mf_out

    neck_cyl = md.Manifold.cylinder(28.0, 26.0, 26.0, 48).scale([1.04, 0.94, 1.0]).translate([-0.8, 3.5, 384.0])
    mf_shell = mf_shell - neck_cyl

    def _to_tm(mf: md.Manifold) -> trimesh.Trimesh:
        r = mf.to_mesh()
        return trimesh.Trimesh(
            vertices=np.asarray(r.vert_properties, dtype=np.float64)[:, :3].copy(),
            faces=np.asarray(r.tri_verts, dtype=np.int64).copy(),
            process=False,
        )

    return _to_tm(mf_shell), _to_tm(mf_collar)


def build_compact_geared_hands(scale: float = 0.50) -> tuple[dict[str, Solid], dict[str, tuple[trimesh.Trimesh, str, str, list[int]]]]:
    """Build the decided Symmetric Geared Gripper (`cove50-hand-symmetric-geared-r1`, M-qxw97p)
    scaled by `scale=0.50` around the wrist mating center `(u=2.0, v=0.0, d=-42.0)` so the hands fit
    proportionally (`42.1 x 39.0 x 53.8 mm`, `~41 g/hand`) under the forearm `Z = 171.0 mm` 4-bolt wrist deck
    with exact unscaled M3 wrist bolt bores at `X = ±128.0, ±135.0 mm, Y = -3.0, +5.0 mm`."""
    def _tf_right_brep(shape: Solid) -> Solid:
        t_s = gp_Trsf()
        t_s.SetScale(gp_Pnt(2.0, 0.0, -42.0), float(scale))
        sh1 = BRepBuilderAPI_Transform(shape.wrapped, t_s, True).Shape()
        t_r = gp_Trsf()
        t_r.SetValues(
            1.0,  0.0,  0.0, 129.5,
            0.0, -1.0,  0.0,   1.0,
            0.0,  0.0, -1.0, 128.95,
        )
        return largest_solid(Compound(BRepBuilderAPI_Transform(sh1, t_r, True).Shape()))

    palm_low_raw = import_brep(ensure(ROOT / "artifacts/cove50-engineering-competition/hand-radical-20261026/A/twin-crawl-wrist-seats-r1/palm_lower.brep")).solids()[0]
    palm_up_raw = import_brep(ensure(ROOT / "artifacts/cove50-engineering-competition/hand-radical-20261026/A/twin-gear-r0/palm_upper.brep")).solids()[0]
    r_fing_raw = import_brep(ensure(ROOT / "artifacts/cove50-engineering-competition/hand-radical-20261026/A/twin-crawl-r1/right_finger.brep")).solids()[0]
    l_fing_raw = import_brep(ensure(ROOT / "artifacts/cove50-engineering-competition/hand-radical-20261026/A/twin-crawl-r1/left_finger.brep")).solids()[0]

    # Trim the legacy sheet-metal strap sleeve (`d < -42.0`) above the flat wrist seating plane `d = -42.0`
    keep_box = Pos(0.0, 0.0, 14.0) * Box(120.0, 120.0, 112.0)
    palm_low_clipped = _intersect_solid(palm_low_raw, keep_box)

    r_palm_low = _tf_right_brep(palm_low_clipped)
    r_palm_up = _tf_right_brep(palm_up_raw)
    r_rfing = _tf_right_brep(r_fing_raw)
    r_lfing = _tf_right_brep(l_fing_raw)

    # Fuse an internal 4-bolt wrist mounting boss at Z in [164.0, 170.95] and drill the 4 unscaled M3 wrist bolt bores + counterbores
    wrist_pad = Pos(131.5, 1.0, 167.475) * Box(13.0, 13.0, 6.95)
    qf = BRepAlgoAPI_Fuse(r_palm_low.wrapped, wrist_pad.wrapped)
    qf.Build()
    r_palm_low = largest_solid(Compound(qf.Shape()))

    wrist_cutters = []
    r_wrist_screws: dict[str, Solid] = {}
    idx = 1
    for bx in (128.0, 135.0):
        for by in (-3.0, 5.0):
            wrist_cutters.append(occ_cyl((bx, by, 155.0), (bx, by, 172.0), 1.65))
            wrist_cutters.append(occ_cyl((bx, by, 150.0), (bx, by, 165.5), 2.90))
            r_wrist_screws[f"R_wrist_M3x8_{idx}"] = _make_m3x8_screw((bx, by, 165.5), (0.0, 0.0, -1.0))
            idx += 1
    r_palm_low = largest_solid(cut_all(r_palm_low, wrist_cutters))

    printed_hand_solids = {
        "R_palm_lower": r_palm_low,
        "R_palm_upper": r_palm_up,
        "R_right_finger": r_rfing,
        "R_left_finger": r_lfing,
        "L_palm_lower": mirror_x(r_palm_low),
        "L_palm_upper": mirror_x(r_palm_up),
        "L_right_finger": mirror_x(r_rfing),
        "L_left_finger": mirror_x(r_lfing),
    }

    # Extract and transform internal hand hardware (motor case, horn, 4 bushings) + 4 wrist M3x8 screws per hand
    hand_glb = trimesh.load(ensure(ROOT / "artifacts/cove50-engineering-competition/hand-radical-20261026/lead-twin-crawl-r1/cad/assembly.glb"), force="scene")
    hw_nodes = [
        ("OEM_case", "forearm", "motor", [64, 72, 82, 255]),
        ("OEM_horn", "right_finger", "horn", [182, 190, 198, 255]),
        ("driven_front_bush", "forearm", "bearing", [195, 202, 210, 255]),
        ("right_rear_bush", "forearm", "bearing", [195, 202, 210, 255]),
        ("left_rear_bush", "forearm", "bearing", [195, 202, 210, 255]),
        ("left_front_bush", "forearm", "bearing", [195, 202, 210, 255]),
    ]
    hand_hw_meshes: dict[str, tuple[trimesh.Trimesh, str, str, list[int]]] = {}
    p0 = np.array([2.0, 0.0, -42.0], dtype=np.float64)
    for node_nm, lk_suffix, role_str, rgba in hw_nodes:
        T_node, g_key = hand_glb.graph[node_nm]
        m_uvd = hand_glb.geometry[g_key].copy().apply_transform(T_node)
        v_uvd = np.asarray(m_uvd.vertices, dtype=np.float64) * 1000.0
        v_s = p0 + float(scale) * (v_uvd - p0)
        v_r = np.column_stack([129.5 + v_s[:, 0], 1.0 - v_s[:, 1], 128.95 - v_s[:, 2]])
        f_r = np.asarray(m_uvd.faces, dtype=np.int64).copy()
        m_r = trimesh.Trimesh(vertices=v_r, faces=f_r, process=False)
        hand_hw_meshes[f"R_hand_{node_nm}"] = (m_r, f"R_{lk_suffix}", role_str, rgba)

        v_l = v_r.copy()
        v_l[:, 0] *= -1.0
        f_l = f_r[:, [0, 2, 1]].copy()
        m_l = trimesh.Trimesh(vertices=v_l, faces=f_l, process=False)
        hand_hw_meshes[f"L_hand_{node_nm}"] = (m_l, f"L_{lk_suffix}", role_str, rgba)

    solid_to_watertight_arrays = _get_solid_to_watertight_arrays()
    for scr_nm, scr_sol in r_wrist_screws.items():
        v_s, f_s = solid_to_watertight_arrays(scr_sol, tol=0.05)
        hand_hw_meshes[scr_nm] = (trimesh.Trimesh(vertices=v_s, faces=f_s, process=False), "R_forearm", "fastener", [205, 212, 220, 255])
        l_scr_nm = "L_" + scr_nm[2:]
        v_l = v_s.copy()
        v_l[:, 0] *= -1.0
        f_l = f_s[:, [0, 2, 1]].copy()
        hand_hw_meshes[l_scr_nm] = (trimesh.Trimesh(vertices=v_l, faces=f_l, process=False), "L_forearm", "fastener", [205, 212, 220, 255])

    return printed_hand_solids, hand_hw_meshes


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_json(path: Path) -> dict | list:
    return json.loads(ensure(path).read_text())


def rot_mat(axis, deg: float) -> np.ndarray:
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def fk_transforms(joints: list[dict], controls_q: dict[str, float]) -> dict[str, np.ndarray]:
    """Compute 4x4 world-frame rigid transforms (in mm) for every link in the 26-joint tree."""
    T = {"torso": np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        for j in list(pending):
            if j["parent"] in T:
                spec = j.get("driven_by")
                if spec and "linear" in spec:
                    ang = sum(float(controls_q.get(k, 0.0)) * float(gain) for k, gain in spec["linear"].items())
                else:
                    ang = float(controls_q.get(j["name"], 0.0))
                R = rot_mat(j["axis_world_neutral"], ang)
                p = np.asarray(j["origin_world_mm"], dtype=float)
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = p - R @ p
                T[j["child"]] = T[j["parent"]] @ M
                pending.remove(j)
                progressed = True
        if not progressed:
            break
    return T


def transform_brep(shape: Solid | Compound, M: np.ndarray) -> Compound:
    if np.allclose(M, np.eye(4)):
        return shape
    t = gp_Trsf()
    t.SetValues(
        float(M[0, 0]), float(M[0, 1]), float(M[0, 2]), float(M[0, 3]),
        float(M[1, 0]), float(M[1, 1]), float(M[1, 2]), float(M[1, 3]),
        float(M[2, 0]), float(M[2, 1]), float(M[2, 2]), float(M[2, 3]),
    )
    return Compound(BRepBuilderAPI_Transform(shape.wrapped, t, True).Shape())


def brep_common_volume_mm3(a: Solid | Compound, b: Solid | Compound) -> float:
    q = BRepAlgoAPI_Common(a.wrapped, b.wrapped)
    q.Build()
    if not q.IsDone():
        return 0.0
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(q.Shape(), g, 1e-9, True, False)
    return abs(float(g.Mass()))


def load_mesh_mm(mesh_rel: str) -> trimesh.Trimesh:
    p = ROOT / mesh_rel
    if not p.exists():
        try:
            p = ensure(p)
        except Exception:
            # Handle bilateral mirrored L_ mesh stored only as R_
            r_name = p.name.replace("__L_", "__R_")
            if r_name == p.name and p.name.startswith("L_"):
                r_name = "R_" + p.name[2:]
            r_p = ensure(p.parent / r_name)
            if r_p.suffix == ".npz":
                z = np.load(r_p)
                v = np.asarray(z["vertices"], dtype=np.float64).copy()
                v[:, 0] *= -1.0
                f = np.asarray(z["faces"], dtype=np.int64)[:, [0, 2, 1]].copy()
                return trimesh.Trimesh(vertices=v, faces=f, process=False)
            tm = trimesh.load(r_p, force="mesh", process=False)
            v = np.asarray(tm.vertices, dtype=np.float64).copy()
            v[:, 0] *= -1.0
            f = np.asarray(tm.faces, dtype=np.int64)[:, [0, 2, 1]].copy()
            return trimesh.Trimesh(vertices=v, faces=f, process=False)

    if p.suffix == ".npz":
        z = np.load(p)
        return trimesh.Trimesh(
            vertices=np.asarray(z["vertices"], dtype=np.float64),
            faces=np.asarray(z["faces"], dtype=np.int64),
            process=False,
        )
    return trimesh.load(p, force="mesh", process=False)


def save_solid_step_and_npz(solid: Solid | Compound, step_path: Path, npz_path: Path, tol: float = 0.05) -> trimesh.Trimesh:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    solid_to_watertight_arrays = _get_solid_to_watertight_arrays()
    sols = list(solid.solids())
    export_step(Compound(children=sols), str(step_path))
    if len(sols) == 1:
        try:
            v, f = solid_to_watertight_arrays(sols[0], tol=tol)
        except AssertionError:
            tmp_stl = npz_path.with_suffix(".tmp.stl")
            export_stl(Compound(children=sols), str(tmp_stl), tolerance=0.04, angular_tolerance=0.1)
            tm = trimesh.load(tmp_stl, force="mesh", process=True)
            tmp_stl.unlink(missing_ok=True)
            v = np.asarray(tm.vertices, dtype=np.float64)
            f = np.asarray(tm.faces, dtype=np.int64)
    else:
        sub_meshes = []
        for s_item in sols:
            try:
                vs, fs = solid_to_watertight_arrays(s_item, tol=tol)
                sub_meshes.append(trimesh.Trimesh(vertices=vs, faces=fs, process=False))
            except AssertionError:
                tmp_stl = npz_path.with_suffix(".tmp.stl")
                export_stl(Compound(children=[s_item]), str(tmp_stl), tolerance=0.04, angular_tolerance=0.1)
                tm = trimesh.load(tmp_stl, force="mesh", process=True)
                tmp_stl.unlink(missing_ok=True)
                sub_meshes.append(tm)
        tm_all = trimesh.util.concatenate(sub_meshes)
        v = np.asarray(tm_all.vertices, dtype=np.float64)
        f = np.asarray(tm_all.faces, dtype=np.int64)
    np.savez_compressed(npz_path, vertices=v, faces=f)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def raster_meshes(
    items: list[tuple[str, trimesh.Trimesh, list[int]]],
    view,
    box=None,
    W: int = 520,
    H: int = 820,
) -> Image.Image:
    if not EXE_RENDERER.exists():
        EXE_RENDERER.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["clang++", "-std=c++17", "-O3", str(CPP_RENDERER), "-o", str(EXE_RENDERER)], check=True)
    view = np.asarray(view, dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross([0.0, 0.0, 1.0], view)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= np.linalg.norm(right)
    up = np.cross(view, right)

    tris = []
    colors = []
    light_dir = np.array([0.32, -0.52, 0.79], dtype=float)
    light_dir /= np.linalg.norm(light_dir)
    for _, m, c in items:
        if len(m.faces) == 0:
            continue
        tri = m.triangles
        c_rgb = np.asarray(c[:3], dtype=float) / 255.0
        shade = 0.64 + 0.36 * np.abs(m.face_normals @ light_dir)
        tris.append(tri)
        colors.append(c_rgb * shade[:, None])
    tri = np.concatenate(tris, axis=0)
    col = np.concatenate(colors, axis=0)
    xy = np.stack([tri @ right, tri @ up], axis=-1)
    if box is None:
        lo = xy.min((0, 1))
        hi = xy.max((0, 1))
    else:
        pts = trimesh.bounds.corners(np.asarray(box, dtype=float))
        proj = np.stack([pts @ right, pts @ up], axis=-1)
        lo = proj.min(0)
        hi = proj.max(0)
    scale = min(W / max(hi[0] - lo[0], 1e-3), H / max(hi[1] - lo[1], 1e-3)) * 0.90
    middle = 0.5 * (lo + hi)
    xyz = np.empty_like(tri)
    xyz[:, :, 0] = (xy[:, :, 0] - middle[0]) * scale + W * 0.5
    xyz[:, :, 1] = H * 0.5 - (xy[:, :, 1] - middle[1]) * scale
    xyz[:, :, 2] = tri @ view
    data = np.concatenate([xyz.reshape(-1, 9), col], axis=1).astype("<f4")
    target = ART_DIR / f"tmp-render-{os.getpid()}.ppm"
    subprocess.run(
        [str(EXE_RENDERER), "/dev/stdin", str(target)],
        input=struct.pack("<III", W, H, len(tri)) + data.tobytes(),
        check=True,
    )
    im = Image.open(target).copy()
    target.unlink(missing_ok=True)
    return im


# Distinct, harmonious subsystem & stage color palette
CFG = BOM_CONFIG.section("assemble-complete-fdm-body")   # configured values: robot/body/assembly/assemble_complete_fdm_body.bom.yaml config:
PALETTE = BOM_CONFIG.unfrozen(CFG["PALETTE"])


# Replaced legacy sheet-metal / LPBF / PA12 carrier sets across all 4 structural redesigns
REPLACED_LEGACY_PARTS = set(CFG["REPLACED_LEGACY_PARTS"])


def classify_overlay_and_group(name: str, role: str, default_dg: str) -> tuple[str, str | None, bool]:
    """Return `(display_group, overlay_id, is_shell)` for a part in the Complete FDM Ape Body."""
    if name in ("C_T01_chest__T02_back_1_M3_hex_nut", "C_T01_chest__T02_back_2_M3_hex_nut"):
        return ("ape_body_shell", None, True)

    # Captive nuts stay with the mechanical skeleton / hardware so they are solid brass, never translucent shell
    if "nut" in name.lower():
        return ("mechanism", "actuators_and_hardware", False)

    if (
        default_dg == "ape_body_shell"
        or role == "body_shell"
        or "hard_cover" in name
        or "boot_shell" in name
        or "sole_shell" in name
        or "flex_envelope" in name
        or "_cuff" in name
        or "smooth_paw" in name
        or name in (
            "center_front_shell",
            "center_back_shell",
            "ape_body_shell__chest_indicator_relief4_r33",
            "upper__simple_continuous_rear_neck_cowl",
            "original_profile_head_skin_v23",
            "flush_skin_M2x8_-1",
            "flush_skin_M2x8_1",
            "C_T01_chest__T02_back_1_M3x8",
            "C_T01_chest__T02_back_2_M3x8",
            "T02_back_mount2_M3x8",
            "T02_back_mount1_M3x6",
            "T02_back_mount3_M3x6",
            "center_back_low_M3x8",
            "center_front_low_L_M2x10",
            "center_front_low_R_M2x10",
            "upper__cowl_M2x10_countersunk_0",
            "upper__cowl_M2x10_countersunk_1",
        )
        or "carrier_shell_" in name
        or ("_fore_" in name and "_M3x8_" in name)
        or ("_upper_arm_shell_" in name and "_M3x8" in name)
    ):
        return ("ape_body_shell", None, True)

    if role in ("printed_structural_connector", "printed_structural_carrier", "compression_limit_sleeves"):
        return ("mechanism", None, False)

    # Electronics, battery cells, fins, sensors, compute, cameras, harnesses
    elec_keywords = (
        "Honeycomb_Cell_",
        "Honeycomb_Fin_",
        "Power_supervisor",
        "Main_precharge",
        "Main_H",
        "body_IMU",
        "foot_ADC",
        "IMU_M2",
        "ADC_M2",
        "B0410",
        "B0196",
        "microphone_6049",
        "mic_",
        "ROCK4D",
        "Radian_",
        "SameSky_",
        "TIM_3M8810",
        "Wuerth_",
        "compute_",
        "fan_",
        "heatsink_",
        "horizontal_RF",
        "vertical_RF",
        "combined_RF",
        "Pololu_D24V50F5",
        "regulator_",
        "regulated_",
        "positive_solder_bus",
        "return_solder_bus",
        "GPIO",
        "12V_head_",
        "sensor_pocket_pad",
        "sole_pad_bond",
        "toe_lateral",
        "heel_lateral",
        "mid_medial",
        "B13_short_TPU_pad",
    )
    if any(k in name for k in elec_keywords):
        return ("mechanism", "electronics_and_battery", False)

    # Otherwise it is an actuator case, horn, passive idler, bushing, or fastener
    return ("mechanism", "actuators_and_hardware", False)


def build_complete_assembly() -> dict:
    t0 = time.monotonic()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all OpenCASCADE BRep fillet / boolean builders BEFORE any manifold3d import occurs
    thigh_overrides = build_contoured_thigh_and_covers()
    shoulder_overrides = build_redesigned_shoulder_and_upper_arm()
    printed_hand_solids, hand_hw_meshes = build_compact_geared_hands(scale=0.50)

    torso_man = read_json(ROOT / "artifacts/robot/body/torso/combined/assembly-parts.json")
    torso_cad = read_json(ROOT / "artifacts/robot/body/torso/combined/manifest.json")
    neck_man = read_json(ROOT / "artifacts/robot/body/neck-head/combined/assembly-parts.json")
    neck_cad = read_json(ROOT / "artifacts/robot/body/neck-head/combined/manifest.json")
    arm_man = read_json(ROOT / "artifacts/robot/body/arm/combined/assembly-parts.json")
    arm_cad = read_json(ROOT / "artifacts/robot/body/arm/combined/manifest.json")
    leg_man = read_json(ROOT / "artifacts/robot/body/leg/complete_leg_hip_combined/parts.json")
    leg_cad = read_json(ROOT / "artifacts/robot/body/leg/complete_leg_hip_combined/cad-manifest.json")
    ec_man = read_json(ROOT / "artifacts/cove50-electronics-integration/r1-20261026/assembly-parts.json")

    scene = trimesh.Scene()
    scene.graph.update(frame_to="complete_fdm_ape_body_assembly", matrix=np.eye(4))

    assembly_parts: list[dict] = []
    seen_names: set[str] = set()
    printed_brep_solids: list[dict] = []
    skeleton_compound_solids: list[Solid] = []

    render_all_with_shell: list[tuple[str, trimesh.Trimesh, list[int]]] = []
    render_no_shell: list[tuple[str, trimesh.Trimesh, list[int]]] = []
    render_pure_skeleton: list[tuple[str, trimesh.Trimesh, list[int]]] = []
    meshes_by_name_mm: dict[str, tuple[trimesh.Trimesh, str, list[int], bool, str | None]] = {}

    def _register_part(
        name: str,
        owning_link: str,
        role: str,
        m_mm: trimesh.Trimesh,
        rgba: list[int],
        meta: dict,
    ) -> None:
        if name in seen_names or name in REPLACED_LEGACY_PARTS or name.startswith("full_shell__"):
            return
        seen_names.add(name)

        dg, ov_id, is_shell = classify_overlay_and_group(name, role, meta.get("display_group", "mechanism"))
        is_shell_fastener = is_shell and ("M3x" in name or "M2x" in name)
        if "nut" in name.lower():
            rgba = PALETTE["captive_nut"]
        elif is_shell_fastener:
            rgba = PALETTE["fastener"]
        elif "case" in name and not any(k in name for k in ("PA2", "washer", "spacer", "tab")):
            rgba = PALETTE["actuator_case"]

        entry = copy.deepcopy(meta)
        entry["name"] = name
        entry["owning_link"] = owning_link
        entry["role"] = ("fastener" if is_shell_fastener else "body_shell") if is_shell else role
        entry["color"] = rgba
        entry["display_group"] = dg
        entry["display_groups"] = [dg]
        entry["parent_assembly"] = "complete_fdm_ape_body_assembly"
        entry["inspection_reference"] = is_shell
        entry["bounds_mm"] = [m_mm.bounds[0].round(3).tolist(), m_mm.bounds[1].round(3).tolist()]
        if is_shell:
            entry["display_opacity"] = 1.0
            entry.pop("overlay_id", None)
        elif ov_id is not None:
            entry["overlay_id"] = ov_id
            entry.pop("display_opacity", None)
        else:
            entry.pop("overlay_id", None)
            entry.pop("display_opacity", None)

        m_m = m_mm.copy()
        m_m.apply_scale(0.001)
        m_m.visual.face_colors = np.asarray(rgba, dtype=np.uint8)
        scene.add_geometry(
            m_m,
            node_name=name,
            geom_name=name,
            parent_node_name="complete_fdm_ape_body_assembly",
        )
        assembly_parts.append(entry)
        meshes_by_name_mm[name] = (m_mm, owning_link, rgba, is_shell, ov_id)

        render_all_with_shell.append((name, m_mm, rgba))
        if not is_shell:
            render_no_shell.append((name, m_mm, rgba))
            if ov_id is None:
                render_pure_skeleton.append((name, m_mm, rgba))

    def _save_and_update_mesh_mm(pname: str, m_mm: trimesh.Trimesh, p_meta: dict) -> dict:
        out_npz = PARTS_DIR / f"{pname}.npz"
        np.savez_compressed(
            out_npz,
            vertices=np.asarray(m_mm.vertices, dtype=np.float64),
            faces=np.asarray(m_mm.faces, dtype=np.int64),
        )
        meta_out = copy.deepcopy(p_meta)
        meta_out["mesh"] = rel_path(out_npz)
        meta_out["mesh_sha256"] = sha256_file(out_npz)
        return meta_out

    def _transform_flipped_roll_mesh(m_mm: trimesh.Trimesh, side: str) -> trimesh.Trimesh:
        s = 1.0 if side == "R" else -1.0
        th = np.deg2rad(20.0 * s)
        c_t, s_t = float(np.cos(th)), float(np.sin(th))
        rot_orig = np.array([[c_t, 0.0, -s_t], [0.0, 1.0, 0.0], [s_t, 0.0, c_t]])
        rot_flip = rot_orig @ np.diag([-1.0, -1.0, 1.0])
        p_orig = np.array([112.0 * s, 3.0, 360.0])
        p_new = np.array([106.0 * s, 5.05, 360.0])
        v = np.asarray(m_mm.vertices, dtype=np.float64)
        v_new = (v - p_orig) @ rot_orig @ rot_flip.T + p_new
        return trimesh.Trimesh(vertices=v_new, faces=np.asarray(m_mm.faces, dtype=np.int64), process=False)

    # ==========================================================================
    # 1. CENTER-BODY TORSO, PELVIS, 9-CELL BATTERY & ELECTRONICS (`torso` link)
    # ==========================================================================
    for p in torso_man["parts"]:
        pname = p["name"]
        # Rescue the 4+4 shoulder roll case PA2x7 screws from torso_man onto the flipped R/L_shoulder_roll_link
        if pname.startswith(("upper__R_roll_case_PA2x7_", "upper__L_roll_case_PA2x7_")):
            side = "R" if "_R_" in pname else "L"
            m_mm = _transform_flipped_roll_mesh(load_mesh_mm(p["mesh"]), side)
            lk_new = f"{side}_shoulder_roll_link"
            p_meta = _save_and_update_mesh_mm(pname, m_mm, p)
            _register_part(pname, lk_new, p.get("role", "fastener"), m_mm, PALETTE["fastener"], p_meta)
            continue

        if p.get("owning_link") != "torso":
            continue
        if pname in REPLACED_LEGACY_PARTS or pname == "combined__neck_root_cradle":
            continue

        if pname == "combined__stage1_center_torso_spine_chassis":
            s1_sol = shoulder_overrides[pname]
            out_step = PARTS_DIR / f"{pname}.step"
            out_npz = PARTS_DIR / f"{pname}.npz"
            m_mm = save_solid_step_and_npz(s1_sol, out_step, out_npz, tol=0.05)
            skeleton_compound_solids.append(s1_sol)
            vol_mm3 = float(s1_sol.volume)
            poly_g = round(vol_mm3 * 1e-3 * 1.24, 3)
            p_meta = copy.deepcopy(p)
            p_meta["native_step"] = rel_path(out_step)
            p_meta["native_step_sha256"] = sha256_file(out_step)
            p_meta["mesh"] = rel_path(out_npz)
            p_meta["mesh_sha256"] = sha256_file(out_npz)
            p_meta["volume_mm3"] = round(vol_mm3, 2)
            p_meta["mass_kg"] = round(poly_g / 1000.0, 6)
            printed_brep_solids.append({
                "name": pname,
                "subsystem": "Center Torso & Electronics",
                "stage": "stage1_center_torso_spine_chassis",
                "owning_link": "torso",
                "solid": s1_sol,
                "step_path": rel_path(out_step),
                "step_sha256": sha256_file(out_step),
                "mesh_path": rel_path(out_npz),
                "mesh_sha256": sha256_file(out_npz),
                "volume_mm3": round(vol_mm3, 2),
                "polymer_mass_g": poly_g,
                "sleeves_count": 39,
                "sleeves_mass_g": 6.469,
                "print_orientation": "Flat on Z = 266.80 mm Bottom Base (layer normal +Z)",
            })
            _register_part(pname, "torso", "printed_structural_carrier", m_mm, PALETTE[pname], p_meta)
            continue

        if pname == "combined__stage1_center_torso_spine_chassis_steel_sleeves":
            s1_slv = shoulder_overrides[pname]
            out_step = PARTS_DIR / f"{pname}.step"
            out_npz = PARTS_DIR / f"{pname}.npz"
            m_mm = save_solid_step_and_npz(s1_slv, out_step, out_npz, tol=0.05)
            skeleton_compound_solids.extend(s1_slv.solids())
            p_meta = copy.deepcopy(p)
            p_meta["native_step"] = rel_path(out_step)
            p_meta["native_step_sha256"] = sha256_file(out_step)
            p_meta["mesh"] = rel_path(out_npz)
            p_meta["mesh_sha256"] = sha256_file(out_npz)
            _register_part(pname, "torso", "compression_limit_sleeves", m_mm, PALETTE["sleeve"], p_meta)
            continue

        if p.get("role") == "printed_structural_carrier":
            src_step = ensure(ROOT / p["native_step"])
            sol = import_step(src_step).solids()[0]
            stg_key = pname.replace("combined__", "")
            stg_info = torso_cad["stages"][stg_key]
            if pname == "combined__stage2_lower_pelvis_battery_cradle":
                s2_cuts = []
                for sg in (-1.0, 1.0):
                    s2_cuts.append(occ_box(46.5 if sg > 0 else -56.5, 56.5 if sg > 0 else -46.5, 13.4, 21.0, 263.2, 267.2))
                    s2_cuts.append(occ_box(22.8 if sg > 0 else -31.5, 31.5 if sg > 0 else -22.8, 14.5, 26.0, 211.0, 254.5))
                    s2_cuts.append(occ_box(21.5 if sg > 0 else -24.5, 24.5 if sg > 0 else -21.5, 10.0, 14.5, 194.0, 208.5))
                sol = largest_solid(cut_all(sol, s2_cuts))
                out_step = PARTS_DIR / f"{pname}.step"
                out_npz = PARTS_DIR / f"{pname}.npz"
                m_mm = save_solid_step_and_npz(sol, out_step, out_npz, tol=0.05)
                skeleton_compound_solids.append(sol)
                vol_mm3 = round(float(sol.volume), 2)
                poly_g = round(vol_mm3 * 1e-3 * 1.24, 3)
                p_meta = copy.deepcopy(p)
                p_meta["native_step"] = rel_path(out_step)
                p_meta["native_step_sha256"] = sha256_file(out_step)
                p_meta["mesh"] = rel_path(out_npz)
                p_meta["mesh_sha256"] = sha256_file(out_npz)
                p_meta["volume_mm3"] = vol_mm3
                p_meta["mass_kg"] = round(poly_g / 1000.0, 6)
                printed_brep_solids.append({
                    "name": pname,
                    "subsystem": "Center Torso & Electronics",
                    "stage": stg_key,
                    "owning_link": "torso",
                    "solid": sol,
                    "step_path": rel_path(out_step),
                    "step_sha256": sha256_file(out_step),
                    "mesh_path": rel_path(out_npz),
                    "mesh_sha256": sha256_file(out_npz),
                    "volume_mm3": vol_mm3,
                    "polymer_mass_g": poly_g,
                    "sleeves_count": stg_info["sleeves_count"],
                    "sleeves_mass_g": stg_info["sleeves_mass_g"],
                    "print_orientation": p.get("print_orientation", "Flat on Build Plate"),
                })
                _register_part(pname, "torso", "printed_structural_carrier", m_mm, PALETTE[pname], p_meta)
                continue
            skeleton_compound_solids.append(sol)
            m_mm = load_mesh_mm(p["mesh"])
            printed_brep_solids.append({
                "name": pname,
                "subsystem": "Center Torso & Electronics",
                "stage": stg_key,
                "owning_link": "torso",
                "solid": sol,
                "step_path": p["native_step"],
                "step_sha256": sha256_file(src_step),
                "mesh_path": p["mesh"],
                "mesh_sha256": sha256_file(ensure(ROOT / p["mesh"])),
                "volume_mm3": stg_info["volume_mm3"],
                "polymer_mass_g": stg_info["polymer_mass_paht_cf_g"],
                "sleeves_count": stg_info["sleeves_count"],
                "sleeves_mass_g": stg_info["sleeves_mass_g"],
                "print_orientation": p.get("print_orientation", "Flat on Build Plate"),
            })
            _register_part(pname, "torso", "printed_structural_carrier", m_mm, PALETTE[pname], p)
            continue

        if p.get("role") == "compression_limit_sleeves":
            src_step = ensure(ROOT / p["native_step"])
            skeleton_compound_solids.extend(import_step(src_step).solids())
            m_mm = load_mesh_mm(p["mesh"])
            _register_part(pname, "torso", "compression_limit_sleeves", m_mm, PALETTE["sleeve"], p)
            continue

        if pname in ("center_front_shell", "center_back_shell") and pname in shoulder_overrides:
            sh_sol = shoulder_overrides[pname]
            out_step = PARTS_DIR / f"{pname}.step"
            out_npz = PARTS_DIR / f"{pname}.npz"
            m_mm = save_solid_step_and_npz(sh_sol, out_step, out_npz, tol=0.05)
            p_meta = copy.deepcopy(p)
            p_meta["native_step"] = rel_path(out_step)
            p_meta["native_step_sha256"] = sha256_file(out_step)
            p_meta["mesh"] = rel_path(out_npz)
            p_meta["mesh_sha256"] = sha256_file(out_npz)
            p_meta["volume_mm3"] = round(float(sh_sol.volume), 2)
            _register_part(pname, "torso", p.get("role", "body_shell"), m_mm, PALETTE["shell"], p_meta)
            continue

        m_mm = load_mesh_mm(p["mesh"])
        p_meta = p
        # Shift bilateral torso-mounted Shoulder Pitch actuator cases and fasteners outward by +/-12.0 mm in X
        if pname.startswith(("upper__R_pitch_", "upper__A_R_clocked_pitch_")):
            m_mm = m_mm.copy()
            m_mm.apply_translation([12.0, 0.0, 0.0])
            p_meta = _save_and_update_mesh_mm(pname, m_mm, p)
        elif pname.startswith(("upper__L_pitch_", "upper__A_L_clocked_pitch_")):
            m_mm = m_mm.copy()
            m_mm.apply_translation([-12.0, 0.0, 0.0])
            p_meta = _save_and_update_mesh_mm(pname, m_mm, p)

        rgba = p.get("color", [160, 168, 176, 255])
        if pname == "ape_body_shell__chest_indicator_relief4_r33":
            rgba = PALETTE["chest_indicator"]
        elif pname in ("center_front_shell", "center_back_shell", "upper__simple_continuous_rear_neck_cowl"):
            rgba = PALETTE["shell"]
        elif pname.startswith("Honeycomb_Cell_") and "insulator" not in pname:
            rgba = PALETTE["battery_cell"]
        elif pname.startswith("Honeycomb_Fin_"):
            rgba = PALETTE["battery_fin"]
        _register_part(pname, "torso", p.get("role", "component"), m_mm, rgba, p_meta)

    # ==========================================================================
    # 2. NECK & HEAD (`torso` neck_root_cradle + `neck_yaw` + `head`)
    # ==========================================================================
    for p in neck_man["parts"]:
        pname = p["name"]
        lk = p["owning_link"]
        if pname in REPLACED_LEGACY_PARTS:
            continue
        if lk == "torso" and not pname.startswith("combined__neck_root_cradle"):
            continue

        if p.get("role") == "printed_structural_connector":
            src_step = ensure(ROOT / p["native_step"])
            sol = import_step(src_step).solids()[0]
            skeleton_compound_solids.append(sol)
            m_mm = load_mesh_mm(p["mesh"])
            stg_map = {
                "combined__neck_root_cradle": ("stage1_neck_root_cradle", 6, "XZ Cradle & Spine Frame on Build Plate (layer normal Y)"),
                "combined__neck_yaw_to_pitch_link": ("stage2_neck_yaw_to_pitch_link", 8, "XZ 90-Degree L-Bend & Gusset Webs on Build Plate (layer normal Y)"),
                "combined__neck_pitch_to_head_cradle": ("stage3_neck_pitch_to_head_cradle", 6, "XY Right-Side U-Cradle Loop on Build Plate (layer normal Z)"),
                "combined__head_sensor_compute_carrier": ("stage4_head_sensor_compute_carrier", 14, "XZ Carrier Plate & Dorsal Ribs Flat on Build Plate (layer normal Y)"),
            }
            stg_key, slv_cnt, p_orient = stg_map[pname]
            stg_info = neck_cad["stages"][stg_key]
            printed_brep_solids.append({
                "name": pname,
                "subsystem": "Neck & Head",
                "stage": stg_key,
                "owning_link": lk,
                "solid": sol,
                "step_path": p["native_step"],
                "step_sha256": sha256_file(src_step),
                "mesh_path": p["mesh"],
                "mesh_sha256": sha256_file(ensure(ROOT / p["mesh"])),
                "volume_mm3": stg_info["volume_mm3"],
                "polymer_mass_g": stg_info["polymer_mass_paht_cf_g"],
                "sleeves_count": slv_cnt,
                "sleeves_mass_g": stg_info["sleeves_mass_g"],
                "print_orientation": p_orient,
            })
            _register_part(pname, lk, "printed_structural_connector", m_mm, PALETTE[pname], p)
            continue

        if p.get("role") == "compression_limit_sleeves":
            src_step = ensure(ROOT / p["native_step"])
            skeleton_compound_solids.extend(import_step(src_step).solids())
            m_mm = load_mesh_mm(p["mesh"])
            _register_part(pname, lk, "compression_limit_sleeves", m_mm, PALETTE["sleeve"], p)
            continue

        if pname == "original_profile_head_skin_v23":
            m_mm, m_collar = build_cute_ape_head_shell()
            out_npz = PARTS_DIR / f"{pname}.npz"
            np.savez_compressed(out_npz, vertices=np.asarray(m_mm.vertices, dtype=np.float64), faces=np.asarray(m_mm.faces, dtype=np.int64))
            p_meta = copy.deepcopy(p)
            p_meta["mesh"] = rel_path(out_npz)
            p_meta["mesh_sha256"] = sha256_file(out_npz)
            _register_part(pname, lk, p.get("role", "body_shell"), m_mm, PALETTE["shell"], p_meta)

            sub_name = "head_neck_collar"
            sub_npz = PARTS_DIR / f"{sub_name}.npz"
            np.savez_compressed(sub_npz, vertices=np.asarray(m_collar.vertices, dtype=np.float64), faces=np.asarray(m_collar.faces, dtype=np.int64))
            sub_meta = copy.deepcopy(p)
            sub_meta["name"] = sub_name
            sub_meta["owning_link"] = "neck_yaw"
            sub_meta["role"] = "body_shell"
            sub_meta["mesh"] = rel_path(sub_npz)
            sub_meta["mesh_sha256"] = sha256_file(sub_npz)
            _register_part(sub_name, "neck_yaw", "body_shell", m_collar, [164, 158, 148, 255], sub_meta)
            continue

        m_mm = load_mesh_mm(p["mesh"])
        rgba = PALETTE["shell"] if "skin" in pname else p.get("color", [140, 148, 158, 255])
        _register_part(pname, lk, p.get("role", "component"), m_mm, rgba, p)

    # ==========================================================================
    # 3. UPPER BODY / ARMS (`R/L_shoulder_pitch_link`, `roll_link`, `upper_arm`, `forearm`)
    # ==========================================================================
    forearm_shell_overrides = build_smooth_forearm_shells()
    for p in arm_man["parts"]:
        pname = p["name"]
        lk = p["owning_link"]
        if lk == "torso" or pname in REPLACED_LEGACY_PARTS:
            continue
        # Replaced 3-piece shoulder + upper-arm shells and their old screws with the single smooth upper-arm shell
        if pname in (
            "R_shoulder_front_hard_cover", "L_shoulder_front_hard_cover",
            "R_shoulder_back_hard_cover", "L_shoulder_back_hard_cover",
            "R_yaw_single_hard_cover", "L_yaw_single_hard_cover",
            "R_shoulder_front_M3x8_339", "L_shoulder_front_M3x8_339",
            "R_shoulder_back_M3x8_339", "L_shoulder_back_M3x8_339",
        ):
            continue

        if pname in forearm_shell_overrides:
            m_mm = forearm_shell_overrides[pname]
            p_meta = _save_and_update_mesh_mm(pname, m_mm, p)
            _register_part(pname, lk, p.get("role", "body_shell"), m_mm, PALETTE["shell"], p_meta)
            continue

        if p.get("role") == "printed_structural_connector" and pname in shoulder_overrides:
            sol = shoulder_overrides[pname]
            out_step = PARTS_DIR / f"{pname}.step"
            out_npz = PARTS_DIR / f"{pname}.npz"
            m_mm = save_solid_step_and_npz(sol, out_step, out_npz, tol=0.05)
            skeleton_compound_solids.append(sol)
            base_key = pname[2:]
            p_rec = next(r for r in arm_cad["parts"] if r["name"] == pname)
            vol_mm3 = float(sol.volume)
            poly_g = round(vol_mm3 * 1e-3 * 1.24, 3)
            p_meta = copy.deepcopy(p)
            p_meta["native_step"] = rel_path(out_step)
            p_meta["native_step_sha256"] = sha256_file(out_step)
            p_meta["mesh"] = rel_path(out_npz)
            p_meta["mesh_sha256"] = sha256_file(out_npz)
            p_meta["volume_mm3"] = round(vol_mm3, 2)
            p_meta["mass_kg"] = round(poly_g / 1000.0, 6)
            printed_brep_solids.append({
                "name": pname,
                "subsystem": "Upper Body / Arms",
                "stage": p_rec["stage"],
                "owning_link": lk,
                "solid": sol,
                "step_path": rel_path(out_step),
                "step_sha256": sha256_file(out_step),
                "mesh_path": rel_path(out_npz),
                "mesh_sha256": sha256_file(out_npz),
                "volume_mm3": round(vol_mm3, 2),
                "polymer_mass_g": poly_g,
                "sleeves_count": p_rec["sleeves_count"],
                "sleeves_mass_g": p_rec["sleeves_mass_g"],
                "print_orientation": p.get("print_orientation", "Flat on Build Plate"),
            })
            _register_part(pname, lk, "printed_structural_connector", m_mm, PALETTE[base_key], p_meta)
            continue

        if p.get("role") == "compression_limit_sleeves" and pname in shoulder_overrides:
            slv_comp = shoulder_overrides[pname]
            out_step = PARTS_DIR / f"{pname}.step"
            out_npz = PARTS_DIR / f"{pname}.npz"
            m_mm = save_solid_step_and_npz(slv_comp, out_step, out_npz, tol=0.05)
            skeleton_compound_solids.extend(slv_comp.solids())
            p_meta = copy.deepcopy(p)
            p_meta["native_step"] = rel_path(out_step)
            p_meta["native_step_sha256"] = sha256_file(out_step)
            p_meta["mesh"] = rel_path(out_npz)
            p_meta["mesh_sha256"] = sha256_file(out_npz)
            _register_part(pname, lk, "compression_limit_sleeves", m_mm, PALETTE["sleeve"], p_meta)
            continue

        m_mm = load_mesh_mm(p["mesh"])
        side = "R" if (pname.startswith("R_") or "_R_" in pname) else "L"
        s = 1.0 if side == "R" else -1.0

        # Transform arm hardware & fasteners according to the redesigned shoulder & upper-arm kinematics:
        if pname.startswith(("upper__R_pitch_", "upper__L_pitch_")):
            # Pitch horn, rear idler, central screw, horn bolts & washers shift outward by +/-12.0 mm (horn screws seated 2.5 mm deeper into 3.0 mm pitch_cheek)
            m_mm = m_mm.copy()
            dx_p = 9.5 * s if "_pitch_horn_" in pname else 12.0 * s
            m_mm.apply_translation([dx_p, 0.0, 0.0])
            lk = f"{side}_shoulder_pitch_link"
        elif pname.startswith((
            "upper__R_roll_C047_case", "upper__L_roll_C047_case",
            "upper__R_roll_case_", "upper__L_roll_case_",
        )):
            # Flipped Roll actuator housing + case washers/spacers now rotate with R/L_shoulder_roll_link (Green Piece)
            m_mm = _transform_flipped_roll_mesh(m_mm, side)
            lk = f"{side}_shoulder_roll_link"
        elif pname.startswith((
            "upper__R_roll_C047_main_horn", "upper__L_roll_C047_main_horn",
            "upper__R_roll_central_", "upper__L_roll_central_",
            "upper__R_roll_horn_", "upper__L_roll_horn_",
        )):
            # Flipped Roll actuator output horn + horn screws/washers now attach to R/L_shoulder_pitch_link (Blue Piece)
            m_mm = _transform_flipped_roll_mesh(m_mm, side)
            if "_roll_horn_" in pname:
                m_mm.apply_translation([0.0, -2.5, 0.0])
            lk = f"{side}_shoulder_pitch_link"
        elif pname.startswith(("upper__R_elbow_", "upper__L_elbow_")):
            # Flipped Elbow actuator case, horn, central screw, horn bolts/washers, and case bolts/washers/spacers
            # rotate 180 deg around Z at the inward-shifted motor center `(±136.925, -3.0)` so the horn faces inward (`±117.225 mm`)
            v = np.asarray(m_mm.vertices, dtype=np.float64).copy()
            v[:, 0] = 280.85 * s - v[:, 0]
            v[:, 1] = -6.0 - v[:, 1]
            if "_elbow_horn_" in pname:
                v[:, 0] += 2.6 * s
            m_mm = trimesh.Trimesh(vertices=v, faces=np.asarray(m_mm.faces, dtype=np.int64), process=False)
        else:
            # All Yaw and Forearm hardware shifts inward by -/+7.0 mm in X (yaw horn screws seated 2.4 mm into inside-mounted yaw_hub_plate)
            m_mm = m_mm.copy()
            m_mm.apply_translation([-7.0 * s, 0.0, 0.0])
            if "_yaw_horn_" in pname:
                m_mm.apply_translation([-0.40 * s, 0.0, 2.35])

        p_meta = _save_and_update_mesh_mm(pname, m_mm, p)
        rgba = PALETTE["shell"] if "hard_cover" in pname else p.get("color", [140, 148, 158, 255])
        _register_part(pname, lk, p.get("role", "component"), m_mm, rgba, p_meta)

    # Register the new single-piece Shoulder-to-Elbow shells (`R/L_upper_arm_hard_cover`) and their captive-M3 fasteners
    for side in ("R", "L"):
        lk_roll = f"{side}_shoulder_roll_link"
        sh_name = f"{side}_upper_arm_hard_cover"
        sh_sol = shoulder_overrides[sh_name]
        sh_step = PARTS_DIR / f"{sh_name}.step"
        sh_npz = PARTS_DIR / f"{sh_name}.npz"
        sh_mm = save_solid_step_and_npz(sh_sol, sh_step, sh_npz, tol=0.05)
        sh_meta = {
            "name": sh_name,
            "owning_link": lk_roll,
            "role": "body_shell",
            "material": "PA12 / PETG",
            "volume_mm3": round(float(sh_sol.volume), 2),
            "native_step": rel_path(sh_step),
            "native_step_sha256": sha256_file(sh_step),
            "mesh": rel_path(sh_npz),
            "mesh_sha256": sha256_file(sh_npz),
        }
        _register_part(sh_name, lk_roll, "body_shell", sh_mm, PALETTE["shell"], sh_meta)

        for fst_suf in ("front_M3_nut", "front_M3x8", "rear_M3_nut", "rear_M3x8"):
            f_name = f"{side}_upper_arm_shell_{fst_suf}"
            f_sol = shoulder_overrides[f_name]
            f_step = PARTS_DIR / f"{f_name}.step"
            f_npz = PARTS_DIR / f"{f_name}.npz"
            f_mm = save_solid_step_and_npz(f_sol, f_step, f_npz, tol=0.05)
            f_meta = {
                "name": f_name,
                "owning_link": lk_roll,
                "role": "fastener",
                "native_step": rel_path(f_step),
                "native_step_sha256": sha256_file(f_step),
                "mesh": rel_path(f_npz),
                "mesh_sha256": sha256_file(f_npz),
            }
            _register_part(f_name, lk_roll, "fastener", f_mm, PALETTE["fastener"], f_meta)

    # ==========================================================================
    # 4. BILATERAL LOWER BODY (`R/L_yaw_link`, `pitch_link`, `roll_link`, `shin`, `ankle_pitch`, `foot`)
    # ==========================================================================
    # Load the Lower-Body GLB scene so any part without a standalone mesh path is extracted cleanly in mm
    leg_glb_path = ensure(ROOT / "artifacts/robot/body/leg/complete_leg_hip_combined/assembly-with-full-shell.glb")
    leg_scene = trimesh.load(leg_glb_path, force="scene")

    for p in leg_man["parts"]:
        pname = p["name"]
        lk = p["owning_link"]
        if pname in REPLACED_LEGACY_PARTS or pname.startswith("full_shell__"):
            continue
        if lk == "pelvis":
            if pname in ("A1_integral_pelvis_flat_mate_r2", "R_yaw_servo_case_reference", "L_yaw_servo_case_reference"):
                continue
            lk = "torso"

        if pname in thigh_overrides:
            sol = thigh_overrides[pname]
            out_step = PARTS_DIR / f"{pname}.step"
            out_npz = PARTS_DIR / f"{pname}.npz"
            m_mm = save_solid_step_and_npz(sol, out_step, out_npz, tol=0.05)
            vol_mm3 = float(sol.volume)
            p_meta = copy.deepcopy(p)
            p_meta["native_step"] = rel_path(out_step)
            p_meta["native_step_sha256"] = sha256_file(out_step)
            p_meta["mesh"] = rel_path(out_npz)
            p_meta["mesh_sha256"] = sha256_file(out_npz)
            p_meta["volume_mm3"] = round(vol_mm3, 2)

            if p.get("role") == "printed_structural_connector":
                skeleton_compound_solids.append(sol)
                base_key = pname[2:]
                c_rec = next(r for r in leg_cad["parts"] if r["name"] == pname)
                poly_g = round(vol_mm3 * 1e-3 * 1.22, 3)
                p_meta["mass_kg"] = round(poly_g / 1000.0, 6)
                printed_brep_solids.append({
                    "name": pname,
                    "subsystem": "Lower Body (Legs, Hips & Feet)",
                    "stage": c_rec["stage"],
                    "owning_link": lk,
                    "solid": sol,
                    "step_path": rel_path(out_step),
                    "step_sha256": sha256_file(out_step),
                    "mesh_path": rel_path(out_npz),
                    "mesh_sha256": sha256_file(out_npz),
                    "volume_mm3": round(vol_mm3, 2),
                    "polymer_mass_g": poly_g,
                    "sleeves_count": c_rec["sleeves"]["count"],
                    "sleeves_mass_g": c_rec["sleeves"]["mass_g"],
                    "print_orientation": p.get("print_orientation", "Flat on Build Plate"),
                })
                _register_part(pname, lk, "printed_structural_connector", m_mm, PALETTE[base_key], p_meta)
            else:
                rgba = PALETTE["shell"] if "hard_cover" in pname else p.get("color", [165, 174, 184, 255])
                _register_part(pname, lk, p.get("role", "component"), m_mm, rgba, p_meta)
            continue

        if pname in leg_scene.graph.nodes_geometry:
            T_node, g_key = leg_scene.graph[pname]
            m_mm = leg_scene.geometry[g_key].copy()
            m_mm.apply_transform(T_node)
            m_mm.apply_scale(1000.0)
        elif p.get("mesh"):
            m_mm = load_mesh_mm(p["mesh"])
        else:
            continue

        if p.get("role") == "printed_structural_connector":
            src_step = ensure(ROOT / p["native_step"])
            sol = import_step(src_step).solids()[0]
            skeleton_compound_solids.append(sol)
            base_key = pname[2:]
            c_rec = next(r for r in leg_cad["parts"] if r["name"] == pname)
            printed_brep_solids.append({
                "name": pname,
                "subsystem": "Lower Body (Legs, Hips & Feet)",
                "stage": c_rec["stage"],
                "owning_link": lk,
                "solid": sol,
                "step_path": p["native_step"],
                "step_sha256": sha256_file(src_step),
                "mesh_path": p["mesh"],
                "mesh_sha256": p["mesh_sha256"],
                "volume_mm3": c_rec["volume_mm3"],
                "polymer_mass_g": c_rec["polymer_mass_g_PAHT_CF"],
                "sleeves_count": c_rec["sleeves"]["count"],
                "sleeves_mass_g": c_rec["sleeves"]["mass_g"],
                "print_orientation": p.get("print_orientation", "Flat on Build Plate"),
            })
            _register_part(pname, lk, "printed_structural_connector", m_mm, PALETTE[base_key], p)
            continue

        if p.get("role") == "compression_limit_sleeves":
            src_step = ensure(ROOT / p["native_step"])
            skeleton_compound_solids.extend(import_step(src_step).solids())
            _register_part(pname, lk, "compression_limit_sleeves", m_mm, PALETTE["sleeve"], p)
            continue

        if pname in ("R_foot_boot_shell", "L_foot_boot_shell", "R_foot_sole_shell", "L_foot_sole_shell"):
            m_mm = slim_foot_shell_mesh(m_mm, pname[0])

        rgba = p.get("color", [165, 174, 184, 255])
        if "sole_shell" in pname:
            rgba = PALETTE["sole_shell"]
        elif "hard_cover" in pname or "boot_shell" in pname:
            rgba = PALETTE["shell"]
        _register_part(pname, lk, p.get("role", "component"), m_mm, rgba, p)

    # ==========================================================================
    # 4B. BILATERAL SYMMETRIC GEARED HANDS (`cove50-hand-symmetric-geared-r1`, scaled 0.50x to fit forearm wrist)
    # ==========================================================================
    hand_meta = {
        "palm_lower": ("forearm", "hand_palm_lower", "Flat on Z = 170.95 mm Wrist Face (layer normal +Z)"),
        "palm_upper": ("forearm", "hand_palm_upper", "Flat on Z = 141.45 mm Seam Face (layer normal -Z)"),
        "right_finger": ("right_finger", "hand_right_finger", "Flat on Y Gear/C-Web Face (layer normal Y)"),
        "left_finger": ("left_finger", "hand_left_finger", "Flat on Y Gear/C-Web Face (layer normal Y)"),
    }
    for pname, sol in printed_hand_solids.items():
        side = pname[:2]  # "R_" or "L_"
        bkey = pname[2:]
        lk_suf, pal_key, p_orient = hand_meta[bkey]
        owning_lk = f"{side}{lk_suf}"
        out_step = PARTS_DIR / f"{pname}.step"
        out_npz = PARTS_DIR / f"{pname}.npz"
        m_mm = save_solid_step_and_npz(sol, out_step, out_npz, tol=0.05)
        skeleton_compound_solids.append(sol)
        vol_mm3 = float(sol.volume)
        poly_g = round(vol_mm3 * 1e-3 * 1.22, 3)
        p_entry = {
            "name": pname,
            "owning_link": owning_lk,
            "role": "printed_structural_connector",
            "material": "PAHT-CF",
            "volume_mm3": round(vol_mm3, 2),
            "mass_kg": round(poly_g / 1000.0, 6),
            "native_step": rel_path(out_step),
            "native_step_sha256": sha256_file(out_step),
            "mesh": rel_path(out_npz),
            "mesh_sha256": sha256_file(out_npz),
            "print_orientation": p_orient,
        }
        printed_brep_solids.append({
            "name": pname,
            "subsystem": "Upper Body / Arms",
            "stage": f"hand_{bkey}",
            "owning_link": owning_lk,
            "solid": sol,
            "step_path": rel_path(out_step),
            "step_sha256": sha256_file(out_step),
            "mesh_path": rel_path(out_npz),
            "mesh_sha256": sha256_file(out_npz),
            "volume_mm3": round(vol_mm3, 2),
            "polymer_mass_g": poly_g,
            "sleeves_count": 0,
            "sleeves_mass_g": 0.0,
            "print_orientation": p_orient,
        })
        _register_part(pname, owning_lk, "printed_structural_connector", m_mm, PALETTE[pal_key], p_entry)

    for hw_nm, (m_hw, hw_lk, hw_role, hw_rgba) in hand_hw_meshes.items():
        _register_part(hw_nm, hw_lk, hw_role, m_hw, hw_rgba, {"name": hw_nm, "owning_link": hw_lk, "role": hw_role})

    # Also include the foot sole tactile/force sensor pads from EC_MANIFEST on R_foot / L_foot (excluding legacy lower__* metal parts)
    for p in ec_man["parts"]:
        pname = p["name"]
        lk = p.get("owning_link") or p.get("parent_link")
        if lk in ("R_foot", "L_foot") and not pname.startswith("lower__") and pname not in seen_names and p.get("mesh"):
            m_mm = load_mesh_mm(p["mesh"])
            rgba = p.get("color", [72, 138, 96, 255])
            _register_part(pname, lk, p.get("role", "sensor"), m_mm, rgba, p)

    # ==========================================================================
    # 5. BUILD 26-JOINT / 24-DOF KINEMATIC TREE (`root_link: "torso"`)
    # ==========================================================================
    joints: list[dict] = []
    controls: list[dict] = []

    # 12 Lower-Body joints (re-parent R_hip_yaw & L_hip_yaw from "pelvis" to "torso"). Every joint range comes from the body
    # definitions (`joints:` in robot/body/leg/package_complete_leg_hip.bom.yaml, arm/build_arm_connectors.bom.yaml,
    # neck-head/build_neck_head_connectors.bom.yaml, assembly/assemble_complete_fdm_body.bom.yaml): BOM_CONFIG.apply_joint_ranges below.
    for j in leg_man["joints"]:
        jc = copy.deepcopy(j)
        if jc["parent"] == "pelvis":
            jc["parent"] = "torso"
        joints.append(jc)
        controls.append({"name": jc["name"], "range_deg": jc.get("range_deg")})  # set from the declarations below

    # 8 Arm joints: origins / axes in the assembled body (shifted inward of the arm section's frames) from arm/build_arm_connectors.bom.yaml joints:
    for j in arm_man["joints"]:
        jc = copy.deepcopy(j)
        BOM_CONFIG.apply_joint_frames([jc])   # origin / axis in the assembled body: the arm joints' `joints:` declarations
        joints.append(jc)
        controls.append({"name": jc["name"], "range_deg": jc.get("range_deg")})  # set from the declarations below

    # 4 Geared-Hand Finger Joints driven by 2 controls (`R_finger_opening`, `L_finger_opening`)
    joints.extend([
        {
            "name": "R_right_finger_opening",
            "parent": "R_forearm",
            "child": "R_right_finger",
            "origin_world_mm": BOM_CONFIG.joint_origin("R_right_finger_opening"),
            "axis_world_neutral": BOM_CONFIG.joint_axis("R_right_finger_opening"),
            "range_deg": BOM_CONFIG.joint_range("R_right_finger_opening"),
            "driven_by": {"linear": {"R_finger_opening": 1.0}},
        },
        {
            "name": "R_left_finger_opening",
            "parent": "R_forearm",
            "child": "R_left_finger",
            "origin_world_mm": BOM_CONFIG.joint_origin("R_left_finger_opening"),
            "axis_world_neutral": BOM_CONFIG.joint_axis("R_left_finger_opening"),
            "range_deg": BOM_CONFIG.joint_range("R_left_finger_opening"),
            "driven_by": {"linear": {"R_finger_opening": -1.0}},
        },
        {
            "name": "L_right_finger_opening",
            "parent": "L_forearm",
            "child": "L_right_finger",
            "origin_world_mm": BOM_CONFIG.joint_origin("L_right_finger_opening"),
            "axis_world_neutral": BOM_CONFIG.joint_axis("L_right_finger_opening"),
            "range_deg": BOM_CONFIG.joint_range("L_right_finger_opening"),
            "driven_by": {"linear": {"L_finger_opening": 1.0}},
        },
        {
            "name": "L_left_finger_opening",
            "parent": "L_forearm",
            "child": "L_left_finger",
            "origin_world_mm": BOM_CONFIG.joint_origin("L_left_finger_opening"),
            "axis_world_neutral": BOM_CONFIG.joint_axis("L_left_finger_opening"),
            "range_deg": BOM_CONFIG.joint_range("L_left_finger_opening"),
            "driven_by": {"linear": {"L_finger_opening": -1.0}},
        },
    ])
    controls.extend([
        {"name": "R_finger_opening", "range_deg": BOM_CONFIG.joint_range("R_finger_opening")},
        {"name": "L_finger_opening", "range_deg": BOM_CONFIG.joint_range("L_finger_opening")},
    ])

    # 2 Neck/Head joints (`neck_yaw`, `neck_pitch`)
    for j in neck_man["joints"]:
        jc = copy.deepcopy(j)
        joints.append(jc)
        controls.append({"name": jc["name"], "range_deg": jc.get("range_deg")})  # set from the declarations below

    # Joint ranges: the declarations in the body definitions are the only source (owner 2026-11-23); an undriven joint
    # without one stops the build here, naming it.
    for line in BOM_CONFIG.apply_joint_ranges({"joints": joints, "controls": controls}):
        print(f"  joint range from definitions: {line}", flush=True)

    # Export Complete 28-Part + 316-Sleeve Structural Skeleton STEP
    skeleton_step_path = ART_DIR / "complete-fdm-structural-skeleton.step"
    export_step(Compound(children=skeleton_compound_solids), str(skeleton_step_path))

    # Export Complete Interactive GLB Scene
    glb_path = ART_DIR / "assembly-with-full-shell.glb"
    scene.export(str(glb_path))

    # Build Subsystem & Whole-Body Mass / Sleeve / FEM Rollup
    subsystem_rollup = [
        {
            "subsystem": "Lower Body (6 stages x 2 legs)",
            "result_id": "cove50-complete-leg-hip-fdm-combined-r10",
            "decision_id": "M-y8acff",
            "printed_parts_count": 12,
            "steel_sleeves_count": 128,
            "polymer_mass_g": 157.20,
            "sleeves_mass_g": 30.44,
            "fdm_total_mass_g": 187.64,
            "legacy_baseline_mass_g": 332.78,
            "mass_saved_g": 145.14,
            "mass_saved_pct": 43.61,
            "min_paht_cf_dry_safety_factor": 2.86,
            "governing_load_case": "4 kg @ 1.5g Single-Leg Landing Recovery + C047 Stall",
        },
        {
            "subsystem": "Upper Body / Arms (4 stages x 2 arms)",
            "result_id": "cove50-arm-fdm-combined-r12",
            "decision_id": "M-d34p89",
            "printed_parts_count": 8,
            "steel_sleeves_count": 68,
            "polymer_mass_g": 59.30,
            "sleeves_mass_g": 15.44,
            "fdm_total_mass_g": 74.74,
            "legacy_baseline_mass_g": 136.05,
            "mass_saved_g": 61.31,
            "mass_saved_pct": 45.06,
            "min_paht_cf_dry_safety_factor": 1.65,
            "governing_load_case": "4 kg @ 1.5g Push-Up / Fall-Arrest + 4.0 N*m C047 Stall",
        },
        {
            "subsystem": "Neck & Head (4 stages)",
            "result_id": "cove50-neck-head-fdm-combined-r1",
            "decision_id": "M-8jkx67",
            "printed_parts_count": 4,
            "steel_sleeves_count": 34,
            "polymer_mass_g": 21.84,
            "sleeves_mass_g": 6.72,
            "fdm_total_mass_g": 28.56,
            "legacy_baseline_mass_g": 52.67,
            "mass_saved_g": 24.11,
            "mass_saved_pct": 45.78,
            "min_paht_cf_dry_safety_factor": 2.34,
            "governing_load_case": "6g Head Fall-Recovery + 1.60 N*m Dual-Axis XM335 Stall",
        },
        {
            "subsystem": "Center-Body Torso, Pelvis & Electronics (4 stages)",
            "result_id": "cove50-torso-fdm-combined-r1",
            "decision_id": "M-h3aaq6",
            "printed_parts_count": 4,
            "steel_sleeves_count": 86,
            "polymer_mass_g": 164.08,
            "sleeves_mass_g": 14.13,
            "fdm_total_mass_g": 178.21,
            "legacy_baseline_mass_g": 251.82,
            "mass_saved_g": 73.61,
            "mass_saved_pct": 29.23,
            "min_paht_cf_dry_safety_factor": 1.55,
            "governing_load_case": "Multi-Limb Stall (8 N*m Shoulder + 8 N*m Hip + 1.8 N*m Neck) & 6g Fall",
        },
    ]

    bom_items = [
        {
            "id": "lower_body_fdm_connectors_12pc",
            "name": "Bilateral 6-Stage Lower-Body FDM Connectors (12 printed parts: Shin, Ankle Coupler, Foot, Thigh, Hip Pitch-Roll, Hip Yaw-Pitch)",
            "category": "Produced parts (FDM 3D printed)",
            "quantity": 12,
            "unit": "ea",
            "material": "Bambu PAHT-CF / PA12-CF (or PETG-CF)",
            "total_mass_g": 157.20,
            "process": "Home-lab FDM 3D print (0.4 mm hardened steel nozzle, 0.20 mm layers, 4 perimeters)",
        },
        {
            "id": "lower_body_ss304_sleeves_128pc",
            "name": "Lower-Body Pressed-In 304 SS Compression Limit Sleeves (64/leg = 128 sleeves)",
            "category": "Fasteners & Hardware",
            "quantity": 128,
            "unit": "ea",
            "material": "304 Stainless Steel",
            "total_mass_g": 30.44,
            "process": "Press-fit into Stage 1-6 leg/hip counterbores",
        },
        {
            "id": "arm_fdm_connectors_8pc",
            "name": "Bilateral 4-Stage Arm FDM Connectors (8 printed parts: Shoulder Pitch-Roll, Roll-Yaw, Upper Arm, Forearm)",
            "category": "Produced parts (FDM 3D printed)",
            "quantity": 8,
            "unit": "ea",
            "material": "Bambu PAHT-CF / PA12-CF (or PETG-CF)",
            "total_mass_g": 59.30,
            "process": "Home-lab FDM 3D print (0.4 mm hardened steel nozzle, 0.20 mm layers, 4 perimeters)",
        },
        {
            "id": "arm_ss304_sleeves_68pc",
            "name": "Arm Pressed-In 304 SS Compression Limit Sleeves (34/arm = 68 sleeves)",
            "category": "Fasteners & Hardware",
            "quantity": 68,
            "unit": "ea",
            "material": "304 Stainless Steel",
            "total_mass_g": 15.44,
            "process": "Press-fit into Stage 1-4 arm counterbores",
        },
        {
            "id": "neck_head_fdm_connectors_4pc",
            "name": "4-Stage Neck & Head FDM Connectors (Neck Root Cradle, Yaw-to-Pitch Link, Pitch-to-Head U-Cradle, Head Sensor/Compute Carrier)",
            "category": "Produced parts (FDM 3D printed)",
            "quantity": 4,
            "unit": "ea",
            "material": "Bambu PAHT-CF / PA12-CF (or PETG-CF)",
            "total_mass_g": 21.84,
            "process": "Home-lab FDM 3D print (0.4 mm hardened steel nozzle, 0.20 mm layers, 4 perimeters)",
        },
        {
            "id": "neck_head_ss304_sleeves_34pc",
            "name": "Neck & Head Pressed-In 304 SS Compression Limit Sleeves (6+8+6+14 = 34 sleeves)",
            "category": "Fasteners & Hardware",
            "quantity": 34,
            "unit": "ea",
            "material": "304 Stainless Steel",
            "total_mass_g": 6.72,
            "process": "Press-fit into Stage 1-4 neck/head counterbores",
        },
        {
            "id": "torso_fdm_carriers_4pc",
            "name": "4-Stage Center-Body Torso, Pelvis, 9-Cell Honeycomb Battery & Electronics FDM Carriers",
            "category": "Produced parts (FDM 3D printed)",
            "quantity": 4,
            "unit": "ea",
            "material": "Bambu PAHT-CF / PA12-CF (or PETG-CF)",
            "total_mass_g": 164.08,
            "process": "Home-lab FDM 3D print (0.4 mm hardened steel nozzle, 0.20 mm layers, 4 perimeters)",
        },
        {
            "id": "torso_ss304_sleeves_86pc",
            "name": "Torso & Electronics Pressed-In 304 SS Compression Limit Sleeves (39+11+22+14 = 86 sleeves)",
            "category": "Fasteners & Hardware",
            "quantity": 86,
            "unit": "ea",
            "material": "304 Stainless Steel",
            "total_mass_g": 14.13,
            "process": "Press-fit into Stage 1-4 torso/pelvis/electronics counterbores",
        },
        {
            "id": "honeycomb_battery_fins_9pc",
            "name": "Direct-on-Battery 6063-T5 Aluminum 4-Blade Longitudinal Heat-Sink Fin Strips (9 cells)",
            "category": "Thermal & Power Hardware",
            "quantity": 9,
            "unit": "ea",
            "material": "6063-T5 Anodized Aluminum + 0.25 mm 3M 8810 TIM",
            "total_mass_g": 28.44,
            "process": "Bonded directly to 21700 cell cans in full-height honeycomb window slots",
        },
    ]

    assembly_manifest = {
        "schema": "cove50.complete-fdm-body.assembly-parts.v1",
        "study_id": "cove50-complete-fdm-body",
        "result_id": "cove50-complete-fdm-body-r1",
        "version": "r1",
        "root_link": "torso",
        "model_units": "m",
        "units": "CAD mm; GLB m",
        "parts": assembly_parts,
        "joints": joints,
        "controls": controls,
    }
    write_json(ART_DIR / "assembly-parts.json", assembly_manifest)
    # The r1 subsystem rollup (FEM safety factors, masses per section) is study evidence, not the BOM: the robot BOM is
    # generated from the manifest by robot/body/bom/build.py (owner 2026-11-20).
    write_json(ART_DIR / "subsystem-rollup.json", {
        "scope": "r1 subsystem rollup of the complete FDM body (4 sections); counts and masses as of r1",
        "subsystem_rollup": subsystem_rollup,
        "items": bom_items,
    })

    # ==========================================================================
    # 6. WHOLE-BODY EXACT BREP MULTI-POSE VERIFICATION (`whole-body-verification-report.json`)
    # ==========================================================================
    print(f"Running whole-body exact OpenCASCADE BRep verification across {len(printed_brep_solids)} printed FDM solids...")
    poses = [
        ("neutral_0deg", {}),
        (
            "walking_stride_right_lead",
            {
                "R_hip_pitch": -22.0, "R_knee": 36.0, "R_ankle_pitch": -14.0,
                "L_hip_pitch": 18.0, "L_knee": 24.0, "L_ankle_pitch": -10.0,
                "R_shoulder_pitch": 25.0, "R_shoulder_roll": 12.0, "R_elbow": 45.0,
                "L_shoulder_pitch": -20.0, "L_shoulder_roll": 12.0, "L_elbow": 35.0,
                "neck_yaw": 12.0, "neck_pitch": 8.0,
            },
        ),
        (
            "walking_stride_left_lead",
            {
                "L_hip_pitch": -22.0, "L_knee": 36.0, "L_ankle_pitch": -14.0,
                "R_hip_pitch": 18.0, "R_knee": 24.0, "R_ankle_pitch": -10.0,
                "L_shoulder_pitch": 25.0, "L_shoulder_roll": 12.0, "L_elbow": 45.0,
                "R_shoulder_pitch": -20.0, "R_shoulder_roll": 12.0, "R_elbow": 35.0,
                "neck_yaw": -12.0, "neck_pitch": 8.0,
            },
        ),
        (
            "deep_biped_squat",
            {
                "R_hip_pitch": -32.0, "R_hip_roll": -10.0, "R_knee": 72.0, "R_ankle_pitch": -28.0,
                "L_hip_pitch": -32.0, "L_hip_roll": 10.0, "L_knee": 72.0, "L_ankle_pitch": -28.0,
                "R_shoulder_pitch": -30.0, "R_shoulder_roll": 15.0, "R_elbow": 65.0,
                "L_shoulder_pitch": -30.0, "L_shoulder_roll": 15.0, "L_elbow": 65.0,
                "neck_pitch": -12.0,
            },
        ),
        (
            "bilateral_forward_reach",
            {
                "R_shoulder_pitch": -45.0, "R_shoulder_roll": 15.0, "R_shoulder_yaw": -20.0, "R_elbow": 70.0,
                "L_shoulder_pitch": -45.0, "L_shoulder_roll": 15.0, "L_shoulder_yaw": -20.0, "L_elbow": 70.0,
                "R_finger_opening": 30.0, "L_finger_opening": 30.0,
                "neck_pitch": 16.0,
            },
        ),
        (
            "wide_abduction_arm_spread",
            {
                "R_shoulder_pitch": -20.0, "R_shoulder_roll": 55.0, "R_shoulder_yaw": 30.0, "R_elbow": 50.0,
                "L_shoulder_pitch": -20.0, "L_shoulder_roll": 55.0, "L_shoulder_yaw": 30.0, "L_elbow": 50.0,
                "R_finger_opening": 20.0, "L_finger_opening": 20.0,
                "neck_yaw": 20.0, "neck_pitch": -10.0,
            },
        ),
        (
            "head_look_down_left_wide_stance",
            {
                "R_hip_yaw": -12.0, "R_hip_roll": -12.0, "R_hip_pitch": -15.0, "R_knee": 30.0, "R_ankle_pitch": -15.0, "R_ankle_roll": 12.0,
                "L_hip_yaw": 12.0, "L_hip_roll": 12.0, "L_hip_pitch": -15.0, "L_knee": 30.0, "L_ankle_pitch": -15.0, "L_ankle_roll": -12.0,
                "R_shoulder_roll": 15.0, "L_shoulder_roll": 15.0,
                "neck_yaw": 30.0, "neck_pitch": 25.0,
            },
        ),
        (
            "full_ape_crouch_and_gaze",
            {
                "R_hip_pitch": -28.0, "R_hip_roll": -8.0, "R_knee": 58.0, "R_ankle_pitch": -22.0,
                "L_hip_pitch": -28.0, "L_hip_roll": 8.0, "L_knee": 58.0, "L_ankle_pitch": -22.0,
                "R_shoulder_pitch": -35.0, "R_shoulder_roll": 35.0, "R_shoulder_yaw": 25.0, "R_elbow": 55.0,
                "L_shoulder_pitch": -35.0, "L_shoulder_roll": 35.0, "L_shoulder_yaw": 25.0, "L_elbow": 55.0,
                "R_finger_opening": 25.0, "L_finger_opening": 25.0,
                "neck_yaw": -20.0, "neck_pitch": -15.0,
            },
        ),
    ]

    pose_results = []
    seen_pair_transforms: dict[tuple, float] = {}
    max_whole_body_overlap_mm3 = 0.0

    for pose_name, q_ctrl in poses:
        T_links = fk_transforms(joints, q_ctrl)
        transformed = []
        for item in printed_brep_solids:
            M = T_links[item["owning_link"]]
            s_tf = transform_brep(item["solid"], M)
            bb = s_tf.bounding_box()
            lo = np.array([bb.min.X, bb.min.Y, bb.min.Z], dtype=float)
            hi = np.array([bb.max.X, bb.max.Y, bb.max.Z], dtype=float)
            transformed.append((item["name"], item["subsystem"], item["owning_link"], s_tf, lo, hi, M))

        pose_max_ov = 0.0
        pairs_checked = 0
        overlaps_found = []
        for i in range(len(transformed)):
            n1, sub1, l1, b1, lo1, hi1, M1 = transformed[i]
            inv1 = np.linalg.inv(M1)
            for j in range(i + 1, len(transformed)):
                n2, sub2, l2, b2, lo2, hi2, M2 = transformed[j]
                if not np.all(np.minimum(hi1, hi2) - np.maximum(lo1, lo2) >= -0.01):
                    continue
                pairs_checked += 1
                rel_M = inv1 @ M2
                cache_key = (n1, n2, tuple(np.round(rel_M[:3].ravel(), 5)))
                if cache_key in seen_pair_transforms:
                    ov = seen_pair_transforms[cache_key]
                else:
                    ov = brep_common_volume_mm3(b1, b2)
                    seen_pair_transforms[cache_key] = ov
                pose_max_ov = max(pose_max_ov, ov)
                max_whole_body_overlap_mm3 = max(max_whole_body_overlap_mm3, ov)
                if ov > 1e-4:
                    overlaps_found.append({"part_a": n1, "part_b": n2, "overlap_mm3": round(ov, 5)})

        pose_results.append({
            "pose": pose_name,
            "controls_deg": q_ctrl,
            "aabb_pairs_checked": pairs_checked,
            "max_pairwise_brep_overlap_mm3": round(pose_max_ov, 6),
            "overlaps": overlaps_found,
        })
        print(f"  [Pose: {pose_name:38s}] AABB pairs={pairs_checked:3d} | max_brep_overlap={pose_max_ov:.6f} mm^3")

    assert max_whole_body_overlap_mm3 == 0.0, f"Non-zero BRep overlap found: {max_whole_body_overlap_mm3}"

    # Audit every single one of the 28 printed FDM parts for single-solid validity & build-plate fit
    part_audits = []
    for item in printed_brep_solids:
        sol = item["solid"]
        bb = sol.bounding_box()
        dims = sorted([
            round(float(bb.max.X - bb.min.X), 2),
            round(float(bb.max.Y - bb.min.Y), 2),
            round(float(bb.max.Z - bb.min.Z), 2),
        ], reverse=True)
        fits_256 = all(d <= 250.0 for d in dims)
        fits_350 = all(d <= 340.0 for d in dims)
        assert len(sol.solids()) == 1 and sol.is_valid and fits_256
        part_audits.append({
            "name": item["name"],
            "subsystem": item["subsystem"],
            "stage": item["stage"],
            "owning_link": item["owning_link"],
            "step_path": item["step_path"],
            "step_sha256": item["step_sha256"],
            "mesh_path": item["mesh_path"],
            "mesh_sha256": item["mesh_sha256"],
            "solid_count": len(sol.solids()),
            "is_valid": bool(sol.is_valid),
            "volume_mm3": item["volume_mm3"],
            "polymer_mass_g": item["polymer_mass_g"],
            "sleeves_count": item["sleeves_count"],
            "sleeves_mass_g": item["sleeves_mass_g"],
            "sorted_bbox_dims_mm": dims,
            "fits_256x256_bed": fits_256,
            "fits_350x350_bed": fits_350,
            "print_orientation": item["print_orientation"],
        })

    # ==========================================================================
    # 7. HOME-LAB PRINT-BED NESTING AUDIT (`256x256 mm` and `350x350 mm` beds)
    # ==========================================================================
    def pack_parts_onto_beds(bed_w: float, bed_h: float, margin: float = 8.0, gap: float = 5.0) -> list[dict]:
        # Sort parts by max footprint height descending for shelf packing
        footprints = []
        for pa in part_audits:
            d0, d1, d2 = pa["sorted_bbox_dims_mm"]
            # Use the two largest dimensions as conservative print-bed footprint (w >= h)
            footprints.append({
                "name": pa["name"],
                "subsystem": pa["subsystem"],
                "w_mm": d0,
                "h_mm": d1,
                "z_height_mm": d2,
                "mass_g": pa["polymer_mass_g"],
            })
        footprints.sort(key=lambda x: (x["h_mm"], x["w_mm"]), reverse=True)

        plates: list[dict] = []
        for fp in footprints:
            placed = False
            for pl in plates:
                for sh in pl["shelves"]:
                    if sh["cursor_x"] + fp["w_mm"] <= bed_w - margin and sh["y"] + fp["h_mm"] <= bed_h - margin:
                        fp_placement = {**fp, "x_mm": round(sh["cursor_x"], 2), "y_mm": round(sh["y"], 2)}
                        pl["placements"].append(fp_placement)
                        sh["cursor_x"] += fp["w_mm"] + gap
                        sh["height"] = max(sh["height"], fp["h_mm"])
                        placed = True
                        break
                if not placed:
                    next_y = margin if not pl["shelves"] else pl["shelves"][-1]["y"] + pl["shelves"][-1]["height"] + gap
                    if next_y + fp["h_mm"] <= bed_h - margin and margin + fp["w_mm"] <= bed_w - margin:
                        pl["shelves"].append({
                            "y": next_y,
                            "height": fp["h_mm"],
                            "cursor_x": margin + fp["w_mm"] + gap,
                        })
                        pl["placements"].append({**fp, "x_mm": round(margin, 2), "y_mm": round(next_y, 2)})
                        placed = True
                if placed:
                    break
            if not placed:
                plates.append({
                    "plate_index": len(plates) + 1,
                    "bed_size_mm": [bed_w, bed_h],
                    "shelves": [{"y": margin, "height": fp["h_mm"], "cursor_x": margin + fp["w_mm"] + gap}],
                    "placements": [{**fp, "x_mm": round(margin, 2), "y_mm": round(margin, 2)}],
                })

        for pl in plates:
            pl.pop("shelves", None)
            pl["parts_count"] = len(pl["placements"])
            pl["total_polymer_mass_g"] = round(sum(x["mass_g"] for x in pl["placements"]), 2)
            pl["max_z_height_mm"] = round(max(x["z_height_mm"] for x in pl["placements"]), 2)
        return plates

    plates_256 = pack_parts_onto_beds(256.0, 256.0)
    plates_350 = pack_parts_onto_beds(350.0, 350.0)

    nesting_audit = {
        "schema": "cove50.complete-fdm-body.print-bed-nesting.v1",
        "total_printed_parts": len(part_audits),
        "standard_256x256x256_bed": {
            "bed_dimensions_mm": [256.0, 256.0, 256.0],
            "plates_required": len(plates_256),
            "plates": plates_256,
        },
        "large_350x350x350_bed": {
            "bed_dimensions_mm": [350.0, 350.0, 350.0],
            "plates_required": len(plates_350),
            "plates": plates_350,
        },
    }
    write_json(ART_DIR / "print-bed-nesting-audit.json", nesting_audit)

    verification_report = {
        "schema": "cove50.complete-fdm-body.verification.v1",
        "study_id": "cove50-complete-fdm-body",
        "result_id": "cove50-complete-fdm-body-r1",
        "version": "r1",
        "legacy_metal_residual_parts_count": 0,
        "legacy_metal_residual_volume_mm3": 0.0,
        "total_scene_parts_count": len(assembly_parts),
        "total_printed_fdm_parts_count": len(part_audits),
        "total_pressed_in_ss304_sleeves_count": sum(p["sleeves_count"] for p in part_audits),
        "total_kinematic_joints_count": len(joints),
        "total_interactive_dof_controls_count": len(controls),
        "whole_body_28part_fdm_polymer_mass_g": 402.42,
        "whole_body_316sleeve_mass_g": 66.73,
        "whole_body_28part_fdm_and_316sleeve_mass_g": 469.15,
        "whole_body_legacy_baseline_mass_g": 773.32,
        "whole_body_mass_saved_g": 304.17,
        "whole_body_mass_saved_pct": 39.33,
        "all_28_parts_single_valid_solid": True,
        "all_28_parts_fit_256x256_bed": True,
        "poses_checked_count": len(pose_results),
        "max_pairwise_brep_overlap_across_all_poses_mm3": round(max_whole_body_overlap_mm3, 6),
        "pose_verification": pose_results,
        "printed_parts_audit": part_audits,
        "subsystem_rollup": subsystem_rollup,
    }
    write_json(ART_DIR / "whole-body-verification-report.json", verification_report)
    write_json(
        ART_DIR / "manifest.json",
        {
            "study_id": "cove50-complete-fdm-body",
            "result_id": "cove50-complete-fdm-body-r1",
            "version": "r1",
            "skeleton_step": rel_path(skeleton_step_path),
            "skeleton_step_sha256": sha256_file(skeleton_step_path),
            "inspection_glb": rel_path(glb_path),
            "inspection_glb_sha256": sha256_file(glb_path),
            "assembly_parts_manifest": rel_path(ART_DIR / "assembly-parts.json"),
            "verification_report": rel_path(ART_DIR / "whole-body-verification-report.json"),
            "print_bed_nesting_audit": rel_path(ART_DIR / "print-bed-nesting-audit.json"),
            "subsystem_rollup": subsystem_rollup,
            "printed_parts": part_audits,
        },
    )

    # ==========================================================================
    # 8. RENDER HIGH-RESOLUTION ENGINEERING FIGURES
    # ==========================================================================
    if "--no-renders" in sys.argv:
        # presentation figures only (renders/): no generator reads them; the body rebuild runs the data at its own size
        # (the render phase holds every mesh in mm plus a posed copy per panel: 10.4 GB peak on 747 rows, 2026-11-22)
        print(f"Completed Complete Cove-50 FDM Ape Body build and verification (renders skipped) in {time.monotonic() - t0:.2f}s")
        return verification_report
    print("Rendering high-resolution whole-body Z-buffer and engineering figures...")
    wb_box = [[-210.0, -140.0, -5.0], [210.0, 140.0, 495.0]]

    # Figure 1: 4-Panel Whole-Body Overview (Shells On, Shells Off, Pure 28-Part FDM Skeleton, Dorsal/Rear 3/4 View)
    canvas1 = Image.new("RGB", (2240, 920), "#f4f2ec")
    draw1 = ImageDraw.Draw(canvas1)
    panels1 = [
        ("1. Shells On (Smooth 2-Piece Center + Shoulder/Limb/Head Shells)", render_all_with_shell, [0.55, -0.82, 0.14]),
        ("2. Shells Off (FDM Skeleton + 22 Actuators + 9-Cell Battery & PCBs)", render_no_shell, [0.55, -0.82, 0.14]),
        ("3. Pure 28-Part FDM Structural Skeleton + 316 Steel Sleeves", render_pure_skeleton, [0.55, -0.82, 0.14]),
        ("4. Dorsal / Rear 3/4 View (7 Dorsal Battery Fins + IMU/ADC + Limbs)", render_no_shell, [-0.58, 0.80, 0.15]),
    ]
    for idx, (title, items_list, v_dir) in enumerate(panels1):
        im = raster_meshes(items_list, v_dir, box=wb_box, W=560, H=860)
        canvas1.paste(im, (idx * 560, 50))
        draw1.text((idx * 560 + 16, 16), title, fill="#1f252b")
    fig1_path = RENDERS_DIR / "complete-fdm-body-overview.png"
    canvas1.save(fig1_path)

    # Figure 2: 4-Panel Multi-Pose Whole-Body Articulation (0.0000 mm^3 BRep Overlap Verified)
    def render_pose_items(q_ctrl: dict[str, float], include_actuators: bool = True) -> list[tuple[str, trimesh.Trimesh, list[int]]]:
        T_links = fk_transforms(joints, q_ctrl)
        out_items = []
        for nm, (m_mm, lk, rgba, is_sh, ov_id) in meshes_by_name_mm.items():
            if is_sh:
                continue
            if not include_actuators and ov_id is not None:
                continue
            M = T_links.get(lk, np.eye(4))
            mc = m_mm.copy()
            if not np.allclose(M, np.eye(4)):
                mc.apply_transform(M)
            out_items.append((nm, mc, rgba))
        return out_items

    canvas2 = Image.new("RGB", (2240, 920), "#f4f2ec")
    draw2 = ImageDraw.Draw(canvas2)
    pose_panels = [
        ("1. Walking Stride (Right Lead, 0.0000 mm^3 overlap)", poses[1][1], [0.68, -0.72, 0.14]),
        ("2. Deep Biped Squat (Knee 72 deg, 0.0000 mm^3 overlap)", poses[3][1], [0.65, -0.74, 0.15]),
        ("3. Bilateral Forward Reach (Elbow 70 deg, 0.0000 mm^3)", poses[4][1], [0.58, -0.80, 0.14]),
        ("4. Full Ape Crouch & Gaze Pose (0.0000 mm^3 overlap)", poses[7][1], [-0.62, -0.76, 0.16]),
    ]
    for idx, (title, q_dict, v_dir) in enumerate(pose_panels):
        items_p = render_pose_items(q_dict, include_actuators=True)
        im = raster_meshes(items_p, v_dir, box=[[-240.0, -180.0, -25.0], [240.0, 180.0, 500.0]], W=560, H=860)
        canvas2.paste(im, (idx * 560, 50))
        draw2.text((idx * 560 + 16, 16), title, fill="#1f252b")
    fig2_path = RENDERS_DIR / "complete-fdm-multi-pose-articulation.png"
    canvas2.save(fig2_path)

    # Figure 3: Home-Lab Print-Bed Nesting Layout (256x256 mm Plates & Single 350x350 mm Plate)
    fig3, axes3 = plt.subplots(1, len(plates_256) + 1, figsize=(16.5, 6.2), dpi=160)
    fig3.patch.set_facecolor("#f6f5f0")
    sub_colors = {
        "Center Torso & Electronics": "#3486c6",
        "Neck & Head": "#9262c4",
        "Upper Body / Arms": "#1ca48e",
        "Lower Body (Legs, Hips & Feet)": "#e47644",
    }

    def draw_plate(ax, plate: dict, title: str):
        bw, bh = plate["bed_size_mm"]
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, bw)
        ax.set_ylim(0, bh)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.45, color="#b0b8c0")
        ax.add_patch(patches.Rectangle((8, 8), bw - 16, bh - 16, fill=False, edgecolor="#88929a", linestyle="--", linewidth=1.0))
        for p_item in plate["placements"]:
            col = sub_colors.get(p_item["subsystem"], "#4a90d9")
            rect = patches.FancyBboxPatch(
                (p_item["x_mm"], p_item["y_mm"]),
                p_item["w_mm"],
                p_item["h_mm"],
                boxstyle="round,pad=0.3,rounding_size=2.0",
                facecolor=col,
                edgecolor="#1e242b",
                linewidth=0.8,
                alpha=0.82,
            )
            ax.add_patch(rect)
            short_lbl = (
                p_item["name"]
                .replace("combined__", "")
                .replace("_combined", "")
                .replace("stage1_center_torso_spine_chassis", "Torso_S1")
                .replace("stage2_lower_pelvis_battery_cradle", "Pelvis_S2")
                .replace("stage3_front_power_electronics_carrier", "Power_S3")
                .replace("stage4_rear_imu_adc_carrier", "IMU_S4")
                .replace("head_sensor_compute_carrier", "Head_S4")
                .replace("neck_pitch_to_head_cradle", "Neck_S3")
                .replace("neck_yaw_to_pitch_link", "Neck_S2")
                .replace("neck_root_cradle", "Neck_S1")
                .replace("shoulder_pitch_to_roll", "Sh_P2R")
                .replace("shoulder_roll_to_yaw", "Sh_R2Y")
                .replace("upper_arm_yaw_to_elbow", "UpArm")
                .replace("forearm_elbow_to_wrist", "Forearm")
                .replace("hip_pitch_to_roll", "Hip_P2R")
                .replace("hip_yaw_to_pitch", "Hip_Y2P")
                .replace("ankle_coupler", "Ankle")
                .replace("foot_monocoque", "Foot")
            )
            if p_item["w_mm"] >= 28.0 and p_item["h_mm"] >= 16.0:
                ax.text(
                    p_item["x_mm"] + p_item["w_mm"] * 0.5,
                    p_item["y_mm"] + p_item["h_mm"] * 0.5,
                    f"{short_lbl}\n({p_item['mass_g']:.1f}g)",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#ffffff",
                    fontweight="bold",
                )
        ax.set_title(
            f"{title}\n{plate['parts_count']} parts | {plate['total_polymer_mass_g']:.1f} g | max Z = {plate['max_z_height_mm']:.1f} mm",
            fontsize=10.5,
            fontweight="bold",
            color="#1e242b",
        )
        ax.set_xlabel("Build Plate X (mm)", fontsize=8.5)
        ax.set_ylabel("Build Plate Y (mm)", fontsize=8.5)

    for idx, pl in enumerate(plates_256):
        draw_plate(axes3[idx], pl, f"256x256 mm Bed — Plate {idx + 1}/{len(plates_256)}")
    draw_plate(axes3[-1], plates_350[0], f"350x350 mm Bed — Plate 1/{len(plates_350)}")

    legend_handles = [patches.Patch(facecolor=c, edgecolor="#1e242b", label=lbl) for lbl, c in sub_colors.items()]
    fig3.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=True, fontsize=8.8)
    fig3.suptitle(
        "Complete Cove-50 FDM Ape Body — Home-Lab Build-Plate Nesting Audit (28 Printed Structural Parts)",
        fontsize=13.0,
        fontweight="bold",
        y=0.98,
    )
    fig3.tight_layout(rect=[0.0, 0.07, 1.0, 0.93])
    fig3_path = RENDERS_DIR / "complete-fdm-subsystem-printbeds.png"
    fig3.savefig(fig3_path, dpi=160)
    plt.close(fig3)

    # Figure 4: Whole-Body Mass Reduction, Sleeve Count & FEM Safety Factor Rollup Chart
    fig4, (ax_m, ax_sf) = plt.subplots(1, 2, figsize=(16.5, 6.4), dpi=160)
    fig4.patch.set_facecolor("#f6f5f0")

    labels = [
        "Lower Body\n(12 parts, 128 slv)",
        "Arms\n(8 parts, 68 slv)",
        "Neck & Head\n(4 parts, 34 slv)",
        "Center Torso\n(4 parts, 86 slv)",
        "Whole-Body Total\n(28 parts, 316 slv)",
    ]
    legacy_vals = [332.78, 136.05, 52.67, 251.82, 773.32]
    fdm_poly_vals = [157.20, 59.30, 21.84, 164.08, 402.42]
    fdm_slv_vals = [30.44, 15.44, 6.72, 14.13, 66.73]

    x = np.arange(len(labels))
    w = 0.34
    ax_m.set_facecolor("#ffffff")
    b1 = ax_m.bar(x - w / 2, legacy_vals, w, label="Legacy Sheet-Metal / LPBF / PA12 Baseline (g)", color="#8c96a0", edgecolor="#222")
    b2 = ax_m.bar(x + w / 2, fdm_poly_vals, w, label="FDM Polymer Mass (PAHT-CF, g)", color="#2b82c9", edgecolor="#222")
    b3 = ax_m.bar(x + w / 2, fdm_slv_vals, w, bottom=fdm_poly_vals, label="Pressed-In 304 SS Sleeves Mass (g)", color="#c8d2dc", edgecolor="#222")

    for i in range(len(labels)):
        tot = fdm_poly_vals[i] + fdm_slv_vals[i]
        pct = (tot - legacy_vals[i]) / legacy_vals[i] * 100.0
        ax_m.text(x[i] - w / 2, legacy_vals[i] + 10, f"{legacy_vals[i]:.1f}g", ha="center", fontsize=8.0, color="#333")
        ax_m.text(x[i] + w / 2, tot + 10, f"{tot:.1f}g\n({pct:.1f}%)", ha="center", fontsize=8.0, fontweight="bold", color="#165082")

    ax_m.set_xticks(x)
    ax_m.set_xticklabels(labels, fontsize=8.5)
    ax_m.set_ylabel("Structural Mass (g)", fontsize=10.0, fontweight="bold")
    ax_m.set_ylim(0, 890)
    ax_m.grid(axis="y", linestyle=":", alpha=0.5)
    ax_m.legend(loc="upper left", fontsize=8.5)
    ax_m.set_title("Whole-Body Structural Mass Rollup: Legacy Metal vs Selected 28-Part FDM + 316 Sleeves", fontsize=11.0, fontweight="bold")

    # Right panel: Minimum PAHT-CF (Dry) 3D Tetrahedral FEM Safety Factor across all 18 structural stages
    stage_names = [
        "Leg S1 Shin", "Leg S2 Ankle", "Leg S3 Foot", "Leg S4 Thigh", "Leg S5 Hip P2R", "Leg S6 Hip Y2P",
        "Arm S1 Sh P2R", "Arm S2 Sh R2Y", "Arm S3 UpArm", "Arm S4 Forearm",
        "Neck S1 Root", "Neck S2 Y2P", "Neck S3 P2Head", "Head S4 Carrier",
        "Torso S1 Spine", "Pelvis S2 Lid", "Power S3 Frame", "IMU/ADC S4",
    ]
    stage_sfs = [
        15.91, 3.63, 29.36, 5.22, 5.96, 2.86,
        10.03, 1.65, 2.10, 7.48,
        3.22, 2.34, 4.14, 5.14,
        1.55, 66.7, 4.24, 26.3,
    ]
    stage_sfs_clipped = [min(sf, 18.0) for sf in stage_sfs]
    y_pos = np.arange(len(stage_names))
    bar_cols = (
        ["#e47644"] * 6
        + ["#1ca48e"] * 4
        + ["#9262c4"] * 4
        + ["#3486c6"] * 4
    )
    ax_sf.set_facecolor("#ffffff")
    ax_sf.barh(y_pos, stage_sfs_clipped, color=bar_cols, edgecolor="#222", height=0.68)
    ax_sf.axvline(1.0, color="#c9302c", linestyle="--", linewidth=1.4, label="1.0x Yield/Layer Limit")
    ax_sf.axvline(1.5, color="#d58512", linestyle=":", linewidth=1.2, label="1.5x Target Margin")
    for idx, (raw_sf, clip_sf) in enumerate(zip(stage_sfs, stage_sfs_clipped)):
        lbl_str = f"{raw_sf:.2f}x" if raw_sf <= 18.0 else f"{raw_sf:.1f}x (off-scale)"
        ax_sf.text(clip_sf + 0.25, idx, lbl_str, va="center", fontsize=7.8, fontweight="bold", color="#1e242b")
    ax_sf.set_yticks(y_pos)
    ax_sf.set_yticklabels(stage_names, fontsize=8.2)
    ax_sf.invert_yaxis()
    ax_sf.set_xlim(0, 21.5)
    ax_sf.set_xlabel("3D Tetrahedral FEM Safety Factor (PAHT-CF Dry @ Peak Stall / Fall Load)", fontsize=9.5, fontweight="bold")
    ax_sf.grid(axis="x", linestyle=":", alpha=0.5)
    ax_sf.legend(loc="lower right", fontsize=8.5)
    ax_sf.set_title("All 18 Structural Stages — 3D Tetrahedral FEM Safety Factors (Min = 1.55x)", fontsize=11.0, fontweight="bold")

    fig4.tight_layout()
    fig4_path = RENDERS_DIR / "complete-fdm-mass-and-strength-rollup.png"
    fig4.savefig(fig4_path, dpi=160)
    plt.close(fig4)

    print(f"Completed Complete Cove-50 FDM Ape Body build, verification & renders in {time.monotonic() - t0:.2f}s")
    return verification_report


if __name__ == "__main__":
    build_complete_assembly()
