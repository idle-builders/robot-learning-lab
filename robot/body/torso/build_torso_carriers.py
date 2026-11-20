"""Step 3 — Blank-Slate 3D-Printed Honeycomb Torso & Electronics Carrier Builder (Family 1: truss_box, Family 2: topopt)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from build123d import Box, Cylinder, Pos, Rot, export_step, export_stl

ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = ROOT / "robot/body/torso"
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import common


def build_stage1_center_torso_spine_chassis(family: str):
    """Build Stage 1 from scratch: 7-Cell Hexagonal Honeycomb Torso Spine, Hip-Yaw Cradles & Shoulder/Neck Tower."""
    upper_cells = [(-44.8, 25.6), (-22.4, 25.6), (0.0, 25.6), (22.4, 25.6), (44.8, 25.6), (-11.2, 6.2), (11.2, 6.2)]

    # 1. 7-Cell Vertical Hexagonal Honeycomb Magazine Core (Z = [266.8, 341.6] mm, continuous solid-web 22.4 mm hex lattice):
    parts_s1 = [Pos(x, y, 304.2) * Cylinder(12.65, 74.8) for x, y in upper_cells]
    parts_s1.extend([
        # Solid continuous interstitial honeycomb webs filling all triangular voids between the 5 dorsal & 2 core battery tubes (zero vertical channels/gaps):
        Pos(0.0, 15.9, 304.2) * Box(44.8, 19.4, 74.8),
        Pos(0.0, 25.6, 304.2) * Box(89.6, 12.0, 74.8),
        Pos(0.0,  6.2, 304.2) * Box(22.4, 12.0, 74.8),
        # 2x Integral Lower-Battery Top-Cap Plugs on the underside of Stage 1 (X = ±11.2, Y = 14.8, Z = [265.65, 268.35] mm)
        # that drop into the top of Stage 2's lower battery tubes and cap Cell_08 & Cell_09 (Z_top = 265.575 mm):
        Pos(-11.2, 14.8, 267.00) * Cylinder(10.75, 2.70),
        Pos( 11.2, 14.8, 267.00) * Cylinder(10.75, 2.70),
    ])

    # 2. Symmetric Bilateral Hip-Yaw C047 Mounting Cradles Flush on Build Plate (Z = [266.8, 338.0] mm):
    parts_s1.extend([
        # Left & Right Hip-Yaw Saddle Rails flush with Z=266.8 build plate (Z = [266.8, 276.0], |X| = [20.0, 52.5], Y = [-10.2, 16.5]):
        Pos(-36.25, 3.15, 271.4) * Box(32.5, 26.7, 9.2),
        Pos( 36.25, 3.15, 271.4) * Box(32.5, 26.7, 9.2),
        # Inboard Hip-Yaw vertical cheek walls connecting saddle rails to central honeycomb:
        Pos(-21.2, 3.15, 291.4) * Box(3.4, 24.0, 49.2),
        Pos( 21.2, 3.15, 291.4) * Box(3.4, 24.0, 49.2),
        # Rear Hip-Yaw cradle webs at Y = [14.5, 18.5], |X| = [20.0, 47.0], Z = [266.8, 316.0]:
        Pos(-33.5, 16.5, 291.4) * Box(27.0, 4.0, 49.2),
        Pos( 33.5, 16.5, 291.4) * Box(27.0, 4.0, 49.2),
        # Upper Hip-Yaw arch ribs at Z = [313.0, 317.5], |X| = [20.0, 45.0], Y = [4.0, 18.5]:
        Pos(-32.5, 11.25, 315.25) * Box(25.0, 14.5, 4.5),
        Pos( 32.5, 11.25, 315.25) * Box(25.0, 14.5, 4.5),
        # 45-deg self-supporting diagonal gussets from Hip-Yaw arches up to Shoulder-Pitch saddles:
        Pos(-26.0, 18.5, 329.0) * Rot(0, 26, 0) * Box(13.0, 4.6, 26.0),
        Pos( 26.0, 18.5, 329.0) * Rot(0, -26, 0) * Box(13.0, 4.6, 26.0),
        # Symmetric flat front vertical stiffening ribs outboard of Cell_06 & Cell_07 vertical fin slots, flush on Z=266.8 build plate (|X| = [14.6, 19.4], Y = [-6.10, 3.0], Z = [266.8, 341.6]):
        Pos(-17.0, -1.55, 304.2) * Box(4.8, 9.10, 74.8),
        Pos( 17.0, -1.55, 304.2) * Box(4.8, 9.10, 74.8),
        # 4x Dorsal valley waist-clevis & battery-lid anchor blocks between Cell_01..05, flush on Z=266.8 build plate (X = ±33.6, ±11.2, Y = [28.0, 35.40], Z = [266.8, 277.2]):
        Pos(-33.6, 31.70, 272.0) * Box(5.6, 7.40, 10.4),
        Pos(-11.2, 31.70, 272.0) * Box(5.6, 7.40, 10.4),
        Pos( 11.2, 31.70, 272.0) * Box(5.6, 7.40, 10.4),
        Pos( 33.6, 31.70, 272.0) * Box(5.6, 7.40, 10.4),
        # 3x Dorsal honeycomb valley mounting rails for Stage 4 IMU/ADC carrier between Cell_01/02/03/04 (X = -33.6, -11.2, +11.2, Y = [30.20, 35.40], Z = [295.0, 340.0]):
        Pos(-33.6, 32.80, 317.5) * Box(5.6, 5.20, 45.0),
        Pos(-11.2, 32.80, 317.5) * Box(5.6, 5.20, 45.0),
        Pos( 11.2, 32.80, 317.5) * Box(5.6, 5.20, 45.0),
        # 2x Lower Dorsal Center-Shell Bosses (center_back_mount1_M3x6 & center_back_mount3_M3x6 at X = ±49.90 mm, Z = 275.46 mm, mating face Y = 38.20 mm, Z = [270.50, 279.50] mm):
        Pos(-49.90, 31.90, 275.00) * Box(5.40, 12.60, 9.00),
        Pos( 49.90, 31.90, 275.00) * Box(5.40, 12.60, 9.00),
    ])

    # 3. Symmetric Upper Shoulder-Pitch C047 Buttress Towers, Dual Arch Bridges & Neck Root Cradle Seat (Z = [339.0, 387.5] mm):
    parts_s1.extend([
        # Left & Right Shoulder Pitch Mounting Cheeks at |X| = [28.45, 32.35], Y = [-3.2, 21.2], Z = [339.0, 387.5]:
        Pos(-30.4, 9.0, 363.25) * Box(3.9, 24.4, 48.5),
        Pos( 30.4, 9.0, 363.25) * Box(3.9, 24.4, 48.5),
        # Upper Inter-Row & Ventral Box-Arch Bridges connecting Left & Right Shoulder Cheeks across X = [-30.0, +30.0] along the solid honeycomb walls (leaving all cell tops open):
        Pos(0.0, 15.90, 343.25) * Box(60.0, 4.6, 8.5),
        Pos(0.0, -1.20, 343.25) * Box(60.0, 3.8, 8.5),
        # 3D-Printed Tapered Buttress Gussets rising from the Arch Bridge up the Shoulder Cheeks (Y = [9.0, 15.0], Z = [343.0, 362.0]):
        Pos(-25.5, 12.0, 352.5) * Rot(0, -12, 0) * Box(5.6, 6.0, 19.0),
        Pos( 25.5, 12.0, 352.5) * Rot(0,  12, 0) * Box(5.6, 6.0, 19.0),
        # Root U-channel flanges stiffening the Shoulder-Pitch cheek roots (Z = [339.0, 355.5], |X| = [26.0, 32.35]):
        Pos(-29.175, -0.85, 347.25) * Box(6.35, 4.7, 16.5),
        Pos( 29.175, -0.85, 347.25) * Box(6.35, 4.7, 16.5),
        Pos(-29.175, 18.85, 347.25) * Box(6.35, 4.7, 16.5),
        Pos( 29.175, 18.85, 347.25) * Box(6.35, 4.7, 16.5),
        # Right Neck Root XM335 Cheek + Shoulder-to-Neck Cross Buttress Web along Y = 16.20 mm + Upper Dorsal Torso Shell Boss (T02_back_mount2 at X=+10.10, Y_mate=36.00, Z=377.46 mm):
        Pos(11.5, 22.8, 352.5) * Box(3.9, 14.8, 25.0),
        Pos(20.75, 16.20, 345.75) * Box(18.5, 4.0, 11.5),
        Pos(11.50, 29.00, 367.50) * Box(3.90, 9.00, 15.0),
        Pos(10.10, 34.075, 375.50) * Box(7.00, 3.85, 9.80),
        # Left Neck Root Cradle Spine Anchor Block connected to Left Shoulder Cheek (Y = [25.20, 29.90], Z = [341.0, 351.5]):
        Pos(-23.0, 27.55, 346.25) * Box(19.0, 4.7, 10.5),
    ])

    s1 = parts_s1[0].fuse(*parts_s1[1:])

    # 4. Precision clearance cuts for 7 Bottom-Loadable Honeycomb Battery Cells, Full-Height Fin-Exposure Slots, Hip Yaw Servos & Horns, and Sleeved Bores:
    cuts_s1 = [
        # Full-width dorsal fin-exposure plane at Y >= 35.85 mm across Z = [280.5, 352.0] mm (behind Stage 4, above lower Z <= 279.5 bosses and below upper Z >= 360.0 boss)
        # + 5x bottom-open 6.80 mm fin-slot channels through Z = [256.5, 281.5] mm (clearing the 6.00 mm fins while preserving the X=±28.0 and X=±49.9 rear mounts):
        Pos(  0.0, 42.0, 316.25) * Box(120.0, 12.3, 71.5),
        Pos(-44.8, 42.0, 269.00) * Box(6.8, 12.3, 25.0),
        Pos(-22.4, 42.0, 269.00) * Box(6.8, 12.3, 25.0),
        Pos(  0.0, 42.0, 269.00) * Box(6.8, 12.3, 25.0),
        Pos( 22.4, 42.0, 269.00) * Box(6.8, 12.3, 25.0),
        Pos( 44.8, 42.0, 269.00) * Box(6.8, 12.3, 25.0),
        # Full-height front (-Y) vertical fin-exposure window slots on Core Cell_06 & Cell_07 (Y <= 6.2 - 10.25 = -4.05 mm, |X| <= 14.6 mm, open through Z = 266.80 mm bottom):
        Pos(0.0, -9.05, 304.2) * Box(29.2, 10.0, 95.0),
        # Bilateral Hip Yaw C047 servo case pockets (|X| >= 23.85, Y = [-10.5, 14.45], Z = [276.1, 312.7]):
        Pos(-47.0, 1.975, 294.4) * Box(46.3, 24.95, 36.6),
        Pos( 47.0, 1.975, 294.4) * Box(46.3, 24.95, 36.6),
        # Back shell lower-outer chamfer clearance below Z = 270.4 mm (strictly below T02_back_mount1/3 bosses at Z = [270.5, 279.5]):
        Pos(-52.0, 35.0, 265.4) * Rot(0, 0, -25) * Box(20.0, 10.0, 10.0),
        Pos( 52.0, 35.0, 265.4) * Rot(0, 0,  25) * Box(20.0, 10.0, 10.0),
        # Bilateral Hip Yaw rotary horn clearance cylinders at (±59.0, 2.0):
        Pos(-59.0, 2.0, 271.0) * Cylinder(11.2, 20.0),
        Pos( 59.0, 2.0, 271.0) * Cylinder(11.2, 20.0),
        # 45-deg self-supporting upper corner bevels on Shoulder Pitch cheeks (Z = 387.5, Y = -3.2 / +21.2):
        Pos(-30.4, -3.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
        Pos( 30.4, -3.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
        Pos(-30.4, 21.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
        Pos( 30.4, 21.2, 387.5) * Rot(45, 0, 0) * Box(12.0, 4.5, 4.5),
    ]
    for x, y in upper_cells:
        # Bottom-open slide-in cell pocket (r=10.95, h=76.0 centered at Z=301.8 -> Z=[263.8, 339.8], 100% open at bottom Z=266.80 mm)
        # + r=8.80 mm upper axial stop ring (Z=[339.8, 341.6]) and unobstructed top terminal/vent chimney:
        cuts_s1.append(Pos(x, y, 301.8) * Cylinder(10.95, 76.0))
        if abs(x - (-22.4)) < 0.5 and abs(y - 25.6) < 0.5:
            # Cell_02 sits under the Left Neck Root M3 anchor block (Y=[25.2, 29.9], Z=[341.0, 351.5]); cut r=8.80 up to Z=341.0 and front terminal chimney at Y=21.2:
            cuts_s1.append(Pos(x, y, 303.5) * Cylinder(8.80, 75.0))
            cuts_s1.append(Pos(x, 21.2, 348.0) * Cylinder(3.80, 18.0))
        else:
            cuts_s1.append(Pos(x, y, 316.0) * Cylinder(8.80, 96.0))

    specs1 = common.stage_interface_specs()["stage1_center_torso_spine_chassis"]
    for b in specs1["support_bores"] + specs1["load_bores"] + specs1.get("aux_bores", []):
        cuts_s1.append(common.aligned_cylinder(b["center"], b["normal"], b["r_od"] + 0.04, 16.0))

    if family == "truss_box":
        for x_sh in [-30.4, 30.4]:
            cuts_s1.append(Pos(x_sh, 9.0, 372.5) * Rot(0, 90, 0) * Cylinder(5.2, 8.0))
        r_in, r_out = 0.60, 0.22
    else:
        for x_sh in [-30.4, 30.4]:
            cuts_s1.append(Pos(x_sh, 9.0, 372.5) * Rot(0, 90, 0) * Cylinder(5.8, 8.0))
            cuts_s1.append(Pos(x_sh, 9.0, 350.0) * Rot(0, 90, 0) * Cylinder(4.2, 8.0))
        r_in, r_out = 0.68, 0.25

    s1 = s1.cut(*cuts_s1)
    body = max(s1.solids(), key=lambda x: x.volume)
    return body, {"r_inner_mm": r_in, "r_outer_mm": r_out}


def build_stage2_lower_pelvis_battery_cradle(family: str):
    """Build Stage 2 from scratch: 2-Cell Lower Pelvis Honeycomb Column + Integral 7-Cell Upper Battery Retention Lid & 6-Bolt Waist U-Clevis."""
    lower_cells = [(-11.2, 14.8), (11.2, 14.8)]
    upper_dorsal_cells = [(-44.8, 25.6), (-22.4, 25.6), (0.0, 25.6), (22.4, 25.6), (44.8, 25.6)]
    upper_core_cells = [(-11.2, 6.2), (11.2, 6.2)]

    # 1. 2-Cell Lower Pelvis Honeycomb Column (Z = [194.50, 266.75] mm, flush up to 0.05 mm below Stage 1's Z=266.80 build-plate base):
    parts_s2 = [Pos(x, y, 230.625) * Cylinder(12.5, 72.25) for x, y in lower_cells]
    parts_s2.extend([
        # Solid inter-cell honeycomb bridge between Cell_08 & Cell_09 (zero gaps/channels):
        Pos(0.0, 14.8, 230.625) * Box(22.4, 12.0, 72.25),
        # Central dorsal spine column at X=0, Y=[23.6, 28.0], Z=[194.50, 266.75] (flush on Z=194.5 build plate):
        Pos(0.0, 25.8, 230.625) * Box(9.0, 4.4, 72.25),
        # Symmetric side ribs & curved front Hip-Pitch Waist Guard at Y=[-10.1, 4.0], X=[-19.4, 19.4], Z=[216.0, 266.75]:
        Pos(-17.0, -2.80, 241.375) * Box(4.8, 13.6, 50.75),
        Pos( 17.0, -2.80, 241.375) * Box(4.8, 13.6, 50.75),
        Pos(0.0, -8.5, 237.0) * Box(38.8, 3.2, 42.0),
        # 2x Front Pelvis Shell-Connector Mounting Bosses (X = ±16.0 mm, Y = [-12.00, -4.00] mm, Z = [216.0, 256.0] mm) mating flush to pelvis_front_carrier at Y = -12.00 mm:
        Pos(-16.0, -8.0, 236.0) * Box(5.6, 8.0, 40.0),
        Pos( 16.0, -8.0, 236.0) * Box(5.6, 8.0, 40.0),
        # Zone A — 2x Front Ventral Upward Clevis Lugs lapping over Stage 1's front pillars (|X| = [14.6, 19.4], Y = [-9.60, -6.15], Z = [216.0, 275.5]):
        Pos(-17.0, -7.875, 245.75) * Box(4.8, 3.45, 59.5),
        Pos( 17.0, -7.875, 245.75) * Box(4.8, 3.45, 59.5),
        # Integral Upper 7-Cell Battery-Retaining Lid (Z = [263.75, 266.75] mm) closing the bottom of Stage 1's 5 dorsal + 2 core cells:
        Pos(0.0, 26.8, 265.25) * Box(89.6, 16.8, 3.0),
        Pos(0.0,  6.2, 265.25) * Box(38.8, 16.0, 3.0),
        # Self-supporting 45-deg diagonal dorsal wing buttresses rising from the central pelvis column (|X|=19, Z=245.5) up to the outer lid wings (|X|=39, Z=265.5):
        Pos(-29.0, 26.5, 255.0) * Rot(0,  45, 0) * Box(27.0, 8.5, 4.0),
        Pos( 29.0, 26.5, 255.0) * Rot(0, -45, 0) * Box(27.0, 8.5, 4.0),
        # Zone B — 4x Rear Dorsal Valley Upward Clevis Lugs (X = ±33.6, ±11.2, Y = [35.45, 38.45], Z = [263.75, 275.5]) + full-width dorsal tie bar:
        Pos(0.0, 31.70, 259.375) * Box(9.0, 13.5, 14.75),
        Pos(0.0, 36.95, 265.25) * Box(72.8, 3.00, 3.0),
        Pos(-33.6, 36.95, 269.625) * Box(5.6, 3.00, 11.75),
        Pos(-11.2, 36.95, 269.625) * Box(5.6, 3.00, 11.75),
        Pos( 11.2, 36.95, 269.625) * Box(5.6, 3.00, 11.75),
        Pos( 33.6, 36.95, 269.625) * Box(5.6, 3.00, 11.75),
    ])
    for ux, uy in upper_dorsal_cells + upper_core_cells:
        # Lower lid flange disk (Z = [263.75, 266.75]) + upward cell-retaining plug ring (r=10.60, Z = [266.75, 268.85]) inside Stage 1's r=10.95 bottom bore:
        parts_s2.append(Pos(ux, uy, 265.25) * Cylinder(12.20, 3.0))
        parts_s2.append(Pos(ux, uy, 267.80) * Cylinder(10.60, 2.10))

    s2 = parts_s2[0].fuse(*parts_s2[1:])

    cuts_s2 = [
        # Full-height rear (+Y) vertical fin-exposure window slots on Lower Pelvis Cell_08 & Cell_09 (Y >= 14.8 + 10.25 = 25.05 mm, Y <= 33.05 mm, strictly below Z=263.75 lid):
        Pos(-11.2, 29.05, 229.0) * Box(13.0, 8.0, 69.5),
        Pos( 11.2, 29.05, 229.0) * Box(13.0, 8.0, 69.5),
        # Also keep rear fin-slot clearance at the top lid for Stage 1 dorsal cells (Y >= 35.85 mm outside the 4 clevis lugs X = ±33.6, ±11.2) and Stage 1 core cells (Y <= -4.05 mm):
        Pos(-44.8, 39.0, 266.5) * Box(14.0, 6.3, 10.0),
        Pos(-22.4, 39.0, 266.5) * Box(14.0, 6.3, 10.0),
        Pos(  0.0, 39.0, 266.5) * Box(14.0, 6.3, 10.0),
        Pos( 22.4, 39.0, 266.5) * Box(14.0, 6.3, 10.0),
        Pos( 44.8, 39.0, 266.5) * Box(14.0, 6.3, 10.0),
        Pos(  0.0, -9.05, 266.5) * Box(29.2, 10.0, 10.0),
        # Back shell lower-outer chamfer clearance at |X| = 52 mm:
        Pos(-52.0, 35.0, 272.0) * Rot(0, 0, -25) * Box(20.0, 10.0, 20.0),
        Pos( 52.0, 35.0, 272.0) * Rot(0, 0,  25) * Box(20.0, 10.0, 20.0),
        # Bilateral Hip Yaw bracket clearance (|X| >= 48.9, Y <= 19.1, Z <= 267.0):
        Pos(-63.0, 2.0, 246.0) * Box(28.2, 34.2, 42.0),
        Pos( 63.0, 2.0, 246.0) * Box(28.2, 34.2, 42.0),
        # 45-deg self-supporting underside bevel on the central dorsal spine flare (Z = 251.5, Y = 38.45, X in [-5.0, +5.0] strictly below Z=261.4):
        Pos(0.0, 38.45, 251.5) * Rot(-45, 0, 0) * Box(10.0, 14.0, 14.0),
        # Bilateral Hip Pitch C047 servo case clearance (|X| >= 21.1, Y = [-10.5, 14.5], Z = [208.5, 254.5]):
        Pos(-39.6, 2.0, 231.5) * Box(37.0, 26.0, 47.0),
        Pos( 39.6, 2.0, 231.5) * Box(37.0, 26.0, 47.0),
        # Central harness hollow between front waist guard and 2-cell honeycomb (strictly below Z=260.0):
        Pos(0.0, -1.8, 237.0) * Box(25.6, 10.2, 46.0),
        # 45-deg self-supporting lower & upper chamfers on front Hip-Pitch Waist Guard & Front Pelvis Shell Bosses:
        Pos(0.0, -11.5, 216.0) * Rot(45, 0, 0) * Box(44.0, 6.0, 6.0),
        Pos(0.0, -10.1, 258.0) * Rot(45, 0, 0) * Box(25.6, 6.0, 6.0),
    ]
    for ux, uy in upper_dorsal_cells + upper_core_cells:
        # r = 8.80 mm (17.6 mm ID) circular bottom ventilation & terminal opening through the Stage 2 upper battery-retaining lid:
        cuts_s2.append(Pos(ux, uy, 266.5) * Cylinder(8.80, 10.0))
    for x, y in lower_cells:
        # Top-open slide-in cell pocket (r=10.95, h=75.2 centered at Z=233.0 -> Z=[195.40, 270.60], 100% open at top)
        # + r=8.80 mm lower floor axial stop ring (Z=[194.50, 195.40]) and bottom terminal/ventilation opening:
        cuts_s2.append(Pos(x, y, 233.00) * Cylinder(10.95, 75.2))
        cuts_s2.append(Pos(x, y, 230.625) * Cylinder(8.80, 76.0))

    specs2 = common.stage_interface_specs()["stage2_lower_pelvis_battery_cradle"]
    for b in specs2["support_bores"] + specs2["load_bores"] + specs2.get("aux_bores", []):
        cuts_s2.append(common.aligned_cylinder(b["center"], b["normal"], b["r_od"] + 0.04, 14.0))

    if family == "truss_box":
        for z_w in [224.0, 237.0, 250.0]:
            cuts_s2.append(Pos(0.0, -8.5, z_w) * Rot(90, 0, 0) * Cylinder(4.5, 8.0))
        r_in, r_out = 0.55, 0.20
    else:
        for z_w in [226.0, 248.0]:
            cuts_s2.append(Pos(-6.5, -8.5, z_w) * Rot(90, 0, 0) * Cylinder(4.2, 8.0))
            cuts_s2.append(Pos( 6.5, -8.5, z_w) * Rot(90, 0, 0) * Cylinder(4.2, 8.0))
        r_in, r_out = 0.62, 0.22

    s2 = s2.cut(*cuts_s2)
    body = max(s2.solids(), key=lambda x: x.volume)
    return body, {"r_inner_mm": r_in, "r_outer_mm": r_out}


def build_stage3_front_power_electronics_carrier(family: str):
    """Build Stage 3 from scratch: 3D-Printed Dual-Deck Front Power Supervisor & Precharge Frame with Integral Rearward Standoff Brackets to Stage 1."""
    rear_plate = (Pos(18.25, -14.505, 319.0) * Box(79.5, 1.91, 97.0)).cut(
        Pos(18.5, -14.5, 322.0) * Box(52.0, 4.0, 68.0),
        Pos(-19.5, -14.5, 284.0) * Box(6.0, 4.0, 20.0),
        Pos(-19.5, -14.5, 355.0) * Box(6.0, 4.0, 26.0),
        Pos( 56.0, -14.5, 284.0) * Box(6.0, 4.0, 20.0),
        Pos( 56.0, -14.5, 355.0) * Box(6.0, 4.0, 26.0),
    )
    # Vertical structural spine rails on rear_plate aligned with Stage 1's flat front ribs at X = -17.0 & +17.0 mm:
    spine_rails = [
        Pos(-17.0, -14.505, 313.0) * Box(5.4, 1.91, 62.0),
        Pos( 17.0, -14.505, 319.0) * Box(5.4, 1.91, 97.0),
    ]
    if family == "truss_box":
        rib_d1 = (Pos(18.5, -14.505, 322.0) * Rot(0, 38, 0) * Box(78.0, 1.91, 4.2)) & (Pos(18.25, -14.505, 319.0) * Box(76.0, 1.91, 92.0))
        rib_d2 = (Pos(18.5, -14.505, 322.0) * Rot(0, -38, 0) * Box(78.0, 1.91, 4.2)) & (Pos(18.25, -14.505, 319.0) * Box(76.0, 1.91, 92.0))
        rear_frame = rear_plate.fuse(*spine_rails, rib_d1, rib_d2)
        r_in, r_out = 0.52, 0.20
    else:
        rib_h = Pos(18.5, -14.505, 313.0) * Box(72.0, 1.91, 5.0)
        hub = Pos(17.0, -14.505, 313.0) * Rot(90, 0, 0) * Cylinder(9.0, 1.91)
        rear_frame = rear_plate.fuse(*spine_rails, rib_h, hub).cut(Pos(17.0, -14.5, 313.0) * Rot(90, 0, 0) * Cylinder(4.5, 4.0))
        r_in, r_out = 0.60, 0.22

    # board bosses reach the PCB face (0.00 contact; before: 0.51 / 1.74 mm short of it)
    ps_len = common.STAGE3_PS_BOSS_Y[1] - common.STAGE3_PS_BOSS_Y[0]
    ps_bosses = [
        Pos(x, common.STAGE3_PS_BOSS_Y[2], z) * Rot(90, 0, 0) * Cylinder(2.8, ps_len)
        for x, z in ((48.5, 280.8), (-11.5, 280.8), (48.0, 364.0), (-11.0, 364.0))
    ]
    # 4x Integral Rearward Standoff Brackets built into Stage 3 (X = ±17.0 mm, Z = 288.0 & 338.0 mm, Y = [-15.46, -6.15] mm)
    # + vertical connecting webs along Y = [-15.46, -11.50] mm, mating flush onto Stage 1's flat Y = -6.10 mm front mounts:
    rearward_core_brackets = [
        Pos(-17.0, -10.805, 288.0) * Box(4.8, 9.31, 8.0),
        Pos(-17.0, -10.805, 338.0) * Box(4.8, 9.31, 8.0),
        Pos( 17.0, -10.805, 288.0) * Box(4.8, 9.31, 8.0),
        Pos( 17.0, -10.805, 338.0) * Box(4.8, 9.31, 8.0),
        Pos(-17.0, -13.48,  313.0) * Box(4.0, 3.96, 54.0),
        Pos( 17.0, -13.48,  313.0) * Box(4.0, 3.96, 54.0),
    ]
    # Front Precharge Deck (Y = [-30.64, -28.72] mm) with top screwdriver access notches at (X = ±17.0, Z = 338.0 mm):
    front_plate = (Pos(18.25, -29.68, 316.5) * Box(79.5, 1.92, 38.0)).cut(
        Pos( 18.5, -29.68, 316.5) * Box(58.0, 4.0, 26.0),
        Pos(-17.0, -29.68, 338.0) * Rot(90, 0, 0) * Cylinder(3.6, 5.0),
        Pos( 17.0, -29.68, 338.0) * Rot(90, 0, 0) * Cylinder(3.6, 5.0),
    )
    mp_len = common.STAGE3_MP_BOSS_Y[1] - common.STAGE3_MP_BOSS_Y[0]
    mp_bosses = [
        Pos(x, common.STAGE3_MP_BOSS_Y[2], z) * Rot(90, 0, 0) * Cylinder(2.8, mp_len)
        for x, z in ((-7.0, 332.0), (37.0, 301.0))
    ]
    side_bridges = [
        Pos(-18.0, -22.1, 316.5) * Box(5.0, 15.2, 30.0),
        Pos( 55.0, -22.1, 316.5) * Box(5.0, 15.2, 30.0),
    ]
    s3 = rear_frame.fuse(*ps_bosses, *rearward_core_brackets, front_plate, *mp_bosses, *side_bridges)

    cuts3 = [
        # Rear nut-access windows in rear_frame coaxial with the 2 Main Precharge PCB mounts (-7.0, 332.0) and (37.0, 301.0):
        Pos(-7.0, -14.5, 332.0) * Rot(90, 0, 0) * Cylinder(4.2, 5.0),
        Pos(37.0, -14.5, 301.0) * Rot(90, 0, 0) * Cylinder(4.2, 5.0),
    ]
    specs3 = common.stage_interface_specs()["stage3_front_power_electronics_carrier"]
    for b in specs3["support_bores"] + specs3["load_bores"]:
        cuts3.append(common.aligned_cylinder(b["center"], b["normal"], b["r_od"] + 0.04, b["length"] + 6.0))

    s3 = s3.cut(*cuts3)
    body = max(s3.solids(), key=lambda x: x.volume)
    return body, {"r_inner_mm": r_in, "r_outer_mm": r_out}


def build_stage4_rear_imu_adc_carrier(family: str):
    """Build Stage 4 from scratch: 3D-Printed Dorsal Body-IMU & Foot-ADC Sensor Carrier Plate with 3x Honeycomb Valley Standoff Rails."""
    # Main dorsal sensor carrier plate (X = [-36.4, +14.0], Y = [39.10, 41.90], Z = [293.5, 350.5])
    # + 3x full-height dorsal honeycomb valley standoff rails at X = -33.6, -11.2, +11.2 mm (Y = [35.45, 41.90], Z = [295.0, 340.0], flush with Stage 1 at Y=35.40):
    s4 = (Pos(-11.2, 40.50, 322.0) * Box(50.4, 2.8, 57.0)).fuse(
        Pos(-33.6, 38.675, 317.5) * Box(5.6, 6.45, 45.0),
        Pos(-11.2, 38.675, 317.5) * Box(5.6, 6.45, 45.0),
        Pos( 11.2, 38.675, 317.5) * Box(5.6, 6.45, 45.0),
    )
    cuts4 = []
    specs4 = common.stage_interface_specs()["stage4_rear_imu_adc_carrier"]
    for b in specs4["support_bores"]:
        cuts4.append(common.aligned_cylinder(b["center"], b["normal"], b["r_od"] + 0.04, 12.0))
    for b in specs4["load_bores"]:
        cuts4.append(common.aligned_cylinder(b["center"], b["normal"], b["r_od"] + 0.04, 12.0))
        cuts4.append(Pos(b["center"][0], 37.0, b["center"][2]) * Rot(90, 0, 0) * Cylinder(2.3, 3.0))

    if family == "truss_box":
        cuts4.extend([
            Pos(-22.4, 40.50, 321.5) * Rot(90, 0, 0) * Cylinder(3.8, 6.0),
            Pos(  0.0, 40.50, 321.5) * Rot(90, 0, 0) * Cylinder(3.8, 6.0),
        ])
        r_in, r_out = 0.55, 0.22
    else:
        cuts4.extend([
            Pos(-22.4, 40.50, 321.5) * Rot(90, 0, 0) * Cylinder(4.4, 6.0),
            Pos(  0.0, 40.50, 321.5) * Rot(90, 0, 0) * Cylinder(4.4, 6.0),
        ])
        r_in, r_out = 0.62, 0.24

    s4 = s4.cut(*cuts4)
    body = max(s4.solids(), key=lambda x: x.volume)
    return body, {"r_inner_mm": r_in, "r_outer_mm": r_out}


BUILDERS = {
    "stage1_center_torso_spine_chassis": build_stage1_center_torso_spine_chassis,
    "stage2_lower_pelvis_battery_cradle": build_stage2_lower_pelvis_battery_cradle,
    "stage3_front_power_electronics_carrier": build_stage3_front_power_electronics_carrier,
    "stage4_rear_imu_adc_carrier": build_stage4_rear_imu_adc_carrier,
}


def main() -> None:
    t0 = time.monotonic()
    manifest = {
        "study_id": "cove50-torso-fdm-carrier-20261105",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
    }

    for family in ("truss_box", "topopt"):
        fam_dir = common.ARTIFACT_DIR / "candidates" / family
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_stages = {}

        for stage in common.STAGES:
            solid, bevel_info = BUILDERS[stage](family)
            sleeves_comp, slv_vol_mm3, slv_mass_g = common.build_stage_sleeves(stage)

            step_path = fam_dir / f"{stage}.step"
            stl_path = fam_dir / f"{stage}.stl"
            slv_step_path = fam_dir / f"{stage}_sleeves.step"
            slv_stl_path = fam_dir / f"{stage}_sleeves.stl"

            exp_info = common.export_stage_step(solid, step_path, stl_path)
            export_step(sleeves_comp, str(slv_step_path))
            export_stl(sleeves_comp, str(slv_stl_path), tolerance=0.04, angular_tolerance=0.15)

            vol_mm3 = float(exp_info["step_volume_mm3"])
            poly_mass_g = vol_mm3 * 1e-3 * common.MATERIALS["PAHT_CF_dry"]["density_g_cm3"]
            petg_mass_g = vol_mm3 * 1e-3 * common.MATERIALS["PETG_CF"]["density_g_cm3"]
            total_mass_g = poly_mass_g + slv_mass_g

            curv = common.compute_edge_curvature_metrics(step_path, bevel_info["r_inner_mm"], bevel_info["r_outer_mm"])

            fam_stages[stage] = {
                **exp_info,
                "stl_path": str(stl_path.relative_to(ROOT)),
                "sleeve_step_path": str(slv_step_path.relative_to(ROOT)),
                "sleeve_stl_path": str(slv_stl_path.relative_to(ROOT)),
                "polymer_volume_mm3": round(vol_mm3, 2),
                "polymer_mass_paht_cf_g": round(poly_mass_g, 3),
                "polymer_mass_petg_cf_g": round(petg_mass_g, 3),
                "sleeve_volume_mm3": round(slv_vol_mm3, 2),
                "sleeve_mass_ss304_g": round(slv_mass_g, 3),
                "total_stage_mass_g": round(total_mass_g, 3),
                "bevel_params": bevel_info,
                "edge_curvature": curv,
            }
            print(
                f"[{family:9s}] {stage:38s}: vol={vol_mm3:9.2f} mm3  "
                f"poly={poly_mass_g:6.2f} g + sleeves={slv_mass_g:5.2f} g = {total_mass_g:6.2f} g  "
                f"solids={exp_info['solid_count']} slivers={curv['sliver_faces_lt_0p05mm2']} shorts={curv['short_edges_lt_0p08mm']}"
            )

        manifest["families"][family] = {
            "stages": fam_stages,
            "total_polymer_mass_paht_cf_g": round(sum(s["polymer_mass_paht_cf_g"] for s in fam_stages.values()), 3),
            "total_sleeve_mass_ss304_g": round(sum(s["sleeve_mass_ss304_g"] for s in fam_stages.values()), 3),
            "total_assembly_mass_g": round(sum(s["total_stage_mass_g"] for s in fam_stages.values()), 3),
        }

    out_path = common.ARTIFACT_DIR / "torso-candidates-manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path.relative_to(ROOT)} in {time.monotonic() - t0:.2f}s")


# Stage winners of the combined family (the selected body's choice; chosen from the old matched FEM comparison, now retired:
# robot/body/checks fem runs the per-part setups on the chosen parts).
COMBINED_WINNERS = {
    "stage1_center_torso_spine_chassis": "truss_box",
    "stage2_lower_pelvis_battery_cradle": "topopt",
    "stage3_front_power_electronics_carrier": "truss_box",
    "stage4_rear_imu_adc_carrier": "topopt",
}


def assemble_combined() -> None:
    """Copy the per-stage winners into candidates/combined and add the `combined` family to torso-candidates-manifest.json."""
    import shutil
    path = common.ARTIFACT_DIR / "torso-candidates-manifest.json"
    man = json.loads(path.read_text())
    comb_dir = common.ARTIFACT_DIR / "candidates" / "combined"
    comb_dir.mkdir(parents=True, exist_ok=True)
    stages, poly, sleeve = {}, 0.0, 0.0
    for stage in common.STAGES:
        winner = COMBINED_WINNERS[stage]
        meta = json.loads(json.dumps(man["families"][winner]["stages"][stage]))
        step = ROOT / meta["step_path"]
        slv = ROOT / meta["sleeve_step_path"]
        shutil.copy2(step, comb_dir / f"{stage}.step")
        shutil.copy2(step.with_suffix(".stl"), comb_dir / f"{stage}.stl")
        shutil.copy2(slv, comb_dir / f"{stage}_sleeves.step")
        meta["step_path"] = str((comb_dir / f"{stage}.step").relative_to(ROOT))
        meta["sleeve_step_path"] = str((comb_dir / f"{stage}_sleeves.step").relative_to(ROOT))
        meta["selected_from_family"] = winner
        stages[stage] = meta
        poly += meta["polymer_mass_paht_cf_g"]
        sleeve += meta["sleeve_mass_ss304_g"]
    man["families"]["combined"] = {"total_polymer_mass_g": round(poly, 3), "total_sleeve_mass_g": round(sleeve, 3),
                                   "total_mass_g": round(poly + sleeve, 3), "stages": stages}
    path.write_text(json.dumps(man, indent=2) + "\n")


if __name__ == "__main__":
    main()
    assemble_combined()
