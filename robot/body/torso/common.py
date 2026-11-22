"""Shared geometry, material properties, and interface helpers for the Cove-50 Center-Body Torso Base Plate & Electronics Carrier FDM Redesign Study."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import (
    Align,
    Axis,
    Box,
    Compound,
    Cylinder,
    Location,
    Plane,
    Pos,
    Rot,
    Shape,
    Solid,
    Sphere,
    Vector,
    Wire,
    chamfer,
    scale,
    export_step,
    export_stl,
    fillet,
    import_step,
)

ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = ROOT / "robot/body/torso"
ARTIFACT_DIR = ROOT / "artifacts/robot/body/torso"
SOURCE_MANIFEST = ROOT / "artifacts/cove50-electronics-integration/r1-20261026/assembly-parts.json"

STAGES = (
    "stage1_center_torso_spine_chassis",
    "stage2_lower_pelvis_battery_cradle",
    "stage3_front_power_electronics_carrier",
    "stage4_rear_imu_adc_carrier",
)

STAGE_LABELS = {
    "stage1_center_torso_spine_chassis": "Stage 1: 7-Cell Hexagonal Honeycomb Torso Spine, Hip-Yaw Cradles & Shoulder/Neck Tower (Z=266.8..387.5 mm)",
    "stage2_lower_pelvis_battery_cradle": "Stage 2: 2-Cell Coaxial Honeycomb Pelvis Column, Interlocking Front/Rear Waist U-Clevis & Hip-Pitch Guard (Z=194.5..275.5 mm)",
    "stage3_front_power_electronics_carrier": "Stage 3: 3D-Printed Dual-Deck Front Power Supervisor & Precharge Frame (Z=270.5..367.5 mm)",
    "stage4_rear_imu_adc_carrier": "Stage 4: 3D-Printed Dorsal Body-IMU & Foot-ADC Sensor Carrier (Z=293.5..350.5 mm)",
}

# Repeated 5+2+2 Vertical Hexagonal Honeycomb 9-Cell Molicel P50B 21700 Battery Pack (all 9 cells parallel to Z):
# - Upper 7-Cell Honeycomb Magazine (Stage 1, Z_center = 304.2 mm, 22.4 mm hex pitch):
#     5 Dorsal cells at Y = +25.6 mm (X = -44.8, -22.4, 0.0, +22.4, +44.8 mm)
#     2 Nested Central Spine cells at Y = +6.2 mm (X = -11.2, +11.2 mm)
# - Lower 2-Cell Honeycomb Pelvis Column (Stage 2, Z_center = 230.5 mm, 22.4 mm pitch):
#     2 Central Pelvis cells at Y = +14.8 mm (X = -11.2, +11.2 mm)
HONEYCOMB_9CELL_BATTERY_LAYOUT = [
    {"name": "Honeycomb_Cell_01_Upper_Dorsal_L2", "fin_name": "Honeycomb_Fin_01_Upper_Dorsal_L2", "stage": "stage1_center_torso_spine_chassis", "center": (-44.8, 25.6, 304.2), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": +1},
    {"name": "Honeycomb_Cell_02_Upper_Dorsal_L1", "fin_name": "Honeycomb_Fin_02_Upper_Dorsal_L1", "stage": "stage1_center_torso_spine_chassis", "center": (-22.4, 25.6, 304.2), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": +1},
    {"name": "Honeycomb_Cell_03_Upper_Dorsal_C0", "fin_name": "Honeycomb_Fin_03_Upper_Dorsal_C0", "stage": "stage1_center_torso_spine_chassis", "center": (  0.0, 25.6, 304.2), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": +1},
    {"name": "Honeycomb_Cell_04_Upper_Dorsal_R1", "fin_name": "Honeycomb_Fin_04_Upper_Dorsal_R1", "stage": "stage1_center_torso_spine_chassis", "center": ( 22.4, 25.6, 304.2), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": +1},
    {"name": "Honeycomb_Cell_05_Upper_Dorsal_R2", "fin_name": "Honeycomb_Fin_05_Upper_Dorsal_R2", "stage": "stage1_center_torso_spine_chassis", "center": ( 44.8, 25.6, 304.2), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": +1},
    {"name": "Honeycomb_Cell_06_Upper_Core_L",    "fin_name": "Honeycomb_Fin_06_Upper_Core_L",    "stage": "stage1_center_torso_spine_chassis", "center": (-11.2,  6.2, 304.2), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": -1},
    {"name": "Honeycomb_Cell_07_Upper_Core_R",    "fin_name": "Honeycomb_Fin_07_Upper_Core_R",    "stage": "stage1_center_torso_spine_chassis", "center": ( 11.2,  6.2, 304.2), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": -1},
    {"name": "Honeycomb_Cell_08_Lower_Pelvis_L",  "fin_name": "Honeycomb_Fin_08_Lower_Pelvis_L",  "stage": "stage2_lower_pelvis_battery_cradle", "center": (-11.2, 14.8, 230.5), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": +1},
    {"name": "Honeycomb_Cell_09_Lower_Pelvis_R",  "fin_name": "Honeycomb_Fin_09_Lower_Pelvis_R",  "stage": "stage2_lower_pelvis_battery_cradle", "center": ( 11.2, 14.8, 230.5), "r_can_mm": 10.75, "h_can_mm": 70.15, "fin_dir_y": +1},
]


def build_cell_aluminum_fin_shape(cell_entry: dict[str, Any]) -> Shape:
    """Build a direct-on-battery 6063-T5 anodized aluminum 4-blade vertical heat-sink fin strip
    that glues directly onto the exposed 21700 battery can inside its vertical honeycomb window slot
    without touching the 3D-printed PAHT-CF carrier (d_cut = 10.25 mm -> |x - cx| <= 3.00 mm sits at |y - cy| >= 10.36 mm).
    """
    from build123d import Box, Cylinder, Pos

    cx, cy, cz = cell_entry["center"]
    s_y = float(cell_entry.get("fin_dir_y", +1))
    h_fin = 62.0  # Fits inside the 70.4..71.2 mm vertical window slot with >4 mm top/bottom ring clearance

    # Dorsal Cell_02 (-22.4, 25.6) and Cell_03 (0.0, 25.6) sit directly in front of the Stage 4 IMU/ADC carrier & M2 screws (Y >= 37.33 mm),
    # so they use a low-profile continuous 4-blade fin (outer tip at Y = 25.6 + 11.62 = 37.22 mm) with zero notches or interruptions;
    # the other 7 unobstructed cells use the full-depth 4-blade fin (outer tip at |y - cy| = 13.22 mm).
    is_under_stage4 = abs(cy - 25.6) < 0.5 and (abs(cx - (-22.4)) < 1.0 or abs(cx - 0.0) < 1.0)
    if is_under_stage4:
        y_base_center = cy + s_y * 10.68   # |y - cy| = [10.36, 11.00] mm
        base_depth = 0.64
        y_blade_center = cy + s_y * 11.26  # |y - cy| = [10.90, 11.62] mm (Y_max = 37.22 mm < 37.33 mm M2 screw tips)
        blade_depth = 0.72
    else:
        y_base_center = cy + s_y * 10.80   # |y - cy| = [10.36, 11.30] mm
        base_depth = 1.00
        y_blade_center = cy + s_y * 12.16  # |y - cy| = [11.12, 13.20] mm
        blade_depth = 2.08

    base_box = Pos(cx, y_base_center, cz) * Box(6.00, base_depth, h_fin)
    blades = [
        Pos(cx + dx, y_blade_center, cz) * Box(0.88, blade_depth, h_fin)
        for dx in (-2.56, -0.85, 0.85, 2.56)
    ]
    fin = base_box.fuse(*blades)

    # Cut inner cylindrical battery interface (r = 10.79 mm -> 0.04 mm TIM clearance to r_can = 10.75 mm cell can):
    fin = fin.cut(Pos(cx, cy, cz) * Cylinder(10.79, h_fin + 4.0))
    return max(fin.solids(), key=lambda s: s.volume)

_MANIFEST_BY_NAME: dict[str, dict[str, Any]] = {
    (p.get("name") or p.get("id")): p for p in json.loads(SOURCE_MANIFEST.read_text())["parts"]
}


def _manifest_step(name: str) -> Path:
    return ROOT / _MANIFEST_BY_NAME[name]["native_step"]


BASELINE_STAGE_PART_NAMES: dict[str, list[str]] = {
    "stage1_center_torso_spine_chassis": [
        "Common_power_carrier_mounting_r33",
        *[f"Cell_capture_bridge_{i:02d}" for i in range(13)],
    ],
    "stage2_lower_pelvis_battery_cradle": [
        "lower_cradle_lugs_224_with_waist_guard_r18",
        "lower_cradle_lugs_252_retained_outer_r18",
        "Cell_capture_bridge_13",
        "Cell_capture_bridge_14",
    ],
    "stage3_front_power_electronics_carrier": [
        "Power_supervisor_PA12_open_tray",
        "Main_precharge_PA12_open_tray",
        "BMS_recessed_fastener_front_guard",
        "Cell_07_removable_front_clamp_L",
        "Cell_07_removable_front_clamp_R",
    ],
    "stage4_rear_imu_adc_carrier": [
        "IMU_ADC_carrier_v17",
    ],
}

BASELINE_STAGE_SOURCES: dict[str, list[Path]] = {
    st: [_manifest_step(nm) for nm in names] for st, names in BASELINE_STAGE_PART_NAMES.items()
}

REPLACED_TORSO_PARTS = (
    "Common_power_carrier_mounting_r33",
    "lower_cradle_lugs_224_with_waist_guard_r18",
    "lower_cradle_lugs_252_retained_outer_r18",
    "Power_supervisor_PA12_open_tray",
    "Main_precharge_PA12_open_tray",
    "BMS_recessed_fastener_front_guard",
    "IMU_ADC_carrier_v17",
    "Cell_07_removable_front_clamp_L",
    "Cell_07_removable_front_clamp_R",
    *[f"Cell_capture_bridge_{i:02d}" for i in range(15)],
)

REPLACED_TORSO_FASTENER_PREFIXES = (
    "Cell_capture_",
    "Cell7_",
    "Cell_0",
    "Cell5_",
    "BMS_",
    "carrier_M2p5",
    "fixed_neck_ribbon",
    "lower_cradle",
    "Supervisor_frame_",
    "Main_frame_",
)

FDM_PRINT_BEDS_MM = {
    "standard_256x256x256": (256.0, 256.0, 256.0),
    "large_350x350x350": (350.0, 350.0, 350.0),
}

MATERIALS = {
    "AlSi10Mg_LPBF": {
        "label": "AlSi10Mg LPBF (Baseline Stage 1 Metal Spine)",
        "E_xy_mpa": 68000.0,
        "E_z_mpa": 68000.0,
        "nu": 0.33,
        "G_mpa": 25560.0,
        "sigma_xy_allow_mpa": 230.0,
        "sigma_z_allow_mpa": 210.0,
        "tau_allow_mpa": 130.0,
        "density_g_cm3": 2.67,
    },
    "PA12_SLS": {
        "label": "PA12 SLS (Baseline Stages 2-4 Trays/Cradles)",
        "E_xy_mpa": 1750.0,
        "E_z_mpa": 1650.0,
        "nu": 0.39,
        "G_mpa": 630.0,
        "sigma_xy_allow_mpa": 46.0,
        "sigma_z_allow_mpa": 40.0,
        "tau_allow_mpa": 26.0,
        "density_g_cm3": 1.01,
    },
    "PAHT_CF_dry": {
        "label": "Bambu PAHT-CF (PPA-CF15, Dry Annealed)",
        "E_xy_mpa": 5200.0,
        "E_z_mpa": 2400.0,
        "nu": 0.35,
        "G_mpa": 1450.0,
        "sigma_xy_allow_mpa": 95.0,
        "sigma_z_allow_mpa": 32.0,
        "tau_allow_mpa": 28.0,
        "density_g_cm3": 1.24,
    },
    "PAHT_CF_wet": {
        "label": "Bambu PAHT-CF (PPA-CF15, 50% RH Conditioned)",
        "E_xy_mpa": 4300.0,
        "E_z_mpa": 2050.0,
        "nu": 0.36,
        "G_mpa": 1200.0,
        "sigma_xy_allow_mpa": 80.0,
        "sigma_z_allow_mpa": 27.0,
        "tau_allow_mpa": 24.0,
        "density_g_cm3": 1.24,
    },
    "PETG_CF": {
        "label": "PETG-CF15 (Low-Creep Stiff Reference)",
        "E_xy_mpa": 3800.0,
        "E_z_mpa": 1900.0,
        "nu": 0.35,
        "G_mpa": 1100.0,
        "sigma_xy_allow_mpa": 65.0,
        "sigma_z_allow_mpa": 22.0,
        "tau_allow_mpa": 20.0,
        "density_g_cm3": 1.29,
    },
    "SS304_sleeve": {
        "label": "304 Stainless Steel Pressed-In Compression Sleeve",
        "E_xy_mpa": 193000.0,
        "E_z_mpa": 193000.0,
        "nu": 0.29,
        "G_mpa": 74800.0,
        "sigma_xy_allow_mpa": 215.0,
        "sigma_z_allow_mpa": 215.0,
        "tau_allow_mpa": 124.0,
        "density_g_cm3": 7.93,
    },
}

# Multi-limb torso loading parameters:
# - Bilateral C047 shoulder pitch actuators: 2 x 4.0 Nm peak stall (8.0 Nm combined pitch moment)
# - Bilateral C047 hip yaw actuators: 2 x 4.0 Nm peak stall
# - Neck root XM335 + 1.05 kg head assembly: 1.8 Nm stall + head gravity/impact inertia
# - Torso + 9-cell P50B battery pack + boards + arms + head suspended mass: ~3.20 kg above hip yaw plane
TORSO_SPRUNG_MASS_KG = 3.20
BATTERY_PACK_MASS_KG = 0.65
GRAVITY_M_S2 = 9.81
SHOULDER_C047_STALL_NM = 4.00
HIP_C047_STALL_NM = 4.00
NECK_XM335_STALL_NM = 1.80


def load_baseline_stage(stage: str) -> Shape:
    paths = BASELINE_STAGE_SOURCES[stage]
    solids = []
    for p in paths:
        s = import_step(str(p))
        solids.extend(list(s.solids()))
    if len(solids) == 1:
        return solids[0]
    return Compound(children=solids)


def aligned_cylinder(center: tuple[float, float, float], normal: tuple[float, float, float], radius: float, length: float) -> Solid:
    nx, ny, nz = normal
    if abs(nx) > 0.9:
        rot = Rot(0, 90, 0)
    elif abs(ny) > 0.9:
        rot = Rot(90, 0, 0)
    else:
        rot = Rot(0, 0, 0)
    return Pos(*center) * rot * Cylinder(radius, length, align=(Align.CENTER, Align.CENTER, Align.CENTER))


# Stage 3 board mounts (mm, body frame). PCB faces on the carrier side, measured on the installed boards
# (Power_supervisor_source_board -20.51, Main_precharge_source_board -33.68); the frame's rear face is -13.55 (rear plate),
# the precharge deck's rear face -28.72. Tuples: (front face Y, rear face Y, centre Y).
STAGE3_PCB_FACE_Y = {"power_supervisor": -20.51, "main_precharge": -33.68}
STAGE3_PS_BOSS_Y = (-20.51, -13.55, -17.03)
STAGE3_MP_BOSS_Y = (-33.68, -28.72, -31.20)


def stage_interface_specs() -> dict[str, dict[str, Any]]:
    """Exact bolt hole centers, axes, and pressed-in 304 SS compression sleeve specs for each center-body torso stage."""
    return {
        "stage1_center_torso_spine_chassis": {
            "support_face_label": "Bilateral Hip Yaw C047 Case Mounts (8x M2 at Z=272.95..275.00 mm, X=-49.6..-28.9 & +28.4..+49.6 mm)",
            "load_face_label": "Bilateral Shoulder Pitch C047 (8x M2 at X=+-27 mm, Z=360.3..384.8 mm) + Neck Root Cradle/XM335 (2x M3 + 4x M2)",
            "print_layer_normal": (0.0, 1.0, 0.0),
            "slice_axis": 2,  # Z slices from hip yaw / pelvis spine (Z=221.0) to shoulder pitch (Z=384.8)
            "support_bores": [
                #Left Hip Yaw C047 Case Mounts on Stage 1 Cheek Rail (4x M2 sleeves)
                {"center": (-49.60, -8.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (-49.60, 12.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (-28.90, -8.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (-28.90, 12.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                # Right Hip Yaw C047 Case Mounts on Stage 1 Cheek Rail (4x M2 sleeves)
                {"center": (49.60, -8.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (49.60, 12.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (28.90, -8.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (28.90, 12.25, 273.75), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
            ],
            "load_bores": [
                # Left Shoulder Pitch C047 (4x M2 sleeves)
                {"center": (-30.4, -1.25, 360.30), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (-30.4, 19.25, 360.30), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (-30.4, -1.25, 384.75), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (-30.4, 19.25, 384.75), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                # Right Shoulder Pitch C047 (4x M2 sleeves)
                {"center": (30.4, -1.25, 360.30), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (30.4, 19.25, 360.30), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (30.4, -1.25, 384.75), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (30.4, 19.25, 384.75), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                # Neck Root XM335 Right-Cheek Mounts (4x M2 sleeves)
                {"center": (11.5, 18.0, 356.75), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (11.5, 27.5, 356.75), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (11.5, 18.0, 362.25), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                {"center": (11.5, 27.5, 362.25), "normal": (1.0, 0.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.6},
                # Neck Root Cradle Spine Anchors (2x M3 sleeves)
                {"center": (-29.0, 27.5, 347.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.60, "r_od": 2.35, "length": 3.6},
                {"center": (-21.0, 27.5, 347.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.60, "r_od": 2.35, "length": 3.6},
            ],
            "aux_bores": [
                # 6x Interlocking Stage-1-to-Stage-2 Waist U-Clevis & Battery-Lid Mating Sleeves at Z=271.50 mm (2 Front Ventral + 4 Rear Dorsal Valley)
                {"center": (-17.0, -3.80, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.8},
                {"center": ( 17.0, -3.80, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.8},
                {"center": (-33.6, 33.70, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                {"center": (-11.2, 33.70, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                {"center": ( 11.2, 33.70, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                {"center": ( 33.6, 33.70, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                # 4x Flat Flush Stage-1-to-Stage-3 Front Spine Mounting Sleeves on Stage 1's existing flat Y=-6.10 mm ribs (X=+-17.0 mm, Z=288.0 & 338.0 mm)
                {"center": (-17.0, -4.15, 288.00), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.8},
                {"center": (-17.0, -4.15, 338.00), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.8},
                {"center": ( 17.0, -4.15, 288.00), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.8},
                {"center": ( 17.0, -4.15, 338.00), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.8},
                # 6x Stage-1-to-Stage-4 Dorsal Honeycomb Valley Mating Sleeves at X=-33.6, -11.2, +11.2 mm, Z=304.0 & 335.0 mm
*[{"center": (x, 33.60, z), "normal": (0.0, 1.0, 0.0), "r_id": 1.30, "r_od": 1.95, "length": 3.0}   # sites: stage4_rail_sites
                  for x, z in stage4_rail_sites()],
                # 3x Direct Dorsal Center-Back-Shell M3 Sleeves on Stage 1:
                # - 2x Mid-Waist Dorsal Shell Bosses (center_back_mount_mid_L & R at X=+-49.90, Y_mate=38.20, Z=275.46 mm)
                {"center": (-49.90, 37.18, 275.46), "normal": (0.0, 1.0, 0.0), "r_id": 1.60, "r_od": 2.05, "length": 2.0},
                {"center": ( 49.90, 37.18, 275.46), "normal": (0.0, 1.0, 0.0), "r_id": 1.60, "r_od": 2.05, "length": 2.0},
                # - 1x Upper Dorsal Shell Boss (center_back_mount_top at X=+10.10, Y_mate=36.00, Z=377.46 mm)
                {"center": ( 10.10, 34.10, 377.46), "normal": (0.0, 1.0, 0.0), "r_id": 1.60, "r_od": 2.35, "length": 3.6},
            ],
            "load_arm_mm": 121.0,
        },
        "stage2_lower_pelvis_battery_cradle": {
            "support_face_label": "6x M2 Interlocking Upper Waist U-Clevis & Battery-Lid Lugs to Stage 1 (Z=271.5 mm) + 4x M2 Flat Front Center-Front-Shell & Waist-Guard Mounts (X=+-16.0 mm, Y_mate=-12.0 mm) + 1x M3 Flat Rear Center-Back-Shell Mount (X=0.0, Y_mate=+28.0, Z=204.0 mm)",
            "load_face_label": "2-Cell Coaxial Lower Battery Honeycomb Column, 7-Cell Upper Battery Retention Lid & Direct Front/Rear Pelvis Shell Seats",
            "print_layer_normal": (0.0, 0.0, 1.0),
            "slice_axis": 1,  # Y slices from flat front shell mounts (Y=-12.0) to rear dorsal clevis lugs (Y=+38.45)
            "support_bores": [
                # 6x Interlocking Stage-2-to-Stage-1 Upward Waist U-Clevis & Battery-Lid Sleeves at Z=271.50 mm (coaxial with Stage 1 aux_bores)
                {"center": (-17.0, -7.875, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.2},
                {"center": ( 17.0, -7.875, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 3.2},
                {"center": (-33.6, 36.95, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                {"center": (-11.2, 36.95, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                {"center": ( 11.2, 36.95, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                {"center": ( 33.6, 36.95, 271.50), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 2.8},
                # 4x Flat Front Center-Front-Shell (Z=224.0 mm, Y_mate=-12.00 mm) & Waist-Guard Mount Sleeves (X=+-16.0 mm, Z=224.0 & 252.0 mm)
                {"center": (-16.0, -8.0, 224.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 7.6},
                {"center": ( 16.0, -8.0, 224.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 7.6},
                {"center": (-16.0, -8.0, 252.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 7.6},
                {"center": ( 16.0, -8.0, 252.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 7.6},
            ],
            "load_bores": [
                # Rear dorsal spine structural tie sleeves in the unified 2-cell honeycomb battery cradle web
                {"center": (0.0, 26.2, 224.0), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.0},
                {"center": (0.0, 26.2, 252.0), "normal": (0.0, 0.0, 1.0), "r_id": 1.05, "r_od": 1.65, "length": 3.0},
            ],
            "aux_bores": [
                # 1x Flat Rear Pelvis Center-Back-Shell M3 Sleeve on Stage 2's Y=+28.00 mm dorsal spine column (X=0.0, Z=204.0 mm)
                {"center": (0.0, 25.80, 204.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.60, "r_od": 2.05, "length": 4.2},
            ],
            "load_arm_mm": 38.0,
        },
        "stage3_front_power_electronics_carrier": {
            "support_face_label": "4x M2 Rearward Standoff Brackets (X=+-17.0 mm, Z=288.0 & 338.0 mm, Y=[-15.46, -6.15] mm) Mating Flush to Stage 1 Flat Front Mounts",
            "load_face_label": "4x M2 Power Supervisor PCB Mounts + 2x M2 Main Precharge PCB Mounts",
            "print_layer_normal": (0.0, 1.0, 0.0),
            "slice_axis": 2,  # Z slices across the 98 mm vertical electronics bay
            "support_bores": [
                # 4x M2 Rearward Standoff Bracket sleeves built into Stage 3 (coaxial with Stage 1's 4 flat front sleeves at X=+-17.0 mm, Y=-4.15 mm)
                {"center": (-17.0, -10.80, 288.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 9.2},
                {"center": (-17.0, -10.80, 338.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 9.2},
                {"center": ( 17.0, -10.80, 288.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 9.2},
                {"center": ( 17.0, -10.80, 338.0), "normal": (0.0, 1.0, 0.0), "r_id": 1.05, "r_od": 1.65, "length": 9.2},
            ],
            "load_bores": [
                # Board mounts: the boss runs from the PCB's carrier-side face (STAGE3_PCB_FACE_Y, 0.00 contact) to the
                # frame's rear face; the sleeve is the full thickness of that stack, flush at both ends (every clamped
                # printed hole sleeved, head on the outer face), M2 bore 2.2 (standard-parts.json sleeve.bore_for).
                # 4x M2 Power Supervisor PCB sleeves
                *[{"center": (x, STAGE3_PS_BOSS_Y[2], z), "normal": (0.0, 1.0, 0.0), "r_id": 1.10, "r_od": 1.70,
                   "length": STAGE3_PS_BOSS_Y[1] - STAGE3_PS_BOSS_Y[0]}
                  for x, z in ((48.5, 280.8), (-11.5, 280.8), (48.0, 364.0), (-11.0, 364.0))],
                # 2x M2 Main Precharge PCB sleeves (inside mp_bosses)
                *[{"center": (x, STAGE3_MP_BOSS_Y[2], z), "normal": (0.0, 1.0, 0.0), "r_id": 1.10, "r_od": 1.70,
                   "length": STAGE3_MP_BOSS_Y[1] - STAGE3_MP_BOSS_Y[0]}
                  for x, z in ((-7.0, 332.0), (37.0, 301.0))],
            ],
            "aux_bores": [],
            "load_arm_mm": 32.0,
        },
        "stage4_rear_imu_adc_carrier": {
            "support_face_label": "6x M2.5 Dorsal Honeycomb Valley Standoff Rails to Stage 1 threaded sleeves (X=-33.6, -11.2, +11.2 mm; heads on the carrier's outer face outside the IMU / ADC footprints)",
            "load_face_label": "4x M2 Body IMU 4517 Mounts + 4x M2 Foot ADC 6494 Mounts (printed pilots in the plate bosses, no sleeves)",
            "print_layer_normal": (0.0, 1.0, 0.0),
            "slice_axis": 2,  # Z slices from IMU (Z=297.65) through spine anchors (Z=304..335) to ADC (Z=345.35)
            # Rail screws: the carrier rail carries the full-thickness flush clearance sleeve (Stage 1 face to the outer face, head
            # on the outer face); the coaxial Stage 1 sleeve is internally threaded (owner 2026-11-22, torso_screws.py).
            "support_bores": [
                {"center": (x, (STAGE4_RAIL_Y0 + STAGE4_FRONT_Y) / 2, z), "normal": (0.0, 1.0, 0.0), "r_id": 1.30, "r_od": 1.95,
                 "length": STAGE4_FRONT_Y - STAGE4_RAIL_Y0}
                for x, z in stage4_rail_sites()
            ],
            # Board screws end in printed pilots of the plate (stage4_board_sites, from the boards' own STEP): no sleeve.
            "load_bores": [],
            "aux_bores": [],
            "load_arm_mm": 27.35,
        },
    }


# Stage 4 IMU / ADC carrier (mm, body frame). The plate's outer face is the boards' seat face (0.00 contact); the rails run from
# Stage 1's dorsal rail face (Y 35.40, builder stage 1) to it. Rail screw sites keep the old X = -33.6 / -11.2 / +11.2 and
# Z 304 / 335, except the two on X = -11.2, which sat under the IMU (Z 295.1-312.9) and the ADC (Z 330.1-347.9) boards: they
# move to the free strip between the boards, +-STAGE4_RAIL_MID_OFFSET about its middle.
STAGE4_RAIL_Y0 = 35.40
STAGE4_PLATE_REAR_Y = 39.10
STAGE4_BOSS_REAR_Y = 37.45          # board-screw boss ends 0.23 mm in front of the low-profile fins of Cell_02 / Cell_03 (tip Y 37.22)
STAGE4_FRONT_Y = 42.00              # = the boards' back face (min Y of both supplier STEPs, checked in stage4_board_sites)
STAGE4_RAIL_MID_OFFSET = 5.0
PA2_PILOT_DIA = 1.6                     # PA2 thread-forming pilot: 80 % of the 2.0 major diameter (board screws into the carrier bosses)
STAGE4_RAIL_X = (-33.6, -11.2, 11.2)    # rail columns; the one source for the sites, the rail boxes of the carrier and Stage 1's sleeves
STAGE4_RAIL_Z = (304.0, 335.0)          # the old screw rows (legacy_stage4_rail_sites)
STAGE4_BOARDS = ("body_IMU_4517", "foot_ADC_6494")


def _board_hole_faces(part: str) -> list[dict]:
    """Mount holes of a supplier board's own STEP: the cylindrical faces of radius 1.1-1.4 mm with their axis along Y (the vias are
    0.5 mm, the board's other cylinders are parts of connectors), grouped by axis position (X, Z): Y span of the faces of one hole
    and the board's extent."""
    from build123d import GeomType, import_step
    shape = import_step(str(_manifest_step(part)))
    bb = shape.bounding_box()
    holes = {}
    for f in shape.faces():
        if f.geom_type != GeomType.CYLINDER or not 1.1 < f.radius < 1.4 or abs(f.axis_of_rotation.direction.Y) < 0.99:
            continue
        a = f.axis_of_rotation.position
        h = holes.setdefault((round(a.X, 2), round(a.Z, 2)), dict(x=round(a.X, 3), z=round(a.Z, 3), r_mm=f.radius, y0=1e9, y1=-1e9))
        fb = f.bounding_box()
        h["y0"], h["y1"] = min(h["y0"], fb.min.Y), max(h["y1"], fb.max.Y)
    return [dict(board=part, board_y=(bb.min.Y, bb.max.Y), board_x=(bb.min.X, bb.max.X), board_z=(bb.min.Z, bb.max.Z), **h)
            for h in holes.values()]


_BOARD_SITES: list[dict] = []


def stage4_board_sites() -> list[dict]:
    """The 8 IMU / ADC mount holes read from the boards' own STEP files, ordered IMU first, then by Z descending and X
    ascending: board, x, z (hole axis, Y), y_seat (the board's back face on the carrier), y_top (the hole's far end, the
    face a head bears on), the board's X / Z extent."""
    if not _BOARD_SITES:
        for part in STAGE4_BOARDS:
            hs = _board_hole_faces(part)
            assert len(hs) == 4, (part, [(h["x"], h["z"]) for h in hs])
            for h in sorted(hs, key=lambda h: (-h["z"], h["x"])):
                _BOARD_SITES.append(dict(board=part, x=h["x"], z=h["z"], hole_dia_mm=round(2 * h["r_mm"], 3), y_seat=round(h["board_y"][0], 3),
                                         y_top=round(h["y1"], 3), board_x=h["board_x"], board_z=h["board_z"]))
    return _BOARD_SITES


_RAIL_SITES: list[tuple[float, float]] = []


def stage4_rail_sites() -> list[tuple[float, float]]:
    """(x, z) of the 6 rail screws: the old sites, the two on X = -11.2 moved into the strip between the boards."""
    if not _RAIL_SITES:
        spans = []
        for b in STAGE4_BOARDS:
            bb = import_step(str(_manifest_step(b))).bounding_box()
            spans.append((bb.min.Z, bb.max.Z))
        zs = sorted(spans)
        mid = (zs[0][1] + zs[1][0]) / 2.0
        moved = {-11.2: (round(mid - STAGE4_RAIL_MID_OFFSET, 3), round(mid + STAGE4_RAIL_MID_OFFSET, 3))}
        _RAIL_SITES.extend((x, z) for x in STAGE4_RAIL_X for z in moved.get(x, STAGE4_RAIL_Z))
    return list(_RAIL_SITES)


def legacy_stage4_rail_sites() -> list[tuple[float, float]]:
    """The old rail sites the selected Stage 1 STEP still has bores and sleeves at (x, z), no longer used by stage4_rail_sites."""
    cur = stage4_rail_sites()
    return [(x, z) for x in STAGE4_RAIL_X for z in STAGE4_RAIL_Z if not any(abs(x - a) < 1e-6 and abs(z - b) < 1e-6 for a, b in cur)]


def build_stage_sleeves(stage: str) -> tuple[Compound, float, float]:
    """Build 304 SS pressed-in compression sleeves for all bolted interfaces of a stage."""
    spec = stage_interface_specs()[stage]
    sleeves = []
    vol_mm3 = 0.0
    for bore in spec["support_bores"] + spec["load_bores"] + spec.get("aux_bores", []):
        outer = aligned_cylinder(bore["center"], bore["normal"], bore["r_od"], bore["length"])
        inner = aligned_cylinder(bore["center"], bore["normal"], bore["r_id"], bore["length"] + 0.4)
        sleeve = outer.cut(inner)
        for sol in sleeve.solids():
            sleeves.append(sol)
            vol_mm3 += float(sol.volume)
    mass_g = vol_mm3 * 1e-3 * MATERIALS["SS304_sleeve"]["density_g_cm3"]
    return Compound(children=sleeves), vol_mm3, mass_g


def bevel_3d_edges(
    shape: Shape,
    r_inner: float = 0.55,
    r_outer: float = 0.22,
    keep_zones: list[Shape] | None = None,
) -> tuple[Shape, dict[str, Any]]:
    """Apply asymmetric 3D edge beveling: larger inner fillets (r_inner > r_outer) for stress relief, smaller outer bevels (r_outer)."""
    solids = list(shape.solids())
    if not solids:
        return shape, {"filleted_inner_edges": 0, "filleted_outer_edges": 0, "r_inner_mm": r_inner, "r_outer_mm": r_outer}
    body = max(solids, key=lambda s: s.volume)
    bb = body.bounding_box()
    cx = 0.5 * (bb.min.X + bb.max.X)
    cy = 0.5 * (bb.min.Y + bb.max.Y)
    cz = 0.5 * (bb.min.Z + bb.max.Z)
    half_diag = 0.5 * math.sqrt((bb.max.X - bb.min.X) ** 2 + (bb.max.Y - bb.min.Y) ** 2 + (bb.max.Z - bb.min.Z) ** 2)

    edges = sorted(body.edges(), key=lambda e: e.length, reverse=True)
    inner_count = 0
    outer_count = 0
    applied_inner_r = r_inner
    applied_outer_r = r_outer

    for e in edges:
        if e.length < 2.2:
            continue
        ec = e.center()
        dist = math.sqrt((ec.X - cx) ** 2 + (ec.Y - cy) ** 2 + (ec.Z - cz) ** 2)
        is_inner = dist < 0.58 * half_diag
        if is_inner and inner_count < 14:
            for r_try in (r_inner, 0.42, 0.34):
                try:
                    cand = fillet([e], radius=r_try)
                    if cand is not None and len(cand.solids()) == 1 and cand.is_valid:
                        body = cand.solids()[0]
                        inner_count += 1
                        applied_inner_r = max(applied_inner_r, r_try)
                        break
                except Exception:
                    continue
        elif (not is_inner) and outer_count < 14:
            for r_try in (r_outer, 0.18, 0.14):
                try:
                    cand = fillet([e], radius=r_try)
                    if cand is not None and len(cand.solids()) == 1 and cand.is_valid:
                        body = cand.solids()[0]
                        outer_count += 1
                        applied_outer_r = min(applied_outer_r, r_try)
                        break
                except Exception:
                    continue
        if inner_count >= 6 and outer_count >= 6:
            break

    return body, {
        "filleted_inner_edges": inner_count,
        "filleted_outer_edges": outer_count,
        "r_inner_mm": round(float(applied_inner_r), 3),
        "r_outer_mm": round(float(applied_outer_r), 3),
        "r_inner_gt_r_outer": bool(applied_inner_r > applied_outer_r),
    }


def solid_to_watertight_mesh(shape: Shape, tol: float = 0.05) -> trimesh.Trimesh:
    """Tessellate an OpenCascade Shape via BRepMesh_IncrementalMesh and weld shared vertices into a watertight trimesh."""
    import manifold3d as md
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    s0 = shape.solids()[0] if hasattr(shape, "solids") and len(shape.solids()) == 1 else shape
    for t_try, dig in [(tol, 5), (tol, 4), (0.03, 4), (0.06, 4), (0.03, 3)]:
        BRepTools.Clean_s(s0.wrapped)
        BRepMesh_IncrementalMesh(s0.wrapped, float(t_try), False, 0.12, True).Perform()
        vertices, faces = [], []
        for f in s0.faces():
            loc = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(f.wrapped, loc)
            if poly is None:
                continue
            off = len(vertices)
            tr = loc.Transformation()
            vertices.extend(
                [[p.X(), p.Y(), p.Z()] for p in [poly.Node(i).Transformed(tr) for i in range(1, poly.NbNodes() + 1)]]
            )
            for i in range(1, poly.NbTriangles() + 1):
                a, b, c = poly.Triangle(i).Get()
                faces.append(
                    [off + a - 1, off + c - 1, off + b - 1]
                    if f.wrapped.Orientation() == TopAbs_REVERSED
                    else [off + a - 1, off + b - 1, off + c - 1]
                )
        m = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=True)
        m.merge_vertices(digits_vertex=dig)
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        if not m.is_watertight:
            trimesh.repair.fill_holes(m)
            trimesh.repair.fix_normals(m)
        if m.is_watertight and m.volume < 0:
            m.invert()
        if m.is_watertight:
            return m
    return m


def export_stage_step(shape: Shape, step_path: Path, stl_path: Path | None = None) -> dict[str, Any]:
    """Export single-solid BRep STEP and verify watertight trimesh conversion."""
    step_path.parent.mkdir(parents=True, exist_ok=True)
    solids = list(shape.solids())
    out_shape = solids[0] if len(solids) == 1 else Compound(children=solids)
    export_step(out_shape, str(step_path))
    tmp_stl = stl_path if stl_path is not None else step_path.with_suffix(".stl")
    mesh = solid_to_watertight_mesh(out_shape, tol=0.04)
    mesh.export(str(tmp_stl))
    step_vol = float(sum(s.volume for s in solids))
    mesh_vol = float(abs(mesh.volume))
    rel_diff = abs(step_vol - mesh_vol) / max(step_vol, 1e-6)
    return {
        "step_path": str(step_path.relative_to(ROOT)),
        "solid_count": len(solids),
        "is_valid_brep": bool(all(s.is_valid for s in solids)),
        "watertight_mesh": bool(mesh.is_watertight),
        "step_volume_mm3": round(step_vol, 3),
        "mesh_volume_mm3": round(mesh_vol, 3),
        "volume_rel_diff": round(rel_diff, 6),
        "bounds_mm": [[round(float(x), 3) for x in mesh.bounds[0]], [round(float(x), 3) for x in mesh.bounds[1]]],
        "extents_mm": [round(float(x), 3) for x in mesh.extents],
    }


def compute_edge_curvature_metrics(step_path: Path, r_inner_mm: float = 0.55, r_outer_mm: float = 0.22) -> dict[str, Any]:
    """Compute BRep face/edge quality metrics (slivers, Plane-Plane linear short edges, sharp dihedral fraction, bevel radii)."""
    from build123d import Edge
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane, GeomAbs_Torus
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    s = import_step(str(step_path))
    solids = list(s.solids())
    faces = list(s.faces())
    edges = list(s.edges())
    sliver_faces = sum(1 for f in faces if f.area < 0.05)

    short_edges = 0
    ef = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(s.wrapped, TopAbs_EDGE, TopAbs_FACE, ef)
    for idx in range(1, ef.Extent() + 1):
        ed = TopoDS.Edge_s(ef.FindKey(idx))
        adj = ef.FindFromIndex(idx)
        if adj.Extent() == 2:
            f1 = TopoDS.Face_s(adj.First())
            f2 = TopoDS.Face_s(adj.Last())
            if BRepAdaptor_Surface(f1).GetType() == GeomAbs_Plane and BRepAdaptor_Surface(f2).GetType() == GeomAbs_Plane:
                if BRepAdaptor_Curve(ed).GetType() == GeomAbs_Line and Edge(ed).length < 0.08:
                    short_edges += 1

    planes = cyls = tori = other = 0
    exp = TopExp_Explorer(s.wrapped, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        surf = BRepAdaptor_Surface(face)
        st = surf.GetType()
        if st == GeomAbs_Plane:
            planes += 1
        elif st == GeomAbs_Cylinder:
            cyls += 1
        elif st == GeomAbs_Torus:
            tori += 1
        else:
            other += 1
        exp.Next()

    tmp_dir = ARTIFACT_DIR / "scratch_stl"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_stl = step_path.with_suffix(".stl") if str(step_path).startswith(str(ARTIFACT_DIR)) else (tmp_dir / f"{step_path.stem}.stl")
    if not tmp_stl.exists():
        export_stl(s, str(tmp_stl), tolerance=0.04, angular_tolerance=0.08)
    mesh = trimesh.load(str(tmp_stl), force="mesh")
    angles = np.degrees(mesh.face_adjacency_angles) if len(mesh.face_adjacency_angles) else np.zeros(1)
    sharp_60 = float(np.mean(angles > 60.0))
    acute_110 = int(np.sum(angles > 110.0))

    return {
        "solid_count": len(solids),
        "face_count": len(faces),
        "edge_count": len(edges),
        "plane_faces": planes,
        "cylinder_faces": cyls,
        "torus_fillet_faces": tori,
        "other_faces": other,
        "sliver_faces_lt_0p05mm2": sliver_faces,
        "short_edges_lt_0p08mm": short_edges,
        "mesh_sharp_dihedral_gt_60deg_fraction": round(sharp_60, 4),
        "mesh_acute_dihedral_gt_110deg_count": acute_110,
        "r_inner_mm": round(float(r_inner_mm), 3),
        "r_outer_mm": round(float(r_outer_mm), 3),
        "r_inner_gt_r_outer": bool(r_inner_mm > r_outer_mm),
    }


# Pivot axes for multi-limb flexion checks around the center-body torso:
LEFT_SHOULDER_PITCH_AXIS_ORIGIN = np.array([-27.0, 9.0, 372.525], dtype=float)
LEFT_SHOULDER_PITCH_AXIS_DIR = np.array([-1.0, 0.0, 0.0], dtype=float)

RIGHT_SHOULDER_PITCH_AXIS_ORIGIN = np.array([27.0, 9.0, 372.525], dtype=float)
RIGHT_SHOULDER_PITCH_AXIS_DIR = np.array([1.0, 0.0, 0.0], dtype=float)

LEFT_HIP_YAW_AXIS_ORIGIN = np.array([-59.0, 2.0, 263.8], dtype=float)
LEFT_HIP_YAW_AXIS_DIR = np.array([0.0, 0.0, 1.0], dtype=float)

RIGHT_HIP_YAW_AXIS_ORIGIN = np.array([59.0, 2.0, 263.8], dtype=float)
RIGHT_HIP_YAW_AXIS_DIR = np.array([0.0, 0.0, 1.0], dtype=float)

NECK_YAW_AXIS_ORIGIN = np.array([0.0, 5.0, 365.0], dtype=float)
NECK_YAW_AXIS_DIR = np.array([0.0, 0.0, 1.0], dtype=float)


def axis_angle_matrix(origin: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return np.eye(4)
    u = axis / np.linalg.norm(axis)
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    ux, uy, uz = u
    R = np.array([
        [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin - R @ origin
    return T


def load_manifest_part_mesh(part_entry: dict[str, Any], tol: float = 0.08) -> trimesh.Trimesh | None:
    """Load a part mesh from SOURCE_MANIFEST: its declared mesh, else its native STEP meshed, each restored by path.

    The .npz cache is an output for the GLB, never read back, so the result does not depend on earlier runs."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from infra.storage.blob_store import ensure
    pid = part_entry.get("name") or part_entry.get("id") or "part"
    cache_dir = ARTIFACT_DIR / "neighbor_meshes_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / f"{pid}.npz"

    mp = part_entry.get("mesh")
    if mp and ensure(ROOT / mp).exists():
        m = trimesh.load(str(ensure(ROOT / mp)), force="mesh")
        np.savez_compressed(cache_npz, vertices=np.asarray(m.vertices, dtype=np.float32), faces=np.asarray(m.faces, dtype=np.int32))
        return m

    sp = part_entry.get("native_step")
    if sp and ensure(ROOT / sp).exists():
        s = import_step(str(ensure(ROOT / sp)))
        m = solid_to_watertight_mesh(s, tol=tol)
        np.savez_compressed(cache_npz, vertices=np.asarray(m.vertices, dtype=np.float32), faces=np.asarray(m.faces, dtype=np.int32))
        return m

    import sys
    if str(ROOT / "infra" / "dashboard") not in sys.path:
        sys.path.insert(0, str(ROOT / "infra" / "dashboard"))
    import results_hub

    for rel in (mp, sp):
        if not rel:
            continue
        try:
            with results_hub.opened_source_asset(rel) as fetched:
                if str(fetched).endswith(".stl"):
                    m = trimesh.load(str(fetched), force="mesh")
                else:
                    s = import_step(str(fetched))
                    m = solid_to_watertight_mesh(s, tol=tol)
                np.savez_compressed(cache_npz, vertices=np.asarray(m.vertices, dtype=np.float32), faces=np.asarray(m.faces, dtype=np.int32))
                return m
        except Exception:
            continue
    return None


def ensure_mounted_shell_and_connector_assets() -> list[dict[str, Any]]:
    """Materialize and cache the 2-piece smooth continuous Center Shell (center_front_shell, center_back_shell),
    the chest indicator lozenge (chest_indicator), and all 10 direct shell fasteners connecting directly
    to Stage 1 and Stage 2 with zero intermediate pelvis carrier brackets.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from infra.storage.blob_store import ensure

    shell_dir = ARTIFACT_DIR / "shell_assets"
    shell_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = shell_dir / "mounted-shell-assets.json"  # rebuilt every run; an earlier run's copy is never reused

    def _save_shape_asset(
        name: str,
        shape: Shape,
        display_group: str,
        role: str,
        tol: float = 0.06,
        save_step: bool = False,
    ) -> dict[str, Any]:
        npz_path = shell_dir / f"{name}.npz"
        m = solid_to_watertight_mesh(shape, tol=tol)
        np.savez_compressed(
            npz_path,
            vertices=np.asarray(m.vertices, dtype=np.float32),
            faces=np.asarray(m.faces, dtype=np.int32),
        )
        entry: dict[str, Any] = {
            "name": name,
            "display_group": display_group,
            "role": role,
            "npz_path": str(npz_path.relative_to(ROOT)),
            "volume_mm3": round(float(abs(m.volume)), 3),
            "bounds_mm": [[round(float(x), 3) for x in m.bounds[0]], [round(float(x), 3) for x in m.bounds[1]]],
        }
        if save_step:
            st_path = shell_dir / f"{name}.step"
            export_step(shape, str(st_path))
            entry["step_path"] = str(st_path.relative_to(ROOT))
        return entry

    # 1. Build the continuous 12-station NURBS loft from crotch (Z=185.71 mm) to neck collar (Z=397.14 mm)
    # matching the reference ape drawing silhouette without any horizontal waist split:
    scale = 500.0 / 350.0
    stations = [
        [185.71, 0.0, 7.00, 22.00, 20.00],
        [193.00, 0.0, 6.50, 36.00, 29.00],
        [204.00, 0.0, 5.50, 64.00, 40.00],
        [218.00, 0.0, 4.60, 76.80, 45.40],
        [242.00, 0.0, 4.10, 77.20, 45.60],
        [270.00, 0.0, 4.09, 78.38, 46.34],
        [284.29, 0.0, 4.25, 82.85, 49.20],
        [327.14, 0.0, 5.61, 90.00, 54.00],
        [361.43, 0.0, 7.14, 90.00, 44.58],
        [377.14, 0.0, 7.14, 85.71, 38.91],
        [391.43, 0.0, 7.14, 50.00, 31.43],
        [397.14, 0.0, 7.14, 32.86, 25.71],
    ]

    def _loft_ellipses(rows: list[list[float]]) -> Solid:
        return Solid.make_loft([Wire.make_ellipse(rx, ry, Plane(origin=(x, y, z))) for z, x, y, rx, ry in rows])

    outer = _loft_ellipses(stations)
    inner_rows = [[z, x, y, rx - 2.0, ry - 2.0] for z, x, y, rx, ry in stations]
    inner_rows[0][0] += 2.0
    inner = _loft_ellipses(inner_rows)
    shell = outer - inner

    # Neck collar opening + rear neck cowl clearance:
    neck_hole = Solid.make_cylinder(20.0 * scale, 40.0 * scale, Plane(origin=(0.0, 4.0 * scale, 260.0 * scale)))
    cowl_relief = Pos(0.0, 32.0, 388.0) * Box(54.0, 20.0, 20.0)
    shell = shell - neck_hole - cowl_relief

    # Smooth spherical thigh openings at pelvis bottom:
    for sg in (-1.0, 1.0):
        leg_tool = Solid.make_sphere(1.0).scale((43.0, 50.0, 42.0)).translate((sg * 54.0, 5.0, 181.0))
        shell = shell - leg_tool

    # Rear-half shoulder-bridge exit openings & inner motor-case relief pockets:
    for sg in (-1.0, 1.0):
        sh_cut = Pos(sg * 82.0, 14.5, 344.2) * Box(26.0, 18.5, 38.0)
        case_rel = Pos(sg * 62.0, 22.0, 352.0) * Box(22.0, 10.0, 30.0)
        shell = shell - sh_cut - case_rel

    # Chest indicator lozenge opening:
    s_ind = import_step(str(ensure(ROOT / _manifest_step("ape_body_shell__chest_indicator_relief4_r33"))))
    bb_ind = s_ind.bounding_box()
    ind_tool = (
        Solid.make_sphere(1.0)
        .scale((1.95 * scale, 4.5 * scale, 5.8 * scale))
        .translate((0.0, 0.5 * (bb_ind.min.Y + bb_ind.max.Y), 244.0 * scale))
    )
    shell = shell - ind_tool

    # Split at Y = 5.7143 mm (0.65 mm vertical side parting seam):
    s_front = shell & (Pos(0.0, -57.307, 290.0) * Box(240.0, 125.389, 250.0))
    s_back = shell & (Pos(0.0, 66.038, 290.0) * Box(240.0, 120.000, 250.0))
    for z_slot in (324.0, 330.0, 336.0, 342.0):
        s_back = s_back.cut(Pos(0.0, 60.0, z_slot) * Box(28.0, 50.0, 2.4))

    # 2. Add bilateral side clamshell screw-post couplers C_T01_chest__T02_back_1/2 at (±84.6213, 5.7143, 292.8571)
    # and direct Stage 1 / Stage 2 inward mounting bosses to center_front_shell & center_back_shell:
    s_t01 = import_step(str(ensure(ROOT / "artifacts/cove50-fitted-shell/r1-20261027/panels/T01_chest.step")))
    s_t02 = import_step(str(ensure(ROOT / "artifacts/cove50-fitted-shell/r1-20261027/panels/T02_back.step")))
    box_couplers = (Pos(-85.0, 5.7143, 293.0) * Box(22.0, 18.0, 16.0)).fuse(Pos(85.0, 5.7143, 293.0) * Box(22.0, 18.0, 16.0))

    # 2x Lower-Pelvis Front Inward Screw Posts on center_front_shell at (X = ±16.0, Y = [-39.9, -12.05], Z = 224.0)
    # mating directly to Stage 2's flat Y = -12.00 mm front mounts:
    f_bosses = []
    f_cuts = []
    for sgn in (-1.0, 1.0):
        fb = (Pos(sgn * 16.0, -26.50, 224.0) * Rot(90, 0, 0) * Cylinder(3.50, 28.90)) & outer
        f_bosses.append(fb)
        # M2 clearance bore (r=1.15 mm) through Y=[-16.50, -12.00] and deep driver counterbore (r=2.30 mm) for Y <= -16.50:
        f_cuts.append(Pos(sgn * 16.0, -26.0, 224.0) * Rot(90, 0, 0) * Cylinder(1.15, 32.0))
        f_cuts.append(Pos(sgn * 16.0, -29.5, 224.0) * Rot(90, 0, 0) * Cylinder(2.30, 26.0))

    center_front = ((s_front - box_couplers).fuse((s_t01 & box_couplers) & outer, *f_bosses)).cut(*f_cuts)
    center_front = max(center_front.solids(), key=lambda s: s.volume)

    # 4x Dorsal Inward Screw Posts on center_back_shell:
    # - Top boss at (+10.10, Y_mate=+36.00, Z=377.457) -> Stage 1
    # - 2x Mid-waist bosses at (±49.90, Y_mate=+38.20, Z=275.457) -> Stage 1
    # - 1x Lower-pelvis boss at (0.00, Y=[+28.05, +45.4], Z=204.00) -> Stage 2's flat Y=+28.00 mm dorsal spine column
    box_mount2 = Pos(10.10, 42.0, 377.457) * Box(14.0, 16.0, 14.0)
    box_m1_src = Pos(-49.90, 41.0, 275.457) * Box(12.0, 14.0, 12.0)
    box_m3_src = Pos(49.90, 41.0, 275.457) * Box(12.0, 14.0, 12.0)
    m1_piece = ((s_t02 & box_m1_src) - (Pos(-49.90, 34.10, 275.457) * Box(16.0, 8.20, 16.0))) & outer
    m3_piece = m1_piece.mirror(Plane.YZ)
    m2_piece = (s_t02 & box_mount2) & outer

    b_low_boss = (Pos(0.0, 37.025, 204.0) * Rot(90, 0, 0) * Cylinder(3.80, 17.95)) & outer
    b_low_cuts = [
        Pos(0.0, 37.0, 204.0) * Rot(90, 0, 0) * Cylinder(1.65, 22.0),
        Pos(0.0, 40.5, 204.0) * Rot(90, 0, 0) * Cylinder(2.85, 16.0),
    ]

    center_back = (
        (s_back - box_couplers - box_mount2 - box_m1_src - box_m3_src)
        .fuse((s_t02 & box_couplers) & outer, m2_piece, m1_piece, m3_piece, b_low_boss)
        .cut(*b_low_cuts)
    )
    center_back = max(center_back.solids(), key=lambda s: s.volume)

    assets: list[dict[str, Any]] = [
        _save_shape_asset("center_front_shell", center_front, "ape_body_shell", "mounted_center_shell", tol=0.08, save_step=True),
        _save_shape_asset("center_back_shell", center_back, "ape_body_shell", "mounted_center_shell", tol=0.08, save_step=True),
    ]

    # 3. Fasteners for side clamshell couplers (2x M3x8 + 2x M3 hex nut) and upper dorsal boss (1x M3x8):
    hw_r1 = "artifacts/cove50-fitted-shell/r1-20261027/hardware"
    for fn in (
        "C_T01_chest__T02_back_1_M3x8",
        "C_T01_chest__T02_back_2_M3x8",
        "C_T01_chest__T02_back_1_M3_hex_nut",
        "C_T01_chest__T02_back_2_M3_hex_nut",
        "T02_back_mount2_M3x8",
    ):
        s_hw = import_step(str(ensure(ROOT / f"{hw_r1}/{fn}.step")))
        assets.append(_save_shape_asset(fn, s_hw, "hardware", "shell_fastener", tol=0.04))

    # 2x Mid-waist dorsal M3x6 screws into Stage 1 at (±49.90, Z=275.457):
    for sgn, m_name in ((-1.0, "T02_back_mount1_M3x6"), (1.0, "T02_back_mount3_M3x6")):
        sh_head = Pos(sgn * 49.90, 40.90, 275.457) * Rot(90, 0, 0) * Cylinder(2.75, 2.40)
        sh_shaft = Pos(sgn * 49.90, 37.975, 275.457) * Rot(90, 0, 0) * Cylinder(1.50, 3.45)
        assets.append(_save_shape_asset(m_name, sh_head.fuse(sh_shaft), "hardware", "shell_fastener", tol=0.04))

    # 1x Lower-pelvis dorsal M3x8 screw into Stage 2 at (0.00, Z=204.00):
    low_b_head = Pos(0.0, 33.70, 204.0) * Rot(90, 0, 0) * Cylinder(2.75, 2.40)
    low_b_shaft = Pos(0.0, 28.50, 204.0) * Rot(90, 0, 0) * Cylinder(1.50, 8.00)
    assets.append(_save_shape_asset("center_back_low_M3x8", low_b_head.fuse(low_b_shaft), "hardware", "shell_fastener", tol=0.04))

    # 2x Lower-pelvis front M2x10 screws into Stage 2 at (±16.00, Z=224.00):
    for sgn, m_name in ((-1.0, "center_front_low_L_M2x10"), (1.0, "center_front_low_R_M2x10")):
        low_f_head = Pos(sgn * 16.0, -17.50, 224.0) * Rot(90, 0, 0) * Cylinder(1.90, 2.00)
        low_f_shaft = Pos(sgn * 16.0, -11.50, 224.0) * Rot(90, 0, 0) * Cylinder(1.00, 10.00)
        assets.append(_save_shape_asset(m_name, low_f_head.fuse(low_f_shaft), "hardware", "shell_fastener", tol=0.04))

    manifest_json.write_text(json.dumps(assets, indent=2))
    return assets


