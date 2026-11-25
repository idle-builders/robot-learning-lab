"""Package GLB assemblies (`assembly-with-full-shell.glb`), `assembly-parts.json` manifests
(with `display_group: "ape_body_shell"` for the "Hide shell" toggle and interactive multi-limb
shoulder/hip/neck joint controls), `carriers-4stage.step`, and high-resolution engineering
renders for all 3 Center-Body Torso FDM Carrier studies:
  1. `cove50-torso-fdm-truss-box-r1` (`truss_box`)
  2. `cove50-torso-fdm-topopt-r1` (`topopt`)
  3. `cove50-torso-fdm-combined-r1` (`combined`)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bom"))
import bom_config as BOM_CONFIG  # noqa: E402  (joint ranges from the body definitions; torso-local joint names map to the body joints)
from build123d import Compound, export_step, import_step
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.storage.blob_store import ensure, require  # noqa: E402  inputs restore by path or the build fails
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def read_json(path: Path):
    """A declared JSON input, restored by path from the bucket when it is not on disk."""
    import sys
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from infra.storage.blob_store import require
    return json.loads(Path(require(path)).read_text())


def write_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(common.ROOT))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

STAGE_COLORS = {
    "stage1_center_torso_spine_chassis": [42, 138, 184, 255],
    "stage2_lower_pelvis_battery_cradle": [28, 164, 142, 255],
    "stage3_front_power_electronics_carrier": [212, 134, 52, 255],
    "stage4_rear_imu_adc_carrier": [64, 156, 212, 255],
    "sleeve": [215, 222, 230, 255],
    "shell": [231, 222, 201, 255],
}

PRINT_ORIENTATIONS = {
    "stage1_center_torso_spine_chassis": "XY Vertical Honeycomb Magazine & Actuator Cheeks Upright on Build Plate (layer normal +Z, 127.5 x 69.2 x 120.7 mm, zero internal honeycomb support, fits 256 x 256 mm bed)",
    "stage2_lower_pelvis_battery_cradle": "XY Lower 2-Cell Coaxial Honeycomb Pelvis Column Upright on Build Plate (layer normal +Z, 50.0 x 43.3 x 78.6 mm, fits 256 x 256 mm bed)",
    "stage3_front_power_electronics_carrier": "XZ Ventral Dual-Deck Power Frame Flat on Build Plate (layer normal -Y, 80.6 x 101.6 x 23.0 mm, fits 256 x 256 mm bed)",
    "stage4_rear_imu_adc_carrier": "XZ Dorsal IMU/ADC Sensor Plate Flat on Build Plate (layer normal +Y, 35.0 x 57.0 x 5.9 mm, fits 256 x 256 mm bed)",
}

CPP_RENDERER = common.ROOT / "robot/electronics/power/cad/zbuffer_renderer.cpp"
EXE_RENDERER = common.ARTIFACT_DIR / "zbuffer_renderer_bin"


def raster_meshes(
    items: list[tuple[str, trimesh.Trimesh, list[int]]],
    view,
    box=None,
    W: int = 480,
    H: int = 720,
) -> Image.Image:
    if not EXE_RENDERER.exists():
        EXE_RENDERER.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["clang++", "-std=c++17", "-O3", str(CPP_RENDERER), "-o", str(EXE_RENDERER)],
            check=True,
        )
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
    l_key = -0.60 * view + 0.45 * right + 0.65 * up
    l_key /= np.linalg.norm(l_key)
    l_fill = -0.55 * view - 0.55 * right + 0.25 * up
    l_fill /= np.linalg.norm(l_fill)
    for _, m, c in items:
        if len(m.faces) == 0:
            continue
        tri = m.triangles
        c_rgb = np.asarray(c[:3], dtype=float) / 255.0
        n_k = np.abs(m.face_normals @ l_key)
        n_f = np.abs(m.face_normals @ l_fill)
        shade = np.clip(0.58 + 0.28 * n_k + 0.18 * n_f, 0.0, 1.04)
        tris.append(tri)
        colors.append(np.clip(c_rgb * shade[:, None], 0.0, 1.0))
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
    scale = min(W / max(hi[0] - lo[0], 1e-3), H / max(hi[1] - lo[1], 1e-3)) * 0.88
    middle = 0.5 * (lo + hi)
    xyz = np.empty_like(tri)
    xyz[:, :, 0] = (xy[:, :, 0] - middle[0]) * scale + W * 0.5
    xyz[:, :, 1] = H * 0.5 - (xy[:, :, 1] - middle[1]) * scale
    xyz[:, :, 2] = tri @ view
    data = np.concatenate([xyz.reshape(-1, 9), col], axis=1).astype("<f4")
    target = common.ARTIFACT_DIR / f"tmp-render-{os.getpid()}.ppm"
    subprocess.run(
        [str(EXE_RENDERER), "/dev/stdin", str(target)],
        input=struct.pack("<III", W, H, len(tri)) + data.tobytes(),
        check=True,
    )
    im = Image.open(target).copy()
    target.unlink(missing_ok=True)
    return im


def save_and_load_npz(mesh: trimesh.Trimesh, npz_path: Path) -> trimesh.Trimesh:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
    )
    return mesh


def source_part_rgba(p: dict) -> list[int]:
    nm = p.get("name", "")
    rl = p.get("role", "")
    if "P50B" in nm or "Cell_" in nm or "cell" in nm:
        if "terminal_PCB" in nm:
            return [42, 134, 82, 255]
        if "vented_terminal_cover" in nm or "insulator" in nm or "insulation" in nm:
            return [196, 152, 64, 255]
        return [84, 142, 168, 255]
    if any(k in nm for k in ("PCB", "board", "BMS", "IMU", "ADC", "ROCK4D", "Pololu", "OV9782", "BNO085", "ReSpeaker")) or "board" in rl:
        return [42, 134, 82, 255]
    if p.get("color") and len(p["color"]) == 4:
        return [int(x) for x in p["color"]]
    if "case" in nm or "C047" in nm or "XM335" in nm:
        return [64, 72, 82, 255]
    if "horn" in nm or "idler" in nm or "flange" in nm:
        return [182, 190, 198, 255]
    return [128, 136, 146, 255]


REPLACED_LIMB_AND_NECK_SHEET_PARTS = {
    "lower__L_A14_plain3_inside_yaw_pitch",
    "lower__R_A14_plain3_inside_yaw_pitch",
    "lower__L_A13_plain2_outside_pitch_roll",
    "lower__R_A13_plain2_outside_pitch_roll",
    "lower__L_A12_plain2_outside_roll_upper",
    "lower__R_A12_plain2_outside_roll_upper",
    "upper__L_pitch_to_roll_one_piece",
    "upper__R_pitch_to_roll_one_piece",
    "upper__L_roll_to_yaw_one_piece",
    "upper__R_roll_to_yaw_one_piece",
    "upper__L_yaw_to_elbow_one_piece",
    "upper__R_yaw_to_elbow_one_piece",
    "upper__L_elbow_to_wrist_one_piece",
    "upper__R_elbow_to_wrist_one_piece",
    "fixed_neck_ribbon_anchor_v21",
    "upper__one_piece_bent_neck_L",
    "compact_neck_U_strap_v23",
    "single_shared_carrier_v23",
}

LOCKED_FDM_NEIGHBOR_CONNECTORS = [
    ("L_hip_yaw_to_pitch_combined", "left_hip_yaw", "artifacts/robot/body/leg/complete_leg_hip_combined/parts/L_hip_yaw_to_pitch_combined.step", [76, 126, 168, 255]),
    ("R_hip_yaw_to_pitch_combined", "right_hip_yaw", "artifacts/robot/body/leg/complete_leg_hip_combined/parts/R_hip_yaw_to_pitch_combined.step", [76, 126, 168, 255]),
    ("L_hip_pitch_to_roll_combined", "left_hip_yaw", "artifacts/robot/body/leg/complete_leg_hip_combined/parts/L_hip_pitch_to_roll_combined.step", [64, 142, 132, 255]),
    ("R_hip_pitch_to_roll_combined", "right_hip_yaw", "artifacts/robot/body/leg/complete_leg_hip_combined/parts/R_hip_pitch_to_roll_combined.step", [64, 142, 132, 255]),
    ("L_shoulder_pitch_to_roll_combined", "left_shoulder_pitch", "artifacts/robot/body/arm/combined/parts/L_shoulder_pitch_to_roll_combined.step", [76, 126, 168, 255]),
    ("R_shoulder_pitch_to_roll_combined", "right_shoulder_pitch", "artifacts/robot/body/arm/combined/parts/R_shoulder_pitch_to_roll_combined.step", [76, 126, 168, 255]),
    ("L_shoulder_roll_to_yaw_combined", "left_shoulder_pitch", "artifacts/robot/body/arm/combined/parts/L_shoulder_roll_to_yaw_combined.step", [64, 142, 132, 255]),
    ("R_shoulder_roll_to_yaw_combined", "right_shoulder_pitch", "artifacts/robot/body/arm/combined/parts/R_shoulder_roll_to_yaw_combined.step", [64, 142, 132, 255]),
    ("combined__neck_root_cradle", "torso", "artifacts/robot/body/neck-head/parts/combined__neck_root_cradle.step", [76, 126, 168, 255]),
    ("combined__neck_yaw_to_pitch_link", "neck_yaw", "artifacts/robot/body/neck-head/parts/combined__neck_yaw_to_pitch_link.step", [64, 142, 132, 255]),
    ("combined__neck_pitch_to_head_cradle", "neck_yaw", "artifacts/robot/body/neck-head/parts/combined__neck_pitch_to_head_cradle.step", [88, 148, 184, 255]),
    ("combined__head_sensor_compute_carrier", "neck_yaw", "artifacts/robot/body/neck-head/parts/combined__head_sensor_compute_carrier.step", [96, 156, 192, 255]),
]


def is_electronics_or_battery_part(pname: str, role: str) -> bool:
    return any(
        k in pname
        for k in (
            "Cell_",
            "BMS_",
            "Power_supervisor_",
            "Main_precharge_",
            "Main_frame_",
            "Main_H",
            "Pololu_",
            "body_IMU_",
            "foot_ADC_",
            "IMU_",
            "ADC_",
            "ROCK4D",
            "OV9782",
            "BNO085",
            "ReSpeaker",
            "antenna",
            " coax",
            "coax",
            "harness",
            "XT30",
            "busbar",
            "standoff",
            "buck_",
        )
    ) or role in ("battery_cell", "pcb", "electronics", "sensor", "harness")


def should_include_neighbor_part(p: dict) -> bool:
    pname = p.get("name", "")
    link = p.get("owning_link", "")
    if pname in common.REPLACED_TORSO_PARTS:
        return False
    if pname in (
        "ape_body_shell__torso_front_relief4_r33",
        "ape_body_shell__torso_back_relief4_r33",
        "ape_body_shell__pelvis_front",
        "ape_body_shell__pelvis_back",
    ):
        return False
    if any(pname.startswith(pr) for pr in common.REPLACED_TORSO_FASTENER_PREFIXES):
        return False
    if pname in REPLACED_LIMB_AND_NECK_SHEET_PARTS:
        return False
    if any(pname.startswith(pr) for pr in ("lower__L_A12_", "lower__R_A12_", "lower__L_A13_", "lower__R_A13_")):
        return False
    if "wrist" in pname or "wrist" in link or "elbow" in pname or "elbow" in link:
        return False
    b = p.get("bounds_mm")
    if not b:
        return False
    bmin, bmax = b[0], b[1]
    # Keep torso, pelvis, hip yaw/pitch/roll roots, shoulder pitch/roll/yaw roots, and neck/head in the center-body scene:
    if bmax[2] < 170.0 or bmin[2] > 475.0:
        return False
    if bmin[0] > 145.0 or bmax[0] < -145.0:
        return False
    return True


def package_family(fam: str, cand_man: dict, fem_rep: dict) -> dict:
    fam_dir = common.ARTIFACT_DIR / fam
    fam_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = common.ARTIFACT_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ec = read_json(common.SOURCE_MANIFEST)

    scene = trimesh.Scene()
    scene.graph.update(frame_to="manufactured_torso_carrier_assembly", matrix=np.eye(4))
    assembly_parts = []
    render_with_shell = []
    render_no_shell = []
    render_carrier_and_joints = []
    render_carrier_only = []
    stage_meshes = {}

    def _add_to_scene(
        name: str,
        m_mm: trimesh.Trimesh,
        rgba: list[int],
        entry: dict,
        is_shell: bool = False,
        is_elec: bool = False,
        is_carrier: bool = False,
    ):
        m_m = m_mm.copy()
        m_m.apply_scale(0.001)
        m_m.visual.face_colors = np.asarray(rgba, dtype=np.uint8)
        scene.add_geometry(
            m_m,
            node_name=name,
            geom_name=name,
            parent_node_name="manufactured_torso_carrier_assembly",
        )
        assembly_parts.append(entry)
        bmin, bmax = m_mm.bounds
        if bmax[2] >= 170.0 and bmin[2] <= 475.0 and bmin[0] <= 145.0 and bmax[0] >= -145.0:
            render_with_shell.append((name, m_mm, rgba))
            if not is_shell:
                render_no_shell.append((name, m_mm, rgba))
                if not is_elec:
                    render_carrier_and_joints.append((name, m_mm, rgba))
            if is_carrier:
                render_carrier_only.append((name, m_mm, rgba))

    # 1. Retained torso, battery, electronics, actuator, and shell parts from SOURCE_MANIFEST
    for p in ec["parts"]:
        if not should_include_neighbor_part(p):
            continue
        pname = p["name"]
        m_mm = common.load_manifest_part_mesh(p)
        if m_mm is None:
            continue
        mp = p.get("mesh")
        if not mp or not ensure(common.ROOT / mp).exists():
            mp = rel(common.ARTIFACT_DIR / "neighbor_meshes_cache" / f"{pname}.npz")
        is_sh = bool(
            p.get("role") == "body_shell"
            or "shell" in pname
            or "cowl" in pname
            or "skin" in pname
        )
        is_el = (not is_sh) and is_electronics_or_battery_part(pname, p.get("role", ""))
        if pname == "ape_body_shell__chest_indicator_relief4_r33":
            rgba = [205, 120, 76, 255]  # Signature orange chest indicator (#cd784c)
        else:
            rgba = STAGE_COLORS["shell"] if is_sh else source_part_rgba(p)
        d_group = "ape_body_shell" if is_sh else "mechanism"
        entry = {
            "name": pname,
            "owning_link": p.get("owning_link", "torso"),
            "role": "body_shell" if is_sh else p.get("role", "component"),
            "mesh": mp,
            "mesh_units": "mm",
            "bounds_mm": [m_mm.bounds[0].round(3).tolist(), m_mm.bounds[1].round(3).tolist()],
            "color": rgba,
            "display_group": d_group,
            "display_groups": [d_group],
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": is_sh,
        }
        if is_sh and pname != "ape_body_shell__chest_indicator_relief4_r33":
            entry["display_opacity"] = 0.42
        elif is_el:
            entry["overlay_id"] = "electronics_and_battery"
        else:
            entry["overlay_id"] = "limb_and_neck_actuators"
        if p.get("native_step"):
            entry["native_step"] = p["native_step"]
        _add_to_scene(pname, m_mm, rgba, entry, is_shell=is_sh, is_elec=is_el, is_carrier=False)

    # 1b. Add locked-in FDM Hip, Shoulder, and Neck/Head connectors so the full Hip-to-Shoulder-to-Head structural chain is complete:
    for conn_name, owning_link, step_rel, conn_rgba in LOCKED_FDM_NEIGHBOR_CONNECTORS:
        m_conn = common.load_manifest_part_mesh({"name": conn_name, "native_step": step_rel})
        if m_conn is None:
            continue
        npz_rel = rel(common.ARTIFACT_DIR / "neighbor_meshes_cache" / f"{conn_name}.npz")
        conn_entry = {
            "name": conn_name,
            "owning_link": owning_link,
            "role": "connected_fdm_joint_bracket",
            "native_step": step_rel,
            "mesh": npz_rel,
            "mesh_units": "mm",
            "bounds_mm": [m_conn.bounds[0].round(3).tolist(), m_conn.bounds[1].round(3).tolist()],
            "color": conn_rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "overlay_id": "limb_and_neck_actuators",
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(conn_name, m_conn, conn_rgba, conn_entry, is_shell=False, is_elec=False, is_carrier=False)

    # 1c. Add the 9 Repositioned Vertical Honeycomb P50B 21700 Battery Cells (5+2 upper spine + 2 lower pelvis):
    nb_cache_dir = common.ARTIFACT_DIR / "neighbor_meshes_cache"
    nb_cache_dir.mkdir(parents=True, exist_ok=True)
    for c in common.HONEYCOMB_9CELL_BATTERY_LAYOUT:
        cname = c["name"]
        cx, cy, cz = c["center"]
        r_can = c["r_can_mm"]
        h_can = c["h_can_mm"]
        m_cell = trimesh.creation.cylinder(radius=r_can, height=h_can, sections=48)
        m_cell.apply_translation([cx, cy, cz])
        c_npz = nb_cache_dir / f"{cname}.npz"
        save_and_load_npz(m_cell, c_npz)
        c_rgba = [84, 142, 168, 255]
        c_entry = {
            "name": cname,
            "owning_link": "torso",
            "role": "battery_cell",
            "mesh": rel(c_npz),
            "mesh_units": "mm",
            "bounds_mm": [m_cell.bounds[0].round(3).tolist(), m_cell.bounds[1].round(3).tolist()],
            "color": c_rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "overlay_id": "electronics_and_battery",
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(cname, m_cell, c_rgba, c_entry, is_shell=False, is_elec=True, is_carrier=False)

        # Top amber dielectric ring on each honeycomb cell for clear visual orientation:
        ring_name = f"{cname}_top_insulator"
        m_ring = trimesh.creation.cylinder(radius=r_can - 0.35, height=1.2, sections=48)
        m_ring.apply_translation([cx, cy, cz + 0.5 * h_can - 0.55])
        r_npz = nb_cache_dir / f"{ring_name}.npz"
        save_and_load_npz(m_ring, r_npz)
        r_rgba = [196, 152, 64, 255]
        r_entry = {
            "name": ring_name,
            "owning_link": "torso",
            "role": "battery_insulator",
            "mesh": rel(r_npz),
            "mesh_units": "mm",
            "bounds_mm": [m_ring.bounds[0].round(3).tolist(), m_ring.bounds[1].round(3).tolist()],
            "color": r_rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "overlay_id": "electronics_and_battery",
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(ring_name, m_ring, r_rgba, r_entry, is_shell=False, is_elec=True, is_carrier=False)

        # Direct-on-battery 6063-T5 anodized aluminum 4-blade vertical heat-sink fin strip inside the honeycomb window slot:
        fin_name = c.get("fin_name", f"{cname}_al_heatsink_fin")
        sh_fin = common.build_cell_aluminum_fin_shape(c)
        m_fin = common.solid_to_watertight_mesh(sh_fin, tol=0.04)
        f_npz = nb_cache_dir / f"{fin_name}.npz"
        save_and_load_npz(m_fin, f_npz)
        f_rgba = [214, 222, 232, 255]  # Bright anodized aluminum silver
        f_entry = {
            "name": fin_name,
            "owning_link": "torso",
            "role": "battery_heatsink_fin",
            "mesh": rel(f_npz),
            "mesh_units": "mm",
            "bounds_mm": [m_fin.bounds[0].round(3).tolist(), m_fin.bounds[1].round(3).tolist()],
            "color": f_rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "overlay_id": "electronics_and_battery",
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(fin_name, m_fin, f_rgba, f_entry, is_shell=False, is_elec=True, is_carrier=False)

    # 1d. Add the Mounted 2-Piece Center Shell (center_front_shell, center_back_shell),
    # the Chest Indicator (chest_indicator), and the 10 Direct Shell Fasteners:
    shell_connector_meshes = []
    for sh_asset in common.ensure_mounted_shell_and_connector_assets():
        sh_name = sh_asset["name"]
        d_npz = np.load(common.ROOT / sh_asset["npz_path"])
        m_sh = trimesh.Trimesh(vertices=d_npz["vertices"], faces=d_npz["faces"], process=False)
        sh_role = sh_asset["role"]
        is_sh_panel = sh_asset["display_group"] == "ape_body_shell"
        if is_sh_panel and sh_role == "mounted_center_shell":
            sh_rgba = STAGE_COLORS["shell"]
            d_grp = "ape_body_shell"
        else:
            sh_rgba = [192, 198, 206, 255]
            d_grp = "ape_body_shell"
            shell_connector_meshes.append((sh_name, m_sh, sh_rgba))
        sh_entry = {
            "name": sh_name,
            "owning_link": "torso",
            "role": "body_shell" if sh_role == "mounted_center_shell" else sh_role,
            "mesh": sh_asset["npz_path"],
            "mesh_units": "mm",
            "bounds_mm": sh_asset["bounds_mm"],
            "color": sh_rgba,
            "display_group": d_grp,
            "display_groups": [d_grp],
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": is_sh_panel,
        }
        if is_sh_panel and sh_role == "mounted_center_shell":
            sh_entry["display_opacity"] = 0.42
        if sh_asset.get("step_path"):
            sh_entry["native_step"] = sh_asset["step_path"]
        _add_to_scene(sh_name, m_sh, sh_rgba, sh_entry, is_shell=(d_grp == "ape_body_shell"), is_elec=False, is_carrier=False)

    # 2. Add the 4 FDM torso carrier stages + 56 pressed-in 304 SS compression sleeves
    compound_solids = []
    bom_items = []
    manifest_stages = {}
    manifest_parts_out = []

    for stg_key in common.STAGES:
        if fam == "combined":
            sel_fam = fem_rep["families"]["combined"]["stage_selections"][stg_key]["selected_family"]
        else:
            sel_fam = fam

        p_rec = cand_man["families"][sel_fam]["stages"][stg_key]
        # always the declared STEP, meshed here: never a leftover STL or an earlier run's npz
        src_step = require(common.ROOT / p_rec["step_path"])
        m_mm = common.solid_to_watertight_mesh(import_step(str(src_step)))
        stage_meshes[stg_key] = m_mm

        part_name = f"{fam}__{stg_key}"
        step_p = parts_dir / f"{part_name}.step"
        npz_p = parts_dir / f"{part_name}.npz"
        shutil.copy2(src_step, step_p)
        save_and_load_npz(m_mm, npz_p)
        compound_solids.extend(import_step(str(step_p)).solids())

        rgba = STAGE_COLORS[stg_key]
        p_orient = PRINT_ORIENTATIONS[stg_key]
        specs = common.stage_interface_specs()[stg_key]
        slv_count = len(specs["support_bores"]) + len(specs["load_bores"]) + len(specs.get("aux_bores", []))

        poly_entry = {
            "name": part_name,
            "owning_link": "torso",
            "role": "printed_structural_carrier",
            "physical_part_id": part_name,
            "bom_group": part_name,
            "native_step": rel(step_p),
            "native_step_sha256": sha256_file(step_p),
            "mesh": rel(npz_p),
            "mesh_sha256": sha256_file(npz_p),
            "mesh_units": "mm",
            "bounds_mm": p_rec["bounds_mm"],
            "native_solid_count": 1,
            "native_validity": [True],
            "material": f"{common.BOM_CONFIG.material(part_name, 'build-torso-carriers')} FDM",
            **common.BOM_CONFIG.mass_record(part_name, p_rec["polymer_volume_mm3"], "build-torso-carriers",
                                            basis="Exact native STEP polymer volume"),   # declared material
            "color": rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": False,
            "volume_mm3": p_rec["polymer_volume_mm3"],
            "print_orientation": p_orient,
        }
        _add_to_scene(part_name, m_mm, rgba, poly_entry, is_shell=False, is_elec=False, is_carrier=True)

        # Stage 304 SS sleeves
        slv_src_step = require(common.ROOT / p_rec["sleeve_step_path"])
        slv_comp_step = fam_dir / f"{stg_key}_sleeves.step"
        slv_comp_npz = fam_dir / f"{stg_key}_sleeves.npz"
        m_slv_all = common.solid_to_watertight_mesh(import_step(str(slv_src_step)))
        shutil.copy2(slv_src_step, slv_comp_step)
        save_and_load_npz(m_slv_all, slv_comp_npz)
        slv_solids = import_step(str(slv_comp_step)).solids()
        compound_solids.extend(slv_solids)

        slv_name = f"{part_name}_steel_sleeves"
        slv_entry = {
            "name": slv_name,
            "owning_link": "torso",
            "role": "compression_limit_sleeves",
            "physical_part_id": slv_name,
            "bom_group": slv_name,
            "native_step": rel(slv_comp_step),
            "native_step_sha256": sha256_file(slv_comp_step),
            "mesh": rel(slv_comp_npz),
            "mesh_sha256": sha256_file(slv_comp_npz),
            "mesh_units": "mm",
            "bounds_mm": [m_slv_all.bounds[0].round(3).tolist(), m_slv_all.bounds[1].round(3).tolist()],
            "native_solid_count": len(slv_solids),
            "native_validity": [True] * len(slv_solids),
            "material": f"304 Stainless Steel compression sleeves ({common.SLEEVE['density_g_cm3']:g} g/cm3, {common.SLEEVE['name']})",
            "mass_kg": round(p_rec["sleeve_mass_ss304_g"] / 1000.0, 6),
            "color": STAGE_COLORS["sleeve"],
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "parent_assembly": "manufactured_torso_carrier_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(slv_name, m_slv_all, STAGE_COLORS["sleeve"], slv_entry, is_shell=False, is_elec=False, is_carrier=True)

        manifest_stages[stg_key] = {
            "stage": stg_key,
            "selected_family": sel_fam,
            "owning_link": "torso",
            "extents_mm": p_rec["extents_mm"],
            "volume_mm3": p_rec["polymer_volume_mm3"],
            "polymer_mass_paht_cf_g": p_rec["polymer_mass_paht_cf_g"],
            "sleeves_count": slv_count,
            "sleeves_mass_g": p_rec["sleeve_mass_ss304_g"],
            "total_stage_mass_paht_cf_g": p_rec["total_stage_mass_g"],
            "step": rel(step_p),
            "mesh": rel(npz_p),
            "sleeves_step": rel(slv_comp_step),
            "sleeves_mesh": rel(slv_comp_npz),
        }
        manifest_parts_out.append(manifest_stages[stg_key])

        bom_items.append({
            "id": stg_key,
            "name": f"{stg_key} FDM structural carrier ({sel_fam})",
            "category": "Produced parts (FDM 3D printed)",
            "quantity": 1,
            "unit": "ea",
            "material": common.BOM_CONFIG.material(part_name, "build-torso-carriers"),
            "unit_mass_g": round(common.BOM_CONFIG.mass_kg(part_name, p_rec["polymer_volume_mm3"], "build-torso-carriers") * 1000.0, 3),
            "process": f"Home-lab FDM 3D print ({p_orient})",
            "notes": f"One-piece 3D-printed structural torso carrier with {slv_count} pressed-in 304 SS compression sleeve seats and asymmetric 3D bevels (r_inner > r_outer). Fits standard 256 x 256 mm FDM print bed.",
        })
        bom_items.append({
            "id": f"{stg_key}_sleeves",
            "name": f"{stg_key} 304 SS compression sleeve set ({slv_count} sleeves)",
            "category": "Fasteners & Hardware",
            "quantity": 1,
            "unit": "set",
            "material": "304 Stainless Steel",
            "unit_mass_g": p_rec["sleeve_mass_ss304_g"],
            "process": "Off-the-shelf 304 SS cylindrical sleeves pressed into printed counterbores",
            "notes": f"{slv_count} steel sleeves prevent polymer preload creep across shoulder/hip/neck actuators, battery cradle lugs, and PCB standoffs.",
        })

    total_poly_g = round(sum(s["polymer_mass_paht_cf_g"] for s in manifest_stages.values()), 3)
    total_slv_g = round(sum(s["sleeves_mass_g"] for s in manifest_stages.values()), 3)
    total_mass_g = round(total_poly_g + total_slv_g, 3)

    write_json(
        fam_dir / "manifest.json",
        {
            "family": fam,
            "total_polymer_mass_paht_cf_g": total_poly_g,
            "total_sleeves_mass_g": total_slv_g,
            "total_4stage_mass_paht_cf_g": total_mass_g,
            "total_sleeves_count": sum(s["sleeves_count"] for s in manifest_stages.values()),
            "stages": manifest_stages,
            "parts": manifest_parts_out,
        },
    )

    step_4stage = fam_dir / "carriers-4stage.step"
    export_step(Compound(children=compound_solids), str(step_4stage))

    glb_path = fam_dir / "assembly-with-full-shell.glb"
    scene.export(str(glb_path))

    torso_joints = [
        {
            "name": "left_shoulder_pitch",
            "parent": "torso",
            "child": "left_shoulder_pitch",
            "origin_world_mm": common.LEFT_SHOULDER_PITCH_AXIS_ORIGIN.tolist(),
            "axis_world_neutral": [-1.0, 0.0, 0.0],
            "range_deg": BOM_CONFIG.joint_range("L_shoulder_pitch"),
        },
        {
            "name": "right_shoulder_pitch",
            "parent": "torso",
            "child": "right_shoulder_pitch",
            "origin_world_mm": common.RIGHT_SHOULDER_PITCH_AXIS_ORIGIN.tolist(),
            "axis_world_neutral": [1.0, 0.0, 0.0],
            "range_deg": BOM_CONFIG.joint_range("R_shoulder_pitch"),
        },
        {
            "name": "left_hip_yaw",
            "parent": "torso",
            "child": "left_hip_yaw",
            "origin_world_mm": common.LEFT_HIP_YAW_AXIS_ORIGIN.tolist(),
            "axis_world_neutral": [0.0, 0.0, 1.0],
            "range_deg": BOM_CONFIG.joint_range("L_hip_yaw"),
        },
        {
            "name": "right_hip_yaw",
            "parent": "torso",
            "child": "right_hip_yaw",
            "origin_world_mm": common.RIGHT_HIP_YAW_AXIS_ORIGIN.tolist(),
            "axis_world_neutral": [0.0, 0.0, 1.0],
            "range_deg": BOM_CONFIG.joint_range("R_hip_yaw"),
        },
        {
            "name": "neck_yaw",
            "parent": "torso",
            "child": "neck_yaw",
            "origin_world_mm": common.NECK_YAW_AXIS_ORIGIN.tolist(),
            "axis_world_neutral": [0.0, 0.0, 1.0],
            "range_deg": BOM_CONFIG.joint_range("neck_yaw"),
        },
    ]

    write_json(
        fam_dir / "assembly-parts.json",
        {
            "family": fam,
            "root_link": "torso",
            "units": "CAD mm; GLB m",
            "parts": assembly_parts,
            "bom": {
                "scope": f"4-stage Center-Body Torso & Electronics FDM Carrier Architecture ({fam}, consolidating 24 legacy torso/pelvis/battery/electronics parts into 4 single-solid stages) and pressed-in 304 SS compression sleeves (count: the bom items)",
                "completeness": "partial",
                "notes": "Consolidates 24 parts (Common_power_carrier_mounting_r33, Cell_capture_bridge_00..14, lower_cradle_lugs_224_with_waist_guard_r18, lower_cradle_lugs_252_retained_outer_r18, Power_supervisor_PA12_open_tray, Main_precharge_PA12_open_tray, BMS_recessed_fastener_front_guard, Cell_07_removable_front_clamp_L/R, and IMU_ADC_carrier_v17) into 4 3D-printable PAHT-CF stages.",
                "exclusions": ["Retained C047/XM335 actuators, 9x P50B battery cells, PCBs, harnesses, and exterior torso/pelvis/head shells"],
                "items": bom_items,
            },
            "joints": torso_joints,
        },
    )

    # Render 4-panel shell-fit, electronics-fit, and bare carrier views
    renders_dir = fam_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1920, 780), "#f3f1eb")
    draw = ImageDraw.Draw(canvas)
    box_torso = [[-135.0, -52.0, 175.0], [135.0, 52.0, 465.0]]
    box_carrier = [[-75.0, -45.0, 185.0], [65.0, 45.0, 395.0]]
    panels = [
        ("1. Shells On (Torso + Pelvis + Head)", render_with_shell, [-0.28, -0.94, 0.14], box_torso),
        ("2. Shells Off: 9-Cell Battery + PCBs + Hip/Shoulder/Head", render_no_shell, [-0.35, -0.92, 0.15], box_torso),
        ("3. Shells & Electronics Hidden: Hip-to-Shoulder-to-Head", render_carrier_and_joints, [-0.38, -0.90, 0.18], box_torso),
        (f"4. Bare 4-Stage {fam} Carrier + Steel Sleeves (Z=190..389 mm)", render_carrier_only, [-0.52, -0.82, 0.22], box_carrier),
    ]
    for idx, (title, items_list, v_dir, b_box) in enumerate(panels):
        im = raster_meshes(items_list, v_dir, box=b_box, W=480, H=720)
        canvas.paste(im, (idx * 480, 40))
        draw.text((idx * 480 + 10, 12), title, fill="#22282e")
    out_shell_fit = renders_dir / "shell-fit-views.png"
    canvas.save(out_shell_fit)

    # Render dedicated 4-panel close-up of the 9 Honeycomb Window Slots & 9x Direct-on-Battery Aluminum Heat-Sink Fins:
    carrier_s1_s2 = [it for it in render_carrier_only if any(k in it[0] for k in ("stage1", "stage2"))]
    batt_and_fins = [it for it in render_no_shell if any(k in it[0] for k in ("Honeycomb_Cell", "Honeycomb_Fin", "top_insulator"))]
    canvas_fins = Image.new("RGB", (1920, 780), "#f3f1eb")
    draw_fins = ImageDraw.Draw(canvas_fins)
    fin_panels = [
        ("1. Dorsal (+Y) View: 7 Rear Honeycomb Slots + 7x Glued Battery Al Fins", carrier_s1_s2 + batt_and_fins, [0.46, 0.85, 0.24], box_carrier),
        ("2. Dorsal (+Y) Bare Honeycomb: 7 Clean Vertical Fin-Exposure Slots (S1+S2)", carrier_s1_s2, [0.46, 0.85, 0.24], box_carrier),
        ("3. Ventral (-Y) View: 2 Front Core Honeycomb Slots + 2x Glued Battery Al Fins", carrier_s1_s2 + batt_and_fins, [-0.46, -0.85, 0.24], box_carrier),
        ("4. Top-3/4 View: 9 Cells, 9x 4-Blade Al Fins in Slots & 3 Interstitial Chimneys", carrier_s1_s2 + batt_and_fins, [0.38, 0.58, 0.72], box_carrier),
    ]
    for idx, (title, items_list, v_dir, b_box) in enumerate(fin_panels):
        im = raster_meshes(items_list, v_dir, box=b_box, W=480, H=720)
        canvas_fins.paste(im, (idx * 480, 40))
        draw_fins.text((idx * 480 + 10, 12), title, fill="#22282e")
    out_fins_view = renders_dir / "battery-fins-cad-views.png"
    canvas_fins.save(out_fins_view)

    # Render dedicated 4-panel close-up of ALL Inter-Stage & Direct 2-Piece Center-Shell Connections to Stage 1 & Stage 2:
    box_waist = [[-88.0, -52.0, 185.0], [88.0, 52.0, 398.0]]
    box_dorsal_s4 = [[-56.0, 10.0, 192.0], [56.0, 48.0, 384.0]]
    front_shell_items = [it for it in render_with_shell if it[0] in ("center_front_shell", "ape_body_shell__chest_indicator_relief4_r33")]
    back_shell_items = [it for it in render_with_shell if it[0] == "center_back_shell"]
    canvas_clevis = Image.new("RGB", (1920, 780), "#f3f1eb")
    draw_clevis = ImageDraw.Draw(canvas_clevis)
    clevis_panels = [
        ("1. Dorsal View: 3x Stage 1 (+10.1,-49.9,+49.9) & 1x Stage 2 (0,204) Shell Mounts", render_carrier_only + batt_and_fins + shell_connector_meshes, [-0.52, 0.82, 0.22], box_dorsal_s4),
        ("2. Ventral View: Stage 3 Standoffs (±17.0) & 2x Stage 2 Front Shell Mounts (±16,224)", render_carrier_only + shell_connector_meshes, [-0.55, -0.80, 0.20], [[-58.0, -40.0, 192.0], [58.0, 48.0, 384.0]]),
        ("3. Inside Back View of center_front_shell: 2x Side Couplers + 2x Lower Inward Posts", front_shell_items + carrier_s1_s2 + shell_connector_meshes, [0.32, 0.92, 0.20], box_waist),
        ("4. Inside Front View of center_back_shell: 4x Dorsal Inward Bosses + 2x Side Couplers", back_shell_items + carrier_s1_s2 + shell_connector_meshes, [-0.32, -0.92, 0.20], box_waist),
    ]
    for idx, (title, items_list, v_dir, b_box) in enumerate(clevis_panels):
        im = raster_meshes(items_list, v_dir, box=b_box, W=480, H=720)
        canvas_clevis.paste(im, (idx * 480, 40))
        draw_clevis.text((idx * 480 + 10, 12), title, fill="#22282e")
    out_clevis_view = renders_dir / "waist-clevis-joint-views.png"
    canvas_clevis.save(out_clevis_view)

    return {
        "glb_path": glb_path,
        "step_path": step_4stage,
        "stage_meshes": stage_meshes,
    }


def render_comparison_charts(stage_meshes_by_fam: dict, fem_rep: dict):
    renders_dir = common.ARTIFACT_DIR / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage-by-stage CAD comparison grid (`baseline_metal` vs `truss_box` vs `topopt` vs `combined`)
    canvas = Image.new("RGB", (1920, 1480), "#f3f1eb")
    draw = ImageDraw.Draw(canvas)
    stage_titles = [
        ("stage1_center_torso_spine_chassis", "Stage 1: Honeycomb Spine (36 sleeves)", [-0.52, -0.82, 0.24]),
        ("stage2_lower_pelvis_battery_cradle", "Stage 2: Pelvis U-Clevis (10 sleeves)", [-0.58, -0.75, 0.30]),
        ("stage3_front_power_electronics_carrier", "Stage 3: Dual-Deck Power (22 sleeves)", [-0.45, -0.85, 0.22]),
        ("stage4_rear_imu_adc_carrier", "Stage 4: Dorsal IMU/ADC (14 sleeves)", [-0.42, 0.88, 0.22]),
    ]

    # Row 0: Baseline parts
    for c_idx, (st_key, st_title, v_dir) in enumerate(stage_titles):
        base_meshes = []
        for p in common.BASELINE_STAGE_SOURCES[st_key]:
            bm = common.solid_to_watertight_mesh(import_step(str(require(p))))
            base_meshes.append((st_key, bm, [142, 150, 160, 255]))
        im = raster_meshes(base_meshes, v_dir, W=480, H=330)
        canvas.paste(im, (c_idx * 480, 35))
        b_mass = fem_rep["families"]["baseline_metal"]["stages"][st_key]["mass_g"]
        draw.text(
            (c_idx * 480 + 12, 10),
            f"{st_title} - Baseline ({b_mass:.2f} g)",
            fill="#22282e",
        )

    row_fams = [
        ("truss_box", "Family 1: Truss/Box"),
        ("topopt", "Family 2: 3D SIMP TopOpt"),
        ("combined", "Combined Optimal 4-Stage Torso"),
    ]
    for r_idx, (fam_key, fam_label) in enumerate(row_fams, start=1):
        man = read_json(common.ARTIFACT_DIR / fam_key / "manifest.json")
        for c_idx, (st_key, st_title, v_dir) in enumerate(stage_titles):
            m = stage_meshes_by_fam[fam_key][st_key]
            slv_npz = common.ARTIFACT_DIR / fam_key / f"{st_key}_sleeves.npz"
            z = np.load(slv_npz)
            m_slv = trimesh.Trimesh(vertices=z["vertices"], faces=z["faces"], process=False)
            im = raster_meshes(
                [(st_key, m, STAGE_COLORS[st_key]), (f"{st_key}_slv", m_slv, STAGE_COLORS["sleeve"])],
                v_dir,
                W=480,
                H=330,
            )
            canvas.paste(im, (c_idx * 480, r_idx * 365 + 35))
            mass_g = man["stages"][st_key]["total_stage_mass_paht_cf_g"]
            sel_tag = f" [{man['stages'][st_key]['selected_family']}]" if fam_key == "combined" else ""
            draw.text(
                (c_idx * 480 + 12, r_idx * 365 + 10),
                f"{st_title} - {fam_label}{sel_tag} ({mass_g:.2f} g)",
                fill="#22282e",
            )
    out_stage_cad = renders_dir / "stage-by-stage-cad.png"
    canvas.save(out_stage_cad)

    # 2. FEM stress, displacement, and Tsai-Hill index chart (`fem-stress-displacement-comparison.png`)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160)
    stages = list(common.STAGES)
    short_labels = ["S1: Torso Spine", "S2: Pelvis Cradle", "S3: Power Tray", "S4: IMU/ADC"]
    x = np.arange(len(stages))
    w = 0.20

    fams_plot = [
        ("baseline_metal", "Baseline (AlSi10Mg + PA12)", "#7a8694"),
        ("truss_box", "Family 1: Truss/Box (PAHT-CF)", "#2a8ab8"),
        ("topopt", "Family 2: SIMP TopOpt (PAHT-CF)", "#1ca48e"),
        ("combined", "Combined Optimal (PAHT-CF)", "#d48634"),
    ]
    for idx, (fk_name, label, col) in enumerate(fams_plot):
        u_vals = []
        p99_vals = []
        sf_vals = []
        for st in stages:
            st_block = fem_rep["families"][fk_name]["stages"][st]
            m_key = st_block.get("baseline_material", "PAHT_CF_dry") if fk_name == "baseline_metal" else "PAHT_CF_dry"
            row = st_block["materials"][m_key]["combined_shoulder_hip_neck_stall"]
            u_vals.append(row["disp_max_mm"])
            p99_vals.append(row["vm_p99_mpa"])
            sf_vals.append(round(1.0 / max(row["tsai_hill_p99"], 0.02), 2))
        axes[0].bar(x + (idx - 1.5) * w, u_vals, width=w, label=label, color=col)
        axes[1].bar(x + (idx - 1.5) * w, p99_vals, width=w, label=label, color=col)
        axes[2].bar(x + (idx - 1.5) * w, sf_vals, width=w, label=label, color=col)

    axes[0].set_title("Peak Deflection @ Multi-Limb Stall (mm)")
    axes[0].set_xticks(x, short_labels, rotation=15)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Von Mises p99 Stress @ Stall (MPa)")
    axes[1].set_xticks(x, short_labels, rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].set_title("Orthotropic Tsai-Hill Margin (1 / TH_p99)")
    axes[2].set_xticks(x, short_labels, rotation=15)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "APE0 4-Stage Center-Body Torso FDM Carriers — Matched 3D Tetrahedral FEM (Combined Multi-Limb Stall)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_fem = renders_dir / "fem-stress-displacement-comparison.png"
    fig.savefig(out_fem)
    plt.close(fig)

    # 3. Weight-to-strength + FDM Print-Bed Footprint comparison (`weight-to-strength-chart.png`)
    base_audit = read_json(common.ARTIFACT_DIR / "torso-baseline-audit.json")
    base_mass_g = float(base_audit["total_baseline_mass_g"])
    man_tb = read_json(common.ARTIFACT_DIR / "truss_box" / "manifest.json")
    man_tp = read_json(common.ARTIFACT_DIR / "topopt" / "manifest.json")
    man_cb = read_json(common.ARTIFACT_DIR / "combined" / "manifest.json")

    def _rss_u(fk: str) -> float:
        vals = []
        for st in stages:
            st_block = fem_rep["families"][fk]["stages"][st]
            m_key = st_block.get("baseline_material", "PAHT_CF_dry") if fk == "baseline_metal" else "PAHT_CF_dry"
            vals.append(st_block["materials"][m_key]["combined_shoulder_hip_neck_stall"]["disp_max_mm"] ** 2)
        return float(np.sqrt(sum(vals)))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=160)
    names = ["Baseline\n(AlSi10Mg + 6 PA12)", "Family 1\n(Truss/Box)", "Family 2\n(SIMP TopOpt)", "Combined\nOptimal"]
    cols = ["#7a8694", "#2a8ab8", "#1ca48e", "#d48634"]
    masses = [
        base_mass_g,
        man_tb["total_4stage_mass_paht_cf_g"],
        man_tp["total_4stage_mass_paht_cf_g"],
        man_cb["total_4stage_mass_paht_cf_g"],
    ]
    rss_u = [_rss_u("baseline_metal"), _rss_u("truss_box"), _rss_u("topopt"), _rss_u("combined")]

    axes[0].bar(names, masses, color=cols, width=0.52)
    axes[0].set_title("4-Stage Torso Mass incl. 56 Sleeves (g)")
    axes[0].set_ylabel("Total 4-stage mass (g)")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(masses):
        axes[0].text(i, v + 3.0, f"{v:.2f} g", ha="center", fontweight="bold", fontsize=9)

    axes[1].bar(names, rss_u, color=cols, width=0.52)
    axes[1].set_title("4-Stage RSS Deflection @ Multi-Limb Stall (mm)")
    axes[1].set_ylabel("RSS Displacement (mm)")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(rss_u):
        axes[1].text(i, v + 0.06, f"{v:.2f} mm", ha="center", fontweight="bold", fontsize=9)

    # Panel 3: Print-bed envelope comparison (max dimension of each stage vs 256 mm and 350 mm FDM print beds)
    s1_max = max(man_cb["stages"]["stage1_center_torso_spine_chassis"]["extents_mm"])
    s2_max = max(man_cb["stages"]["stage2_lower_pelvis_battery_cradle"]["extents_mm"])
    s3_max = max(man_cb["stages"]["stage3_front_power_electronics_carrier"]["extents_mm"])
    s4_max = max(man_cb["stages"]["stage4_rear_imu_adc_carrier"]["extents_mm"])
    bed_labels = [
        f"S1: Honeycomb\n({s1_max:.1f} mm)",
        f"S2: Pelvis\n({s2_max:.1f} mm)",
        f"S3: Power\n({s3_max:.1f} mm)",
        f"S4: IMU/ADC\n({s4_max:.1f} mm)",
        "Full 4-Stage\n(193.0 mm)",
    ]
    max_dims = [s1_max, s2_max, s3_max, s4_max, 193.0]
    axes[2].bar(bed_labels, max_dims, color=["#2a8ab8", "#1ca48e", "#d48634", "#409cd4", "#5c6b7a"], width=0.52)
    axes[2].axhline(256.0, color="#c93b2b", linestyle="--", linewidth=1.8, label="Standard 26x26 cm Bed (256 mm)")
    axes[2].axhline(350.0, color="#2e7d32", linestyle=":", linewidth=1.8, label="Large 35x35 cm Bed (350 mm)")
    axes[2].set_ylim(0, 385)
    axes[2].set_title("Max Span vs 26x26 cm & 35x35 cm FDM Beds (mm)")
    axes[2].set_ylabel("Max Bounding Dimension (mm)")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=8, loc="upper left")
    for i, v in enumerate(max_dims):
        axes[2].text(i, v + 6.0, f"{v:.1f} mm", ha="center", fontweight="bold", fontsize=8.5)

    cb_mass_g = man_cb["total_4stage_mass_paht_cf_g"]
    mass_red_pct = 100.0 * (base_mass_g - cb_mass_g) / base_mass_g
    fig.suptitle(
        f"APE0 Honeycomb Torso — {mass_red_pct:.1f}% Mass Reduction ({base_mass_g:.2f} g -> {cb_mass_g:.2f} g), Chain Deflection & FDM Bed Fit",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    out_w2s = renders_dir / "weight-to-strength-chart.png"
    fig.savefig(out_w2s)
    plt.close(fig)

    # 4. 3D SIMP Topology Optimization density projection plot (`topopt-density-and-loadpaths.png`)
    topopt_npz = np.load(require(common.ARTIFACT_DIR / "torso-topopt-fields.npz"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=160)
    for idx, (prefix, title) in enumerate([
        ("stage1_center_torso_spine_chassis", "Stage 1: Center Torso Spine SIMP"),
        ("stage2_lower_pelvis_battery_cradle", "Stage 2: Pelvis Battery Cradle SIMP"),
        ("stage3_front_power_electronics_carrier", "Stage 3: Front Power Carrier SIMP"),
        ("stage4_rear_imu_adc_carrier", "Stage 4: Rear IMU/ADC Carrier SIMP"),
    ]):
        rho = topopt_npz[f"{prefix}_density"]
        proj = np.max(rho, axis=1).T
        im = axes[idx].imshow(proj, origin="lower", cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        axes[idx].set_title(title, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("X voxel index")
        axes[idx].set_ylabel("Z voxel index")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="SIMP Relative Density ρ")
    fig.suptitle(
        "3D Multi-Load SIMP Topology Optimization Density Projections (Center-Body Torso Stages 1–4)",
        fontsize=12,
        fontweight="bold",
    )
    out_topopt = renders_dir / "topopt-density-and-loadpaths.png"
    fig.savefig(out_topopt, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="combined",
                    help="comma-separated: combined (the selected design; default), truss_box, topopt (their inputs are no longer in the tree)")
    a = ap.parse_args()
    fams = [f for f in a.families.split(",") if f]
    cand_man = read_json(common.ARTIFACT_DIR / "torso-candidates-manifest.json")
    fem_rep = read_json(common.ARTIFACT_DIR / "torso-fem-results.json")
    stage_meshes_by_fam = {}
    for fam in fams:
        res = package_family(fam, cand_man, fem_rep)
        stage_meshes_by_fam[fam] = res["stage_meshes"]
        print(f"Packaged {fam}: {res['glb_path'].name} and {res['step_path'].name}")
    if set(fams) >= {"truss_box", "topopt", "combined"}:
        # the old three-family comparison; the published FEM chart is drawn by the publisher from fem-evidence.json
        render_comparison_charts(stage_meshes_by_fam, fem_rep)
        print("Rendered all comparison figures in", common.ARTIFACT_DIR / "renders")


if __name__ == "__main__":
    main()
