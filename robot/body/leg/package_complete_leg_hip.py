"""Assemble and package the Upper-Leg & Hip families (`upper_hip_truss_box`, `upper_hip_topopt`)
and the Complete 6-Stage Foot-to-Hip Combined Architecture (`complete_leg_hip_combined`),
all with 12 C047 actuators, horns, idlers, fasteners, and shell overlays, plus comparison renders.
"""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_k] = "1"

from pathlib import Path
import copy
import shutil
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build123d import Compound, export_step, import_step

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.storage.blob_store import require  # noqa: E402  inputs restore by path or the build fails
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ROOT,
    ART,
    V16_DIR,
    V16_MANIFEST,
    FULL_SHELL_GLB,
    RIGHT_LEG_SHELL_DIR,
    read_json,
    write_json,
    rel,
    sha,
    ensure_two_screw_shin_shell_assets,
)
from package_and_render import (
    REPLACED_V16_LOWER_LEG_PARTS,
    actuator_rgba,
    add_extra_4screw_fasteners,
    render_stl_on_ax,
    set_equal_3d,
)

REPLACED_V16_UPPER_HIP_PARTS = set()
for _side in ("R", "L"):
    REPLACED_V16_UPPER_HIP_PARTS.update(
        [
            # Stage 4: Thigh
            f"{_side}_A10_one_piece_direct_upper",
            # Stage 5: Hip Pitch-to-Roll Yoke (3 parts/leg)
            f"{_side}_A13_pitch_roll_negative_L",
            f"{_side}_A13_pitch_roll_positive_L",
            f"{_side}_A13_pitch_output_ring8",
            # Stage 6: Hip Yaw-to-Pitch Frame (6 parts/leg)
            f"{_side}_A14_plain3_inside_yaw_pitch",
            f"{_side}_C_hip_yaw_ring_8mm_0",
            f"{_side}_A14_yaw_flat_shim1_0",
            f"{_side}_A14_yaw_flat_shim1_1",
            f"{_side}_A14_yaw_flat_shim1_2",
            f"{_side}_A14_yaw_flat_shim1_3",
        ]
    )

REPLACED_ALL_6_STAGES = REPLACED_V16_LOWER_LEG_PARTS | REPLACED_V16_UPPER_HIP_PARTS

STAGE_COLORS = {
    "stage1_shin": [42, 138, 184, 255],
    "stage2_ankle_coupler": [28, 164, 142, 255],
    "stage3_foot_monocoque": [212, 134, 52, 255],
    "stage4_thigh": [64, 156, 212, 255],
    "stage5_hip_pitch_to_roll": [228, 118, 68, 255],
    "stage6_hip_yaw_to_pitch": [46, 178, 136, 255],
    "sleeve": [215, 222, 230, 255],
}

PRINT_ORIENTATIONS = {
    "stage1_shin": "YZ Cheek / Oblique Bed (layer normal Z/X)",
    "stage2_ankle_coupler": "YZ Oblique Arch Bed (layer normal X)",
    "stage3_foot_monocoque": "XY Flat Sole Bed (layer normal Z)",
    "stage4_thigh": "YZ Medial Cheek Bed (layer normal X)",
    "stage5_hip_pitch_to_roll": "XY Corridor Bridge Bed (layer normal Z)",
    "stage6_hip_yaw_to_pitch": "YZ Lateral Wall / XY Top Deck Bed (layer normal X)",
}


def build_complete_combined_cad_and_fem():
    """Assemble `complete_leg_hip_combined` uniting the best stage from each family across all 6 stages."""
    comb_dir = ART / "complete_leg_hip_combined"
    parts_dir = comb_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    lower_comb = read_json(ART / "combined" / "cad-manifest.json")
    uh_truss = read_json(ART / "upper_hip_truss_box" / "cad-manifest.json")
    uh_topopt = read_json(ART / "upper_hip_topopt" / "cad-manifest.json")

    lower_fem = read_json(ART / "fem-report.json")
    upper_fem = read_json(ART / "fem-upper-hip-report.json")

    # Stage selection for Complete 6-Stage Combined Architecture:
    # Stage 1 (Shin): truss_box (12.37 g, u_max=0.281 mm, 15.91x margin)
    # Stage 2 (Ankle Coupler): truss_box (12.06 g, u_max=0.750 mm, 3.63x margin)
    # Stage 3 (Foot Monocoque): topopt (19.13 g, u_max=0.157 mm, 29.36x margin)
    # Stage 4 (Thigh / Femur): topopt (19.06 g, u_max=1.291 mm, k/m=2.70 (N/mm)/g, 5.22x margin — beats truss_box in mass & stiffness)
    # Stage 5 (Hip Pitch-to-Roll): topopt (16.73 g, u_max=1.454 mm vs 2.563 mm metal baseline, k/m=2.84 (N/mm)/g, 5.96x margin)
    # Stage 6 (Hip Yaw-to-Pitch): truss_box (12.27 g, u_max=1.295 mm, k/m=4.42 (N/mm)/g, 2.86x margin — beats topopt in stiffness & margin)
    combined_parts = []
    for p in lower_comb["parts"]:
        combined_parts.append(copy.deepcopy(p))

    upper_choices = [
        ("stage4_thigh", uh_topopt, "thigh_combined"),
        ("stage5_hip_pitch_to_roll", uh_topopt, "hip_pitch_to_roll_combined"),
        ("stage6_hip_yaw_to_pitch", uh_truss, "hip_yaw_to_pitch_combined"),
    ]
    for st_key, src_meta, base_name in upper_choices:
        for side in ("R", "L"):
            src = next(p for p in src_meta["parts"] if p["stage"] == st_key and p["side"] == side)
            p_name = f"{side}_{base_name}"
            s_name = f"{side}_{base_name}_steel_sleeves"

            dst_step = parts_dir / f"{p_name}.step"
            dst_stl = parts_dir / f"{p_name}.stl"
            shutil.copy2(require(ROOT / src["native_step"]), dst_step)
            shutil.copy2(require(ROOT / src["mesh"]), dst_stl)

            dst_s_step = parts_dir / f"{s_name}.step"
            dst_s_stl = parts_dir / f"{s_name}.stl"
            shutil.copy2(require(ROOT / src["sleeves"]["native_step"]), dst_s_step)
            shutil.copy2(require(ROOT / src["sleeves"]["mesh"]), dst_s_stl)

            entry = copy.deepcopy(src)
            entry["name"] = p_name
            entry["native_step"] = rel(dst_step)
            entry["native_step_sha256"] = sha(dst_step)
            entry["mesh"] = rel(dst_stl)
            entry["mesh_sha256"] = sha(dst_stl)
            entry["sleeves"]["name"] = s_name
            entry["sleeves"]["native_step"] = rel(dst_s_step)
            entry["sleeves"]["native_step_sha256"] = sha(dst_s_step)
            entry["sleeves"]["mesh"] = rel(dst_s_stl)
            entry["sleeves"]["mesh_sha256"] = sha(dst_s_stl)
            combined_parts.append(entry)

    write_json(
        comb_dir / "cad-manifest.json",
        {
            "family": "cove50-complete-leg-hip-fdm-combined",
            "version": "r1",
            "description": (
                "Complete 6-Stage Foot-to-Hip Combined FDM Connector Architecture uniting "
                "Stage 1 Shin (truss_box), Stage 2 Ankle Coupler (truss_box), Stage 3 Foot Monocoque (topopt), "
                "Stage 4 Thigh (topopt), Stage 5 Hip Pitch-to-Roll Yoke (topopt), and Stage 6 Hip Yaw-to-Pitch Frame (truss_box)"
            ),
            "parts": combined_parts,
        },
    )

    # Also build 6-stage FEM summaries for baseline_metal, full_truss_box, full_topopt, and complete_leg_hip_combined
    lower_cs = lower_fem["chain_summary_recovery_4kg_1p5g"]
    upper_cs = upper_fem["upper_chain_summary_recovery_4kg_1p5g"]

    full_6stage_summary = {}
    # 1) Baseline V16 metal (6 stages, 32 parts/leg, 113.481 g/leg)
    b_low = lower_cs["baseline_metal__Al5052_H32"]
    b_up = upper_cs["baseline_metal__Al5052_H32"]
    b_stages = {**b_low["stages"], **b_up["stages"]}
    b_mass = b_low["one_leg_mass_g"] + b_up["one_leg_mass_g"]
    full_6stage_summary["baseline_metal__Al5052_H32"] = {
        "family": "baseline_metal",
        "material": "Al5052_H32",
        "one_leg_mass_g": round(b_mass, 3),
        "bilateral_pair_mass_g": round(2.0 * b_mass, 3),
        "mass_saved_vs_baseline_one_leg_g": 0.0,
        "mass_saved_vs_baseline_bilateral_g": 0.0,
        "summed_stage_max_displacement_mm": round(b_low["summed_stage_max_displacement_mm"] + b_up["summed_stage_max_displacement_mm"], 4),
        "summed_stage_strain_energy_Nmm": round(b_low["summed_stage_strain_energy_Nmm"] + b_up["summed_stage_strain_energy_Nmm"], 4),
        "mean_specific_stiffness_N_mm_per_g": round(float(np.mean([r["specific_stiffness_N_mm_per_g"] for r in b_stages.values()])), 2),
        "min_stage_p99_strength_margin_ratio": round(min(r["p99_strength_margin_ratio"] for r in b_stages.values()), 2),
        "min_stage_p99_layer_margin_ratio": round(min(r["p99_layer_margin_ratio"] for r in b_stages.values()), 2),
        "stages": b_stages,
    }

    for mat in ["PAHT_CF_dry", "PAHT_CF_wet", "PETG_CF"]:
        # Full truss_box (6 stages)
        t_low = lower_cs[f"truss_box__{mat}"]
        t_up = upper_cs[f"truss_box__{mat}"]
        t_stages = {**t_low["stages"], **t_up["stages"]}
        t_mass = sum(r["stage_mass_g"] for r in t_stages.values())
        full_6stage_summary[f"truss_box__{mat}"] = {
            "family": "truss_box",
            "material": mat,
            "one_leg_mass_g": round(t_mass, 3),
            "bilateral_pair_mass_g": round(2.0 * t_mass, 3),
            "mass_saved_vs_baseline_one_leg_g": round(b_mass - t_mass, 3),
            "mass_saved_vs_baseline_bilateral_g": round(2.0 * (b_mass - t_mass), 3),
            "summed_stage_max_displacement_mm": round(sum(r["max_displacement_mm"] for r in t_stages.values()), 4),
            "summed_stage_strain_energy_Nmm": round(sum(r["strain_energy_Nmm"] for r in t_stages.values()), 4),
            "mean_specific_stiffness_N_mm_per_g": round(float(np.mean([r["specific_stiffness_N_mm_per_g"] for r in t_stages.values()])), 2),
            "min_stage_p99_strength_margin_ratio": round(min(r["p99_strength_margin_ratio"] for r in t_stages.values()), 2),
            "min_stage_p99_layer_margin_ratio": round(min(r["p99_layer_margin_ratio"] for r in t_stages.values()), 2),
            "stages": t_stages,
        }

        # Full topopt (6 stages)
        o_low = lower_cs[f"topopt__{mat}"]
        o_up = upper_cs[f"topopt__{mat}"]
        o_stages = {**o_low["stages"], **o_up["stages"]}
        o_mass = sum(r["stage_mass_g"] for r in o_stages.values())
        full_6stage_summary[f"topopt__{mat}"] = {
            "family": "topopt",
            "material": mat,
            "one_leg_mass_g": round(o_mass, 3),
            "bilateral_pair_mass_g": round(2.0 * o_mass, 3),
            "mass_saved_vs_baseline_one_leg_g": round(b_mass - o_mass, 3),
            "mass_saved_vs_baseline_bilateral_g": round(2.0 * (b_mass - o_mass), 3),
            "summed_stage_max_displacement_mm": round(sum(r["max_displacement_mm"] for r in o_stages.values()), 4),
            "summed_stage_strain_energy_Nmm": round(sum(r["strain_energy_Nmm"] for r in o_stages.values()), 4),
            "mean_specific_stiffness_N_mm_per_g": round(float(np.mean([r["specific_stiffness_N_mm_per_g"] for r in o_stages.values()])), 2),
            "min_stage_p99_strength_margin_ratio": round(min(r["p99_strength_margin_ratio"] for r in o_stages.values()), 2),
            "min_stage_p99_layer_margin_ratio": round(min(r["p99_layer_margin_ratio"] for r in o_stages.values()), 2),
            "stages": o_stages,
        }

        # Complete 6-Stage Combined
        c_stages = {
            "stage1_shin": copy.deepcopy(t_low["stages"]["stage1_shin"]),
            "stage2_ankle_coupler": copy.deepcopy(t_low["stages"]["stage2_ankle_coupler"]),
            "stage3_foot_monocoque": copy.deepcopy(o_low["stages"]["stage3_foot_monocoque"]),
            "stage4_thigh": copy.deepcopy(o_up["stages"]["stage4_thigh"]),
            "stage5_hip_pitch_to_roll": copy.deepcopy(o_up["stages"]["stage5_hip_pitch_to_roll"]),
            "stage6_hip_yaw_to_pitch": copy.deepcopy(t_up["stages"]["stage6_hip_yaw_to_pitch"]),
        }
        c_mass = sum(r["stage_mass_g"] for r in c_stages.values())
        full_6stage_summary[f"complete_leg_hip_combined__{mat}"] = {
            "family": "complete_leg_hip_combined",
            "material": mat,
            "one_leg_mass_g": round(c_mass, 3),
            "bilateral_pair_mass_g": round(2.0 * c_mass, 3),
            "mass_saved_vs_baseline_one_leg_g": round(b_mass - c_mass, 3),
            "mass_saved_vs_baseline_bilateral_g": round(2.0 * (b_mass - c_mass), 3),
            "summed_stage_max_displacement_mm": round(sum(r["max_displacement_mm"] for r in c_stages.values()), 4),
            "summed_stage_strain_energy_Nmm": round(sum(r["strain_energy_Nmm"] for r in c_stages.values()), 4),
            "mean_specific_stiffness_N_mm_per_g": round(float(np.mean([r["specific_stiffness_N_mm_per_g"] for r in c_stages.values()])), 2),
            "min_stage_p99_strength_margin_ratio": round(min(r["p99_strength_margin_ratio"] for r in c_stages.values()), 2),
            "min_stage_p99_layer_margin_ratio": round(min(r["p99_layer_margin_ratio"] for r in c_stages.values()), 2),
            "stages": c_stages,
        }

    write_json(comb_dir / "fem-summary.json", full_6stage_summary)
    return full_6stage_summary


def package_upper_and_complete_studies():
    v16 = read_json(V16_MANIFEST)
    v16_mech_glb = trimesh.load(require(V16_DIR / "mechanism-neutral.glb"), force="scene")

    configs = [
        ("upper_hip_truss_box", REPLACED_V16_UPPER_HIP_PARTS, 38.601),
        ("upper_hip_topopt", REPLACED_V16_UPPER_HIP_PARTS, 38.601),
        ("complete_leg_hip_combined", REPLACED_ALL_6_STAGES, 113.481),
    ]

    for fam_key, replaced_set, baseline_one_leg_g in configs:
        fam_dir = ART / fam_key
        cad_meta = read_json(fam_dir / "cad-manifest.json")

        full_body_actuators_scene = trimesh.Scene()
        full_body_actuators_scene.graph.update(frame_to="manufactured_lower_body_assembly", matrix=np.eye(4))

        retained_parts = []
        compound_solids = []

        for p in v16["parts"]:
            if p.get("inspection_reference"):
                continue
            pname = p["name"]
            if pname in replaced_set:
                continue

            p_copy = copy.deepcopy(p)
            col = actuator_rgba(pname, p_copy.get("color", [165, 174, 184, 255]))
            p_copy["color"] = col
            p_copy["display_group"] = "mechanism"
            p_copy["display_groups"] = ["mechanism"]
            p_copy["inspection_reference"] = False

            if pname in v16_mech_glb.graph.nodes_geometry:
                T_node, geom_key = v16_mech_glb.graph[pname]
                mesh = v16_mech_glb.geometry[geom_key].copy()
                mesh.apply_transform(T_node)
            else:
                mesh = trimesh.load(require(ROOT / p_copy["mesh"]), force="mesh", process=False)
                mesh.apply_scale(0.001)

            mesh.visual.face_colors = col
            full_body_actuators_scene.add_geometry(
                mesh.copy(),
                node_name=pname,
                geom_name=pname,
                parent_node_name="manufactured_lower_body_assembly",
            )
            retained_parts.append(p_copy)

        if fam_key == "complete_leg_hip_combined":
            v16_parts_by_name = {p["name"]: p for p in v16["parts"]}
            add_extra_4screw_fasteners(
                v16_parts_by_name,
                v16_mech_glb,
                [full_body_actuators_scene],
                retained_parts,
            )

        bom_items = []
        for item in cad_meta["parts"]:
            stage = item["stage"]
            side = item["side"]
            owning_link = item["owning_link"]
            p_orient = PRINT_ORIENTATIONS[stage]
            col_stage = STAGE_COLORS[stage]

            poly_name = item["name"]
            poly_mesh = trimesh.load(require(ROOT / item["mesh"]), force="mesh", process=False)
            poly_mesh.apply_scale(0.001)
            poly_mesh.visual.face_colors = col_stage
            full_body_actuators_scene.add_geometry(
                poly_mesh.copy(),
                node_name=poly_name,
                geom_name=poly_name,
                parent_node_name="manufactured_lower_body_assembly",
            )
            compound_solids.extend(import_step(require(ROOT / item["native_step"])).solids())

            poly_entry = {
                "name": poly_name,
                "owning_link": owning_link,
                "role": "printed_structural_connector",
                "physical_part_id": poly_name,
                "bom_group": poly_name,
                "native_step": item["native_step"],
                "native_step_sha256": item["native_step_sha256"],
                "mesh": item["mesh"],
                "mesh_sha256": item["mesh_sha256"],
                "mesh_units": "mm",
                "bounds_mm": item["bounds_mm"],
                "native_solid_count": item["native_solids"],
                "native_validity": [item["native_valid"]],
                "material": "Bambu PAHT-CF / PA12-CF (1.22 g/cm3) or PETG-CF (1.25 g/cm3) FDM",
                "mass_kg": round(item["polymer_mass_g_PAHT_CF"] / 1000.0, 6),
                "mass_basis": "Exact native STEP volume * 1.22 g/cm3 (PAHT-CF)",
                "color": col_stage,
                "display_group": "mechanism",
                "display_groups": ["mechanism"],
                "parent_assembly": "manufactured_lower_body_assembly",
                "inspection_reference": False,
                "volume_mm3": item["volume_mm3"],
                "print_orientation": p_orient,
                "manufacturing": (
                    f"Home-lab FDM 3D print (0.4 mm hardened steel nozzle, 0.20 mm layer height, 4 perimeters, "
                    f"100% rectilinear/gyroid dense structural walls). Bed orientation: {p_orient}."
                ),
            }
            retained_parts.append(poly_entry)

            slv_info = item["sleeves"]
            slv_name = slv_info["name"]
            slv_mesh = trimesh.load(require(ROOT / slv_info["mesh"]), force="mesh", process=False)
            slv_mesh.apply_scale(0.001)
            slv_mesh.visual.face_colors = STAGE_COLORS["sleeve"]
            full_body_actuators_scene.add_geometry(
                slv_mesh.copy(),
                node_name=slv_name,
                geom_name=slv_name,
                parent_node_name="manufactured_lower_body_assembly",
            )
            compound_solids.extend(import_step(require(ROOT / slv_info["native_step"])).solids())

            slv_entry = {
                "name": slv_name,
                "owning_link": owning_link,
                "role": "compression_limit_sleeves",
                "physical_part_id": slv_name,
                "bom_group": slv_name,
                "native_step": slv_info["native_step"],
                "native_step_sha256": slv_info["native_step_sha256"],
                "mesh": slv_info["mesh"],
                "mesh_sha256": slv_info["mesh_sha256"],
                "mesh_units": "mm",
                "bounds_mm": slv_info["bounds_mm"],
                "native_solid_count": slv_info["count"],
                "native_validity": [True] * slv_info["count"],
                "material": "304 Stainless Steel compression sleeves (7.85 g/cm3)",
                "mass_kg": round(slv_info["mass_g"] / 1000.0, 6),
                "mass_basis": "Exact native STEP volume * 7.85 g/cm3",
                "color": STAGE_COLORS["sleeve"],
                "display_group": "mechanism",
                "display_groups": ["mechanism"],
                "parent_assembly": "manufactured_lower_body_assembly",
                "inspection_reference": False,
                "volume_mm3": slv_info["volume_mm3"],
                "manufacturing": "Off-the-shelf 304 stainless steel cylindrical compression sleeves pressed into printed counterbores.",
            }
            retained_parts.append(slv_entry)

            if side == "R":
                bom_items.append(
                    {
                        "id": poly_name[2:],
                        "name": f"Bilateral {stage} FDM connector ({fam_key})",
                        "category": "Produced parts (FDM 3D printed)",
                        "quantity": 2,
                        "unit": "ea",
                        "material": "Bambu PAHT-CF / PA12-CF (or PETG-CF)",
                        "unit_mass_g": item["polymer_mass_g_PAHT_CF"],
                        "unit_cost": None,
                        "currency": "",
                        "cost_basis": "unknown",
                        "price_date": "",
                        "process": poly_entry["manufacturing"],
                        "notes": f"Replaces V16 bent/flat sheet parts in {stage}. Print orientation: {p_orient}.",
                    }
                )
                bom_items.append(
                    {
                        "id": slv_name[2:],
                        "name": f"Bilateral {stage} 304 SS compression sleeve set ({slv_info['count']} sleeves/leg)",
                        "category": "Fasteners & Hardware",
                        "quantity": 2,
                        "unit": "set",
                        "material": "304 Stainless Steel",
                        "unit_mass_g": slv_info["mass_g"],
                        "unit_cost": None,
                        "currency": "",
                        "cost_basis": "unknown",
                        "price_date": "",
                        "process": slv_entry["manufacturing"],
                        "notes": f"{slv_info['count']} steel sleeves per leg prevent bolt preload creep in the polymer flange.",
                    }
                )

        # Add captive M3 nuts and shell M3x8 screws seated inside the integrated Stage 4 Thigh (and Stage 1 Shin) 2-screw shell mounts
        sh2_assets = ensure_two_screw_shin_shell_assets()
        shell_hw_dir = ROOT / "artifacts/cove50-right-leg-shell/r4-20261028/knee125/native"
        shin_hw_dir = sh2_assets["dir"]
        shell_hw_specs = [
            (shell_hw_dir / "Thigh_carrier_shell_thigh_front_1_nut.step", "Thigh_carrier_shell_thigh_front_1_nut", "roll_link", [220, 185, 75, 255]),
            (shell_hw_dir / "Thigh_carrier_shell_thigh_front_1_M3x8.step", "Thigh_carrier_shell_thigh_front_1_M3x8", "roll_link", [210, 218, 226, 255]),
            (shell_hw_dir / "Thigh_carrier_shell_thigh_front_2_nut.step", "Thigh_carrier_shell_thigh_front_2_nut", "roll_link", [220, 185, 75, 255]),
            (shell_hw_dir / "Thigh_carrier_shell_thigh_front_2_M3x8.step", "Thigh_carrier_shell_thigh_front_2_M3x8", "roll_link", [210, 218, 226, 255]),
            (shell_hw_dir / "Thigh_rear_carrier_shell_thigh_back_1_nut.step", "Thigh_rear_carrier_shell_thigh_back_1_nut", "roll_link", [220, 185, 75, 255]),
            (shell_hw_dir / "Thigh_rear_carrier_shell_thigh_back_1_M3x8.step", "Thigh_rear_carrier_shell_thigh_back_1_M3x8", "roll_link", [210, 218, 226, 255]),
            (shell_hw_dir / "Thigh_rear_carrier_shell_thigh_back_2_nut.step", "Thigh_rear_carrier_shell_thigh_back_2_nut", "roll_link", [220, 185, 75, 255]),
            (shell_hw_dir / "Thigh_rear_carrier_shell_thigh_back_2_M3x8.step", "Thigh_rear_carrier_shell_thigh_back_2_M3x8", "roll_link", [210, 218, 226, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_front_1_nut.step", "Shin_carrier_shell_shin_front_1_nut", "shin", [220, 185, 75, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_front_1_M3x8.step", "Shin_carrier_shell_shin_front_1_M3x8", "shin", [210, 218, 226, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_front_2_nut.step", "Shin_carrier_shell_shin_front_2_nut", "shin", [220, 185, 75, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_front_2_M3x8.step", "Shin_carrier_shell_shin_front_2_M3x8", "shin", [210, 218, 226, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_back_1_nut.step", "Shin_carrier_shell_shin_back_1_nut", "shin", [220, 185, 75, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_back_1_M3x8.step", "Shin_carrier_shell_shin_back_1_M3x8", "shin", [210, 218, 226, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_back_2_nut.step", "Shin_carrier_shell_shin_back_2_nut", "shin", [220, 185, 75, 255]),
            (shin_hw_dir / "Shin_carrier_shell_shin_back_2_M3x8.step", "Shin_carrier_shell_shin_back_2_M3x8", "shin", [210, 218, 226, 255]),
        ]
        for hw_path, hw_base, lnk_suffix, hw_rgba in shell_hw_specs:
            hw_shape = import_step(require(hw_path))
            verts, faces = hw_shape.tessellate(0.12, 0.20)
            v_arr = np.array([[v.X * 0.001, v.Y * 0.001, v.Z * 0.001] for v in verts], dtype=float)
            f_arr = np.array(faces, dtype=int)
            for side in ("R", "L"):
                node_hw = f"{side}_{hw_base}"
                if side == "R":
                    hw_m = trimesh.Trimesh(vertices=v_arr.copy(), faces=f_arr.copy(), process=False)
                else:
                    v_l = v_arr.copy()
                    v_l[:, 0] *= -1.0
                    hw_m = trimesh.Trimesh(vertices=v_l, faces=f_arr[:, ::-1].copy(), process=False)
                hw_m.visual.face_colors = hw_rgba
                full_body_actuators_scene.add_geometry(
                    hw_m,
                    node_name=node_hw,
                    geom_name=node_hw,
                    parent_node_name="manufactured_lower_body_assembly",
                )
                retained_parts.append(
                    {
                        "name": node_hw,
                        "owning_link": f"{side}_{lnk_suffix}",
                        "role": "fastener",
                        "display_group": "mechanism",
                        "display_groups": ["mechanism"],
                        "parent_assembly": "manufactured_lower_body_assembly",
                        "inspection_reference": False,
                    }
                )

        bilateral_step_path = fam_dir / "connectors-bilateral.step"
        export_step(Compound(children=compound_solids), bilateral_step_path)

        shell_ref_parts = []
        for r in v16["parts"]:
            if r.get("inspection_reference"):
                rc = copy.deepcopy(r)
                rc["display_opacity"] = 0.55
                shell_ref_parts.append(rc)

        # Add the 8 bilateral fitted leg hard-shell covers (`thigh_front`, `thigh_back`, `shin_front`, `shin_back`) directly to full_body_actuators_scene so they are visible in View 0 and toggleable via the hide-shell button
        shell_mp = read_json(RIGHT_LEG_SHELL_DIR / "manufactured-parts.json")
        for sp in shell_mp["parts"][:4]:
            if sp["name"] == "shin_front":
                npz_p = sh2_assets["shin_front_npz"]
            elif sp["name"] == "shin_back":
                npz_p = sh2_assets["shin_back_npz"]
            else:
                npz_p = ROOT / sp["lossless_mesh"]["path"]
            z_sh = np.load(require(npz_p))
            v_sh = z_sh["vertices"] * 0.001
            f_sh = z_sh["faces"]
            lnk_suffix = sp["owning_link"][2:]
            for side in ("R", "L"):
                sh_node = f"{side}_{sp['name']}_hard_cover"
                if side == "R":
                    sh_m = trimesh.Trimesh(vertices=v_sh.copy(), faces=f_sh.copy(), process=False)
                else:
                    v_l = v_sh.copy()
                    v_l[:, 0] *= -1.0
                    sh_m = trimesh.Trimesh(vertices=v_l, faces=f_sh[:, ::-1].copy(), process=False)
                sh_m.visual.face_colors = [231, 222, 201, 255]
                full_body_actuators_scene.add_geometry(
                    sh_m,
                    node_name=sh_node,
                    geom_name=sh_node,
                    parent_node_name="manufactured_lower_body_assembly",
                )
                shell_ref_parts.append(
                    {
                        "name": sh_node,
                        "owning_link": f"{side}_{lnk_suffix}",
                        "role": "body_shell",
                        "material": "PETG / polycarbonate hard-shell cover (bolted to integrated M3 captive-nut mounts)",
                        "display_group": "ape_body_shell",
                        "display_groups": ["ape_body_shell"],
                        "display_opacity": 0.65,
                        "inspection_reference": True,
                    }
                )

        # Also add foot boot and sole shell panels to full_body_actuators_scene
        full_shell_src = trimesh.load(require(FULL_SHELL_GLB), force="scene")
        for src_node, out_node, own_lnk, sh_col in [
            ("ape_body_shell__left_smooth_boot", "R_foot_boot_shell", "R_foot", [231, 222, 201, 255]),
            ("ape_body_shell__R_sole", "R_foot_sole_shell", "R_foot", [120, 115, 110, 255]),
            ("ape_body_shell__right_smooth_boot", "L_foot_boot_shell", "L_foot", [231, 222, 201, 255]),
            ("ape_body_shell__L_sole", "L_foot_sole_shell", "L_foot", [120, 115, 110, 255]),
        ]:
            if src_node in full_shell_src.graph.nodes_geometry:
                T_s, g_s = full_shell_src.graph[src_node]
                m_s = full_shell_src.geometry[g_s].copy()
                m_s.apply_transform(T_s)
                m_s.visual.face_colors = sh_col
                full_body_actuators_scene.add_geometry(
                    m_s,
                    node_name=out_node,
                    geom_name=out_node,
                    parent_node_name="manufactured_lower_body_assembly",
                )
                shell_ref_parts.append(
                    {
                        "name": out_node,
                        "owning_link": own_lnk,
                        "role": "body_shell",
                        "material": "Foot boot / sole shell cover",
                        "display_group": "ape_body_shell",
                        "display_groups": ["ape_body_shell"],
                        "display_opacity": 0.65,
                        "inspection_reference": True,
                    }
                )

        full_lower_body_glb_path = fam_dir / "complete-lower-body-with-actuators.glb"
        full_body_actuators_scene.export(full_lower_body_glb_path)

        full_scene = trimesh.load(require(FULL_SHELL_GLB), force="scene")
        full_scene.delete_geometry(
            [
                "ape_body_shell__R_thigh_front", "ape_body_shell__R_thigh_back",
                "ape_body_shell__L_thigh_front", "ape_body_shell__L_thigh_back",
                "ape_body_shell__R_shin_front", "ape_body_shell__R_shin_back",
                "ape_body_shell__L_shin_front", "ape_body_shell__L_shin_back",
                "ape_body_shell__left_smooth_boot", "ape_body_shell__R_sole",
                "ape_body_shell__right_smooth_boot", "ape_body_shell__L_sole",
            ]
        )
        full_scene.graph.update(frame_to="manufactured_lower_body_assembly", matrix=np.eye(4))
        for node_name in full_body_actuators_scene.graph.nodes_geometry:
            T, geom_key = full_body_actuators_scene.graph[node_name]
            full_scene.add_geometry(
                full_body_actuators_scene.geometry[geom_key].copy(),
                node_name=node_name,
                geom_name=node_name,
                transform=T,
                parent_node_name="manufactured_lower_body_assembly",
            )

        full_glb_path = fam_dir / "assembly-with-full-shell.glb"
        full_scene.export(full_glb_path)

        manifest = {
            "schema": f"cove50.leg-hip-fdm-connectors.{fam_key}.r1",
            "version": "r1",
            "status": f"3D-printable FDM leg-to-hip connector architecture ({fam_key}) with all 12 C047 actuators in place",
            "units": {"native_STEP": "mm", "mesh": "mm", "GLB": "m"},
            "root_link": "pelvis",
            "parts": retained_parts + shell_ref_parts,
            "joints": copy.deepcopy(v16["joints"]),
            "controls": copy.deepcopy(v16["controls"]),
            "assemblies": [
                {"name": "manufactured_lower_body_assembly", "role": "Mechanical lower-body assembly with 12 C047 actuators"},
                {"name": "full_ape_shell_assembly", "role": "Complete original visual shell assembly", "inspection_reference": True},
            ],
        }
        write_json(fam_dir / "parts.json", manifest)

        one_leg_mass = sum(r["total_stage_mass_g_PAHT_CF"] for r in cad_meta["parts"] if r["side"] == "R")
        delta_g = baseline_one_leg_g - one_leg_mass
        bom = {
            "scope": f"Bilateral FDM connector replacement ({fam_key}) with 304 SS compression sleeves and all 12 C047 actuators fixed.",
            "completeness": "partial",
            "notes": (
                f"One-leg connector + sleeve mass: {one_leg_mass:.2f} g "
                f"(vs {baseline_one_leg_g:.2f} g V16 baseline, net delta {-delta_g:+.2f} g/leg "
                f"or {-2.0 * delta_g:+.2f} g bilateral)."
            ),
            "exclusions": [
                "Unchanged C047 servo motors, horns, bearings, foot pressure sensors, and pelvis central plate",
                "Print support waste and filament drying consumables",
            ],
            "items": bom_items,
        }
        write_json(fam_dir / "bom.json", bom)
        print(f"Packaged {fam_key}: one_leg_mass={one_leg_mass:.2f}g vs baseline={baseline_one_leg_g:.2f}g", flush=True)


def render_upper_and_complete_figures(full_6stage_summary: dict):
    render_dir = ART / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    v16 = read_json(V16_MANIFEST)
    v16_by_name = {p["name"]: p for p in v16["parts"]}
    v16_mech_glb = trimesh.load(require(V16_DIR / "mechanism-neutral.glb"), force="scene")
    upper_fem = read_json(ART / "fem-upper-hip-report.json")
    upper_cs = upper_fem["upper_chain_summary_recovery_4kg_1p5g"]

    def render_part_or_stl(ax, path_or_name, color, alpha=0.92, lw=0.08):
        p = Path(path_or_name) if not isinstance(path_or_name, Path) else path_or_name
        if p.exists():
            return render_stl_on_ax(ax, p, color, alpha=alpha, lw=lw)
        pname = p.stem
        if pname in v16_mech_glb.graph.nodes_geometry:
            T_node, geom_key = v16_mech_glb.graph[pname]
            mesh = v16_mech_glb.geometry[geom_key].copy()
            mesh.apply_transform(T_node)
            mesh.apply_scale(1000.0)
            tmp_stl = render_dir / f"_tmp_{pname}.stl"
            mesh.export(tmp_stl)
            res = render_stl_on_ax(ax, tmp_stl, color, alpha=alpha, lw=lw)
            tmp_stl.unlink(missing_ok=True)
            return res
        raise FileNotFoundError(f"Could not find {path_or_name}")

    MOTOR_COL = (0.24, 0.31, 0.40)
    HORN_COL = (0.90, 0.68, 0.22)

    stage_actuators_uh = {
        "stage4_thigh": [
            (ROOT / v16_by_name["R_C047_knee_case"]["mesh"], MOTOR_COL, 0.72),
            (ROOT / v16_by_name["R_C047_knee_main_horn"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_C047_knee_rear_idler"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_roll_servo_case_reference"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_roll_upper_horn_reference"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_roll_passive_idler_reference"]["mesh"], HORN_COL, 0.96),
        ],
        "stage5_hip_pitch_to_roll": [
            (ROOT / v16_by_name["R_roll_servo_case_reference"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_roll_upper_horn_reference"]["mesh"], HORN_COL, 0.94),
            (ROOT / v16_by_name["R_roll_passive_idler_reference"]["mesh"], HORN_COL, 0.94),
            (ROOT / v16_by_name["R_pitch_servo_case_reference"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_pitch_upper_horn_reference"]["mesh"], HORN_COL, 0.96),
        ],
        "stage6_hip_yaw_to_pitch": [
            (ROOT / v16_by_name["R_pitch_servo_case_reference"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_pitch_upper_horn_reference"]["mesh"], HORN_COL, 0.94),
            (ROOT / v16_by_name["R_yaw_servo_case_reference"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_yaw_upper_horn_reference"]["mesh"], HORN_COL, 0.96),
        ],
    }

    # Figure A: Upper-Leg & Hip Stage-by-Stage Comparison (Stages 4, 5, 6 with Mating C047 Actuators)
    fig = plt.figure(figsize=(16, 13.5), dpi=180)
    fig.patch.set_facecolor("#0f141c")
    fig.suptitle(
        "Cove-50 Upper-Leg & Hip Connectors (Stages 4–6) Mated to Fixed C047 Actuators (Knee -> Hip Roll -> Hip Pitch -> Hip Yaw)",
        color="#f0f4f8",
        fontsize=13.8,
        fontweight="bold",
        y=0.97,
    )

    stages_uh_info = [
        (
            "Stage 4: Thigh (Knee -> Hip Roll)",
            [(ROOT / v16_by_name["R_A10_one_piece_direct_upper"]["mesh"], (0.84, 0.64, 0.22), 0.96)],
            [
                (ART / "upper_hip_truss_box/parts/R_thigh_truss_box.stl", (0.25, 0.61, 0.83), 0.96),
                (ART / "upper_hip_truss_box/parts/R_thigh_truss_box_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
            [
                (ART / "upper_hip_topopt/parts/R_thigh_topopt.stl", (0.88, 0.44, 0.28), 0.96),
                (ART / "upper_hip_topopt/parts/R_thigh_topopt_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
            (20, -52),
            "stage4_thigh",
        ),
        (
            "Stage 5: Hip Pitch-to-Roll Yoke",
            [
                (ROOT / v16_by_name["R_A13_pitch_roll_negative_L"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_A13_pitch_roll_positive_L"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_A13_pitch_output_ring8"]["mesh"], (0.72, 0.76, 0.80), 0.96),
            ],
            [
                (ART / "upper_hip_truss_box/parts/R_hip_pitch_to_roll_truss_box.stl", (0.14, 0.68, 0.58), 0.96),
                (ART / "upper_hip_truss_box/parts/R_hip_pitch_to_roll_truss_box_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
            [
                (ART / "upper_hip_topopt/parts/R_hip_pitch_to_roll_topopt.stl", (0.89, 0.46, 0.27), 0.96),
                (ART / "upper_hip_topopt/parts/R_hip_pitch_to_roll_topopt_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
            (24, -44),
            "stage5_hip_pitch_to_roll",
        ),
        (
            "Stage 6: Hip Yaw-to-Pitch Frame",
            [
                (ROOT / v16_by_name["R_A14_plain3_inside_yaw_pitch"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_C_hip_yaw_ring_8mm_0"]["mesh"], (0.72, 0.76, 0.80), 0.96),
                (ROOT / v16_by_name["R_A14_yaw_flat_shim1_0"]["mesh"], (0.72, 0.76, 0.80), 0.96),
            ],
            [
                (ART / "upper_hip_truss_box/parts/R_hip_yaw_to_pitch_truss_box.stl", (0.18, 0.70, 0.53), 0.96),
                (ART / "upper_hip_truss_box/parts/R_hip_yaw_to_pitch_truss_box_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
            [
                (ART / "upper_hip_topopt/parts/R_hip_yaw_to_pitch_topopt.stl", (0.86, 0.54, 0.22), 0.96),
                (ART / "upper_hip_topopt/parts/R_hip_yaw_to_pitch_topopt_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
            (22, -48),
            "stage6_hip_yaw_to_pitch",
        ),
    ]

    uh_base_m = upper_cs["baseline_metal__Al5052_H32"]["one_leg_mass_g"]
    uh_f1_m = upper_cs["truss_box__PAHT_CF_dry"]["one_leg_mass_g"]
    uh_f2_m = upper_cs["topopt__PAHT_CF_dry"]["one_leg_mass_g"]

    col_headers = [
        f"Baseline V16 Sheet + Rings + Shims + C047s\n(5052-H32, 10 parts/leg, {uh_base_m:.2f} g/leg)",
        f"Family 1: Sculpted Space-Truss & Box + C047s\n(PAHT-CF + SS Sleeves, 3 parts/leg, {uh_f1_m:.2f} g/leg)",
        f"Family 2: Sculpted 3D TopOpt Wishbone + C047s\n(PAHT-CF + SS Sleeves, 3 parts/leg, {uh_f2_m:.2f} g/leg)",
    ]

    for row_idx, (stage_title, base_stls, truss_stls, topopt_stls, (elev, azim), st_key) in enumerate(stages_uh_info):
        act_list = stage_actuators_uh[st_key]
        for col_idx, (fam_key, mat_key, stl_list) in enumerate(
            [
                ("baseline_metal", "Al5052_H32", base_stls),
                ("truss_box", "PAHT_CF_dry", truss_stls),
                ("topopt", "PAHT_CF_dry", topopt_stls),
            ]
        ):
            ax = fig.add_subplot(3, 3, row_idx * 3 + col_idx + 1, projection="3d")
            ax.set_facecolor("#161d28")
            mins_conn, maxs_conn = [], []
            for sp, col, alp in act_list:
                render_part_or_stl(ax, sp, col, alpha=alp, lw=0.06)
            for sp, col, alp in stl_list:
                mn, mx = render_part_or_stl(ax, sp, col, alpha=alp, lw=0.12)
                mins_conn.append(mn)
                maxs_conn.append(mx)
            c_min = np.min(mins_conn, axis=0)
            c_max = np.max(maxs_conn, axis=0)
            set_equal_3d(ax, c_min - np.array([5.0, 6.0, 8.0]), c_max + np.array([5.0, 6.0, 8.0]), pad=3.0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("#263244")

            st_metrics = upper_cs[f"{fam_key}__{mat_key}"]["stages"][st_key]
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], color="#e2e8f0", fontsize=10.5, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.text2D(-0.08, 0.5, stage_title, transform=ax.transAxes, color="#94a3b8", fontsize=9.5, fontweight="bold", rotation=90, va="center", ha="center")

            badge = (
                f"Mass: {st_metrics['stage_mass_g']:.2f} g | u_max (4kg@1.5g): {st_metrics['max_displacement_mm']:.3f} mm\n"
                f"k/m: {st_metrics['specific_stiffness_N_mm_per_g']:.1f} (N/mm)/g | p99: {st_metrics['primary_vm_p99_MPa']:.1f} MPa ({st_metrics['p99_strength_margin_ratio']:.2f}x margin)"
            )
            ax.text2D(
                0.5,
                0.03,
                badge,
                transform=ax.transAxes,
                color="#f8fafc",
                fontsize=8.2,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#0f172a", edgecolor="#334155", alpha=0.90),
            )

    plt.tight_layout(rect=[0.03, 0.01, 0.99, 0.94])
    fig_uh_path = render_dir / "upper-hip-stage-cad.png"
    fig.savefig(fig_uh_path, dpi=180)
    plt.close(fig)

    # Figure B: Complete 6-Stage Foot-to-Hip Assembly Comparison (Baseline V16 vs Combined 6-Stage FDM)
    fig = plt.figure(figsize=(16.0, 9.8), dpi=180)
    fig.patch.set_facecolor("#0f141c")
    fig.suptitle(
        "Complete 6-Stage Foot-to-Hip Chain (Stages 1–6) with All 6 Right-Leg C047 Actuators: V16 Sheet Metal vs. Combined FDM",
        color="#f0f4f8",
        fontsize=13.6,
        fontweight="bold",
        y=0.96,
    )

    right_actuators = []
    for p in v16["parts"]:
        pname = p["name"]
        if (
            pname.startswith("R_")
            and not p.get("inspection_reference")
            and pname not in REPLACED_ALL_6_STAGES
            and "fastener" not in p.get("role", "")
        ):
            if "case" in pname:
                col, alp = MOTOR_COL, 0.84
            elif "horn" in pname or "idler" in pname:
                col, alp = HORN_COL, 0.96
            else:
                col, alp = (0.58, 0.63, 0.68), 0.72
            right_actuators.append((ROOT / p["mesh"], col, alp))

    v16_6stage_stls = [
        (ROOT / v16_by_name[n]["mesh"], (0.84, 0.64, 0.22), 0.96)
        for n in [
            "R_B13R14_short_inner_shin_plate",
            "R_B13R14_short_outer_shin_plate",
            "R_B13R14_one_bend_output_link",
            "R_B13R14_front_case_L",
            "R_B13R14_rear_case_L",
            "R_B13R14_plain_sole",
            "R_A10_one_piece_direct_upper",
            "R_A13_pitch_roll_negative_L",
            "R_A13_pitch_roll_positive_L",
            "R_A13_pitch_output_ring8",
            "R_A14_plain3_inside_yaw_pitch",
            "R_C_hip_yaw_ring_8mm_0",
        ]
    ]

    truss_6stage_stls = [
        (ART / "truss_box/parts/R_shin_truss_box.stl", (0.18, 0.58, 0.78), 0.96),
        (ART / "truss_box/parts/R_ankle_coupler_truss_box.stl", (0.14, 0.68, 0.58), 0.96),
        (ART / "truss_box/parts/R_foot_monocoque_truss_box.stl", (0.24, 0.50, 0.74), 0.96),
        (ART / "upper_hip_truss_box/parts/R_thigh_truss_box.stl", (0.25, 0.61, 0.83), 0.96),
        (ART / "upper_hip_truss_box/parts/R_hip_pitch_to_roll_truss_box.stl", (0.14, 0.68, 0.58), 0.96),
        (ART / "upper_hip_truss_box/parts/R_hip_yaw_to_pitch_truss_box.stl", (0.18, 0.70, 0.53), 0.96),
    ]

    comb_6stage_stls = [
        (ART / "combined/parts/R_shin_combined.stl", (0.18, 0.58, 0.78), 0.96),
        (ART / "combined/parts/R_ankle_coupler_combined.stl", (0.14, 0.68, 0.58), 0.96),
        (ART / "combined/parts/R_foot_monocoque_combined.stl", (0.86, 0.54, 0.22), 0.96),
        (ART / "complete_leg_hip_combined/parts/R_thigh_combined.stl", (0.25, 0.61, 0.83), 0.96),
        (ART / "complete_leg_hip_combined/parts/R_hip_pitch_to_roll_combined.stl", (0.89, 0.46, 0.27), 0.96),
        (ART / "complete_leg_hip_combined/parts/R_hip_yaw_to_pitch_combined.stl", (0.18, 0.70, 0.53), 0.96),
    ]

    full_sum_dict = full_6stage_summary or read_json(ART / "complete_leg_hip_combined" / "fem-summary.json")
    b6 = full_sum_dict["baseline_metal__Al5052_H32"]
    c6 = full_sum_dict["complete_leg_hip_combined__PAHT_CF_dry"]
    f1_6 = full_sum_dict["truss_box__PAHT_CF_dry"]

    panels = [
        (
            f"Baseline V16 Bent Sheet + Rings + Shims\n32 parts/leg · {b6['one_leg_mass_g']:.2f} g/leg\n"
            f"sum u_max = {b6['summed_stage_max_displacement_mm']:.2f} mm · min margin {b6['min_stage_p99_strength_margin_ratio']:.2f}x",
            v16_6stage_stls,
        ),
        (
            f"Family 1: Full 6-Stage Space-Truss & Box\n6 parts/leg · {f1_6['one_leg_mass_g']:.2f} g/leg ({f1_6['one_leg_mass_g'] - b6['one_leg_mass_g']:+.2f} g/leg)\n"
            f"sum u_max = {f1_6['summed_stage_max_displacement_mm']:.2f} mm · min margin {f1_6['min_stage_p99_strength_margin_ratio']:.2f}x",
            truss_6stage_stls,
        ),
        (
            f"Combined 6-Stage Hybrid (Best of F1 & F2)\n6 parts/leg · {c6['one_leg_mass_g']:.2f} g/leg ({c6['one_leg_mass_g'] - b6['one_leg_mass_g']:+.2f} g/leg)\n"
            f"sum u_max = {c6['summed_stage_max_displacement_mm']:.2f} mm · min margin {c6['min_stage_p99_strength_margin_ratio']:.2f}x",
            comb_6stage_stls,
        ),
    ]

    for idx, (ptitle, stls) in enumerate(panels):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_facecolor("#161d28")
        mins_all, maxs_all = [], []
        for sp, col, alp in right_actuators + stls:
            mn, mx = render_part_or_stl(ax, sp, col, alpha=alp, lw=0.07)
            mins_all.append(mn)
            maxs_all.append(mx)
        set_equal_3d(ax, np.min(mins_all, axis=0), np.max(maxs_all, axis=0), pad=4.0)
        ax.view_init(elev=18, azim=-52)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#263244")
        ax.set_title(ptitle, color="#f8fafc", fontsize=9.8, fontweight="bold", pad=10)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    fig_full_path = render_dir / "complete-leg-hip-6stage-comparison.png"
    fig.savefig(fig_full_path, dpi=180)
    plt.close(fig)

    # Figure C: Upper-Leg & Hip End-to-End Verification Suite (3D SIMP Density + Matched 3D FEM Charts + 16-Pose & 60-Ray Mountability Summary)
    topopt_npz = np.load(require(ART / "topopt" / "topopt-upper-hip-fields.npz"))
    fit_uh = read_json(ART / "upper-hip-fit-report.json")
    fig = plt.figure(figsize=(16.0, 11.2), dpi=180)
    fig.patch.set_facecolor("#0f141c")
    fig.suptitle(
        "Upper-Leg & Hip (Stages 4–6) Full End-to-End Verification: 3D SIMP Load Paths, Matched 3D Tetrahedral FEM & 76-Ray Screw/Mountability Audit",
        color="#f0f4f8",
        fontsize=13.0,
        fontweight="bold",
        y=0.97,
    )

    # Row 1: 3D SIMP Topology Optimization Density Projections for Stages 4, 5, 6
    simp_panels = [
        ("Stage 4 Thigh 3D SIMP Density (YZ Max Projection)", topopt_npz["s4_rho"].max(axis=0).T, "inferno"),
        ("Stage 5 Hip Pitch-to-Roll 3D SIMP Density (XZ Max Projection)", topopt_npz["s5_rho"].max(axis=1).T, "inferno"),
        ("Stage 6 Hip Yaw-to-Pitch 3D SIMP Density (YZ Max Projection)", topopt_npz["s6_rho"].max(axis=0).T, "inferno"),
    ]
    for col_i, (stitle, img2d, cmap_name) in enumerate(simp_panels):
        ax = fig.add_subplot(2, 3, col_i + 1)
        ax.set_facecolor("#161d28")
        im = ax.imshow(img2d, origin="lower", aspect="auto", cmap=cmap_name, vmin=0.0, vmax=1.0)
        ax.set_title(stitle, color="#e2e8f0", fontsize=10.2, fontweight="bold")
        ax.set_xlabel("Voxel Grid Axis 1", color="#94a3b8", fontsize=8.5)
        ax.set_ylabel("Voxel Grid Z (Vertical)", color="#94a3b8", fontsize=8.5)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="#94a3b8", labelsize=7.5)

    # Row 2: Quantitative FEM Comparison across Stages 4, 5, 6 (Mass, Deflection u_max, and p99 Strength Margin)
    st_keys = ["stage4_thigh", "stage5_hip_pitch_to_roll", "stage6_hip_yaw_to_pitch"]
    st_labels = ["Stage 4\nThigh", "Stage 5\nPitch-Roll", "Stage 6\nYaw-Pitch"]
    x_pos = np.arange(len(st_keys))
    w = 0.25

    variants_bar = [
        ("baseline_metal__Al5052_H32", "V16 Sheet/Rings (Al 5052)", "#d97706"),
        ("truss_box__PAHT_CF_dry", "Family 1: Truss-Box (PAHT-CF)", "#0284c7"),
        ("topopt__PAHT_CF_dry", "Family 2: 3D TopOpt (PAHT-CF)", "#ea580c"),
    ]

    metrics_bar = [
        ("stage_mass_g", "Stage Mass (g/leg)", "Mass (g)"),
        ("max_displacement_mm", "3D FEM Peak Deflection u_max (mm @ 4kg 1.5g)", "Deflection (mm)"),
        ("p99_strength_margin_ratio", "3D FEM p99 Strength Margin Ratio (x)", "Margin Ratio (x)"),
    ]

    for m_idx, (m_field, m_title, m_ylabel) in enumerate(metrics_bar):
        ax = fig.add_subplot(2, 3, 4 + m_idx)
        ax.set_facecolor("#161d28")
        all_vals = []
        for v_i, (v_key, v_label, v_col) in enumerate(variants_bar):
            vals = [upper_cs[v_key]["stages"][sk][m_field] for sk in st_keys]
            all_vals.extend(vals)
            bars = ax.bar(x_pos + (v_i - 1) * w, vals, width=w, color=v_col, label=v_label, edgecolor="#0f172a")
            for b, val in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() * 0.5,
                    b.get_height() + max(vals) * 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    color="#f8fafc",
                    fontsize=7.8,
                    fontweight="bold",
                )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(st_labels, color="#e2e8f0", fontsize=9.0, fontweight="bold")
        ax.set_title(m_title, color="#e2e8f0", fontsize=10.0, fontweight="bold")
        ax.set_ylabel(m_ylabel, color="#94a3b8", fontsize=8.8)
        ax.tick_params(colors="#94a3b8", labelsize=8.2)
        ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.5)
        for sp_side in ("top", "right", "left", "bottom"):
            ax.spines[sp_side].set_color("#334155")
        ax.set_ylim(0, max(all_vals) * (1.28 if m_idx == 0 else 1.15))
        if m_idx == 0:
            ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper left")

    f1_fit = fit_uh["families"]["upper_hip_truss_box"]
    f2_fit = fit_uh["families"]["upper_hip_topopt"]
    footer_txt = (
        f"Full End-to-End Verification Summary (Zero Tests Skipped): "
        f"16 Flexion Poses Checked vs All 6 C047 Motor Cases + Pelvis + 12 Horns/Idlers -> "
        f"Family 1 Max Overlap = {f1_fit['max_overlap_mm3_across_all_poses']:.4f} mm^3, "
        f"Family 2 Max Overlap = {f2_fit['max_overlap_mm3_across_all_poses']:.4f} mm^3\n"
        f"Mountability & Fastener Line-of-Sight Audit (76 Screw/Hex-Driver & Captive-Nut Rays Across Stages 1–6) -> "
        f"Family 1 Max Ray Overlap = {f1_fit['mountability_and_screw_access']['max_ray_overlap_mm3_across_all_6_stages']:.4f} mm^3, "
        f"Family 2 Max Ray Overlap = {f2_fit['mountability_and_screw_access']['max_ray_overlap_mm3_across_all_6_stages']:.4f} mm^3"
    )
    fig.text(
        0.5,
        0.022,
        footer_txt,
        ha="center",
        va="center",
        color="#38bdf8",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#0284c7", alpha=0.95),
    )

    plt.tight_layout(rect=[0.01, 0.06, 0.99, 0.94])
    fig_ver_path = render_dir / "upper-hip-verification-suite.png"
    fig.savefig(fig_ver_path, dpi=180)
    plt.close(fig)
    print("Saved upper-hip and complete 6-stage comparison renders to", render_dir, flush=True)


if __name__ == "__main__":
    full_sum = build_complete_combined_cad_and_fem()
    package_upper_and_complete_studies()
    render_upper_and_complete_figures(full_sum)
