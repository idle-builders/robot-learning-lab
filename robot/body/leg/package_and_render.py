"""Package articulated GLBs (with all C047 actuators), manifests, BOMs, and generate visual comparison figures with actuators."""
import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_k] = "1"

from pathlib import Path
import copy
import json
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from build123d import Compound, export_step, import_step

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.storage.blob_store import ensure, require  # noqa: E402  inputs restore by path or the build fails
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("build-combined")   # configured values: robot/body/leg/build_combined.bom.yaml config:

# Parts in V16 lower leg that are replaced by the 3 new FDM connector stages per leg
REPLACED_V16_LOWER_LEG_PARTS = set()
for _side in ("R", "L"):
    REPLACED_V16_LOWER_LEG_PARTS.update(
        [
            # Stage 1: Shin plates + 4 cross spacers
            f"{_side}_B13R14_short_inner_shin_plate",
            f"{_side}_B13R14_short_outer_shin_plate",
            f"{_side}_B13R14_raised_B11_inner_cut_spacer_0",
            f"{_side}_B13R14_raised_B11_inner_cut_spacer_1",
            f"{_side}_B13R14_raised_B11_outer_cut_spacer_0",
            f"{_side}_B13R14_raised_B11_outer_cut_spacer_1",
            # Stage 2: Ankle Pitch-to-Roll one-bend output link
            f"{_side}_B13R14_one_bend_output_link",
            # Stage 3: Foot front/rear L-brackets + plain sole + 4 corner spacers + 4 corner bolts + 4 corner nuts
            f"{_side}_B13R14_front_case_L",
            f"{_side}_B13R14_rear_case_L",
            f"{_side}_B13R14_plain_sole",
            f"{_side}_B13_front_cut_spacer_0",
            f"{_side}_B13_front_cut_spacer_1",
            f"{_side}_B13_rear_cut_spacer_0",
            f"{_side}_B13_rear_cut_spacer_1",
            f"{_side}_B13R14_front_sole_M3x6_button_0",
            f"{_side}_B13R14_front_sole_M3x6_button_1",
            f"{_side}_B13R14_rear_sole_M3x6_button_0",
            f"{_side}_B13R14_rear_sole_M3x6_button_1",
            f"{_side}_B13R14_front_sole_M3_nut_0",
            f"{_side}_B13R14_front_sole_M3_nut_1",
            f"{_side}_B13R14_rear_sole_M3_nut_0",
            f"{_side}_B13R14_rear_sole_M3_nut_1",
        ]
    )

LOWER_LEG_OWNING_LINKS = set(CFG["LOWER_LEG_OWNING_LINKS"])

EXTRA_4SCREW_SPECS = []
for _side in ("R", "L"):
    EXTRA_4SCREW_SPECS.extend(
        [
            # Stage 1 Knee Main Horn: add 3rd & 4th M3x4 screws at (Y=-3, Z=123) and (Y=11, Z=123) from (Y=4, Z=116)
            (f"{_side}_B_NBK_M3x4_H0p8_front_0_t2_flat", f"{_side}_B_NBK_M3x4_H0p8_front_2_t2_flat", (0.0, -7.0, 7.0)),
            (f"{_side}_B_NBK_M3x4_H0p8_front_0_t2_flat", f"{_side}_B_NBK_M3x4_H0p8_front_3_t2_flat", (0.0, 7.0, 7.0)),
            # Stage 1 Knee Rear Idler: add 3rd & 4th M3x4 screws at (Y=-3, Z=123) and (Y=11, Z=123) from (Y=4, Z=116)
            (f"{_side}_B_NBK_M3x4_H0p8_rear_0_t2_flat", f"{_side}_B_NBK_M3x4_H0p8_rear_2_t2_flat", (0.0, -7.0, 7.0)),
            (f"{_side}_B_NBK_M3x4_H0p8_rear_0_t2_flat", f"{_side}_B_NBK_M3x4_H0p8_rear_3_t2_flat", (0.0, 7.0, 7.0)),
            # Stage 1 Inner Ankle Pitch Cheek: add 3rd & 4th PA2x6 screws at (-6.25, 93.75) and (14.25, 69.30) from (-6.25, 69.30)
            (f"{_side}_B13R14_raised_B11_inner_PA2x6_proposal_0", f"{_side}_B13R14_raised_B11_inner_PA2x6_proposal_2", (0.0, 0.0, 24.45)),
            (f"{_side}_B13R14_raised_B11_inner_PA2x6_proposal_0", f"{_side}_B13R14_raised_B11_inner_PA2x6_proposal_3", (0.0, 20.5, 0.0)),
            # Stage 1 Outer Ankle Pitch Cheek: add 3rd & 4th PA2x6 screws at (-6.25, 90.00) and (14.25, 69.30) from (-6.25, 69.30)
            (f"{_side}_B13R14_raised_B11_outer_PA2x6_proposal_0", f"{_side}_B13R14_raised_B11_outer_PA2x6_proposal_2", (0.0, 0.0, 20.70)),
            (f"{_side}_B13R14_raised_B11_outer_PA2x6_proposal_0", f"{_side}_B13R14_raised_B11_outer_PA2x6_proposal_3", (0.0, 20.5, 0.0)),
            # Stage 2 Ankle Pitch Rear Idler: add 3rd & 4th M3x4 screws at (Y=-3, Z=61) and (Y=11, Z=61) from (Y=4, Z=54)
            (f"{_side}_B16_idler_SECS_M3x4_0", f"{_side}_B16_idler_SECS_M3x4_2", (0.0, -7.0, 7.0)),
            (f"{_side}_B16_idler_SECS_M3x4_0", f"{_side}_B16_idler_SECS_M3x4_3", (0.0, 7.0, 7.0)),
        ]
    )


def actuator_rgba(part_name: str, default_rgba: list[int]) -> list[int]:
    """Assign high-contrast, unmistakable colors to C047 motor cases, horns, idlers, and fasteners."""
    if "case" in part_name and ("C047" in part_name or "servo" in part_name):
        return [52, 68, 88, 255]  # Deep anodized slate-blue motor housing
    if "horn" in part_name or "idler" in part_name:
        return [230, 168, 52, 255]  # Gold/brass output horn & rear idler
    if "M3" in part_name or "PA2" in part_name or "nut" in part_name or "SECS" in part_name or "NBK" in part_name:
        return [195, 204, 214, 255]  # Stainless fastener silver
    return default_rgba


def add_extra_4screw_fasteners(
    v16_parts_by_name: dict,
    v16_mech_glb: trimesh.Scene,
    scenes: list[trimesh.Scene],
    retained_parts: list[dict],
):
    """Clone and translate the 20 newly populated 3D screw solids (10 per leg) so every 4-screw motor/horn/idler interface has visible 3D screws."""
    for src_name, new_name, (dx_mm, dy_mm, dz_mm) in EXTRA_4SCREW_SPECS:
        if src_name not in v16_parts_by_name:
            continue
        p_copy = copy.deepcopy(v16_parts_by_name[src_name])
        p_copy["name"] = new_name
        p_copy["physical_part_id"] = new_name
        col = actuator_rgba(new_name, p_copy.get("color", [195, 204, 214, 255]))
        p_copy["color"] = col
        p_copy["display_group"] = "mechanism"
        p_copy["display_groups"] = ["mechanism"]
        p_copy["inspection_reference"] = False
        if "bounds_mm" in p_copy and len(p_copy["bounds_mm"]) == 2:
            b0, b1 = p_copy["bounds_mm"]
            p_copy["bounds_mm"] = [
                [round(b0[0] + dx_mm, 4), round(b0[1] + dy_mm, 4), round(b0[2] + dz_mm, 4)],
                [round(b1[0] + dx_mm, 4), round(b1[1] + dy_mm, 4), round(b1[2] + dz_mm, 4)],
            ]
        if src_name in v16_mech_glb.graph.nodes_geometry:
            T_node, geom_key = v16_mech_glb.graph[src_name]
            mesh = v16_mech_glb.geometry[geom_key].copy()
            mesh.apply_transform(T_node)
        else:
            mesh = trimesh.load(ROOT / p_copy["mesh"], force="mesh", process=False)
            mesh.apply_scale(0.001)
        mesh.apply_translation([dx_mm * 0.001, dy_mm * 0.001, dz_mm * 0.001])
        mesh.visual.face_colors = col
        for sc in scenes:
            sc.add_geometry(
                mesh.copy(),
                node_name=new_name,
                geom_name=new_name,
                parent_node_name="manufactured_lower_body_assembly",
            )
        retained_parts.append(p_copy)


def build_family_packages():
    """Build articulated GLBs (all including the C047 actuators), parts.json, and bom.json for both families."""
    v16 = read_json(V16_MANIFEST)
    v16_mech_glb = trimesh.load(V16_DIR / "mechanism-neutral.glb", force="scene")

    family_colors = {
        "truss_box": {
            "stage1_shin": [42, 138, 184, 255],
            "stage2_ankle_coupler": [28, 164, 142, 255],
            "stage3_foot_monocoque": [58, 118, 172, 255],
            "sleeve": [215, 222, 230, 255],
        },
        "topopt": {
            "stage1_shin": [224, 108, 68, 255],
            "stage2_ankle_coupler": [46, 168, 132, 255],
            "stage3_foot_monocoque": [212, 134, 52, 255],
            "sleeve": [215, 222, 230, 255],
        },
        "combined": {
            "stage1_shin": [42, 138, 184, 255],
            "stage2_ankle_coupler": [28, 164, 142, 255],
            "stage3_foot_monocoque": [212, 134, 52, 255],
            "sleeve": [215, 222, 230, 255],
        },
    }

    print_orientations = {
        "stage1_shin": "YZ Cheek / Oblique Bed (layer normal Z/X)",
        "stage2_ankle_coupler": "YZ Oblique Arch Bed (layer normal X)",
        "stage3_foot_monocoque": "XY Flat Sole Bed (layer normal Z)",
    }

    for family in ("truss_box", "topopt", "combined"):
        fam_dir = ART / family
        cad_meta = read_json(require(fam_dir / "cad-manifest.json"))
        colors = family_colors[family]

        # Remove any old actuator-less GLB if present
        old_conn_only = fam_dir / "lower-leg-connectors-only.glb"
        if old_conn_only.exists():
            old_conn_only.unlink()

        # Scene 1: Lower-leg mechanism WITH all 6 lower-leg C047 actuators, horns, idlers, fasteners &thigh links
        lower_leg_actuators_scene = trimesh.Scene()
        lower_leg_actuators_scene.graph.update(frame_to="manufactured_lower_body_assembly", matrix=np.eye(4))

        # Scene 2: Complete 12-DOF lower-body mechanism (Pelvis + Hip Yaw/Roll/Pitch C047s + Thighs + Lower-Leg C047s + New FDM Connectors)
        full_body_actuators_scene = trimesh.Scene()
        full_body_actuators_scene.graph.update(frame_to="manufactured_lower_body_assembly", matrix=np.eye(4))

        retained_parts = []
        compound_solids = []

        # Add all unchanged V16 parts (using V16 mechanism-neutral.glb geometry so all 12 C047 actuators across pelvis/hip/knee/ankle/foot are present)
        for p in v16["parts"]:
            if p.get("inspection_reference"):
                continue
            pname = p["name"]
            if pname in REPLACED_V16_LOWER_LEG_PARTS:
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
                mesh = trimesh.load(ROOT / p_copy["mesh"], force="mesh", process=False)
                mesh.apply_scale(0.001)

            mesh.visual.face_colors = col
            full_body_actuators_scene.add_geometry(
                mesh.copy(),
                node_name=pname,
                geom_name=pname,
                parent_node_name="manufactured_lower_body_assembly",
            )
            if p_copy.get("owning_link") in LOWER_LEG_OWNING_LINKS:
                lower_leg_actuators_scene.add_geometry(
                    mesh.copy(),
                    node_name=pname,
                    geom_name=pname,
                    parent_node_name="manufactured_lower_body_assembly",
                )
            retained_parts.append(p_copy)

        v16_parts_by_name = {p["name"]: p for p in v16["parts"]}
        add_extra_4screw_fasteners(
            v16_parts_by_name,
            v16_mech_glb,
            [lower_leg_actuators_scene, full_body_actuators_scene],
            retained_parts,
        )

        # Add the 6 new bilateral printed parts + 6 sleeve sets
        bom_items = []
        for item in cad_meta["parts"]:
            stage = item["stage"]
            side = item["side"]
            owning_link = item["owning_link"]
            p_orient = print_orientations[stage]

            # Polymer part entry
            poly_name = item["name"]
            poly_mesh = trimesh.load(ROOT / item["mesh"], force="mesh", process=False)
            poly_mesh.apply_scale(0.001)
            poly_mesh.visual.face_colors = colors[stage]
            for sc in (lower_leg_actuators_scene, full_body_actuators_scene):
                sc.add_geometry(
                    poly_mesh.copy(),
                    node_name=poly_name,
                    geom_name=poly_name,
                    parent_node_name="manufactured_lower_body_assembly",
                )
            compound_solids.extend(import_step(ROOT / item["native_step"]).solids())

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
                "material": f"{BOM_CONFIG.material(item['name'], 'package-complete-leg-hip')} FDM",
                **BOM_CONFIG.mass_record(item["name"], item["volume_mm3"], "package-complete-leg-hip"),   # declared material
                "color": colors[stage],
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

            # Steel sleeve set entry
            slv_info = item["sleeves"]
            slv_name = slv_info["name"]
            slv_mesh = trimesh.load(ROOT / slv_info["mesh"], force="mesh", process=False)
            slv_mesh.apply_scale(0.001)
            slv_mesh.visual.face_colors = colors["sleeve"]
            for sc in (lower_leg_actuators_scene, full_body_actuators_scene):
                sc.add_geometry(
                    slv_mesh.copy(),
                    node_name=slv_name,
                    geom_name=slv_name,
                    parent_node_name="manufactured_lower_body_assembly",
                )
            compound_solids.extend(import_step(ROOT / slv_info["native_step"]).solids())

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
                "color": colors["sleeve"],
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
                        "name": f"Bilateral {stage} FDM connector ({family})",
                        "category": "Produced parts (FDM 3D printed)",
                        "quantity": 2,
                        "unit": "ea",
                        "material": BOM_CONFIG.material(item["name"], "package-complete-leg-hip"),
                        "unit_mass_g": round(BOM_CONFIG.mass_kg(item["name"], item["volume_mm3"], "package-complete-leg-hip") * 1000.0, 3),
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

        # Add bilateral captive M3 nuts and M3x8 shell screws seated inside integrated Stage 1 Shin 2-screw shell mounts (4 screws + 4 nuts per leg)
        sh2_assets = ensure_two_screw_shin_shell_assets()
        shell_hw_dir = sh2_assets["dir"]
        for hw_base, lnk_suffix, hw_rgba in [
            ("Shin_carrier_shell_shin_front_1_nut", "shin", [220, 185, 75, 255]),
            ("Shin_carrier_shell_shin_front_1_M3x8", "shin", [210, 218, 226, 255]),
            ("Shin_carrier_shell_shin_front_2_nut", "shin", [220, 185, 75, 255]),
            ("Shin_carrier_shell_shin_front_2_M3x8", "shin", [210, 218, 226, 255]),
            ("Shin_carrier_shell_shin_back_1_nut", "shin", [220, 185, 75, 255]),
            ("Shin_carrier_shell_shin_back_1_M3x8", "shin", [210, 218, 226, 255]),
            ("Shin_carrier_shell_shin_back_2_nut", "shin", [220, 185, 75, 255]),
            ("Shin_carrier_shell_shin_back_2_M3x8", "shin", [210, 218, 226, 255]),
        ]:
            hw_shape = import_step(require(shell_hw_dir / f"{hw_base}.step"))
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
                for sc in (lower_leg_actuators_scene, full_body_actuators_scene):
                    sc.add_geometry(
                        hw_m.copy(),
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

        # Build parts.json shell reference parts & add bilateral fitted leg hard-shell covers (`thigh_front`, `thigh_back`, `shin_front`, `shin_back`) + foot shells directly into ALL scenes so the hide-shell button toggles them in every view
        shell_ref_parts = []
        for r in v16["parts"]:
            if r.get("inspection_reference"):
                rc = copy.deepcopy(r)
                rc["display_opacity"] = 0.55
                shell_ref_parts.append(rc)

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
                for sc in (lower_leg_actuators_scene, full_body_actuators_scene):
                    sc.add_geometry(
                        sh_m.copy(),
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

        # Also add foot boot and sole shell panels to lower_leg_actuators_scene and full_body_actuators_scene
        full_shell_src = trimesh.load(FULL_SHELL_GLB, force="scene")
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
                for sc in (lower_leg_actuators_scene, full_body_actuators_scene):
                    sc.add_geometry(
                        m_s.copy(),
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

        # Export bilateral compound STEP and the actuator+shell-populated GLBs
        bilateral_step_path = fam_dir / "lower-leg-connectors-bilateral.step"
        export_step(Compound(children=compound_solids), bilateral_step_path)

        # View 1: Focused lower-leg assembly WITH all 6 lower-leg C047 actuators, horns, idlers, fasteners, thigh links & bilateral leg hard-shell covers
        mech_glb_path = fam_dir / "lower-leg-mechanism-context.glb"
        lower_leg_actuators_scene.export(mech_glb_path)

        # View 2: Complete 12-DOF lower-body assembly WITH all 12 C047 actuators + bilateral leg hard-shell covers
        full_lower_body_glb_path = fam_dir / "complete-lower-body-with-actuators.glb"
        full_body_actuators_scene.export(full_lower_body_glb_path)

        # View 3: Lower-leg mechanism + C047 actuators + Leg Hard Shell
        shell_fit_glb_path = fam_dir / "lower-leg-with-hard-shell.glb"
        lower_leg_actuators_scene.export(shell_fit_glb_path)

        # View 4: Complete 12-DOF lower-body mechanism + all 12 C047 actuators + full 53-node reference shell + fitted leg hard-shell covers
        full_scene = trimesh.load(FULL_SHELL_GLB, force="scene")
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
            "schema": f"cove50.lower-leg-fdm-connectors.{family}.r1",
            "version": "r1",
            "status": f"3D-printable FDM lower-leg connector redesign ({family}) with all C047 actuators in place",
            "units": {"native_STEP": "mm", "mesh": "mm", "GLB": "m"},
            "root_link": "pelvis",
            "parts": retained_parts + shell_ref_parts,
            "joints": copy.deepcopy(v16["joints"]),
            "controls": copy.deepcopy(v16["controls"]),
            "assemblies": [
                {"name": "manufactured_lower_body_assembly", "role": "Mechanical lower-body assembly with C047 actuators"},
                {"name": "full_ape_shell_assembly", "role": "Complete original visual shell assembly", "inspection_reference": True},
            ],
        }
        write_json(fam_dir / "parts.json", manifest)

        one_leg_mass = sum(r["total_stage_mass_g_PAHT_CF"] for r in cad_meta["parts"] if r["side"] == "R")
        saved_one_leg = 74.880 - one_leg_mass
        bom = {
            "scope": (
                f"Bilateral 3-stage lower-leg FDM connector replacement ({family}): "
                "6 printed PAHT-CF/PETG-CF structural parts + 40 stainless steel compression limit sleeves across both legs. "
                "Replaces 44 baseline V16 sheet-metal/spacer/sole-fastener parts across both legs while keeping all C047 actuators fixed."
            ),
            "completeness": "partial",
            "notes": (
                f"One-leg connector + sleeve mass: {one_leg_mass:.2f} g "
                f"(vs 74.88 g V16 baseline, saving {saved_one_leg:.2f} g/leg "
                f"or {2.0 * saved_one_leg:.2f} g bilateral)."
            ),
            "exclusions": [
                "Unchanged C047 servo motors, horns, bearings, foot pressure sensors, and upper-hip links",
                "Print support waste and filament drying consumables",
            ],
            "items": bom_items,
        }
        write_json(fam_dir / "bom.json", bom)
        print(f"Packaged actuator-complete GLBs, STEP, parts.json, and bom.json for {family}", flush=True)


def render_stl_on_ax(ax, stl_path: Path, color, alpha=0.98, edgecolor=(0.05, 0.08, 0.12, 0.10), lw=0.10, max_faces=200000):
    if stl_path.suffix == ".npz":
        z_data = np.load(stl_path)
        verts, faces = z_data["vertices"], z_data["faces"]
    else:
        mesh = trimesh.load(stl_path, force="mesh", process=False)
        verts = mesh.vertices
        faces = mesh.faces
    tris = verts[faces]
    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-12)
    light = np.array([-0.45, -0.65, 0.60])
    light /= np.linalg.norm(light)
    intensity = 0.55 + 0.45 * np.clip(np.abs(normals @ light), 0.0, 1.0)
    base_rgb = np.array(color[:3], dtype=float)
    face_colors = np.column_stack([np.clip(base_rgb[None, :] * intensity[:, None], 0.0, 1.0), np.full(len(tris), alpha)])
    poly = Poly3DCollection(tris, facecolors=face_colors, edgecolors=edgecolor, linewidths=lw, zsort="average")
    ax.add_collection3d(poly)
    return verts.min(axis=0), verts.max(axis=0)


def set_equal_3d(ax, mins, maxs, pad=3.5):
    center = 0.5 * (mins + maxs)
    span = 0.5 * float(np.max(maxs - mins)) + pad
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def generate_figures():
    render_dir = ART / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    v16 = read_json(V16_MANIFEST)
    v16_by_name = {p["name"]: p for p in v16["parts"]}
    fem_report = read_json(ART / "fem-report.json")
    chain_sum = fem_report["chain_summary_recovery_4kg_1p5g"]
    topopt_fields = np.load(ART / "topopt" / "topopt-fields.npz")

    # Mating C047 actuators per stage (rendered with realistic slate-blue motor housings & gold horns/idlers)
    MOTOR_COL = (0.24, 0.31, 0.40)
    HORN_COL = (0.90, 0.68, 0.22)
    stage_actuators = {
        "stage1_shin": [
            (ROOT / v16_by_name["R_C047_knee_case"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_C047_knee_main_horn"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_C047_knee_rear_idler"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_B13R14_raised_C047_ankle_pitch_case"]["mesh"], MOTOR_COL, 0.78),
            (ROOT / v16_by_name["R_B13R14_raised_C047_ankle_pitch_main_horn"]["mesh"], HORN_COL, 0.92),
            (ROOT / v16_by_name["R_B13R14_raised_C047_ankle_pitch_rear_idler"]["mesh"], HORN_COL, 0.92),
        ],
        "stage2_ankle_coupler": [
            (ROOT / v16_by_name["R_B13R14_raised_C047_ankle_pitch_case"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_B13R14_raised_C047_ankle_pitch_main_horn"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_B13R14_raised_C047_ankle_pitch_rear_idler"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_B13_C047_foot_roll_case"]["mesh"], MOTOR_COL, 0.68),
            (ROOT / v16_by_name["R_B13_C047_foot_roll_main_horn"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_B13_C047_foot_roll_rear_idler"]["mesh"], HORN_COL, 0.92),
        ],
        "stage3_foot_monocoque": [
            (ROOT / v16_by_name["R_B13_C047_foot_roll_case"]["mesh"], MOTOR_COL, 0.85),
            (ROOT / v16_by_name["R_B13_C047_foot_roll_main_horn"]["mesh"], HORN_COL, 0.96),
            (ROOT / v16_by_name["R_B13_C047_foot_roll_rear_idler"]["mesh"], HORN_COL, 0.96),
        ],
    }

    # -------------------------------------------------------------------------
    # Figure 1: Stage-by-Stage CAD Comparison WITH Mating C047 Actuators & Horns
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13.5), dpi=180)
    fig.patch.set_facecolor("#0f141c")
    fig.suptitle(
        "APE0 Lower-Leg Connectors Mated to Fixed C047 Actuators (Slate=Motor Case, Gold=Horn/Idler, Silver=SS Sleeves)",
        color="#f0f4f8",
        fontsize=14.2,
        fontweight="bold",
        y=0.97,
    )

    stages_info = [
        (
            "Stage 1: Shin + Knee/Ankle C047s",
            [
                (ROOT / v16_by_name["R_B13R14_short_inner_shin_plate"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_short_outer_shin_plate"]["mesh"], (0.84, 0.64, 0.22), 0.96),
            ],
            [
                (ART / "truss_box/parts/R_shin_truss_box.stl", (0.18, 0.58, 0.78), 0.96),
                (ART / "truss_box/parts/R_shin_truss_box_steel_sleeves.stl", (0.86, 0.89, 0.92), 0.98),
            ],
            [
                (ART / "topopt/parts/R_shin_topopt.stl", (0.88, 0.44, 0.28), 0.96),
                (ART / "topopt/parts/R_shin_topopt_steel_sleeves.stl", (0.86, 0.89, 0.92), 0.98),
            ],
            (20, -56),
            "stage1_shin",
        ),
        (
            "Stage 2: Coupler + Pitch/Roll C047s",
            [
                (ROOT / v16_by_name["R_B13R14_one_bend_output_link"]["mesh"], (0.84, 0.64, 0.22), 0.96),
            ],
            [
                (ART / "truss_box/parts/R_ankle_coupler_truss_box.stl", (0.14, 0.68, 0.58), 0.96),
                (ART / "truss_box/parts/R_ankle_coupler_truss_box_steel_sleeves.stl", (0.86, 0.89, 0.92), 0.98),
            ],
            [
                (ART / "topopt/parts/R_ankle_coupler_topopt.stl", (0.18, 0.72, 0.54), 0.96),
                (ART / "topopt/parts/R_ankle_coupler_topopt_steel_sleeves.stl", (0.86, 0.89, 0.92), 0.98),
            ],
            (22, -46),
            "stage2_ankle_coupler",
        ),
        (
            "Stage 3: Foot + Roll C047 Actuator",
            [
                (ROOT / v16_by_name["R_B13R14_front_case_L"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_rear_case_L"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_plain_sole"]["mesh"], (0.72, 0.76, 0.80), 0.96),
            ],
            [
                (ART / "truss_box/parts/R_foot_monocoque_truss_box.stl", (0.24, 0.50, 0.74), 0.96),
                (ART / "truss_box/parts/R_foot_monocoque_truss_box_steel_sleeves.stl", (0.86, 0.89, 0.92), 0.98),
            ],
            [
                (ART / "topopt/parts/R_foot_monocoque_topopt.stl", (0.86, 0.54, 0.22), 0.96),
                (ART / "topopt/parts/R_foot_monocoque_topopt_steel_sleeves.stl", (0.86, 0.89, 0.92), 0.98),
            ],
            (26, -50),
            "stage3_foot_monocoque",
        ),
    ]

    col_headers = [
        "Baseline V16 Bent Sheet & Plate + C047s\n(5052-H32 / 6061-T6, 22 parts/leg, 74.88 g/leg)",
        "Family 1: FDM Space-Truss & Box + C047s\n(PAHT-CF + SS Sleeves, 3 parts/leg, 45.68 g/leg)",
        "Family 2: 3D Topology-Optimized FDM + C047s\n(PAHT-CF + SS Sleeves, 3 parts/leg, 39.98 g/leg)",
    ]

    for row_idx, (stage_title, base_stls, truss_stls, topopt_stls, (elev, azim), st_key) in enumerate(stages_info):
        act_list = stage_actuators[st_key]
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
            # Render mating C047 actuators first
            for sp, col, alp in act_list:
                render_stl_on_ax(ax, sp, col, alpha=alp, lw=0.06)
            # Render connectors & steel sleeves and frame camera around connector + horn interface
            for sp, col, alp in stl_list:
                mn, mx = render_stl_on_ax(ax, sp, col, alpha=alp, lw=0.12)
                mins_conn.append(mn)
                maxs_conn.append(mx)
            c_min = np.min(mins_conn, axis=0)
            c_max = np.max(maxs_conn, axis=0)
            # Expand framing slightly so the mating C047 actuators are prominently visible
            set_equal_3d(ax, c_min - np.array([4.0, 6.0, 8.0]), c_max + np.array([4.0, 6.0, 10.0]), pad=3.0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("#263244")

            st_metrics = chain_sum[f"{fam_key}__{mat_key}"]["stages"][st_key]
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], color="#e2e8f0", fontsize=10.5, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.text2D(-0.08, 0.5, stage_title, transform=ax.transAxes, color="#94a3b8", fontsize=9.5, fontweight="bold", rotation=90, va="center", ha="center")

            badge = (
                f"Mass: {st_metrics['stage_mass_g']:.2f} g | u_max (4kg@1.5g): {st_metrics['max_displacement_mm']:.3f} mm\n"
                f"k/m: {st_metrics['specific_stiffness_N_mm_per_g']:.1f} (N/mm)/g | p99: {st_metrics['primary_vm_p99_MPa']:.1f} MPa ({st_metrics['p99_strength_margin_ratio']:.1f}x margin)"
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
    fig1_path = render_dir / "stage-by-stage-cad.png"
    fig.savefig(fig1_path, dpi=180)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: 3D SIMP Topology Optimization Density Fields & Extracted BRep Solids WITH Actuators
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15.5, 9.8), dpi=180)
    fig.patch.set_facecolor("#0f141c")
    fig.suptitle(
        "Blank-Space 3D SIMP Topology Optimization Fields (rho > 0.42) & Reconstructed Solids Mated to C047 Actuators",
        color="#f0f4f8",
        fontsize=13.8,
        fontweight="bold",
        y=0.96,
    )

    rho_configs = [
        ("Stage 1: Shin Anterior Arch & K-Struts", "shin_rho", ART / "topopt/parts/R_shin_topopt.stl", ART / "topopt/parts/R_shin_topopt_steel_sleeves.stl", (0.88, 0.44, 0.28), (20, -56), "stage1_shin"),
        ("Stage 2: Ankle Coupler Bifurcated Wishbone", "coupler_rho", ART / "topopt/parts/R_ankle_coupler_topopt.stl", ART / "topopt/parts/R_ankle_coupler_topopt_steel_sleeves.stl", (0.18, 0.72, 0.54), (22, -46), "stage2_ankle_coupler"),
        ("Stage 3: Foot Monocoque Vaulted Sole Ribs", "foot_rho", ART / "topopt/parts/R_foot_monocoque_topopt.stl", ART / "topopt/parts/R_foot_monocoque_topopt_steel_sleeves.stl", (0.86, 0.54, 0.22), (26, -50), "stage3_foot_monocoque"),
    ]

    for col_idx, (title, rho_key, stl_path, slv_path, col, (elev, azim), st_key) in enumerate(rho_configs):
        rho = topopt_fields[rho_key]
        ax_top = fig.add_subplot(2, 3, col_idx + 1, projection="3d")
        ax_top.set_facecolor("#161d28")
        ix, iy, iz = np.nonzero(rho > 0.42)
        vals = rho[ix, iy, iz]
        ax_top.scatter(ix, iy, iz, c=vals, cmap="inferno", s=8, alpha=0.82, vmin=0.35, vmax=1.0)
        ax_top.view_init(elev=elev, azim=azim)
        ax_top.set_title(f"{title}\n3D SIMP Voxel Density Field (rho > 0.42)", color="#e2e8f0", fontsize=10, fontweight="bold")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_zticks([])

        ax_bot = fig.add_subplot(2, 3, col_idx + 4, projection="3d")
        ax_bot.set_facecolor("#161d28")
        for sp, acol, alp in stage_actuators[st_key]:
            render_stl_on_ax(ax_bot, sp, acol, alpha=alp, lw=0.06)
        mn, mx = render_stl_on_ax(ax_bot, stl_path, col, alpha=0.96, lw=0.12)
        render_stl_on_ax(ax_bot, slv_path, (0.86, 0.89, 0.92), alpha=0.98, lw=0.08)
        set_equal_3d(ax_bot, mn - np.array([4.0, 6.0, 8.0]), mx + np.array([4.0, 6.0, 10.0]), pad=3.0)
        ax_bot.view_init(elev=elev, azim=azim)
        ax_bot.set_title(f"Extracted BRep Solid + Fixed C047 Actuators\n({stl_path.stem}, valid=True, solids=1)", color="#e2e8f0", fontsize=10, fontweight="bold")
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
        ax_bot.set_zticks([])

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    fig2_path = render_dir / "topopt-density-and-loadpaths.png"
    fig.savefig(fig2_path, dpi=180)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Complete Lower-Leg Assembly WITH Prominent C047 Actuators & Translucent Shell
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16.0, 9.5), dpi=180)
    fig.patch.set_facecolor("#0f141c")
    fig.suptitle(
        "Complete Right Lower-Leg Assembly with Fixed C047 Actuators (Knee, Ankle Pitch, Foot Roll) & Hard-Shell Envelope",
        color="#f0f4f8",
        fontsize=13.8,
        fontweight="bold",
        y=0.96,
    )

    context_motor_stls = []
    for p in v16["parts"]:
        if (
            p.get("owning_link") in {"R_roll_link", "R_shin", "R_ankle_pitch", "R_foot"}
            and not p.get("inspection_reference")
            and p["name"] not in REPLACED_V16_LOWER_LEG_PARTS
            and "fastener" not in p.get("role", "")
        ):
            pname = p["name"]
            if "case" in pname:
                col, alp = MOTOR_COL, 0.85
            elif "horn" in pname or "idler" in pname:
                col, alp = HORN_COL, 0.96
            else:
                col, alp = (0.58, 0.63, 0.68), 0.72
            context_motor_stls.append((ROOT / p["mesh"], col, alp))
    sh2_assets = ensure_two_screw_shin_shell_assets()
    context_motor_stls.extend(
        [
            (sh2_assets["shin_front_npz"], (0.75, 0.85, 0.95), 0.14),
            (sh2_assets["shin_back_npz"], (0.75, 0.85, 0.95), 0.14),
        ]
    )

    assy_panels = [
        (
            "Baseline V16 Bent Sheet & Plate + 3 C047 Actuators\n(22 connector/hardware parts/leg, 74.88 g/leg)",
            [
                (ROOT / v16_by_name["R_B13R14_short_inner_shin_plate"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_short_outer_shin_plate"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_one_bend_output_link"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_front_case_L"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_rear_case_L"]["mesh"], (0.84, 0.64, 0.22), 0.96),
                (ROOT / v16_by_name["R_B13R14_plain_sole"]["mesh"], (0.72, 0.76, 0.80), 0.96),
            ],
        ),
        (
            "Family 1: FDM Space-Truss & Box + 3 C047 Actuators\n(3 printed parts/leg + SS sleeves, 45.68 g/leg, -39% mass)",
            [
                (ART / "truss_box/parts/R_shin_truss_box.stl", (0.18, 0.58, 0.78), 0.96),
                (ART / "truss_box/parts/R_shin_truss_box_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
                (ART / "truss_box/parts/R_ankle_coupler_truss_box.stl", (0.14, 0.68, 0.58), 0.96),
                (ART / "truss_box/parts/R_ankle_coupler_truss_box_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
                (ART / "truss_box/parts/R_foot_monocoque_truss_box.stl", (0.24, 0.50, 0.74), 0.96),
                (ART / "truss_box/parts/R_foot_monocoque_truss_box_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
        ),
        (
            "Family 2: 3D Topology-Optimized FDM + 3 C047 Actuators\n(3 printed parts/leg + SS sleeves, 39.98 g/leg, -47% mass)",
            [
                (ART / "topopt/parts/R_shin_topopt.stl", (0.88, 0.44, 0.28), 0.96),
                (ART / "topopt/parts/R_shin_topopt_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
                (ART / "topopt/parts/R_ankle_coupler_topopt.stl", (0.18, 0.72, 0.54), 0.96),
                (ART / "topopt/parts/R_ankle_coupler_topopt_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
                (ART / "topopt/parts/R_foot_monocoque_topopt.stl", (0.86, 0.54, 0.22), 0.96),
                (ART / "topopt/parts/R_foot_monocoque_topopt_steel_sleeves.stl", (0.88, 0.90, 0.92), 0.98),
            ],
        ),
    ]

    for col_idx, (p_title, conn_list) in enumerate(assy_panels):
        ax = fig.add_subplot(1, 3, col_idx + 1, projection="3d")
        ax.set_facecolor("#161d28")
        mins_all, maxs_all = [], []
        for sp, col, alp in context_motor_stls + conn_list:
            mn, mx = render_stl_on_ax(ax, require(sp), col, alpha=alp, lw=0.06 if alp < 0.5 else 0.10)
            mins_all.append(mn)
            maxs_all.append(mx)
        set_equal_3d(ax, np.min(mins_all, axis=0), np.max(maxs_all, axis=0), pad=3.5)
        ax.view_init(elev=18, azim=-52)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#263244")
        ax.set_title(p_title, color="#f8fafc", fontsize=10.8, fontweight="bold", pad=10)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    fig5_path = render_dir / "shell-fit-views.png"
    fig.savefig(fig5_path, dpi=180)
    plt.close(fig)
    print("Updated actuator-integrated figures in", render_dir, flush=True)


if __name__ == "__main__":
    build_family_packages()
    generate_figures()
