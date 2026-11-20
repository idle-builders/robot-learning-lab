"""Package GLB assemblies (`assembly-with-full-shell.glb`), `assembly-parts.json` manifests
(with `display_group: "ape_body_shell"` for the "Hide shell" toggle), `connectors-bilateral.step`,
and high-resolution engineering renders for all 3 Arm FDM Connector studies:
  1. `cove50-arm-fdm-truss-box-r1` (`truss_box`)
  2. `cove50-arm-fdm-topopt-r1` (`topopt`)
  3. `cove50-arm-fdm-combined-r1` (`combined`)
"""
from pathlib import Path
import copy
import math
import os
import struct
import subprocess
import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build123d import Compound, export_step, import_step

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from infra.storage.blob_store import ensure, require  # noqa: E402  inputs restore by path or the build fails
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ARM_SHELL_DIR,
    ART,
    BASELINE_STEPS,
    MATERIALS,
    REPLACED_ARM_PARTS,
    ROOT,
    STAGE_LINKS,
    STAGE_PART_BASENAMES,
    ensure_arm_shell_nut_assets,
    read_json,
    ref,
    rel,
    sha,
    solid_to_watertight_arrays,
    write_json,
)

STAGE_COLORS = {
    "stage1_shoulder_pitch_to_roll": [42, 138, 184, 255],
    "stage2_shoulder_roll_to_yaw": [28, 164, 142, 255],
    "stage3_upper_arm_yaw_to_elbow": [212, 134, 52, 255],
    "stage4_forearm_elbow_to_wrist": [64, 156, 212, 255],
    "sleeve": [215, 222, 230, 255],
    "nut": [220, 185, 75, 255],
    "screw": [210, 218, 226, 255],
    "hard_cover": [231, 222, 201, 255],
}

PRINT_ORIENTATIONS = {
    "stage1_shoulder_pitch_to_roll": "XZ Roll-Case Flange Bed (layer normal Y)",
    "stage2_shoulder_roll_to_yaw": "XZ Roll-Horn Flange Bed (layer normal Y)",
    "stage3_upper_arm_yaw_to_elbow": "YZ Elbow-Ear Flange Bed (layer normal X)",
    "stage4_forearm_elbow_to_wrist": "YZ Medial-Spine Bed (layer normal X)",
}

CPP_RENDERER = ROOT / "robot/electronics/power/cad/zbuffer_renderer.cpp"
EXE_RENDERER = ART / "zbuffer_renderer_bin"


def raster_meshes(items: list[tuple[str, trimesh.Trimesh, list[int]]], view, box=None, W: int = 480, H: int = 720) -> Image.Image:
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
    light_dir = np.array([0.3, -0.5, 0.8], dtype=float)
    light_dir /= np.linalg.norm(light_dir)
    for _, m, c in items:
        tri = m.triangles
        c_rgb = np.asarray(c[:3], dtype=float) / 255.0
        shade = 0.66 + 0.34 * np.abs(m.face_normals @ light_dir)
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
    scale = min(W / max(hi[0] - lo[0], 1e-3), H / max(hi[1] - lo[1], 1e-3)) * 0.88
    middle = 0.5 * (lo + hi)
    xyz = np.empty_like(tri)
    xyz[:, :, 0] = (xy[:, :, 0] - middle[0]) * scale + W * 0.5
    xyz[:, :, 1] = H * 0.5 - (xy[:, :, 1] - middle[1]) * scale
    xyz[:, :, 2] = tri @ view
    data = np.concatenate([xyz.reshape(-1, 9), col], axis=1).astype("<f4")
    target = ART / f"tmp-render-{os.getpid()}.ppm"
    subprocess.run(
        [str(EXE_RENDERER), "/dev/stdin", str(target)],
        input=struct.pack("<III", W, H, len(tri)) + data.tobytes(),
        check=True,
    )
    im = Image.open(target).copy()
    target.unlink(missing_ok=True)
    return im


def load_npz_trimesh(npz_path: Path) -> trimesh.Trimesh:
    npz_path = ensure(npz_path)
    if not npz_path.exists():
        # a left part that exists nowhere (built and stored only as the right one) is its mirror
        r_name = npz_path.name.replace("__L_", "__R_")
        if r_name == npz_path.name and npz_path.name.startswith("L_"):
            r_name = "R_" + npz_path.name[2:]
        r_path = ensure(npz_path.parent / r_name)
        z = np.load(r_path)
        v = np.asarray(z["vertices"], dtype=np.float64).copy()
        v[:, 0] *= -1.0
        f = np.asarray(z["faces"], dtype=np.int64)[:, [0, 2, 1]]
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    z = np.load(npz_path)
    return trimesh.Trimesh(vertices=np.asarray(z["vertices"], dtype=np.float64), faces=np.asarray(z["faces"], dtype=np.int64), process=False)


def actuator_rgba(name: str) -> list[int]:
    if "case" in name:
        return [64, 72, 82, 255]
    if "horn" in name or "idler" in name or "bracket" in name:
        return [182, 190, 198, 255]
    if "M3" in name or "PA2" in name or "M2" in name or "screw" in name:
        return [205, 212, 220, 255]
    return [96, 112, 122, 255]


def package_family(fam: str, hw: dict) -> dict:
    fam_dir = ART / fam
    man = read_json(fam_dir / "manifest.json")
    src = read_json(ARM_SHELL_DIR / "source.json")
    mp = read_json(ARM_SHELL_DIR / "manufactured-parts.json")

    replaced = set(mp["replaced_source"]) | REPLACED_ARM_PARTS

    arm_links = {
        "torso",
        "R_shoulder_pitch_link",
        "R_shoulder_roll_link",
        "R_upper_arm",
        "R_forearm",
        "L_shoulder_pitch_link",
        "L_shoulder_roll_link",
        "L_upper_arm",
        "L_forearm",
    }

    scene = trimesh.Scene()
    scene.graph.update(frame_to="manufactured_arm_assembly", matrix=np.eye(4))
    assembly_parts = []
    render_items_r_with_shell = []
    render_items_r_no_shell = []
    render_items_bilateral = []
    stage_meshes_r = {}

    def _add_to_scene(name: str, m_mm: trimesh.Trimesh, rgba: list[int], entry: dict, is_shell: bool = False):
        m_m = m_mm.copy()
        m_m.apply_scale(0.001)
        m_m.visual.face_colors = np.asarray(rgba, dtype=np.uint8)
        scene.add_geometry(
            m_m,
            node_name=name,
            geom_name=name,
            parent_node_name="manufactured_arm_assembly",
        )
        assembly_parts.append(entry)
        owner = entry["owning_link"]
        if owner.startswith("R_") or (owner == "torso" and "R_pitch" in name):
            render_items_r_with_shell.append((name, m_mm, rgba))
            if not is_shell:
                render_items_r_no_shell.append((name, m_mm, rgba))
        if owner.startswith("R_") or owner.startswith("L_") or (owner == "torso" and "pitch" in name):
            render_items_bilateral.append((name, m_mm, rgba))

    # 1. Source motors, horns, screws, and wrist/hand context
    for p in src["parts"]:
        pname = p["name"]
        owner = p["owning_link"]
        if pname in replaced or owner not in arm_links:
            continue
        # Only include arm-relevant torso actuators (`R/L_pitch_C047_case`, `R/L_pitch_main_horn`, `R/L_pitch_horn_M3x8_*`)
        if owner == "torso" and not ("R_pitch" in pname or "L_pitch" in pname):
            continue
        rel_mesh = p["package"]["exact"]["path"]  # always the exact package, never whichever copy is on disk
        m_mm = load_npz_trimesh(ROOT / rel_mesh)
        rgba = actuator_rgba(pname)
        entry = {
            "name": pname,
            "owning_link": owner,
            "role": p.get("role", "source_actuator"),
            "mesh": rel_mesh,
            "mesh_units": "mm",
            "bounds_mm": [m_mm.bounds[0].round(3).tolist(), m_mm.bounds[1].round(3).tolist()],
            "color": rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "parent_assembly": "manufactured_arm_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(pname, m_mm, rgba, entry, is_shell=False)

    # 2. Add the 8 bilateral FDM arm connectors + 8 bilateral sleeve sets (`64` sleeves total)
    compound_solids = []
    bom_items = []
    for item in man["parts"]:
        pname = item["name"]
        stage_key = item["stage"]
        side = item["side"]
        owner = item["owning_link"]
        rgba = STAGE_COLORS[stage_key]
        p_orient = PRINT_ORIENTATIONS[stage_key]

        step_p = require(ROOT / item["step"]["path"])
        npz_p = ROOT / item["mesh"]["path"]
        m_mm = load_npz_trimesh(npz_p)
        if side == "R":
            stage_meshes_r[stage_key] = m_mm
        compound_solids.extend(import_step(step_p).solids())

        poly_entry = {
            "name": pname,
            "owning_link": owner,
            "role": "printed_structural_connector",
            "physical_part_id": pname,
            "bom_group": pname,
            "native_step": item["step"]["path"],
            "native_step_sha256": item["step"]["sha256"],
            "mesh": item["mesh"]["path"],
            "mesh_sha256": item["mesh"]["sha256"],
            "mesh_units": "mm",
            "bounds_mm": [m_mm.bounds[0].round(3).tolist(), m_mm.bounds[1].round(3).tolist()],
            "native_solid_count": 1,
            "native_validity": [True],
            "material": "Bambu PAHT-CF / PA12-CF (1.22 g/cm3) or PETG-CF (1.25 g/cm3) FDM",
            "mass_kg": round(item["polymer_mass_paht_cf_g"] / 1000.0, 6),
            "mass_basis": "Exact native STEP volume * 1.22 g/cm3 (PAHT-CF)",
            "color": rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "parent_assembly": "manufactured_arm_assembly",
            "inspection_reference": False,
            "volume_mm3": item["volume_mm3"],
            "print_orientation": p_orient,
        }
        _add_to_scene(pname, m_mm, rgba, poly_entry, is_shell=False)

        # Sleeves
        slv_name = f"{pname}_steel_sleeves"
        slv_step_p = require(ROOT / item["sleeves_step"]["path"])
        slv_npz_p = ROOT / item["sleeves_mesh"]["path"]
        m_slv = load_npz_trimesh(slv_npz_p)
        compound_solids.extend(import_step(slv_step_p).solids())

        slv_entry = {
            "name": slv_name,
            "owning_link": owner,
            "role": "compression_limit_sleeves",
            "physical_part_id": slv_name,
            "bom_group": slv_name,
            "native_step": item["sleeves_step"]["path"],
            "native_step_sha256": item["sleeves_step"]["sha256"],
            "mesh": item["sleeves_mesh"]["path"],
            "mesh_sha256": item["sleeves_mesh"]["sha256"],
            "mesh_units": "mm",
            "bounds_mm": [m_slv.bounds[0].round(3).tolist(), m_slv.bounds[1].round(3).tolist()],
            "native_solid_count": item["sleeves_count"],
            "native_validity": [True] * item["sleeves_count"],
            "material": "304 Stainless Steel compression sleeves (7.85 g/cm3)",
            "mass_kg": round(item["sleeves_mass_g"] / 1000.0, 6),
            "color": STAGE_COLORS["sleeve"],
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "parent_assembly": "manufactured_arm_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(slv_name, m_slv, STAGE_COLORS["sleeve"], slv_entry, is_shell=False)

        if side == "R":
            bom_items.append({
                "id": pname[2:],
                "name": f"Bilateral {stage_key} FDM arm connector ({fam})",
                "category": "Produced parts (FDM 3D printed)",
                "quantity": 2,
                "unit": "ea",
                "material": "Bambu PAHT-CF / PA12-CF (or PETG-CF)",
                "unit_mass_g": item["polymer_mass_paht_cf_g"],
                "process": f"Home-lab FDM 3D print ({p_orient})",
                "notes": f"Integrates dual-cheek C047 clamping & 2-screw hard-shell mounts on Stages 1 & 4; {item['sleeves_count']} steel sleeve seats.",
            })
            bom_items.append({
                "id": f"{STAGE_PART_BASENAMES[stage_key]}_sleeves",
                "name": f"Bilateral {stage_key} 304 SS compression sleeve set ({item['sleeves_count']} sleeves/arm)",
                "category": "Fasteners & Hardware",
                "quantity": 2,
                "unit": "set",
                "material": "304 Stainless Steel",
                "unit_mass_g": item["sleeves_mass_g"],
                "process": "Off-the-shelf 304 SS cylindrical sleeves pressed into printed counterbores",
                "notes": f"{item['sleeves_count']} steel sleeves per stage prevent bolt preload creep across dual-cheek motor/horn/wrist interfaces.",
            })

    # 3. Add the 16 bilateral captive M3 nuts and 16 shell M3x8 screws + 4 yawcase PA2x9.2 screws + 10 hard-shell covers (`ape_body_shell`)
    for n in hw["nuts"]:
        m_nut = load_npz_trimesh(n["npz_path"])
        n_entry = {
            "name": n["name"],
            "owning_link": n["owning_link"],
            "role": "fastener",
            "native_step": rel(n["step_path"]),
            "native_step_sha256": sha(n["step_path"]),
            "mesh": rel(n["npz_path"]),
            "mesh_sha256": sha(n["npz_path"]),
            "mesh_units": "mm",
            "color": STAGE_COLORS["nut"],
            "display_group": "ape_body_shell",
            "display_groups": ["ape_body_shell"],
            "parent_assembly": "manufactured_arm_assembly",
            "inspection_reference": True,
        }
        _add_to_scene(n["name"], m_nut, STAGE_COLORS["nut"], n_entry, is_shell=True)

    for scr in hw["screws"]:
        m_scr = load_npz_trimesh(scr["npz_path"])
        s_entry = {
            "name": scr["name"],
            "owning_link": scr["owning_link"],
            "role": "fastener",
            "native_step": rel(scr["step_path"]),
            "native_step_sha256": sha(scr["step_path"]),
            "mesh": rel(scr["npz_path"]),
            "mesh_sha256": sha(scr["npz_path"]),
            "mesh_units": "mm",
            "color": STAGE_COLORS["screw"],
            "display_group": "ape_body_shell",
            "display_groups": ["ape_body_shell"],
            "parent_assembly": "manufactured_arm_assembly",
            "inspection_reference": True,
        }
        _add_to_scene(scr["name"], m_scr, STAGE_COLORS["screw"], s_entry, is_shell=True)

    for p in mp["parts"]:
        pname = p["name"]
        if pname in replaced or "insert" in pname or pname in ("R_S_yaw_to_elbow", "L_S_yaw_to_elbow"):
            continue
        m_mp = load_npz_trimesh(ROOT / p["lossless_mesh"]["path"])
        if p["role"] == "hard_cover":
            sh_entry = {
                "name": f"{pname}_hard_cover",
                "owning_link": p["owning_link"],
                "role": "body_shell",
                "material": "PETG / polycarbonate fitted arm hard-shell cover (2 M3 screws per panel)",
                "mesh": p["lossless_mesh"]["path"],
                "mesh_sha256": p["lossless_mesh"]["sha256"],
                "mesh_units": "mm",
                "color": STAGE_COLORS["hard_cover"],
                "display_group": "ape_body_shell",
                "display_groups": ["ape_body_shell"],
                "display_opacity": 0.65,
                "parent_assembly": "manufactured_arm_assembly",
                "inspection_reference": True,
            }
            _add_to_scene(f"{pname}_hard_cover", m_mp, STAGE_COLORS["hard_cover"], sh_entry, is_shell=True)
        else:
            # Shell screws (`R/L_shoulder_front_M3x8_339`, `R/L_shoulder_back_M3x8_339`, `R/L_fore_front_M3x8_*`, `R/L_fore_back_M3x8_*`) toggle with `ape_body_shell`;
            # actuator screws (`R/L_yawcase_PA2x9p2_*`) stay in `mechanism`.
            is_shell_screw = "M3x8" in pname
            d_group = "ape_body_shell" if is_shell_screw else "mechanism"
            f_entry = {
                "name": pname,
                "owning_link": p["owning_link"],
                "role": "fastener",
                "mesh": p["lossless_mesh"]["path"],
                "mesh_sha256": p["lossless_mesh"]["sha256"],
                "mesh_units": "mm",
                "color": STAGE_COLORS["screw"],
                "display_group": d_group,
                "display_groups": [d_group],
                "parent_assembly": "manufactured_arm_assembly",
                "inspection_reference": is_shell_screw,
            }
            _add_to_scene(pname, m_mp, STAGE_COLORS["screw"], f_entry, is_shell=is_shell_screw)

    # Export bilateral STEP compound and single-file GLB assembly with full shell toggle
    bilateral_step = fam_dir / "connectors-bilateral.step"
    export_step(Compound(children=compound_solids), str(bilateral_step))

    glb_path = fam_dir / "assembly-with-full-shell.glb"
    scene.export(str(glb_path))

    arm_joints = [j for j in src["joints"] if "shoulder" in j["name"] or j["name"].endswith("_elbow")]
    arm_controls = [c for c in src["controls"] if "shoulder" in c["name"] or c["name"].endswith("_elbow")]
    write_json(
        fam_dir / "assembly-parts.json",
        {
            "family": fam,
            "root_link": "torso",
            "units": "CAD mm; GLB m",
            "parts": assembly_parts,
            "bom": {
                "scope": f"Bilateral 4-stage Blank-State Multi-Connection FDM arm structural connectors ({fam}), 68 pressed-in 304 SS compression sleeves (34/arm), and 16 captive M3 shell nuts",
                "completeness": "partial",
                "notes": "Replaces upper__R/L_pitch_to_roll_one_piece, upper__R/L_roll_to_yaw_one_piece, R/L_S_yaw_to_elbow, R/L_forearm_frame_drilled, R/L_shoulder_carrier, and R/L_fore_carrier; adds wide-clearance mid-shoulder roll yoke and dual-cheek C047 clamping on the Elbow case.",
                "exclusions": ["Retained C047 servo motors, horns, wrist brackets, and fitted hard-shell panels"],
                "items": bom_items,
            },
            "joints": arm_joints,
            "controls": arm_controls,
        },
    )

    # Render shell-fit and covers-off views for this family
    renders_dir = fam_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1920, 780), "#f3f1eb")
    draw = ImageDraw.Draw(canvas)
    panels = [
        ("Right Arm - Front (Shell On)", render_items_r_with_shell, [0.0, -1.0, 0.10]),
        ("Right Arm - Back (Shell On)", render_items_r_with_shell, [0.0, 1.0, 0.10]),
        ("Right Arm - Lateral (Shell On)", render_items_r_with_shell, [1.0, -0.15, 0.08]),
        (f"Covers Off - {fam} Connectors & Sleeves", render_items_r_no_shell, [0.75, -0.95, 0.12]),
    ]
    for idx, (title, items_list, v_dir) in enumerate(panels):
        im = raster_meshes(items_list, v_dir, W=480, H=720)
        canvas.paste(im, (idx * 480, 40))
        draw.text((idx * 480 + 14, 12), title, fill="#22282e")
    canvas.save(renders_dir / "shell-fit-views.png")

    return {
        "glb_path": glb_path,
        "step_path": bilateral_step,
        "stage_meshes_r": stage_meshes_r,
    }


def render_comparison_charts(stage_meshes_by_fam: dict):
    renders_dir = ART / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    fem_rep = read_json(ART / "fem-evaluation.json")

    # 1. Stage-by-stage CAD comparison grid (`baseline_metal` vs `truss_box` vs `topopt` vs `combined` across Stages 1..4)
    canvas = Image.new("RGB", (1920, 1480), "#f3f1eb")
    draw = ImageDraw.Draw(canvas)
    stage_titles = [
        ("stage1_shoulder_pitch_to_roll", "Stage 1: Box-Cradle Yoke (10 conn)", [0.40, 0.85, 0.35]),
        ("stage2_shoulder_roll_to_yaw", "Stage 2: 3D Windowed Saddle (8 conn)", [0.50, -0.80, 0.35]),
        ("stage3_upper_arm_yaw_to_elbow", "Stage 3: 3D Twin-Branch Wishbone (8 conn)", [0.68, -0.70, 0.28]),
        ("stage4_forearm_elbow_to_wrist", "Stage 4: Dual-Cheek Truss Cage (14 conn)", [0.68, -0.68, 0.28]),
    ]

    # Row 0: Baseline Sheet-Metal STEP parts
    for c_idx, (st_key, st_title, v_dir) in enumerate(stage_titles):
        b_sol = import_step(require(BASELINE_STEPS[st_key]))
        bv, bf = solid_to_watertight_arrays(b_sol, tol=0.05)
        bm = trimesh.Trimesh(vertices=bv, faces=bf, process=False)
        im = raster_meshes([(st_key, bm, [142, 150, 160, 255])], v_dir, W=480, H=330)
        canvas.paste(im, (c_idx * 480, 35))
        b_row = next(
            r for r in fem_rep["rows"]
            if r["stage"] == st_key and r["family"] == "baseline_metal" and r["case_id"] == "recovery_4kg_1p5g"
        )
        draw.text(
            (c_idx * 480 + 12, 10),
            f"{st_title} - Baseline Sheet ({b_row['stage_mass_g']:.2f} g)",
            fill="#22282e",
        )

    row_fams = [
        ("truss_box", "Family 1: Clean-Sheet Truss/Box"),
        ("topopt", "Family 2: Clean-Sheet 3D SIMP TopOpt"),
        ("combined", "Combined Optimal 4-Stage Arm"),
    ]
    for r_idx, (fam_key, fam_label) in enumerate(row_fams, start=1):
        man = read_json(ART / fam_key / "manifest.json")
        for c_idx, (st_key, st_title, v_dir) in enumerate(stage_titles):
            m = stage_meshes_by_fam[fam_key][st_key]
            base_name = STAGE_PART_BASENAMES[st_key]
            m_slv = load_npz_trimesh(ART / fam_key / "meshes" / f"R_{base_name}_sleeves.npz")
            im = raster_meshes(
                [(st_key, m, STAGE_COLORS[st_key]), (f"{st_key}_slv", m_slv, STAGE_COLORS["sleeve"])],
                v_dir,
                W=480,
                H=330,
            )
            canvas.paste(im, (c_idx * 480, r_idx * 365 + 35))
            mass_g = man["stages"][st_key]["total_stage_mass_paht_cf_g"]
            sel_tag = f" [{man['stages'][st_key].get('selected_family', fam_key)}]" if fam_key == "combined" else ""
            draw.text(
                (c_idx * 480 + 12, r_idx * 365 + 10),
                f"{st_title} - {fam_label}{sel_tag} ({mass_g:.2f} g)",
                fill="#22282e",
            )
    canvas.save(renders_dir / "stage-by-stage-cad.png")

    # 2. FEM stress, displacement, and specific stiffness comparison chart (`fem-stress-displacement-comparison.png`)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160)
    stages = list(STAGE_PART_BASENAMES.keys())
    short_labels = ["S1: Pitch->Roll", "S2: Roll->Yaw", "S3: Yaw->Elbow", "S4: Forearm->Wrist"]
    x = np.arange(len(stages))
    w = 0.20

    fams_plot = [
        ("baseline_metal", "Baseline Sheet+Carriers", "#7a8694"),
        ("truss_box", "Family 1: Truss/Box (PAHT-CF)", "#2a8ab8"),
        ("topopt", "Family 2: SIMP TopOpt (PAHT-CF)", "#1ca48e"),
        ("combined", "Combined Optimal (PAHT-CF)", "#d48634"),
    ]
    for idx, (fk_name, label, col) in enumerate(fams_plot):
        u_vals = []
        p99_vals = []
        km_vals = []
        for st in stages:
            row = next(
                r for r in fem_rep["rows"]
                if r["stage"] == st and r["family"] == fk_name
                and (r["material"] == "PAHT_CF_dry" or (fk_name == "baseline_metal" and r["material"] in ("Al5052_H32", "PAHT_CF_dry")))
                and r["case_id"] == "recovery_4kg_1p5g"
            )
            u_vals.append(row["max_displacement_mm"])
            p99_vals.append(row["von_mises_p99_MPa"])
            km_vals.append(row["specific_stiffness_N_per_mm_per_g"])
        axes[0].bar(x + (idx - 1.5) * w, u_vals, width=w, label=label, color=col)
        axes[1].bar(x + (idx - 1.5) * w, p99_vals, width=w, label=label, color=col)
        axes[2].bar(x + (idx - 1.5) * w, km_vals, width=w, label=label, color=col)

    axes[0].set_title("Peak Deflection @ 4kg 1.5g Recovery (mm)")
    axes[0].set_xticks(x, short_labels, rotation=15)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Von Mises p99 Stress @ 4kg 1.5g (MPa)")
    axes[1].set_xticks(x, short_labels, rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].set_title("Specific Stiffness k/m ((N/mm)/g)")
    axes[2].set_xticks(x, short_labels, rotation=15)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Cove-50 4-Stage Arm FDM Connectors — Matched 3D Tetrahedral FEM (4 kg @ 1.5g Recovery Wrench)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(renders_dir / "fem-stress-displacement-comparison.png")
    plt.close(fig)

    # 3. Weight-to-strength & single-arm mass comparison chart (`weight-to-strength-chart.png`)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=160)
    cs = fem_rep["chain_summaries"]
    names = ["Baseline\n(Sheet+Carriers)", "Family 1\n(Truss/Box)", "Family 2\n(SIMP TopOpt)", "Combined\nOptimal"]
    keys = ["baseline_metal", "truss_box", "topopt", "combined"]
    cols = ["#7a8694", "#2a8ab8", "#1ca48e", "#d48634"]
    masses = [cs[k]["single_arm_4stage_mass_g"] for k in keys]
    rss_u = [cs[k]["recovery_4kg_1p5g_rss_displacement_mm"] for k in keys]

    axes[0].bar(names, masses, color=cols, width=0.52)
    axes[0].set_title("Single-Arm 4-Stage Mass Including Shell Mounts & Sleeves (g)")
    axes[0].set_ylabel("Mass per arm (g)")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(masses):
        axes[0].text(i, v + 1.0, f"{v:.1f} g", ha="center", fontweight="bold", fontsize=9)

    axes[1].bar(names, rss_u, color=cols, width=0.52)
    axes[1].set_title("4-Stage RSS Chain Deflection @ 4kg 1.5g Recovery (mm)")
    axes[1].set_ylabel("RSS Displacement (mm)")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(rss_u):
        axes[1].text(i, v + 0.02, f"{v:.3f} mm", ha="center", fontweight="bold", fontsize=9)

    fig.suptitle("Cove-50 Arm Architecture — Mass Savings (Eliminating Separate Carriers) & Chain Deflection", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(renders_dir / "weight-to-strength-chart.png")
    plt.close(fig)

    # 4. 3D SIMP Topology Optimization density projection plot (`topopt-density-and-loadpaths.png`)
    topopt_npz = np.load(require(ART / "topopt/arm-topopt-fields.npz"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=160)
    for idx, (prefix, title) in enumerate([
        ("stage1_shoulder_pitch_to_roll", "Stage 1: Dual-Cheek Pitch->Roll SIMP"),
        ("stage2_shoulder_roll_to_yaw", "Stage 2: 3D Roll->Yaw Saddle SIMP"),
        ("stage3_upper_arm_yaw_to_elbow", "Stage 3: 3D Twin-Branch Wishbone SIMP"),
        ("stage4_forearm_elbow_to_wrist", "Stage 4: Dual-Cheek Forearm->Wrist SIMP"),
    ]):
        rho = topopt_npz[f"{prefix}__rho"] if f"{prefix}__rho" in topopt_npz else topopt_npz[f"{prefix}_rho"]
        proj = np.max(rho, axis=1).T
        im = axes[idx].imshow(proj, origin="lower", cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        axes[idx].set_title(title, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("X voxel index")
        axes[idx].set_ylabel("Z voxel index")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="SIMP Relative Density ρ")
    fig.suptitle("3D Multi-Load SIMP Topology Optimization Density Projections (Stages 1–4)", fontsize=12, fontweight="bold")
    fig.savefig(renders_dir / "topopt-density-and-loadpaths.png", bbox_inches="tight")
    plt.close(fig)


def main():
    hw = ensure_arm_shell_nut_assets()
    stage_meshes_by_fam = {}
    for fam in ("truss_box", "topopt", "combined"):
        res = package_family(fam, hw)
        stage_meshes_by_fam[fam] = res["stage_meshes_r"]
        print(f"Packaged {fam}: {res['glb_path'].name} and {res['step_path'].name}")
    render_comparison_charts(stage_meshes_by_fam)
    print("Rendered all comparison figures in", ART / "renders")


if __name__ == "__main__":
    main()
