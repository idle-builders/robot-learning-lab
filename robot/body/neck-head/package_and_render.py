"""Package GLB assemblies (`assembly-with-full-shell.glb`), `assembly-parts.json` manifests
(with `display_group: "ape_body_shell"` for the "Hide shell" toggle and interactive `neck_yaw`
and `neck_pitch` joint controls), `connectors-4stage.step`, and high-resolution engineering
renders for all 3 Neck & Head FDM Connector studies:
  1. `cove50-neck-head-fdm-truss-box-r1` (`truss_box`)
  2. `cove50-neck-head-fdm-topopt-r1` (`topopt`)
  3. `cove50-neck-head-fdm-combined-r1` (`combined`)
"""
from __future__ import annotations

from pathlib import Path
import os
import struct
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bom"))
import bom_config as BOM_CONFIG  # noqa: E402  (joint ranges from the body definitions)
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
    ART,
    AXIS_NECK_PITCH,
    AXIS_NECK_YAW,
    BASELINE_STEPS,
    EC_MANIFEST,
    MATERIALS,
    P_NECK_PITCH,
    P_NECK_YAW,
    REPLACED_NECK_HEAD_PARTS,
    ROOT,
    STAGE_LINKS,
    STAGE_PART_BASENAMES,
    classify_neck_head_link,
    read_json,
    ref,
    rel,
    sha,
    solid_to_watertight_arrays,
    write_json,
)

CFG = BOM_CONFIG.section("build-neck-head-connectors")   # configured values: robot/body/neck-head/build_neck_head_connectors.bom.yaml config:
STAGE_COLORS = BOM_CONFIG.unfrozen(CFG["STAGE_COLORS"])

PRINT_ORIENTATIONS = CFG["PRINT_ORIENTATIONS"]

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
        if len(m.faces) == 0:
            continue
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
    z = np.load(npz_path)
    return trimesh.Trimesh(
        vertices=np.asarray(z["vertices"], dtype=np.float64),
        faces=np.asarray(z["faces"], dtype=np.int64),
        process=False,
    )


def source_part_rgba(p: dict) -> list[int]:
    nm = p.get("name", "")
    if p.get("color") and len(p["color"]) == 4:
        return [int(x) for x in p["color"]]
    if "case" in nm:
        return [64, 72, 82, 255]
    if "horn" in nm or "idler" in nm:
        return [182, 190, 198, 255]
    if "ROCK4D" in nm or "Camera" in nm or "Microphone" in nm:
        return [48, 118, 78, 255]
    return [128, 136, 146, 255]


def package_family(fam: str, build_sum: dict, fem_rep: dict) -> dict:
    fam_dir = ART / fam
    fam_dir.mkdir(parents=True, exist_ok=True)
    ec = read_json(EC_MANIFEST)

    scene = trimesh.Scene()
    scene.graph.update(frame_to="manufactured_neck_head_assembly", matrix=np.eye(4))
    assembly_parts = []
    render_with_shell = []
    render_no_shell = []
    stage_meshes = {}

    def _add_to_scene(name: str, m_mm: trimesh.Trimesh, rgba: list[int], entry: dict, is_shell: bool = False):
        m_m = m_mm.copy()
        m_m.apply_scale(0.001)
        m_m.visual.face_colors = np.asarray(rgba, dtype=np.uint8)
        scene.add_geometry(
            m_m,
            node_name=name,
            geom_name=name,
            parent_node_name="manufactured_neck_head_assembly",
        )
        assembly_parts.append(entry)
        # Filter torso render items to Z >= 340 mm and |X| <= 75 mm so neck & head fill the viewport cleanly
        bmin, bmax = m_mm.bounds
        if bmax[2] >= 342.0 and bmin[0] <= 75.0 and bmax[0] >= -75.0:
            if is_shell and ("torso_front" in name or "torso_back" in name):
                # Keep upper torso shell out of tight head/neck render panels so head skin & neck cowl are clear
                pass
            else:
                render_with_shell.append((name, m_mm, rgba))
            if not is_shell:
                render_no_shell.append((name, m_mm, rgba))

    # 1. Active head, neck_yaw, and upper torso parts from EC_MANIFEST (excluding REPLACED_NECK_HEAD_PARTS)
    for p in ec["parts"]:
        pname = p.get("name", "")
        if pname in REPLACED_NECK_HEAD_PARTS:
            continue
        lk = classify_neck_head_link(p)
        if lk is None:
            continue
        b = p.get("bounds_mm")
        if lk == "torso" and b and (b[1][2] < 340.0 or b[0][0] > 68.0 or b[1][0] < -68.0):
            # Exclude far-away shoulder/arm torso parts so the neck-head GLB focuses on torso spine + neck + head
            if "torso_front" not in pname and "torso_back" not in pname and "Common_power_carrier" not in pname:
                continue
        mp = p.get("mesh")
        if not mp:
            continue
        m_mm = trimesh.load(require(ROOT / mp), force="mesh")
        is_sh = bool(
            p.get("role") == "body_shell"
            or "shell" in pname
            or "cowl" in pname
            or "skin" in pname
            or pname in ("flush_skin_M2x8_1", "flush_skin_M2x8_-1")
        )
        rgba = STAGE_COLORS["shell"] if is_sh and ("skin" in pname or "cowl" in pname or "shell" in pname) else source_part_rgba(p)
        d_group = "ape_body_shell" if is_sh else "mechanism"
        entry = {
            "name": pname,
            "owning_link": lk,
            "role": "body_shell" if is_sh else p.get("role", "component"),
            "mesh": mp,
            "mesh_units": "mm",
            "bounds_mm": [m_mm.bounds[0].round(3).tolist(), m_mm.bounds[1].round(3).tolist()],
            "color": rgba,
            "display_group": d_group,
            "display_groups": [d_group],
            "parent_assembly": "manufactured_neck_head_assembly",
            "inspection_reference": is_sh,
        }
        if is_sh:
            entry["display_opacity"] = 0.62
        if p.get("native_step"):
            entry["native_step"] = p["native_step"]
        _add_to_scene(pname, m_mm, rgba, entry, is_shell=is_sh)

    # 2. Add the 4 FDM neck & head stage connectors + 34 pressed-in 304 SS compression sleeves
    compound_solids = []
    bom_items = []
    manifest_stages = {}
    manifest_parts_out = []

    for stg_key, bname in STAGE_PART_BASENAMES.items():
        if fam == "combined":
            sel_fam = fem_rep["combined_selection"][stg_key]["selected_family"]
        else:
            sel_fam = fam
        p_rec = next(
            r for r in build_sum["parts"]
            if r["family"] == sel_fam and r["stage"] == stg_key
        )
        part_name = f"{fam}__{bname}"
        step_p = ART / "parts" / f"{part_name}.step"
        npz_p = ART / "parts" / f"{part_name}.npz"
        m_mm = load_npz_trimesh(require(npz_p))
        stage_meshes[stg_key] = m_mm
        compound_solids.extend(import_step(str(require(step_p))).solids())

        owner = STAGE_LINKS[stg_key]
        rgba = STAGE_COLORS[stg_key]
        p_orient = PRINT_ORIENTATIONS[stg_key]

        poly_entry = {
            "name": part_name,
            "owning_link": owner,
            "role": "printed_structural_connector",
            "physical_part_id": part_name,
            "bom_group": part_name,
            "native_step": rel(step_p),
            "native_step_sha256": sha(step_p),
            "mesh": rel(npz_p),
            "mesh_sha256": sha(npz_p),
            "mesh_units": "mm",
            "bounds_mm": [m_mm.bounds[0].round(3).tolist(), m_mm.bounds[1].round(3).tolist()],
            "native_solid_count": 1,
            "native_validity": [True],
            "material": "Bambu PAHT-CF / PA12-CF (1.22 g/cm3) or PETG-CF (1.25 g/cm3) FDM",
            "mass_kg": round(p_rec["polymer_mass_PAHT_CF_g"] / 1000.0, 6),
            "mass_basis": "Exact native STEP volume * 1.22 g/cm3 (PAHT-CF)",
            "color": rgba,
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "parent_assembly": "manufactured_neck_head_assembly",
            "inspection_reference": False,
            "volume_mm3": p_rec["volume_mm3"],
            "print_orientation": p_orient,
        }
        _add_to_scene(part_name, m_mm, rgba, poly_entry, is_shell=False)

        # Combine stage sleeves into one STEP and one NPZ per stage for clean BOM & scene inspection
        slv_list = build_sum["sleeves_by_stage"][stg_key]
        slv_solids = []
        slv_meshes = []
        for s_info in slv_list:
            s_step = require(ROOT / s_info["step_path"])
            s_npz = require(ROOT / s_info["npz_path"])
            slv_solids.extend(import_step(str(s_step)).solids())
            slv_meshes.append(load_npz_trimesh(s_npz))
        compound_solids.extend(slv_solids)

        slv_comp_step = fam_dir / f"{bname}_sleeves.step"
        slv_comp_npz = fam_dir / f"{bname}_sleeves.npz"
        export_step(Compound(children=slv_solids), str(slv_comp_step))
        m_slv_all = trimesh.util.concatenate(slv_meshes)
        np.savez_compressed(
            slv_comp_npz,
            vertices=np.asarray(m_slv_all.vertices, dtype=np.float32),
            faces=np.asarray(m_slv_all.faces, dtype=np.int32),
        )

        slv_name = f"{part_name}_steel_sleeves"
        slv_entry = {
            "name": slv_name,
            "owning_link": owner,
            "role": "compression_limit_sleeves",
            "physical_part_id": slv_name,
            "bom_group": slv_name,
            "native_step": rel(slv_comp_step),
            "native_step_sha256": sha(slv_comp_step),
            "mesh": rel(slv_comp_npz),
            "mesh_sha256": sha(slv_comp_npz),
            "mesh_units": "mm",
            "bounds_mm": [m_slv_all.bounds[0].round(3).tolist(), m_slv_all.bounds[1].round(3).tolist()],
            "native_solid_count": len(slv_list),
            "native_validity": [True] * len(slv_list),
            "material": "304 Stainless Steel compression sleeves (7.85 g/cm3)",
            "mass_kg": round(p_rec["sleeve_mass_g"] / 1000.0, 6),
            "color": STAGE_COLORS["sleeve"],
            "display_group": "mechanism",
            "display_groups": ["mechanism"],
            "parent_assembly": "manufactured_neck_head_assembly",
            "inspection_reference": False,
        }
        _add_to_scene(slv_name, m_slv_all, STAGE_COLORS["sleeve"], slv_entry, is_shell=False)

        manifest_stages[stg_key] = {
            "stage": stg_key,
            "selected_family": sel_fam,
            "owning_link": owner,
            "volume_mm3": p_rec["volume_mm3"],
            "polymer_mass_paht_cf_g": p_rec["polymer_mass_PAHT_CF_g"],
            "sleeves_count": p_rec["sleeve_count"],
            "sleeves_mass_g": p_rec["sleeve_mass_g"],
            "total_stage_mass_paht_cf_g": p_rec["total_stage_mass_g"],
            "step": ref(step_p),
            "mesh": ref(npz_p),
            "sleeves_step": ref(slv_comp_step),
            "sleeves_mesh": ref(slv_comp_npz),
        }
        manifest_parts_out.append(manifest_stages[stg_key])

        bom_items.append({
            "id": bname,
            "name": f"{stg_key} FDM structural connector ({sel_fam})",
            "category": "Produced parts (FDM 3D printed)",
            "quantity": 1,
            "unit": "ea",
            "material": "Bambu PAHT-CF / PA12-CF (or PETG-CF)",
            "unit_mass_g": p_rec["polymer_mass_PAHT_CF_g"],
            "process": f"Home-lab FDM 3D print ({p_orient})",
            "notes": f"One-piece 3D-printed structural connector with {p_rec['sleeve_count']} pressed-in 304 SS compression sleeve seats and asymmetric 3D bevels (r_inner > r_outer).",
        })
        bom_items.append({
            "id": f"{bname}_sleeves",
            "name": f"{stg_key} 304 SS compression sleeve set ({p_rec['sleeve_count']} sleeves)",
            "category": "Fasteners & Hardware",
            "quantity": 1,
            "unit": "set",
            "material": "304 Stainless Steel",
            "unit_mass_g": p_rec["sleeve_mass_g"],
            "process": "Off-the-shelf 304 SS cylindrical sleeves pressed into printed counterbores",
            "notes": f"{p_rec['sleeve_count']} steel sleeves prevent polymer preload creep across XM335 motor cases, horns, and compute/camera standoffs.",
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

    step_4stage = fam_dir / "connectors-4stage.step"
    export_step(Compound(children=compound_solids), str(step_4stage))

    glb_path = fam_dir / "assembly-with-full-shell.glb"
    scene.export(str(glb_path))

    neck_joints = [
        {
            "name": "neck_yaw",
            "parent": "torso",
            "child": "neck_yaw",
            "origin_world_mm": P_NECK_YAW.tolist(),
            "axis_world_neutral": AXIS_NECK_YAW.tolist(),
            "range_deg": BOM_CONFIG.joint_range("neck_yaw"),
        },
        {
            "name": "neck_pitch",
            "parent": "neck_yaw",
            "child": "head",
            "origin_world_mm": P_NECK_PITCH.tolist(),
            "axis_world_neutral": AXIS_NECK_PITCH.tolist(),
            "range_deg": BOM_CONFIG.joint_range("neck_pitch"),
        },
    ]
    neck_controls = [
        {"name": "neck_yaw", "joint": "neck_yaw", "joints": ["neck_yaw"], "range_deg": BOM_CONFIG.joint_range("neck_yaw")},
        {"name": "neck_pitch", "joint": "neck_pitch", "joints": ["neck_pitch"], "range_deg": BOM_CONFIG.joint_range("neck_pitch")},
    ]

    write_json(
        fam_dir / "assembly-parts.json",
        {
            "family": fam,
            "root_link": "torso",
            "units": "CAD mm; GLB m",
            "parts": assembly_parts,
            "bom": {
                "scope": f"4-stage Blank-State 3D Generative FDM neck & head structural connectors ({fam}) and 34 pressed-in 304 SS compression sleeves",
                "completeness": "partial",
                "notes": "Replaces fixed_neck_ribbon_anchor_v21 (+ cap/pads/M2x8 hardware), upper__one_piece_bent_neck_L, compact_neck_U_strap_v23, single_shared_carrier_v23, and microphone_compact_tab_v23 (+ mic_foot_M2x3_0/1).",
                "exclusions": ["Retained XM335 servo motors, horns, ROCK4D SBC, stereo/wide cameras, microphone board, and exterior head/neck/torso shells"],
                "items": bom_items,
            },
            "joints": neck_joints,
            "controls": neck_controls,
        },
    )

    # Render 4-panel shell-fit and covers-off views
    renders_dir = fam_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1920, 780), "#f3f1eb")
    draw = ImageDraw.Draw(canvas)
    box_nh = [[-60.0, -45.0, 340.0], [60.0, 48.0, 500.0]]
    panels = [
        ("Neck & Head - Front (Shell On)", render_with_shell, [0.0, -1.0, 0.08]),
        ("Neck & Head - Rear/3Q (Shell On)", render_with_shell, [-0.55, 0.82, 0.14]),
        ("Neck & Head - Lateral (Shell On)", render_with_shell, [-1.0, -0.18, 0.08]),
        (f"Covers Off - {fam} 4-Stage Connectors & 34 Sleeves", render_no_shell, [-0.65, -0.72, 0.18]),
    ]
    for idx, (title, items_list, v_dir) in enumerate(panels):
        im = raster_meshes(items_list, v_dir, box=box_nh, W=480, H=720)
        canvas.paste(im, (idx * 480, 40))
        draw.text((idx * 480 + 14, 12), title, fill="#22282e")
    canvas.save(renders_dir / "shell-fit-views.png")

    return {
        "glb_path": glb_path,
        "step_path": step_4stage,
        "stage_meshes": stage_meshes,
    }


def render_comparison_charts(stage_meshes_by_fam: dict, fem_rep: dict):
    renders_dir = ART / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage-by-stage CAD comparison grid (`baseline_metal` vs `truss_box` vs `topopt` vs `combined`)
    canvas = Image.new("RGB", (1920, 1480), "#f3f1eb")
    draw = ImageDraw.Draw(canvas)
    stage_titles = [
        ("stage1_neck_root_cradle", "Stage 1: Neck Root Cradle (6 sleeves)", [-0.45, 0.85, 0.35]),
        ("stage2_neck_yaw_to_pitch_link", "Stage 2: Yaw-to-Pitch L-Link (8 sleeves)", [-0.72, -0.62, 0.30]),
        ("stage3_neck_pitch_to_head_cradle", "Stage 3: Pitch-to-Head Cradle (6 sleeves)", [0.75, -0.58, 0.32]),
        ("stage4_head_sensor_compute_carrier", "Stage 4: Sensor/Compute Carrier (14 sleeves)", [0.35, -0.88, 0.25]),
    ]

    # Row 0: Baseline parts
    for c_idx, (st_key, st_title, v_dir) in enumerate(stage_titles):
        b_sol = import_step(str(require(BASELINE_STEPS[st_key])))
        bv, bf = solid_to_watertight_arrays(b_sol, tol=0.05)
        bm = trimesh.Trimesh(vertices=bv, faces=bf, process=False)
        im = raster_meshes([(st_key, bm, [142, 150, 160, 255])], v_dir, W=480, H=330)
        canvas.paste(im, (c_idx * 480, 35))
        b_mass = fem_rep["stages"][st_key]["baseline_metal"]["materials"]["Al5052_H32"]["polymer_or_base_mass_g"]
        draw.text(
            (c_idx * 480 + 12, 10),
            f"{st_title} - Baseline ({b_mass:.2f} g)",
            fill="#22282e",
        )

    row_fams = [
        ("truss_box", "Family 1: Clean-Sheet Truss/Box"),
        ("topopt", "Family 2: Clean-Sheet 3D SIMP TopOpt"),
        ("combined", "Combined Optimal 4-Stage Neck & Head"),
    ]
    for r_idx, (fam_key, fam_label) in enumerate(row_fams, start=1):
        man = read_json(ART / fam_key / "manifest.json")
        for c_idx, (st_key, st_title, v_dir) in enumerate(stage_titles):
            m = stage_meshes_by_fam[fam_key][st_key]
            bname = STAGE_PART_BASENAMES[st_key]
            m_slv = load_npz_trimesh(ART / fam_key / f"{bname}_sleeves.npz")
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
    canvas.save(renders_dir / "stage-by-stage-cad.png")

    # 2. FEM stress, displacement, and strength margin chart (`fem-stress-displacement-comparison.png`)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160)
    stages = list(STAGE_PART_BASENAMES.keys())
    short_labels = ["S1: Root Cradle", "S2: Yaw->Pitch", "S3: Pitch->Head", "S4: Head Carrier"]
    x = np.arange(len(stages))
    w = 0.20

    fams_plot = [
        ("baseline_metal", "Baseline (Metal/PA12)", "#7a8694"),
        ("truss_box", "Family 1: Truss/Box (PAHT-CF)", "#2a8ab8"),
        ("topopt", "Family 2: SIMP TopOpt (PAHT-CF)", "#1ca48e"),
        ("combined", "Combined Optimal (PAHT-CF)", "#d48634"),
    ]
    for idx, (fk_name, label, col) in enumerate(fams_plot):
        u_vals = []
        p99_vals = []
        sf_vals = []
        for st in stages:
            m_key = "Al5052_H32" if fk_name == "baseline_metal" else "PAHT_CF_dry"
            row = fem_rep["stages"][st][fk_name]["materials"][m_key]["load_cases"]["adverse_neck_stall"]
            u_vals.append(row["max_displacement_mm"])
            p99_vals.append(row["p99_von_mises_MPa"])
            sf_vals.append(row["strength_ratio_inplane_p99"])
        axes[0].bar(x + (idx - 1.5) * w, u_vals, width=w, label=label, color=col)
        axes[1].bar(x + (idx - 1.5) * w, p99_vals, width=w, label=label, color=col)
        axes[2].bar(x + (idx - 1.5) * w, sf_vals, width=w, label=label, color=col)

    axes[0].set_title("Peak Deflection @ Dual-Axis Stall (mm)")
    axes[0].set_xticks(x, short_labels, rotation=15)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Von Mises p99 Stress @ Stall (MPa)")
    axes[1].set_xticks(x, short_labels, rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].set_title("In-Plane p99 Strength Ratio (SF)")
    axes[2].set_xticks(x, short_labels, rotation=15)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Cove-50 4-Stage Neck & Head FDM Connectors — Matched 3D Tetrahedral FEM (Dual-Axis Stall Wrench)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(renders_dir / "fem-stress-displacement-comparison.png")
    plt.close(fig)

    # 3. Weight-to-strength & 4-stage mass comparison chart (`weight-to-strength-chart.png`)
    base_audit = read_json(ART / "neck-head-baseline-audit.json")
    base_mass_g = float(base_audit["total_baseline_mass_g"])
    man_tb = read_json(ART / "truss_box" / "manifest.json")
    man_tp = read_json(ART / "topopt" / "manifest.json")
    man_cb = read_json(ART / "combined" / "manifest.json")

    def _rss_u(fk: str) -> float:
        m_key = "Al5052_H32" if fk == "baseline_metal" else "PAHT_CF_dry"
        return float(np.sqrt(sum(
            fem_rep["stages"][st][fk]["materials"][m_key]["load_cases"]["adverse_neck_stall"]["max_displacement_mm"] ** 2
            for st in stages
        )))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=160)
    names = ["Baseline\n(Legacy 14 Parts)", "Family 1\n(Truss/Box)", "Family 2\n(SIMP TopOpt)", "Combined\nOptimal"]
    cols = ["#7a8694", "#2a8ab8", "#1ca48e", "#d48634"]
    masses = [
        base_mass_g,
        man_tb["total_4stage_mass_paht_cf_g"],
        man_tp["total_4stage_mass_paht_cf_g"],
        man_cb["total_4stage_mass_paht_cf_g"],
    ]
    rss_u = [_rss_u("baseline_metal"), _rss_u("truss_box"), _rss_u("topopt"), _rss_u("combined")]

    axes[0].bar(names, masses, color=cols, width=0.52)
    axes[0].set_title("4-Stage Neck & Head Connector Mass Including 34 Sleeves (g)")
    axes[0].set_ylabel("Total 4-stage mass (g)")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(masses):
        axes[0].text(i, v + 0.8, f"{v:.2f} g", ha="center", fontweight="bold", fontsize=9)

    axes[1].bar(names, rss_u, color=cols, width=0.52)
    axes[1].set_title("4-Stage RSS Chain Deflection @ Dual-Axis Stall (mm)")
    axes[1].set_ylabel("RSS Displacement (mm)")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(rss_u):
        axes[1].text(i, v + 0.02, f"{v:.3f} mm", ha="center", fontweight="bold", fontsize=9)

    fig.suptitle("Cove-50 Neck & Head Architecture — Mass Savings (14 Legacy Parts -> 4 Monocoque Stages) & Chain Deflection", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(renders_dir / "weight-to-strength-chart.png")
    plt.close(fig)

    # 4. 3D SIMP Topology Optimization density projection plot (`topopt-density-and-loadpaths.png`)
    topopt_npz = np.load(require(ART / "neck-head-topopt-fields.npz"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=160)
    for idx, (prefix, title) in enumerate([
        ("stage1_neck_root_cradle", "Stage 1: Neck Root Cradle SIMP"),
        ("stage2_neck_yaw_to_pitch_link", "Stage 2: Yaw->Pitch L-Link SIMP"),
        ("stage3_neck_pitch_to_head_cradle", "Stage 3: Pitch->Head U-Cradle SIMP"),
        ("stage4_head_sensor_compute_carrier", "Stage 4: Head Carrier SIMP"),
    ]):
        rho = topopt_npz[f"{prefix}__rho"]
        proj = np.max(rho, axis=1).T
        im = axes[idx].imshow(proj, origin="lower", cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        axes[idx].set_title(title, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("X voxel index")
        axes[idx].set_ylabel("Z voxel index")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="SIMP Relative Density ρ")
    fig.suptitle("3D Multi-Load SIMP Topology Optimization Density Projections (Neck & Head Stages 1–4)", fontsize=12, fontweight="bold")
    fig.savefig(renders_dir / "topopt-density-and-loadpaths.png", bbox_inches="tight")
    plt.close(fig)


def main():
    build_sum = read_json(ART / "neck-head-build-summary.json")
    fem_rep = read_json(ART / "neck-head-fem-evaluation.json")
    stage_meshes_by_fam = {}
    for fam in ("truss_box", "topopt", "combined"):
        res = package_family(fam, build_sum, fem_rep)
        stage_meshes_by_fam[fam] = res["stage_meshes"]
        print(f"Packaged {fam}: {res['glb_path'].name} and {res['step_path'].name}")
    render_comparison_charts(stage_meshes_by_fam, fem_rep)
    print("Rendered all comparison figures in", ART / "renders")


if __name__ == "__main__":
    main()
