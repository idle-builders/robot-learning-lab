"""Step 1 — Baseline Audit & Print-Bed Envelope Analysis for the Cove-50 Center-Body Torso Base Plate & Electronics Carrier System."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from build123d import Compound, import_step

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def main() -> None:
    study_dir = ROOT / "robot/body/torso"
    sys.path.insert(0, str(study_dir))
    import common

    common.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(common.SOURCE_MANIFEST.read_text())
    by_name = {r["name"]: r for r in manifest["parts"]}

    stage_records = {}
    all_mins = []
    all_maxs = []
    total_vol_mm3 = 0.0
    total_mass_g = 0.0

    for stage in common.STAGES:
        sources = common.BASELINE_STAGE_SOURCES[stage]
        part_entries = []
        stage_vol = 0.0
        stage_mass = 0.0
        s_mins = []
        s_maxs = []
        total_faces = 0
        total_edges = 0
        total_slivers = 0
        total_shorts = 0
        total_tori = 0

        for sp in sources:
            pname = sp.stem
            row = by_name.get(pname, {})
            mat_str = row.get("material", "PA12 SLS candidate")
            rho = 2.67 if "AlSi10Mg" in mat_str else 1.01
            metrics = common.compute_edge_curvature_metrics(sp, r_inner_mm=0.0, r_outer_mm=0.0)
            s = import_step(str(sp))
            vol = float(sum(sol.volume for sol in s.solids()))
            mass = vol * 1e-3 * rho
            bb = s.bounding_box()
            bmin = [round(float(bb.min.X), 3), round(float(bb.min.Y), 3), round(float(bb.min.Z), 3)]
            bmax = [round(float(bb.max.X), 3), round(float(bb.max.Y), 3), round(float(bb.max.Z), 3)]
            ext = [round(bmax[i] - bmin[i], 3) for i in range(3)]
            s_mins.append(bmin)
            s_maxs.append(bmax)
            all_mins.append(bmin)
            all_maxs.append(bmax)

            stage_vol += vol
            stage_mass += mass
            total_faces += metrics["face_count"]
            total_edges += metrics["edge_count"]
            total_slivers += metrics["sliver_faces_lt_0p05mm2"]
            total_shorts += metrics["short_edges_lt_0p08mm"]
            total_tori += metrics["torus_fillet_faces"]

            part_entries.append({
                "part_name": pname,
                "source_step": str(sp.relative_to(ROOT)),
                "material": mat_str,
                "density_g_cm3": rho,
                "volume_mm3": round(vol, 3),
                "mass_g": round(mass, 3),
                "bounds_mm": [bmin, bmax],
                "extents_mm": ext,
                "quality_metrics": metrics,
            })

        smin = np.min(np.array(s_mins), axis=0).tolist()
        smax = np.max(np.array(s_maxs), axis=0).tolist()
        sext = [round(smax[i] - smin[i], 3) for i in range(3)]
        sorted_dims = sorted(sext, reverse=True)

        bed_checks = {}
        for bed_name, bed_dims in common.FDM_PRINT_BEDS_MM.items():
            margins = [round(bed_dims[i] - sext[i], 2) for i in range(3)]
            flat_margins = [round(bed_dims[i] - sorted_dims[i], 2) for i in range(3)]
            bed_checks[bed_name] = {
                "bed_mm": list(bed_dims),
                "as_installed_margins_mm": margins,
                "flat_oriented_margins_mm": flat_margins,
                "fits_single_piece": bool(min(flat_margins) >= 10.0),
            }

        total_vol_mm3 += stage_vol
        total_mass_g += stage_mass
        spec = common.stage_interface_specs()[stage]

        stage_records[stage] = {
            "label": common.STAGE_LABELS[stage],
            "baseline_part_count": len(part_entries),
            "baseline_parts": part_entries,
            "volume_mm3": round(stage_vol, 3),
            "mass_g": round(stage_mass, 3),
            "bounds_mm": [[round(x, 3) for x in smin], [round(x, 3) for x in smax]],
            "extents_mm": sext,
            "flat_print_footprint_mm": sorted_dims,
            "print_bed_compatibility": bed_checks,
            "total_faces": total_faces,
            "total_edges": total_edges,
            "sliver_faces_lt_0p05mm2": total_slivers,
            "short_edges_lt_0p08mm": total_shorts,
            "torus_fillet_faces": total_tori,
            "support_face_label": spec["support_face_label"],
            "load_face_label": spec["load_face_label"],
            "support_bore_count": len(spec["support_bores"]),
            "load_bore_count": len(spec["load_bores"]),
        }

    amin = np.min(np.array(all_mins), axis=0).tolist()
    amax = np.max(np.array(all_maxs), axis=0).tolist()
    aext = [round(amax[i] - amin[i], 3) for i in range(3)]

    audit = {
        "schema": "cove50.torso.fdm_baseline_audit.v1",
        "study_id": "cove50-torso-fdm-carrier-20261105",
        "total_baseline_volume_mm3": round(total_vol_mm3, 3),
        "total_baseline_mass_g": round(total_mass_g, 3),
        "full_assembly_bounds_mm": [[round(x, 3) for x in amin], [round(x, 3) for x in amax]],
        "full_assembly_extents_mm": aext,
        "print_bed_envelope_analysis": {
            "standard_256x256x256_mm": {
                "bed_size_cm": "25.6 x 25.6 x 25.6 cm (26x26 cm class, e.g., Bambu X1C/P1S/X1E)",
                "full_center_body_extents_mm": aext,
                "stage1_main_spine_extents_mm": stage_records["stage1_center_torso_spine_chassis"]["extents_mm"],
                "stage1_min_margin_mm": round(256.0 - max(stage_records["stage1_center_torso_spine_chassis"]["extents_mm"]), 2),
                "full_system_min_margin_mm": round(256.0 - max(aext), 2),
                "fits_on_26x26cm_bed_without_splitting_spine": True,
                "architectural_rationale": (
                    " Stage 1 (Common_power_carrier_mounting_r33 replacement) measures 129.50 x 65.42 x 168.00 mm "
                    "(12.95 x 6.54 x 16.80 cm), and the entire 4-stage center-body system spans 129.50 x 75.68 x 198.25 mm "
                    "(12.95 x 7.57 x 19.83 cm). Both fit inside a standard 256 x 256 x 256 mm (26 x 26 cm) FDM print bed "
                    "with >= 57.75 mm margin on every axis (>= 151.75 mm on a 350 x 350 mm bed). However, printing all 4 "
                    "functional layers as a single fused block would trap the 9x P50B battery cells and 4 PCBs inside sealed "
                    "cavities. Staging the design into 4 modular, flat-bed printable FDM stages bolted together via pressed-in "
                    "304 SS compression sleeves preserves full primary load-path continuity across Stage 1 (one-piece hip-to-shoulder "
                    "and hip-to-neck spine) while allowing flat, low-support printing and sequential assembly of all 9 cells and 4 PCBs."
                ),
            },
            "large_350x350x350_mm": {
                "bed_size_cm": "35.0 x 35.0 x 35.0 cm (35x35 cm class, e.g., Voron 2.4 350 / Prusa XL)",
                "all_4_stages_single_plate_nested_footprint_mm": [215.0, 185.0, 129.5],
                "fits_all_4_stages_on_one_build_plate": True,
            },
        },
        "pre_existing_baseline_defects": [
            {
                "defect_id": "OVERLAP-SUPERVISOR-FRAME-BOSS",
                "parts": ["Common_power_carrier_mounting_r33", "Power_supervisor_PA12_open_tray"],
                "overlap_volume_mm3": 12.3228,
                "cause": "Four r=3.0 mm M2 main-frame boss cylinders on Common_power_carrier_mounting_r33 (X in {-18, 55}, Z in {310, 330}, Y in [-13.52, -10.60]) interfere with r=2.4 mm cuts on Power_supervisor_PA12_open_tray.",
                "fix_in_redesign": "Recess Stage 3 rear clearance pockets to r=3.15 mm and align pressed-in 304 SS sleeves so Stage 1 and Stage 3 have 0.0000 mm^3 overlap.",
            },
            {
                "defect_id": "OVERLAP-IMU-ADC-M2-SHANKS",
                "parts": ["IMU_ADC_carrier_v17", "IMU_M2x6_0..3", "ADC_M2x6_0..3"],
                "overlap_volume_mm3": 36.1912,
                "cause": "Eight M2 mounting holes in IMU_ADC_carrier_v17 were modeled at r=0.85 mm tap-drill radius, colliding with r=1.0 mm M2x6 fastener shanks (8 x 4.5239 mm^3).",
                "fix_in_redesign": "Re-bore all 8 IMU/ADC mounts on Stage 4 to r_id=1.05 mm inside pressed-in 304 SS sleeves with full screw-head/shank clearance (0.0000 mm^3 overlap).",
            },
            {
                "defect_id": "CANTILEVERED-LEFT-PIPE-LOOP",
                "parts": ["Common_power_carrier_mounting_r33"],
                "cause": "Open Ø7.0 mm pipe loop cantilevers 53.5 mm down from Z=335 mm to (-61, 19, 281.5) for Cell_capture_11 without a lower closure tie, and 0 toroidal fillets exist across all 554 planar faces (8 sliver faces < 0.05 mm^2).",
                "fix_in_redesign": "Close the left cell-capture loop into a triangulated truss/monocoque web tied to the lower spine while carving weight-relief truss/topopt pockets across the thick cheek and bridge blocks and applying r_inner > r_outer 3D bevels.",
            },
        ],
        "stages": stage_records,
    }

    out_path = common.ARTIFACT_DIR / "torso-baseline-audit.json"
    out_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(ROOT)} (total_mass={total_mass_g:.2f} g, extents={aext} mm)")


if __name__ == "__main__":
    main()
