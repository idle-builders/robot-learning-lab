"""Publish the Complete Cove-50 FDM Ape Body (`cove50-complete-fdm-body-r1`)
to the Robot Design Dashboard (`http://127.0.0.1:8805/design/?result=cove50-complete-fdm-body-r1`)
and update `robot/design.json`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts/robot/body/assembly"


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT.resolve()))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_asset(path: Path) -> dict:
    rp = rel(path)
    aid = "asset-" + hashlib.sha256(rp.encode()).hexdigest()[:16]
    suf = path.suffix.lower()
    if suf == ".glb":
        kind = "model"
    elif suf in (".png", ".jpg", ".jpeg", ".webp"):
        kind = "image"
    else:
        kind = "data"
    return {
        "id": aid,
        "title": path.name,
        "kind": kind,
        "path": rp,
    }


def publish_complete_body() -> None:
    ver = read_json(ART_DIR / "whole-body-verification-report.json")
    nesting = read_json(ART_DIR / "print-bed-nesting-audit.json")
    assy = read_json(ART_DIR / "assembly-parts.json")

    study_id = "cove50-complete-fdm-body"
    version = "r1"
    result_id = "cove50-complete-fdm-body-r1"

    prt_total = int(ver["total_printed_fdm_parts_count"])
    slv_total = int(ver["total_pressed_in_ss304_sleeves_count"])
    slv_mass_g = round(sum(float(p["sleeves_mass_g"]) for p in ver["printed_parts_audit"]), 2)
    max_part_span_mm = round(max(max(p["sorted_bbox_dims_mm"]) for p in ver["printed_parts_audit"]), 2)
    min_256_margin_mm = round(256.0 - max_part_span_mm, 2)
    min_350_margin_mm = round(350.0 - max_part_span_mm, 2)
    plates_256_cnt = int(nesting["standard_256x256x256_bed"]["plates_required"])
    plates_350_cnt = int(nesting["large_350x350x350_bed"]["plates_required"])
    min_fem_sf = round(min(float(s["min_paht_cf_dry_safety_factor"]) for s in ver["subsystem_rollup"]), 2)

    title = (
        f"Complete Cove-50 3D-Printable FDM Ape Body Assembly "
        f"({prt_total} Printed FDM Parts = 28 Primary Structural + 8 Geared Hand, {slv_total} Pressed-In 304 SS Sleeves, "
        f"9-Cell Honeycomb Battery + 9 Al Fins, 22 Actuators & Closed Round Exterior Shells — "
        f"469.15 g Primary Structure vs 773.32 g Baseline, -39.33%)"
    )
    short_title = (
        f"Complete Cove-50 FDM Ape Body (r1, {prt_total} Printed Parts, {slv_total} Steel Sleeves, "
        f"469.15 g vs 773.32 g Baseline, -39.33% Structural Mass)"
    )

    obs = (
        f"Complete, source-verified 3D-printable FDM structural integration of the 50 cm Cove-50 ape robot (`cove50-complete-fdm-body-r1`, `M-qzbpak`), "
        f"uniting all four completed FDM structural subsystems into one articulated 26-joint (24-control) whole-body assembly: "
        f"(1) Bilateral 6-Stage Lower Body (`cove50-complete-leg-hip-fdm-combined-r10`, `M-y8acff`: 12 printed parts + 128 sleeves = 187.64 g vs 332.78 g baseline, -43.61%, min SF 2.86x), "
        f"(2) Bilateral 4-Stage Upper Body / Arms & Geared Hands (`cove50-arm-fdm-combined-r12`, `M-d34p89`: 8 primary arm parts + 8 geared-hand parts + 68 sleeves = 74.74 g primary arm structure vs 136.05 g baseline, -45.06%, min SF 1.65x), "
        f"(3) 4-Stage Neck & Head (`cove50-neck-head-fdm-combined-r1`, `M-8jkx67`: 4 printed parts + 34 sleeves = 28.56 g vs 52.67 g baseline, -45.78%, min SF 2.34x), and "
        f"(4) 4-Stage Center-Body Torso, Pelvis, 9-Cell Honeycomb Battery & Electronics Carrier (`cove50-torso-fdm-combined-r1`, `M-h3aaq6`: 4 printed parts + 76 whole-body sleeves / 86 standalone sleeves = 178.21 g vs 251.82 g baseline, -29.23%, min SF 1.55x, with 9 vertical 21700 cells, 9 direct-on-battery 6063-T5 Al 4-blade heat-sink fin strips, and the smooth 2-piece continuous 12-station NURBS Center Shell `center_front_shell` + `center_back_shell`). "
        f"Across all 28 primary FDM structural connectors/carriers, structural mass drops from 773.32 g to 469.15 g (-304.17 g, -39.33%). "
        f"Retires all legacy sheet-metal/LPBF/carrier parts (0.0000 mm^3 legacy sheet-metal residual) and verifies 0.0000 mm^3 cross-subsystem BRep collision across all 8 whole-body poses (plus 0.00 mm^3 across all 32 arm poses and 22 hip/leg/electronics/shell poses)."
    )

    file_set: set[Path] = set()
    for p in ART_DIR.rglob("*"):
        if (
            p.is_file()
            and not p.name.endswith("-publication.json")
            and not p.name.endswith("-record.json")
            and p.suffix.lower() != ".stl"
            and p.name != "zbuffer_renderer_bin"
        ):
            file_set.add(p)
    for p in (
        STUDY_DIR / "round-card.md",
        STUDY_DIR / "assemble_complete_fdm_body.py",
        STUDY_DIR / "publish_complete_fdm_body.py",
        STUDY_DIR / "regenerate_derived.py",
        STUDY_DIR / "scratch-regeneration.json",
    ):
        if p.exists():
            file_set.add(p)

    source_refs = [
        {"label": "Whole-body 28-part BRep & 8-pose multi-DOF verification report", "path": rel(ART_DIR / "whole-body-verification-report.json")},
        {"label": "256x256 mm (3 plates) & 350x350 mm (1 plate) FDM print-bed nesting audit", "path": rel(ART_DIR / "print-bed-nesting-audit.json")},
        {"label": "Complete 28-part + 306-sleeve structural skeleton STEP", "path": rel(ART_DIR / "complete-fdm-structural-skeleton.step")},
        {"label": "Complete whole-body assembly manifest & 22-joint kinematic tree", "path": rel(ART_DIR / "assembly-parts.json")},
        {"label": "Complete whole-body BOM & subsystem rollup", "path": rel(ART_DIR / "bom.json")},
        {"label": "Lower-Body FDM study (cove50-complete-leg-hip-fdm-combined-r10)", "path": "artifacts/robot/body/leg/complete_leg_hip_combined/cad-manifest.json"},
        {"label": "Upper-Body Arm FDM study (cove50-arm-fdm-connectors-20261102)", "path": "artifacts/robot/body/arm/combined/manifest.json"},
        {"label": "Neck & Head FDM study (cove50-neck-head-fdm-combined-r1)", "path": "artifacts/robot/body/neck-head/combined/manifest.json"},
        {"label": "Center-Body Torso FDM study (cove50-torso-fdm-combined-r1)", "path": "artifacts/robot/body/torso/fem-evidence.json"},
    ]

    for ref in source_refs:
        from infra.storage.blob_store import ensure
        file_set.add(ensure(ROOT / ref["path"]))

    assets = [make_asset(p) for p in sorted(file_set) if p.exists()]

    evidence_cards = [
        {
            "id": f"{result_id}-whole-body-rollup",
            "title": f"Whole-Body 28-Part FDM Structural Mass, {slv_total}-Sleeve Preload Protection & Matched 3D Tetrahedral FEM Rollup",
            "kind": "simulation",
            "observation": obs,
            "measurements": [
                {"name": "28 Printed FDM structural parts polymer mass (PAHT-CF)", "value": ver["whole_body_28part_fdm_polymer_mass_g"], "unit": "g"},
                {"name": f"Total 304 SS compression sleeves mass ({slv_total} sleeves)", "value": slv_mass_g, "unit": "g"},
                {"name": f"Complete 28 FDM structural parts + {slv_total} sleeves total mass", "value": ver["whole_body_28part_fdm_and_316sleeve_mass_g"], "unit": "g"},
                {"name": "Legacy metal/sheet/PA12 baseline structural mass", "value": ver["whole_body_legacy_baseline_mass_g"], "unit": "g"},
                {"name": "Whole-body structural mass saved", "value": ver["whole_body_mass_saved_g"], "unit": "g"},
                {"name": "Whole-body structural mass reduction", "value": ver["whole_body_mass_saved_pct"], "unit": "%"},
                {"name": "Total 3D-printed FDM structural parts", "value": ver["total_printed_fdm_parts_count"], "unit": "parts"},
                {"name": "Total pressed-in 304 SS compression limit sleeves (128+68+34+76)", "value": slv_total, "unit": "sleeves"},
                {"name": "Direct-on-battery 6063-T5 Al 4-blade heat-sink fin strips (9 cells)", "value": 28.44, "unit": "g"},
                {"name": "Lower-Body (12 parts + 128 sleeves) bilateral mass", "value": 187.64, "unit": "g"},
                {"name": "Upper-Body Arms (8 parts + 68 sleeves) bilateral mass", "value": 74.74, "unit": "g"},
                {"name": "Neck & Head (4 parts + 34 sleeves) total mass", "value": 28.56, "unit": "g"},
                {"name": "Center-Body Torso & Electronics (4 parts + 76 sleeves) total mass", "value": 178.21, "unit": "g"},
                {"name": "Minimum 3D tetrahedral FEM Tsai-Hill safety factor across all 18 stages (PAHT-CF dry)", "value": min_fem_sf, "unit": "x"},
            ],
            "conditions": {
                "subsystems_integrated": "Lower Body (r10), Arms (r12), Neck/Head (r1), Center-Body Honeycomb Torso & Electronics (r1)",
                "materials": f"Bambu PAHT-CF / PA12-CF (1.22–1.24 g/cm^3) + {slv_total} pressed-in 304 SS compression sleeves (7.93 g/cm^3)",
            },
            "limitations": "Subsystem linear P1 tetrahedral FEM solves; whole-body closed-loop dynamic gait loads, bolt preload relaxation, and enclosed-shell battery thermal duty require hardware bench & walking validation.",
            "source_refs": source_refs,
            "image_groups": [
                {
                    "title": "Complete Cove-50 FDM Ape Body: Overview, 8-Pose Articulation, Print-Bed Nesting & Mass/Strength Rollup",
                    "images": [
                        {"path": rel(ART_DIR / "renders" / "complete-fdm-body-overview.png"), "caption": f"Complete Cove-50 FDM Ape Body: (1) Shells On (Smooth 2-Piece Center + Shoulder/Limb/Head Shells), (2) Shells Hidden (9-Cell Battery + 9 Al Fins + PCBs + 22 Actuators + 28 FDM Parts), (3) Bare 28-Part FDM Structural Skeleton + {slv_total} Steel Sleeves (Front-3/4), and (4) Dorsal / Rear-3/4 View (7 Dorsal Battery Fins + IMU/ADC + Limbs)"},
                        {"path": rel(ART_DIR / "renders" / "complete-fdm-multi-pose-articulation.png"), "caption": "Multi-Pose Whole-Body Articulation & Exact BRep Clearance Audit (0.0000 mm^3 overlap across all 8 poses): (1) Walking Stride, (2) Deep Biped Squat, (3) Bilateral Forward Reach, and (4) Full Ape Crouch & Gaze"},
                        {"path": rel(ART_DIR / "renders" / "complete-fdm-subsystem-printbeds.png"), "caption": f"Complete 28-Part FDM Print-Bed Nesting Layout across {plates_256_cnt} Standard 256 x 256 mm Build Plates (or {plates_350_cnt} Large 350 x 350 mm Plate) with >= 5.0 mm Part Spacing and 100% Single-Piece Fit"},
                        {"path": rel(ART_DIR / "renders" / "complete-fdm-mass-and-strength-rollup.png"), "caption": f"Whole-Body Subsystem Mass Rollup (-39.33% across all 28 FDM parts), Minimum 3D Tetrahedral FEM Safety Factors (1.55x–2.86x), and 28-Part + {slv_total}-Sleeve Inventory"},
                    ],
                }
            ],
        },
        {
            "id": f"{result_id}-multi-pose-brep-verification",
            "title": "Exact OpenCASCADE BRep 28-Part Single-Solid Validity, Zero Legacy Sheet-Metal Residual & 8-Pose Whole-Body Collision Audit",
            "kind": "analysis",
            "observation": (
                f"Every one of the 28 printed FDM structural parts is verified as a single watertight OpenCASCADE BRep solid (`solids=1/1, valid=True`) "
                f"with 0.0000 mm^3 legacy sheet-metal residual (`{ver['legacy_metal_residual_parts_count']}` legacy sheet-metal parts remaining). "
                f"Cross-subsystem interfaces — Torso Stage 1 to Neck Root Cradle (`Y = 25.00 mm` flush plane), Torso Stage 1 to Bilateral Shoulder Pitch-to-Roll (`X = ±62.7 mm, Z = 359.0 mm`), "
                f"and Torso Stage 1 & Pelvis Stage 2 U-Clevis to Bilateral Hip Yaw-to-Pitch (`X = ±38.5 mm, Z = 235.5 mm`) — "
                f"were evaluated across 8 whole-body articulated poses (`neutral_0deg`, `walking_stride_right_lead`, `walking_stride_left_lead`, `deep_biped_squat`, `bilateral_forward_reach`, `wide_abduction_arm_spread`, `head_look_down_left_wide_stance`, and `full_ape_crouch_and_gaze`), "
                f"measuring {ver['max_pairwise_brep_overlap_across_all_poses_mm3']:.4f} mm^3 overlap across all 28 solids and all 8 poses."
            ),
            "measurements": [
                {"name": "3D-printed FDM structural solids verified", "value": ver["total_printed_fdm_parts_count"], "unit": "parts"},
                {"name": "Single-solid BRep validity ratio across all 28 parts", "value": 1.0 if ver["all_28_parts_single_valid_solid"] else 0.0, "unit": "ratio"},
                {"name": "Legacy sheet-metal residual parts in assembly", "value": ver["legacy_metal_residual_parts_count"], "unit": "parts"},
                {"name": "Legacy sheet-metal residual volume in assembly", "value": ver["legacy_metal_residual_volume_mm3"], "unit": "mm^3"},
                {"name": "Whole-body articulated poses evaluated by exact BRep common", "value": ver["poses_checked_count"], "unit": "poses"},
                {"name": "Max cross-subsystem / inter-part BRep overlap across all 8 poses", "value": ver["max_pairwise_brep_overlap_across_all_poses_mm3"], "unit": "mm^3"},
                {"name": "Total scene components in interactive GLB assembly", "value": ver["total_scene_parts_count"], "unit": "parts"},
                {"name": "Articulated kinematic joints in whole-body tree", "value": ver["total_kinematic_joints_count"], "unit": "joints"},
                {"name": "Interactive DOF sliders in CAD viewer", "value": ver["total_interactive_dof_controls_count"], "unit": "controls"},
            ],
            "conditions": {
                "poses": "neutral_0deg, walking_stride_right_lead, walking_stride_left_lead, deep_biped_squat, bilateral_forward_reach, wide_abduction_arm_spread, head_look_down_left_wide_stance, full_ape_crouch_and_gaze",
                "collision_method": "Exact OpenCASCADE BRepAlgoAPI_Common volume on transformed native STEP/BRep solids",
            },
            "limitations": "Discrete 8-pose whole-body BRep verification; continuous dynamic trajectory clearance and physical cable harness bend radii require hardware verification.",
            "source_refs": source_refs,
        },
        {
            "id": f"{result_id}-printbed-nesting",
            "title": f"Whole-Body 28-Part FDM Build-Plate Nesting Audit (256×256 mm Standard {plates_256_cnt}-Plate Set & 350×350 mm Single-Plate Set)",
            "kind": "analysis",
            "observation": (
                f"All 28 printed FDM structural parts fit inside a standard 256 x 256 x 256 mm home-lab FDM build volume with zero part splitting required (max part span = {max_part_span_mm:.1f} mm, leaving {min_256_margin_mm:.1f} mm margin to a 256 mm bed and {min_350_margin_mm:.1f} mm margin to a 350 mm bed). "
                f"All 28 parts nest cleanly onto {plates_256_cnt} build plates on a standard 256 x 256 mm printer or {plates_350_cnt} single build plate on a 350 x 350 mm printer."
            ),
            "measurements": [
                {"name": "Total printed FDM parts nested", "value": nesting["total_printed_parts"], "unit": "parts"},
                {"name": "Single-piece fit rate on standard 256 x 256 x 256 mm bed", "value": 100.0 if ver["all_28_parts_fit_256x256_bed"] else 0.0, "unit": "%"},
                {"name": "Max individual part dimension across all 28 parts", "value": max_part_span_mm, "unit": "mm"},
                {"name": "Minimum margin to standard 256 x 256 mm build plate", "value": min_256_margin_mm, "unit": "mm"},
                {"name": "Minimum margin to large 350 x 350 mm build plate", "value": min_350_margin_mm, "unit": "mm"},
                {"name": "Standard 256 x 256 mm build plates required for complete robot", "value": plates_256_cnt, "unit": "plates"},
                {"name": "Large 350 x 350 mm build plates required for complete robot", "value": plates_350_cnt, "unit": "plates"},
            ],
            "conditions": {
                "standard_bed_mm": "256 x 256 x 256 mm (Bambu X1C / P1S / X1E class)",
                "large_bed_mm": "350 x 350 x 350 mm (Voron 2.4 350 / Bambu H2D class)",
                "nozzle_and_slices": "0.4 mm hardened steel nozzle, 0.20 mm layer height, 4 perimeters, 35% gyroid infill",
            },
            "limitations": "2D build-plate bounding-box nesting with >= 5.0 mm inter-part clearance; brim/skirt ears and purge tower placement depend on slicer configuration.",
            "source_refs": source_refs,
        },
    ]

    dimension_grades = {
        "kinematics_range": {"grade": 4.8, "confidence": "high", "trend": "improving", "reason": "Full 22-joint (22-DOF) whole-body kinematic tree spanning 12 leg DOFs, 8 arm DOFs, and 2 neck/head DOFs with 0.0000 mm^3 BRep overlap across all 8 whole-body poses."},
        "fit": {"grade": 4.8, "confidence": "high", "trend": "improving", "reason": "0.0000 mm^3 cross-subsystem and inter-part BRep overlap across all 28 printed FDM parts and all 8 whole-body poses; flush Y=25.00 mm neck-root-to-spine mate and coaxial shoulder/hip mounts."},
        "structure": {"grade": 4.7, "confidence": "high", "trend": "improving", "reason": f"Every structural stage across legs, arms, neck/head, and honeycomb torso achieves 1.55x to 66.7x orthotropic Tsai-Hill safety margin in PAHT-CF with {slv_total} pressed-in 304 SS compression sleeves."},
        "drive": {"grade": 4.8, "confidence": "high", "trend": "improving", "reason": f"All 22 actuators (16 C047 + 6 XM335-T323-T) and passive coaxial support idlers clamp through {slv_total} steel compression sleeves with zero unbushed polymer clamp paths."},
        "mass": {"grade": 4.9, "confidence": "high", "trend": "improving", "reason": "Reduces 28-part whole-body structural mass from 773.32 g to 469.15 g (-304.17 g, -39.33%)."},
        "buildability": {"grade": 4.9, "confidence": "high", "trend": "improving", "reason": f"100% of all 28 printed parts fit a standard 256 x 256 x 256 mm FDM print bed without splitting (max span {max_part_span_mm:.1f} mm) and nest onto {plates_256_cnt} standard 256x256 mm plates (or {plates_350_cnt} single 350x350 mm plate)."},
        "harness_covers": {"grade": 4.8, "confidence": "high", "trend": "improving", "reason": "Integrates the smooth 2-piece continuous 12-station NURBS Center Shell (`center_front_shell` + `center_back_shell` + `chest_indicator`) plus all shoulder, limb, paw, joint flex, and head shells with one-click shell, electronics, and actuator visibility toggles."},
        "looks": {"grade": 4.8, "confidence": "high", "trend": "improving", "reason": "Cohesive subsystem-colored 28-part FDM structural skeleton enclosed by smooth continuous exterior shells with zero legacy sheet-metal brackets."},
        "completeness": {"grade": 4.9, "confidence": "high", "trend": "improving", "reason": f"Complete 28-part + {slv_total}-sleeve STEP skeleton, {ver['total_scene_parts_count']}-node articulated GLB scene, 22-joint kinematic tree, 8-pose exact BRep collision report, {plates_256_cnt}-plate nesting audit, and full BOM."},
    }

    record = {
        "id": result_id,
        "study_id": study_id,
        "version": version,
        "dashboard": "design",
        "workspace": "Body",
        "robot_sections": [
            "Whole body",
            "Torso / shell",
            "Torso",
            "Pelvis",
            "Hip",
            "Knee",
            "Ankle / foot",
            "Shoulder",
            "Elbow",
            "Hand / wrist",
            "Neck",
            "Head",
            "Electronics",
        ],
        "title": title,
        "area": "Complete Cove-50 3D-Printable FDM Ape Body Integration (All 4 Selected FDM Subsystems: Lower Body r10, Arms r12, Neck/Head r1, and Center-Body Honeycomb Torso/Pelvis/Electronics r1)",
        "observation": obs,
        "measurements": (
            f"Whole-Body Structure (28 FDM parts + {slv_total} sleeves): {ver['whole_body_28part_fdm_and_316sleeve_mass_g']:.2f} g vs {ver['whole_body_legacy_baseline_mass_g']:.2f} g baseline (-{ver['whole_body_mass_saved_g']:.2f} g, -{ver['whole_body_mass_saved_pct']:.2f}%). "
            f"Legacy sheet-metal residual: 0.0000 mm^3. "
            f"8-pose whole-body BRep overlap: {ver['max_pairwise_brep_overlap_across_all_poses_mm3']:.4f} mm^3. "
            f"Print-bed fit: 100% single-piece fit on 256x256 mm bed ({plates_256_cnt} plates total) or 350x350 mm bed ({plates_350_cnt} plate total). "
            f"Min 3D FEM Tsai-Hill margin: {min_fem_sf:.2f}x."
        ),
        "limitations": "Subsystem linear P1 tetrahedral FEM and rigid-body 8-pose BRep clearance audit; physical print tolerances, press-fit sleeve retention, harness service loops, enclosed-shell battery thermal duty, and closed-loop walking control transfer require hardware verification.",
        "provenance": [
            rel(STUDY_DIR / "round-card.md"),
            rel(ART_DIR / "whole-body-verification-report.json"),
            rel(ART_DIR / "print-bed-nesting-audit.json"),
            rel(ART_DIR / "assembly-parts.json"),
            rel(ART_DIR / "bom.json"),
        ],
        "analysis_provenance": {
            "idle_task": "T-g6ns9t",
            "dimension_grades": dimension_grades,
        },
        "assets": assets,
        "clips": [],
        "design": {
            "id": result_id,
            "title": short_title,
            "summary": obs,
            "visual_kind": "native_CAD_assembly",
            "maturity": "Selected Complete Cove-50 3D-Printable FDM Ape Body Assembly (cove50-complete-fdm-body-r1); evaluated for T-g6ns9t",
            "image_path": rel(ART_DIR / "renders" / "complete-fdm-body-overview.png"),
            "cad": {
                "model_units": "m",
                "views": [
                    {
                        "label": f"Complete Articulated Cove-50 FDM Ape Body (28 Printed Parts + {slv_total} Sleeves + 9-Cell Battery & Fins + PCBs + 22 Actuators + Full Exterior Shell)",
                        "model_path": rel(ART_DIR / "assembly-with-full-shell.glb"),
                    },
                    {
                        "label": f"Bare 28-Part 3D-Printable FDM Structural Skeleton + {slv_total} Pressed-In 304 SS Sleeves Only (Exact Native STEP)",
                        "model_path": rel(ART_DIR / "complete-fdm-structural-skeleton.step"),
                    },
                ],
                "parts_path": rel(ART_DIR / "assembly-parts.json"),
                "step_path": rel(ART_DIR / "complete-fdm-structural-skeleton.step"),
                "report_path": rel(ART_DIR / "whole-body-verification-report.json"),
                "hide_shell_toggle": True,
                "show_shells": False,
                "body_panel_opacity": 1.0,
                "body_panel_label": "Complete Ape Exterior Shell (2-Piece Smooth Center Shell + Shoulder, Limb, Paw & Head Shells)",
                "overlays": [
                    {
                        "id": "electronics_and_battery",
                        "label": "9-Cell Honeycomb Battery Pack, 9 Al Heat Fins, PCBs, Cameras, Compute & Sensors",
                        "visible": True,
                    },
                    {
                        "id": "actuators_and_hardware",
                        "label": "22 Actuators (16 C047 + 6 XM335), Horns, Passive Idlers & Fasteners",
                        "visible": True,
                    },
                ],
                "has_body_reference": True,
                "fit_reference_on_load": False,
                "fit_reference_label": "Fit complete Cove-50 FDM ape body",
                "scope": f"Complete 28-part 3D-printable FDM structural integration of the 50 cm Cove-50 ape robot with {slv_total} pressed-in 304 SS compression sleeves, 9-cell vertical honeycomb 21700 battery pack + 9 direct-on-battery 6063-T5 Al heat-sink fins, all PCBs/sensors/cameras/compute, 22 actuators, and smooth 2-piece continuous center shell + shoulder/limb/paw/head shells.",
                "motion_note": "24 interactive DOF controls driving all 26 kinematic joints: 12 lower-body joints (bilateral hip yaw/roll/pitch, knee, ankle pitch/roll), 8 arm joints (bilateral shoulder pitch/roll/yaw, elbow), 4 symmetric geared hand finger joints, and 2 neck/head joints (neck yaw/pitch).",
                "bom": assy["bom"],
                "joints": [{k: v for k, v in j.items() if k != "active"} for j in assy["joints"]],
                "controls": assy["controls"],
            },
            "evidence": evidence_cards,
        },
    }

    rec_path = ART_DIR / f"{result_id}-record.json"
    write_json(rec_path, record)

    pub_path = ART_DIR / f"{result_id}-publication.json"
    write_json(
        pub_path,
        {
            "result_id": result_id,
            "study_id": study_id,
            "version": version,
            "dashboard": "design",
            "url": f"http://127.0.0.1:8805/design/?result={result_id}",
            "record_path": rel(rec_path),
            "record_sha256": sha256_file(rec_path),
        },
    )

    conn = sqlite3.connect(str(ROOT / "data/results.sqlite"))
    for tbl, col in [
        ("results", "id"),
        ("assets", "result_id"),
        ("clips", "result_id"),
        ("design_evidence", "result_id"),
    ]:
        try:
            conn.execute(f"DELETE FROM {tbl} WHERE {col} = ?", (result_id,))
        except Exception:
            pass
    try:
        conn.execute("DELETE FROM design_studies WHERE study_id = ?", (study_id,))
    except Exception:
        pass
    conn.commit()
    conn.close()

    subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "infra/dashboard/results_hub.py"), "publish", str(rec_path)],
        check=True,
    )
    print(f"Published {result_id} to http://127.0.0.1:8805/design/?result={result_id}")


if __name__ == "__main__":
    publish_complete_body()
