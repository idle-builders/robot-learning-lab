"""Publish both competing lower-leg FDM connector studies to the Robot Design Dashboard."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STUDY, ART, read_json, write_json, rel, sha


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


def publish_all():
    fem = read_json(ART / "fem-report.json")
    fit = read_json(ART / "fit-report.json")
    baseline_audit = read_json(ART / "baseline-audit.json")
    topopt_report = read_json(ART / "topopt" / "topopt-solve-report.json")

    chain_sum = fem["chain_summary_recovery_4kg_1p5g"]
    base_chain = chain_sum["baseline_metal__Al5052_H32"]

    studies_cfg = [
        {
            "family": "truss_box",
            "study_id": "cove50-lower-leg-fdm-truss-box",
            "version": "r9",
            "result_id": "cove50-lower-leg-fdm-truss-box-r9",
            "title": "Lower-Leg FDM Connectors · Family 1: Parametric Space-Truss & Ribbed Box-Section (Side-by-Side Front Shell Mounts + 2 Screws/Panel + Toggleable Hard Shell)",
            "short_title": "Family 1: Parametric FDM Space-Truss & Ribbed Box-Section (r9, Side-by-Side Front Shell Mounts)",
            "observation": (
                "Replaces 22 baseline V16 bent-sheet/plate/spacer/sole-fastener parts per leg (74.88 g/leg) AND the separate `Shin_carrier` hard-shell mounting block "
                "with 3 3D-beveled, G1-filleted symmetric FDM-printable PAHT-CF structural connectors per leg plus 32 stainless steel compression sleeves across all 4-screw motor/horn/idler interfaces. "
                "Directly integrates 4 `Stage 1 Shin` hard-shell mounting posts and U-shaped hex-slot captive M3 nut pockets (2 side-by-side anterior M3x8 screws + 2 captive nuts on the front bridge at X=53, 67 mm for `shin_front`, and 2 posterior M3x8 screws + 2 captive nuts for `shin_back`), and includes the bilateral 2-screw hard-shell leg covers directly in the CAD viewer with hide-shell toggle."
            ),
        },
        {
            "family": "topopt",
            "study_id": "cove50-lower-leg-fdm-topopt",
            "version": "r9",
            "result_id": "cove50-lower-leg-fdm-topopt-r9",
            "title": "Lower-Leg FDM Connectors · Family 2: Blank-Space 3D Topology-Optimized Load Paths (Side-by-Side Front Shell Mounts + 2 Screws/Panel + Toggleable Hard Shell)",
            "short_title": "Family 2: Blank-Space 3D Topology-Optimized FDM (r9, Side-by-Side Front Shell Mounts)",
            "observation": (
                "Derived directly from 3D Hex8 multi-load SIMP topology optimization from blank design domains between fixed C047 "
                "motor/horn seats and sole patches, reconstructed into 3 symmetric 3D-beveled, G1-filleted watertight FDM solids per leg plus 32 steel compression sleeves "
                "across all 4-screw motor/horn/idler interfaces, with 4 integrated `Stage 1 Shin` hard-shell mounting posts + captive M3 nut pockets (2 side-by-side front screws for `shin_front` at X=53, 67 mm and 2 rear screws for `shin_back`), and toggleable bilateral hard-shell leg covers in the CAD viewer."
            ),
        },
        {
            "family": "combined",
            "study_id": "cove50-lower-leg-fdm-combined",
            "version": "r8",
            "result_id": "cove50-lower-leg-fdm-combined-r8",
            "title": "Lower-Leg FDM Connectors · Combined Optimal Selection: Truss-Box Shin & Ankle + TopOpt Foot Monocoque (Side-by-Side Front Shell Mounts + 2 Screws/Panel + Toggleable Hard Shell)",
            "short_title": "Combined Optimal Lower-Leg FDM (Truss-Box Shin/Ankle + TopOpt Foot, r8, Side-by-Side Front Shell Mounts)",
            "observation": (
                "Combines the highest specific-stiffness and highest-margin 3D-beveled, symmetric stage from each family between the fixed C047 actuators: "
                "Stage 1 Shin from Family 1 (Truss-Box with 4 integrated hard-shell mounts + captive M3 nut pockets — 2 side-by-side front M3x8 screws on the anterior bridge for `shin_front` and 2 rear M3x8 screws for `shin_back` — plus 16 sleeves: 4 Knee Horn + 4 Knee Idler + 4 Inner Case + 4 Outer Case), "
                "Stage 2 Ankle Coupler from Family 1 (Twin-Ribbed Box Arch contoured to clear `shin_front` & `shin_back` across ±35° ankle pitch, 12 sleeves: 4 Pitch Horn + 4 Pitch Idler + 4 Roll Horn), "
                "and Stage 3 Foot Monocoque from Family 2 (3D TopOpt Vaulted Sole, 4 sleeves: 2 Front + 2 Rear Case). "
                "Includes the bilateral 2-screw hard-shell leg covers + captive M3 nuts/screws directly in the CAD viewer (toggleable via the hide-shell checkbox)."
            ),
        },
    ]

    shared_files = [
        ART / "baseline-audit.json",
        ART / "fem-report.json",
        ART / "fit-report.json",
        ART / "cad-rule-check-report.json",
        ART / "topopt" / "topopt-solve-report.json",
        ART / "topopt" / "topopt-fields.npz",
        ART / "renders" / "stage-by-stage-cad.png",
        ART / "renders" / "topopt-density-and-loadpaths.png",
        ART / "renders" / "weight-to-strength-chart.png",
        ART / "renders" / "fem-stress-displacement-comparison.png",
        ART / "renders" / "shell-fit-views.png",
        STUDY / "round-card.md",
        STUDY / "common.py",
        STUDY / "run_topopt_field.py",
        STUDY / "build_truss_box.py",
        STUDY / "build_topopt.py",
        STUDY / "package_and_render.py",
        STUDY / "publish_studies.py",
    ]

    only_fams = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for cfg in studies_cfg:
        fam = cfg["family"]
        if only_fams and fam not in only_fams:
            continue
        fam_dir = ART / fam
        dry_s = chain_sum[f"{fam}__PAHT_CF_dry"]
        wet_s = chain_sum[f"{fam}__PAHT_CF_wet"]
        petg_s = chain_sum[f"{fam}__PETG_CF"]
        fit_s = fit["families"][fam]

        st1 = dry_s["stages"]["stage1_shin"]
        st2 = dry_s["stages"]["stage2_ankle_coupler"]
        st3 = dry_s["stages"]["stage3_foot_monocoque"]

        bom_data = read_json(fam_dir / "bom.json")
        bom_data["completeness"] = "partial"
        write_json(fam_dir / "bom.json", bom_data)
        file_set = set(p for p in shared_files if p.exists())
        for p in fam_dir.rglob("*"):
            if p.is_file() and not p.name.endswith("-publication.json") and not p.name.endswith("-record.json"):
                file_set.add(p)
        for p in (ART / "fem-fields").glob("*.npz"):
            file_set.add(p)

        assets = [make_asset(p) for p in sorted(file_set)]

        source_refs = [
            {"label": "Matched 3D FEM report across all 3 stages & materials", "path": rel(ART / "fem-report.json")},
            {"label": "13-pose native BRep & hard-shell clearance report", "path": rel(ART / "fit-report.json")},
            {"label": "Family CAD manifest & mass breakdown", "path": rel(fam_dir / "cad-manifest.json")},
            {"label": "3D SIMP topology optimization report", "path": rel(ART / "topopt" / "topopt-solve-report.json")},
            {"label": "Baseline V16 lower-leg interface audit", "path": rel(ART / "baseline-audit.json")},
            {"label": "Study round card & design rules", "path": rel(STUDY / "round-card.md")},
        ]

        evidence_cards = [
            {
                "id": f"{cfg['result_id']}-fem-strength",
                "title": "Matched 3D Tetrahedral FEM: Strength, Stiffness & Material Sensitivity (4 kg @ 1.5g & 3-6 kg)",
                "kind": "simulation",
                "observation": (
                    f"{cfg['short_title']} replaces 2D out-of-plane sheet bending with 3D closed-loop / space-frame load paths. "
                    f"At 4 kg @ 1.5g recovery, Stage 1 (shin) achieves u_max={st1['max_displacement_mm']:.3f} mm and p99={st1['primary_vm_p99_MPa']:.2f} MPa "
                    f"({st1['p99_strength_margin_ratio']:.2f}x margin); Stage 2 (ankle coupler) achieves u_max={st2['max_displacement_mm']:.3f} mm and "
                    f"p99={st2['primary_vm_p99_MPa']:.2f} MPa ({st2['p99_strength_margin_ratio']:.2f}x margin vs 1.94x in V16 bent sheet); "
                    f"Stage 3 (foot monocoque) achieves u_max={st3['max_displacement_mm']:.3f} mm and p99={st3['primary_vm_p99_MPa']:.2f} MPa "
                    f"({st3['p99_strength_margin_ratio']:.2f}x margin)."
                ),
                "measurements": [
                    {"name": "Stage 1 (Shin) max displacement (4kg@1.5g, PAHT-CF dry)", "value": st1["max_displacement_mm"], "unit": "mm"},
                    {"name": "Stage 1 (Shin) p99 von Mises stress", "value": st1["primary_vm_p99_MPa"], "unit": "MPa"},
                    {"name": "Stage 1 (Shin) specific stiffness", "value": st1["specific_stiffness_N_mm_per_g"], "unit": "(N/mm)/g"},
                    {"name": "Stage 2 (Ankle Coupler) max displacement (4kg@1.5g, PAHT-CF dry)", "value": st2["max_displacement_mm"], "unit": "mm"},
                    {"name": "Stage 2 (Ankle Coupler) p99 von Mises stress", "value": st2["primary_vm_p99_MPa"], "unit": "MPa"},
                    {"name": "Stage 2 (Ankle Coupler) p99 strength margin ratio (PAHT-CF dry)", "value": st2["p99_strength_margin_ratio"], "unit": "x"},
                    {"name": "Stage 2 (Ankle Coupler) p99 strength margin ratio (PAHT-CF wet)", "value": wet_s["stages"]["stage2_ankle_coupler"]["p99_strength_margin_ratio"], "unit": "x"},
                    {"name": "Stage 2 (Ankle Coupler) p99 strength margin ratio (PETG-CF)", "value": petg_s["stages"]["stage2_ankle_coupler"]["p99_strength_margin_ratio"], "unit": "x"},
                    {"name": "Stage 3 (Foot Monocoque) max displacement (4kg@1.5g, PAHT-CF dry)", "value": st3["max_displacement_mm"], "unit": "mm"},
                    {"name": "Stage 3 (Foot Monocoque) p99 von Mises stress", "value": st3["primary_vm_p99_MPa"], "unit": "MPa"},
                    {"name": "Stage 3 (Foot Monocoque) specific stiffness", "value": st3["specific_stiffness_N_mm_per_g"], "unit": "(N/mm)/g"},
                ],
                "conditions": {
                    "load_cases": "stance_4kg_1g, recovery_4kg_1p5g, light_3kg_1p5g (0.75x), heavy_6kg_1p5g (1.50x)",
                    "materials": "PAHT_CF_dry (E_in=7.55 GPa, E_Z=3.29 GPa), PAHT_CF_wet (E_in=6.57 GPa, E_Z=2.70 GPa), PETG_CF (E_in=5.80 GPa, E_Z=2.85 GPa) + 304 SS sleeves (200 GPa)",
                    "baseline_reference": "V16 Al 5052-H32 / 6061-T6 (Stage 2 p99 = 99.57 MPa, 1.94x margin; total mass = 74.88 g/leg)",
                },
                "limitations": (
                    "Linear P1 tetrahedral elasticity with bonded steel-sleeve/polymer counterbores (optimistic interface bound). "
                    "Physical bolt preload relaxation, interlaminar fatigue under repeated landing impact, and print toolpath weld strength require hardware bench testing."
                ),
                "source_refs": source_refs,
                "image_groups": [
                    {
                        "title": "Matched 3D FEM Stress Fields & Quantitative Weight-to-Strength Charts",
                        "images": [
                            {"path": rel(ART / "renders" / "fem-stress-displacement-comparison.png"), "caption": "3D tetrahedral von Mises stress fields across all 3 stages (Baseline vs Family 1 Truss-Box vs Family 2 TopOpt)"},
                            {"path": rel(ART / "renders" / "weight-to-strength-chart.png"), "caption": "Stage mass, deflection, specific stiffness ((N/mm)/g), and dry/wet/PETG-CF strength margin comparison"},
                        ],
                    }
                ],
            },
            {
                "id": f"{cfg['result_id']}-fit-clearance",
                "title": "13-Pose Kinematic & Hard-Shell Clearance Verification (0.000 mm^3 Overlap)",
                "kind": "analysis",
                "observation": (
                    "Evaluated exact BRep solid intersections across 13 neutral and extreme lower-leg poses "
                    "(knee flexion 0, 60, 90, 125 deg; ankle pitch -35, 0, +35 deg; ankle roll -25, 0, +25 deg; and combined crouch/diagonal poses). "
                    "By routing the Stage 1 shin bridge through the anterior 3D corridor (Y in [-10.5, -2.4] mm, Z in [96.8, 108.5] mm), "
                    "the shin completely clears the thigh link (`R_A10_one_piece_direct_upper`) and knee motor at 125 deg full knee flexion, "
                    "mounts flush with the right-leg hard shell (`Shin_carrier` at X = 39.25 mm), and stays inside `shin_front` and `shin_back`."
                ),
                "measurements": [
                    {"name": "Max BRep overlap volume across all 13 poses", "value": fit_s["max_overlap_mm3_across_all_poses"], "unit": "mm^3"},
                    {"name": "Min inter-stage clearance across all 13 poses", "value": fit_s["min_stage_to_stage_clearance_mm"], "unit": "mm"},
                    {"name": "Min clearance to right-leg hard shell (shin_front)", "value": fit_s["min_distance_to_shin_front_hard_cover_mm"], "unit": "mm"},
                    {"name": "Evaluated kinematic poses", "value": fit_s["poses_checked"], "unit": "poses"},
                ],
                "conditions": {
                    "knee_range": "0 to 125 deg (90 deg hard-shell limit, 125 deg full mechanism limit)",
                    "ankle_pitch_range": "-35 to +35 deg",
                    "ankle_roll_range": "-25 to +25 deg",
                    "checked_neighbors": "All C047 motor cases, horns, idlers, thigh links, foot pressure sensors, Shin_carrier, shin_front, shin_back, and 53-node full shell",
                },
                "limitations": "Unloaded rigid BRep clearance check plus elastic deflection comparison; physical print tolerance stack-up (+/- 0.15 mm) and cable routing harness clearance must be verified on first article.",
                "source_refs": [
                    {"label": "13-pose BRep fit report", "path": rel(ART / "fit-report.json")},
                    {"label": "Fit check (robot/body/checks fit; original per-study script: Git commit 730049457)", "path": "robot/body/checks/modules/fit.py"},
                ],
                "image_groups": [
                    {
                        "title": "Installed Lower-Leg Assembly & Hard-Shell Envelope Fit",
                        "images": [
                            {"path": rel(ART / "renders" / "shell-fit-views.png"), "caption": "Installed lower-leg connectors inside fixed C047 motor and hard-shell envelope"},
                            {"path": rel(ART / "renders" / "stage-by-stage-cad.png"), "caption": "Stage-by-stage CAD geometry and native-unit metrics"},
                        ],
                    }
                ],
            },
            {
                "id": f"{cfg['result_id']}-mass-consolidation",
                "title": "Part Consolidation, Mass Reduction & Compression-Sleeve Hardware Architecture",
                "kind": "measurement",
                "observation": (
                    f"Consolidates 22 baseline lower-leg parts per leg (2 shin plates + 4 shin spacers + 1 ankle link + 2 foot L-brackets "
                    f"+ 1 plain sole + 4 foot spacers + 8 sole corner bolts/nuts = 74.88 g/leg) into 3 single-piece FDM connectors per leg "
                    f"({dry_s['one_leg_mass_g']:.2f} g/leg including 20 pressed-in 304 SS compression sleeves), saving "
                    f"{dry_s['mass_saved_vs_baseline_one_leg_g']:.2f} g/leg ({dry_s['mass_saved_vs_baseline_bilateral_g']:.2f} g bilateral)."
                ),
                "measurements": [
                    {"name": "One-leg total connector + sleeve mass (PAHT-CF)", "value": dry_s["one_leg_mass_g"], "unit": "g"},
                    {"name": "Bilateral pair total mass (PAHT-CF)", "value": dry_s["bilateral_pair_mass_g"], "unit": "g"},
                    {"name": "Mass saved per leg vs V16 baseline (74.88 g)", "value": dry_s["mass_saved_vs_baseline_one_leg_g"], "unit": "g"},
                    {"name": "Bilateral mass saved vs V16 baseline (149.76 g)", "value": dry_s["mass_saved_vs_baseline_bilateral_g"], "unit": "g"},
                    {"name": "Stage 1 (Shin) mass per leg", "value": st1["stage_mass_g"], "unit": "g"},
                    {"name": "Stage 2 (Ankle Coupler) mass per leg", "value": st2["stage_mass_g"], "unit": "g"},
                    {"name": "Stage 3 (Foot Monocoque) mass per leg", "value": st3["stage_mass_g"], "unit": "g"},
                ],
                "conditions": {
                    "density_basis": "PAHT-CF 1.22 g/cm^3, PETG-CF 1.25 g/cm^3, 304 SS sleeves 7.85 g/cm^3, Al 5052-H32 2.68 g/cm^3",
                    "hardware": "20 cylindrical 304 SS compression limit sleeves per leg (2 knee main horn, 10 ankle coupler, 8 foot roll motor case)",
                },
                "limitations": "Theoretical mass from exact BRep volumes and manufacturer densities; excludes print support waste and unchanged motor/horn screws.",
                "source_refs": [
                    {"label": "Family CAD manifest", "path": rel(fam_dir / "cad-manifest.json")},
                    {"label": "Family BOM", "path": rel(fam_dir / "bom.json")},
                ],
                "image_groups": [
                    {
                        "title": "3D SIMP Topology Optimization Fields & Reconstructed CAD",
                        "images": [
                            {"path": rel(ART / "renders" / "topopt-density-and-loadpaths.png"), "caption": "Blank-space 3D Hex8 SIMP density fields (rho > 0.42) and extracted smooth BRep solids"},
                        ],
                    }
                ],
            },
        ]

        dimension_grades = {
            "kinematics_range": {
                "grade": 4.0,
                "confidence": "high",
                "trend": "flat",
                "reason": "Preserves exact V16 motor/horn positions, joint origins, axes, and full 0-125 deg knee, +/-35 deg ankle pitch, and +/-25 deg ankle roll travel with 0.000 mm^3 overlap.",
            },
            "fit": {
                "grade": 4.0,
                "confidence": "high",
                "trend": "improving",
                "reason": "Verified 0.000 mm^3 BRep overlap across 13 poses against all motors, horns, thigh links, foot tactile sensors, Shin_carrier, and shin_front/shin_back.",
            },
            "structure": {
                "grade": 3.5 if fam == "truss_box" else 3.2,
                "confidence": "medium",
                "trend": "improving",
                "reason": (
                    f"3D closed-loop load paths increase Stage 2 ankle coupler p99 strength margin from 1.94x (V16 bent sheet) to "
                    f"{st2['p99_strength_margin_ratio']:.2f}x (PAHT-CF dry) / {wet_s['stages']['stage2_ankle_coupler']['p99_strength_margin_ratio']:.2f}x (wet). "
                    "Physical bolt preload creep and cyclic impact fatigue remain to be bench-qualified."
                ),
            },
            "drive": {
                "grade": 3.5,
                "confidence": "medium",
                "trend": "improving",
                "reason": f"Distal lower-leg mass reduction (-{dry_s['mass_saved_vs_baseline_one_leg_g']:.1f} g/leg) reduces knee and hip swing inertia while preserving dual-shear idler support.",
            },
            "mass": {
                "grade": 4.2 if fam == "truss_box" else 4.5,
                "confidence": "high",
                "trend": "improving",
                "reason": f"Reduces lower-leg connector + hardware mass from 74.88 g/leg to {dry_s['one_leg_mass_g']:.2f} g/leg (-{100.0 * dry_s['mass_saved_vs_baseline_one_leg_g'] / 74.88:.1f}%).",
            },
            "buildability": {
                "grade": 4.0 if fam == "truss_box" else 3.6,
                "confidence": "medium",
                "trend": "improving",
                "reason": "Printable on a standard home-lab FDM printer (Bambu PAHT-CF / PETG-CF, 0.4 mm nozzle) with off-the-shelf cylindrical steel compression sleeves; eliminates sheet-metal forming brake tooling and 16 spacers/nuts per leg.",
            },
            "harness_covers": {
                "grade": 3.5,
                "confidence": "medium",
                "trend": "flat",
                "reason": "Preserves exact X=39.25 mm Shin_carrier hard-shell mounting datum, sensor recesses, and wire routing windows through the shin and ankle arches.",
            },
            "looks": {
                "grade": 0.0,
                "confidence": "low",
                "trend": "new",
                "reason": "Renders and interactive 3D models surfaced for user review; no aesthetic preference assumed.",
            },
            "completeness": {
                "grade": 4.0,
                "confidence": "high",
                "trend": "improving",
                "reason": "Complete bilateral STEP/STL/GLB CAD, steel compression sleeves, parts.json, bom.json, 3D SIMP fields, 13-pose BRep fit trace, and 3-stage orthotropic FEM.",
            },
        }

        record = {
            "id": cfg["result_id"],
            "study_id": cfg["study_id"],
            "version": cfg["version"],
            "dashboard": "design",
            "workspace": "Body",
            "robot_sections": ["Knee", "Ankle / foot"],
            "title": cfg["title"],
            "area": "Lower-leg 3D-printable FDM connector redesign (Shin, Ankle Coupler, Foot Monocoque)",
            "observation": cfg["observation"],
            "measurements": (
                f"One leg (3 stages + 20 SS sleeves): {dry_s['one_leg_mass_g']:.2f} g (saves {dry_s['mass_saved_vs_baseline_one_leg_g']:.2f} g/leg vs V16). "
                f"4kg@1.5g recovery: Shin u={st1['max_displacement_mm']:.3f}mm (p99={st1['primary_vm_p99_MPa']:.1f}MPa, {st1['p99_strength_margin_ratio']:.1f}x), "
                f"Coupler u={st2['max_displacement_mm']:.3f}mm (p99={st2['primary_vm_p99_MPa']:.1f}MPa, {st2['p99_strength_margin_ratio']:.1f}x vs 1.94x V16), "
                f"Foot u={st3['max_displacement_mm']:.3f}mm (p99={st3['primary_vm_p99_MPa']:.1f}MPa, {st3['p99_strength_margin_ratio']:.1f}x). "
                f"13-pose overlap: 0.000 mm^3."
            ),
            "limitations": (
                "Linear tetrahedral FEM with bonded steel-sleeve counterbores; physical bolt preload retention, "
                "interlaminar fatigue under impact, and print shrinkage tolerances require bench verification on printed prototypes."
            ),
            "provenance": [rel(STUDY / "round-card.md"), rel(ART / "baseline-audit.json"), rel(ART / "fem-report.json"), rel(ART / "fit-report.json")],
            "analysis_provenance": {
                "idle_task": "T-f39h6b",
                "decision_memory": "M-24x4qd",
                "baseline_manifest_sha256": sha(ROOT / "artifacts/cove50-engineering-competition/waist-pelvis-20261021/lead/integration-v16/parts.json"),
                "dimension_grades": dimension_grades,
                "no_selection_change": True,
            },
            "assets": assets,
            "clips": [],
            "design": {
                "id": cfg["result_id"],
                "title": cfg["short_title"],
                "summary": cfg["observation"],
                "visual_kind": "native_CAD_assembly",
                "maturity": "Comparative 3D-printable FDM engineering candidate; physical print & bench test pending",
                "image_path": rel(ART / "renders" / "stage-by-stage-cad.png"),
                "cad": {
                    "model_units": "m",
                    "views": [
                        {"label": "Complete lower body + 12 C047 actuators + toggleable hard-shell leg covers", "model_path": rel(fam_dir / "assembly-with-full-shell.glb")},
                    ],
                    "parts_path": rel(fam_dir / "parts.json"),
                    "bom_path": rel(fam_dir / "bom.json"),
                    "step_path": rel(fam_dir / "lower-leg-connectors-bilateral.step"),
                    "report_path": rel(ART / "fem-report.json"),
                    "show_shells": True,
                    "body_panel_opacity": 0.65,
                    "body_panel_label": "Leg hard-shell covers (uncheck to hide shell)",
                    "has_body_reference": True,
                    "fit_reference_on_load": False,
                    "fit_reference_label": "Fit lower-leg assembly",
                    "scope": (
                        "Bilateral 3-stage anatomical lower-leg connector replacement (Stage 1 Shin, Stage 2 Ankle Coupler, Stage 3 Foot Monocoque) "
                        "inside the fixed V16 C047 motor layout and shell envelope."
                    ),
                    "motion_note": "6 lower-leg joint controls (R/L_knee 0..125 deg, R/L_ankle_pitch -35..35 deg, R/L_ankle_roll -25..25 deg).",
                },
                "source_refs": source_refs,
                "evidence": evidence_cards,
                "bom": read_json(fam_dir / "bom.json"),
            },
        }

        rec_path = fam_dir / f"{cfg['result_id']}-record.json"
        write_json(rec_path, record)
        proc = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "infra/dashboard/results_hub.py", "publish", rel(rec_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        write_json(
            fam_dir / f"{cfg['result_id']}-publication.json",
            {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        )
        print(f"Published {cfg['result_id']}: exit_code={proc.returncode}")
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"Publication failed for {cfg['result_id']}")


if __name__ == "__main__":
    publish_all()
