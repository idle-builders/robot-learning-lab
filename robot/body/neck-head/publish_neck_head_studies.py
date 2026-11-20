"""Publish all 3 Neck & Head FDM Connector studies (`cove50-neck-head-fdm-truss-box-r1`,
`cove50-neck-head-fdm-topopt-r1`, and `cove50-neck-head-fdm-combined-r1`) to the Robot Design
Dashboard (`http://127.0.0.1:8805/design/`).
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ART, ROOT, STAGE_PART_BASENAMES, STUDY, read_json, rel, write_json

sys.path.insert(0, str(ROOT / "robot/body/checks"))
import fem_evidence as FEV

FEM_STUDY, FEM_CASE = "cove50-neck-head-fdm-combined", "adverse_neck_stall"
DRY = FEV.MATERIALS["dry"]   # the neck setups ran Bambu PAHT-CF only (part_settings fem.materials); PETG-CF is named in no text below unless FEV.materials(ev) lists it
FEM_JSON = ART / "fem-evidence.json"
PUBLISH_CMD = ".venv/bin/python robot/body/neck-head/publish_neck_head_studies.py   (--dry-run builds and prints the record without publishing)"
CHART = ART / "renders" / "fem-stress-displacement-comparison.png"   # drawn from fem-evidence.json by this publisher; not an old-script image


def fem_numbers(ev):
    """Every FEM number of a record: from fem-evidence.json, each stage along its print direction (dual-axis stall)."""
    part = {st: f"combined__{bn}" for st, bn in STAGE_PART_BASENAMES.items()}
    return dict(part=part, txt={st: FEV.stage_text(ev, part[st], DRY, FEM_CASE) for st in part},
                rows={st: FEV.own_row(ev, part[st], DRY, FEM_CASE) for st in part},
                rss_dry=FEV.rss(ev, "neck", DRY, FEM_CASE), rss_text=FEV.chain_text(ev, "neck", FEM_CASE),
                min_dry=FEV.min_ratio(ev, part.values(), DRY, FEM_CASE), min_text=FEV.ratio_text(ev, part.values(), FEM_CASE), materials=FEV.materials(ev))


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


def publish_all(dry_run=False):
    ev = FEV.load(FEM_STUDY)   # the FEM section of every record below comes from this file only
    chk_path = FEV.ROOT / '.cache/work/cad-speed/checks' / f'{FEM_STUDY}-geometry/report.json'
    chk = FEV.load_check(FEM_STUDY)   # fit / CAD-rule sections: the shared check's clearance + geometry report (recreate: fem_evidence.py --study ... --check-command)
    stage_parts = [f"combined__{bn}" for bn in STAGE_PART_BASENAMES.values()]
    FEV.render_chart(ev, CHART, stage_parts, FEM_CASE, [bn.replace("_", " ") for bn in STAGE_PART_BASENAMES.values()], "Neck & head stages")
    lim = FEV.limitations(ev) + (" Sleeve count: the build manifest says 34, the geometry check counts "
                                 f"{FEV.check_numbers(chk)['sleeves']} sleeve solids in the STEP files (not reconciled here). Not produced by the current tool: the 29-pose flexion fit report, "
                                 "the fastener / driver ray audit, the legacy baseline mass and collision comparison, and the topopt report; their numbers are not in this record.")
    stages = list(STAGE_PART_BASENAMES.keys())
    studies_cfg = [
        {
            "family": "truss_box",
            "study_id": "cove50-neck-head-fdm-truss-box",
            "version": "r1",
            "result_id": "cove50-neck-head-fdm-truss-box-r1",
            "title": "Neck & Head FDM Connectors · Family 1: Blank-State 3D Space-Truss & Ribbed Box-Cradle Architecture (34 Pressed-In 304 SS Sleeves Across Stages 1–4)",
            "short_title": "Neck & Head Family 1: Blank-State 3D Space-Truss & Ribbed Box-Cradle (r1, 34 Steel Sleeves)",
        },
        {
            "family": "topopt",
            "study_id": "cove50-neck-head-fdm-topopt",
            "version": "r1",
            "result_id": "cove50-neck-head-fdm-topopt-r1",
            "title": "Neck & Head FDM Connectors · Family 2: Blank-State 3D SIMP Topology-Optimized Monocoque Architecture (34 Pressed-In 304 SS Sleeves Across Stages 1–4)",
            "short_title": "Neck & Head Family 2: Blank-State 3D SIMP Topology-Optimized Monocoque (r1, 34 Steel Sleeves)",
        },
        {
            "family": "combined",
            "study_id": "cove50-neck-head-fdm-combined",
            "version": "r1",
            "result_id": "cove50-neck-head-fdm-combined-r1",
            "title": "Complete 4-Stage Torso-to-Head Blank-State 3D Generative FDM Neck & Head Architecture (34 Pressed-In 304 SS Sleeves + Consolidated Sensor/Compute/Mic Carrier)",
            "short_title": "Complete 4-Stage Combined Blank-State 3D Generative Neck & Head Architecture (r1, 34 Steel Sleeves)",
        },
    ]

    shared_files = [
        FEM_JSON,
        ART / "neck-head-build-summary.json",
        ART / "neck-head-topopt-report.json",
        ART / "neck-head-topopt-fields.npz",
        ART / "renders" / "stage-by-stage-cad.png",
        CHART,
        STUDY / "round-card.md",
        STUDY / "common.py",
        STUDY / "audit_neck_head_baseline.py",
        STUDY / "run_neck_head_topopt.py",
        STUDY / "build_neck_head_connectors.py",
        STUDY / "package_and_render.py",
        STUDY / "regenerate_derived.py",
        STUDY / "scratch-regeneration.json",
    ]

    for cfg in studies_cfg:
        fam = cfg["family"]
        fam_dir = ART / fam
        if fam != "combined":   # only the combined study has check-based FEM evidence; the truss / topopt families are not in the tree
            print(f"skipped {cfg['result_id']}: no FEM evidence from the shared checks"); continue
        man = read_json(fam_dir / "manifest.json")
        assy = read_json(fam_dir / "assembly-parts.json")

        n = fem_numbers(ev)
        cn = FEV.check_numbers(chk)
        txt, rss_u_dry, min_dry_sf = n["txt"], n["rss_dry"], n["min_dry"]

        s1_m = man["stages"]["stage1_neck_root_cradle"]["total_stage_mass_paht_cf_g"]
        s2_m = man["stages"]["stage2_neck_yaw_to_pitch_link"]["total_stage_mass_paht_cf_g"]
        s3_m = man["stages"]["stage3_neck_pitch_to_head_cradle"]["total_stage_mass_paht_cf_g"]
        s4_m = man["stages"]["stage4_head_sensor_compute_carrier"]["total_stage_mass_paht_cf_g"]
        tot_m = man["total_4stage_mass_paht_cf_g"]

        obs = (
            f"Blank-state 4-stage 3D generative FDM neck & head connector architecture ({fam}) with 4 one-piece FDM monocoques and pressed-in 304 SS compression sleeves: "
            f"Stage 1 Neck Root Cradle ({s1_m:.2f} g; stall: {txt['stage1_neck_root_cradle']}), "
            f"Stage 2 Yaw-to-Pitch L-Link ({s2_m:.2f} g; stall: {txt['stage2_neck_yaw_to_pitch_link']}), "
            f"Stage 3 Pitch-to-Head Right-Side U-Cradle ({s3_m:.2f} g; stall: {txt['stage3_neck_pitch_to_head_cradle']}), "
            f"and Stage 4 Consolidated Head Sensor/Compute/Mic Carrier ({s4_m:.2f} g; stall: {txt['stage4_head_sensor_compute_carrier']}). "
            f"Total 4-stage mass including the steel sleeves is {tot_m:.2f} g (build manifest). "
            f"Dual-axis stall RSS chain deflection {n['rss_text']} (materials run: {', '.join(n['materials'])}). "
            f"Shared checks over the pose set: {cn['pairs_with_new_overlap']} pairs with a new overlap (largest {cn['max_new_overlap_mm3']:.1f} mm^3), smallest distance {cn['min_distance_mm']} mm; "
            f"geometry: {cn['sliver_faces']} sliver faces, {cn['micro_edges']} micro-edges, thinnest load-path slab {cn['min_loadpath_mm2']} mm^2."
        )

        file_set = set(p for p in shared_files if p.exists())
        for p in fam_dir.rglob("*"):
            if (
                p.is_file()
                and not p.name.endswith("-publication.json")
                and not p.name.endswith("-record.json")
                and p.suffix.lower() != ".stl"
            ):
                file_set.add(p)
        for stg_key, bname in STAGE_PART_BASENAMES.items():
            sel_fam = man["stages"][stg_key]["selected_family"]
            for p_cand in (
                ART / "parts" / f"{fam}__{bname}.step",
                ART / "parts" / f"{fam}__{bname}.npz",
                ART / "parts" / f"{sel_fam}__{bname}.step",
                ART / "parts" / f"{sel_fam}__{bname}.npz",
            ):
                if p_cand.exists():
                    file_set.add(p_cand)
        assets = [make_asset(p) for p in sorted(file_set)]

        source_refs = [
            {"label": "FEM evidence of the shared checks (fem-evidence.json, Stages 1-4)", "path": rel(FEM_JSON)},
            {"label": "Shared checks clearance + geometry report (recreate: " + FEV.check_command(FEM_STUDY) + ")", "path": str(chk_path)},
            {"label": "Family manifest & stage mass breakdown", "path": rel(fam_dir / "manifest.json")},
        ]
        fit_card, cad_card, _ = FEV.check_cards(FEM_STUDY, chk, cfg["result_id"], source_refs)

        evidence_cards = [
            {
                "id": f"{cfg['result_id']}-fem",
                "title": "Static FEM of the 4 neck & head stages (shared checks: fem, dual-axis stall)",
                "kind": "simulation",
                "observation": obs,
                "measurements": [
                    {"name": "Total 4-stage connector + sleeve mass (PAHT-CF, build manifest)", "value": tot_m, "unit": "g"},
                    {"name": "Total polymer mass across Stages 1-4 (PAHT-CF)", "value": man["total_polymer_mass_paht_cf_g"], "unit": "g"},
                    {"name": "Total sleeves mass (build manifest)", "value": man["total_sleeves_mass_g"], "unit": "g"},
                    {"name": "4-stage RSS chain deflection (dual-axis stall, Bambu PAHT-CF)", "value": round(rss_u_dry, 4), "unit": "mm"},
                    *FEV.stage_measurements(ev, "Stage 1 (Neck Root Cradle)", n["part"]["stage1_neck_root_cradle"], DRY, FEM_CASE, "stall"),
                    *FEV.stage_measurements(ev, "Stage 2 (Yaw-to-Pitch L-Link)", n["part"]["stage2_neck_yaw_to_pitch_link"], DRY, FEM_CASE, "stall"),
                    *FEV.stage_measurements(ev, "Stage 3 (Pitch-to-Head U-Cradle)", n["part"]["stage3_neck_pitch_to_head_cradle"], DRY, FEM_CASE, "stall"),
                    *FEV.stage_measurements(ev, "Stage 4 (Head Sensor/Compute Carrier)", n["part"]["stage4_head_sensor_compute_carrier"], DRY, FEM_CASE, "stall"),
                    {"name": "Minimum stage p99 strength ratio (Bambu PAHT-CF)", "value": round(min_dry_sf, 2), "unit": "x"},
                ],
                "conditions": {
                    "load_cases": "upright_gravity_1g, dynamic_nod_turn_3g, fall_recovery_head_6g, adverse_neck_stall (the numbers above are the stall case)",
                    "materials": ", ".join(n["materials"]) + " (datasheet; only the materials the setups ran) + steel sleeves",
                },
                "limitations": lim,
                "source_refs": source_refs,
                "image_groups": [
                    {
                        "title": "Matched FEM of the 4 stages, drawn from fem-evidence.json",
                        "images": [{"path": rel(CHART), "caption": "p99 von Mises stress, peak displacement and p99 strength ratio of the 4 stages (dual-axis stall), only the materials that ran"}],
                    }
                ],
            },
            fit_card,
            cad_card,
        ]

        record = {
            "id": cfg["result_id"],
            "study_id": cfg["study_id"],
            "version": cfg["version"],
            "dashboard": "design",
            "workspace": "Body",
            "robot_sections": ["Neck", "Head"],
            "title": cfg["title"],
            "area": "Complete 4-Stage Neck & Head 3D-Printable FDM Connector Redesign (Stages 1–4: Neck Root Cradle, Yaw-to-Pitch L-Link, Pitch-to-Head Cradle, Consolidated Head Sensor/Compute/Mic Carrier)",
            "observation": obs,
            "measurements": (
                f"4-stage total mass: {tot_m:.2f} g (build manifest). 4-stage RSS deflection (stall): {n['rss_text']}. "
                f"Min stage p99 strength ratio: {n['min_text']}. "
                f"Pose-swept clearance: {cn['pairs_with_new_overlap']} pairs with a new overlap, largest {cn['max_new_overlap_mm3']:.1f} mm^3, smallest distance {cn['min_distance_mm']} mm."
            ),
            "limitations": lim + " MIPI ribbon flex fatigue also needs bench verification.",
            "provenance": [rel(STUDY / "round-card.md"), rel(FEM_JSON), str(chk_path)],
            "analysis_provenance": {
                "idle_task": "T-zesmm6",
                "idle_decision": "M-8jkx67",
                "no_selection_change": False if fam == "combined" else True,
            },
            "assets": assets,
            "clips": [],
            "design": {
                "id": cfg["result_id"],
                "title": cfg["short_title"],
                "summary": obs,
                "visual_kind": "native_CAD_assembly",
                "maturity": "Selected 3D-printable FDM neck & head connector candidate (cove50-neck-head-fdm-combined-r1); user-accepted for T-zesmm6 (M-8jkx67)" if fam == "combined" else "Comparative 3D-printable FDM engineering candidate; evaluated for T-zesmm6 (M-8jkx67)",
                "image_path": rel(fam_dir / "renders" / "shell-fit-views.png"),
                "cad": {
                    "model_units": "m",
                    "views": [
                        {
                            "label": "Complete 2-DOF neck & head assembly + XM335 actuators + ROCK4D/cameras/mic + steel sleeves + toggleable head/neck/torso shells",
                            "model_path": rel(fam_dir / "assembly-with-full-shell.glb"),
                        }
                    ],
                    "parts_path": rel(fam_dir / "assembly-parts.json"),
                    "step_path": rel(fam_dir / "connectors-4stage.step"),
                    "report_path": rel(FEM_JSON),
                    "show_shells": True,
                    "body_panel_opacity": 0.62,
                    "body_panel_label": "Head skin, neck cowl & torso shells (uncheck to hide shell)",
                    "has_body_reference": True,
                    "fit_reference_on_load": False,
                    "fit_reference_label": "Fit neck & head assembly",
                    "scope": "4-stage Blank-State 3D Generative FDM neck & head connector architecture with 34 pressed-in 304 SS compression sleeves, consolidated one-piece head sensor/compute/microphone carrier, and toggleable head/neck/torso shells.",
                    "motion_note": "Interactive neck_yaw (-60..+60 deg) and neck_pitch (-30..+35 deg) joint controls.",
                },
                "source_refs": source_refs,
                "evidence": evidence_cards,
                "bom": assy["bom"],
            },
        }

        rec_path = fam_dir / f"{cfg['result_id']}-record.json"
        if dry_run:   # build and print the full record; write nothing, publish nothing
            print(json.dumps(record, indent=1))
            print(f"dry run {cfg['result_id']}: record built, {len(assets)} assets, not published; publish with: {PUBLISH_CMD}")
            continue
        write_json(rec_path, record)
        proc = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "infra/dashboard/results_hub.py"), "publish", str(rec_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        (fam_dir / f"{cfg['result_id']}-publication.json").write_text(proc.stdout)
        print(f"Published {cfg['result_id']}:", proc.stdout.strip())


if __name__ == "__main__":
    if "--fem-section" in sys.argv:   # print the FEM numbers of the record from fem-evidence.json only
        print(json.dumps(fem_numbers(FEV.load(FEM_STUDY)), indent=1))
    else:
        publish_all(dry_run="--dry-run" in sys.argv)
