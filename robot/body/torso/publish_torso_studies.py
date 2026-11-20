"""Publish all 3 Center-Body Torso Base Plate & Electronics Carrier FDM studies
(`cove50-torso-fdm-truss-box-r1`, `cove50-torso-fdm-topopt-r1`, and `cove50-torso-fdm-combined-r1`)
to the Robot Design Dashboard (`http://127.0.0.1:8805/design/`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

ART = common.ARTIFACT_DIR
ROOT = common.ROOT
STUDY = common.STUDY_DIR

sys.path.insert(0, str(ROOT / "robot/body/checks"))
import fem_evidence as FEV

FEM_STUDY, FEM_CASE = "cove50-torso-fdm-combined", "combined_shoulder_hip_neck_stall"
DRY = FEV.MATERIALS["dry"]
FEM_JSON = ART / "fem-evidence.json"
CHART = ART / "renders" / "fem-stress-displacement-comparison.png"   # drawn from fem-evidence.json by this publisher
PUBLISH_CMD = ".venv/bin/python robot/body/torso/publish_torso_studies.py   (--dry-run builds and prints the record without publishing)"



def bom_sleeves(bom):
    """Steel sleeves the BOM's items list: the sleeve sets name their count, '(N sleeves)', times the set quantity."""
    import re
    n = 0
    for it in bom.get("items", []):
        m = re.search(r"\((\d+) sleeves\)", str(it.get("name", "")))      # sleeve sets: '... compression sleeve set (41 sleeves)'
        if m:
            n += int(m.group(1)) * int(float(it.get("quantity") or 1))
    return n


def bom_from_items(bom):
    """The packaged BOM with its scope text stating the sleeve count of its own items (the packaged scope string carried an old count)."""
    out = dict(bom)
    n = bom_sleeves(bom)
    out["scope"] = (bom.get("scope", "").replace("56 pressed-in", f"{n} pressed-in") if "56 pressed-in" in bom.get("scope", "")
                    else bom.get("scope", ""))
    return out

def fem_numbers(ev):
    """Every FEM number of a record: from fem-evidence.json. Only stage 1 has a result (stage 2 mesh only, stages 3 and 4 no setup): the other stages carry the reason, and the chain is stage 1 alone."""
    part = {st: f"combined__{st}" for st in common.STAGES}
    return dict(part=part, txt={st: FEV.stage_text(ev, part[st], DRY, FEM_CASE) for st in part},
                rows={st: FEV.own_row(ev, part[st], DRY, FEM_CASE) for st in part}, stages_with_result=[st for st in part if part[st] in ev["parts"]],
                rss_dry=FEV.rss(ev, "torso", DRY, FEM_CASE), rss_text=FEV.chain_text(ev, "torso", FEM_CASE),
                min_dry=FEV.min_ratio(ev, part.values(), DRY, FEM_CASE), min_text=FEV.ratio_text(ev, part.values(), FEM_CASE), materials=FEV.materials(ev))


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def write_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT))


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
    chk = FEV.load_check(FEM_STUDY)   # fit / CAD-rule sections: the shared check's clearance + geometry report (recreate: fem_evidence.py --study ... --check-command)
    chk_path = FEV.ROOT / ".cache/work/cad-speed/checks" / f"{FEM_STUDY}-geometry/report.json"
    cn = FEV.check_numbers(chk)
    stages = list(common.STAGES)
    stage_parts = [f"combined__{st}" for st in stages if f"combined__{st}" in ev["parts"]]
    FEV.render_chart(ev, CHART, stage_parts, FEM_CASE, [p.replace("combined__", "") for p in stage_parts], "Torso stages with a result")
    lim = FEV.limitations(ev) + (" Battery thermal cycling needs bench verification. Not produced by the current tool (not in this record): the honeycomb thermal report "
                                 "(torso-thermal-results.json; the shared `heat` check is a different, per-part measurement and was not run here), the 25-pose multi-limb fit report with fastener rays, "
                                 "the baseline audit and mass comparison, and the topopt report. The geometry check has no native STEP for stage 2 (not audited; "
                                 f"{cn['parts_without_step']} part without one).")

    studies_cfg = [
        {
            "family": "truss_box",
            "study_id": "cove50-torso-fdm-truss-box",
            "version": "r1",
            "result_id": "cove50-torso-fdm-truss-box-r1",
            "title": "Center-Body Torso & Repeated 9-Cell Honeycomb Battery Carrier · Family 1: Blank-Slate 3D-Printed Hexagonal Honeycomb & Space-Truss Architecture (56 Pressed-In 304 SS Sleeves, 256×256 mm Bed Fit)",
            "short_title": "Center-Body Honeycomb Torso Family 1: 3D-Printed Hexagonal Honeycomb & Space-Truss (r1, 56 Steel Sleeves, 162.28 g)",
        },
        {
            "family": "topopt",
            "study_id": "cove50-torso-fdm-topopt",
            "version": "r1",
            "result_id": "cove50-torso-fdm-topopt-r1",
            "title": "Center-Body Torso & Repeated 9-Cell Honeycomb Battery Carrier · Family 2: Blank-Slate 3D SIMP Topology-Optimized Honeycomb Monocoque (56 Pressed-In 304 SS Sleeves, 256×256 mm Bed Fit)",
            "short_title": "Center-Body Honeycomb Torso Family 2: 3D SIMP Honeycomb Monocoque (r1, 56 Steel Sleeves, 161.16 g)",
        },
        {
            "family": "combined",
            "study_id": "cove50-torso-fdm-combined",
            "version": "r1",
            "result_id": "cove50-torso-fdm-combined-r1",
            "title": "Complete 4-Stage Center-Body Torso & Repeated 9-Cell Vertical Honeycomb Battery 3D-Printable FDM Architecture (Pressed-In 304 SS Sleeves, 26×26 cm FDM Bed Fit)",
            "short_title": "Complete 4-Stage Combined Honeycomb Torso & 9-Cell Battery FDM Architecture (r1)",
        },
    ]

    shared_files = [
        FEM_JSON,
        ART / "torso-candidates-manifest.json",
        CHART,
        STUDY / "round-card.md",
        STUDY / "common.py",
        STUDY / "audit_torso_baseline.py",
        STUDY / "run_torso_topopt.py",
        STUDY / "build_torso_carriers.py",
        STUDY / "thermal_evaluate.py",
        STUDY / "package_and_render.py",
        STUDY / "regenerate_derived.py",
        STUDY / "scratch-regeneration.json",
    ]

    for cfg in studies_cfg:
        fam = cfg["family"]
        fam_dir = ART / fam
        if fam != "combined" or not fam_dir.is_dir():   # only the combined study has check-based FEM evidence; the truss / topopt families are not in the tree
            print(f"skipped {cfg['result_id']}: no FEM evidence from the shared checks"); continue
        man = read_json(fam_dir / "manifest.json")
        assy = read_json(fam_dir / "assembly-parts.json")
        n = fem_numbers(ev)
        txt, rss_u_dry, min_dry_sf = n["txt"], n["rss_dry"], n["min_dry"]

        s1_m = man["stages"]["stage1_center_torso_spine_chassis"]["total_stage_mass_paht_cf_g"]
        s2_m = man["stages"]["stage2_lower_pelvis_battery_cradle"]["total_stage_mass_paht_cf_g"]
        s3_m = man["stages"]["stage3_front_power_electronics_carrier"]["total_stage_mass_paht_cf_g"]
        s4_m = man["stages"]["stage4_rear_imu_adc_carrier"]["total_stage_mass_paht_cf_g"]
        tot_m = man["total_4stage_mass_paht_cf_g"]

        obs = (
            f"4-stage 3D-printable PAHT-CF center-body torso, pelvis battery cradle and electronics carrier architecture ({fam}): "
            f"Stage 1 Center Torso Spine Chassis ({s1_m:.2f} g; stall: {txt['stage1_center_torso_spine_chassis']}), "
            f"Stage 2 Lower Pelvis Battery Cradle ({s2_m:.2f} g; stall: {txt['stage2_lower_pelvis_battery_cradle']}), "
            f"Stage 3 Front Power Electronics Carrier ({s3_m:.2f} g; stall: {txt['stage3_front_power_electronics_carrier']}), "
            f"and Stage 4 Rear IMU/ADC Carrier ({s4_m:.2f} g; stall: {txt['stage4_rear_imu_adc_carrier']}). "
            f"Total 4-stage mass including sleeves is {tot_m:.2f} g (build manifest). Chain deflection {n['rss_text']} is stage 1 alone (materials run: {', '.join(n['materials'])}). "
            f"Shared checks ({cn['poses']}, {cn['parts']} parts with a STEP): {cn['pairs_with_new_overlap']} pairs with a new overlap (largest {cn['max_new_overlap_mm3']:.1f} mm^3), smallest distance {cn['min_distance_mm']} mm; "
            f"geometry: {cn['sliver_faces']} sliver faces, {cn['micro_edges']} micro-edges, {cn['sleeves']} sleeves counted, smallest bed margin {cn['min_bed_margin_mm']} mm."
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
        for stg_key in stages:
            sel_fam = man["stages"][stg_key]["selected_family"]
            for p_cand in (
                ART / "parts" / f"{fam}__{stg_key}.step",
                ART / "parts" / f"{fam}__{stg_key}.npz",
                ART / "parts" / f"{sel_fam}__{stg_key}.step",
                ART / "parts" / f"{sel_fam}__{stg_key}.npz",
            ):
                if p_cand.exists():
                    file_set.add(p_cand)
        shell_assets_dir = ART / "shell_assets"
        if shell_assets_dir.exists():
            for p_sh in shell_assets_dir.rglob("*"):
                if p_sh.is_file():
                    file_set.add(p_sh)
        assets = [make_asset(p) for p in sorted(file_set)]

        source_refs = [
            {"label": "FEM evidence of the shared checks (fem-evidence.json; stage 1 only)", "path": rel(FEM_JSON)},
            {"label": "Shared checks clearance + geometry report (recreate: " + FEV.check_command(FEM_STUDY) + ")", "path": str(chk_path)},
            {"label": "Family manifest & stage mass breakdown", "path": rel(fam_dir / "manifest.json")},
        ]
        fit_card, cad_card, _ = FEV.check_cards(FEM_STUDY, chk, cfg["result_id"], source_refs)

        evidence_cards = [
            {
                "id": f"{cfg['result_id']}-fem",
                "title": "3D Tetrahedral FEM (Stage 1 Spine Chassis Only) & Print-Bed Envelope Audit: Center-Body Torso & Electronics Carrier (1g Static, 3g Gait, 6g Fall Impact & Multi-Limb Stall)",
                "kind": "simulation",
                "observation": obs,
                "measurements": [
                    {"name": "Total 4-stage carrier + sleeve mass (PAHT-CF, build manifest)", "value": tot_m, "unit": "g"},
                    {"name": "Total polymer mass across Stages 1-4 (PAHT-CF)", "value": man["total_polymer_mass_paht_cf_g"], "unit": "g"},
                    {"name": "Total sleeves mass (build manifest)", "value": man["total_sleeves_mass_g"], "unit": "g"},
                    {"name": "Chain deflection, stage 1 only (multi-limb stall, Bambu PAHT-CF; stages 2-4 have no result)", "value": round(rss_u_dry, 4), "unit": "mm"},
                    *FEV.stage_measurements(ev, "Stage 1 (Center Torso Spine)", n["part"]["stage1_center_torso_spine_chassis"], DRY, FEM_CASE, "stall"),
                    *FEV.stage_measurements(ev, "Stage 2 (Pelvis & Battery Cradle)", n["part"]["stage2_lower_pelvis_battery_cradle"], DRY, FEM_CASE, "stall"),
                    *FEV.stage_measurements(ev, "Stage 3 (Front Power Carrier)", n["part"]["stage3_front_power_electronics_carrier"], DRY, FEM_CASE, "stall"),
                    *FEV.stage_measurements(ev, "Stage 4 (Rear IMU/ADC Carrier)", n["part"]["stage4_rear_imu_adc_carrier"], DRY, FEM_CASE, "stall"),
                    {"name": "Stage 1 p99 strength ratio (Bambu PAHT-CF; the only stage with a result)", "value": min_dry_sf, "unit": "x"},
                ],
                "conditions": {
                    "load_cases": "1g_upright_battery_torso, 3g_dynamic_gait_arm_swing, 6g_fall_landing_impact, combined_shoulder_hip_neck_stall",
                    "materials": ", ".join(n["materials"]) + " (datasheet; only the materials the setups ran) + steel sleeves",
                },
                "limitations": lim,
                "source_refs": source_refs,
                "image_groups": [
                    {
                        "title": "4-Stage Center-Body Torso CAD, Print-Bed Fit, Shell Fit & Matched 3D FEM Comparisons",
                        "images": [
                            {"path": rel(fam_dir / "renders" / "waist-clevis-joint-views.png"), "caption": f"{cfg['short_title']}: Dorsal, Ventral, and Inside-Shell views of the 2-piece smooth Center Shell (center_front_shell + center_back_shell) bolting directly to flat Stage 1 & Stage 2 mounts"},
                            {"path": rel(fam_dir / "renders" / "battery-fins-cad-views.png"), "caption": f"{cfg['short_title']}: Dorsal (+Y), Ventral (-Y), Bare-Slot, and Top-3/4 views of the 9 honeycomb vertical window slots and 9x direct-on-battery 4-blade aluminum heat-sink fins"},
                            {"path": rel(fam_dir / "renders" / "shell-fit-views.png"), "caption": f"{cfg['short_title']}: Mounted Smooth 2-Piece Center Shell (center_front_shell + center_back_shell + chest_indicator, 181.3 x 55.1 x 211.4 mm each) on the 4-stage FDM core. "
                             f"Render of 2026-11-06: its panel label '+ 56 Sleeves' was a typed label, not the CAD's count ({bom_sleeves(assy['bom'])} sleeves in the BOM); package_and_render.py does not re-render it as it stands (its candidate inputs are not stored; build_torso_carriers.py can rebuild them)"},
                            {"path": rel(CHART), "caption": "p99 von Mises stress, peak displacement and p99 strength ratio of the torso stages that have a result (stage 1; multi-limb stall), only the materials that ran"},
                        ],
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
            "robot_sections": ["Torso / shell", "Torso", "Pelvis", "Shoulder", "Neck / head", "Electronics"],
            "title": cfg["title"],
            "area": "Complete 4-Stage Center-Body Torso Base Plate & Electronics Carrier 3D-Printable FDM Redesign (Stages 1–4: Center Torso Spine, Pelvis Battery Cradle, Front Power Carrier, Rear IMU/ADC Carrier)",
            "observation": obs,
            "measurements": (
                f"4-stage total mass: {tot_m:.2f} g (build manifest). "
                f"Chain deflection, stage 1 only (multi-limb stall): {n['rss_text']}. "
                f"Stage 1 p99 strength ratio: {n['min_text']} (stages 2-4: no FEM result). "
                f"Pose-swept clearance: {cn['pairs_with_new_overlap']} pairs with a new overlap, largest {cn['max_new_overlap_mm3']:.1f} mm^3, smallest distance {cn['min_distance_mm']} mm."
            ),
            "limitations": lim,
            "provenance": [rel(STUDY / "round-card.md"), rel(FEM_JSON), str(chk_path)],
            "analysis_provenance": {
                "idle_task": "T-zdevbv",
                "no_selection_change": False if fam == "combined" else True,
            },
            "assets": assets,
            "clips": [],
            "design": {
                "id": cfg["result_id"],
                "title": cfg["short_title"],
                "summary": obs,
                "visual_kind": "native_CAD_assembly",
                "maturity": "Selected 3D-printable FDM center-body torso & electronics carrier candidate (cove50-torso-fdm-combined-r1); evaluated for T-zdevbv" if fam == "combined" else "Comparative 3D-printable FDM engineering candidate; evaluated for T-zdevbv",
                "image_path": rel(fam_dir / "renders" / "shell-fit-views.png"),
                "cad": {
                    "model_units": "m",
                    "views": [
                        {
                            "label": "Complete Hip-to-Shoulder-to-Head Assembly + 9-Cell Battery & PCBs + Shell & Electronics Toggles",
                            "model_path": rel(fam_dir / "assembly-with-full-shell.glb"),
                        },
                        {
                            "label": "Bare 4-Stage Printable Center-Body Torso Carrier + Steel Sleeves Only (Z=190..389 mm)",
                            "model_path": rel(fam_dir / "carriers-4stage.step"),
                        },
                    ],
                    "parts_path": rel(fam_dir / "assembly-parts.json"),
                    "step_path": rel(fam_dir / "carriers-4stage.step"),
                    "report_path": rel(FEM_JSON),
                    "show_shells": False,
                    "body_panel_opacity": 0.42,
                    "body_panel_label": "Torso, pelvis & head shells (check to show shell, uncheck to hide)",
                    "overlays": [
                        {
                            "id": "electronics_and_battery",
                            "label": "Electronics, PCBs & 9-cell battery pack (uncheck to hide electronics)",
                            "default": True,
                        },
                        {
                            "id": "limb_and_neck_actuators",
                            "label": "Connected Hip, Shoulder & Neck/Head joints",
                            "default": True,
                        },
                    ],
                    "has_body_reference": True,
                    "fit_reference_on_load": False,
                    "fit_reference_label": "Fit center-body torso assembly",
                    "scope": "4-stage Blank-State 3D Generative FDM center-body torso, pelvis, battery & electronics carrier architecture spanning Hip to Shoulder to Head with pressed-in 304 SS compression sleeves and independent toggles for shells and electronics.",
                    "motion_note": "Interactive shoulder_pitch (-115..+75 deg), hip_yaw (-35..+35 deg), and neck_yaw (-60..+60 deg) joint definitions.",
                    "bom": bom_from_items(assy["bom"]),
                    "joints": assy["joints"],
                },
                "evidence": evidence_cards,
            },
        }

        rec_path = fam_dir / f"{cfg['result_id']}-record.json"
        if dry_run:   # build and print the full record; write nothing, publish nothing
            print(json.dumps(record, indent=1))
            print(f"dry run {cfg['result_id']}: record built, {len(assets)} assets, not published; publish with: {PUBLISH_CMD}")
            continue
        write_json(rec_path, record)
        import sqlite3
        conn = sqlite3.connect(str(ROOT / "data/results.sqlite"))
        for tbl, col in [
            ("results", "id"),
            ("assets", "result_id"),
            ("clips", "result_id"),
            ("design_evidence", "result_id"),
        ]:
            try:
                conn.execute(f"DELETE FROM {tbl} WHERE {col} = ?", (cfg["result_id"],))
            except Exception:
                pass
        try:
            conn.execute("DELETE FROM design_studies WHERE study_id = ?", (cfg["study_id"],))
        except Exception:
            pass
        conn.commit()
        conn.close()
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "infra/dashboard/results_hub.py"), "publish", str(rec_path)],
            check=True,
        )
        print(f"Published {cfg['result_id']} to http://127.0.0.1:8805/design/#{cfg['result_id']}")


if __name__ == "__main__":
    if "--fem-section" in sys.argv:   # print the FEM numbers of the record from fem-evidence.json only
        print(json.dumps(fem_numbers(FEV.load(FEM_STUDY)), indent=1))
    else:
        publish_all(dry_run="--dry-run" in sys.argv)
