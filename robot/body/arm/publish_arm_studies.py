"""Publish all 3 Arm FDM Connector studies (`cove50-arm-fdm-truss-box-r1`,
`cove50-arm-fdm-topopt-r1`, and `cove50-arm-fdm-combined-r1`) to the Robot Design Dashboard
(`http://127.0.0.1:8805/design/`).
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ART, ROOT, STAGE_PART_BASENAMES, STUDY, read_json, rel, sha, write_json

sys.path.insert(0, str(ROOT / "robot/body/checks"))
import fem_evidence as FEV

FEM_STUDY, FEM_CASE = "cove50-arm-fdm-combined", "recovery_4kg_1p5g"
DRY = FEV.MATERIALS["dry"]
FEM_JSON = ART / "fem-evidence.json"
CHART = ART / "renders" / "fem-stress-displacement-comparison.png"   # drawn from fem-evidence.json by this publisher
PUBLISH_CMD = ".venv/bin/python robot/body/arm/publish_arm_studies.py   (--dry-run builds and prints the records without publishing)"


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


def fem_numbers(ev, man):
    """Every FEM number of a record: from fem-evidence.json (right arm, each stage along its print direction); stage masses from the family manifest."""
    part = {st: f"R_{bn}_combined" for st, bn in STAGE_PART_BASENAMES.items()}
    mass = {st: man["stages"][st]["total_stage_mass_paht_cf_g"] for st in part}
    return dict(part=part, txt={st: FEV.stage_text(ev, part[st], DRY, FEM_CASE) for st in part},
                rows={st: FEV.own_row(ev, part[st], DRY, FEM_CASE) for st in part}, mass=mass, arm_mass=sum(mass.values()),
                min_dry_ratio=FEV.min_ratio(ev, part.values(), DRY, FEM_CASE), min_text=FEV.ratio_text(ev, part.values(), FEM_CASE),
                rss_dry=FEV.rss(ev, "R", DRY, FEM_CASE), rss_text=FEV.chain_text(ev, "R", FEM_CASE), materials=FEV.materials(ev))


def publish_all(dry_run=False):
    ev = FEV.load(FEM_STUDY)   # the FEM section of every record below comes from this file only
    chk = FEV.load_check(FEM_STUDY)   # fit / CAD-rule sections: the shared check's clearance + geometry report (recreate: fem_evidence.py --study ... --check-command)
    chk_path = FEV.ROOT / ".cache/work/cad-speed/checks" / f"{FEM_STUDY}-geometry/report.json"
    cn = FEV.check_numbers(chk)
    stage_parts = [f"R_{bn}_combined" for bn in STAGE_PART_BASENAMES.values()]
    FEV.render_chart(ev, CHART, stage_parts, FEM_CASE, [bn.replace("_", " ") for bn in STAGE_PART_BASENAMES.values()], "Right arm stages")
    lim = FEV.limitations(ev) + (" Not produced by the current tool (not in this record): the multi-pose flexion fit report with 42 screw / driver rays, the baseline audit, "
                                 "the topopt report and the hard-typed 38-connection count; clearance and geometry come from the shared checks over the 27-pose set.")

    studies_cfg = [
        {
            "family": "truss_box",
            "study_id": "cove50-arm-fdm-truss-box",
            "version": "r12",
            "result_id": "cove50-arm-fdm-truss-box-r12",
            "title": "Arm FDM Connectors · Family 1: Blank-State 3D Space-Truss, C-Channel Shoulder Bridge & Down-First Upper Arm (38 Bolted Connections/Arm: 34 Steel Sleeves + 4 Forearm Shell Mounts)",
            "short_title": "Arm Family 1: Blank-State 3D Space-Truss & C-Channel Shoulder Bridge (r12, 38 Connections/Arm)",
        },
        {
            "family": "topopt",
            "study_id": "cove50-arm-fdm-topopt",
            "version": "r12",
            "result_id": "cove50-arm-fdm-topopt-r12",
            "title": "Arm FDM Connectors · Family 2: Blank-State 3D SIMP Topology-Optimized C-Channel Shoulder Bridge & Down-First Upper Arm (38 Connections/Arm: 34 Steel Sleeves + 4 Forearm Shell Mounts)",
            "short_title": "Arm Family 2: Blank-State 3D SIMP Topology-Optimized (r12, 38 Connections/Arm)",
        },
        {
            "family": "combined",
            "study_id": "cove50-arm-fdm-combined",
            "version": "r12",
            "result_id": "cove50-arm-fdm-combined-r12",
            "title": "Complete 4-Stage Shoulder-to-Wrist Blank-State 3D Generative FDM Arm Architecture (38 Connections/Arm: C-Channel Shoulder Bridge + Down-First Upper Arm + 34 Steel Sleeves + 4 Forearm Shell Mounts)",
            "short_title": "Complete 4-Stage Combined Blank-State 3D Generative Arm Architecture (r12, 38 Connections/Arm)",
        },
    ]

    shared_files = [
        FEM_JSON,
        CHART,
        STUDY / "round-card.md",
        STUDY / "common.py",
        STUDY / "run_arm_topopt.py",
        STUDY / "build_arm_connectors.py",
        STUDY / "package_and_render.py",
        STUDY / "regenerate_derived.py",
        STUDY / "scratch-regeneration.json",
        STUDY / "acceptance.json",
    ]

    for cfg in studies_cfg:
        fam = cfg["family"]
        fam_dir = ART / fam
        if fam != "combined" or not fam_dir.is_dir():   # only the combined study has check-based FEM evidence; the truss / topopt families are not in the tree
            print(f"skipped {cfg['result_id']}: no FEM evidence from the shared checks"); continue
        man = read_json(fam_dir / "manifest.json")
        assy = read_json(fam_dir / "assembly-parts.json")

        n = fem_numbers(ev, man)
        part, txt, rows, mass, arm_mass = n["part"], n["txt"], n["rows"], n["mass"], n["arm_mass"]
        min_dry_ratio, rss_dry = n["min_dry_ratio"], n["rss_dry"]
        s1, s2, s3, s4 = (rows[k] for k in part)

        obs = (
            f"Blank-state 4-stage 3D generative FDM arm connector architecture ({fam}): "
            f"Stage 1 C-Channel Shoulder Pitch-to-Roll Bridge ({mass["stage1_shoulder_pitch_to_roll"]:.2f} g; {txt["stage1_shoulder_pitch_to_roll"]}), "
            f"Stage 2 Roll-to-Yaw Cradle ({mass["stage2_shoulder_roll_to_yaw"]:.2f} g; {txt["stage2_shoulder_roll_to_yaw"]}), "
            f"Stage 3 Upper-Arm Monocoque ({mass["stage3_upper_arm_yaw_to_elbow"]:.2f} g; {txt["stage3_upper_arm_yaw_to_elbow"]}), "
            f"and Stage 4 Forearm-to-Wrist Cage ({mass["stage4_forearm_elbow_to_wrist"]:.2f} g; {txt["stage4_forearm_elbow_to_wrist"]}). "
            f"Single-arm 4-stage mass including sleeves is {arm_mass:.2f} g ({2*arm_mass:.2f} g bilateral, build manifest). "
            f"4-stage RSS chain deflection of the right arm at 4 kg @ 1.5g recovery is {n['rss_text']}, each stage along its own print direction (materials run: {', '.join(n['materials'])}); no baseline-metal comparison in the shared checks. "
            f"Shared checks (both arms, {cn['poses']}): {cn['pairs_with_new_overlap']} pairs with a new overlap (largest {cn['max_new_overlap_mm3']:.1f} mm^3), smallest distance {cn['min_distance_mm']} mm; "
            f"geometry: {cn['sliver_faces']} sliver faces, {cn['micro_edges']} micro-edges, {cn['sleeves']} sleeves counted, largest part-vs-sleeve overlap {cn['max_sleeve_overlap_mm3']} mm^3."
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
        assets = [make_asset(p) for p in sorted(file_set)]

        source_refs = [
            {"label": "FEM evidence of the shared checks (fem-evidence.json, Stages 1-4, R and L arm)", "path": rel(FEM_JSON)},
            {"label": "Shared checks clearance + geometry report (recreate: " + FEV.check_command(FEM_STUDY) + ")", "path": str(chk_path)},
            {"label": "Family manifest & stage mass breakdown", "path": rel(fam_dir / "manifest.json")},
        ]
        fit_card, cad_card, _ = FEV.check_cards(FEM_STUDY, chk, cfg["result_id"], source_refs)

        evidence_cards = [
            {
                "id": f"{cfg['result_id']}-fem",
                "title": "Matched 3D Tetrahedral FEM: 4-Stage Arm Stiffness, Strength & Carrier Consolidation (1.5 kg Reach to 4–6 kg Support)",
                "kind": "simulation",
                "observation": obs,
                "measurements": [
                    {"name": "Single-arm 4-stage connector + sleeve mass (PAHT-CF, build manifest)", "value": round(arm_mass, 2), "unit": "g"},
                    {"name": "Bilateral arms 8-stage total mass (PAHT-CF)", "value": round(2 * arm_mass, 2), "unit": "g"},
                    {"name": "4-stage RSS chain deflection, right arm (4kg@1.5g recovery, Bambu PAHT-CF)", "value": rss_dry, "unit": "mm"},
                    {"name": "Stage 1 (C-Channel Pitch-to-Roll Bridge) max deflection (4kg@1.5g)", "value": s1["max_displacement_mm"], "unit": "mm"},
                    {"name": "Stage 1 (C-Channel Pitch-to-Roll Bridge) p99 strength ratio (Bambu PAHT-CF)", "value": s1["p99_strength_ratio"], "unit": "x"},
                    {"name": "Stage 2 (3D Sculpted Roll-to-Yaw Cradle) max deflection (4kg@1.5g)", "value": s2["max_displacement_mm"], "unit": "mm"},
                    {"name": "Stage 2 (3D Sculpted Roll-to-Yaw Cradle) p99 strength ratio (Bambu PAHT-CF)", "value": s2["p99_strength_ratio"], "unit": "x"},
                    {"name": "Stage 3 (Down-First Wrap-Around Upper-Arm Monocoque) max deflection (4kg@1.5g)", "value": s3["max_displacement_mm"], "unit": "mm"},
                    {"name": "Stage 3 (Down-First Wrap-Around Upper-Arm Monocoque) p99 strength ratio (Bambu PAHT-CF)", "value": s3["p99_strength_ratio"], "unit": "x"},
                    {"name": "Stage 4 (Dual-Cheek Forearm-to-Wrist Cage) max deflection (4kg@1.5g)", "value": s4["max_displacement_mm"], "unit": "mm"},
                    {"name": "Stage 4 (Dual-Cheek Forearm-to-Wrist Cage) p99 strength ratio (Bambu PAHT-CF)", "value": s4["p99_strength_ratio"], "unit": "x"},
                    {"name": "Minimum stage p99 strength ratio (Bambu PAHT-CF)", "value": min_dry_ratio, "unit": "x"},
                ],
                "conditions": {
                    "load_cases": "manip_1p5kg_reach, support_4kg_1g (39.2 N), recovery_4kg_1p5g (58.8 N), heavy_6kg_1g (58.8 N + high moment)",
                    "materials": ", ".join(n["materials"]) + " (datasheet; only the materials the setups ran) + steel sleeves",
                },
                "limitations": lim,
                "source_refs": source_refs,
                "image_groups": [
                    {
                        "title": "4-Stage Arm CAD, Hard-Shell Fit & Matched 3D FEM Comparisons",
                        "images": [
                            {"path": rel(fam_dir / "renders" / "shell-fit-views.png"), "caption": f"{cfg['short_title']}: Right arm front, back, lateral (shell on) and covers-off connector + sleeve inspection"},
                            {"path": rel(CHART), "caption": "p99 von Mises stress, peak displacement and p99 strength ratio of the right-arm stages (4 kg @ 1.5g recovery), only the materials that ran"},
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
            "robot_sections": ["Shoulder", "Elbow"],
            "title": cfg["title"],
            "area": "Complete 4-Stage Arm 3D-Printable FDM Connector Redesign (Stages 1–4: Shoulder Pitch-to-Roll, Roll-to-Yaw, Upper-Arm Yaw-to-Elbow, Forearm-to-Wrist)",
            "observation": obs,
            "measurements": (
                f"Single arm: {arm_mass:.2f} g ({2 * arm_mass:.2f} g bilateral). "
                f"4-stage RSS deflection, right arm (4kg@1.5g): {n['rss_text']}. "
                f"Min stage p99 strength ratio: {n['min_text']}. "
                f"Pose-swept clearance (both arms): {cn['pairs_with_new_overlap']} pairs with a new overlap, largest {cn['max_new_overlap_mm3']:.1f} mm^3."
            ),
            "limitations": lim,
            "provenance": [rel(STUDY / "round-card.md"), rel(FEM_JSON), str(chk_path)],
            "analysis_provenance": {
                "idle_task": "T-zw7f7t",
                "idle_decision": "M-d34p89",
                "no_selection_change": fam != "combined",
            },
            "assets": assets,
            "clips": [],
            "design": {
                "id": cfg["result_id"],
                "title": cfg["short_title"],
                "summary": obs,
                "visual_kind": "native_CAD_assembly",
                "maturity": "Selected 3D-printable FDM arm connector candidate (cove50-arm-fdm-combined-r12); physical print & bench test pending" if fam == "combined" else "Comparative 3D-printable FDM engineering candidate; physical print & bench test pending",
                "image_path": rel(fam_dir / "renders" / "shell-fit-views.png"),
                "cad": {
                    "model_units": "m",
                    "views": [
                        {
                            "label": "Complete bilateral 8-DOF arms + C047 actuators + 68 steel sleeves (38 connections/arm) + toggleable hard-shell covers",
                            "model_path": rel(fam_dir / "assembly-with-full-shell.glb"),
                        }
                    ],
                    "parts_path": rel(fam_dir / "assembly-parts.json"),
                    "step_path": rel(fam_dir / "connectors-bilateral.step"),
                    "report_path": rel(FEM_JSON),
                    "show_shells": True,
                    "body_panel_opacity": 0.65,
                    "body_panel_label": "Arm hard-shell covers (uncheck to hide shell)",
                    "has_body_reference": True,
                    "fit_reference_on_load": False,
                    "fit_reference_label": "Fit bilateral arm assembly",
                    "scope": "Bilateral 4-stage Blank-State 3D Generative FDM arm connector architecture with 68 304 SS compression sleeves (34/arm across dual-cheek C047 yokes, horns, and wrist pad), integrated 2-screw/panel forearm captive-nut shell mounts, and toggleable hard-shell covers.",
                    "motion_note": "All 8 bilateral arm joint controls (R/L_shoulder_pitch, R/L_shoulder_roll, R/L_shoulder_yaw, R/L_elbow).",
                },
                "source_refs": source_refs,
                "evidence": evidence_cards,
                "bom": assy["bom"] if isinstance(assy["bom"], dict) else {
                    "scope": f"Bilateral 4-stage FDM arm structural connectors ({fam}), 68 pressed-in 304 SS compression sleeves, and 8 captive M3 forearm shell nuts",
                    "completeness": "partial",
                    "notes": "Replaces upper__R/L_pitch_to_roll_one_piece, upper__R/L_roll_to_yaw_one_piece, R/L_S_yaw_to_elbow, R/L_forearm_frame_drilled, R/L_shoulder_carrier, and R/L_fore_carrier.",
                    "exclusions": ["Retained C047 servo motors, horns, wrist brackets, and fitted hard-shell panels"],
                    "items": assy["bom"],
                },
            },
        }

        rec_path = fam_dir / f"{cfg['result_id']}-record.json"
        if dry_run:   # build and print the full record; write nothing, publish nothing
            print(json.dumps(record, indent=1))
            print(f"dry run {cfg['result_id']}: record built, {len(assets)} assets, not published; publish with: {PUBLISH_CMD}")
            continue
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
    if "--fem-section" in sys.argv:   # print the FEM numbers of the record from fem-evidence.json only
        print(json.dumps(fem_numbers(FEV.load(FEM_STUDY), read_json(ART / "combined" / "manifest.json")), indent=1))
    else:
        publish_all(dry_run="--dry-run" in sys.argv)
