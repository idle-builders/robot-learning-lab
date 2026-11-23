"""Package and publish the roll-flip hip variant (Idle T-fs5v6d) on the robot design dashboard.

The hip roll motor case rides with the thigh, turned 90 deg about its own shaft so it hangs down (roll axis
unchanged); a new Stage 4 thigh connector cradles it, and a new Stage 5 plate carries the roll horn and idler
(robot/body/hip-shell/DECISIONS.md D-12, D-13, F-04). Inputs come from build_upper_hip_connectors.py --roll-flip and
roll_flip_checks.py.

  .venv/bin/python -B robot/body/leg/publish_roll_flip.py package   # GLB, bilateral STEP, renders
  .venv/bin/python -B robot/body/leg/publish_roll_flip.py record    # writes the publication JSON
  .venv/bin/python infra/dashboard/results_hub.py publish artifacts/robot/body/leg/roll-flip/publication.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bom"))
import bom_config as BOM_CONFIG  # noqa: E402  (joint ranges from the body definitions)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ART = ROOT / "artifacts/robot/body/leg/roll-flip"
MANIFEST = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
SHEETS = ROOT / "robot/body/hip-shell/sections/drawing-sheets/roll-flip"
STUDY = "hip-roll-motor-in-thigh"


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def package():
    sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
    from package import scene  # motor-refit packager: one named node per part, metres
    man = json.load(open(MANIFEST))
    scene(man).export(str(ART / "assembly-with-full-shell.glb"))
    from build123d import Compound, export_step, import_step
    names = [f"{s}_{n}" for s in ("R", "L") for n in ("thigh_roll_flip", "hip_pitch_to_roll_roll_flip")]
    shapes = []
    for n in names:
        for suffix in ("", "_steel_sleeves"):
            shapes.append(import_step(str(ART / "parts" / f"{n}{suffix}.step")))
    export_step(Compound(children=shapes), str(ART / "connectors-bilateral.step"))
    (ART / "renders").mkdir(exist_ok=True)
    for f in sorted(SHEETS.glob("case-*.png")) + [SHEETS / n for n in ("thigh-shell-shaded.png", "shin-cover-mounts.png", "thigh-connector-h-cage.png", "thigh-shell-top.png") if (SHEETS / n).exists()]:
        shutil.copy2(f, ART / "renders" / f.name)
    print("packaged", ART)


def m(name, value, unit, condition=None):
    out = {"name": name, "value": value, "unit": unit}
    if condition:
        out["condition"] = condition
    return out


def record():
    rep = json.load(open(ART / "report.json"))
    fem, mass = rep["fem"], rep["mass"]

    def p99(part, case, normal):
        d = fem[part]["by_layer_normal"][normal]["PAHT_CF_dry"][case]
        return round(d["p99_vm_MPa"], 1), round(d["p99_strength_ratio"], 2)

    n4 = next(iter(fem["stage4_roll_flip"]["by_layer_normal"]))   # the H-cage print direction the FEM ran along
    s4n = {c: p99("stage4_roll_flip", c, n4) for c in ("recovery_4kg_1p5g", "heavy_6kg_1p5g", "roll_stall")}
    s4o = {c: p99("stage4_selected", c, "x") for c in ("recovery_4kg_1p5g", "heavy_6kg_1p5g", "roll_stall")}
    # print orientation of the new Stage 5: the layer normal with the best worst-case of strength and layer ratios
    def governing(part, case="heavy_6kg_1p5g"):
        bl = fem[part]["by_layer_normal"]
        return max(bl, key=lambda n: min(bl[n]["PAHT_CF_dry"][case]["p99_strength_ratio"], bl[n]["PAHT_CF_dry"][case]["p99_layer_ratio"]))
    n5 = governing("stage5_roll_flip")
    s5n = {c: p99("stage5_roll_flip", c, n5) for c in ("recovery_4kg_1p5g", "heavy_6kg_1p5g", "roll_stall")}
    fit = rep["fit_brep"]["worst_by_pair"]
    gap = {k: fit[v]["min_distance_mm"] for k, v in (("s4s5", "stage4 (R_roll_link) x stage5"),
                                                    ("s5s6", "stage5 (R_pitch_link) x R_hip_yaw_to_pitch_combined"),
                                                    ("s5case", "stage5 (R_pitch_link) x R_roll_servo_case_reference (turned)"))}
    s5o = {c: p99("stage5_selected", c, "z") for c in ("recovery_4kg_1p5g", "heavy_6kg_1p5g", "roll_stall")}
    mp = mass["parts"]["R"]
    links = mass["links"]
    renders = [ART / "renders" / n for n in ("thigh-shell-shaded.png", "thigh-connector-h-cage.png", "thigh-shell-top.png", "shin-cover-mounts.png") if (ART / "renders" / n).exists()] + sorted((ART / "renders").glob("case-*.png"))
    src = [rel(HERE / f) for f in ("build_upper_hip_connectors.py", "common.py", "roll_flip_checks.py", "thigh_shell.py", "publish_roll_flip.py")] \
        + [rel(SHEETS.parents[1] / f) for f in ("thigh_connector_sheet.py", "thigh_shell_views.py", "shin_cover_mounts_view.py")]   # published images
    data_assets = [ART / "report.json", ART / "cad-manifest.json", ART / "connectors-bilateral.step", MANIFEST,
                   ROOT / "robot/body/hip-shell/DECISIONS.md"]
    data_assets += sorted((ART / "parts").glob("*_roll_flip*.st*")) + sorted((ART / "parts").glob("*_thigh_shell_*.stl")) \
        + sorted((ART / "parts").glob("*_shin_back_hard_cover.stl")) + sorted((ART / "parts").glob("*_shin_combined.st*"))
    def slug(p):
        return Path(p).name.replace(".", "-").replace("_", "-").lower()

    assets = ([{"id": "neutral", "title": "Complete body, roll motor in the thigh", "kind": "model",
                "path": rel(ART / "assembly-with-full-shell.glb")}]
              + [{"id": slug(p), "title": p.stem.replace("-", " "), "kind": "image", "path": rel(p)} for p in renders]
              + [{"id": slug(p), "title": p.name, "kind": "data", "path": rel(p)} for p in data_assets])
    report_ref = {"label": "Native report (fit, motion, CAD rules, FEM, mass)", "path": rel(ART / "report.json")}
    evidence = [
        dict(id="clearance", title="Clearance over the whole-leg poses", kind="analysis",
             observation="The H-cage thigh connector (owner 2026-11-19: 2 + 2 screws per actuator) adds no overlap in "
                         "the 151 realistic whole-leg poses; its plates sit about 1.4 mm from the swinging pitch-to-roll "
                         "arms (was 0.15 mm). The exact fit only finds contact at poses beyond the joint ranges.",
             measurements=[
                 m("New overlap, new parts, realistic poses", 0.0, "mm3", "151 poses incl. paths (A-11)"),
                 m("Stage 4 vs battery cradle, exact B-rep", fit["stage4 (R_roll_link) x combined__stage2_lower_pelvis_battery_cradle"]["max_overlap_mm3"],
                   "mm3", "hip yaw 40 deg, beyond the 30 deg range"),
                 m("Stage 4 vs shin frame, exact B-rep", fit["stage4 (R_roll_link) x R_shin_combined"]["max_overlap_mm3"], "mm3",
                   "knee 125 deg, beyond the 95 deg range; touching at 115"),
                 m("Smallest gap Stage 4 to Stage 5", gap["s4s5"], "mm", "exact B-rep, fit poses (roll to -45)"),
                 m("Smallest gap Stage 5 to Stage 6", gap["s5s6"], "mm", "exact B-rep, 74 poses"),
                 m("Smallest gap Stage 5 to roll case", gap["s5case"], "mm", "exact B-rep, 74 poses"),
                 m("Knee motor top to hanging roll case", rep["fit_brep"].get("knee_case_top_to_roll_case_bottom_gap_mm"), "mm", "standing"),
             ],
             conditions={"manifest": rel(MANIFEST), "poses": "hip-shell leg_motion library (151) + GRID (72); fit_brep 74 poses"},
             limitations="Kinematic sweep only: no loads, no compliance. Joint-range sweeps (stress test, not a requirement) "
                         "find two two-joint corners where a thigh shell touches a neighbour: back shell vs the torso back "
                         "shell at hip pitch +45 / roll +25, front shell vs pelvis cell 08 at yaw -30 / pitch -58 "
                         "(cells clear the realistic poses only, owner decision T-jbe85r).",
             source_refs=[report_ref]),
        dict(id="fem", title="Static FEM, new vs today", kind="simulation",
             observation=f"The H-cage thigh connector holds the knee motor from both sides and carries the loads at far "
                         f"lower stress than today's U (roll stall p99 {s4n['roll_stall'][0]} MPa against {s4o['roll_stall'][0]}). "
                         f"The strengthened pitch-to-roll plate (thicker arms, taller wings, held from the pitch output side "
                         f"only: there is no pitch idler) is stronger than today's in the heavy landing case (strength ratio "
                         f"{s5n['heavy_6kg_1p5g'][1]} against {s5o['heavy_6kg_1p5g'][1]}) and under roll-motor stall.",
             measurements=[
                 m("Stage 4 p99 von Mises, recovery 4 kg 1.5 g", s4n["recovery_4kg_1p5g"][0], "MPa", f"today {s4o['recovery_4kg_1p5g'][0]}"),
                 m("Stage 4 p99 von Mises, roll stall", s4n["roll_stall"][0], "MPa", f"today {s4o['roll_stall'][0]}"),
                 m("Stage 4 strength / p99, heavy 6 kg 1.5 g", s4n["heavy_6kg_1p5g"][1], "ratio", f"today {s4o['heavy_6kg_1p5g'][1]}"),
                 m("Stage 5 p99 von Mises, heavy 6 kg 1.5 g", s5n["heavy_6kg_1p5g"][0], "MPa", f"today {s5o['heavy_6kg_1p5g'][0]}"),
                 m("Stage 5 strength / p99, heavy 6 kg 1.5 g", s5n["heavy_6kg_1p5g"][1], "ratio", f"today {s5o['heavy_6kg_1p5g'][1]}"),
                 m("Stage 5 p99 von Mises, roll stall", s5n["roll_stall"][0], "MPa", f"today {s5o['roll_stall'][0]}"),
             ],
             conditions={"material": "PAHT-CF dry, orthotropic, bonded 304 SS sleeves", "mesh": fem["method"],
                         "print_layer_normal": f"Stage 4 new {n4}, today x; Stage 5 new {n5} (best worst-case of strength and "
                                               "layer ratios), today z",
                         "patches": fem["patches"]},
             limitations="Linear static, p99 von Mises of the tetra field; no fatigue, bolt preload or contact. Today's "
                         "Stage 5 is evaluated on its motor-refit STEP.",
             source_refs=[report_ref]),
        dict(id="mass", title="Mass and link mass properties", kind="measurement",
             observation="The H-cage is about 1.6 g heavier than today's thigh connector; the 74.5 g roll case moves from the pitch link to the "
                         "thigh, so the leg below the roll joint gets heavier.",
             measurements=[
                 m("Stage 4 with sleeves", mp["stage4"]["roll_flip_with_sleeves_g"], "g", f"today {mp['stage4']['selected_with_sleeves_g']}"),
                 m("Stage 5 with sleeves", mp["stage5"]["roll_flip_with_sleeves_g"], "g", f"today {mp['stage5']['selected_with_sleeves_g']}"),
                 m("Pitch link", links["R_pitch_link"]["roll_flip"]["mass_g"], "g", f"today {links['R_pitch_link']['selected']['mass_g']}"),
                 m("Roll link (thigh)", links["R_roll_link"]["roll_flip"]["mass_g"], "g", f"today {links['R_roll_link']['selected']['mass_g']}"),
             ],
             conditions={"basis": mass.get("basis")},
             limitations=mass.get("note") or "Horn and idler masses null.",
             source_refs=[report_ref]),
        press_evidence(rep, report_ref),
        thigh_shell_evidence(rep, report_ref),
    ]
    rec = {
        "id": STUDY,
        "study_id": STUDY,
        "title": "Roll motor in the thigh, blue thigh shell",
        "source": src,
        "dashboard": "design",
        "workspace": "Body",
        "robot_sections": ["Hip"],
        "observation": "The hip roll motor hangs down inside the thigh (roll axis unchanged), held by the H-cage thigh "
                       "connector (2 + 2 screws per actuator, owner 2026-11-19); a new pitch-to-roll plate carries its horn and idler. No new overlap in 151 "
                       "realistic poses; the strengthened plate is a little stronger than today's in the heavy landing case. "
                       "A smooth blue thigh shell (two halves per leg) replaces today's thigh covers; it is under owner "
                       "review for its look.",
        "measurements": "; ".join(f"{x['name']}: {x['value']} {x['unit']}" + (f" ({x['condition']})" if x.get("condition") else "")
                                  for x in evidence[0]["measurements"][:1] + evidence[1]["measurements"][3:5]
                                  + evidence[2]["measurements"][2:]),   # the results row stores text; rows live in evidence
        "limitations": "First CAD pass for the owner's hip shell idea (hip-shell D-12/D-13). Knee section to be redesigned "
                       "later. Robot model (URDF) and controller not updated. Not printed or physically tested.",
        "assets": assets,
        "clips": [],
        "design": {
            "id": STUDY,
            "title": "Roll motor in the thigh, blue thigh shell",
            "summary": "Roll motor hangs in the thigh; H-cage thigh connector (2 + 2 M2 screws per actuator, about 1.4 mm "
                       "from the pitch-to-roll arms) and pitch-to-roll plate; smooth blue thigh shell with a calm top edge "
                       "(under owner review) replaces today's thigh covers. Clear in all realistic poses.",
            "visual_kind": "native_CAD_assembly",
            "maturity": "CAD first pass with fit, motion and static FEM checks; no print, no physical test, model not updated.",
            "image_path": rel(ART / "renders" / "thigh-shell-shaded.png"),
            "cad": {
                "model_units": "m",
                "views": [{"label": "Complete body, roll motor in the thigh", "model_path": rel(ART / "assembly-with-full-shell.glb")}],
                "parts_path": rel(MANIFEST),
                "step_path": rel(ART / "connectors-bilateral.step"),
                "report_path": rel(ART / "report.json"),
                "hide_shell_toggle": True,
                "show_shells": True,
                "motion_note": "Joint sliders are inspection commands over the joint ranges declared in the body definitions (joints:), not qualified travel.",
                "joint_ranges": BOM_CONFIG.joint_ranges(),
            },
            "source_refs": [report_ref, {"label": "Decisions register (hip shell)", "path": "robot/body/hip-shell/DECISIONS.md"}],
            "evidence": evidence,
        },
    }
    out = ART / "publication.json"
    out.write_text(json.dumps(rec, indent=1))
    print(out)


def press_evidence(rep, report_ref):
    """Sleeve pressability (owner 2026-11-17): tool path, anvil path and flat counter face for every sleeve."""
    pr = rep["press_roll_flip"]
    def rows(part):
        return pr["parts"][part]["rows"]
    def stats(part, inboard=None):
        r = rows(part)
        if inboard is not None:
            r = [x for x in r if (x["centre_mm"][0] < 50) == inboard]
        cov = [x["counter_face"].get("support_pct", x["counter_face"]["coverage_pct"]) for x in r]
        return dict(n=len(r), tool=max(x["tool_tip_obstruction_mm3"] + x["tool_body_obstruction_mm3"] for x in r),
                    a20=max(x["anvil_path_obstruction_mm3"] for x in r),
                    a10=max(x["anvil_dia10_first15mm_obstruction_mm3"] for x in r),
                    cmin=min(cov), cmax=max(cov), flat=max(x["counter_face"]["flatness_mm"] for x in r))
    s5, s4 = stats("R_hip_pitch_to_roll_roll_flip"), stats("R_thigh_roll_flip")
    sel = rep["press_selected"]["by_subsystem"]
    a = pr["assumptions"]
    return dict(
        id="press", title="Sleeve pressability", kind="analysis",
        observation=f"Pitch-to-roll plate: free tool path, spacer-block support {s5['cmin']}-{s5['cmax']} %. H-cage thigh "
                    f"connector ({s4['n']} sleeves): spacer-block support {s4['cmin']}-{s4['cmax']} % (flat face plus "
                    f"integral wall), tool-path obstruction up to {s4['tool']} mm3 on the worst sleeve. In the selected "
                    f"body nearly every sleeve has a press obstruction or less than 90 % support.",
        measurements=[
            m("Stage 5 tool-path obstruction, worst sleeve", s5["tool"], "mm3", f"{s5['n']} sleeves"),
            m("Stage 5 anvil 10 mm x 15 mm obstruction, worst", s5["a10"], "mm3", "anvil 20 mm x 40 mm: up to " f"{round(s5['a20'])} mm3"),
            m("Stage 5 spacer-block support, worst sleeve", s5["cmin"], "%", f"flatness {s5['flat']} mm"),
            m("Stage 4 (H-cage) tool-path obstruction, worst", s4["tool"], "mm3", f"{s4['n']} sleeves"),
            m("Stage 4 (H-cage) spacer-block support, worst sleeve", s4["cmin"], "%", f"flatness {s4['flat']} mm"),
        ] + [m(f"Selected body, {k}: sleeves with any press obstruction or < 90 % flat face", v["any_issue"], "of " f"{v['sleeves']}")
             for k, v in sel.items()],
        conditions={"tool": f"heated tip {a['tool_tip_dia_mm']} mm for {a['tool_tip_length_mm']} mm, body "
                            f"{a['tool_body_dia_mm']} mm, {a['tool_total_length_mm']} mm long",
                    "anvil": f"face {a['anvil_face_dia_mm']} mm at the exit plane; path {a['anvil_path_dia_mm']} mm x "
                             f"{a['anvil_path_length_mm']} mm (also scored as 10 mm x 15 mm)",
                    "counter_face": f"share of the annulus {a['counter_face_annulus_mm']} mm within "
                                    f"{a['counter_face_tolerance_mm']} mm of the exit plane",
                    "entry_side": "the end with less tool obstruction"},
        limitations="Tool and anvil sizes are assumptions, not a measured press. The annulus out to r 5 mm is a strict "
                    "reading of 'flat counter face'. Checked on the single printed part, before assembly.",
        source_refs=[report_ref])

def thigh_shell_evidence(rep, report_ref):
    """Blue thigh shell (hip-shell D-12/D-13 step 2): mass, wall offset and new overlap over the whole-leg poses."""
    b, c = rep["thigh_shell_build"], rep["thigh_shell_check"]
    parts, off = b["parts"], c["inner_wall_offset_from_covered_parts"]
    real = {k: v for fam, pairs in c["new_overlap_by_family"].items() if fam != "all combinations" for k, v in pairs.items()}
    combos = c["new_overlap_by_family"].get("all combinations", {})
    worst_real = max(real.items(), key=lambda kv: kv[1]["max_new_mm3"]) if real else None
    st = b["settings"]
    return dict(
        id="thigh-shell", title="Blue thigh shell", kind="analysis",
        observation="Two halves per leg on the thigh link replace today's two thigh covers. Round superellipse sections "
                    "from the knee to mid-thigh, closing in around the roll motor case toward the top; the outer skin is one "
                    "smooth spline surface from 25 values, one convex front with no crease. The top "
                    "edge is a smooth, capped curve around the shell (low peaks, gentle slopes) as close as the moving parts allow: it "
                    "wraps the roll motor and connector and dips where the yellow arms swing. Below the Stage 4 bracket the "
                    "back slopes in and flattens to hug the knee motor's back face. Below the widest ring the sides run "
                    "straight down, flaring slightly, and the front comes in, to the shin covers' top outline, so thigh and "
                    "shin read as one shell (owner 2026-11-23). The knee edge is one smooth line in the side view: level 1 mm "
                    "above the shin front cover in front of the knee axis, rising behind it along the grey shin back cover's "
                    "top edge turned to full flexion, about 1 mm from the shin covers over the 0-95 deg knee range. The shin "
                    "back cover's top-back edge is a plane through the knee axis, just under its two old top M3 holes; it is "
                    "held by four PA2x8 self-tapping screws, two per side, through bosses on its side wings into pilot holes "
                    "in the shin frame's thick side walls behind the ankle motor. The core torso shells are left out of the check: they are cut at z 271 and redesigned next.",
        measurements=[
            m("Front half", parts["R_thigh_shell_front"]["mass_g"], "g", "Bambu ABS, per leg"),
            m("Back half", parts["R_thigh_shell_back"]["mass_g"], "g", "per leg"),
            m("Wall", st["wall_mm"], "mm"),
            m("Inner wall offset from covered parts, median", off["R_thigh_shell_front"]["median_mm"], "mm", "front half"),
            m("Inner wall offset from covered parts, min", min(off[k]["min_mm"] for k in off if k.startswith("R_")), "mm"),
            m("New overlap, realistic poses, worst pair", worst_real[1]["max_new_mm3"] if worst_real else 0.0, "mm3",
              f"{c['poses']['library_incl_paths']} poses incl. paths" + (f"; {worst_real[0]}" if worst_real else "")),
            m("New overlap, range-limit combinations, sum of per-pair worst",
              round(sum(v["max_new_mm3"] for v in combos.values()), 1), "mm3",
              f"{c['poses']['range_combinations']} combinations; stress test, not a requirement"),
        ] + ([
            *[m(f"Knee edge height, {w}", b["edges"]["bottom_edge_z"][k], "mm (z)", "thigh frame; knee hub top z 132.4")
              for w, k in (("front", "-90"), ("side", "+0"), ("back", "+90"))],
            m("Shin back cover volume after the cut", rep["shin_back_cover_trim"]["volume_after_mm3"], "mm3",
              f'per leg; before {rep["shin_back_cover_trim"]["volume_before_mm3"]}'),
            m("Shin back cover top edge at its back wall", rep["shin_back_cover_trim"]["top_edge_z_at_y"]["29.4"], "mm (z)",
              "shin frame, standing; today 117.9"),
            *([m("Shin back cover side screws: PA2 thread engagement in the frame wall, min",
                 min(s["thread_engagement_mm"] for s in rep["shin_back_cover_trim"]["mounts"]["sites"]), "mm",
                 "4 PA2x8 per leg, two per side wall; rule of thumb 4-6 mm (2-3 x dia) in plastic"),
               m("Shin back cover side screws: frame wall left behind the pilot hole, min",
                 min(s["pilot_bottom_to_wall_inner_mm"] for s in rep["shin_back_cover_trim"]["mounts"]["sites"]), "mm",
                 "pilot dia 1.7 x 4.5 mm; ankle motor case beyond the wall")]
              if "mounts" in rep["shin_back_cover_trim"] else []),
        ] if "shin_back_cover_trim" in rep else []),
        conditions={"method": c["method"], "roll_range_deg": st["roll_range_deg"], "gap_mm": st["gap_mm"]},
        limitations="Mesh-defined (STL), not yet exact B-rep; no FEM. Sleeve exits on flat lands: press support 84-88 % "
                    "(about 90 % is the ceiling, 10 % of the spacer ring lies past the print bed); 1.0 mm to the ankle "
                    "coupler at its closest (climb step). Thigh design approved by the owner (2026-11-23).",
        source_refs=[report_ref])


if __name__ == "__main__":
    {"package": package, "record": record}[sys.argv[1]]()
