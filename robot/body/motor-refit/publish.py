"""Publish the motor refit as the next version of the complete printed body (study cove50-complete-fdm-body).

  .venv/bin/python -B robot/body/motor-refit/publish.py   # writes the record JSON, then
  .venv/bin/python infra/dashboard/results_hub.py publish artifacts/robot/body/motor-refit/record.json
The study sources must be committed first; their commit is the version (docs/operations/design-dashboard.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "artifacts/robot/body/motor-refit"



def generated_bom(manifest=None):
    """The BOM generated from a body manifest by robot/body/bom/build.py (default: the selected body); the manifest
    itself carries no BOM (owner 2026-11-20)."""
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import build as bom_build
    return bom_build.design_bom_for(manifest)

def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


# Known errors in the stored FEM numbers (Idle T-awqa72, 2026-11-20): shown next to the values instead of re-running retired parts.
FEM_NOTES = {
    "hip_pitch_to_roll": ("refit values superseded, not valid: computed with the steel sleeves 4 mm off the part (fem_refit.py step_in_r1_frame "
                          "shifted part and sleeves by -4 mm and ignored native_step_offset_mm); with the sleeves in place the "
                          "displacement is about 55-59 % lower. The part is replaced in the selected hip by the roll-flip Stage 5 "
                          "(R_hip_pitch_to_roll_roll_flip), whose FEM is in the study hip-roll-motor-in-thigh"),
}


def main():
    rep = json.load(open(OUT / "report.json")); man = json.load(open(OUT / "assembly-parts.json"))
    fem = rep["fem"]; m = rep["mass"]
    fem_line = "; ".join(f"{k.replace('_', ' ')} {v['r1']['body_p99_strength_ratio_PAHT_CF_dry']} -> {v['refit']['body_p99_strength_ratio_PAHT_CF_dry']}"
                         + (f" (stall {v['r1']['stall_p99_strength_ratio_PAHT_CF_dry']} -> {v['refit']['stall_p99_strength_ratio_PAHT_CF_dry']})"
                            if v['refit'].get('stall_p99_strength_ratio_PAHT_CF_dry') else "")
                         + (f" [{FEM_NOTES[k]}]" if k in FEM_NOTES else "") for k, v in fem.items() if k != "pelvis_cradle")
    summary = ("Locked three-tier motors fitted with minimal edits: ST3025 at hip pitch, knee and ankle pitch; STS3250 at hip roll, "
               "shoulders and elbows; C047 at hip yaw and ankle roll; neck stays XM335. Hips +4 mm per side, knees and ankles about "
               "14 mm wider, lower-pelvis cells laid horizontally on the midline.")
    measurements = (f"Printed structure {m['printed_r1_g']} -> {m['printed_refit_g']} g; leg and arm servos {m['servos_20_leg_arm_joints_r1_g']} -> {m['servos_20_leg_arm_joints_refit_g']} g. "
                    f"Servo interfaces: " + ", ".join(f"{g['part'].replace('R_', '').replace('_combined', '')} {g['min_gap_mm']} mm" for g in rep["interface_gaps"]) + " (0 mm3 overlap). "
                    f"FEM p99 strength ratio PAHT-CF dry, r1 -> refit, worst body-weight case (stall): {fem_line}. "
                    f"Collision sweeps over full joint ranges with neighbouring joints, compared with r1 measured the same way: see the evidence cards.")
    assets = []
    def add(i, title, kind, path):
        if (ROOT / path).exists():
            assets.append(dict(id=i, title=title, kind=kind, path=path))
    add("neutral", "Neutral assembly (shells, electronics, servos)", "model", rel(OUT / "assembly-with-full-shell.glb"))
    add("step", "Printed structure and sleeves, native STEP", "data", rel(OUT / "complete-fdm-structural-skeleton.step"))
    add("parts", "Named parts and joint metadata", "data", rel(OUT / "assembly-parts.json"))
    add("report", "Refit report: masses, gaps, FEM, collision sweeps", "data", rel(OUT / "report.json"))
    add("self-collision", "Self-collision limits for the WBC (corner contacts between limbs / limb and body)", "data", rel(OUT / "self-collision-limits.json"))
    for n, t in (("refit-vs-r1-shells", "r1 and refit, front and side, shells on"), ("refit-vs-r1-mechanism", "r1 and refit, front and side, mechanism"),
                 ("refit-vs-r1-pelvis", "Pelvis and hips: r1 and refit")):
        add(n, t, "image", rel(OUT / "renders" / f"{n}.png"))
    for p in sorted((OUT / "parts").glob("*.step")):
        add("step-" + p.stem, p.stem, "data", rel(p))
    evidence = [
        dict(id="fem", title="Static FEM, r1 against refit", kind="simulation", summary=fem_line,
             measurements=[dict(name=f"{k.replace('_', ' ')} {w}: {label}" + (" (superseded, not valid: sleeves 4 mm off, see T-awqa72)" if k in FEM_NOTES and w == "refit" else ""),
                                value=v[w][field], unit="reference strength / p99 von Mises")
                           for k, v in fem.items() for w in ("r1", "refit")
                           for label, field in (("worst body-weight case, PAHT-CF dry", "body_p99_strength_ratio_PAHT_CF_dry"),
                                                ("servo stall, PAHT-CF dry", "stall_p99_strength_ratio_PAHT_CF_dry"),
                                                ("worst case, PETG-CF", "min_p99_strength_ratio_PETG_CF"))
                           if v[w].get(field) is not None and not (k == "pelvis_cradle" and w == "r1")],
             source_refs=[dict(path=rel(OUT / "report.json"))]),
        dict(id="collisions", title="Collision over full joint ranges, r1 against refit", kind="analysis",
             summary="Each changed joint with its neighbouring joints over their ranges; pairs with at least 0.5 mm3 overlap in either body "
                     "(hand and foot-sensor pads omitted). Full tables in the report.",
             measurements=[dict(name=f"{j}: {p['a']} x {p['b']} ({body})", value=p[f'{body}_mm3'], unit="mm3")
                           for j, d in rep["collision_sweeps"].items() for p in d["pairs"][:8] for body in ("refit", "r1")],
             source_refs=[dict(path=rel(OUT / "report.json"))]),
        dict(id="final-sweeps", title="Final body: every joint swept, contacts by kind", kind="analysis",
             summary="Design contacts (inside one limb or static) are the defects to remove; self-collision limits are corner contacts "
                     "between limbs or a limb and the body, handled by the controller (owner decision); servo-model contacts are the "
                     "reference servo's own case against its output.",
             measurements=[dict(name=f"{j}: {k}", value=v, unit="pairs") for j, d in rep.get("final_sweep_summary", {}).items() for k, v in d.items()]
                          + [dict(name=f"remaining design contact {r['joint']}: {r['a']} x {r['b']}", value=r["overlap_mm3"], unit="mm3")
                             for r in rep.get("design_contacts_remaining", [])],
             source_refs=[dict(path=rel(OUT / "report.json")), dict(path=rel(OUT / "self-collision-limits.json"))]),
        dict(id="gaps", title="Servo interface gaps", kind="analysis",
             measurements=[dict(name=f"{g['part']} to {g['servo_part']}", value=g["min_gap_mm"], unit="mm") for g in rep["interface_gaps"]]),
        dict(id="open", title="Open items and r1 defects found", kind="analysis",
             summary=" | ".join(rep["open_items"] + rep["r1_defects_found"])),
    ]
    record = {
        "study_id": "cove50-complete-fdm-body", "id": "cove50-complete-fdm-body", "title": "Complete printed body",
        "source": ["robot/body/motor-refit/"], "dashboard": "design", "workspace": "Body",
        "robot_sections": ["Whole body", "Pelvis", "Hip", "Knee", "Ankle / foot", "Torso / shell", "Electronics"],
        "area": "Complete printed body: locked three-tier motor refit", "observation": summary, "measurements": measurements,
        "limitations": ("Linear static FEM with clamped bolt patches (stall peaks are bolt-patch concentrations); kinematic collision checks "
                        "without loads or compliance; print tolerances, sleeve retention and harness routing unmeasured; the WBC model "
                        "still uses R3 hip spacing. The refit hip pitch-to-roll FEM figures are not valid (sleeves 4 mm off the part) and "
                        "that part is replaced by the roll-flip Stage 5 (study hip-roll-motor-in-thigh)."),
        "provenance": [rel(OUT / "report.json"), rel(OUT / "assembly-parts.json"), "robot/model/motor-map/motor-map.json"],
        "analysis_provenance": dict(idle_task="T-jrnxfn", decisions=list(rep["decisions"]), r1_commit=rep["source_r1_commit"], remote_jobs=rep["remote_jobs"]),
        "assets": assets, "clips": [],
        "design": {
            "id": "cove50-complete-fdm-body", "title": "Complete printed body", "summary": summary,
            "description": "Edits to the r1 printed parts (commit 48c38ceda) by exact B-rep stretch, pad and pocket operations; the r1 version stays in the version menu.",
            "image_path": rel(OUT / "renders" / "refit-vs-r1-shells.png"), "visual_kind": "native_CAD_assembly",
            "maturity": "CAD refit with FEM and collision evidence; not selected (robot/design.json unchanged)",
            "cad": {"model_units": "m", "views": [{"label": "Complete body with the three-tier motors", "model_path": rel(OUT / "assembly-with-full-shell.glb")}],
                    "parts_path": rel(OUT / "assembly-parts.json"), "step_path": rel(OUT / "complete-fdm-structural-skeleton.step"),
                    "report_path": rel(OUT / "report.json"), "hide_shell_toggle": True, "show_shells": False,
                    # The r1 parts carry these overlay ids; each needs a declared toggle.
                    "overlays": [{"id": "actuators_and_hardware", "label": "Motors, horns, idlers and fasteners", "visible": True},
                                 {"id": "electronics_and_battery", "label": "Battery, heat fins, electronics and sensors", "visible": True}],
                    "motion_note": "Joint sliders are inspection commands over the r1 ranges, not qualified travel.",
                    "bom": generated_bom(), "joints": [{k: v for k, v in j.items() if k != "active"} for j in man["joints"]], "controls": man["controls"]},
            "evidence": evidence,
        },
    }
    (OUT / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    print("record", rel(OUT / "record.json"), len(assets), "assets")


if __name__ == "__main__":
    main()
