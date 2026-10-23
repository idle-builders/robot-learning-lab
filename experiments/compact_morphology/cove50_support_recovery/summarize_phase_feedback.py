"""Compare frozen COM feedback probes with the selected-body open-loop release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-phase-feedback-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first(rows: list[dict], predicate) -> dict | None:
    return next((row for row in rows if predicate(row)), None)


def summarize(path: Path, controller: str) -> dict:
    d = json.loads(path.read_text())
    assert d["design_pointer_sha256"] == sha(REPO / "robot/ape-current-design.json")
    assert d["body_sha256"] == sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
    assert d["hip_roll_effort_scale_hypothetical"] == 1.0
    assert sha(REPO / d["state_trace"]) == d["state_trace_sha256"]
    assert sha(REPO / d["video"]) == d["video_sha256"]
    rows = d["timeseries"]
    first_rated = first(rows, lambda x: bool(x["rated_leg_joints"]))
    first_overhang = first(rows, lambda x: x["time_s"] >= 3.0 and
                           abs(x["com_minus_stance_world_xy_m"][1]) > .035)
    pair = [d["stance"] == "left", d["stance"] == "right"]
    first_support = first(rows, lambda x: x["feet_grounded"] == pair and x["swing_sole_z_m"] > .025)
    first_cap = first(rows, lambda x: abs(x["stance_roll_feedback_rad"]) >= .179)
    names = d.get("leg_joint_names", [])
    rated = d.get("leg_rated_effort_nm", [])
    effort_peaks = []
    if names and rated:
        for i, name in enumerate(names):
            peak = max(x["leg_effort_nm"][i] for x in rows)
            effort_peaks.append({"joint": name, "peak_effort_nm": round(peak, 4),
                                 "rated_effort_nm": rated[i],
                                 "peak_over_rated": round(peak / rated[i], 4)})
    sample = first(rows, lambda x: x["time_s"] >= 4.2)
    return {
        "case": controller + "/" + d["stance"],
        "controller": controller,
        "stance": d["stance"],
        "report": str(path.relative_to(REPO)),
        "report_sha256": sha(path),
        "trace": d["state_trace"],
        "video": d["video"],
        "gain_rad_per_m": d["stance_com_feedback_gain_rad_per_m"],
        "target_offset_m": d["stance_com_target_offset_m"],
        "prelift_stance_load_share": d["dwell_mean_stance_foot_load_fraction"],
        "duration_s": d["duration_s"],
        "completed_schedule": d["completed_schedule"],
        "longest_single_support_s": d["longest_single_support_s"],
        "rated_leg_fraction": d["rated_leg_fraction"],
        "max_stance_slip_mm": round(1000 * d["max_stance_slip_m"], 2),
        "max_swing_sole_z_mm": round(1000 * d["max_swing_sole_z_m"], 2),
        "max_tilt_deg": max(x["tilt_deg"] for x in rows),
        "first_rated_joint": ", ".join(first_rated["rated_leg_joints"]) if first_rated else None,
        "first_rated_s": first_rated["time_s"] if first_rated else None,
        "first_lateral_overhang_after_lift_s": first_overhang["time_s"] if first_overhang else None,
        "first_exclusive_support_s": first_support["time_s"] if first_support else None,
        "first_feedback_cap_s": first_cap["time_s"] if first_cap else None,
        "com_y_at_4p2_m": sample["com_minus_stance_world_xy_m"][1] if sample else None,
        "feedback_at_4p2_rad": sample["stance_roll_feedback_rad"] if sample else None,
        "self_contact_frames": d["self_contact_frames"],
        "hand_contact_frames": d["hand_contact_frames"],
        "nonsole_floor_contact_frames": d["nonsole_floor_contact_frames"],
        "effort_peaks": effort_peaks,
        "full_gate_pass": d["lift_skill_screen_pass"],
    }


def main() -> None:
    plan_path = ROOT / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["selected_design_pointer_sha256"] == sha(REPO / "robot/ape-current-design.json")
    assert plan["selected_body_sha256"] == sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
    assert plan["probe_source_sha256"] == sha(Path(__file__).with_name("probe_native_feedback.py"))
    cases = []
    for side in ("right", "left"):
        cases.append(summarize(REPO / plan["baseline_reports"][side]["path"], "open-loop"))
    trials = sorted((ROOT / "trials").glob("*.json"))
    assert len(trials) in (6, 8, 10, 12), len(trials)
    for path in trials:
        cases.append(summarize(path, path.stem.split("-")[0]))
    report = {
        "schema": "robot-learning-lab.cove50-phase-feedback-comparison.v1",
        "method": "Authored native MJLab controller diagnostics. Feedback reads simulator true COM and sole; not a deployable or learned policy.",
        "selected_design_pointer_sha256": plan["selected_design_pointer_sha256"],
        "selected_body_sha256": plan["selected_body_sha256"],
        "actuator_catalog_sha256": plan["actuator_catalog_sha256"],
        "plan": {"path": str(plan_path.relative_to(REPO)), "sha256": sha(plan_path)},
        "cases": cases,
    }
    path = ROOT / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(path), "cases": len(cases),
                      "passes": [x["case"] for x in cases if x["full_gate_pass"]]}, indent=2))


if __name__ == "__main__":
    main()
