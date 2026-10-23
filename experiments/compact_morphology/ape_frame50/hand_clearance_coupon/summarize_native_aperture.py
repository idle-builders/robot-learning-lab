"""Score the declared native aperture entry and powered-jaw diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/native-aperture-entry-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "plan.json"
    report_path = OUT / "development/seed-2906.json"
    trace_path = OUT / "development/seed-2906.npz"
    plan = json.loads(plan_path.read_text())
    report = json.loads(report_path.read_text())
    gate = plan["development_gates"]
    phases = report["phases"]
    close_hold = [row for row in report["timeseries"]
                  if row["phase"] in ("close", "hold")]
    both_streak = longest = 0
    for row in close_hold:
        both_streak = both_streak + 1 if row["finger_contact"] and row["palm_contact"] else 0
        longest = max(longest, both_streak)
    two_jaw_s = round(longest * .02, 3)
    close_hold_support = (sum(all(row["feet_grounded"]) for row in close_hold) /
                          len(close_hold)) if close_hold else 0.0
    cube0 = report["entry_cube_xyz_m"]
    xy_drift_mm = max(((((row["cube_xyz_m"][0] - cube0[0]) ** 2 +
                        (row["cube_xyz_m"][1] - cube0[1]) ** 2) ** .5) * 1000
                       for row in close_hold), default=float("inf")) if cube0 else float("inf")
    faults = []

    def require(ok: bool, reason: str) -> None:
        if not ok:
            faults.append(reason)

    reach = phases.get("reach", {})
    entry = phases.get("entry", {})
    hold = phases.get("hold", {})
    require(reach.get("continuous_finger_contact_s", 0) >=
            gate["reach_finger_contact_s_min"], "reach_contact")
    require((report.get("aperture_solve") or {}).get("site_error_mm", float("inf")) <=
            gate["entry_ik_site_error_mm_max"], "ik_site_error")
    require(entry.get("both_feet_fraction", 0) >=
            gate["entry_two_feet_fraction_min"], "entry_support")
    require(entry.get("rated_leg_fraction", 1) <=
            gate["entry_rated_leg_fraction_max"], "entry_rated_leg")
    require(entry.get("max_hand_floor_force_n", float("inf")) <=
            gate["entry_hand_floor_force_n_max"], "entry_hand_floor")
    require(entry.get("self_body_contact_fraction", 1) <=
            gate["entry_self_contact_fraction_max"], "entry_self_contact")
    require(two_jaw_s >= gate["close_hold_two_jaw_contact_s_min"], "two_jaw_contact")
    require(xy_drift_mm <= gate["close_hold_cube_xy_drift_mm_max"], "cube_xy_drift")
    require(hold.get("final_gripper_rad", 0) >=
            gate["hold_actual_gripper_rad_min"], "jaw_tracking")
    require(close_hold_support >= gate["close_hold_two_feet_fraction_min"],
            "close_hold_support")
    require(report["max_abs_unbounded_action"] <= gate["max_abs_unbounded_action"],
            "action_clipping")
    row = {
        "case": "seed-2906", "stage": "development", "body": "unselected narrow-bridge",
        "pass": not faults, "faults": faults,
        "reach_contact_s": reach.get("continuous_finger_contact_s", 0),
        "ik_site_error_mm": (round(report["aperture_solve"]["site_error_mm"], 3)
                             if report.get("aperture_solve") else None),
        "entry_both_feet_fraction": entry.get("both_feet_fraction", 0),
        "entry_rated_leg_fraction": entry.get("rated_leg_fraction", 1),
        "entry_hand_floor_force_n": entry.get("max_hand_floor_force_n"),
        "entry_self_contact_fraction": entry.get("self_body_contact_fraction", 1),
        "close_hold_two_jaw_contact_s": two_jaw_s,
        "close_hold_cube_xy_drift_mm": (round(xy_drift_mm, 3)
                                        if xy_drift_mm != float("inf") else None),
        "close_hold_both_feet_fraction": round(close_hold_support, 4),
        "final_jaw_rad": hold.get("final_gripper_rad", 0),
        "max_abs_unbounded_action": report["max_abs_unbounded_action"],
        "cube_entry_z_m": cube0[2],
        "cube_final_z_m": hold.get("cube_end_z_m"),
        "native_report": str(report_path.relative_to(ROOT)),
        "native_trace": str(trace_path.relative_to(ROOT)),
    }
    result = {
        "schema": "robot-learning-lab.cove50-native-aperture-entry-result.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "plan_sha256": sha(plan_path),
        "evaluator_sha256": report["evaluator_sha256"],
        "candidate_body_sha256": report["body_sha256"],
        "native_report_sha256": sha(report_path),
        "native_trace_sha256": sha(trace_path),
        "summary_script_sha256": sha(Path(__file__)),
        "development_pass": row["pass"],
        "case": row,
        "qualification_limit": plan["limits"],
    }
    (OUT / "evaluation-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-native-aperture-entry-matrix.v1",
        "cases": [row]}, indent=2) + "\n")
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
