"""Apply the declared powered-route gates to the retained 50 Hz report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/native-aperture-entry-v2"
ROUTE = ("retreat", "open", "detour_1", "detour_2", "aperture", "close", "hold")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "plan.json"
    report_path = OUT / "development/seed-2906.json"
    trace_path = OUT / "development/seed-2906.npz"
    plan = json.loads(plan_path.read_text())
    report = json.loads(report_path.read_text())
    gate = plan["development_gates"]
    rows = [r for r in report["timeseries"] if r["phase"] in ROUTE]
    before_close = [r for r in rows if r["phase"] not in ("close", "hold")]
    close_hold = [r for r in rows if r["phase"] in ("close", "hold")]
    both = [r["finger_contact"] and r["palm_contact"] for r in close_hold]
    streak = longest = 0
    for contact in both:
        streak = streak + 1 if contact else 0
        longest = max(longest, streak)
    cube0 = report["route_cube_xyz_m"]

    def drift(part: list[dict]) -> float:
        return max((sum((r["cube_xyz_m"][axis] - cube0[axis]) ** 2 for axis in (0, 1)) ** .5 * 1000
                    for r in part), default=0.0)

    measures = {
        "source_arm_max_abs_error_rad": round(report["route_source_check"]["arm_max_abs_error_rad"], 8),
        "source_cube_xyz_error_mm": round(report["route_source_check"]["cube_xyz_error_mm"], 4),
        "static_goal_site_error_mm": json.loads((ROOT / plan["route_evidence"]).read_text())["goal_site_error_mm"],
        "route_both_feet_fraction": round(sum(all(r["feet_grounded"]) for r in rows) / len(rows), 4),
        "route_rated_leg_fraction": round(sum(r["leg_at_rated"] for r in rows) / len(rows), 4),
        "route_max_hand_floor_force_n": max(r["hand_floor_force_n"] for r in rows),
        "route_self_contact_fraction": round(sum(r["self_body_contact"] for r in rows) / len(rows), 4),
        "cube_xy_drift_before_close_mm": round(drift(before_close), 3),
        "close_hold_two_jaw_contact_s": round(longest * .02, 3),
        "close_hold_cube_xy_drift_mm": round(drift(close_hold), 3),
        "final_actual_jaw_rad": report["phases"]["hold"]["final_gripper_rad"],
        "max_abs_unbounded_action": report["max_abs_unbounded_action"],
    }
    checks = {
        "precontact_source_finger_contact": report["route_source_check"]["finger_contact"] == gate["precontact_source_finger_contact"],
        "precontact_source_palm_contact": report["route_source_check"]["palm_contact"] == gate["precontact_source_palm_contact"],
        "ik_goal_site_error": measures["static_goal_site_error_mm"] <= gate["ik_goal_site_error_mm_max"],
        "route_two_feet": measures["route_both_feet_fraction"] >= gate["route_two_feet_fraction_min"],
        "route_rated_leg": measures["route_rated_leg_fraction"] <= gate["route_rated_leg_fraction_max"],
        "route_hand_floor": measures["route_max_hand_floor_force_n"] <= gate["route_hand_floor_force_n_max"],
        "route_self_contact": measures["route_self_contact_fraction"] <= gate["route_self_contact_fraction_max"],
        "cube_before_close": measures["cube_xy_drift_before_close_mm"] <= gate["cube_xy_drift_before_close_mm_max"],
        "two_jaw_contact": measures["close_hold_two_jaw_contact_s"] >= gate["close_hold_two_jaw_contact_s_min"],
        "close_hold_cube_drift": measures["close_hold_cube_xy_drift_mm"] <= gate["close_hold_cube_xy_drift_mm_max"],
        "jaw_tracking": measures["final_actual_jaw_rad"] >= gate["final_actual_jaw_rad_min"],
        "action_clipping": measures["max_abs_unbounded_action"] <= gate["max_abs_unbounded_action"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    event_times = {
        "first_cube_drift_over_5_mm_s": next((r["time_s"] for r in rows if
            sum((r["cube_xyz_m"][a] - cube0[a]) ** 2 for a in (0, 1)) ** .5 > .005), None),
        "first_rated_leg_s": next((r["time_s"] for r in rows if r["leg_at_rated"]), None),
        "first_foot_support_loss_s": next((r["time_s"] for r in rows if not all(r["feet_grounded"])), None),
        "first_hand_floor_contact_s": next((r["time_s"] for r in rows if r["hand_floor_force_n"] > 1), None),
        "first_self_contact_s": next((r["time_s"] for r in rows if r["self_body_contact"]), None),
    }
    case = {"case": "seed-2906", "stage": "development", "body": "unselected narrow-bridge",
            "pass": not failed, "failed_gates": failed, "measures": measures,
            "event_times": event_times,
            "native_report": str(report_path.relative_to(ROOT)),
            "native_trace": str(trace_path.relative_to(ROOT))}
    result = {
        "schema": "robot-learning-lab.cove50-native-aperture-route-result.v2",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "plan_sha256": sha(plan_path), "evaluator_sha256": report["evaluator_sha256"],
        "candidate_body_sha256": report["body_sha256"],
        "native_report_sha256": sha(report_path), "native_trace_sha256": sha(trace_path),
        "summary_script_sha256": sha(Path(__file__)),
        "development_pass": not failed, "case": case,
        "qualification_limit": plan["limits"],
    }
    (OUT / "evaluation-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-native-aperture-route-matrix.v2",
        "cases": [case]}, indent=2) + "\n")
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
