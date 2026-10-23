"""Independent full-horizon geometry, hold-order and video audit for V460–V461."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-robocasa-layout4-returning-prop-v460/report.json"
POLICY = ROOT / "artifacts/ape-robocasa-layout4-wait-remap-v461/report.json"
OUT = ROOT / "artifacts/ape-robocasa-layout4-wait-remap-v461"


def evaluator_prop_at(case, time_s):
    points = case["waypoints"]
    if time_s <= points[0][0]:
        return np.asarray(points[0][1:3], dtype=float)
    for start, end in zip(points, points[1:]):
        if time_s <= end[0]:
            fraction = (time_s - start[0]) / (end[0] - start[0])
            return np.asarray(start[1:3]) + fraction * (
                np.asarray(end[1:3]) - np.asarray(start[1:3]))
    return np.asarray(points[-1][1:3], dtype=float)


def decision_contract(case):
    trace = case["episode"]["trace"]
    decisions = case["decision_history"]
    if len(trace) != len(decisions):
        return {"pass": False, "reason": "decision/trace length mismatch"}
    holds = []
    for row, decision in zip(trace, decisions):
        if row["plan_status"] != decision["decision"] or abs(row["time_s"] - decision["time_s"]) > 1e-6:
            return {"pass": False, "reason": "decision/trace mismatch"}
        if decision["decision"] == "hold_and_reobserve_moving_object":
            requested = np.asarray(row["requested_xy_m"])
            estimate = np.asarray(row["estimated_xy_m"])
            if np.linalg.norm(requested - estimate) > 1e-9:
                return {"pass": False, "reason": "hold requested translation"}
            if not all(not sub["stopped_before_move"] for sub in row["motion_substeps"]):
                return {"pass": False, "reason": "hold guard interruption"}
            holds.append(row["time_s"])
    return {"pass": True, "hold_count": len(holds),
            "first_hold_s": min(holds) if holds else None,
            "last_hold_s": max(holds) if holds else None,
            "fresh_remap_s": next((d["time_s"] for d in decisions
                                   if d["mode"] == "navigate_after_fresh_remap"), None),
            "owner_help_s": next((d["time_s"] for d in decisions
                                  if d["decision"] == "ask_owner_clear_corridor"), None)}


def main():
    base = json.loads(BASE.read_text())
    policy = json.loads(POLICY.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    common_audit.prop_at = evaluator_prop_at
    results = []
    for label, report in (("baseline", base), ("wait_remap", policy)):
        for case in report["cases"]:
            audit = common_audit.audit_case(case)
            audit["variant"] = label
            if label == "wait_remap":
                audit["decision_contract"] = decision_contract(case)
            results.append(audit)
            print(label, audit["scenario_id"], audit["outcome"],
                  "gap", round(audit["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
                  "prop", round(audit["nearest_ape_prop"]["signed_gap_m"], 3),
                  "phase", audit["nearest_ape_any"]["phase"], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "baseline_report_sha256": sha256(BASE),
              "policy_report_sha256": sha256(POLICY),
              "config_source_sha_pass": all(report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_returning_prop_v460.json"))
                  for report in (base, policy)),
              "baseline_source_sha_pass": base["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_returning_prop_v460.py")),
              "policy_source_sha_pass": policy["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_wait_and_remap_v461.py")),
              "tracker_source_sha_pass": policy["tracker_source_sha256"] == sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "prior_map_sha_pass": policy["prior_map_sha256"] == sha256(
                  ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
              "all_video_pass": all(c["video_sha_pass"] and c["video_frames_pass"]
                                    for c in results),
              "all_prop_paths_nonoverlap": all(c["prop_fixture_nonoverlap_pass"]
                                               for c in results),
              "all_policy_decisions_pass": all(c["decision_contract"]["pass"]
                                               for c in results if c["variant"] == "wait_remap"),
              "cases": results}
    (OUT / "audit-v462.json").write_text(json.dumps(output, indent=2) + "\n")
    print("all_video_pass", output["all_video_pass"],
          "all_prop_paths_nonoverlap", output["all_prop_paths_nonoverlap"],
          "all_policy_decisions_pass", output["all_policy_decisions_pass"], flush=True)


if __name__ == "__main__":
    main()
