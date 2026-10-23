"""Audit paired whole-task pose-source ablation and actual standoff geometry."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics

import audit_scan30_unified_policy as whole_task
from freeze_scan20_shared_policy_sources import sha
import run_scan30_pose_bias as run


COMPARISON = run.ART / "scan30_pose_bias_comparison.json"


def episode_metrics(row: dict, band: tuple[float, float]) -> dict:
    final = ((json.loads((run.ROOT / row["show_summary"]).read_text()).get(
        "action") or {}) if row.get("show_summary") else
        json.loads((run.ROOT / row["base_summary"]).read_text())
        if row.get("base_summary") else {})
    walk_path = final.get("walk_trace")
    walk = json.loads((run.ROOT / walk_path).read_text()) if walk_path else None
    actions = walk["actions"] if walk else []
    packets = [a["pose_increment_packet"] for a in actions
               if a.get("pose_increment_packet")]
    accepted = [p for p in packets if p.get("accepted")]
    corrected = [a["range_pose_correction"] for a in actions
                 if (a.get("range_pose_correction") or {}).get("accepted")]
    nonzero_corrected = [c for c in corrected if sum(float(value) ** 2
                         for value in c["correction_xy_m"]) > 1e-8]
    gap = final.get("evaluator_final_target_gap_m")
    return {"result": row["result"], "walk_result": row.get("walk_result"),
        "motion_authorized": bool(row.get("motion_authorized")),
        "final_action": row.get("final_action"),
        "actual_final_target_gap_m": gap,
        "actual_standoff_in_band": bool(gap is not None and band[0] <= gap <=
                                        band[1]),
        "max_evaluator_pose_error_m": (walk or {}).get(
            "max_evaluator_capture_pose_error_m"),
        "accepted_biased_increments": len(accepted),
        "all_increments_have_frozen_scale": all(p.get(
            "injected_odom_scale") == run.BIAS_SCALE for p in accepted),
        "accepted_range_corrections": len(corrected),
        "nonzero_range_corrections": len(nonzero_corrected),
        "walk_trace": walk_path}


def main() -> None:
    frozen = json.loads(run.PRECOMMIT.read_text())
    band = tuple(frozen["evaluator_standoff_band_m"])
    audits = {}
    arm_rows = {}
    for arm in run.ARMS:
        whole_task.PRECOMMIT = run.PRECOMMIT
        whole_task.REPORT = run.ART / f"scan30_pose_bias_{arm}_report.json"
        whole_task.OUT = run.OUT / arm
        whole_task.AUDIT = run.ART / f"scan30_pose_bias_{arm}_audit.json"
        whole_task.main()
        audits[arm] = json.loads(whole_task.AUDIT.read_text())
        arm_rows[arm] = json.loads(whole_task.REPORT.read_text())["rows"]
    prior = json.loads(run.BASELINE.read_text())["rows"]
    issues = [f"{arm}: {issue}" for arm in run.ARMS
              for issue in audits[arm]["issues"]]
    if sha(run.BASELINE) != frozen["baseline_report_sha256"]:
        issues.append("baseline report changed")
    pairs = []
    for index, case in enumerate(frozen["cases"]):
        key = case["scene_key"]
        if prior[index]["scene_key"] != key or any(arm_rows[arm][index][
                "scene_key"] != key for arm in run.ARMS):
            issues.append(f"paired order differs: {key}")
            continue
        pair = {"scene_key": key, "capture_group": case["capture_group"],
                "baseline_result": prior[index]["result"],
                "baseline": episode_metrics(prior[index], band)}
        for arm in run.ARMS:
            metric = episode_metrics(arm_rows[arm][index], band)
            pair[arm] = metric
            if not metric["all_increments_have_frozen_scale"]:
                issues.append(f"missing injected increment scale: {arm}/{key}")
            if arm == run.ARMS[0] and metric[
                    "accepted_range_corrections"]:
                issues.append(f"disabled arm corrected pose: {key}")
        pairs.append(pair)
    summaries = {}
    for arm in run.ARMS:
        metrics = [p[arm] for p in pairs]
        gaps = [m["actual_final_target_gap_m"] for m in metrics
                if m["actual_final_target_gap_m"] is not None]
        pose = [m["max_evaluator_pose_error_m"] for m in metrics
                if m["max_evaluator_pose_error_m"] is not None]
        summaries[arm] = {"outcomes": dict(Counter(m["result"] for m in metrics)),
            "motion_authorized": sum(m["motion_authorized"] for m in metrics),
            "biased_motion_exposed": sum(m["accepted_biased_increments"] > 0
                                         for m in metrics),
            "accepted_biased_increments": sum(m["accepted_biased_increments"]
                                              for m in metrics),
            "accepted_range_corrections": sum(m["accepted_range_corrections"]
                                              for m in metrics),
            "nonzero_range_corrections": sum(m["nonzero_range_corrections"]
                                             for m in metrics),
            "completed_with_actual_standoff_in_band": sum(m["result"] ==
                "completed" and m["actual_standoff_in_band"] for m in metrics),
            "completed_with_actual_standoff_outside_band": sum(m["result"] ==
                "completed" and not m["actual_standoff_in_band"] for m in metrics),
            "median_actual_final_gap_m": statistics.median(gaps) if gaps else None,
            "median_max_pose_error_m": statistics.median(pose) if pose else None,
            "wrong_candidate_offers": audits[arm][
                "wrong_candidate_offers_in_stage_traces"],
            "videos_decoded": audits[arm]["videos_decoded"]}
    baseline_metrics = [p["baseline"] for p in pairs]
    baseline_gaps = [m["actual_final_target_gap_m"] for m in baseline_metrics
                     if m["actual_final_target_gap_m"] is not None]
    baseline_summary = {"outcomes": dict(Counter(m["result"] for m in
                                            baseline_metrics)),
        "completed_with_actual_standoff_in_band": sum(m["result"] ==
            "completed" and m["actual_standoff_in_band"] for m in
            baseline_metrics),
        "completed_with_actual_standoff_outside_band": sum(m["result"] ==
            "completed" and not m["actual_standoff_in_band"] for m in
            baseline_metrics),
        "median_actual_final_gap_m": statistics.median(baseline_gaps)
            if baseline_gaps else None}
    result = {"schema": "ape.scan30-pose-bias-comparison.v1",
        "precommit_sha256": sha(run.PRECOMMIT),
        "baseline_report_sha256": sha(run.BASELINE),
        "matched_rooms": len(pairs),
        "evaluator_standoff_band_m": band,
        "baseline_outcomes": dict(Counter(r["result"] for r in prior)),
        "baseline": baseline_summary,
        "arms": summaries, "issues": issues, "pairs": pairs,
        "limits": frozen["limits"]}
    COMPARISON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("matched_rooms",
        "baseline_outcomes", "arms", "issues")}, indent=2))


if __name__ == "__main__":
    main()
