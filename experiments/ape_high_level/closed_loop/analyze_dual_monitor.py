"""Score the frozen v18 furnished-room raw-proximity safety comparison."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

from analyze_recovery import lost_view_approach_violations


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v18"
SEEDS = {77, 78, 79, 80}
SCENARIOS = ("furnished_still", "furnished_moving")
CONFIGS = ("chest_rgbd_range_monitored", "chest_rgbd_range_dual_monitor")


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    cases = defaultdict(list)
    for path in RUN.glob("*/trace.json"):
        row = json.loads(path.read_text())
        if row["seed"] in SEEDS:
            cases[row["scenario"], row["config"]].append(row)
    groups = {}
    for scenario in SCENARIOS:
        for config in CONFIGS:
            rows = cases[scenario, config]
            present = [row for row in rows if not row["absent"]]
            absent = [row for row in rows if row["absent"]]
            reached = [row for row in present if row["outcome"] == "approach_stopped"
                       and row["target_center_distance_m"] is not None
                       and .25 <= row["target_center_distance_m"] <= .4
                       and not row["unsafe_obstacle_overlap"]
                       and (scenario != "furnished_moving" or
                            row.get("post_approach_dwell_s", 0) >= 2.0)]
            groups[f"{scenario}:{config}"] = {
                "present_n": len(present), "absent_n": len(absent),
                "safe_approaches_after_dwell": len(reached),
                "path_overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
                "min_path_clearance_m": min((row[
                    "minimum_robot_obstacle_clearance_m"] for row in present), default=None),
                "near_contacts_under_5cm": sum(row[
                    "minimum_robot_obstacle_clearance_m"] < .05 for row in present),
                "motion_interruptions": sum(row.get(
                    "continuous_action_interrupted_evaluator_only", False) for row in present),
                "total_evasions": sum(row.get("safety_evasions", 0) for row in present),
                "max_evasions": max((row.get("safety_evasions", 0) for row in present),
                                     default=0),
                "total_waits": sum(row.get("dynamic_obstacle_waits", 0) for row in present),
                "incomplete_watch_without_contact": sum(
                    row["outcome"] != "sim_collision_failure" and
                    row.get("post_decision_watch_s", 0) < 2.0 for row in present),
                "no_stale_approach_violations": sum(
                    lost_view_approach_violations(row["events"]) for row in present),
                "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                    "actual_final_xy_m_evaluator_only"]) for row in absent),
                "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
                "mean_present_end_time_s": mean([row["events"][-1]["t_s"]
                                                  for row in present]),
                "outcomes": {str(row["seed"]): row["outcome"] for row in present},
            }
    audit = json.loads((RUN / "furnished-geometry-audit.json").read_text())
    routes = json.loads((RUN / "furnished-route-feasibility-audit.json").read_text())
    static = groups["furnished_still:chest_rgbd_range_dual_monitor"]
    moving = groups["furnished_moving:chest_rgbd_range_dual_monitor"]
    dual = [static, moving]
    checks = {
        "all_paired_cases_present": all(group["present_n"] == 4 and group["absent_n"] == 4
                                    for group in groups.values()),
        "valid_trajectory": audit["all_valid"],
        "route_screen_has_5cm_path": all(row["five_cm_margin"]["route_exists"]
                                          for row in routes["rows"]),
        "dual_zero_path_overlap": all(group["path_overlaps"] == 0 for group in dual),
        "dual_min_5cm_clearance": all(group["min_path_clearance_m"] is not None and
                                      group["min_path_clearance_m"] >= .05
                                      for group in dual),
        "dual_zero_absent_motion": all(group["absent_motion"] == 0 for group in dual),
        "dual_no_stale_approach": all(group["no_stale_approach_violations"] == 0
                                      for group in dual),
        "dual_no_static_false_evasion": static["total_evasions"] == 0,
        "dual_bounded_evasion": all(group["max_evasions"] <= 3 for group in dual),
        "dual_completed_safety_watch": all(group["incomplete_watch_without_contact"] == 0
                                           for group in dual),
        "dual_static_progress": static["safe_approaches_after_dwell"] == 4,
    }
    report = {"seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "criteria": "docs/research/ape-raw-proximity-monitor-v18.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, group in groups.items():
        print(name, "n", group["present_n"], "/", group["absent_n"],
              "safe-after-dwell", group["safe_approaches_after_dwell"],
              "overlap", group["path_overlaps"], "evasion", group["total_evasions"],
              "outcomes", group["outcomes"])
    print("CHECKS", checks)


if __name__ == "__main__":
    main()
