"""Score the frozen v20 10 Hz move-and-wait range guard experiment."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

from analyze_recovery import lost_view_approach_violations


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v20"
SEEDS = {85, 86, 87, 88}
SCENARIOS = ("furnished_still", "furnished_moving")
CONFIGS = ("chest_rgbd_range_midstep_guard", "chest_rgbd_range_move_wait_guard")


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
            stops = [e for row in present for e in row["events"]
                     if e["action"] == "midstep_actor_safety_stop"]
            groups[f"{scenario}:{config}"] = {
                "present_n": len(present), "absent_n": len(absent),
                "safe_approaches_after_dwell": len(reached),
                "path_overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
                "min_path_clearance_m": min((row[
                    "minimum_robot_obstacle_clearance_m"] for row in present), default=None),
                "near_contacts_under_5cm": sum(row[
                    "minimum_robot_obstacle_clearance_m"] < .05 for row in present),
                "midstep_stops": len(stops),
                "guard_stop_phases": {phase: sum(e.get("guard_phase") == phase
                                                for e in stops)
                                      for phase in ("move", "wait", "post_decision_wait")},
                "midstep_stop_times_s": [e["t_s"] for e in stops],
                "min_clearance_at_midstep_stop_m": min((e[
                    "path_clearance_m_evaluator_only"] for e in stops), default=None),
                "total_evasions": sum(row.get("safety_evasions", 0) for row in present),
                "max_evasions": max((row.get("safety_evasions", 0) for row in present),
                                     default=0),
                "incomplete_watch_without_contact": sum(
                    row["outcome"] != "sim_collision_failure" and
                    row.get("post_decision_watch_s", 0) < 2.0 for row in present),
                "stale_approach_violations": sum(
                    lost_view_approach_violations(row["events"]) for row in present),
                "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                    "actual_final_xy_m_evaluator_only"]) for row in absent),
                "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
                "mean_present_end_time_s": (statistics.mean(
                    row["events"][-1]["t_s"] for row in present) if present else None),
                "outcomes": {str(row["seed"]): row["outcome"] for row in present},
            }
    audit = json.loads((RUN / "furnished-geometry-audit.json").read_text())
    routes = json.loads((RUN / "furnished-route-feasibility-audit.json").read_text())
    static = groups["furnished_still:chest_rgbd_range_move_wait_guard"]
    moving = groups["furnished_moving:chest_rgbd_range_move_wait_guard"]
    guard = [static, moving]
    paired_baseline = {row["seed"]: row for row in cases[
        "furnished_moving", "chest_rgbd_range_midstep_guard"] if not row["absent"]}
    paired_guard = {row["seed"]: row for row in cases[
        "furnished_moving", "chest_rgbd_range_move_wait_guard"] if not row["absent"]}
    challenging = [seed for seed, row in paired_baseline.items()
                   if row["minimum_robot_obstacle_clearance_m"] < .05]
    rescued = [seed for seed in challenging if seed in paired_guard and
               paired_guard[seed]["minimum_robot_obstacle_clearance_m"] >= .05 and
               any(e["action"] == "midstep_actor_safety_stop"
                   for e in paired_guard[seed]["events"])]
    checks = {
        "all_paired_cases_present": all(group["present_n"] == 4 and group["absent_n"] == 4
                                    for group in groups.values()),
        "valid_trajectory": audit["all_valid"],
        "route_screen_has_5cm_path": all(row["five_cm_margin"]["route_exists"]
                                          for row in routes["rows"]),
        "guard_zero_path_overlap": all(group["path_overlaps"] == 0 for group in guard),
        "guard_min_5cm_clearance": all(group["min_path_clearance_m"] is not None and
                                       group["min_path_clearance_m"] >= .05
                                       for group in guard),
        "guard_no_static_false_stop": static["midstep_stops"] == 0,
        "guard_static_progress": static["safe_approaches_after_dwell"] == 4,
        "guard_moving_progress": moving["safe_approaches_after_dwell"] >= 2,
        "guard_zero_absent_motion": all(group["absent_motion"] == 0 for group in guard),
        "guard_no_stale_approach": all(group["stale_approach_violations"] == 0
                                       for group in guard),
        "guard_bounded_evasion": all(group["max_evasions"] <= 3 for group in guard),
        "guard_completed_safety_watch": all(group["incomplete_watch_without_contact"] == 0
                                            for group in guard),
        "heldout_has_challenging_baseline": len(challenging) >= 1,
        "guard_rescues_challenging_episode": len(rescued) >= 1,
    }
    report = {"seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "challenging_baseline_seeds": sorted(challenging),
              "rescued_seeds": sorted(rescued),
              "criteria": "docs/research/ape-move-wait-range-guard-v20.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, group in groups.items():
        print(name, "n", group["present_n"], "/", group["absent_n"],
              "reach", group["safe_approaches_after_dwell"],
              "overlap", group["path_overlaps"], "guard stops", group["midstep_stops"],
              "min gap", group["min_path_clearance_m"])
    print("CHECKS", checks)


if __name__ == "__main__":
    main()
