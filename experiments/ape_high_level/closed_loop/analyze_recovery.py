"""Score the frozen v13 held-out viewpoint-recovery comparison."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v13"
SEEDS = {57, 58, 59, 60}
SCENARIOS = ("baseline", "dynamic_obstacle")
CONFIGS = ("chest_rgbd_range", "chest_rgbd_range_recovery",
           "chest_rgbd_head_sweep_depth", "chest_rgbd_head_sweep_recovery")


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def lost_view_approach_violations(events: list[dict]) -> int:
    """Check the trace contract: a lost view needs a new matched metric view."""
    lost = False
    violations = 0
    for event in events:
        action = event["action"]
        if action in ("target_reacquisition_stop", "weak_view_before_reorient",
                      "location_loss_stop"):
            lost = True
        elif action in ("search_viewpoint_step", "recovery_turn_and_reobserve"):
            if event.get("recovered_target_xy_m") is not None:
                lost = False
        elif action == "wait_and_reobserve_moving_obstacle":
            if event.get("fresh_target_xy_m") is not None:
                lost = False
        elif action == "safety_evasion_step":
            if event.get("fresh_target_xy_m") is not None:
                lost = False
        elif action == "reorient_and_reobserve" and event.get("similarity", -1) >= .35:
            lost = False
        elif action == "target_refreshed_after_head_sweep":
            lost = False
        elif action == "move_and_reobserve" and lost:
            violations += 1
    return violations


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
            safe = [row for row in present if row["outcome"] == "approach_stopped"
                    and row["target_center_distance_m"] is not None
                    and .25 <= row["target_center_distance_m"] <= .4
                    and not row["unsafe_obstacle_overlap"]]
            search_moves = [row.get("recovery_viewpoint_moves", 0) for row in present]
            groups[f"{scenario}:{config}"] = {
                "present_n": len(present), "absent_n": len(absent),
                "safe_approaches": len(safe),
                "overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
                "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                    "actual_final_xy_m_evaluator_only"]) for row in absent),
                "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
                "total_recovery_viewpoint_moves": sum(search_moves),
                "max_recovery_viewpoint_moves": max(search_moves, default=0),
                "lost_view_approach_violations": sum(
                    lost_view_approach_violations(row["events"]) for row in present),
                "min_clearance_m": min((row["minimum_robot_obstacle_clearance_m"]
                                        for row in present), default=None),
                "mean_present_end_time_s": mean([row["events"][-1]["t_s"] for row in present]),
                "mean_safe_approach_time_s": mean([row["events"][-1]["t_s"] for row in safe]),
                "outcomes": {str(row["seed"]): row["outcome"] for row in present},
            }
    recovery_groups = [groups[f"{scenario}:{config}"] for scenario in SCENARIOS
                       for config in CONFIGS if config.endswith("_recovery")]
    checks = {
        "all_paired_cases_present": all(group["present_n"] == 4 and group["absent_n"] == 4
                                    for group in groups.values()),
        "recovery_zero_overlap": all(group["overlaps"] == 0 for group in recovery_groups),
        "recovery_zero_absent_motion": all(group["absent_motion"] == 0
                                           for group in recovery_groups),
        "bounded_recovery_steps": all(group["max_recovery_viewpoint_moves"] <= 2
                                      for group in recovery_groups),
        "no_lost_view_approach": all(group["lost_view_approach_violations"] == 0
                                     for group in recovery_groups),
        "dynamic_progress": any(groups[f"dynamic_obstacle:{config}"]["safe_approaches"] >= 2
                                for config in CONFIGS if config.endswith("_recovery")),
    }
    report = {"seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "criteria": "docs/research/ape-occlusion-recovery-v13.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, group in groups.items():
        print(name, "n", group["present_n"], "/", group["absent_n"],
              "safe", group["safe_approaches"], "overlap", group["overlaps"],
              "search", group["total_recovery_viewpoint_moves"],
              "outcomes", group["outcomes"])
    print("CHECKS", checks)


if __name__ == "__main__":
    main()
