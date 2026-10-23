"""Score the frozen v23 delayed-view recovery screen."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from analyze_recovery import lost_view_approach_violations


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v23"
SEEDS = {89, 90, 91, 92}
SCENARIOS = ("furnished_still", "furnished_moving")
CONFIGS = ("chest_rgbd_range_guard_reacquire",
           "chest_rgbd_range_guard_delayed_view")


def stale_resume_count(events: list[dict]) -> int:
    confirmed = False
    violations = 0
    for event in events:
        if event["action"] == "resume_target_reobserve":
            confirmed = event.get("fresh_target_xy_m") is not None
        elif event["action"] == "resume_move_and_reobserve":
            violations += not confirmed
            confirmed = False
        elif event["action"] in ("guarded_viewpoint_step", "safety_evasion_step",
                                 "midstep_actor_safety_stop"):
            confirmed = False
    return violations


def reached(row: dict, *, moving: bool) -> bool:
    if (row["outcome"] != "approach_stopped" or
            row["target_center_distance_m"] is None or
            not .25 <= row["target_center_distance_m"] <= .4 or
            row["unsafe_obstacle_overlap"]):
        return False
    if not moving:
        return True
    if row.get("resumed_target_steps", 0):
        return row.get("post_resume_watch_s", 0) >= 2.
    return row.get("post_decision_watch_s", 0) >= 2.


def main() -> None:
    cases: dict[tuple[str, str], list[dict]] = defaultdict(list)
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
            groups[f"{scenario}:{config}"] = {
                "present_n": len(present),
                "absent_n": len(absent),
                "safe_reaches": sum(reached(row, moving=scenario == "furnished_moving")
                                    for row in present),
                "min_clearance_m": min((row["minimum_robot_obstacle_clearance_m"]
                                       for row in present), default=None),
                "overlap_count": sum(row["unsafe_obstacle_overlap"] for row in present),
                "absent_motion_count": sum(any(abs(value) > 1e-6 for value in
                                               row["actual_final_xy_m_evaluator_only"])
                                           for row in absent),
                "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
                "viewpoint_moves": sum(row.get("recovery_viewpoint_moves", 0)
                                       for row in present),
                "resumed_target_steps": sum(row.get("resumed_target_steps", 0)
                                            for row in present),
                "stale_approach_violations": sum(
                    lost_view_approach_violations(row["events"]) +
                    stale_resume_count(row["events"]) for row in present),
                "outcomes": {str(row["seed"]): row["outcome"] for row in present},
                "target_gaps_m": {str(row["seed"]): row["target_center_distance_m"]
                                  for row in present},
            }
    static_base = groups["furnished_still:chest_rgbd_range_guard_reacquire"]
    static_new = groups["furnished_still:chest_rgbd_range_guard_delayed_view"]
    moving_new = groups["furnished_moving:chest_rgbd_range_guard_delayed_view"]
    geometry_path = RUN / "furnished-geometry-audit.json"
    routes_path = RUN / "furnished-route-feasibility-audit.json"
    geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else None
    routes = json.loads(routes_path.read_text()) if routes_path.exists() else None
    checks = {
        "all_paired_cases_present": all(group["present_n"] == 4 and
                                        group["absent_n"] == 4
                                        for group in groups.values()),
        "valid_scene_geometry": geometry is not None and geometry["all_valid"],
        "five_cm_route_available": routes is not None and all(
            row["five_cm_margin"]["route_exists"] for row in routes["rows"]),
        "zero_overlap": all(group["overlap_count"] == 0 for group in groups.values()),
        "zero_absent_motion": all(group["absent_motion_count"] == 0
                                  for group in groups.values()),
        "moving_min_five_cm": moving_new["min_clearance_m"] is not None and
                              moving_new["min_clearance_m"] >= .05,
        "no_stale_approach": all(group["stale_approach_violations"] == 0
                                  for group in groups.values()),
        "static_progress_preserved": static_new["safe_reaches"] >=
                                     static_base["safe_reaches"],
        "moving_progress_two_of_four": moving_new["safe_reaches"] >= 2,
    }
    result = {"seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "criteria": "docs/research/ape-delayed-active-view-v23.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, group in groups.items():
        print(name, "reach", group["safe_reaches"], "/4",
              "min gap", group["min_clearance_m"],
              "resumed", group["resumed_target_steps"],
              "outcomes", group["outcomes"])
    print("CHECKS", checks)


if __name__ == "__main__":
    main()
