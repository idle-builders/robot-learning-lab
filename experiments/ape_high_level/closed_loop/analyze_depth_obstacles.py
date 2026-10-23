"""Score held-out depth-point versus range-obstacle decisions."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v11"
SEEDS = {49, 50, 51, 52}
SCENARIOS = ("baseline", "dynamic_obstacle")
CONFIGS = ("chest_rgbd", "chest_rgbd_depth_obstacles", "chest_rgbd_range")


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
            depth_points = [event["depth_obstacle_point_count"]
                            for row in present for event in row["events"]
                            if event.get("depth_obstacle_point_count") is not None]
            groups[f"{scenario}:{config}"] = {
                "present_n": len(present), "absent_n": len(absent),
                "safe_approaches": sum(row["outcome"] == "approach_stopped" and
                    .25 <= row["target_center_distance_m"] <= .4 and
                    not row["unsafe_obstacle_overlap"] for row in present),
                "present_overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
                "explicit_obstacle_stops": sum(row["events"][-1]["action"] in (
                    "depth_obstacle_stop", "range_obstacle_stop") for row in present),
                "stopped_before_base_move": sum(row["events"][-1]["action"] ==
                    "depth_obstacle_stop" and all(event["action"] != "move_and_reobserve"
                    for event in row["events"]) for row in present),
                "minimum_clearance_m": min((row["minimum_robot_obstacle_clearance_m"]
                    for row in present), default=None),
                "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                    "actual_final_xy_m_evaluator_only"]) for row in absent),
                "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
                "median_depth_obstacle_points": (statistics.median(depth_points)
                    if depth_points else None),
                "present_outcomes": {str(row["seed"]): row["outcome"] for row in present}}
    depth_baseline = groups["baseline:chest_rgbd_depth_obstacles"]
    range_baseline = groups["baseline:chest_rgbd_range"]
    checks = {"depth_no_overlap": all(groups[f"{scenario}:chest_rgbd_depth_obstacles"][
                  "present_n"] == 4 and groups[f"{scenario}:chest_rgbd_depth_obstacles"][
                  "present_overlaps"] == 0 for scenario in SCENARIOS),
              "depth_no_absent_motion": all(groups[f"{scenario}:chest_rgbd_depth_obstacles"][
                  "absent_n"] == 4 and groups[f"{scenario}:chest_rgbd_depth_obstacles"][
                  "absent_motion"] == 0 for scenario in SCENARIOS),
              "range_replacement": (depth_baseline["present_n"] ==
                  range_baseline["present_n"] == 4 and
                  depth_baseline["safe_approaches"] >= range_baseline["safe_approaches"] and
                  depth_baseline["absent_wrong_questions"] <= range_baseline[
                      "absent_wrong_questions"])}
    report = {"seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "criteria": "docs/research/ape-depth-range-obstacles-v11.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, group in groups.items():
        print(key, group["present_n"], "safe", group["safe_approaches"],
              "overlaps", group["present_overlaps"], "stops", group[
                  "explicit_obstacle_stops"])
    print("CHECKS", checks)


if __name__ == "__main__":
    main()
