"""Score held-out active head-depth sweep versus fixed depth and range."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v12"
SEEDS = {53, 54, 55, 56}
SCENARIOS = ("baseline", "dynamic_obstacle")
CONFIGS = ("chest_rgbd_depth_obstacles", "chest_rgbd_head_sweep_depth",
           "chest_rgbd_range")


def average(values: list[float]) -> float | None:
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
            ages = [event["head_sweep_oldest_age_s"] for row in present
                    for event in row["events"]
                    if event.get("head_sweep_oldest_age_s") is not None]
            groups[f"{scenario}:{config}"] = {
                "present_n": len(present), "absent_n": len(absent),
                "safe_approaches": sum(row["outcome"] == "approach_stopped" and
                    row["target_center_distance_m"] is not None and
                    .25 <= row["target_center_distance_m"] <= .4 and
                    not row["unsafe_obstacle_overlap"] for row in present),
                "overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
                "obstacle_stops": sum(row["outcome"] == "obstacle_stop" for row in present),
                "target_reacquisition_stops": sum(row["outcome"] ==
                    "target_reacquisition_requested" for row in present),
                "minimum_clearance_m": min((row["minimum_robot_obstacle_clearance_m"]
                    for row in present), default=None),
                "mean_present_end_time_s": average([row["events"][-1]["t_s"]
                                                      for row in present]),
                "mean_safe_approach_time_s": average([row["events"][-1]["t_s"]
                    for row in present if row["outcome"] == "approach_stopped"]),
                "total_head_depth_samples": sum(event["action"] == "head_depth_sample"
                    for row in present for event in row["events"]),
                "max_oldest_scan_age_s": max(ages, default=None),
                "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                    "actual_final_xy_m_evaluator_only"]) for row in absent),
                "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
                "present_outcomes": {str(row["seed"]): row["outcome"] for row in present}}
    active = groups["baseline:chest_rgbd_head_sweep_depth"]
    checks = {"active_zero_overlap": all(groups[f"{scenario}:chest_rgbd_head_sweep_depth"][
                "present_n"] == 4 and groups[f"{scenario}:chest_rgbd_head_sweep_depth"][
                "overlaps"] == 0 for scenario in SCENARIOS),
              "active_no_absent_motion": all(groups[f"{scenario}:chest_rgbd_head_sweep_depth"][
                "absent_n"] == 4 and groups[f"{scenario}:chest_rgbd_head_sweep_depth"][
                "absent_motion"] == 0 for scenario in SCENARIOS),
              "active_baseline_progress": active["present_n"] == 4 and
                active["safe_approaches"] >= 3}
    report = {"seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "criteria": "docs/research/ape-active-depth-sweep-v12.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, group in groups.items():
        print(key, group["present_n"], "safe", group["safe_approaches"],
              "overlap", group["overlaps"], "obstacle stops", group["obstacle_stops"],
              "mean time", group["mean_present_end_time_s"])
    print("CHECKS", checks)


if __name__ == "__main__":
    main()
