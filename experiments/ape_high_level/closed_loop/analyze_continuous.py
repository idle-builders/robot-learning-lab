"""Score held-out v14 moving-obstacle runs against frozen owner-task gates."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

from analyze_recovery import lost_view_approach_violations


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v14"
SEEDS = {61, 62, 63, 64}
SCENARIOS = ("baseline", "continuous_obstacle")
CONFIGS = ("chest_rgbd_range_recovery", "chest_rgbd_head_sweep_depth",
           "chest_rgbd_head_sweep_recovery")


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sweep_motion(events: list[dict]) -> list[float]:
    motions = []
    batch = []
    for event in events + [{"action": "end"}]:
        if (event["action"] in ("head_depth_sample", "recovery_head_depth_sample")
                and event.get("moving_obstacle_xy_evaluator_only") is not None):
            batch.append(event)
        elif batch:
            if len(batch) >= 2:
                motions.append(float(np.linalg.norm(
                    np.asarray(batch[-1]["moving_obstacle_xy_evaluator_only"]) -
                    np.asarray(batch[0]["moving_obstacle_xy_evaluator_only"]))))
            batch = []
    return motions


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
            ages = [event["head_sweep_oldest_age_s"] for row in present
                    for event in row["events"]
                    if event.get("head_sweep_oldest_age_s") is not None]
            prop_motions = [motion for row in present
                            for motion in sweep_motion(row["events"])]
            groups[f"{scenario}:{config}"] = {
                "present_n": len(present), "absent_n": len(absent),
                "safe_approaches": len(safe),
                "path_overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
                "motion_interruptions": sum(row.get(
                    "continuous_action_interrupted_evaluator_only", False) for row in present),
                "minimum_path_clearance_m": min((
                    row["minimum_robot_obstacle_clearance_m"] for row in present), default=None),
                "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                    "actual_final_xy_m_evaluator_only"]) for row in absent),
                "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
                "total_recovery_steps": sum(row["recovery_viewpoint_moves"] for row in present),
                "max_recovery_steps": max((row["recovery_viewpoint_moves"]
                                           for row in present), default=0),
                "total_obstacle_waits": sum(row.get("dynamic_obstacle_waits", 0)
                                            for row in present),
                "max_obstacle_waits": max((row.get("dynamic_obstacle_waits", 0)
                                           for row in present), default=0),
                "lost_view_approach_violations": sum(
                    lost_view_approach_violations(row["events"]) for row in present),
                "max_oldest_head_scan_age_s": max(ages, default=None),
                "max_prop_motion_during_head_sweep_m": max(prop_motions, default=None),
                "mean_present_end_time_s": mean([row["events"][-1]["t_s"] for row in present]),
                "mean_safe_approach_time_s": mean([row["events"][-1]["t_s"] for row in safe]),
                "outcomes": {str(row["seed"]): row["outcome"] for row in present},
            }
    audit_path = RUN / "scene-trajectory-audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    recovered = [groups[f"{scenario}:{config}"] for scenario in SCENARIOS
                 for config in CONFIGS if config.endswith("_recovery")]
    checks = {
        "all_paired_cases_present": all(group["present_n"] == 4 and group["absent_n"] == 4
                                    for group in groups.values()),
        "scene_trajectory_valid": audit is not None and audit["all_valid"],
        "recovery_zero_path_overlap": all(group["path_overlaps"] == 0
                                          for group in recovered),
        "recovery_zero_absent_motion": all(group["absent_motion"] == 0
                                           for group in recovered),
        "recovery_no_stale_approach": all(group["lost_view_approach_violations"] == 0
                                          for group in recovered),
        "recovery_bounded_search": all(group["max_recovery_steps"] <= 2
                                       for group in recovered),
        "recovery_bounded_wait": all(group["max_obstacle_waits"] <= 6
                                     for group in recovered),
        "range_dynamic_progress": groups[
            "continuous_obstacle:chest_rgbd_range_recovery"]["safe_approaches"] >= 2,
        "head_depth_dynamic_progress": groups[
            "continuous_obstacle:chest_rgbd_head_sweep_recovery"]["safe_approaches"] >= 2,
    }
    report = {"seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "criteria": "docs/research/ape-continuous-obstacle-v14.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, group in groups.items():
        print(name, "n", group["present_n"], "/", group["absent_n"],
              "safe", group["safe_approaches"], "path overlaps", group["path_overlaps"],
              "search", group["total_recovery_steps"], "outcomes", group["outcomes"])
    print("CHECKS", checks)


if __name__ == "__main__":
    main()
