"""Score frozen held-out RGB-D versus range localization episodes."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v10"
SEEDS = {41, 42, 43, 44}
SCENARIOS = ("baseline", "variable_slip", "moved_target", "dynamic_obstacle")
PLAIN = "chest_rgbd_range"
VISUAL = "chest_rgbd_range_visualodom"
RANGE = "chest_rgbd_range_localized"


def average(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    cases: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in RUN.glob("*/trace.json"):
        row = json.loads(path.read_text())
        if row["seed"] in SEEDS:
            cases[row["scenario"], row["config"]].append(row)
    groups = {}
    for (scenario, config), rows in sorted(cases.items()):
        present = [row for row in rows if not row["absent"]]
        absent = [row for row in rows if row["absent"]]
        groups[f"{scenario}:{config}"] = {
            "present_n": len(present), "absent_n": len(absent),
            "mean_pose_error_m": average([row["pose_error_m"] for row in present]),
            "mean_target_location_error_m": average([
                row["last_target_location_error_m_evaluator_only"] for row in present
                if row["last_target_location_error_m_evaluator_only"] is not None]),
            "safe_approaches": sum(row["outcome"] == "approach_stopped" and
                                   .25 <= row["target_center_distance_m"] <= .4 and
                                   not row["unsafe_obstacle_overlap"] for row in present),
            "overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
            "minimum_clearance_m": min((row["minimum_robot_obstacle_clearance_m"]
                                         for row in present), default=None),
            "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                "actual_final_xy_m_evaluator_only"]) for row in absent),
            "absent_questions": sum(row["owner_questions"] for row in absent),
            "present_outcomes": {str(row["seed"]): row["outcome"] for row in present},
            "registered_updates": sum(
                (event.get("localization") or {}).get("status") == "registered"
                for row in present for event in row["events"]),
        }
    slip_plain = groups.get(f"variable_slip:{PLAIN}")
    slip_visual = groups.get(f"variable_slip:{VISUAL}")
    slip_range = groups.get(f"variable_slip:{RANGE}")
    checks = {"visual_slip_pose": bool(slip_plain and slip_visual and
              slip_visual["present_n"] == 4 and
              slip_visual["mean_pose_error_m"] <= .09 and
              slip_visual["mean_pose_error_m"] < slip_plain["mean_pose_error_m"]),
              "range_slip_pose": bool(slip_range and slip_range["present_n"] == 4 and
              slip_range["mean_pose_error_m"] <= .04)}
    for scenario in SCENARIOS:
        plain = groups.get(f"{scenario}:{PLAIN}")
        for config in (VISUAL, RANGE):
            alternate = groups.get(f"{scenario}:{config}")
            checks[f"{scenario}:{config}:behavior"] = bool(
                plain and alternate and plain["present_n"] == alternate["present_n"] == 4 and
                plain["absent_n"] == alternate["absent_n"] == 4 and
                alternate["overlaps"] <= plain["overlaps"] and
                alternate["absent_motion"] == plain["absent_motion"] == 0 and
                alternate["absent_questions"] <= plain["absent_questions"] and
                alternate["safe_approaches"] >= plain["safe_approaches"])
    report = {"held_out_seeds": sorted(SEEDS), "groups": groups, "checks": checks,
              "all_checks_pass": all(checks.values()),
              "frozen_criteria": "docs/research/ape-closed-loop-sensor-ablation-v10.md"}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, group in groups.items():
        print(key, group["present_n"], "pose", group["mean_pose_error_m"],
              "safe", group["safe_approaches"], "overlap", group["overlaps"])
    print("CHECKS", sum(checks.values()), "/", len(checks),
          [key for key, passed in checks.items() if not passed])


if __name__ == "__main__":
    main()
