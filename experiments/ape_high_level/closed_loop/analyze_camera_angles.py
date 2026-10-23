"""Score held-out virtual chest tilt comparison under moving occlusion."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v10"
SEEDS = {45, 46, 47, 48}
CONFIGS = ("chest_rgbd_range_localized",
           "chest_shallow_rgbd_range_localized",
           "chest_steep_rgbd_range_localized")


def main() -> None:
    cases = defaultdict(list)
    for path in RUN.glob("dynamic_obstacle-*/trace.json"):
        row = json.loads(path.read_text())
        if row["seed"] in SEEDS and row["config"] in CONFIGS:
            cases[row["config"]].append(row)
    groups = {}
    for config in CONFIGS:
        present = [row for row in cases[config] if not row["absent"]]
        absent = [row for row in cases[config] if row["absent"]]
        groups[config] = {"present_n": len(present), "absent_n": len(absent),
            "owner_confirmed_target": sum(row["confirmed_target"] for row in present),
            "safe_approaches": sum(row["outcome"] == "approach_stopped" and
                not row["unsafe_obstacle_overlap"] and
                .25 <= row["target_center_distance_m"] <= .4 for row in present),
            "overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
            "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                "actual_final_xy_m_evaluator_only"]) for row in absent),
            "absent_wrong_questions": sum(row["owner_questions"] for row in absent),
            "present_outcomes": {str(row["seed"]): row["outcome"] for row in present}}
    baseline = groups[CONFIGS[0]]
    checks = {}
    for config in CONFIGS[1:]:
        row = groups[config]
        checks[config] = bool(row["present_n"] == row["absent_n"] == 4 and
            baseline["present_n"] == baseline["absent_n"] == 4 and
            row["overlaps"] <= baseline["overlaps"] and
            row["absent_motion"] == 0 and
            (row["safe_approaches"] > baseline["safe_approaches"] or
             row["absent_wrong_questions"] < baseline["absent_wrong_questions"]))
    # The frozen numeric screen omitted present-target confirmation. Keep its
    # result intact and separately apply the owner-facing task goal.
    goal_usable = {config: checks[config] and
                   groups[config]["owner_confirmed_target"] >= baseline[
                       "owner_confirmed_target"] for config in CONFIGS[1:]}
    report = {"seeds": sorted(SEEDS), "groups": groups,
              "frozen_numeric_screen": checks,
              "owner_task_goal_usable": goal_usable,
              "criteria": "docs/research/ape-closed-loop-sensor-ablation-v10.md"}
    (RUN / "summary-camera-angles.json").write_text(json.dumps(report, indent=2) + "\n")
    for config, row in groups.items():
        print(config, row)
    print("FROZEN_SCREEN", checks, "OWNER_GOAL", goal_usable)


if __name__ == "__main__":
    main()
