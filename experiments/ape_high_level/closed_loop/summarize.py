"""Summarize paired Cove-50 closed-loop episodes without hiding failures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted(args.run.glob("*/trace.json"))]
    if not rows:
        raise ValueError("No episode traces")
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("scenario", "baseline"), row["config"], row["fault"])].append(row)
    summary = {"scope": "Small paired synthetic episodes; scripted owner, kinematic base, no physical sensor or locomotion qualification",
               "run": str(args.run), "groups": {}}
    for (scenario, config, fault), cases in sorted(groups.items()):
        present = [row for row in cases if not row["absent"]]
        absent = [row for row in cases if row["absent"]]
        summary["groups"][f"{scenario}:{config}:{fault}"] = {
            "present_episodes": len(present),
            "absent_episodes": len(absent),
            "present_safe_reaches": sum(
                row["outcome"] == "approach_stopped" and
                row["target_center_distance_m"] is not None and
                .25 <= row["target_center_distance_m"] <= .40 and
                row["minimum_robot_obstacle_clearance_m"] >= 0
                for row in present),
            "present_unsafe_overlaps": sum(row["unsafe_obstacle_overlap"] for row in present),
            "present_unresolved": sum(row["outcome"] in (
                "ask_to_show_again", "weak_reobservation_stop",
                "obstacle_stop", "lost_target_stop", "location_loss_stop")
                for row in present),
            "absent_no_motion": sum(row["outcome"].startswith("absent_abstained")
                                    for row in absent),
            "absent_wrong_candidate_questions": sum(row["owner_questions"] for row in absent),
            "present_owner_questions": sum(row["owner_questions"] for row in present),
            "minimum_present_clearance_m": (min(
                row["minimum_robot_obstacle_clearance_m"] for row in present)
                if present else None),
            "mean_final_pose_error_m": (sum(row.get("pose_error_m", 0)
                                             for row in present) / len(present)
                                        if present else None),
            "max_final_pose_error_m": (max(row.get("pose_error_m", 0)
                                            for row in present) if present else None),
            "mean_last_target_location_error_m": (
                sum(row["last_target_location_error_m_evaluator_only"] for row in present
                    if row.get("last_target_location_error_m_evaluator_only") is not None) /
                sum(row.get("last_target_location_error_m_evaluator_only") is not None
                    for row in present)
                if any(row.get("last_target_location_error_m_evaluator_only") is not None
                       for row in present) else None),
            "outcomes": {str(row["seed"]): row["outcome"] for row in present},
        }
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for config, row in summary["groups"].items():
        print(config, "safe reaches", row["present_safe_reaches"], "/",
              row["present_episodes"], "unsafe", row["present_unsafe_overlaps"],
              "absent prompts", row["absent_wrong_candidate_questions"])


if __name__ == "__main__":
    main()
