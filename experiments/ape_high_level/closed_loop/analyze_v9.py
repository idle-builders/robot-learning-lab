"""Score the frozen held-out Cove-50 localization and health-gate comparison."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v9"
HELD_OUT = {37, 38, 39, 40}
CORE = ("baseline", "variable_slip", "dynamic_obstacle", "moved_target")
FAULTS = ("range_stale", "range_dropout", "camera_stale", "camera_dropout")
PLAIN = "chest_rgbd_range"
LOCAL = "chest_rgbd_range_localized"


def average(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def grouped(rows: list[dict]) -> dict:
    result: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        result[row["scenario"], row["config"]].append(row)
    return result


def main() -> None:
    rows = [json.loads(path.read_text()) for path in RUN.glob("*/trace.json")]
    cases = grouped([row for row in rows if row["seed"] in HELD_OUT])
    groups = {}
    for (scenario, config), episode_rows in sorted(cases.items()):
        present = [row for row in episode_rows if not row["absent"]]
        absent = [row for row in episode_rows if row["absent"]]
        key = f"{scenario}:{config}"
        groups[key] = {
            "present_n": len(present), "absent_n": len(absent),
            "mean_final_pose_error_m": average([row["pose_error_m"] for row in present]),
            "max_final_pose_error_m": max((row["pose_error_m"] for row in present), default=None),
            "mean_target_location_error_m": average([
                row["last_target_location_error_m_evaluator_only"] for row in present
                if row["last_target_location_error_m_evaluator_only"] is not None]),
            "safe_approaches": sum(row["outcome"] == "approach_stopped" and
                                   .25 <= row["target_center_distance_m"] <= .40 and
                                   not row["unsafe_obstacle_overlap"] for row in present),
            "present_overlap": sum(row["unsafe_obstacle_overlap"] for row in present),
            "minimum_clearance_m": min((row["minimum_robot_obstacle_clearance_m"]
                                         for row in present), default=None),
            "absent_motion": sum(any(abs(x) > 1e-6 for x in row[
                "actual_final_xy_m_evaluator_only"]) for row in absent),
            "absent_wrong_candidate_questions": sum(row["owner_questions"] for row in absent),
            "present_owner_questions": sum(row["owner_questions"] for row in present),
            "present_outcomes": {str(row["seed"]): row["outcome"] for row in present},
        }
    checks = {}
    for scenario in ("baseline", "variable_slip", "moved_target"):
        dead = groups.get(f"{scenario}:{PLAIN}")
        fused = groups.get(f"{scenario}:{LOCAL}")
        checks[f"{scenario}_pose"] = bool(dead and fused and
            dead["present_n"] == fused["present_n"] == 4 and
            fused["mean_final_pose_error_m"] <= .04 and
            fused["mean_final_pose_error_m"] < dead["mean_final_pose_error_m"])
    for scenario in CORE:
        dead = groups.get(f"{scenario}:{PLAIN}")
        fused = groups.get(f"{scenario}:{LOCAL}")
        checks[f"{scenario}_behavior"] = bool(dead and fused and
            dead["present_n"] == fused["present_n"] == 4 and
            dead["absent_n"] == fused["absent_n"] == 4 and
            fused["present_overlap"] <= dead["present_overlap"] and
            fused["absent_wrong_candidate_questions"] <= dead[
                "absent_wrong_candidate_questions"] and
            fused["safe_approaches"] >= dead["safe_approaches"] and
            fused["absent_motion"] == dead["absent_motion"] == 0)
    for config in (PLAIN, LOCAL):
        group = groups.get(f"dynamic_obstacle:{config}")
        checks[f"dynamic_safe_{config}"] = bool(group and group["present_n"] == 4 and
            group["present_overlap"] == 0 and all(
                status in ("approach_stopped", "weak_reobservation_stop", "obstacle_stop")
                for status in group["present_outcomes"].values()))
        for scenario, action in (("range_stale", "stale_range_stop"),
                                 ("range_dropout", "range_dropout_stop"),
                                 ("camera_stale", "stale_camera_stop"),
                                 ("camera_dropout", "camera_dropout_stop")):
            selected = [row for row in cases.get((scenario, config), []) if not row["absent"]]
            checks[f"{scenario}_gate_{config}"] = len(selected) == 4 and all(
                row["outcome"].endswith("recovery_requested") and
                row["events"][-1]["action"] == action and
                (row["events"][-1].get("range_age_s", 0) >
                 row["events"][-1].get("range_max_age_s", 0)
                 if scenario == "range_stale" else True) and
                (row["events"][-1].get("camera_age_s", 0) >
                 row["events"][-1].get("camera_max_age_s", 0)
                 if scenario == "camera_stale" else True) and
                not row["unsafe_obstacle_overlap"] for row in selected)
    report = {"held_out_seeds": sorted(HELD_OUT),
              "frozen_criteria": "docs/research/ape-closed-loop-localization-v9.md",
              "groups": groups, "checks": checks,
              "all_checks_pass": all(checks.values())}
    (RUN / "summary-heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, group in groups.items():
        print(key, group["present_n"], "pose", group["mean_final_pose_error_m"],
              "safe", group["safe_approaches"], "overlap", group["present_overlap"])
    print("CHECKS", sum(checks.values()), "/", len(checks),
          [key for key, passed in checks.items() if not passed])


if __name__ == "__main__":
    main()
