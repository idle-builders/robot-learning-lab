"""Audit source age, independent scans, and issued commands in an Isaac timing run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    evidence = args.run / "evidence"
    actions = json.loads((evidence / "online-action-trace.json").read_text())["actions"]
    monitor = json.loads((evidence / "continuous-vision-monitor.json").read_text())["frames"]
    assert len(actions) == len(monitor) > 0
    rows = []
    all_gaps = []
    all_wall_gaps = []
    for action, watched in zip(actions, monitor):
        assert action["frame"] == watched["frame"]
        source = action["observation_time_s"]
        decision = action["decision_time_s"]
        assert abs(source - watched["source_time_s"]) < 1e-5
        assert abs(decision - watched["decision_time_s"]) < 1e-5
        samples = watched["samples"]
        times = [sample["time_s"] for sample in samples]
        assert all(source < instant <= decision for instant in times)
        assert times == sorted(times)
        all_gaps.extend(right - left for left, right in zip(times, times[1:]))
        walls = [sample["wall_elapsed_s"] for sample in samples
                 if "wall_elapsed_s" in sample]
        all_wall_gaps.extend(right - left for left, right in zip(walls, walls[1:]))
        age = decision - source
        assert abs(age - watched["source_age_s"]) < 1e-5
        command = action["command_for_next_frame"]
        moved = (abs(command["yaw_delta_deg"]) > 1e-8 or
                 abs(command.get("body_yaw_delta_deg", 0.)) > 1e-8 or
                 abs(command["forward_step_m"]) > 1e-8)
        interlock = action["actor"].get("command_interlock")
        limit = action["actor"].get("command_source_age_limit_s", .20)
        if age > limit or watched["monitor_alarm"]:
            memory_gaze = (action["actor"].get("memory_gaze_only") is True and
                           not watched["monitor_alarm"] and
                           abs(command["forward_step_m"]) < 1e-8 and
                           abs(command.get("body_yaw_delta_deg", 0.)) < 1e-8 and
                           abs(command["yaw_delta_deg"]) <= 10.)
            assert not moved or memory_gaze, (
                f"stale/alarmed command moved at frame {action['frame']}")
            assert interlock in ("stale_vision", "person_sector_motion",
                                 "held_low_proximity",
                                 "fixed_pose_monitor_probe",
                                 "pose_motion_unconfirmed")
        social = action.get("social_actor") or {}
        if social.get("intent") == "new_item_seen_unattributed":
            assert social.get("fresh") is True
            assert age <= .20 and not watched["monitor_alarm"], (
                f"stale/alarmed item claim at frame {action['frame']}")
        rows.append({"frame": action["frame"], "source_age_s": age,
            "command_source_age_limit_s": limit,
            "inference_wall_s": watched["inference_wall_s"],
            "scan_count": len(samples),
            "first_scan_delay_s": times[0] - source if times else None,
            "bystander_displacement_during_scans_m": (
                math.dist(samples[0]["character_xyz_m_evaluator_only"][1],
                          samples[-1]["character_xyz_m_evaluator_only"][1])
                if len(samples) > 1 and "character_xyz_m_evaluator_only" in samples[0]
                else None),
            "alarm": watched["monitor_alarm"],
            "proposed_intent": action["actor"]["intent"],
            "command_interlock": interlock,
            "memory_gaze_only": action["actor"].get("memory_gaze_only") is True,
            "issued_motion": moved})
    report = {"frames": len(rows), "rows": rows,
        "total_monitor_scans": sum(row["scan_count"] for row in rows),
        "median_source_age_s": median(row["source_age_s"] for row in rows),
        "max_scan_gap_sim_s": max(all_gaps) if all_gaps else None,
        "max_scan_gap_wall_s": max(all_wall_gaps) if all_wall_gaps else None,
        "stale_or_alarmed_motion_commands": sum(row["issued_motion"] for row in rows
            if (row["source_age_s"] > row["command_source_age_limit_s"] or row["alarm"])
            and not row["memory_gaze_only"]),
        "bounded_memory_gaze_commands": sum(row["memory_gaze_only"] for row in rows)}
    (args.run / "continuous-vision-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
