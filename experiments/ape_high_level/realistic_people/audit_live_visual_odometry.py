"""Score live Isaac pose/action traces without feeding evaluator pose to actor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def audit(run: Path) -> dict:
    evidence = run / "evidence"
    actions = json.loads((evidence / "online-action-trace.json").read_text())["actions"]
    scene = json.loads((evidence / "scene-info.json").read_text())
    inputs = json.loads((run / "inputs.json").read_text())
    scale = float(inputs["odometry_scale"])
    odometry = np.asarray(scene["camera_eye_xyz_m"][:2], dtype=float)
    heading = 0.0
    rows = []
    for index, action in enumerate(actions):
        if index:
            previous = actions[index - 1]["command_for_next_frame"]
            heading += previous["yaw_delta_deg"]
            odometry += scale * previous["forward_step_m"] * np.array([
                math.sin(math.radians(heading)),
                math.cos(math.radians(heading))])
        truth = (np.asarray(actions[index - 1]["proxy_eye_xyz_m_evaluator_only"][:2])
                 if index else np.asarray(scene["camera_eye_xyz_m"][:2]))
        localization = action.get("localization")
        if localization is None:
            raise ValueError("Run has no live localization trace")
        pose = np.asarray(localization["pose_xy_m"])
        rows.append({"frame": index,
                     "status": localization["status"],
                     "pose_error_m_evaluator_only": float(np.linalg.norm(pose - truth)),
                     "biased_command_odometry_error_m_evaluator_only":
                         float(np.linalg.norm(odometry - truth)),
                     "pose_xy_m_actor": pose.tolist(),
                     "command_for_next_frame": action["command_for_next_frame"],
                     "command_interlock": action["actor"].get("command_interlock"),
                     "source_age_s": action["sensor_pipeline_wall_s"]["decision_source_age_sim_s"]})
    return {"run": str(run), "frames": len(rows),
            "visual_registered": sum(row["status"] == "visual_registered" for row in rows),
            "odometry_fallback": sum(row["status"] == "odometry_fallback" for row in rows),
            "translation_commands": sum(row["command_for_next_frame"]["forward_step_m"] > 0
                                        for row in rows),
            "pose_interlocks": sum(row["command_interlock"] == "pose_motion_unconfirmed"
                                   for row in rows),
            "final_pose_error_m": rows[-1]["pose_error_m_evaluator_only"],
            "max_pose_error_m": max(row["pose_error_m_evaluator_only"] for row in rows),
            "final_biased_odometry_error_m":
                rows[-1]["biased_command_odometry_error_m_evaluator_only"],
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    target = args.run / "live-visual-odometry-audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print({key: value for key, value in result.items() if key != "rows"})


if __name__ == "__main__":
    main()
