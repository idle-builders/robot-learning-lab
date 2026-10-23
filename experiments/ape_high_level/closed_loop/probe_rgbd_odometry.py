"""Compare ape-camera RGB-D odometry with command and range odometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from localization import ScanLocalizer
from rgbd_odometry import RGBDOdometry
from run import OUT, Room, sensed_range


def main() -> None:
    out = OUT / "rgbd-odometry-probe"
    out.mkdir(parents=True, exist_ok=True)
    room = Room()
    rows = []
    try:
        for seed in (35, 37, 38, 39, 40):
            room.reset()
            room.layout(seed, absent=False)
            rgb, depth = room.capture("chest_low50_x45", depth=True)
            visual = RGBDOdometry()
            visual.initialize(rgb, depth, room.estimated,
                              room.camera_pose_estimate("chest_low50_x45")[1])
            scan = ScanLocalizer()
            range_pose = room.estimated.copy()
            scan.initialize(sensed_range(room, seed, -1), range_pose)
            command_pose = room.estimated.copy()
            steps = []
            side = 1 if seed % 2 else -1
            for step in range(8):
                move = np.array([.04, side * .04])
                slip = .58 if step % 2 == 0 else .94
                room.move(*move, slip=slip)
                command_pose[:2] += move
                predicted_range = range_pose.copy()
                predicted_range[:2] += move
                range_pose, range_status = scan.correct(
                    sensed_range(room, seed, 100 + step), predicted_range)
                rgb, depth = room.capture("chest_low50_x45", depth=True)
                room.estimated, visual_status = visual.correct(
                    rgb, depth, room.estimated,
                    room.camera_pose_estimate("chest_low50_x45")[1])
                steps.append({
                    "step": step, "visual_status": visual_status,
                    "range_status": range_status,
                    "command_error_m": float(np.linalg.norm(command_pose[:2] - room.actual[:2])),
                    "rgbd_error_m": float(np.linalg.norm(room.estimated[:2] - room.actual[:2])),
                    "range_error_m": float(np.linalg.norm(range_pose[:2] - room.actual[:2])),
                    "true_xy_evaluator_only": room.actual[:2].round(4).tolist(),
                    "rgbd_estimated_xy": room.estimated[:2].round(4).tolist(),
                })
            rows.append({"seed": seed, "steps": steps,
                         "final_command_error_m": steps[-1]["command_error_m"],
                         "final_rgbd_error_m": steps[-1]["rgbd_error_m"],
                         "final_range_error_m": steps[-1]["range_error_m"],
                         "rgbd_registered_steps": sum(step["visual_status"]["status"] ==
                                                      "registered" for step in steps)})
    finally:
        room.close()
    (out / "report.json").write_text(json.dumps({
        "scope": "Selected-body-derived chest view in a scripted eight-step MuJoCo path; evaluator pose for scoring only; no real gait policy.",
        "rows": rows}, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["rgbd_registered_steps"], "/ 8",
              "final error command/rgbd/range (m)",
              round(row["final_command_error_m"], 3),
              round(row["final_rgbd_error_m"], 3),
              round(row["final_range_error_m"], 3))


if __name__ == "__main__":
    main()
