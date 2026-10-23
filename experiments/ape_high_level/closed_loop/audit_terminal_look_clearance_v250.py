"""V250: evaluator full-body head-sweep clearance for V249 continuations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_same_color_swap_v248 import room_for_swap
from run import ROOT, sha
from run_terminal_owner_clarification_v249 import (
    QUESTION_YAW_RAD, present_room)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v250"
V247_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v247"
V249 = ROOT / "artifacts/ape-closed-loop-cove50-v249/terminal-owner-clarification-audit.json"


def score(seed: int, case: str) -> dict:
    branch = V247_ROOT / f"scene{seed}" / f"seed{seed}"
    trace = json.loads((branch / "trace.json").read_text())
    frame = max(branch.glob("frame-*.npz"),
                key=lambda path: int(path.stem.split("-")[1]))
    with np.load(frame) as saved:
        qpos = saved["actor_qpos"].copy()
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"])
    estimated = np.asarray(trace["final_estimated_pose_xyyaw"])
    memory = np.asarray(trace["final_actor_target_xy_m"])
    if case == "present":
        room = present_room(seed, qpos, actual, estimated)
    else:
        mode = {"ordinary_absent": "ordinary_absent",
                "red_lookalike_absent": "same_red_material"}[case]
        room = room_for_swap(seed, memory, mode, 1., qpos, actual, estimated)
    try:
        initial_yaw = room.head
        rows = []
        for yaw in np.linspace(initial_yaw, QUESTION_YAW_RAD, 21):
            room.head = float(yaw)
            room._forward()
            gap, robot, obstacle = room._current_obstacle_clearance()
            rows.append({"head_yaw_rad": round(float(yaw), 4),
                         "body_scene_clearance_m": round(gap, 5),
                         "robot_geom": robot, "scene_geom": obstacle})
        limiting = min(rows, key=lambda row: row["body_scene_clearance_m"])
        return {"seed": seed, "case": case,
                "v247_final_frame_sha256": sha(frame),
                "minimum_sweep_clearance_m": limiting["body_scene_clearance_m"],
                "limiting": limiting, "samples": rows}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [score(seed, case) for seed in (111, 113)
            for case in ("present", "ordinary_absent", "red_lookalike_absent")]
    report = {"scope": "V250 evaluator-only selected-body versus all scene-solids minimum at 21 uniformly sampled head angles from V247 final head pose to V249 question yaw0.6. Static end-position stool, replayed actual base pose and joints. Actor receives no evaluator geometry. This samples head sweep but does not validate native head dynamics, stop tail or continuous between-sample clearance.",
              "source_sha256": sha(SOURCE), "v249_sha256": sha(V249),
              "rows": rows}
    (OUT / "terminal-look-clearance-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["case"],
              row["minimum_sweep_clearance_m"], flush=True)


if __name__ == "__main__":
    main()
