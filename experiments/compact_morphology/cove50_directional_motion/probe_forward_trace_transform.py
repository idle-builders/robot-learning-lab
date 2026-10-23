"""Cheap exact-body motion-prior diagnostic from a source forward CUDA trace.

Open-loop CPU replay of saved targets is only a feasibility screen. A passing
transformation must later survive a fresh native CUDA closed-loop visit before
it can become a demonstration source or learner baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from search_cardinal_body_reference import BODY, make_env, sha


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/fresh-native-probe-v2/forward-stop-seed47013.npz"
NAMES = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch")
MODES = ("source", "reverse_hip", "reverse_sagittal", "reverse_time")


def transform(target: np.ndarray, offset: np.ndarray, mode: str) -> np.ndarray:
    target = target.copy()
    if mode == "reverse_time":
        return target
    ids = {"source": (), "reverse_hip": (2, 7),
           "reverse_sagittal": (2, 4, 7, 9)}[mode]
    target[list(ids)] = 2 * offset[list(ids)] - target[list(ids)]
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(SOURCE, allow_pickle=False) as saved:
        if tuple(saved["leg_names"].tolist()) != NAMES:
            raise RuntimeError("Source joint order differs")
        raw = saved["raw_action"][:200, 0].copy()
        source_root = saved["root_pose"][:200, 0].copy()
    env = make_env(seed=47013)
    results = []
    try:
        for mode in MODES:
            env.reset(seed=47013)
            term = env.action_manager.get_term("joint_pos")
            if tuple(term._target_names[:10]) != NAMES:
                raise RuntimeError("Target joint order differs")
            offset = term._offset[0, :10].detach().cpu().numpy().copy()
            scale = term._scale[0, :10].detach().cpu().numpy().copy()
            robot = env.scene["robot"]
            sensors = env.scene.sensors
            first_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
            root0 = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
            rows = []
            for step in range(200):
                source_index = 199 - step if mode == "reverse_time" else step
                target = offset + raw[source_index] * scale
                target = transform(target, offset, mode)
                action = np.zeros((1, len(term._target_names)), dtype=np.float32)
                action[0, :10] = (target - offset) / scale
                _, _, ended, timed, _ = env.step(torch.as_tensor(action))
                root = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
                sole = sensors["feet_ground_contact"].data.found[0].reshape(2, -1).any(dim=-1).cpu().numpy()
                rows.append({"step": step + 1, "root_xyz_m": root.tolist(),
                             "sole_found": sole.astype(bool).tolist(),
                             "hand_force_n": float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).amax())})
                if bool(ended[0]) or bool(timed[0]):
                    break
            roots = np.asarray([r["root_xyz_m"] for r in rows])
            soles = np.asarray([r["sole_found"] for r in rows])
            delta = roots[-1] - root0
            touchdowns = ((~soles[:-1]) & soles[1:]).sum(axis=0).tolist()
            results.append({"mode": mode, "steps": len(rows),
                            "initial_qpos_sha256": hashlib.sha256(first_qpos.tobytes()).hexdigest(),
                            "net_root_xyz_m": delta.tolist(),
                            "backward_m": float(-delta[0]),
                            "min_root_z_m": float(roots[:, 2].min()),
                            "touchdowns_left_right": touchdowns,
                            "max_hand_force_n": max(r["hand_force_n"] for r in rows),
                            "source_root_x_gap_at_last_step_m": float(abs(roots[-1, 0] - source_root[len(rows)-1, 0])),
                            "rows": rows})
            print(mode, len(rows), round(float(delta[0]) * 1000, 1), touchdowns, flush=True)
    finally:
        env.close()
    report = {"schema": "robot-learning-lab.forward-motion-transform-cpu.v1",
              "claim_limit": "Open-loop local CPU transform of saved CUDA forward actions; no learner, corrective policy, fresh CUDA score or accepted demonstration",
              "body_sha256": sha(BODY), "source_trace_sha256": sha(SOURCE),
              "script_sha256": sha(Path(__file__)), "seed": 47013,
              "gpu_hours": 0, "optimizer_updates": 0, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
