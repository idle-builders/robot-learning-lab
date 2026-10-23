"""Local exact-body compatibility screen for one historical whole step reference.

This is a diagnostic action replay, not a teacher acceptance or CUDA score.
It deliberately runs the same 50 Hz motor law as the Cardinal evaluator and
records the physical trajectory before any backward/side retarget or fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ape_walk import task as ape
from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                      configure_frame50_motors,
                                      give_soles_contact_priority,
                                      require_direct_frame)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
MOTIONS = ROOT / "experiments/compact_morphology/motions"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(side: str, seed: int, out: Path) -> dict:
    if side not in ("left", "right"):
        raise ValueError(side)
    require_direct_frame(BODY)
    motion_path = MOTIONS / f"ape_single_forward_step_{side}_v1.npz"
    with np.load(motion_path) as motion:
        fps = int(motion["fps"][0])
        target = motion["joint_pos"][:200, :10].astype(np.float32).copy()
    if fps != 50 or target.shape != (200, 10):
        raise RuntimeError("Expected one 4 s historical 50 Hz step reference")
    cfg = load_env_cfg(ape.ELBOW_TASK_ID, play=True)
    cfg.scene.entities["robot"].spec_fn = partial(ape._ape_spec, BODY.resolve())
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    give_soles_contact_priority(cfg)
    configure_frame50_motors(cfg, BODY, rail_v=12.0, delay_s=0.0)
    clip_frame50_leg_targets(cfg, BODY, margin_rad=.02, axes=("ankle_pitch",))
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, .231)
    cfg.scene.entities["robot"].init_state.joint_pos[r".*shoulder_pitch.*"] = 0.0
    cfg.scene.entities["robot"].init_state.joint_pos[r".*elbow_pitch.*"] = 0.0
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.episode_length_s = 4.0
    cfg.auto_reset = False
    cfg.commands["twist"].rel_standing_envs = 0.0
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (.25, .25)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (100.0, 100.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset(seed=seed)
        term = env.action_manager.get_term("joint_pos")
        offset = term._offset[0].detach().cpu().numpy().copy()
        scale = term._scale[0].detach().cpu().numpy().copy()
        if len(offset) != 21 or len(scale) != 21:
            raise RuntimeError("Direct Frame action width changed")
        rated = np.asarray([a.effort_limit for a in cfg.scene.entities["robot"].articulation.actuators[:10]])
        robot = env.scene["robot"]
        sensors = env.scene.sensors
        start_root = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
        rows = []
        over_unit = 0
        for i in range(200):
            raw_leg = (target[i] - offset[:10]) / scale[:10]
            over_unit += int(np.any(np.abs(raw_leg) > 1.0 + 1e-6))
            action = np.zeros((1, 21), dtype=np.float32)
            # Cardinal's saved RSL-RL runner declares clip_actions=None. Its
            # native evaluator forwards unbounded actor actions to this term.
            action[0, :10] = raw_leg
            _, _, terminated, timed_out, _ = env.step(torch.as_tensor(action))
            sole = sensors["feet_ground_contact"].data.found[0].reshape(2, -1).any(dim=-1).detach().cpu().numpy()
            root = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
            effort = np.abs(robot.data.actuator_force[0, :10].detach().cpu().numpy())
            hand_force = float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).amax())
            rows.append({"step": i + 1, "root_xyz_m": root.tolist(),
                         "sole_found": sole.astype(bool).tolist(),
                         "rated_leg_fraction": float(np.mean(effort >= .99 * rated)),
                         "hand_force_n": hand_force,
                         "action_over_unit": bool(np.any(np.abs(raw_leg) > 1.0 + 1e-6))})
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        both = np.asarray([r["sole_found"] for r in rows], dtype=bool)
        root_end = np.asarray(rows[-1]["root_xyz_m"])
        result = {"schema": "robot-learning-lab.historical-step-reference-compatibility.v1",
                  "claim_limit": "Local CPU 3.13 exact-body diagnostic; not a valid native CUDA score or accepted teacher",
                  "body_sha256": sha(BODY), "source_sha256": sha(Path(__file__)),
                  "motion_sha256": sha(motion_path), "motion": str(motion_path.relative_to(ROOT)),
                  "seed": seed, "side": side, "device": "cpu", "fps": fps,
                  "steps": len(rows), "initial_root_xyz_m": start_root.tolist(),
                  "net_root_xyz_m": (root_end - start_root).tolist(),
                  "action_over_unit_steps": over_unit,
                  "both_sole_fraction": float(np.mean(np.all(both, axis=1))),
                  "no_sole_fraction": float(np.mean(~np.any(both, axis=1))),
                  "rated_leg_frame_fraction": float(np.mean([r["rated_leg_fraction"] > 0 for r in rows])),
                  "max_hand_force_n": float(max(r["hand_force_n"] for r in rows)),
                  "min_root_z_m": float(min(r["root_xyz_m"][2] for r in rows)),
                  "rows": rows}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        return {k: result[k] for k in ("side", "steps", "net_root_xyz_m", "action_over_unit_steps", "both_sole_fraction", "rated_leg_frame_fraction", "max_hand_force_n", "min_root_z_m")}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--seed", type=int, default=49307)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.side, args.seed, args.out)))


if __name__ == "__main__":
    main()
