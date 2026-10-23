"""Add reference action labels on states visited by the failing place policy.

This is one DAgger-style dataset aggregation round. The scripted reference
supplies labels; the saved policy, not the reference, drives these new rollouts.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import ape_walk.task  # register native tasks
from ape_walk.place_tracking import BC_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from robomimic.utils.file_utils import policy_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    cfg = load_env_cfg(BC_TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 142
    cfg.auto_reset = False
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    policy, _ = policy_from_checkpoint(ckpt_path=str(args.checkpoint), device=torch.device("cuda:0"))
    action = env.action_manager.get_term("joint_pos")
    indexes = {name: action._target_names.index("right_" + name)
               for name in ("shoulder_pitch", "elbow_pitch", "gripper")}
    cube = env.scene["cube"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(args.source, "r") as source, h5py.File(args.out, "w") as out:
        source.copy("data", out)
        for name, value in source.attrs.items():
            out.attrs[name] = value
        data = out["data"]
        original_count = len(data)
        for number, seed in enumerate(range(args.first_seed, args.first_seed + args.episodes), original_count):
            obs, _ = env.reset(seed=seed)
            policy.start_episode()
            actor, next_actor, reference_actions = [], [], []
            for step in range(round(3.0 / env.step_dt)):
                t = (step + 1) * env.step_dt
                lower = max(0.0, min(1.0, (t - 0.4) / 1.1))
                withdraw = max(0.0, min(1.0, (t - 1.8) / 0.7))
                target = {
                    "shoulder_pitch": 0.103 + lower * (-0.035 - 0.103) + withdraw * (0.103 + 0.035),
                    "elbow_pitch": -1.754 + lower * (-0.986 + 1.754) + withdraw * (-1.754 + 0.986),
                    "gripper": 0.35 * (1 - max(0.0, min(1.0, (t - 1.5) / 0.3))),
                }
                label = np.zeros(action.action_dim, dtype=np.float32)
                for name, value in target.items():
                    i = indexes[name]
                    label[i] = float((value - action._offset[0, i]) / action._scale[0, i])
                state = obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy()
                conditioned = np.concatenate((state, np.array([step / round(3.0 / env.step_dt)], dtype=np.float32)))
                with torch.no_grad():
                    predicted = policy({"actor": conditioned})
                command = torch.as_tensor(predicted, dtype=torch.float32, device="cuda:0").unsqueeze(0)
                actor.append(state)
                reference_actions.append(label)
                obs, _, terminated, timed_out, _ = env.step(command)
                next_actor.append(obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy())
                if bool(terminated[0]) or bool(timed_out[0]):
                    break
            group = data.create_group(f"demo_{number}")
            group.create_dataset("actions", data=np.stack(reference_actions))
            group.create_dataset("obs/actor", data=np.stack(actor))
            group.create_dataset("next_obs/actor", data=np.stack(next_actor))
            group.create_dataset("rewards", data=np.zeros(len(actor), dtype=np.float32))
            dones = np.zeros(len(actor), dtype=np.int64)
            dones[-1] = 1
            group.create_dataset("dones", data=dones)
            group.attrs["num_samples"] = len(actor)
            group.attrs["seed"] = seed
            group.attrs["executed_policy_checkpoint"] = str(args.checkpoint)
            group.attrs["actions_are_expert_labels_not_executed_actions"] = True
            row = {
                "seed": seed,
                "steps": len(actor),
                "final_cube_xyz_m": [round(float(v), 5) for v in cube.data.root_link_pos_w[0].tolist()],
                "expert_labels_on_policy_states": True,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        data.attrs["total"] = sum(int(group.attrs["num_samples"]) for group in data.values())
        out.attrs["aggregation"] = "20 original expert demos plus saved-policy visited states labeled by scripted time reference"
        out.attrs["reference_limitation"] = "scripted label does not adapt its target to a displaced cube"
    args.summary.write_text(json.dumps({
        "task": BC_TASK_ID,
        "source": str(args.source),
        "checkpoint": str(args.checkpoint),
        "startup_seed": 142,
        "original_episodes": original_count,
        "on_policy_labeled_episodes": len(rows),
        "reference_limitation": "time-based scripted teacher ignores displaced cube; this is DAgger-style supervision, not an adaptive human expert",
        "episodes": rows,
    }, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
