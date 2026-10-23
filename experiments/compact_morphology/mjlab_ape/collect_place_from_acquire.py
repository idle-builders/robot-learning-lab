"""Label real saved-policy acquisition handoffs with a checked place teacher."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import ape_walk.task  # register native tasks
from ape_walk.grasp import ACQUIRE_COLLISION_TASK_ID
from ape_walk.place_tracking import BC_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from robomimic.utils.file_utils import policy_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--acquire-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=3000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--goal-x", type=float, default=0.08)
    parser.add_argument("--goal-y", type=float, default=-0.0845)
    parser.add_argument("--release-shoulder-pitch", type=float, default=-0.035)
    parser.add_argument("--release-elbow-pitch", type=float, default=-0.986)
    parser.add_argument("--rollout-checkpoint", type=Path,
                        help="Execute this learned place policy, while saving teacher labels on its visited states")
    parser.add_argument("--goal-scale", type=float, default=1.0,
                        help="Input scaling used by --rollout-checkpoint (training data stays raw)")
    args = parser.parse_args()
    if args.goal_scale <= 0:
        parser.error("--goal-scale must be positive")

    cfg = load_env_cfg(ACQUIRE_COLLISION_TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 142
    cfg.auto_reset = False
    cfg.episode_length_s = 6.0
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    acquire, _ = policy_from_checkpoint(ckpt_path=str(args.acquire_checkpoint), device=torch.device("cuda:0"))
    rollout = None
    if args.rollout_checkpoint:
        rollout, _ = policy_from_checkpoint(ckpt_path=str(args.rollout_checkpoint), device=torch.device("cuda:0"))
    robot, cube = env.scene["robot"], env.scene["cube"]
    command_term = env.command_manager.get_term("lift_height")
    action_term = env.action_manager.get_term("joint_pos")
    tracked = ("right_shoulder_pitch", "right_elbow_pitch", "right_gripper")
    action_indexes = {name: action_term._target_names.index(name) for name in tracked}
    joint_indexes = {name: robot.joint_names.index(name) for name in tracked}
    place_pose = load_env_cfg(BC_TASK_ID, play=True).scene.entities["robot"].init_state.joint_pos
    acquire_defaults = robot.data.default_joint_pos.clone()
    acquire_offsets = action_term._offset.clone()
    sensors = env.scene.sensors
    hand_site = robot.site_names.index("right_hand_center")
    stage_steps = round(3.0 / env.step_dt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with h5py.File(args.source, "r") as source, h5py.File(args.out, "w") as out:
        source.copy("data", out)
        for key, value in source.attrs.items():
            out.attrs[key] = value
        data = out["data"]
        original_count = len(data)
        accepted = 0
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            robot.data.default_joint_pos[:] = acquire_defaults
            action_term._offset[:] = acquire_offsets
            obs, _ = env.reset(seed=seed)
            acquire.start_episode()
            for step in range(stage_steps):
                actor = obs["actor"][0].detach().cpu().numpy()
                actor = np.concatenate((actor, np.array([step / stage_steps], dtype=np.float32)))
                with torch.no_grad():
                    prediction = acquire({"actor": actor})
                obs, _, terminated, timed_out, _ = env.step(
                    torch.as_tensor(prediction, dtype=torch.float32, device="cuda:0").unsqueeze(0))
                if bool(terminated[0]) or bool(timed_out[0]):
                    break
            initial_cube = cube.data.root_link_pos_w[0].detach().cpu().numpy().copy()
            fixed = bool(sensors["right_hand_cube_contact"].data.found[0].any())
            moving = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
            acquired = len(range(step + 1)) == stage_steps and initial_cube[2] >= 0.04 and fixed and moving
            if not acquired:
                results.append({"seed": seed, "acquired": False, "accepted": False})
                continue

            command_term.target_pos[0] = torch.tensor([args.goal_x, args.goal_y, 0.01], device="cuda:0")
            for name in tracked:
                value = place_pose[rf"^{name}$"]
                robot.data.default_joint_pos[0, joint_indexes[name]] = value
                action_term._offset[0, action_indexes[name]] = value
            env.action_manager.reset()
            obs = {"actor": env.observation_manager.compute_group("actor", update_history=False)}
            if rollout:
                rollout.start_episode()
            actor_rows, next_rows, action_rows, samples = [], [], [], []
            clear = longest_clear = 0
            for step in range(stage_steps):
                t = (step + 1) * env.step_dt
                lower = max(0.0, min(1.0, (t - 0.4) / 1.1))
                withdraw = max(0.0, min(1.0, (t - 1.8) / 0.7))
                target = {
                    "right_shoulder_pitch": 0.103 + lower * (args.release_shoulder_pitch - 0.103) + withdraw * (0.103 - args.release_shoulder_pitch),
                    "right_elbow_pitch": -1.754 + lower * (args.release_elbow_pitch + 1.754) + withdraw * (-1.754 - args.release_elbow_pitch),
                    "right_gripper": 0.35 * (1 - max(0.0, min(1.0, (t - 1.5) / 0.3))),
                }
                action = torch.zeros((1, action_term.action_dim), device="cuda:0")
                for name, value in target.items():
                    i = action_indexes[name]
                    action[0, i] = (value - action_term._offset[0, i]) / action_term._scale[0, i]
                actor_rows.append(obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy())
                action_rows.append(action[0].detach().cpu().numpy().astype(np.float32).copy())
                executed_action = action
                if rollout:
                    conditioned = actor_rows[-1].copy()
                    if conditioned.shape[0] != 72:
                        raise ValueError(f"expected 72-D place actor, got {conditioned.shape}")
                    conditioned[47:50] *= args.goal_scale
                    conditioned = np.concatenate((conditioned, np.array([step / stage_steps], dtype=np.float32)))
                    with torch.no_grad():
                        prediction = rollout({"actor": conditioned})
                    executed_action = torch.as_tensor(prediction, dtype=torch.float32, device="cuda:0").unsqueeze(0)
                obs, _, terminated, timed_out, _ = env.step(executed_action)
                next_rows.append(obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy())
                xyz = cube.data.root_link_pos_w[0].detach().cpu().numpy().copy()
                hand = robot.data.site_pos_w[0, hand_site].detach().cpu().numpy().copy()
                fixed = bool(sensors["right_hand_cube_contact"].data.found[0].any())
                moving = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
                grounded = bool(sensors["feet_ground_contact"].data.found[0].all())
                placed = xyz[2] <= 0.015 and not fixed and not moving and hand[2] - xyz[2] >= 0.03
                clear = clear + 1 if placed else 0
                longest_clear = max(longest_clear, clear)
                samples.append((xyz, grounded))
                if bool(terminated[0]) or bool(timed_out[0]):
                    break
            final_xyz = samples[-1][0]
            goal_error = float(np.linalg.norm(final_xyz[:2] - [args.goal_x, args.goal_y]))
            tail_drift = max(float(np.linalg.norm(s[0][:2] - final_xyz[:2])) for s in samples[-25:])
            accepted_run = len(samples) == stage_steps and longest_clear * env.step_dt >= 0.5
            accepted_run = accepted_run and goal_error <= 0.015 and tail_drift <= 0.002
            accepted_run = accepted_run and all(s[1] for s in samples)
            task_success = accepted_run
            if rollout:
                accepted_run = bool(samples)
            result = {
                "seed": seed, "acquired": True,
                "goal_xy_m": [args.goal_x, args.goal_y],
                "release_joint_targets_rad": [args.release_shoulder_pitch, args.release_elbow_pitch],
                "handoff_cube_xyz_m": [round(float(v), 5) for v in initial_cube],
                "final_cube_xyz_m": [round(float(v), 5) for v in final_xyz],
                "longest_clear_place_s": round(longest_clear * env.step_dt, 3),
                "goal_error_m": round(goal_error, 5), "tail_drift_m": round(tail_drift, 5),
                "accepted": accepted_run, "task_success": task_success,
            }
            if accepted_run:
                group = data.create_group(f"demo_{original_count + accepted}")
                group.create_dataset("actions", data=np.stack(action_rows))
                group.create_dataset("obs/actor", data=np.stack(actor_rows))
                group.create_dataset("next_obs/actor", data=np.stack(next_rows))
                group.create_dataset("rewards", data=np.zeros(len(action_rows), dtype=np.float32))
                dones = np.zeros(len(action_rows), dtype=np.int64)
                dones[-1] = 1
                group.create_dataset("dones", data=dones)
                group.attrs["num_samples"] = len(action_rows)
                group.attrs["seed"] = seed
                group.attrs["handoff_from_saved_acquire_checkpoint"] = str(args.acquire_checkpoint)
                group.attrs["goal_xy_m"] = [args.goal_x, args.goal_y]
                if rollout:
                    group.attrs["executed_policy_checkpoint"] = str(args.rollout_checkpoint)
                    group.attrs["actions_are_expert_labels_not_executed_actions"] = True
                accepted += 1
            results.append(result)
            print(json.dumps(result), flush=True)
        data.attrs["total"] = sum(int(g.attrs["num_samples"]) for g in data.values())
        out.attrs["aggregation"] = ("existing place data plus expert labels on saved place-policy visited states"
                                    if rollout else "existing place data plus successful teacher places from saved acquire policy handoffs")
    args.summary.write_text(json.dumps({
        "source": str(args.source), "acquire_checkpoint": str(args.acquire_checkpoint),
        "goal_xy_m": [args.goal_x, args.goal_y],
        "release_joint_targets_rad": [args.release_shoulder_pitch, args.release_elbow_pitch],
        "startup_seed": 142, "attempted": args.episodes, "accepted": accepted,
        "teacher": ("scripted_place_reference_labels_on_learned_place_rollout_states_not_executed_actions"
                    if rollout else "scripted_place_reference_after_learned_acquisition_not_learned_place_success"),
        "rollout_checkpoint": str(args.rollout_checkpoint) if rollout else None,
        "rollout_goal_scale": args.goal_scale if rollout else None,
        "episodes": results,
    }, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
