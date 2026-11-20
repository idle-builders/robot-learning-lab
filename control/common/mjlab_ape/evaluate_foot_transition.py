"""Fresh native MJLab replay of a learned ape foot-transition checkpoint."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import ape_walk.task
from ape_walk.foot_transition_tracking import TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--residual", action="store_true",
                        help="Use reference-centered action task")
    parser.add_argument("--push-trained", action="store_true",
                        help="Evaluate the reference-centered task trained with velocity pushes")
    parser.add_argument("--hold-push-trained", action="store_true",
                        help="Evaluate reference tracking trained on hold-phase pushes")
    parser.add_argument("--right", action="store_true",
                        help="Evaluate the mirrored right-foot reference task")
    parser.add_argument("--step-side", choices=("left", "right"),
                        help="Evaluate the forward single-step task")
    parser.add_argument("--distilled-side", choices=("left", "right"),
                        help="Evaluate an absolute-action policy without reference joints")
    parser.add_argument("--teacher-actions", action="store_true",
                        help="Feed reference joint targets through native actuators; not learning")
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--bc-action-scale", type=float, default=3.0,
                        help="Undo dataset action scaling for Robomimic's tanh actor")
    parser.add_argument("--bc-mask-action-history", action="store_true",
                        help="Zero the actor's last-action observation, matching the BC ablation data")
    parser.add_argument("--reset-jitter", action="store_true",
                        help="Small 1 mm/0.005 rad reference-frame perturbations")
    parser.add_argument("--reset-jitter-level", choices=("tiny", "train", "hard"),
                        help="Initial root/joint perturbation for fresh-start evaluation")
    parser.add_argument("--explicit-reset", action="store_true",
                        help="Diagnostic: reset the constructed environment before replay")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--body-xml", type=Path)
    parser.add_argument("--physics-model", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--qpos-out", type=Path)
    parser.add_argument("--push-time", type=float,
                        help="Time of one lateral root-velocity impulse")
    parser.add_argument("--push-vy", type=float, default=0.0,
                        help="Impulse as an instantaneous world-y velocity change in m/s")
    parser.add_argument("--action-debug", action="store_true")
    args = parser.parse_args()
    if args.physics_model == "candidate" and args.body_xml:
        parser.error("--body-xml and --physics-model candidate are mutually exclusive")
    task_id = TASK_ID
    if args.distilled_side:
        from ape_walk import step_distill  # noqa: F401 - registers both tasks
        task_id = f"Mjlab-Distilled-Ape-Step-{args.distilled_side.title()}"
    elif args.step_side:
        from ape_walk import step  # noqa: F401 - registers the step tasks
        task_id = f"Mjlab-Tracking-Ape-Single-Step-{args.step_side.title()}"
    elif args.right:
        from ape_walk.foot_transition_residual_right import TASK_ID as right_task_id
        task_id = right_task_id
    elif args.hold_push_trained:
        from ape_walk.foot_transition_residual_hold_push import TASK_ID as hold_push_task_id
        task_id = hold_push_task_id
    elif args.push_trained:
        from ape_walk.foot_transition_residual_push import TASK_ID as push_task_id
        task_id = push_task_id
    elif args.residual:
        from ape_walk.foot_transition_residual import TASK_ID as residual_task_id
        task_id = residual_task_id
    if not args.zero_actions and not args.teacher_actions and args.checkpoint is None and args.bc_checkpoint is None:
        parser.error("give a checkpoint, --bc-checkpoint, --zero-actions, or --teacher-actions")

    cfg = load_env_cfg(task_id, play=True)
    candidate_model = None
    if args.physics_model == "candidate":
        from ape_walk.physics.candidate import MODEL, configure_candidate
        cfg = configure_candidate(cfg)
        candidate_model = MODEL
    if args.body_xml:
        cfg.scene.entities["robot"].spec_fn = partial(ape_walk.task._ape_spec, args.body_xml.resolve())
        cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    jitter_level = args.reset_jitter_level or ("tiny" if args.reset_jitter else None)
    if jitter_level:
        position, angle, joint = {
            "tiny": (0.001, 0.005, 0.005),
            "train": (0.005, 0.015, 0.015),
            "hard": (0.01, 0.03, 0.03),
        }[jitter_level]
        cfg.commands["motion"].pose_range = {
            "x": (-position, position), "y": (-position, position),
            "roll": (-angle, angle), "pitch": (-angle, angle),
        }
        cfg.commands["motion"].joint_position_range = (-joint, joint)
    if args.video:
        cfg.viewer.width = 640
        cfg.viewer.height = 480
        cfg.viewer.azimuth = 135
        cfg.viewer.elevation = -18
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0",
                              render_mode="rgb_array" if args.video else None),
        clip_actions=load_rl_cfg(task_id).clip_actions,
    )
    robot = env.unwrapped.scene["robot"]
    if args.teacher_actions:
        teacher_joint_pos = np.load(cfg.commands["motion"].motion_file)["joint_pos"]
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
    elif args.bc_checkpoint:
        from robomimic.utils.file_utils import policy_from_checkpoint
        bc_policy, _ = policy_from_checkpoint(
            ckpt_path=str(args.bc_checkpoint), device=torch.device("cuda:0"))
        bc_policy.start_episode()
        def policy(obs):
            actor = obs["actor"][0].detach().cpu().numpy().copy()
            if args.bc_mask_action_history:
                actor[-robot.data.joint_pos.shape[-1]:] = 0
            with torch.no_grad():
                predicted = bc_policy({"actor": actor})
            return args.bc_action_scale * torch.as_tensor(predicted, dtype=torch.float32, device="cuda:0")[None, :]
    elif args.zero_actions:
        policy = lambda _obs: torch.zeros_like(robot.data.joint_pos)
    else:
        runner = load_runner_cls(task_id)(env, asdict(load_rl_cfg(task_id)), device="cuda:0")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
    sensors = env.unwrapped.scene.sensors
    initial_state_hash = hashlib.sha256()
    for tensor in (env.unwrapped.sim.data.qpos, env.unwrapped.sim.data.qvel):
        initial_state_hash.update(tensor.cpu().numpy().tobytes())
    first_action_hash = None
    if args.explicit_reset:
        env.reset()
        print("initial motion frame", int(env.unwrapped.command_manager.get_term("motion").time_steps[0]))
    sole_ids = [robot.site_names.index(f"{side}_sole_center") for side in ("left", "right")]
    lead_side = args.distilled_side or args.step_side or ("right" if args.right else "left")
    stance_side = "left" if lead_side == "right" else "right"
    stance_index = 0 if stance_side == "left" else 1
    swing_index = 1 - stance_index
    support_pair = [True, False] if stance_side == "left" else [False, True]
    dt = env.unwrapped.step_dt
    rows, frames, states = [], [], []
    first_stance = first_swing = None
    for step in range(round((13.1 if args.step_side or args.distilled_side else 10.4) / dt)):
        with torch.inference_mode():
            if args.push_time is not None and abs(step * dt - args.push_time) < dt / 2:
                velocity = torch.cat((robot.data.root_link_lin_vel_w,
                                      robot.data.root_link_ang_vel_w), dim=-1).clone()
                velocity[:, 1] += args.push_vy
                robot.write_root_link_velocity_to_sim(velocity)
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.forward()
            if args.teacher_actions:
                target = torch.as_tensor(teacher_joint_pos[step], device="cuda:0")[None, :]
                action = (target - action_term.offset) / action_term.scale
            else:
                action = policy(env.get_observations())
            if step == 0:
                first_action_hash = hashlib.sha256(action.cpu().numpy().tobytes()).hexdigest()
            if args.action_debug and step in (0, 1, 10, 20):
                print("action", step, "range", float(action.min()), float(action.max()),
                      "first_10", action[0, :10].detach().cpu().tolist(), flush=True)
            _, reward, done, _ = env.step(action)
        soles = robot.data.site_pos_w[0, sole_ids].cpu().numpy()
        if first_stance is None:
            first_stance = soles[stance_index, :2].copy()
            first_swing = soles[swing_index, :2].copy()
        contacts = sensors["feet_ground_contact"].data.found[0].cpu().tolist()
        self_contact = bool(sensors["self_collision"].data.found[0].any())
        hand_force = float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max())
        root_quat = robot.data.root_link_pose_w[0, 3:7].cpu().numpy()
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (root_quat[1] ** 2 + root_quat[2] ** 2), -1, 1)))
        rows.append({
            "time_s": round((step + 1) * dt, 3),
            "pelvis_x_m": round(float(robot.data.root_link_pos_w[0, 0]), 4),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "pelvis_tilt_deg": round(float(tilt), 2),
            "swing_z_m": round(float(soles[swing_index, 2]), 4),
            "swing_sole_forward_m": round(float(soles[swing_index, 0] - first_swing[0]), 4),
            "feet_grounded": [bool(v) for v in contacts],
            "stance_sole_displacement_m": round(float(np.linalg.norm(soles[stance_index, :2] - first_stance)), 4),
            "body_contact": self_contact,
            "hand_force_n": round(hand_force, 3),
            "max_leg_torque_nm": round(float(robot.data.actuator_force[0, :10].abs().max()), 3),
            "reward": round(float(reward[0]), 4),
        })
        if args.qpos_out:
            states.append(np.concatenate((robot.data.root_link_pose_w[0].cpu().numpy(),
                                          robot.data.joint_pos[0].cpu().numpy())))
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        if bool(done[0]):
            break
    held = [r for r in rows if 7.0 <= r["time_s"] < 8.0]
    is_single = lambda r: r["feet_grounded"] == support_pair and r["swing_z_m"] > 0.03
    longest, current = 0, 0
    for row in rows:
        current = current + 1 if is_single(row) else 0
        longest = max(longest, current)
    result = {
        "task": task_id,
        "body_xml": str(candidate_model or args.body_xml or ape_walk.task.ELBOW_MODEL),
        "body_sha256": hashlib.sha256((candidate_model or args.body_xml or ape_walk.task.ELBOW_MODEL).read_bytes()).hexdigest(),
        "physics_model": args.physics_model,
        "stance_side": stance_side,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        "bc_checkpoint": str(args.bc_checkpoint) if args.bc_checkpoint else None,
        "bc_checkpoint_sha256": hashlib.sha256(args.bc_checkpoint.read_bytes()).hexdigest() if args.bc_checkpoint else None,
        "bc_action_scale": args.bc_action_scale if args.bc_checkpoint else None,
        "bc_mask_action_history": args.bc_mask_action_history if args.bc_checkpoint else None,
        "teacher_actions": args.teacher_actions,
        "reset_jitter": jitter_level,
        "seed": args.seed,
        "initial_state_sha256": initial_state_hash.hexdigest(),
        "first_action_sha256": first_action_hash,
        "push_time_s": args.push_time,
        "push_vy_mps": args.push_vy if args.push_time is not None else None,
        "duration_s": rows[-1]["time_s"],
        "longest_clear_single_support_s": round(longest * dt, 3),
        "hold_7_to_8_s_single_support_fraction": round(sum(is_single(r) for r in held) / len(held), 3) if held else 0.0,
        "final_two_foot_stand": rows[-1]["feet_grounded"] == [True, True] and rows[-1]["pelvis_z_m"] > 0.16,
        "max_stance_sole_displacement_m": max(r["stance_sole_displacement_m"] for r in rows),
        "body_contact_fraction": round(sum(r["body_contact"] for r in rows) / len(rows), 3),
        "max_hand_force_n": max(r["hand_force_n"] for r in rows),
        "max_leg_torque_nm": max(r["max_leg_torque_nm"] for r in rows),
        "timeseries": rows,
    }
    if args.step_side or args.distilled_side:
        airborne = [r for r in rows if r["feet_grounded"] == support_pair and r["swing_z_m"] > 0.02]
        landing = next((r for r in rows if r["time_s"] > 9 and r["feet_grounded"] == [True, True]
                        and r["swing_sole_forward_m"] > 0.02), None)
        result["single_forward_step"] = {
            "lead_foot": lead_side,
            "airborne_duration_s": round(len(airborne) * dt, 3),
            "max_swing_clearance_m": max(r["swing_z_m"] for r in rows),
            "touchdown_time_s": landing["time_s"] if landing else None,
            "final_swing_forward_m": rows[-1]["swing_sole_forward_m"],
            "final_pelvis_forward_m": round(rows[-1]["pelvis_x_m"] - rows[0]["pelvis_x_m"], 4),
            "final_pelvis_tilt_deg": rows[-1]["pelvis_tilt_deg"],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.qpos_out:
        np.savez_compressed(args.qpos_out, qpos=np.asarray(states), dt=dt)
    if args.video:
        import imageio
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    print(json.dumps({k: v for k, v in result.items() if k != "timeseries"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
