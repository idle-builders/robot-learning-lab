"""Replay saved acquire and place policies in one uninterrupted MJLab episode."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio
import numpy as np
import torch

import ape_walk.task  # register native tasks
from ape_walk.grasp import ACQUIRE_COLLISION_TASK_ID
from ape_walk.grasp import COLLISION_MODEL
from ape_walk.place_tracking import BC_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from robomimic.utils.file_utils import policy_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire-checkpoint", type=Path, required=True)
    parser.add_argument("--place-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--state-out", type=Path,
                        help="Save full robot/cube poses for an independent MuJoCo contact-depth audit")
    parser.add_argument("--first-seed", type=int, default=1142)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--goal-x", type=float, default=0.08)
    parser.add_argument("--goal-y", type=float, default=-0.0845)
    parser.add_argument("--goal-scale", type=float, default=1.0)
    parser.add_argument("--goal-xy-only", action="store_true")
    parser.add_argument("--nominal-friction", action="store_true",
                        help="Disable MJLab's startup fingertip-friction randomization for repeatable nominal replay")
    parser.add_argument("--explicit-goal", action="store_true",
                        help="Supply Robomimic goal={'target_xy': [x,y]} to the place policy")
    parser.add_argument("--full-body-contact", action="store_true",
                        help="Preserve v5 XML torso/limb collision masks for a new physics replay")
    parser.add_argument("--physics-model", choices=("baseline", "candidate"), default="baseline")
    args = parser.parse_args()
    if args.goal_scale <= 0:
        parser.error("--goal-scale must be positive")
    if args.goal_xy_only and args.goal_scale == 1.0:
        parser.error("--goal-xy-only requires --goal-scale")

    cfg = load_env_cfg(ACQUIRE_COLLISION_TASK_ID, play=True)
    body_model = COLLISION_MODEL
    if args.physics_model == "candidate":
        from ape_walk.physics.candidate import MODEL, configure_candidate
        cfg = configure_candidate(cfg)
        body_model = MODEL
    if args.full_body_contact:
        # The historical v5 tasks kept MJLab's selective sole/hand import,
        # which disabled unlisted body geoms despite the collision-proxy XML.
        # Opt in here so old checkpoint replays remain exactly reproducible.
        cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (
            ape_walk.task.ContactSensorCfg(
                name="self_body_contact",
                primary=ape_walk.task.ContactMatch(
                    mode="subtree", pattern="pelvis", entity="robot"),
                secondary=ape_walk.task.ContactMatch(
                    mode="subtree", pattern="pelvis", entity="robot"),
                fields=("found", "force"), reduce="none", num_slots=1,
            ),
        )
    cfg.scene.num_envs = 1
    cfg.seed = 142
    cfg.auto_reset = False
    cfg.episode_length_s = 6.0
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    if args.nominal_friction:
        for name in ("fingertip_friction_slide", "fingertip_friction_spin", "fingertip_friction_roll"):
            cfg.events.pop(name)
    if args.video:
        cfg.viewer.width, cfg.viewer.height = 640, 480
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
    acquire, _ = policy_from_checkpoint(ckpt_path=str(args.acquire_checkpoint), device=torch.device("cuda:0"))
    place, _ = policy_from_checkpoint(ckpt_path=str(args.place_checkpoint), device=torch.device("cuda:0"))
    robot, cube = env.scene["robot"], env.scene["cube"]
    command = env.command_manager.get_term("lift_height")
    sensors = env.scene.sensors
    hand_site = robot.site_names.index("right_hand_center")
    tracked = ("right_shoulder_pitch", "right_elbow_pitch", "right_gripper")
    joint_indexes = {name: robot.joint_names.index(name) for name in tracked}
    action_term = env.action_manager.get_term("joint_pos")
    action_indexes = {name: action_term._target_names.index(name) for name in tracked}
    place_pose = load_env_cfg(BC_TASK_ID, play=True).scene.entities["robot"].init_state.joint_pos
    acquire_joint_defaults = robot.data.default_joint_pos.clone()
    acquire_action_offsets = action_term._offset.clone()
    stage_steps = round(3.0 / env.step_dt)
    frames, episodes, pose_rows = [], [], []

    for seed in range(args.first_seed, args.first_seed + args.episodes):
        # Reuse one simulator but return its control coordinates to the acquire
        # task before the next fresh reset.
        robot.data.default_joint_pos[:] = acquire_joint_defaults
        action_term._offset[:] = acquire_action_offsets
        obs, _ = env.reset(seed=seed)
        acquire.start_episode()
        place.start_episode()
        stage, rows, handoff = "acquire", [], None
        for step in range(2 * stage_steps):
            if step == stage_steps:
                stage = "place"
                # Only the goal changes; robot, cube and all physics state continue.
                command.target_pos[0] = torch.tensor([args.goal_x, args.goal_y, 0.01], device="cuda:0")
                # Native joint-relative observations and normalized actions use
                # per-task reset offsets. Switch those control coordinates too.
                for name in tracked:
                    value = place_pose[rf"^{name}$"]
                    robot.data.default_joint_pos[0, joint_indexes[name]] = value
                    action_term._offset[0, action_indexes[name]] = value
                env.action_manager.reset()
                # The step-return observation still contains the old goal and
                # old action history. Recompute it before the first place action.
                obs = {"actor": env.observation_manager.compute_group("actor", update_history=False)}
                handoff = rows[-1].copy()
            actor = obs["actor"][0].detach().cpu().numpy()
            if stage == "place" and args.goal_scale != 1.0:
                if actor.shape[0] != 72:
                    raise ValueError(f"expected 72-D place actor before scaling, got {actor.shape}")
                actor = actor.copy()
                actor[47:49 if args.goal_xy_only else 50] *= args.goal_scale
            actor = np.concatenate((actor, np.array([(step % stage_steps) / stage_steps], dtype=np.float32)))
            with torch.no_grad():
                if stage == "place" and args.explicit_goal:
                    goal = {"target_xy": np.asarray([args.goal_x, args.goal_y], dtype=np.float32)}
                    prediction = place({"actor": actor}, goal=goal)
                else:
                    prediction = (acquire if stage == "acquire" else place)({"actor": actor})
            action = torch.as_tensor(prediction, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            obs, _, terminated, timed_out, _ = env.step(action)
            if args.state_out:
                pose_rows.append((seed,
                    np.concatenate((robot.data.root_link_pose_w[0].detach().cpu().numpy(),
                                    robot.data.joint_pos[0].detach().cpu().numpy())),
                    cube.data.root_link_pose_w[0].detach().cpu().numpy().copy()))
            if args.video and seed == args.first_seed and step % 2 == 0:
                frames.append(env.render())
            cube_pos = cube.data.root_link_pos_w[0].detach().cpu().numpy()
            hand_pos = robot.data.site_pos_w[0, hand_site].detach().cpu().numpy()
            fixed = bool(sensors["right_hand_cube_contact"].data.found[0].any())
            moving = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
            rows.append({
                "time_s": round((step + 1) * env.step_dt, 3),
                "stage": stage,
                "cube_xyz_m": [round(float(v), 5) for v in cube_pos],
                "hand_xyz_m": [round(float(v), 5) for v in hand_pos],
                "jaw_contacts": [fixed, moving],
                "both_feet_grounded": bool(sensors["feet_ground_contact"].data.found[0].all()),
                "self_body_contact": (bool(sensors["self_body_contact"].data.found[0].any())
                                      if args.full_body_contact else None),
                "max_leg_torque_nm": round(float(robot.data.actuator_force[0, :10].abs().max()), 4),
                "max_abs_action": round(float(action.abs().max()), 4),
                "tracked_joint_rad": {name: round(float(robot.data.joint_pos[0, joint_indexes[name]]), 4)
                                      for name in tracked},
                "tracked_action": {name: round(float(action[0, action_indexes[name]]), 4)
                                   for name in tracked},
            })
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        place_rows = [r for r in rows if r["stage"] == "place"]
        acquired = handoff is not None and handoff["cube_xyz_m"][2] >= 0.04 and all(handoff["jaw_contacts"])
        clear_streak = longest_clear = 0
        for row in place_rows:
            clear = row["cube_xyz_m"][2] <= 0.015 and not any(row["jaw_contacts"])
            clear = clear and row["hand_xyz_m"][2] - row["cube_xyz_m"][2] >= 0.03
            clear_streak = clear_streak + 1 if clear else 0
            longest_clear = max(longest_clear, clear_streak)
        final_xy_error = float(np.linalg.norm(np.array(rows[-1]["cube_xyz_m"][:2]) - [args.goal_x, args.goal_y]))
        tail_drift = max(float(np.linalg.norm(np.array(row["cube_xyz_m"][:2])
                                              - np.array(rows[-1]["cube_xyz_m"][:2])))
                         for row in rows[-25:])
        result = {
            "seed": seed,
            "steps": len(rows),
            "handoff": handoff,
            "acquired_at_handoff": acquired,
            "final_cube_xyz_m": rows[-1]["cube_xyz_m"],
            "final_xy_goal_error_m": round(final_xy_error, 5),
            "longest_clear_place_s": round(longest_clear * env.step_dt, 3),
            "last_0p5s_xy_drift_m": round(tail_drift, 5),
            "both_feet_grounded_fraction": round(sum(r["both_feet_grounded"] for r in rows) / len(rows), 3),
            "self_body_contact_fraction": (round(sum(r["self_body_contact"] for r in rows) / len(rows), 3)
                                           if args.full_body_contact else None),
            "composed_success": acquired and longest_clear >= round(0.5 / env.step_dt)
                and final_xy_error <= 0.015 and tail_drift <= 0.002
                and all(r["both_feet_grounded"] for r in rows),
            "timeseries": rows,
        }
        episodes.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "timeseries"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "status": "saved_policies_continuous_closed_loop_handoff",
        "task": ACQUIRE_COLLISION_TASK_ID,
        "physics_model": args.physics_model,
        "body_xml": str(body_model),
        "body_sha256": hashlib.sha256(body_model.read_bytes()).hexdigest(),
        "acquire_checkpoint": str(args.acquire_checkpoint),
        "acquire_checkpoint_sha256": hashlib.sha256(args.acquire_checkpoint.read_bytes()).hexdigest(),
        "place_checkpoint": str(args.place_checkpoint),
        "place_checkpoint_sha256": hashlib.sha256(args.place_checkpoint.read_bytes()).hexdigest(),
        "switch_s": 3.0,
        "goal_after_switch_xyz_m": [args.goal_x, args.goal_y, 0.01],
        "cube_to_goal_scale": args.goal_scale,
        "cube_to_goal_scale_axes": "xy" if args.goal_xy_only else "xyz",
        "nominal_fingertip_friction": args.nominal_friction,
        "explicit_robomimic_goal": args.explicit_goal,
        "full_body_contact_enabled": args.full_body_contact,
        "episodes": episodes,
    }, indent=2) + "\n")
    if args.video:
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_out,
                            seeds=np.asarray([row[0] for row in pose_rows]),
                            qpos=np.asarray([row[1] for row in pose_rows]),
                            cube_pose=np.asarray([row[2] for row in pose_rows]),
                            joint_names=np.asarray(robot.joint_names),
                            dt=env.step_dt)
    env.close()


if __name__ == "__main__":
    main()
