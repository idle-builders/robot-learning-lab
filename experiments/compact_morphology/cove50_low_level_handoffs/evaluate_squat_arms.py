"""One-scene squat → stand → both-arm raise/lower handoff of saved BC actors.

This is an exploratory low-level composition, not a trained combined controller.
The two actors share the physical target contract; only shoulder-pitch targets
come from the arm actor after the switch. The environment is never reset.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

import ape_walk.task as task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--squat-actor", type=Path, required=True)
    parser.add_argument("--arm-actor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=54122)
    parser.add_argument("--rail-v", type=float, default=12.0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Unscored exact-task sentinel may stop after one step")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1000:
        parser.error("--max-steps must be between 1 and 1000")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.compact_morphology.cove50_low_level.multicommand_bc import load_student
    from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
        expose_squat_blend, mask_squat_action_history, retarget,
    )
    from ape_walk.physics.frame50 import (
        clip_frame50_shoulder_targets, configure_frame50_motors,
        reveal_frame50_proxy_geometry_for_video,
    )

    body = args.body.resolve()
    body_sha = hashlib.sha256(body.read_bytes()).hexdigest()
    if body_sha != "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c":
        raise ValueError("compiled Direct Frame body differs from saved actor contract")
    task_id = task.ELBOW_SQUAT_HEADING_OBS_TASK_ID
    cfg = load_env_cfg(task_id, play=True)
    retarget(cfg, body, mujoco.MjModel.from_xml_path(str(body)), "squat", variance=False)
    expose_squat_blend(cfg)
    mask_squat_action_history(cfg)
    configure_frame50_motors(cfg, body, rail_v=args.rail_v, delay_s=args.delay_ms / 1000)
    clip_frame50_shoulder_targets(cfg, body, margin_rad=0.05)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        task.ContactSensorCfg(
            name="handoff_self_contact",
            primary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
        task.ContactSensorCfg(
            name="handoff_body_ground_contact",
            primary=task.ContactMatch(mode="geom", pattern=(
                r"^pelvis_frame_budget$", r"^torso_frame_compute_budget$",
                r"^(left|right)_(thigh|shin|upper_arm|forearm)_frame_budget$",
            ), entity="robot"),
            secondary=task.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = 20.5
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    reset = cfg.events["set_ground_state"].params
    reset["standing_prob"], reset["sitting_prob"] = 1.0, 0.0
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat = (0.0, 0.0, 0.22)
        cfg.viewer.distance = 0.7
        cfg.viewer.azimuth = 90.0
        cfg.viewer.elevation = -12.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=args.device,
                          render_mode="rgb_array" if args.video else None),
        clip_actions=load_rl_cfg(task_id).clip_actions,
    )
    try:
        robot = env.unwrapped.scene["robot"]
        if args.video:
            reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        names = list(action_term.target_names)
        shoulders = [names.index(f"{side}_shoulder_pitch") for side in ("left", "right")]
        squat_policy = load_student(args.squat_actor, args.device, names,
                                    action_term.scale, action_term.offset, "squat")
        arm_policy = load_student(args.arm_actor, args.device, names,
                                  action_term.scale, action_term.offset, "both_arms")
        command = env.unwrapped.command_manager.get_term("twist")
        dt = float(env.unwrapped.step_dt)
        if abs(dt - 0.02) > 1e-8:
            raise ValueError(f"expected 50 Hz, got {dt}")
        site_ids = [robot.site_names.index(f"{side}_hand_center")
                    for side in ("left", "right")]
        sensors = env.unwrapped.scene.sensors
        leg_rated = torch.tensor([a.effort_limit for a in
                                  cfg.scene.entities["robot"].articulation.actuators[:10]],
                                 device=args.device)
        shoulder_rated = [float(cfg.scene.entities["robot"].articulation.actuators[i].effort_limit)
                          for i in shoulders]
        initial_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
        initial_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
        trace = {key: [] for key in ("pre_qpos", "pre_qvel", "post_qpos", "post_qvel",
                                      "actor_obs", "arm_actor_obs", "squat_action",
                                      "arm_action", "applied_action", "processed_target")}
        rows, frames = [], []
        filtered = None
        previous_arm_flag = 0.0
        ended_early = False
        for step in range(args.max_steps):
            t = step * dt
            squat = 2.0 <= t < 6.0
            arm_flag = float(12.0 <= t < 16.0)
            command.vel_command_b[:, 0] = float(squat)
            with torch.inference_mode():
                obs = env.get_observations()
                if obs["actor"].shape[1] != 72:
                    raise ValueError("actor observation width changed")
                arm_obs = obs["actor"].clone()
                # Stand/squat exposes slewed alpha. The arm actor was trained on
                # the prior step's raw raise flag in this same command slot.
                arm_obs[:, 69] = previous_arm_flag
                pre_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
                pre_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
                squat_action = squat_policy(obs)
                arm_action = arm_policy({"actor": arm_obs})
                merged = squat_action.clamp(-1.0, 1.0).clone()
                # Keep the completed squat actor's neutral arms during the
                # two-second standing handoff. The arm actor begins when its
                # own raise command begins, then owns raise and lower.
                if t >= 12.0:
                    # Arm training held the legs at their default position.
                    # Retaining a feedback squat actor here made its legs
                    # chase arm-induced tilt and changed the arm task's base.
                    merged[:, :10] = 0.0
                    merged[:, shoulders] = arm_action[:, shoulders]
                # The source arm task has no unit action clamp. Its 1.0-rad
                # scale can command >1 rad; the squat scene uses 1.2 rad.
                # Clamping the converted action to 1 would change the saved
                # physical target and invalidate this interface test.
                filtered = merged.clone() if filtered is None else 0.2 * merged + 0.8 * filtered
                _, _, done, _ = env.step(filtered)
                previous_arm_flag = arm_flag
            for key, value in (
                ("pre_qpos", pre_qpos), ("pre_qvel", pre_qvel),
                ("post_qpos", env.unwrapped.sim.data.qpos[0].cpu().numpy()),
                ("post_qvel", env.unwrapped.sim.data.qvel[0].cpu().numpy()),
                ("actor_obs", obs["actor"][0].cpu().numpy()),
                ("arm_actor_obs", arm_obs[0].cpu().numpy()),
                ("squat_action", squat_action[0].cpu().numpy()),
                ("arm_action", arm_action[0].cpu().numpy()),
                ("applied_action", filtered[0].cpu().numpy()),
                ("processed_target", action_term._processed_actions[0].cpu().numpy()),
            ):
                trace[key].append(np.asarray(value).copy())
            if args.video and step % 2 == 0:
                frames.append(env.unwrapped.render())
            hands = robot.data.site_pos_w[0, site_ids, 2].cpu().numpy().copy()
            quat = robot.data.root_link_quat_w[0]
            upright = float(torch.clamp(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0))
            leg_force = robot.data.actuator_force[0, :10].abs()
            rows.append({
                "t": round(t + dt, 3), "phase": "squat" if t < 10 else "arms",
                "squat_requested": squat, "arm_requested": bool(arm_flag),
                "pelvis_z": float(robot.data.root_link_pos_w[0, 2]),
                "hand_z": hands.tolist(),
                "tilt_deg": math.degrees(math.acos(upright)),
                "feet_both": bool(sensors["feet_ground_contact"].data.found[0].all()),
                "hand_force_n": float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()),
                "body_ground": bool(sensors["handoff_body_ground_contact"].data.found[0].any()),
                "self_contact": bool(sensors["handoff_self_contact"].data.found[0].any()),
                "leg_duty": bool((leg_force >= leg_rated * 0.995).any()),
                "shoulder_duty": any(
                    abs(float(robot.data.actuator_force[0, i])) >= shoulder_rated[j] * 0.995
                    for j, i in enumerate(shoulders)
                ),
                "shoulder_force_nm": [float(robot.data.actuator_force[0, i]) for i in shoulders],
                "actor_command": float(obs["actor"][0, 69]),
                "arm_actor_command": float(arm_obs[0, 69]),
            })
            if bool(done[0]):
                ended_early = True
                break
        def phase(lo, hi):
            subset = [r for r in rows if lo <= r["t"] < hi]
            return {
                "frames": len(subset),
                "pelvis_z": float(np.mean([r["pelvis_z"] for r in subset])),
                "hand_z": np.mean([r["hand_z"] for r in subset], axis=0).tolist(),
                "feet_both_fraction": float(np.mean([r["feet_both"] for r in subset])),
                "body_ground_fraction": float(np.mean([r["body_ground"] for r in subset])),
                "leg_duty_fraction": float(np.mean([r["leg_duty"] for r in subset])),
                "shoulder_duty_fraction": float(np.mean([r["shoulder_duty"] for r in subset])),
                "max_hand_force_n": max(r["hand_force_n"] for r in subset),
                "max_tilt_deg": max(r["tilt_deg"] for r in subset),
            }
        windows = {name: phase(*span) for name, span in {
            "stand_before": (1, 2), "squat_hold": (5, 6),
            "stand_after": (9, 10), "arm_before": (11, 12),
            "arm_raised": (15, 16), "arm_lowered": (19, 20),
        }.items() if rows[-1]["t"] >= span[1]}
        squat_pass = (all(k in windows for k in ("stand_before", "squat_hold", "stand_after"))
                      and windows["stand_before"]["pelvis_z"] - windows["squat_hold"]["pelvis_z"] >= 0.04
                      and abs(windows["stand_before"]["pelvis_z"] - windows["stand_after"]["pelvis_z"]) <= 0.015)
        arm_pass = (all(k in windows for k in ("arm_before", "arm_raised", "arm_lowered"))
                    and min(np.subtract(windows["arm_raised"]["hand_z"], windows["arm_before"]["hand_z"])) >= 0.15
                    and max(abs(x) for x in np.subtract(windows["arm_lowered"]["hand_z"],
                                                      windows["arm_before"]["hand_z"])) <= 0.04)
        no_contact = all(r["hand_force_n"] < 1 and not r["body_ground"]
                         and not r["self_contact"] for r in rows)
        support = all(w["feet_both_fraction"] >= 0.95 for w in windows.values())
        rated_duty = all(w["leg_duty_fraction"] <= 0.1 and
                         w["shoulder_duty_fraction"] <= 0.1 for w in windows.values())
        continuity_gap = max(float(np.max(abs(trace["post_qpos"][i-1] - trace["pre_qpos"][i])))
                             for i in (500,)) if len(rows) > 500 else None
        report = {
            "schema": "robot-learning-lab.squat-arms-no-reset.v1",
            "claim_limit": "exploratory source-labelled control composition; CPU diagnostic is not CUDA score",
            "body": str(body), "body_sha256": body_sha,
            "checkpoint_sha256": {"squat": hashlib.sha256(args.squat_actor.read_bytes()).hexdigest(),
                                  "arms": hashlib.sha256(args.arm_actor.read_bytes()).hexdigest()},
            "device": args.device, "seed": args.seed, "rail_v": args.rail_v,
            "delay_ms": args.delay_ms, "dt_s": dt, "steps": len(rows),
            "ended_early": ended_early, "initial_qpos": initial_qpos.tolist(),
            "initial_qvel": initial_qvel.tolist(), "switch_qpos_gap": continuity_gap,
            "shoulder_action_indices": shoulders, "windows": windows,
            "squat_depth_return_pass": bool(squat_pass),
            "arm_rise_return_pass": bool(arm_pass),
            "full_continuity_gate_pass": bool(not ended_early and len(rows) == 1000
                                             and squat_pass and arm_pass and no_contact and support
                                             and rated_duty
                                             and continuity_gap is not None and continuity_gap < 1e-9),
            "no_hand_or_body_contact": no_contact,
            "support_windows_pass": support,
            "rated_duty_windows_pass": rated_duty,
            "timeseries": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.trace, initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                                target_names=np.asarray(names), dt_s=np.asarray(dt),
                                **{k: np.asarray(v) for k, v in trace.items()})
        if args.video:
            import imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
        print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
