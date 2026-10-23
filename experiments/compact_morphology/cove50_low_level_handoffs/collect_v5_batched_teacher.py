"""Measure and collect batched full walk-stop-squat-arms teacher episodes.

Every world's first uninterrupted 24 s path is evaluated. Auto-reset worlds
that fail early are excluded from the original episode; they never become a
replacement success. This is teacher data, not a learned-policy score.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

import ape_walk.task as task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


ROOT = Path(__file__).resolve().parents[3]
BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
ACTOR_SHAS = {"gait": "0e8c6862108bd880e330fd8c07d19bf012d4a034f8be6a2ef808de190e5f301f",
              "stop": "1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2",
              "squat": "0b20cc71d765ce6913adb1f70587fcbf59442bcff25b7645acd64a9e643028f4",
              "arms": "d3e718e3632b4e002adf08f16028700e16be18c4392702d975c9c38f24005774"}
SLICES = {"walk_end": (1, 2), "settled": (3, 4), "stand_before": (5, 6),
          "squat_hold": (9, 10), "stand_after": (13, 14),
          "arm_before": (15, 16), "arm_raised": (19, 20), "arm_lowered": (23, 24)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(series: dict[str, np.ndarray], world: int, steps: int, yaw: float,
            continuity: float) -> dict:
    windows = {}
    for name, (lo, hi) in SLICES.items():
        ids = np.arange(steps)[(np.arange(steps) + 1) * 0.02 >= lo]
        ids = ids[(ids + 1) * 0.02 < hi]
        if len(ids) == 0 or steps < hi * 50:
            continue
        windows[name] = {
            "pelvis_z": float(series["pelvis_z"][ids, world].mean()),
            "hand_z": series["hand_z"][ids, world].mean(axis=0).tolist(),
            "feet_both_fraction": float(series["feet_both"][ids, world].mean()),
            "leg_duty_fraction": float(series["leg_duty"][ids, world].mean()),
            "shoulder_duty_fraction": float(series["shoulder_duty"][ids, world].mean()),
        }
    present = all(name in windows for name in SLICES)
    squat = (present and windows["stand_before"]["pelvis_z"]
             - windows["squat_hold"]["pelvis_z"] >= 0.04
             and abs(windows["stand_before"]["pelvis_z"]
                     - windows["stand_after"]["pelvis_z"]) <= 0.015)
    arms = (present and min(np.subtract(windows["arm_raised"]["hand_z"],
                                  windows["arm_before"]["hand_z"])) >= 0.15
            and max(abs(x) for x in np.subtract(windows["arm_lowered"]["hand_z"],
                                              windows["arm_before"]["hand_z"])) <= 0.04)
    if steps >= 200:
        delta = series["pelvis_xy"][199, world] - series["pelvis_xy"][149, world]
        stop_tail = float(np.linalg.norm(delta))
    else:
        stop_tail = None
    if steps >= 100:
        delta = series["pelvis_xy"][99, world] - series["pelvis_xy"][0, world]
        walk = float(np.dot(delta, (math.cos(yaw), math.sin(yaw))))
    else:
        walk = 0.0
    no_contact = bool(not series["body_ground"][:steps, world].any()
                      and not series["self_contact"][:steps, world].any()
                      and np.all(series["hand_force"][:steps, world] < 1.0))
    support = present and all(w["feet_both_fraction"] >= 0.95 for name, w in windows.items()
                              if name != "walk_end")
    duty = present and all(w["leg_duty_fraction"] <= 0.1
                           and w["shoulder_duty_fraction"] <= 0.1
                           for name, w in windows.items() if name != "walk_end")
    motion = bool(steps == 1200 and walk >= 0.06 and stop_tail is not None
                  and stop_tail <= 0.03 and squat and arms and no_contact
                  and support and continuity < 1e-9)
    failures = []
    for ok, name in ((steps == 1200, "early_termination"),
                     (walk >= 0.06, "walk_progress"),
                     (stop_tail is not None and stop_tail <= 0.03, "stop_tail"),
                     (squat, "squat_return"), (arms, "arm_return"),
                     (no_contact, "forbidden_contact"), (support, "support"),
                     (duty, "rated_duty"), (continuity < 1e-9, "continuity")):
        if not ok:
            failures.append(name)
    return {"steps": steps, "walk_distance_m": walk, "stop_tail_xy_m": stop_tail,
            "squat_pass": bool(squat), "arm_pass": bool(arms),
            "no_contact": no_contact, "support_pass": bool(support),
            "rated_duty_pass": bool(duty), "continuity_gap": continuity,
            "motion_complete": motion, "strict_pass": bool(motion and duty),
            "failure_reasons": failures, "windows": windows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--gait-actor", type=Path, required=True)
    parser.add_argument("--stop-actor", type=Path, required=True)
    parser.add_argument("--squat-actor", type=Path, required=True)
    parser.add_argument("--arm-actor", type=Path, required=True)
    parser.add_argument("--worlds", type=int, choices=(1, 64, 128, 512, 1024, 2048), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rail-v", type=float, choices=(10.0, 12.0), required=True)
    parser.add_argument("--delay-ms", type=float, choices=(0.0, 20.0), required=True)
    parser.add_argument("--steps", type=int, choices=(1, 1200), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--retain-whole-traces", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Original batch output exists")
    if args.device == "cpu" and args.worlds != 1:
        parser.error("CPU is only for a local one-world source preflight")
    if args.retain_whole_traces and args.steps != 1200:
        parser.error("Only whole teacher originals can be retained as imitation data")
    if sha(args.body) != BODY_SHA:
        raise RuntimeError("Exact compiled body differs")
    actors = {"gait": args.gait_actor, "stop": args.stop_actor,
              "squat": args.squat_actor, "arms": args.arm_actor}
    if {name: sha(path) for name, path in actors.items()} != ACTOR_SHAS:
        raise RuntimeError("Saved teacher checkpoints differ")
    sys.path.insert(0, str(ROOT))
    from experiments.compact_morphology.cove50_low_level.multicommand_bc import load_student
    from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
        expose_squat_blend, mask_squat_action_history, retarget)
    from ape_walk.physics.frame50 import clip_frame50_shoulder_targets, configure_frame50_motors

    cfg = load_env_cfg(task.ELBOW_SQUAT_HEADING_OBS_TASK_ID, play=True)
    retarget(cfg, args.body.resolve(), mujoco.MjModel.from_xml_path(str(args.body.resolve())),
             "squat", variance=False)
    expose_squat_blend(cfg)
    mask_squat_action_history(cfg)
    configure_frame50_motors(cfg, args.body.resolve(), rail_v=args.rail_v,
                             delay_s=args.delay_ms / 1000)
    clip_frame50_shoulder_targets(cfg, args.body.resolve(), margin_rad=0.05)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        task.ContactSensorCfg(name="handoff_self_contact",
            primary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1),
        task.ContactSensorCfg(name="handoff_body_ground_contact",
            primary=task.ContactMatch(mode="geom", pattern=(
                r"^pelvis_frame_budget$", r"^torso_frame_compute_budget$",
                r"^(left|right)_(thigh|shin|upper_arm|forearm)_frame_budget$"), entity="robot"),
            secondary=task.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1))
    cfg.scene.num_envs, cfg.seed = args.worlds, args.seed
    cfg.auto_reset = True  # A failed world's original is censored on its first done.
    cfg.episode_length_s = 24.5
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    reset = cfg.events["set_ground_state"].params
    reset["standing_prob"], reset["sitting_prob"] = 1.0, 0.0
    clock = time.perf_counter()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device),
                             clip_actions=load_rl_cfg(task.ELBOW_SQUAT_HEADING_OBS_TASK_ID).clip_actions)
    construct_s = time.perf_counter() - clock
    try:
        robot = env.unwrapped.scene["robot"]
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        names = list(action_term.target_names)
        expected = [f"{side}_{joint}" for side in ("left", "right") for joint in task.LEG_JOINTS]
        if names[:10] != expected or len(names) != 21 or abs(env.unwrapped.step_dt - 0.02) > 1e-8:
            raise RuntimeError("Saved teacher action or 50 Hz contract changed")
        shoulders = [names.index(f"{side}_shoulder_pitch") for side in ("left", "right")]
        squat_policy = load_student(args.squat_actor, args.device, names,
                                    action_term.scale, action_term.offset, "squat")
        arm_policy = load_student(args.arm_actor, args.device, names,
                                  action_term.scale, action_term.offset, "both_arms")
        gait_cfg = load_env_cfg(task.ELBOW_TASK_ID, play=True)
        if (not gait_cfg.actions["joint_pos"].use_default_offset
                or not cfg.actions["joint_pos"].use_default_offset
                or not torch.all(action_term.scale[:, :10] > 0)):
            raise RuntimeError("Physical gait target map changed")
        gait_runner = load_runner_cls(task.ELBOW_TASK_ID)(
            env, asdict(load_rl_cfg(task.ELBOW_TASK_ID)), device=args.device)
        gait_runner.load(str(args.gait_actor), load_cfg={"actor": True},
                         strict=True, map_location=args.device)
        gait_policy = gait_runner.get_inference_policy(device=args.device)
        stop_runner = load_runner_cls(task.ELBOW_TASK_ID)(
            env, asdict(load_rl_cfg(task.ELBOW_TASK_ID)), device=args.device)
        stop_runner.load(str(args.stop_actor), load_cfg={"actor": True},
                         strict=True, map_location=args.device)
        stop_policy = stop_runner.get_inference_policy(device=args.device)
        command = env.unwrapped.command_manager.get_term("twist")
        sensors = env.unwrapped.scene.sensors
        site_ids = [robot.site_names.index(f"{side}_hand_center") for side in ("left", "right")]
        leg_rated = torch.tensor([a.effort_limit for a in
                                  cfg.scene.entities["robot"].articulation.actuators[:10]],
                                 device=args.device)
        shoulder_rated = torch.tensor([cfg.scene.entities["robot"].articulation.actuators[i].effort_limit
                                       for i in shoulders], device=args.device)
        speed = torch.tensor([(.20, .25, .30, .35)[i % 4] for i in range(args.worlds)],
                             device=args.device)
        initial_qpos = env.unwrapped.sim.data.qpos.detach().cpu().numpy().copy()
        initial_qvel = env.unwrapped.sim.data.qvel.detach().cpu().numpy().copy()
        q = initial_qpos[:, 3:7]
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                         1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        n, steps = args.worlds, args.steps
        series = {name: np.zeros(shape, dtype=dtype) for name, shape, dtype in (
            ("pelvis_z", (steps, n), np.float32), ("pelvis_xy", (steps, n, 2), np.float32),
            ("hand_z", (steps, n, 2), np.float32), ("feet_both", (steps, n), np.bool_),
            ("leg_duty", (steps, n), np.bool_), ("shoulder_duty", (steps, n), np.bool_),
            ("body_ground", (steps, n), np.bool_), ("self_contact", (steps, n), np.bool_),
            ("hand_force", (steps, n), np.float32))}
        trace = ({key: [] for key in ("pre_qpos", "post_qpos", "actor_obs", "arm_actor_obs",
                                       "applied_action", "processed_target")}
                 if args.retain_whole_traces else None)
        first_done = np.full(n, steps, dtype=np.int32)
        active = np.ones(n, dtype=np.bool_)
        filtered = None
        last_gait = torch.zeros((n, 21), device=args.device)
        previous_arm = 0.0
        continuity = np.zeros(n, dtype=np.float64)
        prior_post = None
        reset_count = 0
        warm_started = time.perf_counter()
        for step in range(steps):
            t = step * 0.02
            walking, stopping = t < 2.0, 2.0 <= t < 4.0
            squat, arm_flag = 6.0 <= t < 10.0, float(16.0 <= t < 20.0)
            command.vel_command_b[:, 0] = speed if walking else float(squat)
            command.vel_command_b[:, 2] = 0.0
            with torch.inference_mode():
                obs = env.get_observations()
                if obs["actor"].shape != (n, 72):
                    raise RuntimeError("Actor observation width changed")
                arm_obs = obs["actor"].clone()
                arm_obs[:, 69] = previous_arm
                pre = env.unwrapped.sim.data.qpos.detach().clone()
                if prior_post is not None:
                    gap = (pre - prior_post).abs().amax(dim=1).cpu().numpy()
                    continuity[active] = np.maximum(continuity[active], gap[active])
                squat_action = squat_policy(obs)
                arm_action = arm_policy({"actor": arm_obs})
                merged = squat_action.clamp(-1.0, 1.0).clone()
                if walking or stopping:
                    gait_obs = obs.clone()
                    gait_obs["actor"][:, 48:69] = last_gait
                    gait_obs["actor"][:, 69:72] = 0.0
                    gait_obs["actor"][:, 69] = speed if walking else 0.0
                    gait_action = (gait_policy if walking else stop_policy)(gait_obs)
                    last_gait = gait_action.clone()
                    merged[:] = 0.0
                    merged[:, :10] = gait_action[:, :10] * 0.3 / action_term.scale[:, :10]
                if t >= 16.0:
                    merged[:, :10] = 0.0
                    merged[:, shoulders] = arm_action[:, shoulders]
                filtered = (merged.clone() if walking or stopping or filtered is None
                            else 0.2 * merged + 0.8 * filtered)
                _, _, done, _ = env.step(filtered)
                previous_arm = arm_flag
                done_np = done.cpu().numpy().astype(bool)
                just_done = active & done_np
                first_done[just_done] = step + 1
                active[just_done] = False
                reset_count += int(done_np.sum())
                post = env.unwrapped.sim.data.qpos.detach().clone()
                prior_post = post
                if trace is not None:
                    for key, value in (("pre_qpos", pre), ("post_qpos", post),
                                       ("actor_obs", obs["actor"]), ("arm_actor_obs", arm_obs),
                                       ("applied_action", filtered),
                                       ("processed_target", action_term._processed_actions)):
                        trace[key].append(value.detach().cpu().numpy().copy())
                values = {
                    "pelvis_z": robot.data.root_link_pos_w[:, 2],
                    "pelvis_xy": robot.data.root_link_pos_w[:, :2],
                    "hand_z": robot.data.site_pos_w[:, site_ids, 2],
                    "feet_both": sensors["feet_ground_contact"].data.found.all(dim=1),
                    "leg_duty": (robot.data.actuator_force[:, :10].abs()
                                 >= leg_rated * 0.995).any(dim=1),
                    "shoulder_duty": (robot.data.actuator_force[:, shoulders].abs()
                                      >= shoulder_rated * 0.995).any(dim=1),
                    "body_ground": sensors["handoff_body_ground_contact"].data.found.any(dim=1),
                    "self_contact": sensors["handoff_self_contact"].data.found.any(dim=1),
                    "hand_force": sensors["hand_ground_contact"].data.force.norm(dim=-1).amax(dim=1),
                }
                for key, value in values.items():
                    series[key][step] = value.detach().cpu().numpy()
        rollout_s = time.perf_counter() - warm_started
        cases = []
        for world in range(n):
            # The first done step is included as a failed terminal frame; any
            # auto-reset continuation is excluded from this original episode.
            count = int(first_done[world])
            result = summary(series, world, count, float(yaw[world]), float(continuity[world]))
            result.update({"world_index": world, "walk_velocity_m_s": float(speed[world]),
                           "initial_qpos": initial_qpos[world].tolist(),
                           "initial_qvel": initial_qvel[world].tolist()})
            cases.append(result)
        args.out.mkdir(parents=True, exist_ok=False)
        metrics_path = args.out / "metrics.npz"
        np.savez_compressed(metrics_path, initial_qpos=initial_qpos,
                            initial_qvel=initial_qvel, first_done=first_done,
                            continuity=continuity, **series)
        if trace is not None:
            arrays = {key: np.stack(value) for key, value in trace.items()}
            for case in cases:
                if not case["motion_complete"]:
                    continue
                i = case["world_index"]
                path = args.out / f"world-{i:04d}.npz"
                np.savez_compressed(path, target_names=np.asarray(names), dt_s=np.asarray(0.02),
                                    initial_qpos=initial_qpos[i], initial_qvel=initial_qvel[i],
                                    **{key: value[:, i] for key, value in arrays.items()})
                case["trace"] = path.name
                case["trace_sha256"] = sha(path)
        report = {
            "schema": "robot-learning-lab.broad-handoff-v5-batched-teacher.v1",
            "classification": "batched native teacher screen; no learner update or skill score",
            "body_sha256": BODY_SHA,
            "checkpoint_sha256": ACTOR_SHAS,
            "source_sha256": sha(Path(__file__)), "device": args.device,
            "metrics_sha256": sha(metrics_path),
            "seed": args.seed, "worlds": n, "rail_v": args.rail_v,
            "delay_ms": args.delay_ms, "steps_requested": steps,
            "retained_motion_traces": trace is not None,
            "constructor_s": construct_s, "rollout_s": rollout_s,
            "total_s": construct_s + rollout_s, "reset_events": reset_count,
            "motion_complete": sum(c["motion_complete"] for c in cases),
            "strict_passes": sum(c["strict_pass"] for c in cases),
            "motion_episodes_per_gpu_hour": (3600 * sum(c["motion_complete"] for c in cases)
                                              / (construct_s + rollout_s)
                                              if args.device.startswith("cuda") else None),
            "strict_episodes_per_gpu_hour": (3600 * sum(c["strict_pass"] for c in cases)
                                              / (construct_s + rollout_s)
                                              if args.device.startswith("cuda") else None),
            "cases": cases,
        }
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in (
            "worlds", "steps_requested", "motion_complete", "strict_passes",
            "reset_events", "total_s", "motion_episodes_per_gpu_hour",
            "strict_episodes_per_gpu_hour")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
