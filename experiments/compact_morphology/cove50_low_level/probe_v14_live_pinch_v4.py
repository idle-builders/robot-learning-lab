"""One live planned native V14 touch-to-two-pad diagnostic; no learner score."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg

import ape_walk.task  # register exact native task
from ape_walk import reach
from collect_free_object_touch import BODY
from ape_walk.physics.frame50 import require_direct_frame
from free_object_touch_env import ROOT
from touch_reference_residual_v14 import residual_touch_env_cfg
from train_touch_contact_phase_v14 import build_train_config
from screen_v14_attained_pinches import screen_model
from live_pinch_planner_v4 import plan_live_pinch


BASE = ROOT / "artifacts/ape-low-level-training-20261015"
V14 = BASE / "object-touch-contact-phase-v14"
OUT = BASE / "object-live-pinch-v4"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
       "right_elbow_pitch", "right_gripper")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sensor(name: str, primary: str, secondary: str, entity: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=primary, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern=secondary, entity=entity),
        fields=("found", "force"), reduce="netforce", num_slots=1)


def force(env, name: str) -> float:
    value = env.scene.sensors[name].data.force[0].detach().cpu().numpy()
    return float(np.linalg.norm(value.reshape(-1, value.shape[-1]), axis=-1).max())


def smooth(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def arm_goal(index: int, start: np.ndarray, path: np.ndarray,
             destination: np.ndarray) -> np.ndarray:
    # 20 open, 150 arm, 20 close, 50 hold controller frames after the switch.
    target = np.r_[start[:4], start[4] * (1 - smooth((index + 1) / 20)) + .5 * smooth((index + 1) / 20)]
    if index >= 20:
        progress = smooth((index - 19) / 150)
        segments = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        distance = progress * segments[-1]
        point = min(len(path) - 2, max(0, int(np.searchsorted(segments, distance, side="right") - 1)))
        weight = (distance - segments[point]) / max(segments[point + 1] - segments[point], 1e-9)
        target[:4] = path[point] * (1 - weight) + path[point + 1] * weight
        target[4] = .5
    if index >= 170:
        target[4] = .5 * (1 - smooth((index - 169) / 20)) + destination[4] * smooth((index - 169) / 20)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    method_path = OUT / "method.json"
    method = json.loads(method_path.read_text())
    checkpoint = V14 / "model_11.pt"
    files = {"runner": Path(__file__), "body": BODY, "checkpoint": checkpoint,
             "planner": Path(__file__).with_name("live_pinch_planner_v4.py"),
             "screen": Path(__file__).with_name("screen_v14_attained_pinches.py"),
             "v14_train": Path(__file__).with_name("train_touch_contact_phase_v14.py"),
             "touch_env": Path(__file__).with_name("free_object_touch_env.py"),
             "v14_action": Path(__file__).with_name("touch_reference_residual_v14.py")}
    if any(sha(path) != method["source_sha256"][name] for name, path in files.items()):
        raise ValueError("Frozen live pinch source changed")
    if args.device != "cuda:0" or OUT.joinpath("report.json").exists():
        raise ValueError("Only one original CUDA diagnostic visit is permitted")
    if method["seed"] != 37301 or method["planned_native_steps"] != 411:
        raise ValueError("Frozen single-visit parameters changed")
    _, train_cfg = build_train_config(seed=37301)
    cfg = residual_touch_env_cfg(cube_xy=(.0465, -.1435))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, 37301, False
    cfg.episode_length_s = 9.0
    rated_effort = np.asarray([act.effort_limit for act in
                               cfg.scene.entities["robot"].articulation.actuators], dtype=float)
    cfg.scene.sensors += (
        sensor("forearm_budget_cube", "right_forearm_frame_budget", "cube_geom", "cube"),
        sensor("forearm_mass_cube", "right_forearm_stock_mass", "cube_geom", "cube"),
        sensor("hand_table", "right_hand", "touch_table_top", "table"),
        sensor("finger_table", "right_finger_tip", "touch_table_top", "table"),
    )
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device),
                            clip_actions=train_cfg.agent.clip_actions)
    native = env.unwrapped
    model = require_direct_frame(BODY)
    try:
        native.reset(seed=37301)
        runner = reach.ManipulationOnPolicyRunner(env, asdict(train_cfg.agent), device=args.device)
        runner.load(str(checkpoint), strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        term = native.action_manager.get_term("joint_pos")
        indices = [term._target_names.index(name) for name in ARM]
        rows = []
        pre_qpos, qpos, qvel, applied, camera, frames, actuator_force = [], [], [], [], [], [], []
        switch = None
        plan = None
        stop_reason = None
        first_cube = native.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
        for step in range(1, 412):
            obs = env.get_observations()
            actor = obs["actor"][0].detach().cpu().numpy().copy()
            pre_qpos.append(native.sim.data.qpos[0].detach().cpu().numpy().copy())
            if step <= 171:
                with torch.inference_mode():
                    action = policy(obs)
            else:
                if switch is None or plan is None or not plan["path_found"]:
                    raise RuntimeError("Live attained-state planner did not authorize transition")
                desired = arm_goal(step - 172, np.asarray(switch["arm_qpos_rad"]),
                                   np.asarray(plan["path_arm_rad"]),
                                   np.asarray(plan["target"]["arm_qpos_rad"]))
                offsets = term._offset[0, indices].detach().cpu().numpy()
                scales = term._scale[0, indices].detach().cpu().numpy()
                normalized = (desired - offsets) / scales
                if not np.isfinite(normalized).all() or np.max(np.abs(normalized)) > 1.0001:
                    raise ValueError(f"Scripted joint target requires clipping at step {step}")
                term._held_alpha[0] = 1.0
                term._teacher_target[0, indices] = torch.as_tensor(
                    desired, dtype=term._teacher_target.dtype, device=args.device)
                action = torch.zeros((1, 21), device=args.device)
            applied.append(action[0].detach().cpu().numpy().copy())
            camera.append(actor[-6:].copy())
            _, _, done, _ = env.step(action)
            current_qpos = native.sim.data.qpos[0].detach().cpu().numpy().copy()
            current_qvel = native.sim.data.qvel[0].detach().cpu().numpy().copy()
            cube = native.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
            effort = native.sim.data.actuator_force[0].detach().cpu().numpy().copy()
            qpos.append(current_qpos)
            qvel.append(current_qvel)
            actuator_force.append(effort)
            frames.append(native.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy())
            sensors = native.scene.sensors
            pad = force(native, "right_hand_cube_contact")
            finger = force(native, "right_finger_tip_cube_contact")
            forbidden = any(bool(sensors[name].data.found[0].any()) for name in
                            ("forearm_budget_cube", "forearm_mass_cube", "hand_table", "finger_table"))
            table = bool(sensors["cube_table_contact"].data.found[0].any())
            feet = bool(sensors["feet_ground_contact"].data.found[0].all())
            row = {"step": step, "time_s": round(step * native.step_dt, 4),
                   "phase": "v14_actor" if step <= 171 else
                       ("open" if step <= 191 else "arm" if step <= 341 else
                        "close" if step <= 361 else "hold"),
                   "pad_force_n": pad, "finger_force_n": finger,
                   "cube_table": table, "feet": feet, "forbidden": forbidden,
                   "any_motor_at_rated": bool(np.any(np.abs(effort) >= .99 * rated_effort)),
                   "cube_xyz_m": cube.tolist(),
                   "pelvis_z_m": float(native.scene["robot"].data.root_link_pos_w[0, 2]),
                   "raw_action": applied[-1].tolist(),
                   "post_arbitration_action": term._raw_actions[0].detach().cpu().tolist(),
                   "actuator_force_nm": effort.tolist(),
                   "rgb_six": actor[-6:].tolist()}
            rows.append(row)
            if step == 171:
                arm_qpos = np.asarray([current_qpos[model.joint(name).qposadr[0]]
                                       for name in ARM], dtype=float)
                switch = {"arm_qpos_rad": arm_qpos.tolist(),
                          "cube_xyz_m": cube.tolist(),
                          "one_pad_contact": bool(max(pad, finger) > .05),
                          "cube_table": table, "feet": feet,
                          "forbidden": forbidden, "pelvis_z_m": row["pelvis_z_m"]}
                if not switch["one_pad_contact"] or not table or not feet or forbidden or row["pelvis_z_m"] < .10:
                    stop_reason = "learned_prefix_lacked_clean_attained_one_pad_entry"
                    break
                if np.max(np.abs(current_qpos[28:31] - cube)) > 1e-6:
                    stop_reason = "native_cube_qpos_identity_fault"
                    break
                plan = plan_live_pinch(screen_model(), current_qpos.astype(float), seed=37301)
                if not plan["target"]["clean_static_target"] or not plan["path_found"]:
                    stop_reason = "no_clean_actual_state_target_and_path"
                    break
            if bool(done[0]):
                stop_reason = "native_episode_terminated_early"
                break
        longest = current = 0
        for row in rows[171:]:
            clean = (row["pad_force_n"] > .05 and row["finger_force_n"] > .05 and
                     row["cube_table"] and row["feet"] and not row["forbidden"] and
                     row["pelvis_z_m"] >= .10 and not row["any_motor_at_rated"])
            current = current + 1 if clean else 0
            longest = max(longest, current)
        OUT.mkdir(parents=True, exist_ok=True)
        trace = OUT / "seed-37301-live-original.npz"
        video = OUT / "seed-37301-live-original.mp4"
        np.savez_compressed(trace, pre_qpos=np.asarray(pre_qpos), qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            policy_action=np.asarray(applied), rgb_six=np.asarray(camera),
                            actuator_force_nm=np.asarray(actuator_force),
                            cube_xyz_m=np.asarray([r["cube_xyz_m"] for r in rows]))
        imageio.mimwrite(video, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        result = {"schema": "robot-learning-lab.v14-live-pinch-v4.v1",
                  "classification": "one uninterrupted native learned-touch to online planned scripted pinch diagnostic; not learned acquisition",
                  "method_sha256": sha(method_path), "body_sha256": sha(BODY),
                  "checkpoint_sha256": sha(checkpoint),
                  "device": args.device, "seed": 37301,
                  "steps": len(rows), "switch": switch, "plan": plan,
                  "stop_reason": stop_reason,
                  "longest_clean_two_pad_s": longest * native.step_dt,
                  "native_entry_pass": bool(longest >= 10 and len(rows) > 171),
                  "peak_pad_force_n": max(r["pad_force_n"] for r in rows),
                  "peak_finger_force_n": max(r["finger_force_n"] for r in rows),
                  "motor_rated_effort_nm": rated_effort.tolist(),
                  "motor_delay_physics_steps_range": [0, 4],
                  "fraction_transition_frames_any_motor_at_rated_effort": float(np.mean(
                      [r["any_motor_at_rated"] for r in rows[171:]])) if len(rows) > 171 else None,
                  "cube_drift_from_reset_mm": float(np.linalg.norm(np.asarray(rows[-1]["cube_xyz_m"][:2]) - first_cube[:2]) * 1000),
                  "cube_drift_from_switch_mm": float(np.linalg.norm(np.asarray(rows[-1]["cube_xyz_m"][:2]) - np.asarray(rows[170]["cube_xyz_m"][:2])) * 1000) if len(rows) >= 171 else None,
                  "forbidden_frames_after_switch": sum(r["forbidden"] for r in rows[171:]),
                  "rows": rows, "trace": str(trace.relative_to(ROOT)), "trace_sha256": sha(trace),
                  "video": str(video.relative_to(ROOT)), "video_sha256": sha(video)}
        (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in
                          ("steps", "longest_clean_two_pad_s", "native_entry_pass", "stop_reason")}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
