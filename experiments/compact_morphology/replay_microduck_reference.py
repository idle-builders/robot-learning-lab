"""Headless replay of Pollen's released Microduck ONNX policies and CPU BAM controller.

Run from the repository root. This calls the upstream inference class and actuator
setup; the only local code is the episode schedule, measurements, and recording.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".cache/upstream/microduck_rl"
POLICIES = ROOT / ".cache/upstream/microduck-policies"
SCENE = UPSTREAM / "src/mjlab_microduck/robot/microduck/scene.xml"
INFER = UPSTREAM / "scripts/infer_policy.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--walk-speed", type=float, default=0.3)
    parser.add_argument("--walk-policy", default="alpha_walking.onnx")
    parser.add_argument("--scene", default="scene.xml")
    parser.add_argument("--vin", type=float, default=7.4)
    parser.add_argument("--vin-min", type=float, default=None)
    parser.add_argument("--delay-control-ticks", type=int, default=0)
    parser.add_argument("--current-limit", type=float, default=0.0)
    parser.add_argument("--behavior", choices=("ground_reach", "kick_left"), default="ground_reach")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("microduck_infer", INFER)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    bam_model = upstream.load_bam_model(
        200, args.vin, args.current_limit if args.current_limit > 0 else None
    )
    scene = SCENE.parent / args.scene
    vin_min = args.vin_min if args.vin_min is not None else min(6.0, args.vin)
    if vin_min > args.vin:
        parser.error("--vin-min cannot exceed --vin")
    model, data, bam_ctrl, _ = upstream.load_mujoco_with_bam(
        str(scene), bam_model, 0.005, 0.1, vin_min
    )
    walk_path = POLICIES / args.walk_policy
    ground_path = POLICIES / "alpha_ground_pick.onnx"
    kick_path = POLICIES / "ball_kick_left.onnx"
    policy = upstream.PolicyInference(
        model, data, walking_onnx_path=str(walk_path),
        ground_pick_onnx_path=str(ground_path), bam_ctrl=bam_ctrl,
        use_projected_gravity=True, new_cmd_obs=True,
        delay_min_lag=args.delay_control_ticks,
        delay_max_lag=args.delay_control_ticks,
        kick_left_onnx_path=str(kick_path) if args.behavior == "kick_left" else None,
        kick_duration=0.5,
    )
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    root_q = int(model.jnt_qposadr[free_id])
    data.qpos[root_q : root_q + 7] = [0, 0, 0.125, 1, 0, 0, 0]
    for i, qidx in enumerate(policy.joint_qpos_indices):
        data.qpos[qidx] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    if policy.ball_qpos_adr is not None:
        data.qpos[policy.ball_qpos_adr : policy.ball_qpos_adr + 7] = [5, 0, 0.035, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    policy.set_vel_cmd(0, 0, 0)

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    mouth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    renderer = writer = None
    if args.render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.25, 0.0, 0.10]
        camera.distance = 0.75
        camera.azimuth = 130
        camera.elevation = -16
        writer = imageio.get_writer(args.out / "replay.mp4", fps=12.5, quality=7)

    trace = []
    last_mode = None
    try:
        for tick in range(600):  # 12 s at the upstream 50 Hz control rate.
            t = tick * 0.02
            behavior_end = 6.5 if args.behavior == "kick_left" else 8.8
            mode = "walk" if 1.0 <= t < 5.0 else args.behavior if 6.0 <= t < behavior_end else "stand"
            if mode != last_mode:
                if mode == "walk":
                    policy.set_vel_cmd(args.walk_speed, 0, 0)
                elif mode == "ground_reach":
                    policy.set_vel_cmd(0, 0, 0)
                    policy.trigger_ground_pick()
                elif mode == "kick_left":
                    policy.set_vel_cmd(0, 0, 0)
                    policy.trigger_behavior("kick_left")
                else:
                    policy.set_vel_cmd(0, 0, 0)
                last_mode = mode
            policy.update_ground_pick_phase(0.02)
            policy.update_behavior(0.02)
            action = policy.infer()
            policy.apply_action(action)
            for _ in range(4):  # 0.005 s dynamics, 0.02 s policy step.
                bam_ctrl.update()
                mujoco.mj_step(model, data)
            feet = [False, False]
            ball_foot_contact = False
            other_ground_geoms = set()
            for ci in range(data.ncon):
                c = data.contact[ci]
                pair = {int(c.geom1), int(c.geom2)}
                if floor in pair and left in pair:
                    feet[0] = True
                if floor in pair and right in pair:
                    feet[1] = True
                if ball_geom in pair and (left in pair or right in pair):
                    ball_foot_contact = True
                if floor in pair:
                    other = pair - {floor, left, right}
                    for geom_id in other:
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        other_ground_geoms.add(name or f"geom_{geom_id}")
            trace.append({
                "t": round(t + 0.02, 3), "mode": mode,
                "active_policy": policy.current_policy,
                "root_xyz": [round(float(x), 5) for x in data.qpos[root_q : root_q + 3]],
                "ball_xyz": [round(float(x), 5) for x in data.qpos[policy.ball_qpos_adr : policy.ball_qpos_adr + 3]] if policy.ball_qpos_adr is not None else None,
                "mouth_z": round(float(data.site_xpos[mouth, 2]), 5),
                "foot_contact": feet,
                "ball_foot_contact": ball_foot_contact,
                "other_ground_contacts": sorted(other_ground_geoms),
                "action_l2": round(float(np.linalg.norm(action)), 5),
                "max_abs_torque": round(float(np.max(np.abs(data.actuator_force))), 5),
            })
            if writer is not None and tick % 4 == 0:
                camera.lookat[0] = max(0.17, float(data.qpos[root_q]) - 0.05)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    def segment(mode):
        samples = [r for r in trace if r["mode"] == mode]
        both = sum(all(r["foot_contact"]) for r in samples)
        single = sum(sum(r["foot_contact"]) == 1 for r in samples)
        return {
            "samples": len(samples), "both_foot_fraction": both / len(samples),
            "single_foot_fraction": single / len(samples),
            "start_x": samples[0]["root_xyz"][0], "end_x": samples[-1]["root_xyz"][0],
            "min_root_z": min(r["root_xyz"][2] for r in samples),
            "min_mouth_z": min(r["mouth_z"] for r in samples),
            "max_abs_torque": max(r["max_abs_torque"] for r in samples),
            "mean_action_l2": round(float(np.mean([r["action_l2"] for r in samples])), 5),
            "other_ground_contact_geoms": sorted({
                g for r in samples for g in r["other_ground_contacts"]
            }),
            "ball_displacement_x": round(samples[-1]["ball_xyz"][0] - samples[0]["ball_xyz"][0], 5) if samples[0]["ball_xyz"] else None,
            "ball_foot_contact_frames": sum(r["ball_foot_contact"] for r in samples),
        }

    summary = {
        "source_repo": "pollen-robotics/microduck_rl",
        "source_commit": "cb70b792312d559a4da09064d92009079671815f",
        "bam_commit": "62bd8ce12154340be97e06f7f41a0ca8f116d967",
        "model_sha256": sha256(scene),
        "scene": args.scene,
        "policies": {p.name: sha256(p) for p in (walk_path, ground_path, *([kick_path] if args.behavior == "kick_left" else []))},
        "robot_mass_kg": round(float(np.sum(model.body_mass) - (model.body_mass[ball_body] if ball_body >= 0 else 0)), 6),
        "actuator_count": model.nu,
        "motor_voltage_v": args.vin,
        "motor_voltage_floor_v": vin_min,
        "command_delay_control_ticks": args.delay_control_ticks,
        "firmware_current_limit_a": args.current_limit or None,
        "command_walk_m_per_s": args.walk_speed,
        "behavior": args.behavior,
        "segments": {name: segment(name) for name in ("stand", "walk", args.behavior)},
        "trace": trace,
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
