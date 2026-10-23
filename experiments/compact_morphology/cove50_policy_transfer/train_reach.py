"""Bounded PPO reach adaptation on the direct body and retained floor scene.

The source scene's cube location stays unchanged. This run tests whether the
bounded right-arm lesson can recover after the body transfer; it does not
change the floor-reaching contract to an easier elevated target.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ape_walk import task as ape
from ape_walk import reach
from ape_walk.physics.frame50 import (configure_frame50_motors,
                                      give_soles_contact_priority,
                                      require_direct_frame)
from reach_observation import TimedRgbTarget


def visible_goal_approach_reward(env, object_name: str = "cube") -> torch.Tensor:
    """Privileged training reward for hand proximity; actor sees only RGB."""
    hand = env.scene["robot"].data.site_pos_w[:,
        env.scene["robot"].find_sites("right_hand_center")[0][0]]
    cube = env.scene[object_name].data.root_link_pos_w
    return torch.exp(-torch.linalg.norm(hand - cube, dim=-1) / 0.05)


def near_target_gate_frame(env, threshold_m: float = 0.025) -> torch.Tensor:
    """Independent frame condition for a later sustained-reach evaluator."""
    hand = env.scene["robot"].data.site_pos_w[:,
        env.scene["robot"].find_sites("right_hand_center")[0][0]]
    cube = env.scene["cube"].data.root_link_pos_w
    feet = env.scene.sensors["feet_ground_contact"].data.found.all(dim=-1)
    no_hand_floor = ~env.scene.sensors["hand_ground_contact"].data.found.any(dim=-1)
    upright = env.scene["robot"].data.root_link_pos_w[:, 2] >= 0.10
    return (torch.linalg.norm(hand - cube, dim=-1) <= threshold_m) & feet & no_hand_floor & upright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--run-name", default="cove50-observed-reach")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-device", default="cpu",
                        help="Runtime for a bounded entry check; separate CUDA evidence from CPU")
    parser.add_argument("--preflight-dir", type=Path, default=Path(
        "artifacts/ape-low-level-training-20261013/observed-reach-entry"))
    args = parser.parse_args()
    body = args.body_xml.resolve()
    model = require_direct_frame(body)
    cfg = load_env_cfg(reach.TASK_ID, play=False)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = partial(ape._ape_spec, body)
    robot.collisions[0].disable_other_geoms = False
    give_soles_contact_priority(cfg)
    robot.init_state.pos = (0.0, 0.0, 0.1794)
    pose = robot.init_state.joint_pos
    pose[r".*hip_pitch.*"] = -1.16
    pose[r".*knee.*"] = 1.46
    pose[r".*ankle_pitch.*"] = -0.30
    # Replace the broad shoulder regex in-place, before the catch-all regex.
    # Appending an exact key after r".*" would leave the old reset unchanged.
    shoulder_pose = {}
    for pattern, value in pose.items():
        if pattern == r".*shoulder_pitch.*":
            shoulder_pose[r"^left_shoulder_pitch$"] = -1.0
            shoulder_pose[r"^right_shoulder_pitch$"] = value
        elif pattern == r".*":
            shoulder_pose[r"^head_yaw$"] = -1.5
            shoulder_pose[pattern] = value
        else:
            shoulder_pose[pattern] = value
    pose.clear()
    pose.update(shoulder_pose)
    # Virtual reach view mounted without mass or contact changes. The earlier
    # frozen +Y view saw late reach states but missed the reset toy; this -Y
    # entry candidate remains an explicit training sensor assumption.
    # The source body's group-4 proxy geoms have alpha zero. MuJoCo Warp's
    # camera drew them opaque during preflight, hiding the cube; exclude them.
    cfg.scene.sensors += (CameraSensorCfg(
        name="reach_rgb", parent_body="robot/head",
        pos=(0.048, -0.045, 0.016),
        quat=(0.70643377, 0.03084356, -0.03084356, -0.70643377),
        fovy=67.0, width=160, height=120,
        data_types=("rgb",), enabled_geom_groups=(0, 1, 2),
        use_shadows=False, use_textures=True,
    ),)
    actor_terms = cfg.observations["actor"].terms
    actor_terms.pop("ee_to_cube")
    actor_terms.pop("cube_to_goal")
    cfg.observations["critic"].terms.pop("cube_to_goal")
    actor_terms["rgb_target"] = ObservationTermCfg(
        func=TimedRgbTarget,
        params={"sensor_name": "reach_rgb", "max_age_s": 0.2,
                "min_pixels": 10},
    )
    # This stage is an observed approach. The inherited YAM lift terms reward
    # an unobserved randomized target height and conflict with this lesson.
    # Exact cube position remains available only to the training reward and
    # critic, while an independent sustained near-target gate scores outcomes.
    cfg.rewards.pop("lift")
    cfg.rewards.pop("lift_precise")
    cfg.rewards["visible_goal_approach"] = RewardTermCfg(
        func=visible_goal_approach_reward, weight=1.0)
    command = cfg.commands["lift_height"]
    command.target_position_range.z = command.object_pose_range.z
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    robot.articulation.actuators = tuple(
        replace(act, delay_min_lag=0, delay_max_lag=4)
        for act in robot.articulation.actuators
    )
    body_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                       for i in range(1, model.nbody))
    cfg.events["cove_link_density"] = EventTermCfg(
        mode="reset", func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=body_names),
                "alpha_range": (math.log(0.9) / 2, math.log(1.1) / 2)},
    )
    cfg.events["cove_sole_friction"] = EventTermCfg(
        mode="reset", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=("left_sole", "right_sole")),
                "operation": "abs", "ranges": (0.65, 1.15),
                "axes": [0], "shared_random": True},
    )
    cfg.events["cove_effort"] = EventTermCfg(
        mode="reset", func=dr.effort_limits,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "effort_limit_range": (0.9, 1.05), "operation": "scale"},
    )
    agent = deepcopy(load_rl_cfg(reach.TASK_ID))
    agent.experiment_name = "ape_cove50_reach"
    task_id = "Mjlab-Cove50-Reach-Floor-Toy"
    ape.register_mjlab_task(task_id=task_id, env_cfg=cfg, play_env_cfg=cfg,
                            rl_cfg=agent, runner_cls=reach.ManipulationOnPolicyRunner)
    train = TrainConfig.from_task(task_id)
    train.env.scene.num_envs = args.worlds
    train.env.sim.mujoco.timestep = 0.002
    train.env.decimation = 10
    train.agent.max_iterations = args.updates
    train.agent.save_interval = 25
    train.agent.run_name = args.run_name
    train.agent.logger = "tensorboard"
    # The observed actor has a different input contract from the old oracle
    # checkpoint. Do not silently reload incompatible privileged input weights.
    train.agent.resume = False
    if args.preflight_only:
        train.env.scene.num_envs = 1
        train.env.seed = 99
        local = ManagerBasedRlEnv(cfg=train.env, device=args.preflight_device)
        observation, _ = local.reset()
        terms = local.observation_manager.active_terms
        assert "ee_to_cube" not in terms["actor"]
        assert "cube_to_goal" not in terms["actor"]
        assert "ee_to_cube" in terms["critic"]
        assert "cube_to_goal" not in terms["critic"]
        assert "lift" not in local.reward_manager.active_terms
        assert "lift_precise" not in local.reward_manager.active_terms
        target_term = local.observation_manager.get_term_cfg("actor", "rgb_target").func
        raw_rgb = local.scene["reach_rgb"].data.rgb.detach().cpu().numpy()[0].copy()
        mj_model = local.sim.mj_model
        native_qpos = local.sim.data.qpos[0].detach().cpu().numpy().copy()
        head_qpos = native_qpos[mj_model.joint("robot/head_yaw").qposadr[0]]
        cube_xyz = native_qpos[mj_model.joint("cube/cube_joint").qposadr[0]:][:3]
        probe = mujoco.MjData(mj_model)
        probe.qpos[:] = native_qpos
        mujoco.mj_forward(mj_model, probe)
        camera_id = mj_model.camera("reach_rgb").id
        camera_rotation = probe.cam_xmat[camera_id].reshape(3, 3)
        cube_camera = camera_rotation.T @ (cube_xyz - probe.cam_xpos[camera_id])
        focal = (120 / 2) / np.tan(np.deg2rad(67 / 2))
        cube_uv = [79.5 + focal * cube_camera[0] / -cube_camera[2],
                   59.5 - focal * cube_camera[1] / -cube_camera[2]]
        cube_geom = mj_model.geom("cube/cube_geom")
        hand_xyz = probe.site("robot/right_hand_center").xpos.copy()
        first = target_term(local, sensor_name="reach_rgb").detach().cpu()[0]
        next_obs, *_ = local.step(torch.zeros(local.action_space.shape,
                                              device=local.device))
        next_frame = next_obs["actor"][0, -6:].detach().cpu()
        retained_rgb = local.scene["reach_rgb"].data.rgb.clone()
        retained_capture_step = int(local.episode_length_buf[0])
        blank = torch.zeros_like(local.scene["reach_rgb"].data.rgb)
        local.episode_length_buf += 1
        missing = target_term(local, sensor_name="reach_rgb", rgb_override=blank).detach().cpu()[0]
        local.episode_length_buf += 11
        stale = target_term(local, sensor_name="reach_rgb",
                            rgb_override=retained_rgb,
                            capture_step_override=retained_capture_step).detach().cpu()[0]
        reset_entries = []
        for index in range(16):
            entry_obs, _ = local.reset()
            rgb = local.scene["reach_rgb"].data.rgb.detach().cpu().numpy()[0]
            red = ((rgb[:, :, 0].astype(float) > 100) &
                   (rgb[:, :, 0] > rgb[:, :, 1] * 1.5) &
                   (rgb[:, :, 0] > rgb[:, :, 2] * 1.5))
            reset_entries.append({
                "index": index,
                "red_pixels": int(red.sum()),
                "fresh": bool(entry_obs["actor"][0, -3].item() == 1),
            })
        report = {
            "schema": "robot-learning-lab.cove50-observed-reach-entry.v1",
            "body": str(body), "body_sha256": sha256(body.read_bytes()).hexdigest(),
            "trainer_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "rgb_term_source_sha256": sha256(Path(__file__).with_name(
                "reach_observation.py").read_bytes()).hexdigest(),
            "actor_terms": terms["actor"],
            "critic_terms": terms["critic"],
            "reward_terms": local.reward_manager.active_terms,
            "near_target_gate_at_reset": bool(near_target_gate_frame(local)[0]),
            "virtual_camera": {"parent": "robot/head", "xyz_m": [0.048, -0.045, 0.016],
                               "extra_downward_pitch_deg": 45,
                               "fovy_deg": 67, "resolution": [160, 120],
                               "render_geom_groups": [0, 1, 2]},
            "actor_shape": list(observation["actor"].shape),
            "critic_shape": list(observation["critic"].shape),
            "action_shape": list(local.action_space.shape),
            "camera_shape": list(blank.shape),
            "runtime_device": args.preflight_device,
            "head_yaw_rad": float(head_qpos),
            "cube_truth_world_xyz_evaluator_only_m": cube_xyz.tolist(),
            "cube_truth_camera_xyz_evaluator_only_m": cube_camera.tolist(),
            "cube_projected_uv_evaluator_only_px": cube_uv,
            "cube_geom_group": int(cube_geom.group[0]),
            "cube_geom_rgba": cube_geom.rgba.tolist(),
            "right_hand_truth_world_xyz_evaluator_only_m": hand_xyz.tolist(),
            "hand_to_cube_truth_m": float(np.linalg.norm(hand_xyz - cube_xyz)),
            "camera_rgb_min_max": [int(raw_rgb.min()), int(raw_rgb.max())],
            "red_dominant_pixel_count": int(((raw_rgb[:, :, 0] > 100) &
                (raw_rgb[:, :, 0] > raw_rgb[:, :, 1] * 1.5) &
                (raw_rgb[:, :, 0] > raw_rgb[:, :, 2] * 1.5)).sum()),
            "first_rgb_target": first.tolist(),
            "after_one_native_zero_action_step": next_frame.tolist(),
            "injected_missing_after_next_20ms": missing.tolist(),
            "injected_frozen_rgb_after_total_260ms": stale.tolist(),
            "first_frame_detected": bool(first[3] == 1),
            "native_step_finite": bool(torch.isfinite(next_obs["actor"]).all()),
            "missing_marked": bool(missing[5] == 1),
            "stale_coordinate_cleared": bool(stale[:4].abs().sum() == 0),
            "varied_reset_entries": reset_entries,
            "varied_reset_detection": sum(row["fresh"] for row in reset_entries),
            "claim_limit": "Bounded entry and one zero-action step on the reported runtime. Missing/stale frames are injected. The camera is a massless virtual training mount and transparent group-4 proxies are excluded; no learned contact score, physical view or realistic self-occlusion has been shown.",
        }
        args.preflight_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.preflight_dir / "first-frame.png", raw_rgb)
        (args.preflight_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        local.close()
        if not (report["first_frame_detected"] and report["native_step_finite"]
                and report["missing_marked"] and report["stale_coordinate_cleared"]
                and report["varied_reset_detection"] == len(reset_entries)):
            raise RuntimeError("Observed reach entry preflight failed; see report.json")
        return
    launch_training(task_id, train)


if __name__ == "__main__":
    main()
