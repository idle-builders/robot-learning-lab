"""Unscored local entry parity for the V3c forward-to-stop learning method.

This deliberately removes training noise and reset randomization only for a
matched first-state comparison. It never trains, opens development seeds, or
substitutes CPU outcomes for CUDA evaluation. Run the same gate on CUDA before
any paid optimizer update.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/mjlab_ape/src"))
sys.path.insert(0, str(ROOT))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from ape_walk import task as ape  # noqa: E402
from ape_walk.physics.frame50 import (  # noqa: E402
    clip_frame50_leg_targets, configure_frame50_motors, give_soles_contact_priority,
)
from preflight_locomotion_full_state_v3b import equal_state, tensor_sha, trainer_cfg  # noqa: E402


ART = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3c"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CHECKPOINT = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
SOURCES = (
    BODY, CHECKPOINT,
    ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py",
    ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py",
    ROOT / "experiments/compact_morphology/cove50_low_level/preflight_locomotion_full_state_v3b.py",
    Path(__file__).resolve(),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nominal_training_cfg(seed: int):
    cfg = trainer_cfg("control", 0.05).env
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.observations["actor"].enable_corruption = False
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (0.0, 0.0, 0.231)
    robot.articulation.actuators = tuple(
        replace(act, delay_min_lag=0, delay_max_lag=0)
        for act in robot.articulation.actuators
    )
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    cfg.events["cove_link_density"].params["alpha_range"] = (0.0, 0.0)
    cfg.events["cove_sole_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["cove_effort"].params["effort_limit_range"] = (1.0, 1.0)
    cfg.events["cove_pd"].params["kp_range"] = (1.0, 1.0)
    cfg.events["cove_pd"].params["kd_range"] = (1.0, 1.0)
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.35, 0.35)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (100.0, 100.0)
    return cfg


def evaluator_cfg(seed: int, root_z: float):
    # Mirrors evaluate.py's --task Elbows --physics-model frame50
    # --ankle-target-margin-rad 0.02 --yaw-only-head --arms-down
    # --velocity-x 0.35 --velocity-y 0 --angular-z 0 --stop-at-s 4.
    cfg = load_env_cfg(ape.ELBOW_TASK_ID, play=True)
    cfg.scene.entities["robot"].spec_fn = partial(ape._ape_spec, BODY)
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    configure_frame50_motors(cfg, BODY, rail_v=12.0, delay_s=0.0)
    give_soles_contact_priority(cfg)
    clip_frame50_leg_targets(cfg, BODY, margin_rad=0.02, axes=("ankle_pitch",))
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    pose[r".*shoulder_pitch.*"] = 0.0
    pose[r".*elbow_pitch.*"] = 0.0
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, root_z)
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = 8.0
    # This is the matched-entry sentinel only. Scored evaluation retains its
    # own reset variation; these zero ranges isolate config/weight parity.
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.35, 0.35)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (100.0, 100.0)
    if hasattr(command, "rel_turn_in_place_envs"):
        command.rel_turn_in_place_envs = 0.0
    return cfg


def first_state(cfg, seed: int):
    torch.manual_seed(seed)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cpu"), clip_actions=None
    )
    try:
        runner = ape.VelocityOnPolicyRunner(
            env, asdict(load_rl_cfg(ape.ELBOW_TASK_ID)), device="cpu"
        )
        runner.load(str(CHECKPOINT), strict=True, map_location="cpu")
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        mismatches = {
            "actor": equal_state(runner.alg.actor.state_dict(), saved["actor_state_dict"]),
            "critic": equal_state(runner.alg.critic.state_dict(), saved["critic_state_dict"]),
            "optimizer": equal_state(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"]),
        }
        assert runner.current_learning_iteration == saved["iter"] == 844
        assert all(not v for v in mismatches.values()), mismatches
        observations = env.get_observations()
        with torch.inference_mode():
            action = runner.alg.actor(observations)
            value = runner.alg.critic(observations)
            actor_latent = runner.alg.actor.get_latent(observations)
        robot = env.unwrapped.scene["robot"]
        return {
            "iteration": runner.current_learning_iteration,
            "strict_full_state_mismatches": mismatches,
            "qpos": env.unwrapped.sim.data.qpos[0].cpu().tolist(),
            "qvel": env.unwrapped.sim.data.qvel[0].cpu().tolist(),
            "command": env.unwrapped.command_manager.get_command("twist")[0, :3].cpu().tolist(),
            "root_z_m": float(robot.data.root_link_pose_w[0, 2]),
            "actor_observation": observations["actor"][0].cpu().tolist(),
            "critic_observation": observations["critic"][0].cpu().tolist(),
            "first_action": action[0].cpu().tolist(),
            "first_value": value[0].cpu().tolist(),
            "actor_observation_sha256": tensor_sha(observations["actor"]),
            "critic_observation_sha256": tensor_sha(observations["critic"]),
            "normalized_actor_latent_sha256": tensor_sha(actor_latent),
            "action_sha256": tensor_sha(action),
            "value_sha256": tensor_sha(value),
        }
    finally:
        env.close()


def max_gap(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    seed = 47991  # unscored, disjoint from V2/V3 development and score
    trained = first_state(nominal_training_cfg(seed), seed)
    primary = first_state(evaluator_cfg(seed, 0.231), seed)
    sensitivity = first_state(evaluator_cfg(47992, 0.181), 47992)
    gaps = {key: max_gap(trained[key], primary[key]) for key in (
        "qpos", "qvel", "command", "actor_observation",
        "critic_observation", "first_action", "first_value",
    )}
    # Exact physical entry and numerical first-decision tolerances are
    # deliberate; CPU success is a local mechanism gate only.
    checks = {
        "body_and_checkpoint_source": sha(BODY) == "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
        and sha(CHECKPOINT) == "1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2",
        "matched_primary_qpos_gap_le_1e-6": gaps["qpos"] <= 1e-6,
        "matched_primary_qvel_gap_le_1e-6": gaps["qvel"] <= 1e-6,
        "matched_primary_command_gap_le_1e-6": gaps["command"] <= 1e-6,
        "matched_primary_actor_observation_gap_le_1e-5": gaps["actor_observation"] <= 1e-5,
        "matched_primary_action_gap_le_1e-5": gaps["first_action"] <= 1e-5,
        "primary_root_z_0p231": abs(primary["root_z_m"] - 0.231) <= 0.003,
        "sensitivity_root_z_0p181": abs(sensitivity["root_z_m"] - 0.181) <= 0.003,
        "sensitivity_disjoint_seed": seed != 47992,
    }
    report = {
        "schema": "robot-learning-lab.cove50-locomotion-entry-v3c.local-preflight.v1",
        "scope": "Unscored CPU first-state parity between mirrored trainer/evaluator configurations; does not execute evaluate.py CLI, CUDA, score, or training",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        "source_v3_method_sha256": sha(ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/method.json"),
        "seeds": {"matched_primary": seed, "disjoint_0p181_sensitivity": 47992},
        "primary_entry_root_z_m": 0.231,
        "sensitivity_entry_root_z_m": 0.181,
        "matched_first_state_gaps": gaps,
        "checks": checks,
        "training_nominal": trained,
        "evaluation_primary": primary,
        "evaluation_0p181_sensitivity": sensitivity,
    }
    (ART / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": checks, "gaps": gaps,
                      "report": str((ART / "report.json").relative_to(ROOT))}, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
