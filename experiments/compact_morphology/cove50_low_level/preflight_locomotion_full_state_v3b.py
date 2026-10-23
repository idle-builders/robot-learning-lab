"""Local V3 stop-task full-state load check; never opens scored cases or trains.

The frozen V3 method and trainer are untouched. This checks the exact trainer
config builder and upstream RSL-RL full checkpoint loader on a one-world CPU
sentinel. CUDA construction, stochastic rollout and capacity remain separate.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/mjlab_ape/src"))
sys.path.insert(0, str(ROOT))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.os import get_checkpoint_path  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.tasks import registry as task_registry  # noqa: E402
from ape_walk import task as ape  # noqa: E402
from ape_walk.physics.frame50 import clip_frame50_leg_targets, configure_frame50_motors  # noqa: E402
from experiments.compact_morphology.cove50_policy_transfer import train_velocity  # noqa: E402


ART = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3b"
BASE = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SEED = 47990  # Unscored sentinel; disjoint from V2, V3 development and sealed score.


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "no installed distribution metadata; source file hash recorded"


def stable(obj):
    """Make a resolved config comparable without object addresses."""
    if isinstance(obj, dict):
        return {str(k): stable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [stable(v) for v in obj]
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type(partial(int, base=10))):
        return {"partial": stable(obj.func), "args": stable(obj.args), "kwargs": stable(obj.keywords)}
    if callable(obj):
        return f"{obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(stable(obj), sort_keys=True).encode()).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    a = t.detach().contiguous().cpu().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def equal_state(actual, expected, path="") -> list[str]:
    """Exact nested comparison, including optimizer moments and normalizers."""
    errors = []
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected.cpu()):
            errors.append(path)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            errors.append(path + ".keys")
        else:
            for key in expected:
                errors.extend(equal_state(actual[key], expected[key], f"{path}.{key}"))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            errors.append(path + ".sequence")
        else:
            for i, item in enumerate(expected):
                errors.extend(equal_state(actual[i], item, f"{path}[{i}]"))
    elif actual != expected:
        errors.append(path)
    return errors


def trainer_cfg(arm: str, pose_std: float):
    """Capture the real trainer builder at its launch boundary; no launch."""
    captured = {}
    original_launch, original_argv = train_velocity.launch_training, sys.argv

    def capture(task_id, cfg):
        captured.update(task_id=task_id, cfg=cfg)

    try:
        # The production trainer registers a fixed task name once per process.
        # This CPU harness builds two arms in one process without modifying it.
        task_registry._REGISTRY.pop("Mjlab-Velocity-Flat-Ape-Cove50-Varied", None)
        train_velocity.launch_training = capture
        sys.argv = [
            str(ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"),
            "--body-xml", str(BODY), "--profile", "stop",
            "--worlds", "512", "--updates", "48", "--run-name", f"v3b-{arm}",
            "--resume-run", f"v3b-{arm}", "--resume-checkpoint", "model_844.pt",
            "--ankle-target-margin-rad", "0.02",
            "--walking-hip-roll-pose-std", str(pose_std),
        ]
        train_velocity.main()
    finally:
        task_registry._REGISTRY.pop("Mjlab-Velocity-Flat-Ape-Cove50-Varied", None)
        sys.argv = original_argv
        train_velocity.launch_training = original_launch
    assert captured["task_id"] == "Mjlab-Velocity-Flat-Ape-Cove50-Varied"
    return captured["cfg"]


def arm_check(arm: str, pose_std: float, source: dict) -> dict:
    arm_root = ART / "resume-source" / f"v3b-{arm}"
    arm_root.mkdir(parents=True, exist_ok=True)
    cp = arm_root / "model_844.pt"
    shutil.copy2(BASE, cp)
    resolved = get_checkpoint_path(ART / "resume-source", f"v3b-{arm}", "model_844.pt")
    assert resolved == cp and sha(cp) == sha(BASE)

    cfg = trainer_cfg(arm, pose_std)
    assert cfg.agent.resume and cfg.agent.load_run == f"v3b-{arm}"
    assert cfg.agent.load_checkpoint == "model_844.pt"
    launch_worlds, launch_updates = cfg.env.scene.num_envs, cfg.agent.max_iterations
    assert (launch_worlds, launch_updates) == (512, 48)
    cfg.env.scene.num_envs = 1  # CPU sentinel only; launch request remains recorded.
    cfg.env.seed = SEED
    cfg.agent.seed = SEED
    cfg.env.auto_reset = False
    full_env_cfg = stable(asdict(cfg.env))
    normalized_env_cfg = json.loads(json.dumps(full_env_cfg))
    for phase in ("std_walking", "std_running"):
        normalized_env_cfg["rewards"]["pose"]["params"][phase][r".*hip_roll.*"] = 0.05
    normalized_agent_cfg = stable(asdict(cfg.agent))
    normalized_agent_cfg["run_name"] = "paired-arm"
    normalized_agent_cfg["load_run"] = "paired-arm"

    torch.manual_seed(SEED)
    with (ART / f"{arm}-local.log").open("w") as log, redirect_stdout(log):
        env = RslRlVecEnvWrapper(
            ManagerBasedRlEnv(cfg=cfg.env, device="cpu"),
            clip_actions=cfg.agent.clip_actions,
        )
        try:
            runner = ape.VelocityOnPolicyRunner(env, asdict(cfg.agent), device="cpu")
            runner.load(str(resolved), strict=True, map_location="cpu")
            assert runner.current_learning_iteration == 844
            mismatches = {
                "actor": equal_state(runner.alg.actor.state_dict(), source["actor_state_dict"]),
                "critic": equal_state(runner.alg.critic.state_dict(), source["critic_state_dict"]),
                "optimizer": equal_state(runner.alg.optimizer.state_dict(), source["optimizer_state_dict"]),
            }
            obs = env.get_observations()
            with torch.inference_mode():
                action = runner.alg.actor(obs)
                value = runner.alg.critic(obs)
                actor_latent = runner.alg.actor.get_latent(obs)
                critic_latent = runner.alg.critic.get_latent(obs)
                torch.manual_seed(SEED + 1)
                sampled_action = runner.alg.act(obs)
            cmd = env.unwrapped.command_manager.get_command("twist")[:, :3].clone()
            term = env.unwrapped.command_manager.get_term("twist")
            root_z = float(env.unwrapped.scene["robot"].data.root_link_pose_w[0, 2])
            ids = torch.arange(env.num_envs, device=env.device)
            term._resample_command(ids)
            first_resample = term.command[:, :3].clone()
            first_stopped = term.just_stopped.clone()
            term._resample_command(ids)
            second_resample = term.command[:, :3].clone()
            second_stopped = term.just_stopped.clone()
            out = {
                "arm": arm, "resume_path": str(resolved.relative_to(ROOT)),
                "resume_sha256": sha(resolved), "strict_state_mismatches": mismatches,
                "iteration": runner.current_learning_iteration,
                "launch_worlds_updates": [launch_worlds, launch_updates],
                "cpu_sentinel_worlds": 1, "seed": SEED,
                "actor_observation_shape": list(obs["actor"].shape),
                "critic_observation_shape": list(obs["critic"].shape),
                "action_shape": list(action.shape), "value_shape": list(value.shape),
                "first_actor_observation_sha256": tensor_sha(obs["actor"]),
                "first_critic_observation_sha256": tensor_sha(obs["critic"]),
                "first_normalized_actor_latent_sha256": tensor_sha(actor_latent),
                "first_normalized_critic_latent_sha256": tensor_sha(critic_latent),
                "first_deterministic_action_sha256": tensor_sha(action),
                "first_seeded_sampled_action_sha256": tensor_sha(sampled_action),
                "first_value_sha256": tensor_sha(value),
                "first_command_m_s_and_rad_s": cmd[0].tolist(),
                "first_root_z_m": root_z,
                "command_class": type(term).__name__,
                "first_manual_resample_command": first_resample[0].tolist(),
                "second_manual_resample_command": second_resample[0].tolist(),
                "first_manual_resample_just_stopped": bool(first_stopped[0]),
                "second_manual_resample_just_stopped": bool(second_stopped[0]),
                "full_resolved_env_config_sha256": digest(full_env_cfg),
                "normalized_resolved_env_config_sha256": digest(normalized_env_cfg),
                "normalized_resolved_agent_config_sha256": digest(normalized_agent_cfg),
                "control_dt_s": cfg.env.sim.mujoco.timestep * cfg.env.decimation,
                "clip_actions": cfg.agent.clip_actions,
                "stop_resample_s": list(cfg.env.commands["twist"].resampling_time_range),
                "standing_fraction": cfg.env.commands["twist"].rel_standing_envs,
                "pose_std_standing_rad": cfg.env.rewards["pose"].params["std_standing"][r".*hip_roll.*"],
                "pose_std_walking_rad": cfg.env.rewards["pose"].params["std_walking"][r".*hip_roll.*"],
                "pose_std_running_rad": cfg.env.rewards["pose"].params["std_running"][r".*hip_roll.*"],
                "initial_root_config_z_m": cfg.env.scene.entities["robot"].init_state.pos[2],
                "motor_delay_steps_min_max": [
                    [act.delay_min_lag, act.delay_max_lag]
                    for act in cfg.env.scene.entities["robot"].articulation.actuators
                ],
                "action_scales": dict(cfg.env.actions["joint_pos"].scale),
            }
        finally:
            env.close()
    return out


def eval_contract() -> dict:
    """Relevant evaluator config for exact V2/V3 forward→stop CLI, before CUDA."""
    cfg = load_env_cfg(ape.ELBOW_TASK_ID, play=True)
    cfg.scene.entities["robot"].spec_fn = partial(ape._ape_spec, BODY)
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    configure_frame50_motors(cfg, BODY, rail_v=12.0, delay_s=0.0)
    clip_frame50_leg_targets(cfg, BODY, margin_rad=0.02, axes=("ankle_pitch",))
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    pose[r".*shoulder_pitch.*"] = 0.0
    pose[r".*elbow_pitch.*"] = 0.0
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.commands["twist"].rel_standing_envs = 0.0
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.35, 0.35)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (100.0, 100.0)
    return {
        "task": ape.ELBOW_TASK_ID,
        "control_dt_s": cfg.sim.mujoco.timestep * cfg.decimation,
        "clip_actions": load_rl_cfg(ape.ELBOW_TASK_ID).clip_actions,
        "action_scales": dict(cfg.actions["joint_pos"].scale),
        "first_command_m_s_and_rad_s": [0.35, 0.0, 0.0],
        "command_resample_s": list(command.resampling_time_range),
        "command_stop_mechanism": "evaluate.py zeros live command at --stop-at-s; no resample before 100 s",
        "actor_observation_corruption": cfg.observations["actor"].enable_corruption,
        "initial_root_z_m": cfg.scene.entities["robot"].init_state.pos[2],
        "motor_delay_steps_min_max": [
            [act.delay_min_lag, act.delay_max_lag]
            for act in cfg.scene.entities["robot"].articulation.actuators
        ],
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    source = torch.load(BASE, map_location="cpu", weights_only=False)
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"}
    assert required <= source.keys() and source["iter"] == 844
    control = arm_check("control", 0.05, source)
    candidate = arm_check("candidate", 0.15, source)
    evalu = eval_contract()
    paired = {
        "exact_cpu_first_actor_observation": control["first_actor_observation_sha256"] == candidate["first_actor_observation_sha256"],
        "exact_cpu_first_critic_observation": control["first_critic_observation_sha256"] == candidate["first_critic_observation_sha256"],
        "exact_cpu_first_deterministic_action": control["first_deterministic_action_sha256"] == candidate["first_deterministic_action_sha256"],
        "exact_cpu_first_seeded_sampled_action": control["first_seeded_sampled_action_sha256"] == candidate["first_seeded_sampled_action_sha256"],
        "exact_cpu_first_value": control["first_value_sha256"] == candidate["first_value_sha256"],
        "exact_actor_normalizer_output": control["first_normalized_actor_latent_sha256"] == candidate["first_normalized_actor_latent_sha256"],
        "exact_critic_normalizer_output": control["first_normalized_critic_latent_sha256"] == candidate["first_normalized_critic_latent_sha256"],
        "full_env_config_same_except_hip_roll_pose_std": control["normalized_resolved_env_config_sha256"] == candidate["normalized_resolved_env_config_sha256"],
        "agent_config_same_except_isolated_run_names": control["normalized_resolved_agent_config_sha256"] == candidate["normalized_resolved_agent_config_sha256"],
        "manual_stop_resamples_hold_zero": all(
            all(abs(v) < 1e-9 for v in a[key])
            for a in (control, candidate)
            for key in ("first_manual_resample_command", "second_manual_resample_command")
        ),
        "only_intended_reward_std_changes": all(control[k] == candidate[k] for k in (
            "pose_std_standing_rad", "action_scales", "control_dt_s", "clip_actions",
            "stop_resample_s", "standing_fraction", "command_class")) and
            control["pose_std_walking_rad"] == 0.05 and candidate["pose_std_walking_rad"] == 0.15,
        "trainer_evaluator_action_scale_and_clip_match": control["action_scales"] == evalu["action_scales"] and control["clip_actions"] == evalu["clip_actions"],
        "trainer_evaluator_control_dt_match": control["control_dt_s"] == evalu["control_dt_s"] == 0.02,
    }
    checks = [*paired.values(), *(not errs for a in (control, candidate) for errs in a["strict_state_mismatches"].values())]
    paths = [
        Path("artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
        Path("artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"),
        Path("experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"),
        Path("experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py"),
        Path("experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py"),
        Path("experiments/compact_morphology/mjlab_ape/evaluate.py"),
        Path(".venv-control/lib/python3.12/site-packages/mjlab/scripts/train.py"),
        Path(".venv-control/lib/python3.12/site-packages/rsl_rl/runners/on_policy_runner.py"),
        Path(".venv-control/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py"),
    ]
    method_path = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/method.json"
    method = json.loads(method_path.read_text())
    frozen_source_match = {
        "body": sha(BODY) == method["source"]["body_sha256"],
        "checkpoint": sha(BASE) == method["source"]["baseline_checkpoint_sha256"],
        "trainer": sha(ROOT / method["source"]["trainer"]) == method["source"]["trainer_sha256"],
        "task": sha(ROOT / method["source"]["task"]) == method["source"]["task_sha256"],
        "evaluator": sha(ROOT / method["source"]["evaluator"]) == method["source"]["evaluator_sha256"],
    }
    checks.extend(frozen_source_match.values())
    report = {
        "schema": "robot-learning-lab.cove50-locomotion-v3b-local-full-state-preflight.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "local_full_state_pass_launch_blocked_on_cuda_and_entry_parity" if all(checks) else "local_preflight_fault",
        "scope": "Unscored one-world CPU sentinels and static evaluator contract; zero optimizer updates, zero V3 development/score starts, zero GPU spend.",
        "sentinel_seed": SEED, "checkpoint_iter": source["iter"],
        "arms": {"control": control, "candidate": candidate},
        "paired_checks": paired, "evaluator_contract": evalu,
        "frozen_method_sha256": sha(method_path),
        "frozen_source_hashes_match": frozen_source_match,
        "observed_train_eval_differences": {
            "initial_root_config_z_m": {"trainer": control["initial_root_config_z_m"], "evaluator": evalu["initial_root_z_m"]},
            "command": "Trainer samples x=0.20-0.45 m/s and yaw=-0.20..0.20 rad/s, then resamples to zero at 3-4 s. Evaluator fixes x=0.25/0.35, yaw=0 and explicitly zeros live command at 3/4 s.",
            "actor_observation_noise": "Trainer enables actor noise; evaluator play config disables corruption.",
            "motor_delay_steps": {"trainer": sorted(set(map(tuple, control["motor_delay_steps_min_max"]))), "evaluator": sorted(set(map(tuple, evalu["motor_delay_steps_min_max"])))},
            "reset_variation": "Trainer varies base tilt/yaw, joint offset/velocity, mass/inertia, friction, effort and PD at reset. Evaluator uses play reset and fixed sensitivity cases. Matched-entry parity is unproved.",
        },
        "source_sha256": {str(path): sha(ROOT / path) for path in paths},
        "local_package_versions": {name: package_version(name) for name in (
            "mjlab", "rsl-rl", "torch", "mujoco", "warp-lang")},
        "required_cuda_gate": [
            "Repeat strict two-arm load, first actor/critic observation, normalizer, deterministic/sampled action and value parity on the exact pinned CUDA closure and matched physical reset. CPU parity is not CUDA parity.",
            "Prove the live trainer resolver finds the correct isolated source checkpoint in logs/rsl_rl/ape_cove50_velocity; abort if it selects another run or starts at iteration 0.",
            "Compare trainer/evaluator realized first physical qpos and commanded action with identical no-noise/reset parameters; evaluate.py uses play-mode no observation corruption, fixed command and zero motor delay while training uses noisy observations, random commands and 0-4 physics-step delay.",
            "Exercise exact stop resample in CUDA plus intended 512/1024/2048-world reset/contact/finite stress and measured PPO throughput before any optimizer update.",
        ],
    }
    (ART / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "paired_checks": paired,
                      "strict_loads": {a: v["strict_state_mismatches"] for a, v in report["arms"].items()},
                      "report": str((ART / "report.json").relative_to(ROOT))}, indent=2))
    if not all(checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
