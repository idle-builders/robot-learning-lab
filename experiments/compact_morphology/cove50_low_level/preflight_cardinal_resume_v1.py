"""One-world local checkpoint/interface check for the cardinal PPO pilot."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
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
from ape_walk import task as ape  # noqa: E402
from experiments.compact_morphology.cove50_policy_transfer import train_velocity  # noqa: E402
from experiments.compact_morphology.cove50_low_level.preflight_locomotion_full_state_v3b import equal_state  # noqa: E402

OUT = ROOT / "artifacts/ape-low-level-training-20261014/directional-planar-entry-v1"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
BASE = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
TRAINER = ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    captured = {}
    original_launch, original_argv = train_velocity.launch_training, sys.argv
    try:
        train_velocity.launch_training = lambda task, cfg: captured.update(task=task, cfg=cfg)
        sys.argv = [str(TRAINER), "--body-xml", str(BODY), "--profile", "cardinal",
                    "--worlds", "1024", "--updates", "40", "--run-name", "cardinal-v1",
                    "--resume-run", "cardinal-v1-source", "--resume-checkpoint", "model_844.pt",
                    "--ankle-target-margin-rad", "0.02"]
        with (OUT / "cardinal-resume-build.log").open("w") as log, redirect_stdout(log):
            train_velocity.main()
    finally:
        train_velocity.launch_training, sys.argv = original_launch, original_argv
    cfg = captured["cfg"]
    assert cfg.agent.resume and cfg.agent.load_checkpoint == "model_844.pt"
    assert cfg.env.scene.num_envs == 1024 and cfg.agent.max_iterations == 40
    cfg.env.scene.num_envs = 1
    cfg.env.auto_reset = False
    cfg.env.seed = cfg.agent.seed = 47991
    saved = torch.load(BASE, map_location="cpu", weights_only=False)
    assert saved["iter"] == 844
    with (OUT / "cardinal-resume-runtime.log").open("w") as log, redirect_stdout(log):
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg.env, device="cpu"),
                                clip_actions=cfg.agent.clip_actions)
        try:
            runner = ape.VelocityOnPolicyRunner(env, asdict(cfg.agent), device="cpu")
            runner.load(str(BASE), strict=True, map_location="cpu")
            mismatches = {
                "actor": equal_state(runner.alg.actor.state_dict(), saved["actor_state_dict"]),
                "critic": equal_state(runner.alg.critic.state_dict(), saved["critic_state_dict"]),
                "optimizer": equal_state(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"]),
            }
            obs = env.get_observations()
            with torch.inference_mode():
                action = runner.alg.actor(obs)
                value = runner.alg.critic(obs)
            command = env.unwrapped.command_manager.get_command("twist")[0, :3].tolist()
            report = {
                "schema": "robot-learning-lab.cardinal-full-state-entry.v1",
                "source_checkpoint_sha256": sha(BASE), "body_sha256": sha(BODY),
                "trainer_sha256": sha(TRAINER),
                "source_iteration": int(saved["iter"]),
                "loaded_iteration": int(runner.current_learning_iteration),
                "strict_state_mismatches": mismatches,
                "actor_observation_shape": list(obs["actor"].shape),
                "critic_observation_shape": list(obs["critic"].shape),
                "action_shape": list(action.shape), "value_shape": list(value.shape),
                "first_command": command,
                "first_action_finite": bool(torch.isfinite(action).all()),
                "first_value_finite": bool(torch.isfinite(value).all()),
                "control_dt_s": cfg.env.sim.mujoco.timestep * cfg.env.decimation,
                "launch_worlds": 1024, "launch_updates": 40,
                "claim_limit": "Local one-world CPU full-state entry only; zero optimizer updates, no timed transition or CUDA score.",
            }
        finally:
            env.close()
    assert report["loaded_iteration"] == 844
    assert all(not v for v in mismatches.values())
    assert report["actor_observation_shape"] == [1, 72]
    assert report["critic_observation_shape"] == [1, 87]
    assert report["action_shape"] == [1, 21]
    assert report["first_action_finite"] and report["first_value_finite"]
    assert report["control_dt_s"] == .02
    (OUT / "cardinal-resume-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
