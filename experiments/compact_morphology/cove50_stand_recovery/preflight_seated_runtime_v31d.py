"""Unscored, disjoint CUDA runtime sentinel for the Cove seated task.

This constructs the exact V30 task on seed 34000 before any new collection
plan. The wrapper performs V30's initial reset. The first observation/action
is saved before one required diagnostic step. No scored visit or fit exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CHECKPOINT = (ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
              / "checkpoints/model_25.pt")
SENTINEL_SEED = 34000
EXPECTED = {
    "torch": "2.9.1+cu128",
    "triton": "3.5.1",
    "warp-lang": "1.12.0",
    "mjlab": "1.3.0",
    "rsl-rl-lib": "5.0.1",
    "mujoco": "3.13.0",
    "mujoco-warp": "3.8.1",
    "mjlab-microduck": "0.1.0",
    "better-actuator-models": "1.0.1",
    "numpy": "2.4.1",
    "scipy": "1.18.0",
    "pyyaml": "6.0.3",
    "huggingface-hub": "1.15.0",
    "onnxruntime": "1.24.4",
    "rustypot": "1.5.0",
    "colorama": "0.4.6",
}
MODULES = ("torch", "triton", "warp", "mjlab", "rsl_rl", "mujoco",
           "mujoco_warp", "mjlab_microduck", "bam")
BAM_COMMIT = "62bd8ce12154340be97e06f7f41a0ca8f116d967"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def run_command(*parts: str) -> str:
    result = subprocess.run([sys.executable, "-m", "pip", *parts],
                            text=True, capture_output=True, check=False,
                            timeout=90)
    if result.returncode:
        raise RuntimeError(f"pip {' '.join(parts)} failed: {result.stderr[-1500:]}")
    return result.stdout


def installed_closure(plan: dict) -> dict:
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux":
        raise RuntimeError("V31d sentinel requires Linux Python 3.12")
    versions = {name: metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"V31d native package versions differ: {versions}")
    # pip check catches declared conflicts; it cannot catch MJLab's private Warp API.
    pip_check = run_command("check").strip()
    inspect = json.loads(run_command("inspect"))
    installed = []
    bam_direct_url = None
    for item in inspect["installed"]:
        meta = item["metadata"]
        name = meta["name"].lower().replace("_", "-")
        if name == "better-actuator-models":
            bam_direct_url = item.get("direct_url")
        installed.append({"name": name,
                          "version": meta["version"],
                          "location": item.get("metadata_location"),
                          "direct_url": item.get("direct_url")})
    installed.sort(key=lambda item: (item["name"], item["version"],
                                     item["location"] or ""))
    duplicates = sorted({item["name"] for item in installed
                         if sum(p["name"] == item["name"] for p in installed) > 1})
    if any(name in duplicates for name in EXPECTED):
        raise RuntimeError(f"Duplicate critical distributions: {duplicates}")
    if (not isinstance(bam_direct_url, dict)
            or bam_direct_url.get("vcs_info", {}).get("commit_id") != BAM_COMMIT
            or bam_direct_url.get("dir_info", {}).get("editable")):
        raise RuntimeError("BAM installed source is not the exact noneditable commit")
    modules = {}
    for name in MODULES:
        module = importlib.import_module(name)
        modules[name] = str(Path(module.__file__).resolve())
    # A source checkout is required: the installed wheel computes the project
    # root incorrectly for eager task registrations. Check both source origins
    # before opening the disjoint CUDA task.
    import ape_walk.task as ape_task
    from ape_walk.physics import frame50 as frame50_source
    for key, module in (("task", ape_task), ("frame50", frame50_source)):
        origin = Path(module.__file__).resolve()
        expected_origin = (ROOT / plan["source_paths"][key]).resolve()
        if origin != expected_origin:
            raise RuntimeError(f"V31d {key} module loaded from wrong source path: {origin}")
        if sha(origin) != plan["source_sha256"][key]:
            raise RuntimeError(f"V31d {key} imported source differs from frozen plan")
        modules["ape_walk.task" if key == "task" else "ape_walk.physics.frame50"] = str(origin)
    import bam.mjlab as bam_mjlab
    import mjlab_microduck.tasks.mdp as upstream_mdp
    upstream_sha = sha(Path(upstream_mdp.__file__))
    bam_sha = sha(Path(bam_mjlab.__file__))
    if upstream_sha != plan["source_sha256"]["upstream_mdp"]:
        raise RuntimeError("Installed Microduck MDP source differs from frozen reference")
    if bam_sha != plan["source_sha256"]["bam_mjlab_reference"]:
        raise RuntimeError("Installed BAM MJLab source differs from frozen reference")
    import torch
    import warp as wp
    if torch.__version__ != EXPECTED["torch"] or torch.version.cuda != "12.8":
        raise RuntimeError("Torch import differs from the CUDA image pin")
    for name in ("torch", "triton"):
        if Path(modules[name]).is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"{name} was overlaid inside the V31d virtual environment")
    if not hasattr(wp, "context") or not hasattr(wp.context, "runtime"):
        raise RuntimeError("MJLab 1.3.0 requires Warp 1.12's context.runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one working CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 4090":
        raise RuntimeError(f"Unexpected CUDA device: {gpu}")
    closure = {
        "schema": "robot-learning-lab.cove50-v31d-installed-runtime.v1",
        "python": sys.version.split()[0], "executable": sys.executable,
        "expected_versions": EXPECTED, "installed": installed,
        "import_origins": modules, "pip_check": pip_check,
        "torch_cuda": torch.version.cuda, "gpu": gpu,
        "warp_context_runtime_present": True,
        "pollen_lock_sha256": plan["source_sha256"]["pollen_lock"],
        "installed_upstream_mdp_sha256": upstream_sha,
        "installed_bam_mjlab_sha256": bam_sha,
        "bam_direct_url": bam_direct_url,
        "pip_freeze": run_command("freeze", "--all").splitlines(),
    }
    closure["content_sha256"] = hashlib.sha256(
        json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return closure


def verify_source(plan: dict) -> None:
    expected = plan["source_sha256"]
    for name, relative in plan["source_paths"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise RuntimeError(f"V31d source path invalid: {name}")
        if expected.get(name) != sha(path):
            raise RuntimeError(f"V31d source drift: {name}")
    if (expected.get("body") != sha(BODY)
            or expected.get("checkpoint") != sha(CHECKPOINT)
            or expected.get("preflight") != sha(Path(__file__).resolve())):
        raise RuntimeError("V31d body/checkpoint/preflight identity differs")
    if (plan["sentinel_seed"] != SENTINEL_SEED
            or plan["scored_original_visits_max"] != 0
            or plan["sentinel_env_steps_max"] != 1
            or plan["optimizer_updates_max"] != 0):
        raise RuntimeError("V31d sentinel/score contract changed")


def task_smoke(plan: dict, closure: dict, out: Path) -> dict:
    import torch
    import ape_walk.task
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
        build_seated_rise_env_cfg,
    )

    task_id = ape_walk.task.ELBOW_SQUAT_HEADING_OBS_TASK_ID
    # Match V30's construction, including its added self-contact sensor.
    cfg = load_env_cfg(task_id, play=False)
    cfg = build_seated_rise_env_cfg(BODY.resolve(), variance=True)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape_walk.task.ContactSensorCfg(
            name="self_body_contact",
            primary=ape_walk.task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ape_walk.task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
    )
    cfg.scene.num_envs = 1
    cfg.seed = SENTINEL_SEED
    cfg.episode_length_s = 12.0
    cfg.auto_reset = False
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    cfg.events["set_ground_state"].params["sitting_prob"] = 1.0
    cfg.events["set_ground_state"].params["standing_prob"] = 0.0
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(raw, clip_actions=load_rl_cfg(task_id).clip_actions)
    try:
        # V30 constructs and loads the runner before the evaluator's first
        # explicit observation. Runner construction itself reads observations
        # and may consume observation-noise RNG; preserve that exact order.
        runner = load_runner_cls(task_id)(wrapped, asdict(load_rl_cfg(task_id)), device="cuda:0")
        runner.load(str(CHECKPOINT), load_cfg={"actor": True, "critic": True},
                    strict=True, map_location="cuda:0")
        actor_policy = runner.get_inference_policy(device="cuda:0")
        critic = runner.alg.critic.eval()
        # RslRlVecEnvWrapper.__init__ already made V30's sole initial reset.
        command = raw.command_manager.get_term("twist")
        command.vel_command_b[:, 0] = 0.0
        observations = wrapped.get_observations()
        actor, critic_obs = observations["actor"], observations["critic"]
        if tuple(actor.shape) != (1, 73) or tuple(critic_obs.shape) != (1, 88):
            raise RuntimeError(f"Unexpected observation shapes: {actor.shape}, {critic_obs.shape}")
        if wrapped.num_actions != 21 or raw.action_manager.total_action_dim != 21:
            raise RuntimeError("Unexpected 21D action interface")
        if not torch.isfinite(actor).all() or not torch.isfinite(critic_obs).all():
            raise RuntimeError("Nonfinite first actor/critic observation")
        with torch.inference_mode():
            action = actor_policy(observations)
            value = critic(observations)
        if tuple(action.shape) != (1, 21) or not torch.isfinite(action).all():
            raise RuntimeError("Saved actor first action is wrong-shaped or nonfinite")
        if not torch.isfinite(value).all():
            raise RuntimeError("Saved critic first value is nonfinite")
        record = {
            "schema": "robot-learning-lab.cove50-v31d-cuda-sentinel.v1",
            "status": "constructor_reset_first_action_passed",
            "claim_limit": "One disjoint unscored CUDA sentinel; no skill score, demonstration, optimizer or collection original.",
            "seed": SENTINEL_SEED, "device": "cuda:0", "gpu": closure["gpu"],
            "body_sha256": sha(BODY), "checkpoint_sha256": sha(CHECKPOINT),
            "source_sha256": plan["source_sha256"],
            "runtime_closure_sha256": closure["content_sha256"],
            "actor_observation": actor[0].detach().cpu().tolist(),
            "critic_observation": critic_obs[0].detach().cpu().tolist(),
            "first_actor_action": action[0].detach().cpu().tolist(),
            "first_critic_value": value[0].detach().cpu().tolist(),
            "initial_qpos": raw.sim.data.qpos[0].detach().cpu().tolist(),
            "initial_qvel": raw.sim.data.qvel[0].detach().cpu().tolist(),
            "action_manager_dim": raw.action_manager.total_action_dim,
            "control_hz": 1.0 / raw.step_dt,
            "pre_action_command_x": float(command.command[0, 0]),
            "pre_action_rise_phase": float(command.alpha[0]),
            "native_steps": 0,
        }
        # Keep the action-free evidence even if the first CUDA step fails.
        write(out, record)
        with torch.inference_mode():
            next_obs, reward, done, _ = wrapped.step(action)
        if (not torch.isfinite(next_obs["actor"]).all()
                or not torch.isfinite(next_obs["critic"]).all()
                or not torch.isfinite(reward).all()):
            raise RuntimeError("Nonfinite one-step CUDA diagnostic")
        record["status"] = "constructor_reset_action_and_one_step_passed"
        record["native_steps"] = 1
        record["one_step_reward"] = float(reward[0])
        record["one_step_done"] = bool(done[0])
        record["one_step_contact_sensor_present"] = (
            "feet_ground_contact" in raw.scene.sensors
            and "hand_ground_contact" in raw.scene.sensors
            and "self_body_contact" in raw.scene.sensors
        )
        if not record["one_step_contact_sensor_present"]:
            raise RuntimeError("Expected seated contact sensors absent")
        return record
    finally:
        wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--closure-out", type=Path, required=True)
    parser.add_argument("--phase", choices=("attest", "smoke"), required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "robot-learning-lab.cove50-seated-runtime.v31d.plan":
        parser.error("Wrong V31d plan")
    verify_source(plan)
    closure = installed_closure(plan)
    if args.phase == "attest":
        write(args.closure_out, closure)
        write(args.out, {"schema": "robot-learning-lab.cove50-v31d-runtime-attestation.v1",
                         "status": "installed_closure_attested_no_cuda_task",
                         "created_at_utc": datetime.now(timezone.utc).isoformat(),
                         "closure_sha256": closure["content_sha256"],
                         "source_sha256": plan["source_sha256"]})
        return
    prior = json.loads(args.closure_out.read_text())
    if closure != prior:
        raise RuntimeError("Installed runtime closure changed after attestation")
    report = task_smoke(plan, closure, args.out)
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.out, report)


if __name__ == "__main__":
    main()
