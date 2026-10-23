"""Run only the frozen V3d CUDA locomotion sentinels on a guarded worker.

No development/score seed or optimizer is accepted. The `contract` phase is
local and read-only; `cuda` requires an independently reviewed live provider
stop guard and its fresh attestation. This file never creates a pod.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3d"
PACKET = ART / "packet.json"
CHECKPOINT = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
V2_PLAN = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/cuda-launch-v2/plan.json"
SENTINEL = 47997


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def packet_contract() -> dict:
    packet = json.loads(PACKET.read_text())
    if (packet["schema"] != "robot-learning-lab.locomotion-v3d-cuda-preflight-packet.v1"
            or packet["sentinel_seed"] != SENTINEL
            or packet["max_worlds"] != 1
            or packet["max_optimizer_updates"] != 0
            or packet["max_native_steps"] != 202
            or packet["provider_guard_required"]["create_permitted_by_packet"] is not False):
        raise RuntimeError("Frozen V3d packet contract changed")
    if SENTINEL in packet["development_seeds_excluded"] + packet["sealed_score_seeds_excluded"]:
        raise RuntimeError("Sentinel overlaps a development or score seed")
    for rel, expected in packet["source_sha256"].items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"Worker source or eager asset differs: {rel}")
    if (sha(BODY) != "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
            or sha(CHECKPOINT) != "1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2"):
        raise RuntimeError("Direct Frame or baseline checkpoint changed")
    return packet


def guarded_host(path: Path, packet: dict) -> dict:
    """Reject a stale/unbound attestation; local watchdog remains mandatory."""
    binding = json.loads(path.read_text())
    now = datetime.now(timezone.utc)
    stamped = datetime.fromisoformat(binding["attested_at_utc"])
    deadline = datetime.fromisoformat(binding["hard_deadline_utc"])
    guard = packet["provider_guard_required"]
    if (binding.get("schema") != "robot-learning-lab.locomotion-v3d-live-stop-guard.v1"
            or binding.get("packet_sha256") != sha(PACKET)
            or binding.get("provider_desired_status") != "RUNNING"
            or binding.get("provider_runtime_status") != "running"
            or binding.get("watchdog_status") != "healthy"
            or binding.get("gpu_id") != "NVIDIA GeForce RTX 4090"
            or binding.get("gpu_count") != 1
            or not isinstance(binding.get("pod_id"), str)
            or binding.get("listed_rate_usd_per_hour", math.inf) > 0.89
            or binding.get("cap_s", math.inf) > guard["max_paid_wall_s"]
            or not 0 <= (now - stamped).total_seconds() <= guard["latest_guard_attestation_age_s"]
            or (deadline - now).total_seconds() < 180):
        raise RuntimeError("Missing, stale, unbound or unhealthy live provider stop guard")
    return binding


def finite(value) -> bool:
    return all(math.isfinite(float(v)) for v in value)


def cuda_full_state(packet: dict) -> dict:
    from checkout_import_gate_v2 import attest_eager_modules, require_checkout_origin

    origin = require_checkout_origin()
    attest_eager_modules(json.loads(V2_PLAN.read_text()))
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.utils.os import get_checkpoint_path
    from ape_walk import task as ape
    from preflight_locomotion_full_state_v3b import equal_state, trainer_cfg
    from preflight_locomotion_entry_v3c import nominal_training_cfg

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one CUDA GPU")
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if source.get("iter") != 844:
        raise RuntimeError("Source checkpoint is not iteration 844")
    results = {}
    log_root = ROOT / "logs/rsl_rl/ape_cove50_velocity"
    for arm, std in (("control", 0.05), ("candidate", 0.15)):
        run_name = f"v3d-preflight-{arm}-{SENTINEL}"
        run_dir = log_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        copy = run_dir / "model_844.pt"
        if copy.exists():
            raise RuntimeError(f"No replacement of isolated resume source: {copy}")
        shutil.copyfile(CHECKPOINT, copy)
        resolved = get_checkpoint_path(log_root, run_name, "model_844.pt")
        if resolved != copy or sha(resolved) != sha(CHECKPOINT):
            raise RuntimeError("Installed trainer checkpoint resolver selected wrong bytes")
        cfg = trainer_cfg(arm, std)
        if (not cfg.agent.resume or cfg.agent.load_checkpoint != "model_844.pt"
                or cfg.env.scene.num_envs not in (512, 1024, 2048)):
            raise RuntimeError("Captured V3 resume configuration differs")
        env_cfg = nominal_training_cfg(SENTINEL)
        for phase in ("std_walking", "std_running"):
            env_cfg.rewards["pose"].params[phase][r".*hip_roll.*"] = std
        torch.manual_seed(SENTINEL)
        env = RslRlVecEnvWrapper(
            ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0"), clip_actions=cfg.agent.clip_actions
        )
        try:
            runner = ape.VelocityOnPolicyRunner(
                env, asdict(load_rl_cfg(ape.ELBOW_TASK_ID)), device="cuda:0"
            )
            runner.load(str(resolved), strict=True, map_location="cuda:0")
            checks = {
                "actor": equal_state(runner.alg.actor.state_dict(), source["actor_state_dict"]),
                "critic": equal_state(runner.alg.critic.state_dict(), source["critic_state_dict"]),
                "optimizer": equal_state(runner.alg.optimizer.state_dict(), source["optimizer_state_dict"]),
            }
            if runner.current_learning_iteration != 844 or any(checks.values()):
                raise RuntimeError(f"Full RSL-RL resume differs in {arm}: {checks}")
            obs = env.get_observations()
            with torch.inference_mode():
                action = runner.alg.actor(obs)
                value = runner.alg.critic(obs)
                actor_latent = runner.alg.actor.get_latent(obs)
                critic_latent = runner.alg.critic.get_latent(obs)
            command = env.unwrapped.command_manager.get_command("twist")[0, :3]
            sim = env.unwrapped.sim.data
            row = {
                "arm": arm, "resolved_checkpoint": str(resolved.relative_to(ROOT)),
                "checkpoint_sha256": sha(resolved), "iteration": 844,
                "qpos": sim.qpos[0].cpu().tolist(), "qvel": sim.qvel[0].cpu().tolist(),
                "command": command.cpu().tolist(),
                "actor_observation": obs["actor"][0].cpu().tolist(),
                "critic_observation": obs["critic"][0].cpu().tolist(),
                "normalized_actor_latent": actor_latent[0].cpu().tolist(),
                "normalized_critic_latent": critic_latent[0].cpu().tolist(),
                "action": action[0].cpu().tolist(), "value": value[0].cpu().tolist(),
                "physical_root_z_m": float(env.unwrapped.scene["robot"].data.root_link_pose_w[0, 2]),
                "strict_state_mismatches": checks,
            }
            if (len(row["actor_observation"]) != 72 or len(row["critic_observation"]) != 87
                    or len(row["action"]) != 21 or not all(finite(row[key]) for key in
                    ("qpos", "qvel", "command", "actor_observation", "critic_observation",
                     "normalized_actor_latent", "normalized_critic_latent", "action", "value"))):
                raise RuntimeError("Nonfinite or wrong-width first CUDA decision")
            term = env.unwrapped.command_manager.get_term("twist")
            ids = torch.arange(1, device="cuda:0")
            term._resample_command(ids)
            row["first_resample_command"] = term.command[0, :3].cpu().tolist()
            row["first_resample_just_stopped"] = bool(term.just_stopped[0])
            term._resample_command(ids)
            row["second_resample_command"] = term.command[0, :3].cpu().tolist()
            if not (all(abs(v) < 1e-9 for v in row["first_resample_command"] + row["second_resample_command"])
                    and row["first_resample_just_stopped"]):
                raise RuntimeError("Trainer stop command resampling failed")
            # A single contact/finite step, after recording the actual first
            # state/action. This is not a skill episode or PPO update.
            with torch.no_grad():
                env.step(action)
            sensors = env.unwrapped.scene.sensors
            row["first_step_contact_found"] = sensors["feet_ground_contact"].data.found[0].cpu().tolist()
            row["first_step_contact_force"] = sensors["feet_ground_contact"].data.force[0].cpu().tolist()
            if not finite(torch.as_tensor(row["first_step_contact_force"]).flatten().tolist()):
                raise RuntimeError("Nonfinite first-step contact force")
            results[arm] = row
        finally:
            env.close()
    comparison = {}
    for key in ("qpos", "qvel", "command", "actor_observation", "critic_observation",
                "normalized_actor_latent", "normalized_critic_latent", "action", "value"):
        comparison[key] = max(abs(a - b) for a, b in zip(results["control"][key], results["candidate"][key]))
    if any(gap > (1e-6 if key in ("qpos", "qvel", "command") else 1e-5)
           for key, gap in comparison.items()):
        raise RuntimeError(f"Two-arm CUDA first decision differs: {comparison}")
    return {"checkout_origin": origin, "arms": results, "paired_max_gaps": comparison,
            "optimizer_updates": 0}


def actual_evaluator_cli(packet: dict) -> dict:
    """Run actual evaluate.py main, instrumented externally without editing it."""
    import torch
    from mjlab.rl import RslRlVecEnvWrapper

    report = ART / "cuda-evaluate-sentinel.json"
    trace = ART / "cuda-evaluate-sentinel.npz"
    if report.exists() or trace.exists():
        raise RuntimeError("No replacement of original CUDA sentinel report/trace")
    evaluator_args = packet["evaluator_args"]
    if evaluator_args != ["--task", "Mjlab-Velocity-Flat-Ape-Elbows", "--physics-model", "frame50", "--initial-root-z", "0.231", "--yaw-only-head", "--arms-down", "--worlds", "1", "--seed", "47997", "--velocity-x", "0.35", "--velocity-y", "0", "--angular-z", "0", "--motor-rail-v", "12", "--command-delay-ms", "0", "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002", "--stop-at-s", "4.0", "--duration-s", "4.04"]:
        raise RuntimeError("Sentinel CLI arguments differ from frozen packet")
    observed = {"first_observation": None, "command_before_action": [], "first_qpos": None,
                "first_qvel": None, "first_action": None, "post_step_contact": []}
    original_get = RslRlVecEnvWrapper.get_observations

    def capture_get(self, *args, **kwargs):
        value = original_get(self, *args, **kwargs)
        if observed["first_observation"] is None:
            observed["first_observation"] = {key: value[key][0].detach().cpu().tolist()
                                             for key in ("actor", "critic")}
        return value

    def watch(frame, event, arg):
        if frame.f_code.co_filename != str(EVALUATOR) or frame.f_code.co_name != "main":
            return None
        if event != "line":
            return watch
        line, local = frame.f_lineno, frame.f_locals
        if line == 418 and local.get("step") in (0, 199, 200, 201):
            step = local["step"]
            env = local["env"]
            command = env.unwrapped.command_manager.get_command("twist")[0, :3]
            observed["command_before_action"].append({"step": step,
                "time_s": round(step * float(local["dt"]), 6),
                "command": command.detach().cpu().tolist()})
            if step == 0:
                observed["first_qpos"] = env.unwrapped.sim.data.qpos[0].detach().cpu().tolist()
                observed["first_qvel"] = env.unwrapped.sim.data.qvel[0].detach().cpu().tolist()
                observed["first_action"] = local["actions"][0].detach().cpu().tolist()
        if line == 419 and local.get("step") in (0, 199, 200, 201):
            sensor = local["sensors"]["feet_ground_contact"]
            observed["post_step_contact"].append({"step": local["step"],
                "found": sensor.data.found[0].detach().cpu().tolist(),
                "force": sensor.data.force[0].detach().cpu().tolist()})
        return watch

    original_argv = sys.argv
    RslRlVecEnvWrapper.get_observations = capture_get
    sys.settrace(watch)
    try:
        sys.argv = [str(EVALUATOR), str(CHECKPOINT), "--body-xml", str(BODY),
                    "--out", str(report), "--diagnostic-trace", str(trace), *evaluator_args]
        runpy.run_path(str(EVALUATOR), run_name="__main__")
    finally:
        sys.settrace(None)
        RslRlVecEnvWrapper.get_observations = original_get
        sys.argv = original_argv
    native = json.loads(report.read_text())
    import numpy as np
    with np.load(trace, allow_pickle=False) as saved:
        if saved["time_s"].shape[0] != 202:
            raise RuntimeError("Expected exactly 202 unscored evaluator steps")
        for key in ("actuator_force", "sole_contact_force", "raw_action", "root_pose"):
            if not np.isfinite(saved[key]).all():
                raise RuntimeError(f"Nonfinite native evaluator trace: {key}")
    switches = observed["command_before_action"]
    if ([row["step"] for row in switches] != [0, 199, 200, 201]
            or not all(abs(v) < 1e-9 for row in switches[:2] for v in row["command"][1:])
            or any(abs(row["command"][0] - 0.35) > 1e-6 for row in switches[:2])
            or any(abs(v) > 1e-9 for row in switches[2:] for v in row["command"])):
        raise RuntimeError(f"Actual evaluator move-to-zero command switch differs: {switches}")
    if (native["seed"] != SENTINEL or native["stop_at_s"] != 4.0
            or native["body_sha256"] != sha(BODY)
            or native["first_action_world0"] != observed["first_action"]
            or len(observed["first_observation"]["actor"]) != 72
            or len(observed["first_observation"]["critic"]) != 87
            or not finite(observed["first_action"])
            or not all(finite(__import__("numpy").asarray(row["force"]).flatten())
                       for row in observed["post_step_contact"])):
        raise RuntimeError("Actual evaluator report/first decision/contact parity differs")
    return {"report_sha256": sha(report), "trace_sha256": sha(trace),
            "first_observation": observed["first_observation"],
            "first_qpos": observed["first_qpos"], "first_qvel": observed["first_qvel"],
            "first_action": observed["first_action"],
            "command_before_action": switches,
            "post_step_contact": observed["post_step_contact"],
            "native_report_first_state_sha256": native["initial_state_sha256"],
            "native_report_first_action_sha256": native["first_action_sha256"],
            "native_steps": 202, "optimizer_updates": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "cuda"))
    parser.add_argument("--guard-attestation", type=Path)
    args = parser.parse_args()
    packet = packet_contract()
    if args.phase == "contract":
        print(json.dumps({"packet_sha256": sha(PACKET),
                          "verified_source_files": len(packet["source_sha256"]),
                          "max_optimizer_updates": 0, "max_native_steps": 202}))
        return
    if ROOT != Path(packet["worker_root"]):
        raise RuntimeError("CUDA sentinel must run from exact pinned worker root")
    if args.guard_attestation is None:
        raise RuntimeError("Reviewed live provider stop guard attestation required")
    guard = guarded_host(args.guard_attestation, packet)
    from checkout_import_gate_v2 import require_checkout_origin

    require_checkout_origin()
    from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v31d as runtime
    closure = runtime.installed_closure(json.loads(V2_PLAN.read_text()))
    if closure["content_sha256"] != packet["runtime_closure_sha256"]:
        raise RuntimeError("Installed Linux/CUDA package closure differs")
    first = cuda_full_state(packet)
    actual = actual_evaluator_cli(packet)
    gaps = {key: max(abs(a - b) for a, b in zip(first["arms"]["control"][key], actual[target]))
            for key, target in (("qpos", "first_qpos"), ("qvel", "first_qvel"),
                                ("action", "first_action"))}
    if any(not math.isfinite(gap) for gap in gaps.values()):
        raise RuntimeError("Nonfinite trainer/evaluator CUDA parity gap")
    report = {"schema": "robot-learning-lab.locomotion-v3d-cuda-preflight-result.v1",
              "status": "unscored_cuda_sentinel_complete_pending_independent_review",
              "claim_limit": "One source-checked CUDA preflight only. No V3 development/score, optimizer or skill qualification.",
              "packet_sha256": sha(PACKET), "guard_pod_id": guard["pod_id"],
              "runtime_closure_sha256": closure["content_sha256"],
              "full_state": first, "actual_evaluator_cli": actual,
              "trainer_vs_actual_evaluator_first_gap": gaps,
              "requires_review": "Trainer first state is zero-jitter nominal; evaluator has play reset variation. Inspect differences before treating action parity as passed. Reward ranking, throughput and full development remain separate gates."}
    output = ART / "cuda-preflight-result.json"
    if output.exists():
        raise RuntimeError("No replacement of original CUDA preflight result")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(output), "trainer_evaluator_gap": gaps,
                      "native_steps": 202, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
