"""Run only the frozen V3e CUDA locomotion sentinels on a guarded worker.

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
import os
from pathlib import Path
import runpy
import shutil
import sys
import threading


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3e"
EVIDENCE = Path("/workspace/robot-learning-lab/ape-cove50-locomotion-learning-v3-20261014/preflight-v3e")
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
    if (packet["schema"] != "robot-learning-lab.locomotion-v3e-cuda-preflight-packet.v1"
            or packet["sentinel_seed"] != SENTINEL
            or packet["max_worlds"] != 1
            or packet["max_optimizer_updates"] != 0
            or packet["max_native_steps"] != 204
            or packet["provider_guard_required"]["create_permitted_by_packet"] is not False
            or packet.get("persistent_evidence_root") != str(EVIDENCE)):
        raise RuntimeError("Frozen V3e packet contract changed")
    lineage = packet.get("evaluator_lineage", {})
    amendment = json.loads((ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3c/plan.json").read_text())
    original = json.loads((ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/method.json").read_text())
    if (lineage.get("original_v3_method_evaluator_sha256") != original["source"]["evaluator_sha256"]
            or lineage.get("v3c_amended_evaluator_sha256") != amendment["exact_sources"]["evaluator_sha256"]
            or lineage.get("active_unscored_sentinel_evaluator_sha256") != sha(EVALUATOR)
            or amendment["amends_without_rewriting"]["v3_method_sha256"]
            != sha(ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/method.json")):
        raise RuntimeError("V3e evaluator lineage differs from frozen method/V3c amendment")
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


def persistent_evidence_contract() -> None:
    if (not Path("/workspace").is_mount()
            or not EVIDENCE.is_dir()
            or not EVIDENCE.resolve().is_relative_to(Path("/workspace").resolve())
            or not (EVIDENCE / "bootstrap.log").is_file()):
        raise RuntimeError("V3e evidence is not on the mounted persistent /workspace volume")
    if any((EVIDENCE / name).exists() for name in
           ("cuda-preflight-result.json", "cuda-evaluate-sentinel.json",
            "cuda-evaluate-sentinel.npz", "cuda-guard-abort.json")):
        raise RuntimeError("V3e persistent evidence already contains an original result")


def durable_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".incoming")
    with temporary.open("x") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def guarded_host(path: Path, packet: dict, min_remaining_s: int) -> dict:
    """Reject a stale/unbound attestation; local watchdog remains mandatory."""
    binding = json.loads(path.read_text())
    now = datetime.now(timezone.utc)
    stamped = datetime.fromisoformat(binding["attested_at_utc"])
    deadline = datetime.fromisoformat(binding["hard_deadline_utc"])
    guard = packet["provider_guard_required"]
    manifest_path = path.with_name("creation-manifest.json")
    state_path = path.with_name("fresh-pod-create-guard.json")
    attempt_path = path.with_name("consumed-create-attempt.json")
    manifest = json.loads(manifest_path.read_text())
    state = json.loads(state_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    manifest_sha = sha(manifest_path)
    checked = datetime.fromisoformat(state["last_check_at_utc"])
    if (binding.get("schema") != "robot-learning-lab.locomotion-v3e-live-stop-guard.v1"
            or binding.get("packet_sha256") != sha(PACKET)
            or manifest.get("packet_sha256") != sha(PACKET)
            or binding.get("creation_manifest_sha256") != manifest_sha
            or binding.get("guard_source_sha256") != manifest.get("guard_source_sha256")
            or state.get("plan_sha256") != sha(PACKET)
            or state.get("creation_manifest_sha256") != manifest_sha
            or binding.get("consumed_attempt_sha256") != sha(attempt_path)
            or attempt.get("packet_sha256") != sha(PACKET)
            or attempt.get("creation_manifest_sha256") != manifest_sha
            or attempt.get("pod_name") != state.get("name")
            or attempt.get("state") != "consumed_before_provider_create"
            or state.get("status") != "running"
            or state.get("stop_requested") is not False
            or state.get("last_error") is not None
            or state.get("pod_ids") != [binding.get("pod_id")]
            or checked < stamped
            or not 0 <= (now - checked).total_seconds() <= 30
            or binding.get("provider_desired_status") != "RUNNING"
            or binding.get("provider_runtime_status") != "running"
            or binding.get("watchdog_status") != "healthy"
            or binding.get("gpu_id") != "NVIDIA GeForce RTX 4090"
            or binding.get("gpu_count") != 1
            or not isinstance(binding.get("pod_id"), str)
            or binding.get("listed_rate_usd_per_hour", math.inf) > 0.89
            or binding.get("cap_s", math.inf) > guard["max_paid_wall_s"]
            or not 0 <= (now - stamped).total_seconds() <= guard["latest_guard_attestation_age_s"]
            or (deadline - now).total_seconds() < min_remaining_s):
        raise RuntimeError("Missing, stale, unbound or unhealthy live provider stop guard")
    return binding


def monitor_guard(path: Path, packet: dict, done: threading.Event) -> None:
    while not done.wait(5):
        try:
            guarded_host(path, packet, 120)
        except Exception as exc:
            failure = {"schema": "robot-learning-lab.locomotion-v3e-guard-abort.v1",
                       "reason": f"{type(exc).__name__}: {exc}",
                       "at_utc": datetime.now(timezone.utc).isoformat()}
            target = EVIDENCE / "cuda-guard-abort.json"
            try:
                if not target.exists():
                    durable_json(target, failure)
            finally:
                os._exit(93)


def finite(value) -> bool:
    return all(math.isfinite(float(v)) for v in value)


def cuda_full_state(packet: dict, guard_path: Path, origin: dict) -> dict:
    from checkout_import_gate_v2 import attest_eager_modules

    attest_eager_modules(json.loads(V2_PLAN.read_text()))
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.utils.os import get_checkpoint_path
    from ape_walk import task as ape
    from preflight_locomotion_full_state_v3b import digest, equal_state, stable, trainer_cfg

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one CUDA GPU")
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if source.get("iter") != 844:
        raise RuntimeError("Source checkpoint is not iteration 844")
    results = {}
    normalized_envs = {}
    agent_semantics = {}
    log_root = EVIDENCE / "trainer-resume"
    for arm, std in (("control", 0.05), ("candidate", 0.15)):
        guarded_host(guard_path, packet, 720 if arm == "control" else 480)
        run_name = f"v3e-preflight-{arm}-{SENTINEL}"
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
        launch_worlds = cfg.env.scene.num_envs
        launch_updates = cfg.agent.max_iterations
        if (not cfg.agent.resume or cfg.agent.load_checkpoint != "model_844.pt"
                or cfg.agent.load_run != f"v3b-{arm}"
                or (launch_worlds, launch_updates) != (512, 48)):
            raise RuntimeError("Captured V3 resume configuration differs")
        captured_agent = asdict(cfg.agent)
        base_agent = asdict(load_rl_cfg(ape.ELBOW_TASK_ID))
        launch_only = {"max_iterations", "save_interval", "experiment_name", "run_name",
                       "logger", "resume", "load_run", "load_checkpoint"}
        differences = {key for key in captured_agent if captured_agent[key] != base_agent[key]}
        if differences != launch_only or captured_agent.keys() != base_agent.keys():
            raise RuntimeError(f"Trainer/base agent semantics drifted: {sorted(differences)}")
        agent_semantics[arm] = digest({key: value for key, value in captured_agent.items()
                                       if key not in launch_only})
        # Only the sentinel world count, seed and automatic reset change. All
        # training noise, command, motor-lag, randomization and rewards remain.
        cfg.env.scene.num_envs = 1
        cfg.env.seed = SENTINEL
        cfg.env.auto_reset = False
        cfg.agent.seed = SENTINEL
        normalized_env = stable(asdict(cfg.env))
        for phase in ("std_walking", "std_running"):
            normalized_env["rewards"]["pose"]["params"][phase][r".*hip_roll.*"] = 0.05
        normalized_envs[arm] = normalized_env
        torch.manual_seed(SENTINEL)
        env = RslRlVecEnvWrapper(
            ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0"), clip_actions=cfg.agent.clip_actions
        )
        try:
            runner = ape.VelocityOnPolicyRunner(
                env, asdict(cfg.agent), device="cuda:0"
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
                "arm": arm, "resolved_checkpoint": str(resolved.relative_to(EVIDENCE)),
                "checkpoint_sha256": sha(resolved), "iteration": 844,
                "launch_worlds_updates": [launch_worlds, launch_updates],
                "sentinel_worlds": 1, "sentinel_seed": SENTINEL,
                "resolved_env_cfg_sha256": digest(asdict(cfg.env)),
                "resolved_agent_cfg_sha256": digest(asdict(cfg.agent)),
                "agent_inference_optimizer_semantics_sha256": agent_semantics[arm],
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
    if normalized_envs["control"] != normalized_envs["candidate"]:
        raise RuntimeError("Trainer arms differ beyond the two hip-roll pose std entries")
    if agent_semantics["control"] != agent_semantics["candidate"]:
        raise RuntimeError("Trainer agent actor/critic/optimizer semantics differ")
    for key in ("qpos", "qvel", "command", "actor_observation", "critic_observation",
                "normalized_actor_latent", "normalized_critic_latent", "action", "value"):
        comparison[key] = max(abs(a - b) for a, b in zip(results["control"][key], results["candidate"][key]))
    if any(gap > (1e-6 if key in ("qpos", "qvel", "command") else 1e-5)
           for key, gap in comparison.items()):
        raise RuntimeError(f"Two-arm CUDA first decision differs: {comparison}")
    return {"checkout_origin": origin, "arms": results, "paired_max_gaps": comparison,
            "paired_normalized_env_cfg_sha256": digest(normalized_envs["control"]),
            "agent_inference_optimizer_semantics_sha256": agent_semantics["control"],
            "native_steps": 2, "optimizer_updates": 0}


def actual_evaluator_cli(packet: dict, guard_path: Path) -> dict:
    """Run actual evaluate.py main, instrumented externally without editing it."""
    import torch
    from mjlab.rl import RslRlVecEnvWrapper

    guarded_host(guard_path, packet, 420)

    report = EVIDENCE / "cuda-evaluate-sentinel.json"
    trace = EVIDENCE / "cuda-evaluate-sentinel.npz"
    if report.exists() or trace.exists():
        raise RuntimeError("No replacement of original CUDA sentinel report/trace")
    evaluator_args = packet["evaluator_args"]
    if evaluator_args != ["--task", "Mjlab-Velocity-Flat-Ape-Elbows", "--physics-model", "frame50", "--initial-root-z", "0.231", "--yaw-only-head", "--arms-down", "--worlds", "1", "--seed", "47997", "--velocity-x", "0.35", "--velocity-y", "0", "--angular-z", "0", "--motor-rail-v", "12", "--command-delay-ms", "0", "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002", "--stop-at-s", "4.0", "--duration-s", "4.04"]:
        raise RuntimeError("Sentinel CLI arguments differ from frozen packet")
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    observed = {"first_observation": None, "latest_observation_tensors": None,
                "command_before_action": [], "first_qpos": None, "first_qvel": None,
                "first_action": None, "first_value": None,
                "equal_input_actor_action_gap": None, "post_step_contact": []}
    original_get = RslRlVecEnvWrapper.get_observations

    def capture_get(self, *args, **kwargs):
        value = original_get(self, *args, **kwargs)
        observed["latest_observation_tensors"] = value
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
                first_obs = observed["latest_observation_tensors"]
                if first_obs is None:
                    raise RuntimeError("Actual evaluator first observation was not captured")
                observed["first_observation"] = {key: first_obs[key][0].detach().cpu().tolist()
                                                 for key in ("actor", "critic")}
                runner = local["runner"]
                # evaluate.py loads the actor only. Load the saved critic solely
                # for a first-value diagnostic; it cannot affect the policy.
                runner.alg.critic.load_state_dict(source["critic_state_dict"], strict=True)
                with torch.inference_mode():
                    replay_action = runner.alg.actor(first_obs)
                    value = runner.alg.critic(first_obs)
                observed["first_value"] = value[0].detach().cpu().tolist()
                observed["equal_input_actor_action_gap"] = float(
                    (replay_action[0] - local["actions"][0]).abs().max())
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
    for path in (report, trace):
        with path.open("rb") as persisted:
            os.fsync(persisted.fileno())
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
            or not finite(observed["first_value"])
            or observed["equal_input_actor_action_gap"] > 1e-5
            or not all(finite(__import__("numpy").asarray(row["force"]).flatten())
                       for row in observed["post_step_contact"])):
        raise RuntimeError("Actual evaluator report/first decision/contact parity differs")
    return {"report_sha256": sha(report), "trace_sha256": sha(trace),
            "first_observation": observed["first_observation"],
            "first_qpos": observed["first_qpos"], "first_qvel": observed["first_qvel"],
            "first_action": observed["first_action"],
            "first_value_from_saved_critic": observed["first_value"],
            "equal_input_actor_action_gap": observed["equal_input_actor_action_gap"],
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
                          "max_optimizer_updates": 0, "max_native_steps": 204}))
        return
    if ROOT != Path(packet["worker_root"]):
        raise RuntimeError("CUDA sentinel must run from exact pinned worker root")
    if args.guard_attestation is None:
        raise RuntimeError("Reviewed live provider stop guard attestation required")
    guard = guarded_host(args.guard_attestation, packet, 900)
    persistent_evidence_contract()
    done = threading.Event()
    monitor = threading.Thread(target=monitor_guard,
                               args=(args.guard_attestation, packet, done), daemon=True)
    monitor.start()
    try:
        from checkout_import_gate_v2 import require_checkout_origin
        origin = require_checkout_origin()
        from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v31d as runtime
        closure = runtime.installed_closure(json.loads(V2_PLAN.read_text()))
        if closure["content_sha256"] != packet["runtime_closure_sha256"]:
            raise RuntimeError("Installed Linux/CUDA package closure differs")
        guarded_host(args.guard_attestation, packet, 720)
        first = cuda_full_state(packet, args.guard_attestation, origin)
        actual = actual_evaluator_cli(packet, args.guard_attestation)
        targets = (("qpos", "first_qpos"), ("qvel", "first_qvel"),
                   ("actor_observation", "first_observation.actor"),
                   ("critic_observation", "first_observation.critic"),
                   ("action", "first_action"),
                   ("value", "first_value_from_saved_critic"))
        gaps = {}
        for source_key, target in targets:
            observed = actual
            for part in target.split("."):
                observed = observed[part]
            gaps[source_key] = max(abs(a - b) for a, b in zip(
                first["arms"]["control"][source_key], observed))
        if any(not math.isfinite(gap) for gap in gaps.values()):
            raise RuntimeError("Nonfinite trainer/evaluator first-decision difference")
        if first["native_steps"] + actual["native_steps"] != 204:
            raise RuntimeError("Explicit native transitions exceed packet limit")
        guarded_host(args.guard_attestation, packet, 120)
        report = {"schema": "robot-learning-lab.locomotion-v3e-cuda-preflight-result.v1",
                  "status": "unscored_cuda_sentinel_complete_pending_independent_review",
                  "claim_limit": "Exact stochastic trainer-arm pair and actual play-mode CLI were observed separately. Their differing first states/actions are measured distribution shift, not raw parity or skill qualification.",
                  "packet_sha256": sha(PACKET), "guard_pod_id": guard["pod_id"],
                  "persistent_evidence_root": str(EVIDENCE),
                  "evaluator_lineage": packet["evaluator_lineage"],
                  "runtime_closure_sha256": closure["content_sha256"],
                  "full_state": first, "actual_evaluator_cli": actual,
                  "trainer_vs_actual_evaluator_first_differences": gaps,
                  "native_steps": 204, "optimizer_updates": 0,
                  "requires_review": "Inspect trainer/play distribution shift and bounded first action/value. Full reward/physical ranking, throughput and V3 development remain separate gates."}
        output = EVIDENCE / "cuda-preflight-result.json"
        if output.exists():
            raise RuntimeError("No replacement of original CUDA preflight result")
        durable_json(output, report)
        print(json.dumps({"result": str(output), "first_decision_differences": gaps,
                          "native_steps": 204, "optimizer_updates": 0}))
    finally:
        done.set()
        monitor.join(timeout=2)


if __name__ == "__main__":
    main()
