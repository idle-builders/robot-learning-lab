"""Run only the frozen V30 first-24 paired diagnostic on an existing CUDA worker.

No pod provision, optimizer, replacement seed, or optional held/tail visit exists
in this runner. A plan must be independently frozen before collection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-v30-first24"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SEEDS = tuple(range(33001, 33025))
ROUTES = ("scripted", "model25")
SCORER = Path(__file__).with_name("score_seated_collection_v30.py")
WATCHDOG = Path(__file__).with_name("create_seated_collection_v30.py")
V26 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
V29 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-intervention-v29"
EVALUATOR = Path(__file__).with_name("evaluate_seated_collection_v30.py")
CHECKPOINT = V25 / "checkpoints/model_25.pt"
SOURCE_PATHS = {
    "body": BODY,
    "checkpoint": CHECKPOINT,
    "design_pointer": ROOT / "robot/ape-current-design.json",
    "v25_plan": V25 / "plan.json",
    "v26_plan": V26 / "plan.json",
    "v26_report": V26 / "report.json",
    "v29_plan": V29 / "plan.json",
    "v29_report": V29 / "report.json",
    "v26_evaluator": ROOT / "experiments/compact_morphology/mjlab_ape/evaluate_posture.py",
    "v26_runner": ROOT / "experiments/compact_morphology/cove50_stand_recovery/run_seated_reference_cuda_v26.py",
    "v26_scorer": Path(__file__).with_name("score_seated_reference_cuda_v26.py"),
    "v29_evaluator": ROOT / "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_intervention_v29.py",
    "v29_runner": ROOT / "experiments/compact_morphology/cove50_stand_recovery/run_seated_intervention_v29.py",
    "v29_scorer": ROOT / "experiments/compact_morphology/cove50_stand_recovery/score_seated_intervention_v29.py",
    "trainer": ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py",
    "task": ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    "frame50": ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    "evaluator": EVALUATOR,
    "runner": Path(__file__),
    "scorer": SCORER,
    "watchdog": WATCHDOG,
}
RUNTIME_PACKAGES = ("torch", "mjlab", "rsl-rl-lib", "mujoco", "mujoco-warp",
                    "mjlab-microduck")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_manifest() -> dict:
    paths = dict(SOURCE_PATHS)
    try:
        import mjlab_microduck.tasks.mdp as upstream_mdp
        paths["upstream_mdp"] = Path(upstream_mdp.__file__)
    except ImportError:
        pass
    runtime = {"python": ".".join(map(str, sys.version_info[:3]))}
    for package in RUNTIME_PACKAGES:
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = None
    source_sha = {name: sha(path) if path.exists() else None for name, path in paths.items()}
    source_sha.setdefault("upstream_mdp", None)
    source_paths = {name: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
                    else str(path) for name, path in paths.items()}
    source_paths.setdefault("upstream_mdp", "installed mjlab_microduck.tasks.mdp (unavailable locally)")
    return {"source_sha256": source_sha,
            "source_paths": source_paths,
            "runtime_versions": runtime,
            "seed_families": list(SEEDS), "routes": list(ROUTES)}


def verify_sources(plan: dict) -> None:
    actual = source_manifest()
    expected = plan["source_sha256"]
    required = set(SOURCE_PATHS) | {"upstream_mdp"}
    if set(expected) != required:
        raise ValueError(f"V30 source manifest keys differ: {sorted(required ^ set(expected))}")
    for name in required:
        if actual["source_sha256"].get(name) != expected[name] or expected[name] is None:
            raise ValueError(f"V30 source drift/missing file: {name}")
    expected_runtime = plan["runtime_versions"]
    actual_runtime = actual["runtime_versions"]
    if (not actual_runtime["python"].startswith(expected_runtime["python"])
            or any(actual_runtime[name] != expected_runtime[name]
                   for name in RUNTIME_PACKAGES)):
        raise ValueError(f"V30 runtime drift: {actual_runtime}")
    if plan["seed_families"] != list(SEEDS) or plan["routes"] != list(ROUTES):
        raise ValueError("V30 first-24 seed/route partition changed")
    if plan["device"] != "cuda:0":
        raise ValueError("V30 collection requires native CUDA")
    if (plan["max_route_wall_s"] != 120 or plan["max_worker_wall_s"] != 4800
            or plan["max_pod_wall_s"] != 6000
            or plan["max_start_readiness_s"] != 180
            or plan["gpu_listed_rate_usd_per_hour"] != 0.89
            or plan["gpu_listed_spend_cap_usd"] != 1.50
            or plan["original_visits_max"] != 48):
        raise ValueError("V30 reviewed route/worker/pod/listed-rate/visit caps changed")
    if plan["diagnostic_fit_seeds"] != list(range(33001, 33017)) or (
            plan["diagnostic_assessment_seeds"] != list(range(33017, 33025))):
        raise ValueError("V30 whole-family 16/8 diagnostic partition changed")
    if plan["v25_held_seeds_sealed"] != list(range(2240, 2252)):
        raise ValueError("V25 held seed seal changed")
    if plan["action_free_tail_sealed"] != [2260, 3279]:
        raise ValueError("V25 action-free tail seal changed")


def verify_fresh_pod_binding(art: Path, plan_path: Path,
                             local_start_request_at_utc: str) -> dict:
    """Consume the locally provider-checked, short-lived host attestation."""
    plan = json.loads(plan_path.read_text())
    manifest_path = art / "fresh-pod-creation-manifest.json"
    attestation_path = art / "fresh-pod-attestation.json"
    manifest = json.loads(manifest_path.read_text())
    attestation = json.loads(attestation_path.read_text())
    plan_sha = sha(plan_path)
    guard_sha = sha(WATCHDOG)
    if (manifest.get("schema") != "robot-learning-lab.v30-fresh-pod-creation.v1"
            or manifest.get("plan_sha256") != plan_sha
            or manifest.get("guard_source_sha256") != guard_sha
            or manifest.get("guard_source_path") != str(WATCHDOG.relative_to(ROOT))
            or attestation.get("schema") != "robot-learning-lab.v30-fresh-pod-attestation.v1"
            or attestation.get("creation_manifest_sha256") != sha(manifest_path)
            or attestation.get("plan_sha256") != plan_sha
            or attestation.get("guard_source_sha256") != guard_sha
            or attestation.get("guard_service") != plan.get("watchdog_service")
            or attestation.get("guard_state_path") != plan.get("watchdog_state_path")
            or attestation.get("image") != plan.get("watchdog_pod_image")
            or attestation.get("gpu_id") != "NVIDIA GeForce RTX 4090"
            or attestation.get("gpu_count") != 1
            or attestation.get("cloud_type") != "SECURE"
            or attestation.get("listed_rate_usd_per_hour") != plan["gpu_listed_rate_usd_per_hour"]
            or attestation.get("provider_desired_status") != "RUNNING"
            or attestation.get("provider_runtime_status") != "running"
            or attestation.get("local_start_request_at_utc") != local_start_request_at_utc):
        raise ValueError("V30 fresh-pod plan, source, cost, or provider binding mismatch")
    pod_id = attestation.get("pod_id")
    pod_name = attestation.get("pod_name")
    if (not isinstance(pod_id, str) or not 10 <= len(pod_id) <= 32
            or not pod_id.isalnum() or not isinstance(pod_name, str)
            or not pod_name.startswith("cove50-v30-")):
        raise ValueError("V30 fresh-pod identity is invalid")
    started = datetime.fromisoformat(local_start_request_at_utc)
    deadline = datetime.fromisoformat(attestation["hard_deadline_utc"])
    attested = datetime.fromisoformat(attestation["attested_at_utc"])
    current = datetime.now(timezone.utc)
    if (started.tzinfo is None or deadline.tzinfo is None or attested.tzinfo is None
            or not started < attested <= current < deadline
            or (current - attested).total_seconds() > 180
            or (deadline - started).total_seconds() != plan["max_pod_wall_s"]):
        raise ValueError("V30 fresh-pod attestation is stale or exceeds the local cap")
    return attestation


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def new_ledger(plan_path: Path, local_start_request_at_utc: str) -> dict:
    return {
        "schema": "robot-learning-lab.cove50-seated-first24-attempts.v30",
        "plan_sha256": sha(plan_path), "seed_families": list(SEEDS),
        "routes": list(ROUTES), "created_at_utc": now(),
        "local_start_request_at_utc": local_start_request_at_utc,
        "billing_bound_basis": "local request before paid pod start; conservative uptime estimate",
        "worker_started_at_utc": now(),
        "posted_pod_charge_usd": None, "serverless_charge_usd": None,
        "families": [{"seed": seed, "routes": {
            route: {"status": "not_attempted"} for route in ROUTES}}
            for seed in SEEDS],
        "no_posthoc_replacement": True,
    }


def validate_completed_report(seed: int, route: str, report_path: Path,
                              native_path: Path) -> None:
    report = json.loads(report_path.read_text())
    if (report["seed"] != seed or report["intervention_route"] != route
            or report["device"] != "cuda:0" or report["body_sha256"] != sha(BODY)
            or report["checkpoint"] != str(CHECKPOINT)
            or not 0 < report["duration_s"] <= 10.0
            or (report["ended_early"] and not report["end_termination_terms"])):
        raise ValueError(f"{seed} {route}: incomplete or mismatched native visit")
    if not native_path.is_file():
        raise ValueError(f"{seed} {route}: native trace missing")


def visit(seed: int, route: str, art: Path, timeout_s: int) -> dict:
    stem = art / f"seed{seed}-{route}"
    report_path, native_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if report_path.exists() or native_path.exists():
        raise ValueError(f"{seed} {route}: original output path already exists; no replacement")
    cmd = [sys.executable, str(EVALUATOR), str(CHECKPOINT), "--out", str(report_path),
           "--native-trajectory-out", str(native_path), "--intervention-route", route,
           "--seed", str(seed), "--device", "cuda:0", "--cove50-body-xml",
           str(BODY), "--mode", "cycle", "--start-squat",
           "--recover-from-start", "--seated-rise-stage", "--randomized-physics"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    visit_started = time.monotonic()
    with stem.with_suffix(".log").open("x") as log:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    if completed.returncode:
        raise RuntimeError(f"{seed} {route}: evaluator exit {completed.returncode}; inspect {stem.with_suffix('.log')}")
    validate_completed_report(seed, route, report_path, native_path)
    return {"status": "completed", "finished_at_utc": now(),
            "report_sha256": sha(report_path), "native_sha256": sha(native_path),
            "duration_s": json.loads(report_path.read_text())["duration_s"],
            "native_visit_wall_s": time.monotonic() - visit_started}


def run(art: Path, local_start_request_at_utc: str) -> None:
    import mujoco
    from experiments.compact_morphology.cove50_stand_recovery import score_seated_collection_v30 as score
    plan_path = art / "plan.json"
    plan = json.loads(plan_path.read_text())
    verify_sources(plan)
    binding = verify_fresh_pod_binding(art, plan_path, local_start_request_at_utc)
    model = mujoco.MjModel.from_xml_path(str(BODY))
    gate = json.loads((V25 / "plan.json").read_text())["physical_gate"]
    ledger_path = art / "attempt-ledger.json"
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists()
              else new_ledger(plan_path, local_start_request_at_utc))
    if ledger["plan_sha256"] != sha(plan_path) or ledger["seed_families"] != list(SEEDS):
        raise ValueError("V30 attempt ledger/plan drift")
    if ledger["local_start_request_at_utc"] != local_start_request_at_utc:
        raise ValueError("V30 local start request differs from first attempt")
    if ledger.get("pod_id") not in (None, binding["pod_id"]):
        raise ValueError("V30 attempt ledger pod binding changed")
    ledger["pod_id"] = binding["pod_id"]
    ledger["pod_name"] = binding["pod_name"]
    ledger["creation_manifest_sha256"] = binding["creation_manifest_sha256"]
    ledger["attestation_sha256"] = sha(art / "fresh-pod-attestation.json")
    if [family["seed"] for family in ledger["families"]] != list(SEEDS):
        raise ValueError("V30 attempt ledger family order changed")
    for family in ledger["families"]:
        for route in ROUTES:
            if family["routes"][route]["status"] == "attempting":
                family["routes"][route] = {
                    "status": "failed_original", "finished_at_utc": now(),
                    "failure_reason": "Interrupted original visit; intentionally not retried",
                }
    write_json_atomic(ledger_path, ledger)
    billing_bound_started = datetime.fromisoformat(local_start_request_at_utc)
    worker_started = datetime.fromisoformat(ledger["worker_started_at_utc"])
    if billing_bound_started.tzinfo is None:
        raise ValueError("Local start request timestamp must include UTC timezone")
    if billing_bound_started > datetime.now(timezone.utc):
        raise ValueError("Local start request timestamp lies in the future")
    def remaining_seconds() -> float:
        worker_elapsed = (datetime.now(timezone.utc) - worker_started).total_seconds()
        worker_remaining = plan["max_worker_wall_s"] - worker_elapsed
        pod_elapsed = (datetime.now(timezone.utc) - billing_bound_started).total_seconds()
        pod_remaining = plan["max_pod_wall_s"] - pod_elapsed
        listed_rate_remaining = (plan["gpu_listed_spend_cap_usd"] * 3600
                                 / plan["gpu_listed_rate_usd_per_hour"] - pod_elapsed)
        return min(worker_remaining, pod_remaining, listed_rate_remaining)

    for family in ledger["families"]:
        seed = family["seed"]
        for route in ROUTES:
            entry = family["routes"][route]
            if entry["status"] == "completed":
                validate_completed_report(seed, route, art / f"seed{seed}-{route}.json",
                                          art / f"seed{seed}-{route}.npz")
                if entry["report_sha256"] != sha(art / f"seed{seed}-{route}.json") or (
                        entry["native_sha256"] != sha(art / f"seed{seed}-{route}.npz")):
                    raise ValueError(f"{seed} {route}: saved original changed")
                continue
            if entry["status"] != "not_attempted":
                continue  # A failed/interrupted original is never rerun or replaced.
            remaining = remaining_seconds()
            if remaining <= 1:
                raise TimeoutError("V30 worker, pod wall, or listed-rate cap reached")
            verify_sources(plan)  # before every paired native visit
            family["routes"][route] = {"status": "attempting", "started_at_utc": now()}
            write_json_atomic(ledger_path, ledger)
            visit_started = time.monotonic()
            try:
                family["routes"][route] = visit(
                    seed, route, art, min(plan["max_route_wall_s"], max(1, int(remaining))))
            except Exception as exc:
                family["routes"][route] = {"status": "failed_original",
                                             "finished_at_utc": now(),
                                             "failure_reason": f"{type(exc).__name__}: {exc}",
                                             "native_visit_wall_s": time.monotonic() - visit_started}
                write_json_atomic(ledger_path, ledger)
                raise  # Stop on a worker fault; never pay for 47 copies of it.
            write_json_atomic(ledger_path, ledger)
            try:
                scored, _ = score.score_visit(seed, route, art, model, gate, sha(BODY))
                family["routes"][route]["source_checked_full_gate"] = scored["full_gate"]
            except Exception as exc:
                family["routes"][route]["status"] = "invalid_original"
                family["routes"][route]["failure_reason"] = (
                    f"Per-route native source check: {type(exc).__name__}: {exc}")
                write_json_atomic(ledger_path, ledger)
                raise
            write_json_atomic(ledger_path, ledger)
        if all(family["routes"][route]["status"] == "completed" for route in ROUTES):
            result = score.score_collection(art, seed=seed)["families"][0]
            family["pair_source_status"] = result["status"]
            if result["status"] != "complete_source_checked_pair":
                family["pair_failure_reason"] = result["failure_reason"]
                write_json_atomic(ledger_path, ledger)
                raise ValueError(f"{seed}: pair source invalid: {result['failure_reason']}")
            family["scripted_full_gate"] = result["scripted"]["full_gate"]
            family["model25_full_gate"] = result["model25"]["full_gate"]
            write_json_atomic(ledger_path, ledger)
        print(json.dumps({"seed": seed, "routes": family["routes"],
                          "pair_source_status": family.get("pair_source_status")}), flush=True)
    result = score.score_collection(art)
    write_json_atomic(art / "report.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--art", type=Path, default=ART)
    parser.add_argument("--print-source-manifest", action="store_true",
                        help="Local read-only prerequisite list; does not start a visit")
    parser.add_argument("--local-start-request-at-utc",
                        help="Guarded local UTC start request before paid pod start; conservative billing bound")
    args = parser.parse_args()
    if args.print_source_manifest:
        print(json.dumps(source_manifest(), indent=2))
        return
    if not args.local_start_request_at_utc:
        parser.error("--local-start-request-at-utc is required for a paid CUDA collection")
    art = args.art.resolve()
    try:
        run(art, args.local_start_request_at_utc)
    finally:
        ledger_path = art / "attempt-ledger.json"
        if ledger_path.exists():
            from experiments.compact_morphology.cove50_stand_recovery import score_seated_collection_v30 as score
            try:
                result = score.score_collection(art)
            except Exception as exc:
                result = {
                    "schema": "robot-learning-lab.cove50-seated-first24.v30",
                    "status": "unscored_source_or_runtime_failure",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "attempt_ledger_sha256": sha(ledger_path),
                    "families": json.loads(ledger_path.read_text())["families"],
                }
            write_json_atomic(art / "report.json", result)


if __name__ == "__main__":
    main()
