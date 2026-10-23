"""Guarded V34 native collection; never provisions a paid host or fits a policy.

An independent provider watchdog must keep the plan-bound guard state fresh and
stop the host at its deadline. An interrupted original is consumed, not retried.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/compact_morphology/mjlab_ape/src"), str(ROOT)]
from experiments.compact_morphology.cove50_stand_recovery import prepare_seated_trigger_v34 as prep
from experiments.compact_morphology.cove50_stand_recovery import score_seated_trigger_v34 as score
from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v31d as v31d

ART, V31D = prep.ART, prep.V31D
BODY = ROOT / prep.SOURCES["body"]
CHECKPOINT = ROOT / prep.SOURCES["checkpoint"]
EVALUATOR = ROOT / prep.SOURCES["evaluator"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("V34 guard times must be UTC with an offset")
    return parsed


def write_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_guard(plan: dict, plan_path: Path, attestation_path: Path,
                 state_path: Path, request_at: datetime) -> datetime:
    attestation = json.loads(attestation_path.read_text())
    state = json.loads(state_path.read_text())
    deadline = parse_time(attestation["hard_deadline_utc"])
    attested = parse_time(attestation["attested_at_utc"])
    checked = parse_time(state["last_check_at_utc"])
    bound = min(
        request_at + timedelta(seconds=plan["max_pod_wall_s"] - plan["stop_reserve_s"]),
        request_at + timedelta(seconds=(plan["gpu_listed_spend_cap_usd"] * 3600 /
                                        plan["gpu_listed_rate_usd_per_hour"] -
                                        plan["stop_reserve_s"])),
    )
    if (attestation["schema"] != "robot-learning-lab.cove50-seated-trigger.v34.guard-attestation"
            or state["schema"] != "robot-learning-lab.cove50-seated-trigger.v34.guard-state"
            or attestation["plan_sha256"] != prep.sha(plan_path)
            or state["plan_sha256"] != prep.sha(plan_path)
            or parse_time(attestation["local_start_request_at_utc"]) != request_at
            or state["pod_id"] != attestation["pod_id"]
            or attestation["provider_desired_status"] != "RUNNING"
            or attestation["provider_runtime_status"] != "running"
            or state["provider_desired_status"] != "RUNNING"
            or state["provider_runtime_status"] != "running"
            or state["status"] != "running"
            or attestation["gpu_id"] != "NVIDIA GeForce RTX 4090"
            or attestation["listed_rate_usd_per_hour"] != plan["gpu_listed_rate_usd_per_hour"]
            or not request_at <= attested <= checked <= now()
            or (now() - checked).total_seconds() > 30
            or deadline > bound or deadline <= now()):
        raise ValueError("V34 provider guard is stale, unbound, not running or over budget")
    return deadline


def remaining_seconds(plan: dict, worker_start: float, deadline: datetime) -> int:
    worker_remaining = plan["max_worker_wall_s"] - (time.monotonic() - worker_start)
    provider_remaining = (deadline - now()).total_seconds()
    remaining = min(worker_remaining, provider_remaining)
    if remaining <= 1:
        raise TimeoutError("V34 worker, pod wall or listed-spend deadline reached")
    return max(1, min(plan["max_route_wall_s"], int(remaining)))


def verify_runtime_closure(plan: dict) -> None:
    prior = json.loads((V31D / "installed-closure.json").read_text())
    closure = v31d.installed_closure(prep.V31D_PLAN)
    if (closure != prior or closure["content_sha256"] !=
            plan["v31d_installed_closure_content_sha256"]):
        raise ValueError("V34 installed package/source closure differs from V31d")


def native_visit(seed: int, route: str, timeout_s: int,
                 *, sentinel: bool = False) -> tuple[Path, Path]:
    stem = ART / ("seed35000-sentinel" if sentinel else f"seed{seed}-{route}")
    report_path, native_path, log_path = (stem.with_suffix(ext) for ext in (".json", ".npz", ".log"))
    if any(path.exists() for path in (report_path, native_path, log_path)):
        raise FileExistsError(f"{stem}: original output already exists; no replacement")
    cmd = [sys.executable, str(EVALUATOR), str(CHECKPOINT),
           "--out", str(report_path), "--native-trajectory-out", str(native_path),
           "--intervention-route", route, "--seed", str(seed), "--device", "cuda:0",
           "--cove50-body-xml", str(BODY), "--mode", "cycle", "--start-squat",
           "--recover-from-start", "--seated-rise-stage", "--randomized-physics"]
    if sentinel:
        cmd.extend(("--preflight-sentinel", "--duration-s", "0.02"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    with log_path.open("x") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    if result.returncode:
        raise RuntimeError(f"{seed} {route}: native evaluator exit {result.returncode}")
    return report_path, native_path


def verify_native(seed: int, route: str, report_path: Path, native_path: Path,
                  plan: dict, *, sentinel: bool = False) -> dict:
    report = json.loads(report_path.read_text())
    with np.load(native_path, allow_pickle=False) as data:
        actor = data["actor_observation"]
        critic = data["critic_observation"]
        action = data["raw_action"]
    n = len(report["timeseries"])
    full_n = round(plan["duration_s"] / 0.02)
    valid_length = n == 1 if sentinel else 1 <= n <= full_n
    valid_termination = (not report["ended_early"] and not report["end_termination_terms"]
                         if sentinel else
                         (n == full_n and not report["ended_early"] or
                          n < full_n and report["ended_early"] and
                          bool(report["end_termination_terms"])))
    if (report["seed"] != seed or report["intervention_route"] != route
            or report["preflight_sentinel"] is not sentinel
            or report["device"] != "cuda:0" or report["body_sha256"] != plan["source_sha256"]["body"]
            or report["checkpoint_sha256"] != plan["source_sha256"]["checkpoint"]
            or report["evaluator_source_sha256"] != plan["source_sha256"]["evaluator"]
            or report["leg_target_margin_rad"] is not None
            or report["control_dt_s"] != 0.02 or not valid_length
            or abs(report["duration_s"] - n * 0.02) > 1e-6
            or not valid_termination
            or actor.shape != (n, 73) or critic.shape != (n, 88)
            or action.shape != (n, 21)
            or not all(np.isfinite(values).all() for values in (actor, critic, action))
            ):
        raise ValueError(f"{seed} {route}: native source, sentinel or trace contract differs")
    return report


def run(attestation_path: Path, state_path: Path, request_at: datetime) -> None:
    plan_path, ledger_path = ART / "plan.json", ART / "attempt-ledger.json"
    plan = json.loads(plan_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    prep.verify_sources(plan)
    if (ledger["plan_sha256"] != prep.sha(plan_path)
            or ledger["seed_families"] != list(prep.SEEDS)
            or ledger["routes"] != list(prep.ROUTES)
            or [family["seed"] for family in ledger["families"]] != list(prep.SEEDS)
            or not ledger["no_posthoc_replacement"]):
        raise ValueError("V34 plan/ledger seed or route identity differs")
    worker_start = time.monotonic()

    def check_before_visit() -> int:
        prep.verify_sources(plan)
        deadline = verify_guard(plan, plan_path, attestation_path, state_path, request_at)
        verify_runtime_closure(plan)
        return remaining_seconds(plan, worker_start, deadline)

    if ledger["sentinel_status"] == "attempting_unscored":
        ledger["sentinel_status"] = "failed_original_unscored"
        ledger["sentinel_failure_reason"] = "Interrupted exact CUDA sentinel; no retry"
        write_atomic(ledger_path, ledger)
    if ledger["sentinel_status"] == "not_attempted_unscored":
        timeout = check_before_visit()
        ledger["sentinel_status"] = "attempting_unscored"
        ledger["sentinel_attempts"] = 1
        write_atomic(ledger_path, ledger)
        try:
            report_path, native_path = native_visit(35000, "model25", timeout, sentinel=True)
            verify_native(35000, "model25", report_path, native_path, plan, sentinel=True)
            ledger["sentinel_status"] = "passed_unscored"
            ledger["sentinel_report_sha256"] = prep.sha(report_path)
            ledger["sentinel_native_sha256"] = prep.sha(native_path)
        except Exception as exc:
            ledger["sentinel_status"] = "failed_original_unscored"
            ledger["sentinel_failure_reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_atomic(ledger_path, ledger)
    if ledger["sentinel_status"] != "passed_unscored" or ledger["sentinel_attempts"] != 1:
        raise ValueError("V34 exact CUDA sentinel did not pass; scored visits forbidden")
    sentinel_report = ART / "seed35000-sentinel.json"
    sentinel_native = ART / "seed35000-sentinel.npz"
    if (ledger["sentinel_report_sha256"] != prep.sha(sentinel_report)
            or ledger["sentinel_native_sha256"] != prep.sha(sentinel_native)):
        raise ValueError("V34 saved CUDA sentinel differs from original")
    verify_native(35000, "model25", sentinel_report, sentinel_native, plan, sentinel=True)

    for family in ledger["families"]:
        seed = family["seed"]
        for route in prep.ROUTES:
            entry = family["routes"][route]
            if entry["status"] == "attempting":
                entry.update(status="failed_original", failure_reason="Interrupted original; no retry")
                write_atomic(ledger_path, ledger)
            if entry["status"] == "completed":
                report_path = ART / f"seed{seed}-{route}.json"
                native_path = ART / f"seed{seed}-{route}.npz"
                if (entry["report_sha256"] != prep.sha(report_path)
                        or entry["native_sha256"] != prep.sha(native_path)):
                    raise ValueError(f"{seed} {route}: completed source bytes changed")
                verify_native(seed, route, report_path, native_path, plan)
                continue
            if entry["status"] != "not_attempted":
                continue
            timeout = check_before_visit()
            entry.update(status="attempting", attempts=1, started_at_utc=now().isoformat())
            write_atomic(ledger_path, ledger)
            try:
                report_path, native_path = native_visit(seed, route, timeout)
                verify_native(seed, route, report_path, native_path, plan)
                score.v26.ART = ART
                row, _ = score.v26.score(seed, route, score.mujoco.MjModel.from_xml_path(str(BODY)),
                                         json.loads((score.V25 / "plan.json").read_text())["physical_gate"],
                                         plan["source_sha256"]["body"])
                entry.update(status="completed", report_sha256=prep.sha(report_path),
                             native_sha256=prep.sha(native_path), full_gate=row["full_gate"],
                             finished_at_utc=now().isoformat())
            except Exception as exc:
                entry.update(status="failed_original", failure_reason=f"{type(exc).__name__}: {exc}",
                             finished_at_utc=now().isoformat())
                raise
            finally:
                write_atomic(ledger_path, ledger)
        if all(family["routes"][route]["status"] == "completed" for route in prep.ROUTES):
            try:
                score.score_seed(seed, plan, ledger,
                                 score.mujoco.MjModel.from_xml_path(str(BODY)),
                                 json.loads((score.V25 / "plan.json").read_text())["physical_gate"])
            except Exception as exc:
                family["family_source_status"] = "invalid_original_family"
                family["family_failure_reason"] = f"{type(exc).__name__}: {exc}"
                write_atomic(ledger_path, ledger)
                raise
            family["family_source_status"] = "complete_source_checked_four_route_family"
            write_atomic(ledger_path, ledger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard-attestation", type=Path, required=True)
    parser.add_argument("--guard-state", type=Path, required=True)
    parser.add_argument("--local-start-request-at-utc", required=True)
    args = parser.parse_args()
    try:
        run(args.guard_attestation, args.guard_state,
            parse_time(args.local_start_request_at_utc))
    finally:
        ledger_path = ART / "attempt-ledger.json"
        if ledger_path.exists():
            try:
                score.main()
            except Exception as exc:
                write_atomic(ART / "score-failure.json", {
                    "status": "source_or_incomplete_score_failure",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "attempt_ledger_sha256": prep.sha(ledger_path),
                })


if __name__ == "__main__":
    main()
