"""Guarded V35 native development evaluation; never provisions or fits.

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
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/compact_morphology/mjlab_ape/src"), str(ROOT)]
from experiments.compact_morphology.cove50_stand_recovery import prepare_seated_bc_eval_v35_a40_r3 as prep
from experiments.compact_morphology.cove50_stand_recovery import score_seated_bc_eval_v35_a40_r3 as score
from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v35_a40_r3 as a40_runtime

ART, V31D = prep.ART, prep.V31D
BODY = ROOT / prep.SOURCES["body"]
BC_CHECKPOINT = ROOT / prep.SOURCES["bc_checkpoint"]
V25_CHECKPOINT = ROOT / prep.SOURCES["v25_checkpoint"]
EVALUATOR = ROOT / prep.SOURCES["evaluator"]
EVALUATOR_ENV = {"MUJOCO_GL": "egl",
                 "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"}


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("V35 guard times must be UTC with an offset")
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
    if (attestation["schema"] != "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r3.guard-attestation"
            or state["schema"] != "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r3.guard-state"
            or attestation["plan_sha256"] != prep.sha(plan_path)
            or state["plan_sha256"] != prep.sha(plan_path)
            or parse_time(attestation["local_start_request_at_utc"]) != request_at
            or state["pod_id"] != attestation["pod_id"]
            or attestation["provider_desired_status"] != "RUNNING"
            or attestation["provider_runtime_status"] != "running"
            or state["provider_desired_status"] != "RUNNING"
            or state["provider_runtime_status"] != "running"
            or state["status"] != "running"
            or attestation["gpu_id"] != plan["gpu_id"]
            or attestation["listed_rate_usd_per_hour"] != plan["gpu_listed_rate_usd_per_hour"]
            or not request_at <= attested <= checked <= now()
            or (now() - checked).total_seconds() > 30
            or deadline > bound or deadline <= now()):
        raise ValueError("V35 provider guard is stale, unbound, not running or over budget")
    return deadline


def remaining_seconds(plan: dict, worker_start: float, deadline: datetime) -> int:
    worker_remaining = plan["max_worker_wall_s"] - (time.monotonic() - worker_start)
    provider_remaining = (deadline - now()).total_seconds()
    remaining = min(worker_remaining, provider_remaining)
    if remaining < plan["max_route_wall_s"] + plan["route_start_reserve_s"]:
        raise TimeoutError("V35 worker/provider remaining time cannot safely start another original")
    return plan["max_route_wall_s"]


def verify_runtime_closure(plan: dict) -> None:
    prior = json.loads((V31D / "installed-closure.json").read_text())
    closure = a40_runtime.closure(prep.V31D_PLAN)
    a40_runtime.compare_except_hardware(
        closure, prior, plan["v31d_installed_closure_content_sha256"])
    saved = json.loads((ART / "installed-closure-v35.json").read_text())
    if closure != saved:
        raise ValueError("V35 A40 installed closure drifted after first runtime attestation")
    runtime = json.loads((ART / "runtime-attestation.json").read_text())
    graphics = runtime.get("graphics", {})
    if (graphics.get("backend") != "egl"
            or "NVIDIA" not in graphics.get("gl_vendor", "").upper()
            or "NVIDIA" not in graphics.get("gl_renderer", "").upper()):
        raise ValueError("V35 A40 live NVIDIA EGL attestation is missing")


def verify_gpu_identity(plan: dict) -> None:
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip().splitlines()
    if (names != [plan["nvidia_smi_name"]] or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_name(0) != plan["cuda_device_name"]):
        raise ValueError("V35 A40 native GPU identity differs before visit")


def native_visit(seed: int, route: str, timeout_s: int,
                 *, sentinel: bool = False, render_preflight: bool = False,
                 timing: bool = False) -> tuple[Path, Path, Path | None]:
    stem = ART / ("seed35500-render-preflight" if render_preflight else
                  "seed35500-sentinel" if sentinel else
                  "seed35002-timing-bc" if timing else f"seed{seed}-{route}")
    report_path, native_path, log_path = (stem.with_suffix(ext) for ext in (".json", ".npz", ".log"))
    video_path = None if sentinel and not render_preflight else stem.with_suffix(".mp4")
    if any(path.exists() for path in (report_path, native_path, log_path, video_path)
           if path is not None):
        raise FileExistsError(f"{stem}: original output already exists; no replacement")
    checkpoint = BC_CHECKPOINT if route == "bc" else V25_CHECKPOINT
    cmd = [sys.executable, str(EVALUATOR), str(checkpoint),
           "--out", str(report_path), "--native-trajectory-out", str(native_path),
           "--intervention-route", route, "--seed", str(seed), "--device", "cuda:0",
           "--cove50-body-xml", str(BODY), "--mode", "cycle", "--start-squat",
           "--recover-from-start", "--seated-rise-stage", "--randomized-physics"]
    if sentinel:
        cmd.extend(("--preflight-sentinel", "--duration-s", "0.02"))
        if render_preflight:
            cmd.extend(("--render-preflight", "--video", str(video_path)))
    else:
        cmd.extend(("--video", str(video_path)))
        if timing:
            cmd.append("--timing-probe")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    environment.update(EVALUATOR_ENV)
    with log_path.open("x") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    if result.returncode:
        raise RuntimeError(f"{seed} {route}: native evaluator exit {result.returncode}")
    return report_path, native_path, video_path


def verify_native(seed: int, route: str, report_path: Path, native_path: Path,
                  plan: dict, *, sentinel: bool = False,
                  render_preflight: bool = False, timing: bool = False) -> dict:
    report = json.loads(report_path.read_text())
    with np.load(native_path, allow_pickle=False) as data:
        actor = data["actor_observation"]
        critic = data["critic_observation"]
        action = data["raw_action"]
        applied = data["applied_action"]
        processed = data["processed_target"]
    n = len(report["timeseries"])
    full_n = round(plan["duration_s"] / 0.02)
    valid_length = n == 1 if sentinel else n == full_n if timing else 1 <= n <= full_n
    valid_termination = (not report["ended_early"] and not report["end_termination_terms"]
                         if sentinel or timing else
                         (n == full_n and not report["ended_early"] or
                          n < full_n and report["ended_early"] and
                          bool(report["end_termination_terms"])))
    if (report["seed"] != seed or report["intervention_route"] != route
            or report["preflight_sentinel"] is not sentinel
            or report["render_preflight"] is not render_preflight
            or report["timing_probe"] is not timing
            or report["device"] != "cuda:0" or report["gpu_id"] != plan["gpu_id"]
            or report["mujoco_gl"] != plan["evaluator_env"]["MUJOCO_GL"]
            or report["nvidia_driver_capabilities"] != plan["evaluator_env"]["NVIDIA_DRIVER_CAPABILITIES"]
            or report["nvidia_smi_name"] != plan["nvidia_smi_name"]
            or report["cuda_device_name"] != plan["cuda_device_name"]
            or report["hardware_shift"] != "fresh_A40_development_no_RTX4090_pooling"
            or report["body_sha256"] != plan["source_sha256"]["body"]
            or report["checkpoint_sha256"] != plan["source_sha256"][
                "bc_checkpoint" if route == "bc" else "v25_checkpoint"]
            or report["evaluator_source_sha256"] != plan["source_sha256"]["evaluator"]
            or report["evaluation_common_step_counter_after_load"] !=
                plan["evaluation_common_step_counter"]
            or report["leg_target_margin_rad"] is not None
            or report["control_dt_s"] != 0.02 or not valid_length
            or abs(report["duration_s"] - n * 0.02) > 1e-6
            or not valid_termination
            or actor.shape != (n, 73) or critic.shape != (n, 88)
            or action.shape != (n, 21) or applied.shape != (n, 21)
            or processed.shape != (n, 21)
            or not all(np.isfinite(values).all() for values in
                       (actor, critic, action, applied, processed))
            ):
        raise ValueError(f"{seed} {route}: native source, sentinel or trace contract differs")
    if route != "scripted":
        checkpoint = BC_CHECKPOINT if route == "bc" else V25_CHECKPOINT
        gap = float(np.max(np.abs(score.actor_first_action(actor[0], checkpoint) - action[0])))
        if gap > 1e-5:
            raise ValueError(f"{seed} {route}: saved actor first-action gap {gap}")
    scale = np.asarray(report["action_process_scale"])
    offset = np.asarray(report["action_process_offset"])
    if (scale.shape != (21,) or offset.shape != (21,)
            or report["wrapper_action_clip"] is not None
            or not np.array_equal(applied[0], action[0])):
        raise ValueError(f"{seed} {route}: first action boundary differs")
    target = applied[0] * scale + offset
    if report["action_process_clip"] is not None:
        bounds = np.asarray(report["action_process_clip"])
        if bounds.shape != (21, 2):
            raise ValueError(f"{seed} {route}: action clip shape differs")
        target = np.clip(target, bounds[:, 0], bounds[:, 1])
    if np.max(np.abs(target - processed[0])) > 2e-6:
        raise ValueError(f"{seed} {route}: first processed target differs")
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
            or not ledger["no_posthoc_replacement"]
            or plan["evaluator_env"] != EVALUATOR_ENV):
        raise ValueError("V35 plan/ledger seed or route identity differs")
    worker_start = time.monotonic()

    def check_before_visit() -> int:
        prep.verify_sources(plan)
        deadline = verify_guard(plan, plan_path, attestation_path, state_path, request_at)
        verify_runtime_closure(plan)
        verify_gpu_identity(plan)
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
            report_path, native_path, _ = native_visit(35500, "bc", timeout, sentinel=True)
            verify_native(35500, "bc", report_path, native_path, plan, sentinel=True)
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
        raise ValueError("V35 exact CUDA sentinel did not pass; scored visits forbidden")
    sentinel_report = ART / "seed35500-sentinel.json"
    sentinel_native = ART / "seed35500-sentinel.npz"
    if (ledger["sentinel_report_sha256"] != prep.sha(sentinel_report)
            or ledger["sentinel_native_sha256"] != prep.sha(sentinel_native)):
        raise ValueError("V35 saved CUDA sentinel differs from original")
    verify_native(35500, "bc", sentinel_report, sentinel_native, plan, sentinel=True)

    if ledger["render_preflight_status"] == "attempting_unscored":
        ledger["render_preflight_status"] = "failed_original_unscored"
        ledger["render_preflight_failure_reason"] = "Interrupted one-step native video preflight; no retry"
        write_atomic(ledger_path, ledger)
    if ledger["render_preflight_status"] == "not_attempted_unscored":
        timeout = check_before_visit()
        ledger["render_preflight_status"] = "attempting_unscored"
        ledger["render_preflight_attempts"] = 1
        write_atomic(ledger_path, ledger)
        try:
            report_path, native_path, video_path = native_visit(
                plan["render_preflight_seed"], "bc", timeout,
                sentinel=True, render_preflight=True)
            verify_native(plan["render_preflight_seed"], "bc", report_path,
                          native_path, plan, sentinel=True, render_preflight=True)
            source = dict(ledger, render_preflight_status="passed_unscored",
                          render_preflight_report_sha256=prep.sha(report_path),
                          render_preflight_native_sha256=prep.sha(native_path),
                          render_preflight_video_sha256=prep.sha(video_path))
            source["render_preflight_metrics"] = score.verify_render_preflight(plan, source)
            ledger.update(source)
        except Exception as exc:
            ledger["render_preflight_status"] = "failed_original_unscored"
            ledger["render_preflight_failure_reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_atomic(ledger_path, ledger)
    if (ledger["render_preflight_status"] != "passed_unscored"
            or ledger["render_preflight_attempts"] != 1):
        raise ValueError("A40 one-step MJLab render/MP4 preflight did not pass; timing forbidden")
    score.verify_render_preflight(plan, ledger)

    if ledger["timing_status"] == "attempting_unscored":
        ledger["timing_status"] = "failed_original_unscored"
        ledger["timing_failure_reason"] = "Interrupted A40 full-video timing original; no retry"
        write_atomic(ledger_path, ledger)
    if ledger["timing_status"] == "not_attempted_unscored":
        timeout = check_before_visit()
        ledger["timing_status"] = "attempting_unscored"
        ledger["timing_attempts"] = 1
        write_atomic(ledger_path, ledger)
        try:
            started = time.monotonic()
            report_path, native_path, video_path = native_visit(
                plan["timing_seed"], plan["timing_route"], timeout, timing=True)
            elapsed = time.monotonic() - started
            verify_native(plan["timing_seed"], plan["timing_route"],
                          report_path, native_path, plan, timing=True)
            score.decode_video_frame(video_path, plan)
            if elapsed > plan["timing_max_elapsed_wall_s"]:
                raise TimeoutError(f"A40 full-video timing {elapsed:.2f} s exceeds frozen throughput gate")
            ledger.update(timing_status="passed_unscored", timing_elapsed_wall_s=elapsed,
                          timing_report_sha256=prep.sha(report_path),
                          timing_native_sha256=prep.sha(native_path),
                          timing_video_sha256=prep.sha(video_path))
        except Exception as exc:
            ledger["timing_status"] = "failed_original_unscored"
            ledger["timing_failure_reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_atomic(ledger_path, ledger)
    if ledger["timing_status"] != "passed_unscored" or ledger["timing_attempts"] != 1:
        raise ValueError("A40 full-video timing visit did not pass; development originals forbidden")
    timing_stem = ART / "seed35002-timing-bc"
    timing_report, timing_native, timing_video = (timing_stem.with_suffix(ext)
                                                for ext in (".json", ".npz", ".mp4"))
    if (ledger["timing_elapsed_wall_s"] > plan["timing_max_elapsed_wall_s"] or
            ledger["timing_report_sha256"] != prep.sha(timing_report) or
            ledger["timing_native_sha256"] != prep.sha(timing_native) or
            ledger["timing_video_sha256"] != prep.sha(timing_video)):
        raise ValueError("A40 saved timing original differs or exceeded throughput gate")
    verify_native(plan["timing_seed"], plan["timing_route"],
                  timing_report, timing_native, plan, timing=True)

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
                video_path = ART / f"seed{seed}-{route}.mp4"
                if (entry["report_sha256"] != prep.sha(report_path)
                        or entry["native_sha256"] != prep.sha(native_path)
                        or entry["video_sha256"] != prep.sha(video_path)):
                    raise ValueError(f"{seed} {route}: completed source bytes changed")
                verify_native(seed, route, report_path, native_path, plan)
                continue
            if entry["status"] != "not_attempted":
                continue
            timeout = check_before_visit()
            entry.update(status="attempting", attempts=1, started_at_utc=now().isoformat())
            write_atomic(ledger_path, ledger)
            try:
                report_path, native_path, video_path = native_visit(seed, route, timeout)
                verify_native(seed, route, report_path, native_path, plan)
                score.v26.ART = ART
                row, _ = score.v26.score(seed, route, score.mujoco.MjModel.from_xml_path(str(BODY)),
                                         json.loads((score.V25 / "plan.json").read_text())["physical_gate"],
                                         plan["source_sha256"]["body"])
                entry.update(status="completed", report_sha256=prep.sha(report_path),
                             native_sha256=prep.sha(native_path),
                             video_sha256=prep.sha(video_path), full_gate=row["full_gate"],
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
            family["family_source_status"] = "complete_source_checked_three_route_family"
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
