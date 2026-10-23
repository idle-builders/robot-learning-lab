"""Freeze prospective V35 native development replay and no-replacement ledger.

Local preparation only: no pod, native visit, optimizer or score reserve opening.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r3"
V31D = ROOT / "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d"
SEEDS = tuple(range(35511, 35519))
ROUTES = ("bc", "model25", "scripted")
V31D_PLAN = json.loads((V31D / "plan.json").read_text())
SOURCES = {
    "body": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "bc_checkpoint": "artifacts/ape-cove50-stand-recovery/seated-bc-v35/model_bc_seed.pt",
    "v25_checkpoint": "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/checkpoints/model_25.pt",
    "v25_plan": "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/plan.json",
    "bank_manifest": "artifacts/ape-cove50-stand-recovery/seated-bc-v35/bank-manifest.json",
    "fit_report": "artifacts/ape-cove50-stand-recovery/seated-bc-v35/fit-report.json",
    "v34_closure": "artifacts/ape-cove50-stand-recovery/seated-trigger-v34/v34-closure.json",
    "unopened_rtx_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35/plan.json",
    "unopened_rtx_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35/attempt-ledger.json",
    "evaluator": "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_v35_a40_r3.py",
    "scorer": "experiments/compact_morphology/cove50_stand_recovery/score_seated_bc_eval_v35_a40_r3.py",
    "runner": "experiments/compact_morphology/cove50_stand_recovery/run_seated_bc_eval_v35_a40_r3.py",
    "preparer": "experiments/compact_morphology/cove50_stand_recovery/prepare_seated_bc_eval_v35_a40_r3.py",
    "v26_scorer": "experiments/compact_morphology/cove50_stand_recovery/score_seated_reference_cuda_v26.py",
    "trainer": "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py",
    "task": "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    "frame50": "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    "v31d_plan": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/plan.json",
    "v31d_report": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/report.json",
    "v31d_installed_closure": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/installed-closure.json",
    "v31d_cuda_sentinel": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/cuda-sentinel.json",
    "failed_a40_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40/plan.json",
    "failed_a40_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40/attempt-ledger.json",
    "failed_a40_runtime_log": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40/first-runtime-failure.log",
    "failed_a40_r1_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/plan.json",
    "failed_a40_r1_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/attempt-ledger.json",
    "failed_a40_r1_sentinel_log": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/seed35100-sentinel.log",
    "failed_a40_r1_runtime_attestation": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1/runtime-attestation.json",
    "failed_a40_r2_plan": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/plan.json",
    "failed_a40_r2_ledger": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/attempt-ledger.json",
    "failed_a40_r2_timing_log": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/seed35001-timing-bc.log",
    "failed_a40_r2_failure_summary": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/failure-summary.json",
    "failed_a40_r2_sentinel_report": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/seed35300-sentinel.json",
    "failed_a40_r2_sentinel_native": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/seed35300-sentinel.npz",
    "failed_a40_r2_runtime_attestation": "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r2/runtime-attestation.json",
    "a40_runtime_repair": "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_runtime_v35_a40_r3.py",
}
SOURCES.update({f"v31d_material_{name}": rel
                for name, rel in V31D_PLAN["source_paths"].items()})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha(ROOT / rel) for name, rel in SOURCES.items()}


def verify_sources(plan: dict) -> None:
    if plan["source_paths"] != SOURCES or plan["source_sha256"] != source_hashes():
        raise ValueError("V35 source/body/checkpoint changed from frozen plan")
    bank = json.loads((ROOT / SOURCES["bank_manifest"]).read_text())
    fit = json.loads((ROOT / SOURCES["fit_report"]).read_text())
    closure = json.loads((ROOT / SOURCES["v34_closure"]).read_text())
    if (bank["train_families"] != list(range(35001, 35007))
            or bank["fresh_development_validation_seed_families"] != list(range(35101, 35109))
            or bank["separate_unopened_score_seed_families"] != list(range(35201, 35213))
            or fit["bank_manifest_sha256"] != sha(ROOT / SOURCES["bank_manifest"])
            or fit["checkpoint_sha256"] != sha(ROOT / SOURCES["bc_checkpoint"])
            or fit["chosen_epochs"] != 12
            or bank["v34_closure_sha256"] != sha(ROOT / SOURCES["v34_closure"])
            or closure["v34_closed"] is not True
            or plan["v31d_material_sha256"] != V31D_PLAN["source_sha256"]):
        raise ValueError("V35 fit, source closure or disjoint seeds differ")
    installed = json.loads((V31D / "installed-closure.json").read_text())
    if installed["content_sha256"] != plan["v31d_installed_closure_content_sha256"]:
        raise ValueError("V35 installed runtime closure differs from V31d")
    prior = json.loads((ROOT / SOURCES["unopened_rtx_ledger"]).read_text())
    if (prior["sentinel_attempts"] != 0 or
            any(route["attempts"] != 0 for family in prior["families"]
                for route in family["routes"].values()) or
            prior["plan_sha256"] != sha(ROOT / SOURCES["unopened_rtx_plan"])):
        raise ValueError("V35 RTX packet was opened; A40 cannot reuse these seed families")
    failed = json.loads((ROOT / SOURCES["failed_a40_ledger"]).read_text())
    if (failed["sentinel_attempts"] != 0 or failed["timing_attempts"] != 0
            or any(route["attempts"] != 0 for family in failed["families"]
                   for route in family["routes"].values())
            or failed["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_plan"])):
        raise ValueError("First A40 packet opened a native visit; fresh original method required")
    failed_r1 = json.loads((ROOT / SOURCES["failed_a40_r1_ledger"]).read_text())
    if (failed_r1["sentinel_seed"] != 35100 or failed_r1["sentinel_attempts"] != 1
            or failed_r1["sentinel_status"] != "failed_original_unscored"
            or failed_r1["timing_attempts"] != 0
            or any(route["attempts"] != 0 for family in failed_r1["families"]
                   for route in family["routes"].values())
            or failed_r1["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_r1_plan"])):
        raise ValueError("A40-r1 stopped sentinel or development source differs")
    failed_r1_log = (ROOT / SOURCES["failed_a40_r1_sentinel_log"]).read_text()
    if "FileNotFoundError" not in failed_r1_log or "evaluate_seated_bc_v35_a40_r1.py" not in failed_r1_log:
        raise ValueError("A40-r1 original failure evidence differs")
    attestation = json.loads((ROOT / SOURCES["failed_a40_r1_runtime_attestation"]).read_text())
    if (attestation["installed_count"] != 227
            or attestation["status"] != "installed_source_attested_before_v35_cuda"
            or attestation["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_r1_plan"])):
        raise ValueError("A40-r1 installed runtime success source differs")
    failed_r2 = json.loads((ROOT / SOURCES["failed_a40_r2_ledger"]).read_text())
    summary = json.loads((ROOT / SOURCES["failed_a40_r2_failure_summary"]).read_text())
    if (failed_r2["sentinel_seed"] != 35300 or failed_r2["sentinel_attempts"] != 1
            or failed_r2["sentinel_status"] != "passed_unscored"
            or failed_r2["timing_seed"] != 35001 or failed_r2["timing_attempts"] != 1
            or failed_r2["timing_status"] != "failed_original_unscored"
            or any(route["attempts"] != 0 for family in failed_r2["families"]
                   for route in family["routes"].values())
            or failed_r2["plan_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_plan"])
            or summary["attempt_ledger_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_ledger"])
            or summary["timing_log_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_timing_log"])
            or summary["sentinel_report_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_sentinel_report"])
            or summary["sentinel_native_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_sentinel_native"])
            or summary["runtime_attestation_sha256"] != sha(ROOT / SOURCES["failed_a40_r2_runtime_attestation"])):
        raise ValueError("A40-r2 consumed render failure source differs")
    if ("an OpenGL platform library has not been loaded" not in
            (ROOT / SOURCES["failed_a40_r2_timing_log"]).read_text()):
        raise ValueError("A40-r2 original headless-render failure differs")


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    plan_path, ledger_path = ART / "plan.json", ART / "attempt-ledger.json"
    if plan_path.exists() or ledger_path.exists():
        raise FileExistsError("V35 plan/ledger exists; method or original cannot be replaced")
    from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v35_a40_r3 as runtime
    runtime.static_smoke()
    plan = {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r3.plan",
        "status": "prospective_headless_render_repair_for_independent_review_before_paid_cuda",
        "claim_limit": "Fresh A40 development evaluation only; no pooling with RTX 4090 or treating BC CPU loss as native control",
        "hardware_shift": "Earlier A40 packets failed an RTX-only closure and an absent evaluator path. A40-r2 passed its native one-step BC CUDA sentinel but consumed its full-video timing visit when headless MJLab rendering lacked an OpenGL platform. This packet sets EGL before the evaluator imports MuJoCo and gates development on paired one-step render and no-render probes, video decode, and a fresh full-video timing visit.",
        "failed_a40_plan_sha256": sha(ROOT / SOURCES["failed_a40_plan"]),
        "failed_a40_ledger_sha256": sha(ROOT / SOURCES["failed_a40_ledger"]),
        "failed_a40_runtime_log_sha256": sha(ROOT / SOURCES["failed_a40_runtime_log"]),
        "failed_a40_r1_plan_sha256": sha(ROOT / SOURCES["failed_a40_r1_plan"]),
        "failed_a40_r1_ledger_sha256": sha(ROOT / SOURCES["failed_a40_r1_ledger"]),
        "failed_a40_r1_sentinel_log_sha256": sha(ROOT / SOURCES["failed_a40_r1_sentinel_log"]),
        "failed_a40_r1_runtime_attestation_sha256": sha(ROOT / SOURCES["failed_a40_r1_runtime_attestation"]),
        "failed_a40_r1_sentinel_consumed": 35100,
        "failed_a40_r1_pod_id": "4bhiy36ik82msf",
        "failed_a40_r1_originals_unopened": list(range(35101, 35109)),
        "failed_a40_r2_plan_sha256": sha(ROOT / SOURCES["failed_a40_r2_plan"]),
        "failed_a40_r2_ledger_sha256": sha(ROOT / SOURCES["failed_a40_r2_ledger"]),
        "failed_a40_r2_timing_log_sha256": sha(ROOT / SOURCES["failed_a40_r2_timing_log"]),
        "failed_a40_r2_failure_summary_sha256": sha(ROOT / SOURCES["failed_a40_r2_failure_summary"]),
        "failed_a40_r2_sentinel_consumed": 35300,
        "failed_a40_r2_timing_consumed": 35001,
        "failed_a40_r2_pod_id": "a9osk1e2qoaat5",
        "failed_a40_r2_originals_unopened": list(range(35301, 35309)),
        "gpu_id": "NVIDIA A40", "cuda_device_name": "NVIDIA A40",
        "nvidia_smi_name": "NVIDIA A40",
        "unopened_rtx_plan_sha256": sha(ROOT / SOURCES["unopened_rtx_plan"]),
        "unopened_rtx_ledger_sha256": sha(ROOT / SOURCES["unopened_rtx_ledger"]),
        "seed_families": list(SEEDS), "routes": list(ROUTES),
        "fresh_seed_rationale": "A40-r1 consumed sentinel 35100; A40-r2 consumed sentinel 35300 and training-family timing 35001. A40-r3 uses new paired one-step seed 35500, previously opened BC training-family seed 35002 for an unscored timing diagnostic, and independent development families 35511–35518. Score reserve 35201–35212 stays sealed.",
        "sentinel_seed": 35500, "sentinel_role": "one-step BC no-video strict CUDA load and first-action/target parity",
        "render_preflight_seed": 35500,
        "render_preflight_role": "second one-step BC visit from same seed with exact MJLab 640x480 render, MP4 encode/decode and physical/action parity to no-video sentinel",
        "timing_seed": 35002,
        "timing_role": "unscored already-opened training seed, full 10 s BC plus inline native video before any development original",
        "timing_route": "bc", "timing_max_elapsed_wall_s": 300,
        "v25_historical_hold_seeds_sealed": list(range(2240, 2252)),
        "v35_independent_score_seeds_sealed": list(range(35201, 35213)),
        "rare_sole_gap_tail_sealed": True,
        "device": "cuda:0", "duration_s": 10.0,
        "original_visits_max": 24, "sentinel_visits_max": 2, "timing_visits_max": 1,
        "evaluator_env": {"MUJOCO_GL": "egl", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"},
        "render_preflight_frame_shape": [480, 640, 3],
        "render_preflight_min_pixel_std": 1.0,
        "render_preflight_physical_max_abs": 1e-4,
        "render_preflight_actor_max_abs": 1e-5,
        "no_posthoc_replacement": True,
        "max_route_wall_s": 360, "route_start_reserve_s": 20,
        "max_worker_wall_s": 9600, "max_pod_wall_s": 10800,
        "stop_reserve_s": 60,
        "gpu_listed_rate_usd_per_hour": 0.59,
        "gpu_listed_spend_cap_usd": 1.85,
        "guard_contract": "Before paired one-step probes, full-video timing probe and every original: current A40 plan-bound provider attestation, EGL/graphics environment before MuJoCo import, nvidia-smi/Torch A40 identity and fresh 227-package closure with all non-hardware V31d fields exact; independent 10-second stop watchdog; no start unless >=380 s remain. A one-step native render must encode and decode a nonblank 640x480 MP4 frame, and remain physically/action aligned to same-seed no-video. Stop before development if either preflight or the full-video timing visit fails or exceeds 300 s. Stop pod promptly on completion.",
        "first_action_parity_max_abs": 1e-5,
        "processed_target_parity_max_abs": 2e-6,
        "evaluation_common_step_counter": 624,
        "physical_gate": "unchanged V25 full ten-second planted gate scored by V26 source",
        "video": "native original mp4 from same evaluator visit as report/trace; one per scored route",
        "interpretation": {
            "pilot_signal": ">=2/8 BC full-gate passes and >=2 paired BC wins over model25",
            "stop_branch": "0/8 BC passes with no prospective physical improvement: stop phase-only BC",
            "qualification": "No learner qualification from BC training loss, incomplete pairs or development score alone; independent 12-family score stays sealed until checkpoint/method freeze",
        },
        "source_paths": SOURCES, "source_sha256": source_hashes(),
        "v31d_material_sha256": V31D_PLAN["source_sha256"],
        "v31d_installed_closure_content_sha256": json.loads(
            (V31D / "installed-closure.json").read_text())["content_sha256"],
    }
    with plan_path.open("x") as output:
        output.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    ledger = {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r3.attempts",
        "plan_sha256": sha(plan_path), "seed_families": list(SEEDS),
        "routes": list(ROUTES), "no_posthoc_replacement": True,
        "sentinel_seed": 35500, "sentinel_status": "not_attempted_unscored",
        "sentinel_attempts": 0,
        "render_preflight_seed": 35500, "render_preflight_status": "not_attempted_unscored",
        "render_preflight_attempts": 0,
        "timing_seed": 35002, "timing_route": "bc",
        "timing_status": "not_attempted_unscored", "timing_attempts": 0,
        "families": [{"seed": seed, "routes": {
            route: {"status": "not_attempted", "attempts": 0}
            for route in ROUTES}}
            for seed in SEEDS],
    }
    with ledger_path.open("x") as output:
        output.write(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"Prepared V35 plan {sha(plan_path)}; zero native visits")


if __name__ == "__main__":
    main()
