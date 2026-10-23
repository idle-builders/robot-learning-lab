"""Freeze prospective V35 native development replay and no-replacement ledger.

Local preparation only: no pod, native visit, optimizer or score reserve opening.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r1"
V31D = ROOT / "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d"
SEEDS = tuple(range(35101, 35109))
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
    "evaluator": "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_v35_a40.py",
    "scorer": "experiments/compact_morphology/cove50_stand_recovery/score_seated_bc_eval_v35_a40_r1.py",
    "runner": "experiments/compact_morphology/cove50_stand_recovery/run_seated_bc_eval_v35_a40_r1.py",
    "preparer": "experiments/compact_morphology/cove50_stand_recovery/prepare_seated_bc_eval_v35_a40_r1.py",
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
    "a40_runtime_repair": "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_runtime_v35_a40_r1.py",
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
            or bank["fresh_development_validation_seed_families"] != list(SEEDS)
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


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    plan_path, ledger_path = ART / "plan.json", ART / "attempt-ledger.json"
    if plan_path.exists() or ledger_path.exists():
        raise FileExistsError("V35 plan/ledger exists; method or original cannot be replaced")
    from experiments.compact_morphology.cove50_stand_recovery import preflight_seated_runtime_v35_a40_r1 as runtime
    runtime.static_smoke()
    plan = {
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r1.plan",
        "status": "prospective_runtime_gate_repair_for_independent_review_before_paid_cuda",
        "claim_limit": "Fresh A40 development evaluation only; no pooling with RTX 4090 or treating BC CPU loss as native control",
        "hardware_shift": "Failed first A40 packet stopped before CUDA sentinel because V31d hardcoded RTX 4090. This repaired packet compares every installed package/import/source field while explicitly pinning A40; fresh A40 runtime/physics parity required.",
        "failed_a40_plan_sha256": sha(ROOT / SOURCES["failed_a40_plan"]),
        "failed_a40_ledger_sha256": sha(ROOT / SOURCES["failed_a40_ledger"]),
        "failed_a40_runtime_log_sha256": sha(ROOT / SOURCES["failed_a40_runtime_log"]),
        "gpu_id": "NVIDIA A40", "cuda_device_name": "NVIDIA A40",
        "nvidia_smi_name": "NVIDIA A40",
        "unopened_rtx_plan_sha256": sha(ROOT / SOURCES["unopened_rtx_plan"]),
        "unopened_rtx_ledger_sha256": sha(ROOT / SOURCES["unopened_rtx_ledger"]),
        "seed_families": list(SEEDS), "routes": list(ROUTES),
        "sentinel_seed": 35100, "sentinel_role": "one-step BC strict CUDA load and first-action/target parity",
        "timing_seed": 35001,
        "timing_role": "unscored already-opened training seed, full 10 s BC plus inline native video before any development original",
        "timing_route": "bc", "timing_max_elapsed_wall_s": 300,
        "v25_historical_hold_seeds_sealed": list(range(2240, 2252)),
        "v35_independent_score_seeds_sealed": list(range(35201, 35213)),
        "rare_sole_gap_tail_sealed": True,
        "device": "cuda:0", "duration_s": 10.0,
        "original_visits_max": 24, "sentinel_visits_max": 1, "timing_visits_max": 1,
        "no_posthoc_replacement": True,
        "max_route_wall_s": 360, "route_start_reserve_s": 20,
        "max_worker_wall_s": 9600, "max_pod_wall_s": 10800,
        "stop_reserve_s": 60,
        "gpu_listed_rate_usd_per_hour": 0.59,
        "gpu_listed_spend_cap_usd": 1.85,
        "guard_contract": "Before sentinel, full-video timing probe and every original: current A40 plan-bound provider attestation, nvidia-smi/Torch A40 identity and fresh 227-package closure with all non-hardware V31d fields exact; independent 10-second stop watchdog; no start unless >=380 s remain. Stop before development collection if the one unscored timing visit fails, exceeds 300 s, or lacks matched video/report/trace. Stop pod promptly on completion.",
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
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r1.attempts",
        "plan_sha256": sha(plan_path), "seed_families": list(SEEDS),
        "routes": list(ROUTES), "no_posthoc_replacement": True,
        "sentinel_seed": 35100, "sentinel_status": "not_attempted_unscored",
        "sentinel_attempts": 0,
        "timing_seed": 35001, "timing_route": "bc",
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
