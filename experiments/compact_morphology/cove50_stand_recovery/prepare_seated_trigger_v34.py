"""Freeze the V34 four-route development method and a no-replacement ledger.

Preparation is local and does not start CUDA, collect a visit, or fit a policy.
The separate 35000 sentinel may check launch mechanics, but never tune the
threshold or enter the scored 35001-35012 families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-trigger-v34"
V31D = ROOT / "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d"
SEEDS = tuple(range(35001, 35013))
ROUTES = ("scripted", "model25", "fixed-0p06", "gravity-switch")
SOURCES = {
    "body": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "checkpoint": "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/checkpoints/model_25.pt",
    "v25_plan": "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/plan.json",
    "v26_scorer": "experiments/compact_morphology/cove50_stand_recovery/score_seated_reference_cuda_v26.py",
    "v29_report": "artifacts/ape-cove50-stand-recovery/seated-intervention-v29/report.json",
    "v33_report": "artifacts/ape-cove50-stand-recovery/seated-early-warning-v33/report.json",
    "trainer": "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py",
    "task": "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    "frame50": "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    "evaluator": "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_trigger_v34.py",
    "scorer": "experiments/compact_morphology/cove50_stand_recovery/score_seated_trigger_v34.py",
    "preparer": "experiments/compact_morphology/cove50_stand_recovery/prepare_seated_trigger_v34.py",
    "runner": "experiments/compact_morphology/cove50_stand_recovery/run_seated_trigger_v34.py",
    "v31d_plan": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/plan.json",
    "v31d_report": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/report.json",
    "v31d_installed_closure": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/installed-closure.json",
    "v31d_cuda_sentinel": "artifacts/ape-cove50-stand-recovery/seated-runtime-v31d/cuda-sentinel.json",
}
V31D_PLAN = json.loads((V31D / "plan.json").read_text())
SOURCES.update({f"v31d_material_{name}": relative
                for name, relative in V31D_PLAN["source_paths"].items()})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha(ROOT / relative) for name, relative in SOURCES.items()}


def verify_sources(plan: dict) -> None:
    expected = plan["source_sha256"]
    if set(expected) != set(SOURCES) or expected != source_hashes():
        raise ValueError("V34 frozen source/body/checkpoint differs")
    if (plan["v31d_material_sha256"] != V31D_PLAN["source_sha256"] or
            any(expected[f"v31d_material_{name}"] != digest
                for name, digest in V31D_PLAN["source_sha256"].items())):
        raise ValueError("V34 V31d material source closure differs")
    closure = json.loads((V31D / "installed-closure.json").read_text())
    sentinel = json.loads((V31D / "cuda-sentinel.json").read_text())
    report = json.loads((V31D / "report.json").read_text())
    if (closure["content_sha256"] != plan["v31d_installed_closure_content_sha256"] or
            sentinel["runtime_closure_sha256"] != closure["content_sha256"] or
            sentinel["source_sha256"] != V31D_PLAN["source_sha256"] or
            sentinel["status"] != "constructor_reset_action_and_one_step_passed" or
            report["status"] != "runtime_sentinel_passed_environment_ready_for_staged_seated_development" or
            report["sentinel_file_sha256"] != sha(V31D / "cuda-sentinel.json")):
        raise ValueError("V34 V31d installed closure or CUDA sentinel evidence differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revise-unattempted", action="store_true",
                        help="Archive a rejected draft only if no sentinel or scored visit was attempted")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    plan_path = ART / "plan.json"
    ledger_path = ART / "attempt-ledger.json"
    if args.revise_unattempted and plan_path.exists() and ledger_path.exists():
        old = json.loads(ledger_path.read_text())
        if (old.get("sentinel_attempts", 0) != 0 or
                any(entry.get("attempts", 0) != 0 or entry["status"] != "not_attempted"
                    for family in old["families"] for entry in family["routes"].values()) or
                any(ART.glob("seed*.json")) or any(ART.glob("seed*.npz"))):
            raise RuntimeError("V34 attempted/native evidence exists; draft revision forbidden")
        prior_plan = ART / "rejected-draft-plan.json"
        prior_ledger = ART / "rejected-draft-attempt-ledger.json"
        with prior_plan.open("x") as stream:
            stream.write(plan_path.read_text())
        with prior_ledger.open("x") as stream:
            stream.write(ledger_path.read_text())
        plan_path.unlink()
        ledger_path.unlink()
    if plan_path.exists() or ledger_path.exists():
        raise FileExistsError("V34 plan or ledger already exists; never replace a method or visit")
    plan = {
        "schema": "robot-learning-lab.cove50-seated-trigger.v34.plan",
        "status": "proposed_for_independent_review_before_cuda",
        "seed_families": list(SEEDS), "routes": list(ROUTES),
        "sentinel_seed": 35000,
        "sentinel_role": "unscored launch-mechanics preflight only; no threshold calibration or replacement",
        "v25_held_seeds_sealed": list(range(2240, 2252)),
        "rare_sole_gap_tail_sealed": True,
        "device": "cuda:0", "duration_s": 10.0,
        "original_visits_max": 48, "sentinel_visits_max": 1,
        "no_posthoc_replacement": True,
        "max_route_wall_s": 120, "max_worker_wall_s": 4800,
        "max_pod_wall_s": 6000, "stop_reserve_s": 60,
        "gpu_listed_rate_usd_per_hour": 0.89,
        "gpu_listed_spend_cap_usd": 1.50,
        "guard_contract": "Before sentinel and each original, require plan-bound live-provider guard attestation and conservative local request-time wall/listed-rate deadline; the independent guard owns provider stop.",
        "gravity_trigger": {"start_pre_s": 0.04, "end_pre_s": 0.10,
                            "projected_gravity_x_lt": -0.034,
                            "latch_script_for_remainder": True},
        "fixed_time_switch_pre_s": 0.06,
        "physical_gate": "unchanged V25 full ten-second planted gate, scored by V26 scorer",
        "interpretation": {
            "correction_feasibility": "all 12 four-route families source checked; scripted full gate >=8 and actor-to-trigger rescues >=6",
            "actor_visible_signal_value": "in addition, at least two non-null trigger times each in >=2 families, at least two paired gravity-only full-gate wins versus fixed-0p06, and zero paired gravity-only losses",
            "clock_equivalence": "same or almost all trigger times, or no paired benefit over fixed-0p06: early clock switch remains sufficient explanation",
            "fit": "no optimizer fit from V34 alone; script is a phase target, not a learner-state feedback expert",
        },
        "source_paths": SOURCES,
        "source_sha256": source_hashes(),
        "v31d_material_sha256": V31D_PLAN["source_sha256"],
        "v31d_installed_closure_content_sha256": json.loads(
            (V31D / "installed-closure.json").read_text())["content_sha256"],
    }
    with plan_path.open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    ledger = {
        "schema": "robot-learning-lab.cove50-seated-trigger.v34.attempts",
        "plan_sha256": sha(plan_path), "seed_families": list(SEEDS),
        "routes": list(ROUTES), "no_posthoc_replacement": True,
        "sentinel_seed": 35000, "sentinel_status": "not_attempted_unscored",
        "sentinel_attempts": 0,
        "families": [{"seed": seed, "routes": {
            route: {"status": "not_attempted", "attempts": 0} for route in ROUTES
        }} for seed in SEEDS],
    }
    with ledger_path.open("x") as stream:
        json.dump(ledger, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"V34 proposed method prepared for review: {plan_path}")


if __name__ == "__main__":
    main()
