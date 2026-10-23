"""Freeze the bounded V3 movement-discovery pilot before CUDA or scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/cardinal-v3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    path = OUT / "method.json"
    if path.exists():
        raise RuntimeError("V3 method already frozen")
    method = json.loads((OLD / "method.json").read_text())
    method["schema"] = "robot-learning-lab.cove50-cardinal-v3.method.v1"
    method["status"] = "method_frozen_before_any_v3_cuda_or_optimizer_visit"
    method["hypothesis"] = ("A symmetric signed-direction progress reward plus a wider walking hip-roll pose prior "
                            "permits the saved V2 forward actor to discover reverse/side footfalls. This is one "
                            "movement-discovery bundle, not an isolated reward causal test.")
    method["rivals"] = ["Reward pays rocking/sliding without net movement or alternating contact",
                        "The actor has no base linear-velocity observation, so feedback cannot track new directions",
                        "No validated reverse/side demonstration exists for this body",
                        "The leg support/effort envelope may make a side step infeasible"]
    method["changed_training_variable"] = "V2 continuation plus signed reward weight 2 and walking hip-roll pose std 0.05→0.2 rad"
    method["learner"]["additional_updates"] = 24
    method["learner"]["expected_environment_steps"] = 1024 * 24 * 24
    method["learner"]["imitation"] = "none_valid_for_reverse_or_side; full-state V2 PPO warm start"
    method["learner"]["directional_progress_weight"] = 2.0
    method["learner"]["walking_hip_roll_pose_std_rad"] = 0.2
    method["learner"]["source_iteration"] = 883
    method["learner"]["expected_last_saved_iteration_min"] = 906
    method["learner"]["save_interval_updates"] = 12
    method["comparison"]["arms"] = ["v2_equal_step_iteration883", "v3_continuation"]
    method["comparison"]["native_visits"] = 16
    method["comparison"]["claim_limit"] = "Bounded changed-mechanism development pilot; 24 more PPO updates, no robustness or stop claim"
    for case in method["development_cases"]:
        case["seed"] += 100
    method["cuda_entry"]["seed"] = 49398
    method["score"]["pilot_signal"] = "at least one new-direction V2-failure rescue and both forward controls preserved; two of six rescues needed for another locomotion allocation"
    method["score"]["inspect_loaded_sole_motion"] = True
    method["score"]["check_net_not_just_velocity_reward"] = True
    method["source"] = {
        "body": method["source"]["body"],
        "v2_checkpoint": {"path": "artifacts/ape-low-level-training-20261014/cardinal-v2/candidate/candidate-final.pt",
                          "sha256": sha(OLD / "candidate/candidate-final.pt")},
        "trainer": {"path": "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py",
                    "sha256": sha(ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py")},
        "reward_audit": {"path": "artifacts/ape-low-level-training-20261015/cardinal-v3/reward-audit.json",
                         "sha256": sha(OUT / "reward-audit.json")},
    }
    method["independent_review"] = "artifacts/ape-low-level-training-20261015/cardinal-v3/INDEPENDENT-METHOD-REVIEW.md"
    method["spend"]["listed_rental_cap_usd"] = .75
    method["spend"]["pod_wall_limit_s"] = 4500
    method["claim_limit"] = "Experimental Direct Frame command-conditioned locomotion development only; no physical body or independent robustness qualification."
    path.write_text(json.dumps(method, indent=2, sort_keys=True) + "\n")
    print(sha(path))


if __name__ == "__main__":
    main()
