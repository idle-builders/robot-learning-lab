"""Freeze a small Direct Frame forward-to-stop PPO continuation pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4"
FILES = {
    "design_pointer": "robot/ape-current-design.json",
    "body": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "source_checkpoint": "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt",
    "trainer": "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py",
    "native_evaluator": "experiments/compact_morphology/mjlab_ape/evaluate.py",
    "task": "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    "motors": "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    "bootstrap": "experiments/compact_morphology/cove50_low_level/bootstrap_locomotion_fast_v4.sh",
    "source_checker": "experiments/compact_morphology/cove50_low_level/check_locomotion_fast_v4.py",
    "sentinel_worker": "experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_sentinel.py",
    "host_watchdog": "experiments/compact_morphology/cove50_low_level/create_locomotion_fast_v4_pod.py",
    "host_launcher": "experiments/compact_morphology/cove50_low_level/launch_locomotion_fast_v4_sentinel.py",
    "runtime_inputs": "experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt",
}
CASES = (
    (48101, .25, 12, 0), (48102, .35, 10, 0),
    (48103, .45, 12, 10), (48104, .25, 10, 10),
    (48105, .35, 12, 10), (48106, .45, 10, 0),
    (48107, .25, 12, 10), (48108, .45, 12, 0),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    pointer = json.loads((ROOT / FILES["design_pointer"]).read_text())
    if pointer.get("status") != "rejected_no_selected_mechanical_design":
        raise ValueError("Exact Direct Frame control-learning role changed")
    sources = {key: {"path": rel, "sha256": sha(ROOT / rel)} for key, rel in FILES.items()}
    cases = [{"seed": seed, "velocity_x_m_s": speed, "motor_rail_v": rail,
              "command_delay_ms": delay, "initial_root_z_m": .231,
              "stop_at_s": 4.0, "duration_s": 8.0}
             for seed, speed, rail, delay in CASES]
    method = {
        "schema": "robot-learning-lab.cove50-locomotion-fast-v4.method.v1",
        "role": "direct_frame_experimental_basic_control_not_mechanical_qualification",
        "why": "Get one fresh saved learner checkpoint and paired physical evidence for forward-to-stop before expanding skills.",
        "source": sources,
        "prior_method": "V3 two-arm hip-roll hypothesis remains frozen and unrun; V3e one-shot preflight consumed before a native step.",
        "learner": {"algorithm": "MJLab/RSL-RL PPO continuation", "source_iteration": 844,
                    "profile": "stop", "worlds": 512, "additional_updates": 24,
                    "expected_environment_steps": 294912,
                    "walking_hip_roll_pose_std_rad": .05,
                    "ankle_target_margin_rad": .02,
                    "save_interval_updates_max": 8,
                    "full_state_resume_required": True},
        "unscored_sentinel": {"seed": 48097, "native_stop_visit_duration_s": 4.04,
                               "stop_command_at_s": 4.0, "worlds": 1,
                               "disposable_ppo_updates": 1,
                               "candidate_ppo_updates": 0,
                               "worker_timeout_s_max": 1500},
        "readiness": ["source and full checkpoint hash match",
                      "one exact CUDA 50 Hz stop-command transition with finite reset/contact",
                      "one disposable PPO update with finite optimizer/normalizer state and measured throughput; it is not part of the candidate lineage",
                      "before the candidate's 24 updates, compare realized reward and physical outcome on useful, fallen, parked and overshooting native examples; name any unavailable class",
                      "separate host hard-stop watchdog and remote process timeout"],
        "development_cases": cases,
        "matched_arms": ["model844_saved_baseline", "v4_new_checkpoint"],
        "physical_basic_gate_per_case": {
            "full_8s_without_fall": True, "forward_travel_m_min": .7,
            "alternating_touchdowns_min": 6, "max_hand_contact_force_n": 1,
            "stop_tail_travel_m_max": .12,
            "late_double_support_fraction_min": .8},
        "motor_measure": {"moving_rated_leg_duty_target_max": .8,
                          "report_exact_fraction_even_if_gate_passes": True},
        "variation_interpretation": "10 V and 10 ms delay are named stress slices outside the nominal 12 V and 0-8 ms training delay envelope; report in-envelope and stress counts separately.",
        "development_promotion_target": "At least 6/8 basic physical passes, no catastrophic regression, and >=0.05 absolute mean moving-rated-duty reduction against paired baseline; report misses and do not infer motor qualification from PPO reward.",
        "sealed_exclusions": ["V3 49001-49024", "V158 held contact seeds", "V147 independent reach score"],
        "spend": {"gpu": "NVIDIA GeForce RTX 4090", "listed_rate_max_usd_h": .89,
                  "pod_wall_limit_s": 5400, "listed_rental_cap_usd": 1.335,
                  "stop_reserve_s": 180,
                  "provider_confirmed_stop_required": True,
                  "posted_bill_allocation": "unknown_until_provider_posts"},
        "claim_limit": "24-update development PPO pilot and eight matched native forward-stop families; no signed multi-direction locomotion, physical transfer or independent score claim.",
        "status": "method_frozen_before_any_v4_cuda_visit_or_optimizer_update",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(method, indent=2, sort_keys=True) + "\n").encode()
    path = OUT / "method.json"
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("Frozen V4 method differs; version the change")
    path.write_bytes(encoded)
    print(json.dumps({"method": str(path.relative_to(ROOT)), "sha256": sha(path),
                      "cases": len(cases), "updates": 24,
                      "expected_environment_steps": 294912}, indent=2))


if __name__ == "__main__":
    main()
