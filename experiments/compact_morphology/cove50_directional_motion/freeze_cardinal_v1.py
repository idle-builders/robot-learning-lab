"""Freeze a short cardinal travel and stop learning pilot before GPU work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v1"
FILES = {
    "design_pointer": "robot/ape-current-design.json",
    "body": "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "source_checkpoint": "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt",
    "trainer": "experiments/compact_morphology/cove50_policy_transfer/train_velocity.py",
    "native_evaluator": "experiments/compact_morphology/mjlab_ape/evaluate.py",
    "task": "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
    "motors": "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
    "scorer": "experiments/compact_morphology/cove50_directional_motion/summarize_cardinal_v1.py",
    "resume_preflight": "experiments/compact_morphology/cove50_low_level/preflight_cardinal_resume_v1.py",
    "runtime_inputs": "experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pointer = json.loads((ROOT / FILES["design_pointer"]).read_text())
    if (pointer["id"] != "cove-50-direct-frame"
            or pointer["status"] != "rejected_no_selected_mechanical_design"):
        raise ValueError("Exact Direct Frame experimental role changed")
    resume = json.loads((ROOT / "artifacts/ape-low-level-training-20261014/directional-planar-entry-v1/cardinal-resume-check.json").read_text())
    coverage = json.loads((ROOT / "artifacts/ape-low-level-training-20261014/directional-planar-entry-v1/cardinal-coverage.json").read_text())
    if (resume["trainer_sha256"] != sha(ROOT / FILES["trainer"])
            or coverage["trainer_sha256"] != resume["trainer_sha256"]
            or resume["loaded_iteration"] != 844
            or any(resume["strict_state_mismatches"].values())
            or coverage["forced_first_resample_zero_count"] != 128):
        raise ValueError("Cardinal local entry or full-state resume not source-matched")
    # Each family differs in reset seed and a named motor slice. They are
    # development, not an independent held-out score.
    cases = [
        (49101, "backward", -.25, 0., 12, 0),
        (49102, "backward", -.35, 0., 10, 10),
        (49103, "left", 0., .14, 12, 0),
        (49104, "left", 0., .20, 10, 10),
        (49105, "right", 0., -.14, 12, 0),
        (49106, "right", 0., -.20, 10, 10),
        (49107, "forward", .25, 0., 12, 0),
        (49108, "forward", .35, 0., 10, 10),
    ]
    method = {
        "schema": "robot-learning-lab.cove50-cardinal-v1.method.v1",
        "role": "direct_frame_experimental_basic_control_not_mechanical_qualification",
        "hypothesis": "A saved forward actor with command-visible cardinal travel and a same-episode zero command can acquire short backward and signed lateral motion without losing basic forward survival.",
        "rival_hypotheses": [
            "The forward gait prior lacks lateral foot placement and will preserve a standing shortcut.",
            "Tracking reward improves by body sway or fast fall rather than signed displacement and alternating touchdowns.",
            "Nominal movement works but motor delay or lower rail makes it fragile.",
            "Zero-command reward favors parking throughout the moving phase or overshooting before settling.",
        ],
        "source": {key: {"path": rel, "sha256": sha(ROOT / rel)} for key, rel in FILES.items()},
        "local_entry": {"coverage_report": "artifacts/ape-low-level-training-20261014/directional-planar-entry-v1/cardinal-coverage.json",
                        "coverage_sha256": sha(ROOT / "artifacts/ape-low-level-training-20261014/directional-planar-entry-v1/cardinal-coverage.json"),
                        "resume_report": "artifacts/ape-low-level-training-20261014/directional-planar-entry-v1/cardinal-resume-check.json",
                        "resume_sha256": sha(ROOT / "artifacts/ape-low-level-training-20261014/directional-planar-entry-v1/cardinal-resume-check.json")},
        "learner": {"algorithm": "MJLab/RSL-RL PPO continuation", "imitation": "none_valid_for_cardinal_commands_demo_free_warm_start",
                    "source_iteration": 844, "profile": "cardinal", "worlds": 1024,
                    "additional_updates": 40, "steps_per_world_per_update": 24,
                    "expected_environment_steps": 983040,
                    "per_world_training_time_s": 19.2, "nominal_eight_second_episodes_per_world": 2.4,
                    "save_interval_updates": 10, "ankle_target_margin_rad": .02,
                    "full_state_resume_required": True},
        "command_curriculum": {"movement_time_s": [3., 4.], "stop_time_s": [4., 5.],
                               "backward_probability": .30, "left_probability": .25,
                               "right_probability": .25, "forward_probability": .20,
                               "backward_x_m_s": [-.40, -.20], "left_y_m_s": [.12, .22],
                               "right_y_m_s": [-.22, -.12], "forward_x_m_s": [.20, .40],
                               "yaw_rad_s": [-.15, .15]},
        "variation": {"body_mass_scale": [.9, 1.1], "sole_grip_scale": [.65, 1.15],
                      "motor_effort_scale": [.9, 1.05], "pd_gain_scale": [.9, 1.1],
                      "actuator_delay_physics_steps": [0, 4],
                      "interpretation": "Source-bounded simulated sensitivity, not a measured hardware uncertainty interval."},
        "reward_and_shortcuts": {"movement": "Native commanded linear and yaw velocity tracking; independently require signed displacement and alternating touchdowns.",
                                 "stop": "Native stillness-at-zero weight 2; independently require stop-tail <=0.12m and late double-sole support >=0.8.",
                                 "effort": "Joint-torques-l2 weight -0.05; report actual rated-leg duty separately.",
                                 "negative_examples": "Park, fall, opposite-direction travel and overshoot are explicit postpilot trace checks; no full realized reward ranking is claimed at entry."},
        "cuda_entry": {"seed": 49097, "worlds": 1024, "duration_s": 8.,
                       "require_native_move_to_zero_and_finite_reset": True,
                       "measure_steps_per_second_and_gpu_memory": True,
                       "os_graphics_packages": {"libegl1": "1.7.0-1build1", "libopengl0": "1.7.0-1build1"},
                       "require_nonblank_decoded_exact_body_video_before_updates": True,
                       "one_disposable_optimizer_update": True},
        "development_cases": [
            {"seed": seed, "family": family, "velocity_x_m_s": x,
             "velocity_y_m_s": y, "motor_rail_v": rail, "command_delay_ms": delay,
             "initial_root_z_m": .231, "stop_at_s": 4., "duration_s": 8.}
            for seed, family, x, y, rail, delay in cases
        ],
        "physical_gate": {"full_duration_s": 8., "backward_signed_progress_m_min": .25,
                          "lateral_signed_progress_m_min": .15, "forward_signed_progress_m_min": .7,
                          "new_direction_alternating_touchdowns_min": 2,
                          "forward_alternating_touchdowns_min": 6,
                          "hand_force_n_max": 1., "stop_tail_m_max": .12,
                          "late_double_support_fraction_min": .8},
        "pilot_signal": "At least two of six new-direction baseline-failure rescues and no catastrophic forward fall/hand regression; otherwise retain the diagnostic and change skill/hypothesis.",
        "motor_measure": "Always report moving rated-leg duty and loaded sole motion; neither is a promotion gate for this basic-motion pilot.",
        "videos": "Render both arms for all eight development cases during native scored visits; source-match to reports and traces.",
        "sealed_exclusions": ["V3 49001-49024", "V147 independent reach score", "V158 held contact seeds"],
        "spend": {"gpu": "NVIDIA A40", "listed_rate_max_usd_h": .60,
                  "pod_wall_limit_s": 5400, "listed_rental_cap_usd": .90,
                  "provider_confirmed_stop_required": True,
                  "posted_bill_allocation": "unknown_until_provider_posts"},
        "claim_limit": "One short development PPO pilot and eight paired native Direct Frame cases; no independent promotion, motor or selected-body transfer claim.",
        "status": "method_frozen_before_any_cardinal_cuda_or_optimizer_visit",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "method.json"
    encoded = (json.dumps(method, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("Frozen method differs; use another version")
    path.write_bytes(encoded)
    print(json.dumps({"method": str(path.relative_to(ROOT)), "sha256": sha(path),
                      "worlds": 1024, "updates": 40, "steps": 983040,
                      "matched_families": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
