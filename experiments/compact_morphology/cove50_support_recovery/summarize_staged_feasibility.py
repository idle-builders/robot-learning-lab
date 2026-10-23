"""Summarize bounded Direct Frame support-stage feasibility and reward contrast."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261013/support-staged-feasibility"
EVALUATOR = Path(__file__).with_name("evaluate_support_cycle.py")
TRAINER = Path(__file__).with_name("train_support_cycle.py")
NAMES = (
    "load-right-8201", "load-left-8201",
    "load-grip065-right-8201", "load-grip065-left-8201",
    "load-grip115-right-8201", "load-grip115-left-8201",
    "load-grip115-shift034-right-8201", "load-grip115-shift034-left-8201",
    "zero-right-8201", "zero-left-8201",
    "release-pitch015-shift032-right-8201",
    "release-pitch030-shift032-right-8201",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def staged_load_reward(rows: list[dict], weight: float = 2.0,
                       dt: float = 0.02) -> dict:
    # Mirrors train_support_cycle.commanded_two_foot_load for saved rows.
    raw = [int(r["command_flag"] > .5 and r["feet_grounded"] == [True, True]
               and r["stance_load_fraction"] >= .75 and r["pelvis_z_m"] >= .20
               and r["tilt_deg"] <= 15.0 and not r["nonsole_floor_contact"]
               and not r["self_contact"] and r["hand_force_n"] < 1.0)
           for r in rows]
    rated_frames = sum(bool(r["rated_leg_joints"]) for r in rows)
    load_contribution = round(sum(raw) * weight * dt, 3)
    rated_contribution = round(-rated_frames * dt, 3)
    return {
        "raw_positive_frames": sum(raw), "raw_term_sum": sum(raw),
        "raw_positive_s": round(sum(raw) * dt, 3),
        "weight": weight, "dt_scale": dt,
        "weighted_dt_scaled_episode_contribution": load_contribution,
        "rated_leg_raw_frames": rated_frames,
        "rated_leg_penalty_weight": -1.0,
        "rated_leg_weighted_dt_scaled_contribution": rated_contribution,
        "two_term_dt_scaled_contribution": round(load_contribution + rated_contribution, 3),
    }


def main() -> None:
    evaluator_hash = sha(EVALUATOR)
    cases = []
    for name in NAMES:
        report_path = BASE / f"{name}.json"
        trace_path = BASE / f"{name}.npz"
        d = json.loads(report_path.read_text())
        with np.load(trace_path) as trace:
            shape = {key: list(trace[key].shape) for key in
                     ("qpos", "qvel", "actor_observation", "action", "joint_target")}
        if d["evaluator_sha256"] != evaluator_hash or d["trace_sha256"] != sha(trace_path):
            raise ValueError(f"Source or trace mismatch: {name}")
        if shape["actor_observation"][0] != len(d["timeseries"]) or \
           shape["actor_observation"][1] != 72 or shape["action"][1] != 21:
            raise ValueError(f"Incomplete actor/action labels: {name}")
        case = {
            "id": name, "stance": d["stance"], "controller": d["controller"],
            "teacher_profile": d["authored_teacher_profile"],
            "grip": d["grip"], "shift_roll_rad": d["teacher_shift_roll_rad"],
            "initial_state_sha256": d["initial_state_sha256"],
            "duration_s": d["duration_s"],
            "load_transfer_gate_pass": d["load_transfer_gate"]["passed"],
            "min_3s_stance_load_fraction": d["load_transfer_gate"]["min_stance_load_fraction"],
            "full_single_support_gate_pass": d["full_gate_pass"],
            "longest_single_support_s": d["longest_single_support_s"],
            "max_swing_sole_center_m": max(r["swing_sole_z_m"] for r in d["timeseries"]),
            "final_two_foot_stand": d["final_two_foot_stand"],
            "max_stance_patch_slip_m": d["max_stance_patch_slip_m"],
            "rated_leg_fraction": d["rated_leg_fraction"],
            "rated_arm_fraction": d["rated_arm_fraction"],
            "bad_contact_frames": d["bad_contact_frames"],
            "candidate_load_reward": staged_load_reward(d["timeseries"]),
            "trace_shapes": shape,
            "report": str(report_path.relative_to(ROOT)), "report_sha256": sha(report_path),
            "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": sha(trace_path),
        }
        if d.get("video"):
            video = Path(d["video"])
            case.update(video=str(video), video_sha256=sha(video))
        cases.append(case)
    load_cases = [c for c in cases if c["teacher_profile"] == "load_transfer"]
    passed = [c for c in load_cases if c["load_transfer_gate_pass"]]
    failed = [c for c in load_cases if not c["load_transfer_gate_pass"]]
    controls = [c for c in cases if c["controller"] == "zero_action"]
    if not (len(passed) == 6 and len(failed) == len(controls) == 2 and
            all(not c["load_transfer_gate_pass"] for c in controls) and
            all(c["candidate_load_reward"]["weighted_dt_scaled_episode_contribution"] > 0
                for c in passed) and
            all(c["candidate_load_reward"]["weighted_dt_scaled_episode_contribution"] == 0
                for c in controls) and
            min(c["candidate_load_reward"]["two_term_dt_scaled_contribution"]
                for c in passed) >
            max(c["candidate_load_reward"]["two_term_dt_scaled_contribution"]
                for c in failed + controls)):
        raise ValueError("Stage-A teacher/negative-control ordering changed")
    teacher_ambiguity = {}
    for side in ("right", "left"):
        with np.load(BASE / f"load-{side}-8201.npz") as nominal, \
             np.load(BASE / f"load-grip115-shift034-{side}-8201.npz") as high:
            teacher_ambiguity[side] = {
                "initial_actor_observation_max_abs_difference": float(np.max(np.abs(
                    nominal["actor_observation"][0] - high["actor_observation"][0]))),
                "at_2s_actor_observation_max_abs_difference": float(np.max(np.abs(
                    nominal["actor_observation"][99] - high["actor_observation"][99]))),
                "at_2s_action_max_abs_difference": float(np.max(np.abs(
                    nominal["action"][99] - high["action"][99]))),
            }
    report = {
        "schema": "robot-learning-lab.cove50-support-staged-feasibility.v1",
        "classification": "Local native MJLab CPU authored-teacher feasibility and reward-term audit; no learned policy or CUDA skill score",
        "body_sha256": json.loads((BASE / "load-right-8201.json").read_text())["body_sha256"],
        "design_pointer_sha256": json.loads((BASE / "load-right-8201.json").read_text())["design_pointer_sha256"],
        "evaluator_sha256": evaluator_hash, "trainer_sha256": sha(TRAINER),
        "summary_script_sha256": sha(Path(__file__)),
        "trainer_local_preflight_log": str((BASE / "trainer-load-preflight.log").relative_to(ROOT)),
        "trainer_local_preflight_log_sha256": sha(BASE / "trainer-load-preflight.log"),
        "stage_a": {
            "gate": "From upright two-foot reset: 2–5 s both soles grounded and requested stance >=75% measured foot-contact-force norm at every 50 Hz frame; root >=0.20 m, tilt <=15 deg, <=15 mm four-corner stance patch slip, <=10% rated leg and arm frames, no hand/nonsole/self contact, 8 s complete and two-foot return.",
            "accepted_whole_teacher_episodes": 6,
            "accepted_sides": ["left", "right"], "independent_reset_families": 1,
            "contact_grip_settings": [0.65, 1.0, 1.15],
            "accepted_contact_grip_settings": [0.65, 1.0, 1.15],
            "failed_whole_teacher_episodes": 2,
            "adapted_high_grip_shift_rad": 0.34,
            "varied_teacher_bank": "Only a bounded three-point contact-friction grid; reset, mass, rail and lag remain unvaried",
            "teacher_labels": "400 native actor-observation/action pairs per accepted episode, 2,400 total; failed whole episodes excluded. Grip-specific teacher variants must not be blindly pooled for BC.",
            "teacher_condition_ambiguity": teacher_ambiguity,
            "imitation_rule": "The actor has no explicit grip input and the selected variants begin with identical actor observations, then use different hip-roll labels. Choose one common robust action or train separate condition-specific pilots with observed feedback; only pool after a measured closed-loop conflict check.",
            "zero_action_negative_control_gate_passes": 0,
            "proposed_reward_term": "Requested two-foot load >=75% with upright height/tilt and no bad contact; 2.0 weight, dt scaled at 0.02 s",
            "reward_negative_control": "The load term alone gives 6.48 to high-grip failed episodes versus 6.40 to adapted passes. The -1.0 rated-leg-frame term ranks all six Stage-A passes above the two failed high-grip episodes and two zero controls in this development set.",
            "reward_comparison": "Two-term offline replay of actual native traces, tuned on this development grid. Full PPO reward, critic fit and independent reward ranking are not measured.",
        },
        "stage_b": {
            "full_gate_passes": 0,
            "bounded_contrast": "With 0.32 rad bilateral shift, 0.15 rad swing hip-pitch release finishes but remains below 25 mm clearance; 0.30 rad release clears 25 mm transiently but falls, rates leg motors and slips far beyond 15 mm.",
            "accepted_teacher_episodes": 0,
            "next_discriminator": "Test a feedback or coupled hip/knee/ankle release from attained Stage-A states under the same physical gates; do not distill current failures or fit PPO on Stage B.",
        },
        "hypotheses": [
            {"factor": "reset and phase exposure", "evidence": "Fixed 0–1 stand, 1–5 request, 5–8 return exists in trainer; Stage-A native target completes on both sides", "next": "Verify command and 8 s horizon at sampled training resets before any fit"},
            {"factor": "two-foot load transfer feasibility", "evidence": "Stage-A 6/8 whole teacher passes across grip0.65/1.0/1.15 and both sides; high-grip 0.36 rad fails but 0.34 rad passes; zero control 0/2", "next": "Collect independent reset, mass, rail and lag variation; retain only whole physical-gate passes"},
            {"factor": "motor-constrained release", "evidence": "Smaller release stays upright below clearance; larger release reaches clearance but falls with rated effort and patch slip", "next": "Try coupled feedback release; require complete two-sided no-reset return"},
            {"factor": "reward shortcut", "evidence": "Load-only term fails on high grip: 6.48 failed versus 6.40 passing. Explicit rated-leg penalty repairs this small development-set ordering; legacy full-cycle reward rose while 0/4 physical passes", "next": "Use independent varied episodes to challenge both term ranking and full reward; inspect critic diagnostics before PPO"},
            {"factor": "imitation label ambiguity", "evidence": "Nominal and adapted high-grip teachers start with identical actor observations, while their later hip-roll labels differ under a hidden grip setting", "next": "Measure conditional BC conflict and choose a common robust action or observed-feedback policy before pooling demonstrations"},
        ],
        "cases": cases,
        "claim_limit": "Stage A has one reset family and three grip points with two adapted high-grip successes, not an independently varied teacher bank or learned skill. Reward penalty was selected on this same development grid. Stage B has no accepted teacher. Local CPU physics cannot fill frozen CUDA policy scores.",
    }
    (BASE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage_a_accepted": 6, "stage_a_rejected": 2,
                      "zero_passes": 0,
                      "stage_b_full_passes": 0}, indent=2))


if __name__ == "__main__":
    main()
