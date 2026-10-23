"""Summarize V7 short native pitch action probes on already opened failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-stand-recovery/pitch-authority-v7"
BASELINE = ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v5"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_pitch_deg(qpos: np.ndarray) -> float:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, qpos[3:7])
    return float(np.degrees(np.arcsin(-matrix.reshape(3, 3)[2, 0])))


def sample_pitch(qpos: np.ndarray, seconds: float) -> float:
    return signed_pitch_deg(qpos[round(seconds / 0.02) - 1])


def main() -> None:
    plan_path = ARTIFACT / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert digest(ARTIFACT / "source/evaluate_posture.py") == plan["source_snapshot_sha256"]
    rows = []
    for seed in plan["opened_failed_seeds"]:
        baseline_report = json.loads((BASELINE / f"seed{seed}.json").read_text())
        with np.load(BASELINE / f"seed{seed}.npz") as arrays:
            baseline_qpos = arrays["qpos"].copy()
        baseline_trace = baseline_report["timeseries"][:60]
        baseline_rated = sum(sample["leg_torque_saturated"] for sample in baseline_trace)
        baseline_pitch = {str(t): sample_pitch(baseline_qpos, t) for t in (0.5, 0.8, 1.2)}
        for variant in plan["variants"]:
            joint = variant["joint"]
            bias = variant["bias_action"]
            name = f"seed{seed}-{joint}-{'minus' if bias < 0 else 'plus'}"
            source_path = ARTIFACT / f"{name}.json"
            native_path = ARTIFACT / f"{name}.npz"
            source = json.loads(source_path.read_text())
            with np.load(native_path) as arrays:
                qpos = arrays["qpos"].copy()
                raw_action = arrays["raw_action"].copy()
                applied_action = arrays["applied_action"].copy()
                targets = arrays["processed_target"].copy()
            trace = source["timeseries"]
            assert source["initial_state_sha256"] == baseline_report["initial_state_sha256"]
            assert source["first_action_sha256"] == baseline_report["first_action_sha256"]
            assert source["body_sha256"] == plan["body_sha256"]
            assert source["probe_joint"] == joint and source["probe_bias_action"] == bias
            assert len(trace) == len(qpos) == len(raw_action) == len(applied_action) == len(targets) == 60
            assert np.array_equal(qpos[:10], baseline_qpos[:10])
            pitch = {str(t): sample_pitch(qpos, t) for t in (0.5, 0.8, 1.2)}
            rated = sum(sample["leg_torque_saturated"] for sample in trace)
            rows.append({
                "seed": seed,
                "joint": joint,
                "bias_action": bias,
                "source_report": str(source_path.relative_to(ROOT)),
                "source_report_sha256": digest(source_path),
                "native_trajectory_sha256": digest(native_path),
                "same_initial_and_first_action_hash_as_v5": True,
                "first_0p2s_physical_qpos_exactly_same_as_v5": True,
                "baseline_pitch_deg": baseline_pitch,
                "probe_pitch_deg": pitch,
                "change_abs_pitch_at_0p8_deg": abs(pitch["0.8"]) - abs(baseline_pitch["0.8"]),
                "baseline_rated_leg_frames_first_1p2s": baseline_rated,
                "probe_rated_leg_frames_first_1p2s": rated,
                "probe_max_tilt_deg": max(sample["tilt_deg"] for sample in trace),
                "probe_floor_nonsole_body_contact_frames": sum(sample["floor_nonsole_body_contact"] for sample in trace),
            })
    same_variant_improves_both = []
    for variant in plan["variants"]:
        pair = [r for r in rows if r["joint"] == variant["joint"] and r["bias_action"] == variant["bias_action"]]
        if all(r["change_abs_pitch_at_0p8_deg"] < 0 and r["probe_rated_leg_frames_first_1p2s"] <= r["baseline_rated_leg_frames_first_1p2s"] for r in pair):
            same_variant_improves_both.append(variant)
    report = {
        "schema": "robot-learning-lab.cove50-pitch-authority-report.v7",
        "status": "completed_opened_failure_probe_no_shared_useful_small_action",
        "plan": str(plan_path.relative_to(ROOT)),
        "plan_sha256": digest(plan_path),
        "source_snapshot_sha256": digest(ARTIFACT / "source/evaluate_posture.py"),
        "scorer_source_sha256": digest(Path(__file__)),
        "runtime": "8 local CPU native MJLab 1.2 s probes; zero paid GPU time",
        "summary": {
            "opened_failed_seed_families": len(plan["opened_failed_seeds"]),
            "short_probes": len(rows),
            "same_variant_reduced_abs_pitch_at_0p8_on_both_without_more_rated_frames": same_variant_improves_both,
            "all_probes_still_floor_contact_by_1p2s": all(r["probe_floor_nonsole_body_contact_frames"] > 0 for r in rows),
        },
        "rows": rows,
        "decision": "No tested small constant hip or ankle bias gives the same useful direction on both failures without a motor-cost increase. This does not rule out timed feedback or coupled joint correction, but it does not justify a simple bias-based teacher. Probe starts are opened failures, not independent score.",
        "learned_policy_claim": False,
    }
    (ARTIFACT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
