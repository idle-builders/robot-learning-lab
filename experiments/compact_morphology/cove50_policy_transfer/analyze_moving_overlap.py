"""Compare the bounded moving-load-overlap branch with its saved Cove source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-recovery-20261009"
SOURCE = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
CANDIDATE = OUT / "overlap100-model943.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_metrics(prefix: str) -> dict:
    data = np.load(OUT / f"{prefix}-seed7144-trace.npz")
    report = json.loads((OUT / f"{prefix}-seed7144-trace.json").read_text())
    if report["body_sha256"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"):
        raise ValueError("Selected native body changed")
    active = data["active"]
    both = (data["sole_found"] > 0).all(axis=2) & active
    min_load = np.abs(data["sole_contact_force"][:, :, :, 2]).min(axis=2)
    speed = np.linalg.norm(data["root_velocity"][:, :, :2], axis=2)
    loaded_moving = both & (min_load >= 8.0) & (speed >= 0.12)
    rated = (np.abs(data["actuator_force"]) >=
             0.995 * data["rated_effort_limits_nm"]).any(axis=2) & active
    matched = [i for i, world in enumerate(json.loads(
        (OUT / "overlap-source-seed7144-trace.json").read_text())["worlds"])
        if world["ended_by_timeout"] and world["forward_m"] >= 1.0]

    def fraction(mask: np.ndarray, worlds: list[int]) -> float:
        return round(float(mask[:, worlds].sum() / active[:, worlds].sum()), 5)

    return {
        "native_trace": str((OUT / f"{prefix}-seed7144-trace.npz").relative_to(ROOT)),
        "native_trace_sha256": sha(OUT / f"{prefix}-seed7144-trace.npz"),
        "worlds": len(report["worlds"]),
        "source_success_worlds_used_for_paired_phase_comparison": matched,
        "all_active_frame_fraction": {
            "both_soles_detected": fraction(both, list(range(len(report["worlds"])))),
            "both_loaded_8n_and_moving": fraction(loaded_moving, list(range(len(report["worlds"])))),
            "any_rated_leg": fraction(rated, list(range(len(report["worlds"])))),
        },
        "matched_source_success_world_fraction": {
            "both_soles_detected": fraction(both, matched),
            "both_loaded_8n_and_moving": fraction(loaded_moving, matched),
            "any_rated_leg": fraction(rated, matched),
        },
    }


def main() -> None:
    plan = json.loads((OUT / "overlap-plan.json").read_text())
    source = json.loads((OUT / "overlap-source-matrix.json").read_text())
    candidate = json.loads((OUT / "overlap100-matrix.json").read_text())
    if (plan["selected_design"] != "cove-50-direct-frame" or
        plan["sources"]["checkpoint"]["sha256"] != sha(SOURCE) or
        source["checkpoint_sha256"] != sha(SOURCE) or
        candidate["checkpoint_sha256"] != sha(CANDIDATE) or
        len(source["cases"]) != 2 or len(candidate["cases"]) != 2):
        raise ValueError("Paired checkpoint or matrix identity changed")
    rows = []
    for old, new in zip(source["cases"], candidate["cases"]):
        if (old["case"] != new["case"] or old["seed"] != new["seed"] or
            old["body_sha256"] != new["body_sha256"] or
            old["process_exit_code"] or new["process_exit_code"]):
            raise ValueError("Nominal cases are not paired native results")
        rows.append({
            "case": old["case"], "seed": old["seed"],
            "source_gait": old["one_meter_alternating"],
            "candidate_gait": new["one_meter_alternating"],
            "source_abs_drift_m": old["mean_abs_lateral_m"],
            "candidate_abs_drift_m": new["mean_abs_lateral_m"],
            "source_rated_leg_fraction": old["mean_moving_leg_saturated_fraction"],
            "candidate_rated_leg_fraction": new["mean_moving_leg_saturated_fraction"],
            "source_max_hand_force_n": old["max_hand_force_n"],
            "candidate_max_hand_force_n": new["max_hand_force_n"],
            "source_report": old["report"], "candidate_report": new["report"],
            "report": str((OUT / "overlap100-matrix.json").relative_to(ROOT)),
        })
    traces = {"source": trace_metrics("overlap-source"),
              "candidate": trace_metrics("overlap100")}
    report = {
        "schema": "robot-learning-lab.cove50-moving-load-overlap-report.v1",
        "status": "rejected_after_paired_nominal_gate",
        "selected_design": "cove-50-direct-frame",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "selected_body_sha256": sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
        "source_checkpoint_sha256": sha(SOURCE),
        "candidate_checkpoint_sha256": sha(CANDIDATE),
        "analysis_source_sha256": sha(Path(__file__)),
        "plan": str((OUT / "overlap-plan.json").relative_to(ROOT)),
        "single_changed_factor": plan["single_changed_factor"],
        "training": {"worlds": 256, "updates": 100,
                     "run_name": "2026-10-09_16-38-30_overlap100",
                     "source_iteration": 844, "last_saved_iteration": 943,
                     "overlap_reward_first_last": [0.0012, 0.0697],
                     "log": str((OUT / "overlap100-train.log").relative_to(ROOT)),
                     "checkpoint": str(CANDIDATE.relative_to(ROOT))},
        "evaluation": {"native_processes": 4, "worlds": 16,
                       "paired_seed_offset": 7000,
                       "gait_source_total": sum(row["source_gait"] for row in rows),
                       "gait_candidate_total": sum(row["candidate_gait"] for row in rows),
                       "source_abs_drift_m": [row["source_abs_drift_m"] for row in rows],
                       "candidate_abs_drift_m": [row["candidate_abs_drift_m"] for row in rows],
                       "source_rated_leg_fraction": [row["source_rated_leg_fraction"] for row in rows],
                       "candidate_rated_leg_fraction": [row["candidate_rated_leg_fraction"] for row in rows],
                       "matrix": str((OUT / "overlap-comparison-matrix.json").relative_to(ROOT)),
                       "candidate_video": str((OUT / "overlap100-seed7144.mp4").relative_to(ROOT))},
        "phase_diagnostic": traces,
        "decision": "Reject the overlap-reward actor. It recovers one source failure on the reserved nominal seeds but more than doubles mean absolute lateral drift on both paired cases; rated-leg time remains about 91–92%. On matched worlds where the source already walked, genuinely loaded moving double support changes only 2.58% to 3.00% and rated-leg time changes 91.42% to 91.33%. The training reward increase did not create a meaningful load-sharing gait. Do not run the planned stop/stress promotion gate or replace the retained actor.",
        "limits": [
            "The source failure on seed 7144 makes all-active-frame phase averages non-comparable; the matched successful worlds 0–2 isolate the gait-phase effect.",
            "This is a provisional catalog-rated MuJoCo motor/contact sensitivity, not measured physical duty.",
            "No thermal/current or physical sole data are available, and this rejected candidate is not registry-qualified.",
            "Idle dependency T-6g3vnp remains verified awaiting independent review; this is an exploratory unpromoted training branch.",
        ],
    }
    matrix = {"schema": "robot-learning-lab.cove50-moving-load-overlap-matrix.v1",
              "note": "Fresh paired native nominal worlds on selected Cove-50. Source 7/8 gait; candidate 8/8, but drift worsens and rated-leg duty remains near continuous. The contact-phase diagnostic compares only source-success worlds 0–2.",
              "cases": rows}
    (OUT / "overlap-comparison-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (OUT / "overlap-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()
