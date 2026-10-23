"""Apply the predeclared fresh native gate to the hip-roll pose-prior branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-recovery-20261009"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "hiproll-plan.json").read_text())
    for name, source in plan["sources"].items():
        if name == "training_source":
            # The training source is pinned to its pre-run bytes. Later analysis
            # files do not change it; this check catches accidental edits.
            pass
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise ValueError(f"Predeclared source changed: {name}")
    source = json.loads((OUT / "hiproll-source-matrix.json").read_text())
    candidate = json.loads((OUT / "hiproll100-matrix.json").read_text())
    source_checkpoint = ROOT / plan["sources"]["checkpoint"]["path"]
    new_checkpoint = OUT / "hiproll100-model943.pt"
    if source["checkpoint_sha256"] != sha(source_checkpoint) or candidate["checkpoint_sha256"] != sha(new_checkpoint):
        raise ValueError("Paired matrix checkpoint identity changed")
    rows = []
    if len(source["cases"]) != 2 or len(candidate["cases"]) != 2:
        raise ValueError("Expected two reserved nominal cases per actor")
    for old, new in zip(source["cases"], candidate["cases"]):
        if (old["case"] != new["case"] or old["seed"] != new["seed"] or
            old["body_sha256"] != new["body_sha256"] or
            old["process_exit_code"] or new["process_exit_code"]):
            raise ValueError("Native cases are not paired")
        rows.append({
            "case": old["case"], "seed": old["seed"],
            "source_gait": old["one_meter_alternating"],
            "candidate_gait": new["one_meter_alternating"],
            "source_abs_drift_m": old["mean_abs_lateral_m"],
            "candidate_abs_drift_m": new["mean_abs_lateral_m"],
            "source_moving_rated_leg_fraction": old["mean_moving_leg_saturated_fraction"],
            "candidate_moving_rated_leg_fraction": new["mean_moving_leg_saturated_fraction"],
            "source_max_hand_force_n": old["max_hand_force_n"],
            "candidate_max_hand_force_n": new["max_hand_force_n"],
            "source_survived": old["survived_8s"],
            "candidate_survived": new["survived_8s"],
            "source_report": old["report"], "candidate_report": new["report"],
        })
    target = plan["nominal_gate"]
    checks = {
        "eight_of_eight_gait": sum(x["candidate_gait"] for x in rows) == 8,
        "abs_drift_each_seed_le_0_35m": all(
            x["candidate_abs_drift_m"] <= target["mean_absolute_lateral_drift_m_each_seed_max"]
            for x in rows),
        "rated_leg_each_seed_le_70pct": all(
            x["candidate_moving_rated_leg_fraction"] <= target["moving_any_rated_leg_fraction_each_seed_max"]
            for x in rows),
        "all_candidate_worlds_survive": all(x["candidate_survived"] == 4 for x in rows),
        "no_candidate_hand_contact": all(x["candidate_max_hand_force_n"] < 1.0 for x in rows),
    }
    passed = all(checks.values())
    comparison = {
        "schema": "robot-learning-lab.cove50-hiproll-pose-matrix.v1",
        "note": "Fresh paired native four-world nominal replays of the same selected Cove body at reserved seeds. Candidate uses only a wider walking hip-roll pose scale. Stop and heavy stress follow only if the predeclared nominal gate passes.",
        "cases": rows,
    }
    report = {
        "schema": "robot-learning-lab.cove50-hiproll-pose-trial.v1",
        "selected_design": plan["selected_design"],
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "selected_body_sha256": sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
        "source_checkpoint_sha256": sha(source_checkpoint),
        "candidate_checkpoint_sha256": sha(new_checkpoint),
        "plan": str((OUT / "hiproll-plan.json").relative_to(ROOT)),
        "changed_factor": plan["single_changed_factor"],
        "training": {"checkpoint": str(new_checkpoint.relative_to(ROOT)),
                     "log": str((OUT / "hiproll100-train.log").relative_to(ROOT)),
                     "env": str((OUT / "hiproll100-env.yaml").relative_to(ROOT)),
                     "agent": str((OUT / "hiproll100-agent.yaml").relative_to(ROOT))},
        "evaluation": {"native_processes": 4, "worlds": 16,
                       "paired_seed_offset": 9000,
                       "matrix": str((OUT / "hiproll-comparison-matrix.json").relative_to(ROOT)),
                       "source_gait": sum(x["source_gait"] for x in rows),
                       "candidate_gait": sum(x["candidate_gait"] for x in rows),
                       "rejected_actor_clip": str((OUT / "hiproll100-seed9144.mp4").relative_to(ROOT))},
        "phase_diagnostic": {
            "report": str((OUT / "hiproll-phase-report.json").relative_to(ROOT)),
            "matrix": str((OUT / "hiproll-phase-matrix.json").relative_to(ROOT)),
            "native_trace_processes": 2,
            "note": "Separate same-seed four-world native traces after nominal rejection explain the geometry and joint-duty response; the clip renders one world."
        },
        "nominal_gate": {"checks": checks, "passed": passed},
        "decision": ("Nominal gate passed; proceed to fresh stop and heavy low-grip 10 V/10 ms cases before any promotion."
                     if passed else
                     "Reject after paired nominal gate. Keep the prior saved actor; do not run stop/heavy stress or promote this branch."),
        "limits": [
            "The selected motor, sole and mass model remains provisional simulation sensitivity, not physical qualification.",
            "Idle prerequisite T-6g3vnp awaits independent review; this is exploratory and unpromoted.",
        ],
    }
    (OUT / "hiproll-comparison-matrix.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (OUT / "hiproll-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": checks, "passed": passed, "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
