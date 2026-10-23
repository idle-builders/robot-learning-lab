"""Summarize rejected direct-actor pilots beside the retained feedback reach."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-cove50-side-reach-20261009"
REFERENCE = BASE / "reach-feedback-v1/evaluation/held-diagonal-a-feedback.npz"
OUT = BASE / "reach-feedback-dagger-v1/result.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def case(name: str) -> dict:
    root = BASE / name
    training = root / "report.json"
    evaluation = root / "evaluation/matrix.json"
    trace = root / "evaluation/held-toy-diagonal-a.npz"
    train_report = read(training)
    eval_report = read(evaluation)
    with np.load(REFERENCE) as reference, np.load(trace) as candidate:
        teacher_action = reference["executed_action"]
        actor_action = candidate["executed_action"]
        first_obs_rms = float(np.sqrt(np.mean((reference["actor_observation"][0] -
                                              candidate["actor_observation"][0]) ** 2)))
        action_mae = {str(n): round(float(np.mean(np.abs(
            teacher_action[:n] - actor_action[:n]))), 5)
            for n in (1, 3, 5, 10) if len(actor_action) >= n}
    row = eval_report["cases"][0]
    return {
        "checkpoint": train_report["checkpoint"],
        "checkpoint_sha256": train_report["checkpoint_sha256"],
        "training_report": str(training.relative_to(ROOT)),
        "training_report_sha256": sha(training),
        "method": train_report["method"],
        "train_samples": train_report["train_samples"],
        "validation_samples": train_report["validation_samples"],
        "best_validation_mse": train_report["best_validation_mse"],
        "evaluation_matrix": str(evaluation.relative_to(ROOT)),
        "evaluation_matrix_sha256": sha(evaluation),
        "first_reserved_case": row,
        "initial_observation_rms_difference": round(first_obs_rms, 7),
        "mean_absolute_action_difference_vs_retained_feedback_first_n_frames": action_mae,
        "failure_video": str((root / "evaluation/held-toy-diagonal-a.mp4").relative_to(ROOT)),
        "failure_video_sha256": sha(root / "evaluation/held-toy-diagonal-a.mp4"),
    }


def main() -> None:
    collection_path = BASE / "reach-feedback-dagger-v1/collection/matrix.json"
    collection = read(collection_path)
    rows = collection["cases"]
    if len(rows) != 10:
        raise ValueError("Expected all ten mixed collection cases")
    report = {
        "schema": "robot-learning-lab.cove50-reach-feedback-dagger-result.v1",
        "status": "both_direct_actors_rejected_after_first_reserved_native_gate",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "selected_nominal_body_sha256": sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
        "retained_feedback_controller": "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt",
        "retained_feedback_controller_sha256": sha(BASE / "reach-feedback-v1/controller.pt"),
        "first_clone": case("reach-feedback-clone-v1"),
        "corrective_clone": case("reach-feedback-dagger-v1"),
        "mixed_collection": {
            "matrix": str(collection_path.relative_to(ROOT)),
            "matrix_sha256": sha(collection_path),
            "cases": len(rows),
            "train_cases": sum(row["split"] == "train" for row in rows),
            "validation_cases": sum(row["split"] == "validation" for row in rows),
            "survived_cases": sum(row["survived"] for row in rows),
            "teacher_execution_probability": collection["teacher_execution_probability"],
            "note": "Teacher interventions mean these are corrective data, not autonomous skill successes.",
        },
        "interpretation": "A low supervised fit and two development feedback traces did not stabilize an all-axis direct actor. Adding labels from mixed rollouts reduced some early action error but the reserved policy still fell sooner. Retain the hybrid feedback controller; do not promote either clone.",
        "limits": [
            "Both direct checkpoints were stopped at the predeclared first reserved case; broader reserved matrices were intentionally not run after that failure.",
            "The feedback controller still uses an authored phase path, explicit PI and simulator-truth target relative observation.",
            "Selected body, motor and contact properties are simulation assumptions; no physical qualification is inferred.",
        ],
        "remote_runtime": "artifacts/ape-cove50-side-reach-20261009/reach-feedback-dagger-v1/runtime.json",
        "remote_pod_removed": True,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "collection_survived": report["mixed_collection"]["survived_cases"],
                      "first_clone": report["first_clone"]["first_reserved_case"],
                      "corrective_clone": report["corrective_clone"]["first_reserved_case"]}, indent=2))


if __name__ == "__main__":
    main()
