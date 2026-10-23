"""Shadow-label opened calibrated RGB actor visits with the frozen PI teacher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/calibrated-fresh-reserved"
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v8/labels"
ARRAYS = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/controller-arrays.npz"
ACTOR = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt"
REFERENCE = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
JOINTS = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
BOUND = .1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(step: int) -> float:
    x = np.clip(step * .02 / 4, 0, 1)
    return float(x * x * (3 - 2 * x))


def main() -> None:
    with np.load(ARRAYS) as data:
        weight = data["pi_weight"].astype(np.float64)
        trajectory = data["pi_trajectory"].astype(np.float64)
        reference = data["v5_reference"].astype(np.float64)
    with np.load(REFERENCE) as data:
        names = data["action_names"].tolist()
    indices = [names.index(name) for name in JOINTS]
    actor_sha = sha(ACTOR)
    matrix = json.loads((SOURCE / "matrix.json").read_text())
    if matrix["case_count"] != 20:
        raise ValueError("Expected the frozen 20 opened RGB episodes")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in matrix["cases"]:
        source = ROOT / row["result"]
        report = json.loads(source.read_text())
        if (report["reach_checkpoint_sha256"] != actor_sha or
                report["reach_controller"] != "learned-residual"):
            raise ValueError(f"Wrong source actor: {source}")
        reach = [item for item in report["timeseries"] if item["phase"] == "reach"]
        if len(reach) != 450:
            raise ValueError(f"Expected 450 reachable actor frames: {source}")
        error = np.asarray([item["controller_target_error_xy_m"] for item in reach], dtype=np.float64)
        accumulated = np.asarray([item["learned_error_integral"] for item in reach], dtype=np.float64)
        actor_correction = np.asarray([item["learned_arm_residual_raw"] for item in reach], dtype=np.float64)
        current = error / .01
        entry = np.repeat(current[:1], len(reach), axis=0)
        previous = np.vstack((np.zeros((1, 4)), actor_correction[:-1])) / BOUND
        phase = np.asarray([smoothstep(step) for step in range(len(reach))])[:, None]
        x = np.column_stack((entry, current, accumulated, previous, phase))
        teacher_integral = np.zeros(21, dtype=np.float64)
        y = np.empty((len(reach), 4), dtype=np.float64)
        for step in range(len(reach)):
            index = min(step, len(reference) - 1)
            live_xy = reference[index] + error[step]
            direction = weight @ (live_xy - trajectory[min(step, len(trajectory) - 1)])
            teacher_integral = np.clip(teacher_integral + 1.5 * .02 * direction, -BOUND, BOUND)
            residual = np.clip(.6 * direction + teacher_integral, -BOUND, BOUND) * phase[step, 0]
            y[step] = residual[indices] / BOUND
        if x.shape != (450, 11) or y.shape != (450, 4) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid labels for {source}")
        out = OUT / f"seed-{row['seed']}.npz"
        np.savez_compressed(out, features=x.astype(np.float32), labels=y.astype(np.float32))
        cases.append({"seed": row["seed"], "actor_pass": row["pass"],
                      "actor_contact_s": row["contact_s"], "source": row["result"],
                      "source_sha256": sha(source), "dataset": str(out.relative_to(ROOT)),
                      "dataset_sha256": sha(out),
                      "mean_abs_actor_teacher_gap_raw": round(float(np.mean(np.abs(actor_correction - y * BOUND))), 5)})
    manifest = {"schema": "robot-learning-lab.cove50-calibrated-rgb-v8-shadow-labels.v1",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "actor_checkpoint_sha256": actor_sha, "controller_arrays_sha256": sha(ARRAYS),
                "method": "Frozen PI recurrence shadow-labeled on opened RGB actor visits; labels were never executed in these episodes.",
                "cases": cases}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"case_count": len(cases), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
