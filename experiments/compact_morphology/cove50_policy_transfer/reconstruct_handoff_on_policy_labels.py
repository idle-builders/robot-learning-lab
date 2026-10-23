"""Label v6 actor-visited handoff states with the frozen PI teacher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
OUT = ARTIFACT / "v7/on-policy-labels"
ARRAYS = ARTIFACT / "controller-arrays.npz"
JOINTS = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
BOUND = .1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(fraction: float) -> float:
    x = np.clip(fraction, 0, 1)
    return float(x * x * (3 - 2 * x))


def main() -> None:
    with np.load(ARRAYS) as data:
        weight = data["pi_weight"].astype(np.float64)
        trajectory = data["pi_trajectory"].astype(np.float64)
        reference = data["v5_reference"].astype(np.float64)
    from_ref = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
    with np.load(from_ref) as data:
        names = data["action_names"].tolist()
    indices = [names.index(name) for name in JOINTS]
    actor_sha = sha(ARTIFACT / "v6/actor.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for split in ("reference", "development", "reserved"):
        matrix = json.loads((ARTIFACT / f"v6/{split}/matrix.json").read_text())
        for row in matrix["cases"]:
            source = ROOT / row["result"]
            report = json.loads(source.read_text())
            if report["reach_checkpoint_sha256"] != actor_sha or report["reach_controller"] != "learned-residual":
                raise ValueError(f"Wrong source actor: {source}")
            reach = [item for item in report["timeseries"] if item["phase"] == "reach"]
            if len(reach) != 400:
                raise ValueError(f"Expected full actor-visited reach: {source}")
            error = np.asarray([item["controller_target_error_xy_m"] for item in reach], dtype=np.float64)
            accumulated = np.asarray([item["learned_error_integral"] for item in reach], dtype=np.float64)
            actor_correction = np.asarray([item["learned_arm_residual_raw"] for item in reach], dtype=np.float64)
            current = error / .01
            entry = np.repeat(current[:1], 400, axis=0)
            previous = np.vstack((np.zeros((1, 4)), actor_correction[:-1])) / BOUND
            phase = np.asarray([smoothstep(step * .02 / 4) for step in range(400)])[:, None]
            x = np.column_stack((entry, current, accumulated, previous, phase))
            teacher_integral = np.zeros(21, dtype=np.float64)
            y = np.empty((400, 4), dtype=np.float64)
            for step in range(400):
                live_xy = reference[step] + error[step]
                direction = weight @ (live_xy - trajectory[step])
                teacher_integral = np.clip(teacher_integral + 1.5 * .02 * direction,
                                           -BOUND, BOUND)
                residual = np.clip(.6 * direction + teacher_integral,
                                   -BOUND, BOUND) * phase[step, 0]
                y[step] = residual[indices] / BOUND
            if x.shape != (400, 11) or y.shape != (400, 4) or not np.isfinite(x).all():
                raise ValueError(f"Invalid on-policy features: {source}")
            out = OUT / f"{split}-{row['case']}.npz"
            np.savez_compressed(out, features=x.astype(np.float32), labels=y.astype(np.float32),
                                actor_correction=actor_correction.astype(np.float32),
                                shadow_pi_correction=(y * BOUND).astype(np.float32))
            case = {"case": row["case"], "split": split, "actor_pass": row["pass"],
                    "actor_contact_s": row["contact_s"],
                    "mean_abs_actor_teacher_gap_raw": round(float(np.mean(np.abs(actor_correction - y * BOUND))), 5),
                    "max_abs_actor_teacher_gap_raw": round(float(np.max(np.abs(actor_correction - y * BOUND))), 5),
                    "source": row["result"], "source_sha256": sha(source),
                    "dataset": str(out.relative_to(ROOT)), "dataset_sha256": sha(out)}
            cases.append(case)
    manifest = {"schema": "robot-learning-lab.cove50-handoff-v7-on-policy-labels.v1",
                "actor_checkpoint_sha256": actor_sha, "controller_arrays_sha256": sha(ARRAYS),
                "method": "Frozen PI recurrence on v6 actor-visited root-local errors, paired with recorded v6 observation features and previous executed residual; shadow labels were not executed in these native episodes.",
                "cases": cases}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
