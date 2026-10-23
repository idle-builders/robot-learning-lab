"""Rank the proposed signed reward on saved V2 native trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2/candidate"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/cardinal-v3/reward-audit.json"
CASES = ((49201, "backward", (-.25, 0.)),
         (49203, "left", (0., .14)),
         (49205, "right", (0., -.14)),
         (49207, "forward", (.25, 0.)))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    for seed, family, xy in CASES:
        command = np.asarray(xy)
        for arm in ("baseline", "v1", "candidate"):
            trace = SOURCE / f"seed{seed}-{arm}.npz"
            report = SOURCE / f"seed{seed}-{arm}.json"
            with np.load(trace, allow_pickle=False) as data:
                world_velocity = data["root_velocity"][:, 0, :2]
                wxyz = data["root_pose"][:, 0, 3:7]
                yaw = Rotation.from_quat(np.column_stack((wxyz[:, 1:], wxyz[:, 0]))).as_euler("xyz")[:, 2]
                velocity = np.column_stack((
                    np.cos(yaw) * world_velocity[:, 0] + np.sin(yaw) * world_velocity[:, 1],
                    -np.sin(yaw) * world_velocity[:, 0] + np.cos(yaw) * world_velocity[:, 1],
                ))
            # Installed MJLab tracking is exp(-||error||^2 / 0.1) * 2.
            gaussian = 2 * np.exp(-np.sum((velocity - command) ** 2, axis=1) / .1)
            signed = 2 * np.clip((velocity @ command) / (command @ command), -1.25, 1.25)
            world = json.loads(report.read_text())["worlds"][0]
            rows.append({"seed": seed, "family": family, "arm": arm,
                         "gaussian_mean": float(gaussian.mean()),
                         "signed_new_mean": float(signed.mean()),
                         "combined_mean": float((gaussian + signed).mean()),
                         "net_signed_progress_m": world["command_progress_m"],
                         "alternating_touchdowns": world["alternating_touchdowns"],
                         "report_sha256": sha(report), "trace_sha256": sha(trace)})
    counterfactual = []
    for _, family, xy in CASES:
        command = np.asarray(xy)
        parked = 2 * np.exp(-np.sum(command ** 2) / .1)
        counterfactual.append({"family": family, "command_m_s": list(xy),
                               "parked_gaussian": parked,
                               "parked_new": 0.,
                               "exact_tracking_gaussian_plus_new": 4.})
    result = {"schema": "robot-learning-lab.cardinal-v3.reward-audit.v1",
              "source": "V2 saved native 4s reports and NPZ traces; rewards reconstructed only for tracking and new signed term",
              "weight_new": 2., "rows": rows, "counterfactual": counterfactual,
              "limitation": "Pose, fall, torque, foot and other reward terms are not reconstructible from these traces; first CUDA sentinel must inspect resolved full reward. Net travel and footfalls remain independent outcome gates."}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "parked_vs_exact": counterfactual}, indent=2))


if __name__ == "__main__":
    main()
