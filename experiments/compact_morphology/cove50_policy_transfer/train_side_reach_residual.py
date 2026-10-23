"""Distill small target shifts into a bounded correction of the native teacher.

The saved linear actor uses the target displacement observed at skill entry and
skill phase. The authored joint-target trajectory remains its base controller;
the learned residual is limited to 0.1 raw action on each axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_bc import (
    DEFAULT_DATA, ROOT, VALIDATION,
)


DEFAULT_OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-residual-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    rows = [row for row in manifest["cases"] if row["split"] == "train" and row["accepted_teacher"]]
    with np.load(ROOT / next(row["dataset"] for row in rows if row["case"] == "train-center-nominal12")) as center:
        center_action = center["teacher_action"].astype(np.float64)
        center_target = center["actor_observation"][0, 42:44].astype(np.float64)
    splits = {"train": [], "validation": []}
    for row in rows:
        with np.load(ROOT / row["dataset"]) as data:
            action = data["teacher_action"].astype(np.float64)
            if action.shape != center_action.shape:
                raise ValueError("Teacher trajectory has a different action timing")
            target = data["actor_observation"][0, 42:44].astype(np.float64)
            phase = data["smoothstep_phase"].astype(np.float64)
            features = (target - center_target)[None, :] * phase[:, None]
            labels = action - center_action
            splits["validation" if row["case"] in VALIDATION else "train"].append((features, labels))
    x_train = np.concatenate([pair[0] for pair in splits["train"]])
    y_train = np.concatenate([pair[1] for pair in splits["train"]])
    x_val = np.concatenate([pair[0] for pair in splits["validation"]])
    y_val = np.concatenate([pair[1] for pair in splits["validation"]])
    # No bias: the exact center target must produce the exact known-feasible
    # native teacher action. This also prevents an action-history feedback loop.
    weights, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    train_mse = float(np.mean((x_train @ weights - y_train) ** 2))
    val_mse = float(np.mean((x_val @ weights - y_val) ** 2))
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "controller.pt"
    torch.save({"method": "bounded_goal_residual_on_native_teacher",
                "weight": torch.from_numpy(weights.T.astype(np.float32)),
                "center_ee_to_cube_xy_m": torch.from_numpy(center_target.astype(np.float32)),
                "residual_action_bound": 0.1,
                "center_teacher_action_sha256": sha(ROOT / next(row["dataset"] for row in rows
                                                            if row["case"] == "train-center-nominal12"))}, checkpoint)
    report = {"schema": "robot-learning-lab.cove50-reach-residual.v1",
              "method": "least_squares_behavior_cloning_of_teacher_target_corrections",
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha(checkpoint),
              "teacher_manifest_sha256": sha(args.data / "manifest.json"),
              "train_cases": [row["case"] for row in rows if row["case"] not in VALIDATION],
              "validation_cases": [row["case"] for row in rows if row["case"] in VALIDATION],
              "train_samples": len(x_train), "validation_samples": len(x_val),
              "train_mse": train_mse, "validation_mse": val_mse,
              "input": "initial ee-to-cube XY displacement relative to center, multiplied by skill phase; MJLab target is simulator truth",
              "output": "21D residual limited to 0.1 raw action, added to authored native joint-target teacher",
              "qualification": "Offline fit only. The full saved controller requires the authored phase trajectory and fresh native contact replay; no perception or locomotion handoff qualification."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
