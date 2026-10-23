"""Fit target sensitivity from varied native teachers for a bounded PI reach pilot.

The saved controller retains the authored phase base. Its fitted component maps
live hand-to-toy XY error to raw arm action; the PI update is explicit control
logic, not a policy learned end to end from off-path teacher states.
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


DEFAULT_OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--proportional-gain", type=float, default=0.6)
    parser.add_argument("--integral-gain-per-s", type=float, default=1.5)
    args = parser.parse_args()
    if args.proportional_gain < 0 or args.integral_gain_per_s < 0:
        raise ValueError("PI gains must be nonnegative")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [row for row in manifest["cases"]
            if row["split"] == "train" and row["accepted_teacher"]]
    center_row = next(row for row in rows if row["case"] == "train-center-nominal12")
    with np.load(ROOT / center_row["dataset"]) as data:
        center_action = data["teacher_action"].astype(np.float64)
        center_xy = data["actor_observation"][:, 42:44].astype(np.float64)
        center_phase = data["smoothstep_phase"].astype(np.float64)
    samples = {"train": [], "validation": []}
    for row in rows:
        with np.load(ROOT / row["dataset"]) as data:
            action = data["teacher_action"].astype(np.float64)
            phase = data["smoothstep_phase"].astype(np.float64)
            observed_xy = data["actor_observation"][:, 42:44].astype(np.float64)
            if action.shape != center_action.shape or not np.allclose(phase, center_phase):
                raise ValueError(f"Mismatched teacher timing: {row['case']}")
            # The demonstrated arm correction is constant at the joint target,
            # and enters the executed raw action through the smoothstep phase.
            # The goal shift is measured at entry before either arm moves.
            feature = (observed_xy[0] - center_xy[0])[None, :] * phase[:, None]
            samples["validation" if row["case"] in VALIDATION else "train"].append(
                (feature, action - center_action))
    x_train = np.concatenate([x for x, _ in samples["train"]])
    y_train = np.concatenate([y for _, y in samples["train"]])
    x_val = np.concatenate([x for x, _ in samples["validation"]])
    y_val = np.concatenate([y for _, y in samples["validation"]])
    weights, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "controller.pt"
    torch.save({
        "method": "bounded_observed_goal_pi_residual_on_native_teacher",
        "weight": torch.from_numpy(weights.T.astype(np.float32)),
        "center_ee_to_cube_xy_trajectory_m": torch.from_numpy(center_xy.astype(np.float32)),
        "residual_action_bound": 0.1,
        "proportional_gain": args.proportional_gain,
        "integral_gain_per_s": args.integral_gain_per_s,
        "reference_trace_sha256": sha(ROOT / center_row["dataset"]),
    }, checkpoint)
    report = {
        "schema": "robot-learning-lab.cove50-reach-feedback-pilot.v1",
        "method": "least_squares_target_sensitivity_plus_explicit_PI_feedback",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha(checkpoint),
        "teacher_manifest_sha256": sha(manifest_path),
        "train_cases": [row["case"] for row in rows if row["case"] not in VALIDATION],
        "validation_cases": [row["case"] for row in rows if row["case"] in VALIDATION],
        "train_samples": len(x_train), "validation_samples": len(x_val),
        "train_mse": float(np.mean((x_train @ weights - y_train) ** 2)),
        "validation_mse": float(np.mean((x_val @ weights - y_val) ** 2)),
        "input": "live MJLab ee_to_cube XY in metres, nominal trajectory and skill phase",
        "output": "bounded 21D raw-action residual added to authored native phase trajectory",
        "proportional_gain": args.proportional_gain,
        "integral_gain_per_s": args.integral_gain_per_s,
        "qualification": "Offline sensitivity fit only; PI correction and contact outcome require native closed-loop replay. Target pose is simulator truth, not sensed perception.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
