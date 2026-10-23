"""Measure opened RGB occlusion against head yaw at fixed native entry poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.calibrated_virtual_rgb_goal import CalibratedVirtualRgbGoal

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v29"
YAW_GRID = (-1.5, -1.45, -1.4, -1.35, -1.3)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "opened-gaze-plan.json"
    plan = json.loads(plan_path.read_text())
    source_matrix = BASE / "v21/fresh-reserved/matrix.json"
    if plan["source_matrix_sha256"] != sha(source_matrix):
        raise ValueError("V21 source matrix changed")
    matrix = json.loads(source_matrix.read_text())
    by_seed = {row["seed"]: row for row in matrix["cases"]}
    body_sha = plan["source_sha256"]["body"]
    calibration_path = BASE / "rgb-calibration-model.npz"
    if sha(calibration_path) != plan["source_sha256"]["calibration"]:
        raise ValueError("Calibration source changed")
    source = CalibratedVirtualRgbGoal(calibration_model=calibration_path,
                                      selected_body_sha256=body_sha, max_age_s=9)
    head = source.model.joint("head_yaw").qposadr[0]
    rows = []
    for case in plan["cases"]:
        seed = case["seed"]
        original = by_seed[seed]
        with np.load(ROOT / original["trace"]) as trace:
            qpos = trace["qpos"][375].copy()
        if abs(float(qpos[head]) + 1.3) > 0.005:
            raise ValueError(f"Expected attained -1.3 rad baseline on {seed}")
        for yaw in YAW_GRID:
            candidate = qpos.copy()
            candidate[head] = yaw
            event, _ = source.capture(candidate, 7.5)
            estimate = event["estimated_goal_world_xyz_m"]
            error = (float(np.linalg.norm((candidate[28:30] - np.asarray(estimate)[:2]) * 1000))
                     if estimate is not None else None)
            rows.append({"seed": seed, "intended_stratum": case["intended_stratum"],
                         "head_yaw_rad": yaw, "orange_pixels": event["orange_pixels"],
                         "accepted": event["accepted"],
                         "centroid_margin_px": event["centroid_margin_px"],
                         "rgb_xy_error_mm_scoring_only": None if error is None else round(error, 4)})
    source.close()
    report = {"schema": "robot-learning-lab.cove50-v29-opened-static-gaze-scan.v1",
              "role": "static_opened_rgb_observability_diagnostic_not_native_action_score",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "source_sha256": sha(Path(__file__)),
              "v21_matrix_sha256": sha(source_matrix),
              "seed_count": len(plan["cases"]), "yaw_grid_rad": YAW_GRID,
              "rows": rows,
              "qualification": "Only head yaw changes at a frozen native qpos; scored truth is used for evaluation, never image inference. Native support and contact require separate replay."}
    (OUT / "opened-static-gaze-scan.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, seed in zip(axes, (10604, 10602, 10608)):
        cohort = [row for row in rows if row["seed"] == seed]
        yaw = [row["head_yaw_rad"] for row in cohort]
        ax.plot(yaw, [row["orange_pixels"] for row in cohort], marker="o", color="#dd8d3f")
        ax.set_xlabel("Head yaw (rad)")
        ax.set_ylabel("Visible orange pixels", color="#a96025")
        ax.grid(alpha=.2)
        other = ax.twinx()
        other.plot(yaw, [row["rgb_xy_error_mm_scoring_only"] for row in cohort],
                   marker="s", color="#246d8f")
        other.set_ylabel("Scored RGB XY error (mm)", color="#246d8f")
        ax.set_title(f"{seed} · {cohort[0]['intended_stratum']}", loc="left", fontweight="bold")
    fig.suptitle("Opened static view: head turn exposes more of the floor toy", fontweight="bold")
    fig.supxlabel("Fixed native reach-entry pose on rejected Direct Frame · simulator truth only scores image error")
    fig.savefig(OUT / "opened-static-gaze-scan.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"seed_count": report["seed_count"], "rows": len(rows),
                      "failure_10604": [row for row in rows if row["seed"] == 10604]}))


if __name__ == "__main__":
    main()
