"""Assess rendered-template RGB at the better exposed legal head angle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    TemplateMatchedVirtualRgbGoal,
)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v31"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float]) -> dict:
    return {"median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(np.max(values)), 4)}


def main() -> None:
    plan_path = OUT / "offline-plan.json"
    plan = json.loads(plan_path.read_text())
    matrix_path = ROOT / plan["source_matrix"]
    paths = {"matrix": matrix_path, "pointer": ROOT / "robot/ape-current-design.json",
             "template": Path(__file__).with_name("template_virtual_rgb_goal_v30.py"),
             "assessor": Path(__file__),
             "calibration": BASE / "rgb-calibration-model.npz"}
    for key, path in paths.items():
        if sha(path) != plan["source_sha256"][key]:
            raise ValueError(f"Offline source changed: {key}")
    matrix = json.loads(matrix_path.read_text())
    if not matrix["complete"] or matrix["case_count"] != 24:
        raise ValueError("Need full opened V21 matrix")
    body_sha = json.loads((ROOT / matrix["cases"][0]["result"]).read_text())["body_sha256"]
    observer = TemplateMatchedVirtualRgbGoal(
        calibration_model=paths["calibration"], selected_body_sha256=body_sha, max_age_s=9)
    head = observer.model.joint("head_yaw").qposadr[0]
    head_lo, head_hi = observer.model.joint("head_yaw").range
    base_cam = observer.model.cam_pos[observer.camera].copy()
    rng = np.random.default_rng(plan["random_seed"])
    rows = []
    for source_row in matrix["cases"]:
        with np.load(ROOT / source_row["trace"]) as trace:
            native_entry = trace["qpos"][375].copy()
        if native_entry.shape != (35,):
            raise ValueError("Native qpos contract changed")
        for sample in range(plan["samples_per_state"]):
            qpos = native_entry.copy()
            qpos[head] = -1.5
            observer.model.cam_pos[observer.camera] = base_cam
            if sample:
                qpos[28:30] += rng.uniform(-plan["toy_jitter_m"], plan["toy_jitter_m"], 2)
                qpos[head] = np.clip(qpos[head] + rng.uniform(
                    -plan["head_yaw_jitter_rad"], plan["head_yaw_jitter_rad"]),
                    head_lo, head_hi)
                observer.model.cam_pos[observer.camera] += rng.uniform(
                    -plan["camera_jitter_m"], plan["camera_jitter_m"], 3)
            event, _ = observer.capture(qpos, 7.5)
            truth = qpos[28:30]
            before = (np.asarray(event["pre_template_goal_world_xyz_m"][:2])
                      if event["template_applied"] else None)
            after = (np.asarray(event["estimated_goal_world_xyz_m"][:2])
                     if event["template_applied"] else None)
            rows.append({"seed": source_row["seed"], "sample": sample,
                         "intended_stratum": source_row["intended_stratum"],
                         "head_yaw_rad": round(float(qpos[head]), 4),
                         "toy_jitter_xy_m": np.round(qpos[28:30] - native_entry[28:30], 5).tolist(),
                         "orange_pixels": event["orange_pixels"],
                         "accepted": event["accepted"],
                         "template_applied": event["template_applied"],
                         "template_pixels_disagree": event.get("template_pixels_disagree"),
                         "template_candidate_count": event.get("template_candidate_count"),
                         "template_best_score_tie_count": event.get("template_best_score_tie_count"),
                         "baseline_error_mm_scoring_only": (None if before is None else
                             round(float(np.linalg.norm((before - truth) * 1000)), 4)),
                         "template_error_mm_scoring_only": (None if after is None else
                             round(float(np.linalg.norm((after - truth) * 1000)), 4))})
        print(source_row["seed"], len(rows), flush=True)
    observer.model.cam_pos[observer.camera] = base_cam
    observer.close()
    baseline = [row["baseline_error_mm_scoring_only"] for row in rows if row["template_applied"]]
    updated = [row["template_error_mm_scoring_only"] for row in rows if row["template_applied"]]
    full = len(rows) == 24 * plan["samples_per_state"] and len(updated) == len(rows)
    gate = (full and max(updated) <= plan["max_validation_error_mm"] and
            next(row["template_error_mm_scoring_only"] for row in rows
                 if row["seed"] == 10604 and row["sample"] == 0) <= 1.0)
    report = {"schema": "robot-learning-lab.cove50-v31-template-gaze-offline.v1",
              "role": "opened_varied_virtual_rgb_stronger_gaze_gate_not_native_contact",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "source_sha256": sha(Path(__file__)),
              "case_count": len(rows), "accepted_count": len(updated),
              "baseline_error_mm": distribution(baseline),
              "template_error_mm": distribution(updated),
              "offline_gate_pass": gate, "rows": rows,
              "qualification": "Uses exact synthetic robot/camera geometry and known upright 20 mm cube; image-only at inference but not qualified on a physical camera or future body."}
    (OUT / "offline-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("case_count", "accepted_count",
                      "baseline_error_mm", "template_error_mm", "offline_gate_pass")}))


if __name__ == "__main__":
    main()
