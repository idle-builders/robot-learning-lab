"""Render synthetic cube placements at opened native entry body states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_goal_features import (
    FEATURE_NAMES, features,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import VirtualRgbGoal

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-image-goal-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(seed: int) -> tuple[Path, Path]:
    folder = "yaw15-development" if seed in (9921, 9973, 9974, 9976) else "yaw15-reserved"
    return OUT / folder / f"seed-{seed}.json", OUT / folder / f"seed-{seed}.npz"


def main() -> None:
    plan_path = OUT / "rgb-calibration-plan.json"
    plan = json.loads(plan_path.read_text())
    for path, key in ((ROOT / "robot/ape-current-design.json", "selected_pointer_sha256"),
                      (ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml", "selected_body_sha256"),
                      (ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt", "saved_reach_actor_sha256"),
                      (ROOT / "artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json", "virtual_camera_freeze_sha256")):
        assert sha(path) == plan[key], f"Source changed: {path}"
    rng = np.random.default_rng(51015)
    source = VirtualRgbGoal(selected_body_sha256=plan["selected_body_sha256"], max_age_s=9)
    original_camera_pos = source.model.cam_pos[source.camera].copy()
    head_qposadr = source.model.joint("head_yaw").qposadr[0]
    inputs = []
    targets = []
    splits = []
    seeds = []
    sample_indices = []
    source_rows = []
    abstentions = []
    for split, seed_list, count in (("train", plan["opened_training_entry_seeds"], 256),
                                    ("validation", plan["opened_validation_entry_seeds"], 128)):
        for seed in seed_list:
            report_path, trace_path = source_paths(seed)
            report = json.loads(report_path.read_text())
            assert report["seed"] == seed and report["body_sha256"] == plan["selected_body_sha256"]
            assert report["gaze_head_yaw_rad"] == -1.5 and report["one_sim_without_skill_reset"]
            with np.load(trace_path) as trace:
                native_entry = trace["qpos"][375].copy()
            assert native_entry.shape == (35,)
            accepted = 0
            for sample in range(count):
                qpos = native_entry.copy()
                if sample:
                    qpos[28:30] += rng.uniform(-.008, .008, size=2)
                    if split == "train":
                        qpos[head_qposadr] = -1.5 + rng.uniform(0, .03)
                        source.model.cam_pos[source.camera] = original_camera_pos + rng.uniform(-.001, .001, size=3)
                    else:
                        source.model.cam_pos[source.camera] = original_camera_pos
                else:
                    source.model.cam_pos[source.camera] = original_camera_pos
                event, rgb = source.capture(qpos, 7.5)
                if not event["accepted"]:
                    abstentions.append({"seed": seed, "sample": sample, "split": split,
                                        "reason": event["abstain_reason"]})
                    continue
                coarse = np.asarray(event["estimated_goal_world_xyz_m"], dtype=float)
                x = features(rgb, coarse, source.data.cam_xpos[source.camera],
                             source.data.cam_xmat[source.camera])
                y = (qpos[28:30] - coarse[:2]) * 1000
                inputs.append(x)
                targets.append(y)
                splits.append(split)
                seeds.append(seed)
                sample_indices.append(sample)
                accepted += 1
            source_rows.append({"seed": seed, "split": split, "source_report": str(report_path.relative_to(ROOT)),
                                "source_report_sha256": sha(report_path),
                                "source_trace": str(trace_path.relative_to(ROOT)),
                                "source_trace_sha256": sha(trace_path),
                                "rendered": count, "valid": accepted, "abstained": count - accepted})
    source.model.cam_pos[source.camera] = original_camera_pos
    source.close()
    x = np.asarray(inputs)
    y = np.asarray(targets)
    split = np.asarray(splits)
    assert x.shape == (len(y), len(FEATURE_NAMES)) and (split == "train").sum() > 2000
    np.savez_compressed(OUT / "rgb-calibration-data.npz", x=x, y_mm=y, split=split,
                        seed=np.asarray(seeds), sample=np.asarray(sample_indices),
                        feature_names=np.asarray(FEATURE_NAMES))
    report = {
        "schema": "robot-learning-lab.cove50-virtual-rgb-calibration-data.v1",
        "plan": str(plan_path.relative_to(ROOT)), "plan_sha256": sha(plan_path),
        "selected_body_sha256": plan["selected_body_sha256"],
        "feature_source_sha256": sha(Path(__file__).with_name("rgb_goal_features.py")),
        "collector_sha256": sha(Path(__file__)),
        "random_seed": 51015, "features": FEATURE_NAMES,
        "train_valid_images": int((split == "train").sum()),
        "validation_valid_images": int((split == "validation").sum()),
        "coarse_train_xy_error_mm": {"median": round(float(np.median(np.linalg.norm(y[split == "train"], axis=1))), 3),
                                      "max": round(float(np.max(np.linalg.norm(y[split == "train"], axis=1))), 3)},
        "coarse_validation_xy_error_mm": {"median": round(float(np.median(np.linalg.norm(y[split == "validation"], axis=1))), 3),
                                           "max": round(float(np.max(np.linalg.norm(y[split == "validation"], axis=1))), 3)},
        "sources": source_rows, "abstentions": abstentions,
        "qualification": "Synthetic appearance, exact selected robot state and capture camera pose; cube truth is a supervised label, never an inference feature. Native camera hardware and calibration remain unqualified.",
    }
    (OUT / "rgb-calibration-data-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("train_valid_images", "validation_valid_images",
                                            "coarse_validation_xy_error_mm")}, indent=2))


if __name__ == "__main__":
    main()
