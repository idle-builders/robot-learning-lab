"""Add opened native camera poses to the historical RGB correction dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_goal_features import FEATURE_NAMES, features
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import VirtualRgbGoal

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V2 = OUT / "calibration-v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "rgb-calibration-v2-plan.json"
    plan = json.loads(plan_path.read_text())
    base_path = ROOT / plan["base_data"]
    matrix_path = ROOT / plan["source_matrix"]
    if sha(base_path) != plan["base_data_sha256"] or sha(matrix_path) != plan["source_matrix_sha256"]:
        raise ValueError("Calibration sources changed")
    body_sha = json.loads((OUT / "rgb-calibration-plan.json").read_text())["selected_body_sha256"]
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("This dataset is historical only")
    with np.load(base_path) as base:
        arrays = {key: base[key].copy() for key in ("x", "y_mm", "split", "seed", "sample")}
        if base["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("RGB feature contract changed")
    matrix = json.loads(matrix_path.read_text())
    by_seed = {row["seed"]: row for row in matrix["cases"]}
    source = VirtualRgbGoal(selected_body_sha256=body_sha, max_age_s=9)
    original_camera = source.model.cam_pos[source.camera].copy()
    head_qposadr = source.model.joint("head_yaw").qposadr[0]
    rng = np.random.default_rng(51016)
    extra = {name: [] for name in ("x", "y_mm", "split", "seed", "sample")}
    rows = []
    for split, seeds, count in (("train", plan["additional_training_seeds"], plan["samples_per_training_state"]),
                                ("validation", plan["additional_validation_seeds"], plan["samples_per_validation_state"])):
        for seed in seeds:
            row = by_seed[seed]
            report_path, trace_path = ROOT / row["result"], ROOT / row["trace"]
            report = json.loads(report_path.read_text())
            if report["seed"] != seed or report["body_sha256"] != body_sha or not report["one_sim_without_skill_reset"]:
                raise ValueError(f"Wrong opened source: {seed}")
            with np.load(trace_path) as trace:
                native_entry = trace["qpos"][375].copy()
            if native_entry.shape != (35,):
                raise ValueError(f"Bad native entry state: {seed}")
            accepted = 0
            for sample in range(count):
                qpos = native_entry.copy()
                source.model.cam_pos[source.camera] = original_camera
                if sample:
                    qpos[28:30] += rng.uniform(-plan["training_toy_jitter_m"],
                                               plan["training_toy_jitter_m"], size=2)
                    if split == "train":
                        qpos[head_qposadr] = rng.uniform(*plan["head_yaw_training_range_rad"])
                        source.model.cam_pos[source.camera] = original_camera + rng.uniform(
                            -plan["camera_jitter_m"], plan["camera_jitter_m"], size=3)
                event, rgb = source.capture(qpos, 7.5)
                if not event["accepted"]:
                    continue
                coarse = np.asarray(event["estimated_goal_world_xyz_m"], dtype=float)
                extra["x"].append(features(rgb, coarse, source.data.cam_xpos[source.camera],
                                           source.data.cam_xmat[source.camera]))
                extra["y_mm"].append((qpos[28:30] - coarse[:2]) * 1000)
                extra["split"].append(split)
                extra["seed"].append(seed)
                extra["sample"].append(sample)
                accepted += 1
            rows.append({"seed": seed, "split": split, "source_report": str(report_path.relative_to(ROOT)),
                         "source_report_sha256": sha(report_path),
                         "source_trace": str(trace_path.relative_to(ROOT)),
                         "source_trace_sha256": sha(trace_path),
                         "rendered": count, "valid": accepted, "abstained": count - accepted})
    source.model.cam_pos[source.camera] = original_camera
    source.close()
    x = np.concatenate((arrays["x"], np.asarray(extra["x"])))
    y = np.concatenate((arrays["y_mm"], np.asarray(extra["y_mm"])))
    split = np.concatenate((arrays["split"], np.asarray(extra["split"])))
    seed = np.concatenate((arrays["seed"], np.asarray(extra["seed"])))
    sample = np.concatenate((arrays["sample"], np.asarray(extra["sample"])))
    if x.shape != (len(y), len(FEATURE_NAMES)) or set(seed[split == "train"]) & set(seed[split == "validation"]):
        raise ValueError("Invalid combined calibration split")
    V2.mkdir(parents=True, exist_ok=True)
    data_path = V2 / "rgb-calibration-data.npz"
    np.savez_compressed(data_path, x=x, y_mm=y, split=split, seed=seed, sample=sample,
                        feature_names=np.asarray(FEATURE_NAMES))
    result = {"schema": "robot-learning-lab.cove50-rgb-calibration-v2-data.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "base_data_sha256": sha(base_path),
              "data_sha256": sha(data_path), "collector_sha256": sha(Path(__file__)),
              "train_valid_images": int((split == "train").sum()),
              "validation_valid_images": int((split == "validation").sum()),
              "sources": rows}
    (V2 / "data-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("train_valid_images", "validation_valid_images")}, indent=2))


if __name__ == "__main__":
    main()
