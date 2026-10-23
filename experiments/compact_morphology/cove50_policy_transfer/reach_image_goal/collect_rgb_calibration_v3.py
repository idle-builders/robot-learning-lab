"""Collect varied opened native RGB views for a state-disjoint correction fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_goal_features import FEATURE_NAMES, features
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import VirtualRgbGoal

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "calibration-v3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "plan.json"
    plan = json.loads(plan_path.read_text())
    pointer = ROOT / "robot/ape-current-design.json"
    matrix_path = ROOT / plan["source_matrix"]
    base_path = ROOT / plan["base_data"]
    expected = {"pointer": pointer, "source_matrix": matrix_path, "base_data": base_path,
                "collector": Path(__file__),
                "feature_source": Path(__file__).with_name("rgb_goal_features.py")}
    for key, path in expected.items():
        if sha(path) != plan["source_sha256"][key]:
            raise ValueError(f"Frozen calibration source changed: {key}")
    if json.loads(pointer.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V3 remains a historical sandbox dataset")
    matrix = json.loads(matrix_path.read_text())
    if not matrix["complete"] or not matrix["case_count"] == 24:
        raise ValueError("Need complete opened V21 source")
    source_rows = {row["seed"]: row for row in matrix["cases"]}
    training_seeds = plan["training_seeds"]
    validation_seeds = plan["validation_seeds"]
    if (set(training_seeds) & set(validation_seeds) or
            set(training_seeds + validation_seeds) != set(source_rows)):
        raise ValueError("V3 split must partition opened states")
    with np.load(base_path) as base:
        original = {key: base[key].copy() for key in ("x", "y_mm", "split", "seed", "sample")}
        if base["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("Calibration feature contract changed")
    first_report = json.loads((ROOT / matrix["cases"][0]["result"]).read_text())
    body_sha = first_report["body_sha256"]
    source = VirtualRgbGoal(selected_body_sha256=body_sha, max_age_s=9)
    original_camera = source.model.cam_pos[source.camera].copy()
    head_qposadr = source.model.joint("head_yaw").qposadr[0]
    head_lo, head_hi = source.model.joint("head_yaw").range
    rng = np.random.default_rng(plan["collector_seed"])
    extra: dict[str, list] = {name: [] for name in original}
    rows = []
    for split, seeds, count in (("train", training_seeds, plan["samples_per_training_state"]),
                                ("validation", validation_seeds, plan["samples_per_validation_state"])):
        for seed in seeds:
            row = source_rows[seed]
            report_path, trace_path = ROOT / row["result"], ROOT / row["trace"]
            report = json.loads(report_path.read_text())
            if (report["seed"] != seed or report["body_sha256"] != body_sha or
                    not report["one_sim_without_skill_reset"]):
                raise ValueError(f"Invalid native source seed {seed}")
            with np.load(trace_path) as trace:
                native_entry = trace["qpos"][375].copy()
            if native_entry.shape != (35,):
                raise ValueError(f"Invalid native entry seed {seed}")
            accepted = 0
            for sample in range(count):
                qpos = native_entry.copy()
                source.model.cam_pos[source.camera] = original_camera
                if sample:
                    qpos[28:30] += rng.uniform(-plan["toy_jitter_m"], plan["toy_jitter_m"], 2)
                    qpos[head_qposadr] = np.clip(
                        qpos[head_qposadr] + rng.uniform(-plan["head_yaw_jitter_rad"],
                                                         plan["head_yaw_jitter_rad"]),
                        head_lo, head_hi)
                    source.model.cam_pos[source.camera] += rng.uniform(
                        -plan["camera_jitter_m"], plan["camera_jitter_m"], 3)
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
            rows.append({"seed": seed, "split": split, "valid": accepted,
                         "rendered": count, "source_report": str(report_path.relative_to(ROOT)),
                         "source_report_sha256": sha(report_path),
                         "source_trace": str(trace_path.relative_to(ROOT)),
                         "source_trace_sha256": sha(trace_path)})
            print(f"{seed} {split}: {accepted}/{count}", flush=True)
    source.model.cam_pos[source.camera] = original_camera
    source.close()
    arrays = {key: np.concatenate((original[key], np.asarray(extra[key]))) for key in original}
    if arrays["x"].shape != (len(arrays["y_mm"]), len(FEATURE_NAMES)):
        raise ValueError("Invalid combined V3 dataset")
    if set(arrays["seed"][arrays["split"] == "train"]) & set(
            arrays["seed"][arrays["split"] == "validation"]):
        raise ValueError("V3 train and validation states overlap")
    data_path = OUT / "rgb-calibration-data.npz"
    np.savez_compressed(data_path, **arrays, feature_names=np.asarray(FEATURE_NAMES))
    result = {"schema": "robot-learning-lab.cove50-rgb-calibration-v3-data.v1",
              "role": "historical_opened_image_correction_training",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "data_sha256": sha(data_path),
              "train_images": int((arrays["split"] == "train").sum()),
              "validation_images": int((arrays["split"] == "validation").sum()),
              "sources": rows}
    (OUT / "data-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("train_images", "validation_images")}))


if __name__ == "__main__":
    main()
