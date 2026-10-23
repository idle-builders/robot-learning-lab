"""Compare fixed rear-side and arm-mounted toy views on saved native reach poses.

This is a sensor-placement replay. The native robot/toy state is never changed
by the camera; simulator segmentation is an evaluator, not an actor detector.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from cove_reach_view_observability import (
    BODY, FRAME_SIZE, ROOT, SAMPLE_EVERY, SEEDS, SOURCE, capture, image_panel,
    model_for_replay, sha,
)


OUTPUT = ROOT / "artifacts/ape-sensors-50cm/reach-side-v1"
CAMERAS = ("fixed_rear_side", "forearm_side")
SIDE_POS = (-0.03, -0.10, 0.08)  # torso local metres; optical center only
SIDE_YAW_DEG = -110.0
SIDE_PITCH_DEG = -45.0


def model_with_side_view() -> mujoco.MjModel:
    model = model_for_replay()
    camera_id = model.camera("belly_down").id
    model.cam_pos[camera_id] = SIDE_POS
    yaw, pitch = map(math.radians, (SIDE_YAW_DEG, SIDE_PITCH_DEG))
    forward = np.array([math.cos(pitch) * math.cos(yaw),
                        math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    camera_matrix = np.column_stack((right, up, -forward))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, camera_matrix.ravel())
    model.cam_quat[camera_id] = quat
    return model


def summarize(rows: list[dict], name: str, phase: str) -> dict:
    chosen = [row for row in rows if row["camera"] == name and
              (row["phase"] == phase if phase != "hand_within_50mm" else
               row["hand_distance_m"] <= 0.05)]
    visible = [row for row in chosen if row["toy_visible_pixels"] > 0]
    return {
        "samples": len(chosen),
        "toy_visible_samples": len(visible),
        "median_visible_pixels": None if not visible else int(np.median(
            [row["toy_visible_pixels"] for row in visible])),
        "median_optical_depth_m": None if not visible else round(float(np.median(
            [row["toy_median_optical_depth_m"] for row in visible])), 4),
        "out_of_frame_samples": sum(row["toy_pixels_without_robot"] == 0
                                    for row in chosen),
        "fully_self_occluded_samples": sum(
            row["toy_visible_pixels"] == 0 and row["toy_pixels_without_robot"] > 0
            for row in chosen),
        "partly_self_occluded_visible_samples": sum(
            row["toy_visible_pixels"] > 0 and row["self_occluded_pixels"] > 0
            for row in chosen),
    }


def main() -> None:
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer.get("status") != "rejected_no_selected_mechanical_design":
        raise ValueError("Re-evaluate camera placement on any newly selected body")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = model_with_side_view()
    data = mujoco.MjData(model)
    normal, ideal = mujoco.MjvOption(), mujoco.MjvOption()
    normal.geomgroup[:] = 0
    normal.geomgroup[0] = normal.geomgroup[1] = 1
    ideal.geomgroup[:] = 0
    ideal.geomgroup[0] = 1
    toy_id = model.geom("audit_toy").id
    camera_ids = {"fixed_rear_side": "belly_down", "forearm_side": "forearm_side"}
    rows, sources = [], []
    for seed in SEEDS:
        name = f"selected-continuous-{seed}-roll025"
        trace_path, report_path = SOURCE / f"{name}.npz", SOURCE / f"{name}.json"
        trace = np.load(trace_path)
        native = json.loads(report_path.read_text())
        if native["body_sha256"] != sha(BODY) or trace["qpos"].shape != (775, 35):
            raise ValueError(f"Native state/source mismatch: {name}")
        video = OUTPUT / f"{name}-side-versus-forearm.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10, (FRAME_SIZE[0] * 2, FRAME_SIZE[1]))
        if not writer.isOpened():
            raise RuntimeError(f"Could not write {video}")
        try:
            with mujoco.Renderer(model, width=FRAME_SIZE[0],
                                 height=FRAME_SIZE[1]) as renderer:
                for index in range(0, len(trace["qpos"]), SAMPLE_EVERY):
                    data.qpos[:] = trace["qpos"][index]
                    mujoco.mj_forward(model, data)
                    point = native["timeseries"][index]
                    distance = float(point["cube_to_hand_distance_m"])
                    phase = str(trace["phase"][index])
                    panels = []
                    for camera in CAMERAS:
                        rgb, visible, ideal_mask, depth = capture(
                            renderer, data, camera_ids[camera], normal, ideal, toy_id)
                        count = int(visible.sum())
                        ideal_count = int(ideal_mask.sum())
                        rows.append({
                            "seed": seed, "sample_index": index,
                            "time_s": round(float(point["time_s"]), 3),
                            "phase": phase, "camera": camera,
                            "hand_distance_m": round(distance, 5),
                            "toy_visible_pixels": count,
                            "toy_pixels_without_robot": ideal_count,
                            "self_occluded_pixels": max(ideal_count - count, 0),
                            "toy_median_optical_depth_m": None if not count else
                            round(float(np.median(depth[visible])), 4),
                        })
                        panels.append(image_panel(
                            rgb, visible, label=camera,
                            t=float(point["time_s"]), phase=phase,
                            distance_mm=distance * 1000))
                    writer.write(np.hstack(panels))
        finally:
            writer.release()
        sources.append({"seed": seed, "trace_sha256": sha(trace_path),
                        "native_report_sha256": sha(report_path),
                        "video": str(video.relative_to(ROOT)),
                        "video_sha256": sha(video), "video_frames": 155})

    with (OUTPUT / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {camera: {phase: summarize(rows, camera, phase)
                        for phase in ("posture", "preposition", "reach",
                                      "hand_within_50mm")}
               for camera in CAMERAS}
    paired = {}
    for phase in ("reach", "hand_within_50mm"):
        pair_rows = {}
        for row in rows:
            if row["phase"] != phase and phase != "hand_within_50mm":
                continue
            if phase == "hand_within_50mm" and row["hand_distance_m"] > 0.05:
                continue
            pair_rows.setdefault((row["seed"], row["sample_index"]), {})[row["camera"]] = (
                row["toy_visible_pixels"] > 0)
        paired[phase] = {
            "both": sum(v["fixed_rear_side"] and v["forearm_side"]
                        for v in pair_rows.values()),
            "fixed_only": sum(v["fixed_rear_side"] and not v["forearm_side"]
                              for v in pair_rows.values()),
            "forearm_only": sum(not v["fixed_rear_side"] and v["forearm_side"]
                                for v in pair_rows.values()),
            "neither": sum(not v["fixed_rear_side"] and not v["forearm_side"]
                           for v in pair_rows.values()),
        }
    result = {
        "schema": "robot-learning-lab.cove50-native-reach-side-view-comparison.v1",
        "classification": "native articulated pose/toy replay; cameras do not affect physics or action",
        "current_pointer_status": pointer["status"],
        "source_body_sha256": sha(BODY),
        "analysis_sha256": sha(Path(__file__)),
        "base_renderer_sha256": sha(ROOT / "experiments/ape_high_level/sensors/cove_reach_view_observability.py"),
        "camera_models": {
            "fixed_rear_side": {"body": "torso", "pos_m": SIDE_POS,
                                "yaw_deg": SIDE_YAW_DEG, "pitch_deg": SIDE_PITCH_DEG,
                                "vertical_fov_deg": 67,
                                "physical_mount_qualified": False},
            "forearm_side": {"body": "right_forearm", "pos_m": (0.03, -0.04, -0.05),
                             "vertical_fov_deg": 67,
                             "physical_mount_qualified": False}},
        "sample_period_s": 0.1, "image_size_px": list(FRAME_SIZE),
        "sources": sources, "summary": summary, "paired_visibility": paired,
        "limits": ["The side orientation was screened on these same traces; this is design data, not held-out placement validation.",
                   "Toy visibility uses evaluator segmentation; no image-guided policy, object identity or possession is tested.",
                   "Optical centers are hypothetical; source-sized module, moving shell, cable, focus, exposure and power are unqualified.",
                   "Depth is renderer geometry, not a physical stereo/ToF measurement."],
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
