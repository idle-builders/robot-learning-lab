"""Audit the paired articulated camera-mount owner episodes from V244p."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_lateral_clutter_depth_v285 import project
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, model_with_room


OUT = ROOT / "artifacts/ape-articulated-depth-mount-v244p"
REPORT = OUT / "report.json"
EXPECTED_SEEDS = (170, 172, 174, 176)
EXPECTED_CONDITIONS = ("head_static", "head_horizon", "chest_static",
                       "chest_horizon")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        count = 0
        for frame in reader:
            assert frame.ndim == 3 and frame.shape[2] == 3
            count += 1
        return count


def main() -> None:
    report = json.loads(REPORT.read_text())
    assert report["schema"] == "robot-learning-lab.ape-articulated-depth-mount.v1"
    assert report["source_sha256"] == digest(
        ROOT / "experiments/ape_high_level/closed_loop/"
               "run_articulated_depth_mount_v244p.py")
    assert report["historical_visual_sha256"] == json.loads(
        (ROOT / "robot/ape-current-design.json").read_text()
    )["evidence"]["visual_body_sha256"]
    rows = report["rows"]
    assert len(rows) == len(EXPECTED_SEEDS) * len(EXPECTED_CONDITIONS)
    assert {(r["case"][0], r["condition"]) for r in rows} == {
        (s, c) for s in EXPECTED_SEEDS for c in EXPECTED_CONDITIONS}
    checked_packets = 0
    checked_videos = 0
    failures = []
    summaries = []
    model = model_with_room(visual=selected_visual())
    pose_data = mujoco.MjData(model)
    for row in rows:
        seed, condition = row["case"][0], row["condition"]
        trace_file = ROOT / row["trace_path"]
        video_file = ROOT / row["video_path"]
        assert digest(trace_file) == row["trace_sha256"]
        assert digest(video_file) == row["video_sha256"]
        trace = json.loads(trace_file.read_text())
        assert trace["case"] == row["case"]
        assert trace["condition"] == condition
        probes = trace["probe_inputs"]
        assert len(probes) == row["probe_count"]
        events = trace["controller_events"]
        assert len([e for e in events if e["action"] == "pre_step_probe"]) == len(probes)
        assert row["evaluator_motion_interrupted"] == trace[
            "evaluator_motion_interrupted"]
        assert np.isclose(row["evaluator_path_min_clearance_m"],
                          trace["evaluator_path_min_clearance"][0], atol=1e-5)
        frames = video_frames(video_file)
        assert frames >= len(probes) + 1
        checked_videos += 1
        camera = "lower_rgb" if condition.startswith("head") else "chest_low50_x45"
        evaluator_prop_points = []
        evaluator_prop_center_pixels = []
        for probe in probes:
            if condition.startswith("head"):
                assert probe["predicted_self_pixels"] > 0
            path = ROOT / probe["actor_frame_path"]
            assert digest(path) == probe["actor_frame_sha256"]
            with np.load(path) as packet:
                measured = packet["measured_depth_m"]
                filtered = packet["filtered_depth_m"]
                points = packet["actor_points_xyz_m"]
                valid = np.isfinite(measured)
                assert not np.any(valid & ((measured < .30) | (measured > 1.00)))
                retained = np.isfinite(filtered)
                assert not np.any(retained & ~valid)
                np.testing.assert_allclose(filtered[retained], measured[retained])
                assert str(packet["camera_name"]) == camera
                assert float(packet["camera_fovy_deg"]) == 55.
                pose_data.qpos[:] = packet["actor_qpos"]
                pose_data.qvel[:] = 0.
                mujoco.mj_forward(model, pose_data)
                camera_id = model.camera(camera).id
                np.testing.assert_allclose(
                    pose_data.cam_xpos[camera_id],
                    packet["camera_origin_estimated_m"], atol=1e-5)
                np.testing.assert_allclose(
                    pose_data.cam_xmat[camera_id].reshape(3, 3),
                    packet["camera_rotation_estimated"], atol=1e-5)
                projected = project(
                    filtered,
                    (packet["camera_origin_estimated_m"],
                     packet["camera_rotation_estimated"]),
                    float(packet["camera_fovy_deg"]))
                projected = projected[projected[:, 2] >= .04]
                np.testing.assert_allclose(points, projected, atol=1e-6)
                assert len(points) == probe["depth_points_above_ground"]
                assert np.isclose(float(packet["capture_time_s"]),
                                  probe["capture_time_s"], atol=5e-5)
                # Evaluator-only count: never passed to the route actor.
                _, _, near, enter_start, enter_end, leave_start, leave_end = row["case"]
                t = float(packet["capture_time_s"])
                if t < enter_start:
                    prop_y = .45
                elif t < enter_end:
                    prop_y = .45 + (near - .45) * (t - enter_start) / (
                        enter_end - enter_start)
                elif t < leave_start:
                    prop_y = near
                elif t < leave_end:
                    prop_y = near + (.45 - near) * (t - leave_start) / (
                        leave_end - leave_start)
                else:
                    prop_y = .45
                near_prop = ((np.abs(points[:, 0]) < .07) &
                             (np.abs(points[:, 1] - prop_y) < .065) &
                             (points[:, 2] >= .04) & (points[:, 2] <= .11))
                evaluator_prop_points.append(int(np.sum(near_prop)))
                center_camera = (np.array([0., prop_y, .04]) -
                                 packet["camera_origin_estimated_m"]) @ packet[
                                     "camera_rotation_estimated"]
                height, width = measured.shape
                focal = height / (2 * math.tan(math.radians(55.) / 2))
                if center_camera[2] < 0:
                    pixel = (width / 2 + center_camera[0] / -center_camera[2] * focal,
                             height / 2 - center_camera[1] / -center_camera[2] * focal)
                    evaluator_prop_center_pixels.append(
                        [round(pixel[0], 1), round(pixel[1], 1)])
                else:
                    evaluator_prop_center_pixels.append(None)
            checked_packets += 1
        if row["evaluator_path_min_clearance_m"] < .05:
            failures.append((seed, condition, "under_5cm"))
        summaries.append({"seed": seed, "condition": condition,
                          "status": row["status"],
                          "min_clearance_m": row["evaluator_path_min_clearance_m"],
                          "probes": len(probes), "video_frames": frames,
                          "owner_questions": row["owner_questions"],
                          "predicted_self_pixel_counts": [
                              p["predicted_self_pixels"] for p in probes],
                          "masked_self_pixel_counts": [
                              p["self_masked_pixels"] for p in probes],
                          "evaluator_near_prop_point_counts": evaluator_prop_points,
                          "evaluator_prop_center_pixels": evaluator_prop_center_pixels})
    audit = {"schema": "robot-learning-lab.ape-articulated-depth-mount-audit.v1",
             "report_sha256": digest(REPORT), "all_packet_and_video_checks_pass": True,
             "episodes": len(rows), "packets": checked_packets,
             "videos": checked_videos, "under_5cm_cases": failures,
             "rows": summaries}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("episodes", "packets", "videos",
                                           "under_5cm_cases")}, indent=2))


if __name__ == "__main__":
    main()
