"""Replay RGB-D visual odometry across the 21 real-capture moving-view routes.

This tests localization evidence on *recorded virtual camera paths*. It is not
an interactive owner task, an IMU test, or a body-qualified navigation result.
Simulator poses are used only after inference to score each pair.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from measure_real_scan_visual_odometry import (angle_deg, feature_packet,
                                               visual_transform)


ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "assets/real_indoor_scenes/real_capture_21_working_set.json"
OUTPUT = ROOT / "assets/real_indoor_scenes/real_capture_21_rgbd_odometry.json"
STRIDE = 4


def camera_pose(frame: dict, scan: bool, height: float) -> np.ndarray:
    pose = np.eye(4)
    if scan:
        forward = np.asarray(frame["forward"], dtype=float)
        forward /= np.linalg.norm(forward)
        up = np.array([0., 1., 0.])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        pose[:3, :3] = np.column_stack((right, -up, forward))
        pose[:3, 3] = frame["eye_m"]
    else:
        yaw = float(frame["yaw_rad"])
        c, s = math.cos(yaw), math.sin(yaw)
        world_from_habitat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pose[:3, :3] = world_from_habitat @ np.diag([1, -1, -1])
        pose[:3, 3] = np.asarray(frame["position_m"]) + [0, height, 0]
    return pose


def run_scene(row: dict) -> dict:
    trace_path = ROOT / row["trace"]
    trace = json.loads(trace_path.read_text())
    frames = trace["frames"]
    folder = trace_path.parent
    scan = row["source_family"] == "scansceneverse"
    if scan:
        width, height = trace["width"], trace["image_height"]
        focal = height / (2 * math.tan(math.radians(trace["vertical_fov_degrees"]) / 2))
        eye_height = trace["height_above_floor_m"]
    else:
        width, height = trace["camera_resolution_px"]
        focal = width / (2 * math.tan(math.radians(trace["camera_hfov_deg"]) / 2))
        eye_height = trace["camera_height_above_nav_m"]
    k = np.array([[focal, 0, (width - 1) / 2],
                  [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    packets = {}

    def packet(index: int) -> dict:
        if index not in packets:
            frame = frames[index]
            image = (folder / f"frame_{index:03d}.png") if scan else folder / frame["image"]
            depth_path = ((folder / f"depth_{index:03d}.png") if scan else
                          (folder / "depth_replay" / frame.get("depth_mm_png", f"depth_{index:04d}.png"))
                          if row["source_family"] == "hm3d_example" else
                          (folder / frame["depth_mm_png"]))
            if image.is_file():
                rgb = np.asarray(Image.open(image).convert("RGB"))
            else:
                video = cv2.VideoCapture(str(ROOT / row["video"]))
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = video.read()
                video.release()
                if not ok:
                    raise FileNotFoundError(image)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000
            if rgb.shape[:2] != depth.shape or rgb.shape[:2] != (height, width):
                raise ValueError(f"Unaligned image/depth: {image} {depth_path}")
            packets[index] = feature_packet(rgb, depth, orb)
        return packets[index]

    pairs = []
    for a in range(0, len(frames) - STRIDE, STRIDE):
        b = a + STRIDE
        forward, fs = visual_transform(packet(a), packet(b), k, matcher)
        reverse, rs = visual_transform(packet(b), packet(a), k, matcher)
        truth = np.linalg.inv(camera_pose(frames[b], scan, eye_height)) @ \
            camera_pose(frames[a], scan, eye_height)
        accepted = False
        closure = None
        if forward is not None and reverse is not None:
            closure = reverse @ forward
            accepted = bool(fs["inliers"] >= 30 and rs["inliers"] >= 30 and
                            fs["inlier_ratio"] >= .25 and rs["inlier_ratio"] >= .25 and
                            fs["median_reprojection_px"] <= 2.5 and
                            rs["median_reprojection_px"] <= 2.5 and
                            np.linalg.norm(closure[:3, 3]) <= .08 and
                            angle_deg(closure[:3, :3]) <= 5.)
        pairs.append({
            "from_frame": a, "to_frame": b,
            "true_translation_m_in_source_scale": float(np.linalg.norm(truth[:3, 3])),
            "forward": fs, "reverse": rs, "accepted": accepted,
            "translation_error_m_in_source_scale":
                float(np.linalg.norm(forward[:3, 3] - truth[:3, 3])) if forward is not None else None,
            "rotation_error_deg":
                angle_deg(forward[:3, :3].T @ truth[:3, :3]) if forward is not None else None,
            "reciprocal_translation_m":
                float(np.linalg.norm(closure[:3, 3])) if closure is not None else None,
            "reciprocal_rotation_deg":
                angle_deg(closure[:3, :3]) if closure is not None else None,
        })
    return {"scene_id": row["scene_id"], "category": row["category"],
            "source_family": row["source_family"], "split_proposal": row["split_proposal"],
            "video": row["video"], "trace": row["trace"],
            "pairs": pairs, "accepted_pairs": sum(p["accepted"] for p in pairs)}


def main() -> None:
    cv2.setRNGSeed(0)
    scenes = [run_scene(row) for row in json.loads(INDEX.read_text())["scenes"]]
    pairs = [p for scene in scenes for p in scene["pairs"]]
    accepted = [p for p in pairs if p["accepted"]]
    report = {
        "schema": "ape.real-capture-working-set-rgbd-odometry.v1",
        "scope": f"{len(scenes)} provisional moving-view routes; bidirectional ORB RGB-D PnP at four-frame stride; poses evaluator-only",
        "camera_height_m": .42,
        "acceptance_gate": ">=30 inliers each way, >=0.25 ratio, <=2.5px median residual, <=8cm/5deg reciprocal closure",
        "metric_caveat": "Most Sketchfab GLB scales are plausible but unmeasured; their translation errors are in assumed source metres.",
        "not_tested": ["closed-loop camera choices", "dynamic owner/object", "IMU", "joint odometry", "contact sensing", "selected body collision"],
        "scene_count": len(scenes), "pair_count": len(pairs),
        "accepted_pair_count": len(accepted),
        "accepted_pair_translation_error_median":
            float(np.median([p["translation_error_m_in_source_scale"] for p in accepted])) if accepted else None,
        "scenes": scenes,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    for scene in scenes:
        print(f'{scene["scene_id"]}: {scene["accepted_pairs"]}/{len(scene["pairs"])} accepted')
    print(f'Total: {len(accepted)}/{len(pairs)} accepted; {OUTPUT}')


if __name__ == "__main__":
    main()
