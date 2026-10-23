"""Compare a timed chest wait and remembered-bearing head look at owner-route stop.

Both branches start from the same saved simulator state after the unchanged
furnished owner actor stops. Target truth/segmentation is evaluator-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
from experiments.ape_high_level.closed_loop import run as loop  # noqa: E402
from experiments.ape_high_level.native_owner_bridge.replay_owner_packets import (  # noqa: E402
    NativeShapedRoom, SOURCE_QPOS, SOURCE_ODOM)


OUT = ROOT / "artifacts/ape-native-owner-reobserve/v1"
CONFIG = "chest_rgbd_range_guard_scan_memory"
SCENARIO = "furnished_moving"
CAMERAS = {"timed_chest": "chest_low50_x45",
           "head_lower": "lower_rgb"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def snapshot(scene: NativeShapedRoom) -> dict:
    return {"actual": scene.actual.copy(), "estimated": scene.estimated.copy(),
            "head": scene.head, "time_s": scene.time_s,
            "path_min_clearance": scene.path_min_clearance,
            "motion_interrupted": scene.motion_interrupted,
            "safety_stop_observation": scene.safety_stop_observation,
            "qvel": scene.data.qvel.copy()}


def restore(scene: NativeShapedRoom, saved: dict) -> None:
    scene.actual[:] = saved["actual"]
    scene.estimated[:] = saved["estimated"]
    scene.head = saved["head"]
    scene.time_s = saved["time_s"]
    scene.path_min_clearance = saved["path_min_clearance"]
    scene.motion_interrupted = saved["motion_interrupted"]
    scene.safety_stop_observation = saved["safety_stop_observation"]
    scene.data.qvel[:] = saved["qvel"]
    scene._update_moving_obstacle()
    scene._forward()


def frame_evidence(scene: NativeShapedRoom, vision: loop.Vision,
                   exemplar: np.ndarray, camera: str, rgb: Image.Image,
                   depth: np.ndarray, expected_xy: np.ndarray) -> dict:
    proposals = vision.detect(rgb)
    scores = (vision.feature([rgb.crop(item["box"]) for item in proposals]) @ exemplar
              if proposals else np.empty(0))
    ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]),
                    reverse=True)
    camera_pose = scene.camera_pose_estimate(camera)
    actor_candidate = None
    for proposal, score in ranked:
        if float(score) < .35:
            break
        located = loop.locate_world(proposal["box"], camera_pose, depth)
        if located is not None and np.linalg.norm(
                np.asarray(located[:2]) - expected_xy) <= .12:
            actor_candidate = {"box_xyxy_px": [round(float(x), 2)
                                             for x in proposal["box"]],
                               "similarity": round(float(score), 4),
                               "located_xyz_m": [round(float(x), 4) for x in located],
                               "distance_from_remembered_xy_m": round(float(
                                   np.linalg.norm(np.asarray(located[:2]) - expected_xy)), 4)}
            break
    top = ranked[0] if ranked else None
    truth = scene.truth_box("TomatoSauce", camera)
    score_box = actor_candidate["box_xyxy_px"] if actor_candidate else None
    return {
        "camera": camera,
        "capture_time_s": round(scene.time_s, 4),
        "actor_pose_xyyaw": scene.estimated.round(5).tolist(),
        "proposal_count": len(proposals),
        "top_similarity": round(float(top[1]), 4) if top else None,
        "top_box_xyxy_px": [round(float(x), 2) for x in top[0]["box"]]
        if top else None,
        "actor_fresh_metric_candidate": actor_candidate,
        "true_target_box_evaluator_only": truth,
        "true_target_pixel_area_evaluator_only": ((truth[2] - truth[0]) *
                                                  (truth[3] - truth[1]))
        if truth else 0,
        "candidate_true_target_iou_evaluator_only": round(
            loop.iou(score_box, truth), 4) if score_box else None,
        "candidate_location_error_evaluator_only_m": round(float(np.linalg.norm(
            np.asarray(actor_candidate["located_xyz_m"][:2]) -
            np.asarray(scene.positions["TomatoSauce"]))), 4)
        if actor_candidate else None,
        "minimum_obstacle_clearance_evaluator_only_m": round(float(
            scene.obstacle_clearance()[0]), 4),
    }


def branch(scene: NativeShapedRoom, mode: str, seed: int,
           expected_xy: np.ndarray, vision: loop.Vision,
           exemplar: np.ndarray, folder: Path) -> tuple[dict, list[np.ndarray]]:
    camera = CAMERAS[mode]
    frames: list[np.ndarray] = []
    samples = []
    def sample(substep: int, fraction: float) -> None:
        rgb, _ = scene.capture(camera, depth=False)
        frames.append(np.asarray(rgb).copy())
        samples.append({"substep": substep, "fraction": round(fraction, 3),
                        "time_s": round(scene.time_s, 4),
                        "head_yaw_rad": round(float(scene.head), 5),
                        "estimated_xyyaw": scene.estimated.round(5).tolist(),
                        "obstacle_clearance_evaluator_only_m": round(float(
                            scene.obstacle_clearance()[0]), 4)})
        return None
    desired_head = None
    if mode == "timed_chest":
        scene.wait(safety_probe=sample)
    else:
        target_bearing = math.atan2(expected_xy[1] - scene.estimated[1],
                                    expected_xy[0] - scene.estimated[0])
        desired_head = wrap(target_bearing - scene.estimated[2])
        joint = scene.model.joint("head_yaw")
        lower, upper = scene.model.jnt_range[joint.id]
        desired_head = float(np.clip(desired_head, lower, upper))
        scene.look(desired_head, camera, safety_probe=sample)
    rgb, depth = scene.capture(camera, depth=True)
    if depth is None:
        raise RuntimeError("Expected depth-aligned view")
    rgb_path = folder / f"{mode}-final-rgb.png"
    depth_path = folder / f"{mode}-final-depth.npy"
    rgb.save(rgb_path)
    np.save(depth_path, depth)
    evidence = frame_evidence(scene, vision, exemplar, camera, rgb, depth,
                              expected_xy)
    evidence.update({
        "mode": mode, "requested_head_yaw_rad": round(desired_head, 5)
        if desired_head is not None else None,
        "action_samples": samples,
        "rgb": str(rgb_path.relative_to(ROOT)), "rgb_sha256": sha(rgb_path),
        "depth": str(depth_path.relative_to(ROOT)), "depth_sha256": sha(depth_path),
        "minimum_clearance_during_action_evaluator_only_m": round(float(
            min(x["obstacle_clearance_evaluator_only_m"] for x in samples)), 4),
    })
    return evidence, frames


def render_episode_and_views(episode: Path, branch_frames: dict[str, list[np.ndarray]],
                             results: dict[str, dict], output: Path) -> int:
    source = cv2.VideoCapture(str(episode))
    if not source.isOpened():
        raise RuntimeError(f"Cannot decode episode {episode}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             4., (1600, 520))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot encode comparison {output}")
    count = 0
    while True:
        ok, frame = source.read()
        if not ok:
            break
        strip = np.zeros((40, 1600, 3), dtype=np.uint8)
        cv2.putText(strip, "Owner task before view recovery | actor camera + evaluator map",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .72, (245, 245, 245), 2)
        writer.write(np.vstack([strip, frame]))
        count += 1
    source.release()
    length = max(len(x) for x in branch_frames.values())
    for index in range(length + 8):
        left = branch_frames["timed_chest"][min(index, len(branch_frames["timed_chest"]) - 1)]
        right = branch_frames["head_lower"][min(index, len(branch_frames["head_lower"]) - 1)]
        if left.shape != (480, 640, 3) or right.shape != (480, 640, 3):
            raise ValueError("Unexpected actor view dimensions")
        panel = np.zeros((480, 320, 3), dtype=np.uint8)
        for j, mode in enumerate(("timed_chest", "head_lower")):
            row = results[mode]
            cv2.putText(panel, mode, (12, 70 + 170 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
            cv2.putText(panel, f"target pixels {row['true_target_pixel_area_evaluator_only']}",
                        (12, 97 + 170 * j), cv2.FONT_HERSHEY_SIMPLEX, .52,
                        (190, 190, 190), 1)
            cv2.putText(panel, f"fresh candidate {bool(row['actor_fresh_metric_candidate'])}",
                        (12, 124 + 170 * j), cv2.FONT_HERSHEY_SIMPLEX, .52,
                        (190, 190, 190), 1)
        strip = np.zeros((40, 1600, 3), dtype=np.uint8)
        cv2.putText(strip, "Matched 0.4 s post-stop: chest wait | head yaw to remembered bearing",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .68,
                    (245, 245, 245), 2)
        writer.write(np.vstack([strip, np.hstack([cv2.cvtColor(left, cv2.COLOR_RGB2BGR),
                                                 cv2.cvtColor(right, cv2.COLOR_RGB2BGR),
                                                 panel])]))
        count += 1
    writer.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[86])
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    old_out = loop.OUT
    vision = loop.Vision("generic")
    teaching = loop.Room()
    try:
        teaching.teach()
        taught_rgb, _ = teaching.capture("upper_rgb", depth=False)
        truth = teaching.truth_box("TomatoSauce", "upper_rgb")
        if truth is None:
            raise ValueError("Teaching object not visible")
        exemplar = vision.feature([taught_rgb.crop(truth)])[0]
    finally:
        teaching.close()
    summaries = []
    try:
        for seed in args.seeds:
            folder = out / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            loop.OUT = folder
            scene = NativeShapedRoom()
            try:
                present = loop.run_episode(
                    scene, vision, exemplar, seed=seed, absent=False,
                    config=CONFIG, scenario=SCENARIO, observer_video=True)
                episode = folder / (f"{SCENARIO}-seed{seed}-0-{CONFIG}-none")
                trace = json.loads((episode / "trace.json").read_text())
                actor_memory = next((entry["fresh_target_xy_m"]
                                     for entry in reversed(trace["events"])
                                     if entry.get("fresh_target_xy_m") is not None), None)
                if actor_memory is None:
                    raise RuntimeError(f"No actor image-derived target memory: {seed}")
                memory = np.asarray(actor_memory, dtype=float)
                base = snapshot(scene)
                branches = {}
                branch_frames = {}
                for mode in CAMERAS:
                    restore(scene, base)
                    result, frames = branch(scene, mode, seed, memory, vision,
                                            exemplar, folder)
                    branches[mode] = result
                    branch_frames[mode] = frames
                movie = folder / "post-stop-view-comparison.mp4"
                video_frames = render_episode_and_views(
                    episode / "episode.mp4", branch_frames, branches, movie)
                restore(scene, base)
                absent = loop.run_episode(
                    scene, vision, exemplar, seed=seed, absent=True,
                    config=CONFIG, scenario=SCENARIO, observer_video=True)
                absent_episode = folder / (f"{SCENARIO}-seed{seed}-1-{CONFIG}-none")
                summaries.append({
                    "seed": seed, "present_owner_outcome": present["outcome"],
                    "present_episode_trace": str((episode / "trace.json").relative_to(ROOT)),
                    "present_episode_trace_sha256": sha(episode / "trace.json"),
                    "present_episode_video_sha256": sha(episode / "episode.mp4"),
                    "actor_remembered_xy_m": memory.round(5).tolist(),
                    "stop_time_s": round(base["time_s"], 4),
                    "stop_estimated_pose_xyyaw": base["estimated"].round(5).tolist(),
                    "stop_true_xy_evaluator_only_m": base["actual"][:2].round(5).tolist(),
                    "branches": branches,
                    "comparison_video": str(movie.relative_to(ROOT)),
                    "comparison_video_sha256": sha(movie),
                    "comparison_video_frames": video_frames,
                    "absent_owner_outcome": absent["outcome"],
                    "absent_episode_trace": str((absent_episode / "trace.json").relative_to(ROOT)),
                    "absent_episode_trace_sha256": sha(absent_episode / "trace.json"),
                    "absent_episode_video_sha256": sha(absent_episode / "episode.mp4"),
                    "absent_no_look_without_memory": True,
                })
                print(seed, present["outcome"], branches["timed_chest"][
                    "actor_fresh_metric_candidate"], branches["head_lower"][
                        "actor_fresh_metric_candidate"], absent["outcome"])
            finally:
                scene.close()
    finally:
        loop.OUT = old_out
    report = {"schema": "robot-learning-lab.native-owner-post-stop-views.v1",
              "task": "T-theejd", "owner_actor_code_sha256": sha(Path(loop.__file__)),
              "native_adapter_code_sha256": sha(ROOT / "experiments/ape_high_level/native_owner_bridge/replay_owner_packets.py"),
              "analysis_code_sha256": sha(Path(__file__)),
              "native_qpos_sha256": sha(SOURCE_QPOS),
              "native_odometry_sha256": sha(SOURCE_ODOM),
              "scope": "Same-state 0.4 s chest wait versus lower-head look using last actor image-derived target coordinate. Counterfactual after unchanged furnished owner actor; synthetic body and camera geometry; no physical sensor or autonomous exact-item claim.",
              "cases": summaries}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
