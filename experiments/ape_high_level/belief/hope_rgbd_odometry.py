"""Replay measured RGB-D camera motion and its effect on object search.

ORB + depth + PnP estimates each next camera pose from the previous frame.
Only the first published HOPE pose anchors the estimate; later published poses
and projected object boxes are evaluator-only. This is a robot-arm tabletop
camera, not the 50 cm ape or its future encoder/IMU odometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from ..grounding.hope_video_scan import DEFAULT_VIDEO, _plane_world_point
    from ..grounding.hope_image_find import _iou
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grounding"))
    from hope_video_scan import DEFAULT_VIDEO, _plane_world_point
    from hope_image_find import _iou


def _published_world_to_camera(video: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    camera = json.loads((video / f"{frame:04d}.json").read_text())["camera"]
    world_to_camera = np.asarray(camera["extrinsics"], dtype=np.float64).copy()
    world_to_camera[:3, 3] *= 100  # HOPE extrinsic translation is in metres.
    return world_to_camera, np.asarray(camera["intrinsics"], dtype=np.float64)


def _camera_error(estimate: np.ndarray, published: np.ndarray) -> tuple[float, float]:
    e, p = np.linalg.inv(estimate), np.linalg.inv(published)
    position_cm = float(np.linalg.norm(e[:3, 3] - p[:3, 3]))
    rotation = e[:3, :3].T @ p[:3, :3]
    angle_deg = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    return position_cm, angle_deg


def _camera_dict(estimate: np.ndarray, intrinsics: np.ndarray) -> dict:
    extrinsics = estimate.copy()
    extrinsics[:3, 3] /= 100
    return {"intrinsics": intrinsics.tolist(), "extrinsics": extrinsics.tolist()}


def _scored_iou(candidate: tuple[float, dict] | None,
                projected_box: list[float] | None) -> float | None:
    if candidate is None:
        return None
    if projected_box is None:
        return 0.0
    return _iou(candidate[1]["box"], projected_box)


def _evaluate_search(scan: dict, poses: dict[int, np.ndarray],
                     intrinsics: np.ndarray, pose_health: dict[int, bool]) -> dict:
    teaching = scan["teach"]["known_plane_world_points_cm"]
    entries = []
    for row in scan["rows"]:
        target, frame = row["target"], row["frame"]
        candidates = [p for p in scan["proposals"] if p["frame"] == frame]
        reference = np.asarray(teaching[target], dtype=np.float64)
        scored = []
        for candidate in candidates:
            known = candidate["known_plane_world_xyz_cm"]
            if known is not None:
                scored.append((float(np.linalg.norm(np.asarray(known) - reference)), candidate))
        scored.sort(key=lambda item: item[0])
        exact = scored[0] if scored else None
        estimated = []
        if frame in poses:
            camera = _camera_dict(poses[frame], intrinsics)
            for candidate in candidates:
                point = _plane_world_point(candidate["box"], camera, scan["known_table_plane_z_cm"])
                if point is not None:
                    estimated.append((float(np.linalg.norm(np.asarray(point) - reference)), candidate))
            estimated.sort(key=lambda item: item[0])
        chosen = estimated[0] if estimated else None
        entries.append({"target": target, "frame": frame,
                        "pose_available": pose_health.get(frame, False),
                        "projected_in_frame": row["projected_box"] is not None,
                        "target_proposed": row["proposal_target_iou"] >= .25,
                        "candidate_count": len(candidates),
                        "exact_nearest_distance_cm": exact[0] if exact else None,
                        "exact_nearest_iou": _scored_iou(exact, row["projected_box"]),
                        "odometry_nearest_distance_cm": chosen[0] if chosen else None,
                        "odometry_nearest_iou": _scored_iou(chosen, row["projected_box"])})
    visible = [e for e in entries if e["projected_in_frame"]]
    out_of_frame = [e for e in entries if not e["projected_in_frame"]]
    sweep = {}
    for radius in [3, 5, 10, 15, 20]:
        sweep[str(radius)] = {}
        for source in ["exact", "odometry"]:
            accepted = [e for e in entries if e[f"{source}_nearest_distance_cm"] is not None
                        and e[f"{source}_nearest_distance_cm"] <= radius]
            trusted = [e for e in accepted if e["pose_available"]]
            sweep[str(radius)][source] = {
                "accepted_visible": sum(e["projected_in_frame"] for e in accepted),
                "correct_visible": sum(e["projected_in_frame"] and
                                       e[f"{source}_nearest_iou"] >= .25 for e in accepted),
                "wrong_visible": sum(e["projected_in_frame"] and
                                     e[f"{source}_nearest_iou"] < .25 for e in accepted),
                "accepted_out_of_frame": sum(not e["projected_in_frame"] for e in accepted),
                "accepted_with_untrusted_pose": sum(not e["pose_available"] for e in accepted)
                if source == "odometry" else 0,
                "trusted_pose_correct_visible": sum(e["projected_in_frame"] and
                                                    e[f"{source}_nearest_iou"] >= .25
                                                    for e in trusted)
                if source == "odometry" else None,
                "trusted_pose_wrong_or_out_of_frame": sum(
                    not e["projected_in_frame"] or e[f"{source}_nearest_iou"] < .25
                    for e in trusted) if source == "odometry" else None,
            }
    return {"rows": entries, "summary": {
        "projected_in_frame": len(visible),
        "projected_out_of_frame": len(out_of_frame),
        "target_proposed_in_frame": sum(e["target_proposed"] for e in visible),
        "exact_nearest_correct_in_frame": sum(e["exact_nearest_iou"] is not None
                                              and e["exact_nearest_iou"] >= .25 for e in visible),
        "odometry_nearest_correct_in_frame": sum(e["odometry_nearest_iou"] is not None
                                                 and e["odometry_nearest_iou"] >= .25 for e in visible),
        "radius_sweep_cm": sweep}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-motion-start", type=int)
    parser.add_argument("--skip-motion-end", type=int)
    args = parser.parse_args()
    if (args.skip_motion_start is None) != (args.skip_motion_end is None):
        raise ValueError("Specify both skip-motion bounds or neither")
    scan = json.loads(args.scan.read_text())
    frames = sorted(int(p.stem.split("_")[0]) for p in args.video.glob("*_rgb.jpg"))
    if frames != list(range(len(frames))):
        raise ValueError("Need consecutive RGB frames starting at 0")
    if not all((args.video / f"{frame:04d}_depth.png").exists() for frame in frames[:-1]):
        raise ValueError("Need depth aligned to every source RGB frame")
    published0, intrinsics = _published_world_to_camera(args.video, 0)
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    image = cv2.imread(str(args.video / "0000_rgb.jpg"), cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    estimate = published0.copy()
    poses = {0: estimate.copy()}
    trajectory = [{"frame": 0, "position_error_cm": 0.0, "angle_error_deg": 0.0}]
    pose_integrity = True
    for frame in frames[1:]:
        new_image = cv2.imread(str(args.video / f"{frame:04d}_rgb.jpg"), cv2.IMREAD_GRAYSCALE)
        new_keypoints, new_descriptors = orb.detectAndCompute(new_image, None)
        depth = cv2.imread(str(args.video / f"{frame - 1:04d}_depth.png"), cv2.IMREAD_UNCHANGED)
        matches = ([one for one, two in matcher.knnMatch(descriptors, new_descriptors, k=2)
                    if one.distance < .75 * two.distance]
                   if descriptors is not None and new_descriptors is not None else [])
        points_3d, points_2d = [], []
        for match in matches:
            x, y = keypoints[match.queryIdx].pt
            u, v = new_keypoints[match.trainIdx].pt
            depth_cm = float(depth[round(y), round(x)]) / 10
            if 7 < depth_cm < 300:
                points_3d.append([(x - intrinsics[0, 2]) * depth_cm / intrinsics[0, 0],
                                  (y - intrinsics[1, 2]) * depth_cm / intrinsics[1, 1], depth_cm])
                points_2d.append([u, v])
        solved, rotation_vector, translation, inliers = (False, None, None, None)
        if len(points_3d) >= 6:
            solved, rotation_vector, translation, inliers = cv2.solvePnPRansac(
                np.asarray(points_3d, np.float32), np.asarray(points_2d, np.float32),
                intrinsics, None, iterationsCount=100, reprojectionError=3,
                confidence=.99, flags=cv2.SOLVEPNP_EPNP)
        inlier_count = len(inliers) if inliers is not None else 0
        motion_skipped = (args.skip_motion_start is not None
                          and args.skip_motion_start <= frame <= args.skip_motion_end)
        if inlier_count < 12 or motion_skipped:
            pose_integrity = False
        if solved and inlier_count >= 12 and not motion_skipped:
            step = np.eye(4)
            step[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            step[:3, 3] = translation[:, 0]
            estimate = step @ estimate
        # After a skipped motion update, this continued pose is a deliberately
        # unsafe counterfactual. The controller must see pose_integrity=False.
        poses[frame] = estimate.copy()
        published, _ = _published_world_to_camera(args.video, frame)
        error = _camera_error(estimate, published)
        trajectory.append({"frame": frame, "ratio_matches": len(matches),
                           "depth_correspondences": len(points_3d),
                           "pnp_inliers": inlier_count, "motion_skipped": motion_skipped,
                           "pose_available": pose_integrity,
                           "position_error_cm": error[0], "angle_error_deg": error[1]})
        keypoints, descriptors = new_keypoints, new_descriptors
    valid = [row for row in trajectory if row.get("pose_available", True)]
    search = _evaluate_search(scan, poses, intrinsics,
                              {row["frame"]: row.get("pose_available", True)
                               for row in trajectory})
    result = {"scope": "Measured HOPE robot-arm RGB-D odometry, anchored only at frame 0. "
                       "Published poses and projected mesh boxes are evaluator-only after anchor. "
                       "After a motion dropout, continued poses are unsafe counterfactuals and marked "
                       "unavailable to high-level control. No ape gait, IMU, encoder fusion, "
                       "relocalization, moving objects or physical sensor.",
              "frames": len(frames), "method": "ORB + source-frame metric depth + PnP RANSAC",
              "motion_dropout_frames": [args.skip_motion_start, args.skip_motion_end]
              if args.skip_motion_start is not None else None,
              "tracking": {"pose_available_frames": len(valid),
                           "first_lost_frame": next((r["frame"] for r in trajectory
                                                     if r.get("pose_available") is False), None),
                           "final_continued_position_error_cm": trajectory[-1]["position_error_cm"],
                           "final_continued_angle_error_deg": trajectory[-1]["angle_error_deg"],
                           "median_position_error_cm": float(np.median([
                               r["position_error_cm"] for r in trajectory])),
                           "max_position_error_cm": max(r["position_error_cm"] for r in trajectory)},
              "trajectory": trajectory, "search": search}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tracking": result["tracking"],
                      "search": search["summary"]}, indent=2))


if __name__ == "__main__":
    main()
