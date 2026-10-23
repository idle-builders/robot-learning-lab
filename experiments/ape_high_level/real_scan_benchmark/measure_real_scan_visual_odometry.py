"""Measure route-local motion from aligned real-scan RGB-D, with pose truth held out.

This is a spatial replay, not a timed IMU/joint sensor or closed owner episode.
The actor-side acceptance rule is frozen in Idle T-thar6h before this run.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes"
INDEX = ASSETS / "real_capture_33_moving_route_index.json"
OUTPUT = ASSETS / "real_capture_rgbd_odometry_v2.json"
STRIDE = 4


def rgb_depth(root, frame, depth_root=None):
    image_path = root / frame["image"]
    if image_path.is_file():
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
    else:
        video = cv2.VideoCapture(str(root / "ape_height_route.mp4"))
        if not video.isOpened():
            raise FileNotFoundError(image_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame["frame"])
        ok, bgr = video.read()
        video.release()
        if not ok:
            raise RuntimeError(f"video frame unavailable: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_folder = depth_root if depth_root is not None else root
    depth_name = frame.get("depth_mm_png", "depth_%04d.png" % frame["frame"])
    depth = np.asarray(Image.open(depth_folder / depth_name), dtype=np.float32) / 1000.0
    if depth.shape != rgb.shape[:2]:
        raise ValueError("RGB and depth resolutions differ")
    return rgb, depth


def intrinsics(trace):
    width, height = trace["camera_resolution_px"]
    fx = width / (2 * math.tan(math.radians(trace["camera_hfov_deg"]) / 2))
    return np.array([[fx, 0, (width - 1) / 2],
                     [0, fx, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)


def point_cloud(depth, k, spacing=8):
    height, width = depth.shape
    vv, uu = np.mgrid[0:height:spacing, 0:width:spacing]
    z = depth[::spacing, ::spacing]
    valid = (z > 0.3) & (z < 8.0)
    z = z[valid]
    x = (uu[valid] - k[0, 2]) * z / k[0, 0]
    y = (vv[valid] - k[1, 2]) * z / k[1, 1]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
    return cloud.voxel_down_sample(0.06)


def feature_packet(rgb, depth, orb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return {"keypoints": keypoints, "descriptors": descriptors, "depth": depth}


def visual_transform(source, target, k, matcher):
    first, second = source["descriptors"], target["descriptors"]
    if first is None or second is None or len(first) < 2 or len(second) < 2:
        return None, {"matches": 0, "depth_matches": 0, "inliers": 0}
    ratios = matcher.knnMatch(first, second, k=2)
    matches = [pair[0] for pair in ratios
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    xyz, uv = [], []
    for match in matches:
        u, v = source["keypoints"][match.queryIdx].pt
        u0, v0 = int(round(u)), int(round(v))
        z = float(source["depth"][v0, u0])
        if not 0.3 < z < 8.0:
            continue
        xyz.append(((u - k[0, 2]) * z / k[0, 0],
                    (v - k[1, 2]) * z / k[1, 1], z))
        uv.append(target["keypoints"][match.trainIdx].pt)
    stats = {"matches": len(matches), "depth_matches": len(xyz), "inliers": 0}
    if len(xyz) < 8:
        return None, stats
    xyz = np.asarray(xyz, dtype=np.float32)
    uv = np.asarray(uv, dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz, uv, k, None, iterationsCount=300, reprojectionError=3.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None, stats
    indexes = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(xyz[indexes], uv[indexes], k, None, rvec, tvec)
    projected, _ = cv2.projectPoints(xyz[indexes], rvec, tvec, k, None)
    residual = np.linalg.norm(projected[:, 0] - uv[indexes], axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec[:, 0]
    stats.update({"inliers": len(indexes),
                  "inlier_ratio": len(indexes) / len(xyz),
                  "median_reprojection_px": float(np.median(residual))})
    return transform, stats


def angle_deg(rotation):
    value = (np.trace(rotation) - 1) / 2
    return math.degrees(math.acos(float(np.clip(value, -1, 1))))


def camera_pose(frame, eye_height):
    if "camera_to_world_rotation" in frame:
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(frame["camera_to_world_rotation"], dtype=float)
        pose[:3, 3] = np.asarray(frame["camera_center_m"], dtype=float)
        return pose
    yaw = frame["yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    world_from_habitat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    habitat_from_opencv = np.diag([1, -1, -1])
    pose = np.eye(4)
    pose[:3, :3] = world_from_habitat @ habitat_from_opencv
    pose[:3, 3] = np.asarray(frame["position_m"]) + [0, eye_height, 0]
    return pose


def pose_errors(estimate, truth):
    return {"translation_error_m": float(np.linalg.norm(
                estimate[:3, 3] - truth[:3, 3])),
            "rotation_error_deg": angle_deg(estimate[:3, :3].T @ truth[:3, :3])}


def run_scene(row):
    trace_path = ROOT / row.get("sensor_trace", row["trace"])
    trace = json.loads(trace_path.read_text())
    frames = trace["frames"]
    folder = trace_path.parent
    depth_folder = folder / "depth_replay" if row["scene_id"].startswith("hm3d_example/") else None
    k = intrinsics(trace)
    orb = cv2.ORB_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    cache = {}

    def observed(index):
        if index not in cache:
            rgb, depth = rgb_depth(folder, frames[index], depth_folder)
            cache[index] = (feature_packet(rgb, depth, orb), point_cloud(depth, k))
        return cache[index]

    pairs = []
    for a in range(0, len(frames) - STRIDE, STRIDE):
        b = a + STRIDE
        (source, cloud_a), (target, cloud_b) = observed(a), observed(b)
        forward, forward_stats = visual_transform(source, target, k, matcher)
        reverse, reverse_stats = visual_transform(target, source, k, matcher)
        truth = np.linalg.inv(camera_pose(frames[b], trace["camera_height_above_nav_m"])) @ \
                camera_pose(frames[a], trace["camera_height_above_nav_m"])
        item = {
            "from_frame": a, "to_frame": b,
            "source_image": str(folder / frames[a]["image"])
                if (folder / frames[a]["image"]).is_file() else None,
            "target_image": str(folder / frames[b]["image"])
                if (folder / frames[b]["image"]).is_file() else None,
            "source_video": str(ROOT / row.get("clean_video", row["video"])),
            "true_translation_m": float(np.linalg.norm(truth[:3, 3])),
            "rgbd_forward": forward_stats,
            "rgbd_reverse": reverse_stats,
        }
        if forward is not None:
            item["rgbd_error"] = pose_errors(forward, truth)
        if forward is not None and reverse is not None:
            closure = reverse @ forward
            item["reciprocal_translation_m"] = float(np.linalg.norm(closure[:3, 3]))
            item["reciprocal_rotation_deg"] = angle_deg(closure[:3, :3])
        item["rgbd_accepted"] = bool(
            forward is not None and reverse is not None and
            forward_stats["inliers"] >= 30 and reverse_stats["inliers"] >= 30 and
            forward_stats["inlier_ratio"] >= 0.25 and
            reverse_stats["inlier_ratio"] >= 0.25 and
            forward_stats["median_reprojection_px"] <= 2.5 and
            reverse_stats["median_reprojection_px"] <= 2.5 and
            item["reciprocal_translation_m"] <= 0.08 and
            item["reciprocal_rotation_deg"] <= 5.0)
        if forward is not None:
            error = item["rgbd_error"]
            item["rgbd_correct"] = bool(error["translation_error_m"] <= 0.08 and
                                        error["rotation_error_deg"] <= 5.0)
        else:
            item["rgbd_correct"] = False
        if len(cloud_a.points) >= 100 and len(cloud_b.points) >= 100:
            icp = o3d.pipelines.registration.registration_icp(
                cloud_a, cloud_b, 0.55, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            item["depth_only_icp"] = {
                "fitness": float(icp.fitness),
                "inlier_rmse_m": float(icp.inlier_rmse),
                **pose_errors(icp.transformation, truth),
            }
        pairs.append(item)
    return {"scene_id": row["scene_id"], "physical_space_id": row["physical_space_id"],
            "owner_controller_split": row["owner_controller_split"],
            "appearance_screen": row["appearance_screen"],
            "splat_mesh_median_depth_difference_m": row.get("splat_mesh_median_depth_difference_m"),
            "pairs": pairs}


def aggregate(scenes):
    pairs = [pair for scene in scenes for pair in scene["pairs"]]
    accepted = [pair for pair in pairs if pair["rgbd_accepted"]]
    return {
        "scene_variants": len(scenes),
        "physical_spaces": len({scene["physical_space_id"] for scene in scenes}),
        "motion_pairs": len(pairs),
        "rgbd_accepted": len(accepted),
        "rgbd_correct_accepted": sum(pair["rgbd_correct"] for pair in accepted),
        "rgbd_wrong_accepted": sum(not pair["rgbd_correct"] for pair in accepted),
        "rgbd_rejected": len(pairs) - len(accepted),
        "depth_only_icp_correct": sum(
            pair.get("depth_only_icp", {}).get("translation_error_m", math.inf) <= 0.08 and
            pair.get("depth_only_icp", {}).get("rotation_error_deg", math.inf) <= 5
            for pair in pairs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", help="one scene_id for a bounded smoke run")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    index = json.loads(INDEX.read_text())
    rows = [row for row in index["scenes"] if "metric_depth_frame_pattern" in row]
    if args.scene:
        rows = [row for row in rows if row["scene_id"] == args.scene]
        if len(rows) != 1:
            raise ValueError("scene not found among aligned RGB-D routes")
    results = []
    for row in rows:
        result = run_scene(row)
        results.append(result)
        accepted = sum(pair["rgbd_accepted"] for pair in result["pairs"])
        correct = sum(pair["rgbd_accepted"] and pair["rgbd_correct"]
                      for pair in result["pairs"])
        print(row["scene_id"], len(result["pairs"]), "accepted", accepted,
              "correct_accepted", correct, flush=True)
    record = {
        "schema": "ape.real-scan-rgbd-odometry.v2",
        "source_index": str(INDEX), "frame_stride": STRIDE,
        "actor_gate": "bidirectional >=30 inliers, >=0.25 ratio, <=2.5px median reprojection, <=0.08m/5deg reciprocal closure",
        "evaluator_success": "<=0.08m translation and <=5deg rotation error against camera pose truth",
        "intrinsics_note": "Per-route pinhole FOV (90deg Replica/HM3D, 91.31deg Habitat-GS); metric depth is treated as optical-axis distance in this replay.",
        "limits": "Spatial replay of static scenes; no measured clock, IMU, joint, contact, person, object or closed-loop owner task.",
        "versions": {"opencv": cv2.__version__, "open3d": o3d.__version__},
        "scenes": results,
    }
    record["summary"] = aggregate(results)
    record["by_source"] = {
        source: aggregate([scene for scene in results if scene["scene_id"].startswith(source + "/")])
        for source in sorted({scene["scene_id"].split("/", 1)[0] for scene in results})
    }
    record["by_split"] = {
        split: aggregate([scene for scene in results if scene["owner_controller_split"] == split])
        for split in sorted({scene["owner_controller_split"] for scene in results})
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
