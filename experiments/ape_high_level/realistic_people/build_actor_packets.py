"""Package Isaac RGB-D and collision rays without evaluator-only identity labels.

This is an offline observation interface, not a closed-loop actor run. It
deliberately omits character paths, hit prims, world positions and move tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_frames(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        okay, bgr = capture.read()
        if not okay:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 12:
        raise RuntimeError(f"Expected 12 RGB frames in {path}, got {len(frames)}")
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence
    ranges = json.loads((evidence / "range-fan-trace.json").read_text())["frames"]
    motion = json.loads((evidence / "character-motion-trace.json").read_text())
    if len(ranges) != 12 or len(motion) != 12:
        raise RuntimeError("Expected twelve aligned range and evaluator frames")
    times = np.array([frame["timeline_time_s"] for frame in ranges], dtype=np.float64)
    if any(frame["frame"] != index or motion[index]["frame"] != index or
           abs(motion[index]["timeline_time_s"] - times[index]) > 1e-6
           for index, frame in enumerate(ranges)):
        raise RuntimeError("Range and RGB-D frame clocks do not align")
    if np.any(np.diff(times) <= 0):
        raise RuntimeError("Nonmonotonic observation times")
    heights = np.array([plane["height_above_floor_m"]
                        for plane in ranges[0]["planes"]], dtype=np.float32)
    angles = np.array([beam["angle_deg"]
                       for beam in ranges[0]["planes"][0]["beams"]], dtype=np.float32)
    range_m = np.full((12, len(heights), len(angles)), np.nan, dtype=np.float32)
    for frame_index, frame in enumerate(ranges):
        for plane_index, plane in enumerate(frame["planes"]):
            if not np.isclose(plane["height_above_floor_m"], heights[plane_index]):
                raise RuntimeError("Range plane height changed")
            for beam_index, beam in enumerate(plane["beams"]):
                if not np.isclose(beam["angle_deg"], angles[beam_index]):
                    raise RuntimeError("Range beam angle changed")
                if beam["distance_m"] is not None:
                    range_m[frame_index, plane_index, beam_index] = beam["distance_m"]
    if not np.isfinite(range_m).any():
        raise RuntimeError("No PhysX range returns")
    manifest = {"schema": "ape-isaac-rgbd-range-v1",
                "scope": "fixed proxy camera; replayable observations, not an actor rollout",
                "clock": "Isaac timeline seconds; RGB, depth and range frame-aligned",
                "depth": "Isaac distance_to_camera radial range, stride-3 image samples",
                "range": "ideal PhysX collision ray fan; 5 degree beams; NaN=no hit within 8 m",
                "camera_height_above_floor_m": 0.45,
                "range_heights_above_floor_m": heights.tolist(),
                "range_height_offsets_from_camera_m": (heights - 0.45).tolist(),
                "range_angles_deg": angles.tolist(),
                "source_sha256": {"build_actor_packets.py": sha256(Path(__file__)),
                                  "range-fan-trace.json": sha256(
                                      evidence / "range-fan-trace.json")},
                "views": []}
    for fov in (70, 90):
        stem = f"ape_social_{fov}deg"
        rgb = video_frames(evidence / f"{stem}.mp4")
        depth = np.load(evidence / f"{stem}-depth-sequence-stride3.npz")["depth_m"]
        if depth.shape[0] != len(times) or rgb.shape[0] != len(times):
            raise RuntimeError("RGB/depth/range frame counts differ")
        height, width = rgb.shape[1:3]
        focal_px = width / (2 * math.tan(math.radians(fov / 2)))
        out = evidence / f"{stem}-actor-observations.npz"
        np.savez_compressed(out, timeline_time_s=times, rgb=rgb,
                            distance_to_camera_m=depth, range_distance_m=range_m,
                            range_angles_deg=angles,
                            range_heights_above_floor_m=heights,
                            camera_intrinsics_rgb=np.array([[focal_px, 0., width / 2],
                                                            [0., focal_px, height / 2],
                                                            [0., 0., 1.]], dtype=np.float32))
        manifest["views"].append({"fov_deg": fov, "file": out.name,
                                  "frames": len(times),
                                  "rgb_shape": list(rgb.shape),
                                  "depth_shape": list(depth.shape),
                                  "range_valid_beams": int(np.isfinite(range_m).sum()),
                                  "source_video_sha256": sha256(evidence / f"{stem}.mp4"),
                                  "source_depth_sha256": sha256(
                                      evidence / f"{stem}-depth-sequence-stride3.npz"),
                                  "packet_sha256": sha256(out)})
    (evidence / "actor-observation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
