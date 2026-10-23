#!/usr/bin/env python3
from __future__ import annotations
"""Export a compact visual/trace bundle from an upstream i4h Arena HDF5 run."""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def export(source: Path, destination: Path, fps: float, rollout_log: Path | None,
           video_stride: int) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    frames_dir = destination / "stills"
    frames_dir.mkdir(exist_ok=True)
    if video_stride < 1:
        raise ValueError("video stride must be at least one")
    info = {"source_bytes": source.stat().st_size, "trace_fps": fps,
            "video_stride": video_stride, "video_fps": fps / video_stride, "datasets": {}}
    events = []
    if rollout_log and rollout_log.exists():
        pattern = re.compile(r"push cart trajectory: step=(\d+) cart=\[([^]]+)\].*?y_progress_m=([0-9.]+).*?xy_distance_m=([0-9.]+).*?pushed=(\w+).*?near_target=(\w+)")
        for line in rollout_log.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                events.append({"step": step, "time_s": step / fps, "cart_xyz": match.group(2),
                               "y_progress_m": float(match.group(3)), "xy_distance_m": float(match.group(4)),
                               "pushed": match.group(5) == "True", "near_target": match.group(6) == "True"})
    selected = None
    with h5py.File(source, "r") as data:
        def inspect(name, obj):
            nonlocal selected
            if isinstance(obj, h5py.Dataset):
                shape = list(obj.shape)
                info["datasets"][name] = {"shape": shape, "dtype": str(obj.dtype)}
                if (selected is None and "/obs/" in ("/" + name) and
                    len(shape) == 4 and shape[-1] in (3, 4) and
                    ("rgb" in name.lower() or "cam" in name.lower())):
                    selected = name
        data.visititems(inspect)
        if selected is None:
            raise RuntimeError("No camera RGB dataset found under data/<demo>/obs; see hdf5-index.json")
        ds = data[selected]
        n, height, width, channels = ds.shape
        if n < 1:
            raise RuntimeError("Camera dataset is empty")
        info.update(camera_dataset=selected, frame_count=n, width=width, height=height)
        still_targets = {"start": 0}
        if events:
            first_xyz = np.fromstring(events[0]["cart_xyz"], sep=",")
            motion = next((event for event in events
                           if np.linalg.norm(np.fromstring(event["cart_xyz"], sep=",") - first_xyz) > 0.03), None)
            goal = next((event for event in events if event["near_target"]), events[-1])
            if motion:
                source_index = round((motion["step"] / max(1, events[-1]["step"])) * (n - 1))
                still_targets["first-cart-motion"] = round(source_index / video_stride) * video_stride
            source_index = round((goal["step"] / max(1, events[-1]["step"])) * (n - 1))
            still_targets["fallback-or-native-goal"] = round(source_index / video_stride) * video_stride
        still_targets = {label: min(n - 1, max(0, i)) for label, i in still_targets.items()}
        # Extract sparse PNGs up front. These remain inspectable if video encoding
        # is interrupted before a regular MP4 index can be finalized.
        sparse_stride = max(1, int(np.ceil(n / 20)))
        sparse_indices = set(range(0, n, sparse_stride))
        sparse_indices.update(min(n - 1, max(0, i)) for i in still_targets.values())
        sparse_indices = sorted(sparse_indices)[:20]
        info["sparse_still_count"] = len(sparse_indices)
        info["still_frames"] = still_targets
        for i in sparse_indices:
            frame = np.asarray(ds[i])
            if frame.dtype != np.uint8:
                frame = np.clip(frame * (255 if frame.max() <= 1 else 1), 0, 255).astype(np.uint8)
            Image.fromarray(np.ascontiguousarray(frame[..., :3])).save(
                frames_dir / f"sample-frame-{i:06d}.png")
        for label, i in still_targets.items():
            frame = np.asarray(ds[i])
            if frame.dtype != np.uint8:
                frame = np.clip(frame * (255 if frame.max() <= 1 else 1), 0, 255).astype(np.uint8)
            Image.fromarray(np.ascontiguousarray(frame[..., :3])).save(
                frames_dir / f"{label}-frame-{i:06d}.png")
        # Convert and stream one frame at a time; the large source recording stays on worker.
        video = destination / "cart-camera.mp4"
        if not video.exists():
            partial_video = destination / "cart-camera.partial.mp4"
            codec = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps / video_stride), "-i", "-", "-an",
                "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "28", "-g", "10",
                "-flush_packets", "1", "-pix_fmt", "yuv420p",
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                str(partial_video)
            ], stdin=subprocess.PIPE)
            try:
                for i in range(0, n, video_stride):
                    frame = np.asarray(ds[i])
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * (255 if frame.max() <= 1 else 1), 0, 255).astype(np.uint8)
                    frame = np.ascontiguousarray(frame[..., :3])
                    assert codec.stdin is not None
                    codec.stdin.write(frame.tobytes())
            finally:
                if codec.stdin:
                    codec.stdin.close()
            if codec.wait() != 0:
                raise RuntimeError("ffmpeg failed while encoding camera dataset")
            partial_video.replace(video)

        # Preserve the native post-step state and pre-step action traces in long-form CSV.
        with (destination / "state-action-trace.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_s", "kind", "dataset", "component", "value"])
            root = data.get("data")
            if root is not None:
                for demo_name in sorted(root.keys()):
                    demo = root[demo_name]
                    for kind in ("actions", "processed_actions", "states"):
                        group = demo.get(kind)
                        if kind == "actions" and group is None and "obs/actions" in demo:
                            group = demo["obs/actions"]
                            if isinstance(group, h5py.Dataset):
                                arrays = [("obs/actions", group)]
                            else:
                                arrays = []
                        else:
                            arrays = []
                            if isinstance(group, h5py.Dataset):
                                arrays = [(kind, group)]
                            elif group is not None:
                                group.visititems(lambda name, obj: arrays.append((name, obj)))
                        for name, obj in arrays:
                            if not isinstance(obj, h5py.Dataset) or not obj.shape:
                                continue
                            arr = obj
                            if len(arr.shape) == 1:
                                for step, value in enumerate(arr):
                                    writer.writerow([step, step / fps, kind, name, 0, float(value)])
                            else:
                                for step in range(arr.shape[0]):
                                    row = np.asarray(arr[step]).reshape(-1)
                                    for component, value in enumerate(row):
                                        writer.writerow([step, step / fps, kind, name, component, float(value)])

    # Keep event labels synchronized to the actual Arena timestep in its rollout log.
    (destination / "cart-events.json").write_text(json.dumps(events, indent=2))
    info["event_count"] = len(events)
    (destination / "hdf5-index.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video-stride", type=int, default=3,
                        help="sample one camera frame every N native timesteps")
    parser.add_argument("--rollout-log", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.recording, args.output, args.fps, args.rollout_log,
                           args.video_stride), indent=2))


if __name__ == "__main__":
    main()
