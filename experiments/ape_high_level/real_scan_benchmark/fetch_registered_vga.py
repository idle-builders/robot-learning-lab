"""Fetch only pose-near real 640x480 frames from Apple's ranged ZIPs.

This keeps the measured low-resolution depth and poses from 3DOD and adds a
higher-resolution photographic RGB stream with its own intrinsics/timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/arkit_v1"
SOURCE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"


def timestamp(name: str) -> float:
    return float(Path(name).stem.split("_")[-1])


def nearest(names: list[str], t: float) -> str:
    return min(names, key=lambda n: abs(timestamp(n) - t))


def enrich(scene_dir: Path) -> None:
    record_path = scene_dir / "scene.json"
    record = json.loads(record_path.read_text())
    base = f"{SOURCE}/raw/{record['fold']}/{record['video_id']}"
    frame_info = []
    with RemoteZip(base + "/vga_wide.zip") as rgb_zip, \
            RemoteZip(base + "/vga_wide_intrinsics.zip") as intr_zip:
        rgb_names = [n for n in rgb_zip.namelist() if n.endswith(".png")]
        intr_names = [n for n in intr_zip.namelist() if n.endswith(".pincam")]
        for frame in record["selected_frames"]:
            target_t = frame["timestamp_s"]
            rgb_name = nearest(rgb_names, target_t)
            rgb_t = timestamp(rgb_name)
            intr_name = nearest(intr_names, rgb_t)
            if abs(rgb_t - target_t) > 0.05 or abs(timestamp(intr_name) - rgb_t) > 0.005:
                raise ValueError(f"{scene_dir.name}: VGA RGB/intrinsics too far from depth pose")
            folder = scene_dir / "frames" / f"{frame['index']:02d}"
            rgb_path = folder / "rgb_vga.png"
            intr_path = folder / "intrinsics_vga.pincam"
            if not rgb_path.exists():
                data = rgb_zip.read(rgb_name)
                if Image.open(BytesIO(data)).size != (640, 480):
                    raise ValueError(f"{scene_dir.name}: unexpected VGA resolution")
                rgb_path.write_bytes(data)
            if not intr_path.exists():
                intr_path.write_bytes(intr_zip.read(intr_name))
            frame_info.append({
                "index": frame["index"], "vga_timestamp_s": rgb_t,
                "offset_from_depth_pose_s": round(rgb_t - target_t, 4),
                "rgb_vga_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                "rgb_vga_source": f"{base}/vga_wide.zip#{rgb_name}",
            })
    record["vga_rgb_frames"] = frame_info
    record["vga_rgb_status"] = "captured_near_pose_not_exactly_synchronous"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{scene_dir.name}: {len(frame_info)} captured VGA frames, "
          f"max offset {max(abs(f['offset_from_depth_pose_s']) for f in frame_info):.3f}s",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    scenes = json.loads((ASSETS / "manifest.json").read_text())["scenes"][args.start:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(enrich, ASSETS / scene["video_id"]): scene["video_id"]
                   for scene in scenes}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"FAILED {futures[future]}: {exc}", flush=True)


if __name__ == "__main__":
    main()
