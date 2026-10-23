"""Stream one high-resolution recorded view per selected ARKitScenes room.

These are appearance references. The 3DOD RGB-D/pose stream is the synchronized
sensor source; a MOV frame is not claimed to be pixel-aligned to its pose.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/arkit_v1"
SOURCE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    scenes = json.loads((ASSETS / "manifest.json").read_text())["scenes"][:args.limit]
    for scene in scenes:
        scene_dir = ASSETS / scene["video_id"]
        record_path = scene_dir / "scene.json"
        record = json.loads(record_path.read_text())
        first = record["selected_frames"][0]["timestamp_s"]
        last = record["selected_frames"][-1]["timestamp_s"]
        seconds = round(0.5 * (last - first), 3)
        output = scene_dir / "hd_preview.jpg"
        if not output.exists():
            url = f"{SOURCE}/raw/{record['fold']}/{record['video_id']}/{record['video_id']}.mov"
            temporary = scene_dir / "hd_preview_raw.jpg"
            try:
                subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error",
                                "-ss", str(seconds), "-i", url, "-frames:v", "1",
                                "-q:v", "3", "-y", str(temporary)],
                               check=True, timeout=30)
                image = Image.open(temporary).convert("RGB")
                rotation = {"Up": None, "Down": Image.Transpose.ROTATE_180,
                            "Left": Image.Transpose.ROTATE_270,
                            "Right": Image.Transpose.ROTATE_90}[record["sky_direction"]]
                if rotation is not None:
                    image = image.transpose(rotation)
                image = ImageOps.contain(image, (1440, 1440))
                image.save(output, quality=88)
                temporary.unlink()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                temporary.unlink(missing_ok=True)
                print(f"FAILED {record['video_id']}: {exc}", flush=True)
                continue
        image = Image.open(output)
        record["hd_recorded_preview"] = {
            "path": str(output.relative_to(ROOT)), "movie_seek_s": seconds,
            "size_px": list(image.size), "pose_alignment": "unverified",
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{record['video_id']} {image.size} t≈{seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
