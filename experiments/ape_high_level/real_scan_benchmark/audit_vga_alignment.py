"""Measure how the captured VGA images align with the posed RGB-D samples."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/arkit_v1"


def main() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    results = []
    for scene in manifest["scenes"]:
        scene_dir = ASSETS / scene["video_id"]
        record = json.loads((scene_dir / "scene.json").read_text())
        if len(record.get("vga_rgb_frames", [])) != 12:
            results.append({"video_id": scene["video_id"], "status": "missing_vga"})
            continue
        correlations, errors, offsets = [], [], []
        for frame in record["vga_rgb_frames"]:
            folder = scene_dir / "frames" / f"{frame['index']:02d}"
            low = cv2.imread(str(folder / "rgb.png"))
            high = cv2.imread(str(folder / "rgb_vga.png"))
            if low is None or high is None or low.shape != (192, 256, 3) or high.shape != (480, 640, 3):
                raise ValueError(f"{scene['video_id']}: image missing or wrong dimensions")
            down = cv2.resize(high, (256, 192), interpolation=cv2.INTER_AREA)
            correlations.append(float(np.corrcoef(low.ravel(), down.ravel())[0, 1]))
            errors.append(float(np.mean(np.abs(low.astype(np.int16) - down.astype(np.int16)))))
            offsets.append(abs(frame["offset_from_depth_pose_s"]))
        result = {
            "video_id": scene["video_id"], "status": "measured",
            "frame_count": len(correlations),
            "max_pose_depth_rgb_offset_s": round(max(offsets), 4),
            "rgb_correlation_min": round(min(correlations), 4),
            "rgb_correlation_median": round(float(np.median(correlations)), 4),
            "downsampled_rgb_mae_median_0_255": round(float(np.median(errors)), 2),
        }
        results.append(result)
        print(f"{scene['video_id']} RGB corr min/med="
              f"{result['rgb_correlation_min']:.3f}/{result['rgb_correlation_median']:.3f} "
              f"max offset={result['max_pose_depth_rgb_offset_s']:.3f}s", flush=True)
    (ASSETS / "vga_alignment_audit.json").write_text(json.dumps({
        "schema": "ape.arkit-vga-alignment-audit.v1",
        "method": "downsample VGA to lowres pixels; compare RGB and timestamp offsets",
        "limitation": "image agreement is not 3D extrinsic or novel-view validation",
        "scenes": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
