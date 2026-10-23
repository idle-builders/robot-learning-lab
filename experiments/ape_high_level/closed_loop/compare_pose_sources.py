"""Make one inspectable three-way Cove-50 pose-source video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


CONFIGS = (
    ("Command + gyro", "chest_rgbd_range"),
    ("RGB-D odometry + gyro", "chest_rgbd_range_visualodom"),
    ("Range matching + gyro", "chest_rgbd_range_localized"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--scenario", default="variable_slip")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    folders = [args.run / f"{args.scenario}-seed{args.seed}-0-{config}-none"
               for _, config in CONFIGS]
    traces = [json.loads((folder / "trace.json").read_text()) for folder in folders]
    readers = [imageio.get_reader(folder / "episode.mp4") for folder in folders]
    try:
        counts = [reader.count_frames() for reader in readers]
        output = args.run / f"triple-{args.scenario}-seed{args.seed}.mp4"
        with imageio.get_writer(output, fps=4, codec="libx264", quality=7) as writer:
            for index in range(max(counts)):
                parts = [reader.get_data(min(index, count - 1))
                         for reader, count in zip(readers, counts)]
                image = Image.fromarray(np.concatenate(parts, axis=1))
                draw = ImageDraw.Draw(image)
                for side, ((title, _), trace) in enumerate(zip(CONFIGS, traces)):
                    x = side * parts[side].shape[1]
                    draw.rectangle((x, 0, x + parts[side].shape[1], 52), fill=(0, 0, 0))
                    draw.text((x + 8, 6), title, fill=(255, 255, 255))
                    draw.text((x + 8, 27),
                              f"Pose error {trace['pose_error_m']*100:.1f} cm | "
                              f"{trace['outcome']}", fill=(230, 230, 230))
                writer.append_data(np.asarray(image))
    finally:
        for reader in readers:
            reader.close()
    output.with_suffix(".json").write_text(json.dumps({
        "inputs": [str(folder / "episode.mp4") for folder in folders],
        "traces": [str(folder / "trace.json") for folder in folders],
        "map_note": "Blue actual pose is evaluator-only; yellow is each controller's estimate.",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
