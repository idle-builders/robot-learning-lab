"""Place paired localization episode videos side by side for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--scenario", default="variable_slip")
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    names = ("chest_rgbd_range", "chest_rgbd_range_localized")
    folders = [args.run / f"{args.scenario}-seed{args.seed}-0-{name}-none"
               for name in names]
    videos = [folder / "episode.mp4" for folder in folders]
    traces = [json.loads((folder / "trace.json").read_text()) for folder in folders]
    if not all(path.exists() for path in videos):
        raise FileNotFoundError(videos)
    output = args.run / f"paired-{args.scenario}-seed{args.seed}.mp4"
    readers = [imageio.get_reader(path) for path in videos]
    try:
        counts = [reader.count_frames() for reader in readers]
        with imageio.get_writer(output, fps=4, codec="libx264", quality=7) as writer:
            for index in range(max(counts)):
                frames = [reader.get_data(min(index, count - 1))
                          for reader, count in zip(readers, counts)]
                paired = Image.fromarray(np.concatenate(frames, axis=1))
                draw = ImageDraw.Draw(paired)
                for side, (name, trace) in enumerate(zip(names, traces)):
                    x = side * frames[side].shape[1]
                    draw.rectangle((x, 0, x + frames[side].shape[1], 52), fill=(0, 0, 0))
                    draw.text((x + 10, 6), "Command odometry" if side == 0 else
                              "Range scan localization", fill=(255, 255, 255))
                    draw.text((x + 10, 27),
                              f"Final pose error {trace['pose_error_m']*100:.1f} cm | "
                              f"{trace['outcome']}", fill=(225, 225, 225))
                writer.append_data(np.asarray(paired))
    finally:
        for reader in readers:
            reader.close()
    output.with_suffix(".json").write_text(json.dumps({
        "videos": [str(path) for path in videos],
        "source_traces": [str(folder / "trace.json") for folder in folders],
        "output": str(output),
        "map_note": "Blue true pose and object locations are evaluator overlays only; yellow pose is the actor estimate.",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
