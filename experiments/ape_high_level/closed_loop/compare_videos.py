"""Make one watchable before/after clip from paired closed-loop episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as imageio_v3
import numpy as np
from PIL import Image, ImageDraw


def title_frame(title: str, subtitle: str, shape: tuple[int, int, int]) -> np.ndarray:
    height, width, _ = shape
    image = Image.new("RGB", (width, height), (24, 31, 38))
    draw = ImageDraw.Draw(image)
    draw.text((28, height // 2 - 26), title, fill=(255, 255, 255))
    draw.text((28, height // 2 + 4), subtitle, fill=(188, 209, 220))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    scenarios = (
        ("RGB-D only", "Target confirmed, but the body overlaps furniture",
         f"seed{args.seed}-0-chest_rgbd-none"),
        ("RGB-D plus range", "Fresh views and footprint-aware detour",
         f"seed{args.seed}-0-chest_rgbd_range-none"),
    )
    output = args.run / f"paired-seed{args.seed}.mp4"
    inputs = [args.run / folder / "episode.mp4" for _, _, folder in scenarios]
    if not all(path.exists() for path in inputs):
        raise FileNotFoundError(inputs)
    shape = imageio_v3.imread(inputs[0], index=0).shape
    with imageio.get_writer(output, fps=4, codec="libx264", quality=7) as writer:
        for (title, subtitle, _), path in zip(scenarios, inputs):
            frame = title_frame(title, subtitle, shape)
            for _ in range(4):
                writer.append_data(frame)
            meta = imageio_v3.immeta(path)
            count = round(meta["duration"] * meta["fps"])
            with imageio.get_reader(path) as reader:
                for index in range(count):
                    writer.append_data(reader.get_data(index))
    output.with_suffix(".json").write_text(json.dumps({
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "scope": "Paired episodes in the same seeded synthetic room; labels describe evaluator outcomes, not physical robot performance.",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
