"""Side-by-side two saved episodes with their actual outcomes and clearance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default="chest_rgbd_range_guard_goal_map")
    parser.add_argument("--baseline-title", required=True)
    parser.add_argument("--candidate-title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folders = [root / f"{args.scenario}-seed{args.seed}-0-{args.config}-none"
               for root in (args.baseline, args.candidate)]
    traces = [json.loads((folder / "trace.json").read_text()) for folder in folders]
    readers = [imageio.get_reader(folder / "episode.mp4") for folder in folders]
    try:
        counts = [reader.count_frames() for reader in readers]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.output, fps=4, codec="libx264", quality=7) as writer:
            for index in range(max(counts)):
                parts = [reader.get_data(min(index, count - 1))
                         for reader, count in zip(readers, counts)]
                image = Image.fromarray(np.concatenate(parts, axis=1))
                draw = ImageDraw.Draw(image)
                for side, (title, trace) in enumerate(zip(
                        (args.baseline_title, args.candidate_title), traces)):
                    x = side * parts[side].shape[1]
                    draw.rectangle((x, 0, x + parts[side].shape[1], 54),
                                   fill=(0, 0, 0))
                    draw.text((x + 8, 6), title, fill=(255, 255, 255))
                    draw.text((x + 8, 28),
                              f"{trace['outcome']} | min model gap "
                              f"{trace['minimum_robot_obstacle_clearance_m']*100:.1f} cm",
                              fill=(220, 220, 220))
                writer.append_data(np.asarray(image))
    finally:
        for reader in readers:
            reader.close()
    args.output.with_suffix(".json").write_text(json.dumps({
        "inputs": [str(folder / "episode.mp4") for folder in folders],
        "output": str(args.output),
        "frame_counts": counts,
        "titles": [args.baseline_title, args.candidate_title],
        "evaluator_panels_hidden_from_actor": True,
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
