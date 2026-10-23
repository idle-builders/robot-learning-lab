"""Show the paired owner-route outcome from one Gaussian capture."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016/capture_scene14"
ARMS = (
    ("Separate two-height range", "gaussian_transfer_route_v4"),
    ("Upright + downward RGB-D", "gaussian_route_dual_rgbd_v4"),
)
OUTPUT = ROOT / "assets/real_indoor_scenes/gaussian_transfer_20261016/route_sensor_pair.mp4"


def main() -> None:
    streams = []
    for label, name in ARMS:
        folder = BASE / name
        video = folder / "initial/owner_approach/first_person.mp4"
        if not video.is_file():
            # The verified raw-media archive leaves this review reel locally.
            video = folder / "whole_episode_review.mp4"
        trace = json.loads((folder / "initial/owner_approach/trace.json").read_text())["trace"]
        reader = imageio.get_reader(video)
        try:
            frames = [np.asarray(frame) for _, frame in zip(range(len(trace)), reader)]
        finally:
            reader.close()
        if len(frames) != len(trace):
            raise ValueError(f"Video and packet count differ for {name}")
        streams.append((label, frames, trace))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(OUTPUT, fps=5, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for i in range(max(len(item[1]) for item in streams)):
            canvas = Image.new("RGB", (1280, 432), (10, 15, 22))
            draw = ImageDraw.Draw(canvas)
            for side, (label, frames, trace) in enumerate(streams):
                index = min(i, len(frames) - 1)
                view = ImageOps.fit(Image.fromarray(frames[index]).convert("RGB"),
                                    (640, 360))
                canvas.paste(view, (640 * side, 0))
                packet = trace[index]
                owner = packet.get("owner_actor") or {}
                depth = owner.get("local_depth_m")
                depth_text = f"{depth:.2f} m" if depth is not None else "unknown"
                decision = packet["planner_decision"]
                draw.text((640 * side + 12, 368), label, fill="white")
                draw.text((640 * side + 12, 388),
                          f"Owner {depth_text} | {decision['intent']} | "
                          f"known rays {packet.get('known_range_beam_count', 0)}",
                          fill=(188, 226, 246))
            writer.append_data(np.asarray(canvas))
        for _ in range(10):
            canvas = Image.new("RGB", (1280, 432), (10, 15, 22))
            draw = ImageDraw.Draw(canvas)
            draw.text((35, 150), "Range: owner reached; object task completed",
                      fill="white")
            draw.text((675, 150), "Dual RGB-D: owner viewpoint unresolved; HOLD",
                      fill="white")
            writer.append_data(np.asarray(canvas))
    print(OUTPUT)


if __name__ == "__main__":
    main()
