"""Make one reviewable first-person reel from a live option episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    folder = args.episode
    if (folder / "learner_summary.json").is_file():
        result = json.loads((folder / "learner_summary.json").read_text())
        source = ROOT / result["source_episode"]
        action = folder / "continuation_action" / result["scene_key"] / "healthy"
        owner_clip = source.parent / "initial/owner_approach/first_person.mp4"
        if not owner_clip.is_file():
            owner_clip = source.parent / "whole_episode_review.mp4"
        stages = [("First owner encounter", owner_clip),
                  ("Owner shows item again", folder / "teach/rgb_000.png"),
                  ("Find shown item", folder / "find_present/episode.mp4")]
        if result.get("owner_help_trace"):
            point = json.loads((ROOT / result["owner_help_trace"]).read_text())
            if point.get("fresh_video"):
                stages.append(("Owner points, then new search",
                               ROOT / point["fresh_video"]))
        stages += [("Fresh view before motion", action / "fresh_before_move/episode.mp4"),
                  ("Guarded approach", action / "guarded_walk/first_person.mp4"),
                  ("Review with owner", action / "final_view/episode.mp4")]
        options = result["authorized"]
    else:
        result = json.loads((folder / "summary.json").read_text())
        stages = [("Approach owner", folder / "initial/owner_approach/first_person.mp4"),
                  ("Owner teaches item", folder / "initial/teach/rgb_000.png"),
                  ("Find taught item", folder / "initial/find_present/episode.mp4")]
        if result.get("owner_reshow_staging"):
            resee = folder / "owner_reshow" / result["scene_key"]
            if result.get("reshow_interrupted_during_inference"):
                stages.append(("Interrupted show discarded",
                               resee / "teach/rgb_000.png"))
                resee = (folder / "owner_reshow/after_person" /
                         result["scene_key"])
            stages.extend([
                ("Owner shows item again", resee / "teach/rgb_000.png"),
                ("Find shown item", resee / "find_present/episode.mp4")])
        for name, label in (("wide_present", "Widen search"),
                            ("help_present", "Owner points to item"),
                            ("fresh_before_move", "Fresh view before motion"),
                            ("fresh_before_move/wide", "Recover lost visual identity"),
                            ("fresh_before_move/help", "Owner helps reidentify"),
                            ("fresh_after_source_recovery", "Reobserve after range recovery"),
                            ("walk", "Guarded approach"),
                            ("fresh_after_route_clear", "Reobserve after owner clears route"),
                            ("walk_after_route_clear", "Guarded renewed approach"),
                            ("final_interrupted", "Person enters during review"),
                            ("final_view", "Review with owner"),
                            ("final_wide", "Widen final search"),
                            ("final_after_wide", "Check final search result")):
            path = folder / name / ("first_person.mp4" if name.startswith(
                "walk") else "episode.mp4")
            stages.append((label, path))
        options = " > ".join(e["executed"] for e in result["events"]
                             if e["phase"] != "owner_point_result")
    stages = [(label, path) for label, path in stages if path.is_file()]
    output = folder / "whole_episode_review.mp4"
    with imageio.get_writer(output, fps=5, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for label, source in stages:
            if source.suffix == ".png":
                pixels = np.asarray(Image.open(source).convert("RGB"))
                frames = (pixels for _ in range(10))
                reader = None
                stride = 1
            else:
                reader = imageio.get_reader(source)
                frames = reader
                fps = float(reader.get_meta_data().get("fps", 5))
                stride = max(1, round(fps / 5))
            try:
                for index, pixels in enumerate(frames):
                    if index % stride:
                        continue
                    frame = ImageOps.fit(Image.fromarray(pixels).convert("RGB"),
                                         (640, 324))
                    canvas = Image.new("RGB", (640, 368), (12, 15, 20))
                    canvas.paste(frame, (0, 0))
                    ImageDraw.Draw(canvas).text((12, 335), label, fill="white")
                    writer.append_data(np.asarray(canvas))
            finally:
                if reader is not None:
                    reader.close()
        ending = Image.new("RGB", (640, 368), (12, 15, 20))
        draw = ImageDraw.Draw(ending)
        draw.text((24, 120), "Outcome: " + result["result"], fill="white")
        draw.text((24, 160), options[:80], fill=(190, 226, 247))
        for _ in range(10):
            writer.append_data(np.asarray(ending))
    with imageio.get_reader(output) as reader:
        frame_count = reader.count_frames()
        if frame_count < 10:
            raise RuntimeError(f"Incomplete episode reel: {output}")
        reader.get_data(frame_count - 1)
    print(json.dumps({"video": str(output), "result": result["result"],
                      "stages": [label for label, _ in stages],
                      "frames_verified": frame_count}))


if __name__ == "__main__":
    main()
