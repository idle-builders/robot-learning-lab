"""Render synchronized task and side views for a saved moving-person run.

The side view is evaluator-only review footage, never an actor observation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from live_option_episode import alternate_item_case, benchmark, make_scene


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    folder = args.episode.resolve()
    summary = json.loads((folder / "summary.json").read_text())
    if summary["fault"] not in ("moving_person_final",
                                "moving_person_final_range_gap"):
        raise ValueError("This episode has no timed moving person")
    case = next(c for c in benchmark.cohort()
                if c["scene_key"] == summary["scene_key"])
    if summary["scenario_variant"] == "alt_pair":
        case = alternate_item_case(case)
    simple = {"scene_key": case["scene_key"],
        "screen": case["item_screen"],
        "target": case["item"]["target"],
        "distractor": case["item"]["distractor"]}
    scene = make_scene(simple, True)
    packets_path = folder / "independent_range/packets.json"
    if packets_path.is_file():
        source = json.loads(packets_path.read_text())
    else:
        with gzip.open(packets_path.with_name(packets_path.name + ".gz"),
                       "rt") as stream:
            source = json.load(stream)
    capture_log = folder / "moving_person_captures.json"
    origin = (json.loads(capture_log.read_text())["origin_wall_s"]
        if capture_log.is_file() else next(p["capture_wall_s"]
            for p in source["packets"] if p.get("moving_person_xz_m") is not None))
    packets = [p for p in source["packets"]
        if p.get("moving_person_xz_m") is not None and
        0 <= p["capture_wall_s"] - origin <= 5.5]
    if len(packets) < 10:
        raise ValueError("Insufficient moving-person range packets")
    output = folder / "moving_person_monitor.mp4"
    with imageio.get_writer(output, fps=5, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for packet in packets[::2]:
            position = np.asarray(packet["position_xz_m"], float)
            person = tuple(packet["moving_person_xz_m"])
            eye = (scene.eye_at(*position) if hasattr(scene, "eye_at") else
                   np.array([position[0], scene.floor_y + .42,
                             position[1]]))
            if eye is None:
                raise ValueError("No camera support at saved monitor pose")
            views = []
            for bearing in (float(packet["heading_deg"]),
                            float(packet["heading_deg"]) + float(summary[
                                "moving_person_entry_relative_deg"])):
                yaw = math.radians(bearing)
                direction = np.array([math.sin(yaw), 0., math.cos(yaw)])
                rgb, _ = scene.observe(eye, direction, owner_xz=person,
                                       width=320, height=180)
                views.append(Image.fromarray(rgb))
            frame = Image.new("RGB", (640, 240), (16, 20, 26))
            frame.paste(views[0], (0, 0))
            frame.paste(views[1], (320, 0))
            draw = ImageDraw.Draw(frame)
            draw.text((8, 185), "TASK CAMERA", fill="white")
            draw.text((328, 185), "EVALUATOR SIDE VIEW", fill="white")
            elapsed = packet["capture_wall_s"] - origin
            status = ("RANGE MISSING: HOLD" if packet["source_health"] != "healthy"
                else "RANGE CHANGE: HOLD" if packet["near_body_change_alarm"]
                else "range change clear")
            draw.text((8, 208), f"t={elapsed:.2f}s  {status}",
                      fill=(255, 170, 145) if packet["source_health"] !=
                          "healthy" or packet["near_body_change_alarm"] else
                          (190, 230, 205))
            writer.append_data(np.asarray(frame))
    print(json.dumps({"video": str(output), "frames": len(packets[::2]),
                      "alarm_frames": sum(p["near_body_change_alarm"] for p
                                          in packets[::2])}))


if __name__ == "__main__":
    main()
