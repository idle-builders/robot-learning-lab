"""Show paired owner-view and independent range packets during re-show.

The red person marker is evaluator-only: it explains what the range source
missed, and is never an actor observation or policy input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"


def load(run: str, key: str) -> tuple[dict, list[dict], Image.Image]:
    folder = BASE / key / run
    summary = json.loads((folder / "summary.json").read_text())
    packets = json.loads((folder / "independent_range/packets.json").read_text())[
        "packets"]
    image = Image.open(folder / "owner_reshow" / key / "teach/rgb_000.png")
    return summary, packets, image.convert("RGB")


def packet_at(packets: list[dict], wall_s: float) -> dict:
    return min(packets, key=lambda p: abs(p["capture_wall_s"] - wall_s))


def panel(title: str, summary: dict, packet: dict, camera: Image.Image,
          elapsed_s: float) -> Image.Image:
    width, height = 640, 700
    canvas = Image.new("RGB", (width, height), (13, 18, 27))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), title, fill=(236, 241, 250))
    draw.text((16, 36), f"Owner re-show  +{elapsed_s:.1f} s",
              fill=(180, 202, 220))
    canvas.paste(ImageOps.fit(camera, (width, 320)), (0, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 393), "Robot-centered low range, 10 Hz",
              fill=(190, 212, 231))
    center = (320, 555)
    scale = 115.0
    for radius in (.5, 1.0):
        r = round(radius * scale)
        draw.ellipse((center[0]-r, center[1]-r, center[0]+r,
                      center[1]+r), outline=(62, 72, 87), width=1)
    draw.line((center[0], center[1], center[0], center[1]-135),
              fill=(90, 110, 129), width=2)
    draw.text((center[0]+8, center[1]-145), "front", fill=(130, 153, 174))
    low = packet.get("low_m") or []
    for angle, distance in zip(range(-180, 180, 5), low):
        if distance is None:
            continue
        radius = min(float(distance), 1.7) * scale
        a = math.radians(angle)
        end = (center[0] + radius * math.sin(a),
               center[1] - radius * math.cos(a))
        draw.line((*center, *end), fill=(48, 135, 167), width=1)
        draw.ellipse((end[0]-2, end[1]-2, end[0]+2, end[1]+2),
                     fill=(74, 221, 238))
    person = packet.get("moving_person_xz_m")
    if person is not None:
        dx = float(person[0] - packet["position_xz_m"][0])
        dz = float(person[1] - packet["position_xz_m"][1])
        yaw = math.radians(float(packet["heading_deg"]))
        right = -dx * math.cos(yaw) + dz * math.sin(yaw)
        forward = dx * math.sin(yaw) + dz * math.cos(yaw)
        px = center[0] + right * scale
        py = center[1] - forward * scale
        draw.ellipse((px-9, py-9, px+9, py+9), fill=(240, 77, 86))
        draw.text((px+12, py-7), "person (evaluator only)",
                  fill=(250, 135, 141))
    draw.ellipse((center[0]-7, center[1]-7, center[0]+7, center[1]+7),
                 fill=(247, 222, 123))
    alarm = bool(packet.get("near_body_change_alarm"))
    draw.text((16, 667), "RANGE ALARM" if alarm else "No covered alarm",
              fill=(250, 104, 109) if alarm else (167, 186, 203))
    if summary.get("reshow_interruption_wall_s") is not None:
        dispatch = summary["reshow_interruption_wall_s"] - summary[
            "reshow_entrant_origin_wall_s"]
        if elapsed_s >= dispatch:
            draw.text((420, 667), "HOLD issued", fill=(250, 184, 87))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="capture_nomad_east_1m")
    parser.add_argument("--all360", default="v4_reshow_monitor_rear_all360_v2")
    parser.add_argument("--front140", default="v4_reshow_monitor_rear_front140_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_summary, all_packets, all_image = load(args.all360, args.scene)
    front_summary, front_packets, front_image = load(args.front140, args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=5, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for tick in range(34):
            elapsed = tick / 5.0
            all_packet = packet_at(all_packets, all_summary[
                "reshow_entrant_origin_wall_s"] + elapsed)
            front_packet = packet_at(front_packets, front_summary[
                "reshow_entrant_origin_wall_s"] + elapsed)
            canvas = Image.new("RGB", (1280, 704), (13, 18, 27))
            canvas.paste(panel("All-around two-height range", all_summary,
                all_packet, all_image, elapsed), (0, 0))
            canvas.paste(panel("Forward 140-degree range", front_summary,
                front_packet, front_image, elapsed), (640, 0))
            writer.append_data(np.asarray(canvas))
    print(json.dumps({"video": str(args.output), "frames": 34,
                      "evaluator_person_marker": True}))


if __name__ == "__main__":
    main()
