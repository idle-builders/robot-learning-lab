"""Compare Habitat views at both horizontal sign conventions for a Replica pose."""

import argparse
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis

from render_replica_ape_route import make_simulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frame = json.loads(args.trace.read_text())["frames"][args.frame]
    position = np.asarray(frame["position_m"], dtype=float)
    args.output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1280, 540), "#222222")
    draw = ImageDraw.Draw(sheet)
    with make_simulator(args.dataset_root, args.scene, 640, 480, 0.42) as sim:
        agent = sim.initialize_agent(0)
        for row, sign in enumerate((1, -1)):
            for col in range(4):
                state = habitat_sim.AgentState()
                state.position = position.copy()
                state.position[2] *= sign
                state.rotation = quat_from_angle_axis(
                    col * math.pi / 2, np.array([0.0, 1.0, 0.0])
                )
                agent.set_state(state)
                observations = sim.get_sensor_observations()
                rgb = np.asarray(observations["rgb"])[..., :3]
                depth = np.asarray(observations["depth"])
                valid = float((np.isfinite(depth) & (depth > 0)).mean())
                thumbnail = Image.fromarray(rgb).resize((320, 240))
                x, y = col * 320, row * 270
                sheet.paste(thumbnail, (x, y + 30))
                draw.text((x + 5, y + 7),
                          f"z sign {sign:+} | yaw {col * 90} | depth {valid:.2f}",
                          fill="white")
    sheet.save(args.output / "pose_sign_sweep.jpg", quality=90)


if __name__ == "__main__":
    main()
