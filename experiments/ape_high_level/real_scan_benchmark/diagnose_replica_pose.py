"""Temporary pose sweep to diagnose Replica route visibility."""

import sys
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_angle_axis

from render_replica_ape_route import make_simulator


root = Path(sys.argv[1])
scene = sys.argv[2]
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
with make_simulator(root, scene, 640, 480, 0.42) as sim:
    points = [sim.pathfinder.get_random_navigable_point() for _ in range(6)]
    rows = []
    for n, point in enumerate(points):
        for yaw in (0, 1.57, 3.14, 4.71):
            state = habitat_sim.AgentState()
            state.position = point
            state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            sim.initialize_agent(0).set_state(state)
            obs = sim.get_sensor_observations()
            color = np.asarray(obs["rgb"])[..., :3]
            depth = np.asarray(obs["depth"])
            frac = float((np.isfinite(depth) & (depth > 0)).mean())
            thumb = Image.fromarray(color)
            thumb.thumbnail((240, 180))
            rows.append((thumb, f"p{n} yaw {yaw:.2f} depth {frac:.2f}"))
    sheet = Image.new("RGB", (240 * 4, 205 * 6), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, (frame, label) in enumerate(rows):
        x = i % 4 * 240
        y = i // 4 * 205
        sheet.paste(frame, (x, y + 25))
        draw.text((x + 5, y + 5), label, fill="white")
    sheet.save(out / f"{scene}_pose_sweep.jpg", quality=88)
