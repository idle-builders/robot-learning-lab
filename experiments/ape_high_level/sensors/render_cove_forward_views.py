"""Render the two virtual camera views along the saved Cove-50 forward walk.

The red spheres are ideal, fixed-world floor markers. This is a visualization
of camera geometry using saved robot poses, not a perception or control rollout.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from cove_forward_visibility import (CAMERAS, SHELL_VISUAL, TARGETS, TRACE,
                                     check_sources, load_model)


OUTPUT = Path(__file__).resolve().parents[3] / "artifacts/ape-sensors-50cm/cove-forward-camera-views.mp4"


def main() -> None:
    design, _, _ = check_sources()
    model = load_model(SHELL_VISUAL, targets={
        name: ((forward, lateral, radius), radius)
        for name, (forward, lateral, radius) in TARGETS.items()})
    saved = np.load(TRACE, allow_pickle=False)
    qpos = saved["qpos"]
    dt = float(saved["dt"])
    if qpos.shape != (400, model.nq) or dt != 0.02:
        raise ValueError("Saved Cove-50 walk changed")

    start = qpos[0]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, start[3:7])
    start_basis = rotation.reshape(3, 3)[:2, :2]
    for name, (forward, lateral, radius) in TARGETS.items():
        xy = start[:2] + start_basis @ np.array([forward, lateral])
        model.body_pos[model.body(f"probe_{name}").id] = (xy[0], xy[1], radius)

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(OUTPUT, fps=10, codec="libx264", quality=7) as writer:
            for index in range(0, 160, 5):
                data.qpos[:] = qpos[index]
                mujoco.mj_forward(model, data)
                panels = []
                for camera, label in zip(CAMERAS, ("Head lower RGB", "Virtual low chest RGB")):
                    renderer.update_scene(data, camera=camera)
                    panel = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, 280, 40), fill=(0, 0, 0))
                    draw.text((10, 8), f"{label}  {index * dt:.1f}s", fill=(255, 255, 255))
                    panels.append(np.asarray(panel))
                writer.append_data(np.concatenate(panels, axis=1))
    finally:
        renderer.close()

    OUTPUT.with_suffix(".json").write_text(json.dumps({
        "selected_design": design["id"],
        "source_trace": str(TRACE),
        "source_frames": len(qpos),
        "source_duration_s": len(qpos) * dt,
        "video_duration_s": 160 * dt,
        "video_fps": 10,
        "cameras": list(CAMERAS),
        "targets": TARGETS,
        "limits": "Saved policy poses, ideal fixed-world markers, virtual cameras; no object recognition, sensor optics, or new control rollout.",
    }, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
