"""Render the saved native support states for inspection; never rescore them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from ape_walk.physics.frame50 import reveal_frame50_proxy_geometry_for_video


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261015/support-assist-feasibility"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    method = json.loads((OUT / "method.json").read_text())
    if sha(BODY) != method["body_sha256"]:
        raise RuntimeError("The render body differs from the native source")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if reveal_frame50_proxy_geometry_for_video(model) == 0:
        raise RuntimeError("The collision proxy body could not be made visible")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.18]
    camera.distance = 0.90
    camera.azimuth = 135
    camera.elevation = -16
    manifest = {"schema": "robot-learning-lab.support-assist-state-render.v1",
                "kind": "posthoc native qpos visualization, no simulator rescore",
                "body_sha256": sha(BODY), "videos": []}
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for side in ("right", "left"):
            for label in ("zero", "assisted"):
                name = f"{side}-{label}"
                report_path = OUT / "native" / f"{name}.json"
                report = json.loads(report_path.read_text())
                trace_path = OUT / "native" / f"{name}.npz"
                if sha(trace_path) != report["trace_sha256"]:
                    raise RuntimeError(f"Native trace differs: {name}")
                with np.load(trace_path, allow_pickle=False) as values:
                    poses = values["qpos"].copy()
                    dt = float(values["dt_s"])
                if poses.shape != (report["steps"], model.nq) or abs(dt - .02) > 1e-9:
                    raise RuntimeError(f"Unexpected state clock: {name}")
                video = OUT / f"{name}-state-replay.mp4"
                with imageio.get_writer(video, fps=25, macro_block_size=16) as writer:
                    for pose in poses[::2]:
                        data.qpos[:] = pose
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera)
                        writer.append_data(renderer.render().copy())
                manifest["videos"].append({
                    "case": name, "source_report": str(report_path.relative_to(ROOT)),
                    "source_report_sha256": sha(report_path),
                    "source_trace": str(trace_path.relative_to(ROOT)),
                    "source_trace_sha256": sha(trace_path),
                    "video": str(video.relative_to(ROOT)), "video_sha256": sha(video),
                    "frames": len(poses[::2]), "fps": 25,
                })
    (OUT / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
