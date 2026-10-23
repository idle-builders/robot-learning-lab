"""Render saved native V4 qpos only; this never reruns policy or physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if sha(BODY) != BODY_SHA or report["body_sha256"] != BODY_SHA:
        raise RuntimeError("Saved trace body differs from exact Direct Frame")
    with np.load(args.trace, allow_pickle=False) as trace:
        qpos = trace["post_qpos"].copy()
        pre_qpos = trace["pre_qpos"].copy()
        dt = float(trace["dt_s"])
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) != report["steps"] or abs(dt - 0.02) > 1e-8:
        raise RuntimeError("Native trace length, qpos or clock differs")
    if (pre_qpos.shape != qpos.shape or not np.isfinite(pre_qpos).all()
            or not np.isfinite(qpos).all()
            or np.max(np.abs(pre_qpos[0] - report["initial_qpos"])) > 1e-8
            or (len(qpos) > 1 and np.max(np.abs(qpos[:-1] - pre_qpos[1:])) > 1e-8)):
        raise RuntimeError("Trace initial state or no-reset continuity differs from report")
    pelvis = np.asarray([[row["pelvis_x"], row["pelvis_y"], row["pelvis_z"]]
                         for row in report["timeseries"]])
    if pelvis.shape != (len(qpos), 3) or np.max(np.abs(qpos[:, :3] - pelvis)) > 1e-8:
        raise RuntimeError("Trace pelvis motion differs from report")
    for geom in range(model.ngeom):
        if (model.geom_bodyid[geom] != 0 and model.geom_group[geom] == 4
                and model.geom_rgba[geom, 3] == 0):
            model.geom_rgba[geom, 3] = 1
            model.geom_group[geom] = 0
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.10, 0.0, 0.22)
    camera.distance = 1.3
    camera.azimuth = 90.0
    camera.elevation = -12.0
    args.video.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = 0
    try:
        with imageio.get_writer(args.video, fps=25, codec="libx264", quality=8,
                                macro_block_size=16) as writer:
            for step in range(0, len(qpos), 2):
                data.qpos[:] = qpos[step]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                frames += 1
    finally:
        renderer.close()
    result = {
        "schema": "robot-learning-lab.broad-handoff-v4-saved-trace-render.v1",
        "classification": "display-only rendering of native post-step qpos; no physics or policy rerun",
        "body_sha256": BODY_SHA, "report_sha256": sha(args.report),
        "trace_sha256": sha(args.trace), "video_sha256": sha(args.video),
        "native_steps": len(qpos), "video_frames": frames,
        "source_device": report["device"], "source_seed": report["seed"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": report["seed"], "frames": frames, "video": str(args.video)}))


if __name__ == "__main__":
    main()
