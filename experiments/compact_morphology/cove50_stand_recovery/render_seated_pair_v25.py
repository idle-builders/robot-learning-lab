"""Render a matched baseline/learner pair from saved CUDA physical states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(checkpoint_id: str, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    report_path = ART / f"checkpoint-{checkpoint_id}-seed{seed}.json"
    trace_path = ART / f"checkpoint-{checkpoint_id}-seed{seed}.npz"
    report = json.loads(report_path.read_text())
    with np.load(trace_path, allow_pickle=False) as file:
        trace = {name: file[name].copy() for name in ("initial_qpos", "initial_qvel", "qpos", "qvel")}
    assert report["device"] == "cuda:0" and report["seed"] == seed
    assert report["body_sha256"] == sha(BODY)
    assert np.array_equal(trace["initial_qpos"], report["initial_qpos"])
    assert np.array_equal(trace["initial_qvel"], report["initial_qvel"])
    return report, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    initial, baseline = load_case("initial", args.seed)
    learned, candidate = load_case("25", args.seed)
    assert initial["initial_state_sha256"] == learned["initial_state_sha256"]
    assert initial["initial_physics"] == learned["initial_physics"]
    assert initial["initial_motors"] == learned["initial_motors"]

    model = mujoco.MjModel.from_xml_path(str(BODY))
    # Exact score states are retained. Only hidden proxy geom visibility changes.
    from ape_walk.physics.frame50 import reveal_frame50_proxy_geometry_for_video
    visible_count = reveal_frame50_proxy_geometry_for_video(model)
    data = [mujoco.MjData(model), mujoco.MjData(model)]
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat = (0, 0, 0.20)
    camera.distance = 0.9
    camera.azimuth = 90
    camera.elevation = -12
    traces = [baseline, candidate]
    labels = ["UNTRAINED · falls", "MODEL 25 · survives, slips"]
    font = ImageFont.load_default()
    output = ART / f"paired-initial-vs-25-seed{args.seed}.mp4"
    with imageio.get_writer(output, fps=25, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for frame in range(250):
            step = frame * 2
            panels = []
            for trace, state, label in zip(traces, data, labels):
                i = min(step, len(trace["qpos"]) - 1)
                state.qpos[:] = trace["qpos"][i]
                state.qvel[:] = trace["qvel"][i]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera=camera)
                panel = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 350, 44), fill=(15, 20, 25))
                draw.text((9, 7), label, font=font, fill=(255, 255, 255))
                draw.text((9, 25), f"seed {args.seed}  t={frame / 25:.2f}s",
                          font=font, fill=(215, 215, 215))
                panels.append(panel)
            combined = Image.new("RGB", (1280, 480))
            combined.paste(panels[0], (0, 0))
            combined.paste(panels[1], (640, 0))
            writer.append_data(np.asarray(combined))
    renderer.close()
    metadata = {
        "schema": "robot-learning-lab.cove50-seated-paired-cuda-state-video.v25",
        "seed": args.seed,
        "video": str(output.relative_to(ROOT)),
        "video_sha256": sha(output),
        "body_sha256": sha(BODY),
        "matching_initial_state_sha256": initial["initial_state_sha256"],
        "baseline_report_sha256": sha(ART / f"checkpoint-initial-seed{args.seed}.json"),
        "baseline_trace_sha256": sha(ART / f"checkpoint-initial-seed{args.seed}.npz"),
        "candidate_report_sha256": sha(ART / f"checkpoint-25-seed{args.seed}.json"),
        "candidate_trace_sha256": sha(ART / f"checkpoint-25-seed{args.seed}.npz"),
        "renderer_source_sha256": sha(Path(__file__)),
        "proxy_geoms_visual_only_revealed": visible_count,
        "frame_rule": "Every second 50Hz CUDA qpos/qvel sample; after early termination hold its last sample. MuJoCo native local renderer changes no state, action or score.",
        "claim_limit": "Exact saved CUDA physical trajectories, rendered locally; not another rollout or independent score.",
    }
    (ART / f"paired-initial-vs-25-seed{args.seed}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps({"video": str(output), "sha256": metadata["video_sha256"]}))


if __name__ == "__main__":
    main()
