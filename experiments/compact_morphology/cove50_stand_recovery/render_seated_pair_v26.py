"""Render the source-checked V26 CUDA learner/reference pair from saved states."""

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
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(seed: int, route: str) -> tuple[dict, dict[str, np.ndarray]]:
    stem = ART / f"seed{seed}-{route}"
    report = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as file:
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
    scored = json.loads((ART / "report.json").read_text())
    assert scored["status"] == "complete_source_checked_cuda_opened_development_diagnostic"
    assert args.seed in {row["seed"] for row in scored["rows"]}
    learned, learner_trace = load(args.seed, "model25")
    scripted, script_trace = load(args.seed, "scripted")
    assert learned["initial_state_sha256"] == scripted["initial_state_sha256"]
    assert learned["initial_physics"] == scripted["initial_physics"]
    assert learned["initial_motors"] == scripted["initial_motors"]
    for field in ("robot_body_names", "robot_body_mass_kg", "robot_body_inertia_kg_m2",
                  "robot_body_ipos_m", "robot_body_iquat_wxyz"):
        assert learned["initial_physics"][field] == scripted["initial_physics"][field]

    model = mujoco.MjModel.from_xml_path(str(BODY))
    from ape_walk.physics.frame50 import reveal_frame50_proxy_geometry_for_video
    visible_count = reveal_frame50_proxy_geometry_for_video(model)
    states = [mujoco.MjData(model), mujoco.MjData(model)]
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat = (0, 0, 0.20)
    camera.distance = 0.9
    camera.azimuth = 90
    camera.elevation = -12
    traces = [learner_trace, script_trace]
    labels = ["MODEL 25", "SCRIPTED PLANTED REFERENCE"]
    font = ImageFont.load_default()
    output = ART / f"paired-model25-vs-script-seed{args.seed}.mp4"
    with imageio.get_writer(output, fps=25, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for frame in range(250):
            panels = []
            for trace, state, label in zip(traces, states, labels):
                i = min(frame * 2, len(trace["qpos"]) - 1)
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
        "schema": "robot-learning-lab.cove50-seated-paired-cuda-state-video.v26",
        "seed": args.seed,
        "video": str(output.relative_to(ROOT)),
        "video_sha256": sha(output),
        "body_sha256": sha(BODY),
        "matching_initial_state_sha256": learned["initial_state_sha256"],
        "model25_report_sha256": sha(ART / f"seed{args.seed}-model25.json"),
        "model25_trace_sha256": sha(ART / f"seed{args.seed}-model25.npz"),
        "scripted_report_sha256": sha(ART / f"seed{args.seed}-scripted.json"),
        "scripted_trace_sha256": sha(ART / f"seed{args.seed}-scripted.npz"),
        "scored_report_sha256": sha(ART / "report.json"),
        "renderer_source_sha256": sha(Path(__file__)),
        "proxy_geoms_visual_only_revealed": visible_count,
        "frame_rule": "Every second 50Hz CUDA qpos/qvel sample; hold the final state after early termination. Local MuJoCo rendering does not change the rollout or score.",
        "claim_limit": "Source-matched saved CUDA states, not another rollout or independent score.",
    }
    (ART / f"paired-model25-vs-script-seed{args.seed}.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"video": str(output), "sha256": metadata["video_sha256"]}))


if __name__ == "__main__":
    main()
