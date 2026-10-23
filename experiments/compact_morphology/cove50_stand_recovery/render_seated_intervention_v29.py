"""Render V29 source-checked CUDA baseline/intervention pairs from saved states."""

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
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-intervention-v29"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
INTERVENTIONS = {
    "script-first-0p2": ("scripted", "SCRIPT 0.20s → ACTOR"),
    "actor-first-0p1": ("model25", "ACTOR 0.10s → SCRIPT"),
    "actor-first-0p2": ("model25", "ACTOR 0.20s → SCRIPT"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(seed: int, route: str, expected: dict) -> tuple[dict, dict[str, np.ndarray]]:
    stem = ART / f"seed{seed}-{route}"
    report_path, native_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    assert sha(report_path) == expected["source_report_sha256"]
    assert sha(native_path) == expected["native_trajectory_sha256"]
    report = json.loads(report_path.read_text())
    with np.load(native_path, allow_pickle=False) as file:
        trace = {name: file[name].copy() for name in
                 ("initial_qpos", "initial_qvel", "qpos", "qvel")}
    assert report["device"] == "cuda:0" and report["seed"] == seed
    assert report["body_sha256"] == sha(BODY)
    assert np.array_equal(trace["initial_qpos"], report["initial_qpos"])
    assert np.array_equal(trace["initial_qvel"], report["initial_qvel"])
    return report, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("route", choices=tuple(INTERVENTIONS))
    args = parser.parse_args()
    scored_path = ART / "report.json"
    scored = json.loads(scored_path.read_text())
    assert scored["status"] == "complete_source_checked_cuda_opened_development_diagnostic"
    row = {(entry["seed"], entry["route"]): entry for entry in scored["rows"]}
    baseline, label = INTERVENTIONS[args.route]
    baseline_report, baseline_trace = load(args.seed, baseline, row[args.seed, baseline])
    intervention_report, intervention_trace = load(args.seed, args.route,
                                                    row[args.seed, args.route])
    assert baseline_report["initial_state_sha256"] == intervention_report["initial_state_sha256"]
    assert baseline_report["initial_physics"] == intervention_report["initial_physics"]
    assert baseline_report["initial_motors"] == intervention_report["initial_motors"]

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
    traces = [baseline_trace, intervention_trace]
    names = [baseline.upper(), label]
    outcomes = [row[args.seed, baseline], row[args.seed, args.route]]
    font = ImageFont.load_default()
    output = ART / f"paired-{baseline}-vs-{args.route}-seed{args.seed}.mp4"
    with imageio.get_writer(output, fps=25, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for frame in range(250):
            panels = []
            for trace, state, name, outcome in zip(traces, states, names, outcomes):
                i = min(frame * 2, len(trace["qpos"]) - 1)
                state.qpos[:] = trace["qpos"][i]
                state.qvel[:] = trace["qvel"][i]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera=camera)
                panel = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 430, 52), fill=(15, 20, 25))
                draw.text((9, 7), name, font=font, fill=(255, 255, 255))
                result = "PASS" if outcome["full_gate"] else "FAIL: " + ", ".join(
                    outcome["failure_reasons"][:2])
                draw.text((9, 25), f"seed {args.seed}  t={frame / 25:.2f}s  {result}",
                          font=font, fill=(215, 215, 215))
                panels.append(panel)
            combined = Image.new("RGB", (1280, 480))
            combined.paste(panels[0], (0, 0))
            combined.paste(panels[1], (640, 0))
            writer.append_data(np.asarray(combined))
    renderer.close()
    metadata = {
        "schema": "robot-learning-lab.cove50-seated-paired-cuda-state-video.v29",
        "seed": args.seed, "baseline": baseline, "intervention": args.route,
        "video": str(output.relative_to(ROOT)), "video_sha256": sha(output),
        "body_sha256": sha(BODY),
        "matching_initial_state_sha256": baseline_report["initial_state_sha256"],
        "source_report_sha256": {route: sha(ART / f"seed{args.seed}-{route}.json")
                                 for route in (baseline, args.route)},
        "source_trace_sha256": {route: sha(ART / f"seed{args.seed}-{route}.npz")
                                for route in (baseline, args.route)},
        "scored_report_sha256": sha(scored_path),
        "renderer_source_sha256": sha(Path(__file__)),
        "proxy_geoms_visual_only_revealed": visible_count,
        "frame_rule": "Every second 50 Hz saved CUDA qpos/qvel sample; hold the final state after early termination. Local MuJoCo rendering changes no score.",
        "claim_limit": "Source-matched saved CUDA states, not another rollout or independent score.",
    }
    (ART / f"paired-{baseline}-vs-{args.route}-seed{args.seed}.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"video": str(output), "sha256": metadata["video_sha256"]}))


if __name__ == "__main__":
    main()
