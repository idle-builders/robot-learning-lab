"""Render four source-verified V34 CUDA trajectories from saved states only."""

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
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-trigger-v34"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ROUTES = ("model25", "scripted", "fixed-0p06", "gravity-switch")
SCORE_KEYS = {"model25": "model25", "scripted": "scripted",
              "fixed-0p06": "fixed_0p06", "gravity-switch": "gravity_switch"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    score_path = ART / "local-score.json"
    score = json.loads(score_path.read_text())
    assert not score["pending_sync"] and not score["verification_failures"]
    rows = [row for row in score["rows"] if row["seed"] == args.seed]
    assert len(rows) == 1 and rows[0]["status"] == "complete_source_checked_four_route_family"
    row = rows[0]
    reports, traces = [], []
    for route in ROUTES:
        stem = ART / f"seed{args.seed}-{route}"
        report = json.loads(stem.with_suffix(".json").read_text())
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as npz:
            trace = {name: npz[name].copy() for name in ("initial_qpos", "initial_qvel", "qpos", "qvel")}
        assert report["device"] == "cuda:0" and report["body_sha256"] == sha(BODY)
        assert report["seed"] == args.seed and report["intervention_route"] == route
        assert sha(stem.with_suffix(".json")) == row[SCORE_KEYS[route]]["source_report_sha256"]
        assert sha(stem.with_suffix(".npz")) == row[SCORE_KEYS[route]]["native_trajectory_sha256"]
        reports.append(report)
        traces.append(trace)
    assert len({r["initial_state_sha256"] for r in reports}) == 1
    assert all(np.array_equal(traces[0]["initial_qpos"], t["initial_qpos"]) and
               np.array_equal(traces[0]["initial_qvel"], t["initial_qvel"]) for t in traces[1:])

    model = mujoco.MjModel.from_xml_path(str(BODY))
    from ape_walk.physics.frame50 import reveal_frame50_proxy_geometry_for_video
    visible_count = reveal_frame50_proxy_geometry_for_video(model)
    states = [mujoco.MjData(model) for _ in ROUTES]
    renderer = mujoco.Renderer(model, height=360, width=480)
    camera = mujoco.MjvCamera()
    camera.lookat = (0, 0, 0.20)
    camera.distance = 0.9
    camera.azimuth = 90
    camera.elevation = -12
    font = ImageFont.load_default()
    output = ART / f"four-routes-seed{args.seed}-saved-state.mp4"
    assert not output.exists(), "Do not overwrite an existing rendered evidence clip"
    with imageio.get_writer(output, fps=25, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for frame in range(250):
            composite = Image.new("RGB", (960, 720))
            for route_i, (route, state, trace) in enumerate(zip(ROUTES, states, traces)):
                i = min(frame * 2, len(trace["qpos"]) - 1)
                state.qpos[:] = trace["qpos"][i]
                state.qvel[:] = trace["qvel"][i]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera=camera)
                panel = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 360, 45), fill=(15, 20, 25))
                verdict = "PASS" if row[SCORE_KEYS[route]]["full_gate"] else "FAIL"
                draw.text((9, 7), f"{route}  {verdict}", font=font, fill=(255, 255, 255))
                draw.text((9, 25), f"seed {args.seed}  t={frame / 25:.2f}s",
                          font=font, fill=(215, 215, 215))
                composite.paste(panel, ((route_i % 2) * 480, (route_i // 2) * 360))
            writer.append_data(np.asarray(composite))
    renderer.close()
    metadata = {
        "schema": "robot-learning-lab.cove50-seated-trigger.v34.saved-state-video",
        "seed": args.seed,
        "video": str(output.relative_to(ROOT)),
        "video_sha256": sha(output),
        "body_sha256": sha(BODY),
        "initial_state_sha256": reports[0]["initial_state_sha256"],
        "routes": {route: {"full_gate": row[SCORE_KEYS[route]]["full_gate"],
                           "report_sha256": sha(ART / f"seed{args.seed}-{route}.json"),
                           "trajectory_sha256": sha(ART / f"seed{args.seed}-{route}.npz")}
                   for route in ROUTES},
        "local_score_sha256": sha(score_path),
        "renderer_source_sha256": sha(Path(__file__)),
        "proxy_geoms_visual_only_revealed": visible_count,
        "frame_rule": "Every second 50 Hz saved CUDA qpos/qvel sample; hold last state after early termination.",
        "claim_limit": "Local saved-state visualization, not another rollout or a fresh score.",
    }
    metadata_path = output.with_suffix(".json")
    assert not metadata_path.exists(), "Do not overwrite video metadata"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"video": str(output), "video_sha256": metadata["video_sha256"]}))


if __name__ == "__main__":
    main()
