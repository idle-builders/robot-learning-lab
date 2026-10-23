"""Side-by-side evaluator replay of a returning prop and two saved actors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
from render_robocasa_layout4_crossing_v456 import at, positions


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-robocasa-layout4-returning-prop-v460/report.json"
POLICY = ROOT / "artifacts/ape-robocasa-layout4-wait-remap-v461/report.json"
OUT = ROOT / "artifacts/ape-robocasa-layout4-wait-remap-v461"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base = next(c for c in json.loads(BASE.read_text())["cases"]
                if c["scenario"]["id"] == "returning_crossing")
    policy = next(c for c in json.loads(POLICY.read_text())["cases"]
                  if c["scenario"]["id"] == "returning_crossing")
    cases = [("BASELINE - stops at 9.2s", base),
             ("TRACK + WAIT + FRESH REMAP", policy)]
    model, data, ape_body, ape_geom = layout4_evaluator_scene(
        base["scenario"]["from_xy_m"])
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    model.geom_rgba[ape_geom] = [0.08, 0.65, 0.95, 1.]
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [3.10, -2.45, 0.]
    camera.distance = 7.0
    camera.azimuth = 0
    camera.elevation = -90
    tracks = [positions(c["episode"]) for _, c in cases]
    frames = []
    records = []
    for time_s in np.arange(0., 29.2 + .2, .4):
        halves = []
        frame_record = {"time_s": round(float(time_s), 4)}
        for (label, case), points in zip(cases, tracks):
            ape_xy = at(points, float(time_s))
            prop_xy = evaluator_prop_at(case["scenario"], float(time_s))
            model.body_pos[ape_body, :2] = ape_xy
            model.body_pos[prop_body, :2] = prop_xy
            mujoco.mj_forward(model, data)
            gap = float(mujoco.mj_geomDistance(model, data, ape_geom, prop_geom, .5, None))
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 75), fill=(0, 0, 0))
            draw.text((10, 6), label, fill="white")
            mode = (next((d["decision"] for d in reversed(policy["decision_history"])
                          if d["time_s"] <= time_s + 1e-7), "navigate")
                    if case is policy else case["episode"]["outcome"]
                    if time_s >= case["episode"]["end_time_s"] else "navigate")
            draw.text((10, 27), f"t={time_s:4.1f}s  {mode}", fill="white")
            measured_gap = np.isfinite(gap) and gap < .5
            gap_label = f"{gap*100:.1f}cm" if measured_gap else ">50cm"
            draw.text((10, 48), f"ape-prop gap={gap_label}", fill="white")
            halves.append(frame)
            frame_record["baseline_gap_m" if case is base else "policy_gap_m"] = (
                gap if measured_gap else None)
        combined = Image.new("RGB", (1280, 480))
        combined.paste(halves[0], (0, 0))
        combined.paste(halves[1], (640, 0))
        frames.append(np.asarray(combined))
        records.append(frame_record)
    renderer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / "returning-prop-baseline-vs-wait-remap.mp4"
    imageio.mimsave(video, frames, fps=6, codec="libx264", quality=7)
    reader = imageio.get_reader(video)
    decoded = reader.count_frames()
    reader.close()
    report = {"source_sha256": sha256(Path(__file__)),
              "baseline_report_sha256": sha256(BASE),
              "policy_report_sha256": sha256(POLICY),
              "scope": "evaluator-only synchronized topdown replay; actor decisions come only from saved policy trace",
              "video": {"path": str(video), "sha256": sha256(video),
                        "frames": len(frames), "decoded_frames": decoded},
              "timeline": records}
    (OUT / "replay-v463.json").write_text(json.dumps(report, indent=2) + "\n")
    print("frames", decoded, "baseline", base["episode"]["outcome"],
          "policy", policy["episode"]["outcome"], flush=True)


if __name__ == "__main__":
    main()
