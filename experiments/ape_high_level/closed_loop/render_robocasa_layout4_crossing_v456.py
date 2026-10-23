"""Render the audited layout-4 late crossing through the post-stop horizon.

This is an evaluator-only replay of saved ape poses and the predeclared prop
trajectory, not an observation or input to either actor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_robocasa_dynamic_suite_v440 import prop_at
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-robocasa-layout4-crossing-v447/report.json"
LOCAL = ROOT / "artifacts/ape-robocasa-layout4-local-risk-v454/report.json"
OUT = ROOT / "artifacts/ape-robocasa-layout4-local-risk-v454"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positions(episode):
    points = []
    for row in episode["trace"]:
        start = np.asarray(row["actual_xy_m"], dtype=float)
        points.append((float(row["time_s"]), start))
        requested = row.get("requested_xy_m")
        if requested is None:
            continue
        delta = ((np.asarray(requested) - np.asarray(row["estimated_xy_m"]))
                 * (1. + .30 / 5.89))
        for index, sub in enumerate(row["motion_substeps"]):
            fraction = (index if sub["stopped_before_move"] else index + 1) / 20
            points.append((float(sub["time_s"]), start + fraction * delta))
    points.append((float(episode["end_time_s"]),
                   np.asarray(episode["end_true_xy_m"], dtype=float)))
    return sorted(points, key=lambda row: row[0])


def at(points, time_s):
    return next((xy for t, xy in reversed(points) if t <= time_s + 1e-7), points[0][1])


def main():
    base = json.loads(BASE.read_text())
    local = json.loads(LOCAL.read_text())
    cases = [("BASELINE - stops in path", next(c for c in base["cases"]
                                              if c["scenario"]["id"] == "late_crossing")),
             ("LOCAL RISK 60cm - blocked", next(c for c in local["cases"]
                                                if c["scenario"]["id"] == "late_crossing"
                                                and c["risk_radius_m"] == .6))]
    model, data, ape_body, ape_geom = layout4_evaluator_scene(
        cases[0][1]["scenario"]["from_xy_m"])
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    model.geom_rgba[ape_geom] = [0.08, 0.65, 0.95, 1.]
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [2.70, -1.50, 0.]
    camera.distance = 3.1
    camera.azimuth = 0
    camera.elevation = -90
    tracks = [positions(c["episode"]) for _, c in cases]
    frames = []
    measurements = []
    for time_s in np.arange(13.8, 14.4 + .01, .02):
        halves = []
        gaps = []
        for (label, case), points in zip(cases, tracks):
            ape_xy = at(points, float(time_s))
            prop_xy = prop_at(case["scenario"], float(time_s))
            model.body_pos[ape_body, :2] = ape_xy
            model.body_pos[prop_body, :2] = prop_xy
            mujoco.mj_forward(model, data)
            gap = float(mujoco.mj_geomDistance(model, data, ape_geom, prop_geom, .5, None))
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 56), fill=(0, 0, 0))
            draw.text((10, 7), label, fill="white")
            draw.text((10, 27), f"t={time_s:.2f}s  full-body prop gap={gap*100:.1f}cm", fill="white")
            halves.append(frame)
            gaps.append(gap)
        combined = Image.new("RGB", (1280, 480))
        combined.paste(halves[0], (0, 0))
        combined.paste(halves[1], (640, 0))
        frames.append(np.asarray(combined))
        measurements.append({"time_s": round(float(time_s), 4),
                             "baseline_gap_m": gaps[0], "local_risk_gap_m": gaps[1]})
    renderer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / "late-crossing-baseline-vs-local-risk-post-stop.mp4"
    imageio.mimsave(video, frames, fps=10, codec="libx264", quality=7)
    reader = imageio.get_reader(video)
    count = reader.count_frames()
    reader.close()
    report = {"source_sha256": sha256(Path(__file__)),
              "baseline_report_sha256": sha256(BASE),
              "local_risk_report_sha256": sha256(LOCAL),
              "scope": "evaluator-only topdown replay of saved actors and predeclared prop through 14.4s",
              "video": {"path": str(video), "sha256": sha256(video),
                        "frames": len(frames), "decoded_frames": count},
              "minimum_gaps_m": {"baseline": min(x["baseline_gap_m"] for x in measurements),
                                 "local_risk": min(x["local_risk_gap_m"] for x in measurements)},
              "timeline": measurements}
    (OUT / "replay-v456.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["minimum_gaps_m"], "frames", count, flush=True)


if __name__ == "__main__":
    main()
