"""Audit whether the moving prop remains clear after an ideal stop.

V423/V425 ended each episode at the stop event. This independent replay holds
the ape's saved terminal position while the already scripted prop finishes its
motion. The actor never receives these evaluator-only distances.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_robocasa_layout2_routes_v409 import evaluator_scene, sha256
from run_robocasa_moving_prop_v423 import MOVE_END_S, SUBSTEP_S, obstacle_xy


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-post-stop-v427"
SCRIPT = Path(__file__)
BASE = ROOT / "artifacts/ape-robocasa-moving-prop-v423/report.json"
STRESS = ROOT / "artifacts/ape-robocasa-moving-guard-stress-v425/report.json"


def replay(label: str, episode: dict, *, video: bool = False) -> dict:
    model, data, ape_body, ape_geom = evaluator_scene([4.95, -3.34])
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    model.geom_rgba[ape_geom] = [0.1, 0.65, 0.95, 1.]
    ape_xy = np.asarray(episode["end_true_xy_m"], dtype=float)
    stop_time = float(episode["end_time_s"])
    times = np.arange(stop_time, MOVE_END_S + SUBSTEP_S / 2, SUBSTEP_S)
    times = np.unique(np.r_[times, MOVE_END_S])
    records = []
    frames = []
    renderer = mujoco.Renderer(model, height=480, width=640) if video else None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [4.45, -3.34, 0.]
    camera.distance = 2.2
    camera.azimuth = 0
    camera.elevation = -90
    for time_s in times:
        prop_xy = obstacle_xy(float(time_s))
        model.body_pos[ape_body, :2] = ape_xy
        model.body_pos[prop_body, :2] = prop_xy
        mujoco.mj_forward(model, data)
        gap = float(mujoco.mj_geomDistance(model, data, ape_geom, prop_geom, .5, None))
        records.append({"time_s": round(float(time_s), 4),
                        "ape_xy_m": ape_xy.tolist(),
                        "prop_xy_m": prop_xy.tolist(),
                        "signed_clearance_m": gap})
        if renderer is not None:
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 55), fill=(0, 0, 0))
            draw.text((10, 7), f"APE HOLDS AFTER {label} | prop keeps moving", fill="white")
            draw.text((10, 27), f"t={time_s:.2f}s  signed gap={gap*100:.1f}cm  5cm margin", fill="white")
            frames.append(np.asarray(frame))
    if renderer is not None:
        renderer.close()
    result = {"case": label, "actor_outcome": episode["outcome"],
              "stop_time_s": stop_time,
              "stop_gap_m": records[0]["signed_clearance_m"],
              "final_gap_m": records[-1]["signed_clearance_m"],
              "minimum_post_stop_gap_m": min(row["signed_clearance_m"] for row in records),
              "margin_held_through_motion": all(row["signed_clearance_m"] >= .05 for row in records),
              "first_margin_breach_s": next((row["time_s"] for row in records
                                              if row["signed_clearance_m"] < .05), None),
              "first_contact_s": next((row["time_s"] for row in records
                                       if row["signed_clearance_m"] < 0), None),
              "trace": records}
    if video:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "nominal-guard-post-stop.mp4"
        imageio.mimsave(path, frames, fps=10, codec="libx264", quality=7)
        result["video"] = {"path": str(path), "sha256": sha256(path), "frames": len(frames)}
        reader = imageio.get_reader(path)
        result["video_frame_count_pass"] = reader.count_frames() == len(frames)
        reader.close()
    return result


def main() -> None:
    base = json.loads(BASE.read_text())
    stress = json.loads(STRESS.read_text())
    results = [replay("nominal 20ms guard", base["episodes"][1], video=True)]
    results.extend(replay(case["mode"], case["episode"])
                   for case in stress["cases"] if case["mode"] != "nominal")
    report = {"source_sha256": sha256(SCRIPT),
              "base_report_sha256": sha256(BASE), "stress_report_sha256": sha256(STRESS),
              "scenario": "saved ape pose held after ideal stop; original prop trajectory continues to 14.4s",
              "required_gap_m": .05, "cases": results}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(result["case"], "stop", round(result["stop_gap_m"], 4),
              "end", round(result["final_gap_m"], 4),
              "first_contact", result["first_contact_s"], flush=True)


if __name__ == "__main__":
    main()
