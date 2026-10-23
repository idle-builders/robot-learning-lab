"""Side-by-side evaluator replay through the second prop's post-stop motion.

This reconstruction reads saved executed ape poses and scripted prop paths.
It is visual evidence, never an actor camera packet or planning input.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_robocasa_layout2_two_props_v483 import evaluator_scene, secondary_at
from render_robocasa_layout4_crossing_v456 import at, positions
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout2-two-props-v482"


def main():
    source = OUT / "report.json"
    report = json.loads(source.read_text())
    cases = [next(case for case in report["cases"]
                  if case["scenario"]["id"] == name)
             for name in ("second_absent", "second_south_toward_route")]
    model, data = evaluator_scene(cases[0]["scenario"])
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    primary_body = model.body("route_obstacle").id
    secondary_body = model.body("second_obstacle").id
    secondary_geom = model.geom("second_obstacle_geom").id
    model.geom_rgba[ape_geom] = [.08, .65, .95, 1.]
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1.55, -3.05, 0.]
    camera.distance = 3.8
    camera.azimuth = 0
    camera.elevation = -90
    tracks = [positions(case["episode"]) for case in cases]
    frames = []
    gaps = []
    for time_s in np.arange(1.5, 10.2 + .01, .1):
        sides = []
        row_gaps = []
        for label, case, track in zip(("ONE PROP - DETOUR", "TWO PROPS - STAGE"),
                                      cases, tracks):
            ape_xy = at(track, float(time_s))
            model.body_pos[ape_body, :2] = ape_xy
            model.body_pos[primary_body, :2] = case["scenario"]["primary_xy_m"]
            model.body_pos[secondary_body, :2] = secondary_at(case["scenario"], float(time_s))
            mujoco.mj_forward(model, data)
            gap = min(.5, float(mujoco.mj_geomDistance(
                model, data, ape_geom, secondary_geom, .5, None)))
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 640, 65), fill=(0, 0, 0))
            draw.text((10, 7), label, fill="white")
            draw.text((10, 27),
                      f"t={time_s:.1f}s  ape-second gap >= 50 cm"
                      if gap >= .5 else
                      f"t={time_s:.1f}s  ape-second gap = {gap * 100:.1f} cm",
                      fill="white")
            draw.text((10, 47), "cyan: ape   white: moving second prop", fill="white")
            sides.append(image)
            row_gaps.append(gap)
        combined = Image.new("RGB", (1280, 480))
        combined.paste(sides[0], (0, 0))
        combined.paste(sides[1], (640, 0))
        frames.append(np.asarray(combined))
        gaps.append({"time_s": round(float(time_s), 3),
                     "one_prop_gap_m_capped_at_50cm": row_gaps[0],
                     "two_prop_gap_m_capped_at_50cm": row_gaps[1]})
    renderer.close()
    video = OUT / "alternate-route-vs-safe-side-post-stop-v484.mp4"
    imageio.mimsave(video, frames, fps=10, codec="libx264", quality=7)
    reader = imageio.get_reader(video)
    count = reader.count_frames()
    reader.close()
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(source),
              "scope": "evaluator-only overhead reconstruction of saved ape poses and scripted two-prop scene through post-stop movement",
              "video": {"path": str(video), "sha256": sha256(video),
                        "frames": len(frames), "decoded_frames": count},
              "minimum_second_gap_m_capped_at_50cm": min(
                  row["two_prop_gap_m_capped_at_50cm"] for row in gaps),
              "timeline": gaps}
    (OUT / "replay-v484.json").write_text(json.dumps(output, indent=2) + "\n")
    print(video, "frames", count,
          "min_second_gap", output["minimum_second_gap_m_capped_at_50cm"], flush=True)


if __name__ == "__main__":
    main()
