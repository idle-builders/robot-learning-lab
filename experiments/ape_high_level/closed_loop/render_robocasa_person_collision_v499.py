"""Append evaluator post-stop replay to the V491 late-crossing route video.

The actor's original route video ends at its proximity stop. This adds the
continuing scripted prop and true signed body gap so the subsequent collision
is visible rather than hidden behind the stop label.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import robocasa_owner_person as person
from audit_robocasa_layout2_routes_v409 import sha256
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "artifacts/ape-robocasa-person-route-v491/report.json"
AUDIT = ROOT / "artifacts/ape-robocasa-person-route-audit-v493/report.json"
OUT = ROOT / "artifacts/ape-robocasa-person-route-v491/late-crossing-post-stop-v499.mp4"
META = ROOT / "artifacts/ape-robocasa-person-route-v491/late-crossing-post-stop-v499.json"


def main():
    report = json.loads(TRIAL.read_text())
    case = next(row for row in report["cases"] if row["scenario"]["id"] == "late_crossing")
    episode = case["episode"]
    audit = json.loads(AUDIT.read_text())
    witness = next(row for row in audit["packages"][0]["cases"]
                   if row["scenario_id"] == "late_crossing")
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = [reader.get_data(i).copy() for i in range(reader.count_frames())]
    reader.close()
    base = Image.fromarray(frames[-1])
    scene = case["scenario"]
    script = motion(scene)
    model = person.scene_with_person(prop_xy=scene["from_xy_m"])
    person.set_owner_pose(model, [1., -2.4], [1.10, -2.24, .43], show_can=False)
    ape = np.asarray(episode["end_true_xy_m"])
    model.body_pos[model.body("audit_ape").id, :2] = ape
    model.cam_pos[model.camera("head").id, :2] = ape
    prop_body = model.body("route_obstacle").id
    ape_geom = model.geom("audit_ape_geom").id
    prop_geom = model.geom("route_obstacle_geom").id
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    appended = []
    for t in np.arange(episode["end_time_s"] + .02, scene["end_s"] + .011, .02):
        model.body_pos[prop_body, :2] = script(float(t))
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="head")
        rgb = renderer.render().copy()
        gap = float(mujoco.mj_geomDistance(model, data, ape_geom, prop_geom, .5, None))
        frame = base.copy()
        frame.paste(Image.fromarray(rgb), (0, 48))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 640, 48), fill="#101619")
        label = f"EVALUATOR replay after stop  t={t:.2f}s  signed gap={gap:+.3f}m"
        draw.text((12, 17), label, fill="#ff665e" if gap < 0 else "#ffe07b")
        # The head faces the cabinetry at this bad stop. Show the physical
        # footprint overlap explicitly rather than relying on an occluded view.
        draw.rectangle((651, 284, 1031, 519), fill="#152128", outline="#a5beca", width=2)
        draw.text((663, 295), "EVALUATOR top view: held ape and crossing prop",
                  fill="#e5f5f8")
        scale = 185.
        center = np.array([840., 408.])
        prop = script(float(t))
        prop_center = center + np.array([prop[0] - ape[0], ape[1] - prop[1]]) * scale
        for xy, radius, fill, outline in ((prop_center, .24, "#bf7627", "#ffe29d"),
                                          (center, .17, "#2e9da8", "#bafaff")):
            x, y = xy
            r = radius * scale
            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline, width=3)
        draw.text((663, 495), "aqua: ape  |  amber: moving prop", fill="#e5f5f8")
        appended.append(np.asarray(frame))
    renderer.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUT, frames + appended, fps=5, codec="libx264", quality=7)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(TRIAL),
              "audit_report_sha256": sha256(AUDIT),
              "original_route_video_sha_pass": sha256(video) == episode["video"]["sha256"],
              "audit_min_signed_prop_gap_m": witness["nearest_ape_prop"]["signed_gap_m"],
              "scope": "V491 actor route video plus labeled evaluator replay of the held ape pose and continuing late prop; not an additional actor observation",
              "video": {"path": str(OUT), "sha256": sha256(OUT),
                        "frames": len(frames) + len(appended)}}
    META.write_text(json.dumps(output, indent=2) + "\n")
    print(OUT, "frames", len(frames) + len(appended),
          "audit gap", output["audit_min_signed_prop_gap_m"], flush=True)


if __name__ == "__main__":
    main()
