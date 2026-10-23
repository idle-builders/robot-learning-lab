"""Replay V701's commanded hold beyond its cutoff against the declared prop path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_robocasa_layout2_routes_v409 import clearance, sha256
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_layout2_side_prop_v701 as trial


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = trial.OUT / "audit-v702.json"
VIDEO = trial.OUT / "post-horizon-held-pose-evaluator-replay.mp4"


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    ep = report["episode"]
    model = person.scene_with_person(trial.START, layout_id=2,
                                     style_id=1, clutter_mode=0)
    torso = interaction.TORSO_XY + [-.3, 0.]
    rest = interaction.REST_HAND + [-.3, 0., 0.]
    show = interaction.SHOW_HAND + [-.3, 0., 0.]
    person.set_owner_pose(model, torso, rest, show_can=False)
    fingerprint = hashlib.sha256(b"".join(array.tobytes() for array in
        (model.body_pos, model.geom_pos, model.geom_quat,
         model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    held_xy = np.asarray(ep["memory_return"]["end_actual_xy_m_evaluator_only"])
    final_t = float(ep["hold_until_s"])
    assert ep["memory_return"]["route_status"] == "evaded_wait_for_owner_clearance"
    person.set_owner_pose(model, torso, show, show_can=True)
    direction = (show[:2] - torso) / np.linalg.norm(show[:2] - torso)
    model.body_pos[model.body("shown_can").id, :2] = show[:2] + .14 * direction
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    prop_body = model.body("route_obstacle").id
    head_id = model.camera("head").id
    renderer = mujoco.Renderer(model, height=480, width=640)
    rows, frames = [], []
    for index, t in enumerate(np.arange(final_t, 22.401, .02)):
        prop = trial.prop_at(float(t))
        model.body_pos[prop_body, :2] = prop
        model.cam_pos[head_id, :2] = held_xy
        gap, geom = clearance(model, data, ape_body, ape_geom, held_xy)
        proxy_gap = float(np.linalg.norm(held_xy - prop) - .17 - .24)
        rows.append({"time_s": round(float(t), 4), "ape_xy_m": held_xy.tolist(),
                     "prop_xy_m": prop.tolist(), "signed_scene_gap_m_capped_at_10cm": gap,
                     "nearest_geom": geom, "proxy_prop_gap_m": proxy_gap})
        if index % 5 == 0:
            renderer.update_scene(data, camera="head")
            image = Image.fromarray(renderer.render().copy()).convert("RGB")
            canvas = Image.new("RGB", (820, 480), "#0d171c")
            canvas.paste(image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((650, 18), "evaluator replay", fill="white")
            draw.text((650, 44), f"time {t:.2f} s", fill="white")
            draw.text((650, 70), f"ape-prop gap {proxy_gap:.3f} m",
                      fill="#ff6868" if proxy_gap < .05 else "#baf0a8")
            draw.text((650, 100), "ape remains at", fill="white")
            draw.text((650, 118), "chosen hold pose", fill="white")
            frames.append(np.asarray(canvas))
    renderer.close()
    imageio.mimsave(VIDEO, frames, fps=10, codec="libx264", quality=7)
    minimum = min(rows, key=lambda row: row["signed_scene_gap_m_capped_at_10cm"])
    first_breach = next((row for row in rows if row["signed_scene_gap_m_capped_at_10cm"] < .05), None)
    first_overlap = next((row for row in rows if row["signed_scene_gap_m_capped_at_10cm"] < 0.), None)
    episode_video = Path(ep["video"]["path"])
    reader = imageio.get_reader(episode_video)
    frame_count = reader.count_frames()
    reader.close()
    result = {"schema": "robot-learning-lab.layout2-side-prop-post-horizon-audit.v1",
              "source_sha256": sha256(SOURCE),
              "trial_report_sha256": sha256(trial.OUT / "report.json"),
              "scene_fingerprint_pass": fingerprint == ep["scene_fingerprint"],
              "actor_end_time_s": final_t,
              "actor_final_intent": ep["memory_return"]["route_status"],
              "held_pose_xy_m": held_xy.tolist(),
              "continuation_assumption": "No further movement is commanded after the saved episode; hold that pose while replaying the predeclared prop path. This is evaluator-only counterfactual continuation, not an actor observation or an executed control response.",
              "samples_20ms": len(rows),
              "minimum_signed_scene_gap": minimum,
              "first_below_5cm": first_breach,
              "first_overlap": first_overlap,
              "actor_episode_video_sha_pass": sha256(episode_video) == ep["video"]["sha256"],
              "actor_episode_video_frames_pass": frame_count == ep["video"]["frames"],
              "continuation_video": {"path": str(VIDEO), "sha256": sha256(VIDEO),
                                       "frames": len(frames)},
              "claim": "The finite actor hold horizon cannot certify this pose against a continuing moving prop; the replay tests the declared continuation only."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("end",final_t,"first <5cm",None if first_breach is None else first_breach["time_s"],
          "first overlap",None if first_overlap is None else first_overlap["time_s"],
          "minimum",minimum["signed_scene_gap_m_capped_at_10cm"],flush=True)


if __name__ == "__main__":
    main()
