"""Render post-step native pose, RGB view and contact from aligned saved data.

The trace stores qpos before each env.step; timeseries row i stores the
resulting post-step contacts and positions. Therefore row i uses qpos[i+1].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import (
    VirtualRgbGoal, orange_bbox,
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=135)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if not report["one_sim_without_skill_reset"]:
        raise ValueError("Expected saved no-reset RGB-goal result")
    with np.load(args.trace) as trace:
        qpos = trace["qpos"].copy()
        dt = float(trace["step_dt_s"])
    if len(qpos) != len(report["timeseries"]):
        raise ValueError("Trace/report frame count differs")
    source = VirtualRgbGoal(selected_body_sha256=report["body_sha256"],
                            max_age_s=report["max_goal_age_s"])
    if qpos.shape[1] != source.model.nq:
        raise ValueError("Trace/render model qpos differs")
    hand_site = mujoco.mj_name2id(source.model, mujoco.mjtObj.mjOBJ_SITE,
                                  "right_hand_center")
    if hand_site < 0:
        raise ValueError("Right hand site missing from saved body")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.10, 0.0, 0.17)
    camera.distance = .65
    camera.azimuth = args.azimuth
    camera.elevation = -20
    external = mujoco.Renderer(source.model, height=480, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    args.video.parent.mkdir(parents=True, exist_ok=True)
    fps = 20
    times = np.arange(0, (len(qpos) - 1) * dt, 1 / fps)
    f_title, f_body, f_meta = font(30), font(23), font(18)
    entry = next(e for e in report["image_goal_capture_events"] if e["used_for_control"])
    entry_time = entry["time_s"]
    entry_error = entry.get("xy_error_truth_mm_scoring_only")
    contact_s = report["phases"]["reach"]["continuous_finger_contact_s"]
    yaw_text = f"yaw {report['gaze_head_yaw_rad']:g}"
    if report.get("reacquisition_head_yaw_rad") is not None:
        yaw_text += f" → {report['reacquisition_head_yaw_rad']:g}"
    with imageio.get_writer(args.video, fps=fps, codec="libx264", quality=8,
                            macro_block_size=16) as writer:
        for t in times:
            index = min(len(qpos) - 2, round(t / dt))
            row = report["timeseries"][index]
            display_time = row["time_s"]
            source.data.qpos[:] = qpos[index + 1]
            source.data.qvel[:] = 0
            mujoco.mj_forward(source.model, source.data)
            if np.linalg.norm(source.data.site_xpos[hand_site] -
                              row["hand_xyz_m"]) > .00002:
                raise ValueError(f"Saved post-step pose/report mismatch at {display_time}s")
            external.update_scene(source.data, camera=camera, scene_option=option)
            world = external.render().copy()
            source.renderer.update_scene(source.data, camera="lower_rgb", scene_option=option)
            rgb = source.renderer.render().copy()
            bbox = orange_bbox(rgb)
            frame = Image.new("RGB", (1280, 608), "#111a26")
            frame.paste(Image.fromarray(world), (0, 64))
            frame.paste(Image.fromarray(rgb).resize((640, 480), Image.Resampling.NEAREST), (640, 64))
            draw = ImageDraw.Draw(frame)
            draw.text((20, 16), "Cove-50 · native image-goal reach", font=f_title, fill="#f5f9fc")
            draw.text((860, 22),
                      f"seed {report['seed']} · {report['rail_v']:g} V · {report['delay_ms']:g} ms · {yaw_text}",
                      font=f_meta, fill="#c6dbe9")
            draw.rectangle((0, 64, 639, 101), fill="#111a26")
            draw.rectangle((640, 64, 1279, 101), fill="#111a26")
            draw.text((16, 73), "Native robot and toy", font=f_body, fill="#e6f4fa")
            draw.text((656, 73), "Virtual lower RGB · exact sim pose", font=f_body, fill="#e6f4fa")
            if bbox:
                x, y, w, h = bbox["xywh_px"]
                draw.rectangle((640 + 2*x, 64 + 2*y, 640 + 2*(x+w), 64 + 2*(y+h)),
                               outline="#00f5a4", width=3)
            phase = row["phase"]
            supported = all(row["feet_grounded"])
            status = "FINGERTIP CONTACT" if row["finger_contact"] else "no fingertip contact"
            draw.text((16, 557), f"t={display_time:4.1f}s   {phase}   feet: {'both' if supported else 'lost'}",
                      font=f_body, fill="#e6f4fa")
            draw.text((16, 583), status, font=f_body,
                      fill="#62e2ac" if row["finger_contact"] else "#e1bcb4")
            if display_time < entry_time:
                camera_status = "Awaiting reach-entry RGB goal"
            elif report["image_goal_lock_after_entry"]:
                camera_status = f"Frozen entry RGB goal · age {display_time-entry_time:.1f}s"
            else:
                camera_status = "Live RGB goal · visible-frame updates"
            draw.text((656, 557), camera_status, font=f_body, fill="#e6f4fa")
            draw.text((656, 583), f"Scored entry error {entry_error:.2f} mm · best contact {contact_s:.2f}s",
                      font=f_body, fill="#b5d1e2")
            writer.append_data(np.asarray(frame))
    external.close()
    source.close()
    print(f"rendered {args.video} from {len(qpos)} native qpos at {dt:g}s")


if __name__ == "__main__":
    main()
