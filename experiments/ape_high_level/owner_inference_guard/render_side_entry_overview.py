"""Render the saved side-entry trajectory from above next to its stale owner view."""

from __future__ import annotations

import json

import cv2
import mujoco
import numpy as np

from run_inference_side_entry import OUT, moving, turn, two


def label(frame, text, at, color=(255, 255, 255), scale=.55):
    cv2.putText(frame, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def main():
    report = json.loads((OUT / "report.json").read_text())
    s = next(x for x in report["summaries"] if x["condition"] == "all_angle_moving")
    t0 = s["moving_start_s"]
    elapsed_alarm = s["guard_alarm_time_s"]-t0
    elapsed_breach = s["geometry_evaluator"]["first_below_5cm_s"]-t0
    start = np.asarray(s["side_start_xy_evaluator_m"])
    target = np.asarray(s["side_target_xy_evaluator_m"])
    ape = np.asarray(s["ape_xy_evaluator_m"])
    packets = s["range_packets"]
    changes = s["range_change_comparison"]
    shown = cv2.imread(str(OUT / "all_angle_moving" / "owner_reteach" / "new-show.png"))
    assert shown is not None
    shown = cv2.resize(shown, (320, 240))

    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    turn._show_owner(model, 1., .4)
    model.body_pos[model.body("audit_ape").id, :2] = ape
    model.geom_rgba[model.geom("audit_ape_geom").id] = [1., .8, .15, 1.]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [ape[0]-.3, ape[1], .3]
    camera.distance = 2.5
    camera.azimuth = 90
    camera.elevation = -85

    output = OUT / "side-entry-overview.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             10., (960, 480))
    assert writer.isOpened()
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for frame_index in range(36):
                dt = frame_index / 10.
                fraction = min(1., .62*dt/np.linalg.norm(target-start))
                two.set_distractor(model, start+(target-start)*fraction)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                overhead = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
                cv2.rectangle(overhead, (0, 0), (640, 46), (16, 16, 16), -1)
                label(overhead, "Evaluator overhead: red person / yellow ape proxy",
                      (12, 28), scale=.58)
                label(overhead, f"{dt:.1f}s since side entry", (18, 455), scale=.65)
                packet_index = min(range(len(packets)),
                    key=lambda i: abs((packets[i]["time_s"]-t0)-dt))
                counts = changes[packet_index]
                right = np.full((480, 320, 3), (24, 24, 24), np.uint8)
                right[:240] = shown
                label(right, "Owner image: captured before entry", (8, 23), scale=.43)
                label(right, f"Owner sector changed: {counts['owner_sector_changed']}",
                      (8, 272), scale=.48)
                label(right, f"Full circle changed: {counts['all_angle_changed']}",
                      (8, 300), scale=.48)
                if dt >= elapsed_alarm:
                    label(right, "Full-circle alarm: discard image", (8, 340),
                          color=(90, 240, 255), scale=.43)
                else:
                    label(right, "No full-circle alarm yet", (8, 340), scale=.48)
                if dt >= elapsed_breach:
                    label(right, "Person breached 5 cm gap", (8, 377),
                          color=(75, 75, 255), scale=.45)
                label(right, "Ape has no escape action here", (8, 448),
                      color=(100, 200, 255), scale=.42)
                writer.write(np.hstack([overhead, right]))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(output))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 36, frames
    print(output, frames)


if __name__ == "__main__":
    main()
