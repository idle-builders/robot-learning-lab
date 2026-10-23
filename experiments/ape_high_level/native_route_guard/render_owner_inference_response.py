"""Evaluator overhead replay of V35's four matched inference responses."""

from __future__ import annotations

import json

import cv2
import mujoco
import numpy as np

import owner_inference_response as trial


CASES = (
    ("sector_buffered_turning", "Owner sector / accepts image"),
    ("wide_buffered_turning", "Full range / discards only"),
    ("wide_respond_fast", "Full range / prompt response"),
    ("wide_respond_slow", "Full range / delayed response"),
)
FPS = 10
FRAMES = 36


def label(frame, value, at, color=(255, 255, 255), scale=.5):
    cv2.putText(frame, value, at, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def panel(model, data, renderer, camera, summary, elapsed, title):
    ape = np.asarray(summary["worker_ape_start_xy_m"])
    response = summary["response_parameters"]
    alarm = summary["actor_alarm_s"]
    plan = summary["response_plan"]
    if response is not None and alarm is not None and plan["known_free"]:
        active = max(0., elapsed-(alarm-summary["worker_start_s"])-response["delay_s"])
        distance = min(trial.RESPONSE_MAX_M, response["speed_mps"]*active)
        ape = ape+[0., -distance]
    fraction = min(1., trial.APPROACH_MPS*elapsed/
                   np.linalg.norm(trial.TURNING_TARGET-trial.SIDE_START))
    person = trial.SIDE_START+(trial.TURNING_TARGET-trial.SIDE_START)*fraction
    model.body_pos[model.body("audit_ape").id, :2] = ape
    trial.two.set_distractor(model, person)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    frame = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (480, 360))
    ape_geom = model.geom("audit_ape_geom").id
    person_geoms = [model.geom(n).id for n in (
        "distractor_torso_geom", "distractor_head_geom",
        "distractor_upper_arm_geom", "distractor_forearm_geom",
        "distractor_hand_geom")]
    gap = min(float(mujoco.mj_geomDistance(
        model, data, ape_geom, p, 2., None)) for p in person_geoms)
    cv2.rectangle(frame, (0, 0), (480, 57), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, 307), (480, 360), (15, 15, 15), -1)
    label(frame, title, (10, 22), scale=.55)
    alarm_elapsed = None if alarm is None else alarm-summary["worker_start_s"]
    message = "alarm" if alarm_elapsed is not None and elapsed >= alarm_elapsed else "no alarm yet"
    if alarm_elapsed is None:
        message = "owner-sector blind"
    label(frame, message, (10, 45), scale=.48)
    color = (65, 70, 250) if gap < .05 else (110, 230, 120)
    label(frame, f"Modeled gap {gap*100:.1f} cm", (10, 336), color, .63)
    return frame


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    summaries = {row["condition"]: row for row in report["summaries"]}
    assert all(condition in summaries for condition, _ in CASES)
    model = mujoco.MjModel.from_xml_path(str(trial.moving.SCENE_XML))
    data = mujoco.MjData(model)
    trial.turn._show_owner(model, 1., .4)
    model.geom_rgba[model.geom("audit_ape_geom").id] = [1., .8, .15, 1.]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1.22, -4.36, .3]
    camera.distance = 2.6
    camera.azimuth = 90
    camera.elevation = -85
    out = trial.OUT / "inference-alert-and-response-comparison.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (960, 720))
    assert writer.isOpened()
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for index in range(FRAMES):
                elapsed = index/FPS
                frames = [panel(model, data, renderer, camera, summaries[key],
                                elapsed, title) for key, title in CASES]
                writer.write(np.vstack((np.hstack(frames[:2]),
                                        np.hstack(frames[2:]))))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(out))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    assert count == FRAMES
    print(out, count)


if __name__ == "__main__":
    main()
