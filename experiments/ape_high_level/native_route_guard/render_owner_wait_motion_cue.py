"""Evaluator overhead replay of paired V34 wait choices and changed paths."""

from __future__ import annotations

import json

import cv2
import mujoco
import numpy as np

import owner_wait_pose_motion_cue as trial


FPS = 10
FRAMES_PER_PATH = 56


def text(frame, value, xy, color=(255, 255, 255), scale=.55):
    cv2.putText(frame, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def pose_at(summary, t, target):
    start_person = np.asarray(summary["bystander_stage_start_evaluator_xy_m"])
    initial_ape = np.asarray(summary["ape_after_episode_xy_m"])
    if summary["condition"].startswith("motion_wait"):
        initial_ape = initial_ape+[0., trial.WAIT_SHIFT_M]
    if t < trial.STAGE_S:
        person = start_person+(trial.SIDE_START-start_person)*t/trial.STAGE_S
        ape = initial_ape
        phase = "observing side motion"
    elif t < trial.STAGE_S+1.:
        person = trial.SIDE_START
        wait_fraction = min(1., t-trial.STAGE_S)
        ape = initial_ape+([0., -trial.WAIT_SHIFT_M*wait_fraction]
                           if summary["condition"].startswith("motion_wait") else [0., 0.])
        phase = "chosen wait action"
    else:
        dt = t-trial.STAGE_S-1.
        fraction = min(1., trial.APPROACH_MPS*dt/
                       np.linalg.norm(target-trial.SIDE_START))
        person = trial.SIDE_START+(target-trial.SIDE_START)*fraction
        ape = np.asarray(summary["ape_after_episode_xy_m"])
        phase = "slow item recognition"
    return ape, person, phase


def make_panel(model, data, renderer, camera, summary, t, target, title):
    ape, person, phase = pose_at(summary, t, target)
    model.body_pos[model.body("audit_ape").id, :2] = ape
    trial.two.set_distractor(model, person)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    panel = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (480, 480))
    ape_id = model.geom("audit_ape_geom").id
    ids = [model.geom(n).id for n in (
        "distractor_torso_geom", "distractor_head_geom",
        "distractor_upper_arm_geom", "distractor_forearm_geom",
        "distractor_hand_geom")]
    gap = min(float(mujoco.mj_geomDistance(model, data, ape_id, p, 2., None))
              for p in ids)
    cv2.rectangle(panel, (0, 0), (480, 74), (16, 16, 16), -1)
    cv2.rectangle(panel, (0, 402), (480, 480), (16, 16, 16), -1)
    text(panel, title, (12, 23), scale=.66)
    text(panel, phase, (12, 50), scale=.55)
    color = (70, 70, 250) if gap < .05 else (110, 235, 130)
    text(panel, f"Modeled person gap: {gap*100:.1f} cm", (12, 431), color, .62)
    text(panel, "Yellow ape proxy / red person", (12, 458), scale=.48)
    return panel


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    rows = {s["condition"]: s for s in report["summaries"]}
    assert set(rows) == set(trial.CONDITIONS)
    model = mujoco.MjModel.from_xml_path(str(trial.moving.SCENE_XML))
    data = mujoco.MjData(model)
    trial.turn._show_owner(model, 1., .4)
    model.geom_rgba[model.geom("audit_ape_geom").id] = [1., .8, .15, 1.]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1.20, -4.27, .3]
    camera.distance = 2.5
    camera.azimuth = 90
    camera.elevation = -85
    output = trial.OUT / "wait-choice-straight-vs-turning.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (960, 480))
    assert writer.isOpened()
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for path_name, target, held_name, wait_name in (
                    ("straight", trial.SIDE_TARGET, "held_moving", "motion_wait_moving"),
                    ("turning", trial.TURNING_TARGET, "held_turning", "motion_wait_turning")):
                for index in range(FRAMES_PER_PATH):
                    t = index/FPS
                    left = make_panel(model, data, renderer, camera,
                                      rows[held_name], t, target,
                                      f"{path_name}: hold current pose")
                    right = make_panel(model, data, renderer, camera,
                                       rows[wait_name], t, target,
                                       f"{path_name}: range-cued wait")
                    writer.write(np.hstack([left, right]))
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(output))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == FRAMES_PER_PATH*2
    print(output, count)


if __name__ == "__main__":
    main()
