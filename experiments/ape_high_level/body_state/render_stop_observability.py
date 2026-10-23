"""Render recorded articulated poses beside synchronized stop-gate decisions."""

from __future__ import annotations

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from stop_observability import BODY, OUTPUT, SOURCE


def load_packets(case: str) -> dict[float, dict]:
    with (OUTPUT / "packets-and-decisions.csv").open(newline="") as handle:
        return {round(float(row["sample_at_s"]), 2): row for row in csv.DictReader(handle)
                if row["case"] == case and row["world"] == "0"}


def panel(row: dict, *, case: str, width: int = 430, height: int = 480) -> np.ndarray:
    view = np.full((height, width, 3), (28, 25, 22), dtype=np.uint8)
    def line(text: str, y: int, color=(230, 235, 239), scale=.59) -> None:
        cv2.putText(view, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    line("RECORDED NATIVE WALK-STOP", 32, (120, 220, 248), .66)
    line(f"{case} / world 0", 62)
    line(f"simulation t = {float(row['sample_at_s']):.2f} s", 96, scale=.7)
    requested = row["stop_command"] == "1"
    line("STOP REQUESTED" if requested else "WALK REQUESTED", 131,
         (65, 174, 255) if requested else (130, 229, 146), .68)
    line("Actor-like 50 Hz packets", 169, (173, 191, 203))
    line(f"IMU yaw       {float(row['imu_yaw_rate_rad_s']):.3f} rad/s", 197)
    line(f"leg encoders  {float(row['leg_encoder_mean_abs_rad_s']):.3f} rad/s", 224)
    line(f"sole forces   {float(row['left_sole_force_n']):.1f} / {float(row['right_sole_force_n']):.1f} N", 251)
    def gate(label: str, key: str, y: int) -> None:
        accepted = row[key] == "1"
        color = (125, 220, 146) if accepted else (95, 105, 110)
        cv2.rectangle(view, (20, y - 17), (38, y + 1), color, -1)
        cv2.putText(view, label, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .59, (220, 230, 232), 1, cv2.LINE_AA)
    gate("IMU + joints + feet", "fused_accepted", 294)
    gate("same gate, IMU health lost", "fused_imu_lost_accepted", 325)
    cv2.line(view, (20, 346), (410, 346), (75, 81, 82), 1)
    line("Evaluator only (never actor input)", 369, (175, 183, 194), .53)
    line(f"root planar speed {float(row['evaluator_planar_speed_m_s']):.3f} m/s", 394)
    gate("body actually stationary", "evaluator_stationary", 425)
    line("Historical rejected 50 cm frame | pose replay", 460, (125, 138, 144), .44)
    return view


def render_case(case: str) -> Path:
    trace = np.load(SOURCE / f"{case}.npz")
    samples = load_packets(case)
    # The local viewer's MuJoCo predates the contact-sensor XML emitted for
    # native evaluation. Sensors do not affect kinematic pose playback.
    xml = ET.fromstring(BODY.read_text())
    xml.remove(xml.find("sensor"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    # The engineering XML hides budget geoms with alpha=0. Make them visible
    # for a pose replay without changing any recorded dynamics.
    model.geom_rgba[model.geom_rgba[:, 3] == 0, 3] = 1
    model.geom_group[:] = 0
    data = mujoco.MjData(model)
    joint_names = [str(x) for x in trace["leg_names"]]
    joint_addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Leg joint missing from body: {name}")
        joint_addrs.append(model.jnt_qposadr[jid])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.95
    camera.azimuth = 55
    camera.elevation = -14
    origin = trace["root_pose"][0, 0, :2].astype(float)
    output = OUTPUT / f"{case}-world0-body-gate.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1070, 480))
    if not writer.isOpened():
        raise RuntimeError("Could not open MP4 writer")
    try:
        with mujoco.Renderer(model, width=640, height=480) as renderer:
            for i in range(0, len(trace["time_s"]), 2):  # 25 fps from 50 Hz.
                root = trace["root_pose"][i, 0].astype(float).copy()
                root[:2] -= origin
                data.qpos[:] = model.qpos0
                data.qpos[:7] = root
                data.qpos[joint_addrs] = trace["joint_pos"][i, 0]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [root[0], root[1], .24]
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                stamp = round(float(trace["time_s"][i]), 2)
                packet = samples.get(stamp)
                if packet is None:
                    raise ValueError(f"Packet missing at t={stamp:.2f}")
                frame = np.hstack((cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), panel(packet, case=case)))
                writer.write(frame)
    finally:
        writer.release()
    return output


if __name__ == "__main__":
    for label in ("forward-stop-early", "forward-stop-late"):
        print(render_case(label))
