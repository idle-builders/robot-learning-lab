"""Replay ideal proprioceptive XY odometry on retained native 50 cm walks.

This is an upper-bound sensor-role screen on the rejected Direct Frame. The
native root position is evaluator-only; ideal orientation, joint positions
and MuJoCo sole-contact bits stand in for candidate physical packets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-transfer-20261008"
BODY = SOURCE / "cove50-direct-training.xml"
OUT = ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1"
CASES = (
    "planar200v3-forward",
    "varied150-forward-seed144",
    "robust10v200-forward-seed144",
)
FEET = ("left", "right")
# Illustrative sensitivity settings, not specifications of chosen hardware.
YAW_DRIFT_DEG_PER_S = 1.0
ENCODER_BIN_DEG = 0.5
CONTACT_DELAY_SAMPLES = 1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrate(feet_xy: np.ndarray, contact: np.ndarray,
              initial_xy: np.ndarray, *, gate_contact: bool) -> np.ndarray:
    result = np.zeros((len(feet_xy), 2), dtype=float)
    result[0] = initial_xy
    for step in range(1, len(feet_xy)):
        usable = (contact[step - 1] & contact[step] if gate_contact
                  else np.ones(len(FEET), dtype=bool))
        delta = (np.mean(feet_xy[step - 1, usable] - feet_xy[step, usable],
                         axis=0) if usable.any() else np.zeros(2))
        result[step] = result[step - 1] + delta
    return result


def score(estimated: np.ndarray, actual: np.ndarray) -> dict:
    error = np.linalg.norm(estimated - actual, axis=1)
    return {
        "final_xy_error_m": round(float(error[-1]), 5),
        "mean_xy_error_m": round(float(error.mean()), 5),
        "p95_xy_error_m": round(float(np.quantile(error, .95)), 5),
        "final_estimated_xy_m": estimated[-1].round(5).tolist(),
    }


def yaw_drifted_quaternion(quat: np.ndarray, elapsed_s: float) -> np.ndarray:
    """Pre-multiply a world-z attitude error into MuJoCo's wxyz quaternion."""
    half_angle = np.deg2rad(YAW_DRIFT_DEG_PER_S * elapsed_s) / 2
    c, s = np.cos(half_angle), np.sin(half_angle)
    w, x, y, z = quat
    return np.array([c * w - s * z, c * x - s * y,
                     c * y + s * x, c * z + s * w])


def replay(model: mujoco.MjModel, path: Path) -> tuple[dict, list[dict]]:
    report_path = path.with_suffix(".json")
    native_report = json.loads(report_path.read_text())
    if native_report["body_sha256"] != sha(BODY):
        raise ValueError(f"Body hash differs from native rollout: {path}")
    with np.load(path, allow_pickle=False) as saved:
        qpos = saved["qpos"].astype(float)
        dt = float(saved["dt"])
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Qpos does not match body: {path}")
    actual = qpos[:, :2].copy()
    contact = np.zeros((len(qpos), len(FEET)), dtype=bool)
    feet_ideal_orientation = np.zeros((len(qpos), len(FEET), 2))
    feet_frozen_orientation = np.zeros_like(feet_ideal_orientation)
    feet_yaw_drift = np.zeros_like(feet_ideal_orientation)
    feet_encoder_quantized = np.zeros_like(feet_ideal_orientation)
    feet_combined = np.zeros_like(feet_ideal_orientation)
    feet_world_evaluator = np.zeros_like(feet_ideal_orientation)
    real = mujoco.MjData(model)
    kinematics = mujoco.MjData(model)
    initial_quat = qpos[0, 3:7].copy()
    for step, state in enumerate(qpos):
        real.qpos[:] = state
        mujoco.mj_forward(model, real)
        for foot_index, foot in enumerate(FEET):
            sensor = model.sensor(f"{foot}_sole_floor_found").id
            contact[step, foot_index] = real.sensordata[model.sensor_adr[sensor]] > 0
            feet_world_evaluator[step, foot_index] = real.site(
                f"{foot}_sole_center").xpos[:2]
        kinematics.qpos[:] = state
        kinematics.qpos[:3] = 0  # Pose estimator never receives native root XY.
        mujoco.mj_forward(model, kinematics)
        for foot_index, foot in enumerate(FEET):
            feet_ideal_orientation[step, foot_index] = kinematics.site(
                f"{foot}_sole_center").xpos[:2]
        kinematics.qpos[3:7] = initial_quat
        mujoco.mj_forward(model, kinematics)
        for foot_index, foot in enumerate(FEET):
            feet_frozen_orientation[step, foot_index] = kinematics.site(
                f"{foot}_sole_center").xpos[:2]
        for destination, drift, quantize in (
            (feet_yaw_drift, True, False),
            (feet_encoder_quantized, False, True),
            (feet_combined, True, True),
        ):
            kinematics.qpos[:] = state
            kinematics.qpos[:3] = 0
            if drift:
                kinematics.qpos[3:7] = yaw_drifted_quaternion(
                    state[3:7], step * dt)
            if quantize:
                bin_rad = np.deg2rad(ENCODER_BIN_DEG)
                kinematics.qpos[7:] = np.round(state[7:] / bin_rad) * bin_rad
            mujoco.mj_forward(model, kinematics)
            for foot_index, foot in enumerate(FEET):
                destination[step, foot_index] = kinematics.site(
                    f"{foot}_sole_center").xpos[:2]
    delayed_contact = np.zeros_like(contact)
    delayed_contact[CONTACT_DELAY_SAMPLES:] = contact[:-CONTACT_DELAY_SAMPLES]
    estimators = {
        "ideal_orientation_joint_contact": integrate(
            feet_ideal_orientation, contact, actual[0], gate_contact=True),
        "ideal_orientation_joint_all_feet": integrate(
            feet_ideal_orientation, contact, actual[0], gate_contact=False),
        "frozen_orientation_joint_contact": integrate(
            feet_frozen_orientation, contact, actual[0], gate_contact=True),
        "yaw_drift_1deg_per_s": integrate(
            feet_yaw_drift, contact, actual[0], gate_contact=True),
        "encoder_quantization_0p5deg": integrate(
            feet_encoder_quantized, contact, actual[0], gate_contact=True),
        "contact_delay_1frame": integrate(
            feet_ideal_orientation, delayed_contact, actual[0],
            gate_contact=True),
        "combined_illustrative_impairments": integrate(
            feet_combined, delayed_contact, actual[0], gate_contact=True),
    }
    # The sole-center world motion is evaluator-only. It can include foot roll
    # and shifting contact patches; do not label it measured floor slip.
    travel = {}
    for foot_index, foot in enumerate(FEET):
        continuous = contact[1:, foot_index] & contact[:-1, foot_index]
        step_motion = np.linalg.norm(np.diff(feet_world_evaluator[:, foot_index],
                                             axis=0), axis=1)
        travel[foot] = round(float(step_motion[continuous].sum()), 5)
    samples = []
    for step in range(len(qpos)):
        samples.append({
            "t_s": round(step * dt, 4),
            "actual_xy_m_evaluator_only": actual[step].round(5).tolist(),
            "contact_found_idealized": contact[step].tolist(),
            "estimated_xy_m": {name: estimate[step].round(5).tolist()
                               for name, estimate in estimators.items()},
        })
    video = path.with_suffix(".mp4")
    summary = {
        "case": path.stem,
        "trace_source": str(path.relative_to(ROOT)),
        "trace_sha256": sha(path),
        "native_report": str(report_path.relative_to(ROOT)),
        "native_report_sha256": sha(report_path),
        "source_video": str(video.relative_to(ROOT)) if video.exists() else None,
        "source_video_sha256": sha(video) if video.exists() else None,
        "dt_s": dt,
        "samples": len(qpos),
        "actual_xy_displacement_m_evaluator_only": round(
            float(np.linalg.norm(actual[-1] - actual[0])), 5),
        "continuous_contact_fraction": {
            foot: round(float(np.mean(contact[1:, i] & contact[:-1, i])), 4)
            for i, foot in enumerate(FEET)
        },
        "illustrative_impairments": {
            "yaw_drift_deg_per_s": YAW_DRIFT_DEG_PER_S,
            "encoder_bin_deg": ENCODER_BIN_DEG,
            "contact_delivery_delay_frames": CONTACT_DELAY_SAMPLES,
        },
        "contact_sole_center_world_travel_m_evaluator_only": travel,
        "scores": {name: score(estimate, actual)
                   for name, estimate in estimators.items()},
    }
    return summary, samples


def render_error_video(source_video: Path, samples: list[dict], output: Path) -> None:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode native walk video: {source_video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_h = 312
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height + panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot encode route video: {output}")
    truth = np.asarray([row["actual_xy_m_evaluator_only"] for row in samples])
    names = ("ideal_orientation_joint_contact", "ideal_orientation_joint_all_feet",
             "frozen_orientation_joint_contact",
             "combined_illustrative_impairments")
    paths = {name: np.asarray([row["estimated_xy_m"][name]
                               for row in samples]) for name in names}
    colors = ((80, 220, 80), (60, 80, 255), (70, 220, 255),
              (220, 170, 70))
    labels = ("Contact + orientation", "Both feet assumed planted",
              "Contact + frozen orientation", "Drift + round + delay")
    max_error = max(float(np.linalg.norm(path - truth, axis=1).max())
                    for path in paths.values())
    axis_max = max(.5, float(np.ceil(max_error * 2) / 2))
    bar_left, bar_right = int(width * .42), width - 75

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        sample_index = min(len(samples) - 1,
                           round((frame_index / fps) / samples[1]["t_s"]))
        panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
        cv2.putText(panel, f"XY pose error against native path at {frame_index / fps:.2f}s",
                    (12, 31), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "Ideal orientation/joints/contact; evaluator scores only",
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (190, 190, 190), 1, cv2.LINE_AA)
        for index, name in enumerate(names):
            y = 105 + index * 53
            error = float(np.linalg.norm(paths[name][sample_index] -
                                         truth[sample_index]))
            cv2.putText(panel, labels[index], (12, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .57, colors[index], 2,
                        cv2.LINE_AA)
            cv2.rectangle(panel, (bar_left, y - 12), (bar_right, y + 13),
                          (45, 45, 45), -1)
            bar_end = bar_left + round((bar_right - bar_left) *
                                        min(1., error / axis_max))
            cv2.rectangle(panel, (bar_left, y - 12), (bar_end, y + 13),
                          colors[index], -1)
            cv2.putText(panel, f"{error:.2f} m", (bar_right + 8, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(np.vstack([frame, panel]))
        frame_index += 1
    capture.release()
    writer.release()
    if frame_index == 0:
        raise RuntimeError("Native video was empty")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if model.nq != 28:
        raise ValueError("Expected 21-joint 50 cm historical model")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        summary, samples = replay(model, SOURCE / f"{case}.npz")
        trace_path = OUT / f"{case}-trace.json"
        trace_path.write_text(json.dumps(samples, indent=2) + "\n")
        summary["derived_trace"] = str(trace_path.relative_to(ROOT))
        summary["derived_trace_sha256"] = sha(trace_path)
        if case == "varied150-forward-seed144":
            source_video = SOURCE / f"{case}.mp4"
            video_path = OUT / f"{case}-odometry.mp4"
            render_error_video(source_video, samples, video_path)
            summary["derived_video"] = str(video_path.relative_to(ROOT))
            summary["derived_video_sha256"] = sha(video_path)
        rows.append(summary)
    result = {
        "schema": "robot-learning-lab.native-proprioceptive-odometry.v1",
        "current_design_status": json.loads(
            (ROOT / "robot/ape-current-design.json").read_text())["status"],
        "historical_body": str(BODY.relative_to(ROOT)),
        "historical_body_sha256": sha(BODY),
        "analysis_code_sha256": sha(Path(__file__)),
        "mujoco_version": mujoco.__version__,
        "scope": "Retained native 50 cm gait qpos with idealized orientation, exact joints and MuJoCo sole-contact bits; root XY used only for scoring and initial alignment. Added illustrative deterministic yaw drift, encoder rounding and one-frame contact delay sensitivity; no measured packet errors, force calibration or owner-loop handoff.",
        "cases": rows,
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{row["case"]: row["scores"]} for row in rows], indent=2))


if __name__ == "__main__":
    main()
