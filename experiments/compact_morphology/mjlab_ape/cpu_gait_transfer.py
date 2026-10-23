"""Small Pollen-style ONNX/MuJoCo probe for the saved ape walker.

Run the identical 20-output actor on its 20-DOF source and the 22-DOF elbow
body. This CPU probe is a diagnostic, not the native MJLab/Warp evaluation or
evidence of a trained 22-DOF policy.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parents[1] / "playground_models"
SOURCE = HERE / "ape_sourced_v1.xml"
ELBOW = HERE / "ape_sourced_v3_elbows_small_head.xml"


def joint_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(1, model.njnt)]


def actuator_joints(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[a, 0])) for a in range(model.nu)]


def defaults(name):
    if name.endswith("hip_pitch") or name.endswith("ankle_pitch"):
        return -0.2
    if name.endswith("knee"):
        return 0.4
    if name.endswith("shoulder_pitch"):
        return -0.6
    if name.endswith("elbow_pitch"):
        return -1.1
    return 0.0


def run(onnx_path: Path, model_path: Path, seconds: float, speed: float):
    old = mujoco.MjModel.from_xml_path(str(SOURCE))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nq in (27, 29) and old.nq == 27
    model.opt.timestep = 0.002
    # MJLab's parent scene uses Euler when it attaches this source MJCF.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    data = mujoco.MjData(model)
    names, old_names = joint_names(model), joint_names(old)
    act_names, old_act_names = actuator_joints(model), actuator_joints(old)
    assert all(name in names and name in act_names for name in old_names)
    data.qpos[:3] = [0.0, 0.0, 0.181]
    data.qpos[3] = 1.0
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = defaults(name)
    default_ctrl = np.array([defaults(name) for name in act_names])
    scale = np.array([0.3 if i < 10 else 0.0 for i in range(model.nu)])
    data.ctrl[:] = default_ctrl
    mujoco.mj_forward(model, data)

    old_joint_qpos = [int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in old_names]
    old_joint_qvel = [int(model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in old_names]
    new_action_for_old = [act_names.index(name) for name in old_act_names]
    old_default = np.array([defaults(name) for name in old_names], dtype=np.float32)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    soles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
             for side in ("left", "right")]
    hand_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_{part}")
                  for side in ("left", "right") for part in ("hand", "finger_tip")}
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_sole_center")
                for side in ("left", "right")]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 69]
    last_action = np.zeros(20, np.float32)
    command = np.array([speed, 0.0, 0.0], np.float32)
    previous_site_xy = np.array([data.site_xpos[i, :2].copy() for i in site_ids])
    air_frames = [0, 0]
    touches = []
    slip = [0.0, 0.0]
    peak_sole_z = [0.0, 0.0]
    sat_frames = 0
    hand_force_peak = 0.0
    first_meter = None
    frames = []
    for step in range(round(seconds / 0.02)):
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        gravity_b = data.xmat[pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        obs = np.concatenate((
            data.qvel[3:6], gravity_b,
            data.qpos[old_joint_qpos] - old_default,
            data.qvel[old_joint_qvel], last_action, command,
        )).astype(np.float32)
        assert obs.shape == (69,)
        action = session.run(None, {session.get_inputs()[0].name: obs[None, :]})[0][0]
        data.ctrl[:] = default_ctrl
        data.ctrl[new_action_for_old] += action * scale[new_action_for_old]
        last_action = action.copy()
        for _ in range(10):
            mujoco.mj_step(model, data)
        force = [0.0, 0.0]
        hand_force = 0.0
        for ci in range(data.ncon):
            contact = data.contact[ci]
            pair = {int(contact.geom1), int(contact.geom2)}
            if floor not in pair:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, wrench)
            magnitude = float(np.linalg.norm(wrench[:3]))
            for side in range(2):
                if soles[side] in pair:
                    force[side] += magnitude
            if pair & hand_geoms:
                hand_force += magnitude
        sites = np.array([data.site_xpos[i].copy() for i in site_ids])
        grounded = [x > 0.5 for x in force]
        for side in range(2):
            peak_sole_z[side] = max(peak_sole_z[side], float(sites[side, 2]))
            if grounded[side]:
                slip[side] += float(np.linalg.norm(sites[side, :2] - previous_site_xy[side]))
                if air_frames[side] >= 4:
                    touches.append(side)
                air_frames[side] = 0
            else:
                air_frames[side] += 1
        previous_site_xy = sites[:, :2].copy()
        sat_frames += int(np.any(abs(data.actuator_force[:10]) >= 1.56))
        hand_force_peak = max(hand_force_peak, hand_force)
        qw, qx, qy, qz = data.qpos[3:7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        frames.append({"t": round((step + 1) * 0.02, 3),
                       "root": data.qpos[:3].round(5).tolist(),
                       "yaw": round(yaw, 4), "feet": grounded,
                       "force_n": np.round(force, 2).tolist(),
                       "site_z": sites[:, 2].round(4).tolist()})
        if first_meter is None and data.qpos[0] >= 1.0:
            first_meter = {"time_s": round((step + 1) * 0.02, 3),
                           "lateral_m": round(float(data.qpos[1]), 3),
                           "yaw_rad": round(yaw, 3),
                           "alternating_touchdowns": sum(a != b for a, b in zip(touches, touches[1:]))}
        if data.qpos[2] < 0.08 or not np.isfinite(data.qpos).all():
            break
    return {"model": model_path.name, "diagnostic_only": True,
            "duration_s": frames[-1]["t"], "first_meter": first_meter,
            "final_root": frames[-1]["root"], "final_yaw": frames[-1]["yaw"],
            "touchdowns_left_right": [touches.count(0), touches.count(1)],
            "alternating_touchdowns": sum(a != b for a, b in zip(touches, touches[1:])),
            "slip_m": np.round(slip, 3).tolist(),
            "peak_sole_z_m": np.round(peak_sole_z, 3).tolist(),
            "fraction_saturated": round(sat_frames / len(frames), 3),
            "hand_force_peak_n": round(hand_force_peak, 2), "frames": frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--speed", type=float, default=0.35)
    args = parser.parse_args()
    reports = [run(args.onnx, path, args.seconds, args.speed) for path in (SOURCE, ELBOW)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
