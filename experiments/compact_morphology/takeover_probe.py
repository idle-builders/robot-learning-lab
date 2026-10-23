"""Measure whether a servo-pose expert can recover saved learner states.

This is a native MuJoCo diagnostic, not policy training or a learned rollout.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from dynamic_screen import SHAPES, ik


MODEL = Path(__file__).with_name("reach_models") / "ape_short_torso.xml"


def goal_controls(model, target):
    values, fit = ik(model, SHAPES["ape_short_torso"], (target[0], abs(target[1]), 0.045))
    goal = np.zeros(model.nu)
    side = "left" if target[1] > 0 else "right"
    for i in range(model.nu):
        actuator = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint = actuator.removesuffix("_servo")
        if joint.endswith("hip_pitch"):
            goal[i] = values[2]
        elif joint.endswith("knee"):
            goal[i] = values[3]
        elif joint.endswith("ankle_pitch"):
            goal[i] = values[4]
        elif joint == f"{side}_shoulder_pitch":
            goal[i] = values[5]
        elif joint == f"{side}_shoulder_roll":
            goal[i] = values[6] * (1 if side == "left" else -1)
    return goal, fit


def run(model, trace, index, goal, duration=6.0):
    data = mujoco.MjData(model)
    data.qpos[:] = trace["qpos"][index]
    data.mocap_pos[:] = trace["mocap_pos"][index]
    if index:
        mujoco.mj_differentiatePos(
            model, data.qvel, 0.02, trace["qpos"][index - 1], trace["qpos"][index]
        )
    mujoco.mj_forward(model, data)
    start = data.qpos[7:].copy()
    sensors = [model.sensor_adr[model.sensor(f"{side}_hand_target_found").id] for side in ("left", "right")]
    imu_site = model.site("imu").id
    pelvis = model.body("pelvis").id
    streak = longest = contacts = 0
    survived = True
    contact_after_approach = 0
    control_steps = int(duration / 0.02)
    for step in range(control_steps):
        fraction = np.clip((step * 0.02 - 0.15) / 1.6, 0, 1)
        fraction = fraction * fraction * (3 - 2 * fraction)
        data.ctrl[:] = start + fraction * (goal - start)
        for _ in range(10):
            mujoco.mj_step(model, data)
        touched = any(data.sensordata[i] > 0 for i in sensors)
        contacts += touched
        contact_after_approach += touched and step >= 100
        streak = streak + 1 if touched else 0
        longest = max(longest, streak)
        upright = data.xpos[pelvis, 2] >= 0.118 and data.site_xmat[imu_site, 8] >= 0.65
        within_limits = np.all(data.qpos[7:] >= model.jnt_range[1:, 0]) and np.all(
            data.qpos[7:] <= model.jnt_range[1:, 1]
        )
        if not upright or not within_limits or not np.isfinite(data.qpos).all():
            survived = False
            break
    return {
        "takeover_after_control_step": index + 1,
        "survived": survived,
        "survival_steps_after_takeover": step + 1,
        "contact_steps": int(contacts),
        "contact_steps_after_approach": int(contact_after_approach),
        "longest_contact_streak_steps": longest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", default="0,20,50,100,150,180,200,250")
    parser.add_argument("--relative-ik", action="store_true")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    with np.load(args.trace) as trace:
        target = trace["mocap_pos"][0, 0].tolist()
        rows = []
        for index in map(int, args.indices.split(",")):
            ik_target = target
            if args.relative_ik:
                state = mujoco.MjData(model)
                state.qpos[:] = trace["qpos"][index]
                state.mocap_pos[:] = trace["mocap_pos"][index]
                mujoco.mj_forward(model, state)
                pelvis = model.body("pelvis").id
                relative = state.xmat[pelvis].reshape(3, 3).T @ (
                    np.asarray(target) - state.xpos[pelvis]
                )
                ik_target = [relative[0], relative[1], relative[2] + SHAPES["ape_short_torso"]["hip"] - target[2] + 0.045]
            goal, fit = goal_controls(model, ik_target)
            row = run(model, trace, index, goal)
            row["ik_target_xyz_m"] = [float(v) for v in ik_target]
            row["ik_hand_error_m"] = fit["hand_error_m"]
            rows.append(row)
    result = {"target_xyz_m": target, "ik_frame": "pelvis_relative" if args.relative_ik else "initial_world", "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
