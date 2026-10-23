"""Bounded XML-servo feedback probe for a screened one-foot reset.

The imported XML caps actuator force but does not model MJLab motor speed,
delay, or training. This probe only identifies plausible support teachers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def trial(endpoint: dict, gain: float, damping: float, desired_roll: float,
          duration: float) -> dict:
    model = mujoco.MjModel.from_xml_path(endpoint["model"])
    data = mujoco.MjData(model)
    x, y, z, roll, pitch = endpoint["root_xyz_roll_pitch"]
    data.qpos[:3] = (x, y, z)
    xyzw = Rotation.from_euler("xy", (roll, pitch)).as_quat()
    data.qpos[3:7] = (xyzw[3], *xyzw[:3])
    for name, value in endpoint["joints"].items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    stance = "right"
    stance_site = model.site(f"{stance}_sole_center").id
    swing_site = model.site("left_sole_center").id
    pelvis = model.body("pelvis").id
    start_stance = data.site_xpos[stance_site].copy()
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, stance_site)
    jac = np.vstack((jp, jr))
    wrench = np.linalg.lstsq(jac[:, :6].T, data.qfrc_bias[:6], rcond=None)[0]
    effort = data.qfrc_bias[6:] - jac[:, 6:].T @ wrench
    base_ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        qadr = model.jnt_qposadr[jid]
        dadr = model.jnt_dofadr[jid] - 6
        kp = model.actuator_gainprm[i, 0]
        base_ctrl[i] = data.qpos[qadr] + effort[dadr] / kp
        if model.actuator_ctrllimited[i]:
            base_ctrl[i] = np.clip(base_ctrl[i], *model.actuator_ctrlrange[i])
    hip = model.actuator("right_hip_roll_servo").id
    hip_joint = model.joint("right_hip_roll")
    trace = []
    for step in range(round(duration / model.opt.timestep)):
        roll_now = Rotation.from_quat([*data.qpos[4:7], data.qpos[3]]).as_euler("xyz")[0]
        roll_speed = data.qvel[3]
        data.ctrl[:] = base_ctrl
        data.ctrl[hip] += gain * (roll_now - desired_roll) + damping * roll_speed
        data.ctrl[hip] = np.clip(data.ctrl[hip], *hip_joint.range)
        mujoco.mj_step(model, data)
        if step % 10:
            continue
        feet = [False, False]
        body_contact = False
        for c in data.contact[:data.ncon]:
            names = (model.geom(c.geom1).name, model.geom(c.geom2).name)
            for j, side in enumerate(("left", "right")):
                feet[j] |= f"{side}_sole" in names and "floor" in names
            if "floor" in names and not any("sole" in n for n in names):
                body_contact = True
        com = data.subtree_com[pelvis]
        stance_pos = data.site_xpos[stance_site]
        swing_pos = data.site_xpos[swing_site]
        trace.append({
            "t": round(data.time, 3),
            "pelvis_z_m": round(float(data.qpos[2]), 4),
            "roll_rad": round(float(roll_now), 4),
            "com_minus_stance_xy_m": np.round(com[:2] - stance_pos[:2], 4).tolist(),
            "swing_z_m": round(float(swing_pos[2]), 4),
            "stance_slip_m": round(float(np.linalg.norm(stance_pos[:2] - start_stance[:2])), 4),
            "feet_grounded": feet,
            "body_contact": body_contact,
            "rated_leg": bool(np.any(np.abs(data.actuator_force[:10]) >= .995 * model.actuator_forcerange[:10, 1])),
        })
        if data.qpos[2] < .09 or abs(roll_now) > math.radians(70):
            break
    after = [x for x in trace if x["t"] >= .5]
    held = (bool(after) and trace[-1]["t"] >= duration - .02
            and min(x["swing_z_m"] for x in after) > .025
            and not any(x["body_contact"] for x in trace)
            and max(x["stance_slip_m"] for x in trace) < .02)
    return {
        "gain": gain, "damping": damping, "desired_roll_rad": desired_roll,
        "duration_s": trace[-1]["t"], "held": held,
        "min_swing_after_half_s_m": min((x["swing_z_m"] for x in after), default=None),
        "max_stance_slip_m": max(x["stance_slip_m"] for x in trace),
        "rated_leg_fraction": round(sum(x["rated_leg"] for x in trace) / len(trace), 4),
        "body_contact_frames": sum(x["body_contact"] for x in trace),
        "final": trace[-1], "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", type=Path)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--damping", type=float, required=True)
    parser.add_argument("--desired-roll", type=float, required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = trial(json.loads(args.endpoint.read_text()), args.gain, args.damping,
                   args.desired_roll, args.duration)
    result["endpoint"] = str(args.endpoint)
    result["model"] = "Imported XML capped-force position servos; no MJLab DC motor speed/command delay"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
