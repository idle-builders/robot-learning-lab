"""Actuator-driven MuJoCo hold check for a screened one-foot endpoint."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", type=Path)
    parser.add_argument("--stance", choices=("left", "right"), default="right")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-out", type=Path,
                        help="Write gravity-preloaded motor targets for a native MJLab control probe")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--integrator-euler", action="store_true",
                        help="Match MJLab's parent Euler integrator for a sensitivity probe")
    parser.add_argument("--video", type=Path,
                        help="Record this XML-servo feasibility probe")
    args = parser.parse_args()
    spec = json.loads(args.endpoint.read_text())
    model = mujoco.MjModel.from_xml_path(spec["model"])
    if args.integrator_euler:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    renderer = None
    if args.video:
        # Selected training proxies are hidden visually. Reveal only this
        # in-memory render model; force/contact behavior is unaffected.
        for geom_id in range(model.ngeom):
            if (model.geom_bodyid[geom_id] != 0 and model.geom_group[geom_id] == 4
                    and model.geom_rgba[geom_id, 3] == 0):
                model.geom_rgba[geom_id, 3] = 1
                model.geom_group[geom_id] = 0
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0, 0, .24)
        camera.distance = .75
        camera.azimuth = 45
        camera.elevation = -14
    data = mujoco.MjData(model)
    x, y, z, roll, pitch = spec["root_xyz_roll_pitch"]
    data.qpos[:3] = (x, y, z)
    quat = Rotation.from_euler("xy", (roll, pitch)).as_quat()
    data.qpos[3:7] = (quat[3], *quat[:3])
    for name, value in spec["joints"].items():
        data.qpos[model.joint(name).qposadr[0]] = value
    for side in ("left", "right"):
        for name, value in (("shoulder_pitch", -0.6), ("elbow_pitch", -1.1)):
            joint = f"{side}_{name}"
            if joint not in spec["joints"]:
                data.qpos[model.joint(joint).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    stance_id = model.site(f"{args.stance}_sole_center").id
    swing_id = model.site(f"{'left' if args.stance == 'right' else 'right'}_sole_center").id
    start_stance = data.site_xpos[stance_id].copy()
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, stance_id)
    jac = np.vstack((jp, jr))
    wrench = np.linalg.lstsq(jac[:, :6].T, data.qfrc_bias[:6], rcond=None)[0]
    effort = data.qfrc_bias[6:] - jac[:, 6:].T @ wrench
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        qadr = model.jnt_qposadr[jid]
        dadr = model.jnt_dofadr[jid] - 6
        kp = model.actuator_gainprm[i, 0]
        target = data.qpos[qadr] + effort[dadr] / kp
        data.ctrl[i] = (np.clip(target, *model.actuator_ctrlrange[i])
                        if model.actuator_ctrllimited[i] else target)
    if args.control_out:
        control_joints = {
            model.joint(model.actuator_trnid[i, 0]).name: float(data.ctrl[i])
            for i in range(model.nu)
        }
        args.control_out.parent.mkdir(parents=True, exist_ok=True)
        args.control_out.write_text(json.dumps({
            "joints": control_joints,
            "source_endpoint": str(args.endpoint),
            "method": "static gravity-preloaded actuator target from stance-foot wrench",
        }, indent=2) + "\n")

    trace = []
    frames = []
    self_pairs = {}
    max_self_penetration = 0.0
    for step in range(round(args.duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if renderer and step % round(.04 / model.opt.timestep) == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
        if step % round(0.02 / model.opt.timestep):
            continue
        stance = data.site_xpos[stance_id].copy()
        swing = data.site_xpos[swing_id].copy()
        quat = data.qpos[3:7]
        tilt = math.degrees(math.acos(np.clip(1 - 2 * (quat[1]**2 + quat[2]**2), -1, 1)))
        grounded = [False, False]
        non_sole_floor_contact = False
        hand_ground_contact = False
        for contact in data.contact[:data.ncon]:
            names = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
            if model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2]:
                pair = tuple(sorted((model.body(model.geom_bodyid[contact.geom1]).name,
                                     model.body(model.geom_bodyid[contact.geom2]).name)))
                self_pairs[pair] = self_pairs.get(pair, 0) + 1
                max_self_penetration = max(max_self_penetration, -float(contact.dist))
            for j, side in enumerate(("left", "right")):
                grounded[j] |= f"{side}_sole" in names and "floor" in names
            if "floor" in names and not any("sole" in name for name in names):
                non_sole_floor_contact = True
                hand_ground_contact |= any("hand" in name or "gripper" in name
                                           for name in names)
        trace.append({
            "t": round(data.time, 3), "pelvis_z_m": round(float(data.qpos[2]), 4),
            "tilt_deg": round(tilt, 2), "swing_z_m": round(float(swing[2]), 4),
            "stance_z_m": round(float(stance[2]), 4),
            "stance_slip_m": round(float(np.linalg.norm(stance[:2]-start_stance[:2])), 4),
            "grounded": grounded,
            "max_leg_effort_nm": round(float(np.max(np.abs(data.actuator_force[:10]))), 3),
            "rated_leg": bool(np.any(np.abs(data.actuator_force[:10]) >=
                                       .995 * model.actuator_forcerange[:10, 1])),
            "non_sole_floor_contact": non_sole_floor_contact,
            "hand_ground_contact": hand_ground_contact,
        })
        if data.qpos[2] < 0.09 or tilt > 75:
            break
    report = {
        "endpoint": str(args.endpoint), "stance_side": args.stance,
        "servo_target_clip": "apply actuator ctrlrange only when ctrllimited is enabled",
        "integrator": "Euler" if args.integrator_euler else "body XML setting",
        "duration_s": trace[-1]["t"],
        "result": "held" if trace[-1]["t"] >= args.duration - 0.02 and
                  min(r["swing_z_m"] for r in trace[25:]) >= 0.025 and
                  max(r["stance_slip_m"] for r in trace) <= 0.02 and
                  not any(r["non_sole_floor_contact"] for r in trace) else "failed",
        "final": trace[-1], "max_stance_slip_m": max(r["stance_slip_m"] for r in trace),
        "min_swing_height_after_half_second_m": min((r["swing_z_m"] for r in trace[25:]), default=None),
        "max_leg_effort_nm": max(r["max_leg_effort_nm"] for r in trace),
        "rated_leg_fraction": round(sum(r["rated_leg"] for r in trace) / len(trace), 4),
        "non_sole_floor_contact_frames": sum(r["non_sole_floor_contact"] for r in trace),
        "hand_ground_contact_frames": sum(r["hand_ground_contact"] for r in trace),
        "self_contact_pairs": [{"bodies": pair, "sampled_contact_frames": count}
                               for pair, count in sorted(self_pairs.items())],
        "max_self_penetration_m": round(max_self_penetration, 6),
        "trace": trace,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
    if renderer:
        import imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
        renderer.close()
        print("video:", args.video)


if __name__ == "__main__":
    main()
