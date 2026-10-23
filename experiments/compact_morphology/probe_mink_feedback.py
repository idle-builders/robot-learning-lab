"""Try Mink's G1-style hand/feet/COM IK as a contact-recovery teacher.

This is a native-MuJoCo diagnostic from a saved learned trace. It is not a
learned policy result, and the trace velocity must be estimated from poses.
"""

import argparse
import json
from pathlib import Path

import mink
import mujoco
import numpy as np


HERE = Path(__file__).parent
MODEL = HERE / "reach_models/ape_short_torso.xml"
BANK = HERE / "contact_reset_bank.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[10, 31, 120, 170, 220])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--free-root", action="store_true", help="Match Mink's G1 example instead of freezing the floating root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.opt.timestep = 0.002
    bank = json.loads(BANK.read_text())["poses"]
    trace = np.load(args.trace)
    contacts = [model.sensor_adr[model.sensor(f"{side}_hand_target_found").id] for side in ("left", "right")]
    pelvis_id = model.body("pelvis").id
    home_z = model.key_qpos[0, 2]

    def start(frame):
        data = mujoco.MjData(model)
        data.qpos[:] = trace["qpos"][frame]
        data.mocap_pos[:] = trace["mocap_pos"][frame]
        if frame:
            mujoco.mj_differentiatePos(model, data.qvel, 0.02, trace["qpos"][frame - 1], trace["qpos"][frame])
        data.ctrl[:] = data.qpos[7:]
        mujoco.mj_forward(model, data)
        return data

    def rollout(frame, controller):
        data = start(frame)
        target = data.mocap_pos[0].copy()
        side = "left" if target[1] >= 0 else "right"
        nearest = min(bank, key=lambda pose: np.linalg.norm(np.asarray(pose["target_xyz_m"]) - target))
        servo = np.asarray(nearest["ctrl"])

        if controller == "mink":
            configuration = mink.Configuration(model, q=data.qpos.copy())
            hand = mink.FrameTask(f"{side}_hand_center", "site", position_cost=5.0, orientation_cost=0.0)
            hand.set_target(mink.SE3.from_translation(np.array([target[0], target[1], 0.045])))
            feet = [mink.FrameTask(f"{s}_sole_center", "site", position_cost=10.0, orientation_cost=1.0)
                    for s in ("left", "right")]
            for foot in feet:
                foot.set_target_from_configuration(configuration)
            posture = mink.PostureTask(model, cost=0.05)
            posture.set_target(np.asarray(nearest["qpos"]))
            com = mink.ComTask(cost=3.0)
            support = np.mean([data.site_xpos[model.site(f"{s}_sole_center").id] for s in ("left", "right")], axis=0)
            com.set_target(np.array([support[0], support[1], data.subtree_com[pelvis_id, 2]]))
            tasks = [hand, *feet, com, posture]
            limits = [mink.ConfigurationLimit(model)]
            frozen_root = [] if args.free_root else [mink.DofFreezingTask(model, list(range(6)))]

        streak = longest = contact_steps = 0
        survival = args.steps
        closest = float("inf")
        for i in range(args.steps):
            if controller == "mink":
                configuration.update(data.qpos)
                velocity = mink.solve_ik(configuration, tasks, 0.02, "daqp", damping=1e-2,
                                         limits=limits, constraints=frozen_root)
                configuration.integrate_inplace(velocity, 0.02)
                control = configuration.q[7:]
            else:
                control = servo
            data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            for _ in range(10):
                mujoco.mj_step(model, data)
            touched = any(data.sensordata[adr] > 0 for adr in contacts)
            contact_steps += int(touched)
            streak = streak + 1 if touched else 0
            longest = max(longest, streak)
            closest = min(closest, *(np.linalg.norm(data.site_xpos[model.site(f"{s}_hand_center").id] - target)
                                     for s in ("left", "right")))
            upright = data.xmat[pelvis_id].reshape(3, 3)[2, 2] >= 0.65 and data.xpos[pelvis_id, 2] >= 0.65 * home_z
            if not upright:
                survival = i + 1
                break
        return {"frame": frame, "controller": controller, "target_xyz_m": target.tolist(),
                "survival_steps": survival, "contact_steps": contact_steps,
                "longest_contact_streak_steps": longest, "closest_hand_m": round(float(closest), 4)}

    results = [rollout(frame, controller) for frame in args.frames for controller in ("servo", "mink")]
    output = json.dumps({"trace": str(args.trace), "model": str(MODEL), "steps": args.steps,
                         "free_root_ik": args.free_root,
                         "velocity_estimate": "mj_differentiatePos from adjacent recorded poses", "trials": results}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
