"""Observe the released ToddlerBot 2XC walk policy without changing its dynamics.

Run with .venv-toddlerbot/bin/python; output is a small JSON trace for gait review.
"""

import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".cache/upstream/toddlerbot"
OUT = ROOT / "artifacts/toddlerbot-2xc-physical-baseline-20260928"
os.chdir(UPSTREAM)  # ToddlerBot resolves model and checkpoint paths from its root.

from toddlerbot.policies.walk import WalkPolicy  # noqa: E402
from toddlerbot.sim.mujoco_sim import MuJoCoSim  # noqa: E402
from toddlerbot.sim.robot import Robot  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command_m_s, steps=650, initial_roll_deg=0.0, render=False):
    robot = Robot("toddlerbot_2xc")
    sim = MuJoCoSim(robot, vis_type="none")
    if initial_roll_deg:
        base_rotation = Rotation.from_quat(sim.data.qpos[3:7], scalar_first=True)
        offset = Rotation.from_euler("x", np.deg2rad(initial_roll_deg))
        sim.data.qpos[3:7] = (offset * base_rotation).as_quat(scalar_first=True)
        mujoco.mj_forward(sim.model, sim.data)
    command = np.zeros(8, dtype=np.float32)
    command[5] = command_m_s
    policy = WalkPolicy(
        name="walk",
        robot=robot,
        init_motor_pos=sim.get_observation().motor_pos,
        path="toddlerbot_2xc_walk_rsl_20251226_114612",
        fixed_command=command,
    )
    floor = sim.model.geom("floor").id
    feet = {side: sim.model.geom(f"{side}_ankle_roll_link_collision").id for side in ("left", "right")}
    leg_names = [n for n in robot.motor_ordering if any(k in n for k in ("hip", "knee", "ankle"))]
    leg_actuator_ids = {n: sim.model.actuator(n).id for n in leg_names}
    renderer = mujoco.Renderer(sim.model, height=480, width=640) if render else None
    writer = imageio.get_writer(OUT / f"walk_{command_m_s:.2f}.mp4", fps=25, codec="libx264", quality=7) if render else None
    trace = []
    for step in range(steps):
        obs = sim.get_observation()
        obs.time = step * policy.control_dt
        _, target = policy.step(obs, sim)
        sim.set_motor_target(target)
        sim.step()
        if renderer is not None and step >= 325 and step % 2 == 0:
            renderer.update_scene(sim.data, camera="perspective")
            writer.append_data(renderer.render())
        contacts = {side: 0.0 for side in feet}
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            for side, geom in feet.items():
                if {contact.geom1, contact.geom2} == {floor, geom}:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(sim.model, sim.data, i, force)
                    contacts[side] += max(0.0, float(force[0]))
        quat = sim.data.body("torso").xquat
        roll, pitch, yaw = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
        trace.append({
            "t_s": round((step + 1) * policy.control_dt, 3),
            "base_xyz_m": np.round(sim.data.qpos[:3], 5).tolist(),
            "torso_rpy_rad": np.round([roll, pitch, yaw], 5).tolist(),
            "foot_normal_force_n": {k: round(v, 3) for k, v in contacts.items()},
            "leg_torque_nm": {name: round(float(sim.data.actuator_force[i]), 4) for name, i in leg_actuator_ids.items()},
        })
        if not np.isfinite(sim.data.qpos).all() or sim.data.qpos[2] < 0.13:
            break
    if writer is not None:
        writer.close()
        renderer.close()
    final = trace[-1]
    return {
        "command_forward_m_s": command_m_s,
        "initial_roll_deg": initial_roll_deg,
        "actions": len(trace),
        "status": "base_below_0.13_m" if len(trace) < steps else "completed_13_s",
        "final_base_xyz_m": final["base_xyz_m"],
        "trace": trace,
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    base = UPSTREAM / "toddlerbot/descriptions"
    checkpoint = UPSTREAM / "ckpts/toddlerbot_2xc_walk_rsl_20251226_114612/model_best.onnx"
    result = {
        "upstream_commit": "e337f3b177b4b53abff70b31d1695a7b66cc6d2e",
        "sources_sha256": {
            "scene.xml": sha256(base / "toddlerbot_2xc/scene.xml"),
            "robot.xml": sha256(base / "toddlerbot_2xc/toddlerbot_2xc.xml"),
            "default.yml": sha256(base / "default.yml"),
            "model_best.onnx": sha256(checkpoint),
        },
        "note": "Released policy and motor controller unchanged; +/-2 degree starting roll is an evaluation perturbation, not physics randomization.",
        "runs": [run(0.0), run(0.05, render=True), run(0.10, render=True)] + [run(0.05, initial_roll_deg=roll) for roll in (-2.0, 2.0)],
    }
    path = OUT / "trace.json"
    path.write_text(json.dumps(result, separators=(",", ":")))
    for item in result["runs"]:
        print(item["command_forward_m_s"], item["initial_roll_deg"], item["status"], item["actions"], item["final_base_xyz_m"])
    print(path)
