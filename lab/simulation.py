import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from .io import SimRobot, JointCommand


def tracking(robot, output, library, duration=3.0):
    output = Path(output)
    c = robot.config
    sim = SimRobot(robot, output / "supported.xml", library)
    stride = c.control.physics_hz // c.control.command_hz
    names = [j.name for j in robot.joints]
    idx = names.index("neck_yaw_joint")
    target = np.zeros(len(robot.joints))
    trace = []
    for seq in range(round(duration * c.control.command_hz)):
        target[idx] = 0.20 * np.sin(2 * np.pi * 0.5 * sim.data.time)
        if not sim.command(JointCommand(seq, tuple(target))):
            raise RuntimeError("controller command rejected")
        state = sim.advance(stride)
        current = c.power.electronics_current_a + sum(
            c.actuators[j.family].idle_current_a
            + abs(u) / c.actuators[j.family].effective_nm_per_a
            for j, u in zip(robot.joints, state.effort_nm)
        )
        if not np.all(np.isfinite(state.position_rad)):
            raise RuntimeError("non-finite simulation state")
        trace.append(
            {
                "time_s": state.time_s,
                "position_rad": list(state.position_rad),
                "velocity_rad_s": list(state.velocity_rad_s),
                "effort_nm": list(state.effort_nm),
                "target_rad": target.tolist(),
                "supervisor_status": state.supervisor_status,
                "estimated_bus_current_a": float(current),
            }
        )
    error = float(
        np.sqrt(
            np.mean(
                [(t["position_rad"][idx] - t["target_rad"][idx]) ** 2 for t in trace]
            )
        )
    )
    (output / "trajectory.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in trace)
    )
    result = {
        "status": "pass"
        if error < 0.08 and all(t["supervisor_status"] == 1 for t in trace)
        else "fail",
        "scenario": "supported neck tracking",
        "duration_s": duration,
        "tracking_rmse_rad": error,
        "threshold_rad": 0.08,
        "peak_estimated_current_a": max(t["estimated_bus_current_a"] for t in trace),
        "base": "fixed test fixture",
        "locomotion": "not_run",
        "manipulation": "not_run",
        "actuator_model": "generic PD torque saturation; not a measured commercial servo",
    }
    return result, trace


def render(output):
    output = Path(output)
    model = mujoco.MjModel.from_xml_path(str(output / "robot.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    height = max(data.geom_xpos[:, 2]) + 0.05
    camera.lookat[:] = [0, 0, height * 0.48]
    camera.distance = height * 2.05
    camera.azimuth = 135
    camera.elevation = -14
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / "robot.png")
        camera.azimuth = 180
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / "front.png")
        camera.azimuth = 0
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / "rear.png")
        renderer.update_scene(data, camera="head_camera")
        Image.fromarray(renderer.render()).save(output / "camera.png")
    return {
        "status": "pass",
        "views": ["robot.png", "front.png", "rear.png", "camera.png"],
    }
