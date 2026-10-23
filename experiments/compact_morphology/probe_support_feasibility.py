"""One-shot nominal MuJoCo probe: can symmetric hip roll move the ape COM?"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    home = model.keyframe("stand_bent_knees")
    results = []
    for offset in (-0.20, -0.18, -0.16, -0.14, 0.0, 0.14, 0.16, 0.18, 0.20):
        data = mujoco.MjData(model)
        data.qpos[:] = home.qpos
        data.ctrl[:] = home.ctrl
        for side in ("left", "right"):
            data.ctrl[model.actuator(f"{side}_hip_roll_servo").id] += offset
        for _ in range(round(2.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        loads = [max(0.0, data.sensordata[model.sensor_adr[
            model.sensor(f"{side}_sole_normal_force").id
        ]].item()) for side in ("left", "right")]
        contacts = [data.sensordata[model.sensor_adr[
            model.sensor(f"{side}_sole_floor_found").id
        ]].item() for side in ("left", "right")]
        feet_y = [data.site_xpos[model.site(f"{side}_sole_center").id, 1].item()
                  for side in ("left", "right")]
        feet_z = [data.site_xpos[model.site(f"{side}_sole_center").id, 2].item()
                  for side in ("left", "right")]
        q = data.qpos[3:7]
        results.append({
            "hip_roll_target_offset_rad": offset,
            "com_y_m": data.subtree_com[model.body("pelvis").id, 1].item(),
            "root_y_m": data.qpos[1].item(),
            "root_z_m": data.qpos[2].item(),
            "upright_z": float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)),
            "sole_site_y_m": feet_y,
            "sole_site_z_m": feet_z,
            "sole_contact_counts": contacts,
            "sole_normal_force_n": loads,
            "left_load_share": loads[0] / max(sum(loads), 1e-9),
            "max_abs_actuator_force_nm": float(max(abs(data.actuator_force))),
        })
    payload = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "method": "Home pose and controls; add the same hip-roll target offset to both sides; step MuJoCo 2 s; sample the final state. This is an authored feasibility probe, not a learned policy or robustness test.",
        "results": results,
    }
    content = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
