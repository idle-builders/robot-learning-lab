"""Native MuJoCo motor closure from matched, static Cove hand contact poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.screen_static_contact import mirror_qpos


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml"
STATES = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1/raised-states.npz"
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/motor-close-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_info(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> dict:
    cube = model.geom("diagnostic_cube_geom").id
    floor = model.geom("floor").id
    palm = model.geom(f"{side}_hand").id
    finger = model.geom(f"{side}_finger_tip").id
    sole_ids = {model.geom(f"{s}_sole").id for s in ("left", "right")}
    info = {"palm": False, "finger": False, "cube_floor": False,
            "sole_floor": 0, "sole_floor_sides": [],
            "other_cube": {}, "nonsole_floor": {}}
    for c in data.contact[:data.ncon]:
        a, b = c.geom1, c.geom2
        if cube in (a, b):
            other = b if a == cube else a
            if other == palm:
                info["palm"] = True
            elif other == finger:
                info["finger"] = True
            elif other == floor:
                info["cube_floor"] = True
            elif c.dist < -.0001:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                info["other_cube"][name] = round(max(
                    info["other_cube"].get(name, 0), -1000 * float(c.dist)), 3)
        elif floor in (a, b):
            other = b if a == floor else a
            if other in sole_ids:
                info["sole_floor"] += 1
                side_name = ("left" if other == model.geom("left_sole").id else "right")
                if side_name not in info["sole_floor_sides"]:
                    info["sole_floor_sides"].append(side_name)
            elif c.dist < -.0001:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                info["nonsole_floor"][name] = round(max(
                    info["nonsole_floor"].get(name, 0), -1000 * float(c.dist)), 3)
    return info


def main() -> None:
    root = ET.parse(BODY).getroot()
    cube = ET.SubElement(root.find("worldbody"), "body", name="diagnostic_cube", pos="0 0 .01")
    ET.SubElement(cube, "freejoint", name="diagnostic_cube_free")
    ET.SubElement(cube, "geom", name="diagnostic_cube_geom", type="box",
                  size=".01 .01 .01", mass=".00625", friction=".8 .01 .001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.opt.timestep != .002:
        raise ValueError("Unexpected motor interface or timestep")
    # The training body hides collision budgets. Reveal them for diagnostic video only.
    model.geom_rgba[:, 3] = 1.0
    with np.load(STATES) as saved:
        states = saved["qpos"].copy()
        starts = saved["start"].copy()
        heights = saved["raise_mm"].copy()
    indices = np.where((starts == 27) & (heights == 6))[0]
    if len(indices) != 1:
        raise ValueError("Missing predeclared start 27 raised 6 mm")
    q_right = states[int(indices[0])]
    cube_adr = model.joint("diagnostic_cube_free").qposadr[0]
    target_adrs = [model.jnt_qposadr[model.actuator_trnid[i, 0]]
                   for i in range(model.nu)]
    leg_actuator_ids = [i for i in range(model.nu) if any(
        part in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for part in ("hip", "knee", "ankle"))]
    rows = []
    summaries = []
    OUT.mkdir(parents=True, exist_ok=True)
    clip_paths = {}
    for side, (dx, dy) in (("right", (-2, -2)), ("left", (-2, 2))):
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0, 0, .21)
        camera.distance = .95
        camera.azimuth = 115 if side == "right" else 65
        camera.elevation = -18
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[:] = 1
        scene_option.sitegroup[:] = 0
        clip_path = OUT / f"{side}-motor-close.mp4"
        video = imageio.get_writer(str(clip_path), fps=50, codec="libx264", quality=7)
        pose = q_right if side == "right" else mirror_qpos(model, q_right)
        data.qpos[:28] = pose
        data.qpos[model.joint(f"{side}_gripper").qposadr[0]] = 0
        mujoco.mj_forward(model, data)
        site = data.site_xpos[model.site(f"{side}_hand_center").id].copy()
        data.qpos[cube_adr:cube_adr + 3] = (site[0] + dx / 1000, site[1] + dy / 1000, .01)
        angle = np.deg2rad(45)
        data.qpos[cube_adr + 3:cube_adr + 7] = (
            np.cos(angle / 2), 0, 0, np.sin(angle / 2))
        mujoco.mj_forward(model, data)
        initial_root_z = float(data.qpos[2])
        initial_cube_xyz = data.qpos[cube_adr:cube_adr + 3].copy()
        targets = pose[target_adrs].copy()
        active = next(i for i in range(model.nu)
                      if model.actuator_trnid[i, 0] == model.joint(f"{side}_gripper").id)
        for step in range(301):
            time_s = step * .002
            targets[active] = min(.5, .5 * time_s / .3)
            data.ctrl[:] = targets
            mujoco.mj_forward(model, data)
            info = contact_info(model, data, side)
            jaw = float(data.qpos[model.joint(f"{side}_gripper").qposadr[0]])
            cube_xyz = data.qpos[cube_adr:cube_adr + 3].copy()
            force_limit = float(model.actuator_forcerange[active, 1])
            leg_duty = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                        abs(float(data.actuator_force[i])) /
                        max(abs(float(model.actuator_forcerange[i, 0])),
                            abs(float(model.actuator_forcerange[i, 1])))
                        for i in leg_actuator_ids}
            rows.append({
                "side": side, "step": step, "time_s": round(time_s, 3),
                "target_jaw_rad": round(float(targets[active]), 4),
                "actual_jaw_rad": round(jaw, 4),
                "active_servo_force_nm": round(float(data.actuator_force[active]), 5),
                "active_servo_at_force_limit": bool(abs(data.actuator_force[active]) >= force_limit - 1e-5),
                "max_leg_force_fraction": round(max(leg_duty.values()), 4),
                "limiting_leg_actuators": [name for name, duty in leg_duty.items()
                                            if duty >= .99],
                "root_z_delta_mm": round((float(data.qpos[2]) - initial_root_z) * 1000, 3),
                "cube_xyz_m": [round(float(v), 6) for v in cube_xyz],
                "cube_xy_drift_mm": round(float(np.linalg.norm((cube_xyz - initial_cube_xyz)[:2])) * 1000, 3),
                "cube_z_delta_mm": round(float(cube_xyz[2] - initial_cube_xyz[2]) * 1000, 3),
                **info,
            })
            if step % 10 == 0:
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                video.append_data(renderer.render())
            if step < 300:
                mujoco.mj_step(model, data)
        video.close()
        renderer.close()
        clip_paths[side] = str(clip_path.relative_to(ROOT))
        run = [r for r in rows if r["side"] == side]
        summaries.append({
            "side": side, "frames": len(run),
            "final_actual_jaw_rad": run[-1]["actual_jaw_rad"],
            "final_target_jaw_rad": run[-1]["target_jaw_rad"],
            "both_jaw_contact_frames": sum(r["palm"] and r["finger"] for r in run),
            "both_jaw_contact_final_100ms_frames": sum(
                r["palm"] and r["finger"] for r in run if r["time_s"] >= .5),
            "active_servo_limit_frames": sum(r["active_servo_at_force_limit"] for r in run),
            "leg_actuator_limit_frames": sum(bool(r["limiting_leg_actuators"]) for r in run),
            "worst_leg_force_fraction": max(r["max_leg_force_fraction"] for r in run),
            "most_frequent_limited_leg_actuators": sorted(
                ((name, sum(name in r["limiting_leg_actuators"] for r in run))
                 for name in {name for r in run for name in r["limiting_leg_actuators"]}),
                key=lambda item: -item[1])[:5],
            "max_cube_xy_drift_mm": max(r["cube_xy_drift_mm"] for r in run),
            "max_abs_root_z_delta_mm": max(abs(r["root_z_delta_mm"]) for r in run),
            "final_cube_xy_drift_mm": run[-1]["cube_xy_drift_mm"],
            "final_cube_z_delta_mm": run[-1]["cube_z_delta_mm"],
            "final_sole_floor_contacts": run[-1]["sole_floor"],
            "nonsole_floor_penetration_frames": sum(bool(r["nonsole_floor"]) for r in run),
        })
    report = {
        "schema": "robot-learning-lab.cove50-narrow-bridge-motor-close.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(BODY),
        "raised_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "mj_version": mujoco.__version__,
        "summaries": summaries,
        "clips": clip_paths,
        "clip_sha256": {side: sha(ROOT / path) for side, path in clip_paths.items()},
        "limits": [
            "The initial body and cube poses came from static IK/contact screening, not a learned or motor-attained approach.",
            "This 0.6 s actuator diagnostic commands jaw closure and holds other joints; it does not command lift or release.",
            "Cube contact, if any, does not by itself show compressive force, retained possession or policy recovery.",
        ],
    }
    (OUT / "trace.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "summary-matrix.json").write_text(json.dumps({"cases": summaries}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
