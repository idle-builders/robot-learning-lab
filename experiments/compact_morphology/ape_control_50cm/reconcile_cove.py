"""Reconcile the selected Cove frame with its provisional control body.

Run locally with .venv/bin/python. This is a provenance and reset smoke run,
not a learned-control or hardware qualification test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
POINTER = ROOT / "robot/ape-current-design.json"
LEDGER = ROOT / "artifacts/ape-cove50-transfer-20261008/body-ledger.json"
OUT = ROOT / "artifacts/ape-control-baseline-50cm-20261009/manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: str, expected: str) -> dict:
    source = ROOT / path
    actual = sha(source)
    if actual != expected:
        raise ValueError(f"Source changed: {path}: {actual} != {expected}")
    return {"path": path, "sha256": actual}


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index)


def main() -> None:
    pointer = json.loads(POINTER.read_text())
    ledger = json.loads(LEDGER.read_text())
    if pointer["id"] != "cove-50-direct-frame" or ledger["selected_design_sha256"] != sha(POINTER):
        raise ValueError("Selected design and body ledger disagree")
    sources = {
        "selection": {"path": str(POINTER.relative_to(ROOT)), "sha256": sha(POINTER)},
        "generator": checked(pointer["generator"]["path"], pointer["generator"]["sha256"]),
        "source_report": checked(pointer["evidence"]["report"], pointer["evidence"]["report_sha256"]),
        "source_visual_body": checked(pointer["evidence"]["visual_body"], pointer["evidence"]["visual_body_sha256"]),
        "provisional_body": checked(ledger["generated_body"], ledger["generated_body_sha256"]),
        "control_body": checked(ledger["training_body"], ledger["training_body_sha256"]),
        "mass_ledger": {"path": str(LEDGER.relative_to(ROOT)), "sha256": sha(LEDGER)},
    }
    model = mujoco.MjModel.from_xml_path(str(ROOT / ledger["training_body"]))
    joints = []
    for index in range(1, model.njnt):  # index 0 is the free root.
        joints.append({"name": name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
                       "range_rad": model.jnt_range[index].tolist(),
                       "limited": bool(model.jnt_limited[index])})
    actions = [name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
               for i in range(model.nu)]
    if len(joints) != 21 or actions != [joint["name"] for joint in joints] or "head_pitch" in actions:
        raise ValueError("Cove actor or actuator order differs from the selected 21-axis contract")
    bodies = []
    for index in range(1, model.nbody):
        bodies.append({
            "name": name(model, mujoco.mjtObj.mjOBJ_BODY, index),
            "mass_kg": float(model.body_mass[index]),
            "com_body_m": model.body_ipos[index].tolist(),
            "inertia_principal_kg_m2": model.body_inertia[index].tolist(),
            "inertia_frame_quat_wxyz": model.body_iquat[index].tolist(),
        })
    mass = sum(body["mass_kg"] for body in bodies)
    if abs(mass - ledger["compiled_mass_kg"]) > 1e-6:
        raise ValueError(f"Compiled mass differs from ledger: {mass}")
    catalog = yaml.safe_load((ROOT / "robot/actuator-catalog.yaml").read_text())["parts"]
    motor_parts = {}
    for joint in actions:
        if joint == "head_yaw" or joint.endswith("_gripper"):
            part_id = "robotis-xl330-m288-t"
        elif "_hip_" in joint or joint.endswith("_knee"):
            part_id = "feetech-sts3250-c001"
        else:
            part_id = "feetech-st3215-c018"
        part = catalog[part_id]
        motor_parts[joint] = {
            "catalog_part": part_id,
            "rated_effort_nm": part.get("rated_effort_nm"),
            "stall_effort_nm": part.get("stall_effort_nm", part.get("stall_effort_nm_at_5v")),
            "no_load_speed_rad_s": part.get("no_load_speed_rad_s", part.get("no_load_speed_rad_s_at_5v")),
            "catalog_voltage_v": part.get("rating_voltage_v", 5.0),
            "continuous_effort_identified": False,
        }
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.tolist()
    reset_contacts = int(data.ncon)
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for index in range(model.ngeom):
        if model.geom_bodyid[index] == 0:
            continue
        rotation = data.geom_xmat[index].reshape(3, 3)
        size = model.geom_size[index]
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX:
            radius = np.abs(rotation) @ size
        elif model.geom_type[index] == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = np.full(3, size[0])
        elif model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = size[0] + np.abs(rotation[:, 2]) * size[1]
        else:
            raise ValueError("Unexpected geom type in neutral model envelope")
        lower = np.minimum(lower, data.geom_xpos[index] - radius)
        upper = np.maximum(upper, data.geom_xpos[index] + radius)
    root_heights, contact_counts, peak_efforts = [], [], []
    for _ in range(round(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        root_heights.append(float(data.qpos[2]))
        contact_counts.append(int(data.ncon))
        peak_efforts.append(float(np.max(np.abs(data.actuator_force))))
    geoms = {}
    for geom_name in ("floor", "left_sole", "right_sole", "left_hand", "right_hand"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if index >= 0:
            geoms[geom_name] = {"type": int(model.geom_type[index]),
                                "half_size_m": model.geom_size[index].tolist(),
                                "friction": model.geom_friction[index].tolist(),
                                "contype": int(model.geom_contype[index]),
                                "conaffinity": int(model.geom_conaffinity[index])}
    paired_paths = {
        "old_report": "artifacts/ape-cove50-transfer-20261008/baseline-old-seed144.json",
        "cove_report": "artifacts/ape-cove50-transfer-20261008/baseline-direct-seed144.json",
        "cove_video": "artifacts/ape-cove50-transfer-20261008/baseline-direct-seed144.mp4",
    }
    old = json.loads((ROOT / paired_paths["old_report"]).read_text())
    cove = json.loads((ROOT / paired_paths["cove_report"]).read_text())
    if (old["initial_state_sha256"] != cove["initial_state_sha256"]
            or old["first_action_sha256"] != cove["first_action_sha256"]
            or cove["body_sha256"] != ledger["training_body_sha256"]):
        raise ValueError("Paired replay does not match the selected body and initial action")
    def paired_stats(run: dict) -> dict:
        worlds = run["worlds"]
        return {"worlds": len(worlds),
                "survived_8s": sum(world["duration_s"] >= 8.0 for world in worlds),
                "mean_forward_m": statistics.mean(world["forward_m"] for world in worlds),
                "mean_lateral_m": statistics.mean(world["lateral_m"] for world in worlds),
                "mean_yaw_rad": statistics.mean(world["net_yaw_rad"] for world in worlds),
                "mean_leg_saturation_frame_fraction": statistics.mean(
                    world["fraction_frames_with_saturated_leg"] for world in worlds),
                "mean_sole_slip_m_left_right": [statistics.mean(
                    world["contact_slip_m_left_right"][side] for world in worlds)
                    for side in (0, 1)],
                "max_hand_force_n": max(world["hand_force_n"] for world in worlds),
                "max_body_contact_frame_fraction": max(
                    world["fraction_frames_with_body_contact"] for world in worlds)}
    report = {
        "schema": "robot-learning-lab.cove50-control-baseline.v1",
        "status": "provisional_compiled_reset_and_paired_replay_available",
        "sources": sources,
        "body": {"mass_kg": mass, "source_body_mass_kg": ledger["source_body_mass_kg"],
                 "mass_model": ledger["mass_model"], "contact_model": ledger["contact_model"],
                 "root_qpos0_m": initial_qpos[:3], "root_quat0_wxyz": initial_qpos[3:7],
                 "neutral_geom_aabb_m": {"minimum": lower.tolist(), "maximum": upper.tolist(),
                                         "extent": (upper - lower).tolist()},
                 "body_count": len(bodies), "bodies": bodies,
                 "geometry_count": model.ngeom, "contact_pair_count": model.npair,
                 "key_geoms": geoms,
                 "sensor_names": [name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)],
                 "camera_names": [name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]},
        "control": {"joints": joints, "actuator_order": actions,
                    "physics_timestep_s": float(model.opt.timestep),
                    "catalog_motor_assumptions": motor_parts,
                    "native_motor_model": "MJLab DcMotorActuator linear sensitivity envelope; catalog rated, stall and no-load points, unmeasured impedance and duty",
                    "native_velocity_actor": {
                        "observation_layout": "base angular velocity (3), projected gravity (3), named joint positions (21), named joint velocities (21), previous named actions (21), velocity command (3)",
                        "observation_width": 72, "action_width": 21,
                        "action_kind": "normalized joint-position offset from reset pose; per-joint scale/clip and motor targets are task-specific",
                        "step_s": 0.02, "episode_s": 8.0,
                        "reset": "MJLab terrain reset and task-specific root/joint pose; training may add bounded root/joint noise. Direct XML qpos0 root height is 0.231 m, which does not itself define the MJLab reset.",
                    },
                    "old_22_to_21_adapter": "experiments/compact_morphology/cove50_policy_transfer/map_ppo_22_to_21.py; drops only head_pitch by joint/action name and clears optimizer state"},
        "direct_xml_reset_smoke": {"duration_s": 0.5, "zero_ctrl": True,
                                   "initial_contacts": reset_contacts,
                                   "final_root_xyz_m": data.qpos[:3].tolist(),
                                   "min_root_height_m": min(root_heights),
                                   "max_contacts": max(contact_counts),
                                   "max_actuator_force_nm": max(peak_efforts),
                                   "finite": bool(np.all(np.isfinite(data.qpos)))},
        "paired_native_replay": {
            "artifacts": {key: {"path": path, "sha256": sha(ROOT / path)}
                          for key, path in paired_paths.items()},
            "checkpoint": checked(cove["checkpoint"], sha(ROOT / cove["checkpoint"])),
            "same_initial_state_sha256": cove["initial_state_sha256"],
            "same_first_action_sha256": cove["first_action_sha256"],
            "old": paired_stats(old), "cove": paired_stats(cove),
            "interpretation": "Retained checkpoint runs but path control degrades on Cove; this is incompatibility evidence, not qualification."},
        "limitations": ["Mass centers derive from CAD solids; inertia uses each part bounding box.",
                        "Direct case contacts and soles are modeled, while other frame contacts may be proxies or absent.",
                        "Catalog ratings and MJLab motor law do not identify measured thermal, current, impedance or continuous behavior.",
                        "Source pointer physics.match_status records the selection-time state, before this later transfer."],
    }
    if not report["direct_xml_reset_smoke"]["finite"]:
        raise ValueError("Reset smoke produced non-finite state")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {len(joints)} axes, {mass:.6f} kg, reset min z={min(root_heights):.4f} m")


if __name__ == "__main__":
    main()
