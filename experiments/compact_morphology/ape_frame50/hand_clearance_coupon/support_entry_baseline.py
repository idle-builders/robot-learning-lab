"""Compare raw-motor support of a saved squat reset and floor-entry IK posture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2"
SELECTED_BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CANDIDATE_BODY = BASE / "candidate-training.xml"
ENTRY_STATES = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1/raised-states.npz"
OUT = BASE / "support-entry-baseline-v1"
LEG_POSE = {"hip_yaw": 0.0, "hip_roll": 0.0,
            "hip_pitch": -1.225, "knee": 1.555, "ankle_pitch": -0.287}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squat_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = model.qpos0.copy()
    q[:7] = (0, 0, .1712, 1, 0, 0, 0)
    for side in ("left", "right"):
        for joint, angle in LEG_POSE.items():
            q[model.joint(f"{side}_{joint}").qposadr[0]] = angle
        q[model.joint(f"{side}_shoulder_pitch").qposadr[0]] = -1.0
    return q


def contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[set[str], dict[str, float]]:
    floor = model.geom("floor").id
    sole_ids = {model.geom(f"{side}_sole").id: side for side in ("left", "right")}
    grounded = set()
    hits = {}
    for contact in data.contact[:data.ncon]:
        if floor not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        if other in sole_ids:
            if contact.dist <= .001:
                grounded.add(sole_ids[other])
        elif contact.dist < -.0001:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            hits[name] = max(hits.get(name, 0), -1000 * float(contact.dist))
    return grounded, hits


def main() -> None:
    with np.load(ENTRY_STATES) as saved:
        indices = np.where((saved["start"] == 27) & (saved["raise_mm"] == 6))[0]
        if len(indices) != 1:
            raise ValueError("Expected the predeclared entry pose")
        entry = saved["qpos"][int(indices[0])].copy()
    rows = []
    trace = []
    for body_kind, body_path in (("selected", SELECTED_BODY),
                                 ("candidate", CANDIDATE_BODY)):
        model = mujoco.MjModel.from_xml_path(str(body_path))
        if (model.nq, model.nu, model.opt.timestep) != (28, 21, .002):
            raise ValueError("Unexpected body contract")
        target_adrs = [model.jnt_qposadr[model.actuator_trnid[i, 0]]
                       for i in range(model.nu)]
        leg_ids = [i for i in range(model.nu) if any(
            part in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for part in ("hip", "knee", "ankle"))]
        for pose_kind, qpos in (("settled_squat_reset", squat_qpos(model)),
                                ("whole_body_floor_entry", entry)):
            data = mujoco.MjData(model)
            data.qpos[:] = qpos
            data.ctrl[:] = qpos[target_adrs]
            mujoco.mj_forward(model, data)
            initial_z = float(data.qpos[2])
            first_nonsole = None
            first_two_sole_loss = None
            max_root = 0.0
            max_xy = 0.0
            both_sole_frames = 0
            nonsole_frames = 0
            leg_limit_frames = 0
            peak_leg_duty = 0.0
            for step in range(1001):
                t = step * .002
                data.ctrl[:] = qpos[target_adrs]
                mujoco.mj_forward(model, data)
                grounded, hits = contacts(model, data)
                if grounded == {"left", "right"}:
                    both_sole_frames += 1
                elif first_two_sole_loss is None:
                    first_two_sole_loss = t
                if hits:
                    nonsole_frames += 1
                    if first_nonsole is None:
                        first_nonsole = t
                drift = float(np.linalg.norm(data.qpos[:2] - qpos[:2])) * 1000
                root_delta = abs((float(data.qpos[2]) - initial_z) * 1000)
                max_root = max(max_root, root_delta)
                max_xy = max(max_xy, drift)
                leg_duty = max(abs(float(data.actuator_force[i])) /
                               max(abs(float(model.actuator_forcerange[i, 0])),
                                   abs(float(model.actuator_forcerange[i, 1])))
                               for i in leg_ids)
                peak_leg_duty = max(peak_leg_duty, leg_duty)
                if leg_duty >= .99:
                    leg_limit_frames += 1
                if step % 10 == 0:
                    trace.append({"body": body_kind, "pose": pose_kind,
                                  "time_s": round(t, 3),
                                  "root_z_m": round(float(data.qpos[2]), 5),
                                  "root_xy_m": [round(float(v), 5) for v in data.qpos[:2]],
                                  "grounded_soles": sorted(grounded),
                                  "nonsole_floor": {k: round(v, 3) for k, v in hits.items()},
                                  "max_leg_force_fraction": round(leg_duty, 4)})
                if step < 1000:
                    mujoco.mj_step(model, data)
            rows.append({
                "case": f"{body_kind} {pose_kind}", "body": body_kind,
                "pose": pose_kind, "initial_root_z_m": round(initial_z, 5),
                "final_root_z_m": round(float(data.qpos[2]), 5),
                "max_abs_root_z_delta_mm": round(max_root, 3),
                "max_root_xy_drift_mm": round(max_xy, 3),
                "both_sole_fraction": round(both_sole_frames / 1001, 4),
                "first_two_sole_loss_s": first_two_sole_loss,
                "first_nonsole_floor_s": first_nonsole,
                "nonsole_floor_fraction": round(nonsole_frames / 1001, 4),
                "leg_force_limit_fraction": round(leg_limit_frames / 1001, 4),
                "peak_leg_force_fraction": round(peak_leg_duty, 4),
            })
    report = {
        "schema": "robot-learning-lab.cove50-support-entry-baseline.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "selected_body_sha256": sha(SELECTED_BODY),
        "candidate_body_sha256": sha(CANDIDATE_BODY),
        "entry_states_sha256": sha(ENTRY_STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "mj_version": mujoco.__version__,
        "cases": rows,
        "limits": [
            "The squat reset is reconstructed from a saved policy reset and leg targets; its full motor-attained state and learned actor are not replayed.",
            "The raw MuJoCo servo XML omits the native MJLab battery, speed and lag model; this is a screening comparison, not 10 V policy qualification.",
            "Static entry IK is not a motor-attained predecessor state. No toy, reach, grasp or lift is tested here.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary-matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "trace.json").write_text(json.dumps({"cases": trace}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
