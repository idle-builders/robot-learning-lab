"""Arm-only static raised hand entries on the unselected Cove hand coupon."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v1/candidate-training.xml"
STATES = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1/candidate-states.npz"
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
       "right_elbow_pitch")
RAISE_MM = (6, 10, 12, 15)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def robot_penetration(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    floor = model.geom("floor").id
    soles = {model.geom(f"{side}_sole").id for side in ("left", "right")}
    hits = {}
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist >= -0.0001:
            continue
        a, b = contact.geom1, contact.geom2
        if floor in (a, b) and (b if a == floor else a) in soles:
            continue
        names = sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                       for geom in (a, b))
        key = "/".join(names)
        hits[key] = max(hits.get(key, 0.0), -float(contact.dist * 1000))
    return {key: round(value, 3) for key, value in hits.items()}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Expected unselected 21-axis candidate body")
    with np.load(STATES) as saved:
        poses = saved["qpos"][saved["screen_pass"]].copy()
        starts = saved["start"][saved["screen_pass"]].copy()
    if len(poses) != 25:
        raise ValueError("Expected fixed 25 sole/jaw-passing source poses")
    data = mujoco.MjData(model)
    site = model.site("right_hand_center").id
    jaw_adr = model.joint("right_gripper").qposadr[0]
    arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
    lower = np.array([model.joint(name).range[0] for name in ARM])
    upper = np.array([model.joint(name).range[1] for name in ARM])
    rows = []
    fitted = []
    for start, source in zip(starts, poses):
        data.qpos[:] = source
        data.qpos[jaw_adr] = 0
        mujoco.mj_forward(model, data)
        entry_site = data.site_xpos[site].copy()
        entry_arm = source[arm_adr].copy()
        guess = np.clip(entry_arm, lower + 1e-7, upper - 1e-7)
        for raise_mm in RAISE_MM:
            target = entry_site + np.array([0, 0, raise_mm / 1000])

            def geometry(x: np.ndarray):
                data.qpos[:] = source
                data.qpos[arm_adr] = x
                data.qpos[jaw_adr] = 0
                mujoco.mj_forward(model, data)
                return data.site_xpos[site].copy()

            def residual(x: np.ndarray) -> np.ndarray:
                site_xyz = geometry(x)
                return np.r_[(site_xyz - target) / 0.001,
                             (x - entry_arm) / 0.4]

            fit = least_squares(residual, guess, bounds=(lower, upper),
                                max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            guess = fit.x.copy()
            xyz = geometry(fit.x)
            delta_mm = (xyz - target) * 1000
            xy_error = float(np.linalg.norm(delta_mm[:2]))
            z_error = abs(float(delta_mm[2]))
            hits = robot_penetration(model, data)
            passed = xy_error <= 2 and z_error <= 2 and not hits
            row = {
                "ik_start": int(start), "raise_target_mm": raise_mm,
                "source_hand_site_z_mm": round(entry_site[2] * 1000, 3),
                "fitted_hand_site_z_mm": round(xyz[2] * 1000, 3),
                "xy_error_mm": round(xy_error, 3),
                "z_error_mm": round(z_error, 3),
                "arm_delta_rad": {name: round(float(value), 6)
                                  for name, value in zip(ARM, fit.x - entry_arm)},
                "arm_joint_rad": {name: round(float(value), 6)
                                  for name, value in zip(ARM, fit.x)},
                "open_jaw_nonsole_floor_or_self_penetration_mm": hits,
                "ik_screen_pass": passed,
                "solver_success": bool(fit.success),
                "solver_nfev": int(fit.nfev),
            }
            rows.append(row)
            fitted.append(data.qpos.copy())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "raised-states.npz", qpos=np.stack(fitted),
                        start=np.array([row["ik_start"] for row in rows]),
                        raise_mm=np.array([row["raise_target_mm"] for row in rows]),
                        screen_pass=np.array([row["ik_screen_pass"] for row in rows]))
    summary = [{"case": f"+{height} mm", "raise_target_mm": height,
                "attempted": sum(row["raise_target_mm"] == height for row in rows),
                "ik_screen_pass": sum(row["raise_target_mm"] == height and
                                      row["ik_screen_pass"] for row in rows),
                "maximum_xy_error_mm": max(row["xy_error_mm"] for row in rows
                                           if row["raise_target_mm"] == height),
                "maximum_z_error_mm": max(row["z_error_mm"] for row in rows
                                          if row["raise_target_mm"] == height)}
               for height in RAISE_MM]
    report = {
        "schema": "robot-learning-lab.cove50-hand-coupon-raised-entry-ik.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(BODY),
        "source_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "raised_states_sha256": sha(OUT / "raised-states.npz"),
        "attempted": len(rows), "passing": sum(row["ik_screen_pass"] for row in rows),
        "summary": summary, "cases": rows,
        "limits": [
            "The source whole-body IK postures and these arm-only revisions are static, not attained by continuous native motor control.",
            "Open-jaw robot contacts and site alignment are screened here; a movable cube and closing jaw are evaluated separately.",
        ],
    }
    (OUT / "ik-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "ik-summary-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-hand-coupon-raised-entry-ik-summary.v1",
        "cases": summary}, indent=2) + "\n")
    print(json.dumps({"attempted": len(rows), "passing": report["passing"],
                      "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
