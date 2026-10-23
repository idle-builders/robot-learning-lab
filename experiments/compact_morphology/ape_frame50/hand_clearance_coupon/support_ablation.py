"""Separate entry-posture support failure from cube contact and jaw motion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.motor_close import contact_info
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.screen_static_contact import mirror_qpos


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2"
BODY = BASE / "candidate-training.xml"
STATES = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1/raised-states.npz"
OUT = BASE / "support-ablation-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = ET.parse(BODY).getroot()
    cube = ET.SubElement(root.find("worldbody"), "body", name="diagnostic_cube", pos="0 0 .01")
    ET.SubElement(cube, "freejoint", name="diagnostic_cube_free")
    ET.SubElement(cube, "geom", name="diagnostic_cube_geom", type="box",
                  size=".01 .01 .01", mass=".00625", friction=".8 .01 .001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    with np.load(STATES) as saved:
        indices = np.where((saved["start"] == 27) & (saved["raise_mm"] == 6))[0]
        if len(indices) != 1:
            raise ValueError("Missing declared state")
        q_right = saved["qpos"][int(indices[0])].copy()
    target_adrs = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    cube_adr = model.joint("diagnostic_cube_free").qposadr[0]
    leg_ids = [i for i in range(model.nu) if any(
        part in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for part in ("hip", "knee", "ankle"))]
    rows = []
    for side, offset in (("right", (-2, -2)), ("left", (-2, 2))):
        for condition in ("no_cube_open", "floor_cube_open", "floor_cube_close"):
            data = mujoco.MjData(model)
            pose = q_right if side == "right" else mirror_qpos(model, q_right)
            data.qpos[:28] = pose
            jaw_adr = model.joint(f"{side}_gripper").qposadr[0]
            data.qpos[jaw_adr] = 0
            mujoco.mj_forward(model, data)
            site = data.site_xpos[model.site(f"{side}_hand_center").id].copy()
            if condition == "no_cube_open":
                data.qpos[cube_adr:cube_adr + 3] = (3, 3, .01)
            else:
                data.qpos[cube_adr:cube_adr + 3] = (
                    site[0] + offset[0]/1000, site[1] + offset[1]/1000, .01)
            angle = np.deg2rad(45)
            data.qpos[cube_adr + 3:cube_adr + 7] = (
                np.cos(angle/2), 0, 0, np.sin(angle/2))
            mujoco.mj_forward(model, data)
            initial_root_z = float(data.qpos[2])
            initial_cube = data.qpos[cube_adr:cube_adr + 3].copy()
            targets = pose[target_adrs].copy()
            active = next(i for i in range(model.nu)
                          if model.actuator_trnid[i, 0] == model.joint(f"{side}_gripper").id)
            first_floor = None
            first_support_loss = None
            max_root = 0.0
            max_cube = 0.0
            leg_limit_frames = 0
            two_jaw_last100 = 0
            for step in range(301):
                t = step * .002
                targets[active] = min(.5, .5*t/.3) if condition == "floor_cube_close" else 0.0
                data.ctrl[:] = targets
                mujoco.mj_forward(model, data)
                info = contact_info(model, data, side)
                if first_floor is None and info["nonsole_floor"]:
                    first_floor = t
                if first_support_loss is None and set(info["sole_floor_sides"]) != {"left", "right"}:
                    first_support_loss = t
                if t >= .5 and info["palm"] and info["finger"]:
                    two_jaw_last100 += 1
                max_root = max(max_root, abs((float(data.qpos[2]) - initial_root_z)*1000))
                max_cube = max(max_cube, float(np.linalg.norm(
                    (data.qpos[cube_adr:cube_adr + 3] - initial_cube)[:2]))*1000)
                if any(abs(float(data.actuator_force[i])) >= .99*max(
                       abs(float(model.actuator_forcerange[i, 0])),
                       abs(float(model.actuator_forcerange[i, 1]))) for i in leg_ids):
                    leg_limit_frames += 1
                if step < 300:
                    mujoco.mj_step(model, data)
            rows.append({
                "case": f"{side} {condition}", "side": side,
                "condition": condition,
                "first_nonsole_floor_s": first_floor,
                "first_two_sole_loss_s": first_support_loss,
                "max_abs_root_z_delta_mm": round(max_root, 3),
                "max_cube_xy_drift_mm": round(max_cube, 3),
                "final_jaw_rad": round(float(data.qpos[jaw_adr]), 4),
                "two_jaw_final_100ms_frames": two_jaw_last100,
                "leg_limit_frames": leg_limit_frames,
            })
    report = {
        "schema": "robot-learning-lab.cove50-entry-support-ablation.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(BODY),
        "raised_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "mj_version": mujoco.__version__,
        "cases": rows,
        "limits": [
            "This is one matched start and its left mirror, not a population estimate.",
            "The cube-away condition tests body support only; all starts are motor-unattained static IK.",
            "The source hand body, gravity, motor limits and free root are retained; no trained controller or lift is tested.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary-matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
