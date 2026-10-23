"""Check a fixed coupled arm/jaw closure from an attained native pregrasp."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import contacts, with_cube

OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/coupled-closure-screen-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "plan.json"
    plan = json.loads(plan_path.read_text())
    body_path = ROOT / plan["body"]
    source_path = ROOT / plan["native_source"]
    model = with_cube(body_path)
    data = mujoco.MjData(model)
    with np.load(source_path) as trace:
        source = trace["qpos"][round(plan["source_time_s"] / float(trace["step_dt_s"]))].copy()
    arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
    jaw_adr = model.joint("right_gripper").qposadr[0]
    cube = model.geom("audit_cube_geom").id
    palm = model.geom("right_hand").id
    finger = model.geom("right_finger_tip").id
    excluded = {cube, model.geom("floor").id, palm, finger}
    other = [i for i in range(model.ngeom) if i not in excluded and model.geom_bodyid[i] != 0]
    x0 = source[arm_adr].copy()
    goal = np.array(plan["arm_target_rad"])
    rows = []
    for index, fraction in enumerate(np.linspace(0, 1, plan["interpolation_samples"])):
        data.qpos[:] = source
        data.qpos[arm_adr] = x0 + fraction * (goal - x0)
        data.qpos[jaw_adr] = source[jaw_adr] + fraction * (plan["jaw_target_rad"] - source[jaw_adr])
        mujoco.mj_forward(model, data)
        contact = contacts(model, data)
        palm_mm = float(mujoco.mj_geomDistance(model, data, cube, palm, .05, np.zeros(6))) * 1000
        finger_mm = float(mujoco.mj_geomDistance(model, data, cube, finger, .05, np.zeros(6))) * 1000
        other_mm = min(float(mujoco.mj_geomDistance(model, data, cube, i, .05,
                       np.zeros(6))) * 1000 for i in other)
        rows.append({"index": index, "fraction": round(float(fraction), 3),
                     "palm_signed_distance_mm": round(palm_mm, 4),
                     "finger_signed_distance_mm": round(finger_mm, 4),
                     "minimum_nonpad_cube_distance_mm": round(other_mm, 4),
                     "two_sole_contact": contact["two_sole_contact"],
                     "nonsole_floor_or_self_penetration_mm": contact["nonsole_floor_or_self_penetration_mm"],
                     "both_pads": contact["palm_contact"] and contact["finger_contact"]})
    gate = plan["gates"]
    end = rows[-1]
    bad_robot = sum(not row["two_sole_contact"] or bool(row["nonsole_floor_or_self_penetration_mm"])
                    for row in rows)
    minimum_other = min(row["minimum_nonpad_cube_distance_mm"] for row in rows)
    screen_pass = (minimum_other >= gate["minimum_nonpad_cube_clearance_mm"] and
                   gate["minimum_palm_closed_overlap_mm"] <= -end["palm_signed_distance_mm"] <=
                   gate["maximum_palm_closed_overlap_mm"] and
                   gate["minimum_finger_closed_overlap_mm"] <= -end["finger_signed_distance_mm"] <=
                   gate["maximum_finger_closed_overlap_mm"] and
                   bad_robot <= gate["bad_robot_or_two_sole_samples"])
    report = {
        "schema": "robot-learning-lab.cove50-coupled-closure-screen.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(body_path), "source_trace_sha256": sha(source_path),
        "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "mujoco_version": mujoco.__version__,
        "source_time_s": plan["source_time_s"], "source_arm_rad": x0.tolist(),
        "source_cube_xyz_m": source[model.nq - 7:model.nq - 4].tolist(),
        "goal_arm_rad": goal.tolist(), "goal_jaw_rad": plan["jaw_target_rad"],
        "minimum_nonpad_cube_distance_mm": minimum_other,
        "bad_robot_or_two_sole_samples": bad_robot,
        "closed_palm_overlap_mm": -end["palm_signed_distance_mm"],
        "closed_finger_overlap_mm": -end["finger_signed_distance_mm"],
        "both_pad_samples": sum(row["both_pads"] for row in rows),
        "screen_pass": screen_pass, "limit": plan["limits"],
    }
    (OUT / "path-matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
