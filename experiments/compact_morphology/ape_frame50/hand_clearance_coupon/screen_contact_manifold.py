"""Find a static two-pad contact path from an attained native open-hand state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import contacts, with_cube

OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/contact-manifold-screen-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
NEAR_CUBE = ("right_palm_rail_frame_budget", "right_jaw_bridge_frame_budget",
             "right_jaw_rail_frame_budget", "right_palm_root_frame_budget",
             "right_jaw_root_frame_budget", "right_gripper_motor_case",
             "right_forearm_frame_budget")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "plan.json"
    plan = json.loads(plan_path.read_text())
    body_path = ROOT / plan["body"]
    trace_path = ROOT / plan["native_source"]
    model = with_cube(body_path)
    data = mujoco.MjData(model)
    with np.load(trace_path) as trace:
        tick = round(plan["source_time_s"] / float(trace["step_dt_s"]))
        source = trace["qpos"][tick].copy()
    arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
    jaw_adr = model.joint("right_gripper").qposadr[0]
    cube = model.geom("audit_cube_geom").id
    palm = model.geom("right_hand").id
    finger = model.geom("right_finger_tip").id
    nearby = [model.geom(name).id for name in NEAR_CUBE]
    all_nonpad = [i for i in range(model.ngeom)
                  if i not in {cube, model.geom("floor").id, palm, finger}
                  and model.geom_bodyid[i] != 0]
    original = source[arm_adr].copy()
    envelope = plan["arm_envelope_rad"]
    low = np.maximum([model.joint(name).range[0] for name in ARM], original - envelope)
    high = np.minimum([model.joint(name).range[1] for name in ARM], original + envelope)
    previous = original.copy()
    rng = np.random.default_rng(plan["optimizer_seed"])
    reference = np.array([.443078, -.102244, .305214, -.229099])
    target = plan["target_pad_overlap_mm"]

    def state(arm: np.ndarray, jaw: float, obstacles: list[int]) -> tuple[float, float, float]:
        data.qpos[:] = source
        data.qpos[arm_adr] = arm
        data.qpos[jaw_adr] = jaw
        mujoco.mj_forward(model, data)
        def distance(geom: int) -> float:
            return float(mujoco.mj_geomDistance(model, data, cube, geom, .05, np.zeros(6))) * 1000
        return distance(palm), distance(finger), min(distance(geom) for geom in obstacles)

    waypoints = []
    for jaw in np.linspace(0, .5, plan["jaw_waypoints"]):
        starts = [previous, previous * .7 + reference * .3,
                  previous * .3 + reference * .7, reference]
        starts += [previous + rng.normal(0, .07, len(ARM)) for _ in range(5)]
        best = None
        for start in starts:
            start = np.clip(start, low + 1e-6, high - 1e-6)

            def residual(arm: np.ndarray) -> np.ndarray:
                palm_mm, finger_mm, other_mm = state(arm, float(jaw), nearby)
                return np.r_[(palm_mm + target) * 5, (finger_mm + target) * 5,
                             max(0, 1 - other_mm) * 3, (arm - previous) * .4]

            fit = least_squares(residual, start, bounds=(low, high), max_nfev=60,
                                ftol=1e-3, xtol=1e-3, gtol=1e-3)
            palm_mm, finger_mm, other_mm = state(fit.x, float(jaw), nearby)
            score = (max(0, palm_mm + .2) * 4 + max(0, finger_mm + .2) * 4
                     + max(0, 1 - other_mm) * 4 + max(0, -palm_mm - .8)
                     + max(0, -finger_mm - .8) + np.linalg.norm(fit.x - previous) * .1)
            if best is None or score < best[0]:
                best = (score, fit.x.copy())
        assert best is not None
        arm = best[1]
        waypoints.append({"jaw_rad": round(float(jaw), 6),
                          "arm_rad": [round(float(v), 8) for v in arm],
                          "arm_step_norm_rad": round(float(np.linalg.norm(arm - previous)), 6)})
        previous = arm

    angles = np.array([row["jaw_rad"] for row in waypoints])
    arms = np.array([row["arm_rad"] for row in waypoints])
    samples = []
    for jaw in np.linspace(0, .5, plan["interpolated_samples"]):
        arm = np.array([np.interp(jaw, angles, arms[:, i]) for i in range(len(ARM))])
        palm_mm, finger_mm, other_mm = state(arm, float(jaw), all_nonpad)
        contact = contacts(model, data)
        samples.append({"jaw_rad": round(float(jaw), 6),
                        "palm_signed_distance_mm": round(palm_mm, 4),
                        "finger_signed_distance_mm": round(finger_mm, 4),
                        "minimum_nonpad_cube_distance_mm": round(other_mm, 4),
                        "both_pads": contact["palm_contact"] and contact["finger_contact"],
                        "two_sole_contact": contact["two_sole_contact"],
                        "nonsole_floor_or_self_penetration_mm": contact["nonsole_floor_or_self_penetration_mm"]})
    minimum_pad_overlap = min(min(-r["palm_signed_distance_mm"], -r["finger_signed_distance_mm"])
                              for r in samples)
    minimum_nonpad = min(r["minimum_nonpad_cube_distance_mm"] for r in samples)
    maximum_step = max(r["arm_step_norm_rad"] for r in waypoints)
    bad_robot = sum(not r["two_sole_contact"] or bool(r["nonsole_floor_or_self_penetration_mm"])
                    for r in samples)
    gate = plan["gates"]
    checks = {
        "each_pad_overlap": minimum_pad_overlap >= gate["minimum_each_pad_overlap_mm"],
        "nonpad_cube_clearance": minimum_nonpad >= gate["minimum_nonpad_cube_clearance_mm"],
        "arm_step": maximum_step <= gate["maximum_waypoint_arm_step_norm_rad"],
        "robot_and_soles": bad_robot <= gate["bad_robot_or_two_sole_samples"],
        "both_pads_all_samples": all(r["both_pads"] for r in samples),
    }
    report = {
        "schema": "robot-learning-lab.cove50-contact-manifold-screen.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(body_path), "source_trace_sha256": sha(trace_path),
        "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "mujoco_version": mujoco.__version__, "source_time_s": plan["source_time_s"],
        "source_cube_xyz_m": source[-7:-4].tolist(),
        "source_arm_rad": original.tolist(),
        "minimum_each_pad_overlap_mm": round(minimum_pad_overlap, 4),
        "minimum_nonpad_cube_distance_mm": round(minimum_nonpad, 4),
        "maximum_waypoint_arm_step_norm_rad": maximum_step,
        "bad_robot_or_two_sole_samples": bad_robot,
        "both_pad_samples": sum(r["both_pads"] for r in samples),
        "interpolated_samples": len(samples), "checks": checks,
        "screen_pass": all(checks.values()), "limit": plan["limits"],
    }
    (OUT / "arm-waypoints.json").write_text(json.dumps({"waypoints": waypoints}, indent=2) + "\n")
    (OUT / "path-matrix.json").write_text(json.dumps({"cases": samples}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
