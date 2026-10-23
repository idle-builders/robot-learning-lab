"""Screen an earlier arm entry against the saved moving support trajectory."""

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

OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/early-aperture-screen-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def main() -> None:
    plan_path = OUT / "plan.json"
    plan = json.loads(plan_path.read_text())
    body_path = ROOT / plan["body"]
    trace_path = ROOT / plan["nominal_native_trace"]
    goal_path = ROOT / plan["seed_goal"]
    model = with_cube(body_path)
    data = mujoco.MjData(model)
    with np.load(trace_path) as saved:
        q_nom = saved["qpos"].copy()
        dt = float(saved["step_dt_s"])
    with np.load(goal_path) as saved:
        seed_goal = saved["goal_arm_rad"].copy()
    if dt != .02 or q_nom.shape[1] != model.nq:
        raise ValueError("Expected saved 50 Hz 21-axis candidate with free cube")
    source_t = plan["source_time_s"]
    goal_t = plan["static_goal_body_time_s"]
    source = q_nom[round(source_t / dt)].copy()
    static_q = q_nom[round(goal_t / dt)].copy()
    cube_pose = source[model.nq - 7:model.nq].copy()
    static_q[model.nq - 7:model.nq] = cube_pose
    arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
    jaw_adr = model.joint("right_gripper").qposadr[0]
    cube_id = model.geom("audit_cube_geom").id
    excluded = {cube_id, model.geom("floor").id, model.geom("right_hand").id,
                model.geom("right_finger_tip").id}
    other_ids = [i for i in range(model.ngeom) if i not in excluded and
                 model.geom_bodyid[i] != 0]
    site_id = model.site("right_hand_center").id
    target = cube_pose[:3] + np.array([0, 0, .018])
    search = plan["goal_search"]
    joint_ranges = np.array([model.joint(name).range for name in ARM])

    def inspect(q: np.ndarray) -> tuple[dict, float, str]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        contact = contacts(model, data)
        distances = [(float(mujoco.mj_geomDistance(model, data, cube_id, i, .05,
                      np.zeros(6))) * 1000,
                      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
                     for i in other_ids]
        clearance, closest = min(distances)
        return contact, clearance, closest

    def static_triplet(x: np.ndarray) -> dict | None:
        if np.any(x < joint_ranges[:, 0]) or np.any(x > joint_ranges[:, 1]):
            return None
        minimum = 999.0
        site_error = None
        for jaw in search["required_jaw_angles_rad"]:
            q = static_q.copy()
            q[arm_adr] = x
            q[jaw_adr] = jaw
            contact, clearance, _ = inspect(q)
            if (not contact["palm_contact"] or not contact["finger_contact"] or
                contact["nonsole_floor_or_self_penetration_mm"] or not contact["two_sole_contact"]):
                return None
            minimum = min(minimum, clearance)
            if jaw == 0:
                site_error = float(np.linalg.norm((data.site_xpos[site_id] - target) * 1000))
        if site_error is None or site_error > search["open_goal_site_error_mm_max"]:
            return None
        return {"arm_rad": x.tolist(), "static_min_nonpad_clearance_mm": minimum,
                "static_goal_site_error_mm": site_error}

    rng = np.random.default_rng(search["seed"])
    candidates = []
    baseline = static_triplet(seed_goal)
    for _ in range(search["samples"]):
        x = seed_goal + rng.normal(0, search["per_joint_std_rad"], len(ARM))
        scored = static_triplet(x)
        if scored is not None:
            candidates.append(scored)
    candidates.sort(key=lambda item: (-item["static_min_nonpad_clearance_mm"],
                                      item["static_goal_site_error_mm"]))
    shortlist = candidates[:20]
    path_cfg = plan["nominal_path"]
    end_t = path_cfg["open_hold_end_s"] + path_cfg["close_duration_s"] + path_cfg["hold_duration_s"]
    ticks = range(round(source_t / dt), round(end_t / dt) + 1)

    def path_score(candidate: dict) -> dict:
        goal = np.array(candidate["arm_rad"])
        minimum = (999.0, "", 0.0)
        first_both = None
        wrong_robot = 0
        for tick in ticks:
            t = tick * dt
            q = q_nom[tick].copy()
            q[model.nq - 7:model.nq] = cube_pose
            fraction = smoothstep((t - source_t) / path_cfg["arm_transition_duration_s"])
            q[arm_adr] = source[arm_adr] + fraction * (goal - source[arm_adr])
            if t < path_cfg["open_hold_end_s"]:
                q[jaw_adr] = source[jaw_adr] * (1 - smoothstep((t - source_t) /
                                                          path_cfg["jaw_open_duration_s"]))
            else:
                q[jaw_adr] = .5 * smoothstep((t - path_cfg["open_hold_end_s"]) /
                                                 path_cfg["close_duration_s"])
            contact, clearance, closest = inspect(q)
            if clearance < minimum[0]:
                minimum = (clearance, closest, t)
            if contact["nonsole_floor_or_self_penetration_mm"] or not contact["two_sole_contact"]:
                wrong_robot += 1
            if first_both is None and contact["palm_contact"] and contact["finger_contact"]:
                first_both = t
        return {**candidate, "moving_body_min_nonpad_clearance_mm": minimum[0],
                "closest_nonpad_geom": minimum[1], "minimum_time_s": round(minimum[2], 3),
                "first_static_two_pad_time_s": (round(first_both, 3) if first_both is not None else None),
                "bad_robot_or_two_sole_frames": wrong_robot,
                "path_screen_pass": (minimum[0] >= search["minimum_nonpad_cube_clearance_mm"] and
                                     wrong_robot == 0 and first_both is not None)}

    path_cases = [path_score(candidate) for candidate in shortlist]
    path_cases.sort(key=lambda item: (-item["moving_body_min_nonpad_clearance_mm"],
                                      item["static_goal_site_error_mm"]))
    selected = next((item for item in path_cases if item["path_screen_pass"]), None)
    report = {
        "schema": "robot-learning-lab.cove50-early-aperture-screen.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(body_path), "nominal_trace_sha256": sha(trace_path),
        "seed_goal_sha256": sha(goal_path), "plan_sha256": sha(plan_path),
        "script_sha256": sha(Path(__file__)), "mujoco_version": mujoco.__version__,
        "samples": search["samples"], "static_triplets_with_site_gate": len(candidates),
        "shortlisted": len(shortlist), "baseline_triplet": baseline,
        "source_arm_rad": source[arm_adr].tolist(),
        "source_jaw_rad": float(source[jaw_adr]),
        "source_cube_xyz_m": cube_pose[:3].tolist(),
        "selected": selected,
        "screen_pass": selected is not None,
        "limit": plan["limits"],
    }
    (OUT / "candidate-matrix.json").write_text(json.dumps({"cases": path_cases}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
