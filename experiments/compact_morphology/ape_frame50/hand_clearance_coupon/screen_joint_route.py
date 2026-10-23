"""Bounded collision-screened arm path from a native precontact reach state."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import (
    contacts, with_cube,
)


OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/joint-route-screen-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
       "right_elbow_pitch")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    body_path = ROOT / plan["body"]
    source_path = ROOT / plan["native_source"]
    model = with_cube(body_path)
    data = mujoco.MjData(model)
    with np.load(source_path) as trace:
        dt = float(trace["step_dt_s"])
        q0 = trace["qpos"][round(plan["source_time_s"] / dt)].astype(float)
    arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
    jaw_adr = model.joint("right_gripper").qposadr[0]
    site_id = model.site("right_hand_center").id
    cube_xyz = q0[28:31].copy()
    x0 = q0[arm_adr].copy()
    joint_lo = np.array([model.joint(name).range[0] for name in ARM])
    joint_hi = np.array([model.joint(name).range[1] for name in ARM])
    fit_lo = np.maximum(joint_lo, x0 - .55)
    fit_hi = np.minimum(joint_hi, x0 + .55)
    target_site = cube_xyz + np.array([0, 0, .018])

    def inspect(x: np.ndarray, jaw: float = 0.0) -> dict:
        data.qpos[:] = q0
        data.qpos[arm_adr] = x
        data.qpos[jaw_adr] = jaw
        mujoco.mj_forward(model, data)
        return contacts(model, data)

    def residual(x: np.ndarray) -> np.ndarray:
        inspect(x)
        return np.r_[(data.site_xpos[site_id] - target_site) * 1000,
                     (x - x0) / .4]

    fit = least_squares(residual, np.clip(x0, fit_lo + 1e-8, fit_hi - 1e-8),
                        bounds=(fit_lo, fit_hi), max_nfev=120,
                        ftol=1e-9, xtol=1e-9, gtol=1e-9)
    goal = fit.x.copy()
    residual(goal)
    goal_error_mm = float(np.linalg.norm((data.site_xpos[site_id] - target_site) * 1000))
    retreat = x0.copy()
    retreat[1] += .2
    low = np.maximum(joint_lo, x0 - plan["sample"]["joint_search_range_from_native_rad"])
    high = np.minimum(joint_hi, x0 + plan["sample"]["joint_search_range_from_native_rad"])
    extension = plan["sample"]["rrt_extension_rad"]
    resolution = plan["sample"]["collision_edge_resolution_rad"]
    stats = {"state_checks": 0, "rejected_cube": 0, "rejected_robot": 0,
             "rejected_early_pad": 0}

    def valid(x: np.ndarray, *, jaw: float = 0.0, late_pad: bool = True) -> bool:
        stats["state_checks"] += 1
        if np.any(x < joint_lo) or np.any(x > joint_hi):
            return False
        c = inspect(x, jaw)
        if c["other_cube_robot_penetration_mm"]:
            stats["rejected_cube"] += 1
            return False
        if c["nonsole_floor_or_self_penetration_mm"] or not c["two_sole_contact"]:
            stats["rejected_robot"] += 1
            return False
        if late_pad and (c["palm_contact"] or c["finger_contact"]) and np.linalg.norm(x - goal) > .12:
            stats["rejected_early_pad"] += 1
            return False
        return True

    def edge(a: np.ndarray, b: np.ndarray, *, jaw_a: float = 0,
             jaw_b: float = 0, late_pad: bool = True) -> bool:
        steps = max(1, math.ceil(float(np.linalg.norm(a - b)) / resolution))
        for fraction in np.linspace(0, 1, steps + 1):
            if not valid(a + fraction * (b - a),
                         jaw=jaw_a + fraction * (jaw_b - jaw_a), late_pad=late_pad):
                return False
        return True

    source_jaw = float(q0[jaw_adr])
    prefix_ok = (edge(x0, retreat, jaw_a=source_jaw, jaw_b=source_jaw,
                      late_pad=False) and
                 edge(retreat, retreat, jaw_a=source_jaw, jaw_b=0,
                      late_pad=False))
    start_ok = valid(retreat)
    goal_ok = valid(goal)

    def extend(tree: list[tuple[np.ndarray, int]], target: np.ndarray) -> int | None:
        positions = np.stack([node[0] for node in tree])
        nearest = int(np.argmin(np.linalg.norm(positions - target, axis=1)))
        a = tree[nearest][0]
        distance = float(np.linalg.norm(target - a))
        if distance < 1e-8:
            return nearest
        b = a + min(1.0, extension / distance) * (target - a)
        if not edge(a, b):
            return None
        tree.append((b, nearest))
        return len(tree) - 1

    def lineage(tree: list[tuple[np.ndarray, int]], index: int) -> list[np.ndarray]:
        result = []
        while index >= 0:
            node, parent = tree[index]
            result.append(node)
            index = parent
        return result[::-1]

    runs = []
    best_path = None
    if prefix_ok and start_ok and goal_ok and goal_error_mm <= 2:
        for seed in plan["sample"]["seeds"]:
            rng = np.random.default_rng(seed)
            start_tree = [(retreat.copy(), -1)]
            goal_tree = [(goal.copy(), -1)]
            connected = False
            for iteration in range(plan["sample"]["maximum_iterations_per_seed"]):
                point = goal if rng.random() < .1 else rng.uniform(low, high)
                added = extend(start_tree, point)
                if added is None:
                    continue
                target = start_tree[added][0]
                while True:
                    other = extend(goal_tree, target)
                    if other is None:
                        break
                    if np.linalg.norm(goal_tree[other][0] - target) < 1e-6:
                        left = lineage(start_tree, added)
                        right = lineage(goal_tree, other)
                        path = left + list(reversed(right))[1:]
                        connected = True
                        break
                if connected:
                    break
            runs.append({"seed": seed, "iterations": iteration + 1,
                         "start_nodes": len(start_tree), "goal_nodes": len(goal_tree),
                         "connected": connected})
            if connected:
                best_path = path
                break
    if best_path is not None:
        rng = np.random.default_rng(1947)
        for _ in range(100):
            if len(best_path) < 3:
                break
            i, j = sorted(rng.choice(len(best_path), 2, replace=False))
            if j > i + 1 and edge(best_path[i], best_path[j]):
                best_path = best_path[:i + 1] + best_path[j:]
        dense = []
        for a, b in zip(best_path, best_path[1:]):
            steps = max(1, math.ceil(float(np.linalg.norm(a - b)) / .01))
            dense.extend(a + fraction * (b - a) for fraction in
                         np.linspace(0, 1, steps, endpoint=False))
        dense.append(best_path[-1])
        path_cases = []
        for index, x in enumerate(dense):
            c = inspect(x)
            path_cases.append({"index": index, "fraction": round(index / max(1, len(dense) - 1), 4),
                               "arm_rad": np.round(x, 6).tolist(),
                               "hand_site_xyz_m": np.round(data.site_xpos[site_id], 6).tolist(),
                               **c})
        early = [row for row in path_cases if row["fraction"] < .9]
        early_pad_fraction = (sum(row["palm_contact"] or row["finger_contact"]
                                  for row in early) / len(early)) if early else 0
        path_pass = (all(not row["other_cube_robot_penetration_mm"] and
                         not row["nonsole_floor_or_self_penetration_mm"] and
                         row["two_sole_contact"] for row in path_cases) and
                     early_pad_fraction <= .1)
        np.savez_compressed(OUT / "route-states.npz", arm_rad=np.stack(dense),
                            waypoint_arm_rad=np.stack(best_path), source_qpos=q0,
                            goal_arm_rad=goal)
        (OUT / "path-matrix.json").write_text(json.dumps({"cases": path_cases}, indent=2) + "\n")
    else:
        dense = []
        early_pad_fraction = None
        path_pass = False
    report = {
        "schema": "robot-learning-lab.cove50-jaw-entry-joint-route.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(body_path), "source_trace_sha256": sha(source_path),
        "plan_sha256": sha(OUT / "plan.json"), "script_sha256": sha(Path(__file__)),
        "mujoco_version": mujoco.__version__,
        "source_time_s": plan["source_time_s"],
        "source_cube_xyz_m": cube_xyz.tolist(),
        "source_arm_rad": x0.tolist(), "source_jaw_rad": source_jaw,
        "retreat_arm_rad": retreat.tolist(), "goal_arm_rad": goal.tolist(),
        "goal_site_error_mm": round(goal_error_mm, 4),
        "prefix_clear": prefix_ok, "start_clear": start_ok, "goal_clear": goal_ok,
        "search_runs": runs, "collision_stats": stats,
        "route_found": best_path is not None,
        "waypoints": len(best_path) if best_path is not None else 0,
        "dense_path_states": len(dense),
        "early_pad_contact_fraction": (round(early_pad_fraction, 4)
                                       if early_pad_fraction is not None else None),
        "static_route_gate_pass": path_pass,
        "limit": plan["limit"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
