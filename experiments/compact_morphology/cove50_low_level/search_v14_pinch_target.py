"""Look for a clean static two-pad target near an attained V14 free-cube touch.

This is a local kinematic target search, not a motor transition or grasp score.
The target must keep the cube on the table and avoid forearm/hand shortcuts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution, minimize

from screen_v14_attained_pinches import BODY, JOINTS, ROOT, TRACES, screen_model


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=range(37301, 37306), required=True)
    parser.add_argument("--source-frame", type=int,
                        help="one attained physical-touch frame; default is the middle touch frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cube-xy-bound-mm", type=float, default=15.0,
                        help="maximum target cube XY shift from attained contact")
    parser.add_argument("--clearance-mm", type=float, default=0.0,
                        help="desired clearance from non-pad cube and table geometry")
    parser.add_argument("--pad-penetration-mm", type=float, default=0.0,
                        help="desired small pad overlap to seed contact force")
    args = parser.parse_args()
    model, data = screen_model(), None
    data = mujoco.MjData(model)
    trace_path = TRACES / f"seed-{args.seed}-residual.npz"
    with np.load(trace_path, allow_pickle=False) as saved:
        hits = np.flatnonzero(saved["physical_touch"])
        if not len(hits):
            raise ValueError("selected source visit never touched the cube")
        frame = int(args.source_frame if args.source_frame is not None
                    else hits[len(hits) // 2])
        if frame not in hits:
            raise ValueError("requested frame is not a V14 physical-touch frame")
        base = saved["qpos"][frame].astype(float)
        cube_recorded = saved["cube_xyz_m"][frame].astype(float)
    if np.linalg.norm(base[28:31] - cube_recorded) > 1e-5:
        raise ValueError("saved cube and qpos disagree")
    arm_ids = np.array([int(model.joint(name).qposadr[0]) for name in JOINTS])
    lower = np.array([model.joint(name).range[0] for name in JOINTS])
    upper = np.array([model.joint(name).range[1] for name in JOINTS])
    ids = {name: model.geom(name).id for name in (
        "right_hand", "right_finger_tip", "right_forearm_frame_budget",
        "right_forearm_stock_mass", "right_gripper_motor_case",
        "screen_cube_geom", "screen_table_top")}
    witness = np.empty(6)

    def measure(candidate: np.ndarray) -> dict:
        qpos = base.copy()
        qpos[arm_ids] = candidate[:5]
        qpos[28:30] = candidate[5:7]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        def gap(a: str, b: str) -> float:
            return float(mujoco.mj_geomDistance(model, data, ids[a], ids[b], 0.2, witness))

        fixed = gap("right_hand", "screen_cube_geom")
        moving = gap("right_finger_tip", "screen_cube_geom")
        others = {name: gap(name, "screen_cube_geom") for name in (
            "right_forearm_frame_budget", "right_forearm_stock_mass",
            "right_gripper_motor_case")}
        table = {name: gap(name, "screen_table_top") for name in (
            "right_hand", "right_finger_tip", "right_forearm_frame_budget")}
        cube_table = gap("screen_cube_geom", "screen_table_top")
        clean = (max(fixed, moving) <= 0.0005
                 and min(fixed, moving) >= -0.002
                 and -0.002 <= cube_table <= 0.0005
                 and min(others.values()) >= -0.0005
                 and min(table.values()) >= -0.0005)
        return {"fixed_pad_gap_m": fixed, "moving_pad_gap_m": moving,
                "cube_table_gap_m": cube_table, "other_cube_gap_m": others,
                "hand_table_gap_m": table, "clean_static_target": bool(clean),
                "arm_qpos_rad": qpos[arm_ids].tolist(),
                "cube_xy_m": qpos[28:30].tolist()}

    def objective(candidate: np.ndarray) -> float:
        m = measure(candidate)
        def interval(gap: float, lo: float, hi: float) -> float:
            return max(lo - gap, 0.0) ** 2 + max(gap - hi, 0.0) ** 2
        pad_goal = -args.pad_penetration_mm / 1000
        clearance_goal = args.clearance_mm / 1000
        millimetres_squared = 1e6 * (
            interval(m["fixed_pad_gap_m"], pad_goal - 0.0002, pad_goal)
            + interval(m["moving_pad_gap_m"], pad_goal - 0.0002, pad_goal)
            + interval(m["cube_table_gap_m"], -0.001, 0.0)
            + sum(max(clearance_goal - gap, 0.0) ** 2
                  for gap in m["other_cube_gap_m"].values())
            + sum(max(clearance_goal - gap, 0.0) ** 2
                  for gap in m["hand_table_gap_m"].values()))
        move = 0.01 * float(np.sum(((candidate[:4] - base[arm_ids[:4]]) / 0.5) ** 2))
        move += 0.01 * float(np.sum(((candidate[5:7] - base[28:30]) / 0.02) ** 2))
        return millimetres_squared + move

    arm_bounds = [(max(base[at] - 0.6, lo), min(base[at] + 0.6, hi))
                  for at, lo, hi in zip(arm_ids[:4], lower[:4], upper[:4])]
    cube_delta = args.cube_xy_bound_mm / 1000
    bounds = arm_bounds + [(0.0, 0.5),
                           (max(base[28] - cube_delta, 0.035),
                            min(base[28] + cube_delta, 0.065)),
                           (max(base[29] - cube_delta, -0.17),
                            min(base[29] + cube_delta, -0.11))]
    start = np.r_[base[arm_ids], base[28:30]]
    initial = measure(start)
    optimum = differential_evolution(objective, bounds, maxiter=args.iterations,
                                     popsize=12, seed=args.seed, polish=False,
                                     updating="immediate", workers=1)
    refined = minimize(objective, optimum.x, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-8})
    # Nelder-Mead can leave its box. The box-constrained candidate remains the
    # reported result unless the refinement is inside every frozen bound.
    within = all(lo <= val <= hi for val, (lo, hi) in zip(refined.x, bounds))
    best = refined.x if within and refined.fun < optimum.fun else optimum.x
    result = {"schema": "robot-learning-lab.v14-pinch-target-search.v1",
              "claim_limit": "Local MuJoCo static kinematic target only; no motor transition, force, lift or native score",
              "seed": args.seed, "source_frame": frame,
              "source_trace": str(trace_path.relative_to(ROOT)),
              "source_trace_sha256": digest(trace_path),
              "body_sha256": digest(BODY),
              "source_sha256": digest(Path(__file__)),
              "local_mujoco_version": mujoco.__version__,
              "bounds": bounds, "iterations": args.iterations,
              "cube_xy_bound_mm": args.cube_xy_bound_mm,
              "desired_clearance_mm": args.clearance_mm,
              "desired_pad_penetration_mm": args.pad_penetration_mm,
              "initial": initial, "best": measure(best),
              "best_objective": float(objective(best)),
              "evaluations": int(optimum.nfev + refined.nfev)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seed": args.seed, "clean": result["best"]["clean_static_target"],
                      "objective": result["best_objective"], "evaluations": result["evaluations"]}))


if __name__ == "__main__":
    main()
