"""Deterministic local pinch target/path proposal for an attained native state.

The result is a scripted teacher proposal. The simulation must still run the
full motor/contact transition and judge actual forces, cube motion and support.
"""

from __future__ import annotations

import numpy as np
import mujoco
from scipy.optimize import differential_evolution, minimize

from screen_v14_attained_pinches import JOINTS


def plan_live_pinch(model: mujoco.MjModel, base: np.ndarray, *, seed: int,
                    samples: int = 5000) -> dict:
    if base.shape != (model.nq,) or model.nq != 35:
        raise ValueError("expected an actual 35-qpos free-cube native state")
    data = mujoco.MjData(model)
    joints = np.array([int(model.joint(name).qposadr[0]) for name in JOINTS])
    start = base[joints].astype(float).copy()
    bounds = [(max(start[i] - .6, model.joint(name).range[0]),
               min(start[i] + .6, model.joint(name).range[1])) for i, name in enumerate(JOINTS[:4])]
    bounds.append((0.0, 0.5))
    ids = {name: model.geom(name).id for name in (
        "right_hand", "right_finger_tip", "right_forearm_frame_budget",
        "right_forearm_stock_mass", "right_gripper_motor_case",
        "screen_cube_geom", "screen_table_top")}
    witness = np.empty(6)

    def measure(arm: np.ndarray) -> dict:
        qpos = base.copy()
        qpos[joints] = arm
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        def gap(a: str, b: str) -> float:
            return float(mujoco.mj_geomDistance(model, data, ids[a], ids[b], .2, witness))

        pads = [gap(name, "screen_cube_geom") for name in ("right_hand", "right_finger_tip")]
        other = {name: gap(name, "screen_cube_geom") for name in (
            "right_forearm_frame_budget", "right_forearm_stock_mass", "right_gripper_motor_case")}
        table = {name: gap(name, "screen_table_top") for name in (
            "right_hand", "right_finger_tip", "right_forearm_frame_budget")}
        cube_table = gap("screen_cube_geom", "screen_table_top")
        clean = (all(-.002 <= value <= -.0001 for value in pads)
                 and min(other.values()) >= -.0002 and min(table.values()) >= -.0002
                 and -.002 <= cube_table <= .0005)
        return {"arm_qpos_rad": arm.tolist(), "pad_gaps_mm": [x * 1000 for x in pads],
                "other_cube_gaps_mm": {k: v * 1000 for k, v in other.items()},
                "hand_table_gaps_mm": {k: v * 1000 for k, v in table.items()},
                "cube_table_gap_mm": cube_table * 1000,
                "clean_static_target": bool(clean)}

    def objective(candidate: np.ndarray) -> float:
        measured = measure(candidate)
        pads = np.asarray(measured["pad_gaps_mm"])
        pad_loss = np.sum(np.maximum(pads + .3, 0) ** 2 + np.maximum(-.5 - pads, 0) ** 2)
        others = np.asarray(list(measured["other_cube_gaps_mm"].values()))
        table = np.asarray(list(measured["hand_table_gaps_mm"].values()))
        collision = np.sum(np.maximum(.5 - others, 0) ** 2) + np.sum(np.maximum(.5 - table, 0) ** 2)
        move = .01 * np.sum(((candidate[:4] - start[:4]) / .5) ** 2)
        return float(pad_loss + collision + move)

    fit = differential_evolution(objective, bounds, maxiter=140, popsize=12,
                                 seed=seed, polish=False, updating="immediate")
    local = minimize(objective, fit.x, method="Nelder-Mead",
                     options={"maxiter": 2500, "xatol": 1e-7, "fatol": 1e-8})
    inside = all(lo <= value <= hi for value, (lo, hi) in zip(local.x, bounds))
    best = local.x if inside and local.fun < fit.fun else fit.x
    target = measure(best)
    result = {"source_step_arm_start_rad": start.tolist(), "cube_xyz_m": base[28:31].tolist(),
              "bounds_rad": bounds, "target": target,
              "target_objective": objective(best), "target_evaluations": int(fit.nfev + local.nfev),
              "path_found": False, "path_arm_rad": None, "path_collision_evaluations": 0}
    if not target["clean_static_target"]:
        return result

    # At open jaw, seek a joint path around the cube/table. The final close is
    # scored by the native motor simulation rather than by this static screen.
    opened = base.copy()
    opened[joints[-1]] = .5
    path_start = start[:4]
    path_goal = best[:4]
    path_checks = 0

    def valid(q: np.ndarray) -> bool:
        nonlocal path_checks
        path_checks += 1
        opened[joints[:4]] = q
        data.qpos[:] = opened
        mujoco.mj_forward(model, data)

        def gap(a: str, b: str) -> float:
            return float(mujoco.mj_geomDistance(model, data, ids[a], ids[b], .2, witness))

        cube = "screen_cube_geom"
        table = "screen_table_top"
        return (gap("right_forearm_frame_budget", cube) >= -.0002
                and gap("right_forearm_stock_mass", cube) >= -.0002
                and gap("right_hand", cube) >= -.001
                and gap("right_finger_tip", cube) >= -.001
                and gap("right_hand", table) >= -.001
                and gap("right_finger_tip", table) >= -.001
                and gap("right_forearm_frame_budget", table) >= -.0002)

    def edge(a: np.ndarray, b: np.ndarray) -> bool:
        count = max(2, int(np.ceil(np.linalg.norm(b - a) / .025)) + 1)
        return all(valid(a * (1 - u) + b * u) for u in np.linspace(0, 1, count))

    if not valid(path_start) or not valid(path_goal):
        result["path_collision_evaluations"] = path_checks
        return result
    rng = np.random.default_rng(seed + 721)
    trees = [([path_start.copy()], [-1]), ([path_goal.copy()], [-1])]
    connection = None
    for sample in range(samples):
        active = sample % 2
        target_q = path_goal if rng.random() < .1 else rng.uniform(np.asarray(bounds[:4])[:, 0],
                                                                   np.asarray(bounds[:4])[:, 1])
        nodes, parents = trees[active]
        near = int(np.argmin(np.sum((np.asarray(nodes) - target_q) ** 2, axis=1)))
        vector = target_q - nodes[near]
        candidate = nodes[near] + vector * min(1., .08 / max(np.linalg.norm(vector), 1e-9))
        if not edge(nodes[near], candidate):
            continue
        nodes.append(candidate)
        parents.append(near)
        new_index = len(nodes) - 1
        other_nodes, other_parents = trees[1 - active]
        while True:
            other_near = int(np.argmin(np.sum((np.asarray(other_nodes) - candidate) ** 2, axis=1)))
            vector = candidate - other_nodes[other_near]
            next_q = other_nodes[other_near] + vector * min(1., .08 / max(np.linalg.norm(vector), 1e-9))
            if not edge(other_nodes[other_near], next_q):
                break
            other_nodes.append(next_q)
            other_parents.append(other_near)
            other_new = len(other_nodes) - 1
            if np.linalg.norm(next_q - candidate) < 1e-6:
                connection = (new_index, other_new) if active == 0 else (other_new, new_index)
                break
            if np.linalg.norm(next_q - other_nodes[other_near]) < 1e-9:
                break
        if connection is not None:
            break

    def backtrack(tree, index):
        nodes, parents = tree
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    result["path_collision_evaluations"] = path_checks
    if connection is None:
        return result
    left = backtrack(trees[0], connection[0])
    right = backtrack(trees[1], connection[1])
    raw = left + right[::-1][1:]
    simplified = [raw[0]]
    at = 0
    while at < len(raw) - 1:
        following = next(i for i in range(len(raw) - 1, at, -1) if edge(raw[at], raw[i]))
        simplified.append(raw[following])
        at = following
    result["path_found"] = True
    result["path_arm_rad"] = [q.tolist() for q in simplified]
    result["path_collision_evaluations"] = path_checks
    return result
