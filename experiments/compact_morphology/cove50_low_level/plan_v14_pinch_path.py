"""Bounded local joint-space clearance plan between V14 touch and static pinch.

The free cube is held at its attained pose only for this geometry search. A
successful path is a teacher proposal, never a physical transition score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from screen_v14_attained_pinches import BODY, JOINTS, ROOT, TRACES, screen_model


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    source = json.loads(args.target.read_text())
    if source["body_sha256"] != sha(BODY) or not source["best"]["clean_static_target"]:
        raise ValueError("target is not exact-body clean")
    model = screen_model()
    data = mujoco.MjData(model)
    with np.load(TRACES / f"seed-{source['seed']}-residual.npz", allow_pickle=False) as trace:
        qpos = trace["qpos"][source["source_frame"]].copy()
    addresses = np.array([model.joint(j).qposadr[0] for j in JOINTS[:4]])
    qpos[model.joint(JOINTS[-1]).qposadr[0]] = 0.5
    ids = {name: model.geom(name).id for name in (
        "screen_cube_geom", "screen_table_top", "right_hand", "right_finger_tip",
        "right_forearm_frame_budget", "right_forearm_stock_mass")}
    witness = np.empty(6)

    def gap(a: str, b: str) -> float:
        return float(mujoco.mj_geomDistance(model, data, ids[a], ids[b], 0.2, witness))

    evaluations = 0

    def valid(q: np.ndarray) -> bool:
        nonlocal evaluations
        evaluations += 1
        qpos[addresses] = q
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        cube = "screen_cube_geom"
        table = "screen_table_top"
        return (gap("right_forearm_frame_budget", cube) >= -0.0002
                and gap("right_forearm_stock_mass", cube) >= -0.0002
                and gap("right_hand", cube) >= -0.001
                and gap("right_finger_tip", cube) >= -0.001
                and gap("right_hand", table) >= -0.0002
                and gap("right_finger_tip", table) >= -0.0002
                and gap("right_forearm_frame_budget", table) >= -0.0002)

    def edge(start: np.ndarray, end: np.ndarray) -> bool:
        length = float(np.linalg.norm(end - start))
        for alpha in np.linspace(0, 1, max(2, int(np.ceil(length / 0.025)) + 1)):
            if not valid(start * (1 - alpha) + end * alpha):
                return False
        return True

    start = trace_start = np.array(source["initial"]["arm_qpos_rad"][:4])
    goal = np.array(source["best"]["arm_qpos_rad"][:4])
    if not valid(start) or not valid(goal):
        raise ValueError("source or goal has a forbidden static contact")
    bounds = np.array(source["bounds"][:4], dtype=float)
    rng = np.random.default_rng(source["seed"] + 721)
    trees = [([start], [-1]), ([goal], [-1])]
    connection = None
    for sample in range(args.samples):
        active = sample % 2
        target = goal if rng.random() < 0.1 else rng.uniform(bounds[:, 0], bounds[:, 1])
        nodes, parents = trees[active]
        near = int(np.argmin(np.sum((np.array(nodes) - target) ** 2, axis=1)))
        vector = target - nodes[near]
        candidate = nodes[near] + vector * min(1.0, 0.08 / max(np.linalg.norm(vector), 1e-9))
        if not edge(nodes[near], candidate):
            continue
        nodes.append(candidate)
        parents.append(near)
        new_index = len(nodes) - 1
        other_nodes, other_parents = trees[1 - active]
        while True:
            other_near = int(np.argmin(np.sum((np.array(other_nodes) - candidate) ** 2, axis=1)))
            vector = candidate - other_nodes[other_near]
            next_q = other_nodes[other_near] + vector * min(1.0, 0.08 / max(np.linalg.norm(vector), 1e-9))
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
        if connection:
            break

    def backtrack(tree: tuple[list[np.ndarray], list[int]], index: int) -> list[np.ndarray]:
        nodes, parents = tree
        result = []
        while index >= 0:
            result.append(nodes[index])
            index = parents[index]
        return result[::-1]

    path = None
    if connection:
        left = backtrack(trees[0], connection[0])
        right = backtrack(trees[1], connection[1])
        path = left + right[::-1][1:]
        assert np.allclose(path[0], trace_start) and np.allclose(path[-1], goal)
        assert all(edge(a, b) for a, b in zip(path, path[1:]))
    raw_waypoints = len(path) if path else 0
    if path:
        simplified = [path[0]]
        position = 0
        while position < len(path) - 1:
            following = next(index for index in range(len(path) - 1, position, -1)
                             if edge(path[position], path[index]))
            simplified.append(path[following])
            position = following
        path = simplified
    result = {"schema": "robot-learning-lab.v14-pinch-static-path.v1",
              "claim_limit": "Local fixed-cube geometry path only; no DC motor, moving cube or native grasp score",
              "source_sha256": sha(Path(__file__)),
              "target": str(args.target), "target_sha256": sha(args.target),
              "body_sha256": sha(BODY), "source_trace_sha256": source["source_trace_sha256"],
              "local_mujoco_version": mujoco.__version__, "samples_requested": args.samples,
              "samples_used": sample + 1, "collision_evaluations": evaluations,
              "raw_waypoints": raw_waypoints,
              "found": path is not None, "waypoints_arm_rad": [x.tolist() for x in path] if path else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"found": result["found"], "samples_used": result["samples_used"],
                      "waypoints": len(path) if path else 0, "evaluations": evaluations}))


if __name__ == "__main__":
    main()
