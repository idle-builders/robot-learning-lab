"""Find a local two-pad target at the *actual* V3 native attained cube state.

This is a fixed-state geometry screen. It does not replay motors, prove contact
force or create an acquisition demonstration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution, minimize

from screen_v14_attained_pinches import BODY, JOINTS, ROOT, screen_model


SOURCE = ROOT / "artifacts/ape-low-level-training-20261015/object-native-pinch-entry-v3/seed-37301-original.npz"
OLD_TARGET = ROOT / "artifacts/ape-low-level-training-20261015/object-attained-pinch-target-v2/seed-37301-force-margin.json"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/object-attained-native-prefix-v4/target-screen.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = screen_model()
    data = mujoco.MjData(model)
    with np.load(SOURCE, allow_pickle=False) as trace:
        if trace["qpos"].shape != (171, model.nq):
            raise ValueError("expected the complete original V3 native prefix")
        base = trace["qpos"][-1].astype(float).copy()
        cube_recorded = trace["cube_xyz_m"][-1].astype(float)
    if np.max(np.abs(base[28:31] - cube_recorded)) > 1e-6:
        raise ValueError("native cube and saved qpos differ")
    ids = {name: model.geom(name).id for name in (
        "right_hand", "right_finger_tip", "right_forearm_frame_budget",
        "right_forearm_stock_mass", "right_gripper_motor_case",
        "screen_cube_geom", "screen_table_top")}
    joints = np.array([int(model.joint(name).qposadr[0]) for name in JOINTS])
    start = base[joints].copy()
    bounds = [(max(start[i] - .6, model.joint(name).range[0]),
               min(start[i] + .6, model.joint(name).range[1])) for i, name in enumerate(JOINTS[:4])]
    bounds.append((0.0, 0.5))
    witness = np.empty(6)

    def measure(candidate: np.ndarray) -> dict:
        qpos = base.copy()
        qpos[joints] = candidate
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
        return {"arm_qpos_rad": candidate.tolist(), "pad_gaps_mm": [x * 1000 for x in pads],
                "other_cube_gaps_mm": {k: v * 1000 for k, v in other.items()},
                "hand_table_gaps_mm": {k: v * 1000 for k, v in table.items()},
                "cube_table_gap_mm": cube_table * 1000,
                "clean_static_target": bool(clean)}

    def objective(candidate: np.ndarray) -> float:
        m = measure(candidate)
        pads = np.asarray(m["pad_gaps_mm"])
        # Aim for a shallow pad overlap. Reject large penetration and use the
        # other geometry only as an explicit collision penalty.
        pad_loss = np.sum(np.maximum(pads + .3, 0) ** 2 + np.maximum(-.5 - pads, 0) ** 2)
        others = np.asarray(list(m["other_cube_gaps_mm"].values()))
        table = np.asarray(list(m["hand_table_gaps_mm"].values()))
        collision = np.sum(np.maximum(.5 - others, 0) ** 2) + np.sum(np.maximum(.5 - table, 0) ** 2)
        move = .01 * np.sum(((candidate[:4] - start[:4]) / .5) ** 2)
        return float(pad_loss + collision + move)

    prior = np.asarray(json.loads(OLD_TARGET.read_text())["best"]["arm_qpos_rad"], dtype=float)
    initial = measure(start)
    prior_on_actual_cube = measure(prior)
    fit = differential_evolution(objective, bounds, maxiter=140, popsize=12,
                                 seed=37301, polish=False, updating="immediate")
    local = minimize(objective, fit.x, method="Nelder-Mead",
                     options={"maxiter": 2500, "xatol": 1e-7, "fatol": 1e-8})
    inside = all(lo <= value <= hi for value, (lo, hi) in zip(local.x, bounds))
    best = local.x if inside and local.fun < fit.fun else fit.x
    result = {
        "schema": "robot-learning-lab.attained-native-pinch-v4-static-screen.v1",
        "claim_limit": "local fixed actual V3 native attained state; no motor transition or contact-force score",
        "source_trace": str(SOURCE.relative_to(ROOT)), "source_trace_sha256": sha(SOURCE),
        "body_sha256": sha(BODY), "search_source_sha256": sha(Path(__file__)),
        "mujoco_version": mujoco.__version__, "source_step": 171,
        "actual_cube_xyz_m": cube_recorded.tolist(), "arm_start_rad": start.tolist(),
        "bounds_rad": bounds, "initial": initial, "old_target_on_actual_cube": prior_on_actual_cube,
        "best": measure(best), "best_objective": objective(best),
        "evaluations": int(fit.nfev + local.nfev),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"clean": result["best"]["clean_static_target"],
                      "pad_gaps_mm": result["best"]["pad_gaps_mm"],
                      "evaluations": result["evaluations"]}))


if __name__ == "__main__":
    main()
