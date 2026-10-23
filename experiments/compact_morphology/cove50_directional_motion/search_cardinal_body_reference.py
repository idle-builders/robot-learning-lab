"""Small local whole-motion feasibility screen on the exact Direct Frame.

This searches scripted target shapes only. A resulting episode is not an
imitation demonstration until it passes the physical gate in fresh CUDA replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ape_walk import task as ape
from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                      configure_frame50_motors,
                                      give_soles_contact_priority,
                                      require_direct_frame)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
EXPECTED_BODY = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_env(seed: int) -> ManagerBasedRlEnv:
    if sha(BODY) != EXPECTED_BODY:
        raise RuntimeError("Direct Frame body changed")
    require_direct_frame(BODY)
    cfg = load_env_cfg(ape.ELBOW_TASK_ID, play=True)
    cfg.scene.entities["robot"].spec_fn = partial(ape._ape_spec, BODY.resolve())
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    give_soles_contact_priority(cfg)
    configure_frame50_motors(cfg, BODY, rail_v=12.0, delay_s=0.0)
    clip_frame50_leg_targets(cfg, BODY, margin_rad=.02, axes=("ankle_pitch",))
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, .231)
    cfg.scene.entities["robot"].init_state.joint_pos[r".*shoulder_pitch.*"] = 0.0
    cfg.scene.entities["robot"].init_state.joint_pos[r".*elbow_pitch.*"] = 0.0
    cfg.scene.num_envs, cfg.seed, cfg.episode_length_s, cfg.auto_reset = 1, seed, 4.0, False
    cfg.commands["twist"].rel_standing_envs = 0.0
    cmd = cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (-.25, -.25)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.resampling_time_range = (100.0, 100.0)
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def smooth(t: float, a: float, b: float) -> float:
    x = np.clip((t - a) / (b - a), 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def target_at(base: np.ndarray, case: dict, t: float) -> np.ndarray:
    """Shift, lift/swing, plant and settle without altering the root directly."""
    q = base.copy()
    swing = case["swing"]
    stance = "left" if swing == "right" else "right"
    index = {"left": 0, "right": 5}
    si, wi = index[stance], index[swing]
    shift = smooth(t, .10, .65) * (1 - smooth(t, 1.70, 2.40))
    airborne = smooth(t, .60, .90) * (1 - smooth(t, 1.35, 1.75))
    plant = smooth(t, .85, 1.35) * (1 - smooth(t, 2.00, 2.60))
    direction = 1 if swing == "left" else -1
    q[si + 1] += direction * case["lean"] * shift
    q[wi + 1] += direction * case["lean"] * shift
    q[wi + 3] += case["knee_lift"] * airborne
    q[wi + 4] -= .5 * case["knee_lift"] * airborne
    if case["direction"] == "backward":
        q[wi + 2] += case["travel"] * plant
        q[wi + 4] -= case["travel"] * plant
    else:
        sign = 1 if case["direction"] == "left" else -1
        q[wi + 1] += sign * case["travel"] * plant
    return q


CASES = [
    {"id": "backward-right-small", "direction": "backward", "swing": "right", "lean": .08, "knee_lift": .25, "travel": .15},
    {"id": "backward-left-small", "direction": "backward", "swing": "left", "lean": .08, "knee_lift": .25, "travel": .15},
    {"id": "left-small", "direction": "left", "swing": "left", "lean": .08, "knee_lift": .25, "travel": .15},
    {"id": "left-medium", "direction": "left", "swing": "left", "lean": .14, "knee_lift": .40, "travel": .25},
    {"id": "right-small", "direction": "right", "swing": "right", "lean": .08, "knee_lift": .25, "travel": .15},
    {"id": "right-medium", "direction": "right", "swing": "right", "lean": .14, "knee_lift": .40, "travel": .25},
]


def visit(env: ManagerBasedRlEnv, case: dict, seed: int) -> dict:
    env.reset(seed=seed)
    term = env.action_manager.get_term("joint_pos")
    offset = term._offset[0].detach().cpu().numpy().copy()
    scale = term._scale[0].detach().cpu().numpy().copy()
    robot = env.scene["robot"]
    sensors = env.scene.sensors
    start = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    first_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
    rated = np.asarray([a.effort_limit for a in env.cfg.scene.entities["robot"].articulation.actuators[:10]])
    rows = []
    for step in range(200):
        time_s = (step + 1) * env.step_dt
        desired = target_at(offset[:10], case, time_s)
        raw = (desired - offset[:10]) / scale[:10]
        action = np.zeros((1, 21), dtype=np.float32)
        action[0, :10] = raw
        _, reward, terminated, timed_out, _ = env.step(torch.as_tensor(action))
        root = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
        sole = sensors["feet_ground_contact"].data.found[0].reshape(2, -1).any(dim=-1).detach().cpu().numpy()
        effort = np.abs(robot.data.actuator_force[0, :10].detach().cpu().numpy())
        hand_force = float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).amax())
        rows.append({"step": step + 1, "root_xyz_m": root.tolist(),
                     "sole_found": sole.astype(bool).tolist(),
                     "max_rated_ratio": float(np.max(effort / rated)),
                     "hand_force_n": hand_force,
                     "reward": float(reward[0])})
        if bool(terminated[0]) or bool(timed_out[0]):
            break
    soles = np.asarray([r["sole_found"] for r in rows], dtype=bool)
    root_end = np.asarray(rows[-1]["root_xyz_m"])
    delta = root_end - start
    progress = -delta[0] if case["direction"] == "backward" else delta[1] * (1 if case["direction"] == "left" else -1)
    touchdown = ((~soles[:-1]) & soles[1:]).sum(axis=0).tolist()
    return {"case": case, "seed": seed, "first_qpos_sha256": hashlib.sha256(first_qpos.tobytes()).hexdigest(),
            "steps": len(rows), "net_root_xyz_m": delta.tolist(), "signed_progress_m": float(progress),
            "min_root_z_m": float(min(r["root_xyz_m"][2] for r in rows)),
            "both_sole_fraction": float(np.mean(np.all(soles, axis=1))),
            "one_sole_frames": int(np.sum(soles.sum(axis=1) == 1)),
            "touchdowns_left_right": touchdown,
            "rated_leg_frame_fraction": float(np.mean([r["max_rated_ratio"] >= .99 for r in rows])),
            "max_hand_force_n": max(r["hand_force_n"] for r in rows),
            "reward_sum": float(sum(r["reward"] for r in rows)),
            "local_physical_screen_pass": bool(len(rows) == 200 and min(r["root_xyz_m"][2] for r in rows) >= .17
                                               and progress >= (.25 if case["direction"] == "backward" else .15)
                                               and sum(touchdown) >= 2 and max(r["hand_force_n"] for r in rows) < 1.0),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    env = make_env(seed=49401)
    results = []
    try:
        for case in CASES[: args.limit or None]:
            result = visit(env, case, seed=49401)
            results.append(result)
            print(json.dumps({k: result[k] for k in ("case", "steps", "signed_progress_m", "one_sole_frames", "touchdowns_left_right", "local_physical_screen_pass")}), flush=True)
    finally:
        env.close()
    report = {"schema": "robot-learning-lab.cardinal-body-reference-local-screen.v1",
              "claim_limit": "Local CPU exact-body exploratory action targets; not accepted demos or CUDA native scores",
              "body_sha256": sha(BODY), "source_sha256": sha(Path(__file__)),
              "seed": 49401, "control_hz": 50, "rail_v": 12.0, "delay_ms": 0,
              "planned_cases": len(CASES), "evaluated_cases": len(results),
              "gpu_hours": 0, "optimizer_updates": 0, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
