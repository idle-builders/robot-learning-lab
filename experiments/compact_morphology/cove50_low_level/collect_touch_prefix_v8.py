"""Collect and replay five gentle observed-RGB whole teacher episodes.

This is a local demonstration-prefix readiness check. Simulator cube/contact
truth accepts or rejects an episode; it is never an actor or teacher input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from ape_walk.physics.frame50 import require_direct_frame
from collect_free_object_touch import BODY
from free_object_touch_env import (ROOT, STAND_TABLE_CUBE_XYZ,
                                   physical_touch_gate_frame, touch_env_cfg)
from probe_observed_free_object_touch import entry_rgb_calibration
from screen_observed_free_object_touch_v4 import action_target, rgb_estimate


SOURCE_LEDGER = ROOT / "artifacts/ape-low-level-training-20261015/free-object-touch-bc-v6/teacher/report.json"
V7_SCREEN = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-trigger-v7/all-sixteen.json"
SEEDS = (33007, 33009, 33012, 33013, 33016)
ALPHA_CAP = .975


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_env(seed: int, cube_xy: tuple[float, float]) -> ManagerBasedRlEnv:
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, seed, False
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def collect(case: dict, coeff: np.ndarray, out: Path) -> dict:
    seed, cube_xy = int(case["seed"]), tuple(case["cube_xy_m"])
    env = make_env(seed, cube_xy)
    try:
        obs, _ = env.reset(seed=seed)
        first_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
        first_qvel = env.sim.data.qvel[0].detach().cpu().numpy().copy()
        first_actor = obs["actor"][0].detach().cpu().numpy().copy()
        rgb = first_actor[-6:]
        estimate = rgb_estimate(rgb, coeff)
        if estimate is None or rgb[3] != 1 or rgb[5] != 0:
            raise RuntimeError(f"seed {seed}: invalid entry RGB")
        target = action_target(env, require_direct_frame(BODY), estimate)
        term = env.action_manager.get_term("joint_pos")
        offset, scale = term._offset[0], term._scale[0]
        last_rgb = rgb.copy()
        hold_alpha = None
        first_finger = None
        running = longest = table_frames = hand_floor_frames = 0
        actor, action, qpos, qvel, cube, gate, finger, table, cue = [], [], [], [], [], [], [], [], []
        for step in range(1, 201):
            actor.append(obs["actor"][0].detach().cpu().numpy().copy())
            alpha = min(ALPHA_CAP, step / 75) if hold_alpha is None else hold_alpha
            desired = offset + alpha * (target - offset)
            raw = torch.where(scale.abs() > 1e-8,
                              (desired - offset) / scale.clamp_min(1e-8),
                              torch.zeros_like(desired))
            a = raw.clamp(-1, 1)[None, :]
            action.append(a[0].detach().cpu().numpy().copy())
            obs, _, terminated, timed_out, _ = env.step(a)
            rgb = obs["actor"][0, -6:].detach().cpu().numpy().copy()
            is_finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
            is_table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
            is_hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
            if is_finger and first_finger is None:
                first_finger = step
            du, dv = rgb[:2] - last_rgb[:2]
            is_cue = bool(rgb[3] == 1 and rgb[5] == 0 and du <= -.0015 and dv <= -.0015)
            if hold_alpha is None and is_cue:
                hold_alpha = alpha
            last_rgb = rgb
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            running = running + 1 if valid else 0
            longest = max(longest, running)
            table_frames += int(is_table)
            hand_floor_frames += int(is_hand_floor)
            qpos.append(env.sim.data.qpos[0].detach().cpu().numpy().copy())
            qvel.append(env.sim.data.qvel[0].detach().cpu().numpy().copy())
            cube.append(env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy())
            gate.append(valid)
            finger.append(is_finger)
            table.append(is_table)
            cue.append(is_cue)
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        path = out / f"seed-{seed}-teacher.npz"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, first_qpos=first_qpos, first_qvel=first_qvel,
                            first_actor=first_actor, actor_obs=np.asarray(actor),
                            action=np.asarray(action), qpos=np.asarray(qpos),
                            qvel=np.asarray(qvel), cube_xyz_m=np.asarray(cube),
                            physical_touch=np.asarray(gate), finger=np.asarray(finger),
                            table=np.asarray(table), rgb_cue=np.asarray(cue))
        cube_shift = np.linalg.norm((np.asarray(cube)[:, :2] - np.asarray(cube)[0, :2]) * 1000, axis=1)
        accepted = bool(len(action) == 200 and longest >= 50 and table_frames >= 190
                        and table[-1] and hand_floor_frames == 0 and cube_shift.max() <= 5)
        return {"seed": seed, "cube_xy_m": list(cube_xy), "episode": str(path.relative_to(ROOT)),
                "episode_sha256": sha(path), "steps": len(action), "first_finger_step": first_finger,
                "longest_supported_touch_s": longest * .02,
                "table_frames": table_frames, "hand_floor_frames": hand_floor_frames,
                "cube_peak_xy_drift_mm": float(cube_shift.max()), "accepted_gentle_whole_teacher": accepted,
                "prefix_steps_precontact_10_20_25": [first_finger - lead for lead in (10, 20, 25)]
                if first_finger is not None else [], "actor_width": int(np.asarray(actor).shape[1])}
    finally:
        env.close()


def replay(case: dict, row: dict) -> dict:
    seed, xy = int(case["seed"]), tuple(case["cube_xy_m"])
    with np.load(ROOT / row["episode"]) as saved:
        first_qpos, first_qvel, first_actor = [saved[k].copy() for k in
                                                ("first_qpos", "first_qvel", "first_actor")]
        action = saved["action"].copy()
        qpos, qvel, actor = [saved[k].copy() for k in ("qpos", "qvel", "actor_obs")]
    env = make_env(seed, xy)
    try:
        obs, _ = env.reset(seed=seed)
        start = {"qpos": float(np.max(np.abs(env.sim.data.qpos[0].detach().cpu().numpy() - first_qpos))),
                 "qvel": float(np.max(np.abs(env.sim.data.qvel[0].detach().cpu().numpy() - first_qvel))),
                 "actor": float(np.max(np.abs(obs["actor"][0].detach().cpu().numpy() - first_actor)))}
        cutoffs = sorted(set(row["prefix_steps_precontact_10_20_25"]))
        if not cutoffs or min(cutoffs) < 1:
            raise RuntimeError(f"seed {seed}: no safe precontact cutoff")
        comparisons = []
        for step in range(1, max(cutoffs) + 1):
            current = obs["actor"][0].detach().cpu().numpy()
            obs_gap = float(np.max(np.abs(current - actor[step - 1])))
            obs, _, _, _, _ = env.step(torch.as_tensor(action[step - 1][None, :]))
            if step in cutoffs:
                nowq = env.sim.data.qpos[0].detach().cpu().numpy()
                nowv = env.sim.data.qvel[0].detach().cpu().numpy()
                comparisons.append({"step": step, "pre_action_actor_max_gap": obs_gap,
                                    "post_action_qpos_max_gap": float(np.max(np.abs(nowq - qpos[step - 1]))),
                                    "post_action_qvel_max_gap": float(np.max(np.abs(nowv - qvel[step - 1])))})
        return {"seed": seed, "entry_max_gaps": start, "cutoffs": comparisons,
                "local_exact_replay": bool(max(start.values()) <= 1e-6 and all(
                    max(c["pre_action_actor_max_gap"], c["post_action_qpos_max_gap"],
                        c["post_action_qvel_max_gap"]) <= 1e-5 for c in comparisons))}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    source = json.loads(SOURCE_LEDGER.read_text())
    screen = json.loads(V7_SCREEN.read_text())
    matches = {int(r["seed"]): r for r in screen["results"]}
    cases = {int(r["seed"]): r for r in source["records"]}
    assert sha(BODY) == source["body_sha256"] == screen["body_sha256"]
    for seed in SEEDS:
        r = matches[seed]
        assert r["full_teacher_gate"] and r["cube_peak_xy_drift_mm"] <= 5 and r["alpha_cap"] == ALPHA_CAP
    coefficients, calibration = entry_rgb_calibration()
    rows = []
    for seed in SEEDS[: args.limit or None]:
        row = collect(cases[seed], coefficients, args.out)
        row["replay"] = replay(cases[seed], row)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("seed", "accepted_gentle_whole_teacher", "replay")}), flush=True)
    report = {"schema": "robot-learning-lab.touch-prefix-v8-local-collection.v1",
              "claim_limit": "Local CPU development demonstration prefixes; no learned actor, CUDA parity or native score",
              "body_sha256": sha(BODY), "source_sha256": sha(Path(__file__)),
              "source_ledger_sha256": sha(SOURCE_LEDGER), "v7_screen_sha256": sha(V7_SCREEN),
              "selected_seeds": list(SEEDS), "calibration": calibration,
              "attempted": len(rows), "accepted_gentle_whole_teacher": sum(r["accepted_gentle_whole_teacher"] for r in rows),
              "local_exact_replay_count": sum(r["replay"]["local_exact_replay"] for r in rows),
              "gpu_hours": 0, "optimizer_updates": 0, "records": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
