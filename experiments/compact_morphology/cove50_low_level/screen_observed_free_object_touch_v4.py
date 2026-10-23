"""Prospective CPU one-pad image-feedback screen on whole cube families.

The intervention uses the free cube joint only to set an external challenge;
the scripted controllers select actions from the actor's RGB observation.
All contact, cube pose and support state are evaluator-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from collect_free_object_touch import BODY, EXPECTED_BODY_SHA, adapted_target
from free_object_touch_env import (ROOT, STAND_TABLE_CUBE_XYZ,
                                   physical_touch_gate_frame, touch_env_cfg)
from probe_observed_free_object_touch import entry_rgb_calibration
from ape_walk.physics.frame50 import require_direct_frame

PLAN = ROOT / "artifacts/ape-low-level-training-20261013/free-object-observed-touch-v4/plan.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb_estimate(rgb: np.ndarray, coefficients: np.ndarray) -> np.ndarray | None:
    if rgb[3] != 1 or rgb[5] != 0:
        return None
    return np.asarray([1.0, *rgb[:3]]) @ coefficients


def action_target(env, model, cube_xy: np.ndarray) -> torch.Tensor:
    ik = adapted_target(model, (float(cube_xy[0]), float(cube_xy[1])))
    term = env.action_manager.get_term("joint_pos")
    target = term._offset[0].clone()
    for name, value in ik.items():
        target[term._target_names.index(name)] = value
    return target


def episode(case: dict, mode: str, coefficients: np.ndarray, out: Path) -> dict:
    seed = int(case["seed"])
    cube_xy = tuple(case["cube_xy_m"])
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        assert env.observation_manager.active_terms["actor"] == [
            "joint_pos", "joint_vel", "actions", "rgb_target"]
        assert list(obs["actor"].shape) == [1, 69]
        initial_rgb = obs["actor"][0, -6:].detach().cpu().numpy()
        initial_est = rgb_estimate(initial_rgb, coefficients)
        assert initial_est is not None
        initial_contact = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any() or
                               env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
        assert not initial_contact
        initial_cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
        assert np.linalg.norm(initial_cube[:2] - np.asarray(cube_xy)) < 1e-4
        model = require_direct_frame(BODY)
        term = env.action_manager.get_term("joint_pos")
        offset, scale = term._offset[0], term._scale[0]
        target_est = initial_est.copy()
        target = action_target(env, model, target_est)
        hold_alpha = None
        cue_step = None
        last_rgb = initial_rgb.copy()
        recent_estimates: list[np.ndarray] = []
        correction_steps = []
        first_touch = None
        running = longest = 0
        hand_floor_frames = table_frames = fixed_frames = both_frames = 0
        early_end = False
        pre_impulse_cube = None
        pre_impulse_contact_valid = False
        trace = []
        for step in range(1, 201):
            if case["challenge"] == "impulse" and step == 110:
                pre_impulse_cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
                pre_impulse_contact_valid = physical_touch_gate_frame(
                    env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
                assert list(env.sim.data.qvel.shape) == [1, 33]
                # 6 root + 21 actuated robot velocities precede the free cube.
                env.sim.data.qvel[0, 28] += 0.4
                env.sim.forward()
            alpha = min(1.0, step / 75) if hold_alpha is None else hold_alpha
            desired = offset + alpha * (target - offset)
            raw = torch.where(scale.abs() > 1e-8,
                              (desired - offset) / scale.clamp_min(1e-8),
                              torch.zeros_like(desired))
            action = raw.clamp(-1, 1)[None, :]
            obs, _, terminated, timed_out, _ = env.step(action)
            rgb = obs["actor"][0, -6:].detach().cpu().numpy()
            du, dv = (rgb[:2] - last_rgb[:2]).tolist()
            if hold_alpha is None and rgb[3] == 1 and rgb[5] == 0 and \
                    du <= -.0015 and dv <= -.0015:
                hold_alpha = alpha
                cue_step = step
            last_rgb = rgb.copy()
            estimate = rgb_estimate(rgb, coefficients)
            if estimate is not None:
                recent_estimates.append(estimate)
                recent_estimates = recent_estimates[-3:]
            else:
                recent_estimates.clear()
            if mode == "rgb_recenter" and step >= 75 and len(recent_estimates) == 3:
                stable = max(np.linalg.norm(v - recent_estimates[-1]) for v in recent_estimates) <= .001
                shift = recent_estimates[-1] - target_est
                if stable and np.linalg.norm(shift) >= .003 and step - (correction_steps[-1] if correction_steps else 0) >= 5:
                    new_est = initial_est + np.clip(recent_estimates[-1] - initial_est, -.010, .010)
                    if np.linalg.norm(new_est - target_est) >= .001:
                        target_est = new_est
                        target = action_target(env, model, target_est)
                        correction_steps.append(step)
            finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
            fixed = bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
            table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
            hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
            physical = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
            running = running + 1 if physical else 0
            longest = max(longest, running)
            if physical and first_touch is None:
                first_touch = step
            hand_floor_frames += int(hand_floor)
            table_frames += int(table)
            fixed_frames += int(fixed)
            both_frames += int(fixed and finger)
            trace.append([step, alpha, *rgb, du, dv, *cube,
                          int(finger), int(fixed), int(table), int(physical),
                          *target_est])
            if bool(terminated[0]) or bool(timed_out[0]):
                early_end = True
                break
        trace_array = np.asarray(trace)
        cube_path = trace_array[:, 10:13]
        out.mkdir(parents=True, exist_ok=True)
        trace_path = out / f"seed-{seed}-{mode}.npz"
        np.savez_compressed(trace_path, step_alpha_rgb_du_dv_cube_contacts_target=trace_array)
        impulse = None
        if case["challenge"] == "impulse" and pre_impulse_cube is not None:
            after = trace_array[109:, :]
            # Rows 109 onward begin at the externally perturbed control step.
            excursion = np.linalg.norm((after[:, 10:12] - pre_impulse_cube[:2]) * 1000, axis=1)
            first_lost = next((int(row[0]) for row in after if row[13] == 0), None)
            recovery_s = 0.0
            if first_lost is not None:
                remaining = [bool(row[16]) for row in trace_array if first_lost < row[0] <= first_lost + 50]
                count = best = 0
                for valid in remaining:
                    count = count + 1 if valid else 0
                    best = max(best, count)
                recovery_s = best * env.step_dt
            impulse = {"pre_cube_xyz_m": pre_impulse_cube.tolist(),
                       "pre_impulse_contact_valid": pre_impulse_contact_valid,
                       "max_post_impulse_xy_shift_mm": float(excursion.max()),
                       "first_lost_finger_step": first_lost,
                       "best_supported_recontact_within_1s_s": round(recovery_s, 3),
                       "valid_challenge": bool(pre_impulse_contact_valid and excursion.max() >= 3 and
                                               first_lost is not None),
                       "off_path_pass": bool(pre_impulse_contact_valid and excursion.max() >= 3 and first_lost is not None and
                                             recovery_s >= .5 and table and hand_floor_frames == 0 and
                                             not early_end)}
        max_drift = float(np.linalg.norm((cube_path - initial_cube) * 1000, axis=1).max())
        nominal_pass = bool(case["challenge"] == "nominal" and longest >= 50 and
                            table_frames >= 190 and table and hand_floor_frames == 0 and
                            not early_end and max_drift <= 5)
        return {"seed": seed, "cube_xy_m": list(cube_xy), "challenge": case["challenge"],
                "mode": mode, "entry_rgb_xy_error_mm": float(np.linalg.norm((initial_est - cube_xy) * 1000)),
                "cue_step": cue_step, "first_physical_touch_step": first_touch,
                "correction_steps": correction_steps,
                "longest_supported_touch_s": round(longest * env.step_dt, 3),
                "cube_peak_drift_mm": max_drift,
                "cube_net_drift_mm": float(np.linalg.norm((cube_path[-1] - initial_cube) * 1000)),
                "cube_max_center_z_m": float(cube_path[:, 2].max()),
                "table_frames": table_frames, "table_at_end": table,
                "hand_floor_frames": hand_floor_frames, "fixed_pad_frames": fixed_frames,
                "both_pad_frames": both_frames, "early_end": early_end,
                "nominal_teacher_pass": nominal_pass, "impulse": impulse,
                "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": digest(trace_path)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    assert digest(BODY) == EXPECTED_BODY_SHA
    plan = json.loads(PLAN.read_text())
    assert plan["body_sha256"] == EXPECTED_BODY_SHA
    coefficients, calibration = entry_rgb_calibration()
    records = []
    for case in plan["families"][:args.limit or None]:
        for mode in ("freeze", "rgb_recenter"):
            record = episode(case, mode, coefficients, args.out)
            records.append(record)
            print(json.dumps({k: record[k] for k in ("seed", "mode", "longest_supported_touch_s",
                                                      "cube_peak_drift_mm", "nominal_teacher_pass", "impulse")}),
                  flush=True)
            partial = {"records": records, "complete": False}
            (args.out / "partial.json").write_text(json.dumps(partial, indent=2) + "\n")
    report = {"schema": "robot-learning-lab.cove50-observed-free-object-touch-screen.v4",
              "plan": str(PLAN.relative_to(ROOT)), "plan_sha256": digest(PLAN),
              "source": str(Path(__file__).resolve().relative_to(ROOT)),
              "source_sha256": digest(Path(__file__)),
              "body_sha256": EXPECTED_BODY_SHA,
              "device": "local CPU", "control_dt_s": .02,
              "calibration": calibration,
              "episodes_complete": len(records) == 12,
              "records": records,
              "claim_limit": "Scripted one-pad native CPU development screen only. No learned actor, two-pad possession, lift, placement, independent held score or paid compute."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
