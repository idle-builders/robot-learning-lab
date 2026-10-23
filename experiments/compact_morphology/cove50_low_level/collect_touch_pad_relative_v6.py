"""Collect whole observed-RGB one-pad teacher episodes for one small BC pilot.

The target and contact-hold cue are image-derived. Truth contact, cube pose and
support are evaluator-only. Rejected episodes remain in the source ledger.
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
from collect_free_object_touch import BODY, EXPECTED_BODY_SHA
from free_object_touch_env import (ROOT, STAND_TABLE_CUBE_XYZ,
                                   physical_touch_gate_frame, touch_env_cfg)
from probe_observed_free_object_touch import entry_rgb_calibration
from screen_observed_free_object_touch_v4 import action_target, rgb_estimate
from touch_pad_relative_v6 import PadRelativeRgb

PLAN = ROOT / "artifacts/ape-low-level-training-20261015/free-object-touch-bc-v6/plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(case: dict, coefficients: np.ndarray, out: Path) -> dict:
    seed = case["seed"]
    cube_xy = tuple(case["cube_xy_m"])
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        initial_rgb = obs["actor"][0, -6:].detach().cpu().numpy()
        initial_xy = rgb_estimate(initial_rgb, coefficients)
        initial_pad_contact = bool(
            env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any() or
            env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
        valid_entry = bool(initial_xy is not None and not initial_pad_contact)
        if not valid_entry:
            return {"seed": seed, "cube_xy_m": list(cube_xy), "split": case["split"],
                    "accepted": False, "reject_reasons": ["missing_entry_rgb_or_initial_pad_contact"]}
        source_model = require_direct_frame(BODY)
        target = action_target(env, source_model, initial_xy)
        term = env.action_manager.get_term("joint_pos")
        offset, scale = term._offset[0], term._scale[0]
        adapter = PadRelativeRgb(env)
        last_rgb = initial_rgb.copy()
        hold_alpha = None
        cue_step = first_touch = None
        running = longest = 0
        table_frames = hand_floor_frames = fixed_frames = 0
        early_end = False
        actor, actions, physical, cube_path = [], [], [], []
        for step in range(1, 201):
            actor.append(adapter.augmented(env, obs["actor"][0]))
            alpha = min(1.0, step / 75) if hold_alpha is None else hold_alpha
            desired = offset + alpha * (target - offset)
            raw = torch.where(scale.abs() > 1e-8,
                              (desired - offset) / scale.clamp_min(1e-8),
                              torch.zeros_like(desired))
            action = raw.clamp(-1, 1)[None, :]
            actions.append(action[0].detach().cpu().numpy().copy())
            obs, _, terminated, timed_out, _ = env.step(action)
            rgb = obs["actor"][0, -6:].detach().cpu().numpy()
            du, dv = rgb[:2] - last_rgb[:2]
            if hold_alpha is None and rgb[3] == 1 and rgb[5] == 0 and \
                    du <= -.0015 and dv <= -.0015:
                hold_alpha = alpha
                cue_step = step
            last_rgb = rgb.copy()
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            physical.append(valid)
            if valid and first_touch is None:
                first_touch = step
            running = running + 1 if valid else 0
            longest = max(longest, running)
            table_frames += int(bool(env.scene.sensors["cube_table_contact"].data.found[0].any()))
            hand_floor_frames += int(bool(env.scene.sensors["hand_ground_contact"].data.found[0].any()))
            fixed_frames += int(bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any()))
            cube_path.append(env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy())
            if bool(terminated[0]) or bool(timed_out[0]):
                early_end = True
                break
        table_end = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
        task_touch = bool(longest >= 10 and not early_end and table_end and hand_floor_frames == 0)
        accepted = bool(task_touch and longest >= 50 and table_frames >= 190)
        reasons = []
        if longest < 10:
            reasons.append("under_200ms_supported_contact")
        if longest < 50:
            reasons.append("under_1s_teacher_contact")
        if table_frames < 190 or not table_end:
            reasons.append("lost_table_support")
        if hand_floor_frames:
            reasons.append("hand_floor_contact")
        if early_end:
            reasons.append("early_termination")
        cube = np.asarray(cube_path)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"seed-{seed}-{case['split']}.npz"
        np.savez_compressed(path, actor_obs=np.asarray(actor, dtype=np.float32),
                            teacher_action=np.asarray(actions, dtype=np.float32),
                            physical_touch=np.asarray(physical, dtype=bool), cube_xyz_m=cube)
        return {"seed": seed, "cube_xy_m": list(cube_xy), "split": case["split"],
                "accepted": accepted, "task_touch_pass": task_touch,
                "reject_reasons": reasons, "steps": len(actor),
                "entry_estimation_error_mm": float(np.linalg.norm((initial_xy - cube_xy) * 1000)),
                "first_touch_step": first_touch, "image_hold_cue_step": cue_step,
                "longest_supported_touch_s": round(longest * env.step_dt, 3),
                "table_frames": table_frames, "table_at_end": table_end,
                "hand_floor_frames": hand_floor_frames, "fixed_pad_frames": fixed_frames,
                "cube_peak_xy_drift_mm": float(np.linalg.norm((cube[:, :2] - cube[0, :2]) * 1000, axis=1).max()),
                "episode": str(path.relative_to(ROOT)), "episode_sha256": sha(path)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    assert sha(BODY) == EXPECTED_BODY_SHA
    plan = json.loads(PLAN.read_text())
    assert plan["body_sha256"] == EXPECTED_BODY_SHA
    coefficients, calibration = entry_rgb_calibration()
    records = []
    args.out.mkdir(parents=True, exist_ok=True)
    for case in plan["teacher_families"][:args.limit or None]:
        row = episode(case, coefficients, args.out)
        records.append(row)
        (args.out / "partial.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
        print(json.dumps({k: row.get(k) for k in
                          ("seed", "split", "accepted", "task_touch_pass",
                           "longest_supported_touch_s", "reject_reasons")}), flush=True)
    report = {"schema": "robot-learning-lab.cove50-observed-touch-teacher.v6",
              "claim_limit": "Local CPU scripted observed-RGB teacher data, not a learned policy or CUDA score.",
              "body_sha256": sha(BODY), "plan_sha256": sha(PLAN),
              "collector_source_sha256": sha(Path(__file__)),
              "adapter_source_sha256": sha(Path(__file__).with_name("touch_pad_relative_v6.py")),
              "calibration": calibration, "device": "local CPU", "control_dt_s": .02,
              "complete": len(records) == len(plan["teacher_families"]),
              "attempted": len(records), "accepted": sum(r["accepted"] for r in records),
              "task_touch_passes": sum(r.get("task_touch_pass", False) for r in records),
              "records": records}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
