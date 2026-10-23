"""Fresh paired native CPU replay of V6 BC and observed-touch baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from mjlab.envs import ManagerBasedRlEnv

from ape_walk.physics.frame50 import require_direct_frame
from collect_free_object_touch import BODY, EXPECTED_BODY_SHA
from fit_touch_pad_relative_v6 import INPUT, OUTPUT
from free_object_touch_env import (ROOT, STAND_TABLE_CUBE_XYZ,
                                   physical_touch_gate_frame, touch_env_cfg)
from probe_observed_free_object_touch import entry_rgb_calibration
from screen_observed_free_object_touch_v4 import action_target, rgb_estimate
from touch_pad_relative_v6 import PadRelativeRgb

PLAN = ROOT / "artifacts/ape-low-level-training-20261015/free-object-touch-bc-v6/plan.json"
OLD = ROOT / "artifacts/ape-low-level-training-20261013/free-object-touch-bc-pilot/v2/actor.pt"
LEDGER = ROOT / "artifacts/ape-low-level-training-20261015/free-object-touch-bc-v6/teacher/report.json"
SCORER = ROOT / "artifacts/ape-low-level-training-20261015/free-object-touch-bc-v6/scorer-amendment.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loaded_actor(path: Path) -> tuple[nn.Module, torch.Tensor, torch.Tensor, list[int], list[int]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["body_sha256"] == EXPECTED_BODY_SHA
    if path != OLD:
        assert saved["input_indices"] == INPUT and saved["action_indices"] == OUTPUT
        assert saved["teacher_ledger_sha256"] == sha(LEDGER)
        assert saved["epochs"] == 60
    inputs = saved["input_indices"]
    outputs = saved["action_indices"]
    model = nn.Sequential(nn.Linear(len(inputs), 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(),
                          nn.Linear(128, len(outputs)), nn.Tanh())
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return (model, torch.as_tensor(saved["obs_mean"], dtype=torch.float32),
            torch.as_tensor(saved["obs_std"], dtype=torch.float32), inputs, outputs)


def episode(case: dict, mode: str, checkpoint: Path, coefficients: np.ndarray,
            out: Path) -> dict:
    seed, cube_xy = case["seed"], tuple(case["cube_xy_m"])
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        first_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
        first_rgb = obs["actor"][0, -6:].detach().cpu().numpy().copy()
        initial_est = rgb_estimate(first_rgb, coefficients)
        initial_contact = bool(
            env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any() or
            env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
        assert initial_est is not None and not initial_contact
        adapter = PadRelativeRgb(env)
        if mode in ("candidate", "old-v2"):
            model, mean, std, indices, outputs = loaded_actor(checkpoint if mode == "candidate" else OLD)
        if mode == "teacher":
            target = action_target(env, require_direct_frame(BODY), initial_est)
            term = env.action_manager.get_term("joint_pos")
            offset, scale = term._offset[0], term._scale[0]
            hold_alpha = None
            last_rgb = first_rgb.copy()
        frames = [env.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy()]
        actions, qpos, qvel, actor, contact, cube = [], [], [], [], [], []
        running = longest = table_frames = hand_floor_frames = fixed_frames = 0
        early_end = False
        cue_step = None
        for step in range(1, 201):
            augmented = adapter.augmented(env, obs["actor"][0])
            actor.append(augmented)
            if mode == "teacher":
                alpha = min(1.0, step / 75) if hold_alpha is None else hold_alpha
                desired = offset + alpha * (target - offset)
                raw = torch.where(scale.abs() > 1e-8,
                                  (desired - offset) / scale.clamp_min(1e-8),
                                  torch.zeros_like(desired))
                action = raw.clamp(-1, 1)[None, :]
            elif mode == "zero":
                action = torch.zeros((1, 21), dtype=torch.float32)
            else:
                with torch.no_grad():
                    feature = (torch.as_tensor(augmented[indices]) - mean) / std
                    predicted = model(feature)
                action = torch.zeros((1, 21), dtype=torch.float32)
                action[0, outputs] = predicted
            actions.append(action[0].detach().cpu().numpy().copy())
            obs, _, terminated, timed_out, _ = env.step(action)
            if mode == "teacher":
                rgb = obs["actor"][0, -6:].detach().cpu().numpy()
                du, dv = rgb[:2] - last_rgb[:2]
                if hold_alpha is None and rgb[3] == 1 and rgb[5] == 0 and \
                        du <= -.0015 and dv <= -.0015:
                    hold_alpha = alpha
                    cue_step = step
                last_rgb = rgb.copy()
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            contact.append(valid)
            running = running + 1 if valid else 0
            longest = max(longest, running)
            table_frames += int(bool(env.scene.sensors["cube_table_contact"].data.found[0].any()))
            hand_floor_frames += int(bool(env.scene.sensors["hand_ground_contact"].data.found[0].any()))
            fixed_frames += int(bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any()))
            qpos.append(env.sim.data.qpos[0].detach().cpu().numpy().copy())
            qvel.append(env.sim.data.qvel[0].detach().cpu().numpy().copy())
            cube.append(env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy())
            frames.append(env.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy())
            if bool(terminated[0]) or bool(timed_out[0]):
                early_end = True
                break
        table_end = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
        out.mkdir(parents=True, exist_ok=True)
        stem = f"seed-{seed}-{mode}"
        trace = out / f"{stem}.npz"
        video = out / f"{stem}.mp4"
        np.savez_compressed(trace, first_qpos=first_qpos, actor_obs=np.asarray(actor),
                            actions=np.asarray(actions), qpos=np.asarray(qpos),
                            qvel=np.asarray(qvel), physical_touch=np.asarray(contact),
                            cube_xyz_m=np.asarray(cube))
        imageio.mimwrite(video, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        loose_touch = bool(longest >= 10 and table_end and hand_floor_frames == 0 and not early_end)
        # A one-frame table sensor gap occurs in six of seven accepted teacher
        # demonstrations. Match the pre-existing teacher support threshold.
        task_pass = bool(loose_touch and table_frames >= 190)
        strict_pass = bool(task_pass and longest >= 50)
        return {"seed": seed, "cube_xy_m": list(cube_xy), "mode": mode,
                "first_qpos_sha256": hashlib.sha256(first_qpos.tobytes()).hexdigest(),
                "steps": len(actions), "video_frames": len(frames),
                "longest_supported_touch_s": round(longest * env.step_dt, 3),
                "loose_touch": loose_touch, "task_pass": task_pass,
                "strict_1s_pass": strict_pass,
                "table_frames": table_frames, "table_at_end": table_end,
                "hand_floor_frames": hand_floor_frames, "fixed_pad_frames": fixed_frames,
                "early_end": early_end, "image_hold_cue_step": cue_step,
                "cube_peak_xy_drift_mm": float(np.linalg.norm((np.asarray(cube)[:, :2] - cube[0][:2]) * 1000, axis=1).max()),
                "trace": str(trace.relative_to(ROOT)), "trace_sha256": sha(trace),
                "video": str(video.relative_to(ROOT)), "video_sha256": sha(video)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.checkpoint = args.checkpoint.resolve()
    assert sha(BODY) == EXPECTED_BODY_SHA
    assert args.checkpoint != OLD
    candidate_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert candidate_payload["body_sha256"] == EXPECTED_BODY_SHA
    assert candidate_payload["input_indices"] == INPUT
    assert candidate_payload["action_indices"] == OUTPUT
    assert candidate_payload["teacher_ledger_sha256"] == sha(LEDGER)
    assert candidate_payload["epochs"] == 60
    plan = json.loads(PLAN.read_text())
    assert plan["body_sha256"] == EXPECTED_BODY_SHA
    coefficients, calibration = entry_rgb_calibration()
    records = []
    for case in plan["fresh_development_families"]:
        for mode in ("teacher", "old-v2", "zero", "candidate"):
            row = episode(case, mode, args.checkpoint, coefficients, args.out)
            records.append(row)
            (args.out / "partial.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in
                              ("seed", "mode", "steps", "longest_supported_touch_s", "task_pass")}), flush=True)
    identity = {str(case["seed"]): len({r["first_qpos_sha256"] for r in records
                                        if r["seed"] == case["seed"]}) == 1
                for case in plan["fresh_development_families"]}
    assert all(identity.values()), "Matched physical reset identity failed"
    report = {"schema": "robot-learning-lab.cove50-observed-touch-bc-development.v6",
              "claim_limit": "Four local CPU fresh matched families, not CUDA qualification, two-pad possession or lift.",
              "source_sha256": sha(Path(__file__)), "plan_sha256": sha(PLAN),
              "scorer_amendment_sha256": sha(SCORER),
              "body_sha256": sha(BODY), "checkpoint_sha256": sha(args.checkpoint),
              "old_checkpoint_sha256": sha(OLD), "calibration": calibration,
              "matched_initial_state": identity,
              "visits": len(records),
              "task_passes_by_mode": {mode: sum(r["task_pass"] for r in records if r["mode"] == mode)
                                      for mode in ("teacher", "old-v2", "zero", "candidate")},
              "candidate_task_passes": sum(r["task_pass"] for r in records if r["mode"] == "candidate"),
              "records": records}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
