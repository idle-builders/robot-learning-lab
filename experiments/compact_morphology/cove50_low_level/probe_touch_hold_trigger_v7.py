"""Local diagnostic: does an early contact hold rescue V6 teacher push-offs?

The cube-specific contact trigger is a simulator-truth upper bound. It is not
deployable actor input or an accepted new imitation teacher by itself.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from ape_walk.physics.frame50 import require_direct_frame
from collect_free_object_touch import BODY
from free_object_touch_env import ROOT, STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame, touch_env_cfg
from probe_observed_free_object_touch import entry_rgb_calibration
from screen_observed_free_object_touch_v4 import action_target, rgb_estimate


def run(case, mode, coefficients, *, alpha_cap=1.0, ramp_steps=75):
    seed, xy = case["seed"], tuple(case["cube_xy_m"])
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=xy)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, seed, False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        first_rgb = obs["actor"][0, -6:].detach().cpu().numpy()
        estimate = rgb_estimate(first_rgb, coefficients)
        if estimate is None:
            raise ValueError("entry RGB was invalid")
        target = action_target(env, require_direct_frame(BODY), estimate)
        term = env.action_manager.get_term("joint_pos")
        offset, scale = term._offset[0], term._scale[0]
        last_rgb = first_rgb.copy()
        first_contact = hold_step = None
        hold_alpha = None
        longest = running = 0
        table_frames = hand_floor_frames = 0
        initial_cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
        rows = []
        for step in range(1, 201):
            alpha = min(alpha_cap, step / ramp_steps) if hold_alpha is None else hold_alpha
            desired = offset + alpha * (target - offset)
            raw = torch.where(scale.abs() > 1e-8,
                              (desired - offset) / scale.clamp_min(1e-8),
                              torch.zeros_like(desired))
            action = raw.clamp(-1, 1)[None, :]
            obs, _, terminated, timed_out, _ = env.step(action)
            rgb = obs["actor"][0, -6:].detach().cpu().numpy()
            finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
            table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
            hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
            if finger and first_contact is None:
                first_contact = step
            du, dv = rgb[:2] - last_rgb[:2]
            rgb_cue = bool(rgb[3] == 1 and rgb[5] == 0 and du <= -.0015 and dv <= -.0015)
            if hold_alpha is None and (finger if mode == "contact_truth" else rgb_cue):
                hold_alpha, hold_step = alpha, step
            last_rgb = rgb.copy()
            physical = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            running = running + 1 if physical else 0
            longest = max(longest, running)
            table_frames += int(table)
            hand_floor_frames += int(hand_floor)
            cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
            rows.append({"step": step, "alpha": float(alpha), "finger": finger,
                         "physical_touch": physical, "table": table,
                         "cube_xy_m": cube[:2].tolist(), "rgb_cue": rgb_cue})
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        drift = max(float(np.linalg.norm(np.asarray(r["cube_xy_m"]) - initial_cube[:2]))
                    for r in rows) * 1000
        return {"seed": seed, "cube_xy_m": list(xy), "mode": mode,
                "alpha_cap": alpha_cap, "ramp_steps": ramp_steps,
                "first_finger_step": first_contact, "hold_step": hold_step,
                "hold_alpha": hold_alpha, "steps": len(rows),
                "longest_supported_touch_s": round(longest * .02, 3),
                "table_frames": table_frames, "table_at_end": rows[-1]["table"],
                "hand_floor_frames": hand_floor_frames,
                "cube_peak_xy_drift_mm": round(drift, 3),
                "full_teacher_gate": bool(len(rows) == 200 and longest >= 50
                                          and table_frames >= 190 and rows[-1]["table"]
                                          and hand_floor_frames == 0),
                "timeseries": rows}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[33003, 33009, 33012])
    parser.add_argument("--target-grid", action="store_true")
    parser.add_argument("--caps", type=float, nargs="+", default=[0.8, 0.9, 1.0])
    args = parser.parse_args()
    source = ROOT / "artifacts/ape-low-level-training-20261015/free-object-touch-bc-v6/teacher/report.json"
    ledger = json.loads(source.read_text())
    cases = {r["seed"]: r for r in ledger["records"]}
    coefficients, calibration = entry_rgb_calibration()
    if args.target_grid:
        results = [run(cases[seed], "rgb_cue", coefficients, alpha_cap=cap)
                   for seed in args.seeds for cap in args.caps]
    else:
        results = [run(cases[seed], mode, coefficients)
                   for seed in args.seeds for mode in ("rgb_cue", "contact_truth")]
    report = {"schema": "robot-learning-lab.touch-hold-trigger-diagnostic.v1",
              "claim_limit": "local CPU simulator-truth trigger upper bound; no new policy, CUDA score or deployable tactile channel",
              "body_sha256": hashlib.sha256(BODY.read_bytes()).hexdigest(),
              "source_ledger_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "calibration": calibration, "results": results,
              "gpu_hours": 0, "optimizer_updates": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print({key: row[key] for key in ("seed", "mode", "first_finger_step", "hold_step",
                                        "alpha_cap", "longest_supported_touch_s", "cube_peak_xy_drift_mm",
                                        "full_teacher_gate")}, flush=True)


if __name__ == "__main__":
    main()
