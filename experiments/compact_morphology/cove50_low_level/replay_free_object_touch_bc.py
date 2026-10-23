"""Fresh native closed-loop diagnostic for the small free-object touch BC pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from mjlab.envs import ManagerBasedRlEnv

from free_object_touch_env import (ROOT, STAND_TABLE_CUBE_XYZ,
                                   physical_touch_gate_frame, touch_env_cfg)

BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zero-fixed-actions", action="store_true",
                        help="Force the 16 zero-scale action channels to exact zero before action-history feedback")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert saved["body_sha256"] == digest(BODY)
    input_indices = saved.get("input_indices", list(range(69)))
    action_indices = saved.get("action_indices", list(range(21)))
    model = nn.Sequential(nn.Linear(len(input_indices), 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(),
                          nn.Linear(128, len(action_indices)), nn.Tanh())
    model.load_state_dict(saved["state_dict"])
    model.eval()
    mean = torch.tensor(saved["obs_mean"], dtype=torch.float32)
    std = torch.tensor(saved["obs_std"], dtype=torch.float32)
    records = []
    # Fresh seeds and cube placements; neither trajectory is in the teacher
    # ledger. This is a development diagnostic, not a held independent score.
    for seed, cube_xy in ((31001, (.049, -.141)), (31002, (.052, -.138))):
        cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
        cfg.scene.num_envs = 1
        cfg.seed = seed
        cfg.auto_reset = False
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        try:
            obs, _ = env.reset(seed=seed)
            first_rgb = obs["actor"][0, -6:].detach().cpu().tolist()
            initial_contact = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any() or
                                   env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
            longest = running = 0
            table_frames = hand_floor_frames = 0
            min_pelvis = float("inf")
            terminated_early = False
            trace = []
            actor_trace = []
            action_trace = []
            hand_cube_distance_trace = []
            hand_site_index = env.scene["robot"].find_sites("right_hand_center")[0][0]
            with torch.no_grad():
                for step in range(200):
                    features = (obs["actor"][0, input_indices] - mean) / std
                    predicted = model(features)
                    action = torch.zeros((1, 21), dtype=predicted.dtype, device=predicted.device)
                    action[0, action_indices] = predicted
                    if args.zero_fixed_actions:
                        action[:, :16] = 0.0
                    actor_trace.append(obs["actor"][0].detach().cpu().numpy().copy())
                    action_trace.append(action[0].detach().cpu().numpy().copy())
                    obs, _, terminated, timed_out, _ = env.step(action)
                    valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
                    running = running + 1 if valid else 0
                    longest = max(longest, running)
                    table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
                    hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
                    table_frames += table
                    hand_floor_frames += hand_floor
                    min_pelvis = min(min_pelvis, float(env.scene["robot"].data.root_link_pos_w[0, 2]))
                    hand = env.scene["robot"].data.site_pos_w[0, hand_site_index]
                    cube = env.scene["cube"].data.root_link_pos_w[0]
                    hand_cube_distance_trace.append(float(torch.linalg.norm(hand - cube)))
                    trace.append((int(valid), int(table), int(hand_floor),
                                  float(env.scene["cube"].data.root_link_pos_w[0, 0]),
                                  float(env.scene["cube"].data.root_link_pos_w[0, 1]),
                                  float(env.scene["cube"].data.root_link_pos_w[0, 2])))
                    if bool(terminated[0]) or bool(timed_out[0]):
                        terminated_early = True
                        break
            trace_path = args.out.parent / f"{args.out.stem}-seed-{seed}-trace.npz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path,
                                physical_trace=np.asarray(trace, dtype=np.float32),
                                actor_observation=np.asarray(actor_trace, dtype=np.float32),
                                actor_action=np.asarray(action_trace, dtype=np.float32),
                                hand_cube_distance_m=np.asarray(hand_cube_distance_trace, dtype=np.float32))
            records.append({
                "seed": seed, "cube_xy_m": list(cube_xy),
                "first_rgb_target": first_rgb, "initial_pad_contact": initial_contact,
                "steps": len(trace), "longest_supported_touch_s": round(longest * env.step_dt, 3),
                "physical_200ms_touch_pass": bool(longest * env.step_dt >= .2 and
                                                  not terminated_early and not initial_contact),
                "table_contact_frames": table_frames,
                "hand_floor_frames": hand_floor_frames,
                "min_pelvis_z_m": round(min_pelvis, 5),
                "min_hand_cube_distance_m": round(min(hand_cube_distance_trace), 6),
                "terminated_early": terminated_early,
                "trace": str(trace_path.resolve().relative_to(ROOT)),
                "trace_sha256": digest(trace_path),
            })
        finally:
            env.close()
    report = {
        "schema": "robot-learning-lab.cove50-free-object-touch-bc-fresh-replay.v1",
        "body_sha256": digest(BODY),
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_sha256": digest(args.checkpoint),
        "teacher_ledger_sha256": saved["teacher_ledger_sha256"],
        "replay_source_sha256": digest(Path(__file__)),
        "actor_contract": saved.get("actor_contract", "full-history"),
        "zero_fixed_actions": args.zero_fixed_actions,
        "device": "cpu", "control_dt_s": .02,
        "records": records,
        "passes": sum(r["physical_200ms_touch_pass"] for r in records),
        "claim_limit": "Two local CPU fresh native development starts only. Not CUDA or independent held score, not robust learned touch, no opposing-pad possession/lift."
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passes": report["passes"], "records": records}, indent=2))


if __name__ == "__main__":
    main()
