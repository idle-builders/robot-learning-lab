"""Collect small, source-bound native teacher episodes for one-pad table touch.

The teacher uses cube truth to author an IK target. The saved actor input does
not: it is the environment's proprioception, previous action and RGB target.
Accepted episodes are imitation examples, not evidence of a learned policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from free_object_touch_env import (ROOT, STAND_TABLE_CUBE_XYZ,
                                   physical_touch_gate_frame, touch_env_cfg)
from ape_walk.physics.frame50 import require_direct_frame

BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
EXPECTED_BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
NOMINAL_TARGET = {
    "right_shoulder_yaw": -.24237,
    "right_shoulder_roll": .11798,
    "right_shoulder_pitch": .22665,
    "right_elbow_pitch": -.74521,
    "right_gripper": .01164,
}
ARM_NAMES = tuple(NOMINAL_TARGET)[:4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapted_target(model: mujoco.MjModel, cube_xy: tuple[float, float]) -> dict[str, float]:
    """Locally linearize the exact body's moving-pad position about the seed teacher."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[2] = .2305
    data.qpos[model.joint("head_yaw").qposadr[0]] = -1.25
    for name, value in NOMINAL_TARGET.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    geom_id = model.geom("right_finger_tip").id
    baseline_xyz = data.geom_xpos[geom_id].copy()
    qvel_ids = [model.joint(name).dofadr[0] for name in ARM_NAMES]
    desired_shift = np.array([cube_xy[0] - STAND_TABLE_CUBE_XYZ[0],
                              cube_xy[1] - STAND_TABLE_CUBE_XYZ[1], 0.0])
    target = dict(NOMINAL_TARGET)
    for _ in range(3):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, jacp, jacr, geom_id)
        jac = jacp[:, qvel_ids]
        residual = baseline_xyz + desired_shift - data.geom_xpos[geom_id]
        step = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(len(qvel_ids)), jac.T @ residual)
        for name, delta in zip(ARM_NAMES, step):
            joint = model.joint(name)
            value = float(np.clip(data.qpos[joint.qposadr[0]] + delta,
                                  joint.range[0], joint.range[1]))
            data.qpos[joint.qposadr[0]] = value
            target[name] = value
        mujoco.mj_forward(model, data)
    return target


def episode(body: Path, cube_xy: tuple[float, float], seed: int,
            ramp_steps: int, out: Path, split: str,
            freeze_on_touch: bool = True) -> dict:
    cfg = touch_env_cfg(body, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        assert env.observation_manager.active_terms["actor"] == [
            "joint_pos", "joint_vel", "actions", "rgb_target"]
        assert list(obs["actor"].shape) == [1, 69]
        entry_rgb = obs["actor"][0, -6:].detach().cpu().numpy()
        cube_start = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy()
        expected = np.array([*cube_xy, STAND_TABLE_CUBE_XYZ[2]])
        reset_gap = float(np.max(np.abs(cube_start - expected)))
        initial_contact = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any() or
                               env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
        model = require_direct_frame(body)
        ik = adapted_target(model, cube_xy)
        term = env.action_manager.get_term("joint_pos")
        names = term._target_names
        offset, scale = term._offset[0], term._scale[0]
        target = offset.clone()
        for name, value in ik.items():
            target[names.index(name)] = value
        actor, actions, gates, cube_positions = [], [], [], []
        hold_alpha = None
        running = longest = 0
        first_touch_s = None
        hand_floor_frames = table_frames = clipped_frames = 0
        min_pelvis = float("inf")
        terminated_early = False
        for step in range(200):
            actor.append(obs["actor"][0].detach().cpu().numpy().copy())
            alpha = min(1.0, (step + 1) / ramp_steps) if hold_alpha is None else hold_alpha
            desired = offset + alpha * (target - offset)
            raw = torch.where(scale.abs() > 1e-8,
                              (desired - offset) / scale.clamp_min(1e-8),
                              torch.zeros_like(desired))
            clipped_frames += int(bool((raw.abs() > 1.00001).any()))
            action = raw.clamp(-1, 1)[None, :]
            actions.append(action[0].detach().cpu().numpy().copy())
            obs, _, terminated, timed_out, _ = env.step(action)
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            gates.append(valid)
            cube_positions.append(env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy())
            if valid and first_touch_s is None:
                first_touch_s = round((step + 1) * env.step_dt, 3)
                if freeze_on_touch:
                    hold_alpha = alpha
            running = running + 1 if valid else 0
            longest = max(longest, running)
            hand_floor_frames += int(bool(env.scene.sensors["hand_ground_contact"].data.found[0].any()))
            table_frames += int(bool(env.scene.sensors["cube_table_contact"].data.found[0].any()))
            min_pelvis = min(min_pelvis, float(env.scene["robot"].data.root_link_pos_w[0, 2]))
            if bool(terminated[0]) or bool(timed_out[0]):
                terminated_early = True
                break
        valid_entry = bool(not initial_contact and reset_gap < 1e-4 and
                           entry_rgb[3] == 1 and entry_rgb[5] == 0 and entry_rgb[2] > 0)
        touch_gate_pass = bool(valid_entry and longest * env.step_dt >= .2 and not terminated_early)
        table_at_end = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
        accepted = bool(touch_gate_pass and longest * env.step_dt >= 1.0 and
                        table_frames >= 190 and table_at_end and
                        hand_floor_frames == 0)
        reasons = []
        if not valid_entry:
            reasons.append("invalid_entry_or_rgb")
        if longest * env.step_dt < .2:
            reasons.append("no_200ms_supported_finger_contact")
        if terminated_early:
            reasons.append("early_termination")
        if longest * env.step_dt < 1.0:
            reasons.append("teacher_hold_under_1s")
        if table_frames < 190 or not table_at_end:
            reasons.append("lost_table_support")
        if hand_floor_frames:
            reasons.append("hand_floor_contact")
        episode_path = (out / f"seed-{seed}-{split}.npz").resolve()
        np.savez_compressed(episode_path, actor_obs=np.asarray(actor, dtype=np.float32),
                            teacher_action=np.asarray(actions, dtype=np.float32),
                            physical_touch=np.asarray(gates, dtype=np.bool_),
                            cube_xyz_m=np.asarray(cube_positions, dtype=np.float32))
        return {
            "seed": seed, "family": f"cube-x{cube_xy[0]:.3f}-y{cube_xy[1]:.3f}",
            "split": split, "cube_xy_m": list(cube_xy), "ramp_steps": ramp_steps,
            "freeze_on_touch": freeze_on_touch,
            "ik_target_rad": ik, "episode": str(episode_path.relative_to(ROOT)),
            "episode_sha256": digest(episode_path), "accepted": accepted,
            "touch_gate_pass": touch_gate_pass,
            "reject_reasons": reasons, "steps": len(actor),
            "entry_rgb_target": entry_rgb.tolist(), "entry_cube_gap_m": reset_gap,
            "initial_pad_contact": initial_contact,
            "first_touch_s": first_touch_s,
            "longest_supported_touch_s": round(longest * env.step_dt, 3),
            "hand_floor_frames": hand_floor_frames, "cube_table_frames": table_frames,
            "cube_table_at_end": table_at_end,
            "min_pelvis_z_m": round(min_pelvis, 5),
            "clipped_action_frames": clipped_frames,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    assert digest(BODY) == EXPECTED_BODY_SHA
    args.out.mkdir(parents=True, exist_ok=True)
    # Each cell is an independent physical reset family. The outer cube
    # placements are validation only; adjacent frames never cross splits.
    grid = [(x, y) for x in (.046, .050, .054)
            for y in (-.144, -.140, -.136)]
    records = []
    for index, cube_xy in enumerate(grid[:args.limit or None]):
        split = "validation" if index in (0, 2, 6, 8) else "train"
        records.append(episode(BODY, cube_xy, 30001 + index,
                               75, args.out, split))
        print(json.dumps({k: records[-1][k] for k in
                          ("seed", "split", "accepted", "longest_supported_touch_s", "reject_reasons")}),
              flush=True)
    report = {
        "schema": "robot-learning-lab.cove50-free-object-touch-teacher-ledger.v2",
        "body": str(BODY.relative_to(ROOT)), "body_sha256": digest(BODY),
        "collector": str(Path(__file__).resolve().relative_to(ROOT)),
        "collector_sha256": digest(Path(__file__)),
        "source_success": "artifacts/ape-low-level-training-20261013/free-object-native-stand-tabletop-ik-probe.json",
        "source_success_sha256": digest(ROOT / "artifacts/ape-low-level-training-20261013/free-object-native-stand-tabletop-ik-probe.json"),
        "control_dt_s": .02, "device": "cpu",
        "object": "free 20 mm, 6.25 g cube on 8 cm fixed table; outside both pads at reset",
        "variation": {"cube_x_m": [.046, .050, .054], "cube_y_m": [-.144, -.140, -.136],
                      "ramp_steps": [75], "source": "authored local 4 mm placement sensitivity around one successful seed; not physical calibration"},
        "actor": "69D: joint position/velocity, previous action, six RGB target/status values; no object truth or privileged contact",
        "teacher": "exact-body offline pad Jacobian adapts original IK target using true start cube coordinates; contact evaluator freezes ramp on first valid touch",
        "gate": "200 ms continuous right-finger/cube contact with table-supported free cube, both soles loaded, no hand-floor, pelvis >=100 mm",
        "imitation_acceptance": "entry valid; physical touch gate; >=1 s continuous valid contact; >=190/200 table-contact frames and table support at final frame; zero hand-floor frames; no early termination",
        "touch_gate_pass": sum(r["touch_gate_pass"] for r in records),
        "accepted": sum(r["accepted"] for r in records),
        "rejected": sum(not r["accepted"] for r in records),
        "train_accepted": sum(r["accepted"] and r["split"] == "train" for r in records),
        "validation_accepted": sum(r["accepted"] and r["split"] == "validation" for r in records),
        "records": records,
        "claim_limit": "CPU native scripted teacher data only. No learned policy, no CUDA score, no two-pad possession or lift; teacher sees cube truth unavailable to actor."
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": report["accepted"], "rejected": report["rejected"],
                      "report": str(args.out / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
