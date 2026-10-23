"""Local staged one-pad entry from accepted V8 precontact poses.

Creates a genuinely new zero-velocity reset at a saved near-contact pose,
then applies the remaining saved RGB teacher actions. This is a CPU entry
feasibility screen, not a learned policy or exact continuation claim.
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

from collect_free_object_touch import BODY
from free_object_touch_env import STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame, touch_env_cfg


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-prefix-v8/local-five/report.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visit(record: dict, lead_steps: int) -> dict:
    seed = record["seed"]
    source_path = ROOT / record["episode"]
    if sha(source_path) != record["episode_sha256"] or not record["accepted_gentle_whole_teacher"]:
        raise RuntimeError(f"Invalid source teacher: {seed}")
    with np.load(source_path, allow_pickle=False) as saved:
        cutoff = int(record["first_finger_step"]) - lead_steps
        start_qpos = saved["qpos"][cutoff - 1].copy()
        actions = saved["action"][cutoff:].copy()
    model = mujoco.MjModel.from_xml_path(str(BODY))
    xy = tuple(record["cube_xy_m"])
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=xy)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, seed, False
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = tuple(map(float, start_qpos[:3]))
    robot.init_state.rot = tuple(map(float, start_qpos[3:7]))
    robot.init_state.joint_pos.clear()
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        robot.init_state.joint_pos[f"^{name}$"] = float(start_qpos[model.jnt_qposadr[joint]])
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        reset_qpos = env.sim.data.qpos[0].cpu().numpy().copy()
        reset_gap = float(np.max(abs(reset_qpos[:28] - start_qpos[:28])))
        initial_finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
        initial_table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
        initial_cube_z = float(env.scene["cube"].data.root_link_pos_w[0, 2])
        rows = []
        running = longest = 0
        ended_early = False
        for step, action in enumerate(actions, 1):
            obs, _, ended, timed, _ = env.step(torch.as_tensor(action[None, :]))
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
            hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
            cube = env.scene["cube"].data.root_link_pos_w[0].cpu().numpy().copy()
            running = running + 1 if valid else 0
            longest = max(longest, running)
            rows.append({"step": step, "physical_touch": valid, "table": table,
                         "hand_floor": hand_floor, "cube_xyz_m": cube.tolist(),
                         "qpos": env.sim.data.qpos[0].cpu().numpy().tolist()})
            if bool(ended[0]) or bool(timed[0]):
                ended_early = step < len(actions)
                break
        cube = np.asarray([r["cube_xyz_m"] for r in rows])
        drift = float(np.linalg.norm((cube[:, :2] - cube[0, :2]) * 1000, axis=1).max())
        # MJLab contact sensors may be empty immediately after reset; require
        # table support at the first physical control step instead.
        gentle = bool(not initial_finger and rows[0]["table"] and not ended_early
                      and len(rows) == len(actions) and longest >= 50
                      and sum(r["table"] for r in rows) >= int(.95 * len(rows))
                      and rows[-1]["table"] and not any(r["hand_floor"] for r in rows)
                      and drift <= 5)
        return {"seed": seed, "lead_steps": lead_steps, "cutoff": cutoff,
                "reset_robot_qpos_max_gap": reset_gap,
                "new_entry_zero_velocity": True,
                "initial_finger": initial_finger, "initial_table_sensor": initial_table,
                "initial_cube_z_m": initial_cube_z,
                "first_step_table": rows[0]["table"],
                "steps": len(rows), "planned_steps": len(actions),
                "longest_supported_touch_s": longest * .02,
                "cube_peak_xy_drift_mm": drift,
                "basic_touch": longest >= 10,
                "gentle_staged_entry": gentle, "ended_early": ended_early,
                "source_episode": record["episode"],
                "source_episode_sha256": record["episode_sha256"],
                "rows": rows}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--lead-steps", type=int, default=20)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    rows = []
    for record in source["records"][:args.limit]:
        row = visit(record, args.lead_steps)
        rows.append(row)
        print(row["seed"], row["steps"], row["basic_touch"], row["gentle_staged_entry"],
              round(row["cube_peak_xy_drift_mm"], 2), flush=True)
    result = {"schema": "robot-learning-lab.touch-near-entry-v13-local.v1",
              "claim_limit": "New zero-velocity near-contact CPU resets from accepted V8 poses with saved future actions; not learned, CUDA, full open-to-contact or corrective-data proof",
              "body_sha256": sha(BODY), "source_report_sha256": sha(SOURCE),
              "runner_sha256": sha(Path(__file__)), "attempted": len(rows),
              "basic_touch": sum(r["basic_touch"] for r in rows),
              "gentle_staged_entries": sum(r["gentle_staged_entry"] for r in rows),
              "gpu_hours": 0, "optimizer_updates": 0, "records": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
