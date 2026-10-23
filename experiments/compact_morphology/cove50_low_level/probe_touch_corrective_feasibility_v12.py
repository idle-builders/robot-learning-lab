"""Replay whole V8 teacher routes after right-arm disturbances; no fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ape_walk.physics.frame50 import require_direct_frame
from collect_free_object_touch import BODY
from collect_touch_prefix_v8 import make_env
from free_object_touch_env import STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-corrective-feasibility-v12"
METHOD = OUT / "method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visit(seed: int, xy: tuple[float, float], saved: dict[str, np.ndarray],
          cutoff: int, sign: int, method: dict) -> dict:
    name = f"seed{seed}-step{cutoff}-sign{'p' if sign > 0 else 'm'}"
    env = make_env(seed, xy)
    model = require_direct_frame(BODY)
    try:
        obs, _ = env.reset(seed=seed)
        gaps = {
            "qpos": float(np.max(np.abs(env.sim.data.qpos[0].cpu().numpy() - saved["first_qpos"]))),
            "qvel": float(np.max(np.abs(env.sim.data.qvel[0].cpu().numpy() - saved["first_qvel"]))),
            "actor": float(np.max(np.abs(obs["actor"][0].cpu().numpy() - saved["first_actor"]))),
        }
        if max(gaps.values()) > 1e-6:
            raise RuntimeError(f"{name}: initial source mismatch {gaps}")
        actor, action, qpos, qvel, cube, gate, finger, table = ([] for _ in range(8))
        observation_change = None
        pre_cutoff_gap = 0.0
        first_finger = None
        longest = running = table_frames = hand_floor_frames = 0
        ended_early = False
        for step in range(1, method["episode_steps"] + 1):
            if step <= cutoff:
                pre_cutoff_gap = max(pre_cutoff_gap, float(np.max(np.abs(
                    obs["actor"][0].cpu().numpy() - saved["actor_obs"][step - 1]))))
            if step == cutoff:
                for joint_name, velocity in method["perturbation"]["joint_velocity_rad_s"].items():
                    dof = model.joint(joint_name).dofadr[0]
                    env.sim.data.qvel[0, dof] += sign * velocity
                env.sim.forward()
                # Recompute the actor-visible proprioception without advancing
                # the observation history or inserting a second control step.
                env.observation_manager._obs_buffer = None
                obs = env.observation_manager.compute(update_history=False)
                observation_change = float(np.max(np.abs(
                    obs["actor"][0].cpu().numpy() - saved["actor_obs"][step - 1])))
            actor.append(obs["actor"][0].cpu().numpy().copy())
            command = torch.as_tensor(saved["action"][step - 1][None, :])
            action.append(command[0].cpu().numpy().copy())
            obs, _, terminated, timed_out, _ = env.step(command)
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            is_finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
            is_table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
            is_hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
            if is_finger and first_finger is None:
                first_finger = step
            running = running + 1 if valid else 0
            longest = max(longest, running)
            table_frames += int(is_table)
            hand_floor_frames += int(is_hand_floor)
            qpos.append(env.sim.data.qpos[0].cpu().numpy().copy())
            qvel.append(env.sim.data.qvel[0].cpu().numpy().copy())
            cube.append(env.scene["cube"].data.root_link_pos_w[0].cpu().numpy().copy())
            gate.append(valid)
            finger.append(is_finger)
            table.append(is_table)
            if bool(terminated[0]) or bool(timed_out[0]):
                ended_early = step < method["episode_steps"]
                break
        out = OUT / f"{name}.npz"
        np.savez_compressed(out, actor_obs=np.asarray(actor), action=np.asarray(action),
                            qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            cube_xyz_m=np.asarray(cube), physical_touch=np.asarray(gate),
                            finger=np.asarray(finger), table=np.asarray(table))
        cube_xyz = np.asarray(cube)
        drift_mm = float(np.max(np.linalg.norm(
            (cube_xyz[:, :2] - cube_xyz[0, :2]) * 1000, axis=1)))
        physical = method["physical_gate"]
        basic = longest * .02 >= physical["basic_supported_touch_s_min"]
        gentle = (longest * .02 >= physical["gentle_supported_touch_s_min"]
                  and table_frames >= physical["table_contact_frames_min"]
                  and bool(table[-1]) and drift_mm <= physical["cube_peak_xy_drift_mm_max"]
                  and hand_floor_frames <= physical["hand_floor_frames_max"]
                  and not ended_early)
        return {"seed": seed, "cube_xy_m": list(xy), "stage": "precontact" if cutoff < saved["first_finger_step"] else "postcontact",
                "cutoff_step": cutoff, "sign": sign, "source_first_finger_step": int(saved["first_finger_step"]),
                "pre_cutoff_actor_max_gap": pre_cutoff_gap,
                "disturbed_actor_max_change": observation_change,
                "episode_steps": len(action), "first_finger_step": first_finger,
                "longest_supported_touch_s": longest * .02,
                "table_frames": table_frames, "table_at_end": bool(table[-1]),
                "hand_floor_frames": hand_floor_frames,
                "cube_peak_xy_drift_mm": drift_mm,
                "ended_early": ended_early,
                "basic_touch": basic, "gentle_whole_episode": gentle,
                "trace": str(out.relative_to(ROOT)), "trace_sha256": sha(out)}
    finally:
        env.close()


def main() -> None:
    method = json.loads(METHOD.read_text())
    if method["schema"] != "robot-learning-lab.touch-corrective-feasibility-v12.method.v1":
        raise RuntimeError("Wrong method")
    for entry in method["source"].values():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Source changed: {entry['path']}")
    report = json.loads((ROOT / method["source"]["v8_report"]["path"]).read_text())
    records = {r["seed"]: r for r in report["records"]}
    rows = []
    for seed in method["seeds"]:
        r = records[seed]
        path = ROOT / r["episode"]
        if sha(path) != r["episode_sha256"] or not r["accepted_gentle_whole_teacher"]:
            raise RuntimeError(f"V8 source episode invalid: {seed}")
        with np.load(path, allow_pickle=False) as data:
            saved = {k: data[k].copy() for k in
                     ("first_qpos", "first_qvel", "first_actor", "actor_obs", "action")}
        saved["first_finger_step"] = r["first_finger_step"]
        for stage, offset in (("precontact", -method["stages"]["precontact_steps_before_first_finger"]),
                              ("postcontact", method["stages"]["postcontact_steps_after_first_finger"])):
            for sign in method["perturbation"]["signs"]:
                cutoff = r["first_finger_step"] + offset
                row = visit(seed, tuple(r["cube_xy_m"]), saved, cutoff, sign, method)
                if row["stage"] != stage:
                    raise RuntimeError("Stage label differs")
                rows.append(row)
                (OUT / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(seed, stage, sign, row["gentle_whole_episode"],
                      round(row["longest_supported_touch_s"], 2),
                      round(row["cube_peak_xy_drift_mm"], 2), flush=True)
    result = {"schema": "robot-learning-lab.touch-corrective-feasibility-v12.result.v1",
              "method_sha256": sha(METHOD), "runner_sha256": sha(Path(__file__)),
              "device": "local CPU", "attempted": len(rows),
              "basic_touch": sum(r["basic_touch"] for r in rows),
              "gentle_whole_episodes": sum(r["gentle_whole_episode"] for r in rows),
              "precontact_gentle": sum(r["gentle_whole_episode"] for r in rows if r["stage"] == "precontact"),
              "postcontact_gentle": sum(r["gentle_whole_episode"] for r in rows if r["stage"] == "postcontact"),
              "gpu_hours": 0, "optimizer_updates": 0,
              "claim_limit": "CPU perturbation and saved teacher-action feasibility only; no CUDA or learned contact score, and no corrective guarantee for displaced cubes.",
              "records": rows}
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("attempted", "basic_touch", "gentle_whole_episodes",
                       "precontact_gentle", "postcontact_gentle")}))


if __name__ == "__main__":
    main()
