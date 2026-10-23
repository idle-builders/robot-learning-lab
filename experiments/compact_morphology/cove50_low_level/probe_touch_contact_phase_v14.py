"""One CPU safety screen of staged V14 authority against V8 source routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from collect_free_object_touch import BODY
from free_object_touch_env import STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame
from touch_reference_residual_v14 import residual_touch_env_cfg


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-prefix-v8/local-five/report.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visit(record: dict, mode: str, gain: float) -> dict:
    seed = int(record["seed"])
    cfg = residual_touch_env_cfg(cube_xy=tuple(record["cube_xy_m"]))
    cfg.actions["joint_pos"].residual_gain = gain
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, seed, False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    with np.load(ROOT / record["episode"], allow_pickle=False) as source:
        source_action = source["action"].copy()
        source_first_qpos = source["first_qpos"].copy()
        source_first_qvel = source["first_qvel"].copy()
    if sha(ROOT / record["episode"]) != record["episode_sha256"]:
        raise RuntimeError("Source teacher changed")
    rng = np.random.default_rng(seed + 14000)
    try:
        obs, _ = env.reset(seed=seed)
        entry_gap = max(float(np.max(abs(env.sim.data.qpos[0].cpu().numpy() - source_first_qpos))),
                        float(np.max(abs(env.sim.data.qvel[0].cpu().numpy() - source_first_qvel))))
        term = env.action_manager.get_term("joint_pos")
        rows = []
        running = longest = 0
        max_source_action_gap = 0.0
        for step in range(1, 201):
            policy_action = np.zeros((1, 21), dtype=np.float32)
            if mode == "bounded_noise":
                policy_action[0, term._arm_indices] = rng.uniform(-1, 1, size=5)
            obs, _, ended, timed, _ = env.step(torch.as_tensor(policy_action))
            teacher = term.teacher_raw[0].cpu().numpy().copy()
            max_source_action_gap = max(max_source_action_gap,
                                        float(np.max(abs(teacher - source_action[step - 1]))))
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
            hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
            cube = env.scene["cube"].data.root_link_pos_w[0].cpu().numpy().copy()
            running = running + 1 if valid else 0
            longest = max(longest, running)
            rows.append({"step": step, "policy_gain": float(term.last_policy_gain[0]),
                         "physical_touch": valid, "table": table,
                         "hand_floor": hand_floor, "cube_xyz_m": cube.tolist()})
            if bool(ended[0]) or bool(timed[0]):
                break
        cube = np.asarray([r["cube_xyz_m"] for r in rows])
        drift = float(np.linalg.norm((cube[:, :2] - cube[0, :2]) * 1000, axis=1).max())
        gentle = bool(len(rows) == 200 and longest >= 50
                      and sum(r["table"] for r in rows) >= 190
                      and rows[-1]["table"] and not any(r["hand_floor"] for r in rows)
                      and drift <= 5)
        return {"seed": seed, "mode": mode, "residual_gain": gain,
                "entry_physical_max_gap": entry_gap,
                "source_teacher_action_max_gap": max_source_action_gap,
                "steps": len(rows), "basic_touch": longest >= 10,
                "gentle_whole_episode": gentle, "longest_supported_touch_s": longest * .02,
                "cube_peak_xy_drift_mm": drift,
                "first_active_step": next((r["step"] for r in rows if r["policy_gain"] > 0), None),
                "max_policy_gain": max(r["policy_gain"] for r in rows),
                "rows": rows}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--gain", type=float, default=0.02)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    rows = []
    for record in source["records"][:args.limit]:
        for mode in ("zero", "bounded_noise"):
            row = visit(record, mode, args.gain)
            rows.append(row)
            print(row["seed"], mode, row["basic_touch"],
                  row["gentle_whole_episode"], round(row["longest_supported_touch_s"], 2),
                  round(row["cube_peak_xy_drift_mm"], 2), flush=True)
    result = {"schema": "robot-learning-lab.touch-contact-phase-v14-cpu.v1",
              "claim_limit": "Local CPU teacher-parity and bounded-noise development only; no learned policy or native CUDA score",
              "body_sha256": sha(BODY), "source_report_sha256": sha(SOURCE),
              "adapter_sha256": sha(Path(__file__).with_name("touch_reference_residual_v14.py")),
              "runner_sha256": sha(Path(__file__)), "visits": len(rows),
              "residual_gain": args.gain,
              "zero_basic": sum(r["basic_touch"] for r in rows if r["mode"] == "zero"),
              "noise_basic": sum(r["basic_touch"] for r in rows if r["mode"] == "bounded_noise"),
              "zero_gentle": sum(r["gentle_whole_episode"] for r in rows if r["mode"] == "zero"),
              "noise_gentle": sum(r["gentle_whole_episode"] for r in rows if r["mode"] == "bounded_noise"),
              "gpu_hours": 0, "optimizer_updates": 0, "records": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
