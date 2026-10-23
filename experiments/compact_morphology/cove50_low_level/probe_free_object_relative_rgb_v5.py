"""Small local observation diagnostic on opened free-object starts.

The controller remains the V4 scripted ramp. Only the initial cube placement
is used to choose its target. Simulator cube truth scores RGB estimates; it
never updates the action. This is not a contact or learner score.
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

from collect_free_object_touch import BODY, EXPECTED_BODY_SHA
from free_object_touch_env import ROOT, touch_env_cfg
from probe_observed_free_object_touch import entry_rgb_calibration
from screen_observed_free_object_touch_v4 import action_target, rgb_estimate
from ape_walk.physics.frame50 import require_direct_frame

PLAN = ROOT / "artifacts/ape-low-level-training-20261013/free-object-observed-touch-v4/plan.json"
CAMERA = "robot/lower_rgb"
MOVING_PAD = "robot/right_finger_tip"
FIXED_PAD = "robot/right_hand"
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 120
PLANE_Z = 0.14


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projection(model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray) -> np.ndarray:
    camera = model.camera(CAMERA).id
    relative = data.cam_xmat[camera].reshape(3, 3).T @ (point - data.cam_xpos[camera])
    depth = -relative[2]
    if depth <= 0:
        return np.full(2, np.nan)
    focal = (IMAGE_HEIGHT / 2) / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)
    return np.array([(IMAGE_WIDTH - 1) / 2 + focal * relative[0] / depth,
                     (IMAGE_HEIGHT - 1) / 2 - focal * relative[1] / depth])


def plane_estimate(model: mujoco.MjModel, data: mujoco.MjData,
                   rgb: np.ndarray) -> np.ndarray | None:
    if rgb[3] != 1 or rgb[5] != 0:
        return None
    camera = model.camera(CAMERA).id
    focal = (IMAGE_HEIGHT / 2) / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)
    u, v = rgb[:2] * [IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1]
    ray_c = np.array([(u - (IMAGE_WIDTH - 1) / 2) / focal,
                      -((v - (IMAGE_HEIGHT - 1) / 2) / focal), -1.0])
    ray_w = data.cam_xmat[camera].reshape(3, 3) @ ray_c
    if ray_w[2] >= -0.05:
        return None
    scale = (PLANE_Z - data.cam_xpos[camera, 2]) / ray_w[2]
    if scale <= 0:
        return None
    return (data.cam_xpos[camera] + scale * ray_w)[:2]


def run(seed: int, cube_xy: tuple[float, float], coefficients: np.ndarray,
        shift_mm: float) -> dict:
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        model = env.sim.mj_model
        probe = mujoco.MjData(model)
        term = env.action_manager.get_term("joint_pos")
        offset, scale = term._offset[0], term._scale[0]
        initial_rgb = obs["actor"][0, -6:].detach().cpu().numpy()
        initial_xy = rgb_estimate(initial_rgb, coefficients)
        assert initial_xy is not None
        probe.qpos[:] = env.sim.data.qpos[0].detach().cpu().numpy()
        mujoco.mj_forward(model, probe)
        initial_dynamic_xy = plane_estimate(model, probe, initial_rgb)
        assert initial_dynamic_xy is not None
        target = action_target(env, require_direct_frame(BODY), initial_xy)
        cube_joint = model.joint("cube/cube_joint").qposadr[0]
        moving_id = model.geom(MOVING_PAD).id
        fixed_id = model.geom(FIXED_PAD).id
        rows = []
        for step in range(1, 101):
            # Apply an unscored external 4 mm challenge well before first V4 touch.
            if shift_mm and step == 35:
                env.sim.data.qpos[0, cube_joint + 1] += shift_mm / 1000
                env.sim.forward()
            alpha = min(1.0, step / 75)
            desired = offset + alpha * (target - offset)
            raw = torch.where(scale.abs() > 1e-8,
                              (desired - offset) / scale.clamp_min(1e-8),
                              torch.zeros_like(desired))
            obs, _, terminated, timed_out, _ = env.step(raw.clamp(-1, 1)[None, :])
            rgb = obs["actor"][0, -6:].detach().cpu().numpy()
            probe.qpos[:] = env.sim.data.qpos[0].detach().cpu().numpy()
            mujoco.mj_forward(model, probe)
            truth = probe.qpos[cube_joint:cube_joint + 2].copy()
            moving_uv = projection(model, probe, probe.geom_xpos[moving_id])
            fixed_uv = projection(model, probe, probe.geom_xpos[fixed_id])
            observed_uv = rgb[:2] * [IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1]
            static_xy = rgb_estimate(rgb, coefficients)
            dynamic_xy = plane_estimate(model, probe, rgb)
            anchored_xy = None if dynamic_xy is None else initial_xy + dynamic_xy - initial_dynamic_xy
            rows.append({
                "step": step,
                "rgb_fresh": bool(rgb[3] == 1 and rgb[5] == 0),
                "truth_xy_m": truth.tolist(),
                "static_xy_m": None if static_xy is None else static_xy.tolist(),
                "dynamic_xy_m": None if dynamic_xy is None else dynamic_xy.tolist(),
                "anchored_xy_m": None if anchored_xy is None else anchored_xy.tolist(),
                "static_error_mm": None if static_xy is None else float(np.linalg.norm(static_xy - truth) * 1000),
                "dynamic_error_mm": None if dynamic_xy is None else float(np.linalg.norm(dynamic_xy - truth) * 1000),
                "anchored_error_mm": None if anchored_xy is None else float(np.linalg.norm(anchored_xy - truth) * 1000),
                "cube_minus_moving_pad_px": (observed_uv - moving_uv).tolist(),
                "cube_minus_fixed_pad_px": (observed_uv - fixed_uv).tolist(),
                "finger_contact": bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any()),
            })
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        return {"seed": seed, "cube_xy_m": list(cube_xy), "external_y_shift_mm": shift_mm,
                "steps": len(rows), "rows": rows}
    finally:
        env.close()


def summarize(record: dict) -> dict:
    rows = record["rows"]
    pre = [r for r in rows if r["step"] <= 60 and not r["finger_contact"] and r["rgb_fresh"]]
    out = {k: record[k] for k in ("seed", "external_y_shift_mm", "steps")}
    out["precontact_fresh_frames"] = len(pre)
    for name in ("static", "dynamic", "anchored"):
        errors = [r[f"{name}_error_mm"] for r in pre if r[f"{name}_error_mm"] is not None]
        out[f"{name}_precontact_median_error_mm"] = None if not errors else float(np.median(errors))
        out[f"{name}_precontact_max_error_mm"] = None if not errors else float(max(errors))
    before = next((r for r in rows if r["step"] == 34), None)
    after = next((r for r in rows if r["step"] == 35), None)
    if before and after:
        out["step35_delta"] = {
            "truth_y_mm": (after["truth_xy_m"][1] - before["truth_xy_m"][1]) * 1000,
            "static_y_mm": None if before["static_xy_m"] is None or after["static_xy_m"] is None else
            (after["static_xy_m"][1] - before["static_xy_m"][1]) * 1000,
            "dynamic_y_mm": None if before["dynamic_xy_m"] is None or after["dynamic_xy_m"] is None else
            (after["dynamic_xy_m"][1] - before["dynamic_xy_m"][1]) * 1000,
            "anchored_y_mm": None if before["anchored_xy_m"] is None or after["anchored_xy_m"] is None else
            (after["anchored_xy_m"][1] - before["anchored_xy_m"][1]) * 1000,
            "moving_pad_relative_uv_px": (np.asarray(after["cube_minus_moving_pad_px"]) -
                                          before["cube_minus_moving_pad_px"]).tolist(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert sha(BODY) == EXPECTED_BODY_SHA
    plan = json.loads(PLAN.read_text())
    coefficients, calibration = entry_rgb_calibration()
    records = [run(case["seed"], tuple(case["cube_xy_m"]), coefficients, shift)
               for case in plan["families"][:3] for shift in (0.0, 4.0)]
    report = {"schema": "robot-learning-lab.cove50-free-object-relative-rgb.v5",
              "claim_limit": "Opened-start local CPU observation diagnostic. No learned policy or CUDA contact score.",
              "body_sha256": sha(BODY), "source_sha256": sha(Path(__file__)),
              "v4_plan_sha256": sha(PLAN), "calibration": calibration,
              "summaries": [summarize(r) for r in records], "records": records}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
