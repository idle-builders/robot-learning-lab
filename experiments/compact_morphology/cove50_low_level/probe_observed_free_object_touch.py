"""Small CPU contrast: RGB-guided scripted touch versus contact-truth freeze.

Calibration uses opened entry images and known cube placements. During each
episode the controller receives only actor RGB/proprioception; contact and
cube truth are recorded after stepping for the physical outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from collect_free_object_touch import (ARM_NAMES, BODY, EXPECTED_BODY_SHA,
                                       NOMINAL_TARGET, adapted_target)
from free_object_touch_env import (ROOT, STAND_TABLE_CUBE_XYZ,
                                   physical_touch_gate_frame, touch_env_cfg)
from ape_walk.physics.frame50 import require_direct_frame

LEDGER = ROOT / "artifacts/ape-low-level-training-20261013/free-object-touch-teacher-v2/report.json"
DEVELOPMENT = ((31001, (.049, -.141)), (31002, (.052, -.138)))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry_rgb_calibration() -> tuple[np.ndarray, dict]:
    ledger = json.loads(LEDGER.read_text())
    assert ledger["body_sha256"] == EXPECTED_BODY_SHA
    records = ledger["records"]
    assert len(records) == 9
    image = np.asarray([[1.0, *r["entry_rgb_target"][:3]] for r in records])
    known = np.asarray([r["cube_xy_m"] for r in records])
    coefficients, *_ = np.linalg.lstsq(image, known, rcond=None)
    fit_error_mm = np.linalg.norm((image @ coefficients - known) * 1000, axis=1)
    return coefficients, {"source": str(LEDGER.relative_to(ROOT)),
                          "source_sha256": digest(LEDGER),
                          "families": len(records),
                          "input": "intercept, RGB red-target u/v/area at reset",
                          "max_calibration_error_mm": float(fit_error_mm.max()),
                          "coefficients": coefficients.tolist()}


def episode(seed: int, cube_xy: tuple[float, float], coefficients: np.ndarray,
            out: Path) -> dict:
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
        initial = obs["actor"][0, -6:].detach().cpu().numpy()
        assert initial[3] == 1 and initial[5] == 0
        initial_pad_contact = bool(
            env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any() or
            env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
        assert not initial_pad_contact
        initial_cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
        estimated_xy = np.asarray([1.0, *initial[:3]]) @ coefficients
        model = require_direct_frame(BODY)
        ik = adapted_target(model, tuple(float(v) for v in estimated_xy))
        term = env.action_manager.get_term("joint_pos")
        names, offset, scale = term._target_names, term._offset[0], term._scale[0]
        target = offset.clone()
        for name, value in ik.items():
            target[names.index(name)] = value
        assert all(name in names for name in (*ARM_NAMES, "right_gripper"))
        hold_alpha = None
        trigger_step = None
        last_rgb = initial.copy()
        running = longest = 0
        first_touch_step = None
        finger_frames = fixed_frames = both_frames = table_frames = hand_floor_frames = 0
        early_end = False
        traces = []
        for step in range(200):
            alpha = min(1.0, (step + 1) / 75) if hold_alpha is None else hold_alpha
            desired = offset + alpha * (target - offset)
            raw = torch.where(scale.abs() > 1e-8,
                              (desired - offset) / scale.clamp_min(1e-8),
                              torch.zeros_like(desired))
            action = raw.clamp(-1, 1)[None, :]
            obs, _, terminated, timed_out, _ = env.step(action)
            rgb = obs["actor"][0, -6:].detach().cpu().numpy()
            du, dv = (rgb[:2] - last_rgb[:2]).tolist()
            # A simultaneous, subpixel left/up target jump is an image-only
            # contact candidate on this opened tabletop geometry. Freeze the
            # arm ramp at the next command. It is not a contact sensor.
            if hold_alpha is None and rgb[3] == 1 and rgb[5] == 0 and \
                    du <= -.0015 and dv <= -.0015:
                hold_alpha = alpha
                trigger_step = step + 1
            last_rgb = rgb.copy()
            finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
            fixed = bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
            table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
            hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
            physical = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            finger_frames += int(finger)
            fixed_frames += int(fixed)
            both_frames += int(finger and fixed)
            table_frames += int(table)
            hand_floor_frames += int(hand_floor)
            running = running + 1 if physical else 0
            longest = max(longest, running)
            if physical and first_touch_step is None:
                first_touch_step = step + 1
            cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy()
            traces.append([step + 1, alpha, *rgb, du, dv, *cube,
                           int(finger), int(fixed), int(table), int(physical)])
            if bool(terminated[0]) or bool(timed_out[0]):
                early_end = True
                break
        out.mkdir(parents=True, exist_ok=True)
        trace = out / f"seed-{seed}-trace.npz"
        np.savez_compressed(trace, step_alpha_rgb_du_dv_cube_contacts=np.asarray(traces))
        cube_path = np.asarray(traces)[:, 10:13]
        true_xy = np.asarray(cube_xy)
        return {"seed": seed, "start_xy_m": list(cube_xy),
                "entry_rgb": initial.tolist(),
                "initial_pad_contact": initial_pad_contact,
                "rgb_estimated_xy_m": estimated_xy.tolist(),
                "entry_estimation_error_mm": float(np.linalg.norm((estimated_xy - true_xy) * 1000)),
                "ik_target_rad": ik,
                "image_trigger_step": trigger_step,
                "image_trigger_after_physical_touch_steps": None if trigger_step is None or first_touch_step is None else trigger_step - first_touch_step,
                "first_physical_touch_step": first_touch_step,
                "longest_supported_touch_s": round(longest * env.step_dt, 3),
                "finger_frames": finger_frames, "fixed_frames": fixed_frames,
                "both_pad_frames": both_frames, "cube_table_frames": table_frames,
                "cube_net_displacement_mm": float(np.linalg.norm((cube_path[-1] - initial_cube) * 1000)),
                "cube_max_displacement_mm": float(np.linalg.norm((cube_path - initial_cube) * 1000, axis=1).max()),
                "cube_max_center_z_m": float(cube_path[:, 2].max()),
                "table_at_end": table, "hand_floor_frames": hand_floor_frames,
                "early_end": early_end, "touch_gate_pass": longest >= 10 and not early_end,
                "strict_imitation_acceptance": bool(longest >= 50 and table_frames >= 190 and table and
                                                    hand_floor_frames == 0 and not early_end),
                "trace": str(trace.relative_to(ROOT)), "trace_sha256": digest(trace)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    assert digest(BODY) == EXPECTED_BODY_SHA
    coefficients, calibration = entry_rgb_calibration()
    records = [episode(seed, xy, coefficients, args.out) for seed, xy in DEVELOPMENT]
    report = {"schema": "robot-learning-lab.cove50-observed-free-object-touch-probe.v1",
              "body": str(BODY.relative_to(ROOT)), "body_sha256": EXPECTED_BODY_SHA,
              "source": str(Path(__file__).resolve().relative_to(ROOT)),
              "source_sha256": digest(Path(__file__)),
              "device": "local CPU", "control_dt_s": .02,
              "calibration": calibration,
              "controller": "offline IK from reset RGB estimate, 75-step ramp, image u/v jump freeze; no live cube pose/contact used for actions",
              "image_trigger": "fresh RGB u and v each decrease >=0.0015 in one 20 ms step",
              "evaluator_only": "true cube start for error, right finger/fixed pad and table contact, object trajectory, floor/feet/pelvis physical gate",
              "records": records,
              "touch_gate_passes": sum(r["touch_gate_pass"] for r in records),
              "strict_imitation_acceptances": sum(r["strict_imitation_acceptance"] for r in records),
              "two_pad_possession": "not established: concurrent frames are contact only; no lift/retention gate run",
              "claim_limit": "Two opened local CPU development starts; scripted image feedback, not learned touch or independent score. No floor acquisition/place/CUDA."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.out / 'report.json'),
                      "records": [{k: r[k] for k in ("seed", "entry_estimation_error_mm",
                                                      "image_trigger_step", "longest_supported_touch_s",
                                                      "both_pad_frames", "strict_imitation_acceptance")}
                                  for r in records]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
