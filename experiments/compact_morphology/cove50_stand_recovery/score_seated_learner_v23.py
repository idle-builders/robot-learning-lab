"""Source-check fresh CUDA seated-rise rollouts of one saved PPO actor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v23"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(checkpoint_id: str, seed: int, model: mujoco.MjModel, gate: dict,
          checkpoint_path: Path) -> dict:
    report_path = ART / f"checkpoint-{checkpoint_id}-seed{seed}.json"
    arrays_path = ART / f"checkpoint-{checkpoint_id}-seed{seed}.npz"
    report = json.loads(report_path.read_text())
    samples = report["timeseries"]
    with np.load(arrays_path) as arrays:
        qpos, qvel = arrays["qpos"].copy(), arrays["qvel"].copy()
        initial_qpos, initial_qvel = arrays["initial_qpos"].copy(), arrays["initial_qvel"].copy()
        actor, raw, applied = (arrays[name].copy() for name in
                               ("actor_observation", "raw_action", "applied_action"))
        targets, reward_rate = arrays["processed_target"].copy(), arrays["reward_rate"].copy()
    assert report["seed"] == seed and report["device"] == "cuda:0"
    assert report["task"] == "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved"
    assert report["seated_rise_stage"] and report["recover_from_start"]
    assert not report["scripted_target_probe"] and not report["zero_actions"]
    assert report["randomized_physics"]
    assert report["env_cfg_play"] is False
    assert report["actor_observation_corruption"] is True
    assert len(report["motor_delay_config"]) == 21
    assert all(d == {"min_lag_steps": 0, "max_lag_steps": 4}
               for d in report["motor_delay_config"])
    assert set(report["observed_motor_delay_lags"]) == set(range(5))
    assert len(report["reward_term_names"]) == 15
    assert "heading_hold" in report["reward_term_names"]
    assert report["scripted_min_blend"] is None
    assert report["initial_physics"] is not None
    motors = report["initial_motors"]
    assert len(motors["target_names"]) == 21
    assert all(0.9 <= x <= 1.05 for x in motors["effort_factor"])
    assert all(0.9 <= x <= 1.1 for x in motors["stiffness_factor"])
    assert all(0.9 <= x <= 1.1 for x in motors["damping_factor"])
    assert Path(report["checkpoint"]).name == checkpoint_path.name
    n = len(samples)
    assert n == len(qpos) == len(qvel) == len(actor) == len(raw) == len(applied) == len(targets) == len(reward_rate)
    assert actor.shape[1] == 73 and raw.shape[1] == applied.shape[1] == targets.shape[1] == 21
    assert np.array_equal(initial_qpos, report["initial_qpos"])
    assert np.array_equal(initial_qvel, report["initial_qvel"])

    data = mujoco.MjData(model)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_sole_center")
             for side in ("left", "right")]
    soles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
             for side in ("left", "right")]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    site_xy, corner_xy, floor_body = [], [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        site_xy.append(data.site_xpos[sites, :2].copy())
        floor_body.append(any(
            (c.geom1 == floor and c.geom2 not in soles)
            or (c.geom2 == floor and c.geom1 not in soles)
            for c in data.contact[: data.ncon]
        ))
        patches = []
        for geom_id in soles:
            half_x, half_y = model.geom_size[geom_id, :2]
            local = np.array([[x * half_x, y * half_y, 0] for x in (-1, 1) for y in (-1, 1)])
            rot = data.geom_xmat[geom_id].reshape(3, 3)
            patches.append(data.geom_xpos[geom_id, :2] + (local @ rot.T)[:, :2])
        corner_xy.append(patches)
    site_xy, corner_xy = np.asarray(site_xy), np.asarray(corner_xy)
    site_gap = float(np.max(np.abs(site_xy - [s["sole_site_xy_m"] for s in samples])))
    assert site_gap <= 5e-5
    assert all(not hit or s["floor_nonsole_body_contact"] for hit, s in zip(floor_body, samples))
    both = np.array([all(s["feet_grounded"]) for s in samples])
    anchor = next((i for i in range(max(0, n - 4)) if both[i:i+5].all()), None)
    anchor_s = samples[anchor]["time_s"] if anchor is not None else None
    site_move = float(np.max(np.linalg.norm(site_xy[anchor:] - site_xy[anchor], axis=2))) if anchor is not None else None
    corner_move = float(np.max(np.linalg.norm(corner_xy[anchor:] - corner_xy[anchor], axis=3))) if anchor is not None else None
    gaps = int(np.count_nonzero(~both[anchor:])) if anchor is not None else None
    tail = [s for s in samples if 9.0 <= s["time_s"] <= 10.0]
    check = {
        "full_duration": n == 500 and report["duration_s"] >= 10.0 and not report["ended_early"],
        "tail_upright": bool(tail) and max(s["tilt_deg"] for s in tail) < gate["tail_tilt_lt_deg"],
        "tail_height": bool(tail) and gate["tail_pelvis_z_m"][0] <= np.mean([s["pelvis_z_m"] for s in tail]) <= gate["tail_pelvis_z_m"][1],
        "tail_double_support": bool(tail) and np.mean([all(s["feet_grounded"]) for s in tail]) >= gate["tail_both_feet_fraction_gte"],
        "anchor_by_0p2s": anchor_s is not None and anchor_s <= gate["first_five_consecutive_both_contact_by_s"],
        "continuous_double_support_after_anchor": gaps == 0,
        "sole_center_stays": site_move is not None and site_move <= gate["sole_center_max_displacement_after_anchor_m_lte"],
        "sole_patch_stays": corner_move is not None and corner_move <= gate["sole_box_corner_max_displacement_after_anchor_m_lte"],
        "no_hand_body_contact": not any(floor_body) and not any(s["body_contact"] or s["hand_force_n"] >= 1 for s in samples),
        "rated_leg_fraction": np.mean([s["leg_torque_saturated"] for s in samples]) <= gate["rated_leg_fraction_lte"],
    }
    check = {k: bool(v) for k, v in check.items()}
    old_v5_gate = all(check.values())
    check["commanded_stand_height"] = bool(
        tail and np.mean([s["pelvis_z_m"] for s in tail]) >= 0.22
    )
    names = report["reward_term_names"]
    contributions = dict(zip(names, (reward_rate.sum(axis=0) * 0.02).tolist()))
    first_s = samples[:50]
    last = samples[-1]
    return {
        "seed": seed, "checkpoint_id": checkpoint_id,
        "source_report": str(report_path.relative_to(ROOT)), "source_report_sha256": sha(report_path),
        "native_trajectory": str(arrays_path.relative_to(ROOT)), "native_trajectory_sha256": sha(arrays_path),
        "initial_state_sha256": report["initial_state_sha256"],
        "initial_physics": report["initial_physics"],
        "initial_motors": motors,
        "duration_s": report["duration_s"], "ended_early": report["ended_early"],
        "source_site_parity_max_abs_m": site_gap,
        "first_s_tilt_max_deg": max(s["tilt_deg"] for s in first_s),
        "last_tilt_deg": last["tilt_deg"], "last_pelvis_z_m": last["pelvis_z_m"],
        "named_termination_terms": report["end_termination_terms"],
        "terminal_failure_signals_inferred": {
            "tilt_gte_20_deg": last["tilt_deg"] >= 20,
            "pelvis_z_lt_0p15m": last["pelvis_z_m"] < 0.15,
            "nonsole_floor_contact": last["floor_nonsole_body_contact"],
            "hand_force_gte_1n": last["hand_force_n"] >= 1,
        },
        "floor_nonsole_contact_frames": sum(floor_body),
        "anchor_s": anchor_s, "post_anchor_both_contact_gap_frames": gaps,
        "max_post_anchor_sole_center_motion_m": site_move,
        "max_post_anchor_sole_corner_motion_m": corner_move,
        "rated_leg_fraction": float(np.mean([s["leg_torque_saturated"] for s in samples])),
        "reward_total_unrounded": float(reward_rate.sum() * 0.02),
        "reward_term_contributions": contributions,
        "physical_gate": check, "old_v5_gate": old_v5_gate,
        "full_gate": all(check.values()),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: score_seated_learner_v23.py CHECKPOINT_ID")
    checkpoint_id = sys.argv[1]
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert checkpoint_id in [str(x) for x in plan["evaluation"]["checkpoints"]]
    checkpoint_path = ART / "checkpoints" / f"model_{checkpoint_id}.pt"
    assert checkpoint_path.is_file()
    model_path = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    assert sha(model_path) == plan["body_sha256"]
    assert sha(ART / "source/train_posture_arm.py") == plan["trainer_source_sha256"]
    assert sha(ART / "source/evaluate_posture.py") == plan["evaluator_source_sha256"]
    assert sha(ART / "source/task.py") == plan["task_source_sha256"]
    assert sha(ART / "source/frame50.py") == plan["motor_source_sha256"]
    assert sha(ART / "source/upstream-mdp.py") == plan["upstream_mdp_source_sha256"]
    assert sha(Path(__file__)) == plan["scorer_source_sha256"]
    gate = json.loads((ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v5/plan.json").read_text())["physical_gate"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    rows = [score(checkpoint_id, seed, model, gate, checkpoint_path)
            for seed in plan["evaluation"]["development_new_seeds"]]
    summary = {
        "fresh_development_starts": len(rows),
        "full_stand_passes": sum(r["full_gate"] for r in rows),
        "early_terminations": sum(r["ended_early"] for r in rows),
        "mean_return": float(np.mean([r["reward_total_unrounded"] for r in rows])),
        "mean_rated_leg_fraction": float(np.mean([r["rated_leg_fraction"] for r in rows])),
    }
    output = {"schema": "robot-learning-lab.cove50-seated-rise-learned-development-score.v23",
              "plan_sha256": sha(plan_path), "body_sha256": sha(model_path),
              "scorer_source_sha256": sha(Path(__file__)),
              "status": "complete_source_checked_cuda_development_score",
              "checkpoint_id": checkpoint_id,
              "checkpoint_sha256": sha(checkpoint_path),
              "summary": summary, "rows": rows,
              "interpretation": "Physical commanded stand gate is independent of PPO reward; compare current actor with saved V22 scripted, parked and zero references.",
              "claim_limit": "Fresh randomized CUDA learned actor development starts; reserved independent score stays sealed until a useful checkpoint is selected."}
    (ART / f"checkpoint-{checkpoint_id}-development-score.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
