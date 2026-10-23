"""Source-check a saved V25 seated-rise actor on frozen CUDA score starts.

The physical gate is independent of return. Development, tail and reserved
families are separate; only a frozen selected candidate can open the latter two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
DT = 0.02
SOURCE_FILES = {
    "train_posture_arm.py": "train_posture_arm.py",
    "evaluate_posture.py": "evaluate_posture.py",
    "task.py": "task.py",
    "frame50.py": "frame50.py",
    "upstream-mdp.py": "upstream-mdp.py",
    "score_seated_learner_v25.py": "score_seated_learner_v25.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def checked_plan() -> tuple[dict, str]:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan_sha = sha(plan_path)
    require(plan["body_sha256"] == sha(BODY), "body SHA differs from V25 freeze")
    for name, snapshot in SOURCE_FILES.items():
        expected = plan["source_sha256"][name]
        require(sha(ART / "source" / snapshot) == expected,
                f"V25 source snapshot changed: {name}")
        if name == "score_seated_learner_v25.py":
            require(sha(Path(__file__)) == expected,
                    "running scorer differs from V25 freeze")
    gate = plan["physical_gate"]
    required_gate = (
        "full_duration_s", "tail_window_s", "tail_tilt_lt_deg",
        "tail_pelvis_z_m", "tail_both_feet_fraction_gte",
        "first_five_consecutive_both_contact_by_s",
        "no_both_contact_gaps_after_anchor",
        "sole_center_max_displacement_after_anchor_m_lte",
        "sole_box_corner_max_displacement_after_anchor_m_lte",
        "no_hand_or_body_contact", "rated_leg_fraction_lte",
        "commanded_stand_height_m_gte",
    )
    require(all(key in gate for key in required_gate), "V25 physical gate incomplete")
    require(gate["full_duration_s"] == 10.0 and
            gate["commanded_stand_height_m_gte"] == 0.22,
            "V25 changed the prospective ten-second rise gate")
    evaluation = plan["evaluation"]
    families = (evaluation["development_new_seeds"],
                evaluation["reserved_unopened_seeds"])
    flattened = [int(seed) for family in families for seed in family]
    require(len(flattened) == len(set(flattened)), "score families overlap")
    return plan, plan_sha


def selected_candidate(checkpoint_id: str, checkpoint_sha: str,
                       plan_sha: str) -> dict:
    selection_path = ART / "selected-candidate.json"
    selected = json.loads(selection_path.read_text())
    require(selected["plan_sha256"] == plan_sha,
            "selected candidate uses another V25 plan")
    require(str(selected["checkpoint_id"]) == checkpoint_id and
            selected["checkpoint_sha256"] == checkpoint_sha,
            "reserved/tail score can open only for the frozen candidate")
    development_path = ART / f"checkpoint-{checkpoint_id}-development-score.json"
    require(sha(development_path) == selected["development_score_sha256"],
            "selected candidate development evidence changed")
    return selected


def checked_tail_reserve(plan: dict, plan_sha: str) -> tuple[list[int], dict]:
    """The action-free gap screen is frozen after the immutable PPO plan."""
    path = ART / "tail-reserve.json"
    reserve = json.loads(path.read_text())
    require(reserve["plan_sha256"] == plan_sha and reserve["action_free_screen"],
            "tail reserve is not bound to the frozen V25 plan/screen")
    screen = ART / reserve["screen_report"]
    require(sha(screen) == reserve["screen_report_sha256"],
            "tail reset-only screen changed")
    entries = reserve["entries"]
    seeds = [int(entry["seed"]) for entry in entries]
    require(len(seeds) >= 1 and len(seeds) == len(set(seeds)) and all(
        float(entry["predicted_sole_gap_m"]) > 0.015 for entry in entries),
        "tail reserve is empty, duplicate or outside the >15 mm slice")
    others = set(map(int, plan["evaluation"]["development_new_seeds"])) | set(
        map(int, plan["evaluation"]["reserved_unopened_seeds"]))
    require(not (set(seeds) & others), "tail seeds overlap development/reserved")
    return seeds, reserve


def score_case(checkpoint_id: str, seed: int, model: mujoco.MjModel,
               gate: dict, checkpoint_path: Path) -> dict:
    report_path = ART / f"checkpoint-{checkpoint_id}-seed{seed}.json"
    arrays_path = ART / f"checkpoint-{checkpoint_id}-seed{seed}.npz"
    report = json.loads(report_path.read_text())
    samples = report["timeseries"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = ("initial_qpos", "initial_qvel", "actor_observation",
                    "raw_action", "applied_action", "processed_target",
                    "qpos", "qvel", "reward_rate")
        require(all(name in arrays for name in required),
                f"seed {seed}: native trajectory missing a required array")
        values = {name: arrays[name].copy() for name in required}
    initial_qpos, initial_qvel = values["initial_qpos"], values["initial_qvel"]
    qpos, qvel = values["qpos"], values["qvel"]
    actor, raw, applied = (values[name] for name in
                           ("actor_observation", "raw_action", "applied_action"))
    targets, reward_rate = values["processed_target"], values["reward_rate"]

    require(report["seed"] == seed and report["device"] == "cuda:0",
            f"seed {seed}: wrong native score seed/device")
    require(report["body_sha256"] == sha(BODY) and
            report["task"] == "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved",
            f"seed {seed}: wrong body/task")
    require(report["seated_rise_stage"] and report["recover_from_start"] and
            report["randomized_physics"] and
            not report["scripted_target_probe"] and
            not report["zero_actions"] and
            report["scripted_min_blend"] is None and
            report["env_cfg_play"] is False and
            report["actor_observation_corruption"] is True,
            f"seed {seed}: evaluator differs from the training task")
    require(Path(report["checkpoint"]).name == checkpoint_path.name,
            f"seed {seed}: wrong checkpoint name")
    require(len(report["motor_delay_config"]) == 21 and all(
        lag == {"min_lag_steps": 0, "max_lag_steps": 4}
        for lag in report["motor_delay_config"]),
        f"seed {seed}: motor-delay contract changed")
    observed_lags = report["observed_motor_delay_lags"]
    require(bool(observed_lags) and set(observed_lags).issubset(set(range(5))),
            f"seed {seed}: observed motor-delay draw outside training envelope")
    motors = report["initial_motors"]
    require(report["initial_physics"] is not None and motors is not None and
            len(motors["target_names"]) == 21,
            f"seed {seed}: sampled physics/motors missing")
    for name, lo, hi in (("effort_factor", 0.9, 1.05),
                         ("stiffness_factor", 0.9, 1.1),
                         ("damping_factor", 0.9, 1.1)):
        require(len(motors[name]) == 21 and all(lo - 1e-5 <= x <= hi + 1e-5
                                               for x in motors[name]),
                f"seed {seed}: {name} outside training envelope")
    require(all(0.65 - 1e-5 <= x <= 1.15 + 1e-5 for x in
                report["initial_physics"]["sole_slide_friction"]),
            f"seed {seed}: sole friction outside training envelope")

    n = len(samples)
    expected_n = round(gate["full_duration_s"] / DT)
    require(0 < n <= expected_n and all(len(v) == n for v in
                                       (qpos, qvel, actor, raw, applied,
                                        targets, reward_rate)),
            f"seed {seed}: trajectory lengths disagree")
    require(qpos.shape == (n, model.nq) and qvel.shape == (n, model.nv)
            and initial_qpos.shape == (model.nq,)
            and initial_qvel.shape == (model.nv,)
            and actor.shape == (n, 73)
            and all(v.shape == (n, 21) for v in (raw, applied, targets)),
            f"seed {seed}: physical or action array shape changed")
    names = report["reward_term_names"]
    require(len(names) == 15 and len(set(names)) == 15 and
            "heading_hold" in names and reward_rate.shape == (n, 15),
            f"seed {seed}: reward term contract changed")
    require(all(np.isfinite(v).all() for v in values.values()),
            f"seed {seed}: nonfinite native trajectory")
    require(np.array_equal(initial_qpos, report["initial_qpos"]) and
            np.array_equal(initial_qvel, report["initial_qvel"]),
            f"seed {seed}: initial arrays/report differ")
    start_hash = hashlib.sha256(initial_qpos.tobytes() + initial_qvel.tobytes()).hexdigest()
    require(start_hash == report["initial_state_sha256"],
            f"seed {seed}: initial state hash differs")
    require(hashlib.sha256(raw[0].tobytes()).hexdigest() == report["first_action_sha256"],
            f"seed {seed}: first action hash differs")
    require(abs(report["duration_s"] - n * DT) <= 1e-6 and all(
        abs(sample["time_s"] - (i + 1) * DT) <= 1e-6
        for i, sample in enumerate(samples)),
        f"seed {seed}: trajectory timebase differs")
    reward_from_terms = reward_rate.sum(axis=1) * DT
    reported_rewards = np.array([sample["step_reward"] for sample in samples])
    reward_gap = float(np.max(np.abs(reward_from_terms - reported_rewards)))
    require(reward_gap <= 5.1e-5,
            f"seed {seed}: reward terms do not reconstruct step reward")
    reported_height = np.array([sample["pelvis_z_m"] for sample in samples])
    height_gap = float(np.max(np.abs(qpos[:, 2] - reported_height)))
    require(height_gap <= 5.1e-5, f"seed {seed}: qpos/report pelvis height differs")

    data = mujoco.MjData(model)
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                  f"{side}_sole_center") for side in ("left", "right")]
    sole_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  f"{side}_sole") for side in ("left", "right")]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    require(all(i >= 0 for i in (*site_ids, *sole_ids, floor_id)),
            "native body lacks a physical score endpoint")
    site_xy, corner_xy, floor_nonsole = [], [], []
    for q, v in zip(qpos, qvel):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        site_xy.append(data.site_xpos[site_ids, :2].copy())
        floor_nonsole.append(any(
            (c.geom1 == floor_id and c.geom2 not in sole_ids) or
            (c.geom2 == floor_id and c.geom1 not in sole_ids)
            for c in data.contact[:data.ncon]
        ))
        corners = []
        for geom_id in sole_ids:
            half_x, half_y = model.geom_size[geom_id, :2]
            local = np.array([[x * half_x, y * half_y, 0.0]
                              for x in (-1, 1) for y in (-1, 1)])
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            corners.append(data.geom_xpos[geom_id, :2] +
                           (local @ rotation.T)[:, :2])
        corner_xy.append(corners)
    site_xy, corner_xy = np.asarray(site_xy), np.asarray(corner_xy)
    site_gap = float(np.max(np.abs(site_xy -
                    np.asarray([sample["sole_site_xy_m"] for sample in samples]))))
    require(site_gap <= 5e-5, f"seed {seed}: native sole-site/report mismatch")
    require(all(not hit or sample["floor_nonsole_body_contact"]
                for hit, sample in zip(floor_nonsole, samples)),
            f"seed {seed}: native floor contact missing in report")

    both = np.array([all(sample["feet_grounded"]) for sample in samples])
    anchor = next((i for i in range(max(0, n - 4))
                   if both[i:i + 5].all()), None)
    anchor_s = samples[anchor]["time_s"] if anchor is not None else None
    site_motion = (float(np.max(np.linalg.norm(site_xy[anchor:] - site_xy[anchor],
                                                axis=2))) if anchor is not None else None)
    corner_motion = (float(np.max(np.linalg.norm(corner_xy[anchor:] - corner_xy[anchor],
                                                  axis=3))) if anchor is not None else None)
    support_gap_frames = (int(np.count_nonzero(~both[anchor:]))
                          if anchor is not None else None)
    tail = [sample for sample in samples if gate["tail_window_s"][0] <=
            sample["time_s"] <= gate["tail_window_s"][1]]
    tail_height = (float(np.mean([sample["pelvis_z_m"] for sample in tail]))
                   if tail else None)
    named_terms = report["end_termination_terms"] or []
    checks = {
        "full_duration": n == expected_n and not report["ended_early"],
        "tail_upright": bool(tail) and max(s["tilt_deg"] for s in tail) <
                        gate["tail_tilt_lt_deg"],
        "tail_height": tail_height is not None and
                       gate["tail_pelvis_z_m"][0] <= tail_height <=
                       gate["tail_pelvis_z_m"][1],
        "tail_double_support": bool(tail) and
                               np.mean([all(s["feet_grounded"]) for s in tail]) >=
                               gate["tail_both_feet_fraction_gte"],
        "anchor_by_0p2s": anchor_s is not None and anchor_s <=
                         gate["first_five_consecutive_both_contact_by_s"],
        "continuous_double_support_after_anchor": support_gap_frames == 0,
        "sole_center_stays": site_motion is not None and site_motion <=
                             gate["sole_center_max_displacement_after_anchor_m_lte"],
        "sole_patch_stays": corner_motion is not None and corner_motion <=
                            gate["sole_box_corner_max_displacement_after_anchor_m_lte"],
        "no_hand_body_contact": not any(floor_nonsole) and not any(
            s["body_contact"] or s["hand_force_n"] >= 1 for s in samples),
        "rated_leg_fraction": np.mean([s["leg_torque_saturated"] for s in samples]) <=
                              gate["rated_leg_fraction_lte"],
        "commanded_stand_height": tail_height is not None and tail_height >=
                                  gate["commanded_stand_height_m_gte"],
    }
    checks = {name: bool(ok) for name, ok in checks.items()}
    failure_reasons = [name for name, ok in checks.items() if not ok]
    failure_reasons.extend(f"termination:{name}" for name in named_terms
                           if name != "time_out")
    contributions = dict(zip(names, (reward_rate.sum(axis=0) * DT).tolist()))
    return {
        "seed": seed,
        "checkpoint_id": checkpoint_id,
        "source_report": str(report_path.relative_to(ROOT)),
        "source_report_sha256": sha(report_path),
        "native_trajectory": str(arrays_path.relative_to(ROOT)),
        "native_trajectory_sha256": sha(arrays_path),
        "initial_state_sha256": start_hash,
        "initial_physics": report["initial_physics"],
        "initial_motors": motors,
        "observed_motor_delay_lags": observed_lags,
        "duration_s": report["duration_s"],
        "ended_early": report["ended_early"],
        "named_termination_terms": named_terms,
        "source_parity": {
            "sole_site_max_abs_m": site_gap,
            "pelvis_height_max_abs_m": height_gap,
            "step_reward_max_abs": reward_gap,
        },
        "initial_pelvis_z_m": float(initial_qpos[2]),
        "max_pelvis_z_m": float(np.max(qpos[:, 2])),
        "tail_mean_pelvis_z_m": tail_height,
        "first_s_max_tilt_deg": max(s["tilt_deg"] for s in samples[:50]),
        "last_tilt_deg": samples[-1]["tilt_deg"],
        "anchor_s": anchor_s,
        "post_anchor_both_contact_gap_frames": support_gap_frames,
        "post_anchor_sole_center_motion_m": site_motion,
        "post_anchor_sole_corner_motion_m": corner_motion,
        "rated_leg_fraction": float(np.mean([s["leg_torque_saturated"]
                                             for s in samples])),
        "reward_total_unrounded": float(reward_from_terms.sum()),
        "reward_term_contributions": contributions,
        "physical_gate": checks,
        "failure_reasons": failure_reasons,
        "full_gate": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_id", help="RSL-RL zero-based checkpoint index")
    parser.add_argument("--stage", choices=("development", "reserved", "tail"),
                        default="development")
    args = parser.parse_args()
    plan, plan_sha = checked_plan()
    checkpoint_id = str(args.checkpoint_id)
    require(checkpoint_id in [str(i) for i in plan["evaluation"]["checkpoints"]],
            "checkpoint is outside the frozen V25 evaluation schedule")
    checkpoint_path = ART / "checkpoints" / f"model_{checkpoint_id}.pt"
    checkpoint_sha = sha(checkpoint_path)
    selection = None
    if args.stage != "development":
        selection = selected_candidate(checkpoint_id, checkpoint_sha, plan_sha)
    tail_reserve = None
    if args.stage == "tail":
        seeds, tail_reserve = checked_tail_reserve(plan, plan_sha)
    else:
        seed_key = ("development_new_seeds" if args.stage == "development"
                    else "reserved_unopened_seeds")
        seeds = [int(seed) for seed in plan["evaluation"][seed_key]]
    require(bool(seeds), f"V25 {args.stage} seed family is empty")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    rows = [score_case(checkpoint_id, seed, model, plan["physical_gate"],
                       checkpoint_path) for seed in seeds]
    if tail_reserve:
        expected = {int(entry["seed"]): entry for entry in tail_reserve["entries"]}
        for row in rows:
            entry = expected[row["seed"]]
            report = json.loads((ART / f"checkpoint-{checkpoint_id}-seed{row['seed']}.json").read_text())
            require(row["initial_state_sha256"] == entry["initial_state_sha256"] and
                    abs(report["initial_predicted_sole_gap_m"] -
                        entry["predicted_sole_gap_m"]) <= 1e-6,
                    f"tail seed {row['seed']}: reset state/gap changed after screen")
    summary = {
        "cases": len(rows),
        "full_gate_passes": sum(row["full_gate"] for row in rows),
        "early_terminations": sum(row["ended_early"] for row in rows),
        "mean_return": float(np.mean([row["reward_total_unrounded"] for row in rows])),
        "mean_rated_leg_fraction": float(np.mean([row["rated_leg_fraction"]
                                                   for row in rows])),
        "observed_motor_delay_lags_across_cases": sorted(set().union(
            *(row["observed_motor_delay_lags"] for row in rows))),
        "failure_reason_counts": {reason: sum(reason in row["failure_reasons"]
                                               for row in rows)
                                  for reason in sorted({reason for row in rows
                                                        for reason in row["failure_reasons"]})},
    }
    result = {
        "schema": f"robot-learning-lab.cove50-seated-rise-learned-{args.stage}-score.v25",
        "status": f"complete_source_checked_cuda_{args.stage}_score",
        "plan_sha256": plan_sha,
        "body_sha256": sha(BODY),
        "scorer_source_sha256": sha(Path(__file__)),
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha,
        "selection_sha256": sha(ART / "selected-candidate.json") if selection else None,
        "tail_reserve_sha256": sha(ART / "tail-reserve.json") if tail_reserve else None,
        "summary": summary,
        "rows": rows,
        "claim_limit": "Fresh CUDA learned-actor simulation score on the rejected Direct Frame control sandbox; no physical robot transfer.",
    }
    output = ART / f"checkpoint-{checkpoint_id}-{args.stage}-score.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
