"""Source-check the opened V26 CUDA scripted/reference feasibility pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
DT = 0.02


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def score(seed: int, route: str, model: mujoco.MjModel,
          gate: dict, source_hash: str) -> tuple[dict, dict]:
    stem = ART / f"seed{seed}-{route}"
    report_path, arrays_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    report = json.loads(report_path.read_text())
    samples = report["timeseries"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in (
            "initial_qpos", "initial_qvel", "actor_observation", "raw_action",
            "applied_action", "processed_target", "qpos", "qvel", "reward_rate"
        )}
    n = len(samples)
    require(0 < n <= 500 and all(len(values[key]) == n for key in (
        "actor_observation", "raw_action", "applied_action",
        "processed_target", "qpos", "qvel", "reward_rate")),
        f"{seed} {route}: native array length mismatch")
    require(report["seed"] == seed and report["device"] == "cuda:0"
            and report["body_sha256"] == source_hash
            and report["task"] == "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved"
            and report["seated_rise_stage"] and report["recover_from_start"]
            and report["randomized_physics"] and report["env_cfg_play"] is False
            and report["actor_observation_corruption"] is True
            and report["scripted_target_probe"] == (route == "scripted")
            and not report["zero_actions"] and report["scripted_min_blend"] is None,
            f"{seed} {route}: wrong native task/source/route")
    require(report["initial_motors"]["delay_lag_steps"] is not None
            and len(report["initial_motors"]["delay_lag_steps"]) == 21,
            f"{seed} {route}: missing exact lag vector")
    require(values["initial_qpos"].shape == (model.nq,)
            and values["initial_qvel"].shape == (model.nv,)
            and values["qpos"].shape == (n, model.nq)
            and values["qvel"].shape == (n, model.nv)
            and values["actor_observation"].shape == (n, 73)
            and all(values[key].shape == (n, 21) for key in (
                "raw_action", "applied_action", "processed_target"))
            and values["reward_rate"].shape == (n, 15),
            f"{seed} {route}: native channel shape mismatch")
    require(all(np.isfinite(value).all() for value in values.values()),
            f"{seed} {route}: nonfinite native array")
    require(np.array_equal(values["initial_qpos"], report["initial_qpos"])
            and np.array_equal(values["initial_qvel"], report["initial_qvel"]),
            f"{seed} {route}: source report/array start mismatch")
    start_hash = hashlib.sha256(values["initial_qpos"].tobytes() +
                                values["initial_qvel"].tobytes()).hexdigest()
    require(start_hash == report["initial_state_sha256"],
            f"{seed} {route}: initial state SHA mismatch")
    require(hashlib.sha256(values["raw_action"][0].tobytes()).hexdigest() ==
            report["first_action_sha256"],
            f"{seed} {route}: first action SHA mismatch")
    require(abs(report["duration_s"] - n * DT) <= 1e-6 and all(
        abs(sample["time_s"] - (i + 1) * DT) <= 1e-6
        for i, sample in enumerate(samples)),
        f"{seed} {route}: timebase mismatch")
    reward = values["reward_rate"].sum(axis=1) * DT
    reward_gap = float(np.max(np.abs(reward - [s["step_reward"] for s in samples])))
    require(reward_gap <= 5.1e-5, f"{seed} {route}: reward reconstruction fault")
    height_gap = float(np.max(np.abs(values["qpos"][:, 2] -
                                     [s["pelvis_z_m"] for s in samples])))
    require(height_gap <= 5.1e-5, f"{seed} {route}: qpos height parity fault")

    data = mujoco.MjData(model)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                              f"{side}_sole_center") for side in ("left", "right")]
    soles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"{side}_sole") for side in ("left", "right")]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    require(min(*sites, *soles, floor) >= 0, "body lacks a physical endpoint")
    site_xy, corner_xy, floor_body = [], [], []
    for q, v in zip(values["qpos"], values["qvel"]):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        site_xy.append(data.site_xpos[sites, :2].copy())
        floor_body.append(any(
            (c.geom1 == floor and c.geom2 not in soles) or
            (c.geom2 == floor and c.geom1 not in soles)
            for c in data.contact[:data.ncon]
        ))
        corners = []
        for geom in soles:
            half_x, half_y = model.geom_size[geom, :2]
            local = np.array([[x * half_x, y * half_y, 0.0]
                              for x in (-1, 1) for y in (-1, 1)])
            rotation = data.geom_xmat[geom].reshape(3, 3)
            corners.append(data.geom_xpos[geom, :2] +
                           (local @ rotation.T)[:, :2])
        corner_xy.append(corners)
    site_xy, corner_xy = np.asarray(site_xy), np.asarray(corner_xy)
    site_gap = float(np.max(np.abs(site_xy -
                    np.asarray([s["sole_site_xy_m"] for s in samples]))))
    require(site_gap <= 5e-5, f"{seed} {route}: sole site parity fault")
    require(all(not hit or sample["floor_nonsole_body_contact"]
                for hit, sample in zip(floor_body, samples)),
            f"{seed} {route}: missing floor-body contact in source report")

    both = np.array([all(s["feet_grounded"]) for s in samples])
    anchor = next((i for i in range(max(0, n - 4)) if both[i:i + 5].all()), None)
    anchor_s = samples[anchor]["time_s"] if anchor is not None else None
    gaps = int(np.count_nonzero(~both[anchor:])) if anchor is not None else None
    center_motion = (float(np.max(np.linalg.norm(site_xy[anchor:] - site_xy[anchor],
                                                axis=2))) if anchor is not None else None)
    patch_motion = (float(np.max(np.linalg.norm(corner_xy[anchor:] - corner_xy[anchor],
                                               axis=3))) if anchor is not None else None)
    tail = [s for s in samples if gate["tail_window_s"][0] <= s["time_s"] <=
            gate["tail_window_s"][1]]
    tail_height = float(np.mean([s["pelvis_z_m"] for s in tail])) if tail else None
    rated = float(np.mean([s["leg_torque_saturated"] for s in samples]))
    checks = {
        "full_duration": n == 500 and not report["ended_early"],
        "tail_upright": bool(tail) and max(s["tilt_deg"] for s in tail) < gate["tail_tilt_lt_deg"],
        "tail_height": tail_height is not None and gate["tail_pelvis_z_m"][0] <= tail_height <= gate["tail_pelvis_z_m"][1],
        "tail_double_support": bool(tail) and np.mean([all(s["feet_grounded"]) for s in tail]) >= gate["tail_both_feet_fraction_gte"],
        "anchor_by_0p2s": anchor_s is not None and anchor_s <= gate["first_five_consecutive_both_contact_by_s"],
        "continuous_double_support_after_anchor": gaps == 0,
        "sole_center_stays": center_motion is not None and center_motion <= gate["sole_center_max_displacement_after_anchor_m_lte"],
        "sole_patch_stays": patch_motion is not None and patch_motion <= gate["sole_box_corner_max_displacement_after_anchor_m_lte"],
        "no_hand_body_contact": not any(floor_body) and not any(s["body_contact"] or s["hand_force_n"] >= 1 for s in samples),
        "rated_leg_fraction": rated <= gate["rated_leg_fraction_lte"],
        "commanded_stand_height": tail_height is not None and tail_height >= gate["commanded_stand_height_m_gte"],
    }
    checks = {name: bool(ok) for name, ok in checks.items()}
    reach_time = next((s["time_s"] for s in samples
                       if s["pelvis_z_m"] >= gate["commanded_stand_height_m_gte"]), None)
    row = {
        "seed": seed, "route": route,
        "source_report": str(report_path.relative_to(ROOT)),
        "source_report_sha256": sha(report_path),
        "native_trajectory": str(arrays_path.relative_to(ROOT)),
        "native_trajectory_sha256": sha(arrays_path),
        "initial_state_sha256": start_hash,
        "duration_s": report["duration_s"],
        "first_height_attainment_s": reach_time,
        "tail_pelvis_z_m": tail_height,
        "anchor_s": anchor_s,
        "post_anchor_double_support_gap_frames": gaps,
        "post_anchor_sole_center_motion_m": center_motion,
        "post_anchor_sole_patch_motion_m": patch_motion,
        "rated_leg_fraction": rated,
        "episode_reward": float(reward.sum()),
        "reward_term_contributions": dict(zip(
            report["reward_term_names"],
            (values["reward_rate"].sum(axis=0) * DT).tolist())),
        "source_parity": {"site_max_abs_m": site_gap,
                          "height_max_abs_m": height_gap,
                          "reward_max_abs": reward_gap},
        "physical_gate": checks,
        "failure_reasons": [name for name, ok in checks.items() if not ok],
        "full_gate": all(checks.values()),
    }
    return row, values


def main() -> None:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(sha(Path(__file__)) == plan["source_sha256"]["scorer"],
            "V26 scorer differs from freeze")
    require(sha(BODY) == plan["source_sha256"]["body"] and
            sha(V25 / "plan.json") == plan["v25_plan_sha256"],
            "source body or V25 plan differs")
    ledger_path = ART / "source-pair-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    require(ledger["plan_sha256"] == sha(plan_path)
            and [row["seed"] for row in ledger["rows"]] ==
            plan["opened_development_seeds"],
            "paired CUDA source ledger incomplete or differs from freeze")
    gate = json.loads((V25 / "plan.json").read_text())["physical_gate"]
    model = mujoco.MjModel.from_xml_path(str(BODY))
    rows, repeat = [], []
    for seed in plan["opened_development_seeds"]:
        learned, learned_arrays = score(seed, "model25", model, gate,
                                        plan["source_sha256"]["body"])
        scripted, _ = score(seed, "scripted", model, gate,
                            plan["source_sha256"]["body"])
        old_report = json.loads((V25 / f"checkpoint-25-seed{seed}.json").read_text())
        with np.load(V25 / f"checkpoint-25-seed{seed}.npz", allow_pickle=False) as old:
            qpos_gap = (float(np.max(np.abs(learned_arrays["qpos"] - old["qpos"])))
                        if learned_arrays["qpos"].shape == old["qpos"].shape else None)
            action_gap = (float(np.max(np.abs(learned_arrays["raw_action"] - old["raw_action"])))
                          if learned_arrays["raw_action"].shape == old["raw_action"].shape else None)
        tolerance = plan["model25_original_repeat_tolerance"]
        require(qpos_gap is not None and qpos_gap <= tolerance["qpos_max_abs_m_or_rad"]
                and action_gap is not None and
                action_gap <= tolerance["raw_action_max_abs"],
                f"{seed}: measurement-only V26 evaluator changed the learned V25 trajectory")
        require(learned["initial_state_sha256"] == scripted["initial_state_sha256"]
                == old_report["initial_state_sha256"],
                f"{seed}: paired/original initial state differs")
        pair_ledger = next(row for row in ledger["rows"] if row["seed"] == seed)
        require(pair_ledger["model25_report_sha256"] == learned["source_report_sha256"]
                and pair_ledger["scripted_report_sha256"] == scripted["source_report_sha256"],
                f"{seed}: pair ledger source differs")
        repeat.append({"seed": seed, "model25_vs_original_qpos_max_abs": qpos_gap,
                       "model25_vs_original_action_max_abs": action_gap})
        rows.extend((learned, scripted))
    scripted_rows = [row for row in rows if row["route"] == "scripted"]
    passes = sum(row["full_gate"] for row in scripted_rows)
    decision = ("no_cuda_planted_reference_stop_fit" if passes == 0 else
                "insufficient_reference_branch_diagnosis_no_fit" if passes <= 3 else
                "reference_feasible_development_only_v26_method_may_be_frozen")
    result = {
        "schema": "robot-learning-lab.cove50-seated-cuda-reference-feasibility.v26",
        "status": "complete_source_checked_cuda_opened_development_diagnostic",
        "plan_sha256": sha(plan_path),
        "body_sha256": sha(BODY),
        "source_pair_ledger_sha256": sha(ledger_path),
        "summary": {
            "opened_seed_families": len(scripted_rows),
            "scripted_planted_passes": passes,
            "scripted_upright_survivals": sum(
                row["physical_gate"]["full_duration"] and
                row["physical_gate"]["tail_upright"]
                for row in scripted_rows),
            "model25_repeat_planted_passes": sum(row["full_gate"] for row in rows
                                                 if row["route"] == "model25"),
            "decision": decision,
        },
        "rows": rows,
        "original_v25_repeat_parity": repeat,
        "claim_limit": "Opened CUDA development teacher feasibility, not held score or learned-skill qualification.",
    }
    output = ART / "report.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
