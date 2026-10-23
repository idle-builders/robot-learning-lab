"""Source-check the frozen half-rise reward counterfactual against V17."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-reward-v18"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(route: str, seed: int, model: mujoco.MjModel, gate: dict) -> dict:
    report_path = ART / f"seed{seed}-{route}.json"
    arrays_path = ART / f"seed{seed}-{route}.npz"
    report = json.loads(report_path.read_text())
    samples = report["timeseries"]
    with np.load(arrays_path) as arrays:
        qpos, qvel = arrays["qpos"].copy(), arrays["qvel"].copy()
        initial_qpos, initial_qvel = arrays["initial_qpos"].copy(), arrays["initial_qvel"].copy()
        actor, raw, applied = (arrays[name].copy() for name in
                               ("actor_observation", "raw_action", "applied_action"))
        targets, reward_rate = arrays["processed_target"].copy(), arrays["reward_rate"].copy()
    assert report["seed"] == seed and report["device"] == "cpu"
    assert report["seated_rise_stage"] and report["recover_from_start"]
    assert route == "parked" and report["scripted_target_probe"]
    assert not report["zero_actions"] and report["scripted_min_blend"] == 0.5
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
    assert site_gap <= 1e-6
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
    names = report["reward_term_names"]
    contributions = dict(zip(names, (reward_rate.sum(axis=0) * 0.02).tolist()))
    first_s = samples[:50]
    last = samples[-1]
    return {
        "seed": seed, "route": route,
        "source_report": str(report_path.relative_to(ROOT)), "source_report_sha256": sha(report_path),
        "native_trajectory": str(arrays_path.relative_to(ROOT)), "native_trajectory_sha256": sha(arrays_path),
        "initial_state_sha256": report["initial_state_sha256"],
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
        "physical_gate": check, "full_gate": all(check.values()),
    }


def main() -> None:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    model_path = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    assert sha(model_path) == plan["body_sha256"]
    assert sha(ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py") == plan["trainer_source_sha256"]
    assert sha(ART / "source/evaluate_posture.py") == plan["evaluator_source_sha256"]
    gate = json.loads((ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v5/plan.json").read_text())["physical_gate"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    rows = [score("parked", seed, model, gate) for seed in plan["opened_v17_seeds"]]
    v17_art = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-native-v17"
    v17 = json.loads((v17_art / "report.json").read_text())
    assert v17["summary"]["scripted_full_gate_passes"] == 6
    scripts = {row["seed"]: row for row in v17["rows"] if row["route"] == "scripted"}
    pairs = []
    for parked in rows:
        seed = parked["seed"]
        script = scripts[seed]
        assert script["initial_state_sha256"] == parked["initial_state_sha256"]
        pairs.append({"seed": seed, "matched_initial_state": True,
                      "script_reward": script["reward_total_unrounded"],
                      "parked_reward": parked["reward_total_unrounded"],
                      "script_minus_parked_reward": script["reward_total_unrounded"] - parked["reward_total_unrounded"],
                      "script_v5_gate": script["full_gate"], "parked_v5_gate": parked["full_gate"]})
    summary = {
        "paired_independent_starts": len(pairs),
        "matched_initial_state_faults": 0,
        "scripted_full_gate_passes": sum(p["script_v5_gate"] for p in pairs),
        "parked_old_v5_gate_passes": sum(p["parked_v5_gate"] for p in pairs),
        "parked_early_failures": sum(r["ended_early"] for r in rows),
        "script_reward_exceeds_parked_pairs": sum(p["script_minus_parked_reward"] > 0 for p in pairs),
        "old_v5_gate_false_positive": any(p["parked_v5_gate"] for p in pairs),
    }
    output = {"schema": "robot-learning-lab.cove50-seated-rise-reward-score.v18",
              "plan_sha256": sha(plan_path), "body_sha256": sha(model_path),
              "scorer_source_sha256": sha(Path(__file__)),
              "status": "complete_local_cpu_opened_reward_audit",
              "summary": summary, "pairs": pairs, "rows": rows,
              "interpretation": "A half-rise parked target is safe for 10s but fails the commanded full stand. The old V5 broad tail-height band may score it as a pass; retain V5 unchanged as historical evidence and prospectively add command attainment.",
              "claim_limit": "Opened development reward and score-contract audit; not BC, PPO or independent qualification."}
    (ART / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
