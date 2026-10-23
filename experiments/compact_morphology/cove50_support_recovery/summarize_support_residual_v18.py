"""Score frozen V18 paired CUDA residual development visits from source traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sole_patch(model: mujoco.MjModel, data: mujoco.MjData,
               qpos: np.ndarray, geom_id: int) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    sx, sy, sz = model.geom_size[geom_id]
    local = np.asarray([(x, y, -sz) for x in (-sx, sx) for y in (-sy, sy)])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return local @ rotation.T + data.geom_xpos[geom_id]


def score_case(root: Path, model: mujoco.MjModel, side: str, label: str,
               seed: int, expected_checkpoint_sha256: str, fixed_lag_steps: int) -> dict:
    stem = f"seed{seed}-{label}"
    report_path, trace_path = root / f"{stem}.json", root / f"{stem}.npz"
    report = json.loads(report_path.read_text())
    episode = report["reward_episode_preflight"]
    assert episode["seed"] == seed and episode["side"] == side
    assert episode["controller"] == ("reference" if label == "reference" else "ppo")
    assert episode["steps"] == len(episode["rows"])
    assert episode["optimizer_updates"] == (0 if label == "reference" else 32)
    assert episode["checkpoint_iteration"] == (None if label == "reference" else 31)
    assert episode["checkpoint_sha256"] == (None if label == "reference" else expected_checkpoint_sha256)
    assert episode["initial_domain_fields"]
    assert set(("body_mass", "body_inertia", "body_ipos", "body_iquat",
                "geom_friction", "actuator_gainprm", "actuator_biasprm",
                "actuator_forcerange")).issubset(episode["initial_domain_fields"])
    assert episode["eval_device"].startswith("cuda") and episode["eval_cuda_device_name"]
    assert episode["eval_fixed_lag_steps"] == fixed_lag_steps
    assert episode["rollout_rng_seed"] == seed + 917133
    assert (episode["lag_buffer_count"] == 0) if fixed_lag_steps == 0 else episode["lag_buffer_count"] > 0
    assert len(episode["lag_trace"]) == episode["steps"]
    lag_hash = hashlib.sha256()
    for sample in episode["lag_trace"]:
        assert len(sample) == episode["lag_buffer_count"]
        assert all(lag == fixed_lag_steps for lag in sample)
        lag_hash.update(np.asarray(sample, dtype=np.int16).tobytes())
    assert lag_hash.hexdigest() == episode["lag_sequence_sha256"]
    with np.load(trace_path) as trace:
        initial = trace["initial_qpos"].copy()
        initial_vel = trace["initial_qvel"].copy()
        states = trace["qpos"].copy()
    assert len(states) == episode["steps"] and len(initial) == model.nq
    assert hashlib.sha256(initial.tobytes() + initial_vel.tobytes()).hexdigest() == episode["initial_state_sha256"]
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
    if geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Expected source box sole")
    data = mujoco.MjData(model)
    start = sole_patch(model, data, initial, geom_id)
    patch_slips = [float(np.linalg.norm(sole_patch(model, data, q, geom_id)[:, :2]
                                        - start[:, :2], axis=1).max()) for q in states]
    rows = episode["rows"]
    for row, patch_slip in zip(rows, patch_slips, strict=True):
        row["offline_patch_slip_m"] = patch_slip
    window = [row for row in rows if 2.0 <= row["time_s"] < 5.0]
    bad = lambda row: row["nonsole_contact"] or row["self_contact"] or row["hand_force_n"] >= 1.0
    valid_load = lambda row: (row["feet_grounded"] == [True, True]
                              and row["stance_load_fraction"] >= .75
                              and row["pelvis_z_m"] >= .20
                              and row["tilt_deg"] <= 15.0
                              and row["offline_patch_slip_m"] <= .015
                              and not bad(row))
    final = rows[-1]
    full_duration = len(rows) == 400 and final["time_s"] >= 7.99
    final_stand = (full_duration and final["feet_grounded"] == [True, True]
                   and final["pelvis_z_m"] >= .20 and final["tilt_deg"] <= 15.0)
    rated_leg = sum(row["rated_leg"] for row in rows) / len(rows)
    rated_arm = sum(row["rated_arm"] for row in rows) / len(rows)
    max_patch = max(patch_slips)
    bad_frames = sum(bool(bad(row)) for row in rows)
    fail = []
    if not full_duration: fail.append("early_termination")
    if not final_stand: fail.append("no_stable_two_foot_return")
    if len(window) != 150 or not all(valid_load(row) for row in window):
        fail.append("incomplete_2_to_5_s_load_window")
    if max_patch > .015: fail.append("sole_patch_slip_over_15mm")
    if rated_leg > .10: fail.append("rated_leg_over_10pct")
    if rated_arm > .10: fail.append("rated_arm_over_10pct")
    if bad_frames: fail.append("forbidden_contact")
    terms = episode["reward_terms_sum"]
    assert abs(sum(terms.values()) - episode["reward_sum"]) < 1e-4
    return {
        "seed": seed, "side": side, "controller": label,
        "initial_state_sha256": episode["initial_state_sha256"],
        "initial_domain_sha256": episode["initial_domain_sha256"],
        "eval_cuda_device_name": episode["eval_cuda_device_name"],
        "fixed_lag_steps": fixed_lag_steps,
        "checkpoint_sha256": episode["checkpoint_sha256"],
        "steps": len(rows), "duration_s": final["time_s"],
        "full_gate_pass": not fail, "failure_reasons": fail,
        "valid_load_frames_2_to_5_s": sum(valid_load(row) for row in window),
        "observed_load_frames_2_to_5_s": len(window),
        "min_load_fraction_2_to_5_s": min((row["stance_load_fraction"] for row in window), default=None),
        "max_patch_slip_m": max_patch,
        "rated_leg_fraction": rated_leg, "rated_arm_fraction": rated_arm,
        "forbidden_contact_frames": bad_frames,
        "reward_sum": episode["reward_sum"],
        "reward_terms_sum": terms,
        "report_sha256": sha(report_path), "trace_sha256": sha(trace_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--evaluation-method", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    method = json.loads(args.evaluation_method.read_text())
    if (method["schema"] != "robot-learning-lab.support-residual-v18.evaluation-method.v5"
            or method["evaluator_sha256"] != sha(Path(__file__).with_name("evaluate_support_residual_v18.py"))
            or method["source_body_sha256"] != sha(args.body_xml)
            or method["checkpoint_sha256"] != sha(args.checkpoint)):
        raise RuntimeError("Frozen V18 evaluation method/body mismatch")
    checkpoint_sha256 = sha(args.checkpoint)
    rows = [score_case(args.root, model, family["side"], label, family["seed"],
                       checkpoint_sha256, family["fixed_lag_steps"])
            for family in method["fresh_families"] for label in ("reference", "checkpoint")]
    comparisons = []
    for family in method["fresh_families"]:
        pair = [row for row in rows if row["seed"] == family["seed"]]
        source = next(row for row in pair if row["controller"] == "reference")
        candidate = next(row for row in pair if row["controller"] == "checkpoint")
        identity = (source["initial_state_sha256"] == candidate["initial_state_sha256"]
                    and source["initial_domain_sha256"] == candidate["initial_domain_sha256"]
                    and source["eval_cuda_device_name"] == candidate["eval_cuda_device_name"])
        if not identity:
            raise RuntimeError(f"Original pair {family['seed']} has mismatched physical or domain start")
        src_episode = json.loads((args.root / f"seed{family['seed']}-reference.json").read_text())["reward_episode_preflight"]
        cand_episode = json.loads((args.root / f"seed{family['seed']}-checkpoint.json").read_text())["reward_episode_preflight"]
        if (src_episode["eval_fixed_lag_steps"] != family["fixed_lag_steps"]
                or cand_episode["eval_fixed_lag_steps"] != family["fixed_lag_steps"]):
            raise RuntimeError(f"Original pair {family['seed']} has mismatched fixed motor lag")
        comparisons.append({"seed": family["seed"], "side": family["side"],
                            "same_initial_state_domain_and_fixed_lag": True,
                            "source_pass": source["full_gate_pass"],
                            "candidate_pass": candidate["full_gate_pass"],
                            "physical_rescue": candidate["full_gate_pass"] and not source["full_gate_pass"],
                            "physical_regression": source["full_gate_pass"] and not candidate["full_gate_pass"],
                            "reward_delta": candidate["reward_sum"] - source["reward_sum"]})
    passes = sum(row["full_gate_pass"] for row in rows if row["controller"] == "checkpoint")
    regressions = sum(pair["physical_regression"] for pair in comparisons)
    gate = method["gate"]
    result = {
        "schema": "robot-learning-lab.support-residual-v18.development-score.v1",
        "body_sha256": sha(args.body_xml),
        "scorer_sha256": sha(Path(__file__)),
        "evaluation_method_sha256": sha(args.evaluation_method),
        "checkpoint_sha256": checkpoint_sha256,
        "method": "Exact-start paired CUDA reference/reloaded-residual episodes, independent full Stage-A physical gate from unrounded rows and source XML sole corners.",
        "rows": rows, "comparisons": comparisons,
        "candidate_full_gate_passes": passes,
        "physical_regressions": regressions,
        "development_gate_pass": (len(comparisons) == gate["required_valid_pairs"]
                                  and passes >= gate["minimum_candidate_passes"]
                                  and regressions <= gate["maximum_regressions"]),
        "claim_limit": "Eight opened Direct Frame development families only. Reference-backed residual is not standalone single-foot support, step or physical transfer.",
    }
    out = args.root / "development-score.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": [{k: v for k, v in row.items() if k not in ("reward_terms_sum",)} for row in rows],
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
