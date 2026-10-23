"""Retrospective actor-visible early-warning audit on the opened V29 pairs.

This is an offline diagnosis, not a fitted controller or an independent score.
The decision-time observation at index i precedes the step ending at (i+1)*0.02 s.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-stand-recovery/seated-intervention-v29"
OUT = ROOT / "artifacts/ape-cove50-stand-recovery/seated-early-warning-v33"
SEEDS = tuple(range(2231, 2237))
ROUTES = ("model25", "scripted")
DT = 0.02
# V32b exact-task preflight lists base_ang_vel, projected_gravity, joint_pos,
# joint_vel, actions, command and rise_phase in this order. The first two are 3D.
PROJECTED_GRAVITY_X = 3
RISE_PHASE = 72


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_report_path = SOURCE / "report.json"
    source = json.loads(source_report_path.read_text())
    rows = {(row["seed"], row["route"]): row for row in source["rows"]}
    term_report_path = ROOT / "artifacts/ape-cove50-stand-recovery/seated-capacity-v32b/preflight-report.json"
    terms = json.loads(term_report_path.read_text())["actor_observation_terms"]
    expected_terms = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "rise_phase"]
    if terms != expected_terms:
        raise ValueError(f"Actor term order changed: {terms}")

    samples: dict[int, dict[str, dict]] = {}
    for seed in SEEDS:
        samples[seed] = {}
        for route in ROUTES:
            row = rows[(seed, route)]
            report_path = ROOT / row["source_report"]
            array_path = ROOT / row["native_trajectory"]
            if sha256(report_path) != row["source_report_sha256"]:
                raise ValueError(f"V29 report hash mismatch: {seed} {route}")
            if sha256(array_path) != row["native_trajectory_sha256"]:
                raise ValueError(f"V29 array hash mismatch: {seed} {route}")
            native = json.loads(report_path.read_text())
            if native["seed"] != seed or native["intervention_route"] != route or native["device"] != "cuda:0":
                raise ValueError(f"Source identity mismatch: {seed} {route}")
            with np.load(array_path) as arrays:
                actor = arrays["actor_observation"].copy()
            if actor.shape != (500, 73) or not np.isfinite(actor).all():
                raise ValueError(f"Invalid actor observations: {seed} {route} {actor.shape}")
            grounded = [all(step["feet_grounded"]) for step in native["timeseries"]]
            anchor = next((i for i in range(len(grounded) - 4)
                           if all(grounded[i:i + 5])), None)
            if anchor is None:
                raise ValueError(f"No five-frame double-support anchor: {seed} {route}")
            loss = next((i for i in range(anchor + 5, len(grounded))
                         if not grounded[i]), None)
            gap = float(native["timeseries"][loss]["time_s"]) if loss is not None else None
            samples[seed][route] = {
                "actor": actor,
                "first_post_anchor_support_gap_s": gap,
                "source_report_sha256": row["source_report_sha256"],
                "native_trajectory_sha256": row["native_trajectory_sha256"],
            }

    if any(samples[seed]["model25"]["first_post_anchor_support_gap_s"] is None or
           samples[seed]["scripted"]["first_post_anchor_support_gap_s"] is not None
           for seed in SEEDS):
        raise ValueError("V29 physical risk labels differ from the opened pair verdict")

    times = []
    for index in range(7):
        risks = [float(samples[seed]["model25"]["actor"][index, PROJECTED_GRAVITY_X]) for seed in SEEDS]
        safes = [float(samples[seed]["scripted"]["actor"][index, PROJECTED_GRAVITY_X]) for seed in SEEDS]
        predictions = []
        thresholds = []
        for held_seed in SEEDS:
            training = [seed for seed in SEEDS if seed != held_seed]
            highest_risk = max(float(samples[seed]["model25"]["actor"][index, PROJECTED_GRAVITY_X]) for seed in training)
            lowest_safe = min(float(samples[seed]["scripted"]["actor"][index, PROJECTED_GRAVITY_X]) for seed in training)
            threshold = (highest_risk + lowest_safe) / 2.0
            thresholds.append(threshold)
            for route, expected_risk in (("model25", True), ("scripted", False)):
                observed_risk = float(samples[held_seed][route]["actor"][index, PROJECTED_GRAVITY_X]) < threshold
                predictions.append(observed_risk == expected_risk)
        phases = [float(samples[seed][route]["actor"][index, RISE_PHASE]) for seed in SEEDS for route in ROUTES]
        phase_predictions = []
        for held_seed in SEEDS:
            training = [seed for seed in SEEDS if seed != held_seed]
            risk_mean = float(np.mean([samples[seed]["model25"]["actor"][index, RISE_PHASE] for seed in training]))
            safe_mean = float(np.mean([samples[seed]["scripted"]["actor"][index, RISE_PHASE] for seed in training]))
            for route, expected_risk in (("model25", True), ("scripted", False)):
                value = float(samples[held_seed][route]["actor"][index, RISE_PHASE])
                observed_risk = abs(value - risk_mean) < abs(value - safe_mean)
                phase_predictions.append(observed_risk == expected_risk)
        times.append({
            "decision_state_time_s": round(index * DT, 2),
            "risk_gravity_x_range": [min(risks), max(risks)],
            "planted_gravity_x_range": [min(safes), max(safes)],
            "gravity_ranges_separated": max(risks) < min(safes),
            "leave_one_seed_whole_out_correct_of_12": sum(predictions),
            "threshold_range_from_five_training_families": [min(thresholds), max(thresholds)],
            "phase_range_both_routes": [min(phases), max(phases)],
            "phase_pair_max_abs_difference": max(abs(float(samples[seed]["model25"]["actor"][index, RISE_PHASE]) -
                                                 float(samples[seed]["scripted"]["actor"][index, RISE_PHASE]))
                                             for seed in SEEDS),
            "phase_only_nearest_centroid_left_out_correct_of_12": sum(phase_predictions),
        })

    output = {
        "schema": "robot-learning-lab.cove50-seated-early-warning.v33.offline-diagnostic",
        "source_v29_report_sha256": sha256(source_report_path),
        "source_v32b_preflight_sha256": sha256(term_report_path),
        "opened_seed_families": list(SEEDS),
        "paired_cuda_trajectories": 12,
        "source_hashes_verified": 24,
        "observation": "Actual corruption-enabled 73-value actor input before each 50 Hz action; index 3 is projected gravity x. No critic/contact/force channel is used by the threshold.",
        "label": "Uninterrupted V29 model25 loses post-anchor double support; matched planted script does not. This labels route outcomes, not unseen disturbances or a deployable intervention.",
        "first_post_anchor_gap_s_by_seed": {str(seed): samples[seed]["model25"]["first_post_anchor_support_gap_s"] for seed in SEEDS},
        "times": times,
        "result": "At 0.06 s, projected-gravity-x separates all six actor-loss trajectories from six matched planted references on already opened seeds; the five-family threshold classifies both routes in each left-out family. Phase-only nearest-centroid classification gets 7/12 and its ranges overlap.",
        "limits": "Posthoc feature/time choice on six opened route pairs. The script and actor take different actions; this is not a causal correction, unseen-seed prediction, complete feedback teacher, learned stand or permission for PPO fitting. Test a frozen trigger on disjoint native starts and require full physical support gates before use.",
        "optimizer_updates": 0,
        "gpu_cost_usd": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    matrix = {
        "schema": "robot-learning-lab.cove50-seated-early-warning.v33.inspector-matrix",
        "source_report": "artifacts/ape-cove50-stand-recovery/seated-early-warning-v33/report.json",
        "claim_limit": output["limits"],
        "cases": [{
            "time_s": row["decision_state_time_s"],
            "gravity_correct_of_12": row["leave_one_seed_whole_out_correct_of_12"],
            "phase_correct_of_12": row["phase_only_nearest_centroid_left_out_correct_of_12"],
            "actor_gravity_x_min": row["risk_gravity_x_range"][0],
            "actor_gravity_x_max": row["risk_gravity_x_range"][1],
            "script_gravity_x_min": row["planted_gravity_x_range"][0],
            "script_gravity_x_max": row["planted_gravity_x_range"][1],
        } for row in times],
    }
    (OUT / "inspector-matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": output["result"], "t0p06": times[3], "output": str(OUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
