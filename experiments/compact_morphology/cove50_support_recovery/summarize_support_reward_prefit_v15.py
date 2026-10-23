"""Score one local Stage-A reward/physical prefit packet from saved source traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


CASES = (
    ("left", "reference"), ("left", "ppo005"), ("left", "ppo020"),
    ("right", "reference"), ("right", "ppo005"), ("right", "ppo020"),
)


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


def score_case(root: Path, model: mujoco.MjModel, side: str, label: str) -> dict:
    stem = f"{side}-{label}"
    report_path, trace_path = root / f"{stem}.json", root / f"{stem}.npz"
    report = json.loads(report_path.read_text())
    episode = report["reward_episode_preflight"]
    assert episode["side"] == side and episode["controller"] == ("reference" if label == "reference" else "ppo")
    assert episode["steps"] == len(episode["rows"]) and episode["optimizer_updates"] == 0
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
        "side": side, "controller": label,
        "initial_state_sha256": episode["initial_state_sha256"],
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
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    rows = [score_case(args.root, model, side, label) for side, label in CASES]
    comparisons = []
    for side in ("left", "right"):
        group = {row["controller"]: row for row in rows if row["side"] == side}
        assert len({row["initial_state_sha256"] for row in group.values()}) == 1
        for label in ("ppo005", "ppo020"):
            candidate = group[label]
            source = group["reference"]
            with open(args.root / f"{side}-reference.json") as f:
                source_rows = json.load(f)["reward_episode_preflight"]["rows"]
            matched_prefix_reward = sum(row["reward"] for row in source_rows[:candidate["steps"]])
            comparisons.append({"side": side, "candidate": label,
                                "same_initial_state": True,
                                "full_return_delta_candidate_minus_reference": candidate["reward_sum"] - source["reward_sum"],
                                "same_duration_prefix_reward_delta": candidate["reward_sum"] - matched_prefix_reward,
                                "physical_rescue": candidate["full_gate_pass"] and not source["full_gate_pass"],
                                "physical_regression": source["full_gate_pass"] and not candidate["full_gate_pass"]})
    result = {
        "schema": "robot-learning-lab.support-reward-prefit-v15.v1",
        "body_sha256": sha(args.body_xml),
        "scorer_sha256": sha(Path(__file__)),
        "method": "Matched local CPU 50 Hz reference/initialized-PPO episodes; full Stage-A physical gate reconstructed from unrounded trainer rows and source XML box-sole patches, with resolved dt-scaled reward terms.",
        "rows": rows, "comparisons": comparisons,
        "claim_limit": "Two opened starts, zero optimizer updates and no CUDA score. Raw-XML offline patch calculation still needs assembled-scene parity before paid fitting.",
    }
    out = args.root / "development-score.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": [{k: v for k, v in row.items() if k not in ("reward_terms_sum",)} for row in rows],
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
