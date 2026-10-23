"""Score a V19 upright release visit from its native report and qpos trace.

Training reward is diagnostic. The verdict comes from the whole physical
episode: actual two-foot load, sole-corner clearance, exclusive support,
stance patch motion, motor duty, forbidden contact, and two-foot return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sole_corners(model: mujoco.MjModel, data: mujoco.MjData,
                 qpos: np.ndarray, side: str) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
    if geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"Missing box sole: {side}")
    sx, sy, sz = model.geom_size[geom_id]
    local = np.asarray([(x, y, -sz) for x in (-sx, sx) for y in (-sy, sy)])
    return local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]


def longest_run(flags: list[bool]) -> int:
    longest = current = 0
    for valid in flags:
        current = current + 1 if valid else 0
        longest = max(longest, current)
    return longest


def score(report_path: Path, trace_path: Path, body_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    episode = report["reward_episode_preflight"]
    if sha(body_path) != report["body_sha256"]:
        raise RuntimeError("Report body and scoring body differ")
    if episode["steps"] != len(episode["rows"]):
        raise RuntimeError("Report step count differs from source rows")
    if abs(sum(episode["reward_terms_sum"].values()) - episode["reward_sum"]) > 1e-4:
        raise RuntimeError("Reward terms do not reconcile")
    with np.load(trace_path) as trace:
        initial = trace["initial_qpos"].copy()
        initial_vel = trace["initial_qvel"].copy()
        states = trace["qpos"].copy()
    if len(states) != episode["steps"]:
        raise RuntimeError("Trace and source report lengths differ")
    state_hash = hashlib.sha256(initial.tobytes() + initial_vel.tobytes()).hexdigest()
    if state_hash != episode["initial_state_sha256"]:
        raise RuntimeError("Initial physical state hash differs")
    model = mujoco.MjModel.from_xml_path(str(body_path))
    data = mujoco.MjData(model)
    side = episode["side"]
    swing = "left" if side == "right" else "right"
    start_patch = sole_corners(model, data, initial, side)
    rows = episode["rows"]
    release_flags = []
    load_flags = []
    max_slip = 0.0
    min_swing_corner = None
    for row, qpos in zip(rows, states, strict=True):
        stance_patch = sole_corners(model, data, qpos, side)
        swing_patch = sole_corners(model, data, qpos, swing)
        patch_slip = float(np.linalg.norm(
            stance_patch[:, :2] - start_patch[:, :2], axis=1).max())
        max_slip = max(max_slip, patch_slip)
        clearance = float(swing_patch[:, 2].min())
        min_swing_corner = clearance if min_swing_corner is None else min(min_swing_corner, clearance)
        bad = (row["nonsole_contact"] or row["self_contact"]
               or row["hand_force_n"] >= 1.0)
        upright = row["pelvis_z_m"] >= .20 and row["tilt_deg"] <= 15
        safe = upright and patch_slip <= .015 and not bad
        load_flags.append(bool(1.8 <= row["time_s"] <= 2.2 and safe
                               and row["feet_grounded"] == [True, True]
                               and row["stance_load_fraction"] >= .75))
        release_flags.append(bool(2.0 <= row["time_s"] < 5.0 and safe
                                  and row["feet_grounded"] ==
                                  ([False, True] if side == "right" else [True, False])
                                  and clearance >= .025))
    load_attained = longest_run(load_flags) >= 10
    run_frames = longest_run(release_flags)
    full_length = len(rows) == 400 and rows[-1]["time_s"] >= 7.99
    final = rows[-1]
    returned = bool(full_length and final["feet_grounded"] == [True, True]
                    and final["pelvis_z_m"] >= .20 and final["tilt_deg"] <= 15
                    and final["returned_two_foot"])
    leg_duty = sum(row["rated_leg"] for row in rows) / len(rows)
    arm_duty = sum(row["rated_arm"] for row in rows) / len(rows)
    bad_frames = sum(row["nonsole_contact"] or row["self_contact"]
                     or row["hand_force_n"] >= 1.0 for row in rows)
    failures = []
    if not load_attained: failures.append("no_attained_two_foot_load")
    if run_frames < 25: failures.append("no_half_second_clean_release")
    if run_frames < 100: failures.append("no_two_second_clean_release")
    if not returned: failures.append("no_two_foot_return")
    if max_slip > .015: failures.append("stance_patch_slip_over_15mm")
    if leg_duty > .10: failures.append("rated_leg_over_10pct")
    if arm_duty > .10: failures.append("rated_arm_over_10pct")
    if bad_frames: failures.append("forbidden_contact")
    return {
        "schema": "robot-learning-lab.cove50-release-v19.native-case.v1",
        "claim": "Source-labelled Direct Frame basic control only",
        "body_sha256": sha(body_path),
        "report_sha256": sha(report_path), "trace_sha256": sha(trace_path),
        "scorer_sha256": sha(Path(__file__)),
        "seed": episode["seed"], "side": side,
        "controller": episode["controller"],
        "checkpoint_sha256": episode["checkpoint_sha256"],
        "initial_state_sha256": state_hash,
        "initial_domain_sha256": episode["initial_domain_sha256"],
        "eval_device": episode["eval_device"],
        "steps": len(rows), "duration_s": final["time_s"],
        "load_attained": load_attained,
        "longest_clean_release_s": round(run_frames * .02, 3),
        "half_second_release": load_attained and run_frames >= 25,
        "two_second_release_and_return": load_attained and run_frames >= 100 and returned,
        "full_gate_pass": not failures,
        "failure_reasons": failures,
        "max_stance_patch_slip_m": max_slip,
        "min_swing_corner_z_m": min_swing_corner,
        "rated_leg_fraction": leg_duty,
        "rated_arm_fraction": arm_duty,
        "forbidden_contact_frames": bad_frames,
        "reward_sum": episode["reward_sum"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.report, args.trace, args.body_xml)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
