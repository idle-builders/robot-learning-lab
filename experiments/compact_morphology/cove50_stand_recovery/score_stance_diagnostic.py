"""Source-check the opened-seed V4 seated-to-stand stance diagnostic.

This is a diagnostic of the previously frozen 5 cm root-drift failure, not a
new recovery score or an accepted imitation dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v4"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
PRIOR = ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v2"
SEEDS = (2203, 2204, 2205)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def largest_norm(x: np.ndarray, axis: int = -1) -> float:
    return float(np.max(np.linalg.norm(x, axis=axis)))


def main() -> None:
    plan_path = ARTIFACT / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["opened_development_seeds"] == list(SEEDS)
    assert sha(BODY) == plan["source_sha256"][str(BODY.relative_to(ROOT))]
    model = mujoco.MjModel.from_xml_path(str(BODY))
    data = mujoco.MjData(model)
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_sole_center") for side in ("left", "right")]
    geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole") for side in ("left", "right")]
    assert min(site_ids + geom_ids) >= 0
    rows = []
    for seed in SEEDS:
        stem = f"seed{seed}-ramp2p0"
        source_path = ARTIFACT / f"{stem}.json"
        native_path = ARTIFACT / f"{stem}.npz"
        prior_path = PRIOR / f"seed{seed}-ramped-target.json"
        source = json.loads(source_path.read_text())
        prior = json.loads(prior_path.read_text())
        with np.load(native_path) as arrays:
            qpos = arrays["qpos"].copy()
            qvel = arrays["qvel"].copy()
            raw_action = arrays["raw_action"].copy()
            applied_action = arrays["applied_action"].copy()
            processed_target = arrays["processed_target"].copy()
            initial_qpos = arrays["initial_qpos"].copy()
            initial_qvel = arrays["initial_qvel"].copy()
        trace = source["timeseries"]
        assert qpos.shape == (500, model.nq) and qvel.shape == (500, model.nv)
        assert raw_action.shape == applied_action.shape == processed_target.shape == (500, 21)
        assert np.array_equal(initial_qpos, source["initial_qpos"])
        assert np.array_equal(initial_qvel, source["initial_qvel"])
        assert source["initial_state_sha256"] == prior["initial_state_sha256"]
        assert len(trace) == len(prior["timeseries"]) == 500
        # The new arrays and site fields may be added, but the old run's
        # physical and action timeseries must remain exactly unchanged.
        for new, old in zip(trace, prior["timeseries"], strict=True):
            assert all(new[key] == value for key, value in old.items())
        assert source["final_hold"] == prior["final_hold"]

        site_xy, corner_xy = [], []
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            site_xy.append(data.site_xpos[site_ids, :2].copy())
            patches = []
            for geom_id in geom_ids:
                half_x, half_y = model.geom_size[geom_id, :2]
                local = np.array([[x * half_x, y * half_y, 0.0] for x in (-1, 1) for y in (-1, 1)])
                orientation = data.geom_xmat[geom_id].reshape(3, 3)
                patches.append(data.geom_xpos[geom_id, :2] + (local @ orientation.T)[:, :2])
            corner_xy.append(patches)
        site_xy = np.asarray(site_xy)
        corner_xy = np.asarray(corner_xy)
        recorded_sites = np.asarray([sample["sole_site_xy_m"] for sample in trace])
        parity_gap = float(np.max(np.abs(site_xy - recorded_sites)))
        assert parity_gap <= 1e-6
        both = np.asarray([all(sample["feet_grounded"]) for sample in trace])
        # First sustained contact: five successive 50 Hz frames, frozen as
        # the operational anchor for this previously specified diagnostic.
        anchor = next((index for index in range(len(both) - 4) if both[index : index + 5].all()), None)
        assert anchor is not None
        site_motion = site_xy[anchor:] - site_xy[anchor]
        corner_motion = corner_xy[anchor:] - corner_xy[anchor]
        root_xy = qpos[:, :2]
        sole_relative_to_root = site_xy - root_xy[:, None, :]
        foot_locked_root_xy = root_xy[anchor] + np.mean(
            sole_relative_to_root[anchor] - sole_relative_to_root, axis=1
        )
        root_prediction_gap = root_xy[anchor:] - foot_locked_root_xy[anchor:]
        max_site_motion = largest_norm(site_motion)
        max_prediction_gap = largest_norm(root_prediction_gap)
        after_anchor_contact_gaps = int(np.count_nonzero(~both[anchor:]))
        no_fall_or_prohibited_support = (
            not source["ended_early"]
            and all(not sample["body_contact"] and sample["hand_force_n"] < 1 for sample in trace)
            and source["final_hold"]["max_tilt_deg"] < 20
            and 0.19 <= source["final_hold"]["mean_pelvis_z_m"] <= 0.26
        )
        frozen_interpretation_pass = (
            max_site_motion <= 0.015
            and max_prediction_gap <= 0.015
            and after_anchor_contact_gaps == 0
            and no_fall_or_prohibited_support
        )
        row = {
            "seed": seed,
            "source_report": str(source_path.relative_to(ROOT)),
            "source_report_sha256": sha(source_path),
            "native_trajectory": str(native_path.relative_to(ROOT)),
            "native_trajectory_sha256": sha(native_path),
            "prior_v2_report_sha256": sha(prior_path),
            "initial_state_sha256": source["initial_state_sha256"],
            "old_trace_fields_exactly_identical_to_v2": True,
            "offline_site_vs_recorded_max_abs_m": parity_gap,
            "first_sustained_both_contact_time_s": trace[anchor]["time_s"],
            "first_sustained_both_contact_frame": anchor,
            "both_contact_gap_frames_after_anchor": after_anchor_contact_gaps,
            "reset_to_anchor_site_shift_m": np.linalg.norm(site_xy[anchor] - source["initial_sole_site_xy_m"], axis=1).tolist(),
            "site_center_max_displacement_after_anchor_m": np.max(np.linalg.norm(site_motion, axis=2), axis=0).tolist(),
            "site_center_final_displacement_after_anchor_m": np.linalg.norm(site_motion[-1], axis=1).tolist(),
            "sole_box_corner_max_displacement_after_anchor_m": np.max(np.linalg.norm(corner_motion, axis=3), axis=(0, 2)).tolist(),
            "root_displacement_after_anchor_m": float(np.linalg.norm(root_xy[-1] - root_xy[anchor])),
            "foot_locked_predicted_root_displacement_after_anchor_m": float(np.linalg.norm(foot_locked_root_xy[-1] - root_xy[anchor])),
            "root_minus_prediction_max_m": max_prediction_gap,
            "root_minus_prediction_final_m": float(np.linalg.norm(root_xy[-1] - foot_locked_root_xy[-1])),
            "original_5cm_gate_final_root_drift_m": float(np.linalg.norm(root_xy[-1] - initial_qpos[:2])),
            "original_5cm_gate_pass": bool(prior["seed"] == seed and np.linalg.norm(root_xy[-1] - initial_qpos[:2]) <= 0.05),
            "no_fall_or_prohibited_support": no_fall_or_prohibited_support,
            "frozen_diagnostic_interpretation_pass": frozen_interpretation_pass,
            "initial_actor_command_x": trace[0]["actor_observed_command_x"],
            "initial_internal_target_blend": trace[0]["target_blend"],
        }
        rows.append(row)
    report = {
        "schema": "robot-learning-lab.cove50-seated-stance-diagnostic-report.v4",
        "status": "completed_opened_seed_diagnostic_not_recovery_score",
        "plan": str(plan_path.relative_to(ROOT)),
        "plan_sha256": sha(plan_path),
        "source_snapshot_sha256": sha(ARTIFACT / "source/evaluate_posture.py"),
        "scorer_source_sha256": sha(Path(__file__)),
        "body_sha256": sha(BODY),
        "runtime": "local CPU native MJLab, 50 Hz; zero paid GPU time",
        "interpretation": "Sole centers and sole box corners move far less than the pelvis. The prior 5 cm absolute-root gate remains failed on 2/3; V4 changes no score and accepts no teacher.",
        "caveat": "Foot-locked root prediction is a kinematic decomposition of the same qpos trajectory; its residual equals average sole-site movement, so it is a consistency measure, not independent model validation. Only opened seeds were measured. Internal target blend is not actor-observed.",
        "summary": {
            "opened_seeds": len(rows),
            "frozen_diagnostic_interpretation_passes": sum(row["frozen_diagnostic_interpretation_pass"] for row in rows),
            "original_5cm_gate_passes": sum(row["original_5cm_gate_pass"] for row in rows),
            "max_site_center_displacement_after_anchor_m": max(max(row["site_center_max_displacement_after_anchor_m"]) for row in rows),
            "max_sole_box_corner_displacement_after_anchor_m": max(max(row["sole_box_corner_max_displacement_after_anchor_m"]) for row in rows),
            "max_root_minus_prediction_m": max(row["root_minus_prediction_max_m"] for row in rows),
        },
        "rows": rows,
        "withheld_score_seeds_still_unopened": plan["withheld_score_seeds"],
        "learned_policy_claim": False,
        "teacher_dataset_accepted": False,
    }
    (ARTIFACT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
