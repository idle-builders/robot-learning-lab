"""Score frozen V9 phase diagnostic against its opened V5 seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-stand-recovery/phase-alignment-v9"
BASELINE = ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v5"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = ARTIFACT / "plan.json"
    plan = json.loads(plan_path.read_text())
    baseline_plan = json.loads((BASELINE / "plan.json").read_text())
    baseline_report = json.loads((BASELINE / "report.json").read_text())
    baseline_rows = {row["seed"]: row for row in baseline_report["rows"]}
    body = ROOT / baseline_plan["body"]
    assert digest(body) == plan["body_sha256"]
    assert digest(ARTIFACT / "source/evaluate_posture.py") == plan["source_snapshot_sha256"]
    model = mujoco.MjModel.from_xml_path(str(body))
    data = mujoco.MjData(model)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_sole_center") for side in ("left", "right")]
    geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole") for side in ("left", "right")]
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    sole_geoms = set(geoms)
    gate = baseline_plan["physical_gate"]
    rows = []
    for seed in plan["opened_v5_score_seeds"]:
        report_path = ARTIFACT / f"seed{seed}.json"
        native_path = ARTIFACT / f"seed{seed}.npz"
        report = json.loads(report_path.read_text())
        samples = report["timeseries"]
        with np.load(native_path) as arrays:
            qpos = arrays["qpos"].copy()
            qvel = arrays["qvel"].copy()
            obs = arrays["actor_observation"].copy()
            raw = arrays["raw_action"].copy()
            applied = arrays["applied_action"].copy()
            targets = arrays["processed_target"].copy()
            initial_qpos = arrays["initial_qpos"].copy()
            initial_qvel = arrays["initial_qvel"].copy()
        assert report["body_sha256"] == plan["body_sha256"]
        assert report["seed"] == seed and report["posture_ramp_s"] == 2.0
        assert report["initial_blend_override"] == 1.0
        assert report["initial_state_sha256"] == baseline_rows[seed]["initial_state_sha256"]
        assert report["device"] == "cpu" and report["recover_from_start"] and report["scripted_target_probe"]
        assert len(samples) == len(qpos) == len(qvel) == len(obs) == len(raw) == len(applied) == len(targets) == 500
        assert obs.shape[1] == 72 and raw.shape[1] == applied.shape[1] == targets.shape[1] == 21
        assert np.array_equal(initial_qpos, report["initial_qpos"])
        assert np.array_equal(initial_qvel, report["initial_qvel"])

        site_xy = []
        corner_xy = []
        floor_nonsole_contact = []
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            floor_nonsole_contact.append(any(
                (contact.geom1 == floor_geom and contact.geom2 not in sole_geoms)
                or (contact.geom2 == floor_geom and contact.geom1 not in sole_geoms)
                for contact in data.contact[: data.ncon]
            ))
            site_xy.append(data.site_xpos[sites, :2].copy())
            patches = []
            for geom_id in geoms:
                half_x, half_y = model.geom_size[geom_id, :2]
                local = np.array([[x * half_x, y * half_y, 0.0] for x in (-1, 1) for y in (-1, 1)])
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                patches.append(data.geom_xpos[geom_id, :2] + (local @ rotation.T)[:, :2])
            corner_xy.append(patches)
        site_xy = np.asarray(site_xy)
        corner_xy = np.asarray(corner_xy)
        site_parity = float(np.max(np.abs(site_xy - np.asarray([s["sole_site_xy_m"] for s in samples]))))
        assert site_parity <= 1e-6
        both = np.asarray([all(s["feet_grounded"]) for s in samples])
        anchor = next((i for i in range(496) if both[i : i + 5].all()), None)
        anchor_time = samples[anchor]["time_s"] if anchor is not None else None
        site_movement = float(np.max(np.linalg.norm(site_xy[anchor:] - site_xy[anchor], axis=2))) if anchor is not None else None
        corner_movement = float(np.max(np.linalg.norm(corner_xy[anchor:] - corner_xy[anchor], axis=3))) if anchor is not None else None
        gaps = int(np.count_nonzero(~both[anchor:])) if anchor is not None else None
        reset_to_anchor_m = np.linalg.norm(site_xy[anchor] - report["initial_sole_site_xy_m"], axis=1).tolist() if anchor is not None else None
        tail = [s for s in samples if 9.0 <= s["time_s"] <= 10.0]
        tail_both = sum(all(s["feet_grounded"]) for s in tail) / len(tail)
        rated_fraction = sum(s["leg_torque_saturated"] for s in samples) / len(samples)
        native_report_body_contact_frames = sum(s["body_contact"] for s in samples)
        assert all(not offline or sample["body_contact"] for offline, sample in zip(floor_nonsole_contact, samples))
        prohibited_contact = any(s["body_contact"] or s["hand_force_n"] >= 1 for s in samples) or any(floor_nonsole_contact)
        checks = {
            "full_duration": report["duration_s"] >= gate["full_duration_s"] and not report["ended_early"],
            "tail_upright": max(s["tilt_deg"] for s in tail) < gate["tail_tilt_lt_deg"],
            "tail_height": gate["tail_pelvis_z_m"][0] <= np.mean([s["pelvis_z_m"] for s in tail]) <= gate["tail_pelvis_z_m"][1],
            "tail_double_support": tail_both >= gate["tail_both_feet_fraction_gte"],
            "anchor_by_0p2s": anchor_time is not None and anchor_time <= gate["first_five_consecutive_both_contact_by_s"],
            "continuous_double_support_after_anchor": gaps == 0,
            "sole_center_stays": site_movement is not None and site_movement <= gate["sole_center_max_displacement_after_anchor_m_lte"],
            "sole_patch_stays": corner_movement is not None and corner_movement <= gate["sole_box_corner_max_displacement_after_anchor_m_lte"],
            "no_hand_body_contact": not prohibited_contact,
            "rated_leg_fraction": rated_fraction <= gate["rated_leg_fraction_lte"],
        }
        checks = {name: bool(value) for name, value in checks.items()}
        rows.append({
            "seed": seed,
            "source_report": str(report_path.relative_to(ROOT)),
            "source_report_sha256": digest(report_path),
            "native_trajectory": str(native_path.relative_to(ROOT)),
            "native_trajectory_sha256": digest(native_path),
            "initial_state_sha256": report["initial_state_sha256"],
            "offline_site_vs_recorded_max_abs_m": site_parity,
            "anchor_time_s": anchor_time,
            "post_anchor_both_contact_gap_frames": gaps,
            "reset_to_anchor_site_shift_m": reset_to_anchor_m,
            "max_post_anchor_site_center_motion_m": site_movement,
            "max_post_anchor_patch_corner_motion_m": corner_movement,
            "tail_both_feet_fraction": tail_both,
            "tail_max_tilt_deg": max(s["tilt_deg"] for s in tail),
            "tail_mean_pelvis_z_m": float(np.mean([s["pelvis_z_m"] for s in tail])),
            "rated_leg_fraction": rated_fraction,
            "native_report_body_contact_frames": native_report_body_contact_frames,
            "offline_floor_nonsole_body_contact_frames": sum(floor_nonsole_contact),
            "offline_first_floor_nonsole_contact_s": next((samples[i]["time_s"] for i, found in enumerate(floor_nonsole_contact) if found), None),
            "absolute_root_xy_drift_m": float(np.linalg.norm(qpos[-1, :2] - initial_qpos[:2])),
            "old_5cm_root_rule_pass": float(np.linalg.norm(qpos[-1, :2] - initial_qpos[:2])) <= 0.05,
            "actor_observed_first_command_x": samples[0]["actor_observed_command_x"],
            "internal_first_target_blend": samples[0]["target_blend"],
            "baseline_full_gate": baseline_rows[seed]["full_gate"],
            "first_0p2s_rated_leg_frames": sum(s["leg_torque_saturated"] for s in samples[:10]),
            "baseline_first_0p2s_rated_leg_frames": sum(s["leg_torque_saturated"] for s in json.loads((BASELINE / f"seed{seed}.json").read_text())["timeseries"][:10]),
            "gate": checks,
            "full_gate": all(checks.values()),
        })
    passes = sum(r["full_gate"] for r in rows)
    old_passes = sum(r["old_5cm_root_rule_pass"] for r in rows)
    prior_falls = [row for row in rows if not row["baseline_full_gate"]]
    prior_passes = [row for row in rows if row["baseline_full_gate"]]
    survives = (len(prior_falls) == 2 and len(prior_passes) == 4
                and all(row["full_gate"] and row["first_0p2s_rated_leg_frames"] < row["baseline_first_0p2s_rated_leg_frames"] for row in prior_falls)
                and all(row["full_gate"] for row in prior_passes))
    summary = {
        "opened_development_seeds": len(rows),
        "physical_full_gate_passes": passes,
        "old_5cm_root_rule_passes": old_passes,
        "falls_or_early_end": sum(not r["gate"]["full_duration"] or not r["gate"]["tail_upright"] for r in rows),
        "source_site_parity_faults": sum(r["offline_site_vs_recorded_max_abs_m"] > 1e-6 for r in rows),
        "offline_floor_nonsole_body_contact_cases": sum(r["offline_floor_nonsole_body_contact_frames"] > 0 for r in rows),
        "former_falls_recovered": sum(row["full_gate"] for row in prior_falls),
        "former_passes_retained": sum(row["full_gate"] for row in prior_passes),
        "first_0p2s_rated_frames_reduced_on_former_falls": sum(row["first_0p2s_rated_leg_frames"] < row["baseline_first_0p2s_rated_leg_frames"] for row in prior_falls),
        "frozen_hypothesis_survives": survives,
    }
    output = {
        "schema": "robot-learning-lab.cove50-seated-phase-alignment-report.v9",
        "status": "complete_opened_seed_diagnostic_no_learned_claim",
        "plan": str(plan_path.relative_to(ROOT)),
        "plan_sha256": digest(plan_path),
        "body_sha256": digest(body),
        "source_snapshot_sha256": digest(ARTIFACT / "source/evaluate_posture.py"),
        "scorer_source_sha256": digest(Path(__file__)),
        "runtime": "local CPU native MJLab, 50 Hz; zero paid GPU time",
        "summary": summary,
        "rows": rows,
        "contact_correction": "Native floor-nonsole contact must cover every offline MuJoCo floor-nonsole contact. Same corrected V5 physical gate; no threshold changed.",
        "claim_limit": "Opened V5 development seeds only, scripted controller, local CPU, zero new independent score and zero learned-control claim.",
    }
    (ARTIFACT / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
