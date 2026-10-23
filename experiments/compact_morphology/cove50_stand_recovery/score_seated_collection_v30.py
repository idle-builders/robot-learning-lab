"""Verify V30 first-24 matched visits and score the unchanged V25 physical gate.

This scorer accepts partial collections. Missing and failed originals remain in
the ledger; only complete, source-checked pairs count toward the 18/24 gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_stand_recovery import score_seated_reference_cuda_v26 as v26


DEFAULT_ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-v30-first24"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SEEDS = tuple(range(33001, 33025))
ROUTES = ("scripted", "model25")
DT = 0.02


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def event(values: dict[str, np.ndarray]) -> dict:
    grounded = values["post_sole_contact"].all(axis=1)
    anchor = next((i for i in range(len(grounded) - 4)
                   if grounded[i:i + 5].all()), None)
    loss = (next((i for i in range(anchor + 5, len(grounded))
                  if not grounded[i]), None) if anchor is not None else None)
    return {
        "five_frame_anchor_start_index": anchor,
        "five_frame_anchor_start_post_time_s": ((anchor + 1) * DT if anchor is not None else None),
        "five_frame_anchor_complete_post_time_s": ((anchor + 5) * DT if anchor is not None else None),
        "first_later_double_support_loss_index": loss,
        "first_later_double_support_loss_post_time_s": ((loss + 1) * DT if loss is not None else None),
        "first_later_double_support_loss_sole_contact": (
            values["post_sole_contact"][loss].tolist() if loss is not None else None),
    }


def check_native(seed: int, route: str, report: dict, values: dict[str, np.ndarray]) -> dict:
    n = len(report["timeseries"])
    require(abs(report["control_dt_s"] - DT) <= 1e-9,
            f"{seed} {route}: control rate differs from 50 Hz")
    shapes = {
        "actor_observation": (n, 73), "critic_observation": (n, 88),
        "raw_action": (n, 21), "applied_action": (n, 21),
        "previous_raw_action": (n, 21), "previous_applied_action": (n, 21),
        "processed_target": (n, 21), "post_sole_contact": (n, 2),
        "post_sole_force_n": (n, 2), "post_sole_site_xyz_m": (n, 2, 3),
        "post_leg_torque_nm": (n, 10), "post_all_actuator_force_nm": (n, 21),
        "post_leg_rated_duty": (n,), "post_pelvis_pose_w": (n, 7),
        "post_hand_site_xyz_m": (n, 2, 3),
        "post_termination_flags": (n, len(report["termination_term_names"])),
        "reward_rate": (n, 15),
    }
    for name, shape in shapes.items():
        require(name in values and values[name].shape == shape,
                f"{seed} {route}: {name} shape {values.get(name, np.array([])).shape} != {shape}")
    for name in ("control_index", "pre_action_time_s", "previous_action_valid", "previous_action_time_s",
                 "previous_action_age_s", "command_squat_pre", "command_x_pre",
                 "rise_phase_pre", "post_time_s", "post_pelvis_z_m", "post_tilt_deg",
                 "post_hand_force_n", "post_body_contact", "post_floor_nonsole_body_contact",
                 "post_command_x", "post_rise_phase", "post_step_reward"):
        require(name in values and values[name].shape == (n,),
                f"{seed} {route}: missing or malformed {name}")
    require(all(np.isfinite(v).all() for v in values.values()),
            f"{seed} {route}: nonfinite native value")
    require(np.array_equal(values["control_index"], np.arange(n))
            and np.allclose(values["pre_action_time_s"], np.arange(n) * DT, atol=1e-8)
            and np.allclose(values["post_time_s"], (np.arange(n) + 1) * DT, atol=1e-8),
            f"{seed} {route}: pre/post timebase drift")
    require(not values["previous_action_valid"][0]
            and values["previous_action_valid"][1:].all()
            and np.allclose(values["previous_raw_action"][0], 0)
            and np.allclose(values["previous_applied_action"][0], 0)
            and np.array_equal(values["previous_raw_action"][1:], values["raw_action"][:-1])
            and np.array_equal(values["previous_applied_action"][1:], values["applied_action"][:-1])
            and np.allclose(values["previous_action_time_s"][1:], np.arange(n - 1) * DT)
            and np.allclose(values["previous_action_age_s"][1:], DT),
            f"{seed} {route}: previous-action values or age do not match actual prior step")
    require(values["selected_action_is_script"].shape == (n,)
            and np.all(values["selected_action_is_script"] == (route == "scripted"))
            and np.array_equal(values["raw_action"],
                               values["script_candidate_action" if route == "scripted"
                                      else "actor_candidate_action"]),
            f"{seed} {route}: route/selected action mismatch")
    clip = report["wrapper_action_clip"]
    expected_applied = (values["raw_action"] if clip is None else
                        np.clip(values["raw_action"], -clip, clip))
    require(np.array_equal(values["applied_action"], expected_applied),
            f"{seed} {route}: actual applied action differs from wrapper clip")
    samples = report["timeseries"]
    require(np.array_equal(values["post_sole_contact"],
                           np.asarray([s["feet_grounded"] for s in samples], dtype=bool)),
            f"{seed} {route}: sole contact report mismatch")
    require(np.max(np.abs(values["post_sole_site_xyz_m"][:, :, :2] -
                          np.asarray([s["sole_site_xy_m"] for s in samples]))) <= 5e-7,
            f"{seed} {route}: sole site report mismatch")
    require(np.max(np.abs(values["post_sole_force_n"] -
                          np.asarray([s["foot_force_n"] for s in samples]))) <= 0.0051,
            f"{seed} {route}: sole force report mismatch")
    require(np.max(np.abs(values["post_leg_torque_nm"] -
                          np.asarray([s["leg_joint_effort_nm"] for s in samples]))) <= 0.00051,
            f"{seed} {route}: torque report mismatch")
    require(np.array_equal(values["post_leg_torque_nm"],
                           values["post_all_actuator_force_nm"][:, :10]),
            f"{seed} {route}: leg/all-actuator force mismatch")
    require(np.array_equal(values["post_leg_rated_duty"],
                           [s["leg_torque_saturated"] for s in samples]),
            f"{seed} {route}: rated duty report mismatch")
    require(np.max(np.abs(values["post_pelvis_z_m"] -
                          [s["pelvis_z_m"] for s in samples])) <= 5.1e-5,
            f"{seed} {route}: height report mismatch")
    require(np.array_equal(values["post_pelvis_z_m"],
                           values["post_pelvis_pose_w"][:, 2]),
            f"{seed} {route}: pelvis pose/height mismatch")
    require(np.max(np.abs(values["post_tilt_deg"] -
                          [s["tilt_deg"] for s in samples])) <= 0.0051,
            f"{seed} {route}: tilt report mismatch")
    require(np.max(np.abs(values["post_hand_force_n"] -
                          [s["hand_force_n"] for s in samples])) <= 0.00051
            and np.max(np.abs(values["post_hand_site_xyz_m"][:, :, 2] -
                                  np.asarray([s["hand_height_m"] for s in samples]))) <= 5.1e-5,
            f"{seed} {route}: hand physical input report mismatch")
    require(np.array_equal(values["post_body_contact"],
                           [s["body_contact"] for s in samples])
            and np.array_equal(values["post_floor_nonsole_body_contact"],
                               [s["floor_nonsole_body_contact"] for s in samples]),
            f"{seed} {route}: forbidden-contact report mismatch")
    require(np.array_equal(values["command_squat_pre"],
                           [s["command_squat"] for s in samples]),
            f"{seed} {route}: command report mismatch")
    require(np.max(np.abs(values["post_rise_phase"] -
                          [s["target_blend"] for s in samples])) <= 0.00051
            and np.max(np.abs(values["post_command_x"] -
                                  [s["actor_observed_command_x"] for s in samples])) <= 5.1e-5
            and np.max(np.abs(values["post_step_reward"] -
                                  [s["step_reward"] for s in samples])) <= 5.1e-5,
            f"{seed} {route}: post phase/reward report mismatch")
    require(np.max(np.abs(values["rise_phase_pre"] - values["actor_observation"][:, -1])) <= 1e-5,
            f"{seed} {route}: rise phase differs from actor-visible phase")
    require(report["actor_observation_terms"][-3:] == ["actions", "command", "rise_phase"]
            and np.max(np.abs(values["actor_observation"][:, -25:-4] -
                              values["previous_applied_action"])) <= 1e-6,
            f"{seed} {route}: actor's previous-action slot differs from actual applied action")
    require(np.max(np.abs(values["actor_observation"][:, -4] -
                          values["command_x_pre"])) <= 1e-6,
            f"{seed} {route}: recorded command differs from actor-visible command")
    if report["end_termination_terms"]:
        require([name for name, active in zip(report["termination_term_names"],
                                              values["post_termination_flags"][-1]) if active]
                == report["end_termination_terms"],
                f"{seed} {route}: named terminal flags mismatch")
    return event(values)


def score_visit(seed: int, route: str, art: Path, model: mujoco.MjModel,
                gate: dict, body_sha: str) -> tuple[dict, dict]:
    v26.ART = art
    row, _ = v26.score(seed, route, model, gate, body_sha)
    report = json.loads((art / f"seed{seed}-{route}.json").read_text())
    with np.load(art / f"seed{seed}-{route}.npz", allow_pickle=False) as source:
        values = {name: source[name].copy() for name in source.files}
    row["support_event"] = check_native(seed, route, report, values)
    require(report["termination_term_names"] and report["native_record_alignment"],
            f"{seed} {route}: missing time/termination provenance")
    return row, report


def pair(seed: int, art: Path, model: mujoco.MjModel, gate: dict,
         body_sha: str) -> dict:
    rows, reports = {}, {}
    for route in ROUTES:
        rows[route], reports[route] = score_visit(seed, route, art, model, gate, body_sha)
    a, b = reports["scripted"], reports["model25"]
    fields = ("initial_state_sha256", "initial_qpos", "initial_qvel",
              "initial_predicted_sole_gap_m", "initial_physics", "initial_motors")
    require(all(a[key] == b[key] for key in fields),
            f"{seed}: paired initial state, full physics, or 21 motor laws/lag differ")
    with np.load(art / f"seed{seed}-scripted.npz", allow_pickle=False) as x, np.load(
            art / f"seed{seed}-model25.npz", allow_pickle=False) as y:
        require(np.array_equal(x["actor_observation"][0], y["actor_observation"][0])
                and np.array_equal(x["critic_observation"][0], y["critic_observation"][0]),
                f"{seed}: first corrupted actor/critic observations differ")
    return {"seed": seed, "status": "complete_source_checked_pair",
            "initial_state_sha256": a["initial_state_sha256"],
            "scripted": rows["scripted"], "model25": rows["model25"]}


def score_collection(art: Path, *, seed: int | None = None) -> dict:
    plan_path = art / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(plan["seed_families"] == list(SEEDS) and plan["routes"] == list(ROUTES),
            "V30 first-24 partition changed")
    require(plan["diagnostic_fit_seeds"] == list(range(33001, 33017))
            and plan["diagnostic_assessment_seeds"] == list(range(33017, 33025))
            and plan["original_visits_max"] == 48,
            "V30 16/8 diagnostic partition or 48-visit cap changed")
    require(sha(BODY) == plan["source_sha256"]["body"]
            and sha(V25 / "plan.json") == plan["source_sha256"]["v25_plan"]
            and sha(V25 / "checkpoints/model_25.pt") == plan["source_sha256"]["checkpoint"],
            "V30 exact body, V25 gate, or checkpoint drift")
    ledger = json.loads((art / "attempt-ledger.json").read_text())
    require(ledger["seed_families"] == list(SEEDS)
            and [family["seed"] for family in ledger["families"]] == list(SEEDS),
            "attempt ledger changed partition")
    require(all(set(family["routes"]) == set(ROUTES) for family in ledger["families"]),
            "attempt ledger changed routes")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    gate = json.loads((V25 / "plan.json").read_text())["physical_gate"]
    results = []
    for family in ledger["families"]:
        current = family["seed"]
        if seed is not None and seed != current:
            continue
        statuses = {route: family["routes"][route]["status"] for route in ROUTES}
        if all(statuses[route] == "completed" for route in ROUTES):
            try:
                result = pair(current, art, model, gate, plan["source_sha256"]["body"])
                result["diagnostic_partition"] = ("fit" if current <= 33016 else "assessment")
                results.append(result)
            except Exception as exc:
                results.append({"seed": current, "status": "source_invalid",
                                "diagnostic_partition": ("fit" if current <= 33016 else "assessment"),
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "route_status": statuses})
        else:
            results.append({"seed": current, "status": "incomplete_or_failed_original",
                            "diagnostic_partition": ("fit" if current <= 33016 else "assessment"),
                            "route_status": statuses,
                            "failure_reasons": {route: family["routes"][route].get("failure_reason")
                                                for route in ROUTES if family["routes"][route].get("failure_reason")}})
    complete = [item for item in results if item["status"] == "complete_source_checked_pair"]
    scripted_passes = sum(item["scripted"]["full_gate"] for item in complete)
    attempted = sum(any(family["routes"][route]["status"] != "not_attempted"
                        for route in ROUTES) for family in ledger["families"])
    visits_attempted = sum(family["routes"][route]["status"] != "not_attempted"
                           for family in ledger["families"] for route in ROUTES)
    require(visits_attempted <= 48, "more than 48 original visits")
    final_gate_ready = attempted == 24 and seed is None
    billing_bound_started = datetime.fromisoformat(ledger["local_start_request_at_utc"])
    worker_started = datetime.fromisoformat(ledger["worker_started_at_utc"])
    pod_elapsed = (datetime.now(timezone.utc) - billing_bound_started).total_seconds()
    native_visit_wall = sum(family["routes"][route].get("native_visit_wall_s", 0.0)
                            for family in ledger["families"] for route in ROUTES)
    return {
        "schema": "robot-learning-lab.cove50-seated-first24.v30",
        "status": ("partial" if attempted < 24 else
                   "complete_with_invalid_source" if any(item["status"] == "source_invalid" for item in results)
                   else "complete_with_failed_originals" if len(complete) < 24 else "complete"),
        "plan_sha256": sha(plan_path), "attempt_ledger_sha256": sha(art / "attempt-ledger.json"),
        "summary": {"original_families": 24, "families_attempted": attempted,
                    "original_visits_attempted": visits_attempted,
                    "source_checked_complete_pairs": len(complete),
                    "scripted_full_gate_passes": scripted_passes,
                    "reference_gate_18_of_24": (len(complete) == 24 and scripted_passes >= 18
                                                if final_gate_ready else None),
                    "optimizer_updates": 0},
        "resource_accounting": {
            "local_start_request_at_utc": ledger["local_start_request_at_utc"],
            "billing_bound_basis": ledger["billing_bound_basis"],
            "worker_started_at_utc": ledger["worker_started_at_utc"],
            "setup_before_worker_upper_bound_s": (worker_started - billing_bound_started).total_seconds(),
            "native_visit_wall_s": native_visit_wall,
            "elapsed_from_local_request_through_report_s": pod_elapsed,
            "listed_rate_gpu_upper_bound_estimate_usd": pod_elapsed * plan["gpu_listed_rate_usd_per_hour"] / 3600,
            "posted_pod_charge_usd": ledger.get("posted_pod_charge_usd"),
            "serverless_charge_usd": ledger.get("serverless_charge_usd"),
        },
        "families": results,
        "claim_limit": "First 24 new training-only seed families. Scripted reference is not learned control; partial or invalid originals are never replaced, and held/tail seeds remain sealed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--art", type=Path, default=DEFAULT_ART)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = score_collection(args.art, seed=args.seed)
    if args.write_report:
        (args.art / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"] if args.seed is None else result["families"], indent=2))


if __name__ == "__main__":
    main()
