"""Locally verify synced V34 families without changing the frozen worker scorer.

The worker reports its absolute checkpoint path. This copy accepts that path
only when it ends at the frozen V25 checkpoint and the report contains its
frozen SHA-256. All physical scoring still uses the unchanged V26 scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import mujoco
import numpy as np

from experiments.compact_morphology.cove50_stand_recovery import score_seated_reference_cuda_v26 as v26
from experiments.compact_morphology.cove50_stand_recovery import prepare_seated_trigger_v34 as prepare


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-trigger-v34"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
SEEDS = tuple(range(35001, 35013))
ROUTES = ("scripted", "model25", "fixed-0p06", "gravity-switch")
DT = 0.02
THRESHOLD = -0.034


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_seed(seed: int, plan: dict, ledger: dict, model: mujoco.MjModel, gate: dict) -> dict:
    v26.ART = ART
    family = next(item for item in ledger["families"] if item["seed"] == seed)
    rows, reports, arrays = {}, {}, {}
    for route in ROUTES:
        item = family["routes"][route]
        require(item["status"] == "completed", f"{seed} {route}: original not complete")
        report_path = ART / f"seed{seed}-{route}.json"
        array_path = ART / f"seed{seed}-{route}.npz"
        require(sha(report_path) == item["report_sha256"] and
                sha(array_path) == item["native_sha256"],
                f"{seed} {route}: source hash differs from atomic ledger")
        row, _ = v26.score(seed, route, model, gate, plan["source_sha256"]["body"])
        report = json.loads(report_path.read_text())
        with np.load(array_path, allow_pickle=False) as native:
            values = {name: native[name].copy() for name in native.files}
        n = len(report["timeseries"])
        require(report["seed"] == seed and
                report["intervention_route"] == route and
                report["preflight_sentinel"] is False and
                report["leg_target_margin_rad"] is None and
                tuple(PurePosixPath(report["checkpoint"]).parts[-3:]) ==
                    ("seated-rise-ppo-pilot-v25", "checkpoints", "model_25.pt") and
                report["checkpoint_sha256"] == plan["source_sha256"]["checkpoint"] and
                report["evaluator_source_sha256"] == plan["source_sha256"]["evaluator"] and
                report["actor_observation_terms"] == ["base_ang_vel", "projected_gravity",
                    "joint_pos", "joint_vel", "actions", "command", "rise_phase"] and
                values["actor_observation"].shape == (n, 73) and
                values["critic_observation"].shape == (n, 88) and
                values["selected_action_is_script"].shape == (n,) and
                values["actor_candidate_action"].shape == (n, 21) and
                values["script_candidate_action"].shape == (n, 21),
                f"{seed} {route}: native observation/action contract differs")
        selected = values["selected_action_is_script"].astype(bool)
        if route == "scripted":
            expected = np.ones(n, dtype=bool)
        elif route == "model25":
            expected = np.zeros(n, dtype=bool)
        elif route == "fixed-0p06":
            expected = np.arange(n) * DT >= 0.06 - 1e-9
        else:
            expected = np.zeros(n, dtype=bool)
            latched = False
            fired = None
            for i in range(n):
                t = i * DT
                if (not latched and 0.04 <= t <= 0.10 + 1e-9 and
                        float(values["actor_observation"][i, 3]) < THRESHOLD):
                    latched = True
                    fired = round(t, 3)
                expected[i] = latched
            require(report["gravity_switch_threshold"] == THRESHOLD and
                    report["gravity_switch_fired_pre_s"] == fired and
                    (fired is None or abs(report["gravity_switch_fired_x"] -
                                          float(values["actor_observation"][round(fired / DT), 3])) <= 1e-6),
                    f"{seed}: trigger report differs from actor input")
            row["trigger_fired_pre_s"] = fired
        require(np.array_equal(selected, expected) and
                np.array_equal(values["raw_action"], np.where(
                    expected[:, None], values["script_candidate_action"],
                    values["actor_candidate_action"])),
                f"{seed} {route}: actual action selection differs from frozen route")
        require(np.array_equal(values["previous_raw_action"][1:], values["raw_action"][:-1]) and
                np.array_equal(values["previous_applied_action"][1:], values["applied_action"][:-1]) and
                np.allclose(values["actor_observation"][:, 72],
                            values["rise_phase_pre"], atol=1e-6) and
                np.array_equal(values["post_sole_contact"],
                               np.asarray([step["feet_grounded"] for step in report["timeseries"]])) and
                np.allclose(values["pre_action_time_s"], np.arange(n) * DT, atol=1e-8),
                f"{seed} {route}: native time, action history or contact trace differs")
        rows[route], reports[route], arrays[route] = row, report, values

    identity = ("initial_state_sha256", "initial_qpos", "initial_qvel",
                "initial_predicted_sole_gap_m", "initial_physics", "initial_motors")
    require(all(reports["scripted"][key] == reports[route][key]
                for route in ROUTES[1:] for key in identity) and
            all(np.array_equal(arrays["scripted"]["actor_observation"][0],
                               arrays[route]["actor_observation"][0]) and
                np.array_equal(arrays["scripted"]["critic_observation"][0],
                               arrays[route]["critic_observation"][0])
                for route in ROUTES[1:]),
            f"{seed}: paired initial state, physical draw or first observation differs")
    trigger = arrays["gravity-switch"]
    actor = arrays["model25"]
    fixed = arrays["fixed-0p06"]
    selected = trigger["selected_action_is_script"].astype(bool)
    prefix = int(np.argmax(selected)) if selected.any() else len(selected)
    require(np.array_equal(trigger["qpos"][:prefix], actor["qpos"][:prefix]) and
            np.array_equal(trigger["raw_action"][:prefix], actor["raw_action"][:prefix]),
            f"{seed}: actor prefix differs before trigger")
    fixed_selected = fixed["selected_action_is_script"].astype(bool)
    fixed_prefix = int(np.argmax(fixed_selected)) if fixed_selected.any() else len(fixed_selected)
    require(np.array_equal(fixed["qpos"][:fixed_prefix], actor["qpos"][:fixed_prefix]) and
            np.array_equal(fixed["raw_action"][:fixed_prefix], actor["raw_action"][:fixed_prefix]),
            f"{seed}: actor prefix differs before fixed switch")
    return {
        "seed": seed, "status": "complete_source_checked_four_route_family",
        "first_observation_and_full_physics_equal": True,
        "actor_prefix_steps_equal": prefix,
        "scripted": rows["scripted"], "model25": rows["model25"],
        "fixed_0p06": rows["fixed-0p06"],
        "gravity_switch": rows["gravity-switch"],
        "trigger_rescue": (not rows["model25"]["full_gate"] and
                           rows["gravity-switch"]["full_gate"]),
        "trigger_regression_vs_script": (rows["scripted"]["full_gate"] and
                                          not rows["gravity-switch"]["full_gate"]),
        "trigger_only_win_vs_fixed": (rows["gravity-switch"]["full_gate"] and
                                      not rows["fixed-0p06"]["full_gate"]),
        "trigger_only_loss_vs_fixed": (rows["fixed-0p06"]["full_gate"] and
                                       not rows["gravity-switch"]["full_gate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ART / "local-score.json")
    args = parser.parse_args()
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    ledger = json.loads((ART / "attempt-ledger.json").read_text())
    prepare.verify_sources(plan)
    require(plan["seed_families"] == list(SEEDS) and
            plan["routes"] == list(ROUTES) and
            plan["gravity_trigger"] == {"start_pre_s": 0.04, "end_pre_s": 0.10,
                                        "projected_gravity_x_lt": THRESHOLD,
                                        "latch_script_for_remainder": True} and
            plan["fixed_time_switch_pre_s"] == 0.06 and
            plan["sentinel_seed"] == 35000 and
            plan["original_visits_max"] == 48 and
            plan["sentinel_visits_max"] == 1 and
            plan["no_posthoc_replacement"] is True and
            plan["v25_held_seeds_sealed"] == list(range(2240, 2252)) and
            ledger["plan_sha256"] == sha(plan_path) and
            ledger["no_posthoc_replacement"] is True and
            ledger["sentinel_seed"] == 35000 and
            ledger["sentinel_status"] in ("not_attempted_unscored", "attempting_unscored",
                                          "failed_original_unscored", "passed_unscored") and
            ledger["sentinel_attempts"] == (0 if ledger["sentinel_status"] ==
                                             "not_attempted_unscored" else 1) and
            ledger["seed_families"] == list(SEEDS) and
            ledger["routes"] == list(ROUTES) and
            [family["seed"] for family in ledger["families"]] == list(SEEDS) and
            all(set(family["routes"]) == set(ROUTES) and
                all(entry["status"] in ("not_attempted", "attempting", "failed_original",
                                         "invalid_original", "completed")
                    and entry["attempts"] == (0 if entry["status"] == "not_attempted" else 1)
                    for entry in family["routes"].values())
                for family in ledger["families"]),
            "V34 plan, trigger or ledger identity differs")
    sentinel_pass = ledger["sentinel_status"] == "passed_unscored"
    if sentinel_pass:
        sentinel_report_path = ART / "seed35000-sentinel.json"
        sentinel_native_path = ART / "seed35000-sentinel.npz"
        require(sha(sentinel_report_path) == ledger["sentinel_report_sha256"] and
                sha(sentinel_native_path) == ledger["sentinel_native_sha256"],
                "V34 exact CUDA sentinel source differs")
        sentinel = json.loads(sentinel_report_path.read_text())
        with np.load(sentinel_native_path, allow_pickle=False) as native:
            require(sentinel["seed"] == 35000 and sentinel["preflight_sentinel"] is True and
                    sentinel["intervention_route"] == "model25" and
                    sentinel["device"] == "cuda:0" and
                    sentinel["leg_target_margin_rad"] is None and
                    sentinel["body_sha256"] == plan["source_sha256"]["body"] and
                    sentinel["checkpoint_sha256"] == plan["source_sha256"]["checkpoint"] and
                    sentinel["evaluator_source_sha256"] == plan["source_sha256"]["evaluator"] and
                    native["actor_observation"].shape == (1, 73) and
                    native["critic_observation"].shape == (1, 88) and
                    native["raw_action"].shape == (1, 21),
                    "V34 exact CUDA sentinel contract differs")
    require(sha(BODY) == plan["source_sha256"]["body"] and
            sha(V25 / "plan.json") == plan["source_sha256"]["v25_plan"],
            "V34 body or unchanged physical gate differs")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    gate = json.loads((V25 / "plan.json").read_text())["physical_gate"]
    completed = []
    pending_sync = []
    verification_failures = []
    for family in ledger["families"]:
        seed = family["seed"]
        for route in ROUTES:
            entry = family["routes"][route]
            if entry["status"] != "completed":
                continue
            for kind, suffix, key in (("report", ".json", "report_sha256"),
                                      ("native", ".npz", "native_sha256")):
                path = ART / f"seed{seed}-{route}{suffix}"
                if not path.is_file():
                    pending_sync.append({"seed": seed, "route": route, "missing": kind})
                elif sha(path) != entry[key]:
                    verification_failures.append({
                        "seed": seed, "route": route,
                        "reason": f"{kind} SHA-256 differs from attempt ledger",
                    })
        if not all(family["routes"][route]["status"] == "completed" for route in ROUTES):
            continue
        if any(row["seed"] == seed for row in (*pending_sync, *verification_failures)):
            continue
        try:
            completed.append(score_seed(seed, plan, ledger, model, gate))
        except Exception as exc:
            verification_failures.append({"seed": seed, "reason": f"{type(exc).__name__}: {exc}"})
    valid_full_collection = (sentinel_pass and len(completed) == len(SEEDS)
                             and not pending_sync and not verification_failures)
    attempted_scored = sum(entry["attempts"] for family in ledger["families"]
                           for entry in family["routes"].values())
    failed_scored = [{"seed": family["seed"], "route": route,
                      "status": entry["status"], "failure_reason": entry.get("failure_reason")}
                     for family in ledger["families"] for route, entry in family["routes"].items()
                     if entry["status"] in ("failed_original", "invalid_original")]
    incomplete_families = [{
        "seed": family["seed"],
        "routes": {route: family["routes"][route]["status"] for route in ROUTES},
    } for family in ledger["families"]
        if not all(family["routes"][route]["status"] == "completed" for route in ROUTES)]
    script_pass = sum(item["scripted"]["full_gate"] for item in completed)
    actor_pass = sum(item["model25"]["full_gate"] for item in completed)
    trigger_pass = sum(item["gravity_switch"]["full_gate"] for item in completed)
    fixed_pass = sum(item["fixed_0p06"]["full_gate"] for item in completed)
    rescues = sum(item["trigger_rescue"] for item in completed)
    trigger_times = [item["gravity_switch"]["trigger_fired_pre_s"] for item in completed]
    time_counts = {str(t): trigger_times.count(t) for t in sorted(set(trigger_times),
                   key=lambda value: (value is None, value if value is not None else 0))}
    variable_trigger_timing = sum(count >= 2 for time, count in time_counts.items()
                                  if time != "None") >= 2
    trigger_only_wins = sum(item["trigger_only_win_vs_fixed"] for item in completed)
    trigger_only_losses = sum(item["trigger_only_loss_vs_fixed"] for item in completed)
    correction_feasible = valid_full_collection and script_pass >= 8 and rescues >= 6
    report = {
        "schema": "robot-learning-lab.cove50-seated-trigger.v34.score",
        "plan_sha256": sha(plan_path), "attempt_ledger_sha256": sha(ART / "attempt-ledger.json"),
        "source_checked_complete_families": len(completed),
        "exact_cuda_sentinel_passed": sentinel_pass,
        "planned_families": len(SEEDS), "native_visits": 4 * len(completed),
        "attempted_scored_visits": attempted_scored,
        "attempted_native_visits_including_sentinel": attempted_scored + ledger["sentinel_attempts"],
        "failed_originals": failed_scored,
        "sentinel_failure": (ledger.get("sentinel_failure_reason")
                             if ledger["sentinel_status"] == "failed_original_unscored" else None),
        "incomplete_families": incomplete_families,
        "scripted_full_gate": script_pass, "model25_full_gate": actor_pass,
        "fixed_0p06_full_gate": fixed_pass,
        "trigger_full_gate": trigger_pass, "trigger_rescues": rescues,
        "trigger_time_counts": time_counts,
        "variable_trigger_timing": variable_trigger_timing,
        "trigger_only_wins_vs_fixed": trigger_only_wins,
        "trigger_only_losses_vs_fixed": trigger_only_losses,
        "trigger_regressions_vs_script": sum(item["trigger_regression_vs_script"] for item in completed),
        "rows": completed,
        "status": ("verification_failed" if verification_failures else
                   "complete_development_diagnostic" if valid_full_collection else
                   "partial_development_diagnostic"),
        "pending_sync": pending_sync,
        "verification_failures": verification_failures,
        "method_gate_met": correction_feasible,
        "actor_visible_signal_value_gate_met": (correction_feasible and variable_trigger_timing
                                                and trigger_only_wins >= 2
                                                and trigger_only_losses == 0),
        "claim_limit": "Development scripted correction only. Gravity benefit needs variable timing and paired wins over the clock control; script is not learner-state feedback, learned stand, held score or physical transfer. No optimizer fit follows V34 alone.",
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({key: report[key] for key in ("status", "source_checked_complete_families",
                  "scripted_full_gate", "model25_full_gate", "trigger_full_gate",
                  "trigger_rescues", "method_gate_met", "pending_sync",
                  "verification_failures")}, indent=2))


if __name__ == "__main__":
    main()
