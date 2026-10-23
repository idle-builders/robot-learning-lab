"""Source-check V29 exact-task CUDA critic and inline switch visits."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.compact_morphology.cove50_stand_recovery import score_seated_reference_cuda_v26 as v26score


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-intervention-v29"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
V26 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ROUTES = ("model25", "scripted", "script-first-0p2", "actor-first-0p1", "actor-first-0p2")
DT = 0.02


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def switch_schedule(route: str, n: int) -> np.ndarray:
    step = np.arange(n)
    if route == "model25":
        return np.zeros(n, dtype=bool)
    if route == "scripted":
        return np.ones(n, dtype=bool)
    if route == "script-first-0p2":
        return step < 10
    if route == "actor-first-0p1":
        return step >= 5
    if route == "actor-first-0p2":
        return step >= 10
    raise ValueError(route)


def post_switch_recovery(report: dict, route: str, row: dict) -> dict | None:
    if route not in ("actor-first-0p1", "actor-first-0p2"):
        return None
    switch = 5 if route == "actor-first-0p1" else 10
    samples = report["timeseries"]
    if len(samples) <= switch:
        return {"switch_s": switch * DT, "reanchor_s": None,
                "pre_switch_support_loss_after_initial_anchor": None,
                "passes_diagnostic": False,
                "failure_reasons": ["episode_ended_before_action_switch"]}
    grounded = np.asarray([all(s["feet_grounded"]) for s in samples], dtype=bool)
    initial_anchor = next((i for i in range(max(0, min(switch - 4, len(samples) - 4)))
                           if grounded[i:i + 5].all()), None)
    pre_switch_support_loss = (initial_anchor is not None
                               and bool(np.any(~grounded[initial_anchor:switch])))
    anchor = next((i for i in range(switch, min(switch + 16, len(samples) - 4))
                   if grounded[i:i + 5].all()), None)
    if anchor is None:
        return {"switch_s": switch * DT, "reanchor_s": None,
                "pre_switch_support_loss_after_initial_anchor": pre_switch_support_loss,
                "passes_diagnostic": False, "failure_reasons": ["no_five_frame_double_support_within_0p4s"]}
    site = np.asarray([s["sole_site_xy_m"] for s in samples], dtype=float)
    motion = float(np.max(np.linalg.norm(site[anchor:] - site[anchor], axis=2)))
    checks = {
        "five_frame_reanchor_completed_within_0p4s": anchor + 4 <= switch + 19,
        "no_later_support_gap": bool(grounded[anchor:].all()),
        "sole_center_after_reanchor_lte_15mm": motion <= 0.015,
        "tail_upright": row["physical_gate"]["tail_upright"],
        "tail_height": row["physical_gate"]["tail_height"],
        "tail_double_support": row["physical_gate"]["tail_double_support"],
        "no_hand_body_contact": row["physical_gate"]["no_hand_body_contact"],
        "rated_leg_fraction": row["physical_gate"]["rated_leg_fraction"],
        "full_duration": row["physical_gate"]["full_duration"],
    }
    if route == "actor-first-0p2":
        checks["actual_pre_switch_support_loss"] = pre_switch_support_loss
    return {
        "switch_s": switch * DT,
        "reanchor_started_s": round((anchor + 1) * DT, 2),
        "five_frame_reanchor_completed_s": round((anchor + 5) * DT, 2),
        "pre_switch_support_loss_after_initial_anchor": pre_switch_support_loss,
        "sole_center_after_reanchor_m": motion,
        "checks": checks,
        "passes_diagnostic": all(checks.values()),
        "failure_reasons": [name for name, passed in checks.items() if not passed],
        "claim_limit": "Post-switch recapture is diagnostic; it does not erase any pre-switch full-gate support gap or count as a learned planted stand.",
    }


def score_case(seed: int, route: str, model: mujoco.MjModel, gate: dict,
               body_sha: str) -> tuple[dict, dict]:
    row, _ = v26score.score(seed, route, model, gate, body_sha)
    report = json.loads((ART / f"seed{seed}-{route}.json").read_text())
    with np.load(ART / f"seed{seed}-{route}.npz", allow_pickle=False) as native:
        arrays = {key: native[key].copy() for key in (
            "initial_qpos", "initial_qvel", "actor_observation",
            "critic_observation", "critic_value", "actor_candidate_action",
            "script_candidate_action", "selected_action_is_script", "raw_action", "qpos", "reward_rate"
        )}
    n = len(report["timeseries"])
    require(report["intervention_route"] == route and report["critic_checkpoint_loaded"],
            f"{seed} {route}: wrong intervention route or critic")
    require(arrays["critic_observation"].shape == (n, 88)
            and arrays["critic_value"].shape == (n, 1)
            and arrays["selected_action_is_script"].shape == (n,)
            and arrays["actor_candidate_action"].shape == (n, 21)
            and arrays["script_candidate_action"].shape == (n, 21),
            f"{seed} {route}: missing critic/action-switch channels")
    require(all(np.isfinite(value).all() for value in arrays.values()),
            f"{seed} {route}: nonfinite native channel")
    expected = switch_schedule(route, n)
    require(np.array_equal(arrays["selected_action_is_script"], expected),
            f"{seed} {route}: action-switch schedule changed")
    chosen = np.where(expected[:, None], arrays["script_candidate_action"],
                      arrays["actor_candidate_action"])
    require(np.array_equal(chosen, arrays["raw_action"]),
            f"{seed} {route}: selected action differs from declared candidate")
    critic_gap = float(np.max(np.abs(arrays["critic_value"][:, 0] -
                                      [s["critic_value"] for s in report["timeseries"]])))
    require(critic_gap <= 5.1e-6, f"{seed} {route}: critic report/array mismatch")
    selection_report = np.asarray([s["selected_action_is_script"] for s in report["timeseries"]])
    require(np.array_equal(selection_report, expected),
            f"{seed} {route}: selection source report mismatch")
    row["critic_initial_value"] = float(arrays["critic_value"][0, 0])
    row["critic_at_0p1_s"] = float(arrays["critic_value"][4, 0]) if n >= 5 else None
    row["critic_at_0p2_s"] = float(arrays["critic_value"][9, 0]) if n >= 10 else None
    row["post_switch_recovery"] = post_switch_recovery(report, route, row)
    row["route"] = route
    return row, {"report": report, "arrays": arrays}


def verify_pair(seed: int, rows: dict[str, dict], sources: dict[str, dict],
                baseline_tolerance: dict) -> dict:
    parity = verify_baselines(seed, rows, sources, baseline_tolerance)
    for route in ROUTES[2:]:
        parity[route] = verify_one_intervention(seed, route, sources, baseline_tolerance)
    return parity


def verify_one_intervention(seed: int, route: str, sources: dict[str, dict],
                            baseline_tolerance: dict) -> dict:
    require(route in ROUTES[2:], "Only a switched route is an intervention")
    original = sources["model25"]["report"]
    report = sources[route]["report"]
    required = ("initial_state_sha256", "initial_qpos", "initial_qvel",
                "initial_predicted_sole_gap_m")
    require(all(report[field] == original[field] for field in required),
            f"{seed} {route}: start differs from baseline")
    require(report["initial_physics"] == original["initial_physics"]
            and report["initial_motors"] == original["initial_motors"],
            f"{seed} {route}: physical or motor draw differs")
    reference_arrays = sources["model25"]["arrays"]
    route_arrays = sources[route]["arrays"]
    require(np.array_equal(route_arrays["actor_observation"][0],
                           reference_arrays["actor_observation"][0])
            and np.array_equal(route_arrays["critic_observation"][0],
                               reference_arrays["critic_observation"][0]),
            f"{seed} {route}: first actor/critic input differs")
    require(abs(float(route_arrays["critic_value"][0, 0]) -
                float(reference_arrays["critic_value"][0, 0])) <= 1e-5,
            f"{seed} {route}: first critic value differs")
    baseline, steps = {
        "script-first-0p2": ("scripted", 10),
        "actor-first-0p1": ("model25", 5),
        "actor-first-0p2": ("model25", 10),
    }[route]
    new = route_arrays
    original_prefix = sources[baseline]["arrays"]
    observed_steps = min(steps, len(new["qpos"]))
    require(observed_steps > 0 and len(original_prefix["qpos"]) >= observed_steps,
            f"{seed} {route}: absent pre-switch prefix")
    qpos_gap = float(np.max(np.abs(new["qpos"][:observed_steps] - original_prefix["qpos"][:observed_steps])))
    action_gap = float(np.max(np.abs(new["raw_action"][:observed_steps] - original_prefix["raw_action"][:observed_steps])))
    require(qpos_gap <= baseline_tolerance["qpos_max_abs_m_or_rad"]
            and action_gap <= baseline_tolerance["raw_action_max_abs"],
            f"{seed} {route}: pre-switch path differs from {baseline}")
    return {"prefix_baseline": baseline, "planned_prefix_steps": steps,
            "observed_prefix_steps": observed_steps,
            "switch_reached": len(new["qpos"]) > steps,
            "qpos_max_abs_m_or_rad": qpos_gap,
            "raw_action_max_abs": action_gap}


def verify_baselines(seed: int, rows: dict[str, dict], sources: dict[str, dict],
                     baseline_tolerance: dict) -> dict:
    original = json.loads((V26 / f"seed{seed}-model25.json").read_text())
    model = sources["model25"]["report"]
    required = ("initial_state_sha256", "initial_qpos", "initial_qvel",
                "initial_predicted_sole_gap_m")
    for route in ROUTES[:2]:
        report = sources[route]["report"]
        require(all(report[field] == original[field] for field in required),
                f"{seed} {route}: start differs from V26")
        require(report["initial_physics"] == model["initial_physics"]
                and report["initial_motors"] == model["initial_motors"],
                f"{seed} {route}: physical or motor draw differs within V29")
        reference_arrays = sources["model25"]["arrays"]
        route_arrays = sources[route]["arrays"]
        require(np.array_equal(route_arrays["actor_observation"][0],
                               reference_arrays["actor_observation"][0])
                and np.array_equal(route_arrays["critic_observation"][0],
                                   reference_arrays["critic_observation"][0]),
                f"{seed} {route}: first actor/critic input differs within V29")
        require(abs(float(route_arrays["critic_value"][0, 0]) -
                    float(reference_arrays["critic_value"][0, 0])) <= 1e-5,
                f"{seed} {route}: first critic value differs within V29")
    for field in ("torso_mass_kg", "sole_slide_friction", "robot_body_mass_kg",
                  "robot_body_inertia_kg_m2", "robot_body_ipos_m", "robot_body_iquat_wxyz"):
        require(model["initial_physics"][field] == original["initial_physics"][field],
                f"{seed}: sampled V29 body differs from V26")
    for field in ("target_names", "rated_effort_nm", "effort_factor",
                  "stiffness_factor", "damping_factor", "delay_lag_steps"):
        require(model["initial_motors"][field] == original["initial_motors"][field],
                f"{seed}: sampled V29 motors differ from V26")
    parity = {}
    for route in ("model25", "scripted"):
        with np.load(V26 / f"seed{seed}-{route}.npz", allow_pickle=False) as old:
            new = sources[route]["arrays"]
            qpos_gap = float(np.max(np.abs(new["qpos"] - old["qpos"])))
            action_gap = float(np.max(np.abs(new["raw_action"] - old["raw_action"])))
        require(qpos_gap <= baseline_tolerance["qpos_max_abs_m_or_rad"]
                and action_gap <= baseline_tolerance["raw_action_max_abs"],
                f"{seed} {route}: V29 baseline differs from original V26 CUDA")
        parity[route] = {"qpos_max_abs_m_or_rad": qpos_gap,
                         "raw_action_max_abs": action_gap}
    require(rows["scripted"]["full_gate"],
            f"{seed}: source scripted baseline no longer passes")
    return parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-seed", type=int)
    parser.add_argument("--partial-seed", type=int)
    parser.add_argument("--partial-route", choices=ROUTES[2:])
    args = parser.parse_args()
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    require(sha(Path(__file__)) == plan["source_sha256"]["scorer"],
            "V29 scorer differs from freeze")
    require(sha(BODY) == plan["source_sha256"]["body"]
            and sha(V26 / "report.json") == plan["source_sha256"]["v26_report"],
            "V29 body or source result differs from freeze")
    require(sha(Path(v26score.__file__)) == plan["source_sha256"]["v26_scorer"],
            "V29 source physical scorer differs")
    v26score.ART = ART
    gate = json.loads((V25 / "plan.json").read_text())["physical_gate"]
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if args.baseline_seed is not None:
        require(args.baseline_seed in plan["opened_development_seeds"],
                "Baseline seed is outside opened development")
        route_rows, route_sources = {}, {}
        for route in ROUTES[:2]:
            route_rows[route], route_sources[route] = score_case(
                args.baseline_seed, route, model, gate, plan["source_sha256"]["body"]
            )
        parity = verify_baselines(args.baseline_seed, route_rows, route_sources,
                                  plan["baseline_repeat_tolerance"])
        print(json.dumps({"seed": args.baseline_seed, "baseline_repeat_parity": parity}))
        return
    if args.partial_seed is not None:
        require(args.partial_seed in plan["opened_development_seeds"]
                and args.partial_route is not None,
                "Partial route/seed outside opened development")
        route_rows, route_sources = {}, {}
        for route in (*ROUTES[:2], args.partial_route):
            route_rows[route], route_sources[route] = score_case(
                args.partial_seed, route, model, gate, plan["source_sha256"]["body"]
            )
        baseline_parity = verify_baselines(args.partial_seed, route_rows,
                                           route_sources, plan["baseline_repeat_tolerance"])
        prefix_parity = verify_one_intervention(args.partial_seed, args.partial_route,
                                                 route_sources, plan["baseline_repeat_tolerance"])
        print(json.dumps({"seed": args.partial_seed, "route": args.partial_route,
                          "baseline_repeat_parity": baseline_parity,
                          "intervention_prefix_parity": prefix_parity,
                          "physical_gate": route_rows[args.partial_route]["full_gate"],
                          "failure_reasons": route_rows[args.partial_route]["failure_reasons"]}))
        return
    rows, pairs = [], []
    for seed in plan["opened_development_seeds"]:
        route_rows, route_sources = {}, {}
        for route in ROUTES:
            route_rows[route], route_sources[route] = score_case(
                seed, route, model, gate, plan["source_sha256"]["body"]
            )
        parity = verify_pair(seed, route_rows, route_sources,
                             plan["baseline_repeat_tolerance"])
        rows.extend(route_rows.values())
        pairs.append({"seed": seed, "baseline_repeat_parity": parity})
    result = {
        "schema": "robot-learning-lab.cove50-seated-critic-inline-switch.v29",
        "status": "complete_source_checked_cuda_opened_development_diagnostic",
        "plan_sha256": sha(plan_path),
        "body_sha256": sha(BODY),
        "source_v26_report_sha256": sha(V26 / "report.json"),
        "summary": {
            "opened_seed_families": len(pairs),
            "native_visits": len(rows),
            "optimizer_updates": 0,
            "script_first_0p2_full_gate_passes": sum(row["full_gate"] for row in rows if row["route"] == "script-first-0p2"),
            "actor_first_0p1_full_gate_passes": sum(row["full_gate"] for row in rows if row["route"] == "actor-first-0p1"),
            "actor_first_0p2_recapture_diagnostic_passes": sum(
                bool(row["post_switch_recovery"] and row["post_switch_recovery"]["passes_diagnostic"])
                for row in rows if row["route"] == "actor-first-0p2"
            ),
        },
        "pairs": pairs,
        "rows": rows,
        "claim_limit": "Opened-start critic/action-switch diagnosis only. Scripted intervention or post-switch recapture is not a learned policy, independent score or physical qualification.",
    }
    (ART / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
