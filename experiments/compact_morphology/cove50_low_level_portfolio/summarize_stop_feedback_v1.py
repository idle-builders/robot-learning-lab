"""Verify frozen stop-residual whole-start pairs and report physical outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/modular-stop-feedback-prep"
NATIVE = BASE / "fresh-score-native"
SOURCE = Path(__file__).parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failures(report: dict) -> list[str]:
    reasons = []
    for name, failed in (
        ("early_end", report["ended_early"] or report["steps"] != 1200),
        ("walk_travel", report["walk_distance_m"] < 0.06),
        ("stop_tail", report["stop_tail_xy_m"] is None or report["stop_tail_xy_m"] > 0.03),
        ("squat", not report["squat_depth_return_pass"]),
        ("arms", not report["arm_rise_return_pass"]),
        ("forbidden_contact", not report["no_hand_or_body_contact"]),
        ("support", not report["support_windows_pass"]),
        ("rated_nonwalk_duty", not report["rated_duty_windows_pass"]),
        ("switch_continuity", report["switch_qpos_gap"] is None
         or report["switch_qpos_gap"] >= 1e-9),
    ):
        if failed:
            reasons.append(name)
    return reasons


def route(case: dict, label: str, method: dict, run_manifest: dict | None) -> dict:
    stem = NATIVE / f"seed-{case['seed']}-{label}"
    paths = {ext: stem.with_suffix(ext) for ext in (".json", ".npz", ".mp4", ".log")}
    if not any(path.exists() for path in paths.values()):
        return {"status": "not_run"}
    if not all(path.exists() for path in paths.values()):
        return {"status": "incomplete_original",
                "present": {ext: path.exists() for ext, path in paths.items()}}
    report = json.loads(paths[".json"].read_text())
    expected_schema = ("robot-learning-lab.modular-low-level-portfolio-breadth.v2"
                       if label == "baseline" else
                       "robot-learning-lab.modular-stop-feedback-v1.score.v1")
    checks = {
        "schema": report.get("schema") == expected_schema,
        "cuda": report.get("device") == "cuda:0",
        "seed": report.get("seed") == case["seed"],
        "body": report.get("body_sha256") == method["source_sha256"]["body"],
        "router": report.get("router_sha256") == method["source_sha256"]["router"],
        "actors": all(report.get("checkpoint_sha256", {}).get(name)
                      == method["source_sha256"][name]
                      for name in ("gait", "stop", "squat", "arms")),
        "case": (report.get("rail_v") == case["rail_v"]
                 and report.get("delay_ms") == case["delay_ms"]
                 and report.get("sole_friction") == case["sole_friction"]
                 and report.get("walk_velocity_m_s") == case["speed_m_s"]
                 and report.get("schedule_s") == {
                     "walk_end": case["walk_s"],
                     "stop_end": case["walk_s"] + case["stop_s"],
                     "squat_start": case["squat_start_s"],
                     "squat_end": case["squat_start_s"] + case["squat_s"],
                     "arm_start": case["arm_start_s"],
                     "arm_end": case["arm_start_s"] + case["arm_s"]}),
    }
    if label == "candidate":
        checks["candidate"] = (run_manifest is not None
                               and report.get("checkpoint_sha256", {}).get("stop_residual")
                               == run_manifest["candidate_checkpoint_sha256"]
                               and report.get("training_report_sha256")
                               == run_manifest["training_report_sha256"]
                               and report.get("score_plan_sha256")
                               == digest(BASE / "fresh-score-plan.json"))
    with np.load(paths[".npz"], allow_pickle=False) as trace:
        steps = {name: trace[name].shape[0] for name in trace.files
                 if name not in ("initial_qpos", "initial_qvel", "target_names", "dt_s")}
        checks["trace"] = (all(value == report["steps"] for value in steps.values())
                           and np.array_equal(trace["initial_qpos"], report["initial_qpos"])
                           and np.array_equal(trace["initial_qvel"], report["initial_qvel"]))
    frame_count = 0
    min_std = None
    shape = None
    for frame in imageio.get_reader(paths[".mp4"]):
        frame_count += 1
        shape = list(frame.shape)
        value = float(np.std(frame))
        min_std = value if min_std is None else min(min_std, value)
    checks["video"] = (frame_count == (report["steps"] + 1) // 2
                       and shape == [480, 640, 3]
                       and min_std is not None and min_std > 1.0)
    reasons = failures(report)
    checks["gate"] = report["full_continuity_gate_pass"] == (not reasons)
    return {"status": "source_checked" if all(checks.values()) else "source_fault",
            "checks": checks, "full_gate_pass": report["full_continuity_gate_pass"],
            "movement_pass_without_duty": not any(x != "rated_nonwalk_duty" for x in reasons),
            "failure_reasons": reasons, "steps": report["steps"],
            "walk_distance_m": report["walk_distance_m"],
            "stop_tail_mm": None if report["stop_tail_xy_m"] is None
            else report["stop_tail_xy_m"] * 1000,
            "settled_support_fraction": report["windows"].get("settled", {}).get("feet_both_fraction"),
            "settled_leg_rated_duty_fraction": report["windows"].get("settled", {}).get("leg_duty_fraction"),
            "report": str(paths[".json"].relative_to(ROOT)),
            "trace": str(paths[".npz"].relative_to(ROOT)),
            "video": str(paths[".mp4"].relative_to(ROOT)),
            "sha256": {ext: digest(paths[ext]) for ext in (".json", ".npz", ".mp4")},
            "decoded_frames": frame_count}


def main() -> None:
    freeze = json.loads((BASE / "candidate-score-freeze.json").read_text())
    files = {"score_plan": BASE / "fresh-score-plan.json",
             "baseline": SOURCE / "evaluate_portfolio_breadth_v2.py",
             "candidate": SOURCE / "evaluate_stop_feedback_v1.py",
             "runner": SOURCE / "run_stop_feedback_v1.py",
             "trainer": SOURCE / "train_stop_feedback_v1.py",
             "training_env": SOURCE / "stop_feedback_env.py",
             "training_rail": SOURCE / "stop_feedback_rail.py",
             "training_reward": SOURCE / "stop_feedback_reward.py",
             "batched_router": SOURCE / "batched_controller.py"}
    if any(digest(path) != freeze["sha256"][key] for key, path in files.items()):
        raise ValueError("frozen source changed")
    plan = json.loads(files["score_plan"].read_text())
    method_path = ROOT / plan["source_method"]
    if digest(method_path) != plan["source_method_sha256"]:
        raise ValueError("source method changed")
    method = json.loads(method_path.read_text())
    run_manifest_path = NATIVE / "score-run-manifest.json"
    run_manifest = (json.loads(run_manifest_path.read_text())
                    if run_manifest_path.exists() else None)
    if run_manifest is not None and run_manifest.get("score_freeze_sha256") != digest(
            BASE / "candidate-score-freeze.json"):
        raise ValueError("score run manifest names a different source freeze")
    cases = []
    for case in plan["cases"]:
        baseline = route(case, "baseline", method, run_manifest)
        candidate = route(case, "candidate", method, run_manifest)
        prefix_path = NATIVE / f"seed-{case['seed']}-prefix-check.json"
        prefix = json.loads(prefix_path.read_text()) if prefix_path.exists() else None
        valid = (baseline["status"] == "source_checked"
                 and candidate["status"] == "source_checked"
                 and prefix is not None and prefix.get("pass") is True
                 and prefix.get("seed") == case["seed"])
        cases.append({"seed": case["seed"], "status": "valid_pair" if valid else "not_valid_pair",
                      "baseline": baseline, "candidate": candidate,
                      "prefix_check": str(prefix_path.relative_to(ROOT)) if prefix else None,
                      "rescue": valid and not baseline["full_gate_pass"] and candidate["full_gate_pass"],
                      "regression": valid and baseline["full_gate_pass"] and not candidate["full_gate_pass"]})
    valid = [case for case in cases if case["status"] == "valid_pair"]
    result = {"schema": "robot-learning-lab.modular-stop-feedback-summary.v1",
              "claim_limit": "source-checked Direct Frame native whole-start development only",
              "planned_pairs": len(cases), "valid_pairs": len(valid),
              "complete": len(valid) == len(cases),
              "baseline_full_passes": sum(case["baseline"]["full_gate_pass"] for case in valid),
              "candidate_full_passes": sum(case["candidate"]["full_gate_pass"] for case in valid),
              "rescues": sum(case["rescue"] for case in valid),
              "regressions": sum(case["regression"] for case in valid),
              "score_freeze_sha256": digest(BASE / "candidate-score-freeze.json"),
              "cases": cases}
    path = BASE / "fresh-score-summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("planned_pairs", "valid_pairs", "complete", "baseline_full_passes",
                       "candidate_full_passes", "rescues", "regressions")}))


if __name__ == "__main__":
    main()
