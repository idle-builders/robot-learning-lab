"""Run frozen fresh whole-seed teacher/student V4 no-reset pairs on one CUDA pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/evaluate_walk_squat_arms.py"
CANDIDATE = ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/evaluate_v4_shared_bc.py"
SCREEN_PLAN = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-screen-method.json"
PILOT_INDICES = (0, 5, 8, 11)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log: Path, deadline_unix: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "experiments/compact_morphology/mjlab_ape/src") + ":" + str(ROOT)
    env["MUJOCO_GL"] = "egl"
    with log.open("x") as output:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, check=False,
                                   timeout=max(1, int(deadline_unix - time.time() - 10)))
    if completed.returncode:
        raise RuntimeError(f"Native visit failed with exit {completed.returncode}: {log}")


def verify(path: Path, trace_path: Path, case: dict, method: dict, role: str) -> dict:
    report = json.loads(path.read_text())
    if (report["seed"] != case["seed"] or report["body_sha256"] != method["body"]["sha256"]
            or report["device"] != "cuda:0" or report["rail_v"] != case["rail_v"]
            or report["delay_ms"] != case["delay_ms"]
            or report["walk_velocity_m_s"] != case["walk_velocity_m_s"]
            or abs(report["dt_s"] - 0.02) > 1e-8):
        raise RuntimeError(f"Native visit identity differs: {path}")
    if role == "teacher":
        if any(report["checkpoint_sha256"][name] != spec["sha256"]
               for name, spec in method["actors"].items()):
            raise RuntimeError(f"Teacher checkpoint identity differs: {path}")
    elif report["student_checkpoint_sha256"] != method["student_checkpoint"]["sha256"]:
        raise RuntimeError(f"Student checkpoint identity differs: {path}")
    with np.load(trace_path, allow_pickle=False) as data:
        pre, post = data["pre_qpos"], data["post_qpos"]
        obs = data["actor_obs"]
        target = data["processed_target"]
        if (len(pre) != report["steps"] or len(post) != len(pre)
                or obs.shape != (len(pre), 72) or target.shape != (len(pre), 21)
                or not all(np.isfinite(item).all() for item in (pre, post, obs, target))
                or np.max(np.abs(pre[0] - report["initial_qpos"])) > 1e-8
                or (len(pre) > 1 and np.max(np.abs(post[:-1] - pre[1:])) > 1e-8)):
            raise RuntimeError(f"Native visit trace differs: {trace_path}")
        first_pre, first_obs = pre[0].copy(), obs[0].copy()
    failures = [name for name, field in (
        ("squat_depth_return", "squat_depth_return_pass"),
        ("arm_rise_return", "arm_rise_return_pass"),
        ("hand_or_body_contact", "no_hand_or_body_contact"),
        ("support_windows", "support_windows_pass"),
        ("rated_duty_windows", "rated_duty_windows_pass")) if not report[field]]
    if report["ended_early"] or report["steps"] < 1200:
        failures.append("early_termination")
    if report["walk_distance_m"] < 0.06:
        failures.append("walk_progress")
    if report["stop_tail_xy_m"] is None or report["stop_tail_xy_m"] > 0.03:
        failures.append("stop_tail_drift")
    if report.get("max_target_mapping_gap", 0) > 1e-5:
        failures.append("action_mapping")
    if report["switch_qpos_gap"] is None or report["switch_qpos_gap"] >= 1e-9:
        failures.append("no_reset_continuity")
    if bool(report["full_continuity_gate_pass"]) != (len(failures) == 0):
        raise RuntimeError(f"Saved full gate and explicit failure reasons differ: {path}")
    return {"report": str(path), "report_sha256": sha(path), "trace": str(trace_path),
            "trace_sha256": sha(trace_path), "steps": report["steps"],
            "pass": bool(report["full_continuity_gate_pass"]),
            "failure_reasons": failures,
            "walk_m": report["walk_distance_m"], "stop_tail_m": report["stop_tail_xy_m"],
            "support": bool(report["support_windows_pass"]),
            "rated_duty": bool(report["rated_duty_windows_pass"]),
            "contact": bool(report["no_hand_or_body_contact"]),
            "squat": bool(report["squat_depth_return_pass"]),
            "arms": bool(report["arm_rise_return_pass"]),
            "mapping_gap": report.get("max_target_mapping_gap"),
            "_first_pre": first_pre, "_first_obs": first_obs}


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pod-start-unix", type=int, required=True)
    parser.add_argument("--deadline-unix", type=int, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    plan = json.loads(SCREEN_PLAN.read_text())
    if (method["screen_plan_sha256"] != sha(SCREEN_PLAN)
            or method["cases"] != [plan["reserved_fresh_evaluation"][i] for i in PILOT_INDICES]
            or len(set(case["seed"] for case in method["cases"])) != len(PILOT_INDICES)
            or set(case["seed"] for case in method["cases"]) &
               set(case["seed"] for case in plan["screen"])):
        raise RuntimeError("Fresh score cases differ from disjoint frozen reserve subset")
    if (method["schema"] != "robot-learning-lab.broad-handoff-v4-l4-paired-score-method.v1"
            or sha(Path(__file__)) != method["paired_runner_sha256"]
            or sha(SOURCE) != method["source_evaluator_sha256"]
            or sha(CANDIDATE) != method["candidate_evaluator_sha256"]
            or sha(ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/v4_action_map.py") != method["action_map_sha256"]
            or sha(ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/build_v4_command_bank.py") != method["bank_builder_sha256"]
            or sha(ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/train_v4_shared_bc.py") != method["fit_source_sha256"]
            or sha(ROOT / method["student_checkpoint"]["path"]) != method["student_checkpoint"]["sha256"]
            or sha(ROOT / method["bank"]["path"]) != method["bank"]["sha256"]
            or sha(ROOT / method["fit_report"]["path"]) != method["fit_report"]["sha256"]
            or sha(ROOT / method["parity_report"]["path"]) != method["parity_report"]["sha256"]
            or sha(ROOT / method["student_sentinel"]["path"]) != method["student_sentinel"]["sha256"]
            or sha(ROOT / method["body"]["path"]) != method["body"]["sha256"]
            or not 0 < args.deadline_unix - args.pod_start_unix <= 3600):
        raise RuntimeError("Frozen score source, checkpoint, body or GPU cap differs")
    import torch
    checkpoint = torch.load(ROOT / method["student_checkpoint"]["path"],
                            map_location="cpu", weights_only=True)
    bank = json.loads((ROOT / method["bank"]["path"]).read_text())
    fit = json.loads((ROOT / method["fit_report"]["path"]).read_text())
    parity = json.loads((ROOT / method["parity_report"]["path"]).read_text())
    sentinel = json.loads((ROOT / method["student_sentinel"]["path"]).read_text())
    bank_seeds = {row["seed"] for row in bank["entries"]}
    parity_entry = next((row for row in bank["entries"] if row["seed"] == parity["seed"]), None)
    if (not parity["pass"] or parity["device"] != "cuda:0"
            or parity_entry is None
            or parity["source_report_sha256"] != parity_entry["report_sha256"]
            or parity["source_trace_sha256"] != parity_entry["trace_sha256"]
            or any(value > 1e-4 for value in parity["gaps"].values())
            or sentinel["device"] != "cuda:0" or sentinel["steps"] != 1
            or sentinel["seed"] not in bank_seeds
            or sentinel["student_checkpoint_sha256"] != method["student_checkpoint"]["sha256"]
            or sentinel["max_target_mapping_gap"] > 1e-5
            or fit["bank_sha256"] != method["bank"]["sha256"]
            or fit["checkpoint_sha256"] != method["student_checkpoint"]["sha256"]
            or set(checkpoint["train_seeds"] + checkpoint["validation_seeds"]) != bank_seeds
            or set(checkpoint["train_seeds"]) & set(checkpoint["validation_seeds"])
            or set(case["seed"] for case in method["cases"]) & bank_seeds):
        raise RuntimeError("Learner fit, opened CUDA sentinels or fresh seed isolation invalid")
    for source in method["actors"].values():
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"Teacher actor changed: {source['path']}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("report.json", "partial.json", "incomplete.json")):
        raise RuntimeError("Scored original already exists; never duplicate a fresh pair")
    rows = []
    for case in method["cases"]:
        if time.time() >= args.deadline_unix - method["minimum_pair_start_reserve_s"]:
            break
        stem = f"seed-{case['seed']}-rail-{case['rail_v']}-delay-{case['delay_ms']}"
        pair = {}
        for role, evaluator in (("teacher", SOURCE), ("student", CANDIDATE)):
            if time.time() >= args.deadline_unix - method["minimum_visit_start_reserve_s"]:
                (out / "incomplete.json").write_text(json.dumps({
                    "reason": "provider deadline before next original visit",
                    "case": case, "next_role": role,
                    "completed_pairs": len(rows)}, indent=2) + "\n")
                return
            report_path = out / f"{stem}-{role}.json"
            trace_path = out / f"{stem}-{role}.npz"
            video_path = out / f"{stem}-{role}.mp4"
            log_path = out / f"{stem}-{role}.log"
            if any(path.exists() for path in (report_path, trace_path, video_path, log_path)):
                raise RuntimeError(f"Original has existing evidence; cannot rerun: {stem}-{role}")
            command = [sys.executable, str(evaluator), "--body", str(ROOT / method["body"]["path"]),
                       "--seed", str(case["seed"]), "--walk-velocity", str(case["walk_velocity_m_s"]),
                       "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                       "--out", str(report_path), "--trace", str(trace_path),
                       "--device", "cuda:0", "--native-evaluation"]
            if role == "teacher":
                for arg, key in (("--gait-actor", "gait"), ("--stop-actor", "stop"),
                                 ("--squat-actor", "squat"), ("--arm-actor", "arms")):
                    command += [arg, str(ROOT / method["actors"][key]["path"])]
            else:
                command += ["--student-actor", str(ROOT / method["student_checkpoint"]["path"])]
            try:
                run(command, log_path, args.deadline_unix)
            except Exception as error:
                (out / "incomplete.json").write_text(json.dumps({
                    "reason": type(error).__name__, "detail": str(error),
                    "case": case, "role": role,
                    "log": str(log_path), "completed_pairs": len(rows)}, indent=2) + "\n")
                raise
            item = verify(report_path, trace_path, case, method, role)
            item.update(log=str(log_path), log_sha256=sha(log_path),
                        video=None, video_source="retained native qpos; local source-bound render pending")
            pair[role] = item
        first_qpos_gap = float(np.max(np.abs(pair["teacher"]["_first_pre"] - pair["student"]["_first_pre"])))
        first_obs_gap = float(np.max(np.abs(pair["teacher"]["_first_obs"] - pair["student"]["_first_obs"])))
        if first_qpos_gap > method["matched_entry_tolerance"] or first_obs_gap > method["matched_entry_tolerance"]:
            raise RuntimeError(f"Matched original entry differs: {case['seed']}: {first_qpos_gap}, {first_obs_gap}")
        rows.append({"case": case, "first_qpos_gap": first_qpos_gap,
                     "first_obs_gap": first_obs_gap,
                     "teacher": public(pair["teacher"]), "student": public(pair["student"]),
                     "rescue": not pair["teacher"]["pass"] and pair["student"]["pass"],
                     "regression": pair["teacher"]["pass"] and not pair["student"]["pass"]})
        (out / "partial.json").write_text(json.dumps({"method_sha256": sha(args.method),
            "planned": len(method["cases"]), "completed": len(rows), "rows": rows}, indent=2) + "\n")
    result = {"schema": "robot-learning-lab.broad-handoff-v4-l4-paired-score.v1",
              "method_sha256": sha(args.method), "planned": len(method["cases"]),
              "completed": len(rows), "teacher_passes": sum(r["teacher"]["pass"] for r in rows),
              "student_passes": sum(r["student"]["pass"] for r in rows),
              "rescues": sum(r["rescue"] for r in rows),
              "regressions": sum(r["regression"] for r in rows), "rows": rows}
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
