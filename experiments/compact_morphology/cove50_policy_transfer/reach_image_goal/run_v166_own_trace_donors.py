"""Collect source-aligned RGB reach imitation episodes on opened development starts.

The instrumented evaluator is the teacher source. This runner makes no V95
trajectory-identity claim and never opens held contact or fresh score families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
METHOD = ROOT / "artifacts/ape-low-level-training-20261015/reach-own-trace-v166/method.json"
PLAN = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v158/new-seed-contact-plan.json"
SCORE_PLAN = ROOT / "artifacts/ape-low-level-training-20261015/reach-own-trace-v166/fresh-score-plan.json"
DEST = ROOT / "artifacts/ape-low-level-training-20261015/reach-own-trace-v166/native-donors"
WRAPPER = Path(__file__).with_name("evaluate_aligned_reach_v165_cli.py")
EVALUATOR = Path(__file__).with_name("evaluate_aligned_reach_v165.py")
VERIFY = Path(__file__).with_name("verify_aligned_reach_v166.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def check_sources(method: dict, plan: dict) -> None:
    expected = method["source_sha256"]
    observed = {
        "runner": sha(Path(__file__)), "v158_plan": sha(PLAN),
        "fresh_score_plan": sha(SCORE_PLAN),
        "v165_evaluator": sha(EVALUATOR), "v165_wrapper": sha(WRAPPER),
        "v166_verifier": sha(VERIFY), "v95_case_runner": sha(Path(native.__file__)),
        "pointer": sha(ROOT / "robot/ape-current-design.json"),
    }
    if observed != expected or plan["native_source_sha256"]["pointer"] != expected["pointer"]:
        raise ValueError("V166 method/source closure differs")
    sources = native.v83.v16.SOURCES
    for key in ("body", "squat", "alternate_actor", "specialist_actor",
                "reference", "geometry", "calibration", "camera_freeze",
                "virtual_rgb_observer", "calibrated_rgb_observer", "pointer"):
        if sha(Path(sources[key])) != plan["native_source_sha256"][key]:
            raise ValueError(f"V158 source changed: {key}")
    if sha(Path(native.v83.v57.ACTOR)) != plan["native_source_sha256"]["actor"]:
        raise ValueError("V158 reach actor changed")
    inherited = method["inherited_own_trace"]
    for key in ("report", "trace", "aligned_trace", "source_check"):
        item = inherited[key]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Inherited own-trace evidence changed: {key}")
    saved = json.loads((ROOT / inherited["source_check"]["path"]).read_text())
    if (saved["seed"] != inherited["seed"] or not saved["source_alignment_pass"]
            or not saved["full_contact_margin_pass"]):
        raise ValueError("Inherited own-trace teacher is not source-valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=("pilot", "remaining"), required=True)
    parser.add_argument("--deadline-unix", type=int, required=True)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    plan = json.loads(PLAN.read_text())
    check_sources(method, plan)
    if (method["schema"] != "robot-learning-lab.reach-v166-own-trace-donor-method.v1"
            or method["claim"] != "own-trajectory aligned development imitation; no V95 parity claim"
            or method["held_contact_seeds"] != [24606, 24612, 24615, 24618, 24621, 24624]
            or method["inherited_own_trace"]["seed"] != 24601):
        raise ValueError("V166 method identity differs")
    chosen = {case["seed"]: case for case in plan["cases"] if case["split"] == "development"}
    score = json.loads(SCORE_PLAN.read_text())
    score_seeds = {case["seed"] for case in score["cases"]}
    if (score["opened_visits"] != 0 or len(score_seeds) != 18 or
            score_seeds & set(chosen) or score_seeds & set(method["held_contact_seeds"])):
        raise ValueError("Fresh score reserve was opened or overlaps development")
    expected = [seed for seed, case in chosen.items()
                if seed != 24601 and method["source_automatic_passes"].get(str(seed))]
    if (set(expected) != set(method["donor_seeds"]) or
            set(method["donor_seeds"]) & set(method["held_contact_seeds"]) or
            set(method["batches"]["pilot"] + method["batches"]["remaining"]) !=
            set(method["donor_seeds"])):
        raise ValueError("V166 donor list differs from opened V158 passing development pool")
    batch = method["batches"][args.batch]
    if not 0 < args.deadline_unix - time.time() <= 3600:
        raise ValueError("Use a future provider-bounded deadline of at most one hour")
    DEST.mkdir(parents=True, exist_ok=True)
    batch_report = DEST / f"{args.batch}-batch.json"
    if batch_report.exists():
        raise FileExistsError("This donor batch already has a result; do not rerun originals")
    rows = []
    for seed in batch:
        if time.time() >= args.deadline_unix - 360:
            break
        case = {**chosen[seed], "id": f"v166-donor-{seed}"}
        stem = f"seed-{seed}-{case['id']}"
        if any((DEST / f"{stem}{suffix}").exists() for suffix in
               (".json", ".npz", ".log", "-aligned.npz")):
            raise FileExistsError(f"Original V166 donor evidence already exists: {stem}")
        original = native.EVALUATOR
        try:
            native.EVALUATOR = WRAPPER
            result = native.run_case(case, {"source_sha256": {
                **plan["native_source_sha256"],
                "evaluator": method["source_sha256"]["v165_evaluator"],
            }}, DEST)
        finally:
            native.EVALUATOR = original
        report = ROOT / result["result"]
        trace = ROOT / result["trace"]
        aligned = trace.with_name(trace.stem + "-aligned.npz")
        command = [sys.executable, str(VERIFY), "--report", str(report),
                   "--trace", str(trace), "--aligned-trace", str(aligned)]
        check = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        source = json.loads(check.stdout)
        if source["seed"] != seed or not source["source_alignment_pass"]:
            raise ValueError(f"V166 synchronized source failed: {seed}")
        row = {"seed": seed, "stratum": case["stratum"],
               "selection_mode": result["selection_mode"],
               "native_full_gate_pass": result["pass"],
               "continuous_contact_s": result["contact_s"],
               "imitation_admit": bool(result["pass"] and source["full_contact_margin_pass"]),
               "faults": result["faults"] + source["physical_or_rgb_faults"],
               "report": result["result"], "report_sha256": sha(report),
               "trace": result["trace"], "trace_sha256": sha(trace),
               "aligned_trace": str(aligned.relative_to(ROOT)),
               "aligned_trace_sha256": sha(aligned),
               "log": result["log"], "log_sha256": sha(ROOT / result["log"])}
        rows.append(row)
        write(DEST / f"{args.batch}-partial.json", {
            "method_sha256": sha(METHOD), "planned": len(batch),
            "completed": len(rows), "rows": rows})
        print(json.dumps({"seed": seed, "admit": row["imitation_admit"],
                          "contact_s": row["continuous_contact_s"]}), flush=True)
    write(batch_report, {"schema": "robot-learning-lab.reach-v166-own-trace-donors.v1",
                         "claim": method["claim"], "method_sha256": sha(METHOD),
                         "batch": args.batch, "planned": len(batch),
                         "completed": len(rows), "accepted": sum(row["imitation_admit"] for row in rows),
                         "rows": rows})


if __name__ == "__main__":
    main()
