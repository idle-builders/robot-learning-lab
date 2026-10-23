"""Run six frozen fresh V167 teacher/student reach pairs on native CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/reach-bc-v167"
LAUNCH = BASE / "launch-r2.json"
RESERVE = ROOT / "artifacts/ape-low-level-training-20261015/reach-own-trace-v166/fresh-score-plan.json"
V158 = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v158/new-seed-contact-plan.json"
TEACHER = Path(__file__).with_name("evaluate_aligned_reach_v165_cli.py")
STUDENT = Path(__file__).with_name("evaluate_reach_bc_v167_cli.py")
TEACHER_EVALUATOR = Path(__file__).with_name("evaluate_aligned_reach_v165.py")
STUDENT_EVALUATOR = Path(__file__).with_name("evaluate_reach_bc_v167.py")
PILOT = (24801, 24802, 24807, 24808, 24813, 24814)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-unix", type=int, required=True)
    parser.add_argument("--out", type=Path, default=BASE / "pilot")
    args = parser.parse_args()
    if not 0 < args.deadline_unix - time.time() <= 3600:
        raise ValueError("Use a provider-bounded deadline no more than one hour away")
    method = json.loads((BASE / "method.json").read_text())
    fit = json.loads((BASE / "fit-report.json").read_text())
    launch = json.loads(LAUNCH.read_text())
    reserve, v158 = json.loads(RESERVE.read_text()), json.loads(V158.read_text())
    if (method["schema"] != "robot-learning-lab.reach-v167-bc-method-freeze.v1"
            or launch["schema"] != "robot-learning-lab.reach-v167-pilot-launch-freeze.v2"
            or launch["source_sha256"]["pair_runner"] != sha(Path(__file__))
            or launch["source_sha256"]["teacher_evaluator"] != sha(TEACHER_EVALUATOR)
            or launch["source_sha256"]["teacher_wrapper"] != sha(TEACHER)
            or launch["source_sha256"]["student_evaluator"] != sha(STUDENT_EVALUATOR)
            or launch["source_sha256"]["student_wrapper"] != sha(STUDENT)
            or launch["source_sha256"]["checkpoint"] != sha(BASE / "checkpoint.pt")
            or launch["source_sha256"]["fit_report"] != sha(BASE / "fit-report.json")
            or launch["source_sha256"]["fresh_reserve"] != sha(RESERVE)
            or launch["source_sha256"]["pointer"] != sha(ROOT / "robot/ape-current-design.json")
            or method["source_sha256"]["fresh_score_reserve"] != sha(RESERVE)
            or method["source_sha256"]["pointer"] != sha(ROOT / "robot/ape-current-design.json")
            or fit["checkpoint_sha256"] != sha(BASE / "checkpoint.pt")
            or reserve["opened_visits"] != 0
            or tuple(method["fresh_pilot_seeds"]) != PILOT):
        raise ValueError("V167 fit/method/fresh score source changed")
    cases = {case["seed"]: case for case in reserve["cases"]}
    if len(cases) != 18 or any(cases[seed]["split"] != "pilot_evaluation" for seed in PILOT):
        raise ValueError("Fresh reserve split changed")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Refusing to replace any V167 pilot originals")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in PILOT:
        if time.time() > args.deadline_unix - 360:
            break
        case = cases[seed]
        pair = {"seed": seed, "stratum": case["stratum"], "case": case}
        for role, wrapper, evaluator in (("teacher", TEACHER, TEACHER_EVALUATOR),
                                         ("student", STUDENT, STUDENT_EVALUATOR)):
            specific = {**case, "id": f"v167-{role}-{seed}"}
            original = native.EVALUATOR
            try:
                native.EVALUATOR = wrapper
                result = native.run_case(specific, {"source_sha256": {
                    **v158["native_source_sha256"], "evaluator": sha(evaluator),
                }}, args.out)
            finally:
                native.EVALUATOR = original
            report_path = ROOT / result["result"]
            trace_path = ROOT / result["trace"]
            aligned_path = trace_path.with_name(trace_path.stem + "-aligned.npz")
            report = json.loads(report_path.read_text())
            if (report["trace_sha256"] != sha(trace_path)
                    or report["aligned_trace_sha256"] != sha(aligned_path)
                    or (role == "student" and
                        report["student_actor_sha256"] != fit["checkpoint_sha256"])):
                raise ValueError(f"Native original source failed: {role} {seed}")
            pair[role] = {**result, "report_sha256": sha(report_path),
                          "trace_sha256": sha(trace_path),
                          "aligned_trace": str(aligned_path.relative_to(ROOT)),
                          "aligned_trace_sha256": sha(aligned_path),
                          "student_route_active": report.get("student_route_active")}
            write(args.out / "partial.json", {
                "schema": "robot-learning-lab.reach-v167-pilot-partial.v1",
                "completed_pairs": len(rows), "open_pair": pair, "rows": rows})
        rows.append(pair)
        write(args.out / "partial.json", {
            "schema": "robot-learning-lab.reach-v167-pilot-partial.v1",
            "completed_pairs": len(rows), "rows": rows})
        print(json.dumps({"seed": seed, "teacher_pass": pair["teacher"]["pass"],
                          "student_pass": pair["student"]["pass"],
                          "student_route_active": pair["student"]["student_route_active"]}), flush=True)
    write(args.out / "run-report.json", {
        "schema": "robot-learning-lab.reach-v167-pilot-run.v1",
        "classification": "fresh native paired development contact on rejected Direct Frame; not independent score",
        "method_sha256": sha(BASE / "method.json"),
        "launch_sha256": sha(LAUNCH),
        "fit_report_sha256": sha(BASE / "fit-report.json"),
        "planned_pairs": len(PILOT), "completed_pairs": len(rows), "rows": rows})


if __name__ == "__main__":
    main()
