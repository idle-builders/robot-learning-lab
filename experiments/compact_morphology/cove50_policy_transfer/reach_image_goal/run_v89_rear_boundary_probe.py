"""Opened native paired check of the V89 rear-camera arm route extension."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v89_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v89"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    older_path = BASE / "v87/prior75-retention-plan.json"
    route_path = BASE / "v87/routes63-retention-plan.json"
    v88_probe_path = BASE / "v88/native-specialist-boundary-probe/matrix.json"
    older = json.loads(older_path.read_text())
    routes = json.loads(route_path.read_text())
    v88_probe = json.loads(v88_probe_path.read_text())
    prior_rescue = next(row for row in v88_probe["cases"] if row["seed"] == 10514)
    if not prior_rescue["pass"] or prior_rescue["contact_s"] < 1.2:
        raise ValueError("Need a passing V88 10514 boundary rescue")
    prior = {case["seed"]: case for case in older["cases"]}
    cases = [{**prior[seed], "expected_selection_mode": mode}
             for seed, mode in ((10433, "v16_specialist_locked"),
                                (10540, "v16_locked"),
                                (10514, "v21_scoped_y_locked"),
                                (10520, "v16_specialist_locked"),
                                (10522, "v21_scoped_y_locked"))]
    cases.append({**next(case for case in routes["cases"]
                         if case["id"] == "large-36-pitch-high"),
                  "expected_selection_mode": "v16_specialist_locked"})
    hashes = {**older["source_sha256"], "evaluator": sha(native.EVALUATOR)}
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical design pointer changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v89-rear-boundary-probe-plan.v1",
        "role": "opened_paired_native_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "v87_prior75_plan_sha256": sha(older_path),
        "v87_routes63_plan_sha256": sha(route_path),
        "v88_10514_report_sha256": sha(ROOT / prior_rescue["result"]),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "hypothesis": "10433 lost contact because its valid RGB entry moved 0.023 mm above "
                      "the existing specialist high-Y boundary and fell to the locked actor.",
        "change": "Extend specialist routing to observed positive-X 0–1.5 mm, Y 6–6.5 mm "
                  "only in the rear-camera quadrant (world X −0.05–0 m, Y<−0.08 m). "
                  "Retain V88's 1.5 mm positive-X specialist/scoped boundary and all "
                  "actor, RGB, timing, body and motor identities.",
        "gate": "Both opened failures 10433 and 10514 regain at least 1.2 s native "
                "contact and expected arm routes; all six paired visits pass full RGB, "
                "support, motor, floor, collision and clipping gates. The 10540 locked "
                "control and specialist/ scoped controls retain their expected routes. "
                "Full variation and retention are separate prerequisites for promotion.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "rear-boundary-probe-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V89 probe/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-rear-boundary-probe"
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        iterator = iter(cases)
        pending = {}
        for _ in range(2):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(native.run_case, case, plan, dest)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            if row["selection_mode"] != case["expected_selection_mode"]:
                row["faults"].append("arm route differs from frozen expectation")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in cases if item["id"] in rows]
            complete = len(ordered) == len(cases)
            rescues = [row for row in ordered if row["seed"] in (10433, 10514)]
            matrix = {
                "schema": "robot-learning-lab.cove50-v89-rear-boundary-probe.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(plan_path), "complete": complete,
                "case_count": len(ordered),
                "pass_count": sum(row["pass"] for row in ordered),
                "minimum_contact_s": min(row["contact_s"] for row in ordered),
                "probe_gate_met": (complete and all(row["pass"] for row in ordered)
                                   and len(rescues) == 2
                                   and all(row["contact_s"] >= 1.2 for row in rescues)),
                "cases": ordered,
            }
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode", "faults")}),
                  flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
