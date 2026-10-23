"""Pair same-worker healthy and live-range-outage whole-task outcomes."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
DEFINITION = ART / "scan30_same_runtime_healthy_precommit.json"
HEALTHY = ART / "scan30_same_runtime_healthy_report.json"
FAULT = ART / "scan30_walk_range_gap_report.json"
HEALTHY_AUDIT = ART / "scan30_same_runtime_healthy_audit.json"
FAULT_AUDIT = ART / "scan30_walk_range_gap_audit.json"
REPORT = ART / "scan30_same_runtime_range_gap_comparison.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(path: str | None) -> dict | None:
    return json.loads((ROOT / path).read_text()) if path else None


def pre_fault(row: dict) -> dict:
    base = summary(row["base_summary"])
    show = summary(row.get("show_summary"))
    action = (show or {}).get("action") or {}
    teaching = (show or {}).get("teaching") or {}
    return {"owner_status": row.get("owner_status"),
        "base_present": base.get("present_action"),
        "base_absent": base.get("absent_action"),
        "base_fresh": base.get("fresh_action"),
        "base_motion_authorized": base.get("motion_authorized"),
        "owner_reshow": bool(show),
        "show_teach_confirmed": teaching.get("teach_confirmed"),
        "show_present": (teaching.get("present") or {}).get("action"),
        "show_absent": (teaching.get("absent") or {}).get("action"),
        "show_fresh": action.get("fresh_action"),
        "show_motion_authorized": action.get("motion_authorized"),
        "goal_xz_m": (action.get("fresh_goal_xz_m") if show else
                      base.get("fresh_goal_xz_m")),
        "wrong_candidate_questions": row.get("wrong_candidate_questions")}


def main() -> None:
    frozen = json.loads(DEFINITION.read_text())
    healthy = json.loads(HEALTHY.read_text())
    fault = json.loads(FAULT.read_text())
    healthy_audit = json.loads(HEALTHY_AUDIT.read_text())
    fault_audit = json.loads(FAULT_AUDIT.read_text())
    issues = []
    if digest(ART / "scan30_unified_policy_precommit.json") != frozen[
            "source_precommit_sha256"] or digest(ART /
            "scan30_walk_range_gap_precommit.json") != frozen[
            "fault_precommit_sha256"]:
        issues.append("Precommitted source/policy changed")
    if digest(ROOT / frozen["wrapper_source"]) != frozen["wrapper_sha256"]:
        issues.append("Same-runtime healthy wrapper changed")
    if healthy_audit["issues"] or fault_audit["issues"]:
        issues.append("One source arm failed independent audit")
    hr, fr = healthy["rows"], fault["rows"]
    if len(hr) != 30 or len(fr) != 30:
        issues.append("Not thirty paired full tasks")
    rows = []
    for nominal, outage in zip(hr, fr):
        key = nominal["scene_key"]
        if key != outage["scene_key"] or nominal[
                "capture_group"] != outage["capture_group"]:
            issues.append(f"Pair identity mismatch: {key}")
            continue
        before, after = pre_fault(nominal), pre_fault(outage)
        nominal_goal, fault_goal = before.pop("goal_xz_m"), after.pop(
            "goal_xz_m")
        gap = (float(np.linalg.norm(np.asarray(nominal_goal, float) -
                                   np.asarray(fault_goal, float)))
               if nominal_goal is not None and fault_goal is not None else None)
        matched = before == after and ((nominal_goal is None and
            fault_goal is None) or (gap is not None and gap <= .02))
        rows.append({"scene_key": key, "capture_group": nominal["capture_group"],
            "pre_fault_matched": matched, "goal_gap_m": gap,
            "healthy_result": nominal["result"],
            "fault_result": outage["result"],
            "healthy_walk_result": nominal.get("walk_result"),
            "fault_walk_result": outage.get("walk_result"),
            "fault_exposure": outage.get("fault_exposure"),
            "fault_observed": outage.get("fault_observed"),
            "pre_fault_healthy": before, "pre_fault_fault": after})
    exposed = [r for r in rows if r["fault_observed"]]
    matched_exposed = [r for r in exposed if r["pre_fault_matched"]]
    unexposed = [r for r in rows if not r["fault_observed"]]
    report = {"schema": "ape.scan30-same-runtime-range-gap-comparison.v1",
        "paired_cases": len(rows),
        "healthy_outcomes": dict(Counter(r["healthy_result"] for r in rows)),
        "fault_outcomes": dict(Counter(r["fault_result"] for r in rows)),
        "fault_exposures": dict(Counter(r["fault_exposure"] for r in rows)),
        "pre_fault_matched": sum(r["pre_fault_matched"] for r in rows),
        "observed_faults": len(exposed),
        "matched_observed_faults": len(matched_exposed),
        "unexposed_identical_outcomes": sum(
            r["healthy_result"] == r["fault_result"] for r in unexposed),
        "observed_faults_withheld_completion": sum(
            r["fault_result"] != "completed" for r in exposed),
        "matched_healthy_completions_stopped": sum(
            r["healthy_result"] == "completed" and
            r["fault_result"] == "safe_unresolved" for r in matched_exposed),
        "rows": rows, "issues": issues,
        "limits": "The two arms share source files, model revisions and one GPU/Python runtime; pre-fault equivalence is checked per room before attributing a changed outcome to the injected range loss. The route planner still uses a simulator-synchronous ideal fan, so this isolates the separate range health/near-body gate, not lidar-versus-camera route sensing. Static captures, scripted owner and unselected 17 cm proxy remain."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("paired_cases",
        "healthy_outcomes", "fault_outcomes", "fault_exposures",
        "pre_fault_matched", "observed_faults",
        "matched_observed_faults", "unexposed_identical_outcomes",
        "observed_faults_withheld_completion",
        "matched_healthy_completions_stopped",
        "issues")}, indent=2))


if __name__ == "__main__":
    main()
