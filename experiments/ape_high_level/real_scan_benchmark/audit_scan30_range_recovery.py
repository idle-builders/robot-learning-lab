"""Audit full-task range recovery against the matched healthy and stop baselines."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import audit_scan30_unified_policy as whole_task
from freeze_scan20_shared_policy_sources import sha
import run_scan30_range_recovery as run


AUDIT = run.ART / "scan30_range_recovery_audit.json"
COMPARISON = run.ART / "scan30_range_recovery_comparison.json"


def packet_path(row: dict) -> Path | None:
    if row.get("show_summary"):
        show = json.loads((run.ROOT / row["show_summary"]).read_text())
        action = show.get("action") or {}
        path = action.get("walk_trace")
        return (run.ROOT / path).parent.parent / \
            "independent_range_packets.json" if path else None
    first = json.loads((run.ROOT / row["base_summary"]).read_text())
    path = first.get("walk_trace")
    return (run.ROOT / path).parent.parent / \
        "independent_range/packets.json" if path else None


def main() -> None:
    whole_task.PRECOMMIT = run.PRECOMMIT
    whole_task.REPORT = run.REPORT
    whole_task.OUT = run.OUT
    whole_task.AUDIT = AUDIT
    whole_task.main()
    audit = json.loads(AUDIT.read_text())
    frozen = json.loads(run.PRECOMMIT.read_text())
    healthy = json.loads(run.HEALTHY.read_text())
    stopped = json.loads(run.FAULT.read_text())
    current = json.loads(run.REPORT.read_text())
    issues = list(audit["issues"])
    for path, key in ((run.HEALTHY, "healthy_report_sha256"),
                      (run.FAULT, "fault_report_sha256"),
                      (run.MOTION, "motion_trace_sha256")):
        if sha(path) != frozen[key]:
            issues.append(f"frozen comparator changed: {path.name}")
    baseline = {r["scene_key"]: r for r in healthy["rows"]}
    fault = {r["scene_key"]: r for r in stopped["rows"]}
    pairs = []
    if len(current["rows"]) != 30 or len(baseline) != 30 or len(fault) != 30:
        issues.append("matched denominator is not thirty")
    exposed = recovered = resumed = 0
    for case, row in zip(frozen["cases"], current["rows"]):
        key = case["scene_key"]
        prior, old_fault = baseline.get(key), fault.get(key)
        if prior is None or old_fault is None:
            issues.append(f"missing matched comparator: {key}")
            continue
        event = row.get("range_fault") or {}
        record = row.get("range_recovery") or {}
        if event:
            exposed += 1
            if not event.get("missing_packet_observed") or not record.get(
                    "outage_observed") or record.get("first_result") != \
                    "source_gap":
                issues.append(f"no stop on observed outage: {key}")
            path = packet_path(row)
            if path is None or not path.is_file():
                issues.append(f"missing independent range packets: {key}")
            else:
                packets = json.loads(path.read_text())["packets"]
                if not any(p["source_health"] == "missing" and
                        p["capture_wall_s"] >= event["gap_wall_s"] for
                        p in packets):
                    issues.append(f"no saved missing packet: {key}")
            if record.get("restored_health", {}).get("authorized"):
                recovered += 1
            if record.get("resumed"):
                resumed += 1
                if record.get("relook_action") != \
                        "owner_confirmed_visual_identity" or not record.get(
                        "relook_source_health", {}).get("authorized") or \
                        record.get("relook_goal_xz_m") is None:
                    issues.append(f"resumed without fresh supported item: {key}")
                first_trace = run.ROOT / record["first_trace"]
                second_trace = run.ROOT / record["second_trace"]
                if not first_trace.is_file() or not second_trace.is_file():
                    issues.append(f"missing segmented walk trace: {key}")
                else:
                    first_walk = json.loads(first_trace.read_text())
                    second_walk = json.loads(second_trace.read_text())
                    if not first_walk["actions"] or first_walk["actions"][-1][
                            "reason"] != "source_gap":
                        issues.append(f"first walk did not hold on gap: {key}")
                    if not second_walk["actions"] or second_walk["actions"][0][
                            "observation_wall_s"] < record[
                            "relook_source_health"]["checked_wall_s"]:
                        issues.append(f"resumed before new-view gate: {key}")
                    for action in first_walk["actions"] + second_walk[
                            "actions"]:
                        if action["decision"] in ("turn", "forward") and not \
                                (action.get("source_guard") or {}).get(
                                    "authorized"):
                            issues.append(f"unguarded move: {key}")
            if row["result"] == "completed" and not record.get("resumed"):
                issues.append(f"completed after outage without recovery: {key}")
        pairs.append({"scene_key": key, "capture_group": case["capture_group"],
            "cohort": case["cohort"],
            "healthy_result": prior["result"],
            "old_gap_result": old_fault["result"],
            "recovery_result": row["result"],
            "outage_exposed": bool(event),
            "source_restored": bool(record.get("restored_health", {}).get(
                "authorized")),
            "fresh_relook_supported": bool(record.get("relook_action") ==
                "owner_confirmed_visual_identity" and record.get(
                "relook_source_health", {}).get("authorized")),
            "resumed": bool(record.get("resumed")),
            "owner_reshow": bool(row.get("show_summary")),
            "wrong_candidate_questions": row.get("wrong_candidate_questions")})
    result = {"schema": "ape.scan30-range-recovery-comparison.v1",
        "precommit_sha256": sha(run.PRECOMMIT),
        "report_sha256": sha(run.REPORT),
        "task_audit_sha256": sha(AUDIT),
        "matched_rooms": len(pairs), "outage_exposed": exposed,
        "source_restored": recovered, "resumed_after_relook": resumed,
        "healthy_outcomes": dict(Counter(p["healthy_result"] for p in pairs)),
        "old_gap_outcomes": dict(Counter(p["old_gap_result"] for p in pairs)),
        "recovery_outcomes": dict(Counter(p["recovery_result"] for p in pairs)),
        "wrong_candidate_offers": audit[
            "wrong_candidate_offers_in_stage_traces"],
        "owner_questions": audit["owner_questions_in_stage_traces"],
        "videos_decoded": audit["videos_decoded"],
        "issues": issues, "pairs": pairs, "limits": frozen["limits"]}
    COMPARISON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "matched_rooms", "outage_exposed", "source_restored",
        "resumed_after_relook", "healthy_outcomes", "old_gap_outcomes",
        "recovery_outcomes", "wrong_candidate_offers", "issues")},
        indent=2))


if __name__ == "__main__":
    main()
