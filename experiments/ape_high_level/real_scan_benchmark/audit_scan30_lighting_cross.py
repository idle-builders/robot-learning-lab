"""Audit complete crossed-lighting owner episodes and compare matched rooms."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import audit_scan30_unified_policy as task_audit
from freeze_scan20_shared_policy_sources import sha
import run_scan30_lighting_cross as cross


AUDIT = cross.ART / "scan30_lighting_cross_audit.json"
COMPARISON = cross.ART / "scan30_lighting_cross_comparison.json"


def main() -> None:
    task_audit.PRECOMMIT = cross.PRECOMMIT
    task_audit.REPORT = cross.REPORT
    task_audit.OUT = cross.OUT
    task_audit.AUDIT = AUDIT
    task_audit.main()
    audit = json.loads(AUDIT.read_text())
    frozen = json.loads(cross.PRECOMMIT.read_text())
    baseline_report = json.loads(cross.BASELINE_REPORT.read_text())
    crossed = json.loads(cross.REPORT.read_text())
    issues = list(audit["issues"])
    if sha(cross.BASELINE_REPORT) != frozen["baseline_report_sha256"]:
        issues.append("matched baseline report changed")
    if sha(cross.MOTION_TRACE) != frozen["historical_motion_trace_sha256"]:
        issues.append("historical motion source changed")
    baseline = {row["scene_key"]: row for row in baseline_report["rows"]}
    pairs = []
    if len(baseline) != 30 or len(crossed["rows"]) != 30:
        issues.append("paired denominator is not 30")
    photometry_traces_checked = 0
    for case, row in zip(frozen["cases"], crossed["rows"]):
        key = case["scene_key"]
        prior = baseline.get(key)
        if prior is None:
            issues.append(f"missing matched baseline: {key}")
            continue
        if prior["capture_group"] != row["capture_group"] or \
                prior["photometry"] != case["baseline_photometry"] or \
                row["photometry"] != case["cross_photometry"] or \
                prior["photometry"] == row["photometry"]:
            issues.append(f"mismatched scene or lighting: {key}")
        search_traces = 0
        for summary in (row.get("base_summary"), row.get("show_summary")):
            if not summary:
                continue
            folder = (cross.ROOT / summary).parent
            for path in folder.rglob("trace.json"):
                value = json.loads(path.read_text()).get("search_photometry")
                if value is None:
                    continue
                search_traces += 1
                photometry_traces_checked += 1
                if value != case["cross_photometry"]:
                    issues.append(f"search lighting mismatch: {key}: {path}")
        if not search_traces and row.get("owner_status") in (
                "teach_find_completed", "teach_find_completed_far_viewpoint"):
            issues.append(f"no observed search lighting trace: {key}")
        pairs.append({"scene_key": key, "capture_group": case["capture_group"],
            "cohort": case["cohort"],
            "baseline_photometry": prior["photometry"],
            "cross_photometry": row["photometry"],
            "baseline_result": prior["result"],
            "cross_result": row["result"],
            "baseline_show_again": bool(prior.get("show_summary")),
            "cross_show_again": bool(row.get("show_summary")),
            "baseline_wrong_questions": prior.get("wrong_candidate_questions"),
            "cross_wrong_questions": row.get("wrong_candidate_questions"),
            "cross_motion_authorized": row.get("motion_authorized"),
            "cross_final_action": row.get("final_action")})
    counts = Counter((p["baseline_result"], p["cross_result"]) for p in pairs)
    result = {"schema": "ape.scan30-lighting-cross-comparison.v1",
        "precommit_sha256": sha(cross.PRECOMMIT),
        "baseline_report_sha256": sha(cross.BASELINE_REPORT),
        "cross_report_sha256": sha(cross.REPORT),
        "task_audit_sha256": sha(AUDIT),
        "matched_rooms": len(pairs),
        "both_completed": counts[("completed", "completed")],
        "baseline_only_completed": sum(n for (a, b), n in counts.items()
            if a == "completed" and b != "completed"),
        "cross_only_completed": sum(n for (a, b), n in counts.items()
            if a != "completed" and b == "completed"),
        "neither_completed": sum(n for (a, b), n in counts.items()
            if a != "completed" and b != "completed"),
        "baseline_outcomes": dict(Counter(p["baseline_result"] for p in pairs)),
        "cross_outcomes": dict(Counter(p["cross_result"] for p in pairs)),
        "baseline_wrong_candidate_questions": sum(int(
            p["baseline_wrong_questions"] or 0) for p in pairs),
        "cross_wrong_candidate_offers": audit[
            "wrong_candidate_offers_in_stage_traces"],
        "baseline_show_again_attempts": sum(p["baseline_show_again"]
                                            for p in pairs),
        "cross_owner_questions": audit["owner_questions_in_stage_traces"],
        "cross_show_again_attempts": audit["show_again_attempts"],
        "cross_videos_decoded": audit["videos_decoded"],
        "photometry_traces_checked": photometry_traces_checked,
        "photometry_scope": "The camera transform changes item search, fresh relook, walk and final views, including those after an owner reshow. The initial owner approach and teaching image, and the reshow teaching image, remain at their original rendering; this does not test dim social or teaching views.",
        "pairs": pairs, "issues": issues,
        "limits": frozen["limits"]}
    COMPARISON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "matched_rooms", "both_completed", "baseline_only_completed",
        "cross_only_completed", "neither_completed", "issues")}, indent=2))


if __name__ == "__main__":
    main()
