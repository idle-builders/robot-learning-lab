"""Independently replay the frozen cost-ordered owner meeting-pose continuation."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART, sha
import audit_meeting_pose_replan_3 as baseline_audit
import run_meeting_pose_cost_order_3 as trial


OUT = ART / "meeting_pose_cost_order_3_audit.json"


def main() -> None:
    frozen = json.loads(trial.PRECOMMIT.read_text())
    report = json.loads(trial.REPORT.read_text())
    baseline = json.loads((ART / "meeting_pose_replan_3_report.json").read_text())
    if (report["processed"] != len(frozen["cases"]) or
            len(report["rows"]) != len(frozen["cases"]) or
            len(baseline["rows"]) != len(frozen["cases"])):
        raise ValueError("Case cohort differs from frozen comparison")
    for case in frozen["cases"]:
        if sha(ROOT / case["source"]) != frozen["source_station_sha256"][case[
                "scene_key"]]:
            raise ValueError("Source station changed after freeze")
    rows = [baseline_audit.audit_one(row, case,
        frozen["required_wait_clearance_m"],
        proposal_provider=trial.ranked_proposals,
        max_actions=trial.MAX_ACTIONS,
        max_candidates=trial.MAX_CANDIDATES,
        max_per_candidate=trial.MAX_PER_CANDIDATE)
        for row, case in zip(report["rows"], frozen["cases"])]
    paired = [{"scene_key": a["scene_key"],
        "fixed_grid_outcome": a["outcome"],
        "cost_order_outcome": b["outcome"],
        "cost_order_accepted_forward_steps": b["accepted_forward_steps"],
        "cost_order_turns": b["turns"]}
        for a,b in zip(baseline["rows"], rows)]
    output = {"schema": "ape.meeting-pose-cost-order-3-audit.v1",
        "precommit": str(trial.PRECOMMIT.relative_to(ROOT)),
        "runs": len(rows),
        "baseline_fresh_owner_confirmed": sum(x["outcome"] ==
            "owner_confirmed_represented_item" for x in baseline["rows"]),
        "cost_order_fresh_owner_confirmed": sum(x["outcome"] ==
            "owner_confirmed_represented_item" for x in rows),
        "paired": paired,
        "limits": "Same three development-exposed stations, no capture-disjoint test; ideal 17cm disk, scripted owner and virtual sensors."}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
