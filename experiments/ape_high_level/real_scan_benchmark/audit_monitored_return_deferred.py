"""Audit deferred diagnostic video against live sensing and item outcomes."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART, relative
import audit_monitored_return_stress as base_audit
import audit_monitored_return_depth_backup as depth_audit
import run_monitored_return_stress_deferred as trial


OUT = ART / "monitored_return_deferred_audit.json"


def audit_one(row: dict, case: dict, lane: dict) -> dict:
    if row["arm"] == "range_outage_depth_deferred_video":
        scored = depth_audit.audit_one(row, case, lane,
            expected_arm="range_outage_depth_deferred_video")
    else:
        scored = base_audit.audit_one(case, row)
    episode = json.loads((ROOT / row["trace"]).read_text())
    intervals = []
    for path in episode["phase_traces"]:
        phase = json.loads((ROOT / path).read_text())
        if phase["monitor_video_render_mode"] != "post_run_diagnostic_replay":
            raise ValueError("Monitor RGB video blocked live sensor acquisition")
        times = [p["capture_wall_s"] for p in phase["monitor_packets"]]
        intervals.extend(b-a for a,b in zip(times,times[1:]))
    return {**scored,
        "max_monitor_packet_interval_s": max(intervals, default=0.),
        "phase_count": len(episode["phase_traces"])}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    cases = {c["id"]: c for c in trial.cohort.case_rows()}
    lanes = {x["case"]: x for x in json.loads(trial.LANES.read_text())["rows"]}
    rows = [audit_one(r, cases[r["case"]], lanes[r["case"]])
            for r in report["rows"]]
    healthy = [r for r in rows if r["arm"] == "healthy_deferred_video"]
    backup = [r for r in rows if r["arm"] ==
              "range_outage_depth_deferred_video"]
    result = {"schema": "ape.monitored-return-deferred-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "healthy_runs": len(healthy), "backup_runs": len(backup),
        "healthy_gap_breaches": sum(r["gap_under_5cm"] for r in healthy),
        "backup_gap_breaches": sum(r["gap_under_5cm"] for r in backup),
        "backup_aimable": sum(r["head_aimable"] for r in backup),
        "backup_aimable_gap_breaches": sum(r["head_aimable"] and
                                           r["gap_under_5cm"] for r in backup),
        "healthy_owner_confirmed": sum(r["outcome"] ==
            "fresh_owner_confirmed" for r in healthy),
        "backup_owner_confirmed": sum(r["outcome"] ==
            "fresh_owner_confirmed" for r in backup),
        "max_monitor_packet_interval_s": max(
            (r["max_monitor_packet_interval_s"] for r in rows), default=0.),
        "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"},indent=2))


if __name__ == "__main__":
    main()
