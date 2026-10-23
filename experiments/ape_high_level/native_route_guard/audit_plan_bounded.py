"""Audit V24's saved no-map packets and actor skill-horizon provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_online_corridor as baseline_audit
import online_corridor_trial as baseline
import plan_bounded_trial as bounded
from kitchen_trial import sha


def main() -> None:
    old_out = baseline.OUT
    baseline.OUT = bounded.OUT
    try:
        baseline_audit.main()
    finally:
        baseline.OUT = old_out
    report_path = bounded.OUT / "report.json"
    report = json.loads(report_path.read_text())
    audit_path = bounded.OUT / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["schema"] = "robot-learning-lab.ape-robocasa-plan-bounded-audit.v1"
    audit["checks"]["v24_source"] = (
        report["plan_bounded_source_sha256"] == sha(Path(bounded.__file__)))
    audit["checks"]["v23_baseline"] = (
        report["v23_report_sha256"] == sha(old_out / "report.json"))
    audit["checks"]["skill_contract"] = (
        report["actor_skill_contract"]["requested_forward_skill_end_s"] ==
        bounded.SKILL_END_S and
        report["actor_skill_contract"]["historical_stop_tail_advance_budget_m"] ==
        bounded.HISTORICAL_STOP_ADVANCE_M)
    for row in report["results"]:
        key = f'{row["world"]}:{row["case"]}'
        with np.load(row["trace"]["path"], allow_pickle=False) as saved:
            packets = [json.loads(raw) for raw in saved["packet_json"]]
        audit["checks"][f"bounded_packets:{key}"] = all(
            abs(p["route_guard"]["skill_remaining_s"] -
                max(0., bounded.SKILL_END_S-p["captured_at_s"])) < 1e-4 and
            abs(p["route_guard"]["forecast_horizon_s"] -
                min(bounded.guard.HORIZON_S,
                    max(0., bounded.SKILL_END_S-p["captured_at_s"]) +
                    bounded.STOP_EQUIVALENT_S)) < 1e-4
            for p in packets)
    audit["pass"] = all(audit["checks"].values())
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pass": audit["pass"],
                      "summary": audit["summary"]}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V24 audit failed")


if __name__ == "__main__":
    main()
