"""Apply the V22 saved-evidence audit to V23's one-rule ablation."""

from __future__ import annotations

import json
from pathlib import Path

import audit_online_cluster_guard as baseline_audit
import online_cluster_guard as baseline
import online_corridor_trial as corridor
from kitchen_trial import sha


def main() -> None:
    old_out = baseline.OUT
    baseline.OUT = corridor.OUT
    try:
        baseline_audit.main()
    finally:
        baseline.OUT = old_out
    report = json.loads((corridor.OUT / "report.json").read_text())
    path = corridor.OUT / "audit.json"
    audit = json.loads(path.read_text())
    audit["schema"] = "robot-learning-lab.ape-robocasa-online-corridor-audit.v1"
    audit["checks"]["wrapper_source"] = (
        report["wrapper_source_sha256"] == sha(Path(corridor.__file__)))
    audit["checks"]["v22_baseline"] = (
        report["v22_report_sha256"] == sha(old_out / "report.json"))
    audit["checks"]["only_lateral_gate"] = (
        report["algorithm"]["unknown_lateral_m"] == baseline.guard.RISK_RADIUS_M)
    audit["pass"] = all(audit["checks"].values())
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "summary": audit["summary"]}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V23 audit failed")


if __name__ == "__main__":
    main()
