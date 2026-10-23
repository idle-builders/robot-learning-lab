"""V23 development ablation: narrow V22's unknown lateral gate to route radius."""

from __future__ import annotations

import json
from pathlib import Path

import online_cluster_guard as online
import run as guard
from kitchen_trial import sha


OUT = online.kitchen.OUT / "v23-online-corridor"


def main() -> None:
    prior_out = online.OUT
    prior_unknown = guard.UNKNOWN_LATERAL_M
    online.OUT = OUT
    guard.UNKNOWN_LATERAL_M = guard.RISK_RADIUS_M
    try:
        online.main()
    finally:
        online.OUT = prior_out
        guard.UNKNOWN_LATERAL_M = prior_unknown
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.ape-robocasa-online-corridor-guard.v1"
    report["method"] += (" V23 development-only ablation changes immature unknown"
                         " lateral gate from .40m+surface extent to .20m+surface"
                         " extent, matching the mature route-radius proxy;")
    report["method"] += " no other actor rule or scene input changes."
    report["wrapper_source_sha256"] = sha(Path(__file__))
    report["v22_report_sha256"] = sha(prior_out / "report.json")
    report["development_on_v22_failures"] = True
    path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
