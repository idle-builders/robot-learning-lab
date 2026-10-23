"""Probe northwest show views beyond the revealed west-station prop endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import probe_layout2_joint_owner_stations_v771 as base


SOURCE = Path(__file__)
OUT = base.side.ROOT / "artifacts/ape-robocasa-layout2-northwest-owner-station-probe-v772"
CANDIDATES = {"far_northwest": [-.3, -2.6],
              "northwest": [-.2, -2.7],
              "northwest_shallow": [-.1, -2.6],
              "north": [0., -2.6],
              "west_control": [.18, -3.01]}


if __name__ == "__main__":
    base.OUT, base.CANDIDATES = OUT, CANDIDATES
    base.main()
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["wrapper_source_sha256"] = base.screen.sha(SOURCE)
    report["development_notice"] = "These candidates were added after V771 showed zero can pixels at five eastern/central views; their fixed scripted prop endpoint is evaluator-only and not actor admission."
    path.write_text(json.dumps(report, indent=2) + "\n")
