"""Render the V134p positive-left-shoulder owner meeting comparison."""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import render_early_meeting_v130p as v130p
import validate_early_forward_arm_v134p as v134p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-forward-arm-video"


def main() -> None:
    original_out = v130p.OUT
    original_target = v130p.TARGET_SHOULDER_RAD
    try:
        v130p.OUT = OUT
        v130p.TARGET_SHOULDER_RAD = v134p.SHOULDER
        v130p.main()
    finally:
        v130p.OUT = original_out
        v130p.TARGET_SHOULDER_RAD = original_target
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    assert report["target_left_shoulder_pitch_rad"] == v134p.SHOULDER
    report["wrapper_source_sha256"] = sha(SOURCE)
    report["v134_report_sha256"] = sha(v134p.OUT / "report.json")
    report["scope"] += (" V134p tests this +0.6rad arm path and fresh "
                        "taught-item RGB-D separately; V135p still leaves "
                        "the actor route unsupported under pose samples.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
