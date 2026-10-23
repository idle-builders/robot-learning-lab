"""Audit active two-view head-pan sensor arm over all 20 rooms."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART
import audit_item_approach_sensor_20 as audit


audit.study.ARMS = ("dual_rgbd_active_pan",)
audit.study.PRECOMMIT = ART / "item_approach_active_depth_20_precommit.json"
audit.study.REPORT = ART / "item_approach_active_depth_20_report.json"
audit.study.OUT = ROOT / "assets/real_indoor_scenes/item_approach_active_depth_20_20261015"
audit.AUDIT = ART / "item_approach_active_depth_20_audit.json"


if __name__ == "__main__":
    frozen = json.loads(audit.study.PRECOMMIT.read_text())
    if "active_pan" not in frozen:
        raise ValueError("Active pan rule was not frozen")
    audit.main()
