"""Paired active head-pan extension of the frozen two-view depth approach."""

from __future__ import annotations

from run_common_policy_20_object_generalization import ROOT, ART
import run_item_approach_sensor_20 as study


study.ARMS = ("dual_rgbd_active_pan",)
study.PRECOMMIT = ART / "item_approach_active_depth_20_precommit.json"
study.REPORT = ART / "item_approach_active_depth_20_report.json"
study.OUT = ROOT / "assets/real_indoor_scenes/item_approach_active_depth_20_20261015"


if __name__ == "__main__":
    study.main()
