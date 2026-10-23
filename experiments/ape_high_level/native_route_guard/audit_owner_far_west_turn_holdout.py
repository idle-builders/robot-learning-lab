"""Independent fixture, packet, geometry and pre-turn audit for V61 failure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_owner_deferred_show as v60_audit
import audit_owner_fixture_clear_prop as fixture_audit
import audit_owner_joint_hazard_response as media_audit
import audit_owner_low_prop_scan_planes as sensor_audit
import owner_far_west_turn_holdout as trial


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-far-west-turn-holdout.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v60_source_sha256", trial.v60.__file__),
                      ("v60_report_sha256", trial.v60.OUT / "report.json"),
                      ("scene_sha256", trial.v60.runner.v35.moving.SCENE_XML),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v60.v52.sha(path)
    source = trial.v60.runner.original()
    policy = json.loads(trial.v60.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    v50_report = json.loads((trial.v60.v50.OUT / "report.json").read_text())
    fixture = fixture_audit.room_gap(trial.END_XY_M, v50_report,
                                     source, scenario)
    assert fixture["minimum_static_fixture_gap_m"] >= .05
    paths = trial.v60.v59.v56.PATHS
    try:
        paths["far_west_turn"] = trial.END_XY_M
        checked = v60_audit.check(
            {"path": "far_west_turn", "run": report["run"]},
            {"checkpoint_s": report["checkpoint_s"]},
            source, scenario, require_margin=False)
    finally:
        del paths["far_west_turn"]
    assert checked["minimum_person_gap_m"] >= .05
    assert checked["minimum_prop_gap_m"] < 0.
    old = json.loads((trial.v60.OUT / "report.json").read_text())
    old_item = next(row for row in old["results"]
                    if row["path"] == "new_west_south_turn")
    old_rows = json.loads(Path(old_item["run"]["trace"]["path"]).read_text())
    new_rows = json.loads(Path(report["run"]["trace"]["path"]).read_text())
    limit = report["checkpoint_s"]+trial.v60.v50.WAYPOINTS[-1][0]
    relevant = ("post_alarm_range_watch", "pre_show_restage_step",
                "deferred_range_watch")
    old_prefix = [row for row in old_rows if row["stage"] in relevant and
                  row["time_s"] <= limit+.000005]
    new_prefix = [row for row in new_rows if row["stage"] in relevant and
                  row["time_s"] <= limit+.000005]
    assert len(old_prefix) == len(new_prefix) and len(old_prefix) >= 40
    for old_row, new_row in zip(old_prefix, new_prefix):
        assert old_row["stage"] == new_row["stage"]
        assert old_row["time_s"] == new_row["time_s"]
        sensor_audit.same_packet(old_row["actor_packet"],
                                 new_row["actor_packet"])
        assert old_row.get("ape_xy_m") == new_row.get("ape_xy_m")
        assert old_row.get("actor_action") == new_row.get("actor_action")
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    video = comparison["video"]
    assert comparison["v60_report_sha256"] == trial.v60.v52.sha(
        trial.v60.OUT / "report.json")
    assert comparison["v61_report_sha256"] == trial.v60.v52.sha(
        trial.OUT / "report.json")
    assert media_audit.sha(video["path"]) == video["sha256"]
    assert media_audit.video_frames(video["path"], (620, 1080, 3)) == \
        video["frames"]
    result = {"passed": True, "row": checked,
              "prop_static_fixture_clearance": fixture,
              "matching_pre_turn_actor_rows": len(old_prefix),
              "comparison_video_frames": video["frames"],
              "scope": "Frozen V60 actor and item-free defer branch on a new fixture-clear final prop turn. Recomputed every packet and 20 ms MuJoCo gap, confirmed identical pre-turn actor history and decoded full episode video. This is a retained failure: the reached west pose is not protected against the larger unmodeled turn."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json")


if __name__ == "__main__":
    main()
