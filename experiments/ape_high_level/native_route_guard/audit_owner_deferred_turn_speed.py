"""Independent changed-speed packet, action, geometry and video audit V65."""

from __future__ import annotations

import json
from pathlib import Path

import audit_owner_deferred_multi_surface as base_audit
import audit_owner_joint_hazard_response as media_audit
import audit_owner_low_prop_scan_planes as sensor_audit
import owner_deferred_turn_speed_stress as trial


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-deferred-turn-speed.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v64_report_sha256", trial.v64.OUT / "report.json"),
                      ("v60_source_sha256", trial.v60.__file__),
                      ("v63_source_sha256", trial.v63.__file__),
                      ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v52.sha(path)
    assert json.loads((trial.v64.OUT / "audit.json").read_text())["passed"]
    assert json.loads((trial.v63.v61.OUT / "audit.json").read_text())["passed"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    baseline = json.loads((trial.v64.OUT / "report.json").read_text())
    checked = []
    prefix_lengths = []
    limit = report["checkpoint_s"]+trial.TURN_START_S
    stages = ("post_alarm_range_watch", "pre_show_restage_step",
              "deferred_range_watch")
    for item in report["results"]:
        endpoint = trial.v64.PATHS[item["base_path"]]
        finish = item["finish_after_checkpoint_s"]
        route_fn = lambda t, endpoint=endpoint, finish=finish: trial.prop_at(
            t, report["checkpoint_s"], endpoint, finish)
        score = base_audit.check(
            {"path": item["variant"], "run": item["run"]},
            {"checkpoint_s": report["checkpoint_s"]},
            source, scenario, route_fn=route_fn, require_margin=False)
        score["base_path"] = item["base_path"]
        score["finish_after_checkpoint_s"] = finish
        checked.append(score)
        old = next(row for row in baseline["results"]
                   if row["path"] == item["base_path"])
        old_rows = json.loads(Path(old["run"]["trace"]["path"]).read_text())
        new_rows = json.loads(Path(item["run"]["trace"]["path"]).read_text())
        old_prefix = [row for row in old_rows if row["stage"] in stages and
                      row["time_s"] <= limit+.000005]
        new_prefix = [row for row in new_rows if row["stage"] in stages and
                      row["time_s"] <= limit+.000005]
        assert len(old_prefix) == len(new_prefix) and len(old_prefix) >= 40
        for a, b in zip(old_prefix, new_prefix):
            assert a["stage"] == b["stage"] and a["time_s"] == b["time_s"]
            sensor_audit.same_packet(a["actor_packet"], b["actor_packet"])
            assert a.get("ape_xy_m") == b.get("ape_xy_m")
        prefix_lengths.append(len(old_prefix))
    failures = [row for row in checked if row["minimum_prop_gap_m"] < .05]
    assert [row["path"] for row in failures] == ["far_west_fast"]
    assert failures[0]["minimum_prop_gap_m"] > 0.
    assert all(row["fresh_item_captures"] == 0 for row in checked)
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    video = comparison["video"]
    assert comparison["v64_report_sha256"] == trial.v52.sha(
        trial.v64.OUT / "report.json")
    assert comparison["v65_report_sha256"] == trial.v52.sha(
        trial.OUT / "report.json")
    assert media_audit.sha(video["path"]) == video["sha256"]
    assert media_audit.video_frames(video["path"], (620, 1080, 3)) == \
        video["frames"]
    result = {"passed": True, "rows": checked,
              "failed_margin_variants": [row["path"] for row in failures],
              "matching_pre_turn_actor_rows": prefix_lengths,
              "comparison_video_frames": video["frames"],
              "scope": "Frozen V64 controller with only the fixture-clear final-segment duration changed. Independently regenerated every low/mid scan, cluster alarm and prior-packet command, 20 ms person/prop gaps and decoded full/comparison videos. Fast far-west turn breaches the provisional 5 cm proxy floor; exact-pose kinematic model and scripted timing are not physical qualification."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json")


if __name__ == "__main__":
    main()
