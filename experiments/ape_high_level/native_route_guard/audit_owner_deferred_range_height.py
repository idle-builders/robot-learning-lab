"""Independent monitor-height packet, action, geometry and video audit V66."""

from __future__ import annotations

import json
from pathlib import Path

import audit_owner_deferred_multi_surface as base_audit
import audit_owner_joint_hazard_response as media_audit
import audit_owner_low_prop_scan_planes as sensor_audit
import owner_deferred_range_height_ablation as trial


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-deferred-range-height.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v64_report_sha256", trial.v64.OUT / "report.json"),
                      ("v65_report_sha256", trial.v65.OUT / "report.json"),
                      ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v52.sha(path)
    assert json.loads((trial.v64.OUT / "audit.json").read_text())["passed"]
    assert json.loads((trial.v65.OUT / "audit.json").read_text())["passed"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    baseline = json.loads((trial.v64.OUT / "report.json").read_text())
    original_scanner = trial.runner.observed_packet
    checked, matched_prefix = [], []
    for item in report["results"]:
        endpoint = trial.v64.PATHS[item["base_path"]]
        finish = item["finish_after_checkpoint_s"]
        route_fn = lambda t, endpoint=endpoint, finish=finish: trial.v65.prop_at(
            t, report["checkpoint_s"], endpoint, finish)
        scanner = trial.v51.v46.plane_packet(
            original_scanner, trial.PROFILES[item["profile"]])
        row = base_audit.check(
            {"path": item["variant"], "run": item["run"]},
            {"checkpoint_s": report["checkpoint_s"]}, source, scenario,
            route_fn=route_fn, monitor_scanner=scanner,
            require_margin=False)
        row["profile"] = item["profile"]
        row["base_path"] = item["base_path"]
        row["finish_after_checkpoint_s"] = finish
        checked.append(row)
        old = next(x for x in baseline["results"]
                   if x["path"] == item["base_path"])
        old_rows = json.loads(Path(old["run"]["trace"]["path"]).read_text())
        new_rows = json.loads(Path(item["run"]["trace"]["path"]).read_text())
        stages = ("post_alarm_range_watch", "pre_show_restage_step")
        old_prefix = [x for x in old_rows if x["stage"] in stages]
        new_prefix = [x for x in new_rows if x["stage"] in stages]
        assert len(old_prefix) == len(new_prefix) and len(old_prefix) >= 30
        for a, b in zip(old_prefix, new_prefix):
            assert a["stage"] == b["stage"] and a["time_s"] == b["time_s"]
            sensor_audit.same_packet(a["actor_packet"], b["actor_packet"])
            assert a.get("ape_xy_m") == b.get("ape_xy_m")
        matched_prefix.append(len(old_prefix))
    lookup = {(row["profile"], row["base_path"],
               row["finish_after_checkpoint_s"]): row for row in checked}
    low_normal = lookup[("low_05cm", "far_west_turn", 5.20)]
    low_fast = lookup[("low_05cm", "far_west_turn", 4.85)]
    mid_normal = lookup[("mid_30cm", "far_west_turn", 5.20)]
    mid_fast = lookup[("mid_30cm", "far_west_turn", 4.85)]
    assert low_normal["alarm_s"] is not None and \
        low_normal["minimum_prop_gap_m"] >= .05
    assert low_fast["alarm_s"] is not None and \
        low_fast["minimum_prop_gap_m"] < .05
    assert mid_normal["alarm_s"] is None and \
        mid_normal["minimum_prop_gap_m"] < 0.
    assert mid_fast["alarm_s"] is None and \
        mid_fast["minimum_prop_gap_m"] < 0.
    for profile in trial.PROFILES:
        outward = lookup[(profile, "outbound_control", 5.20)]
        assert outward["alarm_s"] is None and outward["commanded_m"] == 0.
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    video = comparison["video"]
    assert comparison["report_sha256"] == trial.v52.sha(
        trial.OUT / "report.json")
    assert media_audit.sha(video["path"]) == video["sha256"]
    assert media_audit.video_frames(video["path"], (620, 1080, 3)) == \
        video["frames"]
    result = {"passed": True, "rows": checked,
              "matching_dual_plane_planning_rows": matched_prefix,
              "comparison_video_frames": video["frames"],
              "scope": "Identical fused 5+30cm hold/restage packets, then isolated 5cm or 30cm deferred-monitor packets with frozen multi-surface policy. Independently regenerated each ray packet, cluster alarm and prior-packet response, 20ms person/prop gaps, and decoded all episode/comparison videos. Ideal planes and rejected historical kinematic body do not select physical sensors."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json")


if __name__ == "__main__":
    main()
