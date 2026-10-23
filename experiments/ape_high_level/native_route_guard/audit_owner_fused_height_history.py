"""Audit V48 low/mid packets, range-history choice and full geometry."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audit_owner_joint_hazard_response as prior
import audit_owner_low_prop_scan_planes as v46_audit
import owner_fused_height_history as trial
import render_owner_fused_height_history as renderer


def check(item, report, source, scenario):
    # Reuse V46's independent ray and 20 ms geometry replay for the *same*
    # low-cylinder geometry, but point it at the V48 fused-height run.
    adapter = SimpleNamespace(
        v38=trial.v38, v41=trial.v41,
        PROP_HALF_HEIGHT_M=trial.v46.PROP_HALF_HEIGHT_M,
        PROP_CENTER_Z_M=trial.v46.PROP_CENTER_Z_M,
        PROFILES={"dual_05_30cm": (.05, .30)},
        plane_packet=trial.v46.plane_packet)
    previous_trial = v46_audit.trial
    try:
        v46_audit.trial = adapter
        result = v46_audit.check(
            {**item, "profile": "dual_05_30cm"}, report, source, scenario)
    finally:
        v46_audit.trial = previous_trial
    rows = json.loads(Path(item["run"]["trace"]["path"]).read_text())
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    resume = hold[-1]
    assert resume["actor_action"] == \
        "resume_after_three_fresh_clear_packets"
    expected = trial.v47.novel_forward_returns(
        hold[-2]["actor_packet"], resume["actor_packet"])
    assert expected == item["novel_forward_returns"] and expected
    assert item["choice"] == "northeast"
    start = np.asarray(source["summary"]["ape_after_episode_xy_m"])
    score = trial.v47.v44.endpoint_score(
        resume["actor_packet"], start, trial.v47.NORTHEAST)
    assert score["eligible"]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"]
    old_pose, old_row = start, resume
    for row in steps:
        pose = np.asarray(row["ape_xy_m"])
        distance = row["commanded_step_m"]
        assert abs(row["time_s"]-old_row["time_s"]-
                   trial.v38.HOLD_PERIOD_S) < 1e-5
        assert np.linalg.norm(pose-(old_pose+
                        trial.v47.NORTHEAST*distance)) < 1e-8
        assert row["actor_decision"]["novel_forward_returns"] == expected
        points = trial.v41.JointHazardResponse.endpoints(
            old_row["actor_packet"], old_pose)
        minimum = float(np.min(np.linalg.norm(points-pose, axis=1)))
        assert abs(minimum-row["actor_decision"][
            "minimum_endpoint_gap_m"]) < 1e-8
        assert minimum >= trial.v47.v44.MIN_ENDPOINT_GAP_M
        old_pose, old_row = pose, row
    assert np.linalg.norm(old_pose-start) >= trial.v47.v44.TARGET_M-.001
    assert result["above_provisional_5cm_both"]
    return result


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-fused-height-history.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v38_source_sha256", trial.v38.__file__),
                      ("v46_source_sha256", trial.v46.__file__),
                      ("v47_source_sha256", trial.v47.__file__),
                      ("v46_report_sha256", trial.v46.OUT / "report.json"),
                      ("v47_report_sha256", trial.v47.OUT / "report.json"),
                      ("scene_sha256", trial.v38.v35.moving.SCENE_XML),
                      ("v36_report_sha256", trial.v38.SOURCE),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == prior.sha(path), key
    assert report["scan_heights_m"] == [.05, .30]
    assert report["prop_geometry"] == {
        "radius_m": trial.v46.PROP_RADIUS_M,
        "half_height_m": trial.v46.PROP_HALF_HEIGHT_M,
        "center_z_m": trial.v46.PROP_CENTER_Z_M}
    source = trial.v38.original()
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == \
        "robot-learning-lab.ape-owner-fused-history-video.v1"
    assert comparison["v46_report_sha256"] == prior.sha(
        trial.v46.OUT / "report.json")
    assert comparison["v48_report_sha256"] == prior.sha(
        trial.OUT / "report.json")
    assert comparison["source_sha256"] == prior.sha(renderer.__file__)
    video = comparison["video"]
    assert video["sha256"] == prior.sha(video["path"])
    assert prior.video_frames(video["path"], tuple(video["shape"])) == \
        video["frames"]
    audit = {"passed": True, "rows": checked,
             "comparison_video": video,
             "scope": "Independent V46 packet/20 ms geometry/video replay on V48 data, plus stationary scan-history trigger and prior-packet NE step replay. Historical kinematic body, scripted owner/prop, ideal scan planes."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
