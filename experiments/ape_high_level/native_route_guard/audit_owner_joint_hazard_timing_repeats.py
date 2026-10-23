"""Audit retained V41 repeated full owner continuations."""

from __future__ import annotations

import json
from pathlib import Path

import owner_joint_hazard_timing_repeats as repeats
import audit_owner_joint_hazard_response as base
import render_owner_joint_hazard_timing_repeats as renderer


def main():
    report = json.loads((repeats.OUT / "report.json").read_text())
    assert report["schema"] == \
        "robot-learning-lab.ape-owner-joint-hazard-timing-repeats.v1"
    assert report["source_sha256"] == base.sha(repeats.__file__)
    assert report["v41_source_sha256"] == base.sha(repeats.trial.__file__)
    assert report["v38_source_sha256"] == base.sha(repeats.trial.v40.v38.__file__)
    assert report["scene_sha256"] == base.sha(
        repeats.trial.v40.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == base.sha(repeats.trial.v40.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == base.sha(
        repeats.trial.ROOT / "robot/ape-current-design.json")
    assert report["cases"] == {case: repeats.trial.PROP_CASES[case]
                               for case in repeats.CASES}
    assert len(report["results"]) == report["repeat_count"]*len(repeats.CASES)
    source = repeats.trial.v40.v38.original()
    summary = source["summary"]
    policy = json.loads(repeats.trial.v40.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    rows = [dict(base.audit_result(item, report, summary, source, scenario),
                 repeat=item["repeat"]) for item in report["results"]]
    assert [(row["prop_case"], row["repeat"]) for row in rows] == [
        (case, n) for case in repeats.CASES
        for n in range(1, report["repeat_count"]+1)]
    comparison = json.loads((repeats.OUT / "comparison.json").read_text())
    assert comparison["schema"] == \
        "robot-learning-lab.ape-owner-joint-hazard-repeat-videos.v1"
    assert comparison["report_sha256"] == base.sha(repeats.OUT / "report.json")
    assert comparison["source_sha256"] == base.sha(renderer.__file__)
    assert comparison["base_renderer_sha256"] == base.sha(renderer.base.__file__)
    assert set(comparison["overhead_videos"]) == {
        f"{item['prop_case']}-repeat-{item['repeat']}"
        for item in report["results"]}
    for video in comparison["overhead_videos"].values():
        assert base.sha(video["path"]) == video["sha256"]
        assert base.video_frames(video["path"], (680, 800, 3)) == video["frames"]
    audit = {"passed": True, "rows": rows,
             "overhead_videos": comparison["overhead_videos"],
             "scope": "Repeated full V41 owner-route continuations with unchanged code and scripted worlds. Each repeat independently checks retained RGB-D, packet-causal swerve, 20ms evaluator geometry and decoded complete videos. Variation arises from real inference/render wall scheduling; it is not a calibrated latency distribution."}
    (repeats.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
