"""Audit native-error provenance and owner-scene V142p outcomes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run import ROOT, sha
import stress_early_meeting_native_pose_v142p as v142p
import render_native_pose_meeting_v144p as v144p


SOURCE = Path(__file__)
OUT = v142p.OUT / "audit.json"


def main() -> None:
    path = v142p.OUT / "report.json"
    report = json.loads(path.read_text())
    checks = 0
    assert report["source_sha256"] == sha(v142p.SOURCE)
    assert report["native_heading_report_sha256"] == sha(v142p.NATIVE_REPORT)
    assert len(report["native_error_sources"]) == 3
    assert len(report["rows"]) == 12
    checks += 4
    for source in report["native_error_sources"]:
        trace_path = ROOT / source["trace_path"]
        assert sha(trace_path) == source["trace_sha256"]
        frames = json.loads(trace_path.read_text())
        frame = frames[source["frame"]]
        assert frame["frame"] == source["frame"]
        assert frame["actor_belief_healthy"] == source[
            "native_estimator_reported_healthy"]
        actual = np.asarray(frame["native_xy_evaluator_only_m"])
        estimate = np.asarray(frame["estimated_xy_m"])
        assert np.linalg.norm((actual-estimate)-np.asarray(source[
            "world_actual_minus_estimated_xy_m"])) < 2e-6
        assert abs((frame["native_yaw_evaluator_only_rad"]-
                    frame["estimated_yaw_rad"])-source[
                        "actual_minus_estimated_yaw_rad"]) < 2e-6
        checks += 5
    by_variant = {}
    for row in report["rows"]:
        key = (row["native_case"], row["mirrored_lateral_error"])
        by_variant.setdefault(key, set()).add(row["case"])
        assert row["prescribed_travel"]["minimum_scene_gap_m"] >= .05
        assert row["prescribed_travel"]["minimum_self_gap_m"] >= 0.
        assert row["owner_show"]["minimum_scene_gap_m"] >= .05
        assert row["owner_show"]["image_time_s"] == 17.4
        assert row["owner_show"]["end_time_s"] == 18.2
        assert row["actor_accepted_box"] is not None
        assert row["evaluator_box_iou"] >= .95
        frame_path = ROOT / row["owner_show"]["owner_image"][
            "actor_frame_path"]
        assert sha(frame_path) == row["owner_show"]["owner_image"][
            "actor_frame_sha256"]
        with np.load(frame_path) as packet:
            assert abs(float(packet["capture_time_s"])-17.4) < 1e-6
        checks += 9
    assert len(by_variant) == 6
    assert all(cases == set(v142p.CASES) for cases in by_variant.values())
    checks += 2
    unique_routes = {}
    for row in report["rows"]:
        key = (row["native_case"], row["mirrored_lateral_error"])
        unique_routes[key] = row["actor_swept_route"]["coverage_status"]
    assert "unknown" in unique_routes.values()
    assert not all(status == "observed_free"
                   for status in unique_routes.values())
    checks += 2
    minimum = min(row["owner_show"]["minimum_scene_gap_m"]
                  for row in report["rows"])
    video_report_path = v144p.OUT / "video-report.json"
    video_report = json.loads(video_report_path.read_text())
    assert video_report["source_sha256"] == sha(v144p.SOURCE)
    assert video_report["v142_report_sha256"] == sha(path)
    video_path = ROOT / video_report["video_path"]
    assert sha(video_path) == video_report["video_sha256"]
    reader = imageio.get_reader(video_path)
    try:
        decoded = sum(1 for _ in reader)
    finally:
        reader.close()
    assert decoded == len(video_report["frames"]) == 34
    assert abs(min(frame["native_error_transfer"]["evaluator_gap_m"]
                   for frame in video_report["frames"])-minimum) < 1e-6
    checks += 5
    audit = {"schema": "robot-learning-lab.ape-native-pose-meeting-audit.v1",
             "source_sha256": sha(SOURCE),
             "report_sha256": sha(path),
             "video_report_sha256": sha(video_report_path),
             "checks": checks,
             "minimum_show_gap_m": round(minimum, 5),
             "decoded_video_frames": decoded,
             "unique_route_statuses": {
                 f"{name}:{'mirror' if mirror else 'direct'}": status
                 for (name, mirror), status in unique_routes.items()},
             "scope": "Checks hashed native gait trace provenance, sampled "
             "actual-minus-estimate errors, actor-health fields, image "
             "capture hashes, task proposals and simulated geometry "
             "in V142p. Does not verify a physical reverse gait, a "
             "calibrated pose bound or safe movement permission."}
    OUT.write_text(json.dumps(audit, indent=2)+"\n")
    print(OUT, checks, "minimum", minimum)


if __name__ == "__main__":
    main()
