"""V195p: independent packet replay of V194's static-reference cue."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from compare_standoff_range_layouts_v165p import front_left_roi
from run import ROOT, sha
import run_static_reference_owner_v194p as v194p


SOURCE = Path(__file__)
OUT = v194p.OUT


def replay(row: dict, params: dict) -> int:
    video_path = ROOT / row["video_path"]
    assert sha(video_path) == row["video_sha256"]
    reader = imageio.get_reader(video_path)
    try:
        assert reader.get_data(0).shape[:2] == (480, 1280)
        assert reader.get_data(27).shape[:2] == (480, 1280)
    finally:
        reader.close()
    monitor_path = ROOT / row["monitor"]["path"]
    assert sha(monitor_path) == row["monitor"]["sha256"]
    monitor = json.loads(monitor_path.read_text())
    assert len(monitor["frames"]) == 55
    assert monitor["alarm"] == row["front_alarm"]
    baseline = []
    old_count, old_centroid = 0, None
    checks = 0
    for index, frame in enumerate(monitor["frames"]):
        for path_key, sha_key in (
            ("packet_path", "packet_sha256"),
            ("owner_depth_packet_path", "owner_depth_packet_sha256")):
            path = ROOT / frame[path_key]
            assert sha(path) == frame[sha_key]
            checks += 1
        with np.load(ROOT / frame["packet_path"], allow_pickle=False) as packet:
            points = packet["front_four_points_xyz_m"]
            pose = packet["estimated_pose_xyyaw"]
        selected = points[front_left_roi(points, pose)]
        if index < params["pre_show_frames"]:
            baseline.append(selected)
        reference = np.concatenate(baseline)
        novel = np.empty((0, 3))
        if index >= params["pre_show_frames"] and len(selected):
            distances = np.linalg.norm(
                selected[:, None, :]-reference[None, :, :], axis=2)
            novel = selected[np.min(distances, axis=1) >
                             params["novel_world_point_distance_m"]]
        centroid = np.mean(novel, axis=0) if len(novel) else None
        delta = (float(np.linalg.norm(centroid-old_centroid))
                 if centroid is not None and old_centroid is not None else None)
        cue = (len(novel) >= params["minimum_novel_returns"] and
               (old_count < params["minimum_novel_returns"] or
                (delta is not None and delta >=
                 params["minimum_novel_centroid_motion_m"])))
        assert frame["static_reference_points"] == len(reference)
        assert frame["new_low_surface_returns"] == len(novel)
        assert frame["front_left_cue_now"] == cue
        if delta is None:
            assert frame["new_low_surface_centroid_displacement_m"] is None
        else:
            assert abs(frame["new_low_surface_centroid_displacement_m"]-
                       delta) < 1e-9
        old_count, old_centroid = len(novel), centroid
    alarm = row["front_alarm"]
    if alarm:
        frame = monitor["frames"][alarm["capture_index"]]
        assert frame["capture_time_s"] == alarm["capture_time_s"]
        assert (frame["front_left_first_cue"] if
                alarm["side"] == "range_new_low_surface" else
                frame["owner_depth_first_cue"])
    return checks


def main() -> None:
    index = json.loads((OUT / "index.json").read_text())
    assert index["source_sha256"] == sha(Path(v194p.__file__))
    reports = {}
    packet_checks = 0
    for condition, expected in (("moving", 12), ("stationary_control", 4)):
        path = OUT / condition / "report.json"
        report = json.loads(path.read_text())
        assert report["static_reference_source_sha256"] == sha(
            Path(v194p.__file__))
        assert len(report["rows"]) == expected
        assert all(p["valid_furniture"] for p in report["preflights"])
        for row in report["rows"]:
            packet_checks += replay(row, report["static_reference"])
        reports[condition] = report
    controls = reports["stationary_control"]["rows"]
    assert all(r["front_alarm"] is None for r in controls)
    moving = reports["moving"]["rows"]
    near = [r for r in moving if r["layout"] == "near_side_table"]
    assert all(r["front_alarm"]["side"] == "range_new_low_surface" and
               r["front_alarm"]["capture_time_s"] == 17.3 for r in near)
    assert all(r["evaluator_minimum_scene_gap_m"] > .09 for r in near
               if r["scripted_prop_stop_delay_s"] == .2)
    prior = json.loads((ROOT / "artifacts/ape-closed-loop-cove50-varied-furniture-owner/report.json").read_text())
    prior_by_key = {(r["layout"], r["seed"],
                     r["scripted_prop_stop_delay_s"]): r
                    for r in prior["rows"]}
    assert all(r["evaluator_minimum_scene_gap_m"] ==
               prior_by_key[(r["layout"], r["seed"],
                             r["scripted_prop_stop_delay_s"])][
                                 "evaluator_minimum_scene_gap_m"]
               for r in moving if r["layout"] != "near_side_table" or
               r["scripted_prop_stop_delay_s"] is None)
    report = {"schema": "robot-learning-lab.ape-static-reference-audit.v1",
              "source_sha256": sha(SOURCE),
              "index_sha256": sha(OUT / "index.json"),
              "moving_report_sha256": sha(OUT / "moving/report.json"),
              "stationary_report_sha256": sha(
                  OUT / "stationary_control/report.json"),
              "episodes": 16, "packet_hashes_and_replays": packet_checks,
              "decoded_video_frames": 32,
              "stationary_false_warnings": sum(
                  r["front_alarm"] is not None for r in controls),
              "near_table_timely_warning_count": len(near),
              "near_table_scripted_stop_clear_count": sum(
                  r["evaluator_minimum_scene_gap_m"] > .05
                  for r in near if r["scripted_prop_stop_delay_s"] == .2)}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
