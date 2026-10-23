"""V197p: audit matched new-scene owner turns and the frozen V194 cue."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio

from audit_static_reference_owner_v195p import replay as replay_static
from run import ROOT, sha
import run_heldout_furniture_v196p as v196p


SOURCE = Path(__file__)
OUT = v196p.OUT
EXPECTED = {"pooled_derivative-moving": 8,
            "pooled_derivative-stationary": 4,
            "static_reference-moving": 8,
            "static_reference-stationary": 4}


def audit_row(row: dict, static_params: dict | None) -> int:
    if static_params is not None:
        return replay_static(row, static_params)
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
    checks = 0
    for frame in monitor["frames"]:
        for path_key, sha_key in (
            ("packet_path", "packet_sha256"),
            ("owner_depth_packet_path", "owner_depth_packet_sha256")):
            path = ROOT / frame[path_key]
            assert sha(path) == frame[sha_key]
            checks += 1
    alarm = row["front_alarm"]
    if alarm:
        frame = monitor["frames"][alarm["capture_index"]]
        assert frame["capture_time_s"] == alarm["capture_time_s"]
        assert (frame["front_left_first_cue"] if
                alarm["side"] == "front_left" else
                frame["owner_depth_first_cue"])
    return checks


def main() -> None:
    index = json.loads((OUT / "index.json").read_text())
    assert index["source_sha256"] == sha(Path(v196p.__file__))
    reports = {}
    packet_checks = 0
    for name, expected in EXPECTED.items():
        path = OUT / name / "report.json"
        report = json.loads(path.read_text())
        assert report["heldout_source_sha256"] == sha(Path(v196p.__file__))
        assert report["layouts"] == index["layouts"] if not name.endswith(
            "stationary") else all(k in index["layouts"] for k in report["layouts"])
        assert len(report["rows"]) == expected
        assert all(p["valid_furniture"] for p in report["preflights"])
        params = (report["rows"] and {
            "pre_show_frames": 4,
            "novel_world_point_distance_m": .04,
            "minimum_novel_returns": 4,
            "minimum_novel_centroid_motion_m": .012,
        } if name.startswith("static_reference") else None)
        for row in report["rows"]:
            packet_checks += audit_row(row, params)
        reports[name] = report
    old_controls = reports["pooled_derivative-stationary"]["rows"]
    new_controls = reports["static_reference-stationary"]["rows"]
    assert sum(r["front_alarm"] is not None for r in old_controls) == 2
    assert all(r["front_alarm"] is None for r in new_controls)
    old_moving = reports["pooled_derivative-moving"]["rows"]
    new_moving = reports["static_reference-moving"]["rows"]
    old_left = [r for r in old_moving if r["layout"] == "left_mid"
                and r["scripted_prop_stop_delay_s"] == .2]
    new_left = [r for r in new_moving if r["layout"] == "left_mid"
                and r["scripted_prop_stop_delay_s"] == .2]
    assert sorted(r["front_alarm"]["capture_time_s"] for r in old_left) == [
        16.4, 17.4]
    assert all(r["front_alarm"]["capture_time_s"] == 17.3 and
               r["front_alarm"]["side"] == "range_new_low_surface" and
               r["evaluator_minimum_scene_gap_m"] == .05219
               for r in new_left)
    for condition, group in (("old", old_moving), ("new", new_moving)):
        assert all(r["evaluator_minimum_scene_gap_m"] == -.008
                   for r in group if r["scripted_prop_stop_delay_s"] is None)
    report = {"schema": "robot-learning-lab.ape-heldout-furniture-audit.v1",
              "source_sha256": sha(SOURCE),
              "index_sha256": sha(OUT / "index.json"),
              "report_sha256": {name: sha(OUT / name / "report.json")
                                for name in EXPECTED},
              "episodes": 24,
              "packet_hashes_and_static_rule_replays": packet_checks,
              "decoded_video_frames": 48,
              "old_stationary_false_warnings": 2,
              "new_stationary_false_warnings": 0,
              "new_left_timely_warnings": len(new_left),
              "new_left_stop_gap_m": [r["evaluator_minimum_scene_gap_m"]
                                      for r in new_left],
              "scope": "New scenes were authored and collision-preflighted "
                       "before running the frozen V194 warning. Same held "
                       "historical body/owner and two lighting seeds remain; "
                       "not an independent randomized distribution or "
                       "physical timing qualification."}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
