"""V199p: audit V198 range occlusion and full-turn consequences."""

from __future__ import annotations

import json
from pathlib import Path

from audit_static_reference_owner_v195p import replay
from run import ROOT, sha
import run_heldout_furniture_occlusion_v198p as v198p


SOURCE = Path(__file__)
OUT = v198p.OUT
PARAMS = {"pre_show_frames": 4,
          "novel_world_point_distance_m": .04,
          "minimum_novel_returns": 4,
          "minimum_novel_centroid_motion_m": .012}


def main() -> None:
    index = json.loads((OUT / "index.json").read_text())
    assert index["source_sha256"] == sha(Path(v198p.__file__))
    reports = {}
    packet_checks = removed = 0
    for condition, expected in (("moving", 8), ("stationary", 4)):
        report = json.loads((OUT / condition / "report.json").read_text())
        assert report["source_run_sha256"] == sha(Path(v198p.__file__))
        assert report["injected_forward_pair_front_left_roi_occlusion"]
        assert len(report["rows"]) == expected
        assert all(p["valid_furniture"] for p in report["preflights"])
        for row in report["rows"]:
            packet_checks += replay(row, PARAMS)
            monitor = json.loads((ROOT / row["monitor"]["path"]).read_text())
            removed += sum(frame["front_counts"].get(
                "injected_forward_pair_roi_removed", 0)
                for frame in monitor["frames"])
        reports[condition] = report
    assert removed > 0
    assert all(r["front_alarm"] is None for r in reports["stationary"]["rows"])
    left = [r for r in reports["moving"]["rows"]
            if r["layout"] == "left_mid" and
            r["scripted_prop_stop_delay_s"] == .2]
    assert len(left) == 2
    assert all(r["front_alarm"]["side"] == "range_new_low_surface" and
               r["front_alarm"]["capture_time_s"] == 17.4 and
               r["evaluator_minimum_scene_gap_m"] == .0035
               for r in left)
    right = [r for r in reports["moving"]["rows"]
             if r["layout"] == "right_backward" and
             r["scripted_prop_stop_delay_s"] == .2]
    assert len(right) == 2
    assert all(r["front_alarm"]["side"] == "owner_rgbd_low_region" and
               r["front_alarm"]["capture_time_s"] == 17.2 and
               r["evaluator_minimum_scene_gap_m"] == .10404
               for r in right)
    report = {"schema": "robot-learning-lab.ape-heldout-occlusion-audit.v1",
              "source_sha256": sha(SOURCE),
              "index_sha256": sha(OUT / "index.json"),
              "moving_report_sha256": sha(OUT / "moving/report.json"),
              "stationary_report_sha256": sha(OUT / "stationary/report.json"),
              "episodes": 12,
              "packet_hashes_and_static_rule_replays": packet_checks,
              "decoded_video_frames": 24,
              "forward_roi_returns_removed": removed,
              "left_stop_gap_m": [r["evaluator_minimum_scene_gap_m"]
                                  for r in left],
              "right_stop_gap_m": [r["evaluator_minimum_scene_gap_m"]
                                   for r in right],
              "stationary_false_warnings": 0}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
