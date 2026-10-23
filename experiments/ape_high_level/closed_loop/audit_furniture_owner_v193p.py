"""V193p: verify saved V191/V192 geometry, packets, videos and pairings."""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import run_furniture_sensor_ablation_v192p as v192p
import run_varied_furniture_owner_v191p as v191p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-furniture-ablation"
REPORTS = {
    "visible_moving": v191p.OUT / "report.json",
    "stationary_control": OUT / "stationary_control/report.json",
    "moving_forward_occluded": OUT / "moving_forward_occluded/report.json",
}
EXPECTED = {"visible_moving": 12, "stationary_control": 4,
            "moving_forward_occluded": 8}


def check(condition: str, path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    assert data["source_sha256"] == sha(Path(v191p.__file__))
    assert len(data["rows"]) == EXPECTED[condition]
    assert len(data["preflights"]) == (6 if condition == "visible_moving" else 4)
    assert all(p["valid_furniture"] and
               p["table_other_minimum"][0] >= .02 and
               p["table_robot_minimum"][0] >= .05
               for p in data["preflights"])
    if condition != "visible_moving":
        assert data["ablation_source_sha256"] == sha(Path(v192p.__file__))
        assert data["stationary_prop_control"] == (
            condition == "stationary_control")
        assert data["injected_forward_pair_roi_occlusion"] == (
            condition == "moving_forward_occluded")
    rows = {}
    total_packets = total_removed = 0
    for row in data["rows"]:
        key = (row["layout"], row["seed"],
               row["scripted_prop_stop_delay_s"])
        assert key not in rows
        rows[key] = row
        video = ROOT / row["video_path"]
        assert video.exists() and sha(video) == row["video_sha256"]
        monitor_path = ROOT / row["monitor"]["path"]
        assert sha(monitor_path) == row["monitor"]["sha256"]
        monitor = json.loads(monitor_path.read_text())
        assert len(monitor["frames"]) == 55
        assert monitor["alarm"] == row["front_alarm"]
        for frame in monitor["frames"]:
            for path_key, sha_key in (
                ("packet_path", "packet_sha256"),
                ("owner_depth_packet_path", "owner_depth_packet_sha256")):
                packet = ROOT / frame[path_key]
                assert packet.exists() and sha(packet) == frame[sha_key]
                total_packets += 1
            total_removed += frame["front_counts"].get(
                "injected_forward_pair_roi_removed", 0)
        alarm = row["front_alarm"]
        if alarm is not None:
            frame = monitor["frames"][alarm["capture_index"]]
            assert frame["capture_time_s"] == alarm["capture_time_s"]
            assert (frame["owner_depth_first_cue"] if
                    alarm["side"] == "owner_rgbd_low_region" else
                    frame["front_left_first_cue"])
    if condition == "moving_forward_occluded":
        assert total_removed > 0
    else:
        assert total_removed == 0
    return data, {"episodes": len(rows), "packet_hashes_checked": total_packets,
                  "forward_roi_returns_removed": total_removed,
                  "warnings": sum(r["front_alarm"] is not None
                                  for r in rows.values()),
                  "task_views": sum(r["task_result"] ==
                                    "fresh_unattributed_item_view_after_yield"
                                    for r in rows.values()),
                  "below_5cm": sum(r["evaluator_first_below_5cm_s"]
                                   is not None for r in rows.values()),
                  "max_wall_tick_lateness_s": max(
                      r["max_tick_wall_lateness_s"] for r in rows.values())}


def main() -> None:
    data, summary = {}, {}
    for condition, path in REPORTS.items():
        data[condition], summary[condition] = check(condition, path)
    stationary = data["stationary_control"]["rows"]
    assert all(r["front_alarm"] is not None for r in stationary
               if r["layout"] == "near_side_table")
    assert all(r["front_alarm"] is None for r in stationary
               if r["layout"] == "offset_side_table")
    occluded = data["moving_forward_occluded"]["rows"]
    near_stop = [r for r in occluded if r["layout"] == "near_side_table"
                 and r["scripted_prop_stop_delay_s"] == .2]
    offset_stop = [r for r in occluded if r["layout"] == "offset_side_table"
                   and r["scripted_prop_stop_delay_s"] == .2]
    assert all(r["front_alarm"]["side"] == "owner_rgbd_low_region" and
               r["front_alarm"]["capture_time_s"] == 17.6 and
               r["evaluator_minimum_scene_gap_m"] < 0
               for r in near_stop)
    assert all(r["front_alarm"]["side"] == "owner_rgbd_low_region" and
               r["front_alarm"]["capture_time_s"] == 17.2 and
               r["evaluator_minimum_scene_gap_m"] > .18 and
               r["task_result"] == "owner_turn_unresolved"
               for r in offset_stop)
    report = {"schema": "robot-learning-lab.ape-furniture-owner-audit.v1",
              "source_sha256": sha(SOURCE),
              "reports": {name: {"path": str(path.relative_to(ROOT)),
                                  "sha256": sha(path)}
                          for name, path in REPORTS.items()},
              "summary": summary,
              "checks": "24 episodes, 20ms furniture/prop geometry preflights, "
                        "saved video and 55 synchronized 10Hz range/RGB-D "
                        "monitor frames per turn; hash all 2,640 packets, "
                        "verify source and warning-frame provenance, stationary "
                        "controls and matched occluded-range outcomes."}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
