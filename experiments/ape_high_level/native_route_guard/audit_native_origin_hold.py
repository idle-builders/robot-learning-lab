"""Audit V30 origin-wait geometry, timed range packets and retained media."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import held_body_intrusion_trial as late
import held_monitor_probe as monitor
import kitchen_native as kitchen
import kitchen_trial as trial
import native_early_hold_screen as early
import native_origin_hold_screen as screen


def main() -> None:
    report = json.loads((screen.OUT / "report.json").read_text())
    source = json.loads(screen.REPORT_SOURCE.read_text())
    checks = {
        "screen_source": report["source_sha256"] == trial.sha(Path(screen.__file__)),
        "replay_source": report["replay_source_sha256"] == trial.sha(Path(early.__file__)),
        "range_source": report["range_source_sha256"] == trial.sha(Path(monitor.__file__)),
        "person_script": report["person_script_sha256"] == trial.sha(Path(late.__file__)),
        "scene": report["scene_sha256"] == trial.sha(late.previous.OUT / "scene.xml"),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "zero_source": (report["zero_action_source"]["trace_sha256"] ==
                        trial.sha(screen.SOURCE) and
                        report["zero_action_source"]["report_sha256"] ==
                        trial.sha(screen.REPORT_SOURCE) and
                        source["zero_action"] and
                        all(v == 0 for v in source["first_action_world0"])),
        "different_initial_state_explicit": (
            not report["zero_action_source"]["initial_state_matched_to_v29"] and
            len(report["zero_action_source"]["first_saved_trace_difference"]) == 4 and
            any(row["first_saved_root_pose_max_abs_difference"] > .005
                for row in report["zero_action_source"]["first_saved_trace_difference"])),
        "v29_report": report["v29_report_sha256"] == trial.sha(early.OUT / "report.json"),
        "result_matrix": len(report["results"]) == 8,
        "sensor_matrix": len(report["sensor_results"]) == 32,
    }
    person_gaps = []
    fixture_gaps = []
    videos = 0
    for row in report["results"]:
        key = f"body:{row['world']}:{row['case']}"
        path = Path(row["trace"]["path"])
        checks[key+":source"] = row["trace"]["sha256"] == trial.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_m"], saved["midstep_gap_m"]]
            times = np.r_[saved["time_s"], saved["midstep_time_s"]]
            checks[key+":samples"] = (len(saved["time_s"]) == 400 and
                                       len(saved["midstep_time_s"]) == 399 and
                                       abs(float(times.min())-.02) < 1e-5)
            checks[key+":gap"] = (
                abs(float(gaps.min())-row["min_body_person_gap_m"]) <= 1e-5 and
                bool((gaps <= 0).any()) == row["overlap"] and
                bool((gaps < .05).any()) == row["below_5cm"])
            checks[key+":root"] = abs(float(np.linalg.norm(
                saved["root_xy_m"][-1]-saved["root_xy_m"][0])) -
                row["root_final_progress_m"]) <= 1e-5
        person_gaps.append(row["min_body_person_gap_m"])
        fixture_gaps.append(row["min_sampled_body_fixture_gap_m"])
        if row["world"] == 0:
            video = row["video"]
            movie = Path(video["path"])
            cap = cv2.VideoCapture(str(movie))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            opened, _ = cap.read()
            cap.release()
            checks[key+":video"] = (video["sha256"] == trial.sha(movie) and
                                     video["frames"] == count == 100 and opened)
            videos += 1
        else:
            checks[key+":no_video"] = row["video"] is None
    checks["body_clearance"] = min(person_gaps) > .05
    checks["fixture_clearance"] = min(fixture_gaps) > .2
    checks["two_videos"] = videos == 2

    sensor_rows = {}
    for row in report["sensor_results"]:
        key = f"sensor:{row['mode']}:{row['case']}:{row['world']}"
        path = Path(row["trace"]["path"])
        checks[key+":source"] = row["trace"]["sha256"] == trial.sha(path)
        packets = json.loads(path.read_text())
        sensor_rows[row["mode"], row["case"], row["world"]] = packets
        pre = [p for p in packets if p["time_s"] < late.ENTRY_START_S]
        post = [p for p in packets if p["time_s"] >= late.ENTRY_START_S]
        first_hit = next((p["time_s"] for p in packets
                          if p["evaluator_person_ray_hits"]), None)
        checks[key+":count"] = len(packets) == row["packet_count"]
        checks[key+":packet_format"] = all(
            len(p["actor_packet"]["ranges_m"]) == row["ray_count"] and
            p["actor_packet"]["valid_returns"] == sum(
                r is not None for r in p["actor_packet"]["ranges_m"]) and
            p["actor_packet"]["captured_at_s"] == p["time_s"]
            for p in packets)
        checks[key+":hits"] = (
            sum(p["evaluator_person_ray_hits"] > 0 for p in pre) ==
            row["preentry_person_hit_packets"] and
            sum(p["evaluator_person_ray_hits"] > 0 for p in post) ==
            row["postentry_person_hit_packets"] and
            first_hit == row["first_person_hit_s"])
    indistinguishable = 0
    for mode in monitor.SENSOR_MODES:
        for side in ("left", "right"):
            for world in range(4):
                moving = sensor_rows[mode, f"late_{side}_pause", world]
                stationary = sensor_rows[mode, f"static_{side}", world]
                prefix = [(a, b) for a, b in zip(moving, stationary)
                          if a["time_s"] < late.ENTRY_START_S]
                same = bool(prefix) and all(
                    a["actor_packet"] == b["actor_packet"] for a, b in prefix)
                checks[f"preentry_pair:{mode}:{side}:{world}"] = same
                indistinguishable += int(same)
    audit = {
        "schema": "robot-learning-lab.ape-native-origin-hold-audit.v1",
        "pass": all(checks.values()), "checks": checks,
        "summary": {"minimum_body_person_gap_m": round(min(person_gaps), 5),
                    "minimum_sampled_body_fixture_gap_m": round(min(fixture_gaps), 5),
                    "indistinguishable_preentry_packet_pairs": indistinguishable,
                    "origin_initial_state_matched_to_forward": False,
                    "video_count": videos}}
    (screen.OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pass": audit["pass"], **audit["summary"]}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V30 audit failed")


if __name__ == "__main__":
    main()
