"""Audit V31 packets, held alerts, V28 controls and captioned comparison."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

import body_aware_held_sensor as study
import dense_low_trial as dense
import flank_sensor_trial as flank
import held_body_intrusion_trial as late
import held_monitor_probe as previous
import kitchen_native as kitchen
import kitchen_trial as trial
import online_cluster_guard as online
import render_body_aware_held as render


def main() -> None:
    report_path = study.OUT / "report.json"
    report = json.loads(report_path.read_text())
    previous_report = json.loads((previous.OUT / "report.json").read_text())
    previous_rows = {(r["mode"], r["world"], r["case"]): r
                     for r in previous_report["results"]}
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(study.__file__)),
        "scene": report["scene_sha256"] == trial.sha(late.previous.OUT / "scene.xml"),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "native_trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "tracker": report["tracker_source_sha256"] == trial.sha(Path(online.__file__)),
        "flank": report["flank_source_sha256"] == trial.sha(Path(flank.__file__)),
        "dense": report["dense_source_sha256"] == trial.sha(Path(dense.__file__)),
        "native_pose_source": report["native_pose_source_sha256"] ==
                              trial.sha(Path(kitchen.__file__)),
        "previous_probe_source": report["previous_probe_source_sha256"] ==
                                 trial.sha(Path(previous.__file__)),
        "previous_report": report["previous_report_sha256"] ==
                           trial.sha(previous.OUT / "report.json"),
        "six_modes": set(report["modes"]) == set(study.MODES),
        "complete_matrix": len(report["results"]) == 6*4*4,
    }
    by = {}
    total_self = 0
    total_blocked_person = 0
    for row in report["results"]:
        key = f"{row['mode']}:{row['world']}:{row['case']}"
        by[row["mode"], row["world"], row["case"]] = row
        path = Path(row["trace"]["path"])
        checks[f"trace_hash:{key}"] = row["trace"]["sha256"] == trial.sha(path)
        packets = json.loads(path.read_text())
        checks[f"count:{key}"] = len(packets) == row["packet_count"]
        checks[f"format:{key}"] = all(
            len(p["actor_packet"]["ranges_m"]) == row["rays"] and
            len(p["actor_packet"]["ray_sources"]) == row["rays"] and
            len(p["actor_packet"]["bearings_deg"]) == row["rays"] and
            p["actor_packet"]["valid_returns"] == sum(
                r is not None for r in p["actor_packet"]["ranges_m"]) and
            p["actor_packet"]["captured_at_s"] == p["time_s"]
            for p in packets)
        first_alert = next((p["time_s"] for p in packets
                            if p["actor_alert_streak"] >= 2), None)
        first_visible = next((p["time_s"] for p in packets
                              if p["time_s"] >= late.ENTRY_START_S and
                              p["evaluator"]["person_ray_hits"]), None)
        checks[f"alert:{key}"] = first_alert == row["first_two_packet_actor_alert_s"]
        checks[f"visible:{key}"] = first_visible == row["first_person_ray_after_motion_s"]
        self_count = sum(p["evaluator"]["self_ray_hits"] for p in packets)
        blocked_count = sum(p["evaluator"]["self_blocked_person_rays"]
                            for p in packets)
        checks[f"self_counts:{key}"] = (self_count == row["self_ray_hits"] and
                                        blocked_count == row["self_blocked_person_rays"])
        total_self += self_count
        total_blocked_person += blocked_count
        preentry = sum(p["evaluator"]["person_ray_hits"] > 0 for p in packets
                       if p["time_s"] < late.ENTRY_START_S)
        checks[f"preentry:{key}"] = preentry == row["preentry_person_ray_packets"]
        if row["mode"] in ("front_ideal_15hz", "omni_ideal_10hz"):
            old_mode = ("front_140deg_15hz" if row["mode"].startswith("front")
                        else "ideal_omni_10hz")
            old = previous_rows[old_mode, row["world"], row["case"]]
            checks[f"v28_match:{key}"] = (
                row["first_two_packet_actor_alert_s"] == old["first_two_packet_actor_alert_s"] and
                row["first_evaluator_below_5cm_s"] == old["first_evaluator_below_5cm_s"] and
                row["packet_count"] == old["packet_count"])
    moving = [r for r in report["results"] if r["case"].startswith("late_")]
    stationary = [r for r in report["results"] if r["case"].startswith("static_")]
    count = lambda mode, rows: sum(r["mode"] == mode and
                                   r["first_two_packet_actor_alert_s"] is not None
                                   for r in rows)
    checks["forward_body_same_as_ideal"] = all(
        by["front_body_15hz", world, case]["first_two_packet_actor_alert_s"] ==
        by["front_ideal_15hz", world, case]["first_two_packet_actor_alert_s"]
        for world in range(4) for case in previous.CASES)
    checks["raw_omni_false_alerts"] = count("omni_body_10hz", stationary) == 8
    checks["raw_omni_motion_alerts"] = count("omni_body_10hz", moving) == 8
    checks["masked_omni_controls"] = (
        count("omni_body_masked_10hz", stationary) == 0 and
        count("omni_body_masked_10hz", moving) == 8)
    checks["split_controls"] = (
        count("split_body_10hz", stationary) == 0 and
        count("split_body_10hz", moving) == 8 and
        all(r["self_ray_hits"] == 0 for r in report["results"]
            if r["mode"] == "split_body_10hz"))
    checks["raw_self_returns_present"] = total_self > 0 and total_blocked_person > 0
    video = json.loads((study.OUT / "video.json").read_text())
    cap = cv2.VideoCapture(video["path"])
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded, _ = cap.read()
    cap.release()
    source_video = late.OUT / "bounded_guard/videos/late_right_pause-mapped_low_guard-world0.mp4"
    source_trace = late.OUT / "bounded_guard/episodes/late_right_pause-mapped_low_guard-world0.npz"
    checks["video"] = (video["sha256"] == trial.sha(Path(video["path"])) and
                       video["frames"] == frames == 33 and
                       width == 960 and height == 710 and decoded and
                       video["source_video_sha256"] == trial.sha(source_video) and
                       video["source_trace_sha256"] == trial.sha(source_trace) and
                       video["report_sha256"] == trial.sha(report_path) and
                       video["source_sha256"] == trial.sha(Path(render.__file__)))
    summary = {
        "front_moving_alerts": count("front_body_15hz", moving),
        "raw_omni_static_false_alerts": count("omni_body_10hz", stationary),
        "masked_omni_moving_alerts": count("omni_body_masked_10hz", moving),
        "masked_omni_static_false_alerts": count("omni_body_masked_10hz", stationary),
        "split_moving_alerts": count("split_body_10hz", moving),
        "split_static_false_alerts": count("split_body_10hz", stationary),
        "total_self_ray_hits_all_modes": total_self,
        "total_self_blocked_person_rays_all_modes": total_blocked_person,
    }
    audit = {"schema": "robot-learning-lab.ape-body-aware-held-sensor-audit.v1",
             "pass": all(checks.values()), "checks": checks, "summary": summary}
    (study.OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pass": audit["pass"], "checks": len(checks),
                      **summary}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V31 audit failed")


if __name__ == "__main__":
    main()
