"""Saved packet, geometry, and media audit for V22's no-prior guard."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import kitchen_native as kitchen
import kitchen_trial as trial
import online_cluster_guard as online


def main() -> None:
    report = json.loads((online.OUT / "report.json").read_text())
    control = json.loads((online.clutter.OUT / "report.json").read_text())
    controls = {(r["world"], r["case"]): r for r in control["results"]
                if r["mode"] == "mapped_low_guard"}
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(online.__file__)),
        "dense": report["dense_source_sha256"] ==
                 trial.sha(Path(online.dense.__file__)),
        "runner": report["episode_runner_sha256"] == trial.sha(Path(trial.__file__)),
        "command": report["command_guard_sha256"] ==
                   trial.sha(Path(online.command.__file__)),
        "scene": report["scene_sha256"] ==
                 trial.sha(online.clutter.OUT / "scene.xml"),
        "native_trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "v20": report["v20_report_sha256"] ==
               trial.sha(online.clutter.OUT / "report.json"),
        "no_map_sentinel": report["algorithm"]["no_map_sentinel_xy_m"] ==
                           online.NO_MAP_SENTINEL_XY_M.tolist(),
        "matrix": len(report["results"]) == 24 and
                  len({(r["world"], r["case"]) for r in report["results"]}) == 24,
    }
    summary = {"fast_false_stops": 0, "threat_overlaps": 0,
               "threats_below_5cm": 0, "min_threat_gap_m": float("inf"),
               "changed_stop_decisions_vs_mapped_v20": 0,
               "packets": 0, "online_tracks_reported": 0}
    videos = 0
    for row in report["results"]:
        world, case = row["world"], row["case"]
        key = f"{world}:{case}"
        fast = case in ("fast_clear_held", "fast_right_held")
        summary["fast_false_stops"] += int(fast and row["stop_command_s"] < 3.)
        summary["threat_overlaps"] += int(not fast and row["modeled_person_overlap"])
        summary["threats_below_5cm"] += int(not fast and
                                            row["below_5cm_person_margin"])
        if not fast:
            summary["min_threat_gap_m"] = min(
                summary["min_threat_gap_m"], row["min_modeled_body_person_gap_m"])
        summary["changed_stop_decisions_vs_mapped_v20"] += int(
            row["stop_command_s"] != controls[world, case]["stop_command_s"])
        path = Path(row["trace"]["path"])
        checks[f"trace_hash:{key}"] = trial.sha(path) == row["trace"]["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{key}"] = (
                round(float(gaps.min()), 5) == row["min_modeled_body_person_gap_m"] and
                bool((gaps <= 0).any()) == row["modeled_person_overlap"] and
                bool((gaps < trial.MIN_MARGIN_M).any()) ==
                row["below_5cm_person_margin"])
            packets = [json.loads(raw) for raw in saved["packet_json"]]
            checks[f"packet:{key}"] = bool(packets) and all(
                len(p["ranges_m"]) == 73 and
                p["valid_returns"] == sum(value is not None for value in p["ranges_m"]) and
                p["static_matched_returns"] == 0 and
                p["route_guard"]["map_source"] == "none_online_clusters" and
                p["route_guard"]["online_track_count"] ==
                len(p["route_guard"]["tracks"])
                for p in packets)
            summary["packets"] += len(packets)
            summary["online_tracks_reported"] += sum(
                p["route_guard"]["online_track_count"] for p in packets)
        video = row["video"]
        if video is not None:
            videos += 1
            movie = Path(video["path"])
            capture = cv2.VideoCapture(str(movie))
            frames = 0
            while True:
                okay, frame = capture.read()
                if not okay:
                    break
                frames += int(frame.size > 0)
            capture.release()
            checks[f"video:{key}"] = (trial.sha(movie) == video["sha256"] and
                                       frames == video["frames"])
    checks["five_videos"] = videos == 5
    audit = {"schema": "robot-learning-lab.ape-robocasa-online-cluster-guard-audit.v1",
             "report_sha256": trial.sha(online.OUT / "report.json"),
             "summary": summary, "videos_checked": videos,
             "checks": checks, "pass": all(checks.values())}
    (online.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "summary": summary,
                      "videos_checked": videos}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V22 audit failed")


if __name__ == "__main__":
    main()
