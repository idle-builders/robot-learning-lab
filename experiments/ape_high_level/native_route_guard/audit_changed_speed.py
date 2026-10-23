"""Audit the frozen changed-speed kitchen holdout, including its failures."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import changed_speed_trial as current
import command_route_trial as command
import initial_view_trial as initial
import kitchen_native as kitchen
import kitchen_trial as runner
import run as guard


def audit() -> dict:
    out = current.OUT
    report = json.loads((out / "report.json").read_text())
    checks = {}
    source_paths = {"source_sha256": Path(current.__file__),
                    "episode_runner_sha256": Path(runner.__file__),
                    "command_guard_sha256": Path(command.__file__),
                    "original_guard_sha256": Path(guard.__file__),
                    "native_trace_sha256": kitchen.TRACE,
                    "body_sha256": kitchen.BODY,
                    "v10_report_sha256": command.OUT / "report.json"}
    for name, path in source_paths.items():
        checks[f"hash:{name}"] = report[name] == runner.sha(path)
    checks["scene"] = report["room"]["scene_sha256"] == runner.sha(kitchen.OUT / "scene.xml")
    checks["initial_map"] = (report["sensor"]["prior_map_sha256"] ==
        runner.sha(out / "background_xy.npy") ==
        runner.sha(initial.OUT / "background_xy.npy") and
        report["sensor"]["static_map_points"] == 4)
    rows = report["results"]
    checks["paired_matrix"] = (len(rows) == 24 and
        {(r["world"], r["case"], r["mode"]) for r in rows} ==
        {(w, c, m) for w in range(4) for c in current.CASES for m in runner.MODES})
    matched_static = 0
    decoded_videos = 0
    for row in rows:
        stem = f"{row['case']}-{row['mode']}-world{row['world']}"
        path = out / "episodes" / f"{stem}.npz"
        checks[f"trace_hash:{stem}"] = row["trace"]["sha256"] == runner.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            gap = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{stem}"] = abs(float(gap.min()) - row["min_modeled_body_person_gap_m"]) < 2e-5
            checks[f"overlap:{stem}"] = bool(np.any(gap <= 0)) == row["modeled_person_overlap"]
            packets = [json.loads(str(p)) for p in saved["packet_json"]]
            checks[f"packets:{stem}"] = len(packets) == row["packet_count"]
            matched_static += sum(p["static_matched_returns"] for p in packets)
            trigger = next((b["captured_at_s"] for a, b in zip(packets, packets[1:])
                            if a["route_guard"]["risk"] and b["route_guard"]["risk"]),
                           3.0)
            checks[f"trigger:{stem}"] = abs(trigger-row["stop_command_s"]) < 1e-5
        if row["video"]:
            movie = Path(row["video"]["path"])
            checks[f"video_hash:{stem}"] = row["video"]["sha256"] == runner.sha(movie)
            cap = cv2.VideoCapture(str(movie))
            ok, frame = cap.read()
            checks[f"video_decode:{stem}"] = (ok and frame.shape == (562, 960, 3)
                and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == row["video"]["frames"] == 100)
            cap.release()
            decoded_videos += 1
    by = {(r["world"], r["case"], r["mode"]): r for r in rows}
    controls_overlap = sum(by[w, c, "none"]["modeled_person_overlap"]
                           for w in range(4)
                           for c in ("slow_cross_held", "pause_path_held"))
    guarded_clear = sum(not by[w, c, "mapped_low_guard"]["below_5cm_person_margin"]
                        for w in range(4)
                        for c in ("slow_cross_held", "pause_path_held"))
    false_stops = sum(by[w, "fast_clear_held", "mapped_low_guard"]["stop_command_s"] < 3.0
                      for w in range(4))
    checks["observed_holdout_outcome"] = (controls_overlap == 8 and
                                           guarded_clear == 8 and
                                           false_stops == 2)
    checks["static_returns_exercised"] = matched_static > 0
    comparison = json.loads((out / "comparison-video.json").read_text())
    checks["comparison_pair"] = (comparison["case"] == "fast_clear_held" and
                                  comparison["world"] == 1 and
                                  comparison["source_report_sha256"] ==
                                  runner.sha(out / "report.json"))
    checks["comparison_hash"] = comparison["sha256"] == runner.sha(Path(comparison["video"]))
    for mode, stated in comparison["source_videos_sha256"].items():
        checks[f"comparison_source:{mode}"] = (
            stated == runner.sha(out / "videos" /
                                 f"fast_clear_held-{mode}-world1.mp4"))
    cap = cv2.VideoCapture(comparison["video"])
    ok, frame = cap.read()
    checks["comparison_decode"] = (ok and frame.shape == (544, 1280, 3) and
                                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) ==
                                    comparison["frames"] == 100)
    cap.release()
    result = {"schema": "robot-learning-lab.ape-robocasa-command-route-holdout-audit.v1",
              "report_sha256": runner.sha(out / "report.json"),
              "matched_static_ray_returns": matched_static,
              "unsensed_threat_overlaps": controls_overlap,
              "guarded_threats_above_5cm_margin": guarded_clear,
              "safe_fast_unnecessary_stops": false_stops,
              "videos_checked": decoded_videos,
              "checks": checks, "pass": all(checks.values())}
    (out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("pass", "unsensed_threat_overlaps",
                       "guarded_threats_above_5cm_margin",
                       "safe_fast_unnecessary_stops", "videos_checked")}, indent=2))
    if not result["pass"]:
        print({key: value for key, value in result["checks"].items() if not value})
        raise SystemExit(1)
