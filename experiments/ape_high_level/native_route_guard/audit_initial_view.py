"""Audit the initial-view-map kitchen challenge and its matched V8 control."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import initial_view_trial as current
import kitchen_native as kitchen
import kitchen_trial as runner
import run as guard


def audit() -> dict:
    out = current.OUT
    report = json.loads((out / "report.json").read_text())
    baseline = json.loads((kitchen.OUT / "joined-trial/report.json").read_text())
    checks = {}
    paths = {"source_sha256": Path(current.__file__),
             "episode_runner_sha256": Path(runner.__file__),
             "scene_builder_sha256": Path(kitchen.__file__),
             "guard_source_sha256": Path(guard.__file__),
             "native_trace_sha256": kitchen.TRACE,
             "stop_report_sha256": guard.replay.STOP_REPORT,
             "body_sha256": kitchen.BODY}
    for name, path in paths.items():
        checks[f"hash:{name}"] = report[name] == runner.sha(path)
    checks["same_scene"] = (report["room"]["scene_sha256"] ==
                            runner.sha(kitchen.OUT / "scene.xml") ==
                            baseline["room"]["scene_sha256"])
    points = np.load(out / "background_xy.npy")
    checks["initial_map"] = (points.shape == (report["sensor"]["static_map_points"], 2)
                             and len(points) > 0
                             and report["sensor"]["prior_map_sha256"] ==
                             runner.sha(out / "background_xy.npy")
                             and len(points) < baseline["sensor"]["static_map_points"])
    rows = report["results"]
    checks["matrix"] = (len(rows) == 24 and
        {(r["world"], r["case"], r["mode"]) for r in rows} ==
        {(w, c, m) for w in range(4) for c in runner.CASES for m in runner.MODES})
    old = {(r["world"], r["case"], r["mode"]): r for r in baseline["results"]}
    matched = 0
    videos = 0
    for row in rows:
        key = (row["world"], row["case"], row["mode"])
        stem = f"{row['case']}-{row['mode']}-world{row['world']}"
        path = out / "episodes" / f"{stem}.npz"
        checks[f"trace_hash:{stem}"] = row["trace"]["sha256"] == runner.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            gap = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{stem}"] = abs(float(gap.min()) - row["min_modeled_body_person_gap_m"]) < 2e-5
            checks[f"overlap:{stem}"] = bool(np.any(gap <= 0)) == row["modeled_person_overlap"]
            packets = [json.loads(str(item)) for item in saved["packet_json"]]
            checks[f"packets:{stem}"] = len(packets) == row["packet_count"]
            matched += sum(p.get("static_matched_returns", 0) for p in packets)
            trigger = next((b["captured_at_s"] for a, b in zip(packets, packets[1:])
                            if a["route_guard"]["risk"] and b["route_guard"]["risk"]),
                           3.0)
            checks[f"trigger:{stem}"] = abs(trigger-row["stop_command_s"]) < 1e-5
        if row["mode"] == "none":
            checks[f"same_unsensed_control:{stem}"] = (
                row["min_modeled_body_person_gap_m"] ==
                old[key]["min_modeled_body_person_gap_m"])
        if row["video"]:
            movie = Path(row["video"]["path"])
            checks[f"video_hash:{stem}"] = row["video"]["sha256"] == runner.sha(movie)
            cap = cv2.VideoCapture(str(movie))
            ok, frame = cap.read()
            checks[f"video_decode:{stem}"] = (ok and frame.shape == (562, 960, 3)
                and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == row["video"]["frames"] == 100)
            cap.release()
            videos += 1
    by = {(r["world"], r["case"], r["mode"]): r for r in rows}
    fast_false = sum(by[w, "fast_clear", "mapped_low_guard"]["stop_command_s"] < 3.0
                     for w in range(4))
    threat_clear = sum(not by[w, c, "mapped_low_guard"]["below_5cm_person_margin"]
                       for w in range(4) for c in ("slow_cross", "pause_path"))
    checks["fixed_stool_exercised"] = matched > 0
    checks["observed_failure_and_clearance"] = fast_false == 1 and threat_clear == 8
    result = {"schema": "robot-learning-lab.ape-robocasa-initial-view-audit.v1",
              "report_sha256": runner.sha(out / "report.json"),
              "initial_map_points": len(points),
              "matched_static_ray_returns": matched,
              "safe_fast_unnecessary_stops": fast_false,
              "threats_above_5cm_margin": threat_clear,
              "videos_checked": videos,
              "checks": checks,
              "pass": all(checks.values())}
    (out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("pass", "initial_map_points", "matched_static_ray_returns",
                       "safe_fast_unnecessary_stops", "threats_above_5cm_margin",
                       "videos_checked")}, indent=2))
    if not result["pass"]:
        print({key: value for key, value in result["checks"].items() if not value})
        raise SystemExit(1)
