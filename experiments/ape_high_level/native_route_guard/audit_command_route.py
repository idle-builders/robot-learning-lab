"""Audit the command-route kitchen guard against the retained V9 controls."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import command_route_trial as current
import initial_view_trial as prior
import kitchen_native as kitchen
import kitchen_trial as runner
import run as guard


def audit() -> dict:
    out = current.OUT
    report = json.loads((out / "report.json").read_text())
    control = json.loads((prior.OUT / "report.json").read_text())
    checks = {}
    source_paths = {"source_sha256": Path(current.__file__),
                    "episode_runner_sha256": Path(runner.__file__),
                    "original_guard_sha256": Path(guard.__file__),
                    "native_trace_sha256": kitchen.TRACE,
                    "stop_report_sha256": guard.replay.STOP_REPORT,
                    "body_sha256": kitchen.BODY,
                    "v9_control_report_sha256": prior.OUT / "report.json"}
    for name, path in source_paths.items():
        checks[f"hash:{name}"] = report[name] == runner.sha(path)
    checks["same_scene"] = (report["room"]["scene_sha256"] ==
                            control["room"]["scene_sha256"] ==
                            runner.sha(kitchen.OUT / "scene.xml"))
    points = np.load(out / "background_xy.npy")
    checks["same_initial_map"] = (report["sensor"]["prior_map_sha256"] ==
        runner.sha(out / "background_xy.npy") ==
        control["sensor"]["prior_map_sha256"] and
        len(points) == report["sensor"]["static_map_points"] == 4)
    rows = report["results"]
    checks["matrix"] = (len(rows) == 12 and
        {(r["world"], r["case"]) for r in rows} ==
        {(w, c) for w in range(4) for c in runner.CASES})
    static_returns = 0
    decoded_videos = 0
    for row in rows:
        stem = f"{row['case']}-mapped_low_guard-world{row['world']}"
        path = out / "episodes" / (stem + ".npz")
        checks[f"trace_hash:{stem}"] = row["trace"]["sha256"] == runner.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{stem}"] = abs(float(gaps.min())-row["min_modeled_body_person_gap_m"]) < 2e-5
            checks[f"overlap:{stem}"] = bool(np.any(gaps <= 0)) == row["modeled_person_overlap"]
            packets = [json.loads(str(item)) for item in saved["packet_json"]]
            checks[f"packets:{stem}"] = len(packets) == row["packet_count"]
            static_returns += sum(p["static_matched_returns"] for p in packets)
            checks[f"planned_velocity:{stem}"] = all(
                p["route_guard"]["planned_velocity_xy_mps"] == [.35, 0.]
                for p in packets)
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
    by = {(r["world"], r["case"]): r for r in rows}
    fast_false = sum(by[w, "fast_clear"]["stop_command_s"] < 3.0 for w in range(4))
    threat_clear = sum(not by[w, c]["below_5cm_person_margin"]
                       for w in range(4) for c in ("slow_cross", "pause_path"))
    checks["fixed_clutter_exercised"] = static_returns > 0
    checks["clean_development_matrix"] = fast_false == 0 and threat_clear == 8
    result = {"schema": "robot-learning-lab.ape-robocasa-command-route-audit.v1",
              "report_sha256": runner.sha(out / "report.json"),
              "matched_static_ray_returns": static_returns,
              "safe_fast_unnecessary_stops": fast_false,
              "threats_above_5cm_margin": threat_clear,
              "videos_checked": decoded_videos,
              "checks": checks, "pass": all(checks.values())}
    (out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("pass", "matched_static_ray_returns",
                       "safe_fast_unnecessary_stops", "threats_above_5cm_margin",
                       "videos_checked")}, indent=2))
    if not result["pass"]:
        print({key: value for key, value in result["checks"].items() if not value})
        raise SystemExit(1)
