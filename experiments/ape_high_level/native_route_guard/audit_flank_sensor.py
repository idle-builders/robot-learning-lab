"""Audit V15 front-plus-flank returns, decisions and media."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import flank_sensor_trial as current
import blocked_sector_trial as blocked
import dense_low_trial as healthy
import changed_speed_trial as held
import kitchen_native as kitchen
import kitchen_trial as runner
import command_route_trial as command
import run as guard


def audit() -> dict:
    out = current.OUT
    report = json.loads((out / "report.json").read_text())
    blocked_report = json.loads((blocked.OUT / "report.json").read_text())
    healthy_report = json.loads((healthy.OUT / "report.json").read_text())
    checks = {}
    paths = {"source_sha256": Path(current.__file__),
             "episode_runner_sha256": Path(runner.__file__),
             "dense_sensor_sha256": Path(healthy.__file__),
             "command_guard_sha256": Path(command.__file__),
             "original_guard_sha256": Path(guard.__file__),
             "native_trace_sha256": kitchen.TRACE,
             "body_sha256": kitchen.BODY,
             "v14_report_sha256": blocked.OUT / "report.json",
             "v13_report_sha256": healthy.OUT / "report.json"}
    for key, path in paths.items():
        checks[f"hash:{key}"] = report[key] == runner.sha(path)
    checks["same_scene"] = (report["scene_sha256"] ==
                            blocked_report["scene_sha256"] ==
                            healthy_report["scene_sha256"])
    points = np.load(out / "background_xy.npy")
    checks["initial_map"] = (points.shape == (report["sensor"]["static_map_points"], 2)
                             and report["sensor"]["prior_map_sha256"] ==
                             runner.sha(out / "background_xy.npy") and len(points) > 0)
    rows = report["results"]
    checks["matrix"] = (len(rows) == 12 and
        {(r["world"], r["case"]) for r in rows} ==
        {(w, c) for w in range(4) for c in held.CASES})
    flank_hits = 0
    static_hits = 0
    videos = 0
    progress = {}
    for row in rows:
        world, case = row["world"], row["case"]
        stem = f"{case}-mapped_low_guard-world{world}"
        path = out / "episodes" / f"{stem}.npz"
        checks[f"trace_hash:{stem}"] = row["trace"]["sha256"] == runner.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            gap = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{stem}"] = (abs(float(gap.min())-
                                        row["min_modeled_body_person_gap_m"]) < 2e-5 and
                                        bool(np.any(gap <= 0)) == row["modeled_person_overlap"])
            packets = [json.loads(str(p)) for p in saved["packet_json"]]
            checks[f"packet_sources:{stem}"] = (
                len(packets) == row["packet_count"] and
                all(len(p["ranges_m"]) == len(p["ray_sources"]) == 73 and
                    p["ray_sources"].count("front_low") == 47 and
                    p["ray_sources"].count("left_flank_low") == 26 and
                    sum(not x for x in p["primary_ray_health"]) == 26 and
                    sum(p["flank_ray_health"]) == 26 and
                    all(p["ray_health"]) for p in packets))
            flank_hits += sum(sum(value is not None and source == "left_flank_low"
                                  for value, source in zip(p["ranges_m"], p["ray_sources"]))
                              for p in packets)
            static_hits += sum(p["static_matched_returns"] for p in packets)
            trigger = next((b["captured_at_s"] for a, b in zip(packets, packets[1:])
                            if a["route_guard"]["risk"] and b["route_guard"]["risk"]),
                           3.0)
            checks[f"trigger:{stem}"] = abs(trigger-row["stop_command_s"]) < 1e-5
            root = saved["root_xy_evaluator_m"]
            progress[world, case] = float(root[-1, 0]-root[0, 0])
        if row["video"]:
            movie = Path(row["video"]["path"])
            checks[f"video_hash:{stem}"] = row["video"]["sha256"] == runner.sha(movie)
            cap = cv2.VideoCapture(str(movie))
            ok, frame = cap.read()
            checks[f"video_decode:{stem}"] = (
                ok and frame.shape == (562, 960, 3) and
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == row["video"]["frames"] == 100)
            cap.release()
            videos += 1
    by = {(r["world"], r["case"]): r for r in rows}
    fast_false = sum(by[w, "fast_clear_held"]["stop_command_s"] < 3 for w in range(4))
    threat_clear = sum(not by[w, c]["below_5cm_person_margin"]
                       for w in range(4) for c in ("slow_cross_held", "pause_path_held"))
    checks["flank_observed"] = flank_hits > 0 and static_hits > 0
    checks["outcome"] = fast_false == 0 and threat_clear == 8
    result = {"schema": "robot-learning-lab.ape-robocasa-low-flank-fusion-audit.v1",
              "report_sha256": runner.sha(out / "report.json"),
              "flank_valid_ray_returns": flank_hits,
              "matched_static_returns": static_hits,
              "safe_fast_unnecessary_stops": fast_false,
              "threats_above_5cm_margin": threat_clear,
              "fast_progress_m": [progress[w, "fast_clear_held"] for w in range(4)],
              "videos_checked": videos,
              "checks": checks, "pass": all(checks.values())}
    (out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("pass", "flank_valid_ray_returns", "safe_fast_unnecessary_stops",
                       "threats_above_5cm_margin", "videos_checked")}, indent=2))
    if not result["pass"]:
        print({k: v for k, v in result["checks"].items() if not v})
        raise SystemExit(1)
