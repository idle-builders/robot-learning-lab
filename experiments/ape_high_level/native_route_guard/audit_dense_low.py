"""Audit matched sparse/dense low-scan kitchen evidence."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import dense_low_trial as current
import changed_speed_trial as sparse
import command_route_trial as command
import kitchen_native as kitchen
import kitchen_trial as runner
import run as guard


def audit() -> dict:
    out = current.OUT
    report = json.loads((out / "report.json").read_text())
    comparison = json.loads((sparse.OUT / "report.json").read_text())
    checks = {}
    paths = {"source_sha256": Path(current.__file__),
             "episode_runner_sha256": Path(runner.__file__),
             "command_guard_sha256": Path(command.__file__),
             "original_guard_sha256": Path(guard.__file__),
             "native_trace_sha256": kitchen.TRACE,
             "body_sha256": kitchen.BODY,
             "v11_report_sha256": sparse.OUT / "report.json"}
    for key, path in paths.items():
        checks[f"hash:{key}"] = report[key] == runner.sha(path)
    checks["same_scene"] = report["scene_sha256"] == comparison["room"]["scene_sha256"]
    background = np.load(out / "background_xy.npy")
    checks["dense_initial_map"] = (background.shape == (report["sensor"]["static_map_points"], 2)
                                   and len(background) > comparison["sensor"]["static_map_points"]
                                   and report["sensor"]["prior_map_sha256"] ==
                                   runner.sha(out / "background_xy.npy"))
    rows = report["results"]
    checks["matrix"] = (len(rows) == 12 and
        {(r["world"], r["case"]) for r in rows} ==
        {(w, c) for w in range(4) for c in sparse.CASES})
    matched = 0
    videos = 0
    for row in rows:
        stem = f"{row['case']}-mapped_low_guard-world{row['world']}"
        path = out / "episodes" / f"{stem}.npz"
        checks[f"trace_hash:{stem}"] = row["trace"]["sha256"] == runner.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            gap = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{stem}"] = abs(float(gap.min()) - row["min_modeled_body_person_gap_m"]) < 2e-5
            checks[f"overlap:{stem}"] = bool(np.any(gap <= 0)) == row["modeled_person_overlap"]
            packets = [json.loads(str(p)) for p in saved["packet_json"]]
            checks[f"rays:{stem}"] = (len(packets) == row["packet_count"] and
                all(p["sensor"] == "low_dense" and len(p["ranges_m"]) == 73
                    for p in packets))
            matched += sum(p["static_matched_returns"] for p in packets)
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
            videos += 1
    by = {(r["world"], r["case"]): r for r in rows}
    sparse_by = {(r["world"], r["case"], r["mode"]): r for r in comparison["results"]}
    sparse_false = sum(sparse_by[w, "fast_clear_held", "mapped_low_guard"]["stop_command_s"] < 3
                       for w in range(4))
    dense_false = sum(by[w, "fast_clear_held"]["stop_command_s"] < 3 for w in range(4))
    dense_threat_clear = sum(not by[w, c]["below_5cm_person_margin"]
                             for w in range(4)
                             for c in ("slow_cross_held", "pause_path_held"))
    closest_pause = min(by[w, "pause_path_held"]["min_modeled_body_person_gap_m"]
                        for w in range(4))
    checks["outcome"] = (sparse_false == 2 and dense_false == 0 and
                          dense_threat_clear == 8 and .05 < closest_pause < .07)
    checks["static_returns"] = matched > 0
    media = json.loads((out / "comparison-videos.json").read_text())
    checks["comparison_reports"] = (
        media["sparse_report_sha256"] == runner.sha(sparse.OUT / "report.json") and
        media["dense_report_sha256"] == runner.sha(out / "report.json"))
    checks["comparison_pairs"] = (
        {(c["case"], c["world"]) for c in media["clips"]} ==
        {("fast_clear_held", 1), ("pause_path_held", 0)})
    for clip in media["clips"]:
        case, world = clip["case"], clip["world"]
        stem = f"{case}-mapped_low_guard-world{world}.mp4"
        checks[f"comparison_hash:{case}:{world}"] = (
            clip["sha256"] == runner.sha(Path(clip["path"])) and
            clip["source_17_sha256"] == runner.sha(sparse.OUT / "videos" / stem) and
            clip["source_73_sha256"] == runner.sha(out / "videos" / stem))
        cap = cv2.VideoCapture(clip["path"])
        ok, frame = cap.read()
        checks[f"comparison_decode:{case}:{world}"] = (
            ok and frame.shape == (544, 1280, 3) and
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == clip["frames"] == 100)
        cap.release()
    result = {"schema": "robot-learning-lab.ape-robocasa-dense-low-audit.v1",
              "report_sha256": runner.sha(out / "report.json"),
              "matched_static_ray_returns": matched,
              "sparse_safe_fast_false_stops": sparse_false,
              "dense_safe_fast_false_stops": dense_false,
              "dense_threats_above_5cm_margin": dense_threat_clear,
              "closest_dense_pause_gap_m": closest_pause,
              "videos_checked": videos,
              "checks": checks, "pass": all(checks.values())}
    (out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("pass", "sparse_safe_fast_false_stops",
                       "dense_safe_fast_false_stops", "dense_threats_above_5cm_margin",
                       "closest_dense_pause_gap_m", "videos_checked")}, indent=2))
    if not result["pass"]:
        print({key: value for key, value in result["checks"].items() if not value})
        raise SystemExit(1)
