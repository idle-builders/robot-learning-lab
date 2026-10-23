"""Reconcile the saved joined RoboCasa/native-route experiment.

This checks trace/media integrity and decision timing, not contact dynamics or
physical safety. Running kitchen_trial.py is the experiment itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import kitchen_native as kitchen
import kitchen_trial as trial
import run as route_guard


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    root = trial.OUT
    report = json.loads((root / "report.json").read_text())
    checks = {}
    sources = {
        "source_sha256": Path(trial.__file__),
        "scene_builder_sha256": Path(kitchen.__file__),
        "guard_source_sha256": Path(route_guard.__file__),
        "scene": kitchen.OUT / "scene.xml",
        "native_trace_sha256": kitchen.TRACE,
        "stop_report_sha256": route_guard.replay.STOP_REPORT,
        "body_sha256": kitchen.BODY,
        "prior_map_sha256": root / "background_xy.npy",
    }
    for key, path in sources.items():
        stated = (report["room"]["scene_sha256"] if key == "scene" else
                  report["sensor"][key] if key == "prior_map_sha256" else
                  report[key])
        checks[f"hash_{key}"] = stated == digest(path)
    background = np.load(root / "background_xy.npy")
    checks["nonempty_static_map"] = (background.ndim == 2 and
                                      background.shape[1] == 2 and
                                      len(background) == report["sensor"]["static_map_points"] and
                                      len(background) > 0)
    rows = report["results"]
    checks["paired_matrix"] = (len(rows) == 24 and
        {(r["world"], r["case"], r["mode"]) for r in rows} ==
        {(world, case, mode) for world in range(4)
         for case in trial.CASES for mode in trial.MODES})
    static_returns = 0
    decoded_videos = 0
    for row in rows:
        stem = f"{row['case']}-{row['mode']}-world{row['world']}"
        saved_path = root / "episodes" / (stem + ".npz")
        checks[f"trace_hash:{stem}"] = (row["trace"]["sha256"] ==
                                           digest(saved_path))
        with np.load(saved_path, allow_pickle=False) as saved:
            all_gaps = np.r_[saved["gap_evaluator_m"],
                             saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{stem}"] = (abs(float(all_gaps.min()) -
                                          row["min_modeled_body_person_gap_m"]) < 2e-5)
            checks[f"overlap:{stem}"] = (bool(np.any(all_gaps <= 0)) ==
                                              row["modeled_person_overlap"])
            packets = [json.loads(str(item)) for item in saved["packet_json"]]
            checks[f"packets:{stem}"] = (len(packets) == row["packet_count"] ==
                                           row["trace"]["packets"])
            static_returns += sum(p.get("static_matched_returns", 0)
                                  for p in packets)
            trigger = None
            for first, second in zip(packets, packets[1:]):
                if first["route_guard"]["risk"] and second["route_guard"]["risk"]:
                    trigger = second["captured_at_s"]
                    break
            expected = 3.0 if trigger is None else trigger
            checks[f"trigger:{stem}"] = (abs(expected-row["stop_command_s"]) < 1e-5)
        checks[f"fixture:{stem}"] = row["min_sampled_body_fixture_gap_m"] > 0
        if row["video"]:
            video_path = Path(row["video"]["path"])
            checks[f"video_hash:{stem}"] = digest(video_path) == row["video"]["sha256"]
            capture = cv2.VideoCapture(str(video_path))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            decoded = 0
            for index in (0, count//2, count-1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                decoded += int(ok and frame.shape == (562, 960, 3)) if ok else 0
            capture.release()
            checks[f"video_decode:{stem}"] = (count == row["video"]["frames"] and
                                                decoded == 3)
            decoded_videos += 1
    checks["static_returns_exercised"] = static_returns > 0
    by = {(r["world"], r["case"], r["mode"]): r for r in rows}
    checks["controls_expose_collision"] = all(
        by[w, c, "none"]["modeled_person_overlap"]
        for w in range(4) for c in ("slow_cross", "pause_path"))
    checks["guard_clears_hazards"] = all(
        not by[w, c, "mapped_low_guard"]["below_5cm_person_margin"]
        for w in range(4) for c in ("slow_cross", "pause_path"))
    checks["safe_fast_continues"] = all(
        by[w, "fast_clear", "mapped_low_guard"]["stop_command_s"] == 3.0
        for w in range(4))
    comparison = json.loads((root / "comparison-video.json").read_text())
    checks["comparison_report"] = comparison["source_report_sha256"] == digest(root / "report.json")
    checks["comparison_video"] = comparison["sha256"] == digest(Path(comparison["video"]))
    for mode, stated in comparison["source_videos_sha256"].items():
        checks[f"comparison_source:{mode}"] = (
            stated == digest(root / "videos" / f"slow_cross-{mode}-world0.mp4"))
    capture = cv2.VideoCapture(comparison["video"])
    ok, frame = capture.read()
    checks["comparison_decode"] = (ok and frame.shape == (544, 1280, 3) and
                                    int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) ==
                                    comparison["frames"] == 100)
    capture.release()
    outcome = {"schema": "robot-learning-lab.ape-robocasa-native-guard-audit.v1",
               "report_sha256": digest(root / "report.json"),
               "static_map_points": len(background),
               "matched_static_ray_returns": static_returns,
               "videos_checked": decoded_videos,
               "checks": checks,
               "pass": all(checks.values())}
    (root / "audit.json").write_text(json.dumps(outcome, indent=2)+"\n")
    return outcome


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("pass", "static_map_points", "matched_static_ray_returns",
                       "videos_checked")}, indent=2))
    if not result["pass"]:
        print({key: value for key, value in result["checks"].items() if not value})
        raise SystemExit(1)
