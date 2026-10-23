"""Saved-evidence audit for the V18 single-front sensor comparison."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import kitchen_native as kitchen
import kitchen_trial as trial
import single_front_trial as single


def main() -> None:
    report = json.loads((single.OUT / "report.json").read_text())
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(single.__file__)),
        "dense": report["dense_source_sha256"] ==
                 trial.sha(Path(single.dense.__file__)),
        "v17": report["v17_source_sha256"] ==
               trial.sha(Path(single.right.__file__)),
        "runner": report["episode_runner_sha256"] == trial.sha(Path(trial.__file__)),
        "command": report["command_guard_sha256"] ==
                   trial.sha(Path(single.command.__file__)),
        "scene": report["scene_sha256"] == trial.sha(kitchen.OUT / "scene.xml"),
        "native_trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "v17_report": report["v17_report_sha256"] ==
                      trial.sha(single.right.OUT / "report.json"),
        "matrix": len(report["results"]) == 24 and
                  len({(r["map_mode"], r["world"], r["case"])
                       for r in report["results"]}) == 24,
    }
    outcomes = {}
    videos = 0
    for mode in single.MAP_MODES:
        info = report["maps"][mode]
        checks[f"map:{mode}"] = (info["sha256"] ==
                                 trial.sha(single.OUT / mode / "background_xy.npy"))
        outcomes[mode] = {"fast_false_stops": 0, "threat_overlaps": 0,
                          "threats_below_5cm": 0, "valid_ray_returns": 0}
    for result in report["results"]:
        mode = result["map_mode"]
        key = f"{mode}:{result['world']}:{result['case']}"
        stats = outcomes[mode]
        stats["fast_false_stops"] += int(result["case"] == "fast_right_held" and
                                         result["stop_command_s"] < 3.)
        stats["threat_overlaps"] += int(result["case"] != "fast_right_held" and
                                        result["modeled_person_overlap"])
        stats["threats_below_5cm"] += int(result["case"] != "fast_right_held" and
                                          result["below_5cm_person_margin"])
        trace = Path(result["trace"]["path"])
        checks[f"trace_hash:{key}"] = trial.sha(trace) == result["trace"]["sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{key}"] = (
                round(float(gaps.min()), 5) == result["min_modeled_body_person_gap_m"] and
                bool((gaps <= 0).any()) == result["modeled_person_overlap"] and
                bool((gaps < trial.MIN_MARGIN_M).any()) ==
                result["below_5cm_person_margin"])
            packets = [json.loads(p) for p in saved["packet_json"]]
            checks[f"packet:{key}"] = bool(packets) and all(
                len(p["ranges_m"]) == 73 and
                p["valid_returns"] == sum(value is not None for value in p["ranges_m"])
                for p in packets)
            stats["valid_ray_returns"] += sum(p["valid_returns"] for p in packets)
        video = result["video"]
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
    checks["four_videos"] = videos == 4
    audit = {"schema": "robot-learning-lab.ape-robocasa-single-front-right-entry-audit.v1",
             "report_sha256": trial.sha(single.OUT / "report.json"),
             "outcomes": outcomes, "videos_checked": videos,
             "checks": checks, "pass": all(checks.values())}
    (single.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "outcomes": outcomes,
                      "videos_checked": videos}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V18 audit failed")


if __name__ == "__main__":
    main()
