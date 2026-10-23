"""Independent saved-trace/media consistency check for the V16 quality stress."""

from __future__ import annotations

import json

import cv2
import numpy as np

import range_quality_trial as quality
import flank_sensor_trial as flank
import kitchen_native as kitchen
import kitchen_trial as trial


def main() -> None:
    report = json.loads((quality.OUT / "report.json").read_text())
    checks = {
        "source": report["source_sha256"] == trial.sha(quality.Path(quality.__file__)),
        "v15_source": report["v15_source_sha256"] == trial.sha(quality.Path(flank.__file__)),
        "scene": report["scene_sha256"] == trial.sha(kitchen.OUT / "scene.xml"),
        "native_trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "map": report["sensor"]["initial_map_sha256"] ==
               trial.sha(quality.OUT / "background_xy.npy"),
        "v15": report["v15_report_sha256"] == trial.sha(flank.OUT / "report.json"),
        "mount": report["mount_screen_sha256"] ==
                 trial.sha(flank.OUT / "mount-visibility.json"),
        "matrix": len(report["results"]) == 72 and
                  len({(r["condition"], r["seed"], r["world"], r["case"])
                       for r in report["results"]}) == 72,
    }
    summaries = {}
    videos = 0
    for result in report["results"]:
        condition = result["condition"]
        group = summaries.setdefault(condition, {"episodes": 0, "fast_false_stops": 0,
                                                 "threats_below_5cm": 0,
                                                 "threat_overlaps": 0,
                                                 "unavailable_rays": 0})
        group["episodes"] += 1
        group["fast_false_stops"] += int(result["case"] == "fast_clear_held" and
                                         result["stop_command_s"] < 3.)
        group["threats_below_5cm"] += int(result["case"] != "fast_clear_held" and
                                          result["below_5cm_person_margin"])
        group["threat_overlaps"] += int(result["case"] != "fast_clear_held" and
                                        result["modeled_person_overlap"])
        path = quality.Path(result["trace"]["path"])
        key = f"{condition}:{result['seed']}:{result['world']}:{result['case']}"
        checks[f"trace_hash:{key}"] = trial.sha(path) == result["trace"]["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            all_gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{key}"] = (
                round(float(all_gaps.min()), 5) == result["min_modeled_body_person_gap_m"] and
                bool((all_gaps <= 0).any()) == result["modeled_person_overlap"] and
                bool((all_gaps < trial.MIN_MARGIN_M).any()) ==
                result["below_5cm_person_margin"])
            packets = [json.loads(raw) for raw in saved["packet_json"]]
            checks[f"packet:{key}"] = bool(packets) and all(
                p["quality_condition"] == condition and
                len(p["ranges_m"]) == len(p["ray_health"]) ==
                len(p["ray_sources"]) == 73 and
                p["unavailable_rays"] == len(p["ray_health"])-sum(p["ray_health"]) and
                p["valid_returns"] == sum(x is not None for x in p["ranges_m"]) and
                all(x is None for x, healthy in zip(p["ranges_m"], p["ray_health"])
                    if not healthy)
                for p in packets)
            group["unavailable_rays"] += sum(p["unavailable_rays"] for p in packets)
        video = result["video"]
        if video is not None:
            videos += 1
            movie = quality.Path(video["path"])
            cap = cv2.VideoCapture(str(movie))
            frames = 0
            while True:
                okay, frame = cap.read()
                if not okay:
                    break
                frames += int(frame.size > 0)
            cap.release()
            checks[f"video:{key}"] = (trial.sha(movie) == video["sha256"] and
                                       frames == video["frames"])
    checks["six_videos"] = videos == 6
    result = {"schema": "robot-learning-lab.ape-robocasa-low-fan-range-quality-audit.v1",
              "report_sha256": trial.sha(quality.OUT / "report.json"),
              "summary": summaries, "videos_checked": videos,
              "checks": checks, "pass": all(checks.values())}
    (quality.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "summary": summaries,
                      "videos_checked": videos}, indent=2))
    if not result["pass"]:
        raise SystemExit("V16 audit failed")


if __name__ == "__main__":
    main()
