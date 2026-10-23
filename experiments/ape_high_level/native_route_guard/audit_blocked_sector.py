"""Reconcile V14 blocked-sector sensor-health and full-body outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import blocked_sector_trial as current
import dense_low_trial as healthy
import kitchen_native as kitchen
import kitchen_trial as runner
import command_route_trial as command
import run as guard


def audit() -> dict:
    out = current.OUT
    report = json.loads((out / "report.json").read_text())
    control = json.loads((healthy.OUT / "report.json").read_text())
    checks = {}
    paths = {"source_sha256": Path(current.__file__),
             "episode_runner_sha256": Path(runner.__file__),
             "dense_sensor_sha256": Path(healthy.__file__),
             "command_guard_sha256": Path(command.__file__),
             "original_guard_sha256": Path(guard.__file__),
             "native_trace_sha256": kitchen.TRACE,
             "body_sha256": kitchen.BODY,
             "v13_report_sha256": healthy.OUT / "report.json"}
    for key, path in paths.items():
        checks[f"hash:{key}"] = report[key] == runner.sha(path)
    checks["same_scene"] = report["scene_sha256"] == control["scene_sha256"]
    checks["same_prior"] = (report["sensor"]["prior_map_sha256"] ==
                            control["sensor"]["prior_map_sha256"] ==
                            runner.sha(out / "background_xy.npy"))
    rows = report["results"]
    checks["matrix"] = (len(rows) == 24 and
        {(r["response"], r["world"], r["case"]) for r in rows} ==
        {(response, world, case) for response in current.RESPONSES
         for world in range(4) for case in current.holdout.CASES})
    progress = {}
    videos = 0
    for row in rows:
        response, world, case = row["response"], row["world"], row["case"]
        stem = f"{case}-mapped_low_guard-world{world}"
        path = out / response / "episodes" / f"{stem}.npz"
        checks[f"trace_hash:{response}:{stem}"] = row["trace"]["sha256"] == runner.sha(path)
        with np.load(path, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{response}:{stem}"] = (
                abs(float(gaps.min())-row["min_modeled_body_person_gap_m"]) < 2e-5 and
                bool(np.any(gaps <= 0)) == row["modeled_person_overlap"])
            packets = [json.loads(str(x)) for x in saved["packet_json"]]
            checks[f"packet_fault:{response}:{stem}"] = (
                len(packets) == row["packet_count"] and
                all(len(p["ranges_m"]) == len(p["ray_health"]) == 73 and
                    p["fault"] == "left_mount_sector_occluded" and
                    all(value is None for value, ok in zip(p["ranges_m"], p["ray_health"])
                        if not ok) and
                    sum(not ok for ok in p["ray_health"]) == 26
                    for p in packets))
            trigger = next((b["captured_at_s"] for a, b in zip(packets, packets[1:])
                            if a["route_guard"]["risk"] and b["route_guard"]["risk"]),
                           3.0)
            checks[f"trigger:{response}:{stem}"] = abs(trigger-row["stop_command_s"]) < 1e-5
            root = saved["root_xy_evaluator_m"]
            progress[response, world, case] = float(root[-1, 0]-root[0, 0])
        if row["video"]:
            movie = Path(row["video"]["path"])
            checks[f"video_hash:{response}:{stem}"] = row["video"]["sha256"] == runner.sha(movie)
            cap = cv2.VideoCapture(str(movie))
            ok, frame = cap.read()
            checks[f"video_decode:{response}:{stem}"] = (
                ok and frame.shape == (562, 960, 3) and
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == row["video"]["frames"] == 100)
            cap.release()
            videos += 1
    by = {(r["response"], r["world"], r["case"]): r for r in rows}
    slow_overlap_unaware = sum(by["ignore_health", w, "slow_cross_held"]["modeled_person_overlap"]
                               for w in range(4))
    slow_overlap_hold = sum(by["hold_on_unavailable", w, "slow_cross_held"]["modeled_person_overlap"]
                            for w in range(4))
    all_held_early = all(by["hold_on_unavailable", w, c]["stop_command_s"] == .1
                         for w in range(4) for c in current.holdout.CASES)
    fast_held_progress = [progress["hold_on_unavailable", w, "fast_clear_held"]
                          for w in range(4)]
    fast_healthy_progress = []
    for w in range(4):
        path = healthy.OUT / "episodes" / f"fast_clear_held-mapped_low_guard-world{w}.npz"
        with np.load(path, allow_pickle=False) as saved:
            root = saved["root_xy_evaluator_m"]
            fast_healthy_progress.append(float(root[-1, 0]-root[0, 0]))
    checks["observed_fault_outcome"] = (slow_overlap_unaware == 3 and
                                        slow_overlap_hold == 0 and all_held_early and
                                        max(fast_held_progress) < min(fast_healthy_progress))
    media = json.loads((out / "comparison-videos.json").read_text())
    checks["comparison_report"] = media["report_sha256"] == runner.sha(out / "report.json")
    checks["comparison_pairs"] = (
        {(c["case"], c["world"]) for c in media["clips"]} ==
        {("slow_cross_held", 0), ("fast_clear_held", 1)})
    for clip in media["clips"]:
        case, world = clip["case"], clip["world"]
        stem = f"{case}-mapped_low_guard-world{world}.mp4"
        checks[f"comparison_hash:{case}:{world}"] = (
            clip["sha256"] == runner.sha(Path(clip["path"])) and
            all(clip["source_hashes"][response] ==
                runner.sha(out / response / "videos" / stem)
                for response in current.RESPONSES))
        cap = cv2.VideoCapture(clip["path"])
        ok, frame = cap.read()
        checks[f"comparison_decode:{case}:{world}"] = (
            ok and frame.shape == (544, 1280, 3) and
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == clip["frames"] == 100)
        cap.release()
    result = {"schema": "robot-learning-lab.ape-robocasa-low-sector-loss-audit.v1",
              "report_sha256": runner.sha(out / "report.json"),
              "unaware_slow_overlap_worlds": slow_overlap_unaware,
              "coverage_hold_slow_overlap_worlds": slow_overlap_hold,
              "coverage_hold_early_stops": sum(
                  by["hold_on_unavailable", w, c]["stop_command_s"] == .1
                  for w in range(4) for c in current.holdout.CASES),
              "fast_healthy_progress_m": fast_healthy_progress,
              "fast_coverage_hold_progress_m": fast_held_progress,
              "videos_checked": videos,
              "checks": checks, "pass": all(checks.values())}
    (out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: result[key] for key in
                      ("pass", "unaware_slow_overlap_worlds",
                       "coverage_hold_slow_overlap_worlds",
                       "coverage_hold_early_stops", "videos_checked")}, indent=2))
    if not result["pass"]:
        print({k: v for k, v in result["checks"].items() if not v})
        raise SystemExit(1)
