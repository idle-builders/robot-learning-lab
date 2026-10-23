"""Audit paired realistic-room pose errors and video integrity for V405."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-localization-v405"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    source = ROOT / "experiments/ape_high_level/closed_loop/probe_robocasa_localization_v405.py"
    rows = []
    first_frames = {}
    for episode in report["episodes"]:
        video = episode["video"]
        path = Path(video["path"])
        reader = imageio.get_reader(path)
        frames = reader.count_frames()
        first_frames[(episode["style"], episode["slip_m"], episode["lidar_max_m"])] = (
            reader.get_data(0)[:480].astype(np.int16))
        reader.get_data(frames - 1)
        reader.close()
        trace = episode["trace"]
        rows.append({"style": episode["style"], "slip_m": episode["slip_m"],
                     "lidar_max_m": episode["lidar_max_m"],
                     "scene_seed": episode.get("scene_seed"),
                     "scene_fingerprint": episode.get("scene_fingerprint"),
                     "first_rgb_sha256": trace[0]["rgb_sha256"],
                     "final_error_m": episode["final_error_m"],
                     "mean_error_m": episode["mean_error_m"],
                     "registered_frames": episode["registered_frames"],
                     "mean_scan_returns": statistics.mean(r["range_returns"] for r in trace),
                     "misleading_range_registrations": sum(
                         r["range_status"]["status"] == "registered" and
                         r["range_error_m"] > .10 for r in trace),
                     "trace_frames": len(trace),
                     "video_ok": sha256(path) == video["sha256"] and
                                 frames == video["frames"] == len(trace)})
    expected = {(style, slip, max_m) for style in (1, 2, 3)
                for slip in (0., .12) for max_m in (2.2, 3.0, 5.0)}
    actual = {(r["style"], r["slip_m"], r["lidar_max_m"]) for r in rows}
    first_frame_diffs = []
    for style in (1, 2, 3):
        base = first_frames[(style, 0., 2.2)]
        for slip in (0., .12):
            for max_m in (2.2, 3.0, 5.0):
                diff = np.abs(base - first_frames[(style, slip, max_m)])
                first_frame_diffs.append({"style": style, "slip_m": slip,
                                          "lidar_max_m": max_m,
                                          "mean_absolute_pixel_difference": float(np.mean(diff)),
                                          "fraction_channels_over_15": float(np.mean(diff > 15))})
    by_profile = defaultdict(list)
    for row in rows:
        by_profile[(row["lidar_max_m"], row["slip_m"])].append(row)
    profile_summary = {}
    for (max_m, slip), group in by_profile.items():
        profile_summary[f"range{max_m:.1f}-slip{slip:.2f}"] = {
            "range_final_error_median_m": statistics.median(r["final_error_m"]["range"] for r in group),
            "range_final_error_max_m": max(r["final_error_m"]["range"] for r in group),
            "rgbd_final_error_median_m": statistics.median(r["final_error_m"]["rgbd"] for r in group),
            "command_final_error_median_m": statistics.median(r["final_error_m"]["command"] for r in group),
            "misleading_range_registrations": sum(r["misleading_range_registrations"] for r in group),
        }
    pairs = {(r["style"], r["slip_m"], r["lidar_max_m"]): r for r in rows}
    gates = {
        "source_hash": sha256(source) == report["source_sha256"],
        "matrix_complete": len(rows) == 18 and actual == expected,
        "trace_video_integrity": all(r["trace_frames"] == 48 and r["video_ok"] for r in rows),
        "paired_scene_fingerprint": all(
            len({r["scene_fingerprint"] for r in rows if r["style"] == style}) == 1
            for style in (1, 2, 3)),
        "paired_first_rgb_exact_hash": all(
            len({r["first_rgb_sha256"] for r in rows if r["style"] == style}) == 1
            for style in (1, 2, 3)),
        "paired_scene_render_near_identical": all(
            d["mean_absolute_pixel_difference"] < 1. and
            d["fraction_channels_over_15"] < .001 for d in first_frame_diffs),
        "paired_command_and_rgbd": all(
            abs(pairs[(style, slip, 2.2)]["final_error_m"][source] -
                pairs[(style, slip, max_m)]["final_error_m"][source]) < .015
            for style in (1, 2, 3) for slip in (0., .12)
            for max_m in (3.0, 5.0) for source in ("command", "rgbd")),
        "long_range_no_slip_under_5cm": all(
            pairs[(style, 0., 5.0)]["final_error_m"]["range"] < .05 for style in (1, 2, 3)),
        "long_range_slip_under_6cm": all(
            pairs[(style, .12, 5.0)]["final_error_m"]["range"] < .06 for style in (1, 2, 3)),
        "long_range_beats_short_on_slip": all(
            pairs[(style, .12, 5.0)]["final_error_m"]["range"] <
            pairs[(style, .12, 2.2)]["final_error_m"]["range"] for style in (1, 2, 3)),
        "rgbd_beats_command_on_slip": all(
            pairs[(style, .12, 5.0)]["final_error_m"]["rgbd"] <
            pairs[(style, .12, 5.0)]["final_error_m"]["command"] for style in (1, 2, 3)),
    }
    output = {"scope": "paired aisle-pose replay audit; truth only evaluator; no physical gait or safe-route claim",
              "source_sha256": sha256(Path(__file__)), "gates": gates,
              "profile_summary": profile_summary, "first_frame_diffs": first_frame_diffs,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"gates": gates, "profile_summary": profile_summary}, indent=2))


if __name__ == "__main__":
    main()
