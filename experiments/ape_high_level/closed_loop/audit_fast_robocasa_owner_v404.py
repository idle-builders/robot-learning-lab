"""Audit timing, identity, social outcome and video for V403 fast owner loop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-fast-owner-v403"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a: list[float] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    source = ROOT / "experiments/ape_high_level/closed_loop/run_fast_robocasa_owner_v403.py"
    rows = []
    first_frames = {}
    for episode in report["episodes"]:
        observations = [e for e in episode["events"] if e.get("actor") is not None]
        accepts = [e for e in observations if e["actor"]["status"] in
                   ("metric_localized", "visually_reacquired_unlocalized")]
        wrong = [e for e in accepts if e["evaluator"]["shown_visible_pixels"] < 100 or
                 iou(e["actor"]["box"], e["evaluator"]["shown_box"]) < .3]
        premature = [e for e in accepts if e["stage"] == "actor_search_view"]
        ages = [e["sensors"]["frame_age_s"] for e in observations]
        inference = [e["actor"]["inference_wall_s"] for e in observations]
        overruns = [e["tick_overrun_s"] for e in observations]
        video = episode["video"]
        path = Path(video["path"])
        reader = imageio.get_reader(path)
        frame_count = reader.count_frames()
        first_frames[(episode["style"], episode["response"], episode["depth"])] = (
            reader.get_data(0)[48:528].astype(np.int16))
        reader.get_data(frame_count - 1)
        reader.close()
        video_ok = sha256(path) == video["sha256"] and frame_count == video["frames"]
        rows.append({"style": episode["style"], "response": episode["response"],
                     "depth": episode["depth"], "outcome": episode["outcome"],
                     "scene_seed": episode.get("scene_seed"),
                     "scene_fingerprint": episode.get("scene_fingerprint"),
                     "first_search_rgb_sha256": observations[0]["rgb_sha256"],
                     "accepted_frames": len(accepts), "wrong_accept_frames": len(wrong),
                     "premature_search_accept_frames": len(premature),
                     "first_location_error_m": episode["first_location_error_m"],
                     "asks": sum(e["stage"] in ("actor_asks_owner_to_show", "actor_retries_show_request")
                                 for e in episode["events"]),
                     "last_sim_time_s": max(e["time_s"] for e in episode["events"]),
                     "mean_inference_wall_s": statistics.mean(inference),
                     "median_frame_age_s": statistics.median(ages),
                     "max_frame_age_s": max(ages),
                     "overrun_frames": sum(v > 0 for v in overruns),
                     "accepted_stale_frames": sum(e["sensors"]["frame_age_s"] > .2 for e in accepts),
                     "video_ok": video_ok})
    expected = {(s, r, d) for s in (1, 2, 3)
                for r in ("compliant", "delayed", "ignored") for d in (False, True)}
    actual = {(r["style"], r["response"], r["depth"]) for r in rows}
    first_frame_diffs = []
    for style in (1, 2, 3):
        base = first_frames[(style, "compliant", False)]
        for response in ("compliant", "delayed", "ignored"):
            for depth in (False, True):
                diff = np.abs(base - first_frames[(style, response, depth)])
                first_frame_diffs.append({"style": style, "response": response, "depth": depth,
                                          "mean_absolute_pixel_difference": float(np.mean(diff)),
                                          "fraction_channels_over_15": float(np.mean(diff > 15))})
    gates = {
        "source_hash": sha256(source) == report["source_sha256"],
        "matrix_complete": len(rows) == 18 and actual == expected,
        "paired_scene_fingerprint": all(
            len({r["scene_fingerprint"] for r in rows if r["style"] == style}) == 1
            for style in (1, 2, 3)),
        "paired_first_search_rgb_exact_hash": all(
            len({r["first_search_rgb_sha256"] for r in rows if r["style"] == style}) == 1
            for style in (1, 2, 3)),
        "paired_first_search_rgb_near_identical": all(
            d["mean_absolute_pixel_difference"] < 1. and
            d["fraction_channels_over_15"] < .001 for d in first_frame_diffs),
        "teaching_overlap": iou(report["teaching"]["actor_box"],
                                report["teaching"]["evaluator_box"]) >= .3,
        "no_wrong_accept": all(r["wrong_accept_frames"] == 0 for r in rows),
        "no_premature_search_accept": all(r["premature_search_accept_frames"] == 0 for r in rows),
        "no_stale_accept": all(r["accepted_stale_frames"] == 0 for r in rows),
        "owner_response_separated": all(
            (r["outcome"] == "unknown_ask_for_help" if r["response"] == "ignored"
             else r["outcome"] == ("metric_localized" if r["depth"]
                                    else "visually_reacquired_unlocalized"))
            for r in rows),
        "metric_error_under_8cm": all(r["first_location_error_m"] is not None and
                                       r["first_location_error_m"] <= .08
                                       for r in rows if r["depth"] and r["response"] != "ignored"),
        "all_frames_under_100ms": all(r["overrun_frames"] == 0 for r in rows),
        "videos_decode_and_hash": all(r["video_ok"] for r in rows),
    }
    output = {"scope": "fast frontend timing and live owner-loop audit; no body or native motion",
              "source_sha256": sha256(Path(__file__)), "gates": gates,
              "first_frame_diffs": first_frame_diffs, "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"gates": gates, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
