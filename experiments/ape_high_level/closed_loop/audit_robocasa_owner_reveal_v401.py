"""Independently score live RoboCasa owner-reveal event traces and videos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import imageio.v2 as imageio


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-owner-reveal-v400"


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
    source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_owner_reveal_v400.py"
    source_match = sha256(source) == report["source_sha256"]
    rows: list[dict] = []
    for episode in report["episodes"]:
        observations = [e for e in episode["events"] if e.get("actor") is not None]
        accepts = [e for e in observations if e["actor"]["status"] in
                   ("metric_localized", "visually_reacquired_unlocalized")]
        wrong_accepts = [e for e in accepts if e["evaluator"]["shown_visible_pixels"] < 100 or
                         iou(e["actor"]["box"], e["evaluator"]["shown_box"]) < .3]
        search_accepts = [e for e in accepts if e["stage"] == "actor_search_view"]
        teach = episode["events"][0]
        teach_iou = iou(teach.get("teaching", {}).get("actor_box"),
                        teach["evaluator"]["shown_box"])
        video = episode.get("video")
        if video:
            path = Path(video["path"])
            reader = imageio.get_reader(path)
            frame_count = reader.count_frames()
            # Decode the last frame as well as reading the container's count.
            reader.get_data(frame_count - 1)
            reader.close()
            video_ok = sha256(path) == video["sha256"] and frame_count == video["frames"]
        else:
            video_ok = False
        inference = [e["actor"]["inference_wall_s"] for e in observations]
        row = {"style": episode["style"], "response": episode["response"],
               "depth": episode["depth"], "outcome": episode["outcome"],
               "teaching_iou": teach_iou, "accepted_frames": len(accepts),
               "wrong_accept_frames": len(wrong_accepts),
               "premature_search_accept_frames": len(search_accepts),
               "first_location_error_m": episode.get("first_location_error_m"),
               "asks": sum(e["stage"] in ("actor_asks_owner_to_show", "actor_retries_show_request")
                           for e in episode["events"]),
               "mean_inference_wall_s": statistics.mean(inference) if inference else None,
               "max_inference_wall_s": max(inference) if inference else None,
               "video_ok": video_ok}
        rows.append(row)
    expected = {(style, response, depth) for style in (1, 2)
                for response in ("compliant", "delayed", "ignored")
                for depth in (False, True)}
    actual = {(r["style"], r["response"], r["depth"]) for r in rows}
    gates = {
        "source_hash": source_match,
        "matrix_complete": len(rows) == 12 and actual == expected,
        "teaching_overlap": all(r["teaching_iou"] >= .3 for r in rows),
        "no_wrong_accept": all(r["wrong_accept_frames"] == 0 for r in rows),
        "no_premature_search_accept": all(r["premature_search_accept_frames"] == 0 for r in rows),
        "owner_response_separated": all(
            (r["outcome"] == "unknown_ask_for_help" if r["response"] == "ignored"
             else r["outcome"] == ("metric_localized" if r["depth"]
                                    else "visually_reacquired_unlocalized"))
            for r in rows),
        "metric_error_under_8cm": all(r["first_location_error_m"] is not None and
                                       r["first_location_error_m"] <= .08
                                       for r in rows if r["depth"] and r["response"] != "ignored"),
        "inference_fits_0p4s_search_cadence": all(r["max_inference_wall_s"] is not None and
                                                  r["max_inference_wall_s"] <= .4 for r in rows),
        "videos_decode_and_hash": all(r["video_ok"] for r in rows),
    }
    output = {"scope": "independent trace/video audit; only two kitchen styles, no body or native action",
              "source_sha256": sha256(Path(__file__)), "source_match": source_match,
              "gates": gates, "rows": rows}
    path = OUT / "audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"gates": gates, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
