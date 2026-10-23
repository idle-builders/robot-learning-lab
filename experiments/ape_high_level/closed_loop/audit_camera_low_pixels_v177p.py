"""Audit V176p evaluator segmentation against saved actor depth masks."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from run import ROOT, sha
import direct_structural_rooms_v270 as structural
from articulated_person_scene import add_person
import diagnose_camera_low_pixels_v176p as v176p


SOURCE = Path(__file__)
REPORT = v176p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = segmentation_packets = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v176p.__file__)) == report["source_sha256"],
          "Segmentation diagnostic source changed")
    check(sha(v176p.v171p.OUT / "report.json") ==
          report["v171_report_sha256"],
          "Actor camera source report changed")
    check(len(report["rows"]) == 128,
          "Segmentation diagnostic matrix incomplete")
    old = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old(kind, side))
    room = None
    try:
        room = v176p.v160p.VariantRoom(
            case=(170, "present", "right"),
            pose_source="command", branch=v176p.OUT)
        model = room.model
        for row in report["rows"]:
            check(sha(ROOT / row["source_depth_packet_path"]) ==
                  row["source_depth_packet_sha256"] and
                  sha(ROOT / row["source_rgb_path"]) ==
                  row["source_rgb_sha256"],
                  "Actor source RGB-D packet changed")
            seg_file = ROOT / row["segmentation_packet_path"]
            check(sha(seg_file) == row["segmentation_packet_sha256"],
                  "Evaluator segmentation packet changed")
            with np.load(ROOT / row["source_depth_packet_path"],
                         allow_pickle=False) as depth, np.load(
                    seg_file, allow_pickle=False) as segmentation:
                valid, selected, nearest = v176p.selected_from_packet(depth)
                ids = segmentation["sampled_geom_ids"]
                types = segmentation["sampled_geom_types"]
                time_s = float(segmentation["capture_time_s"])
            check(ids.shape == valid.shape and
                  types.shape == valid.shape and
                  time_s == row["time_s"],
                  "Evaluator segmentation alignment changed")
            selected_ids = ids[valid][selected]
            nearest_ids = ids[valid][nearest]
            all_counts = dict(Counter(
                v176p.category(model, int(gid)) for gid in selected_ids))
            near_counts = dict(Counter(
                v176p.category(model, int(gid)) for gid in nearest_ids))
            check(all_counts == row[
                "low_roi_origin_counts_evaluator_only"] and
                  near_counts == row[
                      "nearest_10pct_origin_counts_evaluator_only"] and
                  len(selected_ids) == row["low_roi_depth_support"],
                  "Pixel origin labels differ from saved masks")
            check(row["rerender_mean_rgb_absolute_difference"] < 2.,
                  "Rerender diverged from actor RGB frame")
            segmentation_packets += 1
    finally:
        if room is not None:
            room.close()
        structural.structural_model = old
    for camera in v176p.v171p.CAMERAS:
        group = [row for row in report["rows"]
                 if row["camera"] == camera]
        item = report["summary"][camera]
        check(item["diagnostic_frames"] == len(group) == 64 and
              item["low_roi_self_pixels"] == sum(row[
                  "low_roi_origin_counts_evaluator_only"].get(
                      "ape_self", 0) for row in group) and
              item["nearest_10pct_self_pixels"] == sum(row[
                  "nearest_10pct_origin_counts_evaluator_only"].get(
                      "ape_self", 0) for row in group),
              "Camera self-origin summary changed")
    pair_file = v176p.v171p.OUT / "paired-video-report.json"
    pair = json.loads(pair_file.read_text())
    video = ROOT / pair["video_path"]
    check(sha(video) == pair["video_sha256"],
          "Paired camera video changed")
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded += 1
        check(image.ndim == 3 and image.shape[0] > 0,
              "Paired camera video frame malformed")
    capture.release()
    check(decoded == pair["frames"] == 16,
          "Paired camera video incomplete")
    result = {
        "schema": "robot-learning-lab.ape-camera-low-pixel-origin-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "segmentation_packets_checked": segmentation_packets,
        "paired_camera_video_frames_decoded": decoded,
        "outcome": "pass",
    }
    (v176p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
