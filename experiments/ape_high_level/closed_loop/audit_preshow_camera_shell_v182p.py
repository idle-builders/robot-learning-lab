"""Rebuild V181p camera-ray maps and low-shell states from saved actor packets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from depth_voxel import DepthVoxelMap
from run import ROOT, sha
import screen_preshow_camera_shell_v181p as v181p


SOURCE = Path(__file__)
REPORT = v181p.OUT / "report.json"


def counts(voxel_map: DepthVoxelMap, samples: np.ndarray,
           time_s: float) -> dict:
    states = [voxel_map.state(point, time_s) for point in samples]
    return {name: states.count(name)
            for name in ("free", "occupied", "unknown")}


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = packets = images = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v181p.__file__)) == report["source_sha256"],
          "Coverage source changed")
    check(sha(v181p.SOURCE.with_name("depth_voxel.py")) ==
          report["depth_voxel_source_sha256"],
          "Ray map source changed")
    check(sha(v181p.SOURCE.with_name("self_depth_mask.py")) ==
          report["self_mask_source_sha256"],
          "Self mask source changed")
    check(sha(v181p.v167p.OUT / "report.json") ==
          report["v167_report_sha256"] and
          len(report["rows"]) == 8,
          "Pre-show comparison source changed")
    totals = {camera: {"free": 0, "occupied": 0, "unknown": 0,
                       "all_observed_free": 0, "cases": 0,
                       "total_shell_cells": 0}
              for camera in (*v181p.CAMERAS, "both_cameras")}
    for row in report["rows"]:
        shell_path = ROOT / row["shell_packet"]
        check(sha(shell_path) == row["shell_packet_sha256"],
              "Actor shell packet changed")
        with np.load(shell_path, allow_pickle=False) as source:
            samples = source["shell_samples_xyz_m"].copy()
            pose = source["estimated_pose_xyyaw"].copy()
        check(len(samples) == row["low_shell_samples"] == 39 and
              pose[0] == row["station_x_m"],
              "Actor shell geometry changed")
        check(sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"],
              "Staging joint/pose packet changed")
        maps = {camera: DepthVoxelMap() for camera in v181p.CAMERAS}
        for camera in v181p.CAMERAS:
            source_rows = [item for item in row["pre_show_camera_packets"]
                           if item["camera"] == camera]
            check([r["capture_time_s"] for r in source_rows] ==
                  list(v181p.v167p.SCAN_TIMES_S),
                  "Camera pre-show timeline changed")
            for index, saved in enumerate(source_rows):
                path = ROOT / saved["packet_path"]
                check(sha(path) == saved["packet_sha256"],
                      "Self-masked depth packet changed")
                with np.load(path, allow_pickle=False) as packet:
                    depth = packet["filtered_depth_m"]
                    origin = packet["camera_origin_xyz_m"]
                    rotation = packet["camera_rotation"]
                    fovy = float(packet["fovy_deg"])
                    time_s = float(packet["capture_time_s"])
                check(time_s == saved["capture_time_s"] and
                      depth.shape == (v181p.H, v181p.W),
                      "Depth frame metadata changed")
                stats = maps[camera].observe(
                    depth, (origin, rotation), time_s, fovy_deg=fovy,
                    stride=v181p.DEPTH_STRIDE, max_depth_m=1.4,
                    self_masked=True)
                check(stats == saved["map"],
                      "Ray map counts differ from saved depth")
                state = counts(maps[camera], samples, time_s)
                check(state == {key: saved["low_shell"][key]
                                for key in state} and
                      saved["low_shell"]["samples"] == len(samples),
                      "Low-shell states differ from saved depth")
                if index == len(source_rows)-1:
                    image_path = ROOT / saved["last_image_path"]
                    check(sha(image_path) == saved["last_image_sha256"],
                          "Last pre-show RGB image changed")
                    with Image.open(image_path) as image:
                        check(image.size == (v181p.W, v181p.H),
                              "Last pre-show RGB image malformed")
                    images += 1
                packets += 1
            final = counts(maps[camera], samples,
                           v181p.v167p.SCAN_TIMES_S[-1])
            check(final == {key: row["final_low_shell"][camera][key]
                            for key in final},
                  "Final camera shell result changed")
        union = DepthVoxelMap()
        for source in maps.values():
            for name in ("free", "occupied"):
                target = getattr(union, name)
                for cell, stamp in getattr(source, name).items():
                    target[cell] = max(target.get(cell, -math.inf), stamp)
        joint = counts(union, samples, v181p.v167p.SCAN_TIMES_S[-1])
        check(joint == {key: row["final_low_shell"][
            "both_cameras"][key] for key in joint},
            "Final combined camera shell result changed")
        for camera in totals:
            result = row["final_low_shell"][camera]
            totals[camera]["cases"] += 1
            totals[camera]["total_shell_cells"] += result["samples"]
            totals[camera]["all_observed_free"] += result[
                "all_observed_free"]
            for name in ("free", "occupied", "unknown"):
                totals[camera][name] += result[name]
    for camera, total in totals.items():
        published = report["summary"][camera]
        check(total == {"cases": published["cases"],
                        "total_shell_cells": published["total_shell_cells"],
                        "all_observed_free": published["all_observed_free"],
                        "free": published["free_shell_cells"],
                        "occupied": published["occupied_shell_cells"],
                        "unknown": published["unknown_shell_cells"]},
              "Published shell summary changed")
    result = {
        "schema": "robot-learning-lab.ape-preshow-camera-shell-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "self_masked_depth_packets_recomputed": packets,
        "last_pre_show_rgb_images_decoded": images,
        "outcome": "pass",
    }
    (v181p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
