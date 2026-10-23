"""Audit saved head-look depth and combine all feasible views with chest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from depth_voxel import DepthVoxelMap
from run import ROOT, sha
import screen_preshow_head_look_v184p as v184p


SOURCE = Path(__file__)
REPORT = v184p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    source = json.loads((v184p.v181p.OUT / "report.json").read_text())
    by_case = {row["source_case_id"]: row for row in source["rows"]}
    checks = packets = images = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v184p.__file__)) == report["source_sha256"] and
          sha(v184p.v181p.OUT / "report.json") ==
          report["v181_report_sha256"] and
          len(report["rows"]) == 8,
          "Head-look source changed")
    totals = {"return_only_free": 0,
              "optimistic_simulator_miss_free": 0,
              "optimistic_unknown": 0,
              "shell_cells": 0}
    rows = []
    for case in report["rows"]:
        origin = by_case[case["source_case_id"]]
        with np.load(ROOT / case["source_shell_packet"],
                     allow_pickle=False) as packet:
            shell = packet["shell_samples_xyz_m"].copy()
            estimated = packet["estimated_pose_xyyaw"].copy()
        check(sha(ROOT / case["source_shell_packet"]) ==
              case["source_shell_packet_sha256"] and
              len(shell) == case["shell_cells"] == 39,
              "Actor low shell changed")
        chest_returned, chest_ideal = v184p.map_saved_chest(origin)
        return_maps = [chest_returned]
        ideal_maps = [chest_ideal]
        check(tuple(row["head_yaw_rad"] for row in case["rows"]) ==
              v184p.HEAD_YAWS_RAD,
              "Head-yaw candidate matrix changed")
        for row in case["rows"]:
            path = ROOT / row["packet_path"]
            image_path = ROOT / row["rgb_path"]
            check(sha(path) == row["packet_sha256"] and
                  sha(image_path) == row["rgb_sha256"],
                  "Head RGB-D packet changed")
            with Image.open(image_path) as image:
                check(image.size == (v184p.W, v184p.H),
                      "Head RGB image malformed")
            images += 1
            with np.load(path, allow_pickle=False) as packet:
                depth = packet["filtered_depth_m"].copy()
                camera_origin = packet["camera_origin_xyz_m"].copy()
                rotation = packet["camera_rotation"].copy()
                fovy = float(packet["fovy_deg"])
                time_s = float(packet["capture_time_s"])
                yaw = float(packet["head_yaw_rad"])
            check(time_s == v184p.TIME_S and yaw == row["head_yaw_rad"],
                  "Head frame metadata changed")
            returned = DepthVoxelMap()
            returned.observe(depth, (camera_origin, rotation), time_s,
                             fovy_deg=fovy,
                             stride=v184p.v181p.DEPTH_STRIDE,
                             self_masked=True)
            ideal = DepthVoxelMap()
            ideal.free = returned.free.copy()
            ideal.occupied = returned.occupied.copy()
            ideal.frames = returned.frames
            ideal.last_time_s = returned.last_time_s
            miss = v184p.v183p.add_simulator_miss_rays(
                ideal, depth, camera_origin, rotation, fovy, time_s)
            check(miss["simulator_far_rays"] == row[
                "simulator_far_rays"] and
                v184p.score(returned, shell) == row[
                    "head_return_only"] and
                v184p.score(ideal, shell) == row[
                    "head_optimistic_miss_as_free"] and
                v184p.score(v184p.v181p.merged(
                    returned, chest_returned), shell) == row[
                    "both_return_only"] and
                v184p.score(v184p.v181p.merged(
                    ideal, chest_ideal), shell) == row[
                    "both_optimistic_miss_as_free"],
                  "Saved head look does not reproduce coverage")
            return_maps.append(returned)
            ideal_maps.append(ideal)
            packets += 1
        returned_union = v184p.v181p.merged(*return_maps)
        ideal_union = v184p.v181p.merged(*ideal_maps)
        all_returned = v184p.score(returned_union, shell)
        all_ideal = v184p.score(ideal_union, shell)
        unknown_body = [np.round(point - np.array((estimated[0],
                         estimated[1], 0.)), 5).tolist() for point in shell
                        if ideal_union.state(point, v184p.TIME_S) == "unknown"]
        rows.append({"source_case_id": case["source_case_id"],
                     "all_six_looks_plus_chest_return_only": all_returned,
                     "all_six_looks_plus_chest_optimistic": all_ideal,
                     "optimistic_unknown_samples_body_xyz_m": unknown_body})
        totals["return_only_free"] += all_returned["free"]
        totals["optimistic_simulator_miss_free"] += all_ideal["free"]
        totals["optimistic_unknown"] += all_ideal["unknown"]
        totals["shell_cells"] += len(shell)
    check(totals["shell_cells"] == 312,
          "Low shell denominator changed")
    result = {
        "schema": "robot-learning-lab.ape-preshow-head-look-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "saved_depth_packets_recomputed": packets,
        "rgb_images_decoded": images,
        "all_head_looks_plus_chest": totals,
        "rows": rows,
        "outcome": "pass",
    }
    (v184p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["all_head_looks_plus_chest"])
    print("audit checks", checks, "packets", packets)


if __name__ == "__main__":
    main()
