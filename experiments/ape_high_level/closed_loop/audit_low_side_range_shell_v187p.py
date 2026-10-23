"""Rebuild V186p shell coverage from saved source-tagged range ray packets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from run import ROOT, sha
import screen_low_side_range_shell_v186p as v186p


SOURCE = Path(__file__)
REPORT = v186p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    source = json.loads((v186p.v181p.OUT / "report.json").read_text())
    by_case = {row["source_case_id"]: row for row in source["rows"]}
    checks = rays_checked = packets = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v186p.__file__)) == report["source_sha256"] and
          sha(v186p.v181p.OUT / "report.json") ==
          report["v181_report_sha256"] and
          len(report["rows"]) == 8,
          "Sensor comparison source changed")
    summary = {name: {"cases": 0, "shell_cells": 0,
                      "range_hit_only_free": 0,
                      "range_max_ray_free": 0,
                      "return_camera_plus_range_max_free": 0,
                      "optimistic_camera_plus_range_max_free": 0}
               for name in v186p.MOUNTS}
    for case in report["rows"]:
        source_row = by_case[case["source_case_id"]]
        shell_path = ROOT / case["source_shell_packet"]
        check(sha(shell_path) == case["source_shell_packet_sha256"],
              "Saved body shell changed")
        with np.load(shell_path, allow_pickle=False) as packet:
            shell = packet["shell_samples_xyz_m"].copy()
        check(len(shell) == case["shell_cells"] == 39,
              "Low shell denominator changed")
        camera_returned, camera_ideal = v186p.camera_maps(source_row)
        check(v186p.state(camera_returned, shell) == case[
            "camera_return_only"] and
            v186p.state(camera_ideal, shell) == case[
                "camera_optimistic_miss_as_free"],
            "Camera baseline changed")
        check({row["mount"] for row in case["mount_rows"]} ==
              set(v186p.MOUNTS),
              "Candidate mount matrix incomplete")
        for row in case["mount_rows"]:
            path = ROOT / row["actor_ray_packet"]
            check(sha(path) == row["actor_ray_packet_sha256"],
                  "Actor range packet changed")
            packet = json.loads(path.read_text())
            rays = packet["rays"]
            check(packet["source_case_id"] == case["source_case_id"] and
                  packet["mount"] == row["mount"] and
                  packet["capture_time_s"] == v186p.TIME_S and
                  len(rays) == row["ray_counts"]["valid"] and
                  sum(ray["endpoint_hit"] for ray in rays) ==
                  row["ray_counts"]["environment_hit"] and
                  sum(not ray["endpoint_hit"] for ray in rays) ==
                  row["ray_counts"]["max_range"],
                  "Range ray source or counts changed")
            check(all(ray["capture_time_s"] == v186p.TIME_S and
                      0.04 <= ray["measured_range_m"] <=
                      v186p.range_model.RANGE_MAX_M for ray in rays),
                  "Range timestamp or distance invalid")
            hit_only = DepthVoxelMap()
            v186p.range_model.add_packet(
                hit_only, [ray for ray in rays if ray["endpoint_hit"]],
                v186p.TIME_S)
            all_returns = DepthVoxelMap()
            v186p.range_model.add_packet(all_returns, rays, v186p.TIME_S)
            check(v186p.state(hit_only, shell) == row[
                "range_hit_only"] and
                v186p.state(all_returns, shell) == row[
                    "range_with_max_distance_rays"] and
                v186p.state(v186p.v181p.merged(
                    camera_returned, hit_only), shell) == row[
                    "cameras_return_only_plus_range_hit_only"] and
                v186p.state(v186p.v181p.merged(
                    camera_returned, all_returns), shell) == row[
                    "cameras_return_only_plus_range_max_rays"] and
                v186p.state(v186p.v181p.merged(
                    camera_ideal, all_returns), shell) == row[
                    "optimistic_cameras_plus_range_max_rays"],
                  "Saved rays do not reproduce shell coverage")
            rays_checked += len(rays)
            packets += 1
            item = summary[row["mount"]]
            item["cases"] += 1
            item["shell_cells"] += len(shell)
            item["range_hit_only_free"] += row[
                "range_hit_only"]["free"]
            item["range_max_ray_free"] += row[
                "range_with_max_distance_rays"]["free"]
            item["return_camera_plus_range_max_free"] += row[
                "cameras_return_only_plus_range_max_rays"]["free"]
            item["optimistic_camera_plus_range_max_free"] += row[
                "optimistic_cameras_plus_range_max_rays"]["free"]
    for name, item in summary.items():
        published = report["summary"][name]
        check(item == {key: published[key] for key in item},
              "Mount summary differs from saved packets")
    result = {
        "schema": "robot-learning-lab.ape-low-side-shell-range-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "source_tagged_packets_recomputed": packets,
        "actor_range_rays_checked": rays_checked,
        "outcome": "pass",
    }
    (v186p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
