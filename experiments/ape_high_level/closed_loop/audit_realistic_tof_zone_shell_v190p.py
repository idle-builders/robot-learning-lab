"""Audit V189p 8x8 zone-center packets against their saved shell scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from run import ROOT, sha
import screen_realistic_tof_zone_shell_v189p as v189p


SOURCE = Path(__file__)
REPORT = v189p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    source = json.loads((v189p.v181p.OUT / "report.json").read_text())
    by_case = {row["source_case_id"]: row for row in source["rows"]}
    checks = packets = rays_checked = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v189p.__file__)) == report["source_sha256"] and
          sha(v189p.v181p.OUT / "report.json") ==
          report["v181_report_sha256"] and
          len(report["rows"]) == 8,
          "8x8 sensor screen source changed")
    summary = {key: {"cases": 0, "shell_cells": 0,
                     "range_hit_only_free": 0,
                     "range_max_ray_free": 0,
                     "return_camera_plus_range_max_free": 0,
                     "optimistic_camera_plus_range_max_free": 0}
               for key in report["summary"]}
    for case in report["rows"]:
        with np.load(ROOT / case["source_shell_packet"],
                     allow_pickle=False) as packet:
            shell = packet["shell_samples_xyz_m"].copy()
        check(sha(ROOT / case["source_shell_packet"]) ==
              case["source_shell_packet_sha256"] and
              len(shell) == case["shell_cells"] == 39,
              "Low shell source changed")
        camera_returned, camera_ideal = v189p.v186p.camera_maps(
            by_case[case["source_case_id"]])
        check(len(case["rows"]) == 12,
              "Per-case sensor candidate matrix incomplete")
        for row in case["rows"]:
            key = (f"{row['footprint']}/"
                   f"{int(row['noise_sigma_m']*1000)}mm/{row['mount']}")
            check(key in summary and row["footprint"] in v189p.FOOTPRINTS
                  and row["mount"] in v189p.MOUNTS and
                  row["noise_sigma_m"] in v189p.NOISE_PROBES_M,
                  "Unknown sensor candidate")
            path = ROOT / row["actor_ray_packet"]
            check(sha(path) == row["actor_ray_packet_sha256"],
                  "8x8 actor ray packet changed")
            packet = json.loads(path.read_text())
            rays = packet["rays"]
            check(packet["source_case_id"] == case["source_case_id"] and
                  packet["footprint"] == row["footprint"] and
                  packet["mount"] == row["mount"] and
                  packet["noise_sigma_m"] == row["noise_sigma_m"] and
                  packet["capture_time_s"] == v189p.TIME_S and
                  len(rays) == row["ray_counts"]["valid"] and
                  len(rays) <= 64 and
                  sum(ray["endpoint_hit"] for ray in rays) ==
                  row["ray_counts"]["environment_hit"] and
                  sum(not ray["endpoint_hit"] for ray in rays) ==
                  row["ray_counts"]["max_range"],
                  "8x8 packet metadata or ray counts changed")
            check(all(ray["capture_time_s"] == v189p.TIME_S and
                      0.04 <= ray["measured_range_m"] <=
                      (v189p.range_model.RANGE_MAX_M + .05
                       if ray["endpoint_hit"] else
                       v189p.range_model.RANGE_MAX_M) for ray in rays),
                  "8x8 ray time or range invalid")
            hits = DepthVoxelMap()
            v189p.range_model.add_packet(
                hits, [ray for ray in rays if ray["endpoint_hit"]],
                v189p.TIME_S)
            all_rays = DepthVoxelMap()
            v189p.range_model.add_packet(all_rays, rays, v189p.TIME_S)
            check(v189p.state(hits, shell) == row[
                "range_hit_only"] and
                v189p.state(all_rays, shell) == row[
                    "range_with_max_distance_rays"] and
                v189p.state(v189p.v181p.merged(
                    camera_returned, hits), shell) == row[
                    "return_cameras_plus_range_hit_only"] and
                v189p.state(v189p.v181p.merged(
                    camera_returned, all_rays), shell) == row[
                    "return_cameras_plus_range_max_rays"] and
                v189p.state(v189p.v181p.merged(
                    camera_ideal, all_rays), shell) == row[
                    "optimistic_cameras_plus_range_max_rays"],
                  "Saved 8x8 rays do not reproduce shell states")
            item = summary[key]
            item["cases"] += 1
            item["shell_cells"] += len(shell)
            item["range_hit_only_free"] += row[
                "range_hit_only"]["free"]
            item["range_max_ray_free"] += row[
                "range_with_max_distance_rays"]["free"]
            item["return_camera_plus_range_max_free"] += row[
                "return_cameras_plus_range_max_rays"]["free"]
            item["optimistic_camera_plus_range_max_free"] += row[
                "optimistic_cameras_plus_range_max_rays"]["free"]
            packets += 1
            rays_checked += len(rays)
    for key, item in summary.items():
        check(item == report["summary"][key],
              "Published 8x8 sensor summary changed")
    result = {
        "schema": "robot-learning-lab.ape-tof-zone-shell-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "source_tagged_packets_recomputed": packets,
        "actor_range_rays_checked": rays_checked,
        "outcome": "pass",
    }
    (v189p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
