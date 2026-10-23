"""V201p: independently replay V200 actor lidar packets and shell states."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from compare_standoff_range_layouts_v165p import front_left_roi
from depth_voxel import DepthVoxelMap
from run import ROOT, sha
import screen_planar_lidar_owner_v200p as v200p
import screen_preshow_camera_shell_v181p as v181p
from screen_standoff_front_cue_v149p import front_left_stat


SOURCE = Path(__file__)
OUT = v200p.OUT


def shell_by_seed() -> dict[int, np.ndarray]:
    report = json.loads((v181p.OUT / "report.json").read_text())
    shells = {}
    for row in report["rows"]:
        if row["seed"] not in (170, 172) or row["station_x_m"] != -.3:
            continue
        path = ROOT / row["shell_packet"]
        assert sha(path) == row["shell_packet_sha256"]
        with np.load(path, allow_pickle=False) as packet:
            points = packet["shell_samples_xyz_m"].copy()
        if row["seed"] in shells:
            assert np.allclose(shells[row["seed"]], points)
        shells[row["seed"]] = points
    assert set(shells) == {170, 172}
    return shells


def replay(row: dict, shell: np.ndarray, device: dict) -> int:
    baseline = []
    old_count, old_centroid = 0, None
    first = None
    preshow = DepthVoxelMap()
    for packet_row in row["packets"]:
        path = ROOT / packet_row["actor_packet"]
        assert sha(path) == packet_row["actor_packet_sha256"]
        with np.load(path, allow_pickle=False) as packet:
            ranges = packet["ranges_m"].copy()
            status = packet["status"].copy()
            angles = packet["azimuth_rad"].copy()
            origin = packet["actor_origin_m"].copy()
            pose = packet["estimated_pose_xyyaw"].copy()
            t = float(packet["capture_time_s"])
        assert len(ranges) == len(status) == len(angles)
        assert all(int(np.count_nonzero(status == code)) ==
                   packet_row["status_counts"][name]
                   for code, name in v200p.STATUSES.items())
        valid = status == 0
        assert np.all(np.isnan(ranges[~valid]))
        assert np.all((ranges[valid] >= device["minimum_m"]) &
                      (ranges[valid] <= device["maximum_m"]))
        directions = np.stack((np.cos(angles+pose[2]),
                               np.sin(angles+pose[2]),
                               np.zeros_like(angles)), axis=1)
        points = origin + ranges[valid, None]*directions[valid]
        selected = points[front_left_roi(points, pose)]
        assert len(selected) == packet_row["front_left_return_count"]
        _, p10 = front_left_stat(points, pose)
        assert p10 == packet_row["front_left_p10_x_m"]
        if t <= v200p.REFERENCE_END_S:
            baseline.append(selected)
            for point in points:
                direction = point-origin
                distance = float(np.linalg.norm(direction))
                if distance <= .06:
                    continue
                for along in np.arange(.04, distance-.06, .04):
                    preshow.free[preshow.cell(
                        origin+along*direction/distance)] = t
            preshow.last_time_s = t
        reference = np.concatenate(baseline) if baseline else np.empty((0, 3))
        novel = np.empty((0, 3))
        if t > v200p.REFERENCE_END_S and len(reference) and len(selected):
            distances = np.linalg.norm(
                selected[:, None, :]-reference[None, :, :], axis=2)
            novel = selected[np.min(distances, axis=1) >
                             v200p.NOVEL_DISTANCE_M]
        centroid = np.mean(novel, axis=0) if len(novel) else None
        shift = (float(np.linalg.norm(centroid-old_centroid))
                 if centroid is not None and old_centroid is not None else None)
        cue = (len(novel) >= v200p.MIN_NOVEL_POINTS and
               (old_count < v200p.MIN_NOVEL_POINTS or
                (shift is not None and shift >= .012)))
        assert len(novel) == packet_row["new_low_returns"]
        assert cue == packet_row["new_surface_cue"]
        if cue and first is None:
            first = t
        old_count, old_centroid = len(novel), centroid
    assert first == row["first_new_low_surface_cue_s"]
    coverage = v181p.shell_state(preshow, shell, v200p.REFERENCE_END_S)
    assert coverage == row["preshow_shell"]
    return len(row["packets"])


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    assert report["source_sha256"] == sha(Path(v200p.__file__))
    assert report["current_design_status"] == "rejected_no_selected_mechanical_design"
    shells = shell_by_seed()
    assert len(report["rows"]) == 48
    checked = 0
    for row in report["rows"]:
        checked += replay(row, shells[row["seed"]],
                          report["devices"][row["device"]])
        assert row["preshow_shell"]["samples"] == 39
        assert row["preshow_shell"]["free"] <= 10
    moving_low = [r for r in report["rows"] if r["motion"] == "moving"
                  and r["mount"].startswith("low_")]
    stationary = [r for r in report["rows"] if r["motion"] == "stationary"]
    mid = [r for r in report["rows"] if r["mount"] == "mid_forward"]
    assert len(moving_low) == 16
    assert all(r["first_new_low_surface_cue_s"] is not None
               for r in moving_low)
    assert all(r["first_new_low_surface_cue_s"] is None for r in stationary+mid)
    a2_low = [r for r in moving_low if r["device"] == "a2_spec_proxy"]
    assert all(r["packets"][-1]["status_counts"]["near_invalid"] > 0
               for r in a2_low)
    result = {"schema": "robot-learning-lab.ape-planar-lidar-owner-audit.v1",
              "source_sha256": sha(SOURCE),
              "report_sha256": sha(OUT / "report.json"),
              "rows": len(report["rows"]),
              "packet_hashes_and_replays": checked,
              "low_moving_timely_cues": len(moving_low),
              "stationary_false_cues": sum(
                  r["first_new_low_surface_cue_s"] is not None
                  for r in stationary),
              "mid_height_cues": sum(
                  r["first_new_low_surface_cue_s"] is not None for r in mid),
              "maximum_preshow_shell_free_cells": max(
                  r["preshow_shell"]["free"] for r in report["rows"])}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
