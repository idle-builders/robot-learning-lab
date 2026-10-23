"""Render the measured-versus-optimistic low-shell coverage comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depth_voxel import DepthVoxelMap
from run import ROOT, sha
import screen_low_side_range_shell_v186p as v186p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-low-side-shell-range"
IMAGE = OUT / "pre-show-low-shell-coverage.png"


def main() -> None:
    report_file = v186p.OUT / "report.json"
    report = json.loads(report_file.read_text())
    camera_source = json.loads((v186p.v181p.OUT / "report.json").read_text())
    case = next(row for row in report["rows"]
                if row["source_case_id"] ==
                "hazard-fast_wide_turn-seed170-x0.30")
    source = next(row for row in camera_source["rows"]
                  if row["source_case_id"] == case["source_case_id"])
    with np.load(ROOT / case["source_shell_packet"],
                 allow_pickle=False) as packet:
        shell = packet["shell_samples_xyz_m"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    camera_returned, camera_ideal = v186p.camera_maps(source)
    maps = [("Camera returns only", camera_returned, None),
            ("Camera + ideal empty rays", camera_ideal, None)]
    for name, label, marker_xy in (
        ("near_left_outward", "Near left range + ideal camera", (0., .095)),
        ("far_left_inward_bound", "Floating inward view + ideal camera",
         (0., .18)),
    ):
        row = next(r for r in case["mount_rows"] if r["mount"] == name)
        packet = json.loads((ROOT / row["actor_ray_packet"]).read_text())
        range_map = DepthVoxelMap()
        v186p.range_model.add_packet(range_map, packet["rays"],
                                     v186p.TIME_S)
        maps.append((label, v186p.v181p.merged(camera_ideal, range_map),
                     marker_xy))
    expected_free = (1, 34, 38, 39)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), sharex=True, sharey=True)
    for axis, (label, voxel_map, mount), expected in zip(
            axes.flat, maps, expected_free):
        state = [voxel_map.state(p, v186p.TIME_S) for p in shell]
        if state.count("free") != expected:
            raise AssertionError("Coverage graphic no longer matches report")
        for status, color in (("unknown", "#a64242"),
                              ("occupied", "#e59b35"),
                              ("free", "#24845e")):
            selected = np.array([p for p, s in zip(shell, state)
                                 if s == status]).reshape(-1, 3)
            if len(selected):
                relative = selected[:, :2]-estimate[:2]
                axis.scatter(relative[:, 0]*100, relative[:, 1]*100,
                             s=50+300*selected[:, 2], c=color,
                             edgecolors="white", linewidths=.4,
                             label=status, alpha=.86)
        if mount is not None:
            axis.scatter(mount[0]*100, mount[1]*100,
                         marker="X", s=130, c="#315caa",
                         edgecolors="white", linewidths=.7)
        axis.set_title(f"{label}\n{state.count('free')}/39 free")
        axis.axhline(0, color="#dddddd", linewidth=.6)
        axis.axvline(0, color="#dddddd", linewidth=.6)
        axis.set_xlim(-9, 14)
        axis.set_ylim(-1, 21)
        axis.grid(alpha=.18)
        axis.set_aspect("equal", adjustable="box")
    axes[1, 0].set_xlabel("Body-relative forward x (cm)")
    axes[1, 1].set_xlabel("Body-relative forward x (cm)")
    axes[0, 0].set_ylabel("Body-relative left y (cm)")
    axes[1, 0].set_ylabel("Body-relative left y (cm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .075))
    fig.suptitle("Front-left low body shell before inviting the owner\n"
                 "39 actor-geometry cells, same held Direct Frame scene",
                 fontsize=13)
    fig.text(.5, .025,
             "Ideal empty rays and range max-distance rays are upper bounds; "
             "a real invalid depth is unknown. Blue X marks virtual range origin.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .92))
    fig.savefig(IMAGE, dpi=170, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "schema": "robot-learning-lab.ape-preshow-shell-figure.v1",
        "source_sha256": sha(SOURCE),
        "v186_report_sha256": sha(report_file),
        "source_case_id": case["source_case_id"],
        "image_path": str(IMAGE.relative_to(ROOT)),
        "image_sha256": sha(IMAGE),
        "free_cells_by_panel": dict(zip(
            ("camera_returns", "camera_ideal_empty",
             "camera_ideal_plus_near_range",
             "camera_ideal_plus_floating_range"), expected_free)),
    }
    (OUT / "figure-report.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
