"""V20: a genuinely new near-field static surface in RoboCasa layout 3.

The added side stool is positioned by fixture clearance before any guard run.
The actor still receives only the initial scan map, timed low rays and ideal
odometry; the static stool is not identified by a scene label.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import changed_layout_trial as layout3
import command_route_trial as command
import dense_low_trial as dense
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = kitchen.OUT / "v20-near-clutter"
NEAR_STOOL_XY_M = (3.10, -2.32)
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy


def make_scene() -> str:
    root = ET.fromstring(layout3.make_scene())
    world = root.find("worldbody")
    body = ET.SubElement(world, "body", name="near_side_stool",
                         pos=f"{NEAR_STOOL_XY_M[0]} {NEAR_STOOL_XY_M[1]} 0")
    ET.SubElement(body, "geom", name="near_stool_column", type="cylinder",
                  pos="0 0 .21", size=".105 .21", group="0",
                  rgba=".37 .29 .22 1", contype="0", conaffinity="0")
    ET.SubElement(body, "geom", name="near_stool_seat", type="cylinder",
                  pos="0 0 .44", size=".18 .025", group="0",
                  rgba=".45 .33 .22 1", contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xml = OUT / "scene.xml"
    xml.write_text(make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {row["world"]: row["command_to_observed_stop_s"]
              for row in stop["rows"] if row["case"] == "forward-stop-late"}
    trial.hazard_xy = layout3.hazard_xy
    guard.ORIGINAL_READ = dense.dense_read
    guard.low_ray_points = dense.dense_points
    guard.FullScanGuard = command.CommandRouteGuard
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            background = dense.initial_map(model, saved)
            map_path = OUT / "background_xy.npy"
            np.save(map_path, background)
            for mode in layout3.MODES:
                trial.OUT = OUT / mode
                for world in range(4):
                    for case in layout3.CASES:
                        video = world == 0 and (case in ("slow_cross_held",
                                                         "slow_right_held") or
                                                mode == "mapped_low_guard" and case in
                                                ("fast_clear_held", "fast_right_held"))
                        result = trial.run_episode(model, saved, background,
                                                   world, case, mode,
                                                   delays[world], video=video)
                        results.append(result)
                        print(mode, world, case, result["stop_command_s"],
                              result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {
        "schema": "robot-learning-lab.ape-robocasa-near-clutter-low-guard.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "V19 upstream RoboCasa layout3/style1, same native trajectory, six crossing scripts, guard and 73-ray front low 15Hz/1.2m sensor. Add one low static stool at (3.10,-2.32)m after pre-run fixture screen; rebuild only initial-view static map. Paired no-sensor control. No guard tuning or new physical rollout.",
        "room": {"layout_id": 3, "style_id": 1,
                 "scene_sha256": trial.sha(xml),
                 "v19_scene_sha256": trial.sha(layout3.OUT / "scene.xml"),
                 "near_stool_xy_m": list(NEAR_STOOL_XY_M)},
        "sensor": {"rays": 73, "period_s": 1/15,
                   "yaw_range_deg": [-70., 70.], "max_range_m": dense.MAX_M,
                   "origin_local_m": [.08, 0., -.15],
                   "initial_map_points": len(background),
                   "initial_map_sha256": trial.sha(map_path),
                   "ideal_odom": True},
        "source_sha256": trial.sha(Path(__file__)),
        "v19_source_sha256": trial.sha(Path(layout3.__file__)),
        "dense_source_sha256": trial.sha(Path(dense.__file__)),
        "episode_runner_sha256": trial.sha(Path(trial.__file__)),
        "command_guard_sha256": trial.sha(Path(command.__file__)),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "body_sha256": trial.sha(kitchen.BODY),
        "v19_report_sha256": trial.sha(layout3.OUT / "report.json"),
        "mujoco_version": mujoco.__version__,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
