"""Frozen V26 challenge: V24 guard on box-shaped furniture/new crossings.

These room/person definitions were fixed before inspecting V25's failed
development result. V24 is the simpler frozen actor under challenge.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import dense_low_trial as dense
import kitchen_native as kitchen
import kitchen_trial as trial
import near_clutter_trial as clutter
import online_cluster_guard as online
import run as guard
import plan_bounded_trial as plan


OUT = kitchen.OUT / "v26-plan-bounded-heldout"
CASES = ("fast_clear_new", "slow_cross_new", "pause_path_new",
         "fast_right_new", "slow_right_new", "pause_right_new")
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy
ORIGINAL_CLASS = guard.FullScanGuard
ORIGINAL_UNKNOWN = guard.UNKNOWN_LATERAL_M


def make_scene() -> str:
    root = ET.fromstring(clutter.make_scene())
    for name, size in (("near_stool_column", ".105 .105 .21"),
                       ("near_stool_seat", ".18 .18 .025")):
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise RuntimeError(f"Missing {name}")
        geom.set("type", "box")
        geom.set("size", size)
    return ET.tostring(root, encoding="unicode")


def hazard_xy(case: str, t: float) -> tuple[float, float]:
    if case == "fast_clear_new":
        y = -1.05-.78*min(t, 4.)
    elif case == "slow_cross_new":
        y = -1.05-.18*min(t, 4.)
    elif case == "pause_path_new":
        y = -1.05-.45*min(t, 1.7)
    elif case == "fast_right_new":
        y = min(-1.00, -2.55+.78*t)
    elif case == "slow_right_new":
        y = -2.55+.18*min(t, 4.)
    elif case == "pause_right_new":
        y = -2.55+.45*min(t, 1.7)
    else:
        raise ValueError(case)
    return 3.55, y


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = OUT / "scene.xml"
    scene.write_text(make_scene())
    model = mujoco.MjModel.from_xml_path(str(scene))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {row["world"]: row["command_to_observed_stop_s"]
              for row in stop["rows"] if row["case"] == "forward-stop-late"}
    guard.ORIGINAL_READ = dense.dense_read
    guard.low_ray_points = dense.dense_points
    trial.hazard_xy = hazard_xy
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            for mode in ("none", "mapped_low_guard"):
                trial.OUT = OUT / ("no_sensor" if mode == "none" else
                                   "online_plan_guard")
                guard.FullScanGuard = (ORIGINAL_CLASS if mode == "none" else
                                      plan.PlanBoundedGuard)
                guard.UNKNOWN_LATERAL_M = guard.RISK_RADIUS_M
                for world in range(4):
                    for case in CASES:
                        video = world == 0 and case in (
                            "fast_clear_new", "slow_cross_new", "fast_right_new",
                            "slow_right_new")
                        row = trial.run_episode(model, saved,
                                                online.NO_MAP_SENTINEL_XY_M,
                                                world, case, mode,
                                                delays[world], video=video)
                        row["configuration"] = ("no_sensor" if mode == "none" else
                                                 "online_plan_guard")
                        results.append(row)
                        print(row["configuration"], world, case,
                              row["stop_command_s"],
                              row["min_modeled_body_person_gap_m"], flush=True)
    finally:
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = ORIGINAL_CLASS
        guard.UNKNOWN_LATERAL_M = ORIGINAL_UNKNOWN
        trial.hazard_xy = ORIGINAL_HAZARD
    report = {
        "schema": "robot-learning-lab.ape-robocasa-plan-bounded-holdout.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "V24 bounded-forecast actor frozen before this changed-scene test: replace the added near stool's cylindrical surfaces by boxes, change left/right fast crossing from .72 to .78m/s, slow from .22 to .18m/s, and pause from .50m/s for1.5s to .45m/s for1.7s. Scene and scripts were defined before inspecting V25's failed shape-fit diagnostic. Same 73-ray 15Hz ideal low fan, ideal pose, historical native trace and stop tail; no hazard-free map. Matched no-sensor controls and full-body sampled gap evaluator.",
        "scene_sha256": trial.sha(scene),
        "source_sha256": trial.sha(Path(__file__)),
        "actor_source_sha256": trial.sha(Path(plan.__file__)),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "body_sha256": trial.sha(kitchen.BODY),
        "v24_report_sha256": trial.sha(plan.OUT / "report.json"),
        "mujoco_version": mujoco.__version__,
        "cases": {case: list(hazard_xy(case, 0.)) for case in CASES},
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run()
