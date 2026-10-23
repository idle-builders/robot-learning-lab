"""V50: fixture-clear rounded prop route in the complete owner continuation.

The prop goes around the island via a checked west aisle. Both paths have the
same checkpoint and staging motion, then one turns toward the ape's held owner
view. Smoothstep segments stop at waypoints; the prop is still scripted, not a
dynamic rigid body pushed by a simulated person.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import owner_turned_prop_holdout as v49


v38 = v49.v38
v41 = v49.v41
v46 = v49.v46
v47 = v49.v47
ROOT = v49.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v50-fixture-clear-prop"
WAYPOINTS = ((0., np.array([3., -1.65])),
             (1.2, np.array([1.8, -1.65])),
             (4.4, np.array([1.8, -5.15])))
END_S = 5.2
PATHS = {"outbound_control": np.array([2.2, -5.15]),
         "inbound_holdout": np.array([1.45, -4.75])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prop_at(time_s, checkpoint_s, end_xy):
    elapsed = np.clip(time_s-checkpoint_s, 0., END_S)
    points = (*WAYPOINTS, (END_S, end_xy))
    for (left_t, left), (right_t, right) in zip(points, points[1:]):
        if elapsed <= right_t:
            u = np.clip((elapsed-left_t)/(right_t-left_t), 0., 1.)
            eased = u*u*(3.-2.*u)
            return left+eased*(right-left)
    return end_xy.copy()


def main():
    source = v38.original()
    policy = json.loads(v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v38.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry = v38.geometry_at
    original_packet = v38.observed_packet
    results = []
    try:
        v38.geometry_at = v46.low_prop_geometry(original_geometry)
        v38.observed_packet = v46.plane_packet(original_packet, (.05, .30))
        for path_name, end_xy in PATHS.items():
            obstacle_fn = lambda t, end_xy=end_xy: prop_at(
                t, checkpoint_s, end_xy)
            for case in v47.CASES:
                history = []

                def retain_actor_packet(packet, time_s, stage):
                    if stage == "post_alarm_range_watch":
                        history.append({"time_s": float(time_s),
                                        "stage": stage, "packet": list(packet)})
                    return packet

                run = v38.run_case(
                    case, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v41.LocalizedFlankCamera(video),
                    flank_response=v38.FLANK_RESPONSE.get(case),
                    require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                    response_controller=(v41.JointHazardResponse()
                                         if case in v38.FLANK_RESPONSE else None),
                    pre_show_restage=v47.HistoryMeetingPose(history),
                    owner_fovy_deg=50., range_transform=retain_actor_packet,
                    output_root=OUT / path_name / case)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                steps = [row for row in rows if row["stage"] ==
                         "pre_show_restage_step"]
                first = steps[0]["actor_decision"] if steps else None
                results.append({"path": path_name, "case": case,
                                "run": run, "restage_steps": len(steps),
                                "choice": first["choice"] if first else None,
                                "novel_forward_returns": (
                                    first["novel_forward_returns"]
                                    if first else [])})
                print(path_name, case, results[-1]["choice"], run["outcome"],
                      run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
                      flush=True)
    finally:
        v38.geometry_at = original_geometry
        v38.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-fixture-clear-prop.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "v38_source_sha256": sha(v38.__file__),
              "v46_source_sha256": sha(v46.__file__),
              "v47_source_sha256": sha(v47.__file__),
              "v49_source_sha256": sha(v49.__file__),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "scan_heights_m": [.05, .30],
              "prop_geometry": {"radius_m": v46.PROP_RADIUS_M,
                                "half_height_m": v46.PROP_HALF_HEIGHT_M,
                                "center_z_m": v46.PROP_CENTER_Z_M},
              "waypoints": [{"after_checkpoint_s": t, "xy_m": xy.tolist()}
                            for t, xy in WAYPOINTS],
              "identical_path_through_s": round(checkpoint_s+WAYPOINTS[-1][0], 5),
              "end_after_checkpoint_s": END_S,
              "path_end_xy_m": {name: xy.tolist() for name, xy in PATHS.items()},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
