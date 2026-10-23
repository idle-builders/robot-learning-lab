"""V77: actor-chosen northward wait pose in the complete owner show episode.

This development selector was introduced after V75/V76 exposed the missing
direction. It consumes only timed pre-move range history, current pose and
the task-known owner station. Scripted prop/person routes remain evaluator
disturbances. The inherited owner loop executes and reports actual task,
near-body, image and geometry outcomes; the policy is not a safety proof.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_deferred_missing_pose_screen as v76


v75, v72, runner = v76.v75, v76.v72, v76.runner
v59, v65 = v72.v64.v60.v59, v75.v74.v65
ROOT = v76.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v77-northward-owner-turn"
MIN_NORTHWARD_M = .30
MIN_STATIC_PROXY_MARGIN_M = .10
MIN_APPROACH_TUBE_MARGIN_M = .0
VARIANTS = v75.VARIANTS


class NorthwardWaitPose(v59.ApproachAwareMeetingPose):
    """Add omitted directions and decline if none has a supported northward reserve."""

    def start(self, xy, packet, time_s):
        super().start(xy, packet, time_s)
        endpoints = v59.v51.v41.JointHazardResponse.endpoints(packet, xy)
        healthy = len([value for value in packet if value is not None]) >= 36
        candidates = []
        for name, direction in v76.DIRECTIONS.items():
            for distance in v76.DISTANCES_M:
                target = self.start_xy+distance*direction
                gaps = [float(np.min(np.linalg.norm(
                    endpoints-(self.start_xy+u*direction), axis=1)))
                    for u in np.linspace(0., distance, 15)]
                owner_range = float(np.linalg.norm(
                    v59.v56.OWNER_SHOW_STATION_XY_M-target))
                static_margin = gaps[-1]-v59.BODY_PROXY_RADIUS_M
                approach_margin = min((v59.time_indexed_surface_margin(
                    self.start_xy, target, distance,
                    np.asarray(surface["surface_xy_m"]))
                    for surface in self.dynamic_surfaces),
                    default=-math.inf)
                northward = float(target[1]-self.start_xy[1])
                eligible = bool(
                    healthy and self.dynamic_surfaces and
                    min(gaps) >= v59.MIN_ENDPOINT_GAP_M and
                    gaps[-1] >= v59.MIN_ENDPOINT_GAP_M and
                    owner_range <= v59.MAX_OWNER_STATION_RANGE_M and
                    northward >= MIN_NORTHWARD_M and
                    static_margin >= MIN_STATIC_PROXY_MARGIN_M and
                    approach_margin > MIN_APPROACH_TUBE_MARGIN_M)
                row = {"name": name, "distance_m": distance,
                       "target_xy_m": target.tolist(),
                       "minimum_path_endpoint_gap_m": min(gaps),
                       "held_endpoint_reserve_m": gaps[-1],
                       "owner_station_range_m": owner_range,
                       "northward_m": northward,
                       "actor_static_margin_m": static_margin,
                       "actor_approach_margin_m": approach_margin,
                       "eligible": eligible}
                candidates.append(row)
                self.scores.append(row)
        ranked = [row for row in candidates if row["eligible"]]
        chosen = max(ranked, key=lambda row: (
            min(row["actor_static_margin_m"],
                row["actor_approach_margin_m"]),
            -row["owner_station_range_m"])) if ranked else None
        self.choice = (f"{chosen['name']}_{int(chosen['distance_m']*100)}cm"
                       if chosen else None)
        self.target_m = chosen["distance_m"] if chosen else 0.
        self.direction = (v76.DIRECTIONS[chosen["name"]]
                          if chosen else np.zeros(2))
        self.choice_score = (min(chosen["actor_static_margin_m"],
                                 chosen["actor_approach_margin_m"])
                             if chosen else None)
        self.no_supported_pose = chosen is None

    def step(self, time_s, xy, packet):
        direction, distance, decision = super().step(time_s, xy, packet)
        decision["northward_wait_policy"] = {
            "minimum_northward_m": MIN_NORTHWARD_M,
            "minimum_static_proxy_margin_m": MIN_STATIC_PROXY_MARGIN_M,
            "minimum_approach_tube_margin_m":
                MIN_APPROACH_TUBE_MARGIN_M,
            "no_supported_pose": self.no_supported_pose}
        return direction, distance, decision


def run_suite(*, output_root=OUT, task_risk_monitor=None,
              report_schema="robot-learning-lab.ape-northward-owner-turn.v1",
              scope_suffix=""):
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    vision = runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(runner.owner.TEACH_CROP).convert("RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry = runner.geometry_at
    original_packet = runner.observed_packet
    results = []
    try:
        runner.geometry_at = v72.v64.v51.v46.low_prop_geometry(
            original_geometry)
        runner.observed_packet = v72.v64.v51.v46.plane_packet(
            original_packet, (.05, .30))
        for variant, path, finish_s in VARIANTS:
            history = []

            def retain(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    history.append({"time_s": float(time_s),
                                    "packet": list(packet)})
                return packet

            endpoint = v72.v64.PATHS[path]
            obstacle = lambda t, endpoint=endpoint, finish_s=finish_s: \
                v65.prop_at(t, checkpoint_s, endpoint, finish_s)
            run = runner.run_case(
                v72.v64.v54.CASE, source, policy, scenario,
                vision, exemplar,
                flank_probe_factory=lambda video:
                    v72.v64.v51.v41.LocalizedFlankCamera(video),
                require_camera_pose=True,
                route_obstacle_fn=obstacle,
                pre_show_restage=NorthwardWaitPose(history),
                owner_fovy_deg=50.,
                range_transform=retain,
                near_body_controller=v72.v64.v52.SurfaceTrackEscape(),
                task_risk_monitor=task_risk_monitor,
                output_root=output_root / variant)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            step = next((row for row in rows if row["stage"] ==
                         "pre_show_restage_step"), None)
            decision = next((row for row in rows if row["stage"] ==
                             "pre_show_restage_decision"), None)
            results.append({"variant": variant, "path": path,
                            "finish_after_checkpoint_s": finish_s,
                            "run": run,
                            "actor_choice": (step["actor_decision"]["choice"]
                                             if step else None),
                            "actor_pose_decision": (
                                decision["actor_decision"]
                                if decision else None),
                            "restage_steps": len([row for row in rows
                                if row["stage"] == "pre_show_restage_step"]),
                            "fresh_item_captures": len([row for row in rows
                                if row["stage"] == "fresh_new_show_capture"])})
            print(variant, results[-1]["actor_choice"],
                  run["outcome"], run["minimum_prop_gap_m"],
                  run["minimum_proxy_gap_m"], flush=True)
    finally:
        runner.geometry_at = original_geometry
        runner.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"schema": report_schema,
              "source_sha256": v72.v52.sha(__file__),
              "v76_report_sha256": v72.v52.sha(v76.OUT / "report.json"),
              "v59_source_sha256": v72.v52.sha(v59.__file__),
              "runner_source_sha256": v72.v52.sha(runner.__file__),
              "scene_sha256": v72.v52.sha(
                  runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "actor_rule": {"minimum_northward_m": MIN_NORTHWARD_M,
                             "minimum_static_proxy_margin_m":
                                 MIN_STATIC_PROXY_MARGIN_M,
                             "minimum_approach_tube_margin_m":
                                 MIN_APPROACH_TUBE_MARGIN_M,
                             "candidate_directions": {
                                 name: value.tolist() for name, value in
                                 v76.DIRECTIONS.items()},
                             "candidate_distances_m": list(v76.DISTANCES_M)},
              "results": results,
              "scope": "Development actor-only northward wait selector in complete V59 owner task with fused ideal low/mid range, RGB-D owner show, flank camera and independent surface-track response. Selected after V75/V76 evaluator screen; six routes are not held out. Historical kinematic proxy, not selected-body or physical qualification. " + scope_suffix}
    target = output_root / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target)


def main():
    run_suite()


if __name__ == "__main__":
    main()
