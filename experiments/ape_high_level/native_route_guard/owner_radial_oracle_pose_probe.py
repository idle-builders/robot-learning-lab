"""V82: full owner episode at V81's evaluator-selected radial wait pose.

This is a task/visibility probe, explicitly not an actor-selected planner.
Future prop scripts selected the pose in V81; the actor in these episodes
still receives only its timed range and RGB-D packets while executing it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_radial_wait_feasibility as v81
import owner_stationary_dynamic_risk_turn as v78


v77 = v81.v77
ROOT = v81.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v82-radial-pose-owner-probe"
CASES = ("outward_normal", "near_northwest")
ANGLE_DEG = 112.5
DISTANCE_M = .85
DIRECTION = np.asarray([
    math.cos(math.radians(ANGLE_DEG)),
    math.sin(math.radians(ANGLE_DEG))])


class OracleRadialPose(v77.v59.ApproachAwareMeetingPose):
    max_ticks = 36  # 85 cm at 2.5 cm/100 ms requires 34 fresh-scan steps.

    def start(self, xy, packet, time_s):
        super().start(xy, packet, time_s)
        self.choice = "evaluator_selected_112p5deg_85cm"
        self.target_m = DISTANCE_M
        self.direction = DIRECTION
        self.choice_score = None

    def step(self, time_s, xy, packet):
        direction, distance, decision = super().step(time_s, xy, packet)
        decision["candidate_source"] = "V81 evaluator future-path screen"
        return direction, distance, decision


def run_suite(*, output_root=OUT, restage_wait_on_block=False,
              max_ticks=36,
              schema="robot-learning-lab.ape-radial-pose-owner-probe.v1"):
    screen = json.loads((v81.OUT / "report.json").read_text())
    survivor = screen["all_eight_clear_candidates"]
    assert len(survivor) == 1 and survivor[0]["angle_deg"] == ANGLE_DEG and \
        survivor[0]["distance_m"] == DISTANCE_M
    source = v77.runner.original()
    policy = json.loads(v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    vision = v77.runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v77.runner.owner.TEACH_CROP).convert("RGB")])[0]
    original_geometry = v77.runner.geometry_at
    original_packet = v77.runner.observed_packet
    original_max_ticks = OracleRadialPose.max_ticks
    results = []
    try:
        OracleRadialPose.max_ticks = max_ticks
        v77.runner.geometry_at = v77.v72.v64.v51.v46.low_prop_geometry(
            original_geometry)
        v77.runner.observed_packet = v77.v72.v64.v51.v46.plane_packet(
            original_packet, (.05, .30))
        for case in CASES:
            path_name, endpoint, finish_s = next(
                item for item in v81.PATHS if item[0] == case)
            history = []

            def retain(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    history.append({"time_s": float(time_s),
                                    "packet": list(packet)})
                return packet

            obstacle = lambda t, endpoint=endpoint, finish_s=finish_s: \
                v77.v65.prop_at(t, checkpoint_s, endpoint, finish_s)
            run = v77.runner.run_case(
                v77.v72.v64.v54.CASE, source, policy, scenario,
                vision, exemplar,
                flank_probe_factory=lambda video:
                    v77.v72.v64.v51.v41.LocalizedFlankCamera(video),
                require_camera_pose=True, route_obstacle_fn=obstacle,
                pre_show_restage=OracleRadialPose(history),
                owner_fovy_deg=50., range_transform=retain,
                near_body_controller=v77.v72.v64.v52.SurfaceTrackEscape(),
                task_risk_monitor=v78.stationary_dynamic_risk,
                output_root=output_root / case,
                **({"restage_wait_on_block": True}
                   if restage_wait_on_block else {}))
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            steps = [r for r in rows if r["stage"] ==
                     "pre_show_restage_step"]
            capture = [r for r in rows if r["stage"] ==
                       "fresh_new_show_capture"]
            candidate = [r for r in rows if r["stage"] ==
                         "fresh_item_candidate"]
            results.append({"path": path_name,
                            "endpoint_xy_m": endpoint.tolist(),
                            "finish_after_checkpoint_s": finish_s,
                            "run": run,
                            "restage_steps": len(steps),
                            "meeting_xy_m": steps[-1]["ape_xy_m"]
                            if steps else None,
                            "fresh_captures": len(capture),
                            "item_valid_depth_pixels": candidate[0][
                                "depth_valid_pixels"] if candidate else None})
            print(case, run["outcome"], run["minimum_prop_gap_m"],
                  run["minimum_proxy_gap_m"], len(steps),
                  len(capture), flush=True)
    finally:
        v77.runner.geometry_at = original_geometry
        v77.runner.observed_packet = original_packet
        OracleRadialPose.max_ticks = original_max_ticks
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"schema": schema,
              "source_sha256": v77.v72.v52.sha(__file__),
              "v81_report_sha256": v77.v72.v52.sha(v81.OUT / "report.json"),
              "runner_source_sha256": v77.v72.v52.sha(v77.runner.__file__),
              "scene_sha256": v77.v72.v52.sha(
                  v77.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v77.v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "candidate_source": "V81 evaluator future-path feasibility",
              "angle_deg": ANGLE_DEG, "distance_m": DISTANCE_M,
              "restage_wait_on_block": restage_wait_on_block,
              "restage_max_ticks": max_ticks,
              "results": results,
              "scope": "Two complete owner episodes at an evaluator-selected radial candidate; image/task and executed packet checks are real within this historical kinematic simulation, but candidate selection is oracle and neither path is held out."}
    target = output_root / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target)


def main():
    run_suite()


if __name__ == "__main__":
    main()
