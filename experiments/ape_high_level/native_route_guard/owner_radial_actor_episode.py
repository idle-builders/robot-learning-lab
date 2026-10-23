"""V86: full owner episodes with the V85 actor-ranked radial destination.

The radial choice uses only pre-move range history. Each 2.5 cm step is
checked against the preceding packet; a blocked step can wait for fresh scans.
The evaluator's prop path is never passed to the pose selector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_radial_actor_rank as v85
import owner_radial_oracle_pose_probe as v82
import owner_restage_wait_runner as runner
import owner_stationary_dynamic_risk_turn as v78


ROOT = v85.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v86-radial-actor-episode"
CASES = ("outward_normal", "far_west_fast", "near_northwest")


class ActorRadialPose(v85.v59.ApproachAwareMeetingPose):
    max_ticks = 50

    def start(self, xy, packet, time_s):
        super().start(xy, packet, time_s)
        chosen, rows, surfaces = v85.rank(packet, self.history, xy)
        self.scores = rows
        self.dynamic_surfaces = surfaces
        self.choice = (f"actor_radial_{chosen['angle_deg']}deg_"
                       f"{int(chosen['distance_m']*100)}cm") if chosen else None
        self.target_m = chosen["distance_m"] if chosen else 0.
        self.direction = (np.asarray([
            math.cos(math.radians(chosen["angle_deg"])),
            math.sin(math.radians(chosen["angle_deg"]))])
            if chosen else np.zeros(2))
        self.choice_score = chosen["actor_combined_margin_m"] if chosen else None
        self.chosen = chosen

    def step(self, time_s, xy, packet):
        direction, distance, decision = super().step(time_s, xy, packet)
        decision["candidate_source"] = "V85 actor range-history radial ranking"
        return direction, distance, decision


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    vision = runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(runner.owner.TEACH_CROP).convert("RGB")])[0]
    original_geometry, original_packet = runner.geometry_at, runner.observed_packet
    results = []
    try:
        runner.geometry_at = v82.v77.v72.v64.v51.v46.low_prop_geometry(
            original_geometry)
        runner.observed_packet = v82.v77.v72.v64.v51.v46.plane_packet(
            original_packet, (.05, .30))
        for case in CASES:
            _, endpoint, finish_s = next(
                item for item in v85.v81.PATHS if item[0] == case)
            history = []

            def retain(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    history.append({"time_s": float(time_s),
                                    "packet": list(packet)})
                return packet

            obstacle = lambda t, endpoint=endpoint, finish_s=finish_s: \
                v82.v77.v65.prop_at(t, checkpoint_s, endpoint, finish_s)
            pose = ActorRadialPose(history)
            run = runner.run_case(
                v82.v77.v72.v64.v54.CASE, source, policy, scenario,
                vision, exemplar,
                flank_probe_factory=lambda video:
                    v82.v77.v72.v64.v51.v41.LocalizedFlankCamera(video),
                require_camera_pose=True, route_obstacle_fn=obstacle,
                pre_show_restage=pose, owner_fovy_deg=50.,
                range_transform=retain,
                near_body_controller=v82.v77.v72.v64.v52.SurfaceTrackEscape(),
                task_risk_monitor=v78.stationary_dynamic_risk,
                restage_wait_on_block=True, output_root=OUT / case)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            steps = [r for r in rows if r["stage"] == "pre_show_restage_step"]
            waits = [r for r in rows if r["stage"] == "pre_show_restage_wait"]
            captures = [r for r in rows if r["stage"] ==
                        "fresh_new_show_capture"]
            candidate = next((r for r in rows if r["stage"] ==
                              "fresh_item_candidate"), None)
            results.append({"path": case, "run": run,
                            "actor_chosen": pose.chosen,
                            "actor_history_rows": len(history),
                            "restage_steps": len(steps),
                            "restage_waits": len(waits),
                            "meeting_xy_m": steps[-1]["ape_xy_m"]
                            if steps else None,
                            "fresh_captures": len(captures),
                            "item_valid_depth_pixels":
                            candidate["depth_valid_pixels"]
                            if candidate else None})
            print(case, run["outcome"], run["minimum_prop_gap_m"],
                  run["minimum_proxy_gap_m"], len(steps), len(waits),
                  len(captures), flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "robot-learning-lab.ape-radial-actor-episode.v1",
        "source_sha256": v82.v77.v72.v52.sha(__file__),
        "v85_report_sha256": v82.v77.v72.v52.sha(v85.OUT / "report.json"),
        "runner_source_sha256": v82.v77.v72.v52.sha(runner.__file__),
        "scene_sha256": v82.v77.v72.v52.sha(runner.v35.moving.SCENE_XML),
        "current_design_pointer_sha256": v82.v77.v72.v52.sha(
            ROOT / "robot/ape-current-design.json"),
        "checkpoint_s": checkpoint_s,
        "candidate_source": "V85 actor-only radial score",
        "restage_wait_on_block": True,
        "results": results,
        "scope": "Complete owner episodes with an actor-only expanded radial choice and fresh-scan checked 2.5 cm steps/blocked-step waits. Historical kinematic proxy, ideal low/mid rays, one authored kitchen; evaluator geometry is post-choice only."}
    target = OUT / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
