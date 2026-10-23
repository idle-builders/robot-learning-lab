"""V63 development monitor for several anonymous range surfaces at once."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_far_west_turn_holdout as v61


v59, v60 = v61.v60.v59, v61.v60
runner, v51, v52, v54 = v59.runner, v59.v51, v59.v52, v59.v54
ROOT = v61.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v63-multi-surface-escape"
PATHS = {"outbound_control": v59.v56.PATHS["outbound_control"],
         "new_west_south_turn": v59.v56.PATHS["new_west_south_turn"],
         "far_west_turn": v61.END_XY_M}
DIRECTIONS = {"north": np.array([0., 1.]),
              "northwest": np.array([-math.sqrt(.5), math.sqrt(.5)]),
              "northeast": np.array([math.sqrt(.5), math.sqrt(.5)]),
              "west": np.array([-1., 0.]),
              "east": np.array([1., 0.]),
              "hold": np.zeros(2)}


def clusters(packet, pose):
    points = []
    for ray, distance in enumerate(packet):
        if distance is None or distance > 1.5:
            continue
        angle = math.radians(-180.+360.*ray/len(packet))
        points.append((ray, np.asarray(pose)+distance*np.array([
            math.cos(angle), math.sin(angle)])))
    groups, group = [], []
    for ray, xy in points:
        if group and (ray != group[-1][0]+1 or
                      np.linalg.norm(xy-group[-1][1]) > .25):
            groups.append(group)
            group = []
        group.append((ray, xy))
    if group:
        groups.append(group)
    return [{"rays": [group[0][0], group[-1][0]],
             "center": np.mean([xy for _, xy in group], axis=0)}
            for group in groups if len(group) >= 3]


def response_scores(pose, packet, center, velocity):
    endpoints = v51.v41.JointHazardResponse.endpoints(packet, pose)
    scores = {}
    for name, direction in DIRECTIONS.items():
        margins, clear = [], True
        for elapsed in np.arange(0., .80001, .1):
            future_pose = np.asarray(pose)+direction*.25*elapsed
            nearest = float(np.min(np.linalg.norm(
                endpoints-future_pose, axis=1)))
            clear = clear and nearest >= v51.STEP_ENDPOINT_GAP_M
            margins.append(float(np.linalg.norm(
                center+velocity*elapsed-future_pose))-.17)
        scores[name] = {"forecast_margin_m": min(margins),
                        "latest_endpoint_corridor_clear": bool(clear)}
    eligible = [name for name, row in scores.items()
                if row["latest_endpoint_corridor_clear"]]
    chosen = max(eligible, key=lambda name: (
        scores[name]["forecast_margin_m"], name != "hold")) if eligible else None
    return chosen, scores


class MultiSurfaceEscape(v51.FullAngleEscape):
    def start(self, xy, time_s, packet):
        super().start(xy, time_s, packet)
        self.previous = clusters(packet, xy)
        self.previous_s = float(time_s)
        self.chosen = None
        self.focus = None
        self.scores = None

    def observe(self, time_s, xy, packet):
        current = clusters(packet, xy)
        tracks = []
        dt = float(time_s)-self.previous_s
        if dt > 0.:
            for group in current:
                if not self.previous:
                    continue
                old = min(self.previous, key=lambda row:
                          np.linalg.norm(group["center"]-row["center"]))
                delta = group["center"]-old["center"]
                if np.linalg.norm(delta) > .30:
                    continue
                velocity = delta/dt
                speed = float(np.linalg.norm(velocity))
                if speed < .20:
                    continue
                tau = float(np.clip(-np.dot(group["center"]-xy, velocity)/
                                    (speed*speed), 0., .80))
                predicted = float(np.linalg.norm(
                    group["center"]+velocity*tau-np.asarray(xy)))
                tracks.append({"rays": group["rays"],
                               "center_xy_m": group["center"].tolist(),
                               "velocity_mps": velocity.tolist(),
                               "predicted_surface_to_center_m": predicted,
                               "closest_at_s": tau})
        risky = [track for track in tracks if
                 track["predicted_surface_to_center_m"] < .40]
        if self.alarm_s is None and risky:
            self.focus = min(risky, key=lambda row:
                             row["predicted_surface_to_center_m"])
            self.chosen, self.scores = response_scores(
                xy, packet, np.asarray(self.focus["center_xy_m"]),
                np.asarray(self.focus["velocity_mps"]))
            if self.chosen not in (None, "hold"):
                self.direction = DIRECTIONS[self.chosen].copy()
                self.alarm_s = float(time_s)
                self.last_command_s = float(time_s)
        self.previous, self.previous_s = current, float(time_s)
        self.last_packet, self.last_packet_s = list(packet), float(time_s)
        return {"alarm": self.alarm_s is not None,
                "capture_time_s": round(float(time_s), 5),
                "valid_rays": sum(value is not None for value in packet),
                "clusters": [{"rays": row["rays"],
                              "center_xy_m": row["center"].tolist()}
                             for row in current],
                "moving_tracks": tracks,
                "risk_tracks": len(risky),
                "chosen_response": self.chosen,
                "response_scores": self.scores,
                "focus": self.focus,
                "escape_direction": self.direction.tolist()
                if self.alarm_s is not None else None}


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(runner.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet = runner.geometry_at, runner.observed_packet
    results = []
    try:
        runner.geometry_at = v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v51.v46.plane_packet(
            original_packet, (.05, .30))
        for path_name, end_xy in PATHS.items():
            history = []

            def retain(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    history.append({"time_s": float(time_s),
                                    "packet": list(packet)})
                return packet

            obstacle_fn = lambda t, end_xy=end_xy: v59.v50.prop_at(
                t, checkpoint_s, end_xy)
            run = runner.run_case(
                v54.CASE, source, policy, scenario, vision, exemplar,
                flank_probe_factory=lambda video:
                    v51.v41.LocalizedFlankCamera(video),
                require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                pre_show_restage=v59.ApproachAwareMeetingPose(history),
                owner_fovy_deg=50., range_transform=retain,
                near_body_controller=MultiSurfaceEscape(),
                output_root=OUT / path_name)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            alarm = next((row for row in rows if row["stage"] ==
                          "near_body_escape_plan"), None)
            commanded = sum(row["near_body_command"]["commanded_step_m"]
                            for row in rows if row["stage"] ==
                            "second_inference_range_watch" and
                            row["near_body_command"] is not None)
            results.append({"path": path_name, "run": run,
                            "alarm_s": alarm["time_s"] if alarm else None,
                            "commanded_m": commanded,
                            "response_direction": (alarm["actor_decision"]
                                ["chosen_response"] if alarm else None)})
            print(path_name, run["outcome"], run["minimum_prop_gap_m"],
                  results[-1]["alarm_s"], commanded,
                  results[-1]["response_direction"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-multi-surface-escape.v1",
              "source_sha256": v52.sha(__file__),
              "v61_report_sha256": v52.sha(v61.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
