"""V41: pilot image-grounded person and range-grounded prop response planning.

The camera motion track and range endpoints are actor observations. Person and
prop evaluator positions are used only by the inherited outcome scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

import owner_post_alarm_moving_flank as v39
import owner_response_corridor_recheck as v40


ROOT = v40.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v41-joint-hazard-response"
PROP_CASES = {
    **v40.PROP_CASES,
    "holdout_fast_early": {"move_begin_after_checkpoint_s": 2.50,
                           "move_duration_s": .35, "end_x_m": 1.90},
    "holdout_fast_late": {"move_begin_after_checkpoint_s": 2.80,
                          "move_duration_s": .35, "end_x_m": 1.90},
    "holdout_closer_finish": {"move_begin_after_checkpoint_s": 2.65,
                              "move_duration_s": .55, "end_x_m": 1.75},
}
SOUTH = np.array([0., -1.])
SOUTHWEST = np.array([-math.sqrt(.5), -math.sqrt(.5)])
CLOSING_X_M = .07


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LocalizedFlankCamera(v39.MovingFlankCamera):
    """Back-project newly foreground RGB-D pixels using timed joint+IMU pose."""

    def __init__(self, video_path: Path):
        super().__init__("joint_imu", video_path)
        self.track = []
        self.raw_evidence = []
        self.raw_path = Path(video_path).with_suffix(".npz")

    def observe(self, model, data, xy, time_s):
        packet = super().observe(model, data, xy, time_s)
        rgb, depth, _ = v40.v38.owner.capture(self.renderer, data,
                                               self.camera, True)
        estimated_yaw = (packet["joint_head_yaw_deg"] +
                         packet["imu_body_yaw_deg"])
        baseline_rgb, baseline_depth = self._registered_baseline(
            rgb, depth, model, estimated_yaw)
        changed = np.max(cv2.absdiff(rgb, baseline_rgb), axis=2) > \
            v40.v38.v36.RGB_CHANGE
        foreground = (changed & np.isfinite(depth) &
                      np.isfinite(baseline_depth) &
                      (baseline_depth-depth > v40.v38.v36.DEPTH_CHANGE_M) &
                      (depth >= .15) & (depth <= 2.8))
        v, u = np.where(foreground)
        camera_id = model.camera(self.camera).id
        if packet["actor_cue_usable"]:
            self.raw_evidence.append({
                "time_s": float(time_s),
                "rgb": rgb.copy(), "depth": depth.copy(),
                "baseline_rgb": baseline_rgb.copy(),
                "baseline_depth": baseline_depth.copy(),
                "camera_rotation": data.cam_xmat[camera_id].reshape(3, 3).copy(),
                "camera_position": data.cam_xpos[camera_id].copy(),
                "fovy_deg": float(model.cam_fovy[camera_id])})
        estimate = None
        if len(u) >= 250 and packet["actor_cue_usable"]:
            # MuJoCo's rendered depth is optical-axis Z. Camera local +x is
            # image right, local +y is image up, and local -z is forward.
            focal = 480. / (2*math.tan(math.radians(
                float(model.cam_fovy[camera_id]))/2))
            z = depth[v, u]
            local = np.stack(((u-320.)*z/focal,
                              -(v-240.)*z/focal, -z))
            rotation = data.cam_xmat[camera_id].reshape(3, 3)
            points = (rotation @ local).T + data.cam_xpos[camera_id]
            middle = np.median(points[:, :2], axis=0)
            estimate = middle.tolist()
            self.track.append((float(time_s), middle))
        velocity = None
        if estimate is not None and len(self.track) >= 2:
            old_t, old_xy = self.track[-2]
            new_t, new_xy = self.track[-1]
            if new_t-old_t > .04:
                velocity = ((new_xy-old_xy)/(new_t-old_t)).tolist()
        packet["foreground_pixels"] = int(len(u))
        packet["foreground_xy_est_m"] = estimate
        packet["foreground_velocity_est_mps"] = velocity
        return packet

    def close(self):
        super().close()
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.raw_path,
                            time_s=np.asarray([row["time_s"] for row in
                                               self.raw_evidence]),
                            rgb=np.asarray([row["rgb"] for row in
                                            self.raw_evidence]),
                            depth=np.asarray([row["depth"] for row in
                                              self.raw_evidence]),
                            baseline_rgb=np.asarray([row["baseline_rgb"] for row
                                                     in self.raw_evidence]),
                            baseline_depth=np.asarray([row["baseline_depth"] for row
                                                       in self.raw_evidence]),
                            camera_rotation=np.asarray([
                                row["camera_rotation"] for row in self.raw_evidence]),
                            camera_position=np.asarray([
                                row["camera_position"] for row in self.raw_evidence]),
                            fovy_deg=np.asarray([row["fovy_deg"] for row in
                                                 self.raw_evidence]))


class JointHazardResponse:
    """Switch away from a newly closing east-south return after a camera alarm.

    This is an actor-side development policy. Its range track is a small
    cluster statistic, and its visual person forecast is deliberately exposed
    rather than treated as a safety guarantee.
    """

    def __init__(self):
        self.camera_alarm_s = None
        self.person_xy = None
        self.person_velocity = None
        self.last_sector_x = None
        self.switched = False

    def on_alarm(self, time_s, xy, camera_packet, range_packet):
        if camera_packet["foreground_xy_est_m"] is not None:
            self.person_xy = np.asarray(camera_packet["foreground_xy_est_m"])
        if camera_packet["foreground_velocity_est_mps"] is not None:
            self.person_velocity = np.asarray(
                camera_packet["foreground_velocity_est_mps"])
        if self.person_xy is not None and self.person_velocity is not None:
            self.camera_alarm_s = time_s

    @staticmethod
    def endpoints(packet, pose):
        if packet is None:
            return np.empty((0, 2))
        angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
        return np.asarray([
            np.asarray(pose) + value*np.array([math.cos(angle), math.sin(angle)])
            for value, angle in zip(packet, angles) if value is not None])

    @staticmethod
    def sector_x(packet, pose):
        if packet is None:
            return None
        angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
        x_hits = [pose[0]+value*math.cos(angle)
                  for value, angle in zip(packet, angles)
                  if value is not None and .2 < value < 1.3 and
                  math.radians(-55) <= angle <= math.radians(-10)]
        return float(np.percentile(x_hits, 20)) if len(x_hits) >= 3 else None

    @staticmethod
    def corridor_clear(points, xy, direction):
        if not len(points):
            return False
        offset = points-np.asarray(xy)
        progress = np.clip(offset @ direction, 0., .30)
        gap = np.linalg.norm(offset-progress[:, None]*direction, axis=1)
        # A 17 cm modeled ape radius and 5 cm provisional clearance.
        return bool(np.all(gap >= .22))

    def step(self, time_s, xy, packet, packet_pose, packet_time_s, step_m):
        if self.camera_alarm_s is None or packet is None or packet_pose is None:
            return np.zeros(2), {"action": "hold_missing_camera_or_range"}
        sector_x = self.sector_x(packet, packet_pose)
        x_drop = (None if sector_x is None or self.last_sector_x is None
                  else self.last_sector_x-sector_x)
        if x_drop is not None and x_drop >= CLOSING_X_M:
            self.switched = True
        if sector_x is not None:
            self.last_sector_x = sector_x
        points = self.endpoints(packet, packet_pose)
        direction = SOUTHWEST if self.switched else SOUTH
        corridor_clear = self.corridor_clear(points, xy, direction)
        if not corridor_clear:
            direction = np.zeros(2)
        forecast = (self.person_xy + self.person_velocity*
                    (time_s-self.camera_alarm_s+.7))
        return direction, {"action": ("swerve_southwest" if self.switched
                                      and corridor_clear else
                                      "continue_south" if corridor_clear else
                                      "hold_unknown_corridor"),
                           "range_packet_time_s": packet_time_s,
                           "east_south_sector_x_m": sector_x,
                           "east_south_x_drop_m": x_drop,
                           "closing_threshold_m": CLOSING_X_M,
                           "camera_person_surface_xy_m": self.person_xy.tolist(),
                           "camera_person_velocity_mps":
                               self.person_velocity.tolist(),
                           "person_surface_forecast_0p7s_xy_m":
                               forecast.tolist(),
                           "corridor_clear_from_range": corridor_clear,
                           "direction_xy": direction.tolist(),
                           "commanded_step_m": step_m}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prop-case", action="append",
                        choices=tuple(PROP_CASES))
    args = parser.parse_args()
    source = v40.v38.original()
    policy = json.loads(v40.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v40.v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v40.v38.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    results = []
    for prop_case in (args.prop_case or tuple(PROP_CASES)):
        config = PROP_CASES[prop_case]
        obstacle_fn = lambda t, config=config: v40.prop_at(t, checkpoint_s,
                                                            config)
        controller = JointHazardResponse()
        run = v40.v38.run_case(
            "blind_return_flank_fast_response", source, policy, scenario,
            vision, exemplar,
            flank_probe_factory=lambda video: LocalizedFlankCamera(video),
            flank_response=v40.v38.FLANK_RESPONSE[
                "blind_return_flank_fast_response"],
            require_camera_pose=True,
            route_obstacle_fn=obstacle_fn,
            response_controller=controller,
            output_root=OUT / prop_case)
        raw = OUT / prop_case / "blind_return_flank_fast_response" / \
            "flank-rgbd-watch.npz"
        results.append({"prop_case": prop_case, "run": run,
                        "camera_rgbd_evidence": {"path": str(raw),
                                                 "sha256": sha(raw)}})
        print(prop_case, run["outcome"], run["minimum_proxy_gap_m"],
              run["minimum_prop_gap_m"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-joint-hazard-response.v1",
              "task": "T-9v3p3n",
              "source_sha256": sha(Path(__file__)),
              "v38_source_sha256": sha(Path(v40.v38.__file__)),
              "v39_source_sha256": sha(Path(v39.__file__)),
              "v40_source_sha256": sha(Path(v40.__file__)),
              "scene_sha256": sha(v40.v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v40.v38.SOURCE),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "prop_cases": {case: PROP_CASES[case] for case in
                             (args.prop_case or tuple(PROP_CASES))},
              "parameters": {"closing_x_m": CLOSING_X_M,
                             "camera_mode": "joint_imu_foreground_backprojection",
                             "response": v40.v38.FLANK_RESPONSE[
                                 "blind_return_flank_fast_response"]},
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
