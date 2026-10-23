"""V42: rank bounded owner-response directions from timed RGB-D and range tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import owner_joint_hazard_response as v41


ROOT = v41.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v42-candidate-occupancy-response"
CASES = (*v41.PROP_CASES, "parked_after_clear")
CANDIDATES = {
    "south": np.array([0., -1.]),
    "southwest": np.array([-math.sqrt(.5), -math.sqrt(.5)]),
    "west": np.array([-1., 0.]),
    "southeast": np.array([math.sqrt(.5), -math.sqrt(.5)]),
    "hold": np.zeros(2),
}
PROP_SPEED_HYPOTHESES_MPS = (0., .3, .6, .9, 1.2)
FUTURE_S = np.arange(0., 1.41, .1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LowerSurfaceFlankCamera(v41.LocalizedFlankCamera):
    """Track the visible lower foreground surface separately from arms/head."""

    def __init__(self, video_path):
        super().__init__(video_path)
        self.lower_track = []
        self.latest_packet = None

    def observe(self, model, data, xy, time_s):
        packet = super().observe(model, data, xy, time_s)
        lower_xy = velocity = None
        count = 0
        if packet["actor_cue_usable"] and self.raw_evidence:
            row = self.raw_evidence[-1]
            rgb, depth = row["rgb"], row["depth"]
            old_rgb, old_depth = row["baseline_rgb"], row["baseline_depth"]
            foreground = ((np.max(cv2.absdiff(rgb, old_rgb), axis=2) >
                           v41.v40.v38.v36.RGB_CHANGE) &
                          np.isfinite(depth) & np.isfinite(old_depth) &
                          (old_depth-depth > v41.v40.v38.v36.DEPTH_CHANGE_M) &
                          (depth >= .15) & (depth <= 2.8))
            ys, xs = np.where(foreground)
            if len(xs):
                focal = 480. / (2*math.tan(math.radians(row["fovy_deg"])/2))
                z = depth[ys, xs]
                local = np.vstack(((xs-320.)*z/focal,
                                   -(ys-240.)*z/focal, -z))
                world = (row["camera_rotation"] @ local).T + \
                    row["camera_position"]
                lower = world[(world[:, 2] >= 0.) & (world[:, 2] <= .30)]
                count = len(lower)
                if count >= 250:
                    lower_xy = np.median(lower[:, :2], axis=0)
                    self.lower_track.append((float(time_s), lower_xy))
                    if len(self.lower_track) >= 2:
                        old_t, old_xy = self.lower_track[-2]
                        if time_s-old_t > .04:
                            velocity = (lower_xy-old_xy)/(time_s-old_t)
        packet["lower_foreground_pixels"] = count
        packet["lower_surface_xy_est_m"] = (lower_xy.tolist()
                                             if lower_xy is not None else None)
        packet["lower_surface_velocity_est_mps"] = (
            velocity.tolist() if velocity is not None else None)
        self.latest_packet = packet
        return packet


class CandidateOccupancyResponse:
    """A development planner with explicit person/low-surface hypotheses."""

    def __init__(self):
        self.person_surface = None
        self.person_velocity = None
        self.person_time_s = None
        self.chosen = None
        self.choice = None
        self.camera = None

    def on_alarm(self, time_s, xy, camera_packet, range_packet):
        surface = camera_packet["lower_surface_xy_est_m"]
        velocity = camera_packet["lower_surface_velocity_est_mps"]
        if surface is not None and velocity is not None:
            self.person_surface = np.asarray(surface)
            self.person_velocity = np.asarray(velocity)
            self.person_time_s = float(time_s)

    @staticmethod
    def low_surface(packet, xy):
        angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
        points = np.asarray([
            np.asarray(xy)+reading*np.array([math.cos(angle), math.sin(angle)])
            for reading, angle in zip(packet, angles)
            if reading is not None and .2 < reading < 1.3 and
            math.radians(-55) <= angle <= math.radians(-10)])
        if len(points) < 3:
            return None
        return np.array([np.percentile(points[:, 0], 20),
                         np.median(points[:, 1])])

    def rank(self, now_s, xy, packet, packet_pose):
        if self.person_surface is None or packet is None:
            return {"chosen": "hold", "reason": "missing_person_or_range",
                    "scores": {}}
        low = self.low_surface(packet, packet_pose)
        if low is None:
            return {"chosen": "hold", "reason": "missing_low_sector_coverage",
                    "scores": {}}
        scores = {}
        for name, direction in CANDIDATES.items():
            person_min = prop_min = float("inf")
            for future in FUTURE_S:
                robot = np.asarray(xy)+direction*min(.25, .25*future)
                person = self.person_surface+self.person_velocity*(
                    now_s-self.person_time_s+future)
                person_min = min(person_min, np.linalg.norm(robot-person)-
                                 .17-.08-.05-.02*future)
                for speed in PROP_SPEED_HYPOTHESES_MPS:
                    point = low+np.array([-speed*future, 0.])
                    prop_min = min(prop_min, np.linalg.norm(robot-point)-
                                   .17-.08-.05-.03*future)
            scores[name] = {"person_min_m": float(person_min),
                            "low_min_m": float(prop_min),
                            "eligible_person": bool(person_min >= 0.)}
        eligible = [name for name, score in scores.items()
                    if score["eligible_person"] and name != "hold"]
        if not eligible:
            chosen, reason = "hold", "no_person_clear_candidate"
        else:
            chosen = max(eligible, key=lambda name: (
                scores[name]["low_min_m"], scores[name]["person_min_m"]))
            reason = ("predicted_joint_margin" if scores[chosen]["low_min_m"] >= 0.
                      else "least_exposed_while_person_approaches")
        return {"chosen": chosen, "reason": reason,
                "person_surface_xy_m": self.person_surface.tolist(),
                "person_velocity_mps": self.person_velocity.tolist(),
                "low_surface_xy_m": low.tolist(),
                "low_speed_hypotheses_mps": PROP_SPEED_HYPOTHESES_MPS,
                "scores": scores}

    def step(self, time_s, xy, packet, packet_pose, packet_time_s, step_m):
        if self.chosen is None:
            if step_m <= 0.:
                return np.zeros(2), {"action": "await_first_movement_tick",
                                      "choice": None,
                                      "range_packet_time_s": packet_time_s,
                                      "corridor_clear_from_range": False,
                                      "direction_xy": [0., 0.],
                                      "commanded_step_m": step_m}
            recent = self.camera.latest_packet if self.camera else None
            if (recent is not None and recent["actor_cue_usable"] and
                    recent["lower_surface_xy_est_m"] is not None and
                    recent["lower_surface_velocity_est_mps"] is not None):
                self.person_surface = np.asarray(
                    recent["lower_surface_xy_est_m"])
                self.person_velocity = np.asarray(
                    recent["lower_surface_velocity_est_mps"])
                self.person_time_s = recent["time_s"]
            self.choice = self.rank(time_s, xy, packet, packet_pose)
            self.chosen = self.choice["chosen"]
        direction = CANDIDATES[self.chosen]
        points = v41.JointHazardResponse.endpoints(packet, packet_pose)
        clear = (v41.JointHazardResponse.corridor_clear(
            points, xy, direction) if self.chosen != "hold" else False)
        applied = direction if clear else np.zeros(2)
        return applied, {"action": ("execute_"+self.chosen if clear
                                     else "hold_unknown_corridor"),
                         "choice": self.choice,
                         "range_packet_time_s": packet_time_s,
                         "corridor_clear_from_range": clear,
                         "direction_xy": applied.tolist(),
                         "commanded_step_m": step_m}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    source = v41.v40.v38.original()
    policy = json.loads(v41.v40.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v41.v40.v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v41.v40.v38.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    results = []
    for case in (args.case or CASES):
        config = v41.PROP_CASES.get(case, v41.PROP_CASES["development"])
        obstacle_fn = lambda t, config=config: v41.v40.prop_at(
            t, checkpoint_s, config)
        controller = CandidateOccupancyResponse()
        def camera_factory(video):
            probe = LowerSurfaceFlankCamera(video)
            controller.camera = probe
            return probe
        condition = ("parked_after_clear" if case == "parked_after_clear" else
                     "blind_return_flank_fast_response")
        run = v41.v40.v38.run_case(
            condition, source, policy, scenario, vision, exemplar,
            flank_probe_factory=camera_factory,
            flank_response=v41.v40.v38.FLANK_RESPONSE[
                "blind_return_flank_fast_response"],
            require_camera_pose=True, route_obstacle_fn=obstacle_fn,
            response_controller=controller, output_root=OUT / case)
        raw = OUT / case / condition / "flank-rgbd-watch.npz"
        results.append({"case": case, "condition": condition,
                        "prop_config": config,
                        "controller_choice": controller.choice,
                        "run": run,
                        "camera_rgbd_evidence": {"path": str(raw),
                                                 "sha256": sha(raw)}})
        print(case, controller.chosen, run["outcome"],
              run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-candidate-occupancy.v1",
              "task": "T-9v3p3n", "source_sha256": sha(Path(__file__)),
              "v41_source_sha256": sha(v41.__file__),
              "v38_source_sha256": sha(v41.v40.v38.__file__),
              "scene_sha256": sha(v41.v40.v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v41.v40.v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "candidate_directions": {name: value.tolist()
                                       for name, value in CANDIDATES.items()},
              "prop_speed_hypotheses_mps": PROP_SPEED_HYPOTHESES_MPS,
              "future_horizon_s": float(FUTURE_S[-1]),
              "cases": list(args.case or CASES), "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
