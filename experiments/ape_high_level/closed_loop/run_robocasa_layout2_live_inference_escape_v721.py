"""Development live range-track interrupt and mapped exit during owner inference."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from held_pose_screen import along_path
from multi_novel_range_objects import MultiNovelRangeObjects
import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_owner_inference_trigger_v716 as trigger
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_moving_prop_v423 as moving
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-live-inference-escape-v721"
GOAL = np.array([1.1, -3.95])
SPEED_M_S = .3
SPEED_EVIDENCE_M_S = .45


class Prop(trigger.CaptureTriggeredProp):
    def __init__(self, moving_profile):
        super().__init__()
        self.moving_profile = moving_profile

    def __call__(self, t):
        return super().__call__(t) if self.moving_profile else side.START.copy()


class LiveMonitor:
    def __init__(self, model, data, actual, script, grid, estimate):
        self.model, self.data, self.actual = model, data, actual
        self.script, self.grid, self.estimate = script, grid, estimate
        with np.load(transfer.PRIOR) as prior:
            self.tracker = MultiNovelRangeObjects(
                prior["xs"], prior["ys"], prior["free"], prior["occupied"])
        self.prop_body = model.body("route_obstacle").id
        self.head = model.camera("head").id
        self.next_scan_s = None
        self.scan_index = 0
        self.centers = []
        self.motion_streak = 0
        self.trigger_time_s = None
        self.plan_status = None
        self.estimated_path = None
        self.actual_path = None
        self.path_length = None
        self.dispatch_time_s = None
        self.stop_time_s = None
        self.start_estimated = None
        self.start_actual = None
        self.response_goal = None

    def __call__(self, t):
        self.model.body_pos[self.prop_body, :2] = self.script(t)
        self.model.cam_pos[self.head, :2] = self.actual
        mujoco.mj_forward(self.model, self.data)
        raw = moving.raw_nearest_range(self.model, self.data, self.actual)
        result = {"actor_guard_range_m": raw, "actual_xy_m": self.actual.tolist(),
                  "estimated_xy_m": self.estimate[:2].tolist(),
                  "invalidate_image": self.trigger_time_s is not None,
                  "actor_response_state": "holding"}
        if self.next_scan_s is None or t + 1.e-6 >= self.next_scan_s:
            packet = joined.SCAN(self.model, self.data, self.actual, 5.,
                                 np.random.default_rng(72100 + self.scan_index))
            self.grid.next_capture_time_s = t
            self.grid.update(packet, self.estimate[:2], 5.)
            snapshot = self.tracker.update(packet, self.estimate[:2], t)
            present = [track for track in snapshot["tracks"]
                       if track["status"] == "confirmed_now"]
            track = max(present, key=lambda item: item["hits"]) if present else None
            self.centers.append((t, None if track is None else
                                 np.asarray(track["center_xy_m"])))
            speed = None
            if (len(self.centers) >= 3 and self.centers[-1][1] is not None and
                    self.centers[-3][1] is not None):
                t0, c0 = self.centers[-3]
                speed = float(np.linalg.norm((self.centers[-1][1] - c0) /
                                             max(.001, t - t0)))
            self.motion_streak = (self.motion_streak + 1 if speed is not None and
                                  speed > SPEED_EVIDENCE_M_S else 0)
            result.update({"range_72_fused_m": packet,
                           "actor_track": track,
                           "actor_confirmed_tracks": present,
                           "actor_track_center_speed_0p2s_m_s": speed,
                           "actor_motion_streak": self.motion_streak,
                           "actor_scan_capture_time_s": t})
            self.scan_index += 1
            self.next_scan_s = t + .1
            admission_gate = getattr(self, "admission_gate", None)
            if (self.trigger_time_s is None and self.motion_streak >= 2 and
                    (admission_gate is None or admission_gate(track, t, result))):
                self.trigger_time_s = t
                self.start_estimated = self.estimate[:2].copy()
                self.start_actual = self.actual.copy()
                selector = getattr(self, "response_selector", None)
                goal, selection = ((GOAL.copy(), None) if selector is None else
                                   selector(track, t, result))
                if goal is None:
                    requested, self.plan_status, path = None, "no_supported_response", []
                else:
                    requested, self.plan_status, path = route.SensorGrid.plan(
                        self.grid, self.start_estimated, goal)
                self.response_goal = None if goal is None else goal.copy()
                if self.plan_status == "known_goal" and requested is not None:
                    self.estimated_path = np.asarray(
                        [self.start_estimated] + [self.grid.xy(cell)
                                                  for cell in path[1:]])
                    self.actual_path = (self.start_actual +
                                        (self.estimated_path - self.start_estimated) *
                                        (1. + .30 / 5.89))
                    self.path_length = float(np.sum(np.linalg.norm(
                        np.diff(self.actual_path, axis=0), axis=1)))
                    self.dispatch_time_s = t
                result.update({"actor_response_trigger_time_s": t,
                               "actor_goal_xy_m": None if goal is None else goal.tolist(),
                               "actor_response_selection": selection,
                               "actor_plan_status": self.plan_status,
                               "actor_estimated_path_xy_m": None if self.estimated_path is None
                                   else self.estimated_path.tolist(),
                               "actor_response_speed_assumption_m_s": SPEED_M_S,
                               "actor_response_qualification": (
                                   "development_only" if selection is None else
                                   selection.get("status", "development_only"))})
        if self.dispatch_time_s is not None and self.stop_time_s is None:
            if raw < moving.GUARD_CENTER_RANGE_M:
                self.stop_time_s = t
                result["actor_response_state"] = "raw_range_stop"
            else:
                covered = min(self.path_length,
                              SPEED_M_S * max(0., t - self.dispatch_time_s + .02))
                self.actual[:] = along_path(self.actual_path, covered)
                self.estimate[:2] = along_path(
                    self.estimated_path, covered / (1. + .30 / 5.89))
                result["actor_response_state"] = (
                    "reached_development_exit" if covered >= self.path_length else
                    "moving_to_development_exit")
                result["actual_xy_m"] = self.actual.tolist()
                result["estimated_xy_m"] = self.estimate[:2].tolist()
        elif self.stop_time_s is not None:
            result["actor_response_state"] = "held_after_raw_range_stop"
        if self.trigger_time_s is not None:
            result["invalidate_image"] = True
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-live-inference-escape.v1",
                   "development_goal_xy_m": GOAL.tolist(),
                   "goal_source": "V720 actor-map-known southern development exit; selected after seeing the V716 path, not a general or held-out policy",
                   "response_speed_m_s": SPEED_M_S,
                   "motion_rule": "prior-map compact track, two consecutive 10Hz 0.2s center-speed estimates >0.45m/s, chosen after V719 development pair",
                   "profiles": ["stationary", "capture_triggered_side_prop"],
                   "no_evaluator_truth_for_actor": True,
                   "source_sha256": route.sha256(SOURCE),
                   "reference_trial_sha256": route.sha256(trigger.OUT / "report.json")}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
           full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
           full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
           transfer.PROP_START.copy(), joined.INFERENCE_MONITOR_FACTORY,
           joined.CAPTURE_EVENT_FACTORY, joined.INTERRUPTED_HOLD_UNTIL_S)
    try:
        full.POST_STOP_UNTIL_S = station.HORIZON_S
        full.STANDOFF_M = station.STANDOFF_M
        full.STATION_LATERAL_M = station.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        full.ACTIVE_VIEW_CORRECTION = True
        transfer.PROP_START = side.START.copy()
        joined.INFERENCE_MONITOR_FACTORY = LiveMonitor
        joined.CAPTURE_EVENT_FACTORY = lambda script: script.arm
        joined.INTERRUPTED_HOLD_UNTIL_S = station.HORIZON_S
        full.RUN_SCOPE = "layout2 full owner task, live 10Hz range-track interrupt and 20ms guarded development exit during item inference; 50cm kinematic proxy"
        for label, is_moving in (("stationary", False), ("moving", True)):
            prop = Prop(is_moving)
            full.OUT = OUT / label
            full.PROP_SCRIPT_FACTORY = lambda _, prop=prop: prop
            full.main()
            path = full.OUT / "report.json"
            report = json.loads(path.read_text())
            report["capture_trigger_time_s"] = prop.capture_time_s
            report["profile"] = label
            report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
            path.write_text(json.dumps(report, indent=2) + "\n")
            ep = report["episode"]
            guard = next(row for row in ep["trace"] if row.get("stage") ==
                         "owner_inference_near_body_monitor")
            response = next((sample for sample in guard["monitor_samples"] if
                             "actor_response_trigger_time_s" in sample), None)
            print(label, ep["route_state"], ep["outcome"],
                  guard["first_alarm_time_s"],
                  None if response is None else response["actor_plan_status"],
                  ep["final_actual_xy_m"], flush=True)
    finally:
        (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
         full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
         full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
         transfer.PROP_START, joined.INFERENCE_MONITOR_FACTORY,
         joined.CAPTURE_EVENT_FACTORY, joined.INTERRUPTED_HOLD_UNTIL_S) = old


if __name__ == "__main__":
    main()
