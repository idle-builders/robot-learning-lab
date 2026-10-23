"""V98: run owner-image inference against a moving simulation clock.

The captured RGB-D is fixed for the recognizer while a separate logical
scheduler advances the furnished owner scene and V95's 10Hz shin monitor.
No real concurrent GPU worker or physical response is claimed. The candidate
is invalidated if the monitor interrupts before inference is released.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import FAR_HAND_XYZ_M, set_person_pose
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
import run_articulated_owner_low_drift_v92 as drift
import run_blocked_owner_help_v317 as owner_module
import run_continuous_owner_guard_v95 as v95
import run_lateral_clutter_shin_search_v284 as search
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_moving_low_clutter_v302 import STOOL_SIZE


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v98"
DRIFT_START_S = 17.2
DRIFT_DURATION_S = .8
INFERENCE_BUDGET_S = .8


class InferenceDriftRoom(drift.OfferDriftRoom):
    def _update_moving_obstacle(self) -> None:
        LateCorridorRoom._update_moving_obstacle(self)
        if self.motion_side is None or self.time_s < drift.OFFER_START_S:
            return
        fraction = min(1., max(0., (self.time_s - DRIFT_START_S) /
                               DRIFT_DURATION_S))
        y = drift.START_Y_M + fraction * (drift.END_Y_M - drift.START_Y_M)
        self.obstacle_xy[3] = (0., y)
        self.model.body_pos[self.model.body("loop_stool").id] = (
            0., y, STOOL_SIZE[2])


class InferenceMonitor(v95.OwnerPhaseMonitor):
    def point_gate(self) -> dict:
        points = super().point_gate()
        # V96/V97 show that occupied-point clearance cannot authorize the
        # arm-volume step, even when its point distance looks ample.
        return {**points, "allowed": False,
                "status": "no_observed_free_swept_arm_corridor",
                "v95_point_gate_would_allow": points["allowed"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = (v95.OUT, v95.OfferDriftRoom, v95.OwnerPhaseMonitor,
                search.run_case)
    raw_run_case = search.run_case

    def inference_scheduler(room, capture_time_s, camera, label, choice, video):
        if not hasattr(room, "v95_monitor") or not label.startswith(
                ("owner-presented", "post-owner")):
            return {"valid": True, "status": "no_delayed_owner_inference",
                    "capture_time_s": round(capture_time_s, 4),
                    "completion_time_s": round(room.time_s, 4)}
        monitor = room.v95_monitor
        start = room.time_s
        while room.time_s - start < INFERENCE_BUDGET_S - 1e-6:
            # Call the class method because V343 temporarily replaces the
            # instance's wait() for the first scripted owner presentation.
            type(room).wait(room, safety_probe=lambda *_: (
                {"reason": "owner_phase_alarm"}
                if monitor.alarm is not None else None))
            rgb, _ = room.capture("lower_rgb", depth=False)
            video.append_data(owner_module.panel(
                room, rgb, f"V98 recognition pending | t{room.time_s:.2f} "
                f"| {'alarm' if monitor.alarm else 'watching'}"))
            if monitor.alarm is not None:
                break
        valid = monitor.alarm is None and not room.motion_interrupted
        return {"valid": valid,
                "status": "fresh_completed" if valid else
                          "invalidated_by_near_body_alarm",
                "capture_time_s": round(capture_time_s, 4),
                "completion_time_s": round(room.time_s, 4),
                "elapsed_s": round(room.time_s - capture_time_s, 4),
                "alarm_capture_time_s": (None if monitor.alarm is None else
                                          monitor.alarm["capture_time_s"]),
                "candidate_before_validity": choice["accepted_box"]}

    def scheduled_run_case(*args, **kwargs):
        kwargs["during_inference"] = inference_scheduler
        return raw_run_case(*args, **kwargs)

    try:
        v95.OUT = OUT
        v95.OfferDriftRoom = InferenceDriftRoom
        v95.OwnerPhaseMonitor = InferenceMonitor
        search.run_case = scheduled_run_case
        v95.main()
    finally:
        (v95.OUT, v95.OfferDriftRoom, v95.OwnerPhaseMonitor,
         search.run_case) = original
    path = OUT / "live-moving-owner-offer-audit.json"
    report = json.loads(path.read_text())
    results = []
    for row in report["rows"]:
        seed = row["case"][0]
        condition = row["condition"]
        branch = (OUT / condition / f"seed{seed}-present" /
                  f"seed{seed}-present")
        trace = json.loads((branch / "episode-trace.json").read_text())
        views = [event for event in trace["controller_events"]
                 if event.get("action") == "view" and event.get(
                     "label", "").startswith(("owner-presented", "post-owner"))]
        monitor = json.loads((branch / "continuous-owner-guard.json").read_text())
        results.append({"seed": seed, "condition": condition,
                        "status": row["status"],
                        "inference_views": views,
                        "monitor_alarm": monitor["alarm"],
                        "monitor_response": monitor["response"],
                        "monitor_packet_count": len(monitor["frames"]),
                        "minimum_evaluator_gap_m": row[
                            "evaluator_path_min_clearance_m"],
                        "episode_video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                        "episode_video_sha256": sha(branch / "episode.mp4")})
    report.update({
        "schema": "robot-learning-lab.ape-owner-inference-clock.v1",
        "source_sha256": sha(SOURCE),
        "search_source_sha256": sha(Path(search.__file__)),
        "v95_reference_report_sha256": sha(
            ROOT / "artifacts/ape-closed-loop-cove50-v95/live-moving-owner-offer-audit.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "delayed_prop": {"start_s": DRIFT_START_S,
                         "duration_s": DRIFT_DURATION_S,
                         "start_y_m": drift.START_Y_M,
                         "end_y_m": drift.END_Y_M},
        "inference_budget_s": INFERENCE_BUDGET_S,
        "results": results,
        "scope": "Full furnished articulated-owner help episode with V95 source-separated 10Hz low-range monitor. A logical simulation scheduler advances moving prop and monitor in 20ms steps while the already-captured image waits for an 0.8s recognition result. Actor invalidates any raw image candidate if a sensor alarm arrives first; no point-only side step is authorized. Parked controls can finish from fresh images. This is an event-clock concurrency test, not actual concurrent inference, a physical stop, or a safe indefinite hold after no-supported-action.",
    })
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    for result in results:
        print(result["seed"], result["condition"], result["status"],
              result["monitor_alarm"],
              [(v["label"], v["capture_time_s"],
                v["inference_completion_time_s"],
                v["inference_validity"]["status"])
               for v in result["inference_views"]])


if __name__ == "__main__":
    main()
