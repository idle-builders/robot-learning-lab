"""Join front-left range warning to scripted prop-yield response timing.

This reuses the concurrent RGB-D/10 Hz range owner turn. The new actor cue
uses the same low-range packets; cooperative prop stops are scene scripts.
The ape itself remains a held historical kinematic body.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import add_person
from audit_direct_structural_entry_v271 import V149, V150
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_continuous_owner_guard_v95 import OwnerPhaseMonitor
from run_moving_low_clutter_v302 import STOOL_SIZE
import run_async_owner_request_v114 as v114
import run_early_standoff_owner_v148p as v148p
import run_pre_show_reserve_screen_v100 as v100
from screen_standoff_front_cue_v149p import (
    MIN_CLOSING_M_PER_0P2S, front_left_stat)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-front-response"
SEEDS = (170, 172)
STATIONS_X_M = (-.30, -.22)
RESPONSE_DELAYS_S = (0.2, 0.5, None)


class ResponsiveRoom(v148p.StandoffRoom):
    profile = "prop_turn_into_standoff"
    response_delay_s: float | None = None

    def _forward(self) -> None:
        if hasattr(self, "model") and hasattr(self, "obstacle_xy"):
            time_s = float(self.time_s)
            alarm_s = getattr(self, "front_alarm_s", None)
            if alarm_s is not None and self.response_delay_s is not None:
                time_s = min(time_s, alarm_s+self.response_delay_s)
            x = v148p.prop_x(self.profile, time_s)
            y = v148p.prop_y(self.profile, time_s)
            self.obstacle_xy[3] = (x, y)
            self.model.body_pos[self.model.body("loop_stool").id] = (
                x, y, STOOL_SIZE[2])
        # Skip StandoffRoom's unconditional prop motion after a scripted stop.
        super(v148p.StandoffRoom, self)._forward()


class FrontMonitor(OwnerPhaseMonitor):
    def __init__(self, room, branch: Path, seed: int) -> None:
        self.front_history: list[tuple[int, float | None]] = []
        super().__init__(room, branch, seed)

    def capture(self) -> None:
        super().capture()
        points = self.points[-1][0]
        support, p10 = front_left_stat(points, self.room.estimated.copy())
        self.front_history.append((support, p10))
        closing = None
        cue = False
        if len(self.front_history) >= 3:
            older = self.front_history[-3][1]
            if older is not None and p10 is not None:
                closing = round(older-p10, 5)
                cue = closing >= MIN_CLOSING_M_PER_0P2S
        first_cue = cue and getattr(self.room, "front_alarm_s", None) is None
        if first_cue:
            self.room.front_alarm_s = round(float(self.room.time_s), 4)
            self.alarm = {"capture_time_s": self.room.front_alarm_s,
                          "capture_index": len(self.frames)-1,
                          "side": "front_left",
                          "closing_0p2s_m": closing}
            self.room.motion_interrupted = True
        self.frames[-1].update({
            "front_left_support": support,
            "front_left_p10_x_m": p10,
            "front_left_closing_0p2s_m": closing,
            "front_left_cue_now": cue,
            "front_left_first_cue": first_cue,
        })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    v98_path = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_path.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old_model = structural.structural_model
    old_room = v114.OfferDriftRoom
    old_monitor = v114.OwnerPhaseMonitor
    old_out = v114.OUT
    old_prop_y = v114.prop_y
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v114.OfferDriftRoom = ResponsiveRoom
    v114.OwnerPhaseMonitor = FrontMonitor
    v114.prop_y = lambda branch, time_s: v148p.prop_y(
        ResponsiveRoom.profile, time_s)
    rows = []
    try:
        for seed in SEEDS:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (v100.V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/side-probe-00.npz")
            with np.load(source, allow_pickle=False) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            probe = ResponsiveRoom(case=tuple(prior["case"]),
                                   pose_source="command", branch=OUT)
            try:
                shoulder = probe.model.joint(
                    "left_shoulder_pitch").qposadr[0]
            finally:
                probe.close()
            for station_x in STATIONS_X_M:
                held_qpos = qpos.copy()
                held_estimate = estimate.copy()
                held_qpos[0] += station_x
                held_estimate[0] += station_x
                held_qpos[shoulder] = .6
                for response_delay in RESPONSE_DELAYS_S:
                    ResponsiveRoom.response_delay_s = response_delay
                    label = ("ignored" if response_delay is None else
                             f"stop-{response_delay:.1f}s")
                    v114.OUT = OUT / f"x{abs(station_x):.2f}" / label
                    result = v114.run_one(
                        seed, "parked_control", 0., tuple(prior["case"]),
                        held_qpos, held_estimate, vision, exemplar,
                        owner_hist, {"request": "yield_after_front_left_alert"})
                    rows.append({
                        "seed": seed,
                        "station_x_m": station_x,
                        "response": label,
                        "scripted_prop_stop_delay_s": response_delay,
                        "staging_packet": str(source.relative_to(ROOT)),
                        "staging_packet_sha256": sha(source),
                        "owner_turn": result,
                    })
                    print(seed, station_x, label,
                          "front alarm", result["monitor"]["alarm"],
                          "first5cm", result["evaluator_first_below_5cm_s"],
                          "minimum", result["evaluator_minimum_scene_gap_m"],
                          "task", result["task_result"], flush=True)
    finally:
        structural.structural_model = old_model
        v114.OfferDriftRoom = old_room
        v114.OwnerPhaseMonitor = old_monitor
        v114.OUT = old_out
        v114.prop_y = old_prop_y
    report = {
        "schema": "robot-learning-lab.ape-standoff-front-response.v1",
        "source_sha256": sha(SOURCE),
        "v148_source_sha256": sha(Path(v148p.__file__)),
        "v149_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/"
                                  "screen_standoff_front_cue_v149p.py"),
        "v98_report_sha256": sha(v98_path),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "scope": "Online replay of the V149 front-left cue on the same "
                 "historical 50cm furnished-owner scene. At first alert a "
                 "scripted cooperative prop stops after0.2s or0.5s; the "
                 "ignored branch keeps moving. Owner show and asynchronous "
                 "RGB-D inference proceed under a continuously sampled 10Hz "
                 "low-range monitor. No message is sent to a real person, "
                 "no robot response is executed, the actor has not selected "
                 "or reached the held station, and evaluator body gaps are "
                 "not available to the actor. The owner-result alarm remains "
                 "latched, so a safe stopped prop does not automatically "
                 "complete the social task.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
