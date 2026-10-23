"""Complete owner turns from an earlier historical 50 cm meeting standoff.

The ape is placed at a candidate approach-time station before the owner show.
The existing independent 10 Hz near-body monitor stays live while a frozen
RGB-D packet is inferred by a separate worker. This tests the held-pose branch
and changed low-prop motion; no native skill moves the ape to either station.
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
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
import run_async_owner_request_v114 as v114
import run_pre_show_reserve_screen_v100 as v100


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-standoff-owner"
STATIONS_X_M = (-.30, -.22)
SEEDS = (170, 172)
PROFILES = ("moving_prop", "hand_close", "prop_toward_standoff",
            "prop_deep_intrusion", "prop_turn_into_standoff")
PROP_APPROACH_START_S = 17.2
PROP_APPROACH_END_S = 18.2
PROP_APPROACH_END_X_M = -.20
PROP_DEEP_END_X_M = -.27
PROP_TURN_END_Y_M = .10


def prop_x(profile: str, time_s: float) -> float:
    if profile not in ("prop_toward_standoff", "prop_deep_intrusion",
                       "prop_turn_into_standoff"):
        return 0.
    fraction = min(1., max(0., (time_s-PROP_APPROACH_START_S) /
                            (PROP_APPROACH_END_S-PROP_APPROACH_START_S)))
    end_x = (PROP_DEEP_END_X_M if profile == "prop_deep_intrusion"
             else PROP_APPROACH_END_X_M)
    return end_x*fraction


def prop_y(profile: str, time_s: float) -> float:
    if profile != "prop_turn_into_standoff":
        return v100.prop_y(time_s)
    fraction = min(1., max(0., (time_s-PROP_APPROACH_START_S) /
                            (PROP_APPROACH_END_S-PROP_APPROACH_START_S)))
    return .306 + (PROP_TURN_END_Y_M-.306)*fraction


class StandoffRoom(OfferDriftRoom):
    profile = "moving_prop"

    def _forward(self) -> None:
        if hasattr(self, "model") and hasattr(self, "obstacle_xy"):
            x = prop_x(self.profile, self.time_s)
            y = float(self.obstacle_xy[3][1])
            self.obstacle_xy[3] = (x, y)
            self.model.body_pos[self.model.body("loop_stool").id] = (
                x, y, STOOL_SIZE[2])
            if self.profile == "hand_close" and self.time_s >= 16.2:
                position, size = CASES["hand_close"]
                self.model.body_pos[self.model.body("loop_box").id] = position
                self.model.geom("loop_box_geom").size[:] = size
        super()._forward()


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
    original_model = structural.structural_model
    original_room = v114.OfferDriftRoom
    original_out = v114.OUT
    original_prop_y = v114.prop_y
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    v114.OfferDriftRoom = StandoffRoom
    v114.prop_y = lambda branch, time_s: prop_y(
        StandoffRoom.profile, time_s)
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
            probe = StandoffRoom(case=tuple(prior["case"]),
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
                for profile in PROFILES:
                    StandoffRoom.profile = profile
                    v114.OUT = OUT / f"x{abs(station_x):.2f}" / profile
                    result = v114.run_one(
                        seed, "parked_control", 0., tuple(prior["case"]),
                        held_qpos, held_estimate, vision, exemplar,
                        owner_hist, {"request": "show_from_reserved_early_pose"})
                    rows.append({
                        "seed": seed,
                        "station_x_m": station_x,
                        "profile": profile,
                        "staging_packet": str(source.relative_to(ROOT)),
                        "staging_packet_sha256": sha(source),
                        "prop_trajectory_x_m": (
                            "0 until17.2, then linearly to -0.20 by18.2"
                            if profile == "prop_toward_standoff" else
                            "0 until17.2, then linearly to -0.27 by18.2"
                            if profile == "prop_deep_intrusion" else
                            "0 until17.2, then linearly to -0.20 by18.2"
                            if profile == "prop_turn_into_standoff" else "0"),
                        "prop_trajectory_y_m": (
                            "0.306 until17.2, then linearly to 0.10 by18.2"
                            if profile == "prop_turn_into_standoff" else
                            "0.306 until17.2, then linearly to 0.210 by18.0"),
                        "owner_turn": result,
                    })
                    print(seed, station_x, profile,
                          result["task_result"],
                          "gap", result["evaluator_minimum_scene_gap_m"],
                          "alarm", result["monitor"]["alarm"] is not None,
                          "inference_packets",
                          result["monitor_packets_during_first_inference"],
                          flush=True)
    finally:
        structural.structural_model = original_model
        v114.OfferDriftRoom = original_room
        v114.OUT = original_out
        v114.prop_y = original_prop_y
    report = {
        "schema": "robot-learning-lab.ape-early-standoff-owner.v1",
        "source_sha256": sha(SOURCE),
        "v114_source_sha256": sha(Path(v114.__file__)),
        "v98_report_sha256": sha(v98_path),
        "historical_visual_sha256": sha(selected_visual()),
        "generic_exemplar_sha256": sha(V149),
        "owner_color_reference_sha256": sha(V150),
        "current_design_status": pointer["status"],
        "scope": "Candidate approach-time held poses at x=-0.30/-0.22m "
                 "on the rejected historical Direct Frame. No native skill "
                 "reaches them. The V114 10Hz mounted-range monitor and "
                 "asynchronous generic RGB-D worker run through the complete "
                 "owner show with a moving low prop; a changed prop path "
                 "approaches the standoff from x=0 to -0.20/-0.27m; one "
                 "turns laterally to y=0.10m. The no-request "
                 "parked-control branch permits the owner show. Evaluator "
                 "gaps score outcomes only; no actor selected the station or "
                 "verified a full-body response route. The added prop path "
                 "is a development stress scene, not measured owner motion.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
