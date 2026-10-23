"""Turn sudden front-left coverage loss into an immediate owner-yield request.

This is a high-level response to a simulated complete packet-field outage.
The prop's cooperation remains scripted; no ape escape skill is executed.
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
import run_continuous_owner_guard_v95 as v95
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_yield_recovery_v153p as v153p
import run_standoff_range_blind_v155p as v155p
from run_standoff_front_response_v150p import FrontMonitor


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-coverage-health"
SEEDS = (170, 172)
RESPONSES_S = (.2, None)
FAULT_START_S = 17.2


class HealthMonitor(FrontMonitor):
    def capture(self) -> None:
        super().capture()
        frame = self.frames[-1]
        previous = (self.frames[-2] if len(self.frames) >= 2 else None)
        lost = bool(previous is not None and
                    previous["front_left_support"] >= 5 and
                    frame["front_left_support"] == 0)
        frame["front_left_coverage_loss_first"] = lost
        if lost and getattr(self.room, "front_alarm_s", None) is None:
            self.room.front_alarm_s = round(float(self.room.time_s), 4)
            self.alarm = {"capture_time_s": self.room.front_alarm_s,
                          "capture_index": len(self.frames)-1,
                          "side": "front_left_coverage_lost",
                          "previous_support": previous[
                              "front_left_support"]}
            self.room.motion_interrupted = True


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
    old_scan = v95.front_scan
    old_monitor = v153p.FrontMonitor
    old_out = v153p.OUT
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v153p.FrontMonitor = HealthMonitor
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
            probe = v153p.ResponsiveRoom(case=tuple(prior["case"]),
                                         pose_source="command", branch=OUT)
            try:
                shoulder = probe.model.joint(
                    "left_shoulder_pitch").qposadr[0]
            finally:
                probe.close()
            qpos[0] += v155p.STATION_X_M
            estimate[0] += v155p.STATION_X_M
            qpos[shoulder] = .6
            for response_delay in RESPONSES_S:
                removed = []

                def fault_scan(room, robot, case):
                    clouds, counts = old_scan(room, robot, case)
                    if room.time_s + 1e-6 < FAULT_START_S:
                        return clouds, counts
                    kept, missing = v155p.without_front_left(
                        clouds["four"], room.actual.copy(),
                        room.estimated.copy())
                    removed.append({"time_s": round(float(room.time_s), 4),
                                    "removed_front_left_returns": missing})
                    clouds["four"] = kept
                    counts["injected_front_left_removed"] = missing
                    return clouds, counts

                v95.front_scan = fault_scan
                v153p.OUT = OUT / ("ignored" if response_delay is None
                                   else "stop-0.2s")
                row = v153p.run_one(
                    seed, v155p.STATION_X_M, response_delay,
                    tuple(prior["case"]), qpos, estimate,
                    vision, exemplar, owner_hist)
                row["fault_start_s"] = FAULT_START_S
                row["coverage_event_rule"] = (
                    "previous packet had >=5 front-left returns; current "
                    "packet has zero")
                row["removed_by_packet"] = removed
                row["staging_packet"] = str(source.relative_to(ROOT))
                row["staging_packet_sha256"] = sha(source)
                rows.append(row)
                print(seed, response_delay, row["front_alarm"],
                      row["task_result"],
                      row["evaluator_minimum_scene_gap_m"], flush=True)
    finally:
        v95.front_scan = old_scan
        v153p.FrontMonitor = old_monitor
        v153p.OUT = old_out
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-standoff-coverage-health.v1",
        "source_sha256": sha(SOURCE),
        "v155_source_sha256": sha(Path(v155p.__file__)),
        "v153_source_sha256": sha(Path(v153p.__file__)),
        "v98_report_sha256": sha(v98_path),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "scope": "Complete furnished held owner turn with front-left actor "
                 "packet-field loss at17.2s. Sudden loss after a supported "
                 "packet becomes an immediate high-level request; the "
                 "prop either stops 0.2s after that request or ignores it. "
                 "The ape remains stationary and the range loss persists, "
                 "so no fresh item result is permitted. Evaluator body gap "
                 "is scoring only. This is neither a physical fault model "
                 "nor a selected-body rescue.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
