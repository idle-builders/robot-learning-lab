"""Inject front-left range blindness into the complete V153p owner turn.

The scene and detector are unchanged. The actor receives packets with every
front-left low return removed after a declared time, as if that field of view
became occluded or unavailable. This is an adversarial packet fault, not an
optical model of a particular physical device.
"""

from __future__ import annotations

import json
import math
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


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-range-blind-owner"
SEEDS = (170, 172)
FAULT_STARTS_S = (17.2, 17.4)
STATION_X_M = -.30
RESPONSE_DELAY_S = .2


def without_front_left(points: np.ndarray, actual: np.ndarray,
                       estimate: np.ndarray) -> tuple[np.ndarray, int]:
    estimated = v95.estimated_shin_points(points, actual, estimate)
    delta = estimated[:, :2] - estimate[:2]
    c, s = math.cos(estimate[2]), math.sin(estimate[2])
    x = c*delta[:, 0] + s*delta[:, 1]
    y = -s*delta[:, 0] + c*delta[:, 1]
    z = estimated[:, 2]
    in_view = ((x > .08) & (x < .50) & (y > .05) & (y < .50) &
               (z > .015) & (z < .10))
    return points[~in_view], int(np.count_nonzero(in_view))


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
    old_out = v153p.OUT
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v153p.OUT = OUT
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
            held_qpos = qpos.copy()
            held_estimate = estimate.copy()
            held_qpos[0] += STATION_X_M
            held_estimate[0] += STATION_X_M
            held_qpos[shoulder] = .6
            for fault_start in FAULT_STARTS_S:
                removed = []
                v153p.OUT = OUT / f"blind-from-{fault_start:.1f}s"

                def fault_scan(room, robot, case):
                    clouds, counts = old_scan(room, robot, case)
                    if room.time_s + 1e-6 < fault_start:
                        return clouds, counts
                    kept, missing = without_front_left(
                        clouds["four"], room.actual.copy(),
                        room.estimated.copy())
                    removed.append({"time_s": round(float(room.time_s), 4),
                                    "removed_front_left_returns": missing})
                    clouds["four"] = kept
                    counts["injected_front_left_removed"] = missing
                    return clouds, counts

                v95.front_scan = fault_scan
                row = v153p.run_one(
                    seed, STATION_X_M, RESPONSE_DELAY_S,
                    tuple(prior["case"]), held_qpos, held_estimate,
                    vision, exemplar, owner_hist)
                row["fault_start_s"] = fault_start
                row["injected_fault"] = "all front-left ROI returns removed"
                row["removed_by_packet"] = removed
                row["staging_packet"] = str(source.relative_to(ROOT))
                row["staging_packet_sha256"] = sha(source)
                rows.append(row)
                print(seed, fault_start, "front cues",
                      [f["capture_time_s"] for f in json.loads(
                          (ROOT / row["monitor"]["path"]).read_text())[
                              "frames"] if f.get("front_left_first_cue")],
                      "result", row["task_result"],
                      "min gap", row["evaluator_minimum_scene_gap_m"],
                      flush=True)
    finally:
        v95.front_scan = old_scan
        structural.structural_model = old_model
        v153p.OUT = old_out
    report = {
        "schema": "robot-learning-lab.ape-standoff-range-blind.v1",
        "source_sha256": sha(SOURCE),
        "v153_source_sha256": sha(Path(v153p.__file__)),
        "v98_report_sha256": sha(v98_path),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "fault": "Remove all front-left low-range point returns from the actor packet "
                 "starting at 17.2s (before the baseline 17.3s warning) or "
                 "17.4s (after the warning). The whole furnished room, "
                 "articulated owner, camera worker and scripted prop path "
                 "remain otherwise unchanged. This is a packet-field "
                 "occlusion stress, not physical optics or a probability "
                 "model. Evaluator gap only scores the result.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
