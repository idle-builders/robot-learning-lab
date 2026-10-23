"""V105: body-relative range evidence for the pre-show bad-start veto.

This paired packet diagnostic does not authorize the rear route. It checks
whether a hypothetical upper side array can distinguish a close arm-height
block at the start under local pose errors, without evaluator labels in rays.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
import audit_blocked_range_mounts_v319 as sensor
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import START_S, V98
from run_pre_show_sensor_coverage_v102 import (FOV_H_DEG, FOV_V_DEG,
                                               GRID_H, GRID_V, MOUNTS)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v105"
PERTURBATIONS = {"nominal": 0., "right_1cm": -.01, "right_2cm": -.02}
SENSORS = ("right_upper", "right_low", "rear_mid_near", "rear_low_near")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    old_mounts = sensor.MOUNTS.copy()
    old_h = sensor.HORIZONTAL_DEG.copy()
    old_v = sensor.VERTICAL_DEG.copy()
    structural.structural_model = lambda kind, side: add_person(old_model(kind, side))
    sensor.MOUNTS.clear()
    sensor.MOUNTS.update(MOUNTS)
    sensor.HORIZONTAL_DEG = np.linspace(-FOV_H_DEG/2, FOV_H_DEG/2, GRID_H)
    sensor.VERTICAL_DEG = np.linspace(-FOV_V_DEG/2, FOV_V_DEG/2, GRID_V)
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            for case, obstacle in CASES.items():
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT / "scene")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    for label, dy in PERTURBATIONS.items():
                        actual_qpos = qpos.copy()
                        actual_qpos[1] += dy
                        room.data.qpos[:] = actual_qpos
                        room.actual[:] = (actual_qpos[0], actual_qpos[1],
                                          2*math.atan2(actual_qpos[6], actual_qpos[3]))
                        room.estimated[:] = estimate
                        room.head = float(actual_qpos[room.head_addr])
                        room.time_s = START_S
                        set_person_pose(room.model, START_HAND_XYZ_M,
                                        carried_item="loop_TomatoSauce")
                        room.obstacle_xy[3] = (0., .306)
                        room.model.body_pos[room.model.body("loop_stool").id] = (
                            0., .306, STOOL_SIZE[2])
                        room._forward()
                        gap, robot_part, obstacle_part = room._current_obstacle_clearance()
                        packets = {}
                        metrics = {}
                        for name in SENSORS:
                            rays, counts = sensor.range_packet(
                                room, estimate, name, seed, START_S)
                            packets[name] = rays
                            hits = [float(ray["measured_range_m"]) for ray in rays
                                    if ray["endpoint_hit"]]
                            metrics[name] = {
                                "min_hit_m": round(min(hits), 5) if hits else None,
                                "hits_under_10cm": sum(value < .10 for value in hits),
                                "hits_under_20cm": sum(value < .20 for value in hits),
                                "under_10cm_fraction_of_valid": round(
                                    sum(value < .10 for value in hits) /
                                    max(1, counts["valid"]), 5),
                                "valid_rays": counts["valid"],
                                "dropout_rays": counts["dropout"],
                            }
                        branch = OUT / f"seed{seed}-{case}-{label}"
                        branch.mkdir(parents=True, exist_ok=True)
                        packet_path = branch / "actor-range.json"
                        packet_path.write_text(json.dumps({
                            "capture_time_s": START_S,
                            "estimated_pose_xyyaw": estimate.tolist(),
                            "sensors": SENSORS,
                            "ray_packets": packets,
                        }, indent=2) + "\n")
                        rows.append({
                            "seed": seed, "case": case, "actual_start_offset_y_m": dy,
                            "perturbation": label,
                            "evaluator_start_gap_m": round(float(gap), 5),
                            "evaluator_closest_pair": [robot_part, obstacle_part],
                            "actor_packet_path": str(packet_path.relative_to(ROOT)),
                            "actor_packet_sha256": sha(packet_path),
                            "source_packet_path": str(source.relative_to(ROOT)),
                            "source_packet_sha256": sha(source),
                            "actor_range_metrics": metrics,
                        })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(old_mounts)
        sensor.HORIZONTAL_DEG = old_h
        sensor.VERTICAL_DEG = old_v
    report = {
        "schema": "robot-learning-lab.ape-pre-show-local-range.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "mounts": {name: MOUNTS[name] for name in SENSORS},
        "fov_deg_h_v": (FOV_H_DEG, FOV_V_DEG),
        "grid_h_v": (GRID_H, GRID_V),
        "perturbations_y_m": PERTURBATIONS,
        "scope": "Paired single-packet 16.2s historical-body range screen for four obstacle cases, two lighting seeds and three actual lateral starts. Estimated pose remains fixed; ray packets have 5mm noise and 15% dropout. Evaluator full-body start gap is used only for scoring. Hypothetical mounts, no route authorization or physical sensor qualification.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"], row["perturbation"],
              row["evaluator_start_gap_m"],
              row["actor_range_metrics"]["right_upper"]["min_hit_m"],
              row["actor_range_metrics"]["right_upper"]["hits_under_10cm"])
    print(path)


if __name__ == "__main__":
    main()
