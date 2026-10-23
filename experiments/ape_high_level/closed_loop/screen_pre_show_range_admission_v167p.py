"""Observe front-left coverage before inviting the owner to show an item.

This reconstructs a stationary 15.4–16.2s pre-show scene from the saved V98
staging pose and scans it at 10Hz. It is a planning preflight, not the same
live simulator process as V160's subsequent full owner turn. A provisional
coverage decision is scored against V165's later warning replay.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
from audit_shin_range_v190 import scan as front_scan
import direct_structural_rooms_v270 as structural
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_continuous_owner_guard_v95 as v95
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
from screen_standoff_front_cue_v149p import front_left_stat
import compare_standoff_range_layouts_v165p as v165p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-pre-show-admission"
SCAN_TIMES_S = tuple(round(15.4+.1*i, 4) for i in range(9))
LAYOUTS = ("two_forward", "four_front_side")
OCCLUSIONS = ("clear_view", "forward_pair_occluded")
MIN_SUPPORTED_OF_NINE = 5


def scan_one(room, robot: RobotGeometry, branch: Path,
             variant_name: str, seed: int, station_x: float,
             index: int, time_s: float) -> dict:
    room.time_s = time_s
    set_person_pose(room.model, START_HAND_XYZ_M,
                    carried_item="loop_TomatoSauce")
    room._forward()
    clouds, counts = front_scan(room, robot, {
        "case": f"v167-{variant_name}-seed{seed}-x{station_x:.2f}"
                f"-packet{index}"})
    points = v95.estimated_shin_points(
        clouds["four"], room.actual.copy(), room.estimated.copy())
    forward_count = counts.get("two_environment_points", 0)
    if len(points) != (forward_count+counts.get(
            "four_environment_points", 0)):
        raise AssertionError("Front-source split mismatch")
    path = branch / f"pre-show-{index:02d}.npz"
    np.savez_compressed(path, front_four_points_xyz_m=points,
                        forward_source_count=forward_count,
                        estimated_pose_xyyaw=room.estimated.copy(),
                        actor_qpos=room.data.qpos.copy(),
                        capture_time_s=time_s)
    return {"time_s": time_s,
            "packet_path": str(path.relative_to(ROOT)),
            "packet_sha256": sha(path),
            "front_counts": counts,
            "forward_count": forward_count}


def decision(packets: list[dict], layout: str,
             occlusion: str) -> dict:
    samples = []
    for source in packets:
        with np.load(ROOT / source["packet_path"], allow_pickle=False) as data:
            front = data["front_four_points_xyz_m"]
            pose = data["estimated_pose_xyyaw"]
            count = int(data["forward_source_count"])
        ids = np.arange(len(front))
        if occlusion == "forward_pair_occluded":
            blocked = v165p.front_left_roi(front, pose) & (ids < count)
            ids = ids[~blocked]
        if layout == "two_forward":
            ids = ids[ids < count]
        support, p10 = front_left_stat(front[ids], pose)
        samples.append({"time_s": source["time_s"],
                        "front_left_support": support,
                        "front_left_p10_x_m": p10})
    supported = sum(s["front_left_support"] >= 5 for s in samples)
    return {
        "supported_packets": supported,
        "latest_packet_supported": samples[-1]["front_left_support"] >= 5,
        "provisional_admission": (supported >= MIN_SUPPORTED_OF_NINE and
                                  samples[-1]["front_left_support"] >= 5),
        "samples": samples,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    future_file = v165p.OUT / "report.json"
    future = json.loads(future_file.read_text())
    future_by_key = {(r["source_case_id"], r["stress"], r["layout"]): r
                     for r in future["rows"] if r["draw"] == 0}
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for variant_name, variant in v160p.VARIANTS.items():
            v160p.VariantRoom.variant = variant
            for seed in v160p.SEEDS:
                prior = next(row for row in v98["rows"]
                             if row["condition"] == "moving" and
                             row["case"][0] == seed)
                staging = (v100.V98 / "moving" / f"seed{seed}-present" /
                           f"seed{seed}-present/side-probe-00.npz")
                with np.load(staging, allow_pickle=False) as data:
                    base_qpos = data["actor_qpos"].copy()
                    base_estimate = data["estimated_pose_xyyaw"].copy()
                for station_x in v160p.STATIONS_X_M:
                    branch = (OUT / variant_name /
                              f"x{abs(station_x):.2f}" / f"seed{seed}")
                    branch.mkdir(parents=True, exist_ok=True)
                    room = v160p.VariantRoom(
                        case=tuple(prior["case"]),
                        pose_source="command", branch=branch)
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        qpos = base_qpos.copy()
                        estimate = base_estimate.copy()
                        qpos[0] += station_x
                        estimate[0] += station_x
                        shoulder = room.model.joint(
                            "left_shoulder_pitch").qposadr[0]
                        qpos[shoulder] = .6
                        room.data.qpos[:] = qpos
                        room.actual[:] = (qpos[0], qpos[1],
                            2*math.atan2(qpos[6], qpos[3]))
                        room.estimated[:] = estimate
                        room.head = float(qpos[room.head_addr])
                        robot = RobotGeometry(room.model)
                        packets = [scan_one(
                            room, robot, branch, variant_name, seed,
                            station_x, i, time_s) for i, time_s in
                            enumerate(SCAN_TIMES_S)]
                    finally:
                        room.close()
                    for layout in LAYOUTS:
                        for occlusion in OCCLUSIONS:
                            result = decision(packets, layout, occlusion)
                            source_id = (
                                f"hazard-{variant_name}-seed{seed}"
                                f"-x{abs(station_x):.2f}")
                            stress = ("saved" if occlusion == "clear_view"
                                      else "forward_pair_occluded")
                            future_row = future_by_key[
                                (source_id, stress, layout)]
                            rows.append({
                                "variant": variant_name,
                                "seed": seed,
                                "station_x_m": station_x,
                                "layout": layout,
                                "occlusion": occlusion,
                                "staging_packet": str(staging.relative_to(ROOT)),
                                "staging_packet_sha256": sha(staging),
                                "pre_show_packets": packets,
                                "pre_show_decision": result,
                                "later_saved_source_case_id": source_id,
                                "later_packet_replay_first_cue_s": future_row[
                                    "first_front_left_cue_s"],
                                "later_packet_replay_timely_warning": future_row[
                                    "timely_hazard_warning"],
                                "later_evaluator_5cm_crossing_s": future_row[
                                    "evaluator_first_below_5cm_s"],
                            })
                            print(variant_name, seed, station_x, layout,
                                  occlusion, result["supported_packets"],
                                  result["provisional_admission"],
                                  future_row["timely_hazard_warning"],
                                  flush=True)
    finally:
        structural.structural_model = old_model
    summary = {}
    for layout in LAYOUTS:
        for occlusion in OCCLUSIONS:
            group = [row for row in rows if row["layout"] == layout and
                     row["occlusion"] == occlusion]
            summary[f"{layout}/{occlusion}"] = {
                "cases": len(group),
                "provisional_admissions": sum(row["pre_show_decision"][
                    "provisional_admission"] for row in group),
                "admitted_without_later_timely_warning": sum(
                    row["pre_show_decision"]["provisional_admission"] and
                    not row["later_packet_replay_timely_warning"]
                    for row in group),
                "deferred_despite_later_timely_warning": sum(
                    not row["pre_show_decision"]["provisional_admission"] and
                    row["later_packet_replay_timely_warning"]
                    for row in group),
            }
    report = {
        "schema": "robot-learning-lab.ape-pre-show-range-admission.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(v98_file),
        "v165_report_sha256": sha(future_file),
        "current_design_status": pointer["status"],
        "scan_times_s": SCAN_TIMES_S,
        "layout_options": LAYOUTS,
        "occlusions": OCCLUSIONS,
        "provisional_rule": "At least 5 of 9 pre-show packets have >=5 "
                            "front-left returns, and the latest packet "
                            "also has >=5 returns.",
        "scope": "Reconstructed stationary pre-show scene at a historical held "
                 "standoff; nine fresh virtual first-hit 10Hz range scans "
                 "per scene with independent scanner noise. The actor "
                 "would admit or defer the owner invitation before motion. "
                 "Later warning labels are joined from the separately "
                 "saved V165 packet replay and are evaluator-only for this "
                 "preflight analysis; no owner branch is executed or "
                 "physically protected by this decision. The rule was "
                 "chosen after seeing development stress evidence, not "
                 "validated on an independent room/body distribution.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
