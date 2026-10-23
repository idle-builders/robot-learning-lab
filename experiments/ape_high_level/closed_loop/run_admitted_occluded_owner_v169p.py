"""Continue only pre-show-admitted owner turns under forward-pair occlusion.

V167p reconstructs the stationary pre-show sensing period; this runner joins
its actor decision to a fresh complete V153p owner turn from the same staged
pose. The simulator process is new at the handoff, so this is a
sensor-equivalent planning continuation rather than one uninterrupted process.
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
import run_standoff_scene_variants_v160p as v160p
import run_standoff_yield_recovery_v153p as v153p
from compare_standoff_range_layouts_v165p import front_left_roi
import screen_pre_show_range_admission_v167p as v167p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-admitted-occluded-owner"
RESPONSES_S = (.2, None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    preflight_file = v167p.OUT / "report.json"
    preflight = json.loads(preflight_file.read_text())
    decisions = [row for row in preflight["rows"]
                 if row["layout"] == "four_front_side" and
                 row["occlusion"] == "forward_pair_occluded"]
    admitted = [row for row in decisions if row["pre_show_decision"][
        "provisional_admission"]]
    if len(decisions) != 8 or len(admitted) != 2:
        raise AssertionError("Pre-show admission matrix changed")
    v98_path = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_path.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old_model = structural.structural_model
    old_scan = v95.front_scan
    old_room = v153p.ResponsiveRoom
    old_out = v153p.OUT
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v153p.ResponsiveRoom = v160p.VariantRoom
    v160p.VariantRoom.variant = v160p.VARIANTS["fast_wide_turn"]
    v153p.OUT = OUT / "admitted-four-front-side"
    rows = []
    try:
        for admission in admitted:
            seed = admission["seed"]
            station_x = admission["station_x_m"]
            variant = admission["variant"]
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = ROOT / admission["staging_packet"]
            with np.load(source, allow_pickle=False) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            probe = v160p.VariantRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT)
            try:
                shoulder = probe.model.joint(
                    "left_shoulder_pitch").qposadr[0]
            finally:
                probe.close()
            qpos[0] += station_x
            estimate[0] += station_x
            qpos[shoulder] = .6
            for delay in RESPONSES_S:
                removed = []

                def occluded_scan(room, robot, case):
                    clouds, counts = old_scan(room, robot, case)
                    world = clouds["four"]
                    first_count = len(clouds["two"])
                    estimated = v95.estimated_shin_points(
                        world, room.actual.copy(), room.estimated.copy())
                    blocked = (front_left_roi(
                        estimated, room.estimated.copy()) &
                        (np.arange(len(world)) < first_count))
                    removed_count = int(np.count_nonzero(blocked))
                    kept = world[~blocked]
                    clouds["two"] = clouds["two"][~blocked[:first_count]]
                    clouds["four"] = kept
                    counts["two_environment_points"] = len(clouds["two"])
                    counts["injected_forward_pair_roi_removed"] = (
                        removed_count)
                    removed.append({
                        "time_s": round(float(room.time_s), 4),
                        "removed_forward_roi_returns": removed_count,
                    })
                    return clouds, counts

                v95.front_scan = occluded_scan
                result = v153p.run_one(
                    seed, station_x, delay, tuple(prior["case"]),
                    qpos, estimate, vision, exemplar, owner_hist)
                result.update({
                    "variant": variant,
                    "pre_show_layout": admission["layout"],
                    "pre_show_occlusion": admission["occlusion"],
                    "pre_show_supported_packets": admission[
                        "pre_show_decision"]["supported_packets"],
                    "pre_show_packet_paths": admission["pre_show_packets"],
                    "staging_packet": admission["staging_packet"],
                    "staging_packet_sha256": admission[
                        "staging_packet_sha256"],
                    "removed_by_live_packet": removed,
                })
                rows.append(result)
                print(seed, delay, result["front_alarm"],
                      result["task_result"],
                      result["evaluator_minimum_scene_gap_m"], flush=True)
    finally:
        v95.front_scan = old_scan
        v153p.ResponsiveRoom = old_room
        v153p.OUT = old_out
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-admitted-occluded-owner.v1",
        "source_sha256": sha(SOURCE),
        "v167_report_sha256": sha(preflight_file),
        "v153_source_sha256": sha(Path(v153p.__file__)),
        "v160_source_sha256": sha(Path(v160p.__file__)),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "admission_decisions": {
            "candidate_scenes": len(decisions),
            "admitted_scenes": len(admitted),
            "deferred_without_owner_turn": len(decisions)-len(admitted),
        },
        "scope": "A reconstructed nine-packet pre-show admission is joined "
                 "at the same staged historical pose to a new uninterrupted "
                 "owner-turn process for the two admitted four-view scenes. "
                 "The forward pair remains occluded in every live actor "
                 "packet. Each scene runs scripted 0.2s prop halt and "
                 "ignored-response branches. Deferred scenes do not run "
                 "an owner turn; this says nothing about physical safety "
                 "if independent clutter continues moving. The stop is a "
                 "scene script, not selected-body action. Evaluator gap "
                 "never enters actor decisions.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
