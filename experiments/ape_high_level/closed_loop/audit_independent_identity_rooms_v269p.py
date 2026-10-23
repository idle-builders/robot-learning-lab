"""Integrity and owner-claim audit for the frozen six-case V268p split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco

from audit_medium_depth_recovery_v247p import check_packet
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, iou, model_with_room
from screen_stopped_alternate_routes_v255p import replay_stage


OUT = ROOT / "artifacts/ape-articulated-depth-route-v268p"
EXPECTED = (186, 188, 190, 192, 194, 196)
SUCCESS = {"owner_confirmed_visual_only",
           "owner_confirmed_forward_visual_only",
           "owner_assisted_visual_reacquisition"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        return sum(1 for frame in reader if frame.ndim == 3 and
                   frame.shape[2] == 3)


def main() -> None:
    declaration_path = OUT / "predeclared-challenge.json"
    preflight_path = OUT / "geometry-preflight.json"
    report_path = OUT / "blocked-forward-viewpoint-audit.json"
    declaration = json.loads(declaration_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    report = json.loads(report_path.read_text())
    wrapper = json.loads((OUT / "report.json").read_text())
    assert wrapper["declaration_sha256"] == digest(declaration_path)
    assert declaration["source_sha256"] == digest(
        OUT / "run_independent_identity_rooms_v268p.py")
    assert wrapper["geometry_preflight_sha256"] == digest(preflight_path)
    assert wrapper["runner_report_sha256"] == digest(report_path)
    assert declaration["historical_visual_sha256"] == digest(selected_visual())
    assert tuple(row["case"][0] for row in preflight["rows"]) == EXPECTED
    assert tuple(row["case"][0] for row in report["rows"]) == EXPECTED
    model = model_with_room(visual=selected_visual())
    data = mujoco.MjData(model)
    robot = RobotGeometry(model)
    rows = []
    head_packets = 0
    rejected_wrong = 0
    wrong_claims = 0
    for row in report["rows"]:
        seed, mode = row["case"][:2]
        assert digest(ROOT / row["trace_path"]) == row["trace_sha256"]
        assert digest(ROOT / row["video_path"]) == row["video_sha256"]
        trace = json.loads((ROOT / row["trace_path"]).read_text())
        assert trace["case"] == row["case"]
        assert trace["evaluator_motion_interrupted"] == row[
            "evaluator_motion_interrupted"]
        assert row["evaluator_path_min_clearance_m"] >= .05
        for entry in trace["side_probe_inputs"]:
            check_packet(ROOT / entry["actor_frame_path"],
                         entry["actor_frame_sha256"],
                         "medium_30cm_1m", model, data)
            head_packets += 1
        for recovery in trace["recovery_inputs"]:
            for entry in recovery["head_inputs"]:
                check_packet(ROOT / entry["actor_frame_path"],
                             entry["actor_frame_sha256"],
                             "medium_30cm_1m", model, data)
                head_packets += 1
            stages = recovery.get("continuation", {}).get("stages", [])
            for stage in stages:
                for entry in stage["head_inputs"]:
                    check_packet(ROOT / entry["actor_frame_path"],
                                 entry["actor_frame_sha256"],
                                 "medium_30cm_1m", model, data)
                    head_packets += 1
            help_event = (recovery.get("initial_veto") or
                          recovery.get("continuation", {}).get("owner_help"))
            if help_event is not None:
                stopped_stage = recovery if "initial_veto" in recovery else \
                    stages[-1]
                alternate = replay_stage(stopped_stage, model, robot)
                assert all(option["status"] == "blocked" for option in
                           alternate["options"].values())
                assert help_event["base_moves_during_help"] == 0
                assert help_event["claim"] == "item_not_found_no_supported_view"
                assert help_event["status"] == row["status"]
        events = trace["controller_events"]
        wrong_candidates = [event for event in events
                            if event["action"] == "view" and
                            event["accepted_box"] is not None and
                            not event["evaluator_correct_item"]]
        if row["status"] in SUCCESS:
            valid = mode == "present"
            if row["status"] == "owner_assisted_visual_reacquisition":
                views = [event for event in events if event["action"] == "view"]
                valid &= (views[-2]["label"] ==
                          "owner-presented fresh candidate" and
                          views[-2]["evaluator_correct_item"] and
                          views[-1]["label"] == "post-owner fresh recheck" and
                          views[-1]["accepted_box"] is not None and
                          iou(views[-2]["accepted_box"],
                              views[-1]["accepted_box"]) >= .6)
            if not valid:
                wrong_claims += 1
        elif row["status"].startswith("owner_rejected"):
            rejected_wrong += 1
        count = frames(ROOT / row["video_path"])
        assert count > 5
        rows.append({"seed": seed, "mode": mode,
                     "status": row["status"],
                     "owner_questions": row["owner_questions"],
                     "base_moves": row["base_moves"],
                     "wrong_candidate_views": len(wrong_candidates),
                     "minimum_modeled_clearance_m": row[
                         "evaluator_path_min_clearance_m"],
                     "video_frames": count})
    audit = {"schema": "robot-learning-lab.independent-identity-room-audit.v1",
             "report_sha256": digest(report_path),
             "head_packets_pose_checked": head_packets,
             "videos_decoded": len(rows),
             "wrong_success_claims": wrong_claims,
             "owner_rejected_cases": rejected_wrong,
             "all_integrity_checks_pass": True,
             "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
