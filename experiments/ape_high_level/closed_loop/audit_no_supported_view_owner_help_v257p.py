"""Pair V256p with V252p and check the no-route/help outcome boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_repeated_viewpoints_v253p import same_capture
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, iou, model_with_room
from screen_stopped_alternate_routes_v255p import replay_stage


BASE = ROOT / "artifacts/ape-articulated-depth-route-v252p"
OUT = ROOT / "artifacts/ape-articulated-depth-route-v256p"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_entries(trace: dict) -> list[dict]:
    recovery = trace["recovery_inputs"][0]
    return (trace["side_probe_inputs"] + recovery["head_inputs"] +
            [entry for stage in recovery["continuation"]["stages"]
             for entry in stage["head_inputs"]])


def frame_count(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        return sum(1 for frame in reader if frame.ndim == 3 and
                   frame.shape[2] == 3)


def main() -> None:
    old_report_path = BASE / "blocked-forward-viewpoint-audit.json"
    new_report_path = OUT / "blocked-forward-viewpoint-audit.json"
    old_report = json.loads(old_report_path.read_text())
    report = json.loads(new_report_path.read_text())
    wrapper = json.loads((OUT / "report.json").read_text())
    assert wrapper["runner_report_sha256"] == digest(new_report_path)
    assert wrapper["source_sha256"] == digest(ROOT /
        "experiments/ape_high_level/closed_loop/run_no_supported_view_owner_help_v256p.py")
    assert len(old_report["rows"]) == len(report["rows"]) == 4
    model = model_with_room(visual=selected_visual())
    robot = RobotGeometry(model)
    paired_captures = 0
    rows = []
    for old, new in zip(old_report["rows"], report["rows"], strict=True):
        assert old["case"] == new["case"]
        assert digest(ROOT / old["trace_path"]) == old["trace_sha256"]
        assert digest(ROOT / new["trace_path"]) == new["trace_sha256"]
        assert digest(ROOT / new["video_path"]) == new["video_sha256"]
        previous = json.loads((ROOT / old["trace_path"]).read_text())
        trace = json.loads((ROOT / new["trace_path"]).read_text())
        assert previous["slips_evaluator_only"] == trace[
            "slips_evaluator_only"]
        assert old["base_moves"] == new["base_moves"] == 7
        for a, b in zip(capture_entries(previous), capture_entries(trace),
                        strict=True):
            assert digest(ROOT / b["actor_frame_path"]) == b[
                "actor_frame_sha256"]
            same_capture(ROOT / a["actor_frame_path"],
                         ROOT / b["actor_frame_path"])
            paired_captures += 1
        recovery = trace["recovery_inputs"][0]
        assert [stage["actor_gate"]["status"] for stage in
                recovery["continuation"]["stages"]] == ["free", "blocked"]
        alternative = replay_stage(
            recovery["continuation"]["stages"][1], model, robot)
        assert all(route["status"] == "blocked" for route in
                   alternative["options"].values())
        help_event = recovery["continuation"]["owner_help"]
        assert help_event["base_moves_during_help"] == 0
        assert help_event["claim"] == "item_not_found_no_supported_view"
        assert help_event["status"] == new["status"]
        resolution = next(event for event in trace["controller_events"]
                          if event["action"] == "route_help_resolution")
        assert resolution["status"] == new["status"]
        assert resolution["owner_questions"] == new["owner_questions"]
        assert len(resolution["actor_gates"]) == 2
        assert all(gate["status"] == "free" for gate in resolution[
            "actor_gates"])
        assert not new["evaluator_motion_interrupted"]
        assert new["evaluator_path_min_clearance_m"] >= .05
        seed = new["case"][0]
        if seed in (170, 174):
            assert new["status"] == "owner_assisted_visual_reacquisition"
            assert new["owner_questions"] == 2
            assert help_event["response_mode_evaluator_script"] == "compliant"
            assert help_event["evaluator_presentation_model_gap_m"] >= .05
            assert help_event["owner_identity_answer"] == "yes"
            views = [event for event in trace["controller_events"]
                     if event["action"] == "view"]
            assert views[-2]["label"] == "owner-presented fresh candidate"
            assert views[-2]["evaluator_correct_item"]
            assert views[-1]["label"] == "post-owner fresh recheck"
            assert views[-1]["accepted_box"] is not None
            assert iou(views[-2]["accepted_box"],
                       views[-1]["accepted_box"]) >= .6
        else:
            assert new["status"] == "owner_help_ignored_unresolved"
            assert new["owner_questions"] == 1
            assert help_event["response_mode_evaluator_script"] == "ignored"
            assert not any(event["action"] == "view" and event["label"] ==
                           "owner-presented fresh candidate" for event in
                           trace["controller_events"])
        frames = frame_count(ROOT / new["video_path"])
        assert frames > 10
        rows.append({"seed": seed, "status": new["status"],
                     "owner_questions": new["owner_questions"],
                     "modeled_minimum_clearance_m": new[
                         "evaluator_path_min_clearance_m"],
                     "video_frames": frames})
    audit = {"schema": "robot-learning-lab.no-supported-view-owner-help-audit.v1",
             "baseline_report_sha256": digest(old_report_path),
             "owner_help_report_sha256": digest(new_report_path),
             "paired_actor_captures_checked": paired_captures,
             "videos_decoded": len(rows),
             "all_checks_pass": True, "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
