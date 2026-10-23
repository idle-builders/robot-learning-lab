"""Paired audit of V265p help after the first depth-loss route veto."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio

from audit_repeated_viewpoints_v253p import same_capture
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, iou, model_with_room
from screen_stopped_alternate_routes_v255p import replay_stage


BEFORE = ROOT / "artifacts/ape-articulated-depth-route-v263p"
AFTER = ROOT / "artifacts/ape-articulated-depth-route-v265p"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        return sum(1 for frame in reader if frame.ndim == 3 and
                   frame.shape[2] == 3)


def input_frames(trace: dict) -> list[dict]:
    frames = trace["side_probe_inputs"].copy()
    for recovery in trace["recovery_inputs"]:
        frames.extend(recovery["head_inputs"])
    return frames


def main() -> None:
    before_path = BEFORE / "blocked-forward-viewpoint-audit.json"
    after_path = AFTER / "blocked-forward-viewpoint-audit.json"
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    wrapper = json.loads((AFTER / "report.json").read_text())
    assert wrapper["runner_report_sha256"] == digest(after_path)
    assert digest(BEFORE / "geometry-preflight.json") == digest(
        AFTER / "geometry-preflight.json")
    old_decl = json.loads((BEFORE / "predeclared-challenge.json").read_text())
    new_decl = json.loads((AFTER / "predeclared-challenge.json").read_text())
    old_decl.pop("source_sha256")
    new_decl.pop("source_sha256")
    assert old_decl.pop("actor_policy") != new_decl.pop("actor_policy")
    assert old_decl == new_decl
    assert len(before["rows"]) == len(after["rows"]) == 4
    model = model_with_room(visual=selected_visual())
    robot = RobotGeometry(model)
    paired = 0
    rows = []
    for old, new in zip(before["rows"], after["rows"], strict=True):
        assert old["case"] == new["case"]
        for row in (old, new):
            assert digest(ROOT / row["trace_path"]) == row["trace_sha256"]
            assert digest(ROOT / row["video_path"]) == row["video_sha256"]
        previous = json.loads((ROOT / old["trace_path"]).read_text())
        trace = json.loads((ROOT / new["trace_path"]).read_text())
        assert previous["slips_evaluator_only"] == trace[
            "slips_evaluator_only"]
        assert old["base_moves"] == new["base_moves"]
        for one, two in zip(input_frames(previous), input_frames(trace),
                            strict=True):
            assert digest(ROOT / two["actor_frame_path"]) == two[
                "actor_frame_sha256"]
            same_capture(ROOT / one["actor_frame_path"],
                         ROOT / two["actor_frame_path"])
            paired += 1
        seed = new["case"][0]
        if seed in (178, 182):
            assert old["status"] == "blocked_after_active_view"
            assert new["status"] == "owner_assisted_visual_reacquisition"
            assert old["owner_questions"] == 0
            assert new["owner_questions"] == 2
            recovery = trace["recovery_inputs"][0]
            assert recovery["actor_gate"]["status"] == "blocked"
            alternate = replay_stage(recovery, model, robot)
            assert all(option["status"] == "blocked" for option in
                       alternate["options"].values())
            help_event = recovery["initial_veto"]
            assert help_event["base_moves_during_help"] == 0
            assert help_event["owner_identity_answer"] == "yes"
            assert help_event["evaluator_presentation_model_gap_m"] >= .05
            views = [event for event in trace["controller_events"]
                     if event["action"] == "view"]
            assert views[-2]["label"] == "owner-presented fresh candidate"
            assert views[-2]["evaluator_correct_item"]
            assert views[-1]["label"] == "post-owner fresh recheck"
            assert views[-1]["accepted_box"] is not None
            assert iou(views[-2]["accepted_box"],
                       views[-1]["accepted_box"]) >= .6
        else:
            assert old["status"] == new["status"] == "owner_confirmed_visual_only"
            assert old["owner_questions"] == new["owner_questions"] == 1
        assert new["evaluator_path_min_clearance_m"] >= .05
        assert not new["evaluator_motion_interrupted"]
        count = frames(ROOT / new["video_path"])
        assert count > 5
        rows.append({"seed": seed, "before": old["status"],
                     "after": new["status"],
                     "base_moves": new["base_moves"],
                     "minimum_modeled_clearance_m": new[
                         "evaluator_path_min_clearance_m"],
                     "video_frames": count})
    audit = {"schema": "robot-learning-lab.initial-veto-help-audit.v1",
             "before_report_sha256": digest(before_path),
             "after_report_sha256": digest(after_path),
             "paired_actor_captures_checked": paired,
             "videos_decoded": len(rows), "all_checks_pass": True,
             "rows": rows}
    (AFTER / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
