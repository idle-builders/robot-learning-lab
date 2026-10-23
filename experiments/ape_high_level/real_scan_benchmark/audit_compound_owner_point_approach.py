"""Audit owner-help goal grounding, guarded motion and fresh confirmation."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned
from run_real_capture_teach_find import SCENES
from audit_hidden_item_rearrangement import check_walk
from run_compound_owner_point_recovery import make_scene
from run_compound_owner_point_approach import PRECOMMIT, REPORT, OUT, confirmed_location


AUDIT = ART / "compound_owner_point_approach_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    if sha(ROOT / frozen["source_help_audit"]) != frozen[
            "source_help_audit_sha256"]:
        raise ValueError("Owner-help evidence changed")
    report = json.loads(REPORT.read_text())
    if len(frozen["cases"]) != 6 or len(report["rows"]) != 6:
        raise ValueError("Expected six helped approach episodes")
    rows = []
    for case, reported in zip(frozen["cases"], report["rows"]):
        key = case["scene_key"]
        if reported["scene_key"] != key:
            raise ValueError("Approach case order changed")
        trace = json.loads((ROOT / reported["trace"]).read_text())
        help_trace = json.loads((ROOT / trace["source_help"]).read_text())
        source = json.loads((ROOT / help_trace["fresh_search"]).read_text())
        if source["action"] != "owner_confirmed_visual_identity":
            raise ValueError("Motion without helped owner confirmation")
        goal = confirmed_location(source)
        if not np.allclose(goal, trace["observed_goal_xz_m"]):
            raise ValueError("Motion goal differs from confirmed RGB-D")
        screen = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in screen["scenes"] if x.get("scene_key") == key)
        true_target = np.asarray(placement["find_target_xz"], float)
        walk = trace["walk"]
        frames, forward = check_walk(walk)
        if (frames != len(walk["actions"]) or len(walk["actions"]) > 25 or
                not np.allclose(walk["actions"][0]["actor_xz_m"],
                                trace["initial_actor_xz_m"])):
            raise ValueError("Walk starts at wrong owner pose or exceeds cap")
        scene = make_scene(case, True)
        for action in walk["actions"]:
            at = np.asarray(action["evaluator_actual_xz_m"], float)
            if not scene.floor_supported(*at):
                raise ValueError("Ape pose left supported floor")
            expected_gap = float(np.linalg.norm(at - true_target))
            if not math.isclose(expected_gap, action[
                    "evaluator_target_gap_m"], abs_tol=1e-7):
                raise ValueError("Target gap disagrees with placement")
            if action["decision"] == "forward":
                heading = math.radians(action["evaluator_actual_yaw_deg"])
                next_xy = at + action["amount"] * np.array([
                    math.sin(heading), math.cos(heading)])
                eye = np.array([next_xy[0], scene.floor_y + .42, next_xy[1]])
                fan = scene.range_fan(eye, action["evaluator_actual_yaw_deg"],
                    owner_xz=None, angles_deg=np.arange(-90., 91., 5.))
                finite = fan[np.isfinite(fan)]
                if (not scene.floor_supported(*next_xy) or len(finite) == 0 or
                        float(finite.min()) < .17 - 1e-8):
                    raise ValueError("Forward move lacks evaluator clearance")
        if (walk["result"] == "observed_target_standoff" and
                walk["actions"][-1]["decision"] != "hold"):
            raise ValueError("Standoff lacks a hold decision")
        final = None
        if trace["final_recheck"] is not None:
            final_path = ROOT / trace["final_recheck"]
            final = json.loads(final_path.read_text())
            if (final["action"] != trace["final_action"] or
                    not composite_aligned(final_path.parent, final) or
                    not all(aligned(final_path.parent / stage["name"])
                            for stage in final["stage_reports"])):
                raise ValueError("Fresh confirmation media mismatch")
            if final["action"] == "owner_confirmed_visual_identity":
                yes = [s for s in final["stage_reports"] if s["owner_answer"]
                    and s["owner_answer"]["action"].startswith("owner_confirmed")]
                if len(yes) != 1:
                    raise ValueError("Final claim lacks owner yes")
                verified = json.loads((ROOT / yes[0]["trace"]).read_text())
                if verified["verification"]["evaluator_selected_target_iou"] < .25:
                    raise ValueError("Wrong item confirmed after motion")
        measured_gap = float(np.linalg.norm(np.asarray(
            walk["actual_final_xz_m"]) - true_target))
        if not math.isclose(measured_gap, trace[
                "evaluator_final_target_gap_m"], abs_tol=1e-7):
            raise ValueError("Final target gap mismatch")
        rows.append({"scene_key": key,
            "capture_group": case.get("capture_group"),
            "walk_result": walk["result"], "forward_steps": forward,
            "walk_video_frames": frames,
            "final_action": trace["final_action"],
            "evaluator_final_target_gap_m": measured_gap,
            "walk_video": walk["video"],
            "final_video": final["video"] if final else None})
    summary = {"rooms": len(rows),
        "observed_standoffs": sum(r["walk_result"] ==
            "observed_target_standoff" for r in rows),
        "fresh_owner_confirmations": sum(r["final_action"] ==
            "owner_confirmed_visual_identity" for r in rows),
        "accepted_forward_steps": sum(r["forward_steps"] for r in rows),
        "final_gap_min_m": min(r["evaluator_final_target_gap_m"] for r in rows),
        "final_gap_max_m": max(r["evaluator_final_target_gap_m"] for r in rows)}
    AUDIT.write_text(json.dumps({"schema":
        "ape.compound-owner-point-approach-audit.v1",
        "precommit": relative(PRECOMMIT), "summary": summary,
        "rows": rows, "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
