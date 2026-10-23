"""Replay the stale-answer → fresh-view → monitored action gate across rooms."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_lighting_helped_action_20 import PRECOMMIT, REPORT, ARMS
from run_real_capture_teach_find import SAME_OBJECT_RADIUS_M
from run_compound_owner_point_recovery import make_scene
from audit_hidden_item_rearrangement import check_walk


OUT = ART / "lighting_helped_action_v2_20_audit.json"


def frame_count(path):
    reader = cv2.VideoCapture(str(path))
    count = 0
    while reader.read()[0]:
        count += 1
    reader.release()
    return count


def search_media(path):
    trace = json.loads(path.read_text())
    if (frame_count(ROOT / trace["video"]) != trace["frames"] or
            len(list(path.parent.glob("rgb_*.png"))) != trace["frames"] or
            len(list(path.parent.glob("depth_*.png"))) != trace["frames"]):
        raise ValueError(f"Search RGB-D/video mismatch: {path}")
    return trace


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    report = json.loads(REPORT.read_text())
    if (sha(ROOT / frozen["source_help_precommit"]) !=
            frozen["source_help_precommit_sha256"] or
            len(frozen["cases"]) != 13 or len(report["rows"]) != 26):
        raise ValueError("Broad paired source or run count changed")
    by_key = {(r["scene_key"], r["arm"]): r for r in report["rows"]}
    if len(by_key) != 26:
        raise ValueError("Duplicate paired run")
    rows = []
    for case in frozen["cases"]:
        key = case["scene_key"]
        pair = {}
        for arm in ARMS:
            row = by_key[(key, arm)]
            trace = json.loads((ROOT / row["trace"]).read_text())
            if trace["motion_authorized"] != row["motion_authorized"]:
                raise ValueError("Policy report disagrees with trace")
            source = json.loads((ROOT / trace["source_help"]).read_text())
            original = json.loads((ROOT / source["fresh_search"]).read_text())
            answer = next(e for e in original["review_events"]
                          if e["event"] == "answer")
            first = search_media(ROOT / trace["fresh_relook"])
            if first["observations"][0]["capture_time_s"] <= answer["answered_at_s"]:
                raise ValueError("Relook predates stale owner answer")
            packets = trace["monitor"]["packets"]
            health = trace["monitor"]["health"]
            before_release = [p for p in packets if p["delivery_wall_s"] <=
                              health["worker_wall_s"]]
            if len(before_release) != health["packet_count"]:
                raise ValueError("Monitor packet count mismatch")
            intervals = [b["capture_wall_s"] - a["capture_wall_s"]
                         for a, b in zip(before_release, before_release[1:])]
            if not math.isclose(max(intervals, default=0.), health[
                    "max_capture_interval_s"], abs_tol=1e-6):
                raise ValueError("Monitor cadence mismatch")
            missing = any(p["source_health"] == "missing"
                          for p in before_release)
            if missing != health["source_gap_seen"]:
                raise ValueError("Source-gap ledger mismatch")
            fresh_goal = trace["fresh_goal_xz_m"]
            gap = (float(np.linalg.norm(np.asarray(fresh_goal) -
                   np.asarray(trace["old_owner_confirmed_goal_xz_m"])))
                   if fresh_goal is not None else None)
            expected_admit = bool(trace["pose_healthy"] and health[
                "authorizes_current_view"] and first["action"] ==
                "candidate_to_owner_for_confirmation" and gap is not None and
                gap <= SAME_OBJECT_RADIUS_M)
            if (expected_admit != trace["motion_authorized"] or
                    (gap is not None and not math.isclose(gap,
                      trace["old_fresh_goal_gap_m"], abs_tol=1e-7))):
                raise ValueError("Motion gate differs from saved actor evidence")
            if arm == "range_gap" and (not missing or trace["walk"] is not None or
                                       trace["final_view"] is not None):
                raise ValueError("Source-gap arm performed item motion")
            walk = trace["walk"]
            final = None
            forward = 0
            if walk is not None:
                frames, forward = check_walk(walk)
                if frames != len(walk["actions"]):
                    raise ValueError("Guarded action video mismatch")
                scene = make_scene(case, True)
                if not all(scene.floor_supported(*action[
                        "evaluator_actual_xz_m"]) for action in walk["actions"]):
                    raise ValueError("Guarded action left reconstructed floor")
            if trace["final_view"] is not None:
                final = search_media(ROOT / trace["final_view"])
                selected = (final.get("verification") or {}).get(
                    "evaluator_selected_target_iou")
                correct = bool(final["action"] ==
                    "candidate_to_owner_for_confirmation" and selected is not None
                    and selected >= .25)
                if correct != trace["scripted_owner_final_yes"]:
                    raise ValueError("Final owner answer disagrees with selected crop")
            pair[arm] = {"fresh_action": first["action"],
                "fresh_rgb": relative((ROOT / trace["fresh_relook"]).parent /
                    first["observations"][0]["rgb_file"]),
                "source_gap": missing,
                "monitor_max_interval_s": health["max_capture_interval_s"],
                "motion_authorized": trace["motion_authorized"],
                "forward_steps": forward,
                "walk_result": walk["result"] if walk else None,
                "final_action": final["action"] if final else None,
                "final_owner_yes": trace["scripted_owner_final_yes"],
                "final_target_gap_m": trace["evaluator_final_target_gap_m"],
                "trace": row["trace"]}
        a, b = pair["healthy"], pair["range_gap"]
        if (sha(ROOT / a["fresh_rgb"]) != sha(ROOT / b["fresh_rgb"]) or
                a["fresh_action"] != b["fresh_action"]):
            raise ValueError("Paired camera or identity inputs changed")
        rows.append({"scene_key": key, "capture_group": case["capture_group"],
                     "healthy": a, "range_gap": b})
    summary = {"rooms": len(rows),
        "healthy_motion_authorized": sum(r["healthy"]["motion_authorized"] for r in rows),
        "healthy_standoffs": sum(r["healthy"]["walk_result"] ==
            "observed_target_standoff" for r in rows),
        "healthy_final_owner_yes": sum(r["healthy"]["final_owner_yes"] is True
                                       for r in rows),
        "range_gap_motion_authorized": sum(r["range_gap"]["motion_authorized"]
                                           for r in rows),
        "max_healthy_monitor_interval_s": max(r["healthy"][
            "monitor_max_interval_s"] for r in rows),
        "accepted_forward_steps": sum(r["healthy"]["forward_steps"] for r in rows)}
    OUT.write_text(json.dumps({"schema": "ape.lighting-helped-action-audit.v1",
        "precommit": relative(PRECOMMIT), "report": relative(REPORT),
        "summary": summary, "rows": rows,
        "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
