"""Recheck fallible owner-point continuations and their saved media."""

from __future__ import annotations

import hashlib
import json

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned
from audit_integrated_owner_20_lighting_shift import video_frames
from run_compound_owner_point_recovery import PRECOMMIT, REPORT, OUT, ARMS


AUDIT = ART / "compound_owner_point_recovery_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for source in frozen["source_audits"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Whole-episode source audit changed")
    report = json.loads(REPORT.read_text())
    if len(frozen["cases"]) != 6 or len(report["rows"]) != 24:
        raise ValueError("Expected six misses in four feedback arms")
    audited = []
    for case in frozen["cases"]:
        original = ROOT / case["source_folder"]
        for name, branch, kind in ARMS:
            rows = [r for r in report["rows"] if r["scene_key"] ==
                case["scene_key"] and r["arm"] == name]
            if len(rows) != 1:
                raise ValueError("Missing or duplicate continuation")
            row = rows[0]
            trace = json.loads((ROOT / row["trace"]).read_text())
            source = json.loads((ROOT / trace["source_episode"]).read_text())
            if (source["action"] != "unidentified_ask_new_view_or_help" or
                    trace["motion_target"] is not None or
                    trace["action"] != row["action"]):
                raise ValueError("Invalid entry or motion claim")
            entry = {"scene_key": case["scene_key"], "arm": name,
                "source_action": source["action"], "action": trace["action"]}
            signal = trace["owner_signal"]
            if signal is None:
                if (kind != "silent" and trace["action"] !=
                        "ask_show_again_hold_unknown") or trace["fresh_search"]:
                    raise ValueError("Missing owner signal promoted")
                entry["status"] = "no_reply_or_no_selectable_point"
                audited.append(entry)
                continue
            if signal["kind"] != kind or kind == "silent":
                raise ValueError("Wrong feedback arm")
            rgb = ROOT / signal["rgb"]
            depth = ROOT / signal["depth"]
            if (hashlib.sha256(rgb.read_bytes()).hexdigest() !=
                    signal["rgb_sha256"] or
                    hashlib.sha256(depth.read_bytes()).hexdigest() !=
                    signal["depth_sha256"]):
                raise ValueError("Point image changed")
            stage_path = original / branch / signal["stage"] / "trace.json"
            stage = json.loads(stage_path.read_text())
            observation = next((o for o in stage["observations"] if
                o["frame"] == signal["frame"]), None)
            if (observation is None or (stage_path.parent / observation[
                    "rgb_file"]) != rgb or (stage_path.parent / observation[
                    "depth_file"]) != depth or observation["capture_time_s"] !=
                    signal["capture_time_s"]):
                raise ValueError("Point not bound to source capture")
            selected = trace["selected_proposal"]
            if selected not in observation["proposals"]:
                raise ValueError("Point selected a nonexistent proposal")
            x, y = signal["click_xy"]
            if (not selected["box"][0] <= x <= selected["box"][2] or
                    not selected["box"][1] <= y <= selected["box"][3] or
                    selected.get("actor_map_xy_m") is None or
                    selected.get("border_crop")):
                raise ValueError("Point lacks a localized actor candidate")
            target_iou = float(selected.get("evaluator_target_iou") or 0.)
            if ((kind == "accurate" and target_iou < .5) or
                    (kind == "wrong" and target_iou >= .1)):
                raise ValueError("Owner intervention label is wrong")
            fresh_path = ROOT / trace["fresh_search"]
            fresh = json.loads(fresh_path.read_text())
            if (fresh["action"] != trace["action"] or
                    fresh["motion_target"] is not None or
                    not composite_aligned(fresh_path.parent, fresh) or
                    not all(aligned(fresh_path.parent / s["name"])
                            for s in fresh["stage_reports"]) or
                    fresh["frames"] != video_frames(ROOT / fresh["video"])):
                raise ValueError("Fresh RGB-D/video/trace mismatch")
            if branch == "find_absent" and fresh["action"] == \
                    "owner_confirmed_visual_identity":
                raise ValueError("Absent object falsely confirmed")
            if fresh["action"] == "owner_confirmed_visual_identity":
                confirmed = [s for s in fresh["stage_reports"] if
                    s["owner_answer"] and s["owner_answer"]["action"].startswith(
                    "owner_confirmed")]
                if len(confirmed) != 1:
                    raise ValueError("Confirmation lacks owner yes")
                stage_confirmed = json.loads((ROOT / confirmed[0]["trace"]).read_text())
                if stage_confirmed["verification"]["evaluator_selected_target_iou"] < .25:
                    raise ValueError("Wrong object confirmed")
            entry.update(status="fresh_reobserved", selected_evaluator_target_iou=
                target_iou, fresh_video=fresh["video"],
                fresh_rgbd_frames=fresh["frames"],
                wrong_candidate_offered=fresh["wrong_confident_candidate"])
            audited.append(entry)
    summary = {"cases": len(frozen["cases"]), "runs": len(audited),
        "accurate_present_confirmed": sum(r["arm"] == "accurate_present" and
            r["action"] == "owner_confirmed_visual_identity" for r in audited),
        "wrong_present_confirmed": sum(r["arm"] == "wrong_present" and
            r["action"] == "owner_confirmed_visual_identity" for r in audited),
        "wrong_absent_confirmed": sum(r["arm"] == "wrong_absent" and
            r["action"] == "owner_confirmed_visual_identity" for r in audited),
        "silent_present_claims": sum(r["arm"] == "silent_present" and
            r["action"] == "owner_confirmed_visual_identity" for r in audited),
        "wrong_candidate_offers": sum(r.get("wrong_candidate_offered", False)
            for r in audited)}
    AUDIT.write_text(json.dumps({"schema":
        "ape.compound-owner-point-recovery-audit.v1",
        "precommit": relative(PRECOMMIT), "summary": summary,
        "rows": audited, "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
