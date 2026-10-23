"""Check source binding, new views and truthful outcomes after owner help."""

from __future__ import annotations

import hashlib
import json

from run_alt_pair_owner_help import ART, ARMS, OUT, PRECOMMIT, REPORT, ROOT, sha
from run_common_policy_20_object_generalization import aligned, relative
from run_common_policy_20_conditional_search import composite_aligned


AUDIT = ART / "alt_pair_owner_help_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    if sha(ROOT / frozen["source"]) != frozen["source_sha256"] or \
            sha(ROOT / frozen["source_precommit"]) != \
            frozen["source_precommit_sha256"]:
        raise ValueError("Source episode set changed")
    for item in frozen["policy_sources"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Policy source changed: {item['path']}")
    report = json.loads(REPORT.read_text())
    if len(report["rows"]) != len(frozen["cases"]) * len(ARMS):
        raise ValueError("Owner-help arm coverage incomplete")
    rows = []
    for case in frozen["cases"]:
        folder = ROOT / case["source_folder"]
        for name, branch, kind in ARMS:
            matches = [r for r in report["rows"] if r["scene_key"] ==
                       case["scene_key"] and r["arm"] == name]
            if len(matches) != 1:
                raise ValueError("Missing or duplicate arm")
            trace = json.loads((ROOT / matches[0]["trace"]).read_text())
            prior = json.loads((folder / branch / "episode_trace.json").read_text())
            if (prior["action"] != "unidentified_ask_new_view_or_help" or
                    trace["action"] != matches[0]["action"] or
                    trace["motion_target"] is not None):
                raise ValueError("Unresolved source promoted to motion")
            row = {"scene_key": case["scene_key"], "arm": name,
                   "action": trace["action"]}
            signal = trace["owner_signal"]
            if signal is None:
                if (trace["action"] != "ask_show_again_hold_unknown" or
                        trace["fresh_search"] is not None):
                    raise ValueError("No signal must leave identity unknown")
                row["status"] = "no_selectable_owner_signal"
                rows.append(row)
                continue
            if signal["kind"] != kind or kind == "silent":
                raise ValueError("Wrong owner feedback condition")
            for channel in ("rgb", "depth"):
                path = ROOT / signal[channel]
                if hashlib.sha256(path.read_bytes()).hexdigest() != \
                        signal[channel + "_sha256"]:
                    raise ValueError("Owner-point source frame changed")
            stage = json.loads((folder / branch / signal["stage"] /
                                "trace.json").read_text())
            observation = next((o for o in stage["observations"] if
                                o["frame"] == signal["frame"]), None)
            selected = trace["selected_proposal"]
            if (observation is None or selected not in observation["proposals"]
                    or selected.get("actor_map_xy_m") is None):
                raise ValueError("Point not grounded in saved observation")
            x, y = signal["click_xy"]
            box = selected["box"]
            if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                raise ValueError("Selected proposal does not contain point")
            iou = float(selected.get("evaluator_target_iou") or 0.)
            if (kind == "accurate" and iou < .5) or \
                    (kind == "wrong" and iou >= .1):
                raise ValueError("Scripted owner condition mislabeled")
            fresh = json.loads((ROOT / trace["fresh_search"]).read_text())
            fresh_folder = (ROOT / trace["fresh_search"]).parent
            if (fresh["action"] != trace["action"] or
                    fresh["motion_target"] is not None or
                    not composite_aligned(fresh_folder, fresh) or
                    not all(aligned(fresh_folder / s["name"])
                            for s in fresh["stage_reports"])):
                raise ValueError("Fresh result or media mismatch")
            if kind == "wrong" and fresh["action"] == \
                    "owner_confirmed_visual_identity":
                raise ValueError("Wrong owner point gained final claim")
            if fresh["action"] == "owner_confirmed_visual_identity":
                confirming = [s for s in fresh["stage_reports"] if
                    s["owner_answer"] and s["owner_answer"]["action"].startswith(
                        "owner_confirmed")]
                if len(confirming) != 1:
                    raise ValueError("Confirmation lacks owner review")
                reviewed = json.loads((ROOT / confirming[0]["trace"]).read_text())
                if reviewed["verification"]["evaluator_selected_target_iou"] < .25:
                    raise ValueError("Wrong object confirmed")
            row.update(status="fresh_reobserved", selected_target_iou=iou,
                       wrong_offer=fresh["wrong_confident_candidate"],
                       video=fresh["video"])
            rows.append(row)
    score = {"source_misses": len(frozen["cases"]), "runs": len(rows),
        "accurate_confirmed": sum(r["arm"] == "accurate_present" and
            r["action"] == "owner_confirmed_visual_identity" for r in rows),
        "wrong_confirmed": sum(r["arm"].startswith("wrong_") and
            r["action"] == "owner_confirmed_visual_identity" for r in rows),
        "silent_confirmed": sum(r["arm"] == "silent_present" and
            r["action"] == "owner_confirmed_visual_identity" for r in rows),
        "wrong_offers": sum(r.get("wrong_offer", False) for r in rows)}
    AUDIT.write_text(json.dumps({"schema": "ape.alt-pair-owner-help-audit.v1",
        "precommit": relative(PRECOMMIT), "score": score, "rows": rows,
        "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
