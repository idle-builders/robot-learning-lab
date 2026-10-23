"""Audit the broad owner-help continuation against its source episodes/media."""

from __future__ import annotations

import hashlib
import json

import cv2

from run_common_policy_20_object_generalization import ROOT, ART, relative
from run_lighting_owner_help_20 import SOURCE, PRECOMMIT, REPORT, ARMS


OUT = ART / "lighting_owner_help_20_audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_frames(path):
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    source = json.loads(SOURCE.read_text())
    report = json.loads(REPORT.read_text())
    if sha(SOURCE) != frozen["source_sha256"]:
        raise ValueError("Source report changed after precommit")
    if len(frozen["cases"]) != 13 or report["processed"] != 39:
        raise ValueError("Expected every one of 13 cases in all three arms")
    by_key = {(r["scene_key"], r["arm"]): r for r in report["rows"]}
    if len(by_key) != 39:
        raise ValueError("Duplicate or missing case/arm")
    source_by_key = {r["scene_key"]: r for r in source["rows"]}
    rows = []
    for case in frozen["cases"]:
        key = case["scene_key"]
        if source_by_key[key]["find_present"]["action"] != \
                "unidentified_ask_new_view_or_help":
            raise ValueError("Recovery was run after a known identity")
        for arm, _, kind in ARMS:
            entry = by_key[(key, arm)]
            trace = json.loads((ROOT / entry["trace"]).read_text())
            if trace["source_episode"] != source_by_key[key]["find_present"]["trace"]:
                raise ValueError("Continuation source episode changed")
            if trace["action"] != entry["action"] or trace["motion_target"] is not None:
                raise ValueError("Action or motion authority mismatch")
            row = {"scene_key": key, "arm": arm, "action": entry["action"],
                   "fresh_media_aligned": None, "owner_answer": None,
                   "answer_image_age_s": None, "wrong_confident_candidate": None,
                   "vision_device": ("none" if kind == "silent" else
                       trace.get("vision_device", "mps_default"))}
            signal = trace["owner_signal"]
            if kind == "silent":
                if signal is not None or trace["fresh_search"] is not None:
                    raise ValueError("Silent owner produced a new search")
                rows.append(row)
                continue
            if (sha(ROOT / signal["rgb"]) != signal["rgb_sha256"] or
                    sha(ROOT / signal["depth"]) != signal["depth_sha256"]):
                raise ValueError("Point did not reference its saved RGB-D packet")
            x, y = signal["click_xy"]
            box = trace["selected_proposal"]["box"]
            if not (box[0] <= x <= box[2] and box[1] <= y <= box[3] and
                    trace["selected_proposal"]["actor_map_xy_m"] is not None):
                raise ValueError("Point lacks actor-localized enclosing proposal")
            search = json.loads((ROOT / trace["fresh_search"]).read_text())
            video_count = decoded_frames(ROOT / search["video"])
            media_ok = (video_count == len(search["video_frames"]) and
                all((ROOT / frame["rgb"]).is_file() and
                    (ROOT / frame["depth"]).is_file()
                    for frame in search["video_frames"]
                    if frame["kind"] == "sensor_frame"))
            if not media_ok:
                raise ValueError("Fresh search video or sensor frames missing")
            first = json.loads((ROOT / trace["fresh_search"]).parent.joinpath(
                "generic/trace.json").read_text())["observations"][0]
            if float(first["capture_time_s"]) <= float(signal["capture_time_s"]):
                raise ValueError("Fresh search did not advance capture time")
            answers = [event for event in search["review_events"]
                       if event["event"] == "answer"]
            row.update(fresh_media_aligned=media_ok,
                owner_answer=answers[-1]["confirmed"] if answers else None,
                answer_image_age_s=answers[-1].get("image_age_at_answer_s")
                    if answers else None,
                wrong_confident_candidate=search["wrong_confident_candidate"])
            if kind == "accurate" and (entry["action"] ==
                    "owner_confirmed_visual_identity") != (row["owner_answer"] is True):
                raise ValueError("Accurate-point final claim lacks owner yes")
            if kind == "wrong" and (entry["action"] ==
                    "owner_confirmed_visual_identity"):
                raise ValueError("Wrong owner point produced final claim")
            rows.append(row)
    baseline = sum(r["find_present"]["action"] ==
                   "owner_confirmed_visual_identity" for r in source["rows"])
    helped = sum(r["arm"] == "accurate_present" and r["action"] ==
                 "owner_confirmed_visual_identity" for r in rows)
    result = {"schema": "ape.lighting-owner-help-20-audit.v1",
        "source": relative(SOURCE), "precommit": relative(PRECOMMIT),
        "report": relative(REPORT), "capture_groups": len(source["rows"]),
        "source_unidentified_present": len(frozen["cases"]),
        "baseline_present_visual_confirmed": baseline,
        "accurate_help_visual_confirmed": helped,
        "joined_visual_confirmed": baseline + helped,
        "wrong_help_final_claims": sum(r["arm"] == "wrong_present" and
            r["action"] == "owner_confirmed_visual_identity" for r in rows),
        "silent_help_final_claims": sum(r["arm"] == "silent_present" and
            r["action"] == "owner_confirmed_visual_identity" for r in rows),
        "fresh_media_aligned": all(r["fresh_media_aligned"]
            for r in rows if r["arm"] != "silent_present"),
        "stale_answered_confirmations": sum(r["arm"] == "accurate_present" and
            r["action"] == "owner_confirmed_visual_identity" and
            r["answer_image_age_s"] is not None and
            r["answer_image_age_s"] > 1. for r in rows),
        "rows": rows,
        "limits": "Development-exposed rooms and scripted evaluator-derived owner points/answers. These are visual identity outcomes, not action-ready claims: motion targets remain null and slow owner answers age the image. Continuations reuse earlier owner approach and teaching; no new physical body, moving person, sensor hardware or 20 photoreal full-task rooms."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "capture_groups", "baseline_present_visual_confirmed",
        "accurate_help_visual_confirmed", "joined_visual_confirmed",
        "wrong_help_final_claims", "silent_help_final_claims",
        "fresh_media_aligned", "stale_answered_confirmations")}, indent=2))


if __name__ == "__main__":
    main()
