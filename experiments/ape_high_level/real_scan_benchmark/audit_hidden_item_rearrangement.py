"""Check the frozen hidden-change trial and compose inspectable episode videos."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_hidden_item_rearrangement import OUT, PRECOMMIT, REPORT, CASES


AUDIT = ART / "hidden_item_rearrangement_audit.json"


def video_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def compose(trace: dict, folder: Path, branch: dict) -> str:
    path = folder / f"{branch['branch']}_full_loop.mp4"
    clips = [("Initial guarded approach", trace["initial_walk"]["video"]),
             ("Fresh recheck after hidden change", branch["recheck_video"])]
    if branch["updated_walk"]:
        clips.append(("Updated goal approach", branch["updated_walk"]["video"]))
    if branch["final_video"]:
        clips.append(("Final owner confirmation", branch["final_video"]))
    with imageio.get_writer(str(path), fps=5, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for title, ref in clips:
            count = 0
            for rgb in video_frames(ROOT / ref):
                cv2.putText(rgb, title, (9, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .56, (255, 255, 255), 3)
                cv2.putText(rgb, title, (9, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .56, (25, 25, 25), 1)
                writer.append_data(rgb)
                count += 1
            if count == 0:
                raise ValueError(f"Empty clip: {ref}")
    return relative(path)


def check_walk(walk: dict) -> tuple[int, int]:
    if not walk["actions"]:
        raise ValueError("Empty action trace")
    video_count = sum(1 for _ in video_frames(ROOT / walk["video"]))
    if video_count != len(walk["actions"]):
        raise ValueError("Walk video/frame count mismatch")
    forward = 0
    for action in walk["actions"]:
        if not (ROOT / action["rgb"]).exists() or not (ROOT / action["depth"]).exists():
            raise ValueError("Walk RGB-D frame missing")
        if action["decision"] == "forward":
            if not (0. < action["amount"] <= .0400001):
                raise ValueError("Forward command outside fixed limit")
            if not action.get("evaluator_motion_accepted", False):
                raise ValueError("Forward motion rejected")
            forward += 1
        if action["decision"] == "turn" and abs(action["amount"]) > 15.00001:
            raise ValueError("Turn command outside fixed limit")
    return video_count, forward


def main() -> None:
    precommit = json.loads(PRECOMMIT.read_text())
    source = ROOT / precommit["source"]
    if sha(source) != precommit["source_sha256"]:
        raise ValueError("Source report changed after freeze")
    cases = json.loads(CASES.read_text())["cases"]
    report = json.loads(REPORT.read_text())
    if len(cases) != 20 or len(report["rows"]) != 20:
        raise ValueError("Expected complete 20-room cohort")
    if len({c["capture_group"] for c in cases}) != 20:
        raise ValueError("Duplicate physical capture")
    rows = []
    for case, reported in zip(cases, report["rows"]):
        key = case["scene_key"]
        if key != reported["scene_key"]:
            raise ValueError("Report order changed")
        frozen = next(x for x in precommit["cases"] if x["scene_key"] == key)
        if sha(ROOT / frozen["screen"]) != frozen["screen_sha256"]:
            raise ValueError("Screen changed after freeze")
        folder = OUT / key
        trace = json.loads((folder / "trace.json").read_text())
        if trace["scheduled_recheck_after_accepted_forward_steps"] != 2:
            raise ValueError("Recheck schedule changed")
        initial_frames, initial_forward = check_walk(trace["initial_walk"])
        if initial_forward > 2:
            raise ValueError("Walk passed recheck boundary")
        moved, absent = trace["branches"]
        if moved["branch"] != "moved" or absent["branch"] != "absent":
            raise ValueError("Branches changed")
        if not np.allclose(moved["evaluator_true_target_xz_m"],
                           trace["evaluator_old_distractor_xz_m"]):
            raise ValueError("Moved target did not swap placement")
        if absent["evaluator_true_target_xz_m"] is not None:
            raise ValueError("Absent branch retains target")
        if absent["new_observed_target_xz_m"] is not None or absent["updated_walk"]:
            raise ValueError("Absent branch generated motion goal")
        if moved["recheck_action"] == "owner_confirmed_visual_identity":
            if moved["new_observed_target_xz_m"] is None:
                raise ValueError("Confirmed move lacks observed localization")
        elif moved["updated_walk"] is not None:
            raise ValueError("Unconfirmed item generated motion goal")
        moved_frames = moved_steps = 0
        if moved["updated_walk"]:
            moved_frames, moved_steps = check_walk(moved["updated_walk"])
        moved_video = compose(trace, folder, moved)
        absent_video = compose(trace, folder, absent)
        rows.append({"scene_key": key, "capture_group": case["capture_group"],
            "initial_forward_steps": initial_forward,
            "initial_frames": initial_frames,
            "moved_reacquired": moved["new_observed_target_xz_m"] is not None,
            "moved_approached": bool(moved["updated_walk"] and
                moved["updated_walk"]["result"] == "observed_target_standoff"),
            "moved_final_confirmed": moved["final_action"] ==
                "owner_confirmed_visual_identity",
            "moved_final_gap_m": moved["evaluator_final_gap_m"],
            "moved_forward_steps": moved_steps,
            "moved_frames": moved_frames,
            "absent_safe": absent["new_observed_target_xz_m"] is None and
                absent["recheck_action"] != "owner_confirmed_visual_identity",
            "moved_full_video": moved_video,
            "absent_full_video": absent_video})
    totals = {"rooms": len(rows),
        "moved_reacquired": sum(x["moved_reacquired"] for x in rows),
        "moved_approached": sum(x["moved_approached"] for x in rows),
        "moved_final_confirmed": sum(x["moved_final_confirmed"] for x in rows),
        "absent_safe": sum(x["absent_safe"] for x in rows),
        "initial_forward_steps": sum(x["initial_forward_steps"] for x in rows),
        "moved_forward_steps": sum(x["moved_forward_steps"] for x in rows)}
    AUDIT.write_text(json.dumps({"schema": "ape.hidden-item-rearrangement-audit.v1",
        "precommit": relative(PRECOMMIT), "report": relative(REPORT),
        "totals": totals, "rows": rows}, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
