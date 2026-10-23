"""Stitch source phases into inspectable interruption/recovery review videos.

Playback is a phase overview; wall and simulation timing remain in the source
traces. The stitched clip is never used to score perception or response.
"""

from __future__ import annotations

import json

import cv2
import imageio.v2 as imageio
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative


PRECOMMIT = ART / "cross_room_recovery_precommit.json"
REPORT = ART / "cross_room_recovery_report.json"
OUT = ART / "cross_room_recovery_review_video_report.json"


def append_video(writer, manifest, path, phase, trace):
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        okay, bgr = capture.read()
        if not okay:
            break
        frame = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
        canvas = cv2.copyMakeBorder(frame, 4, 4, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(10, 10, 10))
        cv2.putText(canvas, phase.replace("_", " ").upper(), (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        manifest.append({"phase": phase, "source_video": relative(path),
                         "source_trace": trace, "source_frame": count})
        count += 1
    capture.release()
    if not count:
        raise ValueError(f"Source video empty: {path}")


def append_teaching(writer, manifest, summary_path):
    summary = json.loads(summary_path.read_text())
    teaching = summary["teaching"]
    rgb = cv2.imread(str(ROOT / teaching["source_rgb"]))
    if rgb is None:
        raise ValueError("Teaching RGB missing")
    canvas = cv2.copyMakeBorder(rgb, 4, 4, 0, 0,
                                cv2.BORDER_CONSTANT, value=(10, 10, 10))
    box = [int(round(x)) for x in teaching["selected_box_xyxy"]]
    cv2.rectangle(canvas, (box[0],box[1]+4), (box[2],box[3]+4),
                  (0,235,255), 2)
    cv2.putText(canvas, "SCRIPTED OWNER TEACHES", (8,25),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    for i in range(5):
        writer.append_data(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        manifest.append({"phase":"scripted_teaching_still",
                         "source_rgb":teaching["source_rgb"],
                         "source_summary":relative(summary_path),
                         "display_frame":i})


def append_result(writer, manifest, result):
    label = ("OWNER CONFIRMED ITEM" if result ==
             "owner_confirmed_visual_identity" else
             "UNIDENTIFIED: ASK FOR ANOTHER VIEW")
    canvas = np.full((368,640,3),25,dtype=np.uint8)
    cv2.putText(canvas,label,(18,180),cv2.FONT_HERSHEY_SIMPLEX,
                .66,(255,255,255),2)
    for i in range(5):
        writer.append_data(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        manifest.append({"phase":"outcome_card","action":result,
                         "display_frame":i})


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    report = json.loads(REPORT.read_text())
    rows = []
    for case in frozen["cases"]:
        row = next(x for x in report["rows"] if x["case"] == case["id"] and
                   x["condition"] == "find_present")
        result = json.loads((ROOT / row["result"]).read_text())
        summary = ROOT / case["teaching_summary"]
        owner = ROOT / case["owner_trace"]
        interrupted = json.loads((ROOT / case["interrupted_trace"]).read_text())
        video_path = (ROOT / row["result"]).parent / "full_interruption_recovery.mp4"
        manifest = []
        with imageio.get_writer(str(video_path), fps=5, codec="libx264",
                                quality=7, macro_block_size=16) as writer:
            append_video(writer,manifest,owner.parent / "first_person.mp4",
                         "owner_approach",relative(owner))
            append_teaching(writer,manifest,summary)
            append_video(writer,manifest,ROOT / interrupted["monitor_video"],
                         "entrant_and_response",case["interrupted_trace"])
            append_video(writer,manifest,ROOT / result["monitor_video"],
                         "fresh_view_monitor",row["result"])
            append_video(writer,manifest,ROOT / result["search_video"],
                         "fresh_search",result["search_trace"])
            append_result(writer,manifest,row["action"])
        path = video_path.with_suffix(".json")
        detail = {"schema":"ape.interruption-recovery-review-video.v1",
                  "case":case["id"],"condition":"find_present",
                  "video":relative(video_path),"frames":len(manifest),
                  "frame_manifest":manifest,
                  "limits":"Stitched 5fps phase overview; source traces retain real worker, capture and action times. Scripted owner and ideal proxy response."}
        path.write_text(json.dumps(detail,indent=2)+"\n")
        capture=cv2.VideoCapture(str(video_path));n=0
        while capture.read()[0]:n+=1
        capture.release()
        if n!=len(manifest):
            raise ValueError("Review video/manifest count mismatch")
        rows.append({"case":case["id"],"video":relative(video_path),
                     "manifest":relative(path),"frames":n,
                     "action":row["action"]})
        print(case["id"],n,row["action"],flush=True)
    OUT.write_text(json.dumps({"schema":"ape.recovery-review-video-report.v1",
        "rows":rows},indent=2)+"\n")


if __name__ == "__main__":
    main()
