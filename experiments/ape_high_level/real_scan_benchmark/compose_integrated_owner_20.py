"""Make reviewable route -> teaching -> search videos from saved episode media.

The stitched video is a phase overview, not a real-time clock. Original trace
timestamps and all source RGB/depth frames remain the timing evidence.
"""

from __future__ import annotations

import argparse
import json

import cv2
import imageio.v2 as imageio

from run_common_policy_20_object_generalization import ROOT, ART, relative


REPORT = ART / "integrated_owner_20_conditional_report.json"


def emit_source_video(writer, manifest: list[dict], path, label: str,
                      source_trace: str, phase: str) -> None:
    capture = cv2.VideoCapture(str(path))
    index = 0
    while True:
        okay, bgr = capture.read()
        if not okay:
            break
        cv2.putText(bgr, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        manifest.append({"phase": phase, "source_video": relative(path),
                         "source_trace": source_trace, "source_frame": index})
        index += 1
    capture.release()
    if index == 0:
        raise ValueError(f"Empty source video: {path}")


def compose(row: dict, condition: str) -> dict:
    folder = (ROOT / row["summary"]).parent
    summary = json.loads((folder / "summary.json").read_text())
    teaching = summary["teaching"]
    owner_video = ROOT / row["owner_video"]
    search_video = ROOT / row[condition]["video"]
    teaching_path = ROOT / teaching["source_rgb"]
    output = folder / f"full_{condition}.mp4"
    manifest = []
    with imageio.get_writer(str(output), fps=5, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        emit_source_video(writer, manifest, owner_video,
                          "OWNER APPROACH | observed route", row["owner_trace"],
                          "owner_approach")
        teaching_bgr = cv2.imread(str(teaching_path))
        if teaching_bgr is None:
            raise ValueError(f"Missing teaching image: {teaching_path}")
        box = [int(round(x)) for x in teaching["selected_box_xyxy"]]
        cv2.rectangle(teaching_bgr, tuple(box[:2]), tuple(box[2:]),
                      (0, 235, 255), 2)
        cv2.putText(teaching_bgr,
                    f"SCRIPTED TEACH | {row['target']} confirmed crop",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (255, 255, 255), 2)
        for index in range(5):
            writer.append_data(cv2.cvtColor(teaching_bgr, cv2.COLOR_BGR2RGB))
            manifest.append({"phase": "scripted_teaching_still",
                             "source_rgb": teaching["source_rgb"],
                             "teaching_summary": row["summary"],
                             "display_frame": index})
        emit_source_video(writer, manifest, search_video,
                          ("FIND PRESENT" if condition == "find_present"
                           else "FIND ABSENT") + " | conditional search",
                          row[condition]["trace"], condition)
    metadata = {"schema": "ape.integrated-owner-review-video.v1",
                "scene_key": row["scene_key"], "condition": condition,
                "video": relative(output), "frames": len(manifest),
                "frame_manifest": manifest,
                "limits": "Phase overview at 5 fps; original owner/search traces contain capture, decision and worker times. Scripted owner and teaching image; no physical grasp or continuous real-time playback."}
    manifest_path = folder / f"full_{condition}.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"video": relative(output), "manifest": relative(manifest_path),
            "frames": len(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default=str(REPORT))
    args = parser.parse_args()
    report = json.loads(open(args.report).read())
    for row in report["rows"]:
        if row["status"] != "teach_find_completed_far_viewpoint":
            continue
        for condition in ("find_present", "find_absent"):
            outcome = compose(row, condition)
            print(row["scene_key"], condition, outcome["frames"], flush=True)


if __name__ == "__main__":
    main()
