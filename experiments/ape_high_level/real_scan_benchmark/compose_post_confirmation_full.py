"""Make one review video per full owner→find→approach→reconfirm episode."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio

from run_common_policy_20_object_generalization import ROOT, ART, relative


REPORT = ART / "post_confirmation_20_approach_v2_report.json"
SOURCE = ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015"


def append_video(writer, manifest: list[dict], video: Path, phase: str,
                 source_rows: list[dict], caption: str) -> None:
    capture = cv2.VideoCapture(str(video))
    index = 0
    while True:
        okay, bgr = capture.read()
        if not okay:
            break
        if caption:
            cv2.putText(bgr, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        manifest.append({"phase": phase, "source_video": relative(video),
                         "source_frame": index, "source": source_rows[index]})
        index += 1
    capture.release()
    if index == 0 or index != len(source_rows):
        raise ValueError(f"Source video/manifest mismatch: {video}")


def main() -> None:
    report = json.loads(REPORT.read_text())
    if len(report["rows"]) != 20:
        raise ValueError("Expected complete 20-room report")
    for row in report["rows"]:
        name = row["scene_key"]
        if row["action_result"] != "observed_target_standoff":
            continue
        folder = (ROOT / row["trace"]).parent
        approach = json.loads((folder / "trace.json").read_text())
        original = SOURCE / name
        prior = json.loads((original / "full_find_present.json").read_text())
        reconfirm = json.loads((folder / "reconfirm/episode_trace.json").read_text())
        manifest = []
        path = folder / "full_owner_find_approach_reconfirm.mp4"
        with imageio.get_writer(str(path), fps=5, codec="libx264", quality=7,
                                macro_block_size=8) as writer:
            append_video(writer, manifest, ROOT / prior["video"],
                         "owner_teach_initial_find", prior["frame_manifest"], "")
            append_video(writer, manifest, ROOT / approach["video"],
                         "guarded_approach", approach["actions"],
                         "APPROACH | fresh range, bounded step")
            append_video(writer, manifest, ROOT / reconfirm["video"],
                         "fresh_reconfirmation", reconfirm["video_frames"],
                         "FRESH ITEM CHECK | owner yes/no")
        review = {"schema": "ape.owner-find-approach-reconfirm-review.v1",
            "scene_key": name, "video": relative(path), "frames": len(manifest),
            "frame_manifest": manifest,
            "limits": "Five-fps phase overview, not real-time playback; source traces carry capture/worker times. Scripted owner, ideal proxy/range, no grasp."}
        (folder / "full_owner_find_approach_reconfirm.json").write_text(
            json.dumps(review, indent=2) + "\n")
        print(name, len(manifest), flush=True)


if __name__ == "__main__":
    main()
