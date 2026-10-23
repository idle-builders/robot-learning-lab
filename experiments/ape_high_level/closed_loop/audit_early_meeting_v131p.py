"""Independent packet, episode and media checks for V126p–V130p."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run import ROOT, sha
import screen_early_show_posture_v126p as v126p
import screen_early_meeting_offset_v127p as v127p
import validate_early_meeting_v128p as v128p
import screen_early_retreat_observability_v129p as v129p
import render_early_meeting_v130p as v130p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-meeting-validation"


def check(condition: bool, message: str, tally: dict) -> None:
    if not condition:
        raise AssertionError(message)
    tally["checks"] += 1


def main() -> None:
    reports = [(v126p.OUT / "screen.json", v126p.SOURCE),
               (v127p.OUT / "screen.json", v127p.SOURCE),
               (v128p.OUT / "report.json", v128p.SOURCE),
               (v129p.OUT / "report.json", v129p.SOURCE),
               (v130p.OUT / "report.json", v130p.SOURCE)]
    tally = {"checks": 0, "decoded_video_frames": {}}
    loaded = []
    for path, source in reports:
        report = json.loads(path.read_text())
        check(report["source_sha256"] == sha(source),
              f"source hash: {source}", tally)
        loaded.append(report)
    posture, meeting, validation, sensing, replay = loaded
    check(len(posture["rows"]) == 18, "posture matrix", tally)
    check(len(meeting["rows"]) == 30, "meeting matrix", tally)
    check(len(validation["rows"]) == 16, "validation matrix", tally)
    check(len(sensing["rows"]) == 2, "sensor matrix", tally)
    check(len(replay["rows"]) == 2, "video matrix", tally)
    neutral = {(row["case"]): row for row in posture["rows"]
               if row["posture"] == "neutral"}
    raised = {(row["case"]): row for row in posture["rows"]
              if row["posture"] == "left_n06"}
    check(neutral["clear"]["minimum_scene_gap_m"] < .05,
          "neutral prop limit", tally)
    check(neutral["hand_close"]["minimum_scene_gap_m"] < .05,
          "neutral hand limit", tally)
    check(raised["clear"]["minimum_scene_gap_m"] > .05,
          "raised clear scene", tally)
    check(raised["hand_close"]["minimum_scene_gap_m"] < .05,
          "raised hand block remains", tally)
    for row in validation["rows"]:
        check(row["travel"]["minimum_scene_gap_m"] >= .05,
              "travel margin", tally)
        show = row["owner_show"]
        check(show["minimum_scene_gap_m"] >= .05,
              "owner show margin", tally)
        check(show["first_below_5cm_s"] is None,
              "no sampled breach", tally)
        check(show["owner_start_s"] == 16.6 and
              show["image_time_s"] == 17.4 and
              show["end_time_s"] == 18.2,
              "owner timing", tally)
        check(row["actor_accepted_box"] is not None and
              row["evaluator_box_iou"] >= .95,
              "taught item gate", tally)
        frame = ROOT / show["owner_image"]["actor_frame_path"]
        check(sha(frame) == show["owner_image"]["actor_frame_sha256"],
              "owner frame hash", tally)
        with np.load(frame) as packet:
            check(abs(float(packet["capture_time_s"])-17.4) < 1e-6,
                  "image capture time", tally)
            check(packet["rgb"].shape == (480, 640, 3) and
                  np.isfinite(packet["depth_m"]).sum() > 0,
                  "rendered RGB-D", tally)
    for row in sensing["rows"]:
        packet_path = ROOT / row["range_packet_path"]
        check(sha(packet_path) == row["range_packet_sha256"],
              "sensor packet hash", tally)
        packets = json.loads(packet_path.read_text())
        for rays in packets["ray_packets"].values():
            check(all(ray["capture_time_s"] == 8.0 for ray in rays),
                  "fresh range timestamp", tally)
        for meta in row["head_packets"].values():
            path = ROOT / meta["path"]
            check(sha(path) == meta["sha256"],
                  "head packet hash", tally)
            with np.load(path) as packet:
                check(abs(float(packet["capture_time_s"])-8.0) < 1e-6,
                      "fresh head timestamp", tally)
        for alternative in row["alternatives"].values():
            for route in alternative["routes"].values():
                check(route["coverage_status"] != "observed_free",
                      "unsupported route must not pass", tally)
    for row in replay["rows"]:
        path = ROOT / row["video_path"]
        check(sha(path) == row["video_sha256"],
              "video hash", tally)
        reader = imageio.get_reader(path)
        try:
            decoded = sum(1 for _ in reader)
        finally:
            reader.close()
        check(decoded == 34, "video frame count", tally)
        tally["decoded_video_frames"][row["case"]] = decoded
        check(abs(row["minimum_sampled_gap_m"] - min(
            frame["evaluator_gap_m"] for frame in row["frames"])) < 1e-6,
            "video gap metadata", tally)
    audit = {"schema": "robot-learning-lab.ape-early-meeting-audit.v1",
             "source_sha256": sha(SOURCE),
             "report_sha256": {str(path.relative_to(ROOT)): sha(path)
                                for path, _ in reports},
             "scope": "Independent consistency checks on retained source, "
             "sensor packet times/hashes, owner RGB-D frames, actor image "
             "accepts, minimum scripted geometry gaps, unknown-route "
             "decision and decoded visual replay. This does not validate "
             "physical safety or actor route authorization.",
             **tally}
    path = OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path, tally)


if __name__ == "__main__":
    main()
