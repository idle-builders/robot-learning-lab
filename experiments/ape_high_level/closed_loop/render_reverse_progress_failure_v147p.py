"""Pair the intended meeting retreat with a saved failed reverse trace.

The native trace supplies the first 1.2 s root path. Both owner-room panes are
kinematic pose transfers on the rejected historical Direct Frame, not native
policy execution in the furnished scene.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import add_person
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import render_native_pose_meeting_v144p as previous
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-reverse-progress-gate"
VIDEO = OUT / "intended-vs-saved-reverse-outcome.mp4"
NATIVE_CASE = "backward-varied300-backward-seed144"
SEED = 170


def local_delta(qpos: np.ndarray, frame: int) -> np.ndarray:
    start_yaw = 2 * math.atan2(float(qpos[0, 6]), float(qpos[0, 3]))
    delta = qpos[frame, :2] - qpos[0, :2]
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    return np.array((c * delta[0] + s * delta[1],
                     -s * delta[0] + c * delta[1]))


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    native_row = next(row for row in report["rows"]
                      if row["native_case"] == NATIVE_CASE)
    native_path = ROOT / native_row["native_trace"]
    if sha(native_path) != native_row["native_trace_sha256"]:
        raise AssertionError("Native reverse trace changed")
    with np.load(native_path, allow_pickle=False) as saved:
        native_qpos = saved["qpos"].astype(float)
        dt = float(saved["dt"])
    if abs(dt - .02) > 1e-10:
        raise AssertionError("Expected 50 Hz native source")
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == SEED)
    staging_path = ROOT / report["staging_packet"]
    if sha(staging_path) != report["staging_packet_sha256"]:
        raise AssertionError("Owner staging packet changed")
    with np.load(staging_path, allow_pickle=False) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()

    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rooms = []
    try:
        for _ in range(2):
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command", branch=OUT)
            room.layout(SEED, absent=False, furnished=True)
            room.data.qpos[:] = qpos
            rooms.append(room)
        timeline = [round(8.0 + .1*i, 3) for i in range(13)] + [
            round(16.2 + .1*i, 3) for i in range(21)]
        frames = []
        with imageio.get_writer(VIDEO, fps=10, codec="libx264",
                                quality=7) as writer:
            for t in timeline:
                fraction = min(1., max(0., (t - 8.0) / 1.2))
                frame = min(60, max(0, round((t - 8.0) / dt)))
                native_offset = local_delta(native_qpos, frame)
                if fraction > 0:
                    passed_offset = native_offset / fraction
                else:
                    passed_offset = native_offset
                intended_image, intended = previous.configure(
                    rooms[0], qpos, estimate, t,
                    np.array((-.12, 0.)), 0., "Intended 12cm retreat")
                actual_image, actual = previous.configure(
                    rooms[1], qpos, estimate, t,
                    passed_offset, 0., "Saved reverse outcome transfer")
                writer.append_data(np.concatenate(
                    (intended_image, actual_image), axis=1))
                frames.append({"time_s": t,
                               "native_source_frame": frame,
                               "native_local_xy_m": native_offset.round(5).tolist(),
                               "intended": intended,
                               "saved_reverse_outcome_transfer": actual})
    finally:
        for room in rooms:
            room.close()
        structural.structural_model = old_model
    video_report = {
        "schema": "robot-learning-lab.ape-reverse-progress-video.v1",
        "source_sha256": sha(SOURCE),
        "v145_report_sha256": sha(report_path),
        "previous_renderer_sha256": sha(Path(previous.__file__)),
        "selected_visual_sha256": sha(selected_visual()),
        "native_trace": native_row["native_trace"],
        "native_trace_sha256": native_row["native_trace_sha256"],
        "staging_packet": report["staging_packet"],
        "staging_packet_sha256": report["staging_packet_sha256"],
        "video_path": str(VIDEO.relative_to(ROOT)),
        "video_sha256": sha(VIDEO),
        "frames": frames,
        "scope": "Side-by-side owner-room kinematic transfer of the intended "
                 "12cm retreat and first1.2s actual XY from a retained native "
                 "historical reverse-policy trace. The native policy is not "
                 "executing in the furnished room; scripted arm, owner and "
                 "prop motions plus evaluator gap labels are only a visual "
                 "counterfactual. The static9.2–16.2s wait is omitted.",
    }
    (OUT / "video-report.json").write_text(json.dumps(
        video_report, indent=2) + "\n")
    print(VIDEO, len(frames),
          min(x["saved_reverse_outcome_transfer"]["evaluator_gap_m"]
              for x in frames))


if __name__ == "__main__":
    main()
