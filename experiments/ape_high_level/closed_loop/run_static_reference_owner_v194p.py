"""V194p: owner turns with a pre-show static-reference range change cue.

Exploratory rule chosen after V191/V192 packet inspection; not a held-out
detector or a physical hazard/identity classifier. The camera cue and V153
task-recovery gate remain unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compare_standoff_range_layouts_v165p import front_left_roi
from run import ROOT, sha
import run_occluded_camera_guard_v174p as v174p
import run_standoff_front_response_v150p as v150p
import run_standoff_scene_variants_v160p as v160p
import run_varied_furniture_owner_v191p as v191p
from run_continuous_owner_guard_v95 import OwnerPhaseMonitor
from screen_standoff_front_cue_v149p import front_left_stat


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-static-reference-owner"
STATIC_FRAMES = 4  # 16.2–16.5 s, before the authored prop approach.
NOVEL_DISTANCE_M = .04
MIN_NOVEL_RETURNS = 4
MIN_NOVEL_CENTROID_MOTION_M = .012


def static_reference_capture(self) -> None:
    """Replace only the pooled p10 warning with a world-point change cue."""
    OwnerPhaseMonitor.capture(self)
    room = self.room
    frame = self.frames[-1]
    index = len(self.frames)-1
    points = self.points[-1][0]
    selected = points[front_left_roi(points, room.estimated.copy())]
    if not hasattr(self, "reference_frames"):
        self.reference_frames = []
    if index < STATIC_FRAMES:
        self.reference_frames.append(selected.copy())
    reference = (np.concatenate(self.reference_frames)
                 if self.reference_frames else np.empty((0, 3)))
    novel = np.empty((0, 3))
    if index >= STATIC_FRAMES and len(reference) and len(selected):
        nearest = np.linalg.norm(
            selected[:, None, :]-reference[None, :, :], axis=2).min(axis=1)
        novel = selected[nearest > NOVEL_DISTANCE_M]
    support, p10 = front_left_stat(points, room.estimated.copy())
    self.front_history.append((support, p10))
    closing = None
    if len(self.front_history) >= 3:
        older = self.front_history[-3][1]
        if older is not None and p10 is not None:
            closing = round(older-p10, 5)
    centroid = np.mean(novel, axis=0) if len(novel) else None
    previous_count = getattr(self, "previous_novel_count", 0)
    previous_centroid = getattr(self, "previous_novel_centroid", None)
    displacement = (float(np.linalg.norm(centroid-previous_centroid))
                    if centroid is not None and
                    previous_centroid is not None else None)
    cue = (len(novel) >= MIN_NOVEL_RETURNS and
           (previous_count < MIN_NOVEL_RETURNS or
            (displacement is not None and
             displacement >= MIN_NOVEL_CENTROID_MOTION_M)))
    self.previous_novel_count = len(novel)
    self.previous_novel_centroid = centroid
    first = cue and getattr(room, "front_alarm_s", None) is None
    if first:
        room.front_alarm_s = round(float(room.time_s), 4)
        self.alarm = {"capture_time_s": room.front_alarm_s,
                      "capture_index": index,
                      "side": "range_new_low_surface",
                      "new_return_count": len(novel)}
        room.motion_interrupted = True
    frame.update({
        "front_left_support": support,
        "front_left_p10_x_m": p10,
        "front_left_closing_0p2s_m": closing,
        "front_left_cue_now": cue,
        "front_left_first_cue": first,
        "static_reference_frames": STATIC_FRAMES,
        "static_reference_points": len(reference),
        "new_low_surface_returns": len(novel),
        "new_low_surface_centroid_displacement_m": displacement,
        "new_low_surface_source": "four_front_side",
    })


def run_condition(name: str, stationary: bool) -> dict:
    old = (v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S)
    try:
        v191p.OUT = OUT / name
        if stationary:
            v191p.LAYOUTS = {
                key: {**value, "prop_variant": "stationary_wide"}
                for key, value in old[1].items()
                if key in ("near_side_table", "offset_side_table")}
            v191p.RESPONSES_S = (None,)
        else:
            v191p.RESPONSES_S = (.2, None)
        v191p.main()
        path = v191p.OUT / "report.json"
        report = json.loads(path.read_text())
        report.update({
            "schema": "robot-learning-lab.ape-static-reference-owner.v1",
            "static_reference_source_sha256": sha(SOURCE),
            "condition": name,
            "static_reference": {
                "pre_show_frames": STATIC_FRAMES,
                "novel_world_point_distance_m": NOVEL_DISTANCE_M,
                "minimum_novel_returns": MIN_NOVEL_RETURNS,
                "minimum_novel_centroid_motion_m":
                    MIN_NOVEL_CENTROID_MOTION_M,
                "static_reference_exploratory_after_v191_inspection": True,
            },
            "scope": "Exploratory actor-visible four-front/side range change "
                     "rule learned from the V191/V192 authored rooms. "
                     "First four held-pose 10Hz packets form a static "
                     "world-point reference; at least four later low "
                     "front-left returns over4cm from it request yield on "
                     "appearance or a further1.2cm centroid shift. "
                     "Frozen V174 camera warning and V153 stable-range/image "
                     "recovery remain active. Complete historical held-pose "
                     "owner turns and 20ms geometry; not held-out scene "
                     "validation or physical motion qualification.",
        })
        path.write_text(json.dumps(report, indent=2) + "\n")
        return {"condition": name, "path": str(path.relative_to(ROOT)),
                "sha256": sha(path), "episodes": len(report["rows"])}
    finally:
        v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S = old


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = (v150p.FrontMonitor.capture, v174p.CameraFrontMonitor)
    v150p.FrontMonitor.capture = static_reference_capture
    v174p.CameraFrontMonitor = old[1]
    v160p.VARIANTS["stationary_wide"] = {
        **v160p.VARIANTS["fast_wide_turn"],
        "approach_start_s": 30., "approach_end_s": 31.}
    try:
        conditions = [run_condition("moving", False),
                      run_condition("stationary_control", True)]
    finally:
        v150p.FrontMonitor.capture = old[0]
        v174p.CameraFrontMonitor = old[1]
        del v160p.VARIANTS["stationary_wide"]
    report = {"schema": "robot-learning-lab.ape-static-reference-index.v1",
              "source_sha256": sha(SOURCE), "conditions": conditions}
    (OUT / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(OUT / "index.json")


if __name__ == "__main__":
    main()
