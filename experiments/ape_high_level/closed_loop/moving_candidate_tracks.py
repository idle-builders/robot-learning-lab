"""Actor-side appearance candidates and short 3D motion tracks for owner show.

This development tracker intentionally keeps every matching proposal. It
can abstain when no unique item moves after a social cue; it does not prove
who moved it, exact-instance identity, or a physical hand connection.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from audit_entry_color_v150 import descriptor
from robust_item_depth import locate_box_center
from run_color_grounded_entry_v151 import DINO_MIN, HUE_MIN


DEDUP_DISTANCE_M = .04
MAX_TRACK_STEP_M = .12
MIN_MOTION_SPAN_M = .08


def appearance_candidates(vision, exemplar: np.ndarray,
                          owner_hist: np.ndarray, rgb: Image.Image,
                          depth: np.ndarray, camera_pose: tuple) -> list[dict]:
    proposals = vision.detect(rgb)
    features = (vision.feature([rgb.crop(row["box"])
                                for row in proposals]) @ exemplar
                if proposals else [])
    ordered = []
    for proposal, score in zip(proposals, features):
        box = proposal["box"]
        hue_hist = np.asarray(descriptor(rgb, box)["hue_histogram"])
        hue = float(np.minimum(owner_hist, hue_hist).sum())
        if score < DINO_MIN or hue < HUE_MIN:
            continue
        point = locate_box_center(box, camera_pose, depth,
                                  fovy_deg=67.)["world_xyz_m"]
        if point is None:
            continue
        ordered.append({"box": box, "world_xyz_m": point,
                        "appearance_score": round(float(score), 4),
                        "hue_overlap": round(hue, 4)})
    ordered.sort(key=lambda row: row["appearance_score"], reverse=True)
    result = []
    for candidate in ordered:
        if all(math.dist(candidate["world_xyz_m"], other["world_xyz_m"])
               > DEDUP_DISTANCE_M for other in result):
            result.append(candidate)
    return result


class MovingCandidateTracks:
    def __init__(self) -> None:
        self.tracks: list[dict] = []
        self.next_id = 1

    @staticmethod
    def _motion_span(track: dict) -> float:
        points = [point["xyz"] for point in track["history"][-3:]]
        if len(points) < 3:
            return 0.
        return max(math.dist(a, b) for a in points for b in points)

    def update(self, index: int, candidates: list[dict]) -> dict:
        edges = sorted((math.dist(track["history"][-1]["xyz"],
                                  candidate["world_xyz_m"]), tid, cid)
                       for tid, track in enumerate(self.tracks)
                       if track["last_index"] == index - 1
                       for cid, candidate in enumerate(candidates))
        used_tracks, used_candidates = set(), set()
        for distance, tid, cid in edges:
            if (distance > MAX_TRACK_STEP_M or tid in used_tracks or
                    cid in used_candidates):
                continue
            track = self.tracks[tid]
            candidate = candidates[cid]
            track["history"].append({
                "index": index, "xyz": candidate["world_xyz_m"],
                "candidate": candidate})
            track["last_index"] = index
            used_tracks.add(tid)
            used_candidates.add(cid)
        for cid, candidate in enumerate(candidates):
            if cid in used_candidates:
                continue
            self.tracks.append({
                "id": self.next_id, "last_index": index,
                "history": [{"index": index,
                             "xyz": candidate["world_xyz_m"],
                             "candidate": candidate}]})
            self.next_id += 1
        moving = [(track, self._motion_span(track)) for track in self.tracks
                  if track["last_index"] == index and
                  self._motion_span(track) >= MIN_MOTION_SPAN_M]
        selected = moving[0][0] if len(moving) == 1 else None
        return {
            "selected_candidate": (None if selected is None else
                                   selected["history"][-1]["candidate"]),
            "selected_track_id": None if selected is None else selected["id"],
            "selected_motion_span_m": (
                None if selected is None else
                round(moving[0][1], 5)),
            "candidate_count": len(candidates),
            "moving_track_count": len(moving),
        }
