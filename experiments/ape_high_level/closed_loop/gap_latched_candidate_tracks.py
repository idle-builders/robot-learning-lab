"""Keep a candidate across one missed 10 Hz image and a later still hold.

The actor needs several distinct observations to prove motion, but once that
evidence exists it should not require the item to keep moving forever. A
track is selected only when its current image exists and it is the unique
motion-confirmed track. Camera-pose health remains an external prerequisite.
"""

from __future__ import annotations

import math

from moving_candidate_tracks import (MAX_TRACK_STEP_M,
                                     MIN_MOTION_SPAN_M)


MAX_FRAME_GAP = 2


class GapLatchedCandidateTracks:
    def __init__(self) -> None:
        self.tracks: list[dict] = []
        self.next_id = 1

    @staticmethod
    def motion_span(track: dict) -> float:
        points = [row["xyz"] for row in track["history"][-3:]]
        if len(points) < 3:
            return 0.
        return max(math.dist(a, b) for a in points for b in points)

    def update(self, index: int, candidates: list[dict]) -> dict:
        edges = []
        for tid, track in enumerate(self.tracks):
            gap = index - track["last_index"]
            if not 1 <= gap <= MAX_FRAME_GAP:
                continue
            for cid, candidate in enumerate(candidates):
                distance = math.dist(track["history"][-1]["xyz"],
                                     candidate["world_xyz_m"])
                if distance <= MAX_TRACK_STEP_M * gap:
                    edges.append((distance, tid, cid))
        used_tracks, used_candidates = set(), set()
        for distance, tid, cid in sorted(edges):
            if tid in used_tracks or cid in used_candidates:
                continue
            track = self.tracks[tid]
            candidate = candidates[cid]
            track["history"].append({
                "index": index, "xyz": candidate["world_xyz_m"],
                "candidate": candidate})
            track["last_index"] = index
            track["motion_confirmed"] |= (
                self.motion_span(track) >= MIN_MOTION_SPAN_M)
            used_tracks.add(tid)
            used_candidates.add(cid)
        for cid, candidate in enumerate(candidates):
            if cid in used_candidates:
                continue
            self.tracks.append({
                "id": self.next_id, "last_index": index,
                "motion_confirmed": False,
                "history": [{"index": index,
                             "xyz": candidate["world_xyz_m"],
                             "candidate": candidate}]})
            self.next_id += 1
        active = [track for track in self.tracks
                  if track["last_index"] == index and
                  track["motion_confirmed"]]
        selected = active[0] if len(active) == 1 else None
        return {
            "selected_candidate": None if selected is None else
                selected["history"][-1]["candidate"],
            "selected_track_id": None if selected is None else
                selected["id"],
            "selected_motion_span_m": None if selected is None else
                round(self.motion_span(selected), 5),
            "candidate_count": len(candidates),
            "moving_track_count": len(active),
            "actor_motion_latched": selected is not None,
        }
