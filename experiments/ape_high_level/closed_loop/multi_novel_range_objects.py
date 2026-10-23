"""Small multi-object belief from timestamped range changes against a prior.

Each supported compact circle gets a persistent ID. Missed observations keep
their last supported position and age; they do not silently become free space.
This is nearest-neighbor association for sparse well-separated props, not a
general person tracker or an occlusion-complete scene representation.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dynamic_range import point_clusters
from novel_range_object import NovelRangeObject


class MultiNovelRangeObjects:
    def __init__(self, xs, ys, prior_free, prior_occupied,
                 max_association_gap_s=None):
        self.extractor = NovelRangeObject(xs, ys, prior_free, prior_occupied)
        self.tracks = {}
        self.next_id = 1
        self.max_association_gap_s = max_association_gap_s

    def update(self, ranges, pose_xy, capture_time_s):
        points = self.extractor._novel_points(ranges, pose_xy)
        groups = point_clusters(points, link_m=.20)
        detections = []
        unresolved = 0
        for group in groups:
            if len(group) < 3:
                unresolved += 1
                continue
            center, radius, residual = self.extractor._fit_circle(group)
            if not (.12 <= radius <= .45 and residual <= .03):
                unresolved += 1
                continue
            detections.append({"center": center, "radius": radius,
                               "residual": residual, "points": len(group)})

        # Associate each new supported shape once to the nearest plausible
        # previous center. Unmatched confirmed tracks remain as stale beliefs.
        pairs = []
        for index, detection in enumerate(detections):
            for track_id, track in self.tracks.items():
                dt = max(0., capture_time_s - track["last_seen_s"])
                if (self.max_association_gap_s is not None and
                        dt > self.max_association_gap_s):
                    continue
                predicted = track["center"]
                if track["velocity"] is not None:
                    predicted = predicted + track["velocity"] * min(dt, 1.0)
                distance = float(np.linalg.norm(detection["center"] - predicted))
                if distance <= .20 + .60 * dt:
                    pairs.append((distance, index, track_id))
        claimed_detections, claimed_tracks = set(), set()
        associations = []
        for _, index, track_id in sorted(pairs):
            if index not in claimed_detections and track_id not in claimed_tracks:
                associations.append((index, track_id))
                claimed_detections.add(index)
                claimed_tracks.add(track_id)
        for index, track_id in associations:
            detection = detections[index]
            track = self.tracks[track_id]
            previous_time = track["last_seen_s"]
            previous_center = track["center"]
            track["center"] = detection["center"]
            track["radius"] = detection["radius"]
            track["residual"] = detection["residual"]
            track["points"] = detection["points"]
            track["last_seen_s"] = capture_time_s
            track["hits"] += 1
            track["history"].append((capture_time_s, detection["center"]))
            if capture_time_s > previous_time:
                track["velocity"] = ((detection["center"] - previous_center) /
                                     (capture_time_s - previous_time))
        for index, detection in enumerate(detections):
            if index in claimed_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "id": track_id, "center": detection["center"],
                "radius": detection["radius"], "residual": detection["residual"],
                "points": detection["points"], "first_seen_s": capture_time_s,
                "last_seen_s": capture_time_s, "hits": 1, "velocity": None,
                "history": deque([(capture_time_s, detection["center"])], maxlen=3)}
        snapshot = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            age = max(0., capture_time_s - track["last_seen_s"])
            snapshot.append({"id": track_id,
                             "status": ("confirmed_now" if age < .001 and track["hits"] >= 2
                                        else "candidate_now" if age < .001 else
                                        "last_seen" if track["hits"] >= 2 else "candidate_lost"),
                             "center_xy_m": track["center"].tolist(),
                             "radius_m": track["radius"],
                             "velocity_xy_m_s": (track["velocity"].tolist()
                                                 if track["velocity"] is not None else None),
                             "first_seen_s": track["first_seen_s"],
                             "last_seen_s": track["last_seen_s"],
                             "age_s": age, "hits": track["hits"],
                             "fit_residual_m": track["residual"],
                             "supporting_points": track["points"]})
        return {"capture_time_s": capture_time_s,
                "novel_return_count": len(points),
                "raw_cluster_count": len(groups),
                "unresolved_cluster_count": unresolved,
                "tracks": snapshot}
