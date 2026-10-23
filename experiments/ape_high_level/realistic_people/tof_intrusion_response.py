"""Experimental stationary near-body cue and bounded surrogate escape.

The cue consumes time-stamped front-left and front-right multizone packets. The separate
escape check consumes ideal 360-degree PhysX low/mid fan distances, not labels.
This is a kinematic eye-proxy screen, not a physical safety controller.
"""

from __future__ import annotations

import bisect
import math


LAG_S = .4
ZONE_DECREASE_M = .045
MIN_CLOSING_ZONES = 4
PERSISTENCE = 2
RESPONSE_DELAY_S = .2
ESCAPE_DISTANCE_M = .16
ESCAPE_SPEED_M_S = .25
PROXY_RADIUS_M = .20
PATH_RESERVE_M = .06
ESCAPE_HEADINGS_BY_MOUNT = {
    "front_left": (90., 135., 180.),
    "front_right": (-90., -135., 180.),
}


class StationaryTofIntrusionResponse:
    def __init__(self, escape_distance_m: float = ESCAPE_DISTANCE_M):
        if not .08 <= escape_distance_m <= .30:
            raise ValueError("Escape distance outside this surrogate screen")
        self.escape_distance_m = escape_distance_m
        self.history = {mount: [] for mount in ESCAPE_HEADINGS_BY_MOUNT}
        self.streak = {mount: 0 for mount in ESCAPE_HEADINGS_BY_MOUNT}
        self.alarm = None
        self.escape_m = 0.
        self.escape_heading_deg = None
        self.last_step_time_s = None
        self.state = "watching"

    def observe(self, packet: dict) -> dict:
        """Observe one packet while the commanded body pose is stationary."""
        now_s = float(packet["sim_time_s"])
        closing_by_mount = {}
        alarms = []
        for mount, sensor in packet["sensors"].items():
            if mount not in self.history:
                continue
            values = [float(distance) if distance is not None and
                      height is not None and height >= .03 else None
                      for distance, height in zip(sensor["range_m_by_zone"],
                                                   sensor["hit_height_above_floor_m_by_zone"])]
            history = self.history[mount]
            times = [row[0] for row in history]
            prior_index = bisect.bisect_right(times, now_s - LAG_S) - 1
            closing_zones = 0
            if self.state == "watching" and prior_index >= 0:
                older = history[prior_index][1]
                closing_zones = sum(old is not None and new is not None and
                                    old - new >= ZONE_DECREASE_M
                                    for old, new in zip(older, values))
                self.streak[mount] = (self.streak[mount] + 1
                    if closing_zones >= MIN_CLOSING_ZONES else 0)
                if self.streak[mount] >= PERSISTENCE:
                    alarms.append((closing_zones, mount))
            history.append((now_s, values))
            closing_by_mount[mount] = closing_zones
        if alarms and self.alarm is None:
            closing_zones, mount = max(alarms)
            self.alarm = {"time_s": now_s, "mount": mount,
                          "closing_zones": closing_zones,
                          "rule": "one_flank_4_zones_4p5cm_over_0p4s_2_packets"}
            self.state = "waiting_for_response"
        return {"time_s": now_s,
                "closing_zones_by_mount": closing_by_mount,
                "streak_by_mount": self.streak.copy(),
                "state": self.state,
                "alarm": self.alarm}

    @staticmethod
    def assess_corridor(planes: list[list[float | None]],
                        angles_deg: list[float], heading_deg: float,
                        step_m: float) -> dict:
        """Check observed low/mid hit points against a swept proxy segment.

        A no-hit from the *ideal PhysX ray* is clear to its 8 m limit. Real
        optical no-return would instead be unknown. Between-beam and other
        height gaps are unproved and must remain explicit in the audit.
        """
        required_m = PROXY_RADIUS_M + PATH_RESERVE_M
        if not planes or any(len(plane) != len(angles_deg) for plane in planes):
            return {"clear": False, "reason": "invalid_coverage"}
        heading = math.radians(heading_deg)
        end_x = step_m * math.sin(heading)
        end_y = step_m * math.cos(heading)
        length_sq = end_x * end_x + end_y * end_y
        nearest = math.inf
        for plane in planes:
            for angle, value in zip(angles_deg, plane):
                if value is None or math.isnan(value) or value <= 0:
                    return {"clear": False, "reason": "unknown_ray",
                            "required_m": required_m}
                if not math.isfinite(value):
                    continue
                bearing = math.radians(angle)
                x = value * math.sin(bearing)
                y = value * math.cos(bearing)
                fraction = (max(0., min(1., (x * end_x + y * end_y) / length_sq))
                            if length_sq > 1e-10 else 0.)
                nearest = min(nearest, math.hypot(x - fraction * end_x,
                                                  y - fraction * end_y))
        return {"clear": nearest >= required_m,
                "reason": "ideal_swept_points_clear" if nearest >= required_m
                          else "swept_hit",
                "required_m": required_m,
                "nearest_swept_point_m": nearest if math.isfinite(nearest) else None,
                "heading_rel_deg": heading_deg,
                "step_m": step_m,
                "checked_beams": len(angles_deg) * len(planes)}

    def step(self, now_s: float, planes: list[list[float | None]] | None,
             angles_deg: list[float] | None) -> dict:
        """Select a visible corridor and return one bounded motion increment."""
        if self.alarm is None:
            return {"state": self.state, "translation_increment_m": 0.}
        if now_s < self.alarm["time_s"] + RESPONSE_DELAY_S:
            return {"state": self.state, "translation_increment_m": 0.}
        if self.escape_m >= self.escape_distance_m - 1e-8:
            self.state = "escape_complete"
            return {"state": self.state, "translation_increment_m": 0.,
                    "escape_total_m": self.escape_m,
                    "heading_rel_deg": self.escape_heading_deg}
        if planes is None or angles_deg is None:
            self.state = "blocked_no_verified_escape"
            return {"state": self.state, "translation_increment_m": 0.,
                    "reason": "missing_corridor_scan"}
        remaining = self.escape_distance_m - self.escape_m
        candidates = (ESCAPE_HEADINGS_BY_MOUNT[self.alarm["mount"]]
                      if self.escape_heading_deg is None
                      else (self.escape_heading_deg,))
        checks = [self.assess_corridor(planes, angles_deg, candidate, remaining)
                  for candidate in candidates]
        clear = [check for check in checks if check["clear"]]
        if not clear:
            self.state = "blocked_no_verified_escape"
            return {"state": self.state, "translation_increment_m": 0.,
                    "candidate_checks": checks}
        chosen = max(clear, key=lambda check: check["nearest_swept_point_m"]
                     if check["nearest_swept_point_m"] is not None else math.inf)
        if self.escape_heading_deg is None:
            self.escape_heading_deg = chosen["heading_rel_deg"]
        elapsed = (1. / 15. if self.last_step_time_s is None
                   else max(0., now_s - self.last_step_time_s))
        increment = min(remaining, ESCAPE_SPEED_M_S * elapsed)
        self.escape_m += increment
        self.last_step_time_s = now_s
        self.state = ("escape_complete" if self.escape_m >= self.escape_distance_m - 1e-8
                      else "escaping")
        return {"state": self.state, "translation_increment_m": increment,
                "heading_rel_deg": self.escape_heading_deg,
                "escape_total_m": self.escape_m,
                "candidate_checks": checks, "chosen_corridor": chosen}

    def summary(self) -> dict:
        return {"state": self.state, "alarm": self.alarm,
                "escape_total_m": self.escape_m,
                "requested_escape_m": self.escape_distance_m,
                "escape_heading_rel_deg": self.escape_heading_deg,
                "kinematic_response_only": True}
