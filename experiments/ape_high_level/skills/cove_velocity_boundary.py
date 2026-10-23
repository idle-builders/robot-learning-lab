"""Narrow high-level handoff for the saved Cove-50 velocity policy.

This boundary does not run the actor or authorize real walking. It records
which simulation skill command is accepted, keeps one motion
owner through a stop request, and distinguishes a zero-twist receipt from an
observed stop. Turn, approach and head-gaze capabilities are not qualified.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path


DESIGN_ID = "cove-50-direct-frame"
CURRENT_DESIGN = Path(__file__).resolve().parents[3] / "robot/ape-current-design.json"


def selected_runtime_available() -> bool:
    """Fail closed when the current design pointer rejects this historical body."""
    try:
        design = json.loads(CURRENT_DESIGN.read_text())
    except (OSError, ValueError):
        return False
    return (design.get("id") == DESIGN_ID and
            design.get("status") == "current_mechanical_design_candidate" and
            design.get("selected_design_id", DESIGN_ID) == DESIGN_ID)


@dataclass(frozen=True)
class PolicyIdentity:
    body_sha256: str
    checkpoint_sha256: str
    actor_action_dim: int
    control_period_ms: int


@dataclass(frozen=True)
class EntryObservation:
    sampled_at_ms: int
    design_id: str
    body_sha256: str
    imu_healthy: bool
    joints_healthy: bool
    base_pose_healthy: bool
    both_feet_grounded: bool
    root_height_m: float


@dataclass(frozen=True)
class MotionRequest:
    request_id: str
    kind: str
    vx_m_s: float
    vy_m_s: float
    wz_rad_s: float
    body_sha256: str
    checkpoint_sha256: str
    deadline_ms: int
    runtime_mode: str = "selected_body_simulation"


@dataclass(frozen=True)
class StopObservation:
    request_id: str
    window_start_ms: int
    window_end_ms: int
    mean_planar_speed_m_s: float
    mean_abs_yaw_speed_rad_s: float
    double_support_fraction: float
    body_sha256: str
    checkpoint_sha256: str
    source: str


@dataclass(frozen=True)
class BodyStopPacket:
    """One actor-visible 50 Hz body packet; no simulator root translation."""

    request_id: str
    sampled_at_ms: int
    body_sha256: str
    checkpoint_sha256: str
    imu_yaw_rate_rad_s: float
    mean_abs_leg_encoder_velocity_rad_s: float
    left_sole_force_n: float
    right_sole_force_n: float
    imu_healthy: bool
    joints_healthy: bool
    sole_forces_healthy: bool
    source: str = "historical_native_sim_idealized_packets"


class CoveVelocityBoundary:
    def __init__(self, identity: PolicyIdentity, *, max_entry_age_ms: int = 100,
                 max_stop_wait_ms: int = 5000):
        if identity.actor_action_dim != 21 or identity.control_period_ms != 20:
            raise ValueError("This boundary expects the saved 21-action, 50 Hz Cove actor")
        self.identity = identity
        self.max_entry_age_ms = max_entry_age_ms  # Interface trial values, not physical limits.
        self.max_stop_wait_ms = max_stop_wait_ms
        self.active_request_id: str | None = None
        self.active_runtime_mode: str | None = None
        self.state = "idle"
        self.stop_requested_at_ms: int | None = None
        self.last_observed_stop_ms: int | None = None
        self.active_deadline_ms: int | None = None
        self.body_quiet_at_ms: int | None = None
        self.last_body_packet_ms: int | None = None
        self._body_quiet_samples: deque[int] = deque(maxlen=20)

    def submit(self, request: MotionRequest, observation: EntryObservation,
               *, now_ms: int) -> dict:
        if (request.body_sha256 != self.identity.body_sha256 or
            request.checkpoint_sha256 != self.identity.checkpoint_sha256 or
            observation.design_id != DESIGN_ID or
            observation.body_sha256 != self.identity.body_sha256):
            return {"status": "rejected", "reason": "body_or_checkpoint_mismatch"}
        if request.runtime_mode not in ("selected_body_simulation", "historical_reference_simulation"):
            return {"status": "rejected", "reason": "physical_motion_unqualified"}
        if (request.runtime_mode == "selected_body_simulation" and
            not selected_runtime_available()):
            return {"status": "rejected", "reason": "no_selected_body_for_runtime"}
        if request.kind != "open_floor_forward":
            return {"status": "rejected", "reason": "skill_not_qualified_for_selected_body"}
        if (request.vx_m_s != 0.35 or request.vy_m_s != 0 or request.wz_rad_s != 0):
            return {"status": "rejected", "reason": "command_outside_replayed_case"}
        if self.active_request_id is not None:
            return {"status": "rejected", "reason": "another_motion_owner_active"}
        if (not request.request_id or request.deadline_ms <= now_ms or
            observation.sampled_at_ms > now_ms or
            now_ms - observation.sampled_at_ms > self.max_entry_age_ms or
            not observation.imu_healthy or not observation.joints_healthy or
            not observation.base_pose_healthy or
            not observation.both_feet_grounded or
            not 0.21 <= observation.root_height_m <= 0.25):
            return {"status": "rejected", "reason": "entry_state_stale_or_invalid"}
        self.active_request_id = request.request_id
        self.active_runtime_mode = request.runtime_mode
        self.state = "moving"
        self.stop_requested_at_ms = None
        self.last_observed_stop_ms = None
        self.body_quiet_at_ms = None
        self.last_body_packet_ms = None
        self._body_quiet_samples.clear()
        self.active_deadline_ms = request.deadline_ms
        status = ("accepted_for_historical_simulation"
                  if request.runtime_mode == "historical_reference_simulation"
                  else "accepted_for_simulation")
        return {"status": status, "request_id": request.request_id,
                "runtime_mode": request.runtime_mode,
                "command_frame": "robot_base", "twist": [0.35, 0.0, 0.0],
                "observed_outcome": "pending"}

    def request_stop(self, *, request_id: str, now_ms: int,
                     cause: str = "owner_interrupt") -> dict:
        if request_id != self.active_request_id:
            return {"status": "ignored", "reason": "not_active_request"}
        if self.state == "stopping":
            return {"status": "already_stopping", "observed_outcome": "pending"}
        if self.state != "moving":
            return {"status": "ignored", "reason": "motion_not_active"}
        self.state = "stopping"
        self.stop_requested_at_ms = now_ms
        self.body_quiet_at_ms = None
        self.last_body_packet_ms = None
        self._body_quiet_samples.clear()
        return {"status": "stop_command_requested", "request_id": request_id,
                "cause": cause, "command_frame": "robot_base",
                "twist": [0.0, 0.0, 0.0], "observed_outcome": "pending"}

    def tick(self, *, now_ms: int) -> dict:
        if (self.state == "moving" and self.active_deadline_ms is not None and
            now_ms >= self.active_deadline_ms):
            return self.request_stop(request_id=self.active_request_id,
                                     now_ms=now_ms, cause="deadline")
        if (self.state in ("stopping", "body_quiet") and self.stop_requested_at_ms is not None and
            now_ms - self.stop_requested_at_ms > self.max_stop_wait_ms):
            self.state = "stop_unverified"
            return {"status": "stop_unverified", "reason": "stop_deadline_expired"}
        return {"status": self.state}

    def observe_body_packet(self, packet: BodyStopPacket, *, now_ms: int) -> dict:
        """Observe body settling without promoting it to a metric stopped pose.

        An independent scene-relative motion/pose observation is still needed
        before the motion owner may be released for coordinate-based action.
        """
        if (packet.request_id != self.active_request_id or
            self.state not in ("stopping", "body_quiet")):
            return {"status": "ignored", "reason": "not_active_stop"}
        values = (packet.imu_yaw_rate_rad_s,
                  packet.mean_abs_leg_encoder_velocity_rad_s,
                  packet.left_sole_force_n, packet.right_sole_force_n)
        if (packet.body_sha256 != self.identity.body_sha256 or
            packet.checkpoint_sha256 != self.identity.checkpoint_sha256 or
            packet.source != ("historical_native_sim_idealized_packets"
                              if self.active_runtime_mode == "historical_reference_simulation"
                              else "selected_body_native_sim_idealized_packets") or
            not all(math.isfinite(v) and v >= 0 for v in values) or
            self.stop_requested_at_ms is None or
            packet.sampled_at_ms <= self.stop_requested_at_ms or
            packet.sampled_at_ms > now_ms or now_ms - packet.sampled_at_ms > 100 or
            (self.last_body_packet_ms is not None and
             packet.sampled_at_ms <= self.last_body_packet_ms)):
            self.state = "stopping"
            self.body_quiet_at_ms = None
            self._body_quiet_samples.clear()
            return {"status": "rejected_feedback", "reason": "stale_or_invalid_body_packet"}
        if now_ms - self.stop_requested_at_ms > self.max_stop_wait_ms:
            self.state = "stop_unverified"
            return {"status": "stop_unverified", "reason": "stop_deadline_expired"}
        gap = (self.last_body_packet_ms is not None and
               packet.sampled_at_ms - self.last_body_packet_ms != 20)
        self.last_body_packet_ms = packet.sampled_at_ms
        if gap:
            self._body_quiet_samples.clear()
            self.body_quiet_at_ms = None
            self.state = "stopping"
        healthy = packet.imu_healthy and packet.joints_healthy and packet.sole_forces_healthy
        quiet = (healthy and packet.imu_yaw_rate_rad_s <= .03 and
                 packet.mean_abs_leg_encoder_velocity_rad_s <= .05 and
                 packet.left_sole_force_n >= 2 and packet.right_sole_force_n >= 2)
        if not quiet:
            self._body_quiet_samples.clear()
            self.body_quiet_at_ms = None
            self.state = "stopping"
            return {"status": "body_settling" if healthy else "body_feedback_unhealthy",
                    "motion_owner": "held", "metric_pose_valid": False}
        self._body_quiet_samples.append(packet.sampled_at_ms)
        if len(self._body_quiet_samples) < 20:
            return {"status": "body_settling", "motion_owner": "held",
                    "metric_pose_valid": False,
                    "quiet_packets": len(self._body_quiet_samples)}
        self.state = "body_quiet"
        self.body_quiet_at_ms = packet.sampled_at_ms
        return {"status": "body_quiet_provisional", "motion_owner": "held",
                "metric_pose_valid": False, "quiet_packets": 20,
                "sampled_at_ms": packet.sampled_at_ms}

    def observe_stop(self, observation: StopObservation, *, now_ms: int) -> dict:
        if observation.request_id != self.active_request_id or self.state != "stopping":
            return {"status": "ignored", "reason": "not_active_stop"}
        if (observation.body_sha256 != self.identity.body_sha256 or
            observation.checkpoint_sha256 != self.identity.checkpoint_sha256 or
            observation.source != ("historical_native_sim_evaluator"
                                   if self.active_runtime_mode == "historical_reference_simulation"
                                   else "selected_body_native_sim_evaluator") or
            not all(math.isfinite(value) for value in (
                observation.mean_planar_speed_m_s,
                observation.mean_abs_yaw_speed_rad_s,
                observation.double_support_fraction)) or
            observation.mean_planar_speed_m_s < 0 or
            observation.mean_abs_yaw_speed_rad_s < 0 or
            not 0 <= observation.double_support_fraction <= 1 or
            self.stop_requested_at_ms is None or
            observation.window_start_ms <= self.stop_requested_at_ms or
            observation.window_end_ms > now_ms or
            observation.window_end_ms <= observation.window_start_ms):
            return {"status": "rejected_feedback", "reason": "wrong_or_unposed_stop_evidence"}
        if now_ms - self.stop_requested_at_ms > self.max_stop_wait_ms:
            self.state = "stop_unverified"
            return {"status": "stop_unverified", "reason": "stop_deadline_expired"}
        if (observation.mean_planar_speed_m_s > 0.01 or
            observation.mean_abs_yaw_speed_rad_s > 0.01 or
            observation.double_support_fraction < 0.99):
            return {"status": "stopping", "observed_outcome": "not_stationary"}
        self.state = "stopped_observed"
        self.last_observed_stop_ms = observation.window_end_ms
        self.active_request_id = None
        self.active_deadline_ms = None
        return {"status": "stopped_observed_in_simulation",
                "source": observation.source,
                "runtime_mode": self.active_runtime_mode,
                "motion_owner": "released_after_observation"}

    def capture_after_stop(self, *, captured_at_ms: int) -> dict:
        if (self.state == "body_quiet" and self.body_quiet_at_ms is not None and
            captured_at_ms > self.body_quiet_at_ms):
            return {"status": "capture_allowed_pose_unverified",
                    "body_quiet_at_ms": self.body_quiet_at_ms,
                    "captured_at_ms": captured_at_ms,
                    "metric_pose_valid": False,
                    "motion_owner": "held"}
        if (self.state != "stopped_observed" or self.last_observed_stop_ms is None or
            captured_at_ms <= self.last_observed_stop_ms):
            return {"status": "reject_capture", "reason": "no_post_stop_frame"}
        return {"status": "accept_fresh_stationary_frame_in_simulation"}
