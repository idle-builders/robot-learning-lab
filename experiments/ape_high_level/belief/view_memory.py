"""View-backed name and location memory for the selected Cove-50 interaction.

Owner-confirmed appearance, model candidates and metric location are separate.
An image similarity score is never treated as identity probability. A location
is usable for a bounded action only with a recent confirmed sighting, matching
calibration/map epoch and healthy capture/current poses with error bounds.
Pose and floor-point estimates enter from sensor/perception code; this module
does not pretend to estimate walking motion or object identity itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math


DESIGN_ID = "cove-50-direct-frame"


@dataclass(frozen=True)
class Pose2:
    map_epoch: str
    x_m: float
    y_m: float
    yaw_rad: float
    sampled_at_ms: int
    healthy: bool
    position_error_bound_m: float | None
    yaw_error_bound_rad: float | None

    def bounded(self) -> bool:
        return (self.healthy and bool(self.map_epoch) and
                self.position_error_bound_m is not None and
                self.yaw_error_bound_rad is not None and
                all(math.isfinite(v) for v in (
                    self.x_m, self.y_m, self.yaw_rad,
                    self.position_error_bound_m, self.yaw_error_bound_rad)) and
                self.position_error_bound_m >= 0 and self.yaw_error_bound_rad >= 0)


@dataclass(frozen=True)
class Frame:
    frame_id: str
    camera: str
    image_sha256: str
    captured_at_ms: int
    calibration_id: str
    image_healthy: bool
    pose: Pose2 | None
    design_id: str = DESIGN_ID

    def validate(self) -> None:
        if (self.design_id != DESIGN_ID or not self.frame_id or not self.camera or
            not self.calibration_id or len(self.image_sha256) != 64 or
            self.captured_at_ms < 0):
            raise ValueError("Frame lacks selected-design image provenance")


@dataclass(frozen=True)
class TaughtView:
    name: str
    frame: Frame
    crop_xyxy: tuple[float, float, float, float]
    feature_ref: str
    owner_confirmed_at_ms: int
    contradicted: bool = False


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    name: str
    frame: Frame
    box_xyxy: tuple[float, float, float, float]
    model_score: float
    floor_xy_in_body_m: tuple[float, float] | None
    floor_point_error_bound_m: float | None
    world_xy_m: tuple[float, float] | None
    world_error_bound_m: float | None


class ViewMemory:
    def __init__(self, *, calibration_ids: dict[str, str],
                 max_action_age_ms: int = 500,
                 max_pose_age_ms: int = 100,
                 max_pose_capture_skew_ms: int = 50,
                 max_action_error_m: float = 0.10):
        self.calibration_ids = dict(calibration_ids)
        self.max_action_age_ms = max_action_age_ms
        self.max_pose_age_ms = max_pose_age_ms
        self.max_pose_capture_skew_ms = max_pose_capture_skew_ms
        self.max_action_error_m = max_action_error_m
        self.teaching: dict[str, TaughtView] = {}
        self.teaching_history: list[TaughtView] = []
        self.candidates: dict[str, Candidate] = {}
        self.confirmed_sighting: dict[str, Candidate] = {}
        self.rejected_candidates: set[str] = set()
        self.owner_answers: list[dict] = []
        self.displayed_candidates: dict[str, dict] = {}

    @staticmethod
    def _box(box: tuple[float, float, float, float]) -> None:
        if (len(box) != 4 or not all(math.isfinite(v) for v in box) or
            box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Candidate crop is not a finite nonempty box")

    def _calibrated(self, frame: Frame) -> bool:
        return self.calibration_ids.get(frame.camera) == frame.calibration_id

    def confirm_teaching(self, *, name: str, frame: Frame,
                         crop_xyxy: tuple[float, float, float, float],
                         feature_ref: str, owner_confirmed_at_ms: int) -> TaughtView:
        frame.validate()
        self._box(crop_xyxy)
        if (not name or not frame.image_healthy or not self._calibrated(frame) or
            not feature_ref or owner_confirmed_at_ms < frame.captured_at_ms):
            raise ValueError("Teaching needs a healthy owner-confirmed image crop")
        if name in self.teaching:
            raise ValueError("Use replace_teaching to correct a known name")
        view = TaughtView(name, frame, crop_xyxy, feature_ref, owner_confirmed_at_ms)
        self.teaching[name] = view
        return view

    def replace_teaching(self, *, name: str, frame: Frame,
                         crop_xyxy: tuple[float, float, float, float],
                         feature_ref: str, owner_confirmed_at_ms: int) -> TaughtView:
        if name not in self.teaching:
            raise ValueError("No teaching view to replace")
        old = self.teaching.pop(name)
        old_sighting = self.confirmed_sighting.pop(name, None)
        try:
            new = self.confirm_teaching(name=name, frame=frame, crop_xyxy=crop_xyxy,
                                        feature_ref=feature_ref,
                                        owner_confirmed_at_ms=owner_confirmed_at_ms)
            self.teaching_history.append(replace(old, contradicted=True))
            return new
        except Exception:
            self.teaching[name] = old
            if old_sighting is not None:
                self.confirmed_sighting[name] = old_sighting
            raise

    def observe_candidate(self, *, candidate_id: str, name: str, frame: Frame,
                          box_xyxy: tuple[float, float, float, float],
                          model_score: float,
                          floor_xy_in_body_m: tuple[float, float] | None,
                          floor_point_error_bound_m: float | None) -> Candidate:
        frame.validate()
        self._box(box_xyxy)
        if (not candidate_id or candidate_id in self.candidates or
            name not in self.teaching or not frame.image_healthy or
            not self._calibrated(frame) or not math.isfinite(model_score)):
            raise ValueError("Invalid candidate or unconfirmed taught name")
        pose = frame.pose
        world = None
        error = None
        if (floor_xy_in_body_m is not None and
            floor_point_error_bound_m is not None and
            len(floor_xy_in_body_m) == 2 and
            all(math.isfinite(v) for v in floor_xy_in_body_m) and
            math.isfinite(floor_point_error_bound_m) and
            floor_point_error_bound_m >= 0 and pose is not None and pose.bounded() and
            0 <= frame.captured_at_ms - pose.sampled_at_ms <= self.max_pose_capture_skew_ms):
            x, y = floor_xy_in_body_m
            c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
            world = (pose.x_m + c * x - s * y,
                     pose.y_m + s * x + c * y)
            distance = math.hypot(x, y)
            error = (floor_point_error_bound_m + pose.position_error_bound_m +
                     distance * pose.yaw_error_bound_rad)
        candidate = Candidate(candidate_id, name, frame, box_xyxy, model_score,
                              floor_xy_in_body_m, floor_point_error_bound_m,
                              world, error)
        self.candidates[candidate_id] = candidate
        return candidate

    def display_candidate(self, candidate_id: str, *, displayed_frame_id: str,
                          displayed_at_ms: int) -> None:
        candidate = self.candidates[candidate_id]
        if (displayed_frame_id != candidate.frame.frame_id or
            displayed_at_ms < candidate.frame.captured_at_ms):
            raise ValueError("Candidate display must identify its source frame")
        self.displayed_candidates[candidate_id] = {
            "frame_id": displayed_frame_id, "displayed_at_ms": displayed_at_ms}

    def owner_answer(self, candidate_id: str, *, confirmed: bool,
                     displayed_frame_id: str, answered_at_ms: int) -> None:
        candidate = self.candidates[candidate_id]
        display = self.displayed_candidates.get(candidate_id)
        if (display is None or displayed_frame_id != display["frame_id"] or
            answered_at_ms < display["displayed_at_ms"]):
            raise ValueError("Owner answer must refer to the displayed candidate frame")
        if confirmed:
            if candidate_id in self.rejected_candidates:
                raise ValueError("A rejected candidate needs a new observation")
            self.confirmed_sighting[candidate.name] = candidate
        else:
            self.rejected_candidates.add(candidate_id)
            if self.confirmed_sighting.get(candidate.name) == candidate:
                self.confirmed_sighting.pop(candidate.name)
        self.owner_answers.append({"candidate_id": candidate_id,
                                   "displayed_frame_id": displayed_frame_id,
                                   "answered_at_ms": answered_at_ms,
                                   "confirmed": confirmed})

    def locate(self, name: str, *, now_ms: int, current_pose: Pose2 | None) -> dict:
        if name not in self.teaching:
            return {"name": name, "status": "unknown_name", "action_target_body_xy_m": None}
        sighting = self.confirmed_sighting.get(name)
        if sighting is None:
            return {"name": name, "status": "learned_name_no_confirmed_location",
                    "action_target_body_xy_m": None,
                    "teaching_frame": self.teaching[name].frame.frame_id}
        base = {"name": name, "source_frame": sighting.frame.frame_id,
                "source_camera": sighting.frame.camera,
                "age_ms": now_ms - sighting.frame.captured_at_ms,
                "action_target_body_xy_m": None}
        if base["age_ms"] < 0:
            raise ValueError("Query time precedes confirmed sighting")
        newer_unconfirmed = any(
            candidate.name == name and
            candidate.frame.captured_at_ms > sighting.frame.captured_at_ms
            for candidate in self.candidates.values())
        if (sighting.world_xy_m is None or sighting.world_error_bound_m is None or
            not self._calibrated(sighting.frame) or current_pose is None or
            not current_pose.bounded() or sighting.frame.pose is None or
            sighting.frame.pose.map_epoch != current_pose.map_epoch or
            not 0 <= now_ms - current_pose.sampled_at_ms <= self.max_pose_age_ms):
            return {**base, "status": "location_unavailable"}
        dx = sighting.world_xy_m[0] - current_pose.x_m
        dy = sighting.world_xy_m[1] - current_pose.y_m
        c, s = math.cos(current_pose.yaw_rad), math.sin(current_pose.yaw_rad)
        body_xy = (c * dx + s * dy, -s * dx + c * dy)
        radius = (sighting.world_error_bound_m + current_pose.position_error_bound_m +
                  math.hypot(dx, dy) * current_pose.yaw_error_bound_rad)
        base.update({"search_hint_body_xy_m": body_xy,
                     "position_error_bound_m": radius})
        if newer_unconfirmed:
            return {**base, "status": "newer_candidate_unconfirmed"}
        if base["age_ms"] > self.max_action_age_ms or radius > self.max_action_error_m:
            return {**base, "status": "search_hint_only"}
        return {**base, "status": "fresh_owner_confirmed_location",
                "action_target_body_xy_m": body_xy}

    def record(self) -> dict:
        return {"design_id": DESIGN_ID,
                "active_calibration_ids": self.calibration_ids,
                "teaching": {name: asdict(view) for name, view in self.teaching.items()},
                "teaching_history": [asdict(view) for view in self.teaching_history],
                "candidates": {key: asdict(value) for key, value in self.candidates.items()},
                "confirmed_sighting_ids": {
                    name: sighting.candidate_id for name, sighting in self.confirmed_sighting.items()},
                "rejected_candidate_ids": sorted(self.rejected_candidates),
                "displayed_candidates": self.displayed_candidates,
                "owner_answers": self.owner_answers}
