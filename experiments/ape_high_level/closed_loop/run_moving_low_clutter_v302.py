"""V302: selected Direct Frame owner search with a continuously moving floor prop.

The prop is outside the route at the start, enters the second lateral corridor
as the ape looks, then leaves. All scene-feasibility labels are evaluator-only.
The actor remains V284's four-shin point-gap search plus V298 pose fusion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_structural_entry_v271 import V149, V150
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_variable_slip_localized_v301 import VariableSlipRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v302"
SEED = 138
CASES = (("stationary_far", "present"),
         ("moving_near", "present"),
         ("moving_near", "absent"),
         ("moving_near", "identical_lookalike"))
POSE_SOURCES = ("command", "motion_checked_rgbd")
STOOL_SIZE = (.045, .04, .04)  # full-height 8 cm moving low prop
Y_FAR_M = .45
Y_NEAR_M = .30


def prop_y(time_s: float, condition: str) -> float:
    if condition == "stationary_far" or time_s < 6.:
        return Y_FAR_M
    if time_s < 8.:
        return Y_FAR_M + (Y_NEAR_M - Y_FAR_M) * (time_s - 6.) / 2.
    if time_s < 10.:
        return Y_NEAR_M
    if time_s < 12.:
        return Y_NEAR_M + (Y_FAR_M - Y_NEAR_M) * (time_s - 10.) / 2.
    return Y_FAR_M


class MovingLowRoom(VariableSlipRoom):
    def __init__(self, *, mode: str, pose_source: str,
                 condition: str, branch: Path) -> None:
        self.condition = condition
        super().__init__(mode=mode, pose_source=pose_source, branch=branch)

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        self.model.geom("loop_stool_geom").size[:] = STOOL_SIZE
        self.motion_side = 1  # activates continuous 20 ms room advancement
        self._update_moving_obstacle()
        self._forward()

    def _update_moving_obstacle(self) -> None:
        if self.motion_side is None:
            return
        xy = (0., prop_y(self.time_s, self.condition))
        self.obstacle_xy[3] = xy
        self.model.body_pos[self.model.body("loop_stool").id] = (
            *xy, STOOL_SIZE[2])


def preflight() -> dict:
    rows = []
    for condition, mode in CASES:
        room = MovingLowRoom(mode=mode, pose_source="command",
                             condition=condition, branch=OUT / "preflight")
        try:
            room.layout(SEED, absent=mode != "present", furnished=True)
            scene = scene_geoms(room)
            bodies = body_geoms(room, scene)
            stool = room.model.geom("loop_stool_geom").id
            scene_without_stool = [gid for gid in scene if gid != stool]
            closest_props = (math.inf, "", 0.)
            closest_origin = (math.inf, 0.)
            corridor_far = (math.inf, 0.)
            for t in np.arange(0., 12.001, .02):
                room.time_s = float(t)
                room._update_moving_obstacle()
                room.actual[:] = (0., 0., 0.)
                room._forward()
                origin_gap = full_body_clearance(room, scene, bodies)[0]
                if origin_gap < closest_origin[0]:
                    closest_origin = (float(origin_gap), float(t))
                for gid in scene_without_stool:
                    gap = float(mujoco.mj_geomDistance(
                        room.model, room.data, stool, gid, 10., None))
                    if gap < closest_props[0]:
                        closest_props = (gap, room.model.geom(gid).name, float(t))
            room.time_s = 12.2
            room._update_moving_obstacle()
            for y in np.linspace(0., .16, 33):
                room.actual[:] = (0., float(y), 0.)
                room._forward()
                gap = full_body_clearance(room, scene, bodies)[0]
                if gap < corridor_far[0]:
                    corridor_far = (float(gap), float(y))
            # A separate scheduled two-step witness establishes that the
            # moving prop can matter *during* an otherwise feasible search.
            scheduled = []
            for start_s, old_y, delta_y, slip in (
                    (7.2, 0., .08, .58),
                    (8.8, .0464, .08, .94)):
                minimum = (math.inf, 0.)
                for substep in range(21):
                    fraction = substep / 20
                    room.time_s = start_s + .4 * fraction
                    room._update_moving_obstacle()
                    room.actual[:] = (0., old_y + slip * delta_y * fraction, 0.)
                    room._forward()
                    gap = float(full_body_clearance(room, scene, bodies)[0])
                    if gap < minimum[0]:
                        minimum = (gap, fraction)
                scheduled.append({"start_time_s": start_s,
                                  "minimum_full_body_clearance_m":
                                      round(minimum[0], 5),
                                  "fraction_at_minimum": minimum[1],
                                  "safe_at_5cm": minimum[0] >= .05})
            if closest_props[0] < .02 or closest_origin[0] < .05 or corridor_far[0] < .05:
                raise ValueError(f"Invalid moving scene {condition}/{mode}: "
                                 f"props={closest_props}, origin={closest_origin}, "
                                 f"late corridor={corridor_far}")
            rows.append({"condition": condition, "mode": mode,
                         "nearest_other_prop_m": round(closest_props[0], 5),
                         "nearest_other_prop": closest_props[1],
                         "minimum_origin_clearance_m": round(closest_origin[0], 5),
                         "late_clear_side_minimum_clearance_m": round(
                             corridor_far[0], 5),
                         "late_clear_side_witness_y_m": round(corridor_far[1], 5),
                         "scheduled_two_step_evaluator_witness": scheduled})
        finally:
            room.close()
    return {"scope": "Evaluator-only selected Direct Frame full-body scene screen: check the continuously moving 8cm-high low prop every20ms against other props and the robot at origin, then check a 0–16cm positive-side route after the prop leaves. These labels never enter actor action choice.",
            "conditions": {"far_y_m": Y_FAR_M, "near_y_m": Y_NEAR_M,
                           "full_prop_size_xyz_m": tuple(2*x for x in STOOL_SIZE),
                           "near_time_s": [8., 10.]},
            "rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = preflight()
    (OUT / "geometry-preflight.json").write_text(
        json.dumps(geometry, indent=2) + "\n")
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    original_scan = search.shin_scan
    for condition, mode in CASES:
        for source in POSE_SOURCES:
            root = OUT / condition / source
            branch = root / f"seed{SEED}-{mode}"
            branch.mkdir(parents=True, exist_ok=True)
            rooms = []
            raw_sensors = []

            def factory(seed: int, requested_mode: str):
                if seed != SEED or requested_mode != mode:
                    raise ValueError("Unexpected room request")
                room = MovingLowRoom(mode=mode, pose_source=source,
                                     condition=condition, branch=branch)
                room.layout(seed, absent=mode != "present", furnished=True)
                rooms.append(room)
                return room

            def retained_scan(room, robot, case):
                clouds, counts = original_scan(room, robot, case)
                name = f"shin-{len(raw_sensors):02d}"
                path = branch / f"{name}.npz"
                angle = float(room.actual[2])  # sensor-adapter world-to-body only
                c, s = math.cos(angle), math.sin(angle)
                rot = np.array(((c, -s), (s, c)))
                local = {}
                for key, points in clouds.items():
                    points = np.asarray(points).reshape(-1, 3)
                    result = points.copy()
                    result[:, :2] = (points[:, :2] - room.actual[:2]) @ rot
                    local[key] = result
                np.savez_compressed(path, **local)
                raw_sensors.append({"case": case["case"],
                                    "capture_time_s": round(room.time_s, 4),
                                    "head_encoder_rad": round(room.head, 5),
                                    "actor_estimated_xyyaw": room.estimated.round(5).tolist(),
                                    "body_local_scan_path": str(path.relative_to(ROOT)),
                                    "body_local_scan_sha256": sha(path),
                                    "counts": dict(counts)})
                return clouds, counts

            search.shin_scan = retained_scan
            try:
                result = search.run_case(SEED, mode, vision, exemplar, owner_hist,
                                         room_factory=factory, output_root=root,
                                         sensor_case_prefix=f"v302-{condition}-{mode}")
            finally:
                search.shin_scan = original_scan
            room = rooms[0]
            trace = branch / "sensor-and-motion-trace.json"
            trace.write_text(json.dumps({
                "condition": condition, "mode": mode, "pose_source": source,
                "shin_observations": raw_sensors,
                "localization_observations": room.localization_events,
                "slips_evaluator_only": room.applied_slips,
                "controller_events": result["events"],
                "evaluator_path_min_clearance": room.path_min_clearance,
                "evaluator_motion_interrupted": room.motion_interrupted,
                "evaluator_final_moving_prop_y_m": prop_y(room.time_s, condition),
                "final_time_s": room.time_s}, indent=2) + "\n")
            row = {"condition": condition, "mode": mode, "pose_source": source,
                   "status": result["status"],
                   "owner_questions": result["owner_questions"],
                   "motion_steps_requested": len(room.applied_slips),
                   "evaluator_min_observed_view_clearance_m":
                       result["evaluator_min_model_clearance_m"],
                   "evaluator_path_min_clearance_m": round(
                       room.path_min_clearance[0], 5),
                   "evaluator_motion_interrupted": room.motion_interrupted,
                   "evaluator_pose_error_m": result["evaluator_pose_error_m"],
                   "route_decisions": [event for event in result["events"]
                                       if event["action"] in (
                                           "lateral_sector", "return_sector")],
                   "trace_path": str(trace.relative_to(ROOT)),
                   "trace_sha256": sha(trace),
                   "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                   "video_sha256": result["video_sha256"]}
            rows.append(row)
            (OUT / "moving-low-clutter-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(condition, mode, source, row["status"],
                  row["evaluator_path_min_clearance_m"],
                  row["motion_steps_requested"], flush=True)
    report = {"scope": "V302 selected Direct Frame live owner teach/find search with a scripted full-height8cm floor prop translating continuously at20ms into and out of the positive-side corridor while the ape looks/moves. Geometry preflight proves positive inter-prop separation, safe origin, and a >=5cm positive-side fixed-yaw route after departure; it does not give the route/time to the actor. Paired stationary-far and moving-near present, plus moving absent and identical-lookalike; command vs frozen motion-checked RGB-D localization, four articulated noisy shin scans, variable hidden slip, owner visual confirmation. Same matched scan seed across pose sources; raw body-local range and RGB/depth frames retained with timestamps. Evaluator path collision/5cm margin and owner result score only. Actor remains V284 point-only instant-stop controller, not a validated dynamic stop, native gait or full free/unknown map.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "geometry_preflight_sha256": sha(OUT / "geometry-preflight.json"),
              "rows": rows}
    (OUT / "moving-low-clutter-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
