"""V305: frozen changed-room/timing split for the V303 wait-and-reobserve rule.

The same owner search, 0.4s range-age limit and eight-wait cap run unchanged.
Each room is geometry-screened before actor exposure; the screen is scoring
only. This is a kinematic dynamic-obstacle challenge, not native stop proof.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_localization_v297 import depth_variant
from audit_direct_structural_entry_v271 import V149, V150
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_moving_low_clutter_v302 import MovingLowRoom, STOOL_SIZE, Y_FAR_M


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v305"
# Predeclared before rendering or sensor decisions: seed, owner mode, near Y,
# arrival start/end, departure start/end (seconds). Even seeds preserve the
# selected screen orientation while changing light and furniture jitter.
CASES = (
    (140, "present", .30, 5.4, 7.4, 9.0, 11.0),
    (142, "present", .32, 6.6, 8.6, 10.0, 12.0),
    (144, "present", .35, 6.0, 8.0, 10.0, 12.0),
    (146, "present", .30, 7.5, 9.5, 9.5, 11.5),
    (148, "absent", .32, 6.0, 8.0, 10.0, 12.0),
    (150, "identical_lookalike", .30, 6.0, 8.0, 10.0, 12.0),
)
POSE_SOURCES = ("command", "motion_checked_rgbd")
MAX_CLEAR_SIDE_WAITS = 8
MAX_RANGE_AGE_S = .4


def scheduled_y(t: float, case: tuple) -> float:
    _, _, near, enter_start, enter_end, leave_start, leave_end = case
    if t < enter_start:
        return Y_FAR_M
    if t < enter_end:
        return Y_FAR_M + (near - Y_FAR_M) * (t - enter_start) / (enter_end - enter_start)
    if t < leave_start:
        return near
    if t < leave_end:
        return near + (Y_FAR_M - near) * (t - leave_start) / (leave_end - leave_start)
    return Y_FAR_M


class HeldoutMovingRoom(MovingLowRoom):
    def __init__(self, *, case: tuple, pose_source: str, branch: Path) -> None:
        self.case = case
        self.scene_seed = case[0]
        super().__init__(mode=case[1], pose_source=pose_source,
                         condition="moving_near", branch=branch)

    def _update_moving_obstacle(self) -> None:
        if self.motion_side is None:
            return
        xy = (0., scheduled_y(self.time_s, self.case))
        self.obstacle_xy[3] = xy
        self.model.body_pos[self.model.body("loop_stool").id] = (
            *xy, STOOL_SIZE[2])

    def capture_localization(self, index: int):
        rgb, ideal = self.capture("upper_rgb", depth=True)
        if ideal is None:
            raise AssertionError("Rendered depth missing")
        depth = depth_variant(ideal, "noisy_long", self.scene_seed, index)
        rgb_path = self.branch / f"localization-{index:02d}-rgb.png"
        depth_path = self.branch / f"localization-{index:02d}-depth.npz"
        rgb.save(rgb_path)
        np.savez_compressed(depth_path, depth_m=depth)
        return (rgb, depth, self.camera_pose_estimate("upper_rgb")[1],
                {"rgb_path": str(rgb_path.relative_to(ROOT)),
                 "depth_path": str(depth_path.relative_to(ROOT)),
                 "rgb_sha256": sha(rgb_path),
                 "depth_sha256": sha(depth_path)})


def screen_case(case: tuple) -> dict:
    seed, mode, *_ = case
    room = HeldoutMovingRoom(case=case, pose_source="command",
                             branch=OUT / "preflight")
    try:
        room.layout(seed, absent=mode != "present", furnished=True)
        scene = scene_geoms(room)
        bodies = body_geoms(room, scene)
        stool = room.model.geom("loop_stool_geom").id
        other = [gid for gid in scene if gid != stool]
        nearest_prop = (math.inf, "", 0.)
        origin = (math.inf, 0.)
        for t in np.arange(0., max(12., case[6]) + .001, .02):
            room.time_s = float(t)
            room._update_moving_obstacle()
            room.actual[:] = (0., 0., 0.)
            room._forward()
            gap = float(full_body_clearance(room, scene, bodies)[0])
            if gap < origin[0]:
                origin = (gap, float(t))
            for gid in other:
                gap = float(mujoco.mj_geomDistance(
                    room.model, room.data, stool, gid, 10., None))
                if gap < nearest_prop[0]:
                    nearest_prop = (gap, room.model.geom(gid).name, float(t))
        room.time_s = case[6] + .2
        room._update_moving_obstacle()
        late = (math.inf, 0.)
        for y in np.linspace(0., .16, 33):
            room.actual[:] = (0., float(y), 0.)
            room._forward()
            gap = float(full_body_clearance(room, scene, bodies)[0])
            if gap < late[0]:
                late = (gap, float(y))
        if nearest_prop[0] < .02 or origin[0] < .05 or late[0] < .05:
            raise ValueError(f"Invalid heldout case {case}: "
                             f"prop={nearest_prop}, origin={origin}, late={late}")
        return {"case": case,
                "nearest_other_prop_m": round(nearest_prop[0], 5),
                "minimum_origin_clearance_m": round(origin[0], 5),
                "late_clear_side_minimum_clearance_m": round(late[0], 5)}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = [screen_case(case) for case in CASES]
    (OUT / "geometry-preflight.json").write_text(json.dumps({
        "scope": "Evaluator-only full-body and inter-prop geometry screen every20ms, plus a fixed-yaw 0–16cm clear-side witness after each scripted prop departs. No route or prop schedule is passed to the actor.",
        "rows": preflight}, indent=2) + "\n")
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    original_scan = search.shin_scan
    for case in CASES:
        seed, mode, *_ = case
        for source in POSE_SOURCES:
            root = OUT / f"seed{seed}-{mode}" / source
            branch = root / f"seed{seed}-{mode}"
            branch.mkdir(parents=True, exist_ok=True)
            rooms = []
            scans = []

            def factory(requested_seed: int, requested_mode: str):
                if requested_seed != seed or requested_mode != mode:
                    raise ValueError("Unexpected room request")
                room = HeldoutMovingRoom(case=case, pose_source=source,
                                         branch=branch)
                room.layout(seed, absent=mode != "present", furnished=True)
                rooms.append(room)
                return room

            def retained_scan(room, robot, request):
                clouds, counts = original_scan(room, robot, request)
                path = branch / f"shin-{len(scans):02d}.npz"
                np.savez_compressed(path, **clouds)
                addr = int(room.model.sensor("accelerometer").adr[0])
                scans.append({"case": request["case"],
                              "capture_time_s": round(room.time_s, 4),
                              "actor_estimated_xyyaw": room.estimated.round(5).tolist(),
                              "joint_qpos": room.data.qpos[7:].round(5).tolist(),
                              "head_encoder_rad": round(room.head, 5),
                              "gyro_xyz_rad_s": room.data.sensordata[
                                  room.gyro_addr:room.gyro_addr+3].round(5).tolist(),
                              "accel_xyz_m_s2": room.data.sensordata[
                                  addr:addr+3].round(5).tolist(),
                              "ray_world_points_path": str(path.relative_to(ROOT)),
                              "ray_world_points_sha256": sha(path),
                              "counts": dict(counts)})
                return clouds, counts

            search.shin_scan = retained_scan
            try:
                result = search.run_case(
                    seed, mode, vision, exemplar, owner_hist,
                    room_factory=factory, output_root=root,
                    sensor_case_prefix=f"v305-seed{seed}-{mode}",
                    max_clear_side_waits=MAX_CLEAR_SIDE_WAITS,
                    max_range_age_s=MAX_RANGE_AGE_S)
            finally:
                search.shin_scan = original_scan
            room = rooms[0]
            trace = branch / "sensor-and-motion-trace.json"
            trace.write_text(json.dumps({
                "case": case, "pose_source": source,
                "shin_observations": scans,
                "localization_observations": room.localization_events,
                "slips_evaluator_only": room.applied_slips,
                "controller_events": result["events"],
                "evaluator_path_min_clearance": room.path_min_clearance,
                "evaluator_motion_interrupted": room.motion_interrupted,
                "final_time_s": room.time_s}, indent=2) + "\n")
            row = {"case": case, "pose_source": source,
                   "status": result["status"],
                   "owner_questions": result["owner_questions"],
                   "motion_steps_requested": len(room.applied_slips),
                   "wait_rechecks": sum(event["action"] == "lateral_sector_recheck"
                                        for event in result["events"]),
                   "evaluator_path_min_clearance_m": round(
                       room.path_min_clearance[0], 5),
                   "evaluator_motion_interrupted": room.motion_interrupted,
                   "evaluator_pose_error_m": result["evaluator_pose_error_m"],
                   "route_decisions": [event for event in result["events"]
                                       if event["action"] in (
                                           "lateral_sector", "lateral_sector_recheck",
                                           "return_sector")],
                   "trace_path": str(trace.relative_to(ROOT)),
                   "trace_sha256": sha(trace),
                   "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                   "video_sha256": result["video_sha256"]}
            rows.append(row)
            (OUT / "heldout-moving-low-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(seed, mode, source, row["status"], "waits",
                  row["wait_rechecks"], "pathgap",
                  row["evaluator_path_min_clearance_m"], flush=True)
    report = {"scope": "V305 frozen changed-seed/timing split for the selected Direct Frame V303 dynamic owner-search policy: six even screen rooms with changed lighting/furniture jitter and independently scripted low moving-prop arrival/departure, four present plus absent and identical-lookalike, paired command/long-RGB-D pose sources. The actor uses one shared deterministic four-shin first-hit range input per pair, frozen 6.5cm point margin, 0.4s range-age limit and at most eight 0.4s wait/rechecks; upper RGB-D has deterministic 1cm noise/15% holes, IMU/joint channels are logged. Geometry and 20ms full-body path clearance are evaluator-only. A late corridor witness proves possible progress after the prop leaves, not physical safety. No native gait/braking, free-volume proof or calibrated physical sensors.",
              "source_sha256": sha(SOURCE),
              "controller_source_sha256": sha(
                  ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py"),
              "selected_visual_sha256": sha(selected_visual()),
              "geometry_preflight_sha256": sha(OUT / "geometry-preflight.json"),
              "case_specs": CASES, "rows": rows}
    (OUT / "heldout-moving-low-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "run_lateral_clutter_shin_search_v284.py").write_bytes(
        (ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py").read_bytes())


if __name__ == "__main__":
    main()
