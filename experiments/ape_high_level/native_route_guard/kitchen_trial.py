"""Native 50 cm replay with a full-scan stop guard in a RoboCasa kitchen.

The owner-sized capsule is scripted; the actor gets only ideal timed rays and
ideal pose. A hazard-free same-room ray survey is the optimistic prior map.
The root/joints follow a saved historical response-law splice, not a new
learned-policy rollout or dynamic interaction with kitchen fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import kitchen_native as kitchen
import run as route_guard


OUT = kitchen.OUT / "joined-trial"
CASES = ("fast_clear", "slow_cross", "pause_path")
MODES = ("none", "mapped_low_guard")
DT = .02
STOP_THRESHOLD_M = .75
MIN_MARGIN_M = .05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hazard_xy(case: str, t: float) -> tuple[float, float]:
    y = {
        "fast_clear": -1.05-.60*min(t, 4.0),
        "slow_cross": -1.05-.25*min(t, 4.0),
        "pause_path": -1.05-.42*min(t, 1.8),
    }[case]
    return 3.55, y


def set_hazard(model: mujoco.MjModel, case: str, t: float) -> int:
    model.body_pos[model.body("approaching_person").id, :2] = hazard_xy(case, t)
    model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
    return model.geom("person_capsule").id


def background_map(model: mujoco.MjModel, saved) -> np.ndarray:
    model.body_pos[model.body("approaching_person").id, :2] = (10., -1.)
    model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
    data = mujoco.MjData(model)
    points = []
    for world in range(4):
        for t in np.arange(.02, 3.0, .06):
            pose = kitchen.put_native_pose(model, data, saved, world, float(t), 3.0)
            packet = route_guard.ORIGINAL_READ(model, data, "low", pose,
                                                "static_offpath", float(t))
            points.extend(route_guard.low_ray_points(packet, pose))
    bins = {tuple(np.round(p/.02).astype(int)) for p in points}
    return np.asarray(sorted(bins), dtype=float).reshape(-1, 2)*.02


def body_gap(model, data, hazard_geom, ape_geoms, holder) -> float:
    return min(float(mujoco.mj_geomDistance(model, data, i, hazard_geom,
                                            2.0, holder)) for i in ape_geoms)


def kitchen_geoms_near_route(model, data) -> list[int]:
    selected = []
    for i in range(model.ngeom):
        if (model.geom_group[i] == 4 or
            model.geom(i).name in ("person_capsule", "pet_box") or
            model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE or
            "floor" in model.geom(i).name.lower()):
            continue
        xy = data.geom_xpos[i, :2]
        radius = model.geom_rbound[i]
        if 2.2-radius < xy[0] < 4.2+radius and -2.5-radius < xy[1] < -1.1+radius:
            selected.append(i)
    return selected


def fixture_gap(model, data, ape_geoms, fixture_geoms, holder) -> float:
    return min(float(mujoco.mj_geomDistance(model, data, i, j, 2.0, holder))
               for i in ape_geoms for j in fixture_geoms)


def overlay(frame: np.ndarray, row: dict, case: str, mode: str) -> np.ndarray:
    panel = np.zeros((82, frame.shape[1], 3), dtype=np.uint8)
    panel[:] = (25, 28, 32)
    lines = [f"RoboCasa layout1 | {case} | {mode} | t={row['time_s']:.2f}s",
             f"{row['action']}  guard={row['guard_risk']}  body-person gap={row['evaluator_gap_m']:.3f}m",
             f"stop command={row['stop_command_s']}  historical native-tail transfer"]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (12, 24+i*24), cv2.FONT_HERSHEY_SIMPLEX,
                    .53, (228, 234, 238), 1, cv2.LINE_AA)
    return np.concatenate([frame, panel], axis=0)


def run_episode(model, saved, background: np.ndarray, world: int,
                case: str, mode: str, stop_delay_s: float, *, video: bool) -> dict:
    data = mujoco.MjData(model)
    ape_geoms = [i for i in range(model.ngeom) if model.geom_group[i] == 4]
    holder = np.zeros(6)
    stop_at = 3.0
    reason = "planned_end"
    streak = 0
    capture_at = .02
    guard = route_guard.FullScanGuard(background)
    packets = []
    rows = []
    qposes = []
    frames = []
    renderer = mujoco.Renderer(model, height=480, width=640) if video else None
    view = mujoco.MjvCamera() if video else None
    options = mujoco.MjvOption() if video else None
    if video:
        view.type = mujoco.mjtCamera.mjCAMERA_FREE
        view.lookat[:] = [3.15, -1.80, .30]
        view.distance = 2.5
        view.elevation = -25
        view.azimuth = 0
        options.geomgroup[4] = 1
    set_hazard(model, case, .02)
    kitchen.put_native_pose(model, data, saved, world, .02, stop_at)
    fixture_geoms = kitchen_geoms_near_route(model, data)
    fixture_min = float("inf")
    for index, time in enumerate(saved["time_s"]):
        t = float(time)
        hazard_geom = set_hazard(model, case, t)
        pose = kitchen.put_native_pose(model, data, saved, world, t, stop_at)
        risk = None
        if mode == "mapped_low_guard" and t <= stop_at and t+1e-8 >= capture_at:
            capture_at += 1/15
            packet = route_guard.ORIGINAL_READ(model, data, "low", pose,
                                                "static_offpath", t)
            endpoints = route_guard.low_ray_points(packet, pose)
            matched = sum(guard.background.query(point)[0] <=
                          route_guard.BACKGROUND_MATCH_M for point in endpoints)
            packet["static_matched_returns"] = matched
            packet["unmatched_returns"] = len(endpoints)-matched
            decision = guard.update(packet, pose, t)
            packet["route_guard"] = decision
            packet["raw_min_forward_range_m"] = packet["min_forward_range_m"]
            packets.append(packet)
            risk = decision["risk"]
            streak = streak+1 if risk else 0
            if streak >= 2 and reason == "planned_end":
                stop_at, reason = t, "two_low_guard_risk_packets"
        gap = body_gap(model, data, hazard_geom, ape_geoms, holder)
        if index % 5 == 0:
            fixture_min = min(fixture_min, fixture_gap(model, data,
                                      ape_geoms, fixture_geoms, holder))
        action = ("forward" if t < stop_at else
                  "stop_tail" if t < stop_at+stop_delay_s else
                  "observed_stationary")
        row = {"time_s": round(t, 4), "action": action,
               "root_xy_evaluator_m": pose[:2].round(5).tolist(),
               "person_xy_evaluator_m": list(hazard_xy(case, t)),
               "evaluator_gap_m": round(gap, 5),
               "guard_risk": risk,
               "stop_command_s": round(stop_at, 4) if t >= stop_at else None}
        rows.append(row)
        qposes.append(data.qpos.copy())
        if renderer and index % 4 == 0:
            renderer.update_scene(data, camera=view, scene_option=options)
            overview = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
            renderer.update_scene(data, camera="ape_lower_rgb")
            head = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
            canvas = np.concatenate([overview, cv2.resize(head, (320, 480))], axis=1)
            frames.append(overlay(canvas, row, case, mode))
    if renderer:
        renderer.close()
    video_meta = None
    if frames:
        movie = OUT / "videos" / f"{case}-{mode}-world{world}.mp4"
        movie.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"),
                                 12.5, (frames[0].shape[1], frames[0].shape[0]))
        if not writer.isOpened():
            raise RuntimeError("MP4 writer failed")
        for frame in frames:
            writer.write(frame)
        writer.release()
        video_meta = {"path": str(movie), "sha256": sha(movie),
                      "frames": len(frames)}
    times = np.asarray([row["time_s"] for row in rows])
    gaps = np.asarray([row["evaluator_gap_m"] for row in rows])
    mids = (times[:-1]+times[1:])/2
    mid_gaps = np.empty(len(mids))
    root_adr = model.joint("ape_root").qposadr[0]
    for i, mid in enumerate(mids):
        set_hazard(model, case, float(mid))
        data.qpos[:] = (qposes[i]+qposes[i+1])/2
        quat = data.qpos[root_adr+3:root_adr+7]
        quat /= np.linalg.norm(quat)
        mujoco.mj_forward(model, data)
        mid_gaps[i] = body_gap(model, data, hazard_geom, ape_geoms, holder)
    swept = np.r_[gaps, mid_gaps]
    trace = OUT / "episodes" / f"{case}-{mode}-world{world}.npz"
    trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace, time_s=times, gap_evaluator_m=gaps,
                        midstep_time_s=mids, midstep_gap_evaluator_m=mid_gaps,
                        root_xy_evaluator_m=np.asarray([r["root_xy_evaluator_m"] for r in rows]),
                        person_xy_evaluator_m=np.asarray([r["person_xy_evaluator_m"] for r in rows]),
                        action=np.asarray([r["action"] for r in rows]),
                        packet_json=np.asarray([json.dumps(p, sort_keys=True)
                                                for p in packets]))
    return {"world": world, "case": case, "mode": mode,
            "stop_command_s": round(stop_at, 4), "stop_reason": reason,
            "native_command_to_observed_stop_s": stop_delay_s,
            "min_modeled_body_person_gap_m": round(float(swept.min()), 5),
            "modeled_person_overlap": bool((swept <= 0).any()),
            "below_5cm_person_margin": bool((swept < MIN_MARGIN_M).any()),
            "min_sampled_body_fixture_gap_m": round(fixture_min, 5),
            "fixture_gap_sample_period_s": .1,
            "packet_count": len(packets),
            "video": video_meta,
            "trace": {"path": str(trace), "sha256": sha(trace),
                      "motion_samples": len(times), "packets": len(packets)}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        background = background_map(model, saved)
        map_path = OUT / "background_xy.npy"
        np.save(map_path, background)
        stop = json.loads(route_guard.replay.STOP_REPORT.read_text())
        delays = {r["world"]: r["command_to_observed_stop_s"]
                  for r in stop["rows"] if r["case"] == "forward-stop-late"}
        results = []
        for world in range(4):
            for case in CASES:
                for mode in MODES:
                    video = world == 0 and (mode == "mapped_low_guard" or
                                           case == "slow_cross")
                    result = run_episode(model, saved, background, world, case,
                                         mode, delays[world], video=video)
                    results.append(result)
                    print(world, case, mode, result["stop_command_s"],
                          result["min_modeled_body_person_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.ape-robocasa-native-guard.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "Historical 50 Hz native path until a timed ideal low-ray guard decision, then shifted saved native stop-tail root/joint states. Static map was built from hazard-free scans of the same scene along all four future native paths. No fresh policy rollout or physical contact response.",
              "room": {"layout_id": 1, "style_id": 1,
                       "scene_sha256": sha(kitchen.OUT / "scene.xml")},
              "sensor": {"name": "low", "period_s": 1/15,
                         "max_range_m": 1.2,
                         "angle_deg": [-70, 70], "rays": 17,
                         "ideal_odom": True,
                         "static_map_points": len(background),
                         "prior_map_sha256": sha(map_path)},
              "source_sha256": sha(Path(__file__)),
              "scene_builder_sha256": sha(Path(kitchen.__file__)),
              "guard_source_sha256": sha(Path(route_guard.__file__)),
              "native_trace_sha256": sha(kitchen.TRACE),
              "stop_report_sha256": sha(route_guard.replay.STOP_REPORT),
              "body_sha256": sha(kitchen.BODY),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
