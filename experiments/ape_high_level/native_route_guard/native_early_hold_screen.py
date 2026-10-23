"""V29: actual early/late native stop traces against V27 late side entry.

This evaluates a proactive shorter forward request using two retained policy
traces from the same seed/body. It is an evaluator feasibility screen: no
actor knows the future person path or selects the earlier stop in this run.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import held_body_intrusion_trial as late
import kitchen_native as kitchen
import kitchen_trial as trial


OUT = kitchen.OUT / "v29-native-early-hold"
TRACE_ROOT = kitchen.ROOT / "artifacts/ape-cove50-effort-envelope"
MODES = {"early_2p5s": (TRACE_ROOT / "forward-stop-early.npz", 2.5),
         "late_3p0s": (TRACE_ROOT / "forward-stop-late.npz", 3.0)}


def direct_pose(model: mujoco.MjModel, data: mujoco.MjData, saved,
                world: int, index: int) -> np.ndarray:
    pose = saved["root_pose"][index, world].copy()
    pose[:2] += kitchen.START_XY-saved["root_pose"][0, world, :2]
    joints = saved["joint_pos"][index, world]
    data.qpos[:] = model.qpos0
    root = model.joint("ape_root").qposadr[0]
    data.qpos[root:root+7] = pose
    for name, angle in zip(saved["leg_names"], joints):
        data.qpos[model.joint("ape_"+str(name)).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    return pose


def render_overlay(frame: np.ndarray, case: str, mode: str,
                   t: float, gap: float, stop_s: float) -> np.ndarray:
    panel = np.full((83, frame.shape[1], 3), (29, 32, 37), dtype=np.uint8)
    for line, y, color in ((
        (f"RoboCasa layout 3 + box stool | {mode} | {case} | t={t:.2f}s",
         24, (227, 233, 238)),
        (f"native zero-twist request={stop_s:.2f}s | full-body person gap={gap:+.3f}m",
         51, (77, 96, 226) if gap < .05 else (227, 233, 238)),
        ("Historical Direct Frame; kinematic scene replay, no live collision response",
         76, (167, 184, 196)),
    )):
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .52, color, 1, cv2.LINE_AA)
    return np.concatenate([frame, panel], axis=0)


def one(model, saved, source: Path, stop_s: float,
        world: int, case: str, mode: str, *, video: bool) -> dict:
    data = mujoco.MjData(model)
    ape_geoms = [i for i in range(model.ngeom) if model.geom_group[i] == 4]
    holder = np.zeros(6)
    person_geom = model.geom("person_capsule").id
    model.body_pos[model.body("approaching_person").id, :2] = \
        late.hazard_xy(case, .02)
    direct_pose(model, data, saved, world, 0)
    fixtures = trial.kitchen_geoms_near_route(model, data)
    fixture_min = float("inf")
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
    for index, raw_t in enumerate(saved["time_s"]):
        t = float(raw_t)
        model.body_pos[model.body("approaching_person").id, :2] = \
            late.hazard_xy(case, t)
        model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
        pose = direct_pose(model, data, saved, world, index)
        gap = trial.body_gap(model, data, person_geom, ape_geoms, holder)
        if index % 5 == 0:
            fixture_min = min(fixture_min, trial.fixture_gap(
                model, data, ape_geoms, fixtures, holder))
        rows.append({"time_s": round(t, 5), "gap_m": round(gap, 5),
                     "root_xy_m": pose[:2].round(5).tolist()})
        qposes.append(data.qpos.copy())
        if renderer and index % 4 == 0:
            renderer.update_scene(data, camera=view, scene_option=options)
            overview = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
            renderer.update_scene(data, camera="ape_lower_rgb")
            head = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
            canvas = np.concatenate([overview, cv2.resize(head, (320, 480))], axis=1)
            frames.append(render_overlay(canvas, case, mode, t, gap, stop_s))
    if renderer:
        renderer.close()
    times = np.asarray([r["time_s"] for r in rows])
    gaps = np.asarray([r["gap_m"] for r in rows])
    mids = (times[:-1]+times[1:])/2
    mid_gaps = np.empty(len(mids))
    root_adr = model.joint("ape_root").qposadr[0]
    for i, t in enumerate(mids):
        model.body_pos[model.body("approaching_person").id, :2] = \
            late.hazard_xy(case, float(t))
        data.qpos[:] = (qposes[i]+qposes[i+1])/2
        quat = data.qpos[root_adr+3:root_adr+7]
        quat /= np.linalg.norm(quat)
        mujoco.mj_forward(model, data)
        mid_gaps[i] = trial.body_gap(model, data, person_geom, ape_geoms, holder)
    all_gaps = np.r_[gaps, mid_gaps]
    all_times = np.r_[times, mids]
    trace = OUT / "episodes" / f"{mode}-{case}-world{world}.npz"
    trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace, time_s=times, gap_m=gaps,
                        midstep_time_s=mids, midstep_gap_m=mid_gaps,
                        root_xy_m=np.asarray([r["root_xy_m"] for r in rows]))
    media = None
    if frames:
        movie = OUT / "videos" / f"{mode}-{case}-world{world}.mp4"
        movie.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"),
                                 12.5, (frames[0].shape[1], frames[0].shape[0]))
        if not writer.isOpened():
            raise RuntimeError("MP4 writer failed")
        for frame in frames:
            writer.write(frame)
        writer.release()
        media = {"path": str(movie), "sha256": trial.sha(movie),
                 "frames": len(frames)}
    first_below = all_times[all_gaps < trial.MIN_MARGIN_M]
    root_xy = np.asarray([r["root_xy_m"] for r in rows])
    stop_index = round(stop_s/.02)-1
    return {"mode": mode, "world": world, "case": case,
            "native_stop_command_s": stop_s,
            "min_body_person_gap_m": round(float(all_gaps.min()), 5),
            "below_5cm": bool((all_gaps < trial.MIN_MARGIN_M).any()),
            "overlap": bool((all_gaps <= 0).any()),
            "first_below_5cm_s": (round(float(first_below.min()), 5)
                                   if len(first_below) else None),
            "min_sampled_body_fixture_gap_m": round(fixture_min, 5),
            "root_command_progress_m": round(float(np.linalg.norm(
                root_xy[stop_index]-root_xy[0])), 5),
            "root_final_progress_m": round(float(np.linalg.norm(
                root_xy[-1]-root_xy[0])), 5),
            "trace_source_sha256": trial.sha(source),
            "trace": {"path": str(trace), "sha256": trial.sha(trace)},
            "video": media}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = late.previous.OUT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    results = []
    native_info = {}
    for mode, (source, stop_s) in MODES.items():
        info = json.loads(source.with_suffix(".json").read_text())
        if info["stop_at_s"] != stop_s:
            raise RuntimeError(f"Unexpected native stop time in {source}")
        native_info[mode] = {"trace_sha256": trial.sha(source),
                             "report_sha256": trial.sha(source.with_suffix(".json")),
                             "initial_state_sha256": info["initial_state_sha256"],
                             "body_sha256": info["body_sha256"],
                             "stop_at_s": stop_s}
        with np.load(source, allow_pickle=False) as saved:
            for world in range(4):
                for case in late.CASES:
                    row = one(model, saved, source, stop_s, world, case, mode,
                              video=(world == 0))
                    results.append(row)
                    print(mode, world, case, row["min_body_person_gap_m"],
                          row["root_final_progress_m"], flush=True)
    report = {"schema": "robot-learning-lab.ape-native-early-hold-screen.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "Evaluator-only paired screen: use actual 2.5s and 3.0s zero-twist native policy traces from the same body, seed and initial state; place each into the V27 box-stool RoboCasa scene and replay the same late left/right capsule. Evaluate articulated full-body person gaps at20ms endpoints plus10ms midpoints and fixture gaps every100ms. No actor chooses the early stop, no reactive escape, and no physical interaction with person or furniture.",
              "scene_sha256": trial.sha(scene),
              "source_sha256": trial.sha(Path(__file__)),
              "body_sha256": trial.sha(kitchen.BODY),
              "v27_report_sha256": trial.sha(late.OUT / "report.json"),
              "native_sources": native_info,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
