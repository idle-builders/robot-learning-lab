"""Timed near-body sensing with a transferred native 50 cm stop tail.

The actor sees only timed range/depth-like rays and idealized yaw. Geometry,
hazard scripts and exact ape pose are evaluator inputs. The post-command path
is a displacement/joint-state splice from a *saved* native Direct Frame stop,
not a fresh learned-policy rollout or a qualified physical stop.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-native-stop-sensing/v3"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TRACE = ROOT / "artifacts/ape-cove50-effort-envelope/forward-stop-late.npz"
STOP_REPORT = ROOT / "artifacts/ape-skill-contract-cove50-20261009/stop-trace.json"
DT = .02
DEFAULT_STOP_S = 3.0
STOP_THRESHOLD_M = .55
MIN_MODELED_GAP_M = .05
MODES = ("none", "mid10", "mid15_matched", "low10_matched",
         "low15", "depth15", "mid_low_depth")
CASES = ("approaching_person", "low_pet", "static_offpath", "low_dropout", "false_return")
SENSORS = {
    "mid": {"period_s": .10, "max_m": 3.0, "origin": (.0, .0, .07),
            "yaw_deg": tuple(np.linspace(-90, 90, 73)), "pitch_deg": (0.,)},
    "low": {"period_s": 1/15, "max_m": 1.2, "origin": (.08, .0, -.15),
            "yaw_deg": tuple(np.linspace(-70, 70, 17)), "pitch_deg": (0.,)},
    "mid_match": {"period_s": 1/15, "max_m": 1.2,
                  "origin": (.08, .0, .07),
                  "yaw_deg": tuple(np.linspace(-70, 70, 17)),
                  "pitch_deg": (0.,)},
    "low10": {"period_s": .10, "max_m": 1.2,
              "origin": (.08, .0, -.15),
              "yaw_deg": tuple(np.linspace(-70, 70, 17)),
              "pitch_deg": (0.,)},
    "depth": {"period_s": 1/15, "max_m": 2.0, "origin": (.04, .0, .17),
              "yaw_deg": tuple(np.linspace(-35, 35, 9)),
              "pitch_deg": (-15., -25., -35., -45., -55.)},
}
MODE_SENSORS = {"none": (), "mid10": ("mid",),
                "mid15_matched": ("mid_match",),
                "low10_matched": ("low10",), "low15": ("low",),
                "depth15": ("depth",),
                "mid_low_depth": ("mid", "low", "depth")}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def make_scene() -> str:
    root = ET.parse(BODY).getroot()
    # Local MuJoCo 3.3.1 lacks MJCF contact-sensor syntax from the source's
    # 3.13.0 run. Saved sole packets are used; geometry/joints stay intact.
    sensor = root.find("sensor")
    for contact in list(sensor.findall("contact")):
        sensor.remove(contact)
    world = root.find("worldbody")
    floor = world.find("geom")
    floor.set("group", "0")
    floor.set("rgba", ".67 .66 .61 1")
    for geom in root.findall(".//geom"):
        if geom is floor:
            continue
        if geom.get("group") == "4" or geom.get("group") is None:
            color = geom.get("rgba")
            if color and float(color.split()[-1]) == 0:
                geom.set("rgba", " ".join(color.split()[:3]) + " .82")
    ET.SubElement(world, "camera", name="overview", pos="1.55 -3.25 1.65",
                  xyaxes="1 0 0 0 .48 .88", fovy="48")
    ET.SubElement(world, "light", pos="1.4 -1.6 2.6", dir="0 0 -1")
    for name, xyz, size, color in (
        ("sofa", "2.5 -.35 .27", ".42 .19 .27", ".25 .42 .47 1"),
        ("table", ".55 -.05 .25", ".22 .20 .25", ".47 .29 .18 1"),
        ("offpath_box", "1.65 -.48 .10", ".12 .11 .10", ".63 .38 .20 1"),
    ):
        ET.SubElement(world, "geom", name=name, type="box", pos=xyz,
                      size=size, group="0", rgba=color, contype="0",
                      conaffinity="0")
    person = ET.SubElement(world, "body", name="person", pos="10 -1 0")
    ET.SubElement(person, "geom", name="person_capsule", type="capsule",
                  fromto="0 0 .10 0 0 1.30", size=".09", group="0",
                  rgba=".75 .42 .36 1")
    pet = ET.SubElement(world, "body", name="pet", pos="10 -1 0")
    ET.SubElement(pet, "geom", name="pet_box", type="box", pos="0 0 .08",
                  size=".18 .08 .08", group="0", rgba=".66 .51 .30 1")
    return ET.tostring(root, encoding="unicode")


def hazard_xy(case: str, t: float) -> tuple[str, tuple[float, float]]:
    if case == "approaching_person":
        return "person", (2.60 - .18*min(t, 4.0), -1.0)
    if case == "low_pet":
        return "pet", (2.25 - .18*min(t, 3.4), -1.0)
    if case == "static_offpath" or case == "false_return":
        return "person", (2.1, -.40)
    if case == "low_dropout":
        return "pet", (2.45 - .18*min(t, 3.7), -1.0)
    raise ValueError(case)


def set_hazard(model: mujoco.MjModel, case: str, t: float) -> int:
    kind, xy = hazard_xy(case, t)
    model.body_pos[model.body("person").id, :2] = xy if kind == "person" else (10., -1.)
    model.body_pos[model.body("pet").id, :2] = xy if kind == "pet" else (10., -1.)
    return model.geom("person_capsule" if kind == "person" else "pet_box").id


def native_pose(saved: np.lib.npyio.NpzFile, world: int, t: float,
                stop_at: float) -> tuple[np.ndarray, np.ndarray]:
    index = min(len(saved["time_s"])-1, max(0, round(t/DT)-1))
    native_stop = round(DEFAULT_STOP_S/DT)-1
    # The four native worlds were spawned at different XY origins, all facing
    # approximately +X. Rebase only translation to world 0's start so they
    # face identical hazard/furniture geometry; retain each world's motion.
    offset = (saved["root_pose"][0, 0, :2] -
              saved["root_pose"][0, world, :2]).astype(float)
    if t <= stop_at:
        pose = saved["root_pose"][index, world].astype(float)
        pose[:2] += offset
        return pose, saved["joint_pos"][index, world].astype(float)
    trigger_index = min(native_stop, max(0, round(stop_at/DT)-1))
    tail_index = min(len(saved["time_s"])-1,
                     native_stop + max(0, round((t-stop_at)/DT)))
    anchor = saved["root_pose"][trigger_index, world].astype(float)
    tail_anchor = saved["root_pose"][native_stop, world].astype(float)
    tail = saved["root_pose"][tail_index, world].astype(float)
    pose = tail.copy()
    pose[:3] = anchor[:3] + tail[:3] - tail_anchor[:3]
    pose[:2] += offset
    return pose, saved["joint_pos"][tail_index, world].astype(float)


def put_pose(model: mujoco.MjModel, data: mujoco.MjData,
             saved: np.lib.npyio.NpzFile, world: int, t: float,
             stop_at: float) -> np.ndarray:
    pose, joints = native_pose(saved, world, t, stop_at)
    data.qpos[:] = model.qpos0
    data.qpos[:7] = pose
    for name, angle in zip(saved["leg_names"], joints):
        data.qpos[model.joint(str(name)).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    return pose


def read_rays(model: mujoco.MjModel, data: mujoco.MjData,
              name: str, pose: np.ndarray, case: str, t: float) -> dict:
    spec = SENSORS[name]
    heading = yaw_from_quat(pose[3:7])
    ch, sh = math.cos(heading), math.sin(heading)
    local = np.asarray(spec["origin"])
    origin = pose[:3] + np.array([ch*local[0]-sh*local[1],
                                   sh*local[0]+ch*local[1], local[2]])
    ranges = []
    bearings = []
    for pitch_deg in spec["pitch_deg"]:
        pitch = math.radians(pitch_deg)
        for yaw_deg in spec["yaw_deg"]:
            bearing = math.radians(yaw_deg)
            angle = heading + bearing
            direction = np.array([math.cos(pitch)*math.cos(angle),
                                  math.cos(pitch)*math.sin(angle), math.sin(pitch)])
            geomid = np.array([-1], dtype=np.int32)
            hit = mujoco.mj_ray(model, data, origin, direction,
                                np.array([1, 0, 0, 0, 0, 0], np.uint8),
                                1, -1, geomid)
            if hit < 0 or hit > spec["max_m"]:
                value = math.inf
            else:
                end_z = origin[2] + hit*direction[2]
                # Floor consistency is geometric calibration, not a sim label.
                value = math.inf if name == "depth" and end_z < .025 else float(hit)
            ranges.append(value)
            bearings.append(bearing)
    if case == "false_return" and name.startswith("low") and abs(t-1.50) < .025:
        center = len(ranges)//2
        ranges[center] = .18
    good = [r for r, b in zip(ranges, bearings)
            if math.isfinite(r) and abs(b) <= math.radians(25)]
    return {"sensor": name, "captured_at_s": round(t, 4),
            "min_forward_range_m": round(min(good), 5) if good else None,
            "valid_returns": sum(math.isfinite(r) for r in ranges),
            "ranges_m": [round(r, 5) if math.isfinite(r) else None for r in ranges]}


def model_gap(model: mujoco.MjModel, data: mujoco.MjData,
              hazard_geom: int, ape_geoms: list[int]) -> float:
    holder = np.zeros(6)
    return min(float(mujoco.mj_geomDistance(model, data, i, hazard_geom,
                                            2.0, holder)) for i in ape_geoms)


def midstep_gaps(model: mujoco.MjModel, data: mujoco.MjData,
                 saved: np.lib.npyio.NpzFile, world: int, case: str,
                 stop_at: float, times: np.ndarray,
                 ape_geoms: list[int]) -> np.ndarray:
    """Score the between-packet swept pose at 10 ms, not just 50 Hz endpoints."""
    result = np.empty(len(times)-1)
    for i in range(len(result)):
        t0, t1 = times[i:i+2]
        p0, j0 = native_pose(saved, world, float(t0), stop_at)
        p1, j1 = native_pose(saved, world, float(t1), stop_at)
        pose = (p0+p1)/2
        pose[3:7] /= np.linalg.norm(pose[3:7])
        data.qpos[:] = model.qpos0
        data.qpos[:7] = pose
        for name, angle in zip(saved["leg_names"], (j0+j1)/2):
            data.qpos[model.joint(str(name)).qposadr[0]] = angle
        hazard = set_hazard(model, case, float((t0+t1)/2))
        mujoco.mj_forward(model, data)
        result[i] = model_gap(model, data, hazard, ape_geoms)
    return result


def overlay(frame: np.ndarray, row: dict, case: str, mode: str) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    panel = np.zeros((82, image.shape[1], 3), np.uint8)
    panel[:] = (26, 28, 31)
    lines = [f"{case} | {mode} | t={row['time_s']:.2f}s | {row['action']}",
             f"min sensor={row['min_sensor_range_m']} m  model gap={row['evaluator_gap_m']:.3f} m",
             f"stop command={row['stop_command_s']} s  (historical native-tail transfer)"]
    for n, line in enumerate(lines):
        cv2.putText(panel, line, (12, 25+n*24), cv2.FONT_HERSHEY_SIMPLEX,
                    .54, (220, 230, 232), 1, cv2.LINE_AA)
    return np.concatenate((image, panel), axis=0)


def run_case(model: mujoco.MjModel, saved: np.lib.npyio.NpzFile,
             world: int, case: str, mode: str, stop_delay_s: float,
             *, video: bool) -> dict:
    data = mujoco.MjData(model)
    model.body_pos[model.body("person").id, :2] = (10., -1.)
    model.body_pos[model.body("pet").id, :2] = (10., -1.)
    configured = MODE_SENSORS[mode]
    next_capture = {name: 0. for name in configured}
    last_capture = {name: None for name in configured}
    streak = {name: 0 for name in configured}
    packet_rows = []
    rows = []
    last_forward_min = None
    recent_range = {name: None for name in configured}
    stop_at = DEFAULT_STOP_S
    stop_reason = "planned_end"
    renderer = mujoco.Renderer(model, height=480, width=640) if video else None
    view_options = mujoco.MjvOption() if video else None
    if view_options is not None:
        view_options.geomgroup[4] = 1
    frames = []
    ape_geoms = [i for i in range(model.ngeom)
                 if model.geom_group[i] == 4 and model.geom_bodyid[i] != 0]
    for index in range(len(saved["time_s"])):
        t = float(saved["time_s"][index])
        hazard_geom = set_hazard(model, case, t)
        pose = put_pose(model, data, saved, world, t, stop_at)
        current_min = []
        if t <= stop_at:
            for name in configured:
                if t + 1e-8 < next_capture[name]:
                    continue
                next_capture[name] += SENSORS[name]["period_s"]
                if case == "low_dropout" and name.startswith("low") and 1.2 <= t < 2.4:
                    packet_rows.append({"sensor": name, "expected_at_s": round(t, 4),
                                        "status": "dropped"})
                    continue
                packet = read_rays(model, data, name, pose, case, t)
                packet_rows.append(packet)
                last_capture[name] = t
                distance = packet["min_forward_range_m"]
                current_min.append(distance)
                recent_range[name] = distance
                last_forward_min = min((v for v in recent_range.values()
                                        if v is not None), default=None)
                streak[name] = streak[name]+1 if (distance is not None and
                                     distance <= STOP_THRESHOLD_M) else 0
                if streak[name] >= 2 and stop_reason == "planned_end":
                    stop_at, stop_reason = t, f"two_{name}_near_returns"
            if stop_reason == "planned_end" and t >= .30:
                for name in configured:
                    last = last_capture[name]
                    if last is None or t-last > 2*SENSORS[name]["period_s"]+.02:
                        stop_at, stop_reason = t, f"{name}_stale"
                        break
        gap = model_gap(model, data, hazard_geom, ape_geoms)
        action = ("forward" if t < stop_at else
                  "stop_tail" if t < stop_at + stop_delay_s else
                  "observed_stationary")
        row = {"time_s": round(t, 4), "action": action,
               "ideal_yaw_rad_actor": round(yaw_from_quat(pose[3:7]), 6),
               "leg_joint_rad_actor": data.qpos[7:17].round(6).tolist(),
               "root_xy_evaluator_m": pose[:2].round(5).tolist(),
               "hazard_xy_evaluator_m": list(hazard_xy(case, t)[1]),
               "evaluator_gap_m": round(gap, 5),
               "min_sensor_range_m": last_forward_min,
               "stop_command_s": round(stop_at, 4) if t >= stop_at else None}
        rows.append(row)
        if renderer and index % 4 == 0:
            renderer.update_scene(data, camera="overview", scene_option=view_options)
            frames.append(overlay(renderer.render(), row, case, mode))
    if renderer:
        renderer.close()
    video_meta = None
    if frames:
        path = OUT / "videos" / f"{case}-{mode}-world{world}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 12.5, (frames[0].shape[1], frames[0].shape[0]))
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")
        for frame in frames:
            writer.write(frame)
        writer.release()
        video_meta = {"path": str(path), "sha256": sha(path), "frames": len(frames)}
    times = np.asarray([row["time_s"] for row in rows])
    gaps = np.asarray([row["evaluator_gap_m"] for row in rows])
    mid_times = (times[:-1]+times[1:])/2
    between_gaps = midstep_gaps(model, data, saved, world, case, stop_at,
                                times, ape_geoms)
    swept_times = np.concatenate((times, mid_times))
    swept_gaps = np.concatenate((gaps, between_gaps))
    last_forward = next((row for row in reversed(rows) if row["action"] == "forward"), rows[0])
    observed_at = stop_at + stop_delay_s
    before_observed = swept_gaps[swept_times <= observed_at]
    after_observed = swept_gaps[swept_times > observed_at]
    result = {"world": world, "case": case, "mode": mode,
              "stop_command_s": round(stop_at, 4), "stop_reason": stop_reason,
              "native_command_to_observed_stop_s": round(stop_delay_s, 4),
              "observed_stop_s": round(observed_at, 4),
              "min_modeled_body_gap_m": round(float(swept_gaps.min()), 5),
              "min_gap_before_observed_stop_m": round(float(before_observed.min()), 5),
              "min_gap_after_observed_stop_m": (round(float(after_observed.min()), 5)
                                                 if len(after_observed) else None),
              "modeled_overlap": bool((swept_gaps <= 0).any()),
              "below_5cm_margin": bool((swept_gaps < MIN_MODELED_GAP_M).any()),
              "first_overlap_s": (round(float(swept_times[swept_gaps <= 0].min()), 4)
                                  if (swept_gaps <= 0).any() else None),
              "pre_stop_progress_m": round(last_forward["root_xy_evaluator_m"][0]-
                                           rows[0]["root_xy_evaluator_m"][0], 5),
              "false_early_stop_on_offpath": bool(case in ("static_offpath", "false_return")
                                                  and stop_at < DEFAULT_STOP_S),
              "packet_count": len(packet_rows), "video": video_meta}
    path = OUT / "episodes" / f"{case}-{mode}-world{world}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
        time_s=np.asarray([r["time_s"] for r in rows]),
        action=np.asarray([r["action"] for r in rows]),
        ideal_yaw_rad_actor=np.asarray([r["ideal_yaw_rad_actor"] for r in rows]),
        leg_joint_rad_actor=np.asarray([r["leg_joint_rad_actor"] for r in rows]),
        root_xy_evaluator_m=np.asarray([r["root_xy_evaluator_m"] for r in rows]),
        hazard_xy_evaluator_m=np.asarray([r["hazard_xy_evaluator_m"] for r in rows]),
        gap_evaluator_m=gaps,
        midstep_time_s=mid_times,
        midstep_gap_evaluator_m=between_gaps,
        packet_json=np.asarray([json.dumps(r, sort_keys=True) for r in packet_rows]))
    result["trace"] = {"path": str(path), "sha256": sha(path),
                       "motion_samples": len(rows), "packets": len(packet_rows)}
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xml = make_scene()
    xml_path = OUT / "scene.xml"
    xml_path.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    saved = np.load(TRACE, allow_pickle=False)
    stop_report = json.loads(STOP_REPORT.read_text())
    if (saved["root_pose"].shape[:2] != (400, 4) or
        not np.isclose(float(saved["step_dt_s"]), DT) or
        stop_report["sources"]["forward-stop-late"]["native_trace_sha256"] != sha(TRACE)):
        raise ValueError("Native stop source changed")
    selected_videos = {("approaching_person", "mid10", 0),
                       ("approaching_person", "mid_low_depth", 0),
                       ("low_pet", "mid10", 0),
                       ("low_pet", "mid15_matched", 0),
                       ("low_pet", "low10_matched", 0),
                       ("low_pet", "low15", 0),
                       ("false_return", "low15", 0)}
    native_delays = {row["world"]: row["command_to_observed_stop_s"]
                     for row in stop_report["rows"]
                     if row["case"] == "forward-stop-late"}
    if set(native_delays) != set(range(4)):
        raise ValueError("Expected four native late-stop worlds")
    results = []
    for world in range(4):
        for case in CASES:
            for mode in MODES:
                result = run_case(model, saved, world, case, mode,
                                  native_delays[world],
                                  video=(case, mode, world) in selected_videos)
                results.append(result)
                print(world, case, mode, result["stop_reason"],
                      result["min_modeled_body_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.native-stop-sensing.v3",
              "task": "T-7f9ef6", "body_status": "historical_rejected_direct_frame",
              "method": "Hybrid closed-loop response-law transfer: each native world is translation-rebased to world 0's start; recorded native forward path runs until actor stop decision, then a shifted saved native stop-tail displacement and joint sequence. No new policy rollout. Timed ideal rays are actor packets. Hazard poses and exact geometry are evaluator-only.",
              "source_sha256": sha(Path(__file__)), "body_sha256": sha(BODY),
              "native_trace_sha256": sha(TRACE), "stop_report_sha256": sha(STOP_REPORT),
              "scene_sha256": sha(xml_path), "mujoco_version": mujoco.__version__,
              "config": {"dt_s": DT, "planned_stop_s": DEFAULT_STOP_S,
                         "stop_threshold_m": STOP_THRESHOLD_M,
                         "minimum_modeled_gap_m": MIN_MODELED_GAP_M,
                         "two_successive_packets_required": True,
                         "sensors": SENSORS, "modes": MODE_SENSORS,
                         "cases": CASES},
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    saved.close()


if __name__ == "__main__":
    main()
