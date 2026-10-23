"""Independently check timed sensor/action packets and proxy geometry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha


PRECOMMIT = ART / "cross_room_inference_response_precommit.json"
REPORT = ART / "cross_room_inference_response_report.json"
OUT = ART / "cross_room_inference_response_audit.json"
BODY_RADIUS = .17
ANGLES = np.arange(-180., 180., 5.)


def video_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def actor_at(start: np.ndarray, direction: np.ndarray | None,
             alarm_s: float | None, t: float) -> np.ndarray:
    if direction is None or alarm_s is None:
        return start.copy()
    distance = min(.6, max(0., t - alarm_s - .1) * .25)
    return start + distance * direction


def hazard_at(start: np.ndarray, direction: np.ndarray, t: float) -> np.ndarray:
    progress = min(1., max(0., (t - .2) / 1.6))
    return start + direction * (1.0 - .95 * progress)


def static_gap(scene: TexturedCaptureScene, xy: np.ndarray) -> float:
    yaws = np.deg2rad(ANGLES)
    rays = np.column_stack((np.full(72, xy[0]),
        np.full(72, scene.floor_y + .1), np.full(72, xy[1]),
        np.sin(yaws), np.zeros(72), np.cos(yaws))).astype(np.float32)
    return float(scene.static.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().min()
                 - BODY_RADIUS)


def audit_row(case: dict, row: dict, scene, hazard_radius: float,
              prop_radii: dict[str, float]) -> dict:
    path = ROOT / row["trace"]
    trace = json.loads(path.read_text())
    if any(trace[k] != row[k] for k in ("case", "arm", "parked")):
        raise ValueError("Report and trace disagree")
    search_path = ROOT / trace["search_trace"]
    if not aligned(search_path.parent):
        raise ValueError("Search RGB/depth/video is not aligned")
    search = json.loads(search_path.read_text())
    if trace["image_candidate_before_monitor"] != (
            search["action"] == "candidate_to_owner_for_confirmation"):
        raise ValueError("Image worker action changed")
    monitor_video = ROOT / trace["monitor_video"]
    frames = trace["monitor_video_frames"]
    if video_count(monitor_video) != len(frames) or len(frames) < 2:
        raise ValueError("Monitor video and source frames disagree")
    baseline_depth = np.asarray(Image.open(path.parent / "baseline_depth.png"))
    packet_by_tick = {x["tick"]: x for x in trace["monitor_packets"]}
    for frame in frames:
        if not (ROOT / frame["rgb"]).is_file() or not (ROOT / frame["depth"]).is_file():
            raise ValueError("Monitor RGB-D frame missing")
        if frame["packet_index"] not in packet_by_tick:
            raise ValueError("Monitor frame lacks source packet")
        if trace["arm"] == "head_depth" and trace["alarm_delivery_s"] is None:
            depth = np.asarray(Image.open(ROOT / frame["depth"]))
            valid = (baseline_depth > 0) & (depth > 0)
            changed = valid & ((baseline_depth.astype(float) - depth) > 80.) & \
                (depth < 900)
            if int(changed.sum()) >= 20:
                raise ValueError("Head-depth arm omitted a visible alarm")
    packets = trace["monitor_packets"]
    times = np.array([p["capture_wall_s"] for p in packets], float)
    if len(packets) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Monitor packets missing or nonmonotonic")
    if len(trace["range_angles_deg"]) != 72 or len(trace["baseline_low_range_m"]) != 72:
        raise ValueError("Unexpected range geometry")
    baseline = np.array([np.inf if x is None else x
                         for x in trace["baseline_low_range_m"]], float)
    expected_first = None
    for packet in packets:
        low = np.array([np.inf if x is None else x for x in packet["low_range_m"]], float)
        with np.errstate(invalid="ignore"):
            early = (baseline - low > .08) & (low < .9)
            late = (baseline - low > .12) & (low < .6)
        front = early & (np.abs(ANGLES) <= 70)
        if (bool(early.any()) != packet["range360_early_alarm"] or
                bool(late.any()) != packet["range360_late_alarm"] or
                bool(front.any()) != packet["front_range_early_alarm"]):
            raise ValueError("Range alarm does not follow saved rays")
        predicate = {"range360_early": bool(early.any()),
                     "range360_late": bool(late.any()),
                     "front_range_early": bool(front.any()),
                     "head_depth": packet["head_depth_alarm"],
                     "capture_only": False}[trace["arm"]]
        if predicate and expected_first is None:
            expected_first = packet
    alarm = trace["alarm_delivery_s"]
    if (expected_first is None) != (alarm is None):
        raise ValueError("First alarm absent or invented")
    if expected_first is not None:
        if trace["arm"] == "head_depth":
            if not (expected_first["range_delivery_wall_s"] <= alarm <=
                    trace["worker_wall_s"]):
                raise ValueError("Head alarm delivered outside image worker")
        elif abs(alarm - expected_first["range_delivery_wall_s"]) > .001:
            raise ValueError("Range response did not use delivery time")
        if abs(trace["alarm_capture_s"] - expected_first["capture_wall_s"]) > .001:
            raise ValueError("Alarm capture time changed")
    owner = json.loads((ROOT / case["owner_trace"]).read_text())["trace"][-1]
    start = np.asarray(owner["evaluator_actual_xz_m"], float)
    yaw = float(owner["evaluator_actual_heading_deg"])
    bearing = math.radians(yaw + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    response = (None if trace["response_direction_xz"] is None else
                np.asarray(trace["response_direction_xz"], float))
    if (response is None) != (alarm is None):
        raise ValueError("Action command without alarm")
    if response is not None:
        angle = math.radians(yaw + trace["alarm_relative_angle_deg"] + 180.)
        if np.linalg.norm(response - [math.sin(angle), math.cos(angle)]) > 1e-5:
            raise ValueError("Response did not follow measured bearing")
    for packet in packets:
        t = packet["capture_wall_s"]
        expected_actor = actor_at(start, response, alarm, t)
        if np.linalg.norm(expected_actor - packet["actor_xz_m"]) > 1e-4:
            raise ValueError("Actor packet contradicts response command")
        if not trace["parked"]:
            expected_hazard = hazard_at(start, incoming, t)
            if np.linalg.norm(expected_hazard - packet["hazard_xz_m"]) > 1e-4:
                raise ValueError("Hazard packet contradicts frozen trajectory")
    scores = trace["evaluator_20ms"]
    if len(scores) < 2 or any(abs((b["t_s"] - a["t_s"]) - .02) > .001
                              for a, b in zip(scores, scores[1:])):
        raise ValueError("Evaluator time grid changed")
    placement = json.loads((ROOT / case["source_screen"]).read_text())
    placement = next(x for x in placement["scenes"] if x["scene_key"] == case["scene_key"])
    gaps = []
    static_gaps = []
    prop_gaps = []
    floor_all = True
    for item in scores:
        t = item["t_s"]
        actor = actor_at(start, response, alarm, t)
        if np.linalg.norm(actor - item["actor_xz_m"]) > 1e-4:
            raise ValueError("Evaluator actor path changed")
        if not trace["parked"]:
            hazard = hazard_at(start, incoming, t)
            gap = float(np.linalg.norm(actor - hazard) - BODY_RADIUS - hazard_radius)
            if abs(gap - item["hazard_gap_m"]) > 1e-5:
                raise ValueError("Evaluator hazard gap changed")
            gaps.append(gap)
        for name, prop in (("target", case["target"]),
                           ("distractor", case["distractor"])):
            pos = np.asarray(placement[f"find_{name}_xz"])
            prop_gaps.append(float(np.linalg.norm(actor - pos) - BODY_RADIUS -
                                   prop_radii[prop]))
        static_gaps.append(static_gap(scene, actor))
        floor_all &= scene.floor_supported(*actor)
    if (not trace["parked"] and abs(min(gaps) - trace["min_hazard_gap_m"]) > 1e-5) or \
            abs(min(static_gaps) - trace["min_static_gap_m"]) > 1e-5 or \
            abs(min(prop_gaps) - trace["min_prop_gap_m"]) > 1e-5 or \
            floor_all != trace["floor_supported_throughout"]:
        raise ValueError("Independent geometry score mismatch")
    candidate = trace["image_candidate_before_monitor"]
    expected_valid = bool(candidate and (alarm is None or
                                         alarm >= trace["worker_wall_s"]))
    if trace["image_valid_after_monitor"] != expected_valid:
        raise ValueError("Stale image validity gate changed")
    if trace["owner_action_after_monitor"] != (
            "ask_for_new_safe_view" if alarm is not None else search["action"]):
        raise ValueError("Owner-facing continuation changed")
    return {"case": row["case"], "arm": row["arm"], "parked": row["parked"],
        "alarm_s": alarm, "min_hazard_gap_m": trace["min_hazard_gap_m"],
        "min_static_gap_m": trace["min_static_gap_m"],
        "min_prop_gap_m": trace["min_prop_gap_m"],
        "floor_supported": floor_all,
        "image_candidate_before_monitor": candidate,
        "image_valid_after_monitor": trace["image_valid_after_monitor"],
        "worker_wall_s": trace["worker_wall_s"],
        "monitor_max_packet_interval_s": float(np.diff(times).max()),
        "monitor_video": trace["monitor_video"],
        "search_video": trace["search_video"], "trace": row["trace"]}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for name, item in frozen["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    report = json.loads(REPORT.read_text())
    expected = [(case["id"], arm, False) for case in frozen["cases"]
                for arm in frozen["arms"]] + [
                (case["id"], "range360_early", True) for case in frozen["cases"]]
    if sorted((x["case"], x["arm"], x["parked"]) for x in report["rows"]) != \
            sorted(expected):
        raise ValueError("Frozen paired cohort incomplete")
    audited = []
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    for case in frozen["cases"]:
        for key, digest in (("teaching_summary", "summary_sha256"),
                            ("source_capture", "source_capture_sha256"),
                            ("source_screen", "source_screen_sha256"),
                            ("source_mesh", "source_mesh_sha256"),
                            ("owner_trace", "owner_trace_sha256")):
            if sha(ROOT / case[key]) != case[digest]:
                raise ValueError(f"Frozen source changed: {case['id']} {key}")
        source = json.loads((ROOT / case["source_capture"]).read_text())
        scene = TexturedCaptureScene(ROOT / case["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.),
            source_z_up=source.get("source_z_up", False))
        radii = {name: TexturedFloorObject(props / f"{name}.obj",
                   scale=.012).footprint_radius_m for name in
                 (case["target"], case["distractor"], case["hazard_object"])}
        for row in report["rows"]:
            if row["case"] == case["id"]:
                audited.append(audit_row(case, row, scene,
                                         radii[case["hazard_object"]], radii))
    by_arm = {}
    for arm in frozen["arms"]:
        xs = [x for x in audited if x["arm"] == arm and not x["parked"]]
        by_arm[arm] = {"runs": len(xs),
            "alarms": sum(x["alarm_s"] is not None for x in xs),
            "gap_breaches_below_5cm": sum(x["min_hazard_gap_m"] < .05 for x in xs),
            "static_or_prop_contact": sum(x["min_static_gap_m"] < 0 or
                                          x["min_prop_gap_m"] < 0 for x in xs),
            "floor_supported": sum(x["floor_supported"] for x in xs),
            "stale_candidates_invalidated": sum(
                x["image_candidate_before_monitor"] and
                not x["image_valid_after_monitor"] for x in xs)}
    summary = {"rooms": len(frozen["cases"]), "runs": len(audited),
        "parked_false_alarms": sum(x["alarm_s"] is not None for x in audited
                                   if x["parked"]),
        "max_monitor_packet_interval_s": max(x[
            "monitor_max_packet_interval_s"] for x in audited),
        "by_arm": by_arm}
    output = {"schema": "ape.cross-room-inference-response-audit.v1",
              "precommit": relative(PRECOMMIT), "summary": summary,
              "rows": audited, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
