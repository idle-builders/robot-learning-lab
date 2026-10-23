"""Audit bounded owner return and freshly monitored presentation after interruption."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_cooperative_return_8 as route_policy
import run_cooperative_represent_probe as presentation
import run_returned_owner_presentation_8 as owner_review
import run_joined_live_hazard_16 as original
import run_joined_backup_recovery_16 as recovery
import audit_joined_backup_recovery_16 as monitor_audit
import audit_shared_world_return as video_audit
import run_post_interrupt_owner_return_3 as trial


OUT = ART / "post_interrupt_owner_return_3_audit.json"


def audit_one(row: dict, case: dict, required: float) -> dict:
    x = json.loads((ROOT / row["trace"]).read_text())
    key = row["scene_key"]
    source = json.loads((recovery.OUT / key / "trace.json").read_text())
    interrupted = json.loads((ROOT / source[
        "source_interruption_trace"]).read_text())
    old = original.actor_state(case)
    delta = np.asarray(source["response_endpoint_xz_m"], float) - np.asarray(
        interrupted["start_xz_m"], float)
    if not np.allclose(x["initial"]["actual_xz_m"], source[
            "response_endpoint_xz_m"]) or not np.allclose(x["initial"][
            "actor_xz_m"], old["actor"] + delta):
        raise ValueError("Return did not start from backed-up response belief")
    cfg = SCENES[key]
    owner = json.loads((ROOT / x["source_owner_station"]).read_text())[
        "trace"][-1]
    goal = np.asarray(cfg["start_xz"], float) + np.asarray(owner[
        "actor_pose_xy_m"], float)
    if not np.allclose(goal, x["actor_goal_xz_m"]):
        raise ValueError("Return goal was not the remembered actor station")
    scene, _, _, _ = original.scene_and_props(case)
    owner_xz = tuple(cfg["owner_xz"])
    actual = np.asarray(x["initial"]["actual_xz_m"], float)
    actor = np.asarray(x["initial"]["actor_xz_m"], float)
    actual_yaw = x["initial"]["actual_yaw_deg"]
    actor_yaw = x["initial"]["actor_yaw_deg"]
    actions = x["actions"]
    if not actions or video_audit.decoded_frames(ROOT / x[
            "video"] if x["fresh_search_video"] is None else ROOT /
            "assets/real_indoor_scenes/post_interrupt_owner_return_3_20261015" /
            key / "route/return.mp4") != len(actions):
        raise ValueError("Return video/action count differs")
    forward = turns = 0
    for index, action in enumerate(actions):
        if action["frame"] != index or not np.allclose(action["actor_xz_m"],
                actor) or not np.allclose(action["actual_xz_m"], actual) or (
                abs(action["actor_yaw_deg"] - actor_yaw) > 1e-6 or
                abs(action["actual_yaw_deg"] - actual_yaw) > 1e-6):
            raise ValueError("Route state/action trace does not replay")
        rgb, depth = ROOT / action["rgb"], ROOT / action["depth"]
        if not rgb.exists() or not depth.exists() or Image.open(rgb).size != (
                Image.open(depth).size):
            raise ValueError("Missing or misaligned route RGB-D")
        eye = np.array([actual[0], scene.floor_y + .42, actual[1]])
        rays = scene.range_fan(eye, actual_yaw, owner_xz=owner_xz,
            angles_deg=np.arange(-90., 91., 5.))
        kind, amount, reason = route_policy.decide(actor, actor_yaw, goal,
            float(owner["actor_heading_deg"]), rays)
        if (kind, reason) != (action["kind"], action["reason"]) or abs(
                amount - action["amount"]) > 1e-6:
            raise ValueError("Actor route decision differs from fresh range")
        if kind == "turn":
            if abs(amount) > route_policy.TURN_DEG + 1e-8:
                raise ValueError("Unbounded turn")
            actual_yaw += amount; actor_yaw += amount; turns += 1
        elif kind == "forward":
            if amount <= 0 or amount > route_policy.STEP_M + 1e-8:
                raise ValueError("Unbounded forward command")
            heading = math.radians(actual_yaw)
            move = amount * np.array([math.sin(heading), math.cos(heading)])
            candidate = actual + move
            at_eye = np.array([candidate[0], scene.floor_y + .42,
                               candidate[1]])
            fresh = scene.range_fan(at_eye, actual_yaw,
                owner_xz=owner_xz, angles_deg=np.arange(-90., 91., 5.))
            finite = fresh[np.isfinite(fresh)]
            accepted = bool(scene.floor_supported(*candidate) and len(finite)
                and float(finite.min()) >= route_policy.BODY_RADIUS_M)
            if action["evaluator_motion_accepted"] != accepted:
                raise ValueError("Accepted route step lacks floor/range support")
            if accepted:
                actual = candidate
                actor += amount * np.array([
                    math.sin(math.radians(actor_yaw)),
                    math.cos(math.radians(actor_yaw))])
                forward += 1
        elif kind == "hold" and index != len(actions) - 1:
            raise ValueError("Hold was followed by more route actions")
    if not np.allclose(actual, x["final_actual_xz_m"]) or not np.allclose(
            actor, x["final_actor_xz_m"]) or x["route_result"] != actions[-1][
            "reason"]:
        raise ValueError("Route endpoint/result differs")
    station_gap = float(np.linalg.norm(actual - np.asarray(x[
        "evaluator_station_xz_m"], float)))
    if x["route_result"] == "returned_to_known_owner_station":
        actor_gap = float(np.linalg.norm(actor - goal))
        if actor_gap > (route_policy.ARRIVAL_M +
                route_policy.STOP_BAND_M + 1e-6):
            raise ValueError("Actor did not reach remembered station")
        wrapped = (scene if case["branch"] == "cooperative" else
            presentation.PersonVisibleScene(scene, owner_xz))
        scan, admitted = presentation.scan_gate(wrapped, actual,
            actual_yaw, required)
        if admitted != x["admitted_entry_bearings_deg"] or not np.allclose(
                [[np.inf if v is None else v for v in x["scan_low_m"]],
                 [np.inf if v is None else v for v in x["scan_mid_m"]]],
                scan, equal_nan=True):
            raise ValueError("Owner-station wait scan cannot be replayed")
    lateral = np.asarray(cfg["owner_lateral"], float)
    lateral /= np.linalg.norm(lateral)
    expected_target = (np.asarray(owner_xz) +
        presentation.OWNER_PLACE_LATERAL_M * lateral)
    if not np.allclose(expected_target, x["scripted_owner_place_xz_m"]):
        raise ValueError("Owner scripted placement differs")
    cadence = None
    if x["fresh_search_trace"] is not None:
        if x["route_result"] != "returned_to_known_owner_station" or not x[
                "admitted_entry_bearings_deg"] or not x[
                "placement_floor_supported"]:
            raise ValueError("Presentation started without route/station gate")
        episode = json.loads((ROOT / x["fresh_search_trace"]).read_text())
        if episode["action"] != x["fresh_search_action"] or (
                video_audit.decoded_frames(ROOT / x["fresh_search_video"]) !=
                len(episode["video_frames"])):
            raise ValueError("Fresh owner episode/video differs")
        cadence = max(monitor_audit.source_packets(x[
            "monitor_primary_packets"], x["monitor_primary_baseline_m"]),
            monitor_audit.source_packets(x["monitor_backup_packets"],
                x["monitor_backup_baseline_m"]))
        if not x["worker_world_captures"] or not any(c["channel"] == "rgbd"
                for c in x["worker_world_captures"]):
            raise ValueError("Owner worker lacks new RGB-D capture")
        alarmed = any(p["changed_return_alarm"] for stream in (
            x["monitor_primary_packets"], x["monitor_backup_packets"])
            for p in stream)
        confirmed = owner_review.truthful_confirmation(ROOT /
            x["fresh_search_trace"])
        expected_outcome = ("changed_scene_hold_unknown" if alarmed else
            "owner_confirmed_represented_item" if confirmed else
            "fresh_visual_identity_unknown")
        if x["presentation_outcome"] != expected_outcome:
            raise ValueError("Owner claim was not freshly supported")
        expected_frames = len(actions) + len(episode["video_frames"])
        if video_audit.decoded_frames(ROOT / x["video"]) != expected_frames:
            raise ValueError("Combined route/presentation video differs")
    return {"scene_key": key, "branch": row["branch"],
        "route_result": x["route_result"],
        "presentation_outcome": x["presentation_outcome"],
        "forward_steps": forward, "turns": turns,
        "actual_station_gap_m": station_gap,
        "wait_admitted": bool(x["admitted_entry_bearings_deg"]),
        "max_packet_interval_s": cadence, "video": x["video"]}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    frozen = json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"]) != 3 or frozen[
            "source_recovery_audit_sha256"] != sha(ART /
            "joined_backup_recovery_16_audit.json"):
        raise ValueError("Incomplete or changed three-case return")
    cases = {c["scene_key"]: c for c in frozen["cases"]}
    if {r["scene_key"] for r in report["rows"]} != set(cases):
        raise ValueError("Return cases missing/repeated")
    required = json.loads(presentation.geometry.OUT.read_text())[
        "required_nearest_two_plane_m"]
    rows = [audit_one(r, cases[r["scene_key"]], required) for r in report[
        "rows"]]
    result = {"schema": "ape.post-interrupt-owner-return-3-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "returned_to_station": sum(r["route_result"] ==
            "returned_to_known_owner_station" for r in rows),
        "station_wait_admitted": sum(r["wait_admitted"] for r in rows),
        "fresh_owner_confirmed": sum(r["presentation_outcome"] ==
            "owner_confirmed_represented_item" for r in rows),
        "accepted_forward_steps": sum(r["forward_steps"] for r in rows),
        "max_actual_station_gap_m": max(r["actual_station_gap_m"]
                                        for r in rows),
        "max_packet_interval_s": max(r["max_packet_interval_s"] for r in rows
            if r["max_packet_interval_s"] is not None),
        "rows": rows, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
