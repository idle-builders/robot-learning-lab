"""Replay observed meeting-pose proposals, bounded exploration and owner evidence."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_post_confirmation_approach_v2 as approach
import run_cooperative_return_8 as route_policy
import run_cooperative_represent_probe as presentation
import run_returned_owner_presentation_8 as owner_review
import audit_joined_backup_recovery_16 as monitor_audit
import audit_shared_world_return as video_audit
import run_meeting_pose_replan_3 as trial


OUT = ART / "meeting_pose_replan_3_audit.json"


def audit_one(row: dict, case: dict, required: float, *,
              proposal_provider=trial.propose,
              max_actions: int = trial.MAX_ACTIONS,
              max_candidates: int = trial.MAX_CANDIDATES,
              max_per_candidate: int = trial.MAX_PER_CANDIDATE) -> dict:
    x = json.loads((ROOT / row["trace"]).read_text())
    source = json.loads((ROOT / case["source"]).read_text())
    if x["source_station_trace"] != case["source"] or x[
            "proposals"] != proposal_provider(source, required):
        raise ValueError("Actor candidate list changed after source scan")
    key = row["scene_key"]
    cfg = SCENES[key]
    placement_doc = json.loads((ROOT / case["screen"]).read_text())
    placement = next(v for v in placement_doc["scenes"] if v[
        "scene_key"] == key)
    scene = approach.base.make_scene(case, placement)
    owner_xz = tuple(cfg["owner_xz"])
    wrapped = presentation.PersonVisibleScene(scene, owner_xz)
    actual = np.asarray(source["final_actual_xz_m"], float)
    actor = np.asarray(source["final_actor_xz_m"], float)
    actual_yaw = float(source["final_actual_yaw_deg"])
    actor_yaw = float(source["final_actor_yaw_deg"])
    desired_yaw = actor_yaw
    actions = x["actions"]
    if len(actions) > max_actions or len(x["attempts"]) > max_candidates:
        raise ValueError("Meeting-pose search exceeded frozen budget")
    forward = turns = rejected = 0
    offset = 0
    for candidate_index, attempt in enumerate(x["attempts"]):
        if attempt["candidate"] != x["proposals"][candidate_index] or (
                attempt["first_frame"] != offset):
            raise ValueError("Candidate attempt order changed")
        goal = np.asarray(attempt["candidate"]["actor_goal_xz_m"], float)
        used = 0
        while offset < len(actions) and actions[offset]["candidate_index"] == (
                candidate_index):
            action = actions[offset]
            used += 1
            if used > max_per_candidate or action["frame"] != offset or (
                    not np.allclose(action["actual_xz_m"], actual) or
                    not np.allclose(action["actor_xz_m"], actor) or
                    abs(action["actual_yaw_deg"] - actual_yaw) > 1e-6 or
                    abs(action["actor_yaw_deg"] - actor_yaw) > 1e-6):
                raise ValueError("Bounded candidate route cannot be replayed")
            rgb, depth = ROOT / action["rgb"], ROOT / action["depth"]
            if not rgb.exists() or not depth.exists() or Image.open(rgb).size != (
                    Image.open(depth).size):
                raise ValueError("Missing/misaligned candidate RGB-D frame")
            eye = np.array([actual[0], scene.floor_y + .42, actual[1]])
            rays = scene.range_fan(eye, actual_yaw, owner_xz=owner_xz,
                angles_deg=np.arange(-90., 91., 5.))
            kind, amount, reason = route_policy.decide(actor, actor_yaw,
                goal, desired_yaw, rays)
            if (kind, reason) != (action["kind"], action["reason"]) or abs(
                    amount - action["amount"]) > 1e-6:
                raise ValueError("Candidate route decision differs from fresh range")
            if kind == "turn":
                if abs(amount) > route_policy.TURN_DEG + 1e-8:
                    raise ValueError("Unbounded candidate turn")
                actual_yaw += amount; actor_yaw += amount; turns += 1
            elif kind == "forward":
                if amount <= 0 or amount > route_policy.STEP_M + 1e-8:
                    raise ValueError("Unbounded candidate step")
                heading = math.radians(actual_yaw)
                step = amount * np.array([math.sin(heading),math.cos(heading)])
                candidate = actual + step
                at_eye = np.array([candidate[0], scene.floor_y + .42,
                                   candidate[1]])
                fresh = scene.range_fan(at_eye, actual_yaw,
                    owner_xz=owner_xz, angles_deg=np.arange(-90.,91.,5.))
                finite = fresh[np.isfinite(fresh)]
                accepted = bool(scene.floor_supported(*candidate) and
                    len(finite) and float(finite.min()) >=
                    route_policy.BODY_RADIUS_M)
                if accepted != action["evaluator_motion_accepted"]:
                    raise ValueError("Candidate step evaluator outcome differs")
                if accepted:
                    actual = candidate
                    actor += amount * np.array([
                        math.sin(math.radians(actor_yaw)),
                        math.cos(math.radians(actor_yaw))])
                    forward += 1
                else:
                    rejected += 1
            offset += 1
        if attempt["result"] == "returned_to_known_owner_station":
            scan, admitted = presentation.scan_gate(wrapped, actual,
                                                     actual_yaw, required)
            if admitted != attempt["fresh_admitted_bearings_deg"] or not (
                    np.allclose(scan[0], [np.inf if v is None else v for v in
                        attempt["fresh_scan_low_m"]], equal_nan=True) and
                    np.allclose(scan[1], [np.inf if v is None else v for v in
                        attempt["fresh_scan_mid_m"]], equal_nan=True)):
                raise ValueError("Moved wait scan/verdict differs")
        elif attempt["fresh_admitted_bearings_deg"]:
            raise ValueError("Unsupported candidate gained owner permission")
    if offset != len(actions) or not np.allclose(actual, x[
            "final_actual_xz_m"]) or not np.allclose(actor, x[
            "final_actor_xz_m"]):
        raise ValueError("Candidate path endpoint differs")
    if bool(x["selected_candidate"]) != bool(x[
            "admitted_entry_bearings_deg"]):
        raise ValueError("Selected meeting pose lacks a fresh corridor")
    if x["selected_candidate"] is not None:
        if x["selected_candidate"] != x["attempts"][-1]["candidate"] or x[
                "route_result"] != "repositioned_supported_meeting_pose":
            raise ValueError("Selected pose did not follow final attempt")
        episode = json.loads((ROOT / x["fresh_search_trace"]).read_text())
        if episode["action"] != x["fresh_search_action"] or (
                video_audit.decoded_frames(ROOT / x["fresh_search_video"]) !=
                len(episode["video_frames"])):
            raise ValueError("New owner search/video disagrees")
        confirmed = owner_review.truthful_confirmation(ROOT /
            x["fresh_search_trace"])
        cadence = max(monitor_audit.source_packets(x[
            "monitor_primary_packets"],x["monitor_primary_baseline_m"]),
            monitor_audit.source_packets(x["monitor_backup_packets"],
                x["monitor_backup_baseline_m"]))
        alarmed = any(p["changed_return_alarm"] for stream in (
            x["monitor_primary_packets"], x["monitor_backup_packets"])
            for p in stream)
        expected = ("changed_scene_hold_unknown" if alarmed else
            "owner_confirmed_represented_item" if confirmed else
            "fresh_visual_identity_unknown")
        if x["presentation_outcome"] != expected or not x[
                "worker_world_captures"]:
            raise ValueError("Owner claim lacks fresh evidence")
        expected_frames = len(actions) + len(episode["video_frames"])
    else:
        cadence = None
        expected_frames = len(actions) if actions else None
        if x["presentation_outcome"] != (
                "no_supported_meeting_pose_hold_unknown"):
            raise ValueError("No supported pose yielded an owner claim")
    if actions and video_audit.decoded_frames(ROOT / x["video"]) != (
            expected_frames):
        raise ValueError("Meeting-pose video/frame count differs")
    if not actions and x["video"] != source["video"]:
        raise ValueError("No-proposal case lost source video")
    return {"scene_key": key, "outcome": x["presentation_outcome"],
        "proposals": len(x["proposals"]), "attempts": len(x["attempts"]),
        "accepted_forward_steps": forward, "turns": turns,
        "rejected_steps": rejected,
        "selected_candidate": x["selected_candidate"],
        "max_packet_interval_s": cadence, "video": x["video"]}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    frozen = json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"]) != 3 or {r["scene_key"] for r in report[
            "rows"]} != {c["scene_key"] for c in frozen["cases"]}:
        raise ValueError("Incomplete meeting-pose cohort")
    if any(sha(ROOT / c["source"]) != frozen["source_station_sha256"][
            c["scene_key"]] for c in frozen["cases"]):
        raise ValueError("Source station evidence changed")
    cases = {c["scene_key"]: c for c in frozen["cases"]}
    rows = [audit_one(r, cases[r["scene_key"]], frozen[
        "required_wait_clearance_m"]) for r in report["rows"]]
    result = {"schema": "ape.meeting-pose-replan-3-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "fresh_owner_confirmed": sum(r["outcome"] ==
            "owner_confirmed_represented_item" for r in rows),
        "no_observed_candidate": sum(r["proposals"] == 0 for r in rows),
        "unresolved_after_bounded_search": sum(r["proposals"] > 0 and r[
            "outcome"] != "owner_confirmed_represented_item" for r in rows),
        "accepted_forward_steps": sum(r["accepted_forward_steps"] for r in rows),
        "rejected_steps": sum(r["rejected_steps"] for r in rows),
        "rows": rows, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__ == "__main__":
    main()
