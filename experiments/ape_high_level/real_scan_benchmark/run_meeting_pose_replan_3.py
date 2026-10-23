"""Search nearby observed owner meeting poses when a remembered station has no wait."""

from __future__ import annotations

import argparse
import json
import math

import imageio.v2 as imageio
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES, camera
from check_scanned_wait_corridor_20 import certified
import screen_20_final_wait_lanes as wait_geometry
import run_post_confirmation_approach_v2 as approach
import run_cooperative_return_8 as route_policy
import run_cooperative_represent_probe as presentation
import run_post_interrupt_owner_return_3 as continued


OUT = ROOT / "assets/real_indoor_scenes/meeting_pose_replan_3_20261015"
PRECOMMIT = ART / "meeting_pose_replan_3_precommit.json"
REPORT = ART / "meeting_pose_replan_3_report.json"
RADII_M = (.45, .35, .25, .15)
WORLD_BEARINGS_DEG = tuple(range(0, 360, 45))
MAX_CANDIDATES = 8
MAX_ACTIONS = 96
MAX_PER_CANDIDATE = 24


def cases() -> list[dict]:
    full = {c["scene_key"]: c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())[
            "cases"]}
    old = json.loads((ART / "cooperative_return_8_v4_report.json").read_text())
    rows = [{"scene_key": r["scene_key"], "source": r["trace"]}
            for r in old["rows"] if not r["admitted_station"]]
    recent = json.loads((ART / "post_interrupt_owner_return_3_report.json").read_text())
    rows += [{"scene_key": r["scene_key"], "source": r["trace"]}
             for r in recent["rows"] if r["presentation_outcome"] ==
             "owner_station_unsupported"]
    if len(rows) != 3 or len({r["scene_key"] for r in rows}) != 3:
        raise ValueError("Expected three distinct unsupported owner stations")
    return [{**full[r["scene_key"]], "branch": "cooperative", **r}
            for r in rows]


def propose(source: dict, required: float, max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    origin = np.asarray(source["final_actor_xz_m"], float)
    yaw = float(source["final_actor_yaw_deg"])
    rays = np.asarray([[np.inf if v is None else v for v in source[
        "scan_low_m"]], [np.inf if v is None else v for v in source[
        "scan_mid_m"]]], float)
    proposals = []
    for radius in RADII_M:
        for bearing in WORLD_BEARINGS_DEG:
            a = math.radians(bearing)
            point = origin + radius * np.array([math.sin(a), math.cos(a)])
            path = [origin + fraction * (point-origin) for fraction in
                    np.linspace(.1, 1., 6)]
            if not certified(path, rays, origin, yaw, required):
                continue
            lanes = []
            for offset in wait_geometry.OFFSETS:
                b = math.radians(yaw + offset)
                direction = np.array([math.sin(b), math.cos(b)])
                samples = [point + d * direction for d in
                           wait_geometry.INBOUND_M]
                samples += [point - d * direction for d in
                            wait_geometry.ESCAPE_M]
                if certified(samples, rays, origin, yaw, required):
                    lanes.append(offset)
            if lanes:
                proposals.append({"radius_m": radius,
                    "world_bearing_deg": bearing,
                    "actor_goal_xz_m": point.tolist(),
                    "predicted_admitted_bearings_deg": lanes})
    return proposals[:max_candidates]


def run_one(case: dict, required: float, *, proposal_provider=propose,
            out=OUT, max_actions: int = MAX_ACTIONS,
            max_per_candidate: int = MAX_PER_CANDIDATE,
            schema: str = "ape.meeting-pose-replan-3.v1") -> dict:
    key = case["scene_key"]
    source = json.loads((ROOT / case["source"]).read_text())
    proposals = proposal_provider(source, required)
    folder = out / key
    folder.mkdir(parents=True, exist_ok=True)
    cfg = SCENES[key]
    placement_doc = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in placement_doc["scenes"] if x[
        "scene_key"] == key)
    scene = approach.base.make_scene(case, placement)
    owner_xz = tuple(cfg["owner_xz"])
    wrapped = presentation.PersonVisibleScene(scene, owner_xz)
    actual = np.asarray(source["final_actual_xz_m"], float)
    actor = np.asarray(source["final_actor_xz_m"], float)
    actual_yaw = float(source["final_actual_yaw_deg"])
    actor_yaw = float(source["final_actor_yaw_deg"])
    desired_yaw = actor_yaw
    actions = []
    attempts = []
    admitted = []
    selected = None
    if proposals:
        with imageio.get_writer(str(folder / "reposition.mp4"), fps=5,
                codec="libx264", quality=7, macro_block_size=8) as writer:
            for proposal in proposals:
                attempt = {"candidate": proposal,
                    "first_frame": len(actions), "result": None,
                    "fresh_admitted_bearings_deg": []}
                goal = np.asarray(proposal["actor_goal_xz_m"], float)
                for _ in range(max_per_candidate):
                    if len(actions) >= max_actions:
                        attempt["result"] = "action_cap"
                        break
                    frame = len(actions)
                    rgb, depth, _ids, ranges = camera(scene, actual,
                        actual_yaw, owner_xz)
                    route_policy.approach.base.save_frame(folder, writer,
                        frame, rgb, depth)
                    kind, amount, reason = route_policy.decide(actor,
                        actor_yaw, goal, desired_yaw, ranges)
                    record = {"frame": frame, "candidate_index": len(attempts),
                        "kind": kind, "amount": amount, "reason": reason,
                        "rgb": relative(folder / f"rgb_{frame:03d}.png"),
                        "depth": relative(folder / f"depth_{frame:03d}.png"),
                        "actor_xz_m": actor.tolist(),
                        "actual_xz_m": actual.tolist(),
                        "actor_yaw_deg": actor_yaw,
                        "actual_yaw_deg": actual_yaw}
                    if kind == "hold":
                        attempt["result"] = reason
                    elif kind == "turn":
                        actor_yaw += amount
                        actual_yaw += amount
                    else:
                        heading = math.radians(actual_yaw)
                        delta = amount * np.array([math.sin(heading),
                                                   math.cos(heading)])
                        candidate = actual + delta
                        eye = np.array([candidate[0], scene.floor_y + .42,
                                        candidate[1]])
                        fan = scene.range_fan(eye, actual_yaw,
                            owner_xz=owner_xz,
                            angles_deg=np.arange(-90., 91., 5.))
                        finite = fan[np.isfinite(fan)]
                        accepted = bool(scene.floor_supported(*candidate) and
                            len(finite) and float(finite.min()) >=
                            route_policy.BODY_RADIUS_M)
                        record["evaluator_motion_accepted"] = accepted
                        if accepted:
                            actual = candidate
                            actor += amount * np.array([
                                math.sin(math.radians(actor_yaw)),
                                math.cos(math.radians(actor_yaw))])
                        else:
                            attempt["result"] = "simulator_rejected_motion"
                    actions.append(record)
                    if attempt["result"] is not None:
                        break
                if attempt["result"] is None:
                    attempt["result"] = "candidate_action_cap"
                if attempt["result"] == "returned_to_known_owner_station":
                    scan, admitted = presentation.scan_gate(wrapped, actual,
                        actual_yaw, required)
                    attempt["fresh_admitted_bearings_deg"] = admitted
                    attempt["fresh_scan_low_m"] = [float(v) if np.isfinite(v)
                        else None for v in scan[0]]
                    attempt["fresh_scan_mid_m"] = [float(v) if np.isfinite(v)
                        else None for v in scan[1]]
                    if admitted:
                        selected = proposal
                        attempts.append(attempt)
                        break
                attempts.append(attempt)
                if len(actions) >= max_actions:
                    break
    route = {"route_result": ("repositioned_supported_meeting_pose"
        if selected else "no_observed_candidate" if not proposals else
        "reposition_unresolved"),
        "final_actual_xz_m": actual.tolist(),
        "final_actor_xz_m": actor.tolist(),
        "final_actual_yaw_deg": actual_yaw,
        "final_actor_yaw_deg": actor_yaw,
        "admitted_entry_bearings_deg": admitted}
    presentation_result = (continued.present_at_station(case, route,
        folder / "owner_presentation") if selected else {
            "presentation_outcome": "no_supported_meeting_pose_hold_unknown",
            "fresh_search_video": None})
    route_video = (relative(folder / "reposition.mp4") if actions else
                   source["video"])
    full_video = continued.combined_video(route_video,
        presentation_result.get("fresh_search_video"),
        folder / "reposition_and_owner_presentation.mp4")
    trace = {"schema": schema,
        "scene_key": key, "source_station_trace": case["source"],
        "initial_actual_xz_m": source["final_actual_xz_m"],
        "initial_actor_xz_m": source["final_actor_xz_m"],
        "initial_actual_yaw_deg": source["final_actual_yaw_deg"],
        "initial_actor_yaw_deg": source["final_actor_yaw_deg"],
        "proposals": proposals, "attempts": attempts,
        "selected_candidate": selected, "actions": actions,
        **route, **presentation_result,
        "video": full_video,
        "limits": "Actor proposals come only from the saved person-aware two-height scan, its believed station pose and a fixed candidate grid. Fresh RGB-D/range guards bound every move and a new local scan admits presentation. Evaluator floor/contact scores candidate execution, not selection. Static owner/item/room, ideal 17cm disk and commanded yaw; no moving entrant or physical mounted-source/stop qualification."}
    path = folder / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": key, "trace": relative(path),
        "outcome": trace["presentation_outcome"],
        "selected_candidate": selected,
        "video": full_video}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    selected = cases()
    required = json.loads(presentation.geometry.OUT.read_text())[
        "required_nearest_two_plane_m"]
    frozen = {"schema": "ape.meeting-pose-replan-3-precommit.v1",
        "cases": selected,
        "source_station_sha256": {c["scene_key"]: sha(ROOT / c["source"])
                                  for c in selected},
        "radii_m": RADII_M,
        "world_bearings_deg": WORLD_BEARINGS_DEG,
        "max_candidates": MAX_CANDIDATES,
        "max_actions": MAX_ACTIONS,
        "max_per_candidate": MAX_PER_CANDIDATE,
        "required_wait_clearance_m": required,
        "policy": "At an unsupported remembered owner station, transform its saved person-aware two-height 360deg scan into a local point cloud. On a fixed 0.15-0.45m grid, require observed full route and possible inbound/escape corridor before proposing a pose. Try at most eight proposals in fixed far-to-near/world-angle order using <=4cm/15deg fresh RGB-D/range-guarded commands. A fresh person-aware wait scan at the reached pose, not the old prediction, must pass before the owner shows the taught item. New RGB-D/owner confirmation runs with independently clocked virtual range streams; otherwise hold unknown.",
        "limits": "Three development-exposed unsupported owner stations, ideal kinematic proxy and static scripted owner; the screen is conservative only for observed ray geometry, not floor gaps or occlusion."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Meeting-pose precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in selected:
        path = OUT / case["scene_key"] / "trace.json"
        if path.exists():
            x = json.loads(path.read_text())
            row = {"scene_key": case["scene_key"],
                "trace": relative(path),
                "outcome": x["presentation_outcome"],
                "selected_candidate": x["selected_candidate"],
                "video": x["video"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_one(case, required)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.meeting-pose-replan-3-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["outcome"],
              row["selected_candidate"], flush=True)


if __name__ == "__main__":
    main()
