"""Approach and reconfirm items recovered by one owner point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
from run_compound_owner_point_recovery import (OUT as HELP_OUT, case_rows,
    make_scene)
from audit_compound_owner_point_recovery import AUDIT as HELP_AUDIT
import run_hidden_item_rearrangement as motion


PRECOMMIT = ART / "compound_owner_point_approach_precommit.json"
REPORT = ART / "compound_owner_point_approach_report.json"
OUT = ROOT / "assets/real_indoor_scenes/compound_owner_point_approach_20261015"


def confirmed_location(episode: dict) -> np.ndarray:
    for stage in episode["stage_reports"]:
        answer = stage["owner_answer"]
        if answer and answer["action"].startswith("owner_confirmed"):
            trace = json.loads((ROOT / stage["trace"]).read_text())
            location = trace["verification"]["candidate_localization"]
            if location["status"] == "located" and location.get(
                    "actor_map_target_xz_m") is not None:
                return np.asarray(location["actor_map_target_xz_m"], float)
    raise ValueError("Helped item lacks an owner-confirmed RGB-D location")


def final_recheck(scene, case: dict, point: dict, state: dict,
                  out: Path) -> dict:
    baseline = motion.base.colour_bank.conditional.baseline
    source = ROOT / case["source_folder"]
    teach = json.loads((source / "teach/trace.json").read_text())
    signal = point["owner_signal"]
    image = np.asarray(Image.open(ROOT / signal["rgb"]).convert("RGB"))
    box = point["selected_proposal"]["box"]
    vision = baseline.Vision("generic-proposal")
    vision._teaching_frame_seen = True
    exemplar = vision.feature([Image.fromarray(image).crop(tuple(
        round(v) for v in box))])[0]
    colour = baseline.colour_histogram(image, box)
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") ==
                     case["scene_key"])
    start = np.asarray(SCENES[case["scene_key"]]["start_xz"], float)
    return baseline.run_search(scene, vision, state["actual"],
        state["actual_yaw"], state["actor"] - start, state["actor_yaw"],
        exemplar, float(teach["acceptance_similarity"]), True, out,
        start_time_s=300.,
        search_xz=tuple(placement["find_target_xz"]),
        search_distractor_xz=tuple(placement["find_distractor_xz"]),
        owner_start_xz=tuple(start), target_name=case["target"],
        distractor_name=case["distractor"], identity_mode="multi_cue",
        colour_exemplar=colour,
        colour_acceptance=float(teach["colour_acceptance_similarity"]),
        search_photometry="cool_dim")


def run_one(case: dict) -> dict:
    key = case["scene_key"]
    source = ROOT / case["source_folder"]
    point = json.loads((HELP_OUT / key / "accurate_present/trace.json").read_text())
    episode = json.loads((ROOT / point["fresh_search"]).read_text())
    if episode["action"] != "owner_confirmed_visual_identity":
        raise ValueError("Cannot move toward an unconfirmed item")
    goal = confirmed_location(episode)
    owner = json.loads((source / "owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    start = np.asarray(SCENES[key]["start_xz"], float)
    state = {"actual": np.asarray(last["evaluator_actual_xz_m"], float),
        "actual_yaw": float(last["evaluator_actual_heading_deg"]),
        "actor": start + np.asarray(last["actor_pose_xy_m"], float),
        "actor_yaw": float(last["actor_heading_deg"])}
    initial_actor = state["actor"].copy()
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") == key)
    target = np.asarray(placement["find_target_xz"], float)
    scene = make_scene(case, True)
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    walk = motion.walk(scene, state, goal, out / "guarded_walk", target,
                       photometry="cool_dim")
    final = (final_recheck(scene, case, point, state, out / "final_recheck")
             if walk["result"] == "observed_target_standoff" else None)
    trace = {"schema": "ape.compound-owner-point-approach.v1",
        "scene_key": key,
        "source_help": relative(HELP_OUT / key / "accurate_present/trace.json"),
        "observed_goal_xz_m": goal.tolist(),
        "initial_actual_xz_m": last["evaluator_actual_xz_m"],
        "initial_actor_xz_m": initial_actor.tolist(),
        "walk": walk,
        "final_recheck": (relative(out / "final_recheck/episode_trace.json")
                          if final else None),
        "final_action": final["action"] if final else None,
        "evaluator_final_target_gap_m": float(np.linalg.norm(state["actual"] - target)),
        "limits": "Kinematic 17cm disk, ideal yaw and two-height range; command-integrated pose during the short approach, static person/room/items, scripted owner. No selected physical body or measured IMU/joint/contact/stop."}
    path = out / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": key, "trace": relative(path),
        "walk_result": walk["result"], "final_action": trace["final_action"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = case_rows()
    frozen = {"schema": "ape.compound-owner-point-approach-precommit.v1",
        "source_help_audit": relative(HELP_AUDIT),
        "source_help_audit_sha256": sha(HELP_AUDIT),
        "cases": cases,
        "policy": "Only an owner-confirmed, fresh RGB-D localized help candidate becomes a metric goal. From the reached owner pose, use unchanged bounded two-height-range guarded v2 walk to an observed 55cm standoff, then a newly rendered cool/dim conditional visual search and another scripted owner yes/no. No movement on silence or wrong-point unknown branches.",
        "limits": "Six development-observed visible-proposal misses, scripted owner and static captured worlds; provisional 17cm disk and ideal range/yaw."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Approach precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in cases:
        path = OUT / case["scene_key"] / "trace.json"
        if path.exists():
            trace = json.loads(path.read_text())
            row = {"scene_key": case["scene_key"], "trace": relative(path),
                "walk_result": trace["walk"]["result"],
                "final_action": trace["final_action"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_one(case)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.compound-owner-point-approach-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(row["scene_key"], row["walk_result"],
              row["final_action"], flush=True)


if __name__ == "__main__":
    main()
