"""Screen one motion source and range pose correction across 20 captured rooms.

Each arm continues a retained, owner-confirmed fresh-start teach/find episode.
The same rendered scene, remembered RGB-D item location, measured motion source,
range-gated route planner and new visual recheck are used in both arms. Only
translation correction from the two-plane range fan differs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from native_shaped_pose import NativeShapedPose
from range_pose_corrector import RangePoseCorrector
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_post_confirmation_approach as approach
import run_hidden_item_rearrangement as motion


COHORT = ART / "integrated_owner_20_conditional_precommit.json"
SOURCE_AUDIT = ART / "integrated_owner_20_conditional_audit.json"
APPROACH_AUDIT = ART / "post_confirmation_20_approach_v2_audit.json"
PRECOMMIT = ART / "pose_fusion_20_precommit.json"
REPORT = ART / "pose_fusion_20_report.json"
OUT = ROOT / "assets/real_indoor_scenes/pose_fusion_20_20261015"
SOURCE = "varied150-forward-seed144"
ARMS = ("motion_only", "motion_plus_range")
ANGLES = np.arange(-90., 91., 5., dtype=np.float32)


def source_case(case: dict) -> dict:
    key = case["scene_key"]
    folder = (ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015"
              / key)
    episode, confirmed = approach.find_confirmed_source(folder)
    absent = json.loads((folder / "find_absent/episode_trace.json").read_text())
    if episode["action"] != "owner_confirmed_visual_identity" or \
            absent["action"] != "unidentified_ask_new_view_or_help":
        raise ValueError(f"Source owner task is not confirmed/safe: {key}")
    owner = json.loads((folder / "owner_approach/trace.json").read_text())
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") == key)
    teach = json.loads((folder / "teach/trace.json").read_text())
    return {"folder": folder, "confirmed": confirmed, "owner": owner,
            "placement": placement, "teach": teach}


def run_case(case: dict) -> dict:
    key = case["scene_key"]
    source = source_case(case)
    folder = source["folder"]
    last = source["owner"]["trace"][-1]
    start = np.asarray(SCENES[key]["start_xz"], dtype=float)
    initial = {"actual": np.asarray(last["evaluator_actual_xz_m"], dtype=float),
        "actual_yaw": float(last["evaluator_actual_heading_deg"]),
        "actor": start + np.asarray(last["actor_pose_xy_m"], dtype=float),
        "actor_yaw": float(last["actor_heading_deg"])}
    goal = np.asarray(source["confirmed"]["location"]["actor_map_target_xz_m"],
                      dtype=float)
    target = np.asarray(source["placement"]["find_target_xz"], dtype=float)
    scene = approach.make_scene(case, source["placement"])
    baseline = approach.colour_bank.conditional.baseline
    vision = baseline.Vision("generic-proposal", device="cuda")
    vision._teaching_frame_seen = True
    teach_rgb = np.asarray(Image.open(folder / "teach/rgb_000.png").convert("RGB"))
    box = tuple(source["teach"]["selected_box_xyxy"])
    exemplar = vision.feature([Image.fromarray(teach_rgb).crop(box)])[0]
    colour = baseline.colour_histogram(teach_rgb, box)
    case_out = OUT / key
    case_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ARMS:
        out = case_out / arm
        state = {name: value.copy() if isinstance(value, np.ndarray) else value
                 for name, value in initial.items()}
        native = NativeShapedPose(SOURCE)
        corrector = RangePoseCorrector(ANGLES) if arm == "motion_plus_range" else None
        initial_gap = float(np.linalg.norm(goal - state["actor"]))
        budget = min(96, max(24,
            math.ceil(max(0., initial_gap - approach.STANDOFF_M) / approach.STEP_M)
            + math.ceil(180. / approach.TURN_DEG) + 50))
        walk = motion.walk(scene, state, goal, out / "walk", target,
            max_actions=budget, route_fallback=True,
            pose_increment=native.advance,
            pose_corrector=corrector.correct if corrector else None,
            record_range_fan=True)
        (out / "walk/trace.json").write_text(json.dumps({
            "schema": "ape.pose-fusion-20-walk.v1", "action_budget": budget,
            "observed_goal_xz_m": goal.tolist(), **walk}, indent=2) + "\n")
        final = None
        if walk["result"] == "observed_target_standoff":
            final = baseline.run_search(scene, vision, state["actual"],
                state["actual_yaw"], state["actor"] - start,
                state["actor_yaw"], exemplar,
                float(source["teach"]["acceptance_similarity"]), True,
                out / "final_view", start_time_s=300.,
                search_xz=tuple(source["placement"]["find_target_xz"]),
                search_distractor_xz=tuple(source["placement"]["find_distractor_xz"]),
                owner_start_xz=tuple(start), target_name=case["target"],
                distractor_name=case["distractor"], identity_mode="multi_cue",
                colour_exemplar=colour,
                colour_acceptance=float(source["teach"]["colour_acceptance_similarity"]))
        row = {"scene_key": key, "capture_group": case["capture_group"],
            "arm": arm, "motion_source": SOURCE,
            "motion_source_sha256": native.source_sha256,
            "source_owner_trace": relative(folder / "owner_approach/trace.json"),
            "source_confirmed_trace": source["confirmed"]["stage"]["trace"],
            "source_absent_trace": relative(folder / "find_absent/episode_trace.json"),
            "walk_result": walk["result"],
            "accepted_forward_steps": walk["accepted_forward_steps"],
            "route_fallback_used": walk["route_fallback_used"],
            "max_evaluator_capture_pose_error_m":
                walk["max_evaluator_capture_pose_error_m"],
            "evaluator_final_pose_error_m": float(np.linalg.norm(
                state["actor"] - state["actual"])),
            "evaluator_final_target_gap_m": float(np.linalg.norm(
                state["actual"] - target)),
            "final_action": final["action"] if final else None,
            "final_wrong_confident": final["wrong_confident_candidate"]
                if final else False,
            "walk_trace": relative(out / "walk/trace.json"),
            "walk_video": walk["video"],
            "final_trace": relative(out / "final_view/episode_trace.json")
                if final else None,
            "final_video": final["video"] if final else None,
            "limits": "Frozen prior owner/teach/find source; route and fresh final view are run now. Historical rejected-body motion-error packets, ideal static RGB-D/two-height range and yaw, 17cm kinematic disk, scripted owner. No independently timed range geometry, current body, physical force/contact or learned task policy."}
        (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(key, arm, row["walk_result"], row["final_action"], flush=True)
    return {"scene_key": key, "capture_group": case["capture_group"],
            "arms": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    frozen = json.loads(PRECOMMIT.read_text())
    assert sha(COHORT) == frozen["cohort_sha256"]
    assert sha(SOURCE_AUDIT) == frozen["source_audit_sha256"]
    assert sha(APPROACH_AUDIT) == frozen["command_approach_audit_sha256"]
    assert sha(NativeShapedPose(SOURCE).path) == frozen["motion_trace_sha256"]
    for item in frozen["policy_sources"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    cases = json.loads(COHORT.read_text())["cases"]
    assert len(cases) == 20 and len({c["capture_group"] for c in cases}) == 20
    rows = []
    launched = 0
    for case in cases:
        if args.scene and args.scene != case["scene_key"]:
            continue
        key = case["scene_key"]
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError(f"Changed placement screen: {key}")
        saved = [OUT / key / arm / "summary.json" for arm in ARMS]
        if all(p.is_file() for p in saved):
            row = {"scene_key": key, "capture_group": case["capture_group"],
                   "arms": [json.loads(p.read_text()) for p in saved]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_case(case)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.pose-fusion-20-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
