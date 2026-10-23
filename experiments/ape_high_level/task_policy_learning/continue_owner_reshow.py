"""Let the learned option policy request one scripted owner reshow at a safe stop.

This continues a saved fresh-start learner episode. The existing reshow skill
stages the owner's new presentation; its later action controller is authored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "real_scan_benchmark"))

import run_scan10_show_again_continuation as show
import run_scan30_unified_policy as benchmark
import run_compound_owner_point_recovery as owner_help
import run_alt_pair_full_action_20 as full_action
from live_option_episode import alternate_item_case
from option_controller import OptionController

PILOT = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"


def run(case: dict, source_run: str, policy: dict,
        destination: str = "owner_reshow_v2") -> dict:
    key = case["scene_key"]
    base = PILOT / key / source_run
    first = json.loads((base / "summary.json").read_text())
    if first["result"] != "owner_or_teaching_unresolved":
        raise ValueError("Owner reshow requires an owner-stage stop")
    owner = json.loads((base / "initial/owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    actor = last.get("owner_actor") or {}
    depth = actor.get("local_depth_m")
    bearing = actor.get("owner_bearing_deg")
    safe_view = bool(last.get("pose_healthy") and depth is not None and
        bearing is not None and float(depth) <= 1.7 and
        abs(float(bearing)) <= 20.)
    controller = OptionController(policy)
    controller.reset()
    observation = controller.observe("owner_result", {
        "owner_ready": False, "teaching_confirmed": False,
        "owner_visible": actor.get("chosen_proposal") is not None,
        "pose_healthy": bool(last.get("pose_healthy")),
        "owner_safe_to_ask": safe_view})
    decision = controller.step()
    prior = [event for event in first.get("events", [])
             if event.get("phase") == "owner_result"]
    if prior:
        if len(prior) != 1 or prior[0]["observation"] != observation:
            raise ValueError("Saved owner option does not match current observation")
        decision = {"requested_option": prior[0]["requested"],
                    "authorized_option": prior[0]["executed"]}
    selected = decision["authorized_option"] if safe_view else "HOLD"
    row = {"schema": "ape.learned-owner-reshow-continuation.v1",
        "scene_key": key, "capture_group": case["capture_group"],
        "source_episode": str((base / "summary.json").relative_to(ROOT)),
        "observation": observation, "requested": decision["requested_option"],
        "authorized": selected, "safe_visible_owner": safe_view,
        "result": "safe_unresolved", "continuation_summary": None,
        "limits": "Learner chooses ASK_RESHOW from actor-observable owner and pose signals at a recorded owner endpoint. The scripted owner and existing authored reshow/action skill execute the continuation. This does not yet train owner approach or continuation action options."}
    out = PILOT / destination / key
    out.mkdir(parents=True, exist_ok=True)
    if selected != "ASK_RESHOW":
        (out / "learner_summary.json").write_text(json.dumps(row, indent=2) + "\n")
        return row
    source_root = PILOT / f"{destination}_sources"
    stub = source_root / key / case["photometry"] / "healthy"
    trace_dest = stub / "initial/owner_approach/trace.json"
    trace_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base / "initial/owner_approach/trace.json", trace_dest)
    shutil.copy2(base / "summary.json", stub / "summary.json")
    if (out / "summary.json").is_file():
        continuation = json.loads((out / "summary.json").read_text())
    else:
        old_prior, old_out, old_owner = show.PRIOR_OUT, show.OUT, show.SOURCE_OWNER
        try:
            show.PRIOR_OUT = source_root
            show.OUT = PILOT / destination
            show.SOURCE_OWNER = show.OUT / "source_owner"
            continuation = show.run_one(case)
        finally:
            show.PRIOR_OUT, show.OUT, show.SOURCE_OWNER = old_prior, old_out, old_owner
    teaching = continuation.get("teaching") or {}
    present = teaching.get("present") or {}
    if (continuation["result"] == "safe_unresolved" and
            teaching.get("teach_confirmed") and
            present.get("action") == "unidentified_ask_new_view_or_help"):
        followup = controller.observe("wide_result", {
            "wide_confirmed": False, "resolved_confirmed": False})
        help_decision = controller.step()
        row["owner_help_observation"] = followup
        row["owner_help_requested"] = help_decision["requested_option"]
        row["owner_help_authorized"] = help_decision["authorized_option"]
        if help_decision["authorized_option"] == "ASK_OWNER":
            point_case = {"scene_key": key,
                "capture_group": case["capture_group"],
                "source_folder": str(out.relative_to(ROOT)),
                "screen": continuation["staging"]["screen"],
                "target": case["item"]["target"],
                "distractor": case["item"]["distractor"]}
            help_root = PILOT / f"{destination}_owner_help"
            trace_path = help_root / key / "accurate_present/trace.json"
            if trace_path.is_file():
                point = json.loads(trace_path.read_text())
            else:
                result = owner_help.run_one(point_case,
                    ("accurate_present", "find_present", "accurate"),
                    output_root=help_root)
                point = json.loads((ROOT / result["trace"]).read_text())
            row["owner_help_action"] = point["action"]
            row["owner_help_trace"] = str(trace_path.relative_to(ROOT))
            if point["action"] == "owner_confirmed_visual_identity":
                saved = (full_action.SOURCE_ROOT, full_action.HELP_ROOT,
                         full_action.OUT)
                try:
                    full_action.SOURCE_ROOT = PILOT / destination
                    full_action.HELP_ROOT = help_root
                    full_action.OUT = out / "continuation_action"
                    continued = {**point_case, "branch": "owner_help"}
                    action_path = full_action.OUT / key / "healthy/summary.json"
                    action = (json.loads(action_path.read_text())
                              if action_path.is_file() else
                              full_action.run_one(continued, "healthy"))
                finally:
                    (full_action.SOURCE_ROOT, full_action.HELP_ROOT,
                     full_action.OUT) = saved
                continuation["action"] = action
                continuation["result"] = ("completed" if
                    action.get("final_action") ==
                    "owner_confirmed_visual_identity" else "safe_unresolved")
                (out / "summary.json").write_text(json.dumps(
                    continuation, indent=2) + "\n")
    row["result"] = continuation["result"]
    row["continuation_summary"] = str((out / "summary.json").relative_to(ROOT))
    (out / "learner_summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-run", default="actor_only_v2_transfer")
    parser.add_argument("--destination", default="owner_reshow_v2")
    parser.add_argument("--policy", type=Path,
                        default=HERE / "event_policy_live_v2.joblib")
    parser.add_argument("--scenario-variant", choices=("standard", "alt_pair"),
                        default="standard")
    args = parser.parse_args()
    case = next(c for c in benchmark.cohort() if c["scene_key"] == args.scene)
    if args.scenario_variant == "alt_pair":
        case = alternate_item_case(case)
    row = run(case, args.source_run, joblib.load(args.policy), args.destination)
    print(json.dumps({k: row[k] for k in ("scene_key", "requested",
        "authorized", "safe_visible_owner", "result", "continuation_summary")},
        indent=2))


if __name__ == "__main__":
    main()
