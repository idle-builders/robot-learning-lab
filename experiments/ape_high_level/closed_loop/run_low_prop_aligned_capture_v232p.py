"""Track delayed low rays at the actor's estimate of their capture pose.

V227p delivered true prior-pose rays but its tracker projected them from the
receipt pose. This adapter keeps one estimated pose per decision, suppresses
duplicate capture packets, and tracks a newly delivered sample at its own
capture time/estimated pose. Compare with and without the age/local-corridor
action gates in complete owner episodes. Evaluator actual pose is never used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import novel_range_object as tracking
from run import ROOT, sha
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_capture_age_v227p as v227
import run_low_prop_local_corridor_v230p as v230
import run_low_prop_held_pose_v224p as v224


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-aligned-capture-v232p"
PROFILES = ("aligned_no_age", "aligned_age_local")


def one(case, profile, prior, vision, exemplar):
    original_update = tracking.NovelRangeObject.update
    original_v227_out, original_v230_out = v227.OUT, v230.OUT
    actor_poses = []
    events = []
    last_source_index = -1

    def aligned_update(tracker, ranges, receipt_pose_xy, receipt_time_s):
        nonlocal last_source_index
        index = len(actor_poses)
        if abs(receipt_time_s - index * .4) > 1e-5:
            raise AssertionError("Tracker decision clock changed")
        actor_poses.append(np.asarray(receipt_pose_xy).copy())
        source_index = (index - v227.LOW_DELIVERY_DELAY_STEPS
                        if v227.DELAY_WINDOW_S[0] <= receipt_time_s <=
                        v227.DELAY_WINDOW_S[1] and
                        index >= v227.LOW_DELIVERY_DELAY_STEPS else index)
        if source_index < last_source_index:
            raise AssertionError("Out-of-order low capture")
        if source_index == last_source_index:
            previous = tracker.last_result
            if previous is None:
                raise AssertionError("Duplicate before first low track")
            result = {**previous, "status": "stale_reused_track",
                      "velocity_xy_m_s": None}
            action = "reused_without_tracker_update"
        else:
            result = original_update(
                tracker, ranges, actor_poses[source_index], source_index * .4)
            last_source_index = source_index
            action = "capture_pose_update"
        result = {**result, "source_capture_index": source_index,
                  "receipt_time_s": receipt_time_s,
                  "actor_capture_pose_xy_m": actor_poses[source_index].tolist()}
        events.append({"decision_index": index,
                       "receipt_time_s": receipt_time_s,
                       "source_capture_index": source_index,
                       "source_capture_time_s": source_index * .4,
                       "actor_capture_pose_xy_m": actor_poses[source_index].tolist(),
                       "actor_receipt_pose_xy_m": actor_poses[index].tolist(),
                       "action": action,
                       "result_status": result["status"]})
        return result

    tracking.NovelRangeObject.update = aligned_update
    try:
        if profile == "aligned_no_age":
            v227.OUT = OUT / profile
            result = v227.one(case, "naive_delivery", prior, vision, exemplar)
        elif profile == "aligned_age_local":
            v230.OUT = OUT / profile
            result = v230.one(case, prior, vision, exemplar)
        else:
            raise ValueError(profile)
    finally:
        tracking.NovelRangeObject.update = original_update
        v227.OUT = original_v227_out
        v230.OUT = original_v230_out
    if len(events) != len(result["route_episode"]["trace"]):
        raise AssertionError("Aligned tracker lost a decision")
    result["aligned_capture_profile"] = profile
    result["aligned_tracker_events"] = events
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    original_window, original_margin = v212.WINDOW_S, v206.route.MARGIN_M
    v212.WINDOW_S = (999., 1000.)
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in v224.CASES:
            for profile in PROFILES:
                result = one(case, profile, prior, vision, exemplar)
                episodes.append(result)
                print(case["id"], profile, result["route_state"],
                      result["outcome"], "duplicate_captures", sum(
                          event["action"] == "reused_without_tracker_update"
                          for event in result["aligned_tracker_events"]),
                      "min_gap", round(result["route_episode"][
                          "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v212.WINDOW_S = original_window
        v206.route.MARGIN_M = original_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-aligned-capture.v1",
        "source_sha256": sha(SOURCE),
        "v227_source_sha256": sha(Path(v227.__file__)),
        "v230_source_sha256": sha(Path(v230.__file__)),
        "tracker_source_sha256": sha(Path(tracking.__file__)),
        "current_design_status": pointer["status"],
        "delay_window_s": v227.DELAY_WINDOW_S,
        "low_delivery_delay_steps": v227.LOW_DELIVERY_DELAY_STEPS,
        "profiles": PROFILES,
        "changed_factor": "Project each delivered low packet from its "
                          "saved actor-estimated capture-decision pose and "
                          "source capture time; do not update tracker twice "
                          "for the same source sample. Compare no action "
                          "age gate to V227 age + V230 local low veto.",
        "scope": "Same three layout4 near-fast owner episodes, actually "
                 "delayed low rays, unchanged mid-height localization "
                 "and RGB-D owner find. Actor capture pose is estimated "
                 "after that decision's mid-scan localization, not hidden "
                 "world truth. No physical sensor timing or native action.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
