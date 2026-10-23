"""Choose a buffered wait before V589 re-teach, then exercise the full episode.

The choice uses a live range packet and the route's known-free grid. The
bystander follows one fixed world path in every trial and starts moving only
during the released RGB-D item worker. The ape motion remains kinematic.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))

import audit_robocasa_social_reteach_v590 as audit  # noqa: E402
import robocasa_first_return_person_reacquire_turn_v576 as turn  # noqa: E402
import run_robocasa_joined_owner_route_v410 as owner  # noqa: E402
import run_robocasa_layout2_multitrack_v485 as multi  # noqa: E402
import run_robocasa_person_joined_v497 as joined  # noqa: E402
import run_robocasa_social_reteach_v589 as reteach  # noqa: E402
import run_robocasa_two_moving_people_v586 as moving  # noqa: E402
import screen_robocasa_two_people_v582 as two  # noqa: E402


OUT = ROOT / "artifacts/ape-native-route-guard/v33-owner-wait-pose"
CONDITIONS = ("held_moving", "buffered_moving", "buffered_static",
              "buffered_no_side", "held_turning", "buffered_turning")
SIDE_START = np.array([.5562619673, -4.2441418102])
SIDE_TARGET = np.array([1.2062619673, -4.2441418102])
TURNING_TARGET = SIDE_TARGET+[0., -.15]
APPROACH_MPS = .62
WAIT_SHIFT_M = .25
WAIT_SPEED_MPS = .25
RANGE_TRIGGER_M = 1.20
CHANGE_M = .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def all_changed(a, b):
    return sum(((old is None) != (new is None)) or
               (old is not None and new is not None and abs(old-new) >= CHANGE_M)
               for old, new in zip(a, b))


def side_range(packet):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    side = [x for x, angle in zip(packet, angles)
            if x is not None and abs((angle+math.pi)%(2*math.pi)-math.pi) >=
            math.radians(120.)]
    return min(side, default=None)


def run_condition(condition, policy, scenario, prior, vision, exemplar):
    original_turn = turn.owner_turn
    original_prop = turn._prop
    original_sector = turn._sector_change
    original_scan = reteach.SCAN
    original_pool = reteach.ThreadPoolExecutor
    original_out = reteach.OUT
    original_path = moving.distractor_xy_at
    state = {"worker": False, "side_staged": False,
             "worker_start_s": None, "current_s": None,
             "last_scan": None, "baseline": None, "packets": [],
             "changes": [], "wait": [], "wait_choice": None,
             "ape_after_wait_xy_m": None, "ape_during_inference_xy_m": None}

    target = TURNING_TARGET if condition.endswith("turning") else SIDE_TARGET

    def bystander(t):
        if condition == "buffered_no_side":
            return np.array([-10., -10.])
        if condition == "buffered_static" or state["worker_start_s"] is None:
            return SIDE_START.copy()
        dt = max(0., t-state["worker_start_s"])
        fraction = min(1., APPROACH_MPS*dt/np.linalg.norm(target-SIDE_START))
        return SIDE_START+(target-SIDE_START)*fraction

    def prop(model, t):
        original_prop(model, t)
        if not state["side_staged"]:
            return
        if state["worker"] and state["worker_start_s"] is None:
            state["worker_start_s"] = float(t)
        two.set_distractor(model, bystander(float(t)))
        state["current_s"] = float(t)

    def scan(model, data, xy, max_range_m, rng):
        packet = original_scan(model, data, xy, max_range_m, rng)
        if state["worker"]:
            state["packets"].append({"time_s": round(state["current_s"], 5),
                                     "ranges_m": packet})
        else:
            state["last_scan"] = packet
        return packet

    def sector(a, b, bearing, width):
        count = all_changed(a, b)
        if state["worker"]:
            state["changes"].append({"time_s": round(state["current_s"], 5),
                                     "all_changed": count,
                                     "owner_changed": original_sector(a, b, bearing, width)})
        return count

    class GuardPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            state["worker"] = True
            state["baseline"] = list(state["last_scan"])
            return super().submit(*args, **kwargs)

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                state["worker"] = False

    def intercept(response, model, data, renderer, actual, estimate,
                  time_s, use_depth, vision, exemplar):
        outcome, original_rows, original_frames, wall, error, t = original_turn(
            response, model, data, renderer, actual, estimate,
            time_s, use_depth, vision, exemplar)
        if outcome not in ("owner_interaction_reset_after_reacquire_motion",
                           "owner_track_uncertain_before_show"):
            return outcome, original_rows, original_frames, wall, error, t
        state["side_staged"] = True
        moving.distractor_xy_at = bystander
        two.set_distractor(model, bystander(t))
        mujoco.mj_forward(model, data)
        packet = reteach.SCAN(model, data, actual, 5., np.random.default_rng(33000))
        nearest = side_range(packet)
        known, cells = turn._known_free_south(multi.CAPTURED[-1], estimate,
                                              WAIT_SHIFT_M)
        should_wait = (condition.startswith("buffered") and known and
                       nearest is not None and nearest < RANGE_TRIGGER_M)
        state["wait_choice"] = {"time_s": round(t, 5),
                                "range_packet": packet,
                                "nearest_side_range_m": nearest,
                                "known_free_south": known,
                                "known_free_cells": cells,
                                "action": "move_south_then_show" if should_wait else
                                          "hold_current_pose"}
        rows = original_rows+[{"stage": "actor_wait_pose_choice", **state["wait_choice"]}]
        frames = original_frames.copy()
        if should_wait:
            start_actual = actual.copy()
            start_estimate = estimate[:2].copy()
            for index in range(1, 51):
                t += .02
                fraction = index/50.
                actual[:] = start_actual+[0., -WAIT_SHIFT_M*fraction]
                estimate[:2] = start_estimate+[0., -WAIT_SHIFT_M*fraction]
                model.body_pos[model.body("audit_ape").id, :2] = actual
                turn._prop(model, t)
                mujoco.mj_forward(model, data)
                if index % 5 == 0:
                    live = reteach.SCAN(model, data, actual, 5.,
                                       np.random.default_rng(33000+index))
                    gap = audit.distances(model, data, actual)["distractor"]
                    state["wait"].append({"time_s": round(t, 5),
                                          "ape_xy_m": actual.tolist(),
                                          "nearest_side_range_m": side_range(live),
                                          "evaluator_person_gap_m": round(gap, 5)})
                    model.cam_pos[model.camera(reteach.CAMERA).id, :2] = actual
                    mujoco.mj_forward(model, data)
                    rgb, _, _ = owner.capture(renderer, data, reteach.CAMERA, False)
                    frames.append(owner.owner_frame(rgb,
                        "moving to a buffered place to wait for the show", None, None))
            rows.append({"stage": "actor_wait_pose_motion",
                         "time_s": round(t, 5),
                         "start_xy_m": start_actual.tolist(),
                         "end_xy_m": actual.tolist(),
                         "distance_m": WAIT_SHIFT_M,
                         "elapsed_s": 1.,
                         "skill_qualification": "kinematic_reference_only"})
        state["ape_after_wait_xy_m"] = actual.tolist()
        return outcome, rows, frames, wall, error, t

    try:
        turn.owner_turn = intercept
        turn._prop = prop
        turn._sector_change = sector
        reteach.SCAN = scan
        reteach.ThreadPoolExecutor = GuardPool
        reteach.OUT = OUT / condition
        result = reteach.run_case("owner_reteach", policy, scenario, prior,
                                  vision, exemplar)
    finally:
        turn.owner_turn = original_turn
        turn._prop = original_prop
        turn._sector_change = original_sector
        reteach.SCAN = original_scan
        reteach.ThreadPoolExecutor = original_pool
        reteach.OUT = original_out
        moving.distractor_xy_at = original_path
    ep = result["result"]["episode"]
    state["ape_during_inference_xy_m"] = ep["final_actual_xy_m"]
    guard = next((row for row in ep["trace"]
                  if row["stage"] == "new_show_inference_guard"), None)
    summary = {"condition": condition, "outcome": ep["outcome"],
               "bystander_target_evaluator_xy_m": target.tolist(),
               "route_state": ep["route_state"],
               "wait_choice": state["wait_choice"],
               "wait_motion_samples": state["wait"],
               "ape_after_wait_xy_m": state["ape_after_wait_xy_m"],
               "ape_after_episode_xy_m": ep["final_actual_xy_m"],
               "worker_start_s": state["worker_start_s"],
               "baseline_range_packet": state["baseline"],
               "range_packets": state["packets"],
               "range_changes": state["changes"],
               "guard_alarm_time_s": guard["alarm_time_s"] if guard else None,
               "video": ep["video"]}
    return result, summary


def score_inference_gap(summary):
    if summary["worker_start_s"] is None:
        return None
    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    ape = np.asarray(summary["ape_after_episode_xy_m"])
    target = np.asarray(summary["bystander_target_evaluator_xy_m"])
    start = summary["worker_start_s"]
    end = summary["range_packets"][-1]["time_s"]
    gaps = []
    for t in np.arange(start, end+.0001, .02):
        if summary["condition"] == "buffered_no_side":
            xy = np.array([-10., -10.])
        elif summary["condition"] == "buffered_static":
            xy = SIDE_START
        else:
            fraction = min(1., APPROACH_MPS*(t-start)/
                           np.linalg.norm(target-SIDE_START))
            xy = SIDE_START+(target-SIDE_START)*fraction
        two.set_distractor(model, xy)
        gap = audit.distances(model, data, ape)["distractor"]
        gaps.append({"time_s": round(float(t), 5), "gap_m": round(gap, 5)})
    return {"minimum_gap_m": min(row["gap_m"] for row in gaps),
            "first_below_5cm_s": next((row["time_s"] for row in gaps
                                       if row["gap_m"] < .05), None),
            "samples": gaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    args = parser.parse_args()
    conditions = tuple(args.condition) if args.condition else CONDITIONS
    policy = json.loads(joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    results, summaries = [], []
    for condition in conditions:
        result, summary = run_condition(condition, policy, scenario, prior,
                                        vision, exemplar)
        summary["inference_geometry_evaluator"] = score_inference_gap(summary)
        results.append(result)
        summaries.append(summary)
        print(condition, summary["wait_choice"]["action"],
              summary["outcome"],
              summary["inference_geometry_evaluator"]["minimum_gap_m"]
              if summary["inference_geometry_evaluator"] else None,
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-wait-pose.v1",
              "task": "T-9v3p3n",
              "scope": "Full V589 owner turn and fixed-world side path. Kinematic pre-show wait motion uses a live 30cm high range packet and map-known south path; actor sees no evaluator person pose. Same ideal all-angle inference guard in every condition. Historical reference geometry only; no qualified selected body or escape skill.",
              "source_sha256": sha(Path(__file__)),
              "base_runner_sha256": sha(reteach.__file__),
              "scene_sha256": sha(moving.SCENE_XML),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "side_start_xy_evaluator_m": SIDE_START.tolist(),
              "side_target_xy_evaluator_m": SIDE_TARGET.tolist(),
              "conditions": conditions, "summaries": summaries,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
