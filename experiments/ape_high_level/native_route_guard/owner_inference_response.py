"""Act on a side-person alarm while V589's slow item inference still runs.

The actor first uses V34's range-cued wait, then compares an owner-facing
sector with full-angle range monitoring and two response envelopes while the
person turns toward the held ape. All ape movement is a kinematic surrogate.
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


OUT = ROOT / "artifacts/ape-native-route-guard/v35-inference-response"
CONDITIONS = ("sector_buffered_turning", "wide_buffered_turning",
              "wide_respond_fast", "wide_respond_slow",
              "wide_buffered_static", "wide_no_entry")
SIDE_START = np.array([.5562619673, -4.2441418102])
SIDE_TARGET = np.array([1.2062619673, -4.2441418102])
TURNING_TARGET = SIDE_TARGET+[0., -.15]
APPROACH_MPS = .62
WAIT_SHIFT_M = .25
WAIT_SPEED_MPS = .25
RANGE_TRIGGER_M = 1.20
CHANGE_M = .05
STAGE_S = 1.2
STAGE_STEP_S = .1
RESPONSE_MAX_M = .25
RESPONSE = {"wide_respond_fast": {"delay_s": .1, "speed_mps": .25},
            "wide_respond_slow": {"delay_s": .5, "speed_mps": .1}}


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


def run_condition(condition, policy, scenario, prior, vision, exemplar,
                  *, inference_probe=None, output_root=OUT):
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
             "ape_after_wait_xy_m": None, "ape_during_inference_xy_m": None,
             "stage_start_s": None, "stage_start_xy_m": None,
             "stage_packets": [], "stage_baseline": None,
             "actual_ref": None, "estimate_ref": None,
             "worker_ape_start_xy_m": None, "worker_estimate_start_xy_m": None,
             "actor_streak": 0, "actor_alarm_s": None,
             "response_plan": None, "response_path": [],
             "first_action_s": None, "last_model": None,
             "last_data": None, "probe_last": None, "probe_samples": []}

    target = TURNING_TARGET

    def bystander(t):
        stage_start = state["stage_start_s"]
        if stage_start is None:
            return np.asarray(original_path(t))
        start_xy = np.asarray(state["stage_start_xy_m"])
        if condition == "wide_no_entry":
            return start_xy.copy()
        if t < stage_start+STAGE_S:
            fraction = min(1., max(0., (t-stage_start)/STAGE_S))
            return start_xy+(SIDE_START-start_xy)*fraction
        if condition == "wide_buffered_static" or state["worker_start_s"] is None:
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
        if state["worker"] and state["actor_alarm_s"] is not None:
            response = RESPONSE.get(condition)
            plan = state["response_plan"]
            if response is not None and plan is not None and plan["known_free"]:
                delay = response["delay_s"]
                speed = response["speed_mps"]
                displacement = min(RESPONSE_MAX_M, max(0.,
                    speed*(t-state["actor_alarm_s"]-delay)))
                actual = state["actual_ref"]
                estimate = state["estimate_ref"]
                actual[:] = np.asarray(state["worker_ape_start_xy_m"])+[0., -displacement]
                estimate[:2] = np.asarray(state["worker_estimate_start_xy_m"])+[0., -displacement]
                model.body_pos[model.body("audit_ape").id, :2] = actual
                model.cam_pos[model.camera(reteach.CAMERA).id, :2] = actual
                if displacement > 0 and state["first_action_s"] is None:
                    state["first_action_s"] = float(t)
                state["response_path"].append({"time_s": round(float(t), 5),
                                                "displacement_m": round(displacement, 5),
                                                "ape_xy_m": actual.tolist()})
        two.set_distractor(model, bystander(float(t)))
        state["current_s"] = float(t)

    def scan(model, data, xy, max_range_m, rng):
        packet = original_scan(model, data, xy, max_range_m, rng)
        state["last_model"] = model
        state["last_data"] = data
        if state["worker"]:
            state["packets"].append({"time_s": round(state["current_s"], 5),
                                     "ranges_m": packet})
            if inference_probe is not None:
                sample = inference_probe.observe(model, data, xy,
                                                 state["current_s"])
                state["probe_last"] = sample
                state["probe_samples"].append(sample)
        else:
            state["last_scan"] = packet
        return packet

    def sector(a, b, bearing, width):
        count = all_changed(a, b)
        owner_count = original_sector(a, b, bearing, width)
        used = owner_count if condition == "sector_buffered_turning" else count
        if inference_probe is not None:
            used = 3 if (state["probe_last"] and
                         state["probe_last"]["actor_motion_cue"]) else 0
        if state["worker"]:
            state["changes"].append({"time_s": round(state["current_s"], 5),
                                     "all_changed": count,
                                     "owner_changed": owner_count,
                                     "actor_used_changed": used})
            state["actor_streak"] = state["actor_streak"]+1 if used >= 3 else 0
            if state["actor_streak"] >= 2 and state["actor_alarm_s"] is None:
                state["actor_alarm_s"] = state["current_s"]
                known, cells = turn._known_free_south(
                    multi.CAPTURED[-1], state["estimate_ref"], RESPONSE_MAX_M)
                state["response_plan"] = {"known_free": known, "cells": cells,
                                          "requested_action": "south_response" if
                                          condition in RESPONSE else "no_response_control"}
        return used

    class GuardPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            if inference_probe is not None:
                inference_probe.start(state["last_model"], state["last_data"],
                                      state["actual_ref"], state["current_s"])
            state["worker"] = True
            state["baseline"] = list(state["last_scan"])
            state["worker_ape_start_xy_m"] = state["actual_ref"].copy().tolist()
            state["worker_estimate_start_xy_m"] = state["estimate_ref"][:2].copy().tolist()
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
        state["actual_ref"] = actual
        state["estimate_ref"] = estimate
        state["stage_start_s"] = float(t)
        state["stage_start_xy_m"] = np.asarray(original_path(t)).tolist()
        moving.distractor_xy_at = bystander
        two.set_distractor(model, bystander(t))
        mujoco.mj_forward(model, data)
        baseline = reteach.SCAN(model, data, actual, 5., np.random.default_rng(34000))
        state["stage_baseline"] = baseline
        streak = 0
        motion_seen = False
        for index in range(1, int(round(STAGE_S/STAGE_STEP_S))+1):
            t = state["stage_start_s"]+index*STAGE_STEP_S
            turn._prop(model, t)
            mujoco.mj_forward(model, data)
            packet = reteach.SCAN(model, data, actual, 5.,
                                 np.random.default_rng(34000+index))
            count = all_changed(baseline, packet)
            streak = streak+1 if count >= 3 else 0
            motion_seen |= streak >= 2
            state["stage_packets"].append({"time_s": round(t, 5),
                                           "ranges_m": packet,
                                           "changed_beams": count,
                                           "motion_streak": streak})
        nearest = side_range(packet)
        known, cells = turn._known_free_south(multi.CAPTURED[-1], estimate,
                                              WAIT_SHIFT_M)
        should_wait = (condition != "wide_no_entry" and motion_seen and known and
                       nearest is not None and nearest < RANGE_TRIGGER_M)
        state["wait_choice"] = {"time_s": round(t, 5),
                                "range_packet": packet,
                                "motion_seen": motion_seen,
                                "stage_changed_beams": [x["changed_beams"]
                                                        for x in state["stage_packets"]],
                                "nearest_side_range_m": nearest,
                                "known_free_south": known,
                                "known_free_cells": cells,
                                "action": "move_south_then_show" if should_wait else
                                          "hold_current_pose"}
        rows = original_rows+[{"stage": "actor_side_motion_observation",
                              "start_time_s": round(state["stage_start_s"], 5),
                              "end_time_s": round(t, 5),
                              "actor_packets": state["stage_packets"]},
                             {"stage": "actor_wait_pose_choice", **state["wait_choice"]}]
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
        reteach.OUT = output_root / condition
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
        if inference_probe is not None:
            inference_probe.close()
    ep = result["result"]["episode"]
    state["ape_during_inference_xy_m"] = ep["final_actual_xy_m"]
    guard = next((row for row in ep["trace"]
                  if row["stage"] == "new_show_inference_guard"), None)
    summary = {"condition": condition, "outcome": ep["outcome"],
               "bystander_target_evaluator_xy_m": target.tolist(),
               "bystander_stage_start_evaluator_xy_m": state["stage_start_xy_m"],
               "route_state": ep["route_state"],
               "stage_baseline_range_packet": state["stage_baseline"],
               "stage_packets": state["stage_packets"],
               "wait_choice": state["wait_choice"],
               "wait_motion_samples": state["wait"],
               "ape_after_wait_xy_m": state["ape_after_wait_xy_m"],
               "ape_after_episode_xy_m": ep["final_actual_xy_m"],
               "worker_start_s": state["worker_start_s"],
               "worker_ape_start_xy_m": state["worker_ape_start_xy_m"],
               "actor_alarm_s": state["actor_alarm_s"],
               "response_parameters": RESPONSE.get(condition),
               "response_plan": state["response_plan"],
               "response_path": state["response_path"],
               "first_action_s": state["first_action_s"],
               "baseline_range_packet": state["baseline"],
               "range_packets": state["packets"],
               "range_changes": state["changes"],
               "inference_sensor": (inference_probe.name if inference_probe
                                    is not None else "ideal_all_angle_range"),
               "inference_probe_samples": state["probe_samples"],
               "guard_alarm_time_s": guard["alarm_time_s"] if guard else None,
               "video": ep["video"]}
    return result, summary


def score_inference_gap(summary):
    if summary["worker_start_s"] is None:
        return None
    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    ape_start = np.asarray(summary["worker_ape_start_xy_m"])
    target = np.asarray(summary["bystander_target_evaluator_xy_m"])
    start = summary["worker_start_s"]
    end = summary["range_packets"][-1]["time_s"]
    gaps = []
    for t in np.arange(start, end+.0001, .02):
        if summary["condition"] == "wide_no_entry":
            xy = np.asarray(summary["bystander_stage_start_evaluator_xy_m"])
        elif summary["condition"] == "wide_buffered_static":
            xy = SIDE_START
        else:
            fraction = min(1., APPROACH_MPS*(t-start)/
                           np.linalg.norm(target-SIDE_START))
            xy = SIDE_START+(target-SIDE_START)*fraction
        response = summary["response_parameters"]
        plan = summary["response_plan"]
        alarm = summary["actor_alarm_s"]
        if response is not None and plan is not None and plan["known_free"]:
            displacement = min(RESPONSE_MAX_M, max(0.,
                response["speed_mps"]*(t-alarm-response["delay_s"])))
            ape = ape_start+[0., -displacement]
        else:
            ape = ape_start
        two.set_distractor(model, xy)
        gap = audit.distances(model, data, ape)["distractor"]
        gaps.append({"time_s": round(float(t), 5), "gap_m": round(gap, 5),
                     "ape_xy_m": ape.tolist()})
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
    report = {"schema": "robot-learning-lab.ape-owner-inference-response.v1",
              "task": "T-9v3p3n",
              "scope": "Full V589 owner turn, range-cued 25cm kinematic wait, then a 15cm turning side-person challenge during slow RGB-D item inference. Compare owner sector, all-angle image rejection, and fast/slow map-checked kinematic south responses on timed range packets. Actor receives no evaluator person pose. No selected body or qualified response skill.",
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
