"""Side-entry range monitoring during V589's complete owner re-teach turn.

The released owner actor and RGB-D item model are unchanged. A bystander is
placed at the held ape's side before the shown image, then moves inward only
while slow item inference runs. The actor sees timed range packets; exact
person pose and body gap are evaluator data.
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
import run_robocasa_person_joined_v497 as joined  # noqa: E402
import run_robocasa_social_reteach_v589 as reteach  # noqa: E402
import run_robocasa_two_moving_people_v586 as moving  # noqa: E402
import screen_robocasa_two_people_v582 as two  # noqa: E402


OUT = ROOT / "artifacts/ape-owner-inference-guard"
CONDITIONS = ("owner_sector_moving", "all_angle_moving", "all_angle_static")
SIDE_START_M = .95
SIDE_TARGET_M = .30
APPROACH_MPS = .62
CHANGE_M = .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changed_all(a, b):
    return sum(((old is None) != (new is None)) or
               (old is not None and new is not None and
                abs(old-new) >= CHANGE_M)
               for old, new in zip(a, b))


def run_condition(condition, policy, scenario, prior, vision, exemplar):
    original_turn = turn.owner_turn
    original_prop = turn._prop
    original_sector = turn._sector_change
    original_scan = reteach.SCAN
    original_pool = reteach.ThreadPoolExecutor
    original_out = reteach.OUT
    original_path = moving.distractor_xy_at
    active = {"worker": False, "start_s": None, "current_s": None,
              "side_start_xy_m": None, "target_xy_m": None,
              "ape_xy_m": None, "baseline": None,
              "last_scan": None, "packets": [], "changes": [],
              "path_samples_evaluator": []}

    def staged_turn(response, model, data, renderer, actual, estimate,
                    time_s, use_depth, vision, exemplar):
        result = original_turn(response, model, data, renderer, actual, estimate,
                               time_s, use_depth, vision, exemplar)
        side = np.asarray(actual[:2], dtype=float)+[-SIDE_START_M, 0.]
        target = np.asarray(actual[:2], dtype=float)+[-SIDE_TARGET_M, 0.]
        active["side_start_xy_m"] = side.tolist()
        active["target_xy_m"] = target.tolist()
        active["ape_xy_m"] = np.asarray(actual[:2]).tolist()
        moving.distractor_xy_at = lambda _t: side.copy()
        two.set_distractor(model, side)
        mujoco.mj_forward(model, data)
        return result

    def position(t):
        start = np.asarray(active["side_start_xy_m"])
        if condition == "all_angle_static" or active["start_s"] is None:
            return start
        target = np.asarray(active["target_xy_m"])
        distance = float(np.linalg.norm(target-start))
        fraction = min(1., max(0., APPROACH_MPS*(t-active["start_s"])/distance))
        return start+(target-start)*fraction

    def prop(model, t):
        original_prop(model, t)
        if active["side_start_xy_m"] is None:
            return
        if active["worker"] and active["start_s"] is None:
            active["start_s"] = float(t)
        xy = position(float(t))
        two.set_distractor(model, xy)
        active["current_s"] = float(t)
        active["path_samples_evaluator"].append(
            {"time_s": round(float(t), 5), "bystander_xy_m": xy.round(5).tolist()})

    def scan(model, data, xy, max_range_m, rng):
        packet = original_scan(model, data, xy, max_range_m, rng)
        if active["worker"]:
            active["packets"].append({"time_s": round(active["current_s"], 5),
                                      "ranges_m": packet})
        else:
            active["last_scan"] = packet
        return packet

    def sector_change(a, b, bearing, half_width_degrees):
        original = original_sector(a, b, bearing, half_width_degrees)
        all_beams = changed_all(a, b)
        if active["worker"]:
            active["changes"].append({"time_s": round(active["current_s"], 5),
                                      "owner_sector_changed": original,
                                      "all_angle_changed": all_beams})
            if condition.startswith("all_angle"):
                return all_beams
        return original

    class GuardPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            active["worker"] = True
            active["baseline"] = list(active["last_scan"])
            return super().submit(*args, **kwargs)

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                active["worker"] = False

    try:
        turn.owner_turn = staged_turn
        turn._prop = prop
        turn._sector_change = sector_change
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
    guard_row = next((row for row in ep["trace"]
                      if row["stage"] == "new_show_inference_guard"), None)
    summary = {"condition": condition, "outcome": ep["outcome"],
               "route_state": ep["route_state"],
               "old_owner_identity_restored": "owner_confirmed" in ep["outcome"],
               "stale_image_discarded": any(row["stage"] ==
                   "new_show_stale_image_discarded" for row in ep["trace"]),
               "new_metric_item_seen": any(row["stage"] ==
                   "new_show_metric_item_seen" for row in ep["trace"]),
               "guard_scan_count": guard_row["scan_count"] if guard_row else None,
               "guard_alarm_time_s": guard_row["alarm_time_s"] if guard_row else None,
               "worker_wall_s": guard_row["worker_wall_s"] if guard_row else None,
               "moving_start_s": active["start_s"],
               "side_start_xy_evaluator_m": active["side_start_xy_m"],
               "side_target_xy_evaluator_m": active["target_xy_m"],
               "ape_xy_evaluator_m": active["ape_xy_m"],
               "baseline_range_packet": active["baseline"],
               "range_packets": active["packets"],
               "range_change_comparison": active["changes"],
               "path_samples_evaluator": active["path_samples_evaluator"],
               "video": ep["video"]}
    return result, summary


def score_gap(summary):
    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    ape = np.asarray(summary["ape_xy_evaluator_m"])
    start = np.asarray(summary["side_start_xy_evaluator_m"])
    target = np.asarray(summary["side_target_xy_evaluator_m"])
    start_s = summary["moving_start_s"]
    duration = summary["range_packets"][-1]["time_s"]-start_s
    rows = []
    for dt in np.arange(0., duration+.0001, .02):
        t = start_s+float(dt)
        if summary["condition"] == "all_angle_static":
            xy = start
        else:
            distance = float(np.linalg.norm(target-start))
            xy = start+(target-start)*min(1., APPROACH_MPS*dt/distance)
        two.set_distractor(model, xy)
        mujoco.mj_forward(model, data)
        gap = audit.distances(model, data, ape)["distractor"]
        rows.append({"time_s": round(t, 5), "gap_m": round(gap, 5)})
    minimum = min(row["gap_m"] for row in rows)
    first_low = next((row["time_s"] for row in rows if row["gap_m"] < .05), None)
    return {"sample_period_s": .02, "min_bystander_ape_gap_m": minimum,
            "first_below_5cm_s": first_low,
            "overlap": minimum <= 0,
            "samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, action="append")
    args = parser.parse_args()
    conditions = tuple(args.condition) if args.condition else CONDITIONS
    policy = json.loads(joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    results = []
    summaries = []
    for condition in conditions:
        result, summary = run_condition(condition, policy, scenario, prior,
                                        vision, exemplar)
        summary["geometry_evaluator"] = score_gap(summary)
        results.append(result)
        summaries.append(summary)
        print(condition, summary["outcome"], summary["guard_alarm_time_s"],
              summary["geometry_evaluator"]["min_bystander_ape_gap_m"],
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-inference-side-entry.v1",
              "task": "T-p2sgs7",
              "scope": "Complete V589 owner re-teach episodes; scripted side capsule enters only during slow image inference. Original owner-sector versus all-angle change check over the same ideal 30cm 360-degree range packets, plus static control. Actor uses no evaluator person IDs or positions. Kinematic ape proxy, scripted people, ideal odometry and no physical response.",
              "source_sha256": sha(Path(__file__)),
              "base_runner_sha256": sha(Path(reteach.__file__)),
              "two_person_runner_sha256": sha(Path(moving.__file__)),
              "scene_sha256": sha(moving.SCENE_XML),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "conditions": conditions,
              "summaries": summaries,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
