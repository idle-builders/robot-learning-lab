"""Compare saved two-front, four-front and rear-added range layouts.

The V190 scanner concatenates the two forward shin views before the two
angled views. Its per-packet counts let this screen separate actual saved
returns without inventing points. The rear-right array is a negative control
for this front-left threat. No optical or package qualification follows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
from screen_standoff_front_cue_v149p import (
    MIN_CLOSING_M_PER_0P2S, front_left_stat)
import stress_standoff_range_packets_v162p as v162p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-range-layout-comparison"
LAYOUTS = ("two_forward", "four_front_side", "four_plus_right_rear")
STRESSES = {
    "saved": {"dropout": 0., "xyz_sigma_m": 0.,
              "forward_pair_front_left_occluded": False, "draws": 1},
    "moderate": {"dropout": .25, "xyz_sigma_m": .005,
                 "forward_pair_front_left_occluded": False, "draws": 5},
    "forward_pair_occluded": {"dropout": 0., "xyz_sigma_m": 0.,
                              "forward_pair_front_left_occluded": True,
                              "draws": 1},
}


def front_left_roi(points: np.ndarray,
                   pose: np.ndarray) -> np.ndarray:
    delta = points[:, :2]-pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    x = c*delta[:, 0]+s*delta[:, 1]
    y = -s*delta[:, 0]+c*delta[:, 1]
    z = points[:, 2]
    return ((x > .08) & (x < .50) &
            (y > .05) & (y < .50) &
            (z > .015) & (z < .10))


def read_case(case: dict) -> list[dict]:
    monitor_file = ROOT / case["monitor_path"]
    if sha(monitor_file) != case["monitor_sha256"]:
        raise AssertionError("Source monitor changed")
    monitor = json.loads(monitor_file.read_text())
    packets = []
    for frame in monitor["frames"]:
        path = ROOT / frame["packet_path"]
        if sha(path) != frame["packet_sha256"]:
            raise AssertionError("Source packet changed")
        with np.load(path, allow_pickle=False) as packet:
            front = packet["front_four_points_xyz_m"].copy()
            rear = packet["right_rear_points_xyz_m"].copy()
            pose = packet["estimated_pose_xyyaw"].copy()
        forward_count = frame["front_counts"].get(
            "two_environment_points", 0)
        side_count = frame["front_counts"].get(
            "four_environment_points", 0)
        if len(front) != forward_count+side_count:
            raise AssertionError("Cannot separate front sensor sources")
        packets.append({"time_s": frame["capture_time_s"],
                        "front": front, "rear": rear,
                        "forward_count": forward_count,
                        "pose": pose})
    return packets


def perturb(packet: dict, case_id: str, stress_name: str,
            stress: dict, draw: int, index: int) -> dict:
    front = packet["front"]
    rear = packet["rear"]
    n_forward = packet["forward_count"]
    pose = packet["pose"]
    front_ids = np.arange(len(front))
    if stress["forward_pair_front_left_occluded"]:
        blocked = front_left_roi(front, pose) & (front_ids < n_forward)
        front_ids = front_ids[~blocked]
    rng = v162p.rng_for(case_id, stress_name, draw, index)
    if stress["dropout"]:
        front_ids = front_ids[
            rng.random(len(front_ids)) >= stress["dropout"]]
    front_points = front[front_ids].copy()
    rear_points = rear.copy()
    if stress["dropout"]:
        rear_points = rear_points[
            rng.random(len(rear_points)) >= stress["dropout"]]
    if stress["xyz_sigma_m"]:
        front_points += rng.normal(
            0., stress["xyz_sigma_m"], size=front_points.shape)
        rear_points += rng.normal(
            0., stress["xyz_sigma_m"], size=rear_points.shape)
    forward = front_points[front_ids < n_forward]
    return {"two_forward": forward,
            "four_front_side": front_points,
            "four_plus_right_rear": np.concatenate(
                (front_points, rear_points), axis=0)}


def evaluate(case: dict, packets: list[dict], stress_name: str,
             stress: dict, draw: int) -> list[dict]:
    samples = {layout: [] for layout in LAYOUTS}
    for index, packet in enumerate(packets):
        views = perturb(packet, case["id"], stress_name, stress,
                        draw, index)
        for layout in LAYOUTS:
            support, p10 = front_left_stat(views[layout], packet["pose"])
            prior = samples[layout]
            closing = None
            cue = False
            if index >= 2 and p10 is not None:
                older = prior[-2]["front_left_p10_x_m"]
                if older is not None:
                    closing = round(older-p10, 5)
                    cue = closing >= MIN_CLOSING_M_PER_0P2S
            prior.append({"time_s": packet["time_s"],
                          "front_left_support": support,
                          "front_left_p10_x_m": p10,
                          "closing_0p2s_m": closing,
                          "cue_now": cue})
    rows = []
    for layout, sequence in samples.items():
        first = next((s["time_s"] for s in sequence if s["cue_now"]), None)
        breach = case["first_below_5cm_s"]
        support_pre = sum(s["front_left_support"] >= 5
                          for s in sequence[:9])
        rows.append({
            "source_case_id": case["id"],
            "class": case["class"],
            "stress": stress_name,
            "draw": draw,
            "layout": layout,
            "first_front_left_cue_s": first,
            "evaluator_first_below_5cm_s": breach,
            "cue_lead_to_5cm_s_evaluator_only": (
                round(breach-first, 4)
                if breach is not None and first is not None else None),
            "timely_hazard_warning": (case["class"] == "hazard" and
                                      first is not None and first <= breach),
            "false_control_warning": (case["class"] == "control" and
                                      first is not None),
            "supported_pre_show_packets_16p2_to_17p0": support_pre,
            "illustrative_admission_5_of_9": support_pre >= 5,
            "samples": sequence,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior_file = v162p.OUT / "report.json"
    prior = json.loads(prior_file.read_text())
    source_cases = prior["source_cases"]
    if len(source_cases) != 16:
        raise AssertionError("Source case matrix incomplete")
    rows = []
    for case in source_cases:
        packets = read_case(case)
        for stress_name, stress in STRESSES.items():
            for draw in range(stress["draws"]):
                rows.extend(evaluate(case, packets, stress_name,
                                     stress, draw))
    summary = {}
    for stress in STRESSES:
        summary[stress] = {}
        for layout in LAYOUTS:
            group = [row for row in rows if row["stress"] == stress and
                     row["layout"] == layout]
            hazards = [row for row in group if row["class"] == "hazard"]
            controls = [row for row in group if row["class"] == "control"]
            summary[stress][layout] = {
                "hazard_trials": len(hazards),
                "timely_hazard_warnings": sum(
                    row["timely_hazard_warning"] for row in hazards),
                "controls": len(controls),
                "false_control_warnings": sum(
                    row["false_control_warning"] for row in controls),
                "illustrative_pre_show_admissions": sum(
                    row["illustrative_admission_5_of_9"] for row in group),
                "minimum_timely_lead_s": min((
                    row["cue_lead_to_5cm_s_evaluator_only"]
                    for row in hazards if row["timely_hazard_warning"]),
                    default=None),
            }
    report = {
        "schema": "robot-learning-lab.ape-standoff-range-layouts.v1",
        "source_sha256": sha(SOURCE),
        "v162_report_sha256": sha(prior_file),
        "source_cases": source_cases,
        "layouts": LAYOUTS,
        "stresses": STRESSES,
        "frozen_front_cue_threshold_m_per_0p2s":
            MIN_CLOSING_M_PER_0P2S,
        "scope": "Exact saved two forward shin views versus their four-view "
                 "front+angled union; adding the saved right-rear array is "
                 "a negative control for a front-left hazard. Counts split "
                 "the first-hit returns by source. Matched packet stresses "
                 "are illustrative, and the forward-pair occlusion removes "
                 "only its relevant front-left returns while preserving "
                 "angled views. The 5-of-9 pre-show support screen is "
                 "post-hoc and informational, not an authorized actor gate. "
                 "No owner turn, response or image worker is rerun. The "
                 "sensor mounts remain virtual on a rejected historical body.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
