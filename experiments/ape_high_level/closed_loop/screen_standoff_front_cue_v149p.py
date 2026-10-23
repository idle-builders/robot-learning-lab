"""Replay an extra front-left closing cue on V148p's actor range packets.

The low shin sensors are unchanged. This is a revealed-development detector
screen, not an online V148p controller or a physical response qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
import run_early_standoff_owner_v148p as v148p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-standoff-front-cue"
PRIOR = v148p.OUT / "report.json"
MIN_SUPPORT = 5
MIN_CLOSING_M_PER_0P2S = .018


def front_left_stat(points: np.ndarray,
                    pose: np.ndarray) -> tuple[int, float | None]:
    delta = points[:, :2] - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    x = c*delta[:, 0] + s*delta[:, 1]
    y = -s*delta[:, 0] + c*delta[:, 1]
    height = points[:, 2]
    selected = x[(x > .08) & (x < .50) &
                 (y > .05) & (y < .50) &
                 (height > .015) & (height < .10)]
    return len(selected), (round(float(np.quantile(selected, .1)), 5)
                           if len(selected) >= MIN_SUPPORT else None)


def front_left_p10_x(packet: Path) -> tuple[int, float | None]:
    with np.load(packet, allow_pickle=False) as saved:
        return front_left_stat(saved["front_four_points_xyz_m"].astype(float),
                               saved["estimated_pose_xyyaw"].astype(float))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR.read_text())
    rows = []
    for entry in prior["rows"]:
        turn = entry["owner_turn"]
        monitor_path = ROOT / turn["monitor"]["path"]
        if sha(monitor_path) != turn["monitor"]["sha256"]:
            raise AssertionError("Source monitor changed")
        monitor = json.loads(monitor_path.read_text())
        samples = []
        for frame in monitor["frames"]:
            packet_path = ROOT / frame["packet_path"]
            if sha(packet_path) != frame["packet_sha256"]:
                raise AssertionError("Actor range packet changed")
            support, p10 = front_left_p10_x(packet_path)
            samples.append({
                "capture_time_s": frame["capture_time_s"],
                "source_packet": frame["packet_path"],
                "source_packet_sha256": frame["packet_sha256"],
                "front_left_support": support,
                "front_left_p10_x_m": p10,
                "closing_0p2s_m": None,
                "cue_now": False,
            })
            if len(samples) < 3:
                continue
            older = samples[-3]
            if older["front_left_p10_x_m"] is None or p10 is None:
                continue
            closing = round(older["front_left_p10_x_m"] - p10, 5)
            samples[-1]["closing_0p2s_m"] = closing
            samples[-1]["cue_now"] = closing >= MIN_CLOSING_M_PER_0P2S
        first = next((s for s in samples if s["cue_now"]), None)
        first_time = first["capture_time_s"] if first else None
        old_alarm = turn["monitor"]["alarm"]
        breach = turn["evaluator_first_below_5cm_s"]
        rows.append({
            "seed": entry["seed"],
            "station_x_m": entry["station_x_m"],
            "profile": entry["profile"],
            "source_monitor": turn["monitor"]["path"],
            "source_monitor_sha256": turn["monitor"]["sha256"],
            "first_front_left_cue_s": first_time,
            "original_side_alarm_s": (
                old_alarm["capture_time_s"] if old_alarm else None),
            "evaluator_first_below_5cm_s": breach,
            "front_cue_to_5cm_lead_s_evaluator_only": (
                round(breach-first_time, 4)
                if first_time is not None and breach is not None else None),
            "original_side_alarm_to_5cm_lead_s_evaluator_only": (
                round(breach-old_alarm["capture_time_s"], 4)
                if old_alarm is not None and breach is not None else None),
            "evaluator_minimum_scene_gap_m": turn[
                "evaluator_minimum_scene_gap_m"],
            "first_image_capture_s": turn["first_image"]["capture_time_s"],
            "samples": samples,
        })
        print(entry["seed"], entry["station_x_m"], entry["profile"],
              "front", first_time,
              "side", old_alarm["capture_time_s"] if old_alarm else None,
              "5cm", breach, flush=True)
    report = {
        "schema": "robot-learning-lab.ape-standoff-front-cue.v1",
        "source_sha256": sha(SOURCE),
        "v148_report_sha256": sha(PRIOR),
        "sensor_input": "unchanged V148 four-front/side 10Hz shin-range point packets",
        "actor_rule": {
            "roi_robot_local_m": "0.08<x<0.50, 0.05<y<0.50, 0.015<z<0.10",
            "statistic": "10th percentile forward x over at least five returns",
            "comparison_interval_s": .2,
            "closing_threshold_m": MIN_CLOSING_M_PER_0P2S,
        },
        "scope": "Read-only front-left closing cue on saved actor packets. "
                 "Evaluator 5cm crossing is used only for scoring. "
                 "Parameters were selected after seeing V148's miss and "
                 "are not independently validated; seed172 changes sensor "
                 "draw/lighting but shares geometry and scripted paths. "
                 "No owner turn is rerun with this cue, no response executes, "
                 "and no physical sensor count is selected.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
