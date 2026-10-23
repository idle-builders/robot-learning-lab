"""Stress the frozen front-left cue on changed-scene and control packets.

This is a deterministic sensor-packet replay, not another owner simulation.
It compares prompt-warning coverage and false cues under point dropout, metric
noise and a partial field occlusion without retuning the V149p cue.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
import run_early_standoff_owner_v148p as v148p
import run_standoff_scene_variants_v160p as v160p
from screen_standoff_front_cue_v149p import (
    MIN_CLOSING_M_PER_0P2S, front_left_stat)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-standoff-range-stress"
PROFILES = {
    "saved": {"dropout": 0., "xyz_sigma_m": 0.,
              "half_front_left_occluded": False, "draws": 1},
    "moderate": {"dropout": .25, "xyz_sigma_m": .005,
                 "half_front_left_occluded": False, "draws": 5},
    "severe": {"dropout": .50, "xyz_sigma_m": .010,
               "half_front_left_occluded": False, "draws": 5},
    "half_view": {"dropout": .25, "xyz_sigma_m": .005,
                  "half_front_left_occluded": True, "draws": 5},
}


def rng_for(*parts: object) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(
        "|".join(str(part) for part in parts).encode()).digest()[:8],
        "little")
    return np.random.default_rng(seed)


def altered_points(points: np.ndarray, pose: np.ndarray,
                   profile: dict, rng: np.random.Generator) -> np.ndarray:
    kept = points.copy()
    if profile["half_front_left_occluded"]:
        delta = kept[:, :2]-pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        x = c*delta[:, 0]+s*delta[:, 1]
        y = -s*delta[:, 0]+c*delta[:, 1]
        z = kept[:, 2]
        blocked = ((x > .08) & (x < .50) &
                   (y > .05) & (y < .275) &
                   (z > .015) & (z < .10))
        kept = kept[~blocked]
    dropout = profile["dropout"]
    if dropout:
        kept = kept[rng.random(len(kept)) >= dropout]
    sigma = profile["xyz_sigma_m"]
    if sigma:
        kept = kept+rng.normal(0., sigma, size=kept.shape)
    return kept


def read_case(case: dict) -> list[dict]:
    monitor_file = ROOT / case["monitor_path"]
    if sha(monitor_file) != case["monitor_sha256"]:
        raise AssertionError("Source monitor changed")
    monitor = json.loads(monitor_file.read_text())
    rows = []
    for frame in monitor["frames"]:
        packet_file = ROOT / frame["packet_path"]
        if sha(packet_file) != frame["packet_sha256"]:
            raise AssertionError("Source range packet changed")
        with np.load(packet_file, allow_pickle=False) as packet:
            rows.append({
                "time_s": frame["capture_time_s"],
                "points": packet["front_four_points_xyz_m"].copy(),
                "pose": packet["estimated_pose_xyyaw"].copy(),
            })
    return rows


def evaluate(case: dict, packets: list[dict], profile_name: str,
             profile: dict, draw: int) -> dict:
    samples = []
    for index, packet in enumerate(packets):
        rng = rng_for(case["id"], profile_name, draw, index)
        points = altered_points(packet["points"], packet["pose"],
                                profile, rng)
        support, p10 = front_left_stat(points, packet["pose"])
        sample = {"time_s": packet["time_s"],
                  "front_left_support": support,
                  "front_left_p10_x_m": p10,
                  "closing_0p2s_m": None,
                  "cue_now": False}
        if index >= 2 and p10 is not None:
            older = samples[-2]["front_left_p10_x_m"]
            if older is not None:
                closing = round(older-p10, 5)
                sample["closing_0p2s_m"] = closing
                sample["cue_now"] = (
                    closing >= MIN_CLOSING_M_PER_0P2S)
        samples.append(sample)
    first = next((s["time_s"] for s in samples if s["cue_now"]), None)
    breach = case["first_below_5cm_s"]
    return {
        "source_case_id": case["id"],
        "class": case["class"],
        "profile": profile_name,
        "draw": draw,
        "first_front_left_cue_s": first,
        "evaluator_first_below_5cm_s": breach,
        "cue_lead_to_5cm_s_evaluator_only": (
            round(breach-first, 4)
            if breach is not None and first is not None else None),
        "timely_hazard_warning": (case["class"] == "hazard" and
                                  first is not None and first <= breach),
        "false_control_warning": (case["class"] == "control" and
                                  first is not None),
        "supported_packet_fraction_16p9_to_18p3": round(
            sum(s["front_left_support"] >= 5 for s in samples
                if 16.9-1e-6 <= s["time_s"] <= 18.3+1e-6) /
            sum(16.9-1e-6 <= s["time_s"] <= 18.3+1e-6
                for s in samples), 4),
        "samples": samples,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants_path = v160p.OUT / "report.json"
    controls_path = v148p.OUT / "report.json"
    variants = json.loads(variants_path.read_text())
    controls = json.loads(controls_path.read_text())
    source_cases = []
    for row in variants["rows"]:
        if row["scripted_prop_stop_delay_s"] is not None:
            continue
        monitor = row["monitor"]
        source_cases.append({
            "id": f"hazard-{row['variant']}-seed{row['seed']}"
                  f"-x{abs(row['station_x_m']):.2f}",
            "class": "hazard",
            "variant": row["variant"],
            "seed": row["seed"],
            "station_x_m": row["station_x_m"],
            "monitor_path": monitor["path"],
            "monitor_sha256": monitor["sha256"],
            "video_path": row["video_path"],
            "video_sha256": row["video_sha256"],
            "first_below_5cm_s": row["evaluator_first_below_5cm_s"],
        })
    for row in controls["rows"]:
        if row["profile"] not in ("moving_prop", "hand_close"):
            continue
        turn = row["owner_turn"]
        monitor = turn["monitor"]
        source_cases.append({
            "id": f"control-{row['profile']}-seed{row['seed']}"
                  f"-x{abs(row['station_x_m']):.2f}",
            "class": "control",
            "variant": row["profile"],
            "seed": row["seed"],
            "station_x_m": row["station_x_m"],
            "monitor_path": monitor["path"],
            "monitor_sha256": monitor["sha256"],
            "video_path": turn["video_path"],
            "video_sha256": turn["video_sha256"],
            "first_below_5cm_s": turn["evaluator_first_below_5cm_s"],
        })
    if len(source_cases) != 16:
        raise AssertionError("Source case matrix incomplete")
    rows = []
    for case in source_cases:
        packets = read_case(case)
        for profile_name, profile in PROFILES.items():
            for draw in range(profile["draws"]):
                rows.append(evaluate(case, packets, profile_name,
                                     profile, draw))
    summary = {}
    for name in PROFILES:
        group = [row for row in rows if row["profile"] == name]
        hazards = [row for row in group if row["class"] == "hazard"]
        controls = [row for row in group if row["class"] == "control"]
        summary[name] = {
            "hazard_trials": len(hazards),
            "timely_hazard_warnings": sum(
                row["timely_hazard_warning"] for row in hazards),
            "controls": len(controls),
            "false_control_warnings": sum(
                row["false_control_warning"] for row in controls),
            "minimum_timely_lead_s": min((
                row["cue_lead_to_5cm_s_evaluator_only"] for row in hazards
                if row["timely_hazard_warning"]), default=None),
            "mean_supported_packet_fraction": round(float(np.mean([
                row["supported_packet_fraction_16p9_to_18p3"]
                for row in group])), 4),
        }
    report = {
        "schema": "robot-learning-lab.ape-standoff-range-packet-stress.v1",
        "source_sha256": sha(SOURCE),
        "v160_report_sha256": sha(variants_path),
        "v148_report_sha256": sha(controls_path),
        "frozen_front_cue_threshold_m_per_0p2s":
            MIN_CLOSING_M_PER_0P2S,
        "profiles": PROFILES,
        "scope": "Offline deterministic perturbation of saved actor point "
                 "packets from eight changed-scene hazard turns and eight "
                 "ordinary/static-block controls. Gaussian point error, "
                 "independent return dropout and partial front-left ROI "
                 "occlusion are illustrative stresses, not device optics. "
                 "The existing V149 front-left threshold is unchanged. "
                 "Evaluator 5cm crossing scores warning lead only; no "
                 "response, owner turn or image worker is rerun here. "
                 "The source cases share one historical body and room.",
        "source_cases": source_cases,
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary)
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
