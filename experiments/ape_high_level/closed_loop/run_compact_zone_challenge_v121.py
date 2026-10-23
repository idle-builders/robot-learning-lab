"""Probe finite 8x8 zone footprints during the V120 owner/prop challenge.

Five first-hit subrays approximate each zone. Nearest and median subray
selection are intentionally simple bracketing rules, not an ST sensor model.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from articulated_person_scene import add_person
from audit_shin_range_v190 import ROOM_GROUP, SELF_GROUP
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
from run_pre_show_reserve_screen_v100 import V98
import run_compact_shin_owner_watch_v119 as base
from run_compact_shin_challenge_v120 import motion_y, supported_alarm


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v121"
PROFILES = ("fast_ignored", "slow_clear")
MODES = ("nearest", "median")
CELL_WIDTH_RAD = base.FOV_RAD/8
SUBRAY_OFFSETS = ((0.,0.),(-.35,-.35),(-.35,.35),(.35,-.35),(.35,.35))


def zone_scan(room, seed: int, branch: str, index: int,
              mode: str, sensor_trial: int):
    clouds = {"two": [], "extra": []}
    counts = {"rays": 0, "self_first": 0, "floor_first": 0,
              "dropout": 0, "environment_points": 0,
              "zones_with_any_hit": 0}
    for side, body_name in ((1, "left_shin"), (-1, "right_shin")):
        bid = room.model.body(body_name).id
        rotation = room.data.xmat[bid].reshape(3,3)
        origin = room.data.xpos[bid] + rotation @ np.array(
            (.02, side*.04, -.05))
        for yaw_deg in (side*22.5, side*67.5):
            layout = "two" if abs(yaw_deg) < 30 else "extra"
            for row in range(8):
                for column in range(8):
                    hits = []
                    for delta_row, delta_col in SUBRAY_OFFSETS:
                        elevation = (base.PITCH_RAD +
                                     (row+.5-4+delta_row)*CELL_WIDTH_RAD)
                        azimuth = (math.radians(yaw_deg) +
                                   (column+.5-4+delta_col)*CELL_WIDTH_RAD)
                        local = np.array((
                            math.cos(elevation)*math.cos(azimuth),
                            math.cos(elevation)*math.sin(azimuth),
                            math.sin(elevation)))
                        direction = rotation @ local
                        scene_id = np.array([-1], dtype=np.int32)
                        self_id = np.array([-1], dtype=np.int32)
                        distance = mujoco.mj_ray(
                            room.model, room.data, origin, direction,
                            ROOM_GROUP, True, -1, scene_id)
                        self_distance = mujoco.mj_ray(
                            room.model, room.data, origin, direction,
                            SELF_GROUP, True, -1, self_id)
                        counts["rays"] += 1
                        if (self_distance >= 0 and
                                (distance < 0 or self_distance <= distance)):
                            counts["self_first"] += 1
                            hits.append((self_distance, direction, "self"))
                        elif 0 <= distance <= base.RANGE_MAX_M:
                            point = origin+distance*direction
                            kind = "floor" if point[2] < .015 else "scene"
                            if kind == "floor":
                                counts["floor_first"] += 1
                            hits.append((distance, direction, kind))
                    if not hits:
                        continue
                    counts["zones_with_any_hit"] += 1
                    ordered = sorted(hits, key=lambda hit: hit[0])
                    chosen = (ordered[0] if mode == "nearest" else
                              ordered[len(ordered)//2])
                    distance, direction, kind = chosen
                    if kind != "scene":
                        continue
                    source = (f"{seed}-{branch}-{index}-{mode}-"
                              f"{sensor_trial}-{body_name}-{yaw_deg}-"
                              f"{row}-{column}")
                    rng_seed = int.from_bytes(hashlib.sha256(
                        source.encode()).digest()[:8], "little")
                    rng = np.random.default_rng(rng_seed)
                    if rng.random() < base.DROPOUT:
                        counts["dropout"] += 1
                        continue
                    noisy = origin + max(.02, distance+rng.normal(
                        0., base.NOISE_STD_M))*direction
                    clouds[layout].append(noisy)
                    counts["environment_points"] += 1
    two = np.asarray(clouds["two"], dtype=float).reshape(-1,3)
    extra = np.asarray(clouds["extra"], dtype=float).reshape(-1,3)
    return {"two": two, "four": np.concatenate((two,extra))}, counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    original_model = structural.structural_model
    original_prop = base.prop_y
    original_scan = base.compact_scan
    original_out = base.OUT
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    rows = []
    try:
        for seed in (170,172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-00.npz")
            with np.load(source) as saved:
                qpos = saved["actor_qpos"].copy()
                estimate = saved["estimated_pose_xyyaw"].copy()
            for profile in PROFILES:
                for mode in MODES:
                    for trial in (1,2):
                        base.OUT = OUT / mode / f"trial{trial}"
                        base.prop_y = lambda branch, time_s, profile=profile: (
                            motion_y(profile, time_s))
                        base.compact_scan = (
                            lambda room, scan_seed, branch, index,
                                   mode=mode, trial=trial: zone_scan(
                                       room, scan_seed, branch, index,
                                       mode, trial))
                        row = base.run_one(seed, profile,
                                           tuple(prior["case"]), qpos, estimate)
                        row["mode"] = mode
                        row["sensor_trial"] = trial
                        row["source_packet_path"] = str(source.relative_to(ROOT))
                        row["source_packet_sha256"] = sha(source)
                        row["audited_alarms"] = {
                            layout: supported_alarm(row, layout)
                            for layout in ("two", "four")}
                        event = row["audited_alarms"]["four"][
                            "first_supported"]
                        breach = row["evaluator_first_below_5cm_s"]
                        row["supported_alarm_lead_before_5cm_s"] = (
                            None if event is None or breach is None else
                            round(breach-event["time_s"],4))
                        rows.append(row)
                        print(seed, profile, mode, trial, "alarm",
                              event["time_s"] if event else None,
                              "breach", breach, flush=True)
    finally:
        base.OUT = original_out
        base.prop_y = original_prop
        base.compact_scan = original_scan
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-compact-zone-challenge.v1",
        "source_sha256": sha(SOURCE),
        "v119_source_sha256": sha(base.SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "profiles": list(PROFILES), "zone_modes": list(MODES),
        "subray_offsets_in_zone_width": SUBRAY_OFFSETS,
        "sensor_trials": [1,2],
        "minimum_returns_each_packet": 5,
        "scope": "Same frozen ape pose and V120 fast-ignored/slow-clear scripted prop paths. Five first-hit rays per 8x8 zone, nearest or median range aggregation, 12mm range noise and 15% per-zone dropout at simulated 15Hz. Bracketing proxies only: no measured ST impulse response, target reflectivity, histograms, multi-target selection, wall-clock bus latency, selected-body actuation, or fresh RGB-D inference.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
