"""Stress the frozen owner-camera low-depth cue with illustrative faults.

All conditions replay the same saved scene frames and actor packet interface.
The fault magnitudes are development probes, not measured camera statistics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run import ROOT, sha
import screen_standoff_camera_depth_v171p as v171p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-camera-depth-stress"
DRAWS = 5
CONDITIONS = {
    "baseline": (0., 0., 0., 0., 0.),
    "moderate": (.25, .005, .005, 2., 0.),
    "severe": (.50, .010, .015, 5., 0.),
    "moderate_contiguous_40pct": (.25, .005, .005, 2., .40),
}


def rotation_error(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = map(math.radians, (yaw_deg, pitch_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array(((cy, sy*sp, sy*cp),
                     (0., cp, -sp),
                     (-sy, cy*sp, cy*cp)))


def low_depth(packet, rng: np.random.Generator,
              condition: tuple[float, float, float, float, float],
              mount_offset: np.ndarray, mount_error: np.ndarray,
              block_start: float) -> tuple[int, float | None]:
    dropout, noise_m, _, _, block_width = condition
    sampled = packet["sampled_depth_m"].astype(float)
    height, width = map(int, packet["full_shape_hw"])
    rows, cols = np.mgrid[v171p.DEPTH_OFFSET:height:v171p.DEPTH_STRIDE,
                          v171p.DEPTH_OFFSET:width:v171p.DEPTH_STRIDE]
    valid = np.isfinite(sampled) & (sampled > .08) & (
        sampled < v171p.MAX_DEPTH_M)
    if dropout:
        valid &= rng.random(sampled.shape) >= dropout
    if block_width:
        normalized_column = cols / width
        valid &= ~((normalized_column >= block_start) &
                   (normalized_column < block_start+block_width))
    depth = sampled[valid]
    if noise_m:
        depth += rng.normal(0., noise_m, len(depth))
    valid_depth = (depth > .08) & (depth < v171p.MAX_DEPTH_M)
    depth = depth[valid_depth]
    focal = height/(2*math.tan(math.radians(float(packet["fovy_deg"]))/2))
    ray = np.stack(((cols[valid]-width/2)/focal,
                    -(rows[valid]-height/2)/focal,
                    -np.ones(np.count_nonzero(valid))), axis=-1)[valid_depth]
    original_rotation = packet["camera_rotation"]
    rotation = original_rotation @ mount_error
    origin = packet["camera_origin_xyz_m"] + (
        original_rotation @ mount_offset)
    xyz = origin+(ray*depth[:, None])@rotation.T
    estimate = packet["estimated_pose_xyyaw"]
    delta = xyz[:, :2]-estimate[:2]
    c, s = math.cos(float(estimate[2])), math.sin(float(estimate[2]))
    x = c*delta[:, 0]+s*delta[:, 1]
    y = -s*delta[:, 0]+c*delta[:, 1]
    z = xyz[:, 2]
    low = x[(x > .08) & (x < .50) &
            (y > .05) & (y < .50) &
            (z > .015) & (z < .10)]
    p10 = (round(float(np.quantile(low, .1)), 5)
           if len(low) >= v171p.MIN_SUPPORT else None)
    return len(low), p10


def scan(row: dict, condition_name: str,
         draw: int) -> dict:
    condition = CONDITIONS[condition_name]
    dropout, noise_m, offset_m, rotation_deg, block_width = condition
    camera_index = v171p.CAMERAS.index(row["camera"])
    case_id = row["source_case_id"]
    stable_seed = sum((index+1)*ord(char)
                      for index, char in enumerate(case_id))
    seed = 179000 + stable_seed*7 + camera_index*1009 + draw*83 + (
        list(CONDITIONS).index(condition_name)*1297)
    rng = np.random.default_rng(seed)
    mount_offset = rng.uniform(-offset_m, offset_m, 3)
    angles = rng.uniform(-rotation_deg, rotation_deg, 2)
    mount_error = rotation_error(*angles)
    block_start = float(rng.choice((0., 1.-block_width)))
    samples = []
    for saved in row["samples"]:
        path = ROOT / saved["depth_packet_path"]
        if sha(path) != saved["depth_packet_sha256"]:
            raise AssertionError("Saved actor depth packet changed")
        with np.load(path, allow_pickle=False) as packet:
            support, p10 = low_depth(packet, rng, condition,
                                     mount_offset, mount_error,
                                     block_start)
            time_s = float(packet["capture_time_s"])
        if time_s != saved["capture_time_s"]:
            raise AssertionError("Saved actor depth time changed")
        if condition_name == "baseline" and (
            support != saved["front_left_low_depth_support"] or
            p10 != saved["front_left_low_p10_x_m"]):
            raise AssertionError("Baseline depth statistic does not reproduce")
        samples.append({"time_s": time_s, "low_support": support,
                        "p10_x_m": p10, "closing_0p2s_m": None,
                        "cue_now": False})
    for index in range(2, len(samples)):
        older = samples[index-2]["p10_x_m"]
        current = samples[index]["p10_x_m"]
        if older is not None and current is not None:
            delta = round(older-current, 5)
            samples[index]["closing_0p2s_m"] = delta
            samples[index]["cue_now"] = (
                delta >= v171p.MIN_CLOSING_M_PER_0P2S)
    first = next((s["time_s"] for s in samples if s["cue_now"]), None)
    breach = row["evaluator_first_below_5cm_s"]
    return {
        "source_case_id": case_id,
        "class": row["class"],
        "camera": row["camera"],
        "condition": condition_name,
        "draw": draw,
        "seed": seed,
        "mount_offset_camera_xyz_m": mount_offset.tolist(),
        "mount_yaw_pitch_error_deg": angles.tolist(),
        "blocked_column_fraction_start": block_start if block_width else None,
        "first_cue_s": first,
        "evaluator_first_below_5cm_s": breach,
        "timely_hazard_warning": (
            row["class"] == "hazard" and first is not None and
            breach is not None and first <= breach),
        "false_control_warning": (
            row["class"] == "control" and first is not None),
        "minimum_support_17p2_to_18p2": min(
            s["low_support"] for s in samples
            if 17.2-1e-6 <= s["time_s"] <= 18.2+1e-6),
        "samples": samples,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = v171p.OUT / "report.json"
    source = json.loads(source_file.read_text())
    rows = []
    for row in source["rows"]:
        for name in CONDITIONS:
            for draw in range(1 if name == "baseline" else DRAWS):
                rows.append(scan(row, name, draw))
        print(row["source_case_id"], row["camera"], flush=True)
    summary = {}
    for name in CONDITIONS:
        summary[name] = {}
        for camera in v171p.CAMERAS:
            group = [r for r in rows if r["condition"] == name and
                     r["camera"] == camera]
            hazard = [r for r in group if r["class"] == "hazard"]
            control = [r for r in group if r["class"] == "control"]
            summary[name][camera] = {
                "hazard_trials": len(hazard),
                "timely_hazard_warnings": sum(
                    r["timely_hazard_warning"] for r in hazard),
                "control_trials": len(control),
                "false_control_warnings": sum(
                    r["false_control_warning"] for r in control),
                "unsupported_anytime_17p2_to_18p2": sum(
                    r["minimum_support_17p2_to_18p2"] < v171p.MIN_SUPPORT
                    for r in group),
            }
        paired = {}
        for row in rows:
            if row["condition"] == name:
                paired.setdefault((row["source_case_id"], row["draw"]),
                                  {})[row["camera"]] = row
        hazard_pairs = [pair for pair in paired.values()
                        if next(iter(pair.values()))["class"] == "hazard"]
        control_pairs = [pair for pair in paired.values()
                         if next(iter(pair.values()))["class"] == "control"]
        summary[name]["either_camera_cue"] = {
            "hazard_trials": len(hazard_pairs),
            "timely_hazard_warnings": sum(
                any(r["timely_hazard_warning"] for r in pair.values())
                for pair in hazard_pairs),
            "control_trials": len(control_pairs),
            "false_control_warnings": sum(
                any(r["false_control_warning"] for r in pair.values())
                for pair in control_pairs),
        }
    report = {
        "schema": "robot-learning-lab.ape-camera-depth-stress.v1",
        "source_sha256": sha(SOURCE),
        "v171_report_sha256": sha(source_file),
        "draws_per_fault_condition_per_case": DRAWS,
        "conditions": {name: {
            "independent_sample_dropout": values[0],
            "depth_noise_sigma_m": values[1],
            "static_mount_translation_half_width_m": values[2],
            "static_mount_yaw_pitch_half_width_deg": values[3],
            "contiguous_image_column_block_fraction": values[4],
        } for name, values in CONDITIONS.items()},
        "scope": "Frozen V171p cue on repeated perturbations of the same 16 "
                 "saved scene trajectories. Synthetic point dropout, depth "
                 "noise, fixed camera extrinsic error and contiguous column "
                 "loss are illustrative, not calibrated optics or new rooms. "
                 "The actor never sees evaluator breach time; it scores cue "
                 "timing only. No action or recovery is simulated here.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
