"""Source-check fresh full episodes and extract native-unit basic-motion facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261016/basic-sweep-posture-locomotion/method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(case: dict, folder: Path, method: dict) -> dict:
    stem = folder / case["name"]
    paths = {suffix: stem.with_suffix("." + suffix) for suffix in ("json", "npz", "mp4", "log")}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["json"].read_text())
    if report["body_sha256"] != method["sources"]["body"]["sha256"] or report["seed"] != case["seed"]:
        raise ValueError(f"native body or seed differs: {case['name']}")
    source = {suffix: {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha(path)}
              for suffix, path in paths.items()}
    with np.load(paths["npz"], allow_pickle=False) as trace:
        n = len(trace["raw_action"])
        if not math.isclose(n * 0.02, report.get("duration_s", report.get("worlds", [{}])[0].get("duration_s", -1)), abs_tol=0.021):
            raise ValueError(f"trace clock/report duration differs: {case['name']}")
        if case["kind"] in ("stand", "squat", "seated"):
            if n > case["duration_s"] * 50 or trace["raw_action"].shape[1] != 21:
                raise ValueError(f"posture action trace differs: {case['name']}")
        else:
            if n > 200 or trace["raw_action"].shape[1:] != (1, 10):
                raise ValueError(f"locomotion action trace differs: {case['name']}")
            if not np.isfinite(trace["root_pose"]).all():
                raise ValueError(f"nonfinite trajectory: {case['name']}")
    duration = n * 0.02
    result = {"case": case["name"], "kind": case["kind"], "seed": case["seed"],
              "requested_s": case["duration_s"], "observed_s": round(duration, 3),
              "observed_control_steps": n,
              "requested_horizon_coverage": round(duration / case["duration_s"], 4),
              "source": source}
    if case["kind"] in ("stand", "squat", "seated"):
        rows = report["timeseries"]
        if len(rows) != n:
            raise ValueError(f"posture timeseries differs: {case['name']}")
        z = np.asarray([row["pelvis_z_m"] for row in rows], float)
        support = np.asarray([all(row["feet_grounded"]) for row in rows], bool)
        duty = np.asarray([row["leg_torque_saturated"] for row in rows], bool)
        result["facts"] = {
            "pelvis_z_min_m": round(float(z.min()), 4),
            "pelvis_z_max_m": round(float(z.max()), 4),
            "pelvis_z_final_m": round(float(z[-1]), 4),
            "pelvis_vertical_range_mm": round(float((z.max() - z.min()) * 1000), 1),
            "two_sole_time_s": round(float(support.sum()) * 0.02, 3),
            "rated_leg_time_s": round(float(duty.sum()) * 0.02, 3),
            "hand_force_peak_n": round(float(max(row["hand_force_n"] for row in rows)), 3),
            "nonsole_floor_contact_s": round(sum(bool(row["floor_nonsole_body_contact"]) for row in rows) * 0.02, 3),
            "tilt_peak_deg": round(float(max(row["tilt_deg"] for row in rows)), 2),
            "ended_early": report["ended_early"],
        }
    else:
        world = report["worlds"][0]
        with np.load(paths["npz"], allow_pickle=False) as trace:
            pose = trace["root_pose"][:, 0]
            world_velocity = trace["root_velocity"][:, 0]
            native_command = trace["command"][:, 0]
            w, x, y, zq = (pose[:, i] for i in (3, 4, 5, 6))
            yaw = np.unwrap(np.arctan2(2 * (w * zq + x * y),
                                       1 - 2 * (y * y + zq * zq)))
            vx_body = np.cos(yaw) * world_velocity[:, 0] + np.sin(yaw) * world_velocity[:, 1]
            vy_body = -np.sin(yaw) * world_velocity[:, 0] + np.cos(yaw) * world_velocity[:, 1]
            if n > 1:
                yaw_rate = np.diff(yaw) / 0.02
                command_error = np.column_stack((native_command[1:, 0] - vx_body[1:],
                                                  native_command[1:, 1] - vy_body[1:],
                                                  native_command[1:, 2] - yaw_rate))
                quantiles = {}
                for axis, units in ((0, "x_m_s"), (1, "y_m_s"), (2, "yaw_rad_s")):
                    values = np.abs(command_error[:, axis])
                    quantiles[units] = {"n": len(values),
                                        "median": round(float(np.median(values)), 4),
                                        "p10": round(float(np.quantile(values, 0.1)), 4),
                                        "p90": round(float(np.quantile(values, 0.9)), 4),
                                        "max": round(float(values.max()), 4)}
            else:
                quantiles = None
        result["facts"] = {
            "forward_m": world["forward_m"],
            "lateral_m": world["lateral_m"],
            "net_yaw_rad": world["net_yaw_rad"],
            "command_progress_m": world["command_progress_m"],
            "command_yaw_progress_rad": world["command_yaw_progress_rad"],
            "contact_slip_m_left_right": world["contact_slip_m_left_right"],
            "rated_leg_time_s": round(world["fraction_frames_with_saturated_leg"] * duration, 3),
            "hand_force_peak_n": world["hand_force_n"],
            "body_contact_time_s": round(world["fraction_frames_with_body_contact"] * duration, 3),
            "alternating_touchdowns": world["alternating_touchdowns"],
            "ended_by_timeout": world["ended_by_timeout"],
            "stop": world.get("stop"),
            "active_command_absolute_error": quantiles,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    rows = []
    missing = []
    for case in method["cases"]:
        if not (args.folder / (case["name"] + ".json")).is_file():
            missing.append(case["name"])
            continue
        rows.append(episode(case, args.folder, method))
    if missing and not args.partial:
        raise RuntimeError(f"incomplete original set: {missing}")
    result = {"schema": "robot-learning-lab.basic-sweep.posture-locomotion.measurements.v1",
              "method_sha256": sha(METHOD), "original_episodes": len(rows),
              "independent_start_families": len({row["seed"] for row in rows}),
              "observed_control_steps": sum(row["observed_control_steps"] for row in rows),
              "requested_control_steps": sum(case["duration_s"] * 50 for case in method["cases"]),
              "optimizer_updates": 0, "missing_originals": missing, "rows": rows,
              "interpretation": "Native-unit facts only. Distinct conditions are not averaged into a combined score. No mid-episode disturbance was introduced."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"original_episodes": len(rows), "missing": missing,
                      "observed_control_steps": result["observed_control_steps"]}))


if __name__ == "__main__":
    main()
