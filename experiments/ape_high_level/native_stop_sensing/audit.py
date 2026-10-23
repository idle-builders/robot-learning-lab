"""Reconcile the native-tail sensor comparison against its saved traces/media."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-native-stop-sensing/v3"
REPORT = OUT / "report.json"
SOURCE = ROOT / "experiments/ape_high_level/native_stop_sensing/run.py"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
NATIVE = ROOT / "artifacts/ape-cove50-effort-envelope/forward-stop-late.npz"
STOP = ROOT / "artifacts/ape-skill-contract-cove50-20261009/stop-trace.json"
CASES = ("approaching_person", "low_pet", "static_offpath", "low_dropout", "false_return")
MODES = ("none", "mid10", "mid15_matched", "low10_matched",
         "low15", "depth15", "mid_low_depth")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_check(metadata: dict | None) -> dict | None:
    if metadata is None:
        return None
    path = Path(metadata["path"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {path}")
    count = 0
    while True:
        okay, frame = cap.read()
        if not okay:
            break
        if frame.shape != (562, 640, 3):
            raise ValueError(f"Unexpected frame shape {path}: {frame.shape}")
        count += 1
    cap.release()
    return {"path": str(path), "hash_ok": sha(path) == metadata["sha256"],
            "decoded_frames": count, "frames_ok": count == metadata["frames"] == 100}


def audit_episode(row: dict, native: np.lib.npyio.NpzFile,
                  native_delay: float, sensor_periods: dict) -> dict:
    path = Path(row["trace"]["path"])
    with np.load(path, allow_pickle=False) as saved:
        t = saved["time_s"]
        xy = saved["root_xy_evaluator_m"]
        gap = saved["gap_evaluator_m"]
        mid_t = saved["midstep_time_s"]
        mid_gap = saved["midstep_gap_evaluator_m"]
        action = saved["action"]
        joints = saved["leg_joint_rad_actor"]
        yaw = saved["ideal_yaw_rad_actor"]
        packet = [json.loads(x) for x in saved["packet_json"]]
    all_t = np.r_[t, mid_t]
    all_gap = np.r_[gap, mid_gap]
    first_overlap = (float(all_t[all_gap <= 0].min()) if (all_gap <= 0).any()
                     else None)
    expected_start = native["root_pose"][0, 0, :2]
    stop_at = row["stop_command_s"]
    observed_at = round(stop_at + native_delay, 4)
    stage = np.where(t < stop_at-1e-8, "forward",
                     np.where(t < observed_at-1e-8,
                              "stop_tail", "observed_stationary"))
    away_from_float_boundary = ((np.abs(t-stop_at) > 1e-6) &
                                (np.abs(t-observed_at) > 1e-6))
    false_stop = row["case"] in ("static_offpath", "false_return") and stop_at < 3.0
    scanned = [p for p in packet if "captured_at_s" in p]
    per_sensor = {}
    for item in scanned:
        per_sensor.setdefault(item["sensor"], []).append(item)
    trigger_supported = True
    if row["stop_reason"].startswith("two_"):
        name = row["stop_reason"].removeprefix("two_").removesuffix("_near_returns")
        rows = per_sensor.get(name, [])
        candidates = [p for p in rows if p["captured_at_s"] <= stop_at+1e-5]
        trigger_supported = len(candidates) >= 2 and all(
            p["min_forward_range_m"] is not None and
            p["min_forward_range_m"] <= .55 for p in candidates[-2:])
    elif row["stop_reason"].endswith("_stale"):
        name = row["stop_reason"].removesuffix("_stale")
        latest = max((p["captured_at_s"] for p in per_sensor.get(name, [])
                      if p["captured_at_s"] <= stop_at), default=None)
        period = sensor_periods[name]
        trigger_supported = latest is None or stop_at-latest > 2*period
    elif row["stop_reason"] == "planned_end":
        trigger_supported = abs(stop_at-3.0) <= 1e-6
    checks = {
        "trace_sha": sha(path) == row["trace"]["sha256"],
        "time_and_shape": (t.shape == (400,) and mid_t.shape == (399,) and
                           xy.shape == (400, 2) and joints.shape == (400, 10) and
                           yaw.shape == (400,) and len(packet) == row["packet_count"] and
                           np.allclose(np.diff(t), .02) and
                           np.allclose(mid_t, (t[:-1]+t[1:])/2)),
        "common_rebased_start": np.linalg.norm(xy[0]-expected_start) < .0001,
        "action_from_command_and_native_delay": (np.array_equal(
              action[away_from_float_boundary], stage[away_from_float_boundary]) and
              abs(row["observed_stop_s"]-observed_at) < .00011),
        "swept_geometry_summary": (abs(row["min_modeled_body_gap_m"]-all_gap.min()) < .00002 and
              row["modeled_overlap"] == bool((all_gap <= 0).any()) and
              row["below_5cm_margin"] == bool((all_gap < .05).any()) and
              (row["first_overlap_s"] is None if first_overlap is None else
               abs(row["first_overlap_s"]-first_overlap) < .00011)),
        "offpath_false_stop_summary": row["false_early_stop_on_offpath"] == false_stop,
        "sensor_trigger_supported": trigger_supported,
        "packet_monotonic": all(np.all(np.diff([p["captured_at_s"] for p in rows]) > 0)
                                for rows in per_sensor.values()),
    }
    return {"world": row["world"], "case": row["case"], "mode": row["mode"],
            "checks": {key: bool(value) for key, value in checks.items()},
            "video": video_check(row["video"]),
            "min_gap_m": row["min_modeled_body_gap_m"],
            "stop_s": stop_at, "reason": row["stop_reason"]}


def main() -> None:
    report = json.loads(REPORT.read_text())
    stop = json.loads(STOP.read_text())
    delays = {x["world"]: x["command_to_observed_stop_s"]
              for x in stop["rows"] if x["case"] == "forward-stop-late"}
    sensors = report["config"]["sensors"]
    mid_match, low, low10 = (sensors[name] for name in
                             ("mid_match", "low", "low10"))
    source_checks = {
        "runner_sha": sha(SOURCE) == report["source_sha256"],
        "body_sha": sha(BODY) == report["body_sha256"],
        "native_trace_sha": sha(NATIVE) == report["native_trace_sha256"],
        "stop_report_sha": sha(STOP) == report["stop_report_sha256"],
        "scene_sha": sha(OUT / "scene.xml") == report["scene_sha256"],
        "matched_height_and_rate_controls": (
            all(mid_match[k] == low[k] for k in ("period_s", "max_m",
                                               "yaw_deg", "pitch_deg")) and
            mid_match["origin"][:2] == low["origin"][:2] and
            mid_match["origin"][2] != low["origin"][2] and
            all(low10[k] == low[k] for k in ("max_m", "origin",
                                               "yaw_deg", "pitch_deg")) and
            low10["period_s"] != low["period_s"]),
        "factorial_matrix": len(report["results"]) == 140 and
            {(x["world"], x["case"], x["mode"]) for x in report["results"]} ==
            {(w, c, m) for w in range(4) for c in CASES for m in MODES},
    }
    native = np.load(NATIVE, allow_pickle=False)
    periods = {name: value["period_s"] for name, value in sensors.items()}
    rows = [audit_episode(x, native, delays[x["world"]], periods)
            for x in report["results"]]
    native.close()
    videos = [x["video"] for x in rows if x["video"]]
    checks = {**source_checks,
              "all_episode_checks": all(all(x["checks"].values()) for x in rows),
              "seven_videos_decoded_and_hashed": (len(videos) == 7 and
                  all(v["hash_ok"] and v["frames_ok"] for v in videos)),
              "false_return_control": all(x["reason"] == "planned_end"
                  for x in rows if x["case"] == "false_return"),
              "low_hazard_mid_blind": all(x["reason"] == "planned_end"
                  for x in rows if x["case"] == "low_pet" and
                  x["mode"] in ("mid10", "mid15_matched")),
              "low_dropout_declared_stop": all(
                  x["reason"] == ("low10_stale" if x["mode"] == "low10_matched"
                                  else "low_stale")
                  for x in rows if x["case"] == "low_dropout" and
                  x["mode"] in ("low10_matched", "low15", "mid_low_depth"))}
    checks = {key: bool(value) for key, value in checks.items()}
    result = {"schema": "robot-learning-lab.native-stop-sensing-audit.v1",
              "source_sha256": sha(Path(__file__)), "report_sha256": sha(REPORT),
              "checks": checks, "all_evidence_checks_pass": all(checks.values()),
              "episodes": rows,
              "scope_limit": "Hybrid replay of historical rejected Direct Frame motion in a stylized room with ideal ray/depth-like packets; body geometry is the source's budget solids, not a selected shell or qualified physical stop."}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks,
                      "minimum_gaps_by_case_mode": {
                          f"{case}/{mode}": [x["min_gap_m"] for x in rows
                                            if x["case"] == case and x["mode"] == mode]
                          for case in CASES for mode in MODES}}, indent=2))
    if not result["all_evidence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
