"""Reconcile route-guard reports, packet traces, declared stop triggers and video."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/ape-native-route-guard"
RUNS = {
    "v1-close": ("run_v1.py", 12),
    "v2-mapped-room": ("run_v2.py", 12),
    "v3-mapped-room": ("run_v3.py", 12),
    "v4-changed-crossing": ("holdout.py", 32),
    "v5-sensor-fusion": ("fused.py", 24),
    "v6-odom-sensitivity": ("odom_sensitivity.py", 24),
    "v7-stop-envelope": ("envelope.py", 12),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(version: str, source: str, expected: int) -> dict:
    directory = BASE / version
    report = json.loads((directory / "report.json").read_text())
    checks = {
        "source_hash": sha(CODE / source) == report["source_sha256"],
        "scene_hash": sha(directory / "scene.xml") == report["scene_sha256"],
        "body_hash": sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") == report["body_sha256"],
        "native_trace_hash": sha(ROOT / "artifacts/ape-cove50-effort-envelope/forward-stop-late.npz") == report["native_trace_sha256"],
        "episode_count": len(report["results"]) == expected,
    }
    if "background_map_sha256" in report:
        checks["background_hash"] = (sha(directory / "background_xy.npy") ==
                                     report["background_map_sha256"])
    if "static_maps" in report:
        checks["background_hash"] = all(
            sha(Path(item["path"])) == item["sha256"]
            for item in report["static_maps"].values())
    for key in ("frozen_guard_sha256", "low_guard_sha256", "guard_source_sha256"):
        if key in report:
            checks["guard_source_hash"] = sha(CODE / "run.py") == report[key]
    if "stop_report_sha256" in report:
        checks["stop_report_hash"] = sha(
            ROOT / "artifacts/ape-skill-contract-cove50-20261009/stop-trace.json"
        ) == report["stop_report_sha256"]
    episode_checks = []
    video_checks = []
    for row in report["results"]:
        trace_path = Path(row["trace"]["path"])
        with np.load(trace_path, allow_pickle=False) as trace:
            t = trace["time_s"]
            gap = np.r_[trace["gap_evaluator_m"], trace["midstep_gap_evaluator_m"]]
            packets = [json.loads(x) for x in trace["packet_json"]]
        trigger = True
        reason = row["stop_reason"]
        if reason.startswith("two_"):
            sensor = reason.removeprefix("two_").removesuffix("_near_returns")
            history = [p for p in packets if p.get("sensor") == sensor and
                       p.get("captured_at_s", 9) <= row["stop_command_s"]+1e-5]
            trigger = len(history) >= 2 and all(
                p.get("min_forward_range_m") is not None and
                p["min_forward_range_m"] <= .75 for p in history[-2:])
        elif reason.endswith("_stale"):
            sensor = reason.removesuffix("_stale")
            history = [p["captured_at_s"] for p in packets
                       if p.get("sensor") == sensor and "captured_at_s" in p]
            period = 1/15 if sensor == "low" else .1
            trigger = not history or row["stop_command_s"]-max(
                t for t in history if t <= row["stop_command_s"]) > 2*period
        elif reason == "planned_end":
            trigger = row["stop_command_s"] == 3.0
        episode_checks.append(all((
            sha(trace_path) == row["trace"]["sha256"],
            t.shape == (400,) and np.allclose(np.diff(t), .02),
            len(packets) == row["packet_count"],
            abs(float(gap.min())-row["min_modeled_body_gap_m"]) < .00002,
            bool((gap <= 0).any()) == row["modeled_overlap"],
            bool((gap < .05).any()) == row["below_5cm_margin"],
            trigger,
        )))
        if row["video"]:
            movie = Path(row["video"]["path"])
            cap = cv2.VideoCapture(str(movie))
            count = 0
            if cap.isOpened():
                while True:
                    okay, frame = cap.read()
                    if not okay:
                        break
                    count += 1
                cap.release()
            video_checks.append(sha(movie) == row["video"]["sha256"] and
                                count == row["video"]["frames"] == 100)
    checks["all_episode_traces"] = all(episode_checks)
    checks["all_videos_decoded_and_hashed"] = bool(video_checks) and all(video_checks)
    output = {"schema": "robot-learning-lab.native-route-guard.audit.v1",
              "report_sha256": sha(directory / "report.json"),
              "auditor_sha256": sha(Path(__file__)),
              "checks": checks,
              "all_evidence_checks_pass": all(checks.values()),
              "episode_count": len(episode_checks),
              "video_count": len(video_checks),
              "scope_limit": "Checks stored source/media/trace provenance and stop bookkeeping. It does not independently re-simulate collision geometry, validate a physical sensor or qualify a real body."}
    (directory / "audit.json").write_text(json.dumps(output, indent=2)+"\n")
    return output


def main() -> None:
    results = {version: audit_run(version, *settings)
               for version, settings in RUNS.items()}
    print(json.dumps({version: {"pass": row["all_evidence_checks_pass"],
                                "episodes": row["episode_count"],
                                "videos": row["video_count"],
                                "checks": row["checks"]}
                      for version, row in results.items()}, indent=2))
    if not all(row["all_evidence_checks_pass"] for row in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
