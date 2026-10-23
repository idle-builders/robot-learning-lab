"""Read-only reconciliation of the development sweeps and crossing challenges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-native-stop-sensing"
CODE = ROOT / "experiments/ape_high_level/native_stop_sensing"
SOURCE = {
    "v4-threshold-sweep": "sweep_threshold.py",
    "v5-motion-gate": "motion_gate_pilot.py",
    "v6-motion-gate": "motion_gate_v6.py",
    "v7-motion-gate": "motion_gate.py",
    "v8-crossing-challenge": "holdout_crossing.py",
    "v9-close-crossing": "holdout_close.py",
    "v10-close-sensor-combos": "sensor_fusion_challenge.py",
}
EXPECTED = {
    "v4-threshold-sweep": 64,
    "v5-motion-gate": 20,
    "v6-motion-gate": 20,
    "v7-motion-gate": 20,
    "v8-crossing-challenge": 24,
    "v9-close-crossing": 24,
    "v10-close-sensor-combos": 24,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_one(directory: str) -> dict:
    base = BASE / directory
    report = json.loads((base / "report.json").read_text())
    body = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    native = ROOT / "artifacts/ape-cove50-effort-envelope/forward-stop-late.npz"
    checks = {
        "source_hash": sha(CODE / SOURCE[directory]) == report["source_sha256"],
        "runner_hash": sha(CODE / "run.py") == report["v3_runner_sha256"],
        "scene_hash": sha(base / "scene.xml") == report["scene_sha256"],
        "body_hash": sha(body) == report["body_sha256"],
        "native_hash": sha(native) == report["native_trace_sha256"],
        "episode_count": len(report["results"]) == EXPECTED[directory],
    }
    row_checks = []
    video_checks = []
    for row in report["results"]:
        path = Path(row["trace"]["path"])
        with np.load(path, allow_pickle=False) as trace:
            t = trace["time_s"]
            gaps = np.r_[trace["gap_evaluator_m"],
                         trace["midstep_gap_evaluator_m"]]
            packets = [json.loads(x) for x in trace["packet_json"]]
        triggered = True
        if row["stop_reason"].startswith("two_"):
            sensor = row["stop_reason"].removeprefix("two_").removesuffix("_near_returns")
            threshold = row.get("threshold_m", report.get("threshold_m", .75))
            if directory == "v4-threshold-sweep":
                threshold = row["threshold_m"]
            history = [p for p in packets if p.get("sensor") == sensor and
                       p.get("captured_at_s", 9) <= row["stop_command_s"]+1e-5]
            triggered = len(history) >= 2 and all(
                p.get("min_forward_range_m") is not None and
                p["min_forward_range_m"] <= threshold+1e-5 for p in history[-2:])
        elif row["stop_reason"] == "planned_end":
            triggered = row["stop_command_s"] == 3.0
        checks_here = {
            "trace_hash": sha(path) == row["trace"]["sha256"],
            "time_base": t.shape == (400,) and np.allclose(np.diff(t), .02),
            "gap_summary": abs(float(gaps.min())-row["min_modeled_body_gap_m"]) < .00002,
            "overlap_summary": bool((gaps <= 0).any()) == row["modeled_overlap"],
            "margin_summary": bool((gaps < .05).any()) == row["below_5cm_margin"],
            "packet_count": len(packets) == row["packet_count"],
            "trigger_supported": triggered,
        }
        row_checks.append(all(checks_here.values()))
        if row["video"]:
            info = row["video"]
            movie = Path(info["path"])
            cap = cv2.VideoCapture(str(movie))
            count = 0
            if cap.isOpened():
                while True:
                    okay, frame = cap.read()
                    if not okay:
                        break
                    count += 1
                cap.release()
            video_checks.append(sha(movie) == info["sha256"] and
                                count == info["frames"] == 100)
    checks["all_episode_traces"] = all(row_checks)
    checks["all_videos_decoded_and_hashed"] = all(video_checks) and bool(video_checks)
    output = {"schema": "robot-learning-lab.native-stop-sensing.followup-audit.v1",
              "report_sha256": sha(base / "report.json"),
              "auditor_sha256": sha(Path(__file__)),
              "checks": checks,
              "all_evidence_checks_pass": all(checks.values()),
              "episode_count": len(row_checks),
              "video_count": len(video_checks),
              "scope_limit": "Reconciles stored traces/media and trigger bookkeeping; does not independently recompute MuJoCo geometry or validate a real sensor, physical body, or new learned policy."}
    (base / "audit.json").write_text(json.dumps(output, indent=2)+"\n")
    return output


def main() -> None:
    results = {key: audit_one(key) for key in SOURCE}
    print(json.dumps({key: {"pass": value["all_evidence_checks_pass"],
                             "episodes": value["episode_count"],
                             "videos": value["video_count"],
                             "checks": value["checks"]}
                      for key, value in results.items()}, indent=2))
    if not all(v["all_evidence_checks_pass"] for v in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
