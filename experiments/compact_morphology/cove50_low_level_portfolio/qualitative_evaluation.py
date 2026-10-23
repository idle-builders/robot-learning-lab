"""Make an evidence packet for a human review of an uninterrupted controller run.

This does not assign a score. The reviewer watches the source video, inspects
the trace, and records concrete observations and a plain-language judgment.
Historical boolean gates remain in their original reports for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHASES = ("walk", "stop", "squat", "arms", "handoff", "recovery")
def source(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def facts(report: dict) -> dict:
    windows = report.get("windows", {})
    dt = report["dt_s"]
    window_facts = {}
    for name, window in windows.items():
        frames = window["frames"]
        window_facts[name] = {
            "observed_s": round(frames * dt, 3),
            "two_foot_support_s": round(frames * dt * window["feet_both_fraction"], 3),
            "rated_leg_force_s": round(frames * dt * window["leg_duty_fraction"], 3),
            "body_ground_contact_s": round(frames * dt * window["body_ground_fraction"], 3),
            "pelvis_height_m": window["pelvis_z"],
            "hand_height_m": window["hand_z"],
            "max_tilt_deg": window["max_tilt_deg"],
        }
    return {
        "duration_s": round(report["steps"] * dt, 3),
        "walk_distance_m": report.get("walk_distance_m"),
        "stop_tail_drift_m": report.get("stop_tail_xy_m"),
        "max_hand_ground_force_n": max((row.get("hand_force_n", 0.0)
                                         for row in report["timeseries"]), default=0.0),
        "body_ground_contact_s": round(dt * sum(bool(row.get("body_ground"))
                                               for row in report["timeseries"]), 3),
        "self_contact_s": round(dt * sum(bool(row.get("self_contact"))
                                        for row in report["timeseries"]), 3),
        "phase_windows": window_facts,
    }


def make_packet(report_path: Path, video_path: Path, trace_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    if "timeseries" not in report or not report.get("steps") or not report.get("dt_s"):
        raise ValueError("expected a complete source report with per-step observations")
    if not trace_path.is_file() or not video_path.is_file():
        raise ValueError("source trace and video are required for a behavior review")
    if len({report_path.stem, trace_path.stem, video_path.stem}) != 1:
        raise ValueError("report, trace and original video must share one source stem")
    return {
        "schema": "robot-learning-lab.low-level-qualitative-review.v1",
        "status": "review_needed",
        "source": {"report": source(report_path), "trace": source(trace_path),
                   "video": source(video_path)},
        "case": {"seed": report.get("seed"), "body_sha256": report.get("body_sha256"),
                 "device": report.get("device"), "facts": facts(report)},
        "review": {
            "overall_judgment": "",
            "phase_observations": {phase: "" for phase in PHASES},
            "motion_quality": "",
            "progress_toward_command": "",
            "adaptation_and_recovery": "not observed; describe an actual disturbance if tested",
            "failure_severity_and_timing": "",
            "engineering_limits": "",
            "video_timepoints_s": [],
            "reviewer": "",
        },
        "interpretation": "Describe useful movement and limitations in words, anchored to the "
                          "measured facts and video timepoints. A new start condition is not a "
                          "recovery test. Motor and contact measurements stay separate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("review packet already exists; original reviews are immutable")
    packet = make_packet(args.report, args.video, args.trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(packet, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
