"""Matched from-start Scan30 tasks with a live range outage after first translation.

This comparison keeps the frozen Scan30 visual and task policy. It changes
only the separately timed range stream: after the first accepted translation,
the source drops packets for one second. Tasks with no translation cannot
exercise a during-walk outage and are reported as unexposed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from freeze_scan20_shared_policy_sources import relative, sha
import run_scan30_unified_policy as unified


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
OUT = ROOT / "assets/real_indoor_scenes/scan30_walk_range_gap_20261015"
PRECOMMIT = ART / "scan30_walk_range_gap_precommit.json"
REPORT = ART / "scan30_walk_range_gap_report.json"
BASELINE = ART / "scan30_unified_policy_report.json"
EVENTS: dict[str, list[dict]] = {}


def freeze(cases: list[dict]) -> dict:
    here = Path(__file__).resolve().parent
    baseline = json.loads(BASELINE.read_text())
    if baseline["processed"] != 30 or len(baseline["rows"]) != 30:
        raise ValueError("Healthy reference is not the complete Scan30 run")
    sources = [Path(__file__), here / "run_scan30_unified_policy.py",
        here / "run_scan20_shared_owner_policy.py",
        here / "run_scan10_recovery_policy.py",
        here / "run_scan10_show_again_continuation.py",
        here / "run_alt_pair_full_action_20.py",
        here / "capture_range_monitor.py"]
    return {"schema": "ape.scan30-walk-range-gap-precommit.v1",
        "reference_report": relative(BASELINE),
        "reference_sha256": sha(BASELINE),
        "reference_precommit_sha256": sha(unified.PRECOMMIT),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in sources],
        "cases": [{"scene_key": c["scene_key"],
                   "capture_group": c["capture_group"],
                   "source_mesh_sha256": c["source_mesh_sha256"],
                   "owner_screen_sha256": c["owner_screen_sha256"],
                   "item_screen_sha256": c["item_screen_sha256"],
                   "photometry": c["photometry"]} for c in cases],
        "fault": "Keep the reference full task and sensor gate. In the direct branch, command one 1.0-second gap in the independent 10 Hz two-height range worker after the first accepted translation; confirm a missing packet before the next decision. In an owner-reshow branch, use the existing range-gap action arm, which injects after the first accepted forward step or, if no step occurs, before final review. Tasks without a translated direct walk are unexposed and must not be counted as a successful live-walk fault test. No scene-specific rule changes.",
        "score": ["paired full task result", "pre-fault owner and object progress",
            "fault exposure and packet timing", "guarded walk response",
            "wrong confident offer", "owner help", "retained video and trace"],
        "limits": "Development-exposed rooms and scripted rigid owner in static real-capture meshes; virtual fault in a separately clocked simulated range worker, not a physical lidar failure. Historical 17 cm motion proxy and ideal RGB-D/yaw remain. The two Habitat test interiors fail the photographic low-camera appearance gate."}


def install_fault(policy, recovery, show) -> None:
    import run_alt_pair_full_action_20 as action

    original_monitor = policy.CaptureRangeMonitor

    class WalkGapMonitor(original_monitor):
        def __init__(self, case, position, heading_deg):
            super().__init__(case, position, heading_deg)
            self.scene_key = case["scene_key"]
            self.last_position = np.asarray(position, float)
            self.gap_done = False

        def update_pose(self, position, heading_deg):
            current = np.asarray(position, float)
            translated = float(np.linalg.norm(current - self.last_position))
            super().update_pose(position, heading_deg)
            self.last_position = current
            if not self.gap_done and translated > .01:
                started = self.inject_gap(1.)
                seen = self.wait_for_gap(started, timeout_s=.45)
                event = {"stage": "first_translated_walk_action",
                    "command_wall_s": started, "observed_missing_packet": seen,
                    "translation_m": translated, "pose_seq": self.pose_seq}
                EVENTS.setdefault(self.scene_key, []).append(event)
                self.gap_done = True
                if not seen:
                    raise RuntimeError("Walk range gap command produced no packet")

    policy.CaptureRangeMonitor = WalkGapMonitor
    original_show = show.run_one
    original_action = action.run_one

    def fault_show(case):
        # The already implemented action arm injects into this branch's own
        # independently timed range worker after its first accepted move.
        def range_gap_action(staged_case, arm):
            if arm != "healthy":
                raise ValueError(f"Unexpected owner-reshow arm: {arm}")
            return original_action(staged_case, "range_gap")

        action.run_one = range_gap_action
        try:
            return original_show(case)
        finally:
            action.run_one = original_action

    show.run_one = fault_show


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = unified.cohort()
    definition = freeze(cases)
    if not args.pilot:
        if PRECOMMIT.is_file():
            if json.loads(PRECOMMIT.read_text()) != definition:
                raise ValueError("Range-gap fault or frozen source changed")
        else:
            PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(cases))
        return
    output = OUT if not args.pilot else ROOT / \
        "assets/real_indoor_scenes/scan30_walk_range_gap_pilot"
    report = REPORT if not args.pilot else ART / \
        "scan30_walk_range_gap_pilot_report.json"
    import run_scan20_shared_owner_policy as policy
    import run_scan10_recovery_policy as recovery
    import run_scan10_show_again_continuation as show
    unified.OUT = output
    unified.BASE_OUT = output / "base"
    unified.SHOW_OUT = output / "show"
    policy.OUT = unified.BASE_OUT
    recovery.OUT = unified.BASE_OUT
    show.PRIOR_OUT = unified.BASE_OUT
    show.OUT = unified.SHOW_OUT
    show.SOURCE_OWNER = unified.SHOW_OUT / "source_owner"
    recovery.install_recoveries(policy)
    install_fault(policy, recovery, show)
    rows = []
    new = 0
    for case in cases:
        if args.scene and args.scene != case["scene_key"]:
            continue
        if sha(ROOT / case["source_mesh_resolved"]) != \
                case["source_mesh_sha256"]:
            raise ValueError(f"Changed room mesh: {case['scene_key']}")
        for key, digest in (("owner_screen", "owner_screen_sha256"),
                            ("item_screen", "item_screen_sha256")):
            if case[key] is None:
                if case[digest] is not None:
                    raise ValueError(f"Unexpected missing {key}: {case['scene_key']}")
            elif sha(ROOT / case[key]) != case[digest]:
                raise ValueError(f"Changed {key}: {case['scene_key']}")
        result_file = output / case["scene_key"] / "combined_summary.json"
        if result_file.is_file():
            row = json.loads(result_file.read_text())
        elif args.max_new and new >= args.max_new:
            break
        else:
            row = unified.attempt(case, policy, recovery, show)
            event = EVENTS.get(case["scene_key"], [])
            show_row = json.loads((ROOT / row["show_summary"]).read_text()) \
                if row.get("show_summary") else None
            show_action = (show_row or {}).get("action") or {}
            exposure = ("direct_walk" if event else
                "show_walk" if show_action.get("gap_stage") == "walk" else
                "show_final" if show_action.get("gap_stage") == "final" else
                "unexposed")
            row["fault_exposure"] = exposure
            row["fault_events"] = event
            row["show_gap_stage"] = show_action.get("gap_stage")
            row["fault_observed"] = bool(event and event[0][
                "observed_missing_packet"] or show_action.get("gap_stage"))
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(row, indent=2) + "\n")
            new += 1
        rows.append(row)
        report.write_text(json.dumps({"schema": "ape.scan30-walk-range-gap-report.v1",
            "precommit": relative(PRECOMMIT) if not args.pilot else None,
            "processed": len(rows), "outcomes": dict(Counter(r["result"] for r in rows)),
            "exposures": dict(Counter(r.get("fault_exposure") for r in rows)),
            "rows": rows}, indent=2) + "\n")
        print(case["cohort"], case["scene_key"], row["result"],
              row.get("fault_exposure"), flush=True)


if __name__ == "__main__":
    main()
