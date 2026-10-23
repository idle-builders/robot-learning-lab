"""Frozen Scan30 full tasks with one observed live range gap and safe recovery.

The original policy and owner-task scripts remain untouched. A one-second
outage follows the first translated step. The actor holds, waits for three
fresh healthy source packets, reobserves and gets a renewed owner answer,
then may make one more bounded route attempt. No room-specific thresholds.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from freeze_scan20_shared_policy_sources import relative, sha
import run_scan30_unified_policy as unified
from run_scan30_lighting_cross import require_inference_weights, verify_case


ROOT = unified.ROOT
ART = unified.ART
HERE = Path(__file__).resolve().parent
OUT = ROOT / "assets/real_indoor_scenes/scan30_range_recovery_20261015"
PRECOMMIT = ART / "scan30_range_recovery_precommit.json"
REPORT = ART / "scan30_range_recovery_report.json"
HEALTHY = ART / "scan30_same_runtime_healthy_report.json"
FAULT = ART / "scan30_walk_range_gap_report.json"
MOTION = ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1/varied150-forward-seed144-trace.json"

ACTIVE = None
CONTEXT: dict = {}
EVENTS: dict[str, dict] = {}


def freeze(cases: list[dict]) -> dict:
    baseline = json.loads(unified.PRECOMMIT.read_text())
    sources = [Path(__file__), HERE / "run_scan30_unified_policy.py",
        HERE / "run_scan20_shared_owner_policy.py",
        HERE / "run_scan10_recovery_policy.py",
        HERE / "run_scan10_show_again_continuation.py",
        HERE / "run_alt_pair_full_action_20.py",
        HERE / "run_hidden_item_rearrangement.py",
        HERE / "capture_range_monitor.py",
        HERE / "native_shaped_pose.py",
        HERE / "range_pose_corrector.py"]
    return {"schema": "ape.scan30-range-recovery-precommit.v1",
        "healthy_report_sha256": sha(HEALTHY),
        "fault_report_sha256": sha(FAULT),
        "baseline_precommit_sha256": sha(unified.PRECOMMIT),
        "motion_trace_sha256": sha(MOTION),
        "development_manifest": baseline["development_manifest"],
        "development_sha256": baseline["development_sha256"],
        "transfer_manifest": baseline["transfer_manifest"],
        "transfer_sha256": baseline["transfer_sha256"],
        "policy_sources": [{"path": relative(p), "sha256": sha(p)}
                           for p in sources],
        "models": baseline["models"],
        "cases": [{"scene_key": c["scene_key"],
                   "capture_group": c["capture_group"],
                   "cohort": c["cohort"],
                   "photometry": c["photometry"],
                   "source_mesh_sha256": c["source_mesh_sha256"],
                   "owner_screen_sha256": c["owner_screen_sha256"],
                   "item_screen_sha256": c["item_screen_sha256"]}
                  for c in cases],
        "policy": "Same Scan30 from-start owner task and single owner reshow. After first accepted translated step, the independent 10 Hz two-height range worker drops packets for 1 s. The original walk must stop on source_gap. Hold with no commands until three consecutive current-pose healthy packets have arrived within 2.5 s; then make a new present-item RGB-D search from the stopped pose. Require a new owner-confirmed visual identity, a fresh healthy range interval spanning that search, and an available actor pose before one bounded continuation. No second recovery. Do not accept stale target coordinates or bypass the existing route/final-review gates. Count all 30 rooms and preserve failures.",
        "score": ["paired from-start task outcome", "observed missing packet",
                  "stop and recovery latency", "new item location and owner answer",
                  "wrong candidate offer", "resumed movement", "fresh final review",
                  "first-person video and sensor/pose trace"],
        "limits": "Previously exposed recorded rooms and scripted static owner/props; ideal geometry, virtual independently timed range, historical rejected-body pose-shaped increments and 17 cm kinematic proxy. The route planner still uses ideal synchronous scan geometry. This tests software recovery from a one-second worker outage, not a hardware lidar or 50 cm body stop."}


def clean_packets(monitor, timeout_s: float = 2.5) -> dict:
    """Require three new healthy current-pose packets, then the normal gate."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with monitor.lock:
            recent = list(monitor.packets[-3:])
            source_error = monitor.error
        if source_error:
            return {"authorized": False, "reason": "source_error",
                    "error": source_error}
        if len(recent) == 3 and all(p["source_health"] == "healthy" and
                p["pose_seq"] == monitor.pose_seq for p in recent):
            state = monitor.health(recent[0]["capture_wall_s"])
            if state["authorized"] and state["packet_count"] >= 3:
                return state
        time.sleep(.01)
    return {"authorized": False, "reason": "recovery_timeout"}


def install_recovery(policy, show) -> None:
    import run_alt_pair_full_action_20 as action
    import run_hidden_item_rearrangement as motion
    import capture_range_monitor as range_source

    original_walk = motion.walk
    original_policy_search = policy.search
    original_action_search = action.fresh_search
    original_monitor = range_source.CaptureRangeMonitor

    class GapMonitor(original_monitor):
        def __init__(self, case, position, heading_deg):
            global ACTIVE
            super().__init__(case, position, heading_deg)
            self.scene_key = case["scene_key"]
            self.last_position = np.asarray(position, float)
            self.gap_done = False
            self.gap_wall_s = None
            ACTIVE = self

        def update_pose(self, position, heading_deg):
            current = np.asarray(position, float)
            translated = float(np.linalg.norm(current - self.last_position))
            super().update_pose(position, heading_deg)
            self.last_position = current
            if not self.gap_done and translated > .01:
                began = self.inject_gap(1.)
                observed = self.wait_for_gap(began, timeout_s=.45)
                self.gap_wall_s = began
                self.gap_done = True
                EVENTS[self.scene_key] = {"injected_after_translation_m": translated,
                    "gap_wall_s": began, "missing_packet_observed": observed,
                    "pose_seq": self.pose_seq}
                if not observed:
                    raise RuntimeError("Range outage was not observed")

        def close(self):
            global ACTIVE
            packets = super().close()
            if ACTIVE is self:
                ACTIVE = None
            return packets

    def policy_search(scene, case, state, start, memory, present, out,
                      photometry, *, wide=False):
        if present:
            CONTEXT.clear()
            CONTEXT.update(kind="direct", scene=scene, case=case,
                state=state, start=start, memory=memory,
                photometry=photometry)
        return original_policy_search(scene, case, state, start, memory,
                                      present, out, photometry, wide=wide)

    def action_search(scene, case, point, state, out, vision):
        CONTEXT.clear()
        CONTEXT.update(kind="reshow", scene=scene, case=case,
            point=point, state=state, vision=vision)
        return original_action_search(scene, case, point, state, out, vision)

    def recovering_walk(scene, state, observed_target, out, true_target,
                        **kwargs):
        monitor = ACTIVE
        if monitor is None:
            return original_walk(scene, state, observed_target, out,
                                 true_target, **kwargs)
        first = original_walk(scene, state, observed_target, out / "before_gap",
                              true_target, **kwargs)
        record = {"attempted": False, "outage_observed": bool(monitor and
            monitor.gap_done), "first_result": first["result"],
            "first_video": first["video"],
            "first_trace": relative(out / "before_gap/segment.json")}
        (out / "before_gap/segment.json").write_text(json.dumps(first,
            indent=2) + "\n")
        if first["result"] != "source_gap" or monitor is None or \
                not monitor.gap_done:
            first["recovery"] = record
            return first
        record["attempted"] = True
        record["hold_started_wall_s"] = time.perf_counter()
        restored = clean_packets(monitor)
        record["restored_health"] = restored
        record["hold_elapsed_s"] = time.perf_counter() - \
            record["hold_started_wall_s"]
        if not restored["authorized"]:
            first["recovery"] = record
            return first
        context = dict(CONTEXT)
        if context.get("scene") is not scene or context.get("state") is not state:
            record["reason"] = "missing_relook_context"
            first["recovery"] = record
            return first
        relook_start = monitor.mark()
        if context["kind"] == "direct":
            fresh = original_policy_search(scene, context["case"], state,
                context["start"], context["memory"], True,
                out / "recovery_relook", context["photometry"])
            goal = policy.confirmed_goal(fresh)
            renewed_action = fresh["action"]
        else:
            fresh = original_action_search(scene, context["case"],
                context["point"], state, out / "recovery_relook",
                context["vision"])
            goal = fresh["goal"]
            renewed_action = fresh["result"]["action"]
        relook_health = monitor.wait_health(relook_start)
        record.update(relook_action=renewed_action,
            relook_source_health=relook_health,
            relook_goal_xz_m=goal,
            relook_trace=relative(out / "recovery_relook/episode_trace.json"))
        if not relook_health["authorized"] or \
                renewed_action != "owner_confirmed_visual_identity" or \
                goal is None:
            record["reason"] = "new_view_or_source_unresolved"
            first["recovery"] = record
            return first
        # A new owner-confirmed image supplies the next target. The original
        # remembered coordinate is never used after the interruption.
        remaining = max(12, int(kwargs.get("max_actions") or 96) -
                        len(first["actions"]))
        resume_started = monitor.mark()
        def resumed_guard(_action, _state):
            return monitor.wait_health(resume_started)
        continued = original_walk(scene, state, np.asarray(goal, float),
            out / "after_relook", true_target,
            **{**kwargs, "max_actions": remaining,
               "decision_guard": resumed_guard})
        (out / "after_relook/segment.json").write_text(json.dumps(
            continued, indent=2) + "\n")
        record.update(resumed=True, second_result=continued["result"],
            second_video=continued["video"],
            second_trace=relative(out / "after_relook/segment.json"))
        return {**continued, "actions": first["actions"] +
            continued["actions"],
            "accepted_forward_steps": first["accepted_forward_steps"] +
            continued["accepted_forward_steps"],
            "route_fallback_used": first["route_fallback_used"] or
            continued["route_fallback_used"],
            "max_evaluator_capture_pose_error_m": max(first[
                "max_evaluator_capture_pose_error_m"], continued[
                "max_evaluator_capture_pose_error_m"]),
            "recovery": record}

    policy.CaptureRangeMonitor = GapMonitor
    action.CaptureRangeMonitor = GapMonitor
    policy.search = policy_search
    action.fresh_search = action_search
    motion.walk = recovering_walk


def recovery_record(row: dict) -> dict | None:
    if row.get("show_summary"):
        show = json.loads((ROOT / row["show_summary"]).read_text())
        path = (show.get("action") or {}).get("walk_trace")
    else:
        first = json.loads((ROOT / row["base_summary"]).read_text())
        path = first.get("walk_trace")
    if path:
        return json.loads((ROOT / path).read_text()).get("recovery")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = unified.cohort()
    definition = freeze(cases)
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen recovery definition changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(cases))
        return
    if sha(MOTION) != definition["motion_trace_sha256"]:
        raise ValueError("Motion trace changed")
    require_inference_weights()
    import run_scan20_shared_owner_policy as policy
    import run_scan10_recovery_policy as recovery
    import run_scan10_show_again_continuation as show
    unified.BASE_OUT = OUT / "base"
    unified.SHOW_OUT = OUT / "show"
    policy.OUT = unified.BASE_OUT
    recovery.OUT = unified.BASE_OUT
    show.PRIOR_OUT = unified.BASE_OUT
    show.OUT = unified.SHOW_OUT
    show.SOURCE_OWNER = unified.SHOW_OUT / "source_owner"
    recovery.install_recoveries(policy)
    install_recovery(policy, show)
    rows = []
    launched = 0
    for case in cases:
        if args.scene and case["scene_key"] != args.scene:
            continue
        verify_case(case)
        file = OUT / case["scene_key"] / "combined_summary.json"
        if file.is_file():
            row = json.loads(file.read_text())
        elif args.max_new and launched >= args.max_new:
            break
        else:
            EVENTS.pop(case["scene_key"], None)
            try:
                row = unified.attempt(case, policy, recovery, show)
                record = recovery_record(row)
                row["range_fault"] = EVENTS.get(case["scene_key"])
                row["range_recovery"] = record
            except Exception as error:
                row = {"scene_key": case["scene_key"],
                    "capture_group": case["capture_group"],
                    "cohort": case["cohort"],
                    "photometry": case["photometry"],
                    "result": "runner_failed", "error": repr(error),
                    "error_traceback": traceback.format_exc()[-3000:]}
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(row, indent=2) + "\n")
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.scan30-range-recovery-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["result"],
              (row.get("range_recovery") or {}).get("second_result"),
              flush=True)


if __name__ == "__main__":
    main()
