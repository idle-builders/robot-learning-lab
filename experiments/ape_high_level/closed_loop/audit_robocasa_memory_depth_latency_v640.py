"""Audit paired simulated depth processing waits and full owner episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_depth_handoff_v634 as handoff_audit
import audit_robocasa_memory_moving_return_v604 as base_audit
import run_robocasa_memory_depth_latency_v639 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def check_wait(ep: dict, latency_s: float) -> dict:
    summary = ep["memory_return"]
    assert summary["head_depth_processing_latency_s_requested"] == latency_s
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision" and
                 row.get("actor_source") == "head_depth_after_range_loss"]
    waits = [row for row in ep["trace"]
             if row["stage"] == "memory_head_depth_processing_wait"]
    if latency_s == 0. or not decisions or decisions[0]["actor_selected_goal_xy_m"] is None:
        assert not waits
        return {"waits": 0, "guarded_wait_samples": 0,
                "decision_to_dispatch_s": None}
    assert len(decisions) == len(waits) == 1
    decision, wait = decisions[0], waits[0]
    assert wait["actor_capture_decision_time_s"] == decision["time_s"]
    assert abs(wait["elapsed_s"] - latency_s) < .00001
    assert abs(wait["time_s"] - decision["time_s"] - latency_s) < .00001
    guarded = [row for row in ep["trace"]
               if row["stage"] == "memory_post_stop_hold" and
               row.get("actor_depth_processing_wait")]
    assert len(guarded) == int(round(latency_s / .02))
    assert all(abs(b["time_s"] - a["time_s"] - .02) < .00001
               for a, b in zip(guarded, guarded[1:]))
    assert abs(summary["held_reactive_escape_time_s"] - wait["time_s"]) < .00001
    return {"waits": len(waits), "guarded_wait_samples": len(guarded),
            "decision_to_dispatch_s": round(wait["time_s"] - decision["time_s"], 4)}


def compare_pair(first: dict, second: dict) -> dict:
    pre = handoff_audit.compare_before_loss(first, second)
    assert pre["identical"] == pre["compared"]
    def captures(ep):
        return [row for row in ep["trace"]
                if row["stage"] == "memory_head_depth_fallback"]
    a, b = captures(first), captures(second)
    common = min(len(a), len(b))
    matched = sum(a[index]["time_s"] == b[index]["time_s"] and
                  a[index]["depth_sha256"] == b[index]["depth_sha256"]
                  for index in range(common))
    return {**pre, "common_head_depth_captures": common,
            "exact_head_depth_prefix": matched}


def main() -> None:
    declaration_path = OUT / "predeclared-depth-latency.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    assert declaration["paired_scan_seeds"] == list(trial.SEEDS)
    assert declaration["processing_latency_s"] == list(trial.LATENCIES_S)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declaration[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    prior = np.load(joined.PRIOR)
    camera_model = joined.scene_with_owner_and_prop(crossing.START)
    rows = []
    for ep, (source, label, dropout, depth_fallback, seed, latency) in zip(
            report["episodes"], trial.PROFILES):
        summary = ep["memory_return"]
        assert summary["run_label"] == (
            f"late_seed{seed}_drop_depth_lag{int(latency * 1000)}ms_held")
        assert summary["paired_scan_seed_requested"] == seed
        normalized_summary = {**summary, "run_label": "late_drop_depth_held"}
        normalized_ep = {**ep, "memory_return": normalized_summary}
        audit = handoff_audit.audit_episode(
            normalized_ep, source, label, dropout, depth_fallback,
            prior, camera_model)
        rows.append({"paired_scan_seed": seed, "latency_s": latency,
                     **audit, **check_wait(ep, latency)})
    pairs = {}
    for index in range(0, len(report["episodes"]), 2):
        source, _, _, _, seed, _ = trial.PROFILES[index]
        pairs[f"{source}_seed{seed}"] = compare_pair(
            report["episodes"][index], report["episodes"][index + 1])
    result = {
        "schema": "robot-learning-lab.robocasa-memory-depth-latency-audit.v1",
        "declaration_sha256": base_audit.sha(declaration_path),
        "report_sha256": base_audit.sha(report_path),
        "videos_decoded": len(rows),
        "pair_comparison": pairs,
        "head_depth_captures_replayed": sum(
            len(row["head_depth_captures_checked"]) for row in rows),
        "processing_wait_samples_20ms": sum(row["guarded_wait_samples"] for row in rows),
        "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
        "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
