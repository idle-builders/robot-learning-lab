"""Replay V654 fast/slow warnings, map changes, motion and saved videos."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_trail_motion_v651 as trail_audit
import audit_robocasa_memory_future_hold_v642 as screen_audit
import audit_robocasa_memory_warning_confirmation_v653 as warning_audit
import dynamic_trail_map
import held_pose_screen
import run_robocasa_memory_warning_speed_v654 as trial
import run_robocasa_memory_reactive_speed_v627 as speed_trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def main() -> None:
    declaration = OUT / "predeclared-warning-speed.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("trail_source_sha256", Path(dynamic_trail_map.__file__)),
                        ("screen_source_sha256", Path(held_pose_screen.__file__)),
                        ("speed_source_sha256", speed_trial.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declared[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (speed, source, count) in zip(report["episodes"], trial.PROFILES):
            original = old_motion("moving_across_return")

            def independent_motion(time_s: float):
                shifted = time_s - .4
                if shifted > crossing.ENTRY_END_S:
                    shifted = crossing.ENTRY_END_S + speed * (
                        shifted - crossing.ENTRY_END_S)
                return original(shifted)

            crossing.prop_motion = lambda _, fn=independent_motion: fn
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == (
                f"speed{speed:.1f}_seed{trial.SEED}_{count}warning_held")
            rows.append({"speed": speed, "source": source,
                         "confirmations_required": count,
                         **base_audit.audit_episode(ep, "late"),
                         **trail_audit.audit_map_evidence(ep, True),
                         **warning_audit.audit_warnings(ep, count)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {}
    for index in range(0, len(trial.PROFILES), 2):
        speed, source, _ = trial.PROFILES[index]
        pairs[f"speed{speed:.1f}_{source}"] = (
            screen_audit.compare_predecision(report["episodes"][index],
                                             report["episodes"][index + 1]))
    result = {
        "schema": "robot-learning-lab.robocasa-memory-warning-speed-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Slower/faster prop and one/two-warning response in the full owner task",
        "videos_decoded": len(rows),
        "predecision_pair_comparison": pairs,
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
