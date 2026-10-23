"""Independently replay the cold warm-up and slow-speed comparison in V629."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_reactive_speed_v628 as speed_audit
import audit_robocasa_memory_moving_return_v604 as base_audit
import run_robocasa_memory_reactive_speed_v627 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.ROOT / "artifacts/ape-robocasa-memory-reactive-warm-speed-v629"
SOURCE = Path(__file__).resolve()


def main() -> None:
    declaration_path = OUT / "predeclared-reactive-speed.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    for key, source in (("source_sha256", Path(__file__).with_name(
                            "run_robocasa_memory_reactive_warm_speed_v629.py")),
                        ("memory_source_sha256", memory.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declaration[key] == base_audit.sha(OUT / source.name)
    assert declaration["profiles"] == [[1., False], [.7, False], [.7, True]]
    assert len(report["episodes"]) == 3
    prior = np.load(joined.PRIOR)
    rows = [speed_audit.audit_episode(ep, speed, reaction, prior)
            for ep, (speed, reaction) in zip(
                report["episodes"], ((1., False), (.7, False), (.7, True)))]
    pair = speed_audit.compare_pair(report["episodes"][1], report["episodes"][2])
    result = {
        "schema": "robot-learning-lab.robocasa-memory-reactive-warm-speed-audit.v1",
        "declaration_sha256": base_audit.sha(declaration_path),
        "report_sha256": base_audit.sha(report_path),
        "videos_decoded": len(rows),
        "slow_pair_pre_action": pair,
        "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
        "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
