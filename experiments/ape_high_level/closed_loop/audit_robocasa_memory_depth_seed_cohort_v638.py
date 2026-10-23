"""Audit paired late-crossing range-noise seeds with/without head depth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_depth_handoff_v634 as handoff_audit
import audit_robocasa_memory_moving_return_v604 as base_audit
import run_robocasa_memory_depth_seed_cohort_v637 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def main() -> None:
    declaration_path = OUT / "predeclared-depth-seed-cohort.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    assert declaration["paired_scan_seeds"] == list(trial.SEEDS)
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
    for ep, (source, label, dropout, depth_fallback, seed) in zip(
            report["episodes"], trial.PROFILES):
        summary = ep["memory_return"]
        assert summary["run_label"] == (
            f"late_seed{seed}_drop_"
            f"{'depth' if depth_fallback else 'range_only'}_held")
        assert summary["paired_scan_seed_requested"] == seed
        # Reuse the independent episode replay with its fixed V633 label check.
        normalized_summary = {**summary,
                              "run_label": f"late_drop_"
                                           f"{'depth' if depth_fallback else 'range_only'}_held"}
        normalized_ep = {**ep, "memory_return": normalized_summary}
        audit = handoff_audit.audit_episode(
            normalized_ep, source, label, dropout, depth_fallback,
            prior, camera_model)
        rows.append({"paired_scan_seed": seed, **audit})
    pairs = {}
    for index in range(0, len(report["episodes"]), 2):
        source, _, _, _, seed = trial.PROFILES[index]
        comparison = handoff_audit.compare_before_loss(
            report["episodes"][index], report["episodes"][index + 1])
        assert comparison["identical"] == comparison["compared"]
        pairs[f"{source}_seed{seed}"] = comparison
    result = {
        "schema": "robot-learning-lab.robocasa-memory-depth-seed-cohort-audit.v1",
        "declaration_sha256": base_audit.sha(declaration_path),
        "report_sha256": base_audit.sha(report_path),
        "videos_decoded": len(rows),
        "pre_loss_pair_comparison": pairs,
        "head_depth_captures_replayed": sum(
            len(row["head_depth_captures_checked"]) for row in rows),
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
