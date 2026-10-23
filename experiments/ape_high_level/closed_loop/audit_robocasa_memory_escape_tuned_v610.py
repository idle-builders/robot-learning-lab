"""Replay V609 owner, scan cue, requested escape, and complete gap horizon."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import run_robocasa_memory_escape_tuned_v609 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


OUT = trial.OUT


def audit_escape(ep: dict, moving: bool) -> dict:
    summary = ep["memory_return"]
    scans = [row for row in ep["trace"] if row["stage"] == "predeparture_range_scan"]
    guards = [row for row in ep["trace"] if row["stage"] == "predeparture_guard"]
    decisions = [row for row in ep["trace"] if row["stage"] == "memory_route_decision"]
    help_rows = [row for row in ep["trace"] if row["stage"] == "predeparture_owner_help"]
    assert guards and scans
    assert len(guards) <= int(round(trial.WATCH_S / memory.SUBSTEP_S))
    assert all(abs(b["time_s"] - a["time_s"] - .02) <= .00001
               for a, b in zip(guards, guards[1:]))
    assert summary["predeparture_escape_requested"]
    assert summary["predeparture_watch_s_requested"] == trial.WATCH_S
    previous = None
    changes = []
    first_actor_cue = None
    for row in scans:
        raw = np.array([np.nan if value is None else value
                        for value in row["range_72_beams_m"]], dtype=float)
        event = row.get("actor_beam_change")
        expected = None
        if previous is not None:
            shortening = previous - raw
            indices = np.flatnonzero(np.isfinite(shortening) &
                                     (shortening >= .07) & (raw < 1.0))
            if len(indices) >= 3 and indices[-1] - indices[0] <= 8:
                center = float(np.average(indices, weights=shortening[indices]))
                expected = {"time_s": row["time_s"],
                            "beam_indices": indices.tolist(),
                            "weighted_center_beam": round(center, 4),
                            "largest_shortening_m": round(
                                float(np.max(shortening[indices])), 4)}
                changes.append(expected)
                if first_actor_cue is None:
                    first_actor_cue = row["time_s"]
        assert event == expected
        previous = raw
        candidates = row.get("actor_escape_candidates")
        if candidates is not None:
            assert len(changes) >= 2 and event is not None
            assert changes[-2]["time_s"] >= row["time_s"] - .4001
            drift = changes[-1]["weighted_center_beam"] - changes[-2]["weighted_center_beam"]
            assert abs(drift - row["actor_bearing_change_beams"]) <= .0001
            assert abs(drift) >= .5
            assert len(candidates) >= 1
            # The owner route carries a zero-heading base estimate; scan's
            # 5 m argument is its range limit, not base yaw.
            bearing = -np.pi + 2 * np.pi * event["weighted_center_beam"] / 72
            tangent = (np.array([-np.sin(bearing), np.cos(bearing)]) if drift < 0
                       else np.array([np.sin(bearing), -np.cos(bearing)]))
            pose = np.asarray(row["actor_estimated_xy_m"])
            for candidate, distance in zip(candidates, summary["escape_distances_m_requested"]):
                assert np.linalg.norm(np.asarray(candidate["goal_xy_m"]) -
                                      (pose + distance * tangent)) <= .00002
            chosen = next((candidate for candidate in candidates
                           if candidate["plan_status"] == "known_goal" and
                           candidate["has_request"]), None)
            if chosen is not None:
                assert np.linalg.norm(np.asarray(chosen["goal_xy_m"]) -
                                      summary["predeparture_escape_goal_xy_m"]) <= .00002
            else:
                assert summary["predeparture_escape_goal_xy_m"] is None
    if moving:
        assert summary["predeparture_watch_trigger"] == "beam_change_escape"
        assert first_actor_cue is not None
        assert summary["predeparture_escape_goal_xy_m"] is not None
        assert decisions and all(row["phase"] == "evade" for row in decisions)
        if summary["route_status"] == "evaded_wait_for_owner_clearance":
            assert help_rows and help_rows[-1]["actor_reason"] == "beam_change_escape"
        else:
            assert not help_rows
        assert summary["item_status"] == "not_rechecked"
    else:
        assert summary["predeparture_watch_trigger"] is None
        assert summary["predeparture_escape_goal_xy_m"] is None
        assert all(row["phase"] in ("away", "return") for row in decisions)
        assert not help_rows
        assert summary["item_status"] == "fresh_metric_item_refound"
    return {"watch_scans": len(scans), "watch_guard_samples_20ms": len(guards),
            "first_beam_change_s": first_actor_cue,
            "escape_goal_xy_m": summary["predeparture_escape_goal_xy_m"],
            "escape_decisions": sum(row["phase"] == "evade" for row in decisions),
            "owner_help_events": len(help_rows)}


def main() -> None:
    declaration_path = OUT / "predeclared-escape-tuned.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__))):
        assert declaration[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    rows = []
    for ep, (source, motion_label) in zip(report["episodes"], trial.PROFILES):
        summary = ep["memory_return"]
        assert summary["source"] == source
        assert summary["run_label"] == f"{motion_label}_escape_tuned"
        rows.append({**base_audit.audit_episode(ep, motion_label),
                     **audit_escape(ep, motion_label != "stationary")})
    result = {"schema": "robot-learning-lab.robocasa-memory-escape-tuned-audit.v1",
              "declaration_sha256": base_audit.sha(declaration_path),
              "report_sha256": base_audit.sha(report_path),
              "videos_decoded": len(rows),
              "watch_guard_samples_20ms": sum(row["watch_guard_samples_20ms"]
                                              for row in rows),
              "moving_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "post_stop_samples_20ms": sum(row["post_stop_samples_20ms"]
                                             for row in rows),
              "all_integrity_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
