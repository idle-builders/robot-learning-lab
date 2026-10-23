"""Audit the observed-range departure choice and full-horizon V605 outcome."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_predeparture_watch_v605 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


OUT = trial.OUT


def audit_watch(ep: dict) -> dict:
    summary = ep["memory_return"]
    scans = [row for row in ep["trace"]
             if row["stage"] == "predeparture_range_scan"]
    guards = [row for row in ep["trace"]
              if row["stage"] == "predeparture_guard"]
    assert guards and scans
    assert len(guards) <= int(round(trial.WATCH_S / memory.SUBSTEP_S))
    for previous, current in zip(guards, guards[1:]):
        assert abs(current["time_s"] - previous["time_s"] - .02) <= .00001
    centers = []
    range_trigger = None
    for row in scans:
        center = row["track_center_xy_m"]
        if center is None:
            continue
        center = np.asarray(center)
        if centers:
            earlier_time, earlier_center = centers[-1]
            dt = row["time_s"] - earlier_time
            speed = float(np.linalg.norm(center - earlier_center) / dt)
            pose = np.asarray(row["actor_estimated_xy_m"])
            closing = float((np.linalg.norm(earlier_center - pose) -
                             np.linalg.norm(center - pose)) / dt)
            distance = float(np.linalg.norm(center - pose))
            assert abs(speed - row["actor_track_speed_mps"]) <= .0002
            assert abs(closing - row["actor_closing_speed_mps"]) <= .0002
            assert abs(distance - row["actor_track_distance_m"]) <= .0002
            if (.15 <= speed <= 1.0 and closing >= .10 and
                    distance < 1.2 and range_trigger is None):
                range_trigger = row["time_s"]
        centers.append((row["time_s"], center))
    guard_trigger = next((row["time_s"] for row in guards
                          if row["actor_guard_range_m"] < .30), None)
    expected = None
    if range_trigger is not None and (guard_trigger is None or
                                      range_trigger <= guard_trigger):
        expected = "range_track_closing"
    elif guard_trigger is not None:
        expected = "near_body_alarm"
    assert summary["predeparture_watch_trigger"] == expected
    assert summary["predeparture_watch_s_requested"] == trial.WATCH_S
    help_rows = [row for row in ep["trace"]
                 if row["stage"] == "predeparture_owner_help"]
    if expected is not None:
        assert help_rows and help_rows[0]["actor_reason"] == expected
        assert summary["moves"] == 0
        assert summary["item_status"] == "not_rechecked"
    else:
        assert not help_rows
    return {"watch_scans": len(scans), "watch_guard_samples_20ms": len(guards),
            "trigger": expected, "range_trigger_time_s": range_trigger,
            "near_body_trigger_time_s": guard_trigger}


def main() -> None:
    declaration_path = OUT / "predeclared-watch.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    assert declaration["source_sha256"] == base_audit.sha(OUT / trial.SOURCE.name)
    assert declaration["memory_source_sha256"] == base_audit.sha(
        OUT / memory.SOURCE.name)
    assert declaration["motion_source_sha256"] == base_audit.sha(
        OUT / crossing.SOURCE.name)
    assert declaration["joined_source_sha256"] == base_audit.sha(
        OUT / Path(joined.__file__).name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    rows = []
    for ep, (source, motion_label) in zip(report["episodes"], trial.PROFILES):
        summary = ep["memory_return"]
        assert summary["source"] == source
        assert summary["run_label"] == f"{motion_label}_watch"
        rows.append({**base_audit.audit_episode(ep, motion_label),
                     **audit_watch(ep)})
    result = {"schema": "robot-learning-lab.robocasa-memory-watch-audit.v1",
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
