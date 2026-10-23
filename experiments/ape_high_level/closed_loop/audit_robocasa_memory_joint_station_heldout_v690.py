"""Independent geometry/media replay of fresh-seed shifted V689 episodes."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_memory_joint_station_v687 as pilot_audit
import audit_robocasa_memory_moving_return_v604 as memory_audit
import audit_robocasa_person_standoff_v508 as meeting_audit
import run_robocasa_memory_joint_station_heldout_v689 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory


OUT = trial.OUT
PREFLIGHT = trial.ROOT / "artifacts/ape-robocasa-meeting-refuge-preflight-v685/report.json"
PILOT_AUDIT = trial.ROOT / "artifacts/ape-robocasa-memory-joint-station-v686/audit.json"
BASELINE = trial.ROOT / "artifacts/ape-robocasa-memory-meeting-standoff-v680/report.json"


def main() -> None:
    declaration_path = OUT / "predeclared.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    snapshot = json.loads(PREFLIGHT.read_text())["actor_map"]
    pilot = json.loads(PILOT_AUDIT.read_text())
    baseline = json.loads(BASELINE.read_text())
    assert pilot["all_integrity_checks_pass"]
    assert declaration["frozen_pilot_report_sha256"] == pilot[
        "report_sha256"]
    assert report["declaration_sha256"] == memory.digest(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES) == 10
    for key, source in (("source_sha256", trial.SOURCE),
                        ("selector_source_sha256", Path(trial.joint_meeting_refuge.__file__)),
                        ("standoff_source_sha256", Path(trial.standoff.__file__)),
                        ("memory_source_sha256", memory.SOURCE)):
        assert declaration[key] == memory.digest(OUT / source.name)
    old_motion = crossing.prop_motion
    old_horizon = crossing.POST_STOP_UNTIL_S
    old_meeting_motion = meeting_audit.motion
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = 20.8
        for ep, (direction, shift_s, source) in zip(
                report["episodes"], trial.PROFILES):
            crossing.prop_motion = old_motion
            base_script = (old_motion("stationary") if direction == "stationary"
                           else trial.bearing.challenge_motion(direction))
            script = lambda t, base_script=base_script, shift_s=shift_s: base_script(t + shift_s)
            crossing.prop_motion = lambda _, script=script: script
            meeting_audit.motion = lambda _, script=script: script
            assert ep["scene_fingerprint"] == baseline["episodes"][0][
                "scene_fingerprint"]
            assert pilot_audit.route_core(ep) == pilot_audit.route_core(
                baseline["episodes"][0])
            assert ep["outcome"] == "owner_confirmed_metric_find"
            assert ep["memory_return"]["source"] == source
            assert ep["memory_return"]["predeparture_confirmed_track_hold_select_requested"]
            assert not ep["memory_return"]["evaluator_abort_on_margin_breach_requested"]
            station_result = pilot_audit.audit_station(ep, snapshot)
            memory_result = memory_audit.audit_episode(ep, direction)
            rows.append({"direction": direction, "shift_s": shift_s,
                         "source": source, **station_result, **memory_result})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
        meeting_audit.motion = old_meeting_motion
    result = {
        "schema": "robot-learning-lab.robocasa-memory-joint-station-heldout-audit.v1",
        "declaration_sha256": memory.digest(declaration_path),
        "report_sha256": memory.digest(report_path),
        "pilot_audit_sha256": memory.digest(PILOT_AUDIT),
        "scope": "Fresh-seed and crossing-phase owner/memory geometry, complete videos and actor-map station/refuge check; one room and kinematic proxy",
        "videos_decoded": len(rows),
        "pre_memory_geometry_samples": sum(row["pre_memory_geometry_samples"]
                                           for row in rows),
        "memory_movement_samples_20ms": sum(row["movement_samples_20ms"]
                                            for row in rows),
        "memory_held_samples_20ms": sum(row["post_stop_samples_20ms"]
                                        for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
