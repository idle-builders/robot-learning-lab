"""Audit sensor-loss timing, range handoff and full owner videos for V601."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_memory_return_v600 as base_audit
import run_robocasa_memory_pose_failover_v601 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


OUT = trial.OUT


def audit_fault(ep: dict) -> dict:
    summary = ep["memory_return"]
    profile = summary["fault_profile"]
    memory_rows = [row for row in ep["trace"]
                   if row["stage"] == "memory_route_decision"]
    fault_rows = [row for row in memory_rows
                  if row.get("post_move_localization", {}).get("status") ==
                  "depth_packet_missing"]
    assert len(fault_rows) == (0 if profile == "none" else 1)
    if not fault_rows:
        assert summary["pose_source_at_return"] == "rgbd"
        return {"profile": profile, "fault_seen": False,
                "fresh_range_handoff": False}
    fault = fault_rows[0]
    fault_time = fault["post_move_localization"]["capture_time_s"]
    assert summary["fault_after_move"] == 3
    assert fault["post_move_localization"]["policy"] == profile
    later = memory_rows[memory_rows.index(fault) + 1:]
    if profile == "stop_on_rgbd_loss":
        assert summary["route_status"] == "pose_health_stop"
        assert summary["item_status"] == "not_rechecked"
        assert not later
        assert not any(row["stage"] == "memory_fresh_view" for row in ep["trace"])
        return {"profile": profile, "fault_seen": True,
                "fresh_range_handoff": False}
    assert profile == "range_fallback"
    assert later
    first = later[0]
    assert first["pose_source_used"] == "range_recovery"
    assert first["range_capture_time_s"] >= fault_time
    assert first["localization"]["status"] in ("mapped", "registered")
    assert all(row["pose_source_used"] == "range_recovery" for row in later)
    assert summary["pose_source_at_return"] == "range_recovery"
    return {"profile": profile, "fault_seen": True,
            "fault_time_s": fault_time,
            "fresh_range_handoff": True,
            "first_range_time_s": first["range_capture_time_s"],
            "first_range_localization": first["localization"]["status"]}


def main() -> None:
    declaration_path = OUT / "predeclared-pose-failover.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    assert declaration["source_sha256"] == base_audit.sha(OUT / trial.SOURCE.name)
    assert declaration["memory_source_sha256"] == base_audit.sha(OUT / memory.SOURCE.name)
    assert declaration["joined_source_sha256"] == base_audit.sha(
        OUT / Path(joined.__file__).name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    assert [(ep["memory_return"]["item_state"],
             ep["memory_return"]["fault_profile"])
            for ep in report["episodes"]] == list(trial.PROFILES)
    rows = []
    route_prefixes = []
    for ep in report["episodes"]:
        assert ep["outcome"] == "owner_confirmed_metric_find"
        route_prefixes.append([(row["rgb_sha256"], row["actual_xy_m"],
                                row["requested_xy_m"])
                               for row in ep["route_episode"]["trace"]])
        rows.append({**base_audit.audit_episode(ep), **audit_fault(ep)})
    assert all(prefix == route_prefixes[0] for prefix in route_prefixes)
    assert rows[0]["item_status"] == "fresh_metric_item_refound"
    assert rows[1]["item_status"] == "not_rechecked"
    assert rows[2]["item_status"] == "fresh_metric_item_refound"
    assert rows[3]["item_status"] == "item_not_refound_ask_owner"
    assert not any(row["wrong_claim"] for row in rows)
    result = {"schema": "robot-learning-lab.robocasa-memory-pose-failover-audit.v1",
              "declaration_sha256": base_audit.sha(declaration_path),
              "report_sha256": base_audit.sha(report_path),
              "videos_decoded": len(rows),
              "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "wrong_metric_item_claims": 0,
              "pre_memory_route_prefixes_match": True,
              "all_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
