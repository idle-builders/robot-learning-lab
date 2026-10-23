"""Verify V28's saved actor packets, evaluator visibility and alert timing."""

from __future__ import annotations

import json
from pathlib import Path

import held_body_intrusion_trial as late
import held_monitor_probe as probe
import kitchen_trial as trial


def main() -> None:
    path = probe.OUT / "report.json"
    report = json.loads(path.read_text())
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(probe.__file__)),
        "online": report["online_source_sha256"] ==
                  trial.sha(Path(probe.online.__file__)),
        "dense": report["dense_source_sha256"] ==
                 trial.sha(Path(probe.dense.__file__)),
        "scene": report["scene_sha256"] ==
                 trial.sha(late.previous.OUT / "scene.xml"),
        "native": report["native_trace_sha256"] == trial.sha(probe.kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(probe.kitchen.BODY),
        "v27": report["v27_report_sha256"] == trial.sha(late.OUT / "report.json"),
        "matrix": len(report["results"]) == 32 and len({
            (r["mode"], r["world"], r["case"]) for r in report["results"]}) == 32,
    }
    summary = {}
    for mode, (bearings, rate) in probe.SENSOR_MODES.items():
        rows = [r for r in report["results"] if r["mode"] == mode]
        moving = [r for r in rows if r["case"] in late.CASES]
        stationary = [r for r in rows if r["case"] in probe.STATIC_CASES]
        leads = [r["alert_lead_to_5cm_s"] for r in moving
                 if r["alert_lead_to_5cm_s"] is not None]
        summary[mode] = {
            "moving_person_visible": sum(r["first_person_ray_after_motion_s"]
                                         is not None for r in moving),
            "moving_actor_alerts": sum(r["first_two_packet_actor_alert_s"]
                                       is not None for r in moving),
            "stationary_false_alerts": sum(r["first_two_packet_actor_alert_s"]
                                           is not None for r in stationary),
            "alerts_after_5cm_breach": sum(lead < 0 for lead in leads),
            "min_alert_lead_s": min(leads) if leads else None,
            "max_alert_lead_s": max(leads) if leads else None,
        }
        for row in rows:
            key = f'{mode}:{row["world"]}:{row["case"]}'
            trace = Path(row["trace"]["path"])
            checks[f"trace:{key}"] = trial.sha(trace) == row["trace"]["sha256"]
            packets = json.loads(trace.read_text())
            checks[f"packet:{key}"] = (
                len(packets) == row["packet_count"] and
                all(len(p["actor_packet"]["ranges_m"]) == len(bearings) and
                    p["actor_packet"]["valid_returns"] == sum(
                        x is not None for x in p["actor_packet"]["ranges_m"])
                    and set(p["actor_packet"]) == {
                        "captured_at_s", "ranges_m", "valid_returns"}
                    and p["actor_decision"]["map_source"] ==
                    "none_online_clusters"
                    for p in packets))
            visible = next((p["time_s"] for p in packets
                            if p["time_s"] >= late.ENTRY_START_S and
                            p["evaluator_person_ray_hits"] > 0), None)
            alert = next((p["time_s"] for p in packets
                          if p["actor_alert_streak"] >= 2), None)
            below = probe.evaluator_first_below(row["case"], row["world"])
            lead = (round(below-alert, 5) if below is not None and
                    alert is not None else None)
            checks[f"timing:{key}"] = (
                visible == row["first_person_ray_after_motion_s"] and
                alert == row["first_two_packet_actor_alert_s"] and
                below == row["first_evaluator_below_5cm_s"] and
                lead == row["alert_lead_to_5cm_s"])
    audit = {"schema": "robot-learning-lab.ape-held-monitor-audit.v1",
             "report_sha256": trial.sha(path), "summary": summary,
             "checks": checks, "pass": all(checks.values())}
    (probe.OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pass": audit["pass"], "summary": summary}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V28 audit failed")


if __name__ == "__main__":
    main()
