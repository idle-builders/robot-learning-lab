"""Check V122p source packets, alert provenance and sampled path claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run import ROOT, sha


SOURCE = Path(__file__)
REPORT = ROOT / "artifacts/ape-closed-loop-cove50-alert-response/report.json"
V121 = ROOT / "artifacts/ape-closed-loop-cove50-v121/report.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-alert-response/audit.json"


def main() -> None:
    data = json.loads(REPORT.read_text())
    source = ROOT / "experiments/ape_high_level/closed_loop/audit_alert_response_envelope_v122p.py"
    assert sha(source) == data["source_sha256"]
    assert sha(V121) == data["v121_report_sha256"]
    sensor = json.loads(V121.read_text())
    checks = 2
    summaries = []
    for row in data["rows"]:
        assert sha(ROOT / row["source_packet_path"]) == row[
            "source_packet_sha256"]
        checks += 1
        upstream = next(r for r in sensor["rows"] if
                        r["seed"] == row["seed"] and
                        r["mode"] == row["sensor_mode"] and
                        r["sensor_trial"] == row["sensor_trial"] and
                        r["branch"] == "fast_ignored")
        event = upstream["audited_alarms"]["four"]["first_supported"]
        assert event is not None and event["time_s"] == row["alert_time_s"]
        packet = next(f for f in upstream["frames"] if
                      f["capture_time_s"] == event["time_s"])
        assert packet["packet_path"] == row["alert_packet_path"]
        assert sha(ROOT / packet["packet_path"]) == packet["packet_sha256"]
        checks += 3
        passed = 0
        for trajectory in row["trajectories"]:
            path = trajectory["path"]
            assert path[0]["time_s"] == row["alert_time_s"]
            assert path[-1]["time_s"] == 18.2
            assert all(b["time_s"] > a["time_s"] and
                       b["time_s"]-a["time_s"] <= .0201
                       for a,b in zip(path,path[1:]))
            smallest = min(p["evaluator_scene_gap_m"] for p in path)
            assert abs(smallest-trajectory["minimum_scene_gap_m"]) <= 1e-5
            first = next((p["time_s"] for p in path
                          if p["evaluator_scene_gap_m"] < .05),None)
            assert first == trajectory["first_below_5cm_s"]
            assert (smallest >= .05) == trajectory["provisional_5cm_pass"]
            assert all(0. <= p["executed_fraction"] <= 1. for p in path)
            checks += 7
            passed += trajectory["provisional_5cm_pass"]
        summaries.append({"seed":row["seed"],"case":row["case"],
                          "alert_time_s":row["alert_time_s"],
                          "passing_trajectories":passed,
                          "total_trajectories":len(row["trajectories"])})
    audit={"schema":"robot-learning-lab.ape-alert-response-audit.v1",
           "source_sha256":sha(SOURCE),
           "report_sha256":sha(REPORT),
           "checks":checks,
           "trajectories":sum(x["total_trajectories"] for x in summaries),
           "scope":"Checks source and alert packet hashes, event-to-path provenance, 20ms path ordering, recorded minima and breach labels. Does not independently recompute MuJoCo collision geometry or qualify kinematic motion as a skill.",
           "summaries":summaries}
    OUT.write_text(json.dumps(audit,indent=2)+"\n")
    print(OUT,checks,"checks")


if __name__=="__main__":
    main()
