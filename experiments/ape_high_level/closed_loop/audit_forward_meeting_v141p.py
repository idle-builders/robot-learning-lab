"""Audit the forward-station evaluator/map comparison."""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import screen_forward_meeting_v139p as v139p
import screen_forward_route_observability_v140p as v140p


SOURCE = Path(__file__)
OUT = v140p.OUT / "audit.json"


def main() -> None:
    geometry_path = v139p.OUT / "screen.json"
    map_path = v140p.OUT / "report.json"
    geometry = json.loads(geometry_path.read_text())
    route = json.loads(map_path.read_text())
    checks = 0
    assert geometry["source_sha256"] == sha(v139p.SOURCE)
    assert route["source_sha256"] == sha(v140p.SOURCE)
    assert route["v139_report_sha256"] == sha(geometry_path)
    checks += 3
    assert len(geometry["rows"]) == 24
    assert len(route["rows"]) == 2
    checks += 2
    passed = [r for r in geometry["rows"]
              if r["minimum_scene_gap_m"] >= .05]
    assert any(r["case"] == "clear" and
               r["meeting_offset_xy_m"] == [.12, 0.]
               for r in passed)
    assert any(r["case"] == "hand_close" and
               r["meeting_offset_xy_m"] == [.12, 0.]
               for r in passed)
    checks += 2
    supported = 0
    for record in route["rows"]:
        path = ROOT / record["new_packet_path"]
        assert sha(path) == record["new_packet_sha256"]
        packet = json.loads(path.read_text())
        for rays in packet["ray_packets"].values():
            assert rays and all(ray["capture_time_s"] == 8.0
                                for ray in rays)
            checks += 1
        for alternative in record["alternatives"].values():
            for result in alternative["routes"].values():
                supported += result["coverage_status"] == "observed_free"
                checks += 1
        checks += 1
    assert supported == 0
    checks += 1
    audit = {"schema": "robot-learning-lab.ape-forward-meeting-audit.v1",
             "source_sha256": sha(SOURCE),
             "geometry_report_sha256": sha(geometry_path),
             "actor_report_sha256": sha(map_path),
             "checks": checks,
             "geometrically_passing_scene_rows": len(passed),
             "actor_supported_routes": supported,
             "scope": "Source/packet/timestamp and geometry-versus-actor "
             "consistency checks. Evaluator clearance does not authorize "
             "an actor move or qualify hardware."}
    OUT.write_text(json.dumps(audit, indent=2)+"\n")
    print(OUT, checks)


if __name__ == "__main__":
    main()
