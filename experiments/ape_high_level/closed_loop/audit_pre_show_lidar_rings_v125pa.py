"""Verify lidar-style packet provenance and conservative route labels."""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha


SOURCE=Path(__file__)
BASE=ROOT/"artifacts/ape-closed-loop-cove50-lidar-rings"
REPORT=BASE/"report.json"
V123=ROOT/"artifacts/ape-closed-loop-cove50-active-pan/report.json"
RUNNER=ROOT/"experiments/ape_high_level/closed_loop/run_pre_show_lidar_rings_v125p.py"
OUT=BASE/"audit.json"


def main() -> None:
    data=json.loads(REPORT.read_text())
    prior=json.loads(V123.read_text())
    assert sha(RUNNER)==data["source_sha256"]
    assert sha(V123)==data["v123_report_sha256"]
    checks=2
    rows=[]
    for row in data["rows"]:
        match=next(r for r in prior["rows"] if
                   r["seed"]==row["seed"] and r["case"]==row["case"])
        assert row["v123_range_packet_path"]==match[
            "range_packet_path"]
        assert sha(ROOT/row["v123_range_packet_path"])==row[
            "v123_range_packet_sha256"]
        assert sha(ROOT/row["new_packet_path"])==row[
            "new_packet_sha256"]
        packet=json.loads((ROOT/row["new_packet_path"]).read_text())
        assert packet["capture_time_s"]==16.6
        assert all(ray["capture_time_s"]==16.6
                   for rays in packet["ray_packets"].values()
                   for ray in rays)
        checks+=5
        supported=[]
        for name,layout in row["layouts"].items():
            named=[route for route,result in layout["routes"].items()
                   if result["coverage_status"]=="observed_free"]
            assert named==layout["map_supported_routes"]
            if layout["pose_screen_scope"]=="estimated_only_rejects_all":
                assert not named
            else:
                assert layout["pose_screen_scope"]=="all_five_offsets"
            checks+=2
            if named:
                supported.append({"layout":name,"routes":named})
        for label,previous in (
            ("outward_plus_none",
             "yaw+1.4-lower_depth-outward"),
            ("dense_inward_plus_none",
             "yaw+1.4-lower_depth-outward_plus_dense_inward_diagnostic")):
            assert row["layouts"][label]["routes"]==match[
                "alternatives"][previous]["routes"]
            checks+=1
        rows.append({"seed":row["seed"],"case":row["case"],
                     "supported":supported,
                     "mount_gaps_m":row[
                         "mount_origin_to_historical_solid_gap_m"]})
    result={"schema":"robot-learning-lab.ape-lidar-ring-audit.v1",
            "source_sha256":sha(SOURCE),
            "report_sha256":sha(REPORT),"checks":checks,
            "scope":"Checks source/packet hashes, 16.6s ray timing, V123p unchanged baselines, and route label consistency. Does not independently recast rays or recompute collision geometry.",
            "rows":rows}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(OUT,checks)


if __name__=="__main__":
    main()
