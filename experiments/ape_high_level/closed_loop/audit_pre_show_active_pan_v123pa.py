"""Check active-look packet timing, hashes and route refusal labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha


SOURCE=Path(__file__)
ROOT_OUT=ROOT/"artifacts/ape-closed-loop-cove50-active-pan"
REPORT=ROOT_OUT/"report.json"
OUT=ROOT_OUT/"audit.json"
RUNNER=ROOT/"experiments/ape_high_level/closed_loop/run_pre_show_active_pan_v123p.py"


def main() -> None:
    data=json.loads(REPORT.read_text())
    assert sha(RUNNER)==data["source_sha256"]
    checks=1
    summaries=[]
    for row in data["rows"]:
        assert sha(ROOT/row["source_packet_path"])==row[
            "source_packet_sha256"]
        assert sha(ROOT/row["old_head_path"])==row[
            "old_head_sha256"]
        checks+=2
        for angle,cameras in row["head_packets"].items():
            for name,meta in cameras.items():
                assert sha(ROOT/meta["path"])==meta["sha256"]
                with np.load(ROOT/meta["path"]) as packet:
                    assert float(packet["capture_time_s"])==16.6
                    assert np.isfinite(packet["actor_qpos"]).all()
                    assert packet["filtered_depth_m"].ndim==2
                checks+=4
        assert sha(ROOT/row["range_packet_path"])==row[
            "range_packet_sha256"]
        packet=json.loads((ROOT/row["range_packet_path"]).read_text())
        assert packet["capture_time_s"]==16.6
        assert all(ray["capture_time_s"]==16.6
                   for rays in packet["ray_packets"].values()
                   for ray in rays)
        checks+=3
        supported=[]
        for key,alt in row["alternatives"].items():
            named=[name for name,route in alt["routes"].items()
                   if route["coverage_status"]=="observed_free"]
            assert named==alt["map_supported_routes"]
            assert alt["pose_screen_scope"] in (
                "estimated_only_rejects_all","all_five_offsets")
            if alt["pose_screen_scope"]=="estimated_only_rejects_all":
                assert not named
                assert all(len(route["pose_samples"])==1
                           for route in alt["routes"].values())
            else:
                assert all(len(route["pose_samples"])==5
                           for route in alt["routes"].values())
            checks+=4
            if named:
                supported.append({"alternative":key,"routes":named})
        summaries.append({"seed":row["seed"],"case":row["case"],
                          "alternatives":len(row["alternatives"]),
                          "supported":supported})
    audit={"schema":"robot-learning-lab.ape-pre-show-active-pan-audit.v1",
           "source_sha256":sha(SOURCE),"report_sha256":sha(REPORT),
           "checks":checks,
           "scope":"Checks exact source/packet hashes, 16.6s capture stamps including every range ray, image payload shape, and consistency of map-supported route labels. Does not independently recast rays or recompute voxel coverage.",
           "summaries":summaries}
    OUT.write_text(json.dumps(audit,indent=2)+"\n")
    print(OUT,checks)


if __name__=="__main__":
    main()
