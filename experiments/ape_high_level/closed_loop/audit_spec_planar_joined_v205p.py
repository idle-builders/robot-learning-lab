"""Audit typed scan provenance, route geometry and owner outcomes for V204p."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_robocasa_joined_owner_route_v411 import audit_case
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-spec-planar-joined-v204p"
REPORT = OUT / "report.json"


def one(case: dict, profile: dict) -> dict:
    base = audit_case(case)
    nav = [row for row in case["trace"] if row["stage"] == "navigate"]
    packet_rows = []
    for index, (source, row) in enumerate(zip(case["sensor_packets"], nav)):
        path = ROOT / source["packet_path"]
        with np.load(path, allow_pickle=False) as packet:
            status = packet["status"]
            ranges = packet["ranges_m"]
            sample_index = int(packet["sample_index"])
        valid = status == 0
        counts = {"valid_hit": int(np.count_nonzero(valid)),
                  "too_near": int(np.count_nonzero(status == 1)),
                  "no_target": int(np.count_nonzero(status == 2))}
        packet_rows.append({
            "sample_index": index,
            "hash_match": sha(path) == source["packet_sha256"],
            "index_match": sample_index == index,
            "beam_count_match": len(status) == profile["beams"],
            "counts_match": counts == source["status_counts"],
            "actor_return_count_match": int(row["range_returns"]) == counts["valid_hit"],
            "invalid_nan": bool(np.all(np.isnan(ranges[~valid]))),
            "valid_bounded": bool(np.all(
                np.isfinite(ranges[valid]) &
                (ranges[valid] >= profile["minimum_m"]) &
                (ranges[valid] <= 5.02))),
        })
    base.update({
        "device": case["device"],
        "scan_packets": len(case["sensor_packets"]),
        "packet_count_match": len(packet_rows) == len(nav) == case["route_packet_count"],
        "packet_replay_pass": all(all(value for key, value in row.items()
                                      if key != "sample_index") for row in packet_rows),
        "first_packet_failure": next((row for row in packet_rows
                                      if not all(value for key, value in row.items()
                                                 if key != "sample_index")), None),
    })
    return base


def main() -> None:
    report = json.loads(REPORT.read_text())
    rows = [one(case, report["device_profiles"][case["device"]])
            for case in report["episodes"]]
    output = {
        "schema": "robot-learning-lab.ape-spec-planar-joined-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "joined_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_spec_planar_joined_v204p.py")),
        "v202_source_match": report["v202_source_sha256"] == sha(
            SOURCE.with_name("run_spec_planar_route_v202p.py")),
        "v410_source_match": report["v410_source_sha256"] == sha(
            SOURCE.with_name("run_robocasa_joined_owner_route_v410.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "cases": rows,
    }
    output["all_pass"] = (output["joined_source_match"] and
                          output["v202_source_match"] and
                          output["v410_source_match"] and
                          output["matched_scene_fingerprint"] and
                          all(row["proxy_margin_pass"] and
                              row["fresh_claim_evidence_pass"] and
                              row["response_contract_pass"] and
                              row["video_sha_pass"] and
                              row["video_frame_count_pass"] and
                              row["packet_count_match"] and
                              row["packet_replay_pass"] for row in rows))
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["device"], row["response"], row["outcome"],
              row["route_steps"], row["packet_replay_pass"], flush=True)
    print("audit passed", output["all_pass"])
    if not output["all_pass"]:
        raise AssertionError("Joined scan/owner audit failed")


if __name__ == "__main__":
    main()
