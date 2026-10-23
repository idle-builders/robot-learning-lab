"""Paired V268p fault: all four recovery near-range fans have no returns.

The earlier owner search, head RGB-D, shin sensing, scenes, object identities
and owner responses are fixed. Missing range packets add no free or occupied
rays. The actor is not given the evaluator's geometry or fault schedule.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_blocked_forward_viewpoint_v321 as live
import run_independent_identity_rooms_v268p as challenge
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v270p"
BASE = ROOT / "artifacts/ape-articulated-depth-route-v268p"
CHALLENGE_SOURCE = Path(challenge.__file__)


def missing_near_packets(room, pose, seed, time_s, voxel_map):
    del room, pose, seed, time_s, voxel_map
    packets = {name: [] for name in live.NEAR_MOUNTS}
    counts = {name: {"valid": 0, "dropout": 1581, "self_first": 0,
                     "environment_hit": 0, "max_range": 0,
                     "health": "missing_packet"}
              for name in live.NEAR_MOUNTS}
    return packets, counts


def main() -> None:
    old_out, old_source = challenge.OUT, challenge.SOURCE
    old_profile = challenge.SENSOR_PROFILE
    old_packets = live.add_near_packets
    try:
        challenge.OUT, challenge.SOURCE = OUT, SOURCE
        challenge.SENSOR_PROFILE = (
            "healthy 30cm-1m head RGB-D and four-shin first-hit through "
            "all captures; every recovery near-body fan packet missing "
            "(no free or occupied rays)")
        live.add_near_packets = missing_near_packets
        challenge.main()
    finally:
        challenge.OUT, challenge.SOURCE = old_out, old_source
        challenge.SENSOR_PROFILE = old_profile
        live.add_near_packets = old_packets
    (OUT / CHALLENGE_SOURCE.name).write_bytes(CHALLENGE_SOURCE.read_bytes())
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report.update({
        "challenge_source_sha256": sha(CHALLENGE_SOURCE),
        "paired_v268_report_sha256": sha(BASE / "report.json"),
        "fault": "At every recovery head-map decision, all four virtual "
                 "near-body fan packets contain zero valid rays. Earlier "
                 "search, head RGB-D, shin points and scripted owner "
                 "responses are unchanged.",
    })
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
