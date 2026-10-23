"""V127p: evaluator screen of early meeting base offsets plus raised arm.

This assumes the ape reaches the candidate station before the owner offer;
travel, actor selection and joint skill are separate gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import add_person
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import screen_early_show_posture_v126p as v126p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-meeting-offset"
V98 = v126p.V98
SEED = 170
X_OFFSETS_M = (0., -.04, -.08, -.12, -.16)
Y_OFFSETS_M = (-.04, 0., .04)
POSTURE = (-.6, 0.)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"] if row["condition"] == "moving"
                 and row["case"][0] == SEED)
    source = (V98 / "moving" / f"seed{SEED}-present" /
              f"seed{SEED}-present/side-probe-00.npz")
    with np.load(source) as packet:
        original_qpos = packet["actor_qpos"].copy()
        original_estimate = packet["estimated_pose_xyyaw"].copy()
        assert abs(float(packet["capture_time_s"])-7.6) < 1e-5
    original_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    rows = []
    try:
        room = OfferDriftRoom(case=tuple(prior["case"]),
                              pose_source="command", branch=OUT)
        try:
            for x in X_OFFSETS_M:
                for y in Y_OFFSETS_M:
                    qpos = original_qpos.copy()
                    qpos[:2] += (x, y)
                    estimate = original_estimate.copy()
                    estimate[:2] += (x, y)
                    pair = []
                    for case in v126p.CASES_TO_SCREEN:
                        row = v126p.one(room, qpos, estimate, case,
                                        "left_n06", POSTURE)
                        row["meeting_offset_xy_m"] = [x, y]
                        rows.append(row)
                        pair.append(round(row["minimum_scene_gap_m"]*100, 2))
                    print(f"x{x:+.2f} y{y:+.2f}: clear,hand {pair}cm",
                          flush=True)
        finally:
            room.close()
    finally:
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-early-meeting-offset-screen.v1",
        "source_sha256": sha(SOURCE),
        "posture_screen_source_sha256": sha(v126p.SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "source_packet_path": str(source.relative_to(ROOT)),
        "source_packet_sha256": sha(source),
        "selected_visual_sha256": sha(selected_visual()),
        "scope": "Evaluator-only early meeting offset sweep from V98's "
        "saved7.6s staging pose. At16.2s, assumes each offset has already "
        "been achieved and checks a direct0.4s left-shoulder transition, "
        "0.4s articulated owner show, and moving prop through18.2s at20ms. "
        "The clear and hand-height side-block cases are run. No route to "
        "the station, actor-sensor authorization, image identity, motor "
        "response or physical clearance certification is claimed.",
        "rows": rows,
    }
    (OUT / "screen.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "screen.json")


if __name__ == "__main__":
    main()
