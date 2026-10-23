"""V139p: geometry screen of front-visible early meeting stations.

This is an evaluator-only alternative to rear retreat. The owner remains at
the same world show position, so advancing may make person clearance worse.
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
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-forward-meeting"
X_OFFSETS_M = (.04, .08, .12, .16)
Y_OFFSETS_M = (-.04, 0., .04)
POSTURE = (.6, 0.)
CASES = ("clear", "hand_close")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    seed = v126p.SEED
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and
                 row["case"][0] == seed)
    source = (v128p.V98 / "moving" / f"seed{seed}-present" /
              f"seed{seed}-present/side-probe-00.npz")
    with np.load(source) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
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
                    station_qpos = qpos.copy()
                    station_qpos[:2] += (x, y)
                    station_estimate = estimate.copy()
                    station_estimate[:2] += (x, y)
                    pair = []
                    for case in CASES:
                        row = v126p.one(room, station_qpos,
                                        station_estimate, case,
                                        "left_p06", POSTURE)
                        row["meeting_offset_xy_m"] = [x, y]
                        rows.append(row)
                        pair.append(round(row["minimum_scene_gap_m"]*100, 2))
                    print(f"x{x:+.2f} y{y:+.2f}: clear,hand {pair}cm",
                          flush=True)
        finally:
            room.close()
    finally:
        structural.structural_model = original_model
    report = {"schema": "robot-learning-lab.ape-forward-meeting-screen.v1",
              "source_sha256": sha(SOURCE),
              "source_packet_path": str(source.relative_to(ROOT)),
              "source_packet_sha256": sha(source),
              "selected_visual_sha256": sha(selected_visual()),
              "scope": "Evaluator-only front/side station sweep from the "
              "saved7.6s staging pose on historical Direct Frame geometry. "
              "The +0.6rad left shoulder transitions16.2-16.6s, owner "
              "shows16.6-17.0s, prop moves17.2-18.0s, and full-body "
              "clearance is sampled20ms through18.2s. No station path, "
              "actor observation, item recognition or skill is implied.",
              "rows": rows}
    (OUT / "screen.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "screen.json")


if __name__ == "__main__":
    main()
