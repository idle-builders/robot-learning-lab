"""V339: side-held can in the same live owner-help motion as V338.

The V338 co-located can/hand geometry buried the can inside the hand. This
variant places its center 8 cm to the hand's side, leaving a small modeled
surface gap and a visible label. All task and hazard logic is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import run_blocked_owner_help_v317 as owner
import run_live_moving_owner_offer_v338 as prior


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v339"
SIDE_HELD_LOCAL_XYZ_M = (0., -.24, 0.)
SIDE_HELD_SHOW_XYZ_M = (.35, 0., .20)


def main() -> None:
    original = (prior.HAND_LOCAL_XYZ_M, prior.OUT,
                owner.OWNER_SHOW_XYZ_M)
    try:
        prior.HAND_LOCAL_XYZ_M = SIDE_HELD_LOCAL_XYZ_M
        prior.OUT = OUT
        owner.OWNER_SHOW_XYZ_M = SIDE_HELD_SHOW_XYZ_M
        prior.main()
    finally:
        (prior.HAND_LOCAL_XYZ_M, prior.OUT,
         owner.OWNER_SHOW_XYZ_M) = original
    audit_path = OUT / "live-moving-owner-offer-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["scope"] = ("V339 repeats V338 on the same two seeds and "
                      "moving/parked prop, present/absent/identical-lookalike "
                      "owner modes. A scripted hand carries the can with its "
                      "center 8cm to one side of the hand rather than "
                      "co-located inside it; the same V317 task vision, "
                      "owner answer and V332 low-prop guard remain. The "
                      "position is a development geometry change, not a "
                      "physical grasp or a general hand-object detector.")
    audit["source_sha256"] = sha(SOURCE)
    audit["v338_source_sha256"] = sha(Path(prior.__file__))
    audit["side_held_local_xyz_m"] = SIDE_HELD_LOCAL_XYZ_M
    audit["side_held_show_xyz_m"] = SIDE_HELD_SHOW_XYZ_M
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
