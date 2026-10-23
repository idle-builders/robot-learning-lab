"""V341: rerun V340 with the item outside the owner-hand scene.

This isolates the hand/forearm sensor timing from the side-held can that
became a short-depth early-warning proxy in V340.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_owner_reach_speed_v340 as prior
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v341"


def main() -> None:
    original = (prior.ITEM_SIDE_OFFSET_M.copy(), prior.OUT)
    try:
        prior.ITEM_SIDE_OFFSET_M = np.asarray((10., 0., 0.))
        prior.OUT = OUT
        prior.main()
    finally:
        prior.ITEM_SIDE_OFFSET_M, prior.OUT = original
    path = OUT / "owner-reach-speed-audit.json"
    audit = json.loads(path.read_text())
    audit["scope"] = ("V341 paired V340 0.4/0.8/1.2s reach-speed audit with the "
                      "carried can moved outside the camera and hand volume. "
                      "Only the visible scripted forearm/hand contributes to "
                      "the RGB motion and short/long depth warning. Actor "
                      "packets still exclude simulator segmentation and true "
                      "distance; evaluator-only human clearances score the "
                      "same 200ms scripted pause. This does not model a "
                      "biomechanical person or hardware depth camera.")
    audit["source_sha256"] = sha(SOURCE)
    audit["v340_source_sha256"] = sha(Path(prior.__file__))
    audit["carried_item"] = False
    path.write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
