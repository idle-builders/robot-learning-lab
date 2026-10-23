"""V377: lose the final two camera callbacks during independent hand show.

The V375 live hand scene and V373 live decision loop are reused. Only the
present-case capture availability changes, so the answer must withhold a
fresh find when the last point is 200 ms old.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_blocked_owner_help_v317 as owner_module
import run_live_independent_hand_help_v375 as hand
import run_live_moving_distant_help_v373 as base
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v377"


def frame_available(seed: int, mode: str, index: int) -> bool:
    return index not in (12, 13)


def main() -> None:
    old_frame, old_out, old_cases = (
        base.frame_available, hand.OUT, owner_module.CASES)
    try:
        base.frame_available = frame_available
        hand.OUT = OUT
        owner_module.CASES = tuple(case for case in old_cases
                                  if case[1] == "present")
        hand.main()
    finally:
        base.frame_available, hand.OUT, owner_module.CASES = (
            old_frame, old_out, old_cases)
    path = OUT / "blocked-owner-help-audit.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V377 selected Cove-50 guarded live owner-help with the V375 "
        "independently moving hand/can and V373 actor. Two present seeds "
        "lose the final two scheduled 10Hz camera callbacks entirely. "
        "The last visual/depth point is 200ms old at the scripted yes; "
        "the <=120ms fresh-answer gate must refuse visual reacquisition. "
        "This is injected packet loss, not optical occlusion, and owner "
        "speech and hand dynamics remain procedural. Inherited V373 video "
        "panel text is cosmetic and does not mark missing frames.")
    report["v377_source_sha256"] = sha(SOURCE)
    report["capture_loss_indices"] = [12, 13]
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
