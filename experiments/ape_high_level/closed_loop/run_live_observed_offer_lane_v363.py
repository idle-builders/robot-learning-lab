"""V363: enforce the requested far offer with fresh actor object position.

This reuses V360's research-only kinematic body preparation. After the
incoming item is freshly seen, its RGB-D world position must match the
public requested zone before the owner identity question or visual accept.
It does not authorize physical transfer or the prior retreat.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_live_handoff_zone_v353 as live
import run_live_repositioned_handoff_v360 as reposition
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v363"


def main() -> None:
    live.main(out=OUT, prepare=reposition.prepare, variant="V363",
              allow_research_reposition=True,
              enforce_invitation_zone=True)
    path = OUT / "live-handoff-zone-audit.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V363 repeats the V360 selected Cove-50 live blocked-route owner-"
        "help episodes with a fresh actor RGB-D accepted-item world-position "
        "check against the far invitation point before owner identity/visual "
        "acceptance. The horizontal zone tolerance is 10cm, frozen from the "
        "V360 seed170 development pair; seed172 is a paired same-geometry "
        "challenge. The kinematic 10cm back+5cm side retreat still uses the "
        "research-only evaluator-bounded continuation, not an actor-"
        "authorized path or native skill. No physical handoff, real owner "
        "response or stopping is proved.")
    report["v363_source_sha256"] = sha(SOURCE)
    report["v360_preparation_source_sha256"] = sha(Path(reposition.__file__))
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in report["rows"]:
        print(row["condition"], row["case"][0], row["case"][1],
              row["status"], flush=True)


if __name__ == "__main__":
    main()
