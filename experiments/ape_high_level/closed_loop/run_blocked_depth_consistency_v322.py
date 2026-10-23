"""V322: paired live depth-spike rejection at the blocked viewpoint gate.

Only sparse head depth rays entering the route map receive a 5x5 local
consistency check. RGB identity, raw depth evidence, shin scans, range packets,
controller thresholds, room motion and owner behavior remain V321's.
"""

from __future__ import annotations

from pathlib import Path

from depth_quality import reject_isolated_depth_spikes
from run import ROOT
from run_blocked_forward_viewpoint_v321 import main as run_live


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v322"


if __name__ == "__main__":
    run_live(output_root=OUT, source=SOURCE,
             map_depth_filter=reject_isolated_depth_spikes,
             experiment_label="V322")
