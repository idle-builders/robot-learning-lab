"""Paired V246p continuation: lose head depth only at the recovery looks.

The full side-search depth packets and route history stay medium-range and
pixel-matched to V246p. Only recovery serials 400+ have no head depth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_quality import reject_isolated_depth_spikes
from run import ROOT, sha
import run_blocked_forward_viewpoint_v321 as live
from run_medium_depth_recovery_v246p import CASES, MediumWindowMap, profile_depth


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v248p"


def main() -> None:
    original_cases = live.CASES
    original_depth = live.noisy_depth
    original_map = live.DepthVoxelMap
    medium = profile_depth("medium_30cm_1m", original_depth)

    def recovery_loss(ideal: np.ndarray, mode: str, seed: int,
                      serial: int) -> np.ndarray:
        if mode == "short" and serial >= 400:
            return np.full_like(ideal, np.nan)
        return medium(ideal, mode, seed, serial)

    try:
        live.CASES = CASES
        live.noisy_depth = recovery_loss
        live.DepthVoxelMap = MediumWindowMap
        live.main(output_root=OUT, source=SOURCE,
                  map_depth_filter=reject_isolated_depth_spikes,
                  experiment_label="V248p recovery-only head-depth dropout")
    finally:
        live.CASES = original_cases
        live.noisy_depth = original_depth
        live.DepthVoxelMap = original_map
    path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(path.read_text())
    report["virtual_head_depth_profile"] = "recovery_only_depth_dropout"
    report["scope_override"] = (
        "V246p's medium 30cm-1m head depth is retained for the full side "
        "search. At the three blocked-route recovery head looks only, all "
        "depth pixels are missing. Hypothetical near-range and fresh four-"
        "shin packets remain available. Actor does not receive the fault "
        "schedule or obstacle truth.")
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.recovery-only-depth-loss.v1",
        "source_sha256": sha(SOURCE),
        "v246_source_sha256": sha(Path(__import__(
            "run_medium_depth_recovery_v246p").__file__)),
        "cases": CASES,
        "runner_report_path": str(path.relative_to(ROOT)),
        "runner_report_sha256": sha(path),
        "rows": [{"case": row["case"], "status": row["status"],
                  "gate": row["recovery_gate"]["status"],
                  "clearance_m": row["evaluator_path_min_clearance_m"]}
                 for row in report["rows"]]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
