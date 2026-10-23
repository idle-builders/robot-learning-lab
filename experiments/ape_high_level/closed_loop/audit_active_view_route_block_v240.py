"""V240 paired diagnostic: missing crate depth versus upper free-evidence gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import audit_origin_route_v152 as route
import run_fourth_split_live_v206 as baseline
import run_bounded_recent_view_v209 as recent
import run_moving_stool_short_depth_v229 as moving
from audit_crate_depth_holes_v232 import FRAME_ROOT as HOLE_FRAMES
from audit_head_short_depth_v225 import FRAME_ROOT as IDEAL_FRAMES
from audit_upper_free_evidence_gate_v234 import UpperFreeEvidenceClearance
from run import ROOT, sha
from run_moving_crate_hole_v238 import MovingCrateHoleRoom
from static_supported_localizer_v233 import StaticSupportedScanLocalizer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v240"
V225 = ROOT / "artifacts/ape-closed-loop-cove50-v225/head-short-depth-route-audit.json"
V232 = ROOT / "artifacts/ape-closed-loop-cove50-v232/crate-depth-hole-route-audit.json"


def main() -> None:
    prior = (moving.OUT, moving.SOURCE, moving.FRAME_ROOT, moving.V225,
             moving.DynamicShortRoom, route.TwoHeightClearance,
             baseline.controller, recent.TwoHeightClearance,
             recent.ScanLocalizer, recent.run, sys.argv[:])

    def with_recent_view(*args, **kwargs):
        kwargs["allow_recent_information_extension"] = True
        return prior[9](*args, **kwargs)

    conditions = (("hole_without_gate", HOLE_FRAMES, V232,
                   MovingCrateHoleRoom, prior[5]),
                  ("ideal_with_gate", IDEAL_FRAMES, V225,
                   prior[4], UpperFreeEvidenceClearance))
    reports = []
    try:
        baseline.controller = recent
        recent.ScanLocalizer = StaticSupportedScanLocalizer
        recent.run = with_recent_view
        for name, frames, audit, room_type, clearance in conditions:
            moving.OUT = OUT / name
            moving.SOURCE = SOURCE
            moving.FRAME_ROOT = frames
            moving.V225 = audit
            moving.DynamicShortRoom = room_type
            route.TwoHeightClearance = clearance
            recent.TwoHeightClearance = clearance
            sys.argv = [str(SOURCE), "--seeds", "111", "113"]
            moving.main()
            raw = moving.OUT / "moving-stool-short-depth-live-audit.json"
            report = json.loads(raw.read_text())
            reports.append({"condition": name,
                            "observation": "all_crate_depth_missing" if
                                name.startswith("hole") else "ideal_short_depth",
                            "upper_free_evidence_gate":
                                clearance is UpperFreeEvidenceClearance,
                            "audit_path": str(raw.relative_to(ROOT)),
                            "rows": [{k: r.get(k) for k in (
                                "seed", "status", "moves",
                                "evaluator_actual_target_gap_m",
                                "evaluator_min_model_clearance_m",
                                "evaluator_max_pose_error_m",
                                "scan_rejections", "information_extensions",
                                "final_fresh_correct",
                                "wrong_object_acceptances",
                                "matched_absent_accepted", "range_alarms",
                                "video_sha256")}
                                for r in report["rows"]]})
            print(name, [(r["seed"], r["status"], r["moves"],
                          r["evaluator_min_model_clearance_m"])
                         for r in report["rows"]], flush=True)
    finally:
        (moving.OUT, moving.SOURCE, moving.FRAME_ROOT, moving.V225,
         moving.DynamicShortRoom, route.TwoHeightClearance,
         baseline.controller, recent.TwoHeightClearance,
         recent.ScanLocalizer, recent.run, sys.argv) = prior
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "V240 revealed paired diagnostic for V239 seeds111/113: same moving stool, static-supported virtual 11cm localization, bounded recent-view extension, RGB floor target, short depth, kinematic selected body and instant virtual low-range stop. Pair A retains all-crate-depth hole but removes the necessary upper free-evidence gate; pair B restores ideal short depth but retains the gate. V239 supplies hole+gate comparator. Simulator segmentation creates missing depth, never actor truth. Evaluator scores after decisions. No physical sensor or native stop qualification.",
              "source_sha256": sha(SOURCE),
              "v239_sha256": sha(ROOT / "artifacts/ape-closed-loop-cove50-v239/active-view-moving-hole-live-audit.json"),
              "rows": reports}
    (OUT / "active-view-route-block-ablation.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
